use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::sync::Arc;

use vulkanalia::prelude::v1_4::*;
use vulkanalia_bootstrap::Device;
use vulkanalia_vma as vma;

use crate::gpu_resources::{GpuBuffer, GpuImage};
use crate::pipeline::{
    allocate_descriptor_set, create_compute_pipeline, create_descriptor_set_layout,
};
use crate::tile_cache::{SdfCacheFile, TileLoadQueue};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// World-space size of each tile in meters.
pub const TILE_SIZE: f32 = 64.0;
/// Texels per tile dimension for the SDF atlas (signed_dist, s_coord).
pub const TILE_RESOLUTION: u32 = 64;
/// Border texels on each side of an SDF tile for bilinear cross-tile interpolation.
pub const TILE_BORDER: u32 = 1;
/// Total texels per tile slot in the atlas (data + 2 × border).
pub const TILE_SLOT_SIZE: u32 = TILE_RESOLUTION + 2 * TILE_BORDER;
/// Texels per tile dimension for the road-ID atlas.
pub const ROAD_ID_RESOLUTION: u32 = 32;
/// Maximum number of dirty tiles dispatched per frame to avoid GPU stalls.
const MAX_DIRTY_DISPATCH: usize = 16;

// ---------------------------------------------------------------------------
// Push constants — must match the shader
// ---------------------------------------------------------------------------

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct SdfPushConstants {
    atlas_offset_x: u32,
    atlas_offset_y: u32,
    tile_world_x: f32,
    tile_world_y: f32,
    tile_size: f32,
    tile_resolution: u32,
    tile_index: u32,
    road_id_atlas_offset_x: u32,
    road_id_atlas_offset_y: u32,
    road_id_resolution: u32,
    tile_border: u32,
    _pad: u32,
}

// ---------------------------------------------------------------------------
// Tile key + atlas slot
// ---------------------------------------------------------------------------

/// Integer tile coordinates (column, row) in world space.
/// Tile (ix, iy) covers [ix*TILE_SIZE .. (ix+1)*TILE_SIZE] × [iy*TILE_SIZE .. (iy+1)*TILE_SIZE].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TileKey {
    pub ix: i32,
    pub iy: i32,
}

impl TileKey {
    pub fn from_world(x: f32, y: f32) -> Self {
        Self {
            ix: (x / TILE_SIZE).floor() as i32,
            iy: (y / TILE_SIZE).floor() as i32,
        }
    }

    /// Bottom-left corner of this tile in world space.
    pub fn world_origin(&self) -> (f32, f32) {
        (self.ix as f32 * TILE_SIZE, self.iy as f32 * TILE_SIZE)
    }
}

/// CPU-side tile data.
pub struct TileInfo {
    /// Index into the tile_headers GPU buffer.
    pub header_index: u32,
}

// ---------------------------------------------------------------------------
// TileMap — CPU-side bookkeeping for tile ↔ segment overlap
// ---------------------------------------------------------------------------

/// Per-tile header (matches GPU TileHeader).
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct TileHeader {
    pub offset: u32,
    pub count: u32,
}

/// Manages the mapping from tile keys to atlas slots and the overlap data.
pub struct TileMap {
    /// Active tiles keyed by tile coordinate.
    pub tiles: HashMap<TileKey, TileInfo>,
    /// Tile headers (one per active tile, indexed by header_index).
    pub headers: Vec<TileHeader>,
    /// Flat list of segment indices per tile (offsets stored in headers).
    pub segment_indices: Vec<u32>,
    /// Road index per entry in segment_indices (parallel array).
    pub road_indices: Vec<u32>,
}

impl TileMap {
    pub fn new() -> Self {
        Self {
            tiles: HashMap::new(),
            headers: Vec::new(),
            segment_indices: Vec::new(),
            road_indices: Vec::new(),
        }
    }

    /// Rebuild tile data from road segments.
    /// `segments` is the flat array of (origin, heading, length, segment_index, road_index)
    /// tuples — we compute AABB overlap with each tile.
    ///
    /// Returns the set of tile keys whose segment content changed (added, removed,
    /// or segment set differs). This can be used to invalidate stale cache entries.
    pub fn rebuild(&mut self, segment_data: &[SegmentAABB]) -> HashSet<TileKey> {
        // Snapshot old per-tile segment sets for change detection
        let old_tile_segs: HashMap<TileKey, Vec<(u32, u32)>> = self
            .tiles
            .keys()
            .map(|&key| {
                let info = &self.tiles[&key];
                let hdr = &self.headers[info.header_index as usize];
                let start = hdr.offset as usize;
                let end = start + hdr.count as usize;
                let mut segs: Vec<(u32, u32)> = self.segment_indices[start..end]
                    .iter()
                    .zip(&self.road_indices[start..end])
                    .map(|(&s, &r)| (s, r))
                    .collect();
                segs.sort();
                (key, segs)
            })
            .collect();

        // Clear old data
        self.tiles.clear();
        self.headers.clear();
        self.segment_indices.clear();
        self.road_indices.clear();

        // Collect which segments overlap which tiles
        let mut tile_segments: HashMap<TileKey, Vec<(u32, u32)>> = HashMap::new();

        for seg in segment_data {
            // Determine which tiles this segment's AABB overlaps
            let min_ix = (seg.aabb_min.0 / TILE_SIZE).floor() as i32;
            let min_iy = (seg.aabb_min.1 / TILE_SIZE).floor() as i32;
            let max_ix = (seg.aabb_max.0 / TILE_SIZE).floor() as i32;
            let max_iy = (seg.aabb_max.1 / TILE_SIZE).floor() as i32;

            for iy in min_iy..=max_iy {
                for ix in min_ix..=max_ix {
                    let key = TileKey { ix, iy };
                    tile_segments
                        .entry(key)
                        .or_default()
                        .push((seg.segment_index, seg.road_index));
                }
            }
        }

        // Build headers (no atlas slot assignment — that's the streaming cache's job)
        for (key, seg_list) in &tile_segments {
            let header_index = self.headers.len() as u32;
            let offset = self.segment_indices.len() as u32;
            let count = seg_list.len() as u32;

            for &(seg_idx, road_idx) in seg_list {
                self.segment_indices.push(seg_idx);
                self.road_indices.push(road_idx);
            }

            self.headers.push(TileHeader { offset, count });
            self.tiles.insert(*key, TileInfo { header_index });
        }

        // Compute changed tiles
        let mut changed = HashSet::new();

        // Tiles that existed before but are gone or have different segments
        for (key, old_segs) in &old_tile_segs {
            match tile_segments.get(key) {
                None => {
                    changed.insert(*key);
                } // removed
                Some(new_segs) => {
                    let mut sorted_new = new_segs.clone();
                    sorted_new.sort();
                    if *old_segs != sorted_new {
                        changed.insert(*key);
                    }
                }
            }
        }

        // Tiles that are new (didn't exist before)
        for key in tile_segments.keys() {
            if !old_tile_segs.contains_key(key) {
                changed.insert(*key);
            }
        }

        changed
    }
}

/// Precomputed AABB for a segment in world space.
pub struct SegmentAABB {
    pub segment_index: u32,
    pub road_index: u32,
    pub aabb_min: (f32, f32),
    pub aabb_max: (f32, f32),
}

// ---------------------------------------------------------------------------
// SdfTileManager — GPU resources for SDF tile generation
// ---------------------------------------------------------------------------

/// Default atlas dimension in tiles (32 × 32 = 1024 slots, 8192 × 8192 texels).
const DEFAULT_ATLAS_TILES: u32 = 32;

pub struct SdfTileManager {
    // SDF atlas (RG16F — signed_dist + s_coord, bilinear sampled)
    pub sdf_atlas: GpuImage,
    // Road ID atlas (R16_UINT — discrete road id, nearest sampled)
    pub road_id_atlas: GpuImage,
    /// Atlas dimension in tiles (fixed at creation).
    pub atlas_tiles_dim: u32,

    // Compute pipeline
    pipeline: vk::Pipeline,
    pipeline_layout: vk::PipelineLayout,
    descriptor_set_layout: vk::DescriptorSetLayout,
    descriptor_pool: vk::DescriptorPool,
    descriptor_set: vk::DescriptorSet,

    // GPU buffers for tile data
    tile_header_buffer: Option<GpuBuffer>,
    tile_segment_index_buffer: Option<GpuBuffer>,
    tile_road_index_buffer: Option<GpuBuffer>,

    // Tile bookkeeping (infinite)
    pub tile_map: TileMap,

    // Atlas streaming cache
    /// Maps atlas slot index → tile key currently occupying it (None if free).
    slot_to_tile: Vec<Option<TileKey>>,
    /// Maps tile key → atlas slot for tiles currently loaded in the atlas.
    pub tile_to_slot: HashMap<TileKey, u32>,
    /// Free atlas slot stack.
    free_slots: Vec<u32>,
    /// Tiles that need their SDF re-generated (newly assigned slots).
    pub dirty_tiles: HashSet<TileKey>,
    /// Set to true when the set of loaded tiles changes (for re-uploading tile instances).
    pub tiles_changed: bool,

    // Disk cache (optional)
    cache: Option<SdfCacheFile>,
    load_queue: Option<TileLoadQueue>,
    /// Tiles currently being loaded from disk cache.
    loading_tiles: HashSet<TileKey>,

    // Staging buffer for cache→GPU uploads
    staging_buffer: Option<GpuBuffer>,
    staging_ptr: *mut u8,

    /// Current image layout of both atlas textures (they always share the same layout).
    pub atlas_layout: vk::ImageLayout,
}

impl SdfTileManager {
    /// Create the SDF tile manager. Takes pre-compiled SPIR-V, creates the atlas and pipeline.
    pub fn new(
        device: &Arc<Device>,
        allocator: &vma::Allocator,
        spirv: &[u32],
    ) -> anyhow::Result<Self> {
        let atlas_dim = DEFAULT_ATLAS_TILES;

        // SDF atlas: RG16F, bilinear-sampled (signed_dist + s_coord)
        let sdf_atlas_size = atlas_dim * TILE_SLOT_SIZE;
        let sdf_atlas = GpuImage::new_storage_2d(
            device,
            allocator,
            sdf_atlas_size,
            sdf_atlas_size,
            vk::Format::R16G16_SFLOAT,
        )?;

        // Road ID atlas: R16_SFLOAT, nearest-sampled (discrete road id stored as float)
        let road_id_atlas_size = atlas_dim * ROAD_ID_RESOLUTION;
        let road_id_atlas = GpuImage::new_storage_2d(
            device,
            allocator,
            road_id_atlas_size,
            road_id_atlas_size,
            vk::Format::R16_SFLOAT,
        )?;

        // Descriptor set layout:
        //   binding 0: storage image (sdf atlas) — write
        //   binding 1: storage buffer (segments) — read
        //   binding 2: storage buffer (tile_headers) — read
        //   binding 3: storage buffer (tile_segment_indices) — read
        //   binding 4: storage buffer (road_indices) — read
        //   binding 5: storage image (road_id atlas) — write
        let bindings = [
            vk::DescriptorSetLayoutBinding::builder()
                .binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
                .build(),
            vk::DescriptorSetLayoutBinding::builder()
                .binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
                .build(),
            vk::DescriptorSetLayoutBinding::builder()
                .binding(2)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
                .build(),
            vk::DescriptorSetLayoutBinding::builder()
                .binding(3)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
                .build(),
            vk::DescriptorSetLayoutBinding::builder()
                .binding(4)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
                .build(),
            vk::DescriptorSetLayoutBinding::builder()
                .binding(5)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
                .build(),
        ];

        let descriptor_set_layout = create_descriptor_set_layout(device, &bindings)?;

        // Descriptor pool
        let pool_sizes = [
            vk::DescriptorPoolSize::builder()
                .type_(vk::DescriptorType::STORAGE_IMAGE)
                .descriptor_count(2)
                .build(),
            vk::DescriptorPoolSize::builder()
                .type_(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(4)
                .build(),
        ];
        let pool_info = vk::DescriptorPoolCreateInfo::builder()
            .max_sets(1)
            .pool_sizes(&pool_sizes);
        let descriptor_pool = unsafe { device.create_descriptor_pool(&pool_info, None)? };

        let descriptor_set =
            allocate_descriptor_set(device, descriptor_pool, descriptor_set_layout)?;

        // Push constant range
        let push_constant_ranges = [vk::PushConstantRange::builder()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .offset(0)
            .size(std::mem::size_of::<SdfPushConstants>() as u32)
            .build()];

        let (pipeline, pipeline_layout) = create_compute_pipeline(
            device,
            spirv,
            "main",
            &[descriptor_set_layout],
            &push_constant_ranges,
        )?;

        // Write atlas descriptors (binding 0: sdf atlas, binding 5: road_id atlas)
        let sdf_image_info = [vk::DescriptorImageInfo::builder()
            .image_view(sdf_atlas.view)
            .image_layout(vk::ImageLayout::GENERAL)
            .build()];
        let road_id_image_info = [vk::DescriptorImageInfo::builder()
            .image_view(road_id_atlas.view)
            .image_layout(vk::ImageLayout::GENERAL)
            .build()];
        let writes = [
            vk::WriteDescriptorSet::builder()
                .dst_set(descriptor_set)
                .dst_binding(0)
                .dst_array_element(0)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .image_info(&sdf_image_info)
                .build(),
            vk::WriteDescriptorSet::builder()
                .dst_set(descriptor_set)
                .dst_binding(5)
                .dst_array_element(0)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .image_info(&road_id_image_info)
                .build(),
        ];
        unsafe {
            device.update_descriptor_sets(&writes, &[] as &[vk::CopyDescriptorSet]);
        }

        let total_slots = atlas_dim * atlas_dim;
        Ok(Self {
            sdf_atlas,
            road_id_atlas,
            atlas_tiles_dim: atlas_dim,
            pipeline,
            pipeline_layout,
            descriptor_set_layout,
            descriptor_pool,
            descriptor_set,
            tile_header_buffer: None,
            tile_segment_index_buffer: None,
            tile_road_index_buffer: None,
            tile_map: TileMap::new(),
            slot_to_tile: vec![None; total_slots as usize],
            tile_to_slot: HashMap::new(),
            free_slots: (0..total_slots).rev().collect(),
            dirty_tiles: HashSet::new(),
            tiles_changed: false,
            cache: None,
            load_queue: None,
            loading_tiles: HashSet::new(),
            staging_buffer: None,
            staging_ptr: std::ptr::null_mut(),
            atlas_layout: vk::ImageLayout::UNDEFINED,
        })
    }

    /// Update which tiles are loaded in the atlas based on the camera view.
    /// Tiles visible in the AABB are loaded; non-visible tiles may be evicted.
    /// Call this each frame before `dispatch_dirty_tiles`.
    pub fn update_visible(&mut self, min_x: f32, min_y: f32, max_x: f32, max_y: f32) {
        if self.tile_map.tiles.is_empty() {
            return;
        }

        // Pad the view bounds by one tile in each direction to prevent
        // thrashing when the camera sits exactly on a tile boundary and to
        // pre-load tiles just outside the viewport for smoother panning.
        let min_x = min_x - TILE_SIZE;
        let min_y = min_y - TILE_SIZE;
        let max_x = max_x + TILE_SIZE;
        let max_y = max_y + TILE_SIZE;

        // Determine which tile keys overlap the padded view AABB
        let tile_min_ix = (min_x / TILE_SIZE).floor() as i32;
        let tile_min_iy = (min_y / TILE_SIZE).floor() as i32;
        let tile_max_ix = (max_x / TILE_SIZE).floor() as i32;
        let tile_max_iy = (max_y / TILE_SIZE).floor() as i32;

        let mut visible_set: HashSet<TileKey> = HashSet::new();
        for iy in tile_min_iy..=tile_max_iy {
            for ix in tile_min_ix..=tile_max_ix {
                let key = TileKey { ix, iy };
                if self.tile_map.tiles.contains_key(&key) {
                    visible_set.insert(key);
                }
            }
        }

        // Find visible tiles not yet in the atlas
        let mut to_load: Vec<TileKey> = Vec::new();
        for &key in &visible_set {
            if !self.tile_to_slot.contains_key(&key) {
                to_load.push(key);
            }
        }

        if to_load.is_empty() {
            return;
        }

        // Evict non-visible tiles if we need more free slots
        if self.free_slots.len() < to_load.len() {
            // Collect currently loaded but non-visible tiles
            let mut evict_candidates: Vec<TileKey> = self
                .tile_to_slot
                .keys()
                .filter(|k| !visible_set.contains(k))
                .cloned()
                .collect();

            let needed = to_load.len() - self.free_slots.len();
            // Evict as many as needed
            for key in evict_candidates.drain(..needed.min(evict_candidates.len())) {
                if let Some(slot) = self.tile_to_slot.remove(&key) {
                    self.slot_to_tile[slot as usize] = None;
                    self.free_slots.push(slot);
                }
            }
        }

        // Assign slots to newly visible tiles
        for key in to_load {
            if let Some(slot) = self.free_slots.pop() {
                self.slot_to_tile[slot as usize] = Some(key);
                self.tile_to_slot.insert(key, slot);
                self.tiles_changed = true;

                // Check if tile is in the disk cache
                let in_cache = self
                    .cache
                    .as_ref()
                    .map(|c| c.has_tile(key))
                    .unwrap_or(false);

                if in_cache {
                    // Submit async load from disk
                    if let Some(lq) = self.load_queue.as_mut() {
                        if lq.request(key, slot) {
                            self.loading_tiles.insert(key);
                        } else {
                            // Queue full, fall back to GPU generation
                            self.dirty_tiles.insert(key);
                        }
                    } else {
                        self.dirty_tiles.insert(key);
                    }
                } else {
                    // Not cached — GPU-generate
                    self.dirty_tiles.insert(key);
                }
            }
            // If no free slot available, this tile won't be rendered — atlas is full
        }
    }

    /// Clear all atlas slot assignments (call when roads are rebuilt).
    pub fn clear_slots(&mut self) {
        let total_slots = self.atlas_tiles_dim * self.atlas_tiles_dim;
        self.slot_to_tile.fill(None);
        self.tile_to_slot.clear();
        self.free_slots.clear();
        self.free_slots.extend((0..total_slots).rev());
        self.dirty_tiles.clear();
        self.tiles_changed = true;
        // Keep atlas_layout as-is so the next transition preserves existing
        // pixel data; stale tiles will be overwritten once dispatched.
    }

    /// Upload tile map data to GPU and update descriptors for segment/tile buffers.
    /// `segment_buffer` is the SSBO containing GpuSegment data from GpuRoadData.
    pub fn upload_tile_data(
        &mut self,
        device: &Device,
        allocator: &vma::Allocator,
        segment_buffer: vk::Buffer,
        segment_buffer_size: u64,
    ) -> anyhow::Result<()> {
        // Clear atlas slot assignments since tile indices may have changed
        self.clear_slots();

        // Destroy old buffers
        if let Some(b) = self.tile_header_buffer.take() {
            b.destroy(allocator);
        }
        if let Some(b) = self.tile_segment_index_buffer.take() {
            b.destroy(allocator);
        }
        if let Some(b) = self.tile_road_index_buffer.take() {
            b.destroy(allocator);
        }

        if self.tile_map.headers.is_empty() {
            return Ok(());
        }

        let usage = vk::BufferUsageFlags::STORAGE_BUFFER;

        // Upload tile headers
        let header_data = bytemuck::cast_slice::<_, u8>(&self.tile_map.headers);
        let (header_buf, header_ptr) =
            GpuBuffer::new_mapped(allocator, header_data.len() as u64, usage)?;
        unsafe {
            std::ptr::copy_nonoverlapping(header_data.as_ptr(), header_ptr, header_data.len())
        };
        self.tile_header_buffer = Some(header_buf);

        // Upload segment indices (ensure at least 4 bytes for Vulkan)
        let seg_idx_data = if self.tile_map.segment_indices.is_empty() {
            vec![0u32]
        } else {
            self.tile_map.segment_indices.clone()
        };
        let seg_idx_bytes = bytemuck::cast_slice::<_, u8>(&seg_idx_data);
        let (seg_idx_buf, seg_idx_ptr) =
            GpuBuffer::new_mapped(allocator, seg_idx_bytes.len() as u64, usage)?;
        unsafe {
            std::ptr::copy_nonoverlapping(seg_idx_bytes.as_ptr(), seg_idx_ptr, seg_idx_bytes.len())
        };
        self.tile_segment_index_buffer = Some(seg_idx_buf);

        // Upload road indices
        let road_idx_data = if self.tile_map.road_indices.is_empty() {
            vec![0u32]
        } else {
            self.tile_map.road_indices.clone()
        };
        let road_idx_bytes = bytemuck::cast_slice::<_, u8>(&road_idx_data);
        let (road_idx_buf, road_idx_ptr) =
            GpuBuffer::new_mapped(allocator, road_idx_bytes.len() as u64, usage)?;
        unsafe {
            std::ptr::copy_nonoverlapping(
                road_idx_bytes.as_ptr(),
                road_idx_ptr,
                road_idx_bytes.len(),
            )
        };
        self.tile_road_index_buffer = Some(road_idx_buf);

        // Update descriptors (bindings 1-4)
        let seg_buf_info = [vk::DescriptorBufferInfo::builder()
            .buffer(segment_buffer)
            .offset(0)
            .range(segment_buffer_size)
            .build()];
        let header_buf_info = [vk::DescriptorBufferInfo::builder()
            .buffer(self.tile_header_buffer.as_ref().unwrap().buffer)
            .offset(0)
            .range(self.tile_header_buffer.as_ref().unwrap().size)
            .build()];
        let seg_idx_buf_info = [vk::DescriptorBufferInfo::builder()
            .buffer(self.tile_segment_index_buffer.as_ref().unwrap().buffer)
            .offset(0)
            .range(self.tile_segment_index_buffer.as_ref().unwrap().size)
            .build()];
        let road_idx_buf_info = [vk::DescriptorBufferInfo::builder()
            .buffer(self.tile_road_index_buffer.as_ref().unwrap().buffer)
            .offset(0)
            .range(self.tile_road_index_buffer.as_ref().unwrap().size)
            .build()];

        let writes = [
            vk::WriteDescriptorSet::builder()
                .dst_set(self.descriptor_set)
                .dst_binding(1)
                .dst_array_element(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&seg_buf_info)
                .build(),
            vk::WriteDescriptorSet::builder()
                .dst_set(self.descriptor_set)
                .dst_binding(2)
                .dst_array_element(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&header_buf_info)
                .build(),
            vk::WriteDescriptorSet::builder()
                .dst_set(self.descriptor_set)
                .dst_binding(3)
                .dst_array_element(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&seg_idx_buf_info)
                .build(),
            vk::WriteDescriptorSet::builder()
                .dst_set(self.descriptor_set)
                .dst_binding(4)
                .dst_array_element(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&road_idx_buf_info)
                .build(),
        ];

        unsafe {
            device.update_descriptor_sets(&writes, &[] as &[vk::CopyDescriptorSet]);
        }

        Ok(())
    }

    /// Record compute dispatches for dirty tiles. Call this during command
    /// buffer recording. The atlas image must be in GENERAL layout.
    ///
    /// When `flush_all` is true every dirty tile is dispatched in one go
    /// (used after a road rebuild). Otherwise at most `MAX_DIRTY_DISPATCH`
    /// tiles are processed to spread streaming cost across frames.
    pub fn dispatch_dirty_tiles(
        &mut self,
        device: &Device,
        cmd: vk::CommandBuffer,
        flush_all: bool,
    ) {
        if self.dirty_tiles.is_empty() {
            return;
        }

        unsafe {
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.pipeline);
            device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline_layout,
                0,
                &[self.descriptor_set],
                &[],
            );
        }

        let dirty: Vec<TileKey> = if flush_all {
            self.dirty_tiles.iter().copied().collect()
        } else {
            self.dirty_tiles
                .iter()
                .copied()
                .take(MAX_DIRTY_DISPATCH)
                .collect()
        };
        for key in &dirty {
            self.dirty_tiles.remove(key);
        }
        for key in &dirty {
            let info = match self.tile_map.tiles.get(key) {
                Some(info) => info,
                None => continue,
            };
            let slot = match self.tile_to_slot.get(key) {
                Some(&s) => s,
                None => continue,
            };

            // SDF atlas texel offset from slot index (includes border)
            let slot_x = slot % self.atlas_tiles_dim;
            let slot_y = slot / self.atlas_tiles_dim;
            let atlas_offset_x = slot_x * TILE_SLOT_SIZE;
            let atlas_offset_y = slot_y * TILE_SLOT_SIZE;

            // Road ID atlas texel offset
            let road_id_atlas_offset_x = slot_x * ROAD_ID_RESOLUTION;
            let road_id_atlas_offset_y = slot_y * ROAD_ID_RESOLUTION;

            let (wx, wy) = key.world_origin();

            let pc = SdfPushConstants {
                atlas_offset_x,
                atlas_offset_y,
                tile_world_x: wx,
                tile_world_y: wy,
                tile_size: TILE_SIZE,
                tile_resolution: TILE_RESOLUTION,
                tile_index: info.header_index,
                road_id_atlas_offset_x,
                road_id_atlas_offset_y,
                road_id_resolution: ROAD_ID_RESOLUTION,
                tile_border: TILE_BORDER,
                _pad: 0,
            };

            unsafe {
                device.cmd_push_constants(
                    cmd,
                    self.pipeline_layout,
                    vk::ShaderStageFlags::COMPUTE,
                    0,
                    bytemuck::bytes_of(&pc),
                );

                // Dispatch: ceil(TILE_SLOT_SIZE/16)^2 workgroups per tile
                let wg = (TILE_SLOT_SIZE + 15) / 16;
                device.cmd_dispatch(cmd, wg, wg, 1);
            }
        }
    }

    /// Check if there are dirty tiles that need dispatching.
    pub fn has_dirty_tiles(&self) -> bool {
        !self.dirty_tiles.is_empty()
    }

    /// Check if the cache load queue is active and may have pending uploads.
    pub fn load_queue_active(&self) -> bool {
        self.load_queue.is_some() && !self.loading_tiles.is_empty()
    }

    /// Invalidate tiles in the disk cache whose road data has changed.
    /// Call this after `tile_map.rebuild()` with the returned set of changed keys.
    pub fn invalidate_cached_tiles(&mut self, changed: &HashSet<TileKey>) {
        if changed.is_empty() {
            return;
        }
        if let Some(cache) = &mut self.cache {
            for &key in changed {
                if let Err(e) = cache.invalidate_tile(key) {
                    log::warn!("failed to invalidate cache tile {:?}: {}", key, e);
                }
            }
            log::info!("Invalidated {} tile(s) in disk cache", changed.len());
        }
    }

    /// Open a disk cache file and spawn the background I/O thread.
    pub fn open_cache(
        &mut self,
        cache_path: &Path,
        allocator: &vma::Allocator,
    ) -> anyhow::Result<()> {
        let cache = SdfCacheFile::open(cache_path)
            .map_err(|e| anyhow::anyhow!("failed to open tile cache: {}", e))?;
        let load_queue = TileLoadQueue::new(cache_path)
            .map_err(|e| anyhow::anyhow!("failed to start tile I/O thread: {}", e))?;

        // Create a staging buffer large enough for one tile (SDF + road_id)
        let sdf_bytes = (TILE_SLOT_SIZE * TILE_SLOT_SIZE * 4) as u64; // RG16F
        let rid_bytes = (ROAD_ID_RESOLUTION * ROAD_ID_RESOLUTION * 2) as u64; // R16_SFLOAT
        let staging_size = sdf_bytes + rid_bytes;
        let (staging, ptr) = GpuBuffer::new_mapped(
            allocator,
            staging_size * 16, // room for 16 uploads per frame
            vk::BufferUsageFlags::TRANSFER_SRC,
        )?;

        self.cache = Some(cache);
        self.load_queue = Some(load_queue);
        self.staging_buffer = Some(staging);
        self.staging_ptr = ptr;
        Ok(())
    }

    /// Process completed tile loads from the I/O thread and upload to GPU.
    /// Must be called during command buffer recording with both atlas images in
    /// GENERAL layout. Returns the number of tiles uploaded.
    pub fn upload_cached_tiles(&mut self, device: &Device, cmd: vk::CommandBuffer) -> u32 {
        let load_queue = match self.load_queue.as_mut() {
            Some(q) => q,
            None => return 0,
        };

        let results = load_queue.drain_completed(16);
        if results.is_empty() {
            return 0;
        }

        let sdf_tile_bytes = (TILE_SLOT_SIZE * TILE_SLOT_SIZE * 4) as usize;
        let rid_tile_bytes = (ROAD_ID_RESOLUTION * ROAD_ID_RESOLUTION * 2) as usize;
        let per_tile = sdf_tile_bytes + rid_tile_bytes;
        let mut count = 0u32;

        for (i, result) in results.into_iter().enumerate() {
            self.loading_tiles.remove(&result.key);

            // Verify the tile still occupies the expected slot
            if self.tile_to_slot.get(&result.key) != Some(&result.slot) {
                continue;
            }

            // Copy pixel data to staging buffer
            let staging_offset = i * per_tile;
            unsafe {
                let dst = self.staging_ptr.add(staging_offset);
                std::ptr::copy_nonoverlapping(result.data.sdf_pixels.as_ptr(), dst, sdf_tile_bytes);
                std::ptr::copy_nonoverlapping(
                    result.data.road_id_pixels.as_ptr(),
                    dst.add(sdf_tile_bytes),
                    rid_tile_bytes,
                );
            }

            let slot_x = result.slot % self.atlas_tiles_dim;
            let slot_y = result.slot / self.atlas_tiles_dim;

            // Copy SDF data: staging → sdf_atlas
            let sdf_region = vk::BufferImageCopy::builder()
                .buffer_offset(staging_offset as u64)
                .buffer_row_length(0)
                .buffer_image_height(0)
                .image_subresource(
                    vk::ImageSubresourceLayers::builder()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .mip_level(0)
                        .base_array_layer(0)
                        .layer_count(1)
                        .build(),
                )
                .image_offset(vk::Offset3D {
                    x: (slot_x * TILE_SLOT_SIZE) as i32,
                    y: (slot_y * TILE_SLOT_SIZE) as i32,
                    z: 0,
                })
                .image_extent(vk::Extent3D {
                    width: TILE_SLOT_SIZE,
                    height: TILE_SLOT_SIZE,
                    depth: 1,
                })
                .build();

            unsafe {
                device.cmd_copy_buffer_to_image(
                    cmd,
                    self.staging_buffer.as_ref().unwrap().buffer,
                    self.sdf_atlas.image,
                    vk::ImageLayout::GENERAL,
                    &[sdf_region],
                );
            }

            // Copy road_id data: staging → road_id_atlas
            let rid_region = vk::BufferImageCopy::builder()
                .buffer_offset((staging_offset + sdf_tile_bytes) as u64)
                .buffer_row_length(0)
                .buffer_image_height(0)
                .image_subresource(
                    vk::ImageSubresourceLayers::builder()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .mip_level(0)
                        .base_array_layer(0)
                        .layer_count(1)
                        .build(),
                )
                .image_offset(vk::Offset3D {
                    x: (slot_x * ROAD_ID_RESOLUTION) as i32,
                    y: (slot_y * ROAD_ID_RESOLUTION) as i32,
                    z: 0,
                })
                .image_extent(vk::Extent3D {
                    width: ROAD_ID_RESOLUTION,
                    height: ROAD_ID_RESOLUTION,
                    depth: 1,
                })
                .build();

            unsafe {
                device.cmd_copy_buffer_to_image(
                    cmd,
                    self.staging_buffer.as_ref().unwrap().buffer,
                    self.road_id_atlas.image,
                    vk::ImageLayout::GENERAL,
                    &[rid_region],
                );
            }

            self.tiles_changed = true;
            count += 1;
        }
        count
    }

    /// Generate all tiles with road segments and write them to a disk cache.
    ///
    /// Batches tiles into atlas-sized groups (up to `atlas_tiles_dim²` per batch),
    /// dispatches the SDF compute shader, reads the result back to CPU, and
    /// compresses each tile into the cache file.
    ///
    /// Both atlas images are left in UNDEFINED layout after this call.
    /// Returns the total number of tiles written.
    pub fn pregenerate_to_cache(
        &mut self,
        device: &Arc<Device>,
        allocator: &vma::Allocator,
        queue: vk::Queue,
        queue_family: u32,
        cache_path: &Path,
        world_origin: (f32, f32),
        world_size: (f32, f32),
    ) -> anyhow::Result<u32> {
        use crate::tile_cache::SdfCacheFile;

        let mut cache = SdfCacheFile::create(
            cache_path,
            world_origin,
            world_size,
            TILE_SIZE,
            TILE_SLOT_SIZE,
            ROAD_ID_RESOLUTION,
        )
        .map_err(|e| anyhow::anyhow!("failed to create tile cache: {}", e))?;

        let all_keys: Vec<TileKey> = self.tile_map.tiles.keys().cloned().collect();
        if all_keys.is_empty() {
            log::info!("No tiles to pregenerate");
            return Ok(0);
        }

        let total_slots = (self.atlas_tiles_dim * self.atlas_tiles_dim) as usize;

        // Readback buffers for the full atlas
        let sdf_atlas_px = (self.atlas_tiles_dim * TILE_SLOT_SIZE) as usize;
        let sdf_readback_bytes = (sdf_atlas_px * sdf_atlas_px * 4) as u64;
        let (sdf_rb, sdf_rb_ptr) = GpuBuffer::new_mapped(
            allocator,
            sdf_readback_bytes,
            vk::BufferUsageFlags::TRANSFER_DST,
        )?;

        let rid_atlas_px = (self.atlas_tiles_dim * ROAD_ID_RESOLUTION) as usize;
        let rid_readback_bytes = (rid_atlas_px * rid_atlas_px * 2) as u64;
        let (rid_rb, rid_rb_ptr) = GpuBuffer::new_mapped(
            allocator,
            rid_readback_bytes,
            vk::BufferUsageFlags::TRANSFER_DST,
        )?;

        // One-shot command pool + fence
        let pool_info = vk::CommandPoolCreateInfo::builder()
            .queue_family_index(queue_family)
            .flags(vk::CommandPoolCreateFlags::TRANSIENT);
        let cmd_pool = unsafe { device.create_command_pool(&pool_info, None)? };
        let alloc_info = vk::CommandBufferAllocateInfo::builder()
            .command_pool(cmd_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let cmd = unsafe { device.allocate_command_buffers(&alloc_info)?[0] };
        let fence_info = vk::FenceCreateInfo::default();
        let fence = unsafe { device.create_fence(&fence_info, None)? };

        let mut total_written = 0u32;
        let num_batches = (all_keys.len() + total_slots - 1) / total_slots;

        for (batch_idx, chunk) in all_keys.chunks(total_slots).enumerate() {
            log::info!(
                "Pregenerating batch {}/{} ({} tiles)",
                batch_idx + 1,
                num_batches,
                chunk.len()
            );

            // Assign tiles to atlas slots
            self.clear_slots();
            for (i, &key) in chunk.iter().enumerate() {
                let slot = i as u32;
                self.slot_to_tile[slot as usize] = Some(key);
                self.tile_to_slot.insert(key, slot);
                self.dirty_tiles.insert(key);
            }

            // Record command buffer
            unsafe {
                device.reset_command_pool(cmd_pool, vk::CommandPoolResetFlags::empty())?;
                device.begin_command_buffer(
                    cmd,
                    &vk::CommandBufferBeginInfo::builder()
                        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                )?;
            }

            crate::transition_image(
                device,
                cmd,
                self.sdf_atlas.image,
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::GENERAL,
            );
            crate::transition_image(
                device,
                cmd,
                self.road_id_atlas.image,
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::GENERAL,
            );

            self.dispatch_dirty_tiles(device, cmd, true);

            // Barrier: compute writes → transfer reads
            unsafe {
                let barrier = vk::MemoryBarrier2::builder()
                    .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                    .src_access_mask(vk::AccessFlags2::SHADER_WRITE)
                    .dst_stage_mask(vk::PipelineStageFlags2::ALL_TRANSFER)
                    .dst_access_mask(vk::AccessFlags2::TRANSFER_READ);
                let dep =
                    vk::DependencyInfo::builder().memory_barriers(std::slice::from_ref(&barrier));
                device.cmd_pipeline_barrier2(cmd, &dep);
            }

            crate::transition_image(
                device,
                cmd,
                self.sdf_atlas.image,
                vk::ImageLayout::GENERAL,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            );
            crate::transition_image(
                device,
                cmd,
                self.road_id_atlas.image,
                vk::ImageLayout::GENERAL,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            );

            // Copy atlas images → readback buffers
            let sdf_copy = vk::BufferImageCopy::builder()
                .buffer_offset(0)
                .buffer_row_length(0)
                .buffer_image_height(0)
                .image_subresource(
                    vk::ImageSubresourceLayers::builder()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .mip_level(0)
                        .base_array_layer(0)
                        .layer_count(1)
                        .build(),
                )
                .image_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
                .image_extent(vk::Extent3D {
                    width: sdf_atlas_px as u32,
                    height: sdf_atlas_px as u32,
                    depth: 1,
                })
                .build();
            unsafe {
                device.cmd_copy_image_to_buffer(
                    cmd,
                    self.sdf_atlas.image,
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    sdf_rb.buffer,
                    &[sdf_copy],
                );
            }

            let rid_copy = vk::BufferImageCopy::builder()
                .buffer_offset(0)
                .buffer_row_length(0)
                .buffer_image_height(0)
                .image_subresource(
                    vk::ImageSubresourceLayers::builder()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .mip_level(0)
                        .base_array_layer(0)
                        .layer_count(1)
                        .build(),
                )
                .image_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
                .image_extent(vk::Extent3D {
                    width: rid_atlas_px as u32,
                    height: rid_atlas_px as u32,
                    depth: 1,
                })
                .build();
            unsafe {
                device.cmd_copy_image_to_buffer(
                    cmd,
                    self.road_id_atlas.image,
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    rid_rb.buffer,
                    &[rid_copy],
                );
            }

            // Submit and wait
            unsafe {
                device.end_command_buffer(cmd)?;
                let cmd_info = vk::CommandBufferSubmitInfo::builder()
                    .command_buffer(cmd)
                    .build();
                let submit = vk::SubmitInfo2::builder()
                    .command_buffer_infos(std::slice::from_ref(&cmd_info));
                device.queue_submit2(queue, std::slice::from_ref(&submit), fence)?;
                device.wait_for_fences(&[fence], true, u64::MAX)?;
                device.reset_fences(&[fence])?;
            }

            // Extract per-tile pixel data and write to cache
            let sdf_bpp = 4usize; // RG16F = 4 bytes/pixel
            let rid_bpp = 2usize; // R16_SFLOAT = 2 bytes/pixel
            let sdf_row_pitch = sdf_atlas_px * sdf_bpp;
            let rid_row_pitch = rid_atlas_px * rid_bpp;

            for (i, &key) in chunk.iter().enumerate() {
                let slot = i as u32;
                let slot_x = slot % self.atlas_tiles_dim;
                let slot_y = slot / self.atlas_tiles_dim;

                // Extract SDF tile (including border)
                let sdf_tile_row = (TILE_SLOT_SIZE as usize) * sdf_bpp;
                let mut sdf_data = vec![0u8; (TILE_SLOT_SIZE * TILE_SLOT_SIZE) as usize * sdf_bpp];
                let sdf_origin_x = (slot_x * TILE_SLOT_SIZE) as usize;
                let sdf_origin_y = (slot_y * TILE_SLOT_SIZE) as usize;
                for row in 0..TILE_SLOT_SIZE as usize {
                    let src = (sdf_origin_y + row) * sdf_row_pitch + sdf_origin_x * sdf_bpp;
                    let dst = row * sdf_tile_row;
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            sdf_rb_ptr.add(src),
                            sdf_data.as_mut_ptr().add(dst),
                            sdf_tile_row,
                        );
                    }
                }

                // Extract road_id tile
                let rid_tile_row = (ROAD_ID_RESOLUTION as usize) * rid_bpp;
                let mut rid_data =
                    vec![0u8; (ROAD_ID_RESOLUTION * ROAD_ID_RESOLUTION) as usize * rid_bpp];
                let rid_origin_x = (slot_x * ROAD_ID_RESOLUTION) as usize;
                let rid_origin_y = (slot_y * ROAD_ID_RESOLUTION) as usize;
                for row in 0..ROAD_ID_RESOLUTION as usize {
                    let src = (rid_origin_y + row) * rid_row_pitch + rid_origin_x * rid_bpp;
                    let dst = row * rid_tile_row;
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            rid_rb_ptr.add(src),
                            rid_data.as_mut_ptr().add(dst),
                            rid_tile_row,
                        );
                    }
                }

                cache
                    .write_tile(key, &sdf_data, &rid_data)
                    .map_err(|e| anyhow::anyhow!("cache write error: {}", e))?;
                total_written += 1;
            }
        }

        // Cleanup
        self.clear_slots();
        unsafe {
            device.destroy_fence(fence, None);
            device.destroy_command_pool(cmd_pool, None);
        }
        sdf_rb.destroy(allocator);
        rid_rb.destroy(allocator);

        log::info!("Pregenerated {} tiles to {:?}", total_written, cache_path);
        Ok(total_written)
    }

    /// Destroy all GPU resources.
    pub fn destroy(&mut self, device: &Device, allocator: &vma::Allocator) {
        // Shut down I/O thread first
        if let Some(mut lq) = self.load_queue.take() {
            lq.shutdown();
        }
        self.cache = None;

        if let Some(b) = self.staging_buffer.take() {
            b.destroy(allocator);
        }
        if let Some(b) = self.tile_header_buffer.take() {
            b.destroy(allocator);
        }
        if let Some(b) = self.tile_segment_index_buffer.take() {
            b.destroy(allocator);
        }
        if let Some(b) = self.tile_road_index_buffer.take() {
            b.destroy(allocator);
        }
        self.sdf_atlas.destroy(device, allocator);
        self.road_id_atlas.destroy(device, allocator);
        unsafe {
            device.destroy_pipeline(self.pipeline, None);
            device.destroy_pipeline_layout(self.pipeline_layout, None);
            device.destroy_descriptor_pool(self.descriptor_pool, None);
            device.destroy_descriptor_set_layout(self.descriptor_set_layout, None);
        }
    }
}

// ---------------------------------------------------------------------------
// Helper: compute AABBs for all segments in a road network
// ---------------------------------------------------------------------------

/// Compute world-space AABBs for all segments across all roads.
/// Uses a conservative estimate by sampling the segment at multiple points
/// and adding a margin for road width.
pub fn compute_segment_aabbs(roads: &[crate::sdf::RoadSegmentInfo]) -> Vec<SegmentAABB> {
    let margin = 20.0; // generous margin for road width + shoulder

    let mut result = Vec::new();
    for info in roads {
        let mut min_x = f32::MAX;
        let mut min_y = f32::MAX;
        let mut max_x = f32::MIN;
        let mut max_y = f32::MIN;

        // Sample the segment to find bounds
        let n_samples = ((info.length / 2.0).ceil() as usize).max(2);
        for i in 0..=n_samples {
            let s = info.length * i as f32 / n_samples as f32;

            // Evaluate segment in local frame, then rotate+translate to world
            let (lx, ly) = eval_segment_xy(info.seg_type, s, info.k_start, info.k_end, info.length);
            let c = info.heading.cos();
            let sn = info.heading.sin();
            let wx = info.origin_x + c * lx - sn * ly;
            let wy = info.origin_y + sn * lx + c * ly;

            min_x = min_x.min(wx);
            min_y = min_y.min(wy);
            max_x = max_x.max(wx);
            max_y = max_y.max(wy);
        }

        result.push(SegmentAABB {
            segment_index: info.segment_index,
            road_index: info.road_index,
            aabb_min: (min_x - margin, min_y - margin),
            aabb_max: (max_x + margin, max_y + margin),
        });
    }
    result
}

/// Segment info needed for AABB computation (passed from the app).
pub struct RoadSegmentInfo {
    pub segment_index: u32,
    pub road_index: u32,
    pub seg_type: u32,
    pub origin_x: f32,
    pub origin_y: f32,
    pub heading: f32,
    pub length: f32,
    pub k_start: f32,
    pub k_end: f32,
}

/// Evaluate segment position in local frame (no heading rotation).
fn eval_segment_xy(seg_type: u32, s: f32, k_start: f32, k_end: f32, length: f32) -> (f32, f32) {
    match seg_type {
        0 => (s, 0.0), // Line
        1 => {
            // Arc
            let curvature = k_start;
            if curvature.abs() < 1e-9 {
                return (s, 0.0);
            }
            let r = 1.0 / curvature;
            let theta = s * curvature;
            (r * theta.sin(), r * (1.0 - theta.cos()))
        }
        _ => {
            // Spiral — Simpson's rule (matching primitives.rs)
            let dk = if length > 1e-9 {
                (k_end - k_start) / length
            } else {
                0.0
            };
            let n = 16usize;
            let h = s / n as f32;
            let mut sum_x = 0.0f32;
            let mut sum_y = 0.0f32;
            for i in 0..=n {
                let t = i as f32 * h;
                let theta = k_start * t + 0.5 * dk * t * t;
                let w = if i == 0 || i == n {
                    1.0
                } else if i % 2 == 1 {
                    4.0
                } else {
                    2.0
                };
                sum_x += w * theta.cos();
                sum_y += w * theta.sin();
            }
            (sum_x * h / 3.0, sum_y * h / 3.0)
        }
    }
}

// ---------------------------------------------------------------------------
// DraftSdfTile — dedicated single-tile SDF for draft road preview
// ---------------------------------------------------------------------------

/// Maximum texel resolution for the draft SDF tile (capped for performance).
const DRAFT_MAX_RESOLUTION: u32 = 256;
/// Target texels per world meter for the draft tile.
const DRAFT_TEXELS_PER_METER: f32 = 2.0;
/// Maximum number of road segments the draft tile can handle.
const DRAFT_MAX_SEGMENTS: usize = 128;

/// A dedicated small SDF texture for rendering a draft (in-progress) road.
/// All GPU resources are pre-allocated at max capacity to avoid per-frame
/// reallocation that would conflict with in-flight command buffers.
pub struct DraftSdfTile {
    /// SDF texture (RG16F — signed_dist + s_coord), always DRAFT_MAX_RESOLUTION².
    pub sdf_image: GpuImage,
    /// Dummy 1×1 road-ID image (R16F) — satisfies the compute shader binding.
    road_id_dummy: GpuImage,

    // Compute pipeline
    pipeline: vk::Pipeline,
    pipeline_layout: vk::PipelineLayout,
    descriptor_set_layout: vk::DescriptorSetLayout,
    descriptor_pool: vk::DescriptorPool,
    descriptor_set: vk::DescriptorSet,

    // Pre-allocated GPU buffers (persistently mapped)
    segment_buffer: GpuBuffer,
    segment_ptr: *mut u8,
    tile_header_buffer: GpuBuffer,
    tile_header_ptr: *mut u8,
    segment_index_buffer: GpuBuffer,
    segment_index_ptr: *mut u8,
    road_index_buffer: GpuBuffer,
    road_index_ptr: *mut u8,

    /// World-space origin (bottom-left) of the tile.
    pub world_origin: [f32; 2],
    /// World-space size of the tile.
    pub world_size: [f32; 2],
    /// Current image layout.
    pub layout: vk::ImageLayout,
}

impl DraftSdfTile {
    /// Create a new draft SDF tile with all resources pre-allocated at max capacity.
    pub fn new(
        device: &Arc<Device>,
        allocator: &vma::Allocator,
        spirv: &[u32],
    ) -> anyhow::Result<Self> {
        let resolution = DRAFT_MAX_RESOLUTION;

        let sdf_image = GpuImage::new_storage_2d(
            device,
            allocator,
            resolution,
            resolution,
            vk::Format::R16G16_SFLOAT,
        )?;

        let road_id_dummy = GpuImage::new_storage_2d(
            device,
            allocator,
            1,
            1,
            vk::Format::R16_SFLOAT,
        )?;

        // Descriptor set layout — same bindings as main SDF system
        let bindings = [
            vk::DescriptorSetLayoutBinding::builder()
                .binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
                .build(),
            vk::DescriptorSetLayoutBinding::builder()
                .binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
                .build(),
            vk::DescriptorSetLayoutBinding::builder()
                .binding(2)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
                .build(),
            vk::DescriptorSetLayoutBinding::builder()
                .binding(3)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
                .build(),
            vk::DescriptorSetLayoutBinding::builder()
                .binding(4)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
                .build(),
            vk::DescriptorSetLayoutBinding::builder()
                .binding(5)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
                .build(),
        ];

        let descriptor_set_layout =
            crate::pipeline::create_descriptor_set_layout(device, &bindings)?;

        let pool_sizes = [
            vk::DescriptorPoolSize::builder()
                .type_(vk::DescriptorType::STORAGE_IMAGE)
                .descriptor_count(2)
                .build(),
            vk::DescriptorPoolSize::builder()
                .type_(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(4)
                .build(),
        ];
        let pool_info = vk::DescriptorPoolCreateInfo::builder()
            .max_sets(1)
            .pool_sizes(&pool_sizes);
        let descriptor_pool = unsafe { device.create_descriptor_pool(&pool_info, None)? };

        let descriptor_set =
            crate::pipeline::allocate_descriptor_set(device, descriptor_pool, descriptor_set_layout)?;

        let push_constant_ranges = [vk::PushConstantRange::builder()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .offset(0)
            .size(std::mem::size_of::<SdfPushConstants>() as u32)
            .build()];

        let (pipeline, pipeline_layout) = crate::pipeline::create_compute_pipeline(
            device,
            spirv,
            "main",
            &[descriptor_set_layout],
            &push_constant_ranges,
        )?;

        // Write SDF image descriptor (binding 0) and road_id dummy (binding 5)
        let sdf_image_info = [vk::DescriptorImageInfo::builder()
            .image_view(sdf_image.view)
            .image_layout(vk::ImageLayout::GENERAL)
            .build()];
        let road_id_image_info = [vk::DescriptorImageInfo::builder()
            .image_view(road_id_dummy.view)
            .image_layout(vk::ImageLayout::GENERAL)
            .build()];

        // Pre-allocate buffers at max capacity (persistently mapped)
        let usage = vk::BufferUsageFlags::STORAGE_BUFFER;
        let seg_size = (DRAFT_MAX_SEGMENTS * std::mem::size_of::<gpu_shared::GpuSegment>()) as u64;
        let hdr_size = std::mem::size_of::<TileHeader>() as u64;
        let idx_size = (DRAFT_MAX_SEGMENTS * std::mem::size_of::<u32>()) as u64;
        let rid_size = (DRAFT_MAX_SEGMENTS * std::mem::size_of::<u32>()) as u64;

        let (segment_buffer, segment_ptr) = GpuBuffer::new_mapped(allocator, seg_size, usage)?;
        let (tile_header_buffer, tile_header_ptr) = GpuBuffer::new_mapped(allocator, hdr_size, usage)?;
        let (segment_index_buffer, segment_index_ptr) = GpuBuffer::new_mapped(allocator, idx_size, usage)?;
        let (road_index_buffer, road_index_ptr) = GpuBuffer::new_mapped(allocator, rid_size, usage)?;

        // Write all descriptor bindings once — they never change after this
        let seg_buf_info = [vk::DescriptorBufferInfo::builder()
            .buffer(segment_buffer.buffer)
            .offset(0)
            .range(seg_size)
            .build()];
        let hdr_buf_info = [vk::DescriptorBufferInfo::builder()
            .buffer(tile_header_buffer.buffer)
            .offset(0)
            .range(hdr_size)
            .build()];
        let idx_buf_info = [vk::DescriptorBufferInfo::builder()
            .buffer(segment_index_buffer.buffer)
            .offset(0)
            .range(idx_size)
            .build()];
        let rid_buf_info = [vk::DescriptorBufferInfo::builder()
            .buffer(road_index_buffer.buffer)
            .offset(0)
            .range(rid_size)
            .build()];

        let writes = [
            vk::WriteDescriptorSet::builder()
                .dst_set(descriptor_set)
                .dst_binding(0)
                .dst_array_element(0)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .image_info(&sdf_image_info)
                .build(),
            vk::WriteDescriptorSet::builder()
                .dst_set(descriptor_set)
                .dst_binding(1)
                .dst_array_element(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&seg_buf_info)
                .build(),
            vk::WriteDescriptorSet::builder()
                .dst_set(descriptor_set)
                .dst_binding(2)
                .dst_array_element(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&hdr_buf_info)
                .build(),
            vk::WriteDescriptorSet::builder()
                .dst_set(descriptor_set)
                .dst_binding(3)
                .dst_array_element(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&idx_buf_info)
                .build(),
            vk::WriteDescriptorSet::builder()
                .dst_set(descriptor_set)
                .dst_binding(4)
                .dst_array_element(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&rid_buf_info)
                .build(),
            vk::WriteDescriptorSet::builder()
                .dst_set(descriptor_set)
                .dst_binding(5)
                .dst_array_element(0)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .image_info(&road_id_image_info)
                .build(),
        ];
        unsafe {
            device.update_descriptor_sets(&writes, &[] as &[vk::CopyDescriptorSet]);
        }

        Ok(Self {
            sdf_image,
            road_id_dummy,
            pipeline,
            pipeline_layout,
            descriptor_set_layout,
            descriptor_pool,
            descriptor_set,
            segment_buffer,
            segment_ptr,
            tile_header_buffer,
            tile_header_ptr,
            segment_index_buffer,
            segment_index_ptr,
            road_index_buffer,
            road_index_ptr,
            world_origin: [0.0; 2],
            world_size: [1.0; 2],
            layout: vk::ImageLayout::UNDEFINED,
        })
    }

    /// Upload segment data and dispatch the SDF compute shader for the draft road.
    /// The SDF image must be in GENERAL layout before calling this.
    pub fn update(
        &mut self,
        device: &Arc<Device>,
        cmd: vk::CommandBuffer,
        segments: &[RoadSegmentInfo],
    ) -> anyhow::Result<()> {
        if segments.is_empty() {
            return Ok(());
        }

        let seg_count = segments.len().min(DRAFT_MAX_SEGMENTS);
        let segments = &segments[..seg_count];

        // Compute AABB from segments
        let aabbs = compute_segment_aabbs(segments);
        let mut min_x = f32::MAX;
        let mut min_y = f32::MAX;
        let mut max_x = f32::MIN;
        let mut max_y = f32::MIN;
        for aabb in &aabbs {
            min_x = min_x.min(aabb.aabb_min.0);
            min_y = min_y.min(aabb.aabb_min.1);
            max_x = max_x.max(aabb.aabb_max.0);
            max_y = max_y.max(aabb.aabb_max.1);
        }

        let size_x = (max_x - min_x).max(1.0);
        let size_y = (max_y - min_y).max(1.0);
        let tile_size = size_x.max(size_y);

        self.world_origin = [min_x, min_y];
        self.world_size = [tile_size, tile_size];

        // Build GPU segment data and memcpy into pre-allocated buffers
        let gpu_segments: Vec<gpu_shared::GpuSegment> = segments
            .iter()
            .map(|s| gpu_shared::GpuSegment {
                segment_type: s.seg_type,
                s_start: 0.0,
                origin: [s.origin_x, s.origin_y],
                heading: s.heading,
                length: s.length,
                k_start: s.k_start,
                k_end: s.k_end,
            })
            .collect();

        let seg_data = bytemuck::cast_slice::<_, u8>(&gpu_segments);
        unsafe { std::ptr::copy_nonoverlapping(seg_data.as_ptr(), self.segment_ptr, seg_data.len()) };

        let header = TileHeader {
            offset: 0,
            count: seg_count as u32,
        };
        let header_data = bytemuck::bytes_of(&header);
        unsafe { std::ptr::copy_nonoverlapping(header_data.as_ptr(), self.tile_header_ptr, header_data.len()) };

        let indices: Vec<u32> = (0..seg_count as u32).collect();
        let idx_data = bytemuck::cast_slice::<_, u8>(&indices);
        unsafe { std::ptr::copy_nonoverlapping(idx_data.as_ptr(), self.segment_index_ptr, idx_data.len()) };

        let road_ids: Vec<u32> = vec![0; seg_count];
        let rid_data = bytemuck::cast_slice::<_, u8>(&road_ids);
        unsafe { std::ptr::copy_nonoverlapping(rid_data.as_ptr(), self.road_index_ptr, rid_data.len()) };

        // Dispatch compute — use actual needed resolution for workgroup count
        let needed_res = ((tile_size * DRAFT_TEXELS_PER_METER) as u32)
            .next_power_of_two()
            .clamp(64, DRAFT_MAX_RESOLUTION);

        let pc = SdfPushConstants {
            atlas_offset_x: 0,
            atlas_offset_y: 0,
            tile_world_x: min_x,
            tile_world_y: min_y,
            tile_size,
            tile_resolution: needed_res,
            tile_index: 0,
            road_id_atlas_offset_x: 0,
            road_id_atlas_offset_y: 0,
            road_id_resolution: 1,
            tile_border: 0,
            _pad: 0,
        };

        unsafe {
            // Transition road_id dummy to GENERAL (discards prior content)
            crate::core::transition_image(
                device,
                cmd,
                self.road_id_dummy.image,
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::GENERAL,
            );

            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.pipeline);
            device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline_layout,
                0,
                &[self.descriptor_set],
                &[],
            );
            device.cmd_push_constants(
                cmd,
                self.pipeline_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                bytemuck::bytes_of(&pc),
            );

            let wg = (needed_res + 15) / 16;
            device.cmd_dispatch(cmd, wg, wg, 1);
        }

        Ok(())
    }

    /// Destroy all GPU resources.
    pub fn destroy(&mut self, device: &Device, allocator: &vma::Allocator) {
        self.segment_buffer.destroy(allocator);
        self.tile_header_buffer.destroy(allocator);
        self.segment_index_buffer.destroy(allocator);
        self.road_index_buffer.destroy(allocator);
        self.sdf_image.destroy(device, allocator);
        self.road_id_dummy.destroy(device, allocator);
        unsafe {
            device.destroy_pipeline(self.pipeline, None);
            device.destroy_pipeline_layout(self.pipeline_layout, None);
            device.destroy_descriptor_pool(self.descriptor_pool, None);
            device.destroy_descriptor_set_layout(self.descriptor_set_layout, None);
        }
    }
}
