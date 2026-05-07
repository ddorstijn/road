use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use vulkanalia::prelude::v1_4::*;
use vulkanalia_bootstrap::Device;
use vulkanalia_vma as vma;

use crate::gpu_resources::{GpuBuffer, GpuImage};
use crate::pipeline::{
    allocate_descriptor_set, compile_wgsl, create_compute_pipeline, create_descriptor_set_layout,
};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// World-space size of each tile in meters.
pub const TILE_SIZE: f32 = 64.0;
/// Texels per tile dimension.
pub const TILE_RESOLUTION: u32 = 256;
/// Atlas dimension in tiles (16 × 16 = 256 tile slots).
const ATLAS_TILES: u32 = 16;
/// Atlas dimension in texels.
const ATLAS_SIZE: u32 = ATLAS_TILES * TILE_RESOLUTION; // 4096

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
    /// Atlas slot index (0..ATLAS_TILES*ATLAS_TILES).
    pub atlas_slot: u32,
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
    /// Free atlas slot stack.
    free_slots: Vec<u32>,
    /// Tile headers (one per active tile, indexed by header_index).
    pub headers: Vec<TileHeader>,
    /// Flat list of segment indices per tile (offsets stored in headers).
    pub segment_indices: Vec<u32>,
    /// Road index per entry in segment_indices (parallel array).
    pub road_indices: Vec<u32>,
    /// Set of tiles that need re-dispatch.
    pub dirty_tiles: HashSet<TileKey>,
}

impl TileMap {
    pub fn new() -> Self {
        let total_slots = (ATLAS_TILES * ATLAS_TILES) as u32;
        Self {
            tiles: HashMap::new(),
            free_slots: (0..total_slots).rev().collect(),
            headers: Vec::new(),
            segment_indices: Vec::new(),
            road_indices: Vec::new(),
            dirty_tiles: HashSet::new(),
        }
    }

    /// Rebuild tile data from road segments.
    /// `segments` is the flat array of (origin, heading, length, segment_index, road_index)
    /// tuples — we compute AABB overlap with each tile.
    pub fn rebuild(
        &mut self,
        segment_data: &[SegmentAABB],
    ) {
        // Clear old data
        let old_keys: Vec<TileKey> = self.tiles.keys().cloned().collect();
        for key in &old_keys {
            if let Some(info) = self.tiles.remove(key) {
                self.free_slots.push(info.atlas_slot);
            }
        }
        self.headers.clear();
        self.segment_indices.clear();
        self.road_indices.clear();
        self.dirty_tiles.clear();

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

        // Allocate atlas slots and build headers
        for (key, seg_list) in &tile_segments {
            let slot = match self.free_slots.pop() {
                Some(s) => s,
                None => {
                    // Atlas full — skip tile
                    continue;
                }
            };

            let header_index = self.headers.len() as u32;
            let offset = self.segment_indices.len() as u32;
            let count = seg_list.len() as u32;

            for &(seg_idx, road_idx) in seg_list {
                self.segment_indices.push(seg_idx);
                self.road_indices.push(road_idx);
            }

            self.headers.push(TileHeader { offset, count });
            self.tiles.insert(*key, TileInfo {
                atlas_slot: slot,
                header_index,
            });
            self.dirty_tiles.insert(*key);
        }
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

pub struct SdfTileManager {
    // Atlas image (RGBA32F, ATLAS_SIZE × ATLAS_SIZE)
    pub atlas: GpuImage,

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

    // Tile bookkeeping
    pub tile_map: TileMap,
}

impl SdfTileManager {
    /// Create the SDF tile manager. Compiles the shader, creates the atlas and pipeline.
    pub fn new(
        device: &Arc<Device>,
        allocator: &vma::Allocator,
        types_wgsl: &str,
        road_eval_wgsl: &str,
        sdf_gen_wgsl: &str,
    ) -> anyhow::Result<Self> {
        // Create atlas image (RGBA32F for signed_dist, s, road_id, unused)
        let atlas = GpuImage::new_storage_2d(
            device,
            allocator,
            ATLAS_SIZE,
            ATLAS_SIZE,
            vk::Format::R32G32B32A32_SFLOAT,
        )?;

        // Compose the full WGSL source by concatenating shared modules
        let full_source = format!("{}\n{}\n{}", types_wgsl, road_eval_wgsl, sdf_gen_wgsl);
        let spirv = compile_wgsl(&full_source)?;

        // Descriptor set layout:
        //   binding 0: storage image (atlas) — write
        //   binding 1: storage buffer (segments) — read
        //   binding 2: storage buffer (tile_headers) — read
        //   binding 3: storage buffer (tile_segment_indices) — read
        //   binding 4: storage buffer (road_indices) — read
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
        ];

        let descriptor_set_layout = create_descriptor_set_layout(device, &bindings)?;

        // Descriptor pool
        let pool_sizes = [
            vk::DescriptorPoolSize::builder()
                .type_(vk::DescriptorType::STORAGE_IMAGE)
                .descriptor_count(1)
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
            &spirv,
            &[descriptor_set_layout],
            &push_constant_ranges,
        )?;

        // Write atlas descriptor (binding 0) — the image is always the same
        let image_info = [vk::DescriptorImageInfo::builder()
            .image_view(atlas.view)
            .image_layout(vk::ImageLayout::GENERAL)
            .build()];
        let writes = [vk::WriteDescriptorSet::builder()
            .dst_set(descriptor_set)
            .dst_binding(0)
            .dst_array_element(0)
            .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
            .image_info(&image_info)
            .build()];
        unsafe {
            device.update_descriptor_sets(&writes, &[] as &[vk::CopyDescriptorSet]);
        }

        Ok(Self {
            atlas,
            pipeline,
            pipeline_layout,
            descriptor_set_layout,
            descriptor_pool,
            descriptor_set,
            tile_header_buffer: None,
            tile_segment_index_buffer: None,
            tile_road_index_buffer: None,
            tile_map: TileMap::new(),
        })
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

    /// Record compute dispatches for all dirty tiles. Call this during command
    /// buffer recording. The atlas image must be in GENERAL layout.
    pub fn dispatch_dirty_tiles(
        &mut self,
        device: &Device,
        cmd: vk::CommandBuffer,
    ) {
        if self.tile_map.dirty_tiles.is_empty() {
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

        let dirty: Vec<TileKey> = self.tile_map.dirty_tiles.drain().collect();
        for key in &dirty {
            let info = match self.tile_map.tiles.get(key) {
                Some(info) => info,
                None => continue,
            };

            // Compute atlas texel offset from slot index
            let slot = info.atlas_slot;
            let slot_x = slot % ATLAS_TILES;
            let slot_y = slot / ATLAS_TILES;
            let atlas_offset_x = slot_x * TILE_RESOLUTION;
            let atlas_offset_y = slot_y * TILE_RESOLUTION;

            let (wx, wy) = key.world_origin();

            let pc = SdfPushConstants {
                atlas_offset_x,
                atlas_offset_y,
                tile_world_x: wx,
                tile_world_y: wy,
                tile_size: TILE_SIZE,
                tile_resolution: TILE_RESOLUTION,
                tile_index: info.header_index,
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

                // Dispatch: (TILE_RESOLUTION/16)^2 workgroups per tile
                let wg = TILE_RESOLUTION / 16;
                device.cmd_dispatch(cmd, wg, wg, 1);
            }
        }
    }

    /// Check if there are dirty tiles that need dispatching.
    pub fn has_dirty_tiles(&self) -> bool {
        !self.tile_map.dirty_tiles.is_empty()
    }

    /// Destroy all GPU resources.
    pub fn destroy(&mut self, device: &Device, allocator: &vma::Allocator) {
        if let Some(b) = self.tile_header_buffer.take() {
            b.destroy(allocator);
        }
        if let Some(b) = self.tile_segment_index_buffer.take() {
            b.destroy(allocator);
        }
        if let Some(b) = self.tile_road_index_buffer.take() {
            b.destroy(allocator);
        }
        self.atlas.destroy(device, allocator);
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
pub fn compute_segment_aabbs(
    roads: &[crate::sdf::RoadSegmentInfo],
) -> Vec<SegmentAABB> {
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
