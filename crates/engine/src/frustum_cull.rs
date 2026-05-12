//! GPU frustum culling for cars and tiles.
//!
//! Dispatches compute shaders that test each car/tile against the camera's
//! world-space AABB and produce compacted indirect draw command buffers.

use vulkanalia::prelude::v1_4::*;
use vulkanalia_bootstrap::Device;
use vulkanalia_vma as vma;

use crate::camera::Camera2D;
use crate::gpu_resources::GpuBuffer;
use crate::pipeline::{write_storage_buffers, ComputePass};
use crate::sdf::{SdfTileManager, ROAD_ID_RESOLUTION, TILE_RESOLUTION, TILE_SIZE};

use gpu_shared::{DrawIndirectCommand, GpuTileInstance};

// ---------------------------------------------------------------------------
// Push constants — must match the shader
// ---------------------------------------------------------------------------

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct CarCullPushConstants {
    view_min_x: f32,
    view_min_y: f32,
    view_max_x: f32,
    view_max_y: f32,
    car_count: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct TileCullPushConstants {
    view_min_x: f32,
    view_min_y: f32,
    view_max_x: f32,
    view_max_y: f32,
    tile_count: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

// ---------------------------------------------------------------------------
// Compute camera world-space bounds (AABB for ortho projection)
// ---------------------------------------------------------------------------

/// Returns (min_x, min_y, max_x, max_y) in world space.
pub fn camera_world_bounds(camera: &Camera2D, aspect: f32) -> (f32, f32, f32, f32) {
    let half_h = 1.0 / camera.zoom;
    let half_w = half_h * aspect;
    (
        camera.position.x - half_w,
        camera.position.y - half_h,
        camera.position.x + half_w,
        camera.position.y + half_h,
    )
}

// ---------------------------------------------------------------------------
// CarCullPass — frustum culling for cars
// ---------------------------------------------------------------------------

pub struct CarCullPass {
    pass: ComputePass,
    /// Buffer holding the indirect draw command (DrawIndirectCommand = 16 bytes).
    pub draw_indirect_buf: GpuBuffer,
    /// Buffer holding visible car indices (u32 per visible car).
    pub visible_indices_buf: GpuBuffer,
    descriptors_valid: bool,
}

impl CarCullPass {
    pub fn new(
        device: &Device,
        allocator: &vma::Allocator,
        spirv: &[u32],
        max_cars: u32,
    ) -> anyhow::Result<Self> {
        // 9 bindings: 0-2 car SoA (road_id, s, lane), 3-6 road data, 7 draw_indirect, 8 visible_indices
        let pass = ComputePass::new(
            device,
            spirv,
            "frustum_cull::car_cull_main",
            9, // num_storage_buffers
            std::mem::size_of::<CarCullPushConstants>() as u32,
            1,
        )?;

        let draw_indirect_buf = GpuBuffer::new(
            allocator,
            std::mem::size_of::<DrawIndirectCommand>() as u64,
            vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::INDIRECT_BUFFER
                | vk::BufferUsageFlags::TRANSFER_DST,
        )?;

        let visible_indices_buf = GpuBuffer::new(
            allocator,
            (max_cars as u64) * 4,
            vk::BufferUsageFlags::STORAGE_BUFFER,
        )?;

        Ok(Self {
            pass,
            draw_indirect_buf,
            visible_indices_buf,
            descriptors_valid: false,
        })
    }

    /// Update descriptors. Call after car buffers or road buffers change.
    pub fn update_descriptors(
        &mut self,
        device: &Device,
        car_road_id: &GpuBuffer,
        car_s: &GpuBuffer,
        car_lane: &GpuBuffer,
        segment_buf: &GpuBuffer,
        road_buf: &GpuBuffer,
        lane_section_buf: &GpuBuffer,
        lane_buf: &GpuBuffer,
    ) {
        write_storage_buffers(
            device,
            self.pass.set(),
            0,
            &[
                car_road_id,
                car_s,
                car_lane,
                segment_buf,
                road_buf,
                lane_section_buf,
                lane_buf,
                &self.draw_indirect_buf,
                &self.visible_indices_buf,
            ],
        );
        self.descriptors_valid = true;
    }

    /// Record the cull dispatch. Resets the indirect draw command, then dispatches.
    pub fn dispatch(
        &self,
        device: &Device,
        cmd: vk::CommandBuffer,
        car_count: u32,
        camera: &Camera2D,
        aspect: f32,
    ) {
        if !self.descriptors_valid || car_count == 0 {
            return;
        }

        // Reset indirect draw command: vertex_count=6, instance_count=0, first_vertex=0, first_instance=0
        let reset_cmd = DrawIndirectCommand {
            vertex_count: 6,
            instance_count: 0,
            first_vertex: 0,
            first_instance: 0,
        };
        unsafe {
            device.cmd_update_buffer(
                cmd,
                self.draw_indirect_buf.buffer,
                0,
                bytemuck::bytes_of(&reset_cmd),
            );

            // Barrier: transfer write → compute read/write
            let barrier = vk::MemoryBarrier2::builder()
                .src_stage_mask(vk::PipelineStageFlags2::ALL_TRANSFER)
                .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                .dst_access_mask(vk::AccessFlags2::SHADER_READ | vk::AccessFlags2::SHADER_WRITE);
            let dep = vk::DependencyInfo::builder().memory_barriers(std::slice::from_ref(&barrier));
            device.cmd_pipeline_barrier2(cmd, &dep);
        }

        let (min_x, min_y, max_x, max_y) = camera_world_bounds(camera, aspect);
        let pc = CarCullPushConstants {
            view_min_x: min_x,
            view_min_y: min_y,
            view_max_x: max_x,
            view_max_y: max_y,
            car_count,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };

        unsafe {
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.pass.pipeline);
            device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                self.pass.pipeline_layout,
                0,
                &[self.pass.set()],
                &[],
            );
            device.cmd_push_constants(
                cmd,
                self.pass.pipeline_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                bytemuck::bytes_of(&pc),
            );
            device.cmd_dispatch(cmd, car_count.div_ceil(256), 1, 1);

            // Barrier: compute write → indirect draw read + vertex shader read
            let barrier = vk::MemoryBarrier2::builder()
                .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                .src_access_mask(vk::AccessFlags2::SHADER_WRITE)
                .dst_stage_mask(
                    vk::PipelineStageFlags2::DRAW_INDIRECT | vk::PipelineStageFlags2::VERTEX_SHADER,
                )
                .dst_access_mask(
                    vk::AccessFlags2::INDIRECT_COMMAND_READ | vk::AccessFlags2::SHADER_READ,
                );
            let dep = vk::DependencyInfo::builder().memory_barriers(std::slice::from_ref(&barrier));
            device.cmd_pipeline_barrier2(cmd, &dep);
        }
    }

    pub fn destroy(&mut self, device: &Device, allocator: &vma::Allocator) {
        self.pass.destroy(device);
        self.draw_indirect_buf.destroy(allocator);
        self.visible_indices_buf.destroy(allocator);
    }
}

// ---------------------------------------------------------------------------
// TileCullPass — frustum culling for road tiles
// ---------------------------------------------------------------------------

pub struct TileCullPass {
    pass: ComputePass,
    /// Buffer holding the indirect draw command.
    pub draw_indirect_buf: GpuBuffer,
    /// Buffer holding all tile instances (uploaded from CPU).
    pub all_tiles_buf: Option<GpuBuffer>,
    /// Buffer holding visible (compacted) tile instances.
    pub visible_tiles_buf: GpuBuffer,
    /// Number of tiles currently in all_tiles_buf.
    pub tile_count: u32,
    /// Max tiles this pass was allocated for.
    max_tiles: u32,
    descriptors_dirty: bool,
}

impl TileCullPass {
    pub fn new(
        device: &Device,
        allocator: &vma::Allocator,
        spirv: &[u32],
        max_tiles: u32,
    ) -> anyhow::Result<Self> {
        // 3 bindings: 0 all_tiles, 1 draw_indirect, 2 visible_tiles
        let pass = ComputePass::new(
            device,
            spirv,
            "frustum_cull::tile_cull_main",
            3,
            std::mem::size_of::<TileCullPushConstants>() as u32,
            1,
        )?;

        let draw_indirect_buf = GpuBuffer::new(
            allocator,
            std::mem::size_of::<DrawIndirectCommand>() as u64,
            vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::INDIRECT_BUFFER
                | vk::BufferUsageFlags::TRANSFER_DST,
        )?;

        let visible_tiles_buf = GpuBuffer::new(
            allocator,
            (max_tiles as u64) * std::mem::size_of::<GpuTileInstance>() as u64,
            vk::BufferUsageFlags::STORAGE_BUFFER,
        )?;

        Ok(Self {
            pass,
            draw_indirect_buf,
            all_tiles_buf: None,
            visible_tiles_buf,
            tile_count: 0,
            max_tiles,
            descriptors_dirty: true,
        })
    }

    /// Upload tile instance data from the SDF tile manager.
    /// Builds a GpuTileInstance for each tile that has an atlas slot.
    pub fn upload_tile_instances(
        &mut self,
        allocator: &vma::Allocator,
        sdf: &SdfTileManager,
    ) -> anyhow::Result<()> {
        // Destroy old buffer
        if let Some(b) = self.all_tiles_buf.take() {
            b.destroy(allocator);
        }

        let tile_count = sdf.tile_to_slot.len();
        if tile_count == 0 {
            self.tile_count = 0;
            return Ok(());
        }

        // Reallocate visible_tiles buffer if needed
        if tile_count as u32 > self.max_tiles {
            self.visible_tiles_buf.destroy(allocator);
            let new_max = tile_count as u32;
            self.visible_tiles_buf = GpuBuffer::new(
                allocator,
                (new_max as u64) * std::mem::size_of::<GpuTileInstance>() as u64,
                vk::BufferUsageFlags::STORAGE_BUFFER,
            )?;
            self.max_tiles = new_max;
        }

        let atlas_tiles = sdf.atlas_tiles_dim;
        let atlas_size_f = (atlas_tiles * TILE_RESOLUTION) as f32;
        let half_texel = 0.5 / atlas_size_f;

        let road_id_atlas_size_f = (atlas_tiles * ROAD_ID_RESOLUTION) as f32;
        let road_id_half_texel = 0.5 / road_id_atlas_size_f;

        let mut instances = Vec::with_capacity(tile_count);
        for (key, &slot) in &sdf.tile_to_slot {
            let slot_x = slot % atlas_tiles;
            let slot_y = slot / atlas_tiles;

            // SDF atlas UVs
            let uv_offset_x = (slot_x * TILE_RESOLUTION) as f32 / atlas_size_f + half_texel;
            let uv_offset_y = (slot_y * TILE_RESOLUTION) as f32 / atlas_size_f + half_texel;
            let uv_scale = TILE_RESOLUTION as f32 / atlas_size_f - 2.0 * half_texel;

            // Road ID atlas UVs
            let rid_uv_offset_x = (slot_x * ROAD_ID_RESOLUTION) as f32 / road_id_atlas_size_f + road_id_half_texel;
            let rid_uv_offset_y = (slot_y * ROAD_ID_RESOLUTION) as f32 / road_id_atlas_size_f + road_id_half_texel;
            let rid_uv_scale = ROAD_ID_RESOLUTION as f32 / road_id_atlas_size_f - 2.0 * road_id_half_texel;

            let (wx, wy) = key.world_origin();

            instances.push(GpuTileInstance {
                atlas_uv_offset: [uv_offset_x, uv_offset_y],
                atlas_uv_scale: [uv_scale, uv_scale],
                tile_world_origin: [wx, wy],
                tile_world_size: [TILE_SIZE, TILE_SIZE],
                road_id_uv_offset: [rid_uv_offset_x, rid_uv_offset_y],
                road_id_uv_scale: [rid_uv_scale, rid_uv_scale],
            });
        }

        let data = bytemuck::cast_slice::<_, u8>(&instances);
        let (buf, ptr) = GpuBuffer::new_mapped(
            allocator,
            data.len() as u64,
            vk::BufferUsageFlags::STORAGE_BUFFER,
        )?;
        unsafe { std::ptr::copy_nonoverlapping(data.as_ptr(), ptr, data.len()) };
        self.all_tiles_buf = Some(buf);
        self.tile_count = tile_count as u32;
        self.descriptors_dirty = true;

        Ok(())
    }

    /// Update descriptors if needed.
    pub fn update_descriptors(&mut self, device: &Device) {
        if !self.descriptors_dirty {
            return;
        }
        let all_buf = match &self.all_tiles_buf {
            Some(b) => b,
            None => return,
        };

        write_storage_buffers(
            device,
            self.pass.set(),
            0,
            &[all_buf, &self.draw_indirect_buf, &self.visible_tiles_buf],
        );
        self.descriptors_dirty = false;
    }

    /// Record the cull dispatch.
    pub fn dispatch(
        &self,
        device: &Device,
        cmd: vk::CommandBuffer,
        camera: &Camera2D,
        aspect: f32,
    ) {
        if self.tile_count == 0 || self.all_tiles_buf.is_none() {
            return;
        }

        // Reset indirect draw command: vertex_count=6, instance_count=0
        let reset_cmd = DrawIndirectCommand {
            vertex_count: 6,
            instance_count: 0,
            first_vertex: 0,
            first_instance: 0,
        };
        unsafe {
            device.cmd_update_buffer(
                cmd,
                self.draw_indirect_buf.buffer,
                0,
                bytemuck::bytes_of(&reset_cmd),
            );

            let barrier = vk::MemoryBarrier2::builder()
                .src_stage_mask(vk::PipelineStageFlags2::ALL_TRANSFER)
                .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                .dst_access_mask(vk::AccessFlags2::SHADER_READ | vk::AccessFlags2::SHADER_WRITE);
            let dep = vk::DependencyInfo::builder().memory_barriers(std::slice::from_ref(&barrier));
            device.cmd_pipeline_barrier2(cmd, &dep);
        }

        let (min_x, min_y, max_x, max_y) = camera_world_bounds(camera, aspect);
        let pc = TileCullPushConstants {
            view_min_x: min_x,
            view_min_y: min_y,
            view_max_x: max_x,
            view_max_y: max_y,
            tile_count: self.tile_count,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };

        unsafe {
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.pass.pipeline);
            device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                self.pass.pipeline_layout,
                0,
                &[self.pass.set()],
                &[],
            );
            device.cmd_push_constants(
                cmd,
                self.pass.pipeline_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                bytemuck::bytes_of(&pc),
            );
            device.cmd_dispatch(cmd, self.tile_count.div_ceil(64), 1, 1);

            // Barrier: compute write → indirect draw read + vertex shader read
            let barrier = vk::MemoryBarrier2::builder()
                .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                .src_access_mask(vk::AccessFlags2::SHADER_WRITE)
                .dst_stage_mask(
                    vk::PipelineStageFlags2::DRAW_INDIRECT | vk::PipelineStageFlags2::VERTEX_SHADER,
                )
                .dst_access_mask(
                    vk::AccessFlags2::INDIRECT_COMMAND_READ | vk::AccessFlags2::SHADER_READ,
                );
            let dep = vk::DependencyInfo::builder().memory_barriers(std::slice::from_ref(&barrier));
            device.cmd_pipeline_barrier2(cmd, &dep);
        }
    }

    pub fn destroy(&mut self, device: &Device, allocator: &vma::Allocator) {
        self.pass.destroy(device);
        self.draw_indirect_buf.destroy(allocator);
        if let Some(b) = self.all_tiles_buf.take() {
            b.destroy(allocator);
        }
        self.visible_tiles_buf.destroy(allocator);
    }
}
