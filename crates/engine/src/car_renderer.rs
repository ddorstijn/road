//! Car renderer — instanced rendering of cars as oriented rectangles.
//!
//! Reads car SoA SSBOs + road data SSBOs, evaluates road position in the vertex shader,
//! and renders each car as a 6-vertex quad (2 triangles) with speed-based coloring,
//! outline, and windshield indicator.

use vulkanalia::prelude::v1_4::*;
use vulkanalia_bootstrap::Device;

use crate::gpu_resources::GpuBuffer;
use crate::pipeline::{
    GraphicsPipelineDesc, allocate_descriptor_set, create_descriptor_set_layout,
    create_graphics_pipeline,
};

// ---------------------------------------------------------------------------
// Push constants
// ---------------------------------------------------------------------------

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CarRenderPushConstants {
    pub view_proj: [[f32; 4]; 4],
    pub car_count: u32,
    pub _pad0: u32,
    pub _pad1: u32,
    pub _pad2: u32,
}

// ---------------------------------------------------------------------------
// CarRenderer
// ---------------------------------------------------------------------------

pub struct CarRenderer {
    pipeline: vk::Pipeline,
    pipeline_layout: vk::PipelineLayout,
    descriptor_set_layout: vk::DescriptorSetLayout,
    descriptor_pool: vk::DescriptorPool,
    descriptor_set: vk::DescriptorSet,
    descriptors_valid: bool,
}

impl CarRenderer {
    /// Create the car rendering pipeline.
    ///
    /// `spirv` is the pre-compiled SPIR-V bytecode containing vs_main and fs_main entry points.
    pub fn new(
        device: &Device,
        color_attachment_format: vk::Format,
        spirv: &[u32],
    ) -> anyhow::Result<Self> {
        // 9 storage buffer bindings: 0-4 car SoA, 5-8 road data
        let bindings: Vec<vk::DescriptorSetLayoutBinding> = (0..9)
            .map(|i| {
                vk::DescriptorSetLayoutBinding::builder()
                    .binding(i)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::VERTEX)
                    .build()
            })
            .collect();

        let descriptor_set_layout = create_descriptor_set_layout(device, &bindings)?;

        let pool_sizes = [vk::DescriptorPoolSize::builder()
            .type_(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(9)
            .build()];
        let pool_info = vk::DescriptorPoolCreateInfo::builder()
            .max_sets(1)
            .pool_sizes(&pool_sizes);
        let descriptor_pool = unsafe { device.create_descriptor_pool(&pool_info, None)? };

        let descriptor_set =
            allocate_descriptor_set(device, descriptor_pool, descriptor_set_layout)?;

        let push_constant_ranges = [vk::PushConstantRange::builder()
            .stage_flags(vk::ShaderStageFlags::VERTEX)
            .offset(0)
            .size(std::mem::size_of::<CarRenderPushConstants>() as u32)
            .build()];

        let desc = GraphicsPipelineDesc {
            vertex_spirv: &spirv,
            fragment_spirv: &spirv,
            vertex_entry: "car_render::vs_main",
            fragment_entry: "car_render::fs_main",
            vertex_binding_descriptions: &[],
            vertex_attribute_descriptions: &[],
            topology: vk::PrimitiveTopology::TRIANGLE_LIST,
            color_attachment_format,
            push_constant_ranges: &push_constant_ranges,
            descriptor_set_layouts: &[descriptor_set_layout],
            line_width: 1.0,
        };

        let (pipeline, pipeline_layout) = create_graphics_pipeline(device, &desc)?;

        Ok(Self {
            pipeline,
            pipeline_layout,
            descriptor_set_layout,
            descriptor_pool,
            descriptor_set,
            descriptors_valid: false,
        })
    }

    /// Update descriptor set with current car + road buffers.
    ///
    /// Car buffers (bindings 0-4): road_id, s, lane, speed, desired_speed
    /// Road buffers (bindings 5-8): segments, roads, lane_sections, lanes
    pub fn update_descriptors(
        &mut self,
        device: &Device,
        car_road_id: &GpuBuffer,
        car_s: &GpuBuffer,
        car_lane: &GpuBuffer,
        car_speed: &GpuBuffer,
        car_desired_speed: &GpuBuffer,
        segment_buf: &GpuBuffer,
        road_buf: &GpuBuffer,
        lane_section_buf: &GpuBuffer,
        lane_buf: &GpuBuffer,
    ) {
        let bufs: [&GpuBuffer; 9] = [
            car_road_id,
            car_s,
            car_lane,
            car_speed,
            car_desired_speed,
            segment_buf,
            road_buf,
            lane_section_buf,
            lane_buf,
        ];

        let buf_infos: Vec<[vk::DescriptorBufferInfo; 1]> = bufs
            .iter()
            .map(|b| {
                [vk::DescriptorBufferInfo::builder()
                    .buffer(b.buffer)
                    .offset(0)
                    .range(b.size)
                    .build()]
            })
            .collect();

        let writes: Vec<vk::WriteDescriptorSet> = buf_infos
            .iter()
            .enumerate()
            .map(|(i, info)| {
                vk::WriteDescriptorSet::builder()
                    .dst_set(self.descriptor_set)
                    .dst_binding(i as u32)
                    .dst_array_element(0)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(info)
                    .build()
            })
            .collect();

        unsafe {
            device.update_descriptor_sets(&writes, &[] as &[vk::CopyDescriptorSet]);
        }

        self.descriptors_valid = true;
    }

    /// Returns whether descriptors have been updated and drawing is possible.
    pub fn is_ready(&self) -> bool {
        self.descriptors_valid
    }

    /// Record draw commands for all cars.
    ///
    /// The caller must have already begun dynamic rendering on the command buffer.
    pub fn draw(&self, device: &Device, cmd: vk::CommandBuffer, pc: &CarRenderPushConstants) {
        if !self.descriptors_valid || pc.car_count == 0 {
            return;
        }

        unsafe {
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, self.pipeline);

            device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline_layout,
                0,
                &[self.descriptor_set],
                &[],
            );

            device.cmd_push_constants(
                cmd,
                self.pipeline_layout,
                vk::ShaderStageFlags::VERTEX,
                0,
                bytemuck::bytes_of(pc),
            );

            device.cmd_draw(cmd, 6, pc.car_count, 0, 0);
        }
    }

    /// Destroy Vulkan resources. Must be called before device destruction.
    pub fn destroy(&self, device: &Device) {
        unsafe {
            device.destroy_pipeline(self.pipeline, None);
            device.destroy_pipeline_layout(self.pipeline_layout, None);
            device.destroy_descriptor_pool(self.descriptor_pool, None);
            device.destroy_descriptor_set_layout(self.descriptor_set_layout, None);
        }
    }
}
