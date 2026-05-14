use std::fs;
use std::sync::Mutex;

use vulkanalia::bytecode::Bytecode;
use vulkanalia::prelude::v1_4::*;
use vulkanalia_bootstrap::Device;

// ---------------------------------------------------------------------------
// Global Vulkan pipeline cache
// ---------------------------------------------------------------------------

/// Global Vulkan pipeline cache handle (set after device creation).
static VK_PIPELINE_CACHE: Mutex<Option<vk::PipelineCache>> = Mutex::new(None);

const PIPELINE_CACHE_FILE: &str = "target/shader_cache/vk_pipeline_cache.bin";

/// Create and store a global Vulkan pipeline cache, loading previous data from disk if available.
/// If the existing cache data is rejected by the driver (e.g. different GPU or corrupt file),
/// falls back to an empty cache.
pub fn init_pipeline_cache(device: &Device) -> anyhow::Result<()> {
    let initial_data = fs::read(PIPELINE_CACHE_FILE).unwrap_or_default();

    let cache = if !initial_data.is_empty() {
        let info = vk::PipelineCacheCreateInfo::builder().initial_data(&initial_data);
        match unsafe { device.create_pipeline_cache(&info, None) } {
            Ok(c) => c,
            Err(e) => {
                log::warn!("Failed to load pipeline cache, creating empty cache: {e}");
                let _ = fs::remove_file(PIPELINE_CACHE_FILE);
                let info = vk::PipelineCacheCreateInfo::builder();
                unsafe { device.create_pipeline_cache(&info, None)? }
            }
        }
    } else {
        let info = vk::PipelineCacheCreateInfo::builder();
        unsafe { device.create_pipeline_cache(&info, None)? }
    };

    *VK_PIPELINE_CACHE.lock().unwrap() = Some(cache);
    Ok(())
}

/// Save the Vulkan pipeline cache to disk and destroy it.
pub fn destroy_pipeline_cache(device: &Device) {
    let mut guard = VK_PIPELINE_CACHE.lock().unwrap();
    if let Some(cache) = guard.take() {
        // Try to persist cache data
        if let Ok(data) = unsafe { device.get_pipeline_cache_data(cache) } {
            let _ = fs::create_dir_all("target/shader_cache");
            let _ = fs::write(PIPELINE_CACHE_FILE, &data);
        }
        unsafe {
            device.destroy_pipeline_cache(cache, None);
        }
    }
}

fn get_vk_pipeline_cache() -> vk::PipelineCache {
    VK_PIPELINE_CACHE
        .lock()
        .unwrap()
        .unwrap_or(vk::PipelineCache::null())
}

/// Create a compute pipeline from SPIR-V words.
/// Returns `(pipeline, pipeline_layout)`.
pub fn create_compute_pipeline(
    device: &Device,
    spirv: &[u32],
    entry_point_name: &str,
    descriptor_set_layouts: &[vk::DescriptorSetLayout],
    push_constant_ranges: &[vk::PushConstantRange],
) -> anyhow::Result<(vk::Pipeline, vk::PipelineLayout)> {
    // Shader module
    let bytecode = Bytecode::new(bytemuck::cast_slice(spirv))?;
    let module_info = vk::ShaderModuleCreateInfo::builder()
        .code(bytecode.code())
        .code_size(bytecode.code_size());

    let shader_module = unsafe { device.create_shader_module(&module_info, None)? };

    // Pipeline layout
    let layout_info = vk::PipelineLayoutCreateInfo::builder()
        .set_layouts(descriptor_set_layouts)
        .push_constant_ranges(push_constant_ranges);

    let pipeline_layout = unsafe { device.create_pipeline_layout(&layout_info, None)? };

    // Pipeline
    let mut entry_buf = entry_point_name.as_bytes().to_vec();
    entry_buf.push(0);
    let stage_info = vk::PipelineShaderStageCreateInfo::builder()
        .stage(vk::ShaderStageFlags::COMPUTE)
        .module(shader_module)
        .name(&entry_buf);

    let pipeline_info = vk::ComputePipelineCreateInfo::builder()
        .layout(pipeline_layout)
        .stage(*stage_info);

    let pipeline = unsafe {
        device
            .create_compute_pipelines(get_vk_pipeline_cache(), &[*pipeline_info], None)?
            .0
            .into_iter()
            .next()
            .ok_or(anyhow::anyhow!("Failed to create compute pipeline"))?
    };

    // Shader module no longer needed
    unsafe {
        device.destroy_shader_module(shader_module, None);
    }

    Ok((pipeline, pipeline_layout))
}

// ---------------------------------------------------------------------------
// ComputePass — bundles pipeline + layout + descriptor set(s)
// ---------------------------------------------------------------------------

/// A complete compute pipeline with its descriptor set layout, pool, and set(s).
///
/// Reduces the per-pipeline boilerplate from ~5 fields to 1.
pub struct ComputePass {
    pub pipeline: vk::Pipeline,
    pub pipeline_layout: vk::PipelineLayout,
    pub descriptor_set_layout: vk::DescriptorSetLayout,
    pub descriptor_pool: vk::DescriptorPool,
    pub descriptor_sets: Vec<vk::DescriptorSet>,
}

impl ComputePass {
    /// Create a compute pass from pre-compiled SPIR-V.
    ///
    /// `num_storage_buffers` is the number of SSBO bindings (binding 0..N-1).
    /// `num_sets` is how many descriptor sets to allocate (typically 1 or 2 for ping-pong).
    pub fn new(
        device: &Device,
        spirv: &[u32],
        entry_point: &str,
        num_storage_buffers: u32,
        push_constant_size: u32,
        num_sets: u32,
    ) -> anyhow::Result<Self> {
        let bindings: Vec<vk::DescriptorSetLayoutBinding> = (0..num_storage_buffers)
            .map(|i| {
                vk::DescriptorSetLayoutBinding::builder()
                    .binding(i)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE)
                    .build()
            })
            .collect();

        let descriptor_set_layout = create_descriptor_set_layout(device, &bindings)?;

        let pool_sizes = [vk::DescriptorPoolSize::builder()
            .type_(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(num_storage_buffers * num_sets)
            .build()];
        let pool_info = vk::DescriptorPoolCreateInfo::builder()
            .max_sets(num_sets)
            .pool_sizes(&pool_sizes);
        let descriptor_pool = unsafe { device.create_descriptor_pool(&pool_info, None)? };

        let mut descriptor_sets = Vec::with_capacity(num_sets as usize);
        for _ in 0..num_sets {
            descriptor_sets.push(allocate_descriptor_set(
                device,
                descriptor_pool,
                descriptor_set_layout,
            )?);
        }

        let push_constant_ranges = [vk::PushConstantRange::builder()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .offset(0)
            .size(push_constant_size)
            .build()];

        let (pipeline, pipeline_layout) = create_compute_pipeline(
            device,
            spirv,
            entry_point,
            &[descriptor_set_layout],
            &push_constant_ranges,
        )?;

        Ok(Self {
            pipeline,
            pipeline_layout,
            descriptor_set_layout,
            descriptor_pool,
            descriptor_sets,
        })
    }

    /// The first (or only) descriptor set.
    pub fn set(&self) -> vk::DescriptorSet {
        self.descriptor_sets[0]
    }

    /// Descriptor set by index (for ping-pong patterns).
    pub fn set_at(&self, index: usize) -> vk::DescriptorSet {
        self.descriptor_sets[index]
    }

    /// Destroy all Vulkan resources.
    pub fn destroy(&self, device: &Device) {
        unsafe {
            device.destroy_pipeline(self.pipeline, None);
            device.destroy_pipeline_layout(self.pipeline_layout, None);
            device.destroy_descriptor_pool(self.descriptor_pool, None);
            device.destroy_descriptor_set_layout(self.descriptor_set_layout, None);
        }
    }
}

// ---------------------------------------------------------------------------
// Descriptor write helpers
// ---------------------------------------------------------------------------

use crate::gpu_resources::GpuBuffer;

/// Write storage buffer descriptors starting at `start_binding`.
///
/// Binds `buffers[0]` to `start_binding`, `buffers[1]` to `start_binding + 1`, etc.
pub fn write_storage_buffers(
    device: &Device,
    set: vk::DescriptorSet,
    start_binding: u32,
    buffers: &[&GpuBuffer],
) {
    let buf_infos: Vec<[vk::DescriptorBufferInfo; 1]> = buffers
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
                .dst_set(set)
                .dst_binding(start_binding + i as u32)
                .dst_array_element(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(info)
                .build()
        })
        .collect();

    unsafe {
        device.update_descriptor_sets(&writes, &[] as &[vk::CopyDescriptorSet]);
    }
}

/// Convenience: create a descriptor set layout from a list of bindings.
pub fn create_descriptor_set_layout(
    device: &Device,
    bindings: &[vk::DescriptorSetLayoutBinding],
) -> anyhow::Result<vk::DescriptorSetLayout> {
    let info = vk::DescriptorSetLayoutCreateInfo::builder().bindings(bindings);
    Ok(unsafe { device.create_descriptor_set_layout(&info, None)? })
}

/// Convenience: allocate a descriptor set from a pool.
pub fn allocate_descriptor_set(
    device: &Device,
    pool: vk::DescriptorPool,
    layout: vk::DescriptorSetLayout,
) -> anyhow::Result<vk::DescriptorSet> {
    let layouts = [layout];
    let alloc_info = vk::DescriptorSetAllocateInfo::builder()
        .descriptor_pool(pool)
        .set_layouts(&layouts);

    let sets = unsafe { device.allocate_descriptor_sets(&alloc_info)? };
    sets.into_iter()
        .next()
        .ok_or(anyhow::anyhow!("Failed to allocate descriptor set"))
}

// ---------------------------------------------------------------------------
// Graphics pipeline (dynamic rendering)
// ---------------------------------------------------------------------------

/// Configuration for creating a simple graphics pipeline with dynamic rendering.
pub struct GraphicsPipelineDesc<'a> {
    pub vertex_spirv: &'a [u32],
    pub fragment_spirv: &'a [u32],
    pub vertex_entry: &'a str,
    pub fragment_entry: &'a str,
    pub vertex_binding_descriptions: &'a [vk::VertexInputBindingDescription],
    pub vertex_attribute_descriptions: &'a [vk::VertexInputAttributeDescription],
    pub topology: vk::PrimitiveTopology,
    pub color_attachment_format: vk::Format,
    pub push_constant_ranges: &'a [vk::PushConstantRange],
    pub descriptor_set_layouts: &'a [vk::DescriptorSetLayout],
    pub line_width: f32,
}

/// Create a graphics pipeline using dynamic rendering (no render pass).
/// Returns `(pipeline, pipeline_layout)`.
pub fn create_graphics_pipeline(
    device: &Device,
    desc: &GraphicsPipelineDesc,
) -> anyhow::Result<(vk::Pipeline, vk::PipelineLayout)> {
    // Build null-terminated entry point names
    let mut vert_entry_buf = desc.vertex_entry.as_bytes().to_vec();
    vert_entry_buf.push(0);
    let mut frag_entry_buf = desc.fragment_entry.as_bytes().to_vec();
    frag_entry_buf.push(0);

    // Shader modules
    let vert_bytecode = Bytecode::new(bytemuck::cast_slice(desc.vertex_spirv))?;
    let vert_module_info = vk::ShaderModuleCreateInfo::builder()
        .code(vert_bytecode.code())
        .code_size(vert_bytecode.code_size());
    let vert_module = unsafe { device.create_shader_module(&vert_module_info, None)? };

    let frag_bytecode = Bytecode::new(bytemuck::cast_slice(desc.fragment_spirv))?;
    let frag_module_info = vk::ShaderModuleCreateInfo::builder()
        .code(frag_bytecode.code())
        .code_size(frag_bytecode.code_size());
    let frag_module = unsafe { device.create_shader_module(&frag_module_info, None)? };

    let stages = [
        vk::PipelineShaderStageCreateInfo::builder()
            .stage(vk::ShaderStageFlags::VERTEX)
            .module(vert_module)
            .name(&vert_entry_buf)
            .build(),
        vk::PipelineShaderStageCreateInfo::builder()
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .module(frag_module)
            .name(&frag_entry_buf)
            .build(),
    ];

    // Vertex input
    let vertex_input = vk::PipelineVertexInputStateCreateInfo::builder()
        .vertex_binding_descriptions(desc.vertex_binding_descriptions)
        .vertex_attribute_descriptions(desc.vertex_attribute_descriptions);

    // Input assembly
    let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::builder()
        .topology(desc.topology)
        .primitive_restart_enable(false);

    // Viewport/scissor — dynamic state
    let viewport_state = vk::PipelineViewportStateCreateInfo::builder()
        .viewport_count(1)
        .scissor_count(1);

    // Rasterization
    let rasterization = vk::PipelineRasterizationStateCreateInfo::builder()
        .polygon_mode(vk::PolygonMode::FILL)
        .cull_mode(vk::CullModeFlags::NONE)
        .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
        .line_width(desc.line_width);

    // Multisample
    let multisample = vk::PipelineMultisampleStateCreateInfo::builder()
        .rasterization_samples(vk::SampleCountFlags::_1);

    // Color blend (alpha blending)
    let color_blend_attachments = [vk::PipelineColorBlendAttachmentState::builder()
        .blend_enable(true)
        .src_color_blend_factor(vk::BlendFactor::SRC_ALPHA)
        .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
        .color_blend_op(vk::BlendOp::ADD)
        .src_alpha_blend_factor(vk::BlendFactor::ONE)
        .dst_alpha_blend_factor(vk::BlendFactor::ZERO)
        .alpha_blend_op(vk::BlendOp::ADD)
        .color_write_mask(vk::ColorComponentFlags::all())
        .build()];

    let color_blend =
        vk::PipelineColorBlendStateCreateInfo::builder().attachments(&color_blend_attachments);

    // Dynamic state
    let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
    let dynamic_state =
        vk::PipelineDynamicStateCreateInfo::builder().dynamic_states(&dynamic_states);

    // Dynamic rendering info (no render pass)
    let color_formats = [desc.color_attachment_format];
    let mut rendering_info =
        vk::PipelineRenderingCreateInfo::builder().color_attachment_formats(&color_formats);

    // Pipeline layout
    let layout_info = vk::PipelineLayoutCreateInfo::builder()
        .set_layouts(desc.descriptor_set_layouts)
        .push_constant_ranges(desc.push_constant_ranges);
    let pipeline_layout = unsafe { device.create_pipeline_layout(&layout_info, None)? };

    // Create pipeline
    let pipeline_info = vk::GraphicsPipelineCreateInfo::builder()
        .stages(&stages)
        .vertex_input_state(&vertex_input)
        .input_assembly_state(&input_assembly)
        .viewport_state(&viewport_state)
        .rasterization_state(&rasterization)
        .multisample_state(&multisample)
        .color_blend_state(&color_blend)
        .dynamic_state(&dynamic_state)
        .layout(pipeline_layout)
        .push_next(&mut rendering_info);

    let pipeline = unsafe {
        device
            .create_graphics_pipelines(get_vk_pipeline_cache(), &[*pipeline_info], None)?
            .0
            .into_iter()
            .next()
            .ok_or(anyhow::anyhow!("Failed to create graphics pipeline"))?
    };

    // Clean up shader modules
    unsafe {
        device.destroy_shader_module(vert_module, None);
        device.destroy_shader_module(frag_module, None);
    }

    Ok((pipeline, pipeline_layout))
}
