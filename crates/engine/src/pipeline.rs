use std::ffi::CStr;

use naga::back::spv;
use naga::front::wgsl;
use vulkanalia::bytecode::Bytecode;
use vulkanalia::prelude::v1_4::*;
use vulkanalia_bootstrap::Device;

/// Compile WGSL source to SPIR-V words.
pub fn compile_wgsl(source: &str) -> anyhow::Result<Vec<u32>> {
    let module = wgsl::parse_str(source)?;

    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)?;

    let mut options = spv::Options::default();
    options
        .flags
        .insert(spv::WriterFlags::ADJUST_COORDINATE_SPACE);

    let binary = spv::write_vec(&module, &info, &options, None)?;
    Ok(binary)
}

/// Create a compute pipeline from SPIR-V words.
/// Returns `(pipeline, pipeline_layout)`.
pub fn create_compute_pipeline(
    device: &Device,
    spirv: &[u32],
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
    let entry_point = CStr::from_bytes_with_nul(b"main\0").unwrap();
    let stage_info = vk::PipelineShaderStageCreateInfo::builder()
        .stage(vk::ShaderStageFlags::COMPUTE)
        .module(shader_module)
        .name(entry_point.to_bytes_with_nul());

    let pipeline_info = vk::ComputePipelineCreateInfo::builder()
        .layout(pipeline_layout)
        .stage(*stage_info);

    let pipeline = unsafe {
        device
            .create_compute_pipelines(vk::PipelineCache::null(), &[*pipeline_info], None)?
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

    let color_blend = vk::PipelineColorBlendStateCreateInfo::builder()
        .attachments(&color_blend_attachments);

    // Dynamic state
    let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
    let dynamic_state =
        vk::PipelineDynamicStateCreateInfo::builder().dynamic_states(&dynamic_states);

    // Dynamic rendering info (no render pass)
    let color_formats = [desc.color_attachment_format];
    let mut rendering_info = vk::PipelineRenderingCreateInfo::builder()
        .color_attachment_formats(&color_formats);

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
            .create_graphics_pipelines(vk::PipelineCache::null(), &[*pipeline_info], None)?
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
