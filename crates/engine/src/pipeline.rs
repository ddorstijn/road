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
