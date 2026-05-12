use spirv_builder::{Capability, SpirvBuilder};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut builder = SpirvBuilder::new("../shaders", "spirv-unknown-vulkan1.4")
        .capability(Capability::StorageImageWriteWithoutFormat)
        .capability(Capability::ImageQuery)
        .capability(Capability::VulkanMemoryModelDeviceScope);
    builder.build_script.defaults = true;
    builder.build_script.env_shader_spv_path = Some(true);
    builder.build()?;
    Ok(())
}
