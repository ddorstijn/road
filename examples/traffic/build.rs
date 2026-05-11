use spirv_builder::SpirvBuilder;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    SpirvBuilder::new("../../shaders", "spirv-unknown-vulkan1.3")
        .print_metadata(spirv_builder::MetadataPrintout::Full)
        .build()?;
    Ok(())
}
