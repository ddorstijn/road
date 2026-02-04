use std::sync::Arc;

use naga::{back::spv, front::wgsl};
use vulkanalia::vk::{self, DeviceV1_3, HasBuilder};
use vulkanalia_bootstrap::Device;

pub fn image_subresource_range(aspect_mask: vk::ImageAspectFlags) -> vk::ImageSubresourceRange {
    vk::ImageSubresourceRange::builder()
        .aspect_mask(aspect_mask)
        .base_mip_level(0)
        .level_count(vk::REMAINING_MIP_LEVELS)
        .base_array_layer(0)
        .layer_count(vk::REMAINING_ARRAY_LAYERS)
        .build()
}

pub fn transition_image(
    device: Arc<Device>,
    cmd: vk::CommandBuffer,
    image: vk::Image,
    current_layout: vk::ImageLayout,
    new_layout: vk::ImageLayout,
) {
    let aspect_mask = if new_layout == vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL {
        vk::ImageAspectFlags::DEPTH
    } else {
        vk::ImageAspectFlags::COLOR
    };

    let image_barriers = [vk::ImageMemoryBarrier2::builder()
        .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
        .src_access_mask(vk::AccessFlags2::MEMORY_WRITE)
        .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
        .dst_access_mask(vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE)
        .old_layout(current_layout)
        .new_layout(new_layout)
        .subresource_range(image_subresource_range(aspect_mask))
        .image(image)];

    let dep_info = vk::DependencyInfo::builder().image_memory_barriers(&image_barriers);

    unsafe {
        device.cmd_pipeline_barrier2(cmd, &dep_info);
    }
}

pub fn copy_image_to_image(
    device: Arc<Device>,
    cmd: vk::CommandBuffer,
    source: vk::Image,
    destination: vk::Image,
    src_size: vk::Extent2D,
    dst_size: vk::Extent2D,
) {
    let src_offset = vk::Offset3D::builder()
        .x(src_size.width as _)
        .y(src_size.height as _)
        .z(1)
        .build();

    // FIX: Set base_array_layer to 0 and layer_count to 1
    let src_subresource = vk::ImageSubresourceLayers::builder()
        .aspect_mask(vk::ImageAspectFlags::COLOR)
        .base_array_layer(0)
        .layer_count(1);

    let dst_offset = vk::Offset3D::builder()
        .x(dst_size.width as _)
        .y(dst_size.height as _)
        .z(1)
        .build();

    // FIX: Set base_array_layer to 0 and layer_count to 1
    let dst_subresource = vk::ImageSubresourceLayers::builder()
        .aspect_mask(vk::ImageAspectFlags::COLOR)
        .base_array_layer(0)
        .layer_count(1);

    let blit_regions = [vk::ImageBlit2::builder()
        .src_offsets([vk::Offset3D::default(), src_offset])
        .dst_offsets([vk::Offset3D::default(), dst_offset])
        .src_subresource(src_subresource)
        .dst_subresource(dst_subresource)];

    // ... remainder of function ...
    let blit_image_info = vk::BlitImageInfo2::builder()
        .dst_image(destination)
        .dst_image_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
        .src_image(source)
        .src_image_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
        .filter(vk::Filter::LINEAR)
        .regions(&blit_regions);

    unsafe {
        device.cmd_blit_image2(cmd, &blit_image_info);
    }
}

pub fn compile_wgsl(source: &str) -> Result<Vec<u32>, anyhow::Error> {
    // 1. Parse WGSL
    let module = wgsl::parse_str(source)?;

    // 2. Validate (Checks types, bindings, etc.)
    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)?;

    // 3. Write SPIR-V
    let mut options = spv::Options::default();
    // Important: Vulkan 1.1+ conventions
    options
        .flags
        .insert(spv::WriterFlags::ADJUST_COORDINATE_SPACE);

    let binary = spv::write_vec(&module, &info, &options, None)?;
    Ok(binary)
}
