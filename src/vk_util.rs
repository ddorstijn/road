use std::sync::Arc;

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
