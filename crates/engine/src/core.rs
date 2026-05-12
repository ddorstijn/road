use std::mem::ManuallyDrop;
use std::sync::Arc;
use std::time::Duration;

use vulkanalia::prelude::v1_4::*;
use vulkanalia::vk::KhrSwapchainExtensionDeviceCommands;
use vulkanalia::Version;
use vulkanalia_bootstrap::{
    Device, DeviceBuilder, Instance, InstanceBuilder, PhysicalDeviceSelector, PreferredDeviceType,
    QueueType, Swapchain, SwapchainBuilder,
};
use vulkanalia_vma::{self as vma};
use winit::dpi::PhysicalSize;
use winit::window::Window;

use crate::gpu_resources::GpuImage;

/// Per-frame synchronization and command recording state.
/// Double-buffered: one per frame-in-flight.
pub(crate) struct FrameData {
    pub command_pool: vk::CommandPool,
    pub command_buffer: vk::CommandBuffer,
    pub swapchain_semaphore: vk::Semaphore,
    pub render_fence: vk::Fence,
}

/// A swapchain image paired with its semaphore for present synchronization.
#[allow(dead_code)]
pub(crate) struct SwapchainImage {
    pub image: vk::Image,
    pub view: vk::ImageView,
    pub render_semaphore: vk::Semaphore,
}

/// The core Vulkan context: device, allocator, swapchain, frames-in-flight.
/// Owned by the engine runner; passed to App callbacks via `EngineContext`.
#[allow(dead_code)]
pub struct Core {
    pub(crate) instance: Arc<Instance>,
    pub(crate) device: Arc<Device>,
    pub(crate) allocator: ManuallyDrop<vma::Allocator>,
    pub(crate) graphics_queue: vk::Queue,
    pub(crate) graphics_queue_family: u32,

    pub(crate) swapchain: Swapchain,
    pub(crate) swapchain_images: Vec<SwapchainImage>,
    pub(crate) draw_image: GpuImage,

    pub(crate) frames: Vec<FrameData>,
    pub(crate) frame_number: usize,
    pub(crate) timestamp_period: f32,
}

const FRAME_OVERLAP: usize = 2;

impl Core {
    pub fn new(window: Arc<Window>) -> anyhow::Result<Self> {
        // --- Instance & Device ---
        let instance = InstanceBuilder::new(Some(window.clone()))
            .app_name("traffic-sim")
            .engine_name("road-engine")
            .request_validation_layers(true)
            .minimum_instance_version(Version::new(1, 3, 0))
            .require_api_version(Version::new(1, 3, 0))
            .build()?;

        let features12 = vk::PhysicalDeviceVulkan12Features::builder()
            .buffer_device_address(true)
            .descriptor_indexing(true)
            .vulkan_memory_model(true)
            .vulkan_memory_model_device_scope(true);

        let features13 = vk::PhysicalDeviceVulkan13Features::builder()
            .synchronization2(true)
            .dynamic_rendering(true);

        let base_features = vk::PhysicalDeviceFeatures::builder()
            .pipeline_statistics_query(true)
            .robust_buffer_access(true);

        let physical_device = PhysicalDeviceSelector::new(instance.clone())
            .preferred_device_type(PreferredDeviceType::Discrete)
            .add_required_features(*base_features)
            .add_required_extension_feature(*features12)
            .add_required_extension_feature(*features13)
            .select()?;

        let device = Arc::new(DeviceBuilder::new(physical_device, instance.clone()).build()?);

        // --- VMA allocator ---
        let allocator_options = vma::AllocatorOptions::new(
            instance.as_ref().as_ref(),
            device.as_ref(),
            *device.physical_device().as_ref(),
        );
        let allocator = unsafe { vma::Allocator::new(&allocator_options) }?;

        // --- Queue ---
        let (graphics_queue_family, graphics_queue) = device.get_queue(QueueType::Graphics)?;
        let graphics_queue_family = graphics_queue_family as u32;

        // --- Swapchain ---
        let PhysicalSize { width, height } = window.inner_size();
        let (swapchain, swapchain_images) =
            Self::create_swapchain(instance.clone(), device.clone(), width, height)?;

        // --- Draw image (offscreen storage image blitted to swapchain) ---
        let draw_image = GpuImage::new_storage_2d(
            &device,
            &allocator,
            width,
            height,
            vk::Format::R16G16B16A16_SFLOAT,
        )?;

        // --- Timestamp period (ns per tick) ---
        let timestamp_period = unsafe {
            let inst: &vulkanalia::Instance = instance.as_ref().as_ref();
            let props = inst.get_physical_device_properties(*device.physical_device().as_ref());
            props.limits.timestamp_period
        };

        // --- Frame data ---
        let frames = Self::create_frame_data(&device, graphics_queue_family, FRAME_OVERLAP)?;

        Ok(Self {
            instance,
            device,
            allocator: ManuallyDrop::new(allocator),
            graphics_queue,
            graphics_queue_family,
            swapchain,
            swapchain_images,
            draw_image,
            frames,
            frame_number: 0,
            timestamp_period,
        })
    }

    /// Record commands, submit, and present one frame.
    /// `record_fn` is called between begin/end command buffer with the draw image
    /// in GENERAL layout, ready for compute writes.
    pub fn draw_frame(
        &mut self,
        record_fn: impl FnOnce(&Device, vk::CommandBuffer, &GpuImage),
    ) -> anyhow::Result<()> {
        let frame = &self.frames[self.frame_number % FRAME_OVERLAP];

        unsafe {
            self.device.wait_for_fences(
                &[frame.render_fence],
                true,
                Duration::from_secs(1).as_nanos() as u64,
            )?;
            self.device.reset_fences(&[frame.render_fence])?;

            let (image_index, result) = self.device.acquire_next_image_khr(
                *self.swapchain.as_ref(),
                u64::MAX,
                frame.swapchain_semaphore,
                vk::Fence::null(),
            )?;

            if matches!(result, vk::SuccessCode::TIMEOUT) {
                return Ok(());
            }

            let swap_img = &self.swapchain_images[image_index as usize];
            let cmd = frame.command_buffer;

            self.device
                .reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty())?;
            self.device.begin_command_buffer(
                cmd,
                &vk::CommandBufferBeginInfo::builder()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )?;

            // Transition draw image to GENERAL for compute writes
            transition_image(
                &self.device,
                cmd,
                self.draw_image.image,
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::GENERAL,
            );

            // --- App records commands here ---
            record_fn(&self.device, cmd, &self.draw_image);

            // Transition for blit: draw → TRANSFER_SRC, swapchain → TRANSFER_DST
            transition_image(
                &self.device,
                cmd,
                self.draw_image.image,
                vk::ImageLayout::GENERAL,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            );
            transition_image(
                &self.device,
                cmd,
                swap_img.image,
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            );

            // Blit draw image → swapchain image
            blit_image_to_image(
                &self.device,
                cmd,
                self.draw_image.image,
                swap_img.image,
                self.draw_image.extent_2d(),
                self.swapchain.extent,
            );

            // Transition swapchain image to PRESENT
            transition_image(
                &self.device,
                cmd,
                swap_img.image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                vk::ImageLayout::PRESENT_SRC_KHR,
            );

            self.device.end_command_buffer(cmd)?;

            // Submit
            let cmd_info = [vk::CommandBufferSubmitInfo::builder().command_buffer(cmd)];
            let wait_info = [vk::SemaphoreSubmitInfo::builder()
                .semaphore(frame.swapchain_semaphore)
                .stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
                .value(1)];
            let signal_info = [vk::SemaphoreSubmitInfo::builder()
                .semaphore(swap_img.render_semaphore)
                .stage_mask(vk::PipelineStageFlags2::ALL_GRAPHICS)
                .value(1)];

            let submit = vk::SubmitInfo2::builder()
                .command_buffer_infos(&cmd_info)
                .wait_semaphore_infos(&wait_info)
                .signal_semaphore_infos(&signal_info);

            self.device
                .queue_submit2(self.graphics_queue, &[submit], frame.render_fence)?;

            // Present
            let wait_semaphores = [swap_img.render_semaphore];
            let swapchains = [*self.swapchain.as_ref()];
            let image_indices = [image_index];

            let present_info = vk::PresentInfoKHR::builder()
                .wait_semaphores(&wait_semaphores)
                .swapchains(&swapchains)
                .image_indices(&image_indices);

            self.device
                .queue_present_khr(self.graphics_queue, &present_info)?;
        }

        self.frame_number += 1;
        Ok(())
    }

    /// Recreate the swapchain and draw image after a window resize.
    pub fn handle_resize(&mut self, width: u32, height: u32) -> anyhow::Result<()> {
        if width == 0 || height == 0 {
            return Ok(());
        }

        unsafe {
            self.device.device_wait_idle()?;
        }

        // Destroy old swapchain resources
        self.destroy_swapchain_resources();

        // Destroy old draw image
        self.draw_image.destroy(&self.device, &self.allocator);

        // Recreate
        let (swapchain, swapchain_images) =
            Self::create_swapchain(self.instance.clone(), self.device.clone(), width, height)?;
        self.swapchain = swapchain;
        self.swapchain_images = swapchain_images;

        self.draw_image = GpuImage::new_storage_2d(
            &self.device,
            &self.allocator,
            width,
            height,
            vk::Format::R16G16B16A16_SFLOAT,
        )?;

        Ok(())
    }

    pub fn device(&self) -> &Arc<Device> {
        &self.device
    }

    pub fn allocator(&self) -> &vma::Allocator {
        &self.allocator
    }

    pub fn draw_image(&self) -> &GpuImage {
        &self.draw_image
    }

    pub fn frame_number(&self) -> usize {
        self.frame_number
    }

    pub fn timestamp_period(&self) -> f32 {
        self.timestamp_period
    }

    pub fn window_extent(&self) -> vk::Extent2D {
        self.swapchain.extent
    }

    // --- Private helpers ---

    fn create_swapchain(
        instance: Arc<Instance>,
        device: Arc<Device>,
        width: u32,
        height: u32,
    ) -> anyhow::Result<(Swapchain, Vec<SwapchainImage>)> {
        let swapchain = SwapchainBuilder::new(instance, device.clone())
            .desired_format(
                vk::SurfaceFormat2KHR::builder()
                    .surface_format(
                        vk::SurfaceFormatKHR::builder()
                            .format(vk::Format::B8G8R8A8_UNORM)
                            .color_space(vk::ColorSpaceKHR::SRGB_NONLINEAR)
                            .build(),
                    )
                    .build(),
            )
            .add_image_usage_flags(vk::ImageUsageFlags::TRANSFER_DST)
            .desired_size(vk::Extent2D::builder().width(width).height(height).build())
            .use_default_present_modes()
            .build()?;

        let images = swapchain.get_images()?;
        let views = swapchain.get_image_views()?;

        let swapchain_images = images
            .iter()
            .zip(views)
            .map(|(&image, view)| {
                let render_semaphore =
                    unsafe { device.create_semaphore(&vk::SemaphoreCreateInfo::default(), None)? };
                Ok(SwapchainImage {
                    image,
                    view,
                    render_semaphore,
                })
            })
            .collect::<anyhow::Result<Vec<_>>>()?;

        Ok((swapchain, swapchain_images))
    }

    fn create_frame_data(
        device: &Device,
        queue_family: u32,
        count: usize,
    ) -> anyhow::Result<Vec<FrameData>> {
        let pool_info = vk::CommandPoolCreateInfo::builder()
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
            .queue_family_index(queue_family);

        (0..count)
            .map(|_| unsafe {
                let command_pool = device.create_command_pool(&pool_info, None)?;
                let command_buffer = *device
                    .allocate_command_buffers(
                        &vk::CommandBufferAllocateInfo::builder()
                            .command_pool(command_pool)
                            .command_buffer_count(1)
                            .level(vk::CommandBufferLevel::PRIMARY),
                    )?
                    .first()
                    .ok_or(anyhow::anyhow!("Failed to allocate command buffer"))?;

                let swapchain_semaphore =
                    device.create_semaphore(&vk::SemaphoreCreateInfo::default(), None)?;
                let render_fence = device.create_fence(
                    &vk::FenceCreateInfo::builder().flags(vk::FenceCreateFlags::SIGNALED),
                    None,
                )?;

                Ok(FrameData {
                    command_pool,
                    command_buffer,
                    swapchain_semaphore,
                    render_fence,
                })
            })
            .collect()
    }

    fn destroy_swapchain_resources(&self) {
        self.swapchain.destroy_image_views().ok();
        for img in &self.swapchain_images {
            unsafe {
                self.device.destroy_semaphore(img.render_semaphore, None);
            }
        }
        self.swapchain.destroy();
    }
}

impl Drop for Core {
    fn drop(&mut self) {
        unsafe {
            self.device.device_wait_idle().unwrap();
        }

        // Draw image
        self.draw_image.destroy(&self.device, &self.allocator);

        // Frame data
        for frame in &self.frames {
            unsafe {
                self.device
                    .free_command_buffers(frame.command_pool, &[frame.command_buffer]);
                self.device.destroy_command_pool(frame.command_pool, None);
                self.device.destroy_fence(frame.render_fence, None);
                self.device
                    .destroy_semaphore(frame.swapchain_semaphore, None);
            }
        }

        // Swapchain
        self.destroy_swapchain_resources();

        // VMA allocator (must drop before device)
        unsafe {
            ManuallyDrop::drop(&mut self.allocator);
        }

        // Device & instance
        self.device.destroy();
        self.instance.destroy();
    }
}

// ---------------------------------------------------------------------------
// Vulkan utility functions
// ---------------------------------------------------------------------------

pub fn transition_image(
    device: &Device,
    cmd: vk::CommandBuffer,
    image: vk::Image,
    old_layout: vk::ImageLayout,
    new_layout: vk::ImageLayout,
) {
    let aspect_mask = if new_layout == vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL {
        vk::ImageAspectFlags::DEPTH
    } else {
        vk::ImageAspectFlags::COLOR
    };

    let subresource_range = vk::ImageSubresourceRange::builder()
        .aspect_mask(aspect_mask)
        .base_mip_level(0)
        .level_count(vk::REMAINING_MIP_LEVELS)
        .base_array_layer(0)
        .layer_count(vk::REMAINING_ARRAY_LAYERS)
        .build();

    let barriers = [vk::ImageMemoryBarrier2::builder()
        .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
        .src_access_mask(vk::AccessFlags2::MEMORY_WRITE)
        .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
        .dst_access_mask(vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE)
        .old_layout(old_layout)
        .new_layout(new_layout)
        .subresource_range(subresource_range)
        .image(image)];

    let dep_info = vk::DependencyInfo::builder().image_memory_barriers(&barriers);
    unsafe {
        device.cmd_pipeline_barrier2(cmd, &dep_info);
    }
}

pub(crate) fn blit_image_to_image(
    device: &Device,
    cmd: vk::CommandBuffer,
    src: vk::Image,
    dst: vk::Image,
    src_extent: vk::Extent2D,
    dst_extent: vk::Extent2D,
) {
    let src_subresource = vk::ImageSubresourceLayers::builder()
        .aspect_mask(vk::ImageAspectFlags::COLOR)
        .base_array_layer(0)
        .layer_count(1);
    let dst_subresource = vk::ImageSubresourceLayers::builder()
        .aspect_mask(vk::ImageAspectFlags::COLOR)
        .base_array_layer(0)
        .layer_count(1);

    let src_offset = vk::Offset3D::builder()
        .x(src_extent.width as i32)
        .y(src_extent.height as i32)
        .z(1)
        .build();
    let dst_offset = vk::Offset3D::builder()
        .x(dst_extent.width as i32)
        .y(dst_extent.height as i32)
        .z(1)
        .build();

    let regions = [vk::ImageBlit2::builder()
        .src_offsets([vk::Offset3D::default(), src_offset])
        .dst_offsets([vk::Offset3D::default(), dst_offset])
        .src_subresource(src_subresource)
        .dst_subresource(dst_subresource)];

    let blit_info = vk::BlitImageInfo2::builder()
        .src_image(src)
        .src_image_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
        .dst_image(dst)
        .dst_image_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
        .filter(vk::Filter::LINEAR)
        .regions(&regions);

    unsafe {
        device.cmd_blit_image2(cmd, &blit_info);
    }
}
