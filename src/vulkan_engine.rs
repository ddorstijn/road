use std::{sync::Arc, time::Duration};

use crate::vk_util::{image_subresource_range, transition_image};
use vulkanalia::{
    Version,
    vk::{self, DeviceV1_0, DeviceV1_3, Handle, HasBuilder, KhrSwapchainExtensionDeviceCommands},
};
use vulkanalia_bootstrap::{
    Device, DeviceBuilder, Instance, InstanceBuilder, PhysicalDeviceSelector, PreferredDeviceType,
    QueueType, Swapchain, SwapchainBuilder,
};
use vulkanalia_vma::{self as vma};
use winit::{dpi::PhysicalSize, window::Window};

#[derive(Debug)]
struct FrameData {
    command_pool: vk::CommandPool,
    command_buffer: vk::CommandBuffer,
    swapchain_semaphore: vk::Semaphore,
    render_fence: vk::Fence,
}

#[derive(Debug)]
struct RenderImage {
    image: vk::Image,
    view: vk::ImageView,
    semaphore: vk::Semaphore,
}

#[derive(Debug)]
pub struct VulkanEngine {
    pub window: Arc<Window>,
    instance: Arc<Instance>,
    device: Arc<Device>,
    swapchain: Swapchain,
    render_images: Vec<RenderImage>,
    graphics_queue: vk::Queue,

    frames: Vec<FrameData>,
    frame_number: usize,

    vma_allocator: vma::Allocator,
}

impl VulkanEngine {
    pub fn new(window: Arc<Window>) -> anyhow::Result<Self> {
        let (instance, device) = Self::setup_vulkan(window.clone())?;

        // Create an allocator.
        let allocator_options = vma::AllocatorOptions::new(
            instance.as_ref().as_ref(),
            device.as_ref(),
            *device.physical_device().as_ref(),
        );
        let vma_allocator = unsafe { vma::Allocator::new(&allocator_options) }.unwrap();

        let (graphics_queue_index, graphics_queue) = device.get_queue(QueueType::Graphics)?;

        let PhysicalSize {
            width: window_width,
            height: window_height,
        } = window.inner_size();

        let (swapchain, render_images) = Self::create_swapchain(
            instance.clone(),
            device.clone(),
            window_width,
            window_height,
        )?;

        let frames = Self::setup_framedata(
            device.clone(),
            graphics_queue_index as _,
            render_images.len(),
        )?;

        Ok(Self {
            window,
            instance,
            device,
            vma_allocator,
            swapchain,
            render_images,
            graphics_queue,
            frame_number: 0,
            frames,
        })
    }

    pub fn draw(&mut self) -> anyhow::Result<()> {
        let current_frame = self.get_current_frame();

        unsafe {
            self.device.wait_for_fences(
                &[current_frame.render_fence],
                true,
                Duration::from_secs(1).as_nanos() as _,
            )?;

            self.device.reset_fences(&[current_frame.render_fence])?;

            let (swapchain_image_index, err) = self.device.acquire_next_image_khr(
                *self.swapchain.as_ref(),
                // use a large timeout to avoid spurious TIMEOUT results on some platforms
                u64::MAX,
                current_frame.swapchain_semaphore,
                vk::Fence::null(),
            )?;

            // Handle common success codes: SUCCESS and SUBOPTIMAL are fine.
            // If we receive a TIMEOUT, skip this frame instead of failing the whole render loop.
            if matches!(err, vk::SuccessCode::TIMEOUT) {
                eprintln!("acquire_next_image_khr timed out, skipping frame");
                return Ok(());
            }

            if !matches!(
                err,
                vk::SuccessCode::SUCCESS | vk::SuccessCode::SUBOPTIMAL_KHR
            ) {
                return Err(anyhow::anyhow!(
                    "Failed acquiring next swapchain image, {}",
                    err
                ));
            }

            let current_image = &self.render_images[swapchain_image_index as usize];
            let cmd = current_frame.command_buffer;

            self.device
                .reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty())?;

            self.device.begin_command_buffer(
                cmd,
                &vk::CommandBufferBeginInfo::builder()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )?;

            //make the swapchain image into writeable mode before rendering
            transition_image(
                self.device.clone(),
                cmd,
                current_image.image,
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::GENERAL,
            );

            let flash = (self.frame_number as f32 / 120f32).sin().abs();
            let clear_value = vk::ClearColorValue {
                float32: { [0.0f32, 0.0f32, flash, 1.0f32] },
            };

            let clear_range = image_subresource_range(vk::ImageAspectFlags::COLOR);

            self.device.cmd_clear_color_image(
                cmd,
                current_image.image,
                vk::ImageLayout::GENERAL,
                &clear_value,
                &[clear_range],
            );

            // Make the swapchain image into presentable mode
            transition_image(
                self.device.clone(),
                cmd,
                current_image.image,
                vk::ImageLayout::GENERAL,
                vk::ImageLayout::PRESENT_SRC_KHR,
            );

            // Finalize the command buffer (we can no longer add commands, but it can now be executed)
            self.device.end_command_buffer(cmd)?;

            //prepare the submission to the queue.
            //we want to wait on the _presentSemaphore, as that semaphore is signaled when the swapchain is ready
            //we will signal the _renderSemaphore, to signal that rendering has finished

            let cmd_info = [vk::CommandBufferSubmitInfo::builder().command_buffer(cmd)];

            let wait_info = [vk::SemaphoreSubmitInfo::builder()
                .semaphore(current_frame.swapchain_semaphore)
                .stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
                .value(1)];
            let signal_info = [vk::SemaphoreSubmitInfo::builder()
                .semaphore(current_image.semaphore)
                .stage_mask(vk::PipelineStageFlags2::ALL_GRAPHICS)
                .value(1)];

            let submit_info = vk::SubmitInfo2::builder()
                .command_buffer_infos(&cmd_info)
                .signal_semaphore_infos(&signal_info)
                .wait_semaphore_infos(&wait_info);

            //submit command buffer to the queue and execute it.
            // _renderFence will now block until the graphic commands finish execution
            self.device.queue_submit2(
                self.graphics_queue,
                &[submit_info],
                current_frame.render_fence,
            )?;

            // Present the image back to the swapchain so it becomes available for acquisition again.
            let wait_semaphores = [current_image.semaphore];
            let swapchains = [*self.swapchain.as_ref()];
            let image_indices = [swapchain_image_index];

            let present_info = vk::PresentInfoKHR::builder()
                .wait_semaphores(&wait_semaphores)
                .swapchains(&swapchains)
                .image_indices(&image_indices);

            // queue_present_khr is provided by the swapchain extension trait.
            self.device
                .queue_present_khr(self.graphics_queue, &present_info)?;
        }

        self.frame_number += 1;

        Ok(())
    }

    // UTILITY

    fn setup_vulkan(window: Arc<Window>) -> anyhow::Result<(Arc<Instance>, Arc<Device>)> {
        let instance = InstanceBuilder::new(Some(window.clone()))
            .app_name("vk-guide example")
            .engine_name("vulkanalia-bootstrap")
            .request_validation_layers(true)
            .minimum_instance_version(Version::new(1, 3, 0))
            .require_api_version(Version::new(1, 3, 0))
            .build()?;

        let features12 = vk::PhysicalDeviceVulkan12Features::builder()
            .buffer_device_address(true)
            .descriptor_indexing(true);

        let features13 = vk::PhysicalDeviceVulkan13Features::builder()
            .synchronization2(true)
            .dynamic_rendering(true);

        let physical_device = PhysicalDeviceSelector::new(instance.clone())
            .preferred_device_type(PreferredDeviceType::Discrete)
            .add_required_extension_feature(*features12)
            .add_required_extension_feature(*features13)
            .select()?;

        let device = Arc::new(DeviceBuilder::new(physical_device, instance.clone()).build()?);

        Ok((instance, device))
    }

    fn create_swapchain(
        instance: Arc<Instance>,
        device: Arc<Device>,
        width: u32,
        height: u32,
    ) -> anyhow::Result<(Swapchain, Vec<RenderImage>)> {
        let swapchain_builder = SwapchainBuilder::new(instance.clone(), device.clone())
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
            .use_default_present_modes();

        let swapchain = swapchain_builder.build()?;
        let swapchain_images = swapchain.get_images()?;
        let swachain_image_views = swapchain.get_image_views()?;

        let render_images = swapchain_images
            .iter()
            .zip(swachain_image_views)
            .map(|(image, view)| {
                let image = *image;

                let semaphore =
                    unsafe { device.create_semaphore(&vk::SemaphoreCreateInfo::default(), None)? };

                Ok(RenderImage {
                    image,
                    view,
                    semaphore,
                })
            })
            .collect::<anyhow::Result<Vec<RenderImage>>>()?;

        Ok((swapchain, render_images))
    }

    fn setup_framedata(
        device: Arc<Device>,
        graphics_queue_index: u32,
        frame_overlap: usize,
    ) -> anyhow::Result<Vec<FrameData>> {
        //create a command pool for commands submitted to the graphics queue.
        //we also want the pool to allow for resetting of individual command buffers
        let command_pool_info = vk::CommandPoolCreateInfo::builder()
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
            .queue_family_index(graphics_queue_index);

        let frames = (0..frame_overlap)
            .map(|_| {
                unsafe {
                    let command_pool = device.create_command_pool(&command_pool_info, None)?;

                    // allocate the default command buffer that we will use for rendering
                    let cmd_alloc_info = vk::CommandBufferAllocateInfo::builder()
                        .command_pool(command_pool)
                        .command_buffer_count(1)
                        .level(vk::CommandBufferLevel::PRIMARY);

                    let command_buffer = *device
                        .allocate_command_buffers(&cmd_alloc_info)?
                        .first()
                        .ok_or(anyhow::anyhow!("No command buffer allocated"))?;

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
                }
            })
            .collect::<anyhow::Result<Vec<FrameData>>>()?;

        Ok(frames)
    }

    fn get_current_frame(&self) -> &FrameData {
        &self.frames[self.frame_number % self.render_images.len()]
    }

    fn destroy_swapchain(&self) {
        // Destroy image views via the swapchain helper before destroying the swapchain/device
        self.swapchain.destroy_image_views().ok();

        for render_image in &self.render_images {
            unsafe {
                self.device.destroy_semaphore(render_image.semaphore, None);
            }
        }

        self.swapchain.destroy();
    }
}

impl Drop for VulkanEngine {
    fn drop(&mut self) {
        unsafe { self.device.device_wait_idle().unwrap() };

        for frame in self.frames.iter_mut() {
            unsafe {
                self.device
                    .free_command_buffers(frame.command_pool, &[frame.command_buffer]);
                self.device.destroy_command_pool(frame.command_pool, None);
                self.device.destroy_fence(frame.render_fence, None);
                self.device
                    .destroy_semaphore(frame.swapchain_semaphore, None);
            }
        }

        self.destroy_swapchain();

        // Cleanup and destroy swapchain/device/instance
        self.device.destroy();
        self.instance.destroy();
    }
}
