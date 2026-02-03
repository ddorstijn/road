use std::{ffi::CStr, mem::ManuallyDrop, sync::Arc, time::Duration};

use crate::{
    vk_type::AllocatedImage,
    vk_util::{copy_image_to_image, transition_image},
};
use vulkanalia::{
    Version, bytecode::Bytecode, prelude::v1_4::*, vk::KhrSwapchainExtensionDeviceCommands,
};
use vulkanalia_bootstrap::{
    Device, DeviceBuilder, Instance, InstanceBuilder, PhysicalDeviceSelector, PreferredDeviceType,
    QueueType, Swapchain, SwapchainBuilder,
};
use vulkanalia_vma::{self as vma, Alloc};
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
struct ComputePipeline {
    pipeline: vk::Pipeline,
    pipeline_layout: vk::PipelineLayout,
    descriptor_set_layout: vk::DescriptorSetLayout,
    descriptor_set: vk::DescriptorSet,
    descriptor_pool: vk::DescriptorPool,
}

#[derive(Debug)]
pub struct VulkanEngine {
    pub window: Arc<Window>,
    instance: Arc<Instance>,
    device: Arc<Device>,
    swapchain: Swapchain,
    render_images: Vec<RenderImage>,
    draw_image: AllocatedImage,
    graphics_queue: vk::Queue,

    gradient_compute: ComputePipeline,
    frames: Vec<FrameData>,
    frame_number: usize,
    vma_allocator: ManuallyDrop<vma::Allocator>,
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

        //draw image size will match the window
        let draw_image_format = vk::Format::R16G16B16A16_SFLOAT;
        let draw_image_extent = vk::Extent3D::builder()
            .width(window_width)
            .height(window_height)
            .depth(1)
            .build();
        let draw_image_usage = vk::ImageUsageFlags::TRANSFER_SRC
            | vk::ImageUsageFlags::TRANSFER_DST
            | vk::ImageUsageFlags::STORAGE
            | vk::ImageUsageFlags::COLOR_ATTACHMENT;
        let draw_image_create_info = vk::ImageCreateInfo::builder()
            .image_type(vk::ImageType::_2D)
            .format(draw_image_format)
            .extent(draw_image_extent)
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(draw_image_usage);

        let allocation_options = vma::AllocationOptions::default();
        let (draw_image, allocation) =
            unsafe { vma_allocator.create_image(draw_image_create_info, &allocation_options) }
                .unwrap();
        let draw_image_view_create_info = vk::ImageViewCreateInfo::builder()
            .view_type(vk::ImageViewType::_2D)
            .image(draw_image)
            .format(draw_image_format)
            .subresource_range(
                vk::ImageSubresourceRange::builder()
                    .base_mip_level(0)
                    .level_count(1)
                    .base_array_layer(0)
                    .layer_count(1)
                    .aspect_mask(vk::ImageAspectFlags::COLOR),
            );

        let draw_image_view =
            unsafe { device.create_image_view(&draw_image_view_create_info, None) }?;

        let draw_image = AllocatedImage {
            image_format: draw_image_format,
            image_extent: draw_image_extent,
            image_view: draw_image_view,
            image: draw_image,
            allocation,
        };

        let gradient_compute = Self::create_gradient_compute_pipeline(device.clone(), &draw_image)?;

        Ok(Self {
            window,
            instance,
            device,
            vma_allocator: ManuallyDrop::new(vma_allocator),
            swapchain,
            render_images,
            draw_image,
            graphics_queue,
            gradient_compute,
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
            let draw_extent = vk::Extent2D::builder()
                .width(self.draw_image.image_extent.width)
                .height(self.draw_image.image_extent.height)
                .build();
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
                self.draw_image.image,
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::GENERAL,
            );

            self.draw_background(cmd);

            //transition the draw image and the swapchain image into their correct transfer layouts
            transition_image(
                self.device.clone(),
                cmd,
                self.draw_image.image,
                vk::ImageLayout::GENERAL,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            );

            transition_image(
                self.device.clone(),
                cmd,
                current_image.image,
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            );

            // execute a copy from the draw image into the swapchain
            copy_image_to_image(
                self.device.clone(),
                cmd,
                self.draw_image.image,
                current_image.image,
                draw_extent,
                self.swapchain.extent,
            );

            // set swapchain image layout to Present so we can show it on the screen
            transition_image(
                self.device.clone(),
                cmd,
                current_image.image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
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

    fn draw_background(&self, cmd: vk::CommandBuffer) {
        self.draw_gradient(cmd);
    }

    fn draw_gradient(&self, cmd: vk::CommandBuffer) {
        unsafe {
            self.device.cmd_bind_pipeline(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                self.gradient_compute.pipeline,
            );

            self.device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                self.gradient_compute.pipeline_layout,
                0,
                &[self.gradient_compute.descriptor_set],
                &[],
            );

            // Calculate work groups needed for the image
            let work_group_x = (self.draw_image.image_extent.width + 15) / 16;
            let work_group_y = (self.draw_image.image_extent.height + 15) / 16;

            self.device.cmd_dispatch(cmd, work_group_x, work_group_y, 1);
        }
    }

    // UTILITY

    fn create_gradient_compute_pipeline(
        device: Arc<Device>,
        draw_image: &AllocatedImage,
    ) -> anyhow::Result<ComputePipeline> {
        let shader_module = {
            let shader_content = include_bytes!("../assets/shaders/gradient.comp.spv");
            let bytecode = Bytecode::new(&shader_content[..]).unwrap();
            let info = vk::ShaderModuleCreateInfo::builder()
                .code(bytecode.code())
                .code_size(bytecode.code_size());

            unsafe { device.create_shader_module(&info, None) }
        }?;

        // Create descriptor set layout for the storage image
        let descriptor_set_layout_bindings = [vk::DescriptorSetLayoutBinding::builder()
            .binding(0)
            .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .build()];

        let descriptor_set_layout_create_info =
            vk::DescriptorSetLayoutCreateInfo::builder().bindings(&descriptor_set_layout_bindings);

        let descriptor_set_layout = unsafe {
            device.create_descriptor_set_layout(&descriptor_set_layout_create_info, None)?
        };

        // Create descriptor pool
        let descriptor_pool_sizes = [vk::DescriptorPoolSize::builder()
            .type_(vk::DescriptorType::STORAGE_IMAGE)
            .descriptor_count(1)
            .build()];

        let descriptor_pool_create_info = vk::DescriptorPoolCreateInfo::builder()
            .max_sets(1)
            .pool_sizes(&descriptor_pool_sizes);

        let descriptor_pool =
            unsafe { device.create_descriptor_pool(&descriptor_pool_create_info, None)? };

        // Allocate descriptor set
        let descriptor_set_layouts = [descriptor_set_layout];
        let descriptor_set_allocate_info = vk::DescriptorSetAllocateInfo::builder()
            .descriptor_pool(descriptor_pool)
            .set_layouts(&descriptor_set_layouts);

        let descriptor_set = unsafe {
            device
                .allocate_descriptor_sets(&descriptor_set_allocate_info)?
                .pop()
                .ok_or(anyhow::anyhow!("No descriptor set allocated"))?
        };

        // Update descriptor set with the draw image
        let image_info = [vk::DescriptorImageInfo::builder()
            .image_view(draw_image.image_view)
            .image_layout(vk::ImageLayout::GENERAL)
            .build()];

        let write_descriptor_set = vk::WriteDescriptorSet::builder()
            .dst_set(descriptor_set)
            .dst_binding(0)
            .dst_array_element(0)
            .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
            .image_info(&image_info);

        unsafe {
            device
                .update_descriptor_sets(&[*write_descriptor_set], &[] as &[vk::CopyDescriptorSet]);
        }

        // Create pipeline layout
        let descriptor_set_layouts = [descriptor_set_layout];
        let pipeline_layout_create_info =
            vk::PipelineLayoutCreateInfo::builder().set_layouts(&descriptor_set_layouts);

        let pipeline_layout =
            unsafe { device.create_pipeline_layout(&pipeline_layout_create_info, None)? };

        // Create compute pipeline
        let entry_point = CStr::from_bytes_with_nul(b"main\0").unwrap();
        let pipeline_stage_create_info = vk::PipelineShaderStageCreateInfo::builder()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(shader_module)
            .name(entry_point.to_bytes_with_nul());

        let pipeline_create_info = vk::ComputePipelineCreateInfo::builder()
            .layout(pipeline_layout)
            .stage(*pipeline_stage_create_info);

        let pipeline = unsafe {
            device
                .create_compute_pipelines(
                    vk::PipelineCache::null(),
                    &[*pipeline_create_info],
                    None,
                )?
                .0
                .pop()
                .ok_or(anyhow::anyhow!("No pipeline created"))?
        };

        // Destroy shader module as it's no longer needed after pipeline creation
        unsafe {
            device.destroy_shader_module(shader_module, None);
        }

        Ok(ComputePipeline {
            pipeline,
            pipeline_layout,
            descriptor_set_layout,
            descriptor_set,
            descriptor_pool,
        })
    }

    // UTILITY (continued)

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

        // Cleanup compute pipeline
        unsafe {
            self.device
                .destroy_descriptor_pool(self.gradient_compute.descriptor_pool, None);
            self.device
                .destroy_descriptor_set_layout(self.gradient_compute.descriptor_set_layout, None);
            self.device
                .destroy_pipeline(self.gradient_compute.pipeline, None);
            self.device
                .destroy_pipeline_layout(self.gradient_compute.pipeline_layout, None);
        }

        unsafe {
            self.vma_allocator
                .destroy_image(self.draw_image.image, self.draw_image.allocation);
        }
        unsafe {
            self.device
                .destroy_image_view(self.draw_image.image_view, None);
        }

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

        unsafe {
            ManuallyDrop::drop(&mut self.vma_allocator);
        }

        // Cleanup and destroy swapchain/device/instance
        self.device.destroy();
        self.instance.destroy();
    }
}
