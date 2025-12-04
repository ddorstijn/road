use std::sync::Arc;

use imgui::Context;
use imgui_winit_support::{HiDpiMode, WinitPlatform};
use vulkano::{
    Validated, VulkanError,
    command_buffer::{
        AutoCommandBufferBuilder, BlitImageInfo, CommandBufferUsage, PrimaryAutoCommandBuffer,
        RenderingAttachmentInfo, RenderingInfo, allocator::StandardCommandBufferAllocator,
    },
    descriptor_set::allocator::StandardDescriptorSetAllocator,
    device::{Device, Queue},
    format::Format,
    image::{Image, ImageCreateInfo, ImageUsage, view::ImageView},
    memory::allocator::{AllocationCreateInfo, StandardMemoryAllocator},
    render_pass::{AttachmentLoadOp, AttachmentStoreOp},
    swapchain::{Surface, Swapchain, SwapchainPresentInfo, acquire_next_image},
    sync::{self, GpuFuture},
};
use winit::{
    application::ApplicationHandler,
    dpi::PhysicalSize,
    event::WindowEvent,
    event_loop::ActiveEventLoop,
    window::{Window, WindowId},
};

use crate::{
    component::GameComponent,
    imgui_renderer::{self, ImguiVulkanoRenderer},
    vulkan_util::{create_swapchain, init_vulkan},
};

pub struct Gui {
    context: imgui::Context,
    platform: WinitPlatform,
    renderer: imgui_renderer::ImguiVulkanoRenderer,
}

impl Gui {
    pub fn new(window: Arc<Window>, ctx: &VulkanContext) -> Self {
        let mut imgui = Context::create();
        let mut platform = WinitPlatform::new(&mut imgui);
        platform.attach_window(imgui.io_mut(), &window, HiDpiMode::Default);

        let renderer = ImguiVulkanoRenderer::new(
            &mut imgui,
            ctx.device.clone(),
            ctx.queue.clone(),
            ctx.command_buffer_allocator.clone(),
            ctx.descriptor_set_allocator.clone(),
            ctx.memory_allocator.clone(),
            ctx.swapchain.image_format(),
        );

        Self {
            context: imgui,
            platform,
            renderer,
        }
    }
}

pub struct VulkanContext {
    pub device: Arc<Device>,
    pub queue: Arc<Queue>,
    pub memory_allocator: Arc<StandardMemoryAllocator>,
    pub descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
    pub command_buffer_allocator: Arc<StandardCommandBufferAllocator>,

    pub swapchain: Arc<Swapchain>,
    pub swapchain_images: Vec<Arc<ImageView>>,
    pub swapchain_suboptimal: bool,

    // Resolution Control
    pub draw_image: Arc<ImageView>,
    internal_resolution: [u32; 2],
}

impl VulkanContext {
    pub fn new(window: Arc<Window>, event_loop: &ActiveEventLoop) -> Self {
        let (instance, device, queue) = init_vulkan(event_loop);

        let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));
        let descriptor_set_allocator = Arc::new(StandardDescriptorSetAllocator::new(
            device.clone(),
            Default::default(),
        ));
        let command_buffer_allocator = Arc::new(StandardCommandBufferAllocator::new(
            device.clone(),
            Default::default(),
        ));

        let internal_resolution = [1920, 1080];
        let draw_image = Self::create_draw_image(&memory_allocator, internal_resolution);

        let surface = Surface::from_window(instance.clone(), window.clone()).unwrap();
        let window_size = window.inner_size();
        let (swapchain, swapchain_images) =
            create_swapchain(window_size, surface, device.clone(), None);

        Self {
            device,
            queue,
            memory_allocator,
            descriptor_set_allocator,
            command_buffer_allocator,
            swapchain,
            swapchain_images,
            swapchain_suboptimal: false,
            internal_resolution,
            draw_image,
        }
    }

    pub fn recreate_swapchain(&mut self, new_size: PhysicalSize<u32>) {
        let (new_swapchain, new_images) = create_swapchain(
            new_size,
            self.swapchain.surface().clone(),
            self.device.clone(),
            Some(self.swapchain.clone()),
        );

        self.swapchain = new_swapchain;
        self.swapchain_images = new_images;
        self.swapchain_suboptimal = false;
    }

    fn create_draw_image(
        allocator: &Arc<StandardMemoryAllocator>,
        extent: [u32; 2],
    ) -> Arc<ImageView> {
        let [width, height] = extent;

        let image = Image::new(
            allocator.clone(),
            ImageCreateInfo {
                extent: [width, height, 1],
                // High precision format for internal rendering (HDR capable)
                format: Format::R16G16B16A16_SFLOAT,
                usage: ImageUsage::TRANSFER_SRC
                    | ImageUsage::TRANSFER_DST
                    | ImageUsage::STORAGE
                    | ImageUsage::COLOR_ATTACHMENT,
                ..Default::default()
            },
            AllocationCreateInfo::default(),
        )
        .unwrap();

        ImageView::new_default(image).unwrap()
    }

    pub fn set_internal_resolution(&mut self, resolution: [u32; 2]) {
        if resolution[0] == 0 || resolution[1] == 0 {
            return;
        }

        self.internal_resolution = resolution;
        self.draw_image = Self::create_draw_image(&self.memory_allocator, resolution);
    }
}

pub(crate) struct EngineRuntime {
    window_title: String,
    components: Vec<Box<dyn GameComponent>>,
    context: Option<VulkanContext>,
    window: Option<Arc<Window>>,
    gui: Option<Gui>,
    frame_sync: Option<Box<dyn GpuFuture>>,
}

impl EngineRuntime {
    pub fn new(window_title: String, components: Vec<Box<dyn GameComponent + 'static>>) -> Self {
        Self {
            window_title,
            components,
            context: None,
            window: None,
            gui: None,
            frame_sync: None,
        }
    }

    /// Helper to render game components.
    /// Takes specific references instead of `&mut self` to avoid borrow conflicts.
    fn render_components(
        components: &mut [Box<dyn GameComponent>],
        ctx: &VulkanContext,
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
    ) {
        for component in components {
            component.render(ctx, builder);
        }
    }

    /// Helper to render UI.
    fn render_ui(
        gui: &mut Gui,
        window: &Window,
        components: &mut [Box<dyn GameComponent>],
        ctx: &VulkanContext,
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        current_image: Arc<ImageView>,
    ) {
        gui.platform
            .prepare_frame(gui.context.io_mut(), window)
            .unwrap();

        let ui = gui.context.new_frame();
        for component in components {
            component.ui(ctx, ui);
        }

        let attachment_info = RenderingAttachmentInfo {
            load_op: AttachmentLoadOp::Load,
            store_op: AttachmentStoreOp::Store,
            ..RenderingAttachmentInfo::image_view(current_image.clone())
        };

        builder
            .begin_rendering(RenderingInfo {
                color_attachments: vec![Some(attachment_info)],
                ..Default::default()
            })
            .unwrap();

        gui.renderer.draw(builder, gui.context.render());
        builder.end_rendering().unwrap();
    }
}

impl ApplicationHandler for EngineRuntime {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }

        let window = Arc::new(
            event_loop
                .create_window(Window::default_attributes().with_title(&self.window_title))
                .unwrap(),
        );
        let ctx = VulkanContext::new(window.clone(), event_loop);
        let gui = Gui::new(window.clone(), &ctx);

        for component in &mut self.components {
            component.start(&ctx);
        }

        self.frame_sync = Some(sync::now(ctx.device.clone()).boxed());
        self.gui = Some(gui);
        self.context = Some(ctx);
        self.window = Some(window);
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        window_id: WindowId,
        event: WindowEvent,
    ) {
        if let Some(gui) = &mut self.gui {
            let window = self.window.as_ref().unwrap();

            let winit_event: winit::event::Event<()> = winit::event::Event::WindowEvent {
                window_id,
                event: event.clone(),
            };
            gui.platform
                .handle_event(gui.context.io_mut(), &window.clone(), &winit_event);
        }

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::RedrawRequested => {
                let window = self.window.as_ref().unwrap();
                let window_size = window.inner_size();

                if window_size.width == 0 || window_size.height == 0 {
                    return;
                }

                self.frame_sync.as_mut().unwrap().cleanup_finished();

                if let Some(ctx) = self.context.as_mut() {
                    if ctx.swapchain_suboptimal {
                        ctx.recreate_swapchain(window_size);
                    }
                }

                let ctx = self.context.as_mut().unwrap();

                let (image_index, suboptimal, acquire_future) = {
                    match acquire_next_image(ctx.swapchain.clone(), None).map_err(Validated::unwrap)
                    {
                        Ok(r) => r,
                        Err(VulkanError::OutOfDate) => {
                            ctx.swapchain_suboptimal = true;
                            return;
                        }
                        Err(e) => panic!("failed to acquire next image: {e}"),
                    }
                };

                if suboptimal {
                    ctx.swapchain_suboptimal = true;
                }

                let command_buffer = {
                    let gui = self.gui.as_mut().unwrap();

                    let mut builder = AutoCommandBufferBuilder::primary(
                        ctx.command_buffer_allocator.clone(),
                        ctx.queue.queue_family_index(),
                        CommandBufferUsage::OneTimeSubmit,
                    )
                    .unwrap();

                    let current_image = ctx.swapchain_images[image_index as usize].clone();

                    Self::render_components(&mut self.components, ctx, &mut builder);

                    builder
                        .blit_image(BlitImageInfo::images(
                            ctx.draw_image.image().clone(),
                            current_image.image().clone(),
                        ))
                        .unwrap();

                    Self::render_ui(
                        gui,
                        window,
                        &mut self.components,
                        ctx,
                        &mut builder,
                        current_image.clone(),
                    );

                    builder.build().unwrap()
                };

                let future = self
                    .frame_sync
                    .take()
                    .unwrap()
                    .join(acquire_future)
                    .then_execute(ctx.queue.clone(), command_buffer)
                    .unwrap()
                    .then_swapchain_present(
                        ctx.queue.clone(),
                        SwapchainPresentInfo::swapchain_image_index(
                            ctx.swapchain.clone(),
                            image_index,
                        ),
                    )
                    .then_signal_fence_and_flush();

                self.frame_sync = Some(match future.map_err(Validated::unwrap) {
                    Ok(future) => future.boxed(),
                    Err(VulkanError::OutOfDate) => {
                        ctx.swapchain_suboptimal = true;
                        sync::now(ctx.device.clone()).boxed()
                    }
                    Err(e) => {
                        println!("failed to flush future: {e}");
                        sync::now(ctx.device.clone()).boxed()
                    }
                });
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        self.window.as_ref().unwrap().request_redraw();
    }
}
