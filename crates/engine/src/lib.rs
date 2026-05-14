pub mod camera;
pub mod car_renderer;
pub mod core;
pub mod egui_renderer;
pub mod frustum_cull;
pub mod gpu_pipeline_stats;
pub mod gpu_resources;
pub mod gpu_timestamps;
pub mod pipeline;
pub mod sdf;
pub mod tile_cache;

use std::collections::HashSet;
use std::sync::Arc;
use std::time::Instant;

use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::{fmt, EnvFilter};
use vulkanalia_bootstrap::Device as BootstrapDevice;
use winit::application::ApplicationHandler;
use winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::window::{Window, WindowAttributes, WindowId};

use crate::camera::Camera2D;
use crate::core::Core;
use crate::gpu_resources::GpuImage;

// ---------------------------------------------------------------------------
// Public re-exports for convenience
// ---------------------------------------------------------------------------

pub use egui;
pub use vulkanalia::{vk, vk::DeviceV1_0, vk::DeviceV1_3, vk::Handle, vk::HasBuilder};
pub use vulkanalia_bootstrap::Device as VkDevice;
pub use vulkanalia_vma as vma;
pub use winit;

pub use crate::core::transition_image;
pub use crate::pipeline::{
    destroy_pipeline_cache, init_pipeline_cache, write_storage_buffers, ComputePass,
};

// ---------------------------------------------------------------------------
// App trait — implemented by the game
// ---------------------------------------------------------------------------

/// Context passed to App callbacks. Provides access to the Vulkan device,
/// allocator, draw image, and input state.
pub struct EngineContext<'a> {
    pub device: &'a Arc<BootstrapDevice>,
    pub allocator: &'a vma::Allocator,
    pub draw_image: &'a GpuImage,
    pub input: &'a InputState,
    pub camera: &'a Camera2D,
    pub dt: f32,
    pub window_width: u32,
    pub window_height: u32,
    pub frame_number: u64,
    pub window: Option<&'a Window>,
    pub timestamp_period: f32,
    pub graphics_queue: vk::Queue,
    pub graphics_queue_family: u32,
}

pub trait App {
    /// Called once after Vulkan is initialized.
    fn init(&mut self, ctx: &EngineContext) -> anyhow::Result<()>;

    /// Called when the window is resized. Recreate any size-dependent resources.
    fn resize(&mut self, ctx: &EngineContext) -> anyhow::Result<()>;

    /// Called every frame. Record GPU commands into `cmd`.
    /// The draw image is already in GENERAL layout, ready for compute writes.
    fn render(&mut self, ctx: &EngineContext, cmd: vk::CommandBuffer) -> anyhow::Result<()>;

    /// Called every frame to build egui UI. Override to add panels, windows, etc.
    fn ui(&mut self, _egui_ctx: &egui::Context, _ctx: &EngineContext) {}

    /// Called on shutdown, before Vulkan resources are destroyed.
    fn shutdown(&mut self, _ctx: &EngineContext) {}
}

// ---------------------------------------------------------------------------
// Input state
// ---------------------------------------------------------------------------

#[derive(Debug, Default)]
pub struct InputState {
    pub mouse_x: f64,
    pub mouse_y: f64,
    pub mouse_dx: f64,
    pub mouse_dy: f64,
    pub scroll_delta: f32,
    pub left_mouse: bool,
    pub right_mouse: bool,
    pub middle_mouse: bool,
    /// True for one frame when left mouse was just pressed.
    pub left_mouse_pressed: bool,
    /// True for one frame when right mouse was just pressed.
    pub right_mouse_pressed: bool,
    /// True for one frame when Escape was just pressed.
    pub escape_pressed: bool,
    /// Keys pressed this frame (one-shot, cleared each frame).
    pub keys_pressed: HashSet<winit::keyboard::Key>,
    /// Keys currently held down.
    pub keys_held: HashSet<winit::keyboard::Key>,
}

impl InputState {
    fn end_frame(&mut self) {
        self.mouse_dx = 0.0;
        self.mouse_dy = 0.0;
        self.scroll_delta = 0.0;
        self.left_mouse_pressed = false;
        self.right_mouse_pressed = false;
        self.escape_pressed = false;
        self.keys_pressed.clear();
    }
}

// ---------------------------------------------------------------------------
// Engine runner (winit ApplicationHandler)
// ---------------------------------------------------------------------------

struct EngineRunner<A: App> {
    app: A,
    core: Option<Core>,
    window: Option<Arc<Window>>,
    input: InputState,
    camera: Camera2D,
    last_frame_time: Option<Instant>,
    dt: f32,
    // egui state
    egui_ctx: egui::Context,
    egui_state: Option<egui_winit::State>,
    egui_renderer: Option<egui_renderer::EguiRenderer>,
    /// Cached egui output for the current frame (set before draw_frame, consumed inside).
    egui_clipped_primitives: Vec<egui::ClippedPrimitive>,
    egui_textures_delta: egui::TexturesDelta,
}

impl<A: App> EngineRunner<A> {
    fn new(app: A) -> Self {
        Self {
            app,
            core: None,
            window: None,
            input: InputState::default(),
            camera: Camera2D::default(),
            last_frame_time: None,
            dt: 0.0,
            egui_ctx: egui::Context::default(),
            egui_state: None,
            egui_renderer: None,
            egui_clipped_primitives: Vec::new(),
            egui_textures_delta: egui::TexturesDelta::default(),
        }
    }
}

impl<A: App> ApplicationHandler for EngineRunner<A> {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window = Arc::new(
            event_loop
                .create_window(WindowAttributes::default().with_title("Traffic Simulation"))
                .expect("Failed to create window"),
        );

        let core = Core::new(window.clone()).expect("Failed to initialize Vulkan");

        // Initialize egui
        let egui_state = egui_winit::State::new(
            self.egui_ctx.clone(),
            egui::ViewportId::ROOT,
            &window,
            None,
            None,
            None,
        );
        let egui_renderer = egui_renderer::EguiRenderer::new(
            core.device(),
            core.allocator(),
            core.draw_image().format,
        )
        .expect("Failed to create egui renderer");

        self.egui_state = Some(egui_state);
        self.egui_renderer = Some(egui_renderer);
        self.core = Some(core);
        self.window = Some(window);

        // Build context inline to avoid borrow conflict with self.app
        self.last_frame_time = Some(Instant::now());
        let core = self.core.as_ref().unwrap();
        let ctx = EngineContext {
            device: core.device(),
            allocator: core.allocator(),
            draw_image: core.draw_image(),
            input: &self.input,
            camera: &self.camera,
            dt: 0.0,
            window_width: core.window_extent().width,
            window_height: core.window_extent().height,
            frame_number: 0,
            window: self.window.as_deref(),
            timestamp_period: core.timestamp_period(),
            graphics_queue: core.graphics_queue(),
            graphics_queue_family: core.graphics_queue_family(),
        };
        self.app.init(&ctx).expect("App::init failed");
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        // Forward events to egui first
        let egui_consumed = if let (Some(egui_state), Some(window)) =
            (self.egui_state.as_mut(), self.window.as_ref())
        {
            let response = egui_state.on_window_event(window, &event);
            response.consumed
        } else {
            false
        };

        match event {
            WindowEvent::CloseRequested => {
                if let Some(core) = &self.core {
                    unsafe {
                        core.device().device_wait_idle().ok();
                    }
                    let ctx = EngineContext {
                        device: core.device(),
                        allocator: core.allocator(),
                        draw_image: core.draw_image(),
                        input: &self.input,
                        camera: &self.camera,
                        dt: self.dt,
                        window_width: core.window_extent().width,
                        window_height: core.window_extent().height,
                        frame_number: 0,
                        window: None,
                        timestamp_period: core.timestamp_period(),
                        graphics_queue: core.graphics_queue(),
                        graphics_queue_family: core.graphics_queue_family(),
                    };
                    self.app.shutdown(&ctx);
                    if let Some(mut r) = self.egui_renderer.take() {
                        r.destroy(core.device(), core.allocator());
                    }
                }
                event_loop.exit();
            }

            WindowEvent::Resized(size) => {
                if let Some(core) = self.core.as_mut() {
                    core.handle_resize(size.width, size.height)
                        .expect("Resize failed");
                    let ctx = EngineContext {
                        device: core.device(),
                        allocator: core.allocator(),
                        draw_image: core.draw_image(),
                        input: &self.input,
                        camera: &self.camera,
                        dt: self.dt,
                        window_width: core.window_extent().width,
                        window_height: core.window_extent().height,
                        frame_number: 0,
                        window: None,
                        timestamp_period: core.timestamp_period(),
                        graphics_queue: core.graphics_queue(),
                        graphics_queue_family: core.graphics_queue_family(),
                    };
                    self.app.resize(&ctx).expect("App::resize failed");
                }
            }

            WindowEvent::CursorMoved { position, .. } if !egui_consumed => {
                self.input.mouse_dx += position.x - self.input.mouse_x;
                self.input.mouse_dy += position.y - self.input.mouse_y;
                self.input.mouse_x = position.x;
                self.input.mouse_y = position.y;
            }

            // Always track cursor position for camera zoom-at-cursor, even when
            // egui has focus. Only mouse *actions* are suppressed.
            WindowEvent::CursorMoved { position, .. } => {
                self.input.mouse_x = position.x;
                self.input.mouse_y = position.y;
            }

            WindowEvent::MouseInput { state, button, .. } if !egui_consumed => {
                let pressed = state == ElementState::Pressed;
                match button {
                    MouseButton::Left => {
                        if pressed && !self.input.left_mouse {
                            self.input.left_mouse_pressed = true;
                        }
                        self.input.left_mouse = pressed;
                    }
                    MouseButton::Right => {
                        if pressed && !self.input.right_mouse {
                            self.input.right_mouse_pressed = true;
                        }
                        self.input.right_mouse = pressed;
                    }
                    MouseButton::Middle => self.input.middle_mouse = pressed,
                    _ => {}
                }
            }

            WindowEvent::MouseWheel { delta, .. } if !egui_consumed => {
                self.input.scroll_delta += match delta {
                    MouseScrollDelta::LineDelta(_, y) => y,
                    MouseScrollDelta::PixelDelta(p) => p.y as f32 / 100.0,
                };
            }

            WindowEvent::KeyboardInput { event, .. } if !egui_consumed => {
                use winit::keyboard::{Key, NamedKey};
                if event.state == ElementState::Pressed {
                    if event.logical_key == Key::Named(NamedKey::Escape) {
                        self.input.escape_pressed = true;
                    }
                    self.input.keys_pressed.insert(event.logical_key.clone());
                    self.input.keys_held.insert(event.logical_key.clone());
                } else {
                    self.input.keys_held.remove(&event.logical_key);
                }
            }

            WindowEvent::RedrawRequested => {
                if let Some(core) = self.core.as_mut() {
                    // Delta time
                    let now = Instant::now();
                    self.dt = self
                        .last_frame_time
                        .map(|t| now.duration_since(t).as_secs_f32())
                        .unwrap_or(1.0 / 60.0);
                    self.last_frame_time = Some(now);

                    // Camera: middle-mouse pan
                    let window_extent = core.window_extent();
                    if self.input.middle_mouse {
                        self.camera.pan_by_pixels(
                            self.input.mouse_dx,
                            self.input.mouse_dy,
                            window_extent.width,
                            window_extent.height,
                        );
                    }

                    // Camera: arrow key panning
                    {
                        use winit::keyboard::{Key, NamedKey};
                        let pan_speed = 2.0 / self.camera.zoom * self.dt;
                        if self
                            .input
                            .keys_held
                            .contains(&Key::Named(NamedKey::ArrowLeft))
                        {
                            self.camera.position.x -= pan_speed;
                        }
                        if self
                            .input
                            .keys_held
                            .contains(&Key::Named(NamedKey::ArrowRight))
                        {
                            self.camera.position.x += pan_speed;
                        }
                        if self
                            .input
                            .keys_held
                            .contains(&Key::Named(NamedKey::ArrowDown))
                        {
                            self.camera.position.y -= pan_speed;
                        }
                        if self
                            .input
                            .keys_held
                            .contains(&Key::Named(NamedKey::ArrowUp))
                        {
                            self.camera.position.y += pan_speed;
                        }
                    }

                    // Camera: scroll zoom centered on cursor
                    if self.input.scroll_delta != 0.0 {
                        let cursor_world = self.camera.screen_to_world(
                            glam::Vec2::new(self.input.mouse_x as f32, self.input.mouse_y as f32),
                            glam::Vec2::new(
                                window_extent.width as f32,
                                window_extent.height as f32,
                            ),
                        );
                        self.camera.zoom_at(self.input.scroll_delta, cursor_world);
                    }

                    // --- egui frame ---
                    let egui_input = self
                        .egui_state
                        .as_mut()
                        .unwrap()
                        .take_egui_input(self.window.as_ref().unwrap());

                    let input = &self.input;
                    let camera = &self.camera;
                    let dt = self.dt;
                    let frame_number = core.frame_number() as u64;
                    let device_ref = core.device().clone();
                    let allocator_ref: *const vma::Allocator = core.allocator();
                    let draw_image_ref: *const GpuImage = core.draw_image();
                    let window_ref: Option<&Window> = self.window.as_deref();
                    let timestamp_period = core.timestamp_period();
                    let gfx_queue = core.graphics_queue();
                    let gfx_queue_family = core.graphics_queue_family();

                    // Run egui — app builds its UI here
                    let full_output = self.egui_ctx.run(egui_input, |egui_ctx| {
                        let ctx = EngineContext {
                            device: &device_ref,
                            allocator: unsafe { &*allocator_ref },
                            draw_image: unsafe { &*draw_image_ref },
                            input,
                            camera,
                            dt,
                            window_width: window_extent.width,
                            window_height: window_extent.height,
                            frame_number,
                            window: window_ref,
                            timestamp_period,
                            graphics_queue: gfx_queue,
                            graphics_queue_family: gfx_queue_family,
                        };
                        self.app.ui(egui_ctx, &ctx);
                    });

                    let pixels_per_point = full_output.pixels_per_point;
                    self.egui_clipped_primitives = self
                        .egui_ctx
                        .tessellate(full_output.shapes, pixels_per_point);
                    self.egui_textures_delta = full_output.textures_delta;

                    self.egui_state.as_mut().unwrap().handle_platform_output(
                        self.window.as_ref().unwrap(),
                        full_output.platform_output,
                    );

                    let screen_size = [window_extent.width, window_extent.height];
                    let egui_prims = &self.egui_clipped_primitives;
                    let egui_tex_delta = &self.egui_textures_delta;
                    let egui_renderer = self.egui_renderer.as_mut().unwrap();
                    let frame_idx = core.frame_number();

                    core.draw_frame(|device, cmd, _draw_image| {
                        let ctx = EngineContext {
                            device: &device_ref,
                            allocator: unsafe { &*allocator_ref },
                            draw_image: unsafe { &*draw_image_ref },
                            input,
                            camera,
                            dt,
                            window_width: window_extent.width,
                            window_height: window_extent.height,
                            frame_number,
                            window: window_ref,
                            timestamp_period,
                            graphics_queue: gfx_queue,
                            graphics_queue_family: gfx_queue_family,
                        };
                        self.app.render(&ctx, cmd).expect("App::render failed");

                        // egui rendering (after scene, before blit)
                        egui_renderer
                            .update_textures(
                                device,
                                unsafe { &*allocator_ref },
                                cmd,
                                egui_tex_delta,
                            )
                            .expect("egui texture update failed");
                        egui_renderer.paint(
                            device,
                            cmd,
                            unsafe { &*draw_image_ref },
                            egui_prims,
                            screen_size,
                            pixels_per_point,
                            frame_idx,
                        );
                    })
                    .expect("draw_frame failed");

                    self.input.end_frame();

                    if let Some(w) = &self.window {
                        w.request_redraw();
                    }
                }
            }

            _ => {}
        }
    }
}

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

/// Run the application. This blocks until the window is closed.
pub fn run(app: impl App + 'static) -> anyhow::Result<()> {
    tracing_subscriber::registry()
        .with(fmt::layer())
        .with(EnvFilter::from_default_env())
        .init();

    let event_loop = EventLoop::new()?;
    let mut runner = EngineRunner::new(app);
    event_loop.run_app(&mut runner)?;
    Ok(())
}
