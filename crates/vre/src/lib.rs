//! Chapter 1 of vk_guide

use bytemuck::Pod;
use std::sync::Arc;
use std::time::Duration;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::{EnvFilter, fmt};
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::window::{WindowAttributes, WindowId};

use crate::vulkan_engine::VulkanEngine;

pub mod vk {
    pub use vulkanalia::vk::{BufferUsageFlags, Format, ImageUsageFlags};
}

pub(crate) mod vk_type;
pub(crate) mod vk_util;
mod vulkan_engine;

#[derive(Debug)]
pub enum PipelineConfig {
    PerFrame,
    FixedTimestep(Duration),
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PassType {
    Render,
    Compute,
}

#[derive(Debug)]
pub(crate) struct PassDef {
    pub name: String,
    pub ty: PassType,
    pub shader_path: String,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    pub push_constants: Vec<u8>,
}

pub struct ShaderPass<'a, T: Pod> {
    pub name: &'a str,
    pub ty: PassType,
    pub shader: &'a str,
    pub inputs: &'a [&'a str],
    pub outputs: &'a [&'a str],
    pub constants: T,
}

#[derive(Debug)]
pub(crate) struct PipelineDef {
    pub name: String,
    pub config: PipelineConfig,
    pub passes: Vec<PassDef>,
}

#[derive(Debug)]
pub struct PipelineBuilder {
    passes: Vec<PassDef>,
}

impl PipelineBuilder {
    pub fn add_pass<T: Pod>(&mut self, pass: ShaderPass<T>) {
        self.passes.push(PassDef {
            name: pass.name.to_string(),
            ty: pass.ty,
            shader_path: pass.shader.to_string(),
            inputs: pass.inputs.iter().map(|s| s.to_string()).collect(),
            outputs: pass.outputs.iter().map(|s| s.to_string()).collect(),
            push_constants: bytemuck::bytes_of(&pass.constants).to_vec(),
        });
    }
}

#[derive(Debug)]
pub struct BufferDef {
    pub name: String,
    pub count: usize,
    pub usage: vk::BufferUsageFlags,
    pub initial_data: Option<Vec<u8>>,
}

#[derive(Clone, Copy, Debug)]
pub enum TextureSize {
    ViewportRelative(f32),
    Fixed(u32, u32),
    Native, // Only valid if path is set
}

pub struct TextureSource<'a> {
    pub path: Option<&'a str>,

    pub format: vk::Format,
    pub size: TextureSize,
    pub usage: vk::ImageUsageFlags,
}

// 3. Internal Storage (Owned)
#[derive(Debug)]
pub(crate) struct TextureDef {
    pub name: String,
    pub path: Option<String>, // Owned string
    pub format: vk::Format,
    pub size: TextureSize,
    pub usage: vk::ImageUsageFlags,
}

#[derive(Default, Debug)]
pub struct RenderApp {
    // We store raw bytes for initialization to keep the App struct non-generic
    buffer_defs: Vec<BufferDef>,
    texture_defs: Vec<TextureDef>,
    pipelines: Vec<PipelineDef>,
    vulkan_engine: Option<VulkanEngine>,
}

impl RenderApp {
    pub fn new() -> Self {
        tracing_subscriber::registry()
            .with(fmt::layer())
            .with(EnvFilter::from_default_env())
            .init();

        Self::default()
    }

    pub fn add_buffer<T: Pod>(
        mut self,
        name: &str,
        count: usize,
        usage: vk::BufferUsageFlags,
        init_fn: Option<fn(usize) -> Vec<T>>,
    ) -> Self {
        let initial_data = init_fn.map(|init| bytemuck::cast_slice(&init(count)).to_vec());

        self.buffer_defs.push(BufferDef {
            name: name.to_string(),
            count,
            usage,
            initial_data,
        });
        self
    }

    pub fn add_texture(mut self, name: &str, source: TextureSource) -> Self {
        self.texture_defs.push(TextureDef {
            name: name.to_string(),
            path: source.path.map(|p| p.to_string()),
            format: source.format,
            size: source.size,
            usage: source.usage,
        });
        self
    }

    pub fn add_pipeline(
        mut self,
        name: &str,
        config: PipelineConfig,
        setup_fn: impl FnOnce(&mut PipelineBuilder),
    ) -> Self {
        let mut builder = PipelineBuilder { passes: Vec::new() };
        setup_fn(&mut builder);

        self.pipelines.push(PipelineDef {
            name: name.to_string(),
            config,
            passes: builder.passes,
        });
        self
    }

    pub fn run(mut self) -> anyhow::Result<()> {
        let event_loop = EventLoop::new()?;
        event_loop.run_app(&mut self)?;

        Ok(())
    }
}

impl ApplicationHandler for RenderApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let init_vulkan = || -> anyhow::Result<VulkanEngine> {
            let window = Arc::new(event_loop.create_window(WindowAttributes::default())?);

            VulkanEngine::new(window)
        };

        match init_vulkan() {
            Ok(vulkan) => {
                self.vulkan_engine.replace(vulkan);
            }
            Err(e) => {
                panic!("Could not initialize window: {}", e);
            }
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::RedrawRequested => {
                let vulkan_engine = self.vulkan_engine.as_mut().unwrap();
                vulkan_engine.draw().unwrap();
                vulkan_engine.window.request_redraw();
            }
            _ => (),
        }
    }
}
