//! Chapter 1 of vk_guide

use std::sync::Arc;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::{EnvFilter, fmt};
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::window::{WindowAttributes, WindowId};

use crate::vulkan_engine::VulkanEngine;

pub(crate) mod vk_util;
mod vulkan_engine;

#[derive(Default, Debug)]
struct App {
    vulkan_engine: Option<VulkanEngine>,
}

impl ApplicationHandler for App {
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

fn main() -> anyhow::Result<()> {
    // Initialize a simple tracing subscriber so example logs are visible
    tracing_subscriber::registry()
        .with(fmt::layer())
        .with(EnvFilter::from_default_env())
        .init();

    let event_loop = EventLoop::new()?;
    let mut app = App::default();
    event_loop.run_app(&mut app)?;

    Ok(())
}
