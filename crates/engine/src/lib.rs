use winit::event_loop::EventLoop;

use crate::{component::GameComponent, engine::EngineRuntime};

pub mod component;
pub mod engine;
mod imgui_renderer;
mod vulkan_util;
mod world;

pub struct Engine {
    window_title: String,
    components: Vec<Box<dyn GameComponent>>,
}

impl Engine {
    pub fn new(title: &str) -> Self {
        Self {
            window_title: title.to_string(),
            components: Vec::new(),
        }
    }

    pub fn add_component<T>(&mut self)
    where
        T: GameComponent + Default + 'static,
    {
        // We can now safely call T::default() because we know T is concrete here
        let component = T::default();

        // We box it up and store it as a trait object
        self.components.push(Box::new(component));
    }

    /// This consumes the Engine builder and starts the loop
    pub fn run(self) {
        let event_loop = EventLoop::new().unwrap();
        let mut runtime = EngineRuntime::new(self.window_title, self.components);
        event_loop.run_app(&mut runtime).unwrap();
    }
}
