use app::App;
use engine::Engine;

fn main() {
    let mut engine = Engine::new("My Vulkan App");
    engine.add_component::<App>();
    engine.run();
}
