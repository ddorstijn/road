use vre::RenderApp;

use glam::Vec2;

struct PointBuffer {
    position: Vec2,
}

fn init_points() -> Vec<PointBuffer> {
    (0..10)
        .map(|p| PointBuffer {
            position: Vec2::splat(p as f32),
        })
        .collect()
}

fn main() {
    // Pseudo code of what the interface should somewhat look like
    // Should we use register_buffer and register_texture or just register_resource?
    // Is it possible to detect input and output buffers for the graph or explicitly specify resources used
    RenderApp::new()
        .register_buffer::<PointBuffer>(init_points())
        .add_node(vre::Pipeline::Compute, "assets/shaders/update_balls.comp")
        .add_node(vre::Pipeline::Graphics, "assets/shaders/render_balls.frag");

    // What this code should do:
    // Fill buffer with random points on the cpu
    // Update points with compute shader
    // Render points with fragment shader with radius constant
}
