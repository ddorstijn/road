use vre::RenderApp;

use glam::Vec2;

struct PointBuffer {
    position: Vec2,
}

fn main() {
    RenderApp::new().add_buffer();

    // Fill buffer with random points on the cpu
    // Update points with compute shader
    // Render points with fragment shader with radius constant
}
