use bytemuck::{Pod, Zeroable};
use glam::Vec2;
use vre::{PipelineConfig, RenderApp, ShaderPass, TextureSource, vk};

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct Ball {
    position: Vec2,
    velocity: Vec2,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct PhysicsParams {
    dt: f32,
    gravity: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct DrawParams {
    ball_size: f32,
    blur_strength: f32,
}

fn create_balls(count: usize) -> Vec<Ball> {
    let mut balls = Vec::with_capacity(count);

    for i in 0..count {
        // Random-ish setup
        let x = (i as f32 % 50.0) - 25.0;
        let y = (i as f32 / 50.0) - 10.0;
        balls.push(Ball {
            position: Vec2::new(x, y),
            velocity: Vec2::new(1.0, 5.0),
        });
    }
    balls
}

fn main() -> anyhow::Result<()> {
    RenderApp::new()
        // Resources
        .add_buffer(
            "Balls",
            10,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            Some(create_balls),
        )
        .add_texture(
            "SceneColor",
            TextureSource {
                format: vk::Format::R16G16B16A16_SFLOAT,
                size: vre::TextureSize::ViewportRelative(1.0),
                usage: vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED,
                path: None,
            },
        )
        // --- PIPELINE: Render ---
        .add_pipeline("Render", PipelineConfig::PerFrame, |pipeline| {
            pipeline.add_pass(ShaderPass {
                name: "DrawBalls",
                ty: vre::PassType::Render,
                shader: "assets/shaders/ball.wgsl",
                inputs: &["Balls"],
                outputs: &["SceneColor"],
                constants: DrawParams {
                    ball_size: 0.5,
                    blur_strength: 0.0,
                },
            });
            pipeline.add_pass(ShaderPass {
                name: "MotionBlur",
                ty: vre::PassType::Compute,
                shader: "assets/shaders/blur.wgsl",
                inputs: &["SceneColor"],
                outputs: &["$SWAPCHAIN"],
                constants: DrawParams {
                    ball_size: 0.0,
                    blur_strength: 0.8,
                },
            });
        })
        .run()
}
