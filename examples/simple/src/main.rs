use bytemuck::{Pod, Zeroable};
use glam::Vec2;
use vre::{
    ComputePass, Pass, PipelineBuilder, PipelineConfig, RenderApp, RenderPass, TextureFormat,
    TextureUsage,
};

// --- 1. Struct Definitions (Shared with Shaders) ---

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

// --- 2. Initialization Function ---
// Returns the exact array of structs to upload
fn create_balls() -> Vec<Ball> {
    const COUNT: usize = 1000;
    let mut balls = Vec::with_capacity(COUNT);

    for i in 0..COUNT {
        // Random-ish setup
        let x = (i as f32 % 50.0) - 25.0;
        let y = (i as f32 / 50.0) - 10.0;
        balls.push(Ball {
            position: Vec2::new(x, y),
            velocity: Vec2::new(1.0, 5.0), // Start with some bounce
        });
    }
    balls
}

fn main() -> anyhow::Result<()> {
    RenderApp::new()
        // Resources
        .add_buffer("Balls", vk::BufferUsageFlags::STORAGE_BUFFER, create_balls)
        .add_texture(
            "SceneColor",
            vk::Format::R16G16B16A16_SFLOAT,
            vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED,
            1.0,
        )
        // --- PIPELINE: Render ---
        .add_pipeline("Render", PipelineConfig::PerFrame, |pipeline| {
            // 1. Render Pass
            // The engine looks for "vs_main" and "fs_main" in "ball.wgsl"
            pipeline.add_pass(RenderPass {
                name: "DrawBalls",
                shader: "assets/shaders/ball.wgsl",
                inputs: &["Balls"],
                outputs: &["SceneColor"],
                constants: DrawParams {
                    ball_size: 0.5,
                    blur_strength: 0.0,
                },
            });

            // 2. Compute Pass (Post-Process)
            // The engine looks for "main" in "blur.wgsl"
            pipeline.add_pass(ComputePass {
                name: "MotionBlur",
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
