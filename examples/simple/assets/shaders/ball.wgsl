// 1. Structs (Shared)
struct Constants {
    ball_size: f32,
    blur_strength: f32,
}
var<push_constant> params: Constants;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec3<f32>,
}

struct Ball {
    position: vec2<f32>,
    velocity: vec2<f32>,
}

// Storage Buffer (Read-Only in Vertex)
@group(0) @binding(0)
var<storage, read> balls: array<Ball>;

// --- VERTEX ENTRY POINT ---
@vertex
fn vs_main(@builtin(vertex_index) in_vertex_index: u32) -> VertexOutput {
    var out: VertexOutput;
    
    // Logic (Standard Rust-like syntax)
    let ball = balls[in_vertex_index];
    
    out.clip_position = vec4<f32>(ball.position, 0.0, 1.0);
    out.color = vec3<f32>(1.0, 0.0, 0.0);
    
    return out;
}

// --- FRAGMENT ENTRY POINT ---
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(in.color, 1.0);
}