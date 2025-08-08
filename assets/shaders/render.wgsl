

// shaders/render.wgsl
struct Particle {
    pos: vec2<f32>,
    vel: vec2<f32>,
}

@group(0) @binding(0) var<storage, read> particles_a: array<Particle>;
@group(0) @binding(1) var<storage, read> particles_b: array<Particle>;

struct Uniforms {
    alpha: f32,
}
@group(0) @binding(2) var<uniform> uniforms: Uniforms;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec3<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    let instance_index = vertex_index / 6u;
    let quad_vertex_index = vertex_index % 6u;

    let particle_a = particles_a[instance_index];
    let particle_b = particles_b[instance_index];

    let interpolated_pos = mix(particle_a.pos, particle_b.pos, uniforms.alpha);

    let quad_offsets = array<vec2<f32>, 6>(
        vec2<f32>(-0.01, -0.01),
        vec2<f32>(0.01, -0.01),
        vec2<f32>(-0.01, 0.01),
        vec2<f32>(-0.01, 0.01),
        vec2<f32>(0.01, -0.01),
        vec2<f32>(0.01, 0.01),
    );

    var out: VertexOutput;
    out.clip_position = vec4<f32>(interpolated_pos + quad_offsets[quad_vertex_index], 0.0, 1.0);

    let speed = length(mix(particle_a.vel, particle_b.vel, uniforms.alpha));
    out.color = vec3<f32>(min(speed * 2.0, 1.0), max(1.0 - speed, 0.2), 0.5);

    return out;
}
