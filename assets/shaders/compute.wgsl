// shaders/compute.wgsl
struct Particle {
    pos: vec2<f32>,
    vel: vec2<f32>,
}

@group(0) @binding(0) var<storage, read> particles_in: array<Particle>;
@group(0) @binding(1) var<storage, read_write> particles_out: array<Particle>;

const DT: f32 = 0.016;
const CENTER: vec2<f32> = vec2<f32>(0.0, 0.0);
const GRAVITY: f32 = -100.0;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    let particle_in = particles_in[index];
    let to_center = CENTER - particle_in.pos;
    let dist = length(to_center);
    if dist < 0.1 {
        particles_out[index] = particle_in;
        return;
    }
    let gravity_force = normalize(to_center) * GRAVITY / (dist * dist);
    let tangent_force = vec2<f32>(-to_center.y, to_center.x) * 0.1 / dist;
    var new_vel = particle_in.vel + (gravity_force + tangent_force) * DT;
    new_vel = new_vel * 0.995;
    let new_pos = particle_in.pos + new_vel * DT;
    particles_out[index].pos = new_pos;
    particles_out[index].vel = new_vel;
}