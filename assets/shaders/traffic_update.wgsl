// Traffic update compute shader: advances cars along roads.
// Each car: s += speed * dt; wraps to 0 when s >= road_length.

struct PushConstants {
    dt: f32,
    car_count: u32,
}

var<immediate> pc: PushConstants;

@group(0) @binding(0) var<storage, read_write> car_road_id: array<u32>;
@group(0) @binding(1) var<storage, read_write> car_s: array<f32>;
@group(0) @binding(2) var<storage, read_write> car_lane: array<i32>;
@group(0) @binding(3) var<storage, read_write> car_speed: array<f32>;
@group(0) @binding(4) var<storage, read_write> car_desired_speed: array<f32>;
@group(0) @binding(5) var<storage, read> road_total_lengths: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= pc.car_count {
        return;
    }

    let speed = car_speed[idx];
    var s = car_s[idx] + speed * pc.dt;

    let road_id = car_road_id[idx];
    let road_len = road_total_lengths[road_id];

    // Wrap around when reaching end of road
    if s >= road_len {
        s = s - road_len;
    }
    // Clamp negative (shouldn't happen but safety)
    if s < 0.0 {
        s = 0.0;
    }

    car_s[idx] = s;
}
