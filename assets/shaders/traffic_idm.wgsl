// IDM (Intelligent Driver Model) car-following compute shader.
// After sorting, cars are ordered by (road_id, lane, s).
// For each car at sorted index i, the leader is at sorted index i+1
// (if same road and lane). Computes acceleration and updates speed.

struct PushConstants {
    dt: f32,
    car_count: u32,
    // IDM parameters
    a_max: f32,        // Maximum acceleration (m/s²), typically 1.5
    b_comfort: f32,    // Comfortable deceleration (m/s²), typically 2.0
    s0: f32,           // Minimum gap (m), typically 2.0
    time_headway: f32, // Desired time headway (s), typically 1.5
    car_length: f32,   // Car length (m), typically 4.5
    _pad: u32,
}

var<immediate> pc: PushConstants;

// Car SoA buffers
@group(0) @binding(0) var<storage, read> car_road_id: array<u32>;
@group(0) @binding(1) var<storage, read_write> car_s: array<f32>;
@group(0) @binding(2) var<storage, read> car_lane: array<i32>;
@group(0) @binding(3) var<storage, read_write> car_speed: array<f32>;
@group(0) @binding(4) var<storage, read> car_desired_speed: array<f32>;
@group(0) @binding(5) var<storage, read> road_total_lengths: array<f32>;

// Sorted indices (output of radix sort)
@group(0) @binding(6) var<storage, read> sorted_indices: array<u32>;

/// Compute IDM acceleration given current state and leader info.
fn idm_acceleration(v: f32, v0: f32, delta_v: f32, gap: f32) -> f32 {
    // Desired gap: s* = s0 + v*T + v*dv / (2*sqrt(a*b))
    let sqrt_ab = sqrt(pc.a_max * pc.b_comfort);
    let s_star = pc.s0 + max(0.0, v * pc.time_headway + v * delta_v / (2.0 * sqrt_ab));

    // IDM acceleration: a = a_max * [1 - (v/v0)^4 - (s*/gap)^2]
    let v_ratio = v / max(v0, 0.1);
    let v_term = v_ratio * v_ratio * v_ratio * v_ratio; // (v/v0)^4

    let gap_clamped = max(gap, 0.1); // Avoid division by zero
    let s_term = (s_star / gap_clamped) * (s_star / gap_clamped);

    return pc.a_max * (1.0 - v_term - s_term);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let sorted_idx = gid.x;
    if sorted_idx >= pc.car_count {
        return;
    }

    let car_idx = sorted_indices[sorted_idx];
    let road_id = car_road_id[car_idx];
    let lane = car_lane[car_idx];
    let s = car_s[car_idx];
    let speed = car_speed[car_idx];
    let desired_speed = car_desired_speed[car_idx];
    let road_len = road_total_lengths[road_id];

    // Left lanes (lane < 0) drive in the DECREASING s direction.
    // In sorted order (ascending s), the "leader" for a left-lane car
    // is at sorted_idx - 1 (lower s = further ahead in travel direction).
    let is_left = lane < 0;

    // Find leader: depends on travel direction
    var gap = 1000.0;
    var delta_v = 0.0;

    if !is_left {
        // RIGHT LANE: leader is next in sorted order (higher s)
        let next_sorted_idx = sorted_idx + 1u;
        if next_sorted_idx < pc.car_count {
            let next_car_idx = sorted_indices[next_sorted_idx];
            if car_road_id[next_car_idx] == road_id && car_lane[next_car_idx] == lane {
                let leader_s = car_s[next_car_idx];
                var raw_gap = leader_s - s;
                if raw_gap < 0.0 { raw_gap += road_len; }
                gap = raw_gap - pc.car_length;
                delta_v = speed - car_speed[next_car_idx];
            }
        }
        // Wraparound: first car in group wraps as leader
        if gap > 900.0 {
            var search_idx = sorted_idx;
            while search_idx > 0u {
                let prev_car = sorted_indices[search_idx - 1u];
                if car_road_id[prev_car] != road_id || car_lane[prev_car] != lane { break; }
                search_idx -= 1u;
            }
            if search_idx != sorted_idx {
                let first_car = sorted_indices[search_idx];
                let wrap_gap = (road_len - s) + car_s[first_car] - pc.car_length;
                if wrap_gap < gap {
                    gap = wrap_gap;
                    delta_v = speed - car_speed[first_car];
                }
            }
        }
    } else {
        // LEFT LANE: travel in decreasing s. Leader has LOWER s (sorted_idx - 1).
        if sorted_idx > 0u {
            let prev_car_idx = sorted_indices[sorted_idx - 1u];
            if car_road_id[prev_car_idx] == road_id && car_lane[prev_car_idx] == lane {
                let leader_s = car_s[prev_car_idx];
                var raw_gap = s - leader_s; // We're at higher s, leader at lower s
                if raw_gap < 0.0 { raw_gap += road_len; }
                gap = raw_gap - pc.car_length;
                delta_v = speed - car_speed[prev_car_idx];
            }
        }
        // Wraparound: last car in group wraps as leader
        if gap > 900.0 {
            var search_idx = sorted_idx;
            while search_idx + 1u < pc.car_count {
                let next_car = sorted_indices[search_idx + 1u];
                if car_road_id[next_car] != road_id || car_lane[next_car] != lane { break; }
                search_idx += 1u;
            }
            if search_idx != sorted_idx {
                let last_car = sorted_indices[search_idx];
                let wrap_gap = s + (road_len - car_s[last_car]) - pc.car_length;
                if wrap_gap < gap {
                    gap = wrap_gap;
                    delta_v = speed - car_speed[last_car];
                }
            }
        }
    }

    // Compute IDM acceleration
    let accel = idm_acceleration(speed, desired_speed, delta_v, gap);

    // Update speed
    var new_speed = speed + accel * pc.dt;

    // --- Overlap resolution ---
    if gap < 0.0 {
        new_speed = 0.0;
    }

    new_speed = max(0.0, new_speed);
    new_speed = min(new_speed, desired_speed * 1.5);

    car_speed[car_idx] = new_speed;

    // Advance position: direction depends on lane
    var new_s: f32;
    if is_left {
        new_s = s - new_speed * pc.dt; // Decreasing s
        if new_s < 0.0 {
            new_s += road_len; // Wrap around to end
        }
    } else {
        new_s = s + new_speed * pc.dt; // Increasing s
        if new_s >= road_len {
            new_s -= road_len;
        }
    }
    car_s[car_idx] = new_s;
}
