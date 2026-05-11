// MOBIL (Minimizing Overall Braking Induced by Lane changes) lane change shader.
// European model: asymmetric keep-right rule. Cars prefer the rightmost lane
// unless overtaking a slower vehicle. Returning right has a lower threshold
// and a positive bias incentive.
//
// Dispatched every Nth simulation tick to reduce cost.
// For each car, checks adjacent lanes for incentive and safety criteria.
//
// Sorted order is (road_id, lane, s) — within a road, all cars in one lane
// are grouped together. To find cars in an adjacent lane we must search both
// forward AND backward from the current sorted position.

struct PushConstants {
    car_count: u32,
    // IDM parameters (needed to compute accelerations)
    a_max: f32,
    b_comfort: f32,
    s0: f32,
    time_headway: f32,
    car_length: f32,
    // MOBIL parameters
    politeness: f32,       // p: politeness factor, typically 0.5
    threshold: f32,        // Minimum incentive for overtaking (m/s²), typically 0.5
    b_safe: f32,           // Maximum safe deceleration for follower (m/s²), typically 4.0
    max_right_lanes: i32,  // Max right lane index (0-based)
    max_left_lanes: i32,   // Max left lane count (negative index)
    stagger_phase: u32,    // Stagger phase for distributing evaluations
    keep_right_bias: f32,  // Incentive bias for returning to the right lane (m/s²)
    _pad: vec3<u32>,
}

var<immediate> pc: PushConstants;

// Car SoA buffers
@group(0) @binding(0) var<storage, read> car_road_id: array<u32>;
@group(0) @binding(1) var<storage, read> car_s: array<f32>;
@group(0) @binding(2) var<storage, read_write> car_lane: array<i32>;
@group(0) @binding(3) var<storage, read> car_speed: array<f32>;
@group(0) @binding(4) var<storage, read> car_desired_speed: array<f32>;
@group(0) @binding(5) var<storage, read> road_total_lengths: array<f32>;

// Sorted indices
@group(0) @binding(6) var<storage, read> sorted_indices: array<u32>;

/// IDM acceleration computation (same as in traffic_idm.wgsl)
fn idm_accel(v: f32, v0: f32, delta_v: f32, gap: f32) -> f32 {
    let sqrt_ab = sqrt(pc.a_max * pc.b_comfort);
    let s_star = pc.s0 + max(0.0, v * pc.time_headway + v * delta_v / (2.0 * sqrt_ab));
    let v_ratio = v / max(v0, 0.1);
    let v_term = v_ratio * v_ratio * v_ratio * v_ratio;
    let gap_clamped = max(gap, 0.1);
    let s_term = (s_star / gap_clamped) * (s_star / gap_clamped);
    return pc.a_max * (1.0 - v_term - s_term);
}

/// Compute the forward gap from `s` to `other_s` on a circular road.
fn forward_gap(s: f32, other_s: f32, road_len: f32) -> f32 {
    var g = other_s - s;
    if g < 0.0 {
        g += road_len;
    }
    return g - pc.car_length;
}

/// Compute gap to `other_s` in the travel direction.
/// Right lanes (lane >= 0): travel in increasing s.
/// Left lanes (lane < 0): travel in decreasing s.
fn directional_leader_gap(my_s: f32, other_s: f32, road_len: f32, is_left: bool) -> f32 {
    if is_left {
        // Leader is BEHIND in s (lower s = further ahead in travel)
        return forward_gap(other_s, my_s, road_len);
    } else {
        return forward_gap(my_s, other_s, road_len);
    }
}

fn directional_follower_gap(my_s: f32, other_s: f32, road_len: f32, is_left: bool) -> f32 {
    if is_left {
        // Follower is AHEAD in s (higher s = behind in travel)
        return forward_gap(my_s, other_s, road_len);
    } else {
        return forward_gap(other_s, my_s, road_len);
    }
}

/// Find the nearest LEADER (car ahead in travel direction) in `target_lane`.
/// Returns vec2(gap, leader_speed). gap = 1000.0 means no leader found.
fn find_leader_in_lane(
    road_id: u32,
    target_lane: i32,
    s: f32,
    road_len: f32,
    sorted_idx: u32,
) -> vec2<f32> {
    var best_gap = 1000.0;
    var best_speed = 0.0;
    let max_scan = 512u;
    let is_left = target_lane < 0;

    // --- Search FORWARD in sorted order ---
    var scan = sorted_idx + 1u;
    var count = 0u;
    var passed_target = false;
    while scan < pc.car_count && count < max_scan {
        let idx = sorted_indices[scan];
        if car_road_id[idx] != road_id { break; }
        let l = car_lane[idx];
        if l == target_lane {
            passed_target = true;
            let g = directional_leader_gap(s, car_s[idx], road_len, is_left);
            if g >= 0.0 && g < best_gap {
                best_gap = g;
                best_speed = car_speed[idx];
            }
        } else if passed_target {
            break;
        }
        scan += 1u;
        count += 1u;
    }

    // --- Search BACKWARD in sorted order ---
    if sorted_idx > 0u {
        scan = sorted_idx - 1u;
        count = 0u;
        passed_target = false;
        loop {
            let idx = sorted_indices[scan];
            if car_road_id[idx] != road_id { break; }
            let l = car_lane[idx];
            if l == target_lane {
                passed_target = true;
                let g = directional_leader_gap(s, car_s[idx], road_len, is_left);
                if g >= 0.0 && g < best_gap {
                    best_gap = g;
                    best_speed = car_speed[idx];
                }
            } else if passed_target {
                break;
            }
            if scan == 0u { break; }
            count += 1u;
            if count >= max_scan { break; }
            scan -= 1u;
        }
    }

    return vec2<f32>(best_gap, best_speed);
}

/// Find the nearest FOLLOWER (car behind in travel direction) in `target_lane`.
/// Returns vec2(gap, follower_speed). gap = 1000.0 means no follower.
fn find_follower_in_lane(
    road_id: u32,
    target_lane: i32,
    s: f32,
    road_len: f32,
    sorted_idx: u32,
) -> vec2<f32> {
    var best_gap = 1000.0;
    var best_speed = 0.0;
    let max_scan = 512u;
    let is_left = target_lane < 0;

    // --- Search FORWARD in sorted order ---
    var scan = sorted_idx + 1u;
    var count = 0u;
    var passed_target = false;
    while scan < pc.car_count && count < max_scan {
        let idx = sorted_indices[scan];
        if car_road_id[idx] != road_id { break; }
        let l = car_lane[idx];
        if l == target_lane {
            passed_target = true;
            let g = directional_follower_gap(s, car_s[idx], road_len, is_left);
            if g >= 0.0 && g < best_gap {
                best_gap = g;
                best_speed = car_speed[idx];
            }
        } else if passed_target {
            break;
        }
        scan += 1u;
        count += 1u;
    }

    // --- Search BACKWARD in sorted order ---
    if sorted_idx > 0u {
        scan = sorted_idx - 1u;
        count = 0u;
        passed_target = false;
        loop {
            let idx = sorted_indices[scan];
            if car_road_id[idx] != road_id { break; }
            let l = car_lane[idx];
            if l == target_lane {
                passed_target = true;
                let g = directional_follower_gap(s, car_s[idx], road_len, is_left);
                if g >= 0.0 && g < best_gap {
                    best_gap = g;
                    best_speed = car_speed[idx];
                }
            } else if passed_target {
                break;
            }
            if scan == 0u { break; }
            count += 1u;
            if count >= max_scan { break; }
            scan -= 1u;
        }
    }

    return vec2<f32>(best_gap, best_speed);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let sorted_idx = gid.x;
    if sorted_idx >= pc.car_count {
        return;
    }

    let car_idx = sorted_indices[sorted_idx];

    // --- Stagger: only ~1/4 of cars evaluate lane changes per dispatch ---
    // Use a simple hash of car_idx to determine eligibility.
    // This prevents many neighbouring cars from switching simultaneously.
    if (car_idx % 4u) != (pc.stagger_phase % 4u) {
        return;
    }

    let road_id = car_road_id[car_idx];
    let lane = car_lane[car_idx];
    let s = car_s[car_idx];
    let speed = car_speed[car_idx];
    let desired_speed = car_desired_speed[car_idx];
    let road_len = road_total_lengths[road_id];

    // Find current lane acceleration (with current leader)
    let cur_leader = find_leader_in_lane(road_id, lane, s, road_len, sorted_idx);
    let a_current = idm_accel(speed, desired_speed, speed - cur_leader.y, cur_leader.x);

    // Try both adjacent lanes
    var best_lane = lane;
    var best_incentive = 0.0;

    // Minimum gap required in target lane (leader AND follower) to allow change
    let min_gap = pc.car_length * 3.0; // ~13.5m

    for (var dir = -1; dir <= 1; dir += 2) {
        let target_lane = lane + dir;

        // ---- CROSS-DIRECTION GUARD ----
        // Right lanes are >= 0, left (oncoming) lanes are < 0.
        // Never cross between them.
        if lane >= 0 && target_lane < 0 {
            continue;
        }
        if lane < 0 && target_lane >= 0 {
            continue;
        }

        // Check lane bounds
        if target_lane >= 0 && target_lane >= pc.max_right_lanes {
            continue;
        }
        if target_lane < 0 && target_lane < -pc.max_left_lanes {
            continue;
        }

        // Find leader in target lane
        let tgt_leader = find_leader_in_lane(road_id, target_lane, s, road_len, sorted_idx);
        // Reject if leader too close
        if tgt_leader.x < min_gap {
            continue;
        }
        let a_target = idm_accel(speed, desired_speed, speed - tgt_leader.y, tgt_leader.x);

        // Find follower in target lane
        let tgt_follower = find_follower_in_lane(road_id, target_lane, s, road_len, sorted_idx);
        let follower_speed = tgt_follower.y;
        let follower_gap = tgt_follower.x;

        // Reject if follower too close
        if follower_gap < min_gap {
            continue;
        }

        // Safety criterion: follower must not need to brake harder than b_safe
        if follower_gap < 999.0 {
            let a_follower_new = idm_accel(
                follower_speed, follower_speed,
                follower_speed - speed,
                follower_gap,
            );
            if a_follower_new < -pc.b_safe {
                continue;
            }
        }

        // MOBIL incentive criterion (European asymmetric model)
        // Rightward = toward lane 0: keep-right bias is added, threshold is 0.
        // Leftward = overtaking: no bias, full threshold required.
        let is_rightward = (lane >= 0 && target_lane < lane) || (lane < 0 && target_lane > lane);
        var incentive = a_target - a_current;
        var effective_threshold = pc.threshold;

        if is_rightward {
            // European keep-right: add bias to encourage returning right
            incentive += pc.keep_right_bias;
            // Lower threshold: move right even without a speed advantage
            effective_threshold = 0.0;
        }

        if incentive > effective_threshold && incentive > best_incentive {
            best_incentive = incentive;
            best_lane = target_lane;
        }
    }

    // Execute lane change if beneficial
    if best_lane != lane {
        car_lane[car_idx] = best_lane;
    }
}
