use spirv_std::glam::{UVec2, UVec3, Vec3};
use spirv_std::num_traits::Float;
use spirv_std::spirv;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct PushConstants {
    pub car_count: u32,
    pub a_max: f32,
    pub b_comfort: f32,
    pub s0: f32,
    pub time_headway: f32,
    pub car_length: f32,
    pub politeness: f32,
    pub threshold: f32,
    pub b_safe: f32,
    pub stagger_phase: u32,
    pub keep_right_bias: f32,
    pub _pad: u32,
}

fn idm_accel(pc: &PushConstants, v: f32, v0: f32, delta_v: f32, gap: f32) -> f32 {
    let sqrt_ab = (pc.a_max * pc.b_comfort).sqrt();
    let s_star = pc.s0 + (v * pc.time_headway + v * delta_v / (2.0 * sqrt_ab)).max(0.0);
    let v_ratio = v / v0.max(0.1);
    let v_term = v_ratio * v_ratio * v_ratio * v_ratio;
    let gap_clamped = gap.max(0.1);
    let s_term = (s_star / gap_clamped) * (s_star / gap_clamped);
    pc.a_max * (1.0 - v_term - s_term)
}

fn forward_gap(s: f32, other_s: f32, road_len: f32, car_length: f32) -> f32 {
    let mut g = other_s - s;
    if g < 0.0 {
        g += road_len;
    }
    g - car_length
}

fn directional_leader_gap(
    my_s: f32,
    other_s: f32,
    road_len: f32,
    is_left: bool,
    car_length: f32,
) -> f32 {
    if is_left {
        forward_gap(other_s, my_s, road_len, car_length)
    } else {
        forward_gap(my_s, other_s, road_len, car_length)
    }
}

fn directional_follower_gap(
    my_s: f32,
    other_s: f32,
    road_len: f32,
    is_left: bool,
    car_length: f32,
) -> f32 {
    if is_left {
        forward_gap(my_s, other_s, road_len, car_length)
    } else {
        forward_gap(other_s, my_s, road_len, car_length)
    }
}

/// Find the nearest LEADER in `target_lane`. Returns (gap, leader_speed, leader_desired_speed).
fn find_leader_in_lane(
    road_id: u32,
    target_lane: i32,
    s: f32,
    road_len: f32,
    sorted_idx: u32,
    pc: &PushConstants,
    car_road_id: &[u32],
    car_s: &[f32],
    car_lane: &[i32],
    car_speed: &[f32],
    car_desired_speed: &[f32],
    sorted_indices: &[u32],
) -> Vec3 {
    let mut best_gap = 1000.0f32;
    let mut best_speed = 0.0f32;
    let mut best_desired = 0.0f32;
    let max_scan = 4u32;
    let is_left = target_lane < 0;

    // Search FORWARD
    let mut scan = sorted_idx + 1;
    let mut count = 0u32;
    let mut passed_target = false;
    while scan < pc.car_count && count < max_scan {
        let idx = sorted_indices[scan as usize] as usize;
        if car_road_id[idx] != road_id {
            break;
        }
        let l = car_lane[idx];
        if l == target_lane {
            passed_target = true;
            let g = directional_leader_gap(s, car_s[idx], road_len, is_left, pc.car_length);
            if g >= 0.0 && g < best_gap {
                best_gap = g;
                best_speed = car_speed[idx];
                best_desired = car_desired_speed[idx];
            }
        } else if passed_target {
            break;
        }
        scan += 1;
        count += 1;
    }

    // Search BACKWARD
    if sorted_idx > 0 {
        scan = sorted_idx - 1;
        count = 0;
        passed_target = false;
        let mut running = true;
        while running && count < max_scan {
            let idx = sorted_indices[scan as usize] as usize;
            if car_road_id[idx] != road_id {
                running = false;
            } else {
                let l = car_lane[idx];
                if l == target_lane {
                    passed_target = true;
                    let g = directional_leader_gap(s, car_s[idx], road_len, is_left, pc.car_length);
                    if g >= 0.0 && g < best_gap {
                        best_gap = g;
                        best_speed = car_speed[idx];
                        best_desired = car_desired_speed[idx];
                    }
                } else if passed_target {
                    running = false;
                }
                if scan == 0 {
                    running = false;
                } else {
                    scan -= 1;
                }
                count += 1;
            }
        }
    }

    Vec3::new(best_gap, best_speed, best_desired)
}

/// Find the nearest FOLLOWER in `target_lane`. Returns (gap, follower_speed, follower_desired_speed).
fn find_follower_in_lane(
    road_id: u32,
    target_lane: i32,
    s: f32,
    road_len: f32,
    sorted_idx: u32,
    pc: &PushConstants,
    car_road_id: &[u32],
    car_s: &[f32],
    car_lane: &[i32],
    car_speed: &[f32],
    car_desired_speed: &[f32],
    sorted_indices: &[u32],
) -> Vec3 {
    let mut best_gap = 1000.0f32;
    let mut best_speed = 0.0f32;
    let mut best_desired = 0.0f32;
    let max_scan = 4u32;
    let is_left = target_lane < 0;

    // Search FORWARD
    let mut scan = sorted_idx + 1;
    let mut count = 0u32;
    let mut passed_target = false;
    while scan < pc.car_count && count < max_scan {
        let idx = sorted_indices[scan as usize] as usize;
        if car_road_id[idx] != road_id {
            break;
        }
        let l = car_lane[idx];
        if l == target_lane {
            passed_target = true;
            let g = directional_follower_gap(s, car_s[idx], road_len, is_left, pc.car_length);
            if g >= 0.0 && g < best_gap {
                best_gap = g;
                best_speed = car_speed[idx];
                best_desired = car_desired_speed[idx];
            }
        } else if passed_target {
            break;
        }
        scan += 1;
        count += 1;
    }

    // Search BACKWARD
    if sorted_idx > 0 {
        scan = sorted_idx - 1;
        count = 0;
        passed_target = false;
        let mut running = true;
        while running && count < max_scan {
            let idx = sorted_indices[scan as usize] as usize;
            if car_road_id[idx] != road_id {
                running = false;
            } else {
                let l = car_lane[idx];
                if l == target_lane {
                    passed_target = true;
                    let g = directional_follower_gap(s, car_s[idx], road_len, is_left, pc.car_length);
                    if g >= 0.0 && g < best_gap {
                        best_gap = g;
                        best_speed = car_speed[idx];
                        best_desired = car_desired_speed[idx];
                    }
                } else if passed_target {
                    running = false;
                }
                if scan == 0 {
                    running = false;
                } else {
                    scan -= 1;
                }
                count += 1;
            }
        }
    }

    Vec3::new(best_gap, best_speed, best_desired)
}

#[spirv(compute(threads(256)))]
pub fn traffic_lane_change_main(
    #[spirv(global_invocation_id)] gid: UVec3,
    #[spirv(push_constant)] pc: &PushConstants,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] car_road_id: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] car_s: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] car_lane: &mut [i32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] car_speed: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] car_desired_speed: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 5)] road_total_lengths: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 6)] sorted_indices: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 7)] road_lane_counts: &[UVec2],
) {
    let sorted_idx = gid.x;
    if sorted_idx >= pc.car_count {
        return;
    }

    let car_idx = sorted_indices[sorted_idx as usize] as usize;

    // Stagger: only ~1/4 of cars evaluate lane changes per dispatch
    if (car_idx as u32 % 4) != (pc.stagger_phase % 4) {
        return;
    }

    let road_id = car_road_id[car_idx];
    let lane = car_lane[car_idx];
    let s = car_s[car_idx];
    let speed = car_speed[car_idx];
    let desired_speed = car_desired_speed[car_idx];
    let road_len = road_total_lengths[road_id as usize];

    let lane_counts = road_lane_counts[road_id as usize];
    let max_right = lane_counts.x as i32;
    let max_left = lane_counts.y as i32;

    // Find current lane acceleration
    let cur_leader = find_leader_in_lane(
        road_id,
        lane,
        s,
        road_len,
        sorted_idx,
        pc,
        car_road_id,
        car_s,
        car_lane,
        car_speed,
        car_desired_speed,
        sorted_indices,
    );
    let a_current = idm_accel(pc, speed, desired_speed, speed - cur_leader.y, cur_leader.x);

    let mut best_lane = lane;
    let mut best_incentive = 0.0f32;

    let min_gap = (pc.car_length * 2.0).max(speed * pc.time_headway);

    let mut dir = -1i32;
    while dir <= 1 {
        if dir == 0 {
            dir += 1;
            continue;
        }
        let target_lane = lane + dir;

        // Cross-direction guard
        if lane >= 0 && target_lane < 0 {
            dir += 2;
            continue;
        }
        if lane < 0 && target_lane >= 0 {
            dir += 2;
            continue;
        }

        // Check lane bounds
        if target_lane >= 0 && target_lane >= max_right {
            dir += 2;
            continue;
        }
        if target_lane < 0 && target_lane < -max_left {
            dir += 2;
            continue;
        }

        // Find leader in target lane
        let tgt_leader = find_leader_in_lane(
            road_id,
            target_lane,
            s,
            road_len,
            sorted_idx,
            pc,
            car_road_id,
            car_s,
            car_lane,
            car_speed,
            car_desired_speed,
            sorted_indices,
        );
        if tgt_leader.x < min_gap {
            dir += 2;
            continue;
        }
        let a_target = idm_accel(pc, speed, desired_speed, speed - tgt_leader.y, tgt_leader.x);

        // Find follower in target lane
        let tgt_follower = find_follower_in_lane(
            road_id,
            target_lane,
            s,
            road_len,
            sorted_idx,
            pc,
            car_road_id,
            car_s,
            car_lane,
            car_speed,
            car_desired_speed,
            sorted_indices,
        );
        let follower_speed = tgt_follower.y;
        let follower_desired = tgt_follower.z;
        let follower_gap = tgt_follower.x;

        let is_rightward = (lane >= 0 && target_lane > lane) || (lane < 0 && target_lane < lane);

        // Follower clearance: relaxed for rightward returns (follower is behind
        // and slower), stricter for leftward overtakes (don't cut off faster car)
        let follower_clearance = if is_rightward {
            pc.car_length * 3.0
        } else {
            min_gap
        };
        if follower_gap < follower_clearance {
            dir += 2;
            continue;
        }

        // Don't return right if slower car close ahead
        let keep_left_dist = speed * pc.time_headway * 3.0 + pc.car_length * 4.0;
        if is_rightward && tgt_leader.x < keep_left_dist && tgt_leader.y < speed * 0.85 {
            dir += 2;
            continue;
        }

        // Safety criterion
        let mut a_follower_new = 0.0f32;
        let mut a_follower_old = 0.0f32;
        if follower_gap < 999.0 {
            a_follower_new = idm_accel(
                pc,
                follower_speed,
                follower_desired,
                follower_speed - speed,
                follower_gap,
            );
            if a_follower_new < -pc.b_safe {
                dir += 2;
                continue;
            }
            let follower_leader_gap = tgt_leader.x + follower_gap + pc.car_length;
            a_follower_old = idm_accel(
                pc,
                follower_speed,
                follower_desired,
                follower_speed - tgt_leader.y,
                follower_leader_gap,
            );
        }

        // MOBIL incentive criterion
        let mut incentive =
            (a_target - a_current) + pc.politeness * (a_follower_new - a_follower_old);
        let mut effective_threshold = pc.threshold;

        if is_rightward {
            incentive += pc.keep_right_bias;
            effective_threshold = 0.0;
        }

        if incentive > effective_threshold && incentive > best_incentive {
            best_incentive = incentive;
            best_lane = target_lane;
        }

        dir += 2;
    }

    // Execute lane change if beneficial
    if best_lane != lane {
        car_lane[car_idx] = best_lane;
    }
}
