use crate::road_eval::{compute_lane_offset, curvature_at_s};
use gpu_shared::{GpuLane, GpuLaneSection, GpuRoad, GpuSegment};
use spirv_std::glam::UVec3;
use spirv_std::num_traits::Float;
use spirv_std::spirv;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct PushConstants {
    pub dt: f32,
    pub car_count: u32,
    pub a_max: f32,
    pub b_comfort: f32,
    pub s0: f32,
    pub time_headway: f32,
    pub car_length: f32,
    pub _pad: u32,
}

fn idm_acceleration(pc: &PushConstants, v: f32, v0: f32, delta_v: f32, gap: f32) -> f32 {
    let sqrt_ab = (pc.a_max * pc.b_comfort).sqrt();
    let s_star = pc.s0 + (v * pc.time_headway + v * delta_v / (2.0 * sqrt_ab)).max(0.0);
    let v_ratio = v / v0.max(0.1);
    let v_term = v_ratio * v_ratio * v_ratio * v_ratio;
    let gap_clamped = gap.max(0.1);
    let s_term = (s_star / gap_clamped) * (s_star / gap_clamped);
    pc.a_max * (1.0 - v_term - s_term)
}

#[spirv(compute(threads(256)))]
pub fn traffic_idm_main(
    #[spirv(global_invocation_id)] gid: UVec3,
    #[spirv(push_constant)] pc: &PushConstants,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] car_road_id: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] car_s: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] car_lane: &[i32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] car_speed: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] car_desired_speed: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 5)] road_total_lengths: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 6)] sorted_indices: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 7)] roads: &[GpuRoad],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 8)] segments: &[GpuSegment],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 9)] lane_sections: &[GpuLaneSection],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 10)] lanes: &[GpuLane],
) {
    let sorted_idx = gid.x;
    if sorted_idx >= pc.car_count {
        return;
    }

    let car_idx = sorted_indices[sorted_idx as usize] as usize;
    let road_id = car_road_id[car_idx];
    let lane = car_lane[car_idx];
    let s = car_s[car_idx];
    let speed = car_speed[car_idx];
    let desired_speed = car_desired_speed[car_idx];
    let road_len = road_total_lengths[road_id as usize];

    // Curvature correction: convert between reference-line s and actual lane arc-length.
    // For a lane at lateral offset t from a reference line with curvature kappa,
    // ds_ref = v_actual * dt / (1 - kappa * t)
    let kappa = curvature_at_s(road_id, s, roads, segments);
    let t = compute_lane_offset(road_id, lane, s, roads, lane_sections, lanes);
    let kt = kappa * t;
    // Clamp to avoid singularity (lane offset should never exceed radius of curvature)
    let lane_factor = (1.0 - kt).clamp(0.25, 4.0);
    // correction converts actual speed to reference-line ds/dt
    let ref_correction = 1.0 / lane_factor;

    let is_left = lane < 0;

    let mut gap = 1000.0f32;
    let mut delta_v = 0.0f32;

    if !is_left {
        // RIGHT LANE: leader is next in sorted order (higher s)
        let next_sorted_idx = sorted_idx + 1;
        if next_sorted_idx < pc.car_count {
            let next_car_idx = sorted_indices[next_sorted_idx as usize] as usize;
            if car_road_id[next_car_idx] == road_id && car_lane[next_car_idx] == lane {
                let leader_s = car_s[next_car_idx];
                let mut raw_gap = leader_s - s;
                if raw_gap < 0.0 {
                    raw_gap += road_len;
                }
                gap = raw_gap - pc.car_length;
                delta_v = speed - car_speed[next_car_idx];
            }
        }
        // Wraparound
        if gap > 900.0 {
            let mut search_idx = sorted_idx;
            while search_idx > 0 {
                let prev_car = sorted_indices[(search_idx - 1) as usize] as usize;
                if car_road_id[prev_car] != road_id || car_lane[prev_car] != lane {
                    break;
                }
                search_idx -= 1;
            }
            if search_idx != sorted_idx {
                let first_car = sorted_indices[search_idx as usize] as usize;
                let wrap_gap = (road_len - s) + car_s[first_car] - pc.car_length;
                if wrap_gap < gap {
                    gap = wrap_gap;
                    delta_v = speed - car_speed[first_car];
                }
            }
        }
    } else {
        // LEFT LANE: travel in decreasing s. Leader has LOWER s (sorted_idx - 1).
        if sorted_idx > 0 {
            let prev_car_idx = sorted_indices[(sorted_idx - 1) as usize] as usize;
            if car_road_id[prev_car_idx] == road_id && car_lane[prev_car_idx] == lane {
                let leader_s = car_s[prev_car_idx];
                let mut raw_gap = s - leader_s;
                if raw_gap < 0.0 {
                    raw_gap += road_len;
                }
                gap = raw_gap - pc.car_length;
                delta_v = speed - car_speed[prev_car_idx];
            }
        }
        // Wraparound
        if gap > 900.0 {
            let mut search_idx = sorted_idx;
            while search_idx + 1 < pc.car_count {
                let next_car = sorted_indices[(search_idx + 1) as usize] as usize;
                if car_road_id[next_car] != road_id || car_lane[next_car] != lane {
                    break;
                }
                search_idx += 1;
            }
            if search_idx != sorted_idx {
                let last_car = sorted_indices[search_idx as usize] as usize;
                let wrap_gap = s + (road_len - car_s[last_car]) - pc.car_length;
                if wrap_gap < gap {
                    gap = wrap_gap;
                    delta_v = speed - car_speed[last_car];
                }
            }
        }
    }

    // Convert reference-line gap to actual lane arc-length gap
    let lane_gap = gap * lane_factor;

    // Compute IDM acceleration using actual lane gap
    let accel = idm_acceleration(pc, speed, desired_speed, delta_v, lane_gap);

    // Update speed
    let mut new_speed = speed + accel * pc.dt;
    if gap < 0.0 {
        new_speed = 0.0;
    }
    new_speed = new_speed.max(0.0).min(desired_speed * 1.5);
    // Guard against NaN/inf propagation
    if !(new_speed >= 0.0 && new_speed <= 200.0) {
        new_speed = 0.0;
    }
    car_speed[car_idx] = new_speed;

    // Advance position (convert actual speed to reference-line ds/dt)
    let ds = new_speed * pc.dt * ref_correction;
    if is_left {
        let mut new_s = s - ds;
        if new_s < 0.0 {
            new_s += road_len;
        }
        if !(new_s >= 0.0 && new_s < road_len) {
            new_s = 0.0;
        }
        car_s[car_idx] = new_s;
    } else {
        let mut new_s = s + ds;
        if new_s >= road_len {
            new_s -= road_len;
        }
        if !(new_s >= 0.0 && new_s < road_len) {
            new_s = 0.0;
        }
        car_s[car_idx] = new_s;
    }
}
