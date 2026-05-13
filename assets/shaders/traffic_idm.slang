// Traffic IDM (Intelligent Driver Model) simulation

import gpu_shared;
import road_eval;

struct PushConstants {
    float dt;
    uint car_count;
    float a_max;
    float b_comfort;
    float s0;
    float time_headway;
    float car_length;
    uint _pad;
};

[[vk::push_constant]] ConstantBuffer<PushConstants> pc;
[[vk::binding(0, 0)]] StructuredBuffer<uint> car_road_id;
[[vk::binding(1, 0)]] RWStructuredBuffer<float> car_s;
[[vk::binding(2, 0)]] StructuredBuffer<int> car_lane;
[[vk::binding(3, 0)]] RWStructuredBuffer<float> car_speed;
[[vk::binding(4, 0)]] StructuredBuffer<float> car_desired_speed;
[[vk::binding(5, 0)]] StructuredBuffer<float> road_total_lengths;
[[vk::binding(6, 0)]] StructuredBuffer<uint> sorted_indices;
[[vk::binding(7, 0)]] StructuredBuffer<GpuRoad> roads_buf;
[[vk::binding(8, 0)]] StructuredBuffer<GpuSegment> segments;
[[vk::binding(9, 0)]] StructuredBuffer<GpuLaneSection> lane_sections;
[[vk::binding(10, 0)]] StructuredBuffer<GpuLane> lanes;

float idm_acceleration(float v, float v0, float delta_v, float gap) {
    float sqrt_ab = sqrt(pc.a_max * pc.b_comfort);
    float s_star = pc.s0 + max(v * pc.time_headway + v * delta_v / (2.0 * sqrt_ab), 0.0);
    float v_ratio = v / max(v0, 0.1);
    float v_term = v_ratio * v_ratio * v_ratio * v_ratio;
    float gap_clamped = max(gap, 0.1);
    float s_term = (s_star / gap_clamped) * (s_star / gap_clamped);
    return pc.a_max * (1.0 - v_term - s_term);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void traffic_idm_main(uint3 gid : SV_DispatchThreadID) {
    uint sorted_idx = gid.x;
    if (sorted_idx >= pc.car_count)
        return;

    uint car_idx = sorted_indices[sorted_idx];
    uint road_id = car_road_id[car_idx];
    int lane = car_lane[car_idx];
    float s = car_s[car_idx];
    float speed = car_speed[car_idx];
    float desired_speed = car_desired_speed[car_idx];
    float road_len = road_total_lengths[road_id];

    // Curvature correction
    float kappa = curvature_at_s(road_id, s, roads_buf, segments);
    float t = compute_lane_offset(road_id, lane, s, roads_buf, lane_sections, lanes);
    float kt = kappa * t;
    float lane_factor = clamp(1.0 - kt, 0.25, 4.0);
    float ref_correction = 1.0 / lane_factor;

    bool is_left = lane < 0;

    float gap = 1000.0;
    float delta_v = 0.0;

    if (!is_left) {
        // RIGHT LANE: leader is next in sorted order (higher s)
        uint next_sorted_idx = sorted_idx + 1;
        if (next_sorted_idx < pc.car_count) {
            uint next_car_idx = sorted_indices[next_sorted_idx];
            if (car_road_id[next_car_idx] == road_id && car_lane[next_car_idx] == lane) {
                float leader_s = car_s[next_car_idx];
                float raw_gap = leader_s - s;
                if (raw_gap < 0.0) raw_gap += road_len;
                gap = raw_gap - pc.car_length;
                delta_v = speed - car_speed[next_car_idx];
            }
        }
        // Wraparound
        if (gap > 900.0) {
            uint search_idx = sorted_idx;
            while (search_idx > 0) {
                uint prev_car = sorted_indices[search_idx - 1];
                if (car_road_id[prev_car] != road_id || car_lane[prev_car] != lane) break;
                search_idx--;
            }
            if (search_idx != sorted_idx) {
                uint first_car = sorted_indices[search_idx];
                float wrap_gap = (road_len - s) + car_s[first_car] - pc.car_length;
                if (wrap_gap < gap) {
                    gap = wrap_gap;
                    delta_v = speed - car_speed[first_car];
                }
            }
        }
    } else {
        // LEFT LANE: travel in decreasing s
        if (sorted_idx > 0) {
            uint prev_car_idx = sorted_indices[sorted_idx - 1];
            if (car_road_id[prev_car_idx] == road_id && car_lane[prev_car_idx] == lane) {
                float leader_s = car_s[prev_car_idx];
                float raw_gap = s - leader_s;
                if (raw_gap < 0.0) raw_gap += road_len;
                gap = raw_gap - pc.car_length;
                delta_v = speed - car_speed[prev_car_idx];
            }
        }
        // Wraparound
        if (gap > 900.0) {
            uint search_idx = sorted_idx;
            while (search_idx + 1 < pc.car_count) {
                uint next_car = sorted_indices[search_idx + 1];
                if (car_road_id[next_car] != road_id || car_lane[next_car] != lane) break;
                search_idx++;
            }
            if (search_idx != sorted_idx) {
                uint last_car = sorted_indices[search_idx];
                float wrap_gap = s + (road_len - car_s[last_car]) - pc.car_length;
                if (wrap_gap < gap) {
                    gap = wrap_gap;
                    delta_v = speed - car_speed[last_car];
                }
            }
        }
    }

    // Convert reference-line gap to actual lane arc-length gap
    float lane_gap = gap * lane_factor;

    float accel = idm_acceleration(speed, desired_speed, delta_v, lane_gap);

    float new_speed = speed + accel * pc.dt;
    if (gap < 0.0) new_speed = 0.0;
    new_speed = clamp(new_speed, 0.0, desired_speed * 1.5);
    if (!(new_speed >= 0.0 && new_speed <= 200.0)) new_speed = 0.0;
    car_speed[car_idx] = new_speed;

    // Advance position
    float ds = new_speed * pc.dt * ref_correction;
    if (is_left) {
        float new_s = s - ds;
        if (new_s < 0.0) new_s += road_len;
        if (!(new_s >= 0.0 && new_s < road_len)) new_s = 0.0;
        car_s[car_idx] = new_s;
    } else {
        float new_s = s + ds;
        if (new_s >= road_len) new_s -= road_len;
        if (!(new_s >= 0.0 && new_s < road_len)) new_s = 0.0;
        car_s[car_idx] = new_s;
    }
}
