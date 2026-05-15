// Traffic MOBIL lane change model

struct PushConstants {
    uint car_count;
    float a_max;
    float b_comfort;
    float s0;
    float time_headway;
    float car_length;
    float politeness;
    float threshold;
    float b_safe;
    uint stagger_phase;
    float keep_right_bias;
    uint _pad;
};

[[vk::push_constant]] ConstantBuffer<PushConstants> pc;
[[vk::binding(0, 0)]] StructuredBuffer<uint> car_road_id;
[[vk::binding(1, 0)]] StructuredBuffer<float> car_s;
[[vk::binding(2, 0)]] RWStructuredBuffer<int> car_lane;
[[vk::binding(3, 0)]] StructuredBuffer<float> car_speed;
[[vk::binding(4, 0)]] StructuredBuffer<float> car_desired_speed;
[[vk::binding(5, 0)]] StructuredBuffer<float> road_total_lengths;
[[vk::binding(6, 0)]] StructuredBuffer<uint> sorted_indices;
[[vk::binding(7, 0)]] StructuredBuffer<uint2> road_lane_counts;
[[vk::binding(8, 0)]] StructuredBuffer<float> car_preferred_headway;

float idm_accel(float v, float v0, float delta_v, float gap) {
    float sqrt_ab = sqrt(pc.a_max * pc.b_comfort);
    float s_star = pc.s0 + max(v * pc.time_headway + v * delta_v / (2.0 * sqrt_ab), 0.0);
    float v_ratio = v / max(v0, 0.1);
    float v_term = v_ratio * v_ratio * v_ratio * v_ratio;
    float gap_clamped = max(gap, 0.1);
    float s_term = (s_star / gap_clamped) * (s_star / gap_clamped);
    return pc.a_max * (1.0 - v_term - s_term);
}

float forward_gap(float s, float other_s, float road_len, float car_length) {
    float g = other_s - s;
    if (g < 0.0) g += road_len;
    return g - car_length;
}

float directional_leader_gap(float my_s, float other_s, float road_len, bool is_left, float car_length) {
    return is_left ?
        forward_gap(other_s, my_s, road_len, car_length) :
        forward_gap(my_s, other_s, road_len, car_length);
}

float directional_follower_gap(float my_s, float other_s, float road_len, bool is_left, float car_length) {
    return is_left ?
        forward_gap(my_s, other_s, road_len, car_length) :
        forward_gap(other_s, my_s, road_len, car_length);
}

// Returns (gap, speed, desired_speed) packed in float3
float3 find_leader_in_lane(
    uint road_id, int target_lane, float s, float road_len, uint sorted_idx)
{
    float best_gap = 1000.0;
    float best_speed = 0.0;
    float best_desired = 0.0;
    uint max_scan = 4;
    bool is_left = target_lane < 0;

    // Search FORWARD
    uint scan = sorted_idx + 1;
    uint count = 0;
    bool passed_target = false;
    while (scan < pc.car_count && count < max_scan) {
        uint idx = sorted_indices[scan];
        if (car_road_id[idx] != road_id) break;
        int l = car_lane[idx];
        if (l == target_lane) {
            passed_target = true;
            float g = directional_leader_gap(s, car_s[idx], road_len, is_left, pc.car_length);
            if (g >= 0.0 && g < best_gap) {
                best_gap = g;
                best_speed = car_speed[idx];
                best_desired = car_desired_speed[idx];
            }
        } else if (passed_target) break;
        scan++;
        count++;
    }

    // Search BACKWARD
    if (sorted_idx > 0) {
        scan = sorted_idx - 1;
        count = 0;
        passed_target = false;
        bool running = true;
        while (running && count < max_scan) {
            uint idx = sorted_indices[scan];
            if (car_road_id[idx] != road_id) {
                running = false;
            } else {
                int l = car_lane[idx];
                if (l == target_lane) {
                    passed_target = true;
                    float g = directional_leader_gap(s, car_s[idx], road_len, is_left, pc.car_length);
                    if (g >= 0.0 && g < best_gap) {
                        best_gap = g;
                        best_speed = car_speed[idx];
                        best_desired = car_desired_speed[idx];
                    }
                } else if (passed_target) running = false;
                if (scan == 0) running = false;
                else scan--;
                count++;
            }
        }
    }

    return float3(best_gap, best_speed, best_desired);
}

float3 find_follower_in_lane(
    uint road_id, int target_lane, float s, float road_len, uint sorted_idx)
{
    float best_gap = 1000.0;
    float best_speed = 0.0;
    float best_desired = 0.0;
    uint max_scan = 4;
    bool is_left = target_lane < 0;

    // Search FORWARD
    uint scan = sorted_idx + 1;
    uint count = 0;
    bool passed_target = false;
    while (scan < pc.car_count && count < max_scan) {
        uint idx = sorted_indices[scan];
        if (car_road_id[idx] != road_id) break;
        int l = car_lane[idx];
        if (l == target_lane) {
            passed_target = true;
            float g = directional_follower_gap(s, car_s[idx], road_len, is_left, pc.car_length);
            if (g < best_gap) {
                best_gap = g;
                best_speed = car_speed[idx];
                best_desired = car_desired_speed[idx];
            }
        } else if (passed_target) break;
        scan++;
        count++;
    }

    // Search BACKWARD
    if (sorted_idx > 0) {
        scan = sorted_idx - 1;
        count = 0;
        passed_target = false;
        bool running = true;
        while (running && count < max_scan) {
            uint idx = sorted_indices[scan];
            if (car_road_id[idx] != road_id) {
                running = false;
            } else {
                int l = car_lane[idx];
                if (l == target_lane) {
                    passed_target = true;
                    float g = directional_follower_gap(s, car_s[idx], road_len, is_left, pc.car_length);
                    if (g < best_gap) {
                        best_gap = g;
                        best_speed = car_speed[idx];
                        best_desired = car_desired_speed[idx];
                    }
                } else if (passed_target) running = false;
                if (scan == 0) running = false;
                else scan--;
                count++;
            }
        }
    }

    return float3(best_gap, best_speed, best_desired);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void traffic_lane_change_main(uint3 gid : SV_DispatchThreadID) {
    uint sorted_idx = gid.x;
    if (sorted_idx >= pc.car_count)
        return;

    uint car_idx = sorted_indices[sorted_idx];

    // Stagger: only ~1/4 of cars evaluate lane changes per dispatch
    if ((car_idx % 4) != (pc.stagger_phase % 4))
        return;

    uint road_id = car_road_id[car_idx];
    int lane = car_lane[car_idx];
    float s = car_s[car_idx];
    float speed = car_speed[car_idx];
    float desired_speed = car_desired_speed[car_idx];
    float road_len = road_total_lengths[road_id];
    float preferred_headway = car_preferred_headway[car_idx];

    uint2 lane_counts = road_lane_counts[road_id];
    int max_right = int(lane_counts.x);
    int max_left = int(lane_counts.y);

    // Find current lane acceleration
    float3 cur_leader = find_leader_in_lane(road_id, lane, s, road_len, sorted_idx);
    float a_current = idm_accel(speed, desired_speed, speed - cur_leader.y, cur_leader.x);

    int best_lane = lane;
    float best_incentive = 0.0;

    float min_gap = max(pc.car_length * 2.0, speed * preferred_headway);

    for (int dir = -1; dir <= 1; dir += 2) {
        int target_lane = lane + dir;

        // Cross-direction guard
        if (lane >= 0 && target_lane < 0) continue;
        if (lane < 0 && target_lane >= 0) continue;

        // Check lane bounds
        if (target_lane >= 0 && target_lane >= max_right) continue;
        if (target_lane < 0 && target_lane < -max_left) continue;

        // Find leader in target lane
        float3 tgt_leader = find_leader_in_lane(road_id, target_lane, s, road_len, sorted_idx);
        if (tgt_leader.x < min_gap) continue;
        float a_target = idm_accel(speed, desired_speed, speed - tgt_leader.y, tgt_leader.x);

        // Find follower in target lane
        float3 tgt_follower = find_follower_in_lane(road_id, target_lane, s, road_len, sorted_idx);
        float follower_speed = tgt_follower.y;
        float follower_desired = tgt_follower.z;
        float follower_gap = tgt_follower.x;

        bool is_rightward = (lane >= 0 && target_lane > lane) || (lane < 0 && target_lane < lane);

        // Only consider overtaking when actually slowed below desired speed
        if (!is_rightward && speed >= desired_speed * 0.98) continue;

        float follower_clearance;
        if (is_rightward)
            follower_clearance = max(follower_speed * preferred_headway + pc.car_length, pc.car_length * 3.0);
        else
            follower_clearance = min_gap;
        if (follower_gap < follower_clearance) continue;

        // Don't return right if slower car close ahead
        float keep_left_dist = speed * preferred_headway * 3.0 + pc.car_length * 4.0;
        if (is_rightward && tgt_leader.x < keep_left_dist && tgt_leader.y < speed * 0.85) continue;

        // Safety criterion
        float a_follower_new = 0.0;
        float a_follower_old = 0.0;
        if (follower_gap < 999.0) {
            a_follower_new = idm_accel(follower_speed, follower_desired, follower_speed - speed, follower_gap);
            if (a_follower_new < -pc.b_safe) continue;
            float follower_leader_gap = tgt_leader.x + follower_gap + pc.car_length;
            a_follower_old = idm_accel(follower_speed, follower_desired, follower_speed - tgt_leader.y, follower_leader_gap);
        }

        // MOBIL incentive criterion
        float incentive = (a_target - a_current) + pc.politeness * (a_follower_new - a_follower_old);
        float effective_threshold = pc.threshold;

        if (is_rightward) {
            incentive += pc.keep_right_bias;
            effective_threshold = 0.0;
        }

        if (incentive > effective_threshold && incentive > best_incentive) {
            best_incentive = incentive;
            best_lane = target_lane;
        }
    }

    if (best_lane != lane)
        car_lane[car_idx] = best_lane;
}
