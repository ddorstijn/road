// Car rendering — instanced vertex + fragment shader

import gpu_shared;
import road_eval;

struct PushConstants {
    float4x4 view_proj;
    uint car_count;
    uint _pad0;
    uint _pad1;
    uint _pad2;
};

[[vk::push_constant]] ConstantBuffer<PushConstants> pc;
[[vk::binding(0, 0)]] StructuredBuffer<uint> car_road_id;
[[vk::binding(1, 0)]] StructuredBuffer<float> car_s;
[[vk::binding(2, 0)]] StructuredBuffer<int> car_lane;
[[vk::binding(3, 0)]] StructuredBuffer<float> car_speed;
[[vk::binding(4, 0)]] StructuredBuffer<float> car_desired_speed;
[[vk::binding(5, 0)]] StructuredBuffer<GpuSegment> segments;
[[vk::binding(6, 0)]] StructuredBuffer<GpuRoad> roads_buf;
[[vk::binding(7, 0)]] StructuredBuffer<GpuLaneSection> lane_sections;
[[vk::binding(8, 0)]] StructuredBuffer<GpuLane> lanes_buf;
[[vk::binding(9, 0)]] StructuredBuffer<uint> visible_indices;

static const float CAR_LENGTH = 4.5;
static const float CAR_WIDTH = 2.0;

struct VsOutput {
    float4 position : SV_Position;
    float3 color : COLOR0;
    float2 uv : TEXCOORD0;
    float speed_ratio : TEXCOORD1;
};

float smoothstep_manual(float edge0, float edge1, float x) {
    float t = clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0);
    return t * t * (3.0 - 2.0 * t);
}

float step_manual(float edge, float x) {
    return (x < edge) ? 0.0 : 1.0;
}

float3 mix_vec3(float3 a, float3 b, float t) {
    return a * (1.0 - t) + b * t;
}

[shader("vertex")]
VsOutput vs_main(uint vi : SV_VertexID, uint ii : SV_InstanceID) {
    uint car_idx = visible_indices[ii];

    uint road_id = car_road_id[car_idx];
    float s = car_s[car_idx];
    int lane = car_lane[car_idx];
    float speed = car_speed[car_idx];
    float desired = car_desired_speed[car_idx];

    PoseResult pose = evaluate_road_at_s(road_id, s, roads_buf, segments);
    float lat_offset = compute_lane_offset(road_id, lane, s, roads_buf, lane_sections, lanes_buf);

    float2 normal = float2(-sin(pose.heading), cos(pose.heading));
    float2 center = pose.position + normal * lat_offset;

    float qx, qy;
    switch (vi) {
    case 0: qx = -0.5; qy = -0.5; break;
    case 1: qx =  0.5; qy = -0.5; break;
    case 2: qx =  0.5; qy =  0.5; break;
    case 3: qx = -0.5; qy = -0.5; break;
    case 4: qx =  0.5; qy =  0.5; break;
    case 5: qx = -0.5; qy =  0.5; break;
    default: qx = 0.0; qy = 0.0; break;
    }

    VsOutput output;
    output.uv = float2(qx + 0.5, qy + 0.5);

    float2 local = float2(qx * CAR_LENGTH, qy * CAR_WIDTH);
    float c = cos(pose.heading);
    float sn = sin(pose.heading);
    float2 rotated = float2(c * local.x - sn * local.y, sn * local.x + c * local.y);
    float2 world_pos = center + rotated;

    output.position = mul(pc.view_proj, float4(world_pos, 0.0, 1.0));

    float ratio = clamp(speed / max(desired, 0.1), 0.0, 1.0);
    output.speed_ratio = ratio;

    float r = clamp(1.0 - ratio * 1.5, 0.0, 1.0);
    float g = clamp(ratio * 1.5, 0.0, 1.0);
    output.color = float3(r, g, 0.1);

    return output;
}

float sdf_rounded_rect(float2 p, float2 half_size, float radius) {
    float2 q = abs(p) - half_size + float2(radius, radius);
    return length(max(q, float2(0.0, 0.0))) + min(max(q.x, q.y), 0.0) - radius;
}

[shader("fragment")]
float4 fs_main(VsOutput input) : SV_Target {
    float2 centered = input.uv - float2(0.5, 0.5);
    float3 color = input.color;

    float body_sdf = sdf_rounded_rect(centered, float2(0.45, 0.42), 0.08);

    if (body_sdf > 0.0)
        discard;

    float aa = clamp(fwidth(body_sdf), 0.001, 0.05);
    float edge_alpha = 1.0 - smoothstep_manual(-aa * 1.5, 0.0, body_sdf);

    // Outline
    float outline_width = 0.04;
    float outline_factor = smoothstep_manual(-outline_width - aa, -outline_width, body_sdf);
    float3 outline_color = float3(0.05, 0.05, 0.05);

    // Windshield
    float windshield_start = 0.68;
    float windshield_end = 0.82;
    float windshield_side = 0.32;
    float is_windshield = step_manual(windshield_start, input.uv.x)
        * (1.0 - step_manual(windshield_end, input.uv.x))
        * step_manual(0.5 - windshield_side, input.uv.y)
        * (1.0 - step_manual(0.5 + windshield_side, input.uv.y));
    float3 windshield_color = float3(0.55, 0.7, 0.85);

    float3 final_color = mix_vec3(color, outline_color, outline_factor);
    final_color = mix_vec3(final_color, windshield_color, is_windshield * 0.7);

    return float4(final_color, edge_alpha);
}
