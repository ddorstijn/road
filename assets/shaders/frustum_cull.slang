// GPU frustum culling: car culling compute shader

import gpu_shared;
import road_eval;

struct CarCullPushConstants {
    float view_min_x;
    float view_min_y;
    float view_max_x;
    float view_max_y;
    uint car_count;
    uint _pad0;
    uint _pad1;
    uint _pad2;
};

[[vk::push_constant]] ConstantBuffer<CarCullPushConstants> pc;
[[vk::binding(0, 0)]] StructuredBuffer<uint> car_road_id;
[[vk::binding(1, 0)]] StructuredBuffer<float> car_s;
[[vk::binding(2, 0)]] StructuredBuffer<int> car_lane;
[[vk::binding(3, 0)]] StructuredBuffer<GpuSegment> segments;
[[vk::binding(4, 0)]] StructuredBuffer<GpuRoad> roads_buf;
[[vk::binding(5, 0)]] StructuredBuffer<GpuLaneSection> lane_sections;
[[vk::binding(6, 0)]] StructuredBuffer<GpuLane> lanes_buf;
[[vk::binding(7, 0)]] RWStructuredBuffer<uint> draw_indirect;
[[vk::binding(8, 0)]] RWStructuredBuffer<uint> visible_indices;

static const float CAR_MARGIN = 3.5;

[shader("compute")]
[numthreads(256, 1, 1)]
void car_cull_main(uint3 gid : SV_DispatchThreadID) {
    uint idx = gid.x;
    if (idx >= pc.car_count)
        return;

    uint road_id = car_road_id[idx];
    float s = car_s[idx];
    int lane = car_lane[idx];

    PoseResult pose = evaluate_road_at_s(road_id, s, roads_buf, segments);
    float lat_offset = compute_lane_offset(road_id, lane, s, roads_buf, lane_sections, lanes_buf);
    float2 normal = float2(-sin(pose.heading), cos(pose.heading));
    float2 center = pose.position + normal * lat_offset;

    if (center.x + CAR_MARGIN < pc.view_min_x ||
        center.x - CAR_MARGIN > pc.view_max_x ||
        center.y + CAR_MARGIN < pc.view_min_y ||
        center.y - CAR_MARGIN > pc.view_max_y)
        return;

    uint slot;
    InterlockedAdd(draw_indirect[1], 1, slot);
    visible_indices[slot] = idx;
}
