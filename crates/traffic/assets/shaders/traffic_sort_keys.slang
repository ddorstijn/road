// Traffic sort: key generation

struct PushConstants {
    uint car_count;
    uint _pad0;
    uint _pad1;
    uint _pad2;
};

[[vk::push_constant]] ConstantBuffer<PushConstants> pc;
[[vk::binding(0, 0)]] StructuredBuffer<uint> car_road_id;
[[vk::binding(1, 0)]] StructuredBuffer<float> car_s;
[[vk::binding(2, 0)]] StructuredBuffer<int> car_lane;
[[vk::binding(3, 0)]] StructuredBuffer<float> road_total_lengths;
[[vk::binding(4, 0)]] RWStructuredBuffer<uint> sort_keys;
[[vk::binding(5, 0)]] RWStructuredBuffer<uint> sort_vals;

[shader("compute")]
[numthreads(256, 1, 1)]
void traffic_sort_keys_main(uint3 gid : SV_DispatchThreadID) {
    uint idx = gid.x;
    if (idx >= pc.car_count) {
        sort_keys[idx] = 0xFFFFFFFF;
        sort_vals[idx] = idx;
        return;
    }

    uint road_id = car_road_id[idx];
    float s = car_s[idx];
    int lane = car_lane[idx];
    float road_len = road_total_lengths[road_id];

    // Quantize s to 16 bits (0..65535)
    float s_norm = clamp(s / max(road_len, 0.001), 0.0, 1.0);
    uint s_quant = uint(s_norm * 65535.0);

    // Map lane to unsigned: lane + 128 -> 0..255
    uint lane_u = uint(clamp(lane + 128, 0, 255));

    // Build key: road_id[31:24] | lane[23:16] | s[15:0]
    uint key = (road_id << 24) | (lane_u << 16) | s_quant;

    sort_keys[idx] = key;
    sort_vals[idx] = idx;
}
