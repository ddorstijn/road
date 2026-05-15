// Traffic sort: histogram pass

struct PushConstants {
    uint pass_id;
    uint car_count;
    uint num_workgroups;
    uint _pad;
};

static const uint TILE_SIZE = 256;

[[vk::push_constant]] ConstantBuffer<PushConstants> pc;
[[vk::binding(0, 0)]] StructuredBuffer<uint> keys_in;
[[vk::binding(1, 0)]] RWStructuredBuffer<uint> histograms;

[shader("compute")]
[numthreads(256, 1, 1)]
void traffic_sort_histogram_main(
    uint3 lid : SV_GroupThreadID,
    uint3 wg_id : SV_GroupID)
{
    uint local_idx = lid.x;
    uint group_id = wg_id.x;
    uint global_idx = group_id * TILE_SIZE + local_idx;

    if (global_idx < pc.car_count) {
        uint key = keys_in[global_idx];
        uint digit = (key >> (pc.pass_id * 8)) & 0xFF;
        uint hist_idx = digit * pc.num_workgroups + group_id;
        uint original;
        InterlockedAdd(histograms[hist_idx], 1, original);
    }
}
