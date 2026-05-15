// Traffic sort: scatter pass

struct PushConstants {
    uint pass_id;
    uint car_count;
    uint num_workgroups;
    uint _pad;
};

static const uint TILE_SIZE = 256;

[[vk::push_constant]] ConstantBuffer<PushConstants> pc;
[[vk::binding(0, 0)]] StructuredBuffer<uint> keys_in;
[[vk::binding(1, 0)]] StructuredBuffer<uint> vals_in;
[[vk::binding(2, 0)]] RWStructuredBuffer<uint> keys_out;
[[vk::binding(3, 0)]] RWStructuredBuffer<uint> vals_out;
[[vk::binding(4, 0)]] StructuredBuffer<uint> histograms;

[shader("compute")]
[numthreads(256, 1, 1)]
void traffic_sort_scatter_main(
    uint3 lid : SV_GroupThreadID,
    uint3 wg_id : SV_GroupID)
{
    uint local_idx = lid.x;
    uint group_id = wg_id.x;
    uint global_idx = group_id * TILE_SIZE + local_idx;

    if (global_idx >= pc.car_count)
        return;

    uint key = keys_in[global_idx];
    uint val = vals_in[global_idx];
    uint digit = (key >> (pc.pass_id * 8)) & 0xFF;

    // Compute local rank: count how many preceding elements in this tile
    // have the same digit (for stable sort)
    uint local_rank = 0;
    uint tile_start = group_id * TILE_SIZE;
    for (uint i = 0; i < local_idx; i++) {
        uint other_idx = tile_start + i;
        if (other_idx < pc.car_count) {
            uint other_key = keys_in[other_idx];
            uint other_digit = (other_key >> (pc.pass_id * 8)) & 0xFF;
            if (other_digit == digit)
                local_rank++;
        }
    }

    // Output position = global prefix sum offset + local rank
    uint global_offset = histograms[digit * pc.num_workgroups + group_id];
    uint dst = global_offset + local_rank;

    keys_out[dst] = key;
    vals_out[dst] = val;
}
