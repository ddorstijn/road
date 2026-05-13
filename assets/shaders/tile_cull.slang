// GPU frustum culling: tile culling compute shader

import gpu_shared;

struct TileCullPushConstants {
    float view_min_x;
    float view_min_y;
    float view_max_x;
    float view_max_y;
    uint tile_count;
    uint _pad0;
    uint _pad1;
    uint _pad2;
};

[[vk::push_constant]] ConstantBuffer<TileCullPushConstants> pc;
[[vk::binding(0, 0)]] StructuredBuffer<GpuTileInstance> all_tiles;
[[vk::binding(1, 0)]] RWStructuredBuffer<uint> draw_indirect;
[[vk::binding(2, 0)]] RWStructuredBuffer<GpuTileInstance> visible_tiles;

[shader("compute")]
[numthreads(64, 1, 1)]
void tile_cull_main(uint3 gid : SV_DispatchThreadID) {
    uint idx = gid.x;
    if (idx >= pc.tile_count)
        return;

    GpuTileInstance tile = all_tiles[idx];
    float tile_min_x = tile.tile_world_origin.x;
    float tile_min_y = tile.tile_world_origin.y;
    float tile_max_x = tile_min_x + tile.tile_world_size.x;
    float tile_max_y = tile_min_y + tile.tile_world_size.y;

    if (tile_max_x < pc.view_min_x ||
        tile_min_x > pc.view_max_x ||
        tile_max_y < pc.view_min_y ||
        tile_min_y > pc.view_max_y)
        return;

    uint slot;
    InterlockedAdd(draw_indirect[1], 1, slot);
    visible_tiles[slot] = tile;
}
