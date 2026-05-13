// SDF tile generation compute shader

import gpu_shared;
import road_eval;

struct SdfPushConstants {
    uint atlas_offset_x;
    uint atlas_offset_y;
    float tile_world_x;
    float tile_world_y;
    float tile_size;
    uint tile_resolution;
    uint tile_index;
    uint road_id_atlas_offset_x;
    uint road_id_atlas_offset_y;
    uint road_id_resolution;
    uint tile_border;
    uint _pad;
};

[[vk::push_constant]] ConstantBuffer<SdfPushConstants> pc;
[[vk::binding(0, 0), vk::image_format("rg16f")]] RWTexture2D<float2> sdf_atlas;
[[vk::binding(1, 0)]] StructuredBuffer<GpuSegment> segments;
[[vk::binding(2, 0)]] StructuredBuffer<TileHeader> tile_headers;
[[vk::binding(3, 0)]] StructuredBuffer<uint> tile_segment_indices;
[[vk::binding(4, 0)]] StructuredBuffer<uint> road_indices;
[[vk::binding(5, 0), vk::image_format("r16f")]] RWTexture2D<float> road_id_atlas;

[shader("compute")]
[numthreads(16, 16, 1)]
void sdf_generate_main(uint3 id : SV_DispatchThreadID) {
    // Total texels per tile slot = data + 2 * border
    uint dispatch_size = pc.tile_resolution + 2 * pc.tile_border;
    if (id.x >= dispatch_size || id.y >= dispatch_size)
        return;

    float texel_size = pc.tile_size / float(pc.tile_resolution);
    float world_x = pc.tile_world_x + (float(id.x) - float(pc.tile_border) + 0.5) * texel_size;
    float world_y = pc.tile_world_y + (float(id.y) - float(pc.tile_border) + 0.5) * texel_size;
    float2 world_pos = float2(world_x, world_y);

    TileHeader header = tile_headers[pc.tile_index];

    float min_dist = 1e10;
    float best_s = 0.0;
    float best_road = -1.0;
    float best_signed_dist = 1e10;

    for (uint i = 0; i < header.count; i++) {
        uint seg_idx = tile_segment_indices[header.offset + i];
        uint road_idx = road_indices[header.offset + i];
        GpuSegment seg = segments[seg_idx];

        float2 origin = seg.origin;
        float2 local_pt = world_to_segment_local(world_pos, origin, seg.heading);

        ClosestResult cp = closest_point_on_segment(
            seg.segment_type, local_pt, seg.length, seg.k_start, seg.k_end);

        float abs_dist = abs(cp.signed_dist);
        if (abs_dist < min_dist) {
            min_dist = abs_dist;
            best_signed_dist = cp.signed_dist;
            best_s = seg.s_start + cp.s;
            best_road = float(road_idx);
        }
    }

    if (min_dist > 20.0)
        best_road = -1.0;

    // Write SDF atlas (RG16F): signed_dist, s_coord
    int2 atlas_pos = int2(pc.atlas_offset_x + id.x, pc.atlas_offset_y + id.y);
    sdf_atlas[atlas_pos] = float2(best_signed_dist, best_s);

    // Write road ID atlas (R16_SFLOAT) at lower resolution
    uint interior_x = id.x - pc.tile_border;
    uint interior_y = id.y - pc.tile_border;
    if (interior_x < pc.tile_resolution && interior_y < pc.tile_resolution) {
        uint ratio = pc.tile_resolution / pc.road_id_resolution;
        if (interior_x % ratio == 0 && interior_y % ratio == 0) {
            uint rid_x = interior_x / ratio;
            uint rid_y = interior_y / ratio;
            int2 road_id_pos = int2(pc.road_id_atlas_offset_x + rid_x,
                                    pc.road_id_atlas_offset_y + rid_y);
            road_id_atlas[road_id_pos] = best_road;
        }
    }
}
