// Road surface rendering with lane markings from SDF atlas

import gpu_shared;

struct PushConstants {
    float4x4 view_proj;
};

[[vk::push_constant]] ConstantBuffer<PushConstants> pc;
[[vk::binding(0, 0)]] Texture2D<float4> sdf_tex;
[[vk::binding(1, 0)]] SamplerState sdf_sampler;
[[vk::binding(2, 0)]] StructuredBuffer<GpuRoad> roads;
[[vk::binding(3, 0)]] StructuredBuffer<GpuLaneSection> lane_sections_buf;
[[vk::binding(4, 0)]] StructuredBuffer<GpuLane> lanes_buf;
[[vk::binding(5, 0)]] StructuredBuffer<GpuTileInstance> tile_instances;
[[vk::binding(6, 0)]] Texture2D<float4> road_id_tex;
[[vk::binding(7, 0)]] SamplerState road_id_sampler;

// Marking constants
static const float MARKING_HALF_WIDTH = 0.12;
static const float MARKING_AA = 0.06;
static const float DASH_LENGTH = 3.0;
static const float SHOULDER_WIDTH = 0.5;

// Colors
static const float3 ASPHALT = float3(0.20, 0.20, 0.22);
static const float3 SHOULDER_COLOR = float3(0.35, 0.35, 0.33);
static const float3 WHITE = float3(0.95, 0.95, 0.95);
static const float3 YELLOW = float3(1.0, 0.85, 0.0);

float smoothstep_manual(float edge0, float edge1, float x) {
    float t = clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0);
    return t * t * (3.0 - 2.0 * t);
}

float3 mix_vec3(float3 a, float3 b, float t) {
    return a * (1.0 - t) + b * t;
}

float marking_visibility(float dist_to_boundary, float sdf_pixel_w) {
    float aa = max(MARKING_AA, sdf_pixel_w);
    float vis = 1.0 - smoothstep_manual(max(MARKING_HALF_WIDTH - aa, 0.0),
                                         MARKING_HALF_WIDTH + aa,
                                         dist_to_boundary);
    if (sdf_pixel_w > MARKING_AA) {
        return (vis > 0.25) ? 1.0 : 0.0;
    }
    return vis;
}

float3 marking_color(uint marking_type) {
    return (marking_type >= 3) ? YELLOW : WHITE;
}

bool is_dashed(uint marking_type) {
    return marking_type == 2 || marking_type == 4;
}

struct VsOutput {
    float4 position : SV_Position;
    float2 uv : TEXCOORD0;
    float2 road_id_uv : TEXCOORD1;
};

[shader("vertex")]
VsOutput vs_main(uint vi : SV_VertexID, uint ii : SV_InstanceID) {
    GpuTileInstance tile = tile_instances[ii];
    float2 tile_world_origin = tile.tile_world_origin;
    float2 tile_world_size = tile.tile_world_size;
    float2 atlas_uv_offset = tile.atlas_uv_offset;
    float2 atlas_uv_scale = tile.atlas_uv_scale;
    float2 road_id_uv_offset = tile.road_id_uv_offset;
    float2 road_id_uv_scale = tile.road_id_uv_scale;

    float qx, qy;
    switch (vi) {
    case 0: qx = 0.0; qy = 0.0; break;
    case 1: qx = 1.0; qy = 0.0; break;
    case 2: qx = 1.0; qy = 1.0; break;
    case 3: qx = 0.0; qy = 0.0; break;
    case 4: qx = 1.0; qy = 1.0; break;
    default: qx = 0.0; qy = 1.0; break;
    }
    float2 local = float2(qx, qy);
    float2 world = tile_world_origin + local * tile_world_size;

    VsOutput output;
    output.position = mul(pc.view_proj, float4(world, 0.0, 1.0));
    output.uv = atlas_uv_offset + local * atlas_uv_scale;
    output.road_id_uv = road_id_uv_offset + local * road_id_uv_scale;
    return output;
}

[shader("fragment")]
float4 fs_main(VsOutput input) : SV_Target {
    float2 uv = input.uv;
    float2 road_id_uv = input.road_id_uv;

    // Sample SDF atlas (bilinear): signed_dist + s_coord
    float4 sdf_val = sdf_tex.Sample(sdf_sampler, uv);
    float signed_dist = sdf_val.x;
    float s_coord = sdf_val.y;

    // Sample road ID atlas (nearest)
    float4 road_id_val = road_id_tex.Sample(road_id_sampler, road_id_uv);
    float road_id_f = road_id_val.x;

    float s_mod_period = fmod(s_coord, 6.0);

    if (!(road_id_f >= 0.0 && road_id_f < 255.0))
        discard;

    uint road_id = uint(road_id_f);
    GpuRoad road = roads[road_id];

    if (s_coord < -0.5 || s_coord > road.total_length + 0.5)
        discard;

    // Find lane section at this s-coordinate
    uint section_idx = road.lane_section_offset;
    for (uint i = 0; i < road.lane_section_count; i++) {
        uint idx = road.lane_section_offset + i;
        GpuLaneSection ls = lane_sections_buf[idx];
        if (s_coord >= ls.s_start && s_coord <= ls.s_end) {
            section_idx = idx;
            break;
        }
    }
    GpuLaneSection section = lane_sections_buf[section_idx];

    float abs_dist = abs(signed_dist);
    bool on_left = signed_dist >= 0.0;

    uint lane_start, lane_end;
    if (on_left) {
        lane_start = section.lane_offset;
        lane_end = section.lane_offset + section.left_lane_count;
    } else {
        lane_start = section.lane_offset + section.left_lane_count;
        lane_end = section.lane_offset + section.lane_count;
    }

    // Compute total road width on this side
    float total_side_width = 0.0;
    for (uint j = lane_start; j < lane_end; j++)
        total_side_width += lanes_buf[j].width;

    if (abs_dist > total_side_width + SHOULDER_WIDTH)
        discard;

    float3 color = ASPHALT;
    float alpha = 1.0;

    // Shoulder zone
    if (abs_dist > total_side_width) {
        float shoulder_t = (abs_dist - total_side_width) / SHOULDER_WIDTH;
        color = mix_vec3(ASPHALT, SHOULDER_COLOR, shoulder_t);
        alpha = 1.0 - smoothstep_manual(0.0, 1.0, shoulder_t);
    }

    // Screen-space pixel width from smooth signed_dist
    float sdf_pixel_w = min(fwidth(signed_dist), MARKING_HALF_WIDTH * 2.0);

    // Marking pass
    float best_marking_vis = 0.0;
    float3 best_marking_color = WHITE;

    float accum = 0.0;
    for (uint k = lane_start; k < lane_end; k++) {
        GpuLane lane = lanes_buf[k];
        float outer_edge = accum + lane.width;

        if (lane.marking_type != 0) {
            float boundary_dist = abs(abs_dist - outer_edge);
            float vis = marking_visibility(boundary_dist, sdf_pixel_w);

            if (is_dashed(lane.marking_type)) {
                if (s_mod_period > DASH_LENGTH)
                    vis = 0.0;
            }

            if (vis > best_marking_vis) {
                best_marking_vis = vis;
                best_marking_color = marking_color(lane.marking_type);
            }
        }

        accum = outer_edge;
    }

    // Center line
    float center_vis = marking_visibility(abs(signed_dist), sdf_pixel_w);
    if (center_vis > best_marking_vis) {
        best_marking_vis = center_vis;
        best_marking_color = YELLOW;
    }

    color = mix_vec3(color, best_marking_color, best_marking_vis);

    return float4(color, alpha);
}
