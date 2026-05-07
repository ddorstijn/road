// Road rendering shader: converts SDF atlas data into final road visuals.
// Renders asphalt surface, lane markings (solid/dashed, white/yellow),
// center line, and shoulder with anti-aliased edges.

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

struct PushConstants {
    view_proj: mat4x4<f32>,
    atlas_uv_offset: vec2<f32>,
    atlas_uv_scale: vec2<f32>,
    tile_world_origin: vec2<f32>,
    tile_world_size: vec2<f32>,
}

var<immediate> pc: PushConstants;

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VertexOutput {
    var out: VertexOutput;

    // Unit quad: 2 triangles (6 vertices)
    var qx: f32;
    var qy: f32;
    switch vi {
        case 0u { qx = 0.0; qy = 0.0; }
        case 1u { qx = 1.0; qy = 0.0; }
        case 2u { qx = 1.0; qy = 1.0; }
        case 3u { qx = 0.0; qy = 0.0; }
        case 4u { qx = 1.0; qy = 1.0; }
        default { qx = 0.0; qy = 1.0; }
    }
    let local = vec2<f32>(qx, qy);
    let world = pc.tile_world_origin + local * pc.tile_world_size;
    out.position = pc.view_proj * vec4<f32>(world, 0.0, 1.0);
    out.uv = pc.atlas_uv_offset + local * pc.atlas_uv_scale;
    return out;
}

// --- Bindings ---
@group(0) @binding(0) var sdf_tex: texture_2d<f32>;
@group(0) @binding(1) var sdf_sampler: sampler;
@group(0) @binding(2) var<storage, read> roads: array<GpuRoad>;
@group(0) @binding(3) var<storage, read> lane_sections: array<GpuLaneSection>;
@group(0) @binding(4) var<storage, read> lanes: array<GpuLane>;

// Marking constants
const MARKING_HALF_WIDTH: f32 = 0.12;  // 12 cm half-width
const MARKING_AA: f32 = 0.06;          // anti-aliasing transition
const DASH_LENGTH: f32 = 3.0;          // 3 m dash
const GAP_LENGTH: f32 = 3.0;           // 3 m gap
const SHOULDER_WIDTH: f32 = 0.5;       // 0.5 m shoulder zone

// Colors
const ASPHALT: vec3<f32> = vec3<f32>(0.20, 0.20, 0.22);
const SHOULDER_COLOR: vec3<f32> = vec3<f32>(0.35, 0.35, 0.33);
const WHITE: vec3<f32> = vec3<f32>(0.95, 0.95, 0.95);
const YELLOW: vec3<f32> = vec3<f32>(1.0, 0.85, 0.0);

fn marking_visibility(dist_to_boundary: f32) -> f32 {
    return 1.0 - smoothstep(MARKING_HALF_WIDTH - MARKING_AA, MARKING_HALF_WIDTH + MARKING_AA, dist_to_boundary);
}

fn marking_color(marking_type: u32) -> vec3<f32> {
    // 1 = SolidWhite, 2 = DashedWhite, 3 = SolidYellow, 4 = DashedYellow, 5 = DoubleSolidYellow
    if marking_type >= 3u {
        return YELLOW;
    }
    return WHITE;
}

fn is_dashed(marking_type: u32) -> bool {
    return marking_type == 2u || marking_type == 4u;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let sdf_val = textureSample(sdf_tex, sdf_sampler, in.uv);
    let signed_dist = sdf_val.x;
    let s_coord = sdf_val.y;
    let road_id_f = sdf_val.z;
    let s_mod_period = sdf_val.w;  // s % 6.0, high-precision for dash patterns

    // No road nearby — fully transparent
    if road_id_f < 0.0 {
        discard;
    }

    let road_id = u32(road_id_f);
    let road = roads[road_id];

    // Find lane section at this s-coordinate
    var section_idx = road.lane_section_offset;
    for (var i = 0u; i < road.lane_section_count; i++) {
        let idx = road.lane_section_offset + i;
        let ls = lane_sections[idx];
        if s_coord >= ls.s_start && s_coord <= ls.s_end {
            section_idx = idx;
            break;
        }
    }
    let section = lane_sections[section_idx];

    let abs_dist = abs(signed_dist);
    let on_left = signed_dist >= 0.0;

    // Determine which lanes to check based on side
    var lane_start: u32;
    var lane_end: u32;
    if on_left {
        lane_start = section.lane_offset;
        lane_end = section.lane_offset + section.left_lane_count;
    } else {
        lane_start = section.lane_offset + section.left_lane_count;
        lane_end = section.lane_offset + section.lane_count;
    }

    // Compute total road width on this side (for edge detection)
    var total_side_width = 0.0;
    for (var i = lane_start; i < lane_end; i++) {
        total_side_width += lanes[i].width;
    }

    // Check if outside road surface (with shoulder fade)
    if abs_dist > total_side_width + SHOULDER_WIDTH {
        discard;
    }

    // Start with base color
    var color = ASPHALT;
    var alpha = 1.0;

    // Shoulder zone — fade from asphalt to transparent at the edge
    if abs_dist > total_side_width {
        let shoulder_t = (abs_dist - total_side_width) / SHOULDER_WIDTH;
        color = mix(ASPHALT, SHOULDER_COLOR, shoulder_t);
        alpha = 1.0 - smoothstep(0.0, 1.0, shoulder_t);
        // Still apply edge marking below
    }

    // --- Marking pass: iterate lane boundaries on this side ---
    var best_marking_vis = 0.0;
    var best_marking_color = WHITE;

    var accum = 0.0;
    for (var i = lane_start; i < lane_end; i++) {
        let lane = lanes[i];
        let outer_edge = accum + lane.width;

        // Check outer boundary marking of this lane
        if lane.marking_type != 0u {
            let boundary_dist = abs(abs_dist - outer_edge);
            var vis = marking_visibility(boundary_dist);

            if is_dashed(lane.marking_type) {
                // Use pre-computed s % 6.0 from atlas alpha for fp16 precision
                if s_mod_period > DASH_LENGTH {
                    vis = 0.0;
                }
            }

            if vis > best_marking_vis {
                best_marking_vis = vis;
                best_marking_color = marking_color(lane.marking_type);
            }
        }

        accum = outer_edge;
    }

    // --- Center line (double solid yellow at signed_dist ≈ 0) ---
    let center_gap = 0.06;  // 6 cm gap between the two yellow lines
    let d1 = abs(signed_dist - center_gap);
    let d2 = abs(signed_dist + center_gap);
    let c1 = marking_visibility(d1);
    let c2 = marking_visibility(d2);
    let center_vis = max(c1, c2);

    if center_vis > best_marking_vis {
        best_marking_vis = center_vis;
        best_marking_color = YELLOW;
    }

    // Composite marking on top of base color
    color = mix(color, best_marking_color, best_marking_vis);

    return vec4<f32>(color, alpha);
}
