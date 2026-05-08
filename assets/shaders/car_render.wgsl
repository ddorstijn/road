// Car rendering shader: renders cars as oriented rectangles with visual detail.
// Instanced drawing: 6 vertices per car (two triangles forming a quad).
// Reads car SoA SSBOs + road data, evaluates road at (road_id, s) → world position.

struct PushConstants {
    view_proj: mat4x4<f32>,
    car_count: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

var<immediate> pc: PushConstants;

// Car SoA SSBOs
@group(0) @binding(0) var<storage, read> car_road_id: array<u32>;
@group(0) @binding(1) var<storage, read> car_s: array<f32>;
@group(0) @binding(2) var<storage, read> car_lane: array<i32>;
@group(0) @binding(3) var<storage, read> car_speed: array<f32>;
@group(0) @binding(4) var<storage, read> car_desired_speed: array<f32>;

// Road data
@group(0) @binding(5) var<storage, read> segments: array<GpuSegment>;
@group(0) @binding(6) var<storage, read> roads: array<GpuRoad>;
@group(0) @binding(7) var<storage, read> lane_sections: array<GpuLaneSection>;
@group(0) @binding(8) var<storage, read> lanes: array<GpuLane>;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec3<f32>,
    @location(1) uv: vec2<f32>,            // local quad coords [-0.5, 0.5]
    @location(2) speed_ratio: f32,
}

const CAR_LENGTH: f32 = 4.5;
const CAR_WIDTH: f32 = 2.0;

/// Evaluate road reference line at s → (world_pos, heading)
fn evaluate_road_at_s(road_id: u32, s_global: f32) -> PoseResult {
    let road = roads[road_id];

    var seg_idx = road.segment_offset;
    var local_s = s_global;

    for (var i = 0u; i < road.segment_count; i++) {
        let seg = segments[road.segment_offset + i];
        if s_global < seg.s_start + seg.length || i == road.segment_count - 1u {
            seg_idx = road.segment_offset + i;
            local_s = s_global - seg.s_start;
            local_s = clamp(local_s, 0.0, seg.length);
            break;
        }
    }

    let seg = segments[seg_idx];
    let pose = eval_segment(seg.segment_type, local_s, seg.k_start, seg.k_end, seg.length);

    var world_pose: PoseResult;
    world_pose.position = segment_local_to_world(pose.position, seg.origin, seg.heading);
    world_pose.heading = seg.heading + pose.heading;
    return world_pose;
}

/// Compute lateral offset for a given lane at s position
fn compute_lane_offset(road_id: u32, lane_idx: i32, s_global: f32) -> f32 {
    let road = roads[road_id];

    var section_idx = road.lane_section_offset;
    for (var i = 0u; i < road.lane_section_count; i++) {
        let ls = lane_sections[road.lane_section_offset + i];
        if s_global >= ls.s_start && s_global < ls.s_end {
            section_idx = road.lane_section_offset + i;
            break;
        }
        section_idx = road.lane_section_offset + i;
    }

    let section = lane_sections[section_idx];
    let left_count = i32(section.left_lane_count);

    var offset = 0.0;

    if lane_idx >= 0 {
        let right_start = u32(left_count);
        for (var i = 0u; i < u32(lane_idx); i++) {
            let l = lanes[section.lane_offset + right_start + i];
            offset += l.width;
        }
        let current = lanes[section.lane_offset + right_start + u32(lane_idx)];
        offset += current.width * 0.5;
        offset = -offset;
    } else {
        let left_idx = u32(-lane_idx - 1);
        for (var i = 0u; i < left_idx; i++) {
            let l = lanes[section.lane_offset + i];
            offset += l.width;
        }
        let current = lanes[section.lane_offset + left_idx];
        offset += current.width * 0.5;
    }

    return offset;
}

@vertex
fn vs_main(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VertexOutput {
    var out: VertexOutput;

    let road_id = car_road_id[ii];
    let s = car_s[ii];
    let lane = car_lane[ii];
    let speed = car_speed[ii];
    let desired = car_desired_speed[ii];

    let pose = evaluate_road_at_s(road_id, s);
    let lat_offset = compute_lane_offset(road_id, lane, s);

    let normal = vec2<f32>(-sin(pose.heading), cos(pose.heading));
    let center = pose.position + normal * lat_offset;

    // Unit quad vertex positions (6 verts = 2 triangles)
    var qx: f32;
    var qy: f32;
    switch vi {
        case 0u { qx = -0.5; qy = -0.5; }
        case 1u { qx = 0.5; qy = -0.5; }
        case 2u { qx = 0.5; qy = 0.5; }
        case 3u { qx = -0.5; qy = -0.5; }
        case 4u { qx = 0.5; qy = 0.5; }
        case 5u { qx = -0.5; qy = 0.5; }
        default { qx = 0.0; qy = 0.0; }
    }

    out.uv = vec2<f32>(qx + 0.5, qy + 0.5); // [0,1] range

    // Scale by car dimensions and rotate by heading
    let local = vec2<f32>(qx * CAR_LENGTH, qy * CAR_WIDTH);
    let c = cos(pose.heading);
    let sn = sin(pose.heading);
    let rotated = vec2<f32>(c * local.x - sn * local.y, sn * local.x + c * local.y);
    let world_pos = center + rotated;

    out.position = pc.view_proj * vec4<f32>(world_pos, 0.0, 1.0);

    // Speed ratio for coloring
    let ratio = clamp(speed / max(desired, 0.1), 0.0, 1.0);
    out.speed_ratio = ratio;

    // Base body color: red (stopped) → yellow (slow) → green (free flow)
    let r = clamp(1.0 - ratio * 1.5, 0.0, 1.0);
    let g = clamp(ratio * 1.5, 0.0, 1.0);
    out.color = vec3<f32>(r, g, 0.1);

    return out;
}

// Rounded rectangle SDF for the car body shape
fn sdf_rounded_rect(p: vec2<f32>, half_size: vec2<f32>, radius: f32) -> f32 {
    let q = abs(p) - half_size + vec2<f32>(radius);
    return min(max(q.x, q.y), 0.0) + length(max(q, vec2<f32>(0.0))) - radius;
}

@fragment
fn fs_main(
    @location(0) color: vec3<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) speed_ratio: f32,
) -> @location(0) vec4<f32> {
    // UV is in [0,1], center at (0.5, 0.5)
    let centered = uv - vec2<f32>(0.5);  // [-0.5, 0.5]

    // Car body: rounded rectangle
    let body_sdf = sdf_rounded_rect(centered, vec2<f32>(0.45, 0.42), 0.08);

    // Discard outside car body
    if body_sdf > 0.0 {
        discard;
    }

    // Anti-aliased edge
    let aa = fwidth(body_sdf);
    let edge_alpha = 1.0 - smoothstep(-aa * 1.5, 0.0, body_sdf);

    // Outline: darken near edge
    let outline_width = 0.04;
    let outline_factor = smoothstep(-outline_width - aa, -outline_width, body_sdf);
    let outline_color = vec3<f32>(0.05, 0.05, 0.05);

    // Windshield: front portion of car (u > 0.7)
    let windshield_start = 0.68;
    let windshield_end = 0.82;
    let windshield_side = 0.32;
    let is_windshield = step(windshield_start, uv.x)
                      * (1.0 - step(windshield_end, uv.x))
                      * step(0.5 - windshield_side, uv.y)
                      * (1.0 - step(0.5 + windshield_side, uv.y));
    let windshield_color = vec3<f32>(0.55, 0.7, 0.85);

    // Rear window: back portion
    let rear_start = 0.12;
    let rear_end = 0.25;
    let rear_side = 0.28;
    let is_rear = step(rear_start, uv.x)
                * (1.0 - step(rear_end, uv.x))
                * step(0.5 - rear_side, uv.y)
                * (1.0 - step(0.5 + rear_side, uv.y));

    // Compose final color
    var final_color = color;

    // Apply windshield
    final_color = mix(final_color, windshield_color, is_windshield * 0.85);

    // Apply rear window
    final_color = mix(final_color, windshield_color * 0.8, is_rear * 0.7);

    // Apply outline on top
    final_color = mix(final_color, outline_color, outline_factor);

    return vec4<f32>(final_color, edge_alpha);
}
