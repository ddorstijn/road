use crate::road_eval::{eval_segment, segment_local_to_world, PoseResult};
use gpu_shared::{GpuLane, GpuLaneSection, GpuRoad, GpuSegment};
use spirv_std::arch::Derivative;
use spirv_std::glam::{Mat4, Vec2, Vec3, Vec4};
use spirv_std::num_traits::Float;
use spirv_std::spirv;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct PushConstants {
    pub view_proj: Mat4,
    pub car_count: u32,
    pub _pad0: u32,
    pub _pad1: u32,
    pub _pad2: u32,
}

const CAR_LENGTH: f32 = 4.5;
const CAR_WIDTH: f32 = 2.0;

fn evaluate_road_at_s(
    road_id: u32,
    s_global: f32,
    roads: &[GpuRoad],
    segments: &[GpuSegment],
) -> PoseResult {
    let road = roads[road_id as usize];
    let mut seg_idx = road.segment_offset;
    let mut local_s = s_global;

    let mut i = 0u32;
    while i < road.segment_count {
        let seg = segments[(road.segment_offset + i) as usize];
        if s_global < seg.s_start + seg.length || i == road.segment_count - 1 {
            seg_idx = road.segment_offset + i;
            local_s = (s_global - seg.s_start).clamp(0.0, seg.length);
            break;
        }
        i += 1;
    }

    let seg = segments[seg_idx as usize];
    let pose = eval_segment(
        seg.segment_type,
        local_s,
        seg.k_start,
        seg.k_end,
        seg.length,
    );

    let origin = Vec2::new(seg.origin[0], seg.origin[1]);
    PoseResult {
        position: segment_local_to_world(pose.position, origin, seg.heading),
        heading: seg.heading + pose.heading,
    }
}

fn compute_lane_offset(
    road_id: u32,
    lane_idx: i32,
    s_global: f32,
    roads: &[GpuRoad],
    lane_sections: &[GpuLaneSection],
    lanes: &[GpuLane],
) -> f32 {
    let road = roads[road_id as usize];

    let mut section_idx = road.lane_section_offset;
    let mut i = 0u32;
    while i < road.lane_section_count {
        let ls = lane_sections[(road.lane_section_offset + i) as usize];
        if s_global >= ls.s_start && s_global < ls.s_end {
            section_idx = road.lane_section_offset + i;
            break;
        }
        section_idx = road.lane_section_offset + i;
        i += 1;
    }

    let section = lane_sections[section_idx as usize];
    let left_count = section.left_lane_count as i32;
    let mut offset = 0.0f32;

    if lane_idx >= 0 {
        let right_start = left_count as u32;
        let mut j = 0u32;
        while j < lane_idx as u32 {
            let l = lanes[(section.lane_offset + right_start + j) as usize];
            offset += l.width;
            j += 1;
        }
        let current = lanes[(section.lane_offset + right_start + lane_idx as u32) as usize];
        offset += current.width * 0.5;
        offset = -offset;
    } else {
        let left_idx = (-lane_idx - 1) as u32;
        let mut j = 0u32;
        while j < left_idx {
            let l = lanes[(section.lane_offset + j) as usize];
            offset += l.width;
            j += 1;
        }
        let current = lanes[(section.lane_offset + left_idx) as usize];
        offset += current.width * 0.5;
    }

    offset
}

#[spirv(vertex)]
pub fn vs_main(
    #[spirv(vertex_index)] vi: u32,
    #[spirv(instance_index)] ii: u32,
    #[spirv(push_constant)] pc: &PushConstants,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] car_road_id: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] car_s: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] car_lane: &[i32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] car_speed: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] car_desired_speed: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 5)] segments: &[GpuSegment],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 6)] roads: &[GpuRoad],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 7)] lane_sections: &[GpuLaneSection],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 8)] lanes_buf: &[GpuLane],
    #[spirv(position)] out_pos: &mut Vec4,
    #[spirv(location = 0)] out_color: &mut Vec3,
    #[spirv(location = 1)] out_uv: &mut Vec2,
    #[spirv(location = 2)] out_speed_ratio: &mut f32,
) {
    let road_id = car_road_id[ii as usize];
    let s = car_s[ii as usize];
    let lane = car_lane[ii as usize];
    let speed = car_speed[ii as usize];
    let desired = car_desired_speed[ii as usize];

    let pose = evaluate_road_at_s(road_id, s, roads, segments);
    let lat_offset = compute_lane_offset(road_id, lane, s, roads, lane_sections, lanes_buf);

    let normal = Vec2::new(-pose.heading.sin(), pose.heading.cos());
    let center = pose.position + normal * lat_offset;

    // Unit quad vertex positions (6 verts = 2 triangles)
    let (qx, qy) = match vi {
        0 => (-0.5f32, -0.5f32),
        1 => (0.5, -0.5),
        2 => (0.5, 0.5),
        3 => (-0.5, -0.5),
        4 => (0.5, 0.5),
        5 => (-0.5, 0.5),
        _ => (0.0, 0.0),
    };

    *out_uv = Vec2::new(qx + 0.5, qy + 0.5);

    // Scale by car dimensions and rotate by heading
    let local = Vec2::new(qx * CAR_LENGTH, qy * CAR_WIDTH);
    let c = pose.heading.cos();
    let sn = pose.heading.sin();
    let rotated = Vec2::new(c * local.x - sn * local.y, sn * local.x + c * local.y);
    let world_pos = center + rotated;

    *out_pos = pc.view_proj * Vec4::new(world_pos.x, world_pos.y, 0.0, 1.0);

    // Speed ratio for coloring
    let ratio = (speed / desired.max(0.1)).clamp(0.0, 1.0);
    *out_speed_ratio = ratio;

    // Base body color: red (stopped) → yellow (slow) → green (free flow)
    let r = (1.0 - ratio * 1.5).clamp(0.0, 1.0);
    let g = (ratio * 1.5).clamp(0.0, 1.0);
    *out_color = Vec3::new(r, g, 0.1);
}

fn sdf_rounded_rect(p: Vec2, half_size: Vec2, radius: f32) -> f32 {
    let q = p.abs() - half_size + Vec2::splat(radius);
    q.max(Vec2::ZERO).length() + q.x.max(q.y).min(0.0) - radius
}

#[spirv(fragment)]
pub fn fs_main(
    #[spirv(location = 0)] color: Vec3,
    #[spirv(location = 1)] uv: Vec2,
    #[spirv(location = 2)] _speed_ratio: f32,
    output: &mut Vec4,
) {
    let centered = uv - Vec2::splat(0.5);

    let body_sdf = sdf_rounded_rect(centered, Vec2::new(0.45, 0.42), 0.08);

    if body_sdf > 0.0 {
        spirv_std::arch::kill();
    }

    // Anti-aliased edge
    let aa = body_sdf.fwidth();
    let edge_alpha = 1.0 - smoothstep(-aa * 1.5, 0.0, body_sdf);

    // Outline: darken near edge
    let outline_width = 0.04;
    let outline_factor = smoothstep(-outline_width - aa, -outline_width, body_sdf);
    let outline_color = Vec3::new(0.05, 0.05, 0.05);

    // Windshield
    let windshield_start = 0.68;
    let windshield_end = 0.82;
    let windshield_side = 0.32;
    let is_windshield = step(windshield_start, uv.x)
        * (1.0 - step(windshield_end, uv.x))
        * step(0.5 - windshield_side, uv.y)
        * (1.0 - step(0.5 + windshield_side, uv.y));
    let windshield_color = Vec3::new(0.55, 0.7, 0.85);

    // Composite: body → outline → windshield
    let mut final_color = mix_vec3(color, outline_color, outline_factor);
    final_color = mix_vec3(final_color, windshield_color, is_windshield * 0.7);

    *output = Vec4::new(final_color.x, final_color.y, final_color.z, edge_alpha);
}

#[inline]
fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

#[inline]
fn step(edge: f32, x: f32) -> f32 {
    if x < edge {
        0.0
    } else {
        1.0
    }
}

#[inline]
fn mix_vec3(a: Vec3, b: Vec3, t: f32) -> Vec3 {
    a * (1.0 - t) + b * t
}
