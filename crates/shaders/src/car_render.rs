use crate::road_eval::{evaluate_road_at_s, compute_lane_offset};
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
    #[spirv(storage_buffer, descriptor_set = 0, binding = 9)] visible_indices: &[u32],
    #[spirv(position)] out_pos: &mut Vec4,
    #[spirv(location = 0)] out_color: &mut Vec3,
    #[spirv(location = 1)] out_uv: &mut Vec2,
    #[spirv(location = 2)] out_speed_ratio: &mut f32,
) {
    // Indirect draw: look up actual car index from visibility buffer
    let car_idx = visible_indices[ii as usize] as usize;

    let road_id = car_road_id[car_idx];
    let s = car_s[car_idx];
    let lane = car_lane[car_idx];
    let speed = car_speed[car_idx];
    let desired = car_desired_speed[car_idx];

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
