use gpu_shared::{GpuLane, GpuLaneSection, GpuRoad, GpuTileInstance};
use spirv_std::glam::{Mat4, Vec2, Vec3, Vec4};
use spirv_std::image::Image2d;
use spirv_std::spirv;
use spirv_std::Sampler;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct PushConstants {
    pub view_proj: Mat4,
}

// Marking constants
const MARKING_HALF_WIDTH: f32 = 0.12;
const MARKING_AA: f32 = 0.06;
const DASH_LENGTH: f32 = 3.0;
const SHOULDER_WIDTH: f32 = 0.5;

// Colors
const ASPHALT: Vec3 = Vec3::new(0.20, 0.20, 0.22);
const SHOULDER_COLOR: Vec3 = Vec3::new(0.35, 0.35, 0.33);
const WHITE: Vec3 = Vec3::new(0.95, 0.95, 0.95);
const YELLOW: Vec3 = Vec3::new(1.0, 0.85, 0.0);

#[spirv(vertex)]
pub fn vs_main(
    #[spirv(vertex_index)] vi: u32,
    #[spirv(instance_index)] ii: u32,
    #[spirv(push_constant)] pc: &PushConstants,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 5)] tile_instances: &[GpuTileInstance],
    #[spirv(position)] out_pos: &mut Vec4,
    #[spirv(location = 0)] out_uv: &mut Vec2,
) {
    let tile = tile_instances[ii as usize];
    let tile_world_origin = Vec2::new(tile.tile_world_origin[0], tile.tile_world_origin[1]);
    let tile_world_size = Vec2::new(tile.tile_world_size[0], tile.tile_world_size[1]);
    let atlas_uv_offset = Vec2::new(tile.atlas_uv_offset[0], tile.atlas_uv_offset[1]);
    let atlas_uv_scale = Vec2::new(tile.atlas_uv_scale[0], tile.atlas_uv_scale[1]);

    let (qx, qy) = match vi {
        0 => (0.0f32, 0.0f32),
        1 => (1.0, 0.0),
        2 => (1.0, 1.0),
        3 => (0.0, 0.0),
        4 => (1.0, 1.0),
        _ => (0.0, 1.0),
    };
    let local = Vec2::new(qx, qy);
    let world = tile_world_origin + local * tile_world_size;
    *out_pos = pc.view_proj * Vec4::new(world.x, world.y, 0.0, 1.0);
    *out_uv = atlas_uv_offset + local * atlas_uv_scale;
}

#[spirv(fragment)]
pub fn fs_main(
    #[spirv(location = 0)] uv: Vec2,
    #[spirv(descriptor_set = 0, binding = 0)] sdf_tex: &Image2d,
    #[spirv(descriptor_set = 0, binding = 1)] sdf_sampler: &Sampler,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] roads: &[GpuRoad],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)]
    lane_sections_buf: &[GpuLaneSection],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] lanes_buf: &[GpuLane],
    output: &mut Vec4,
) {
    let sdf_val: Vec4 = sdf_tex.sample(*sdf_sampler, uv);

    let signed_dist = sdf_val.x;
    let s_coord = sdf_val.y;
    let road_id_f = sdf_val.z;
    let s_mod_period = sdf_val.w;

    if road_id_f < 0.0 {
        spirv_std::arch::kill();
    }

    let road_id = road_id_f as u32;
    let road = roads[road_id as usize];

    // Find lane section at this s-coordinate
    let mut section_idx = road.lane_section_offset;
    let mut i = 0u32;
    while i < road.lane_section_count {
        let idx = road.lane_section_offset + i;
        let ls = lane_sections_buf[idx as usize];
        if s_coord >= ls.s_start && s_coord <= ls.s_end {
            section_idx = idx;
            break;
        }
        i += 1;
    }
    let section = lane_sections_buf[section_idx as usize];

    let abs_dist = signed_dist.abs();
    let on_left = signed_dist >= 0.0;

    let (lane_start, lane_end) = if on_left {
        (
            section.lane_offset,
            section.lane_offset + section.left_lane_count,
        )
    } else {
        (
            section.lane_offset + section.left_lane_count,
            section.lane_offset + section.lane_count,
        )
    };

    // Compute total road width on this side
    let mut total_side_width = 0.0f32;
    let mut j = lane_start;
    while j < lane_end {
        total_side_width += lanes_buf[j as usize].width;
        j += 1;
    }

    if abs_dist > total_side_width + SHOULDER_WIDTH {
        spirv_std::arch::kill();
    }

    // Start with base color
    let mut color = ASPHALT;
    let mut alpha = 1.0f32;

    // Shoulder zone
    if abs_dist > total_side_width {
        let shoulder_t = (abs_dist - total_side_width) / SHOULDER_WIDTH;
        color = mix_vec3(ASPHALT, SHOULDER_COLOR, shoulder_t);
        alpha = 1.0 - smoothstep(0.0, 1.0, shoulder_t);
    }

    // Marking pass: iterate lane boundaries on this side
    let mut best_marking_vis = 0.0f32;
    let mut best_marking_color = WHITE;

    let mut accum = 0.0f32;
    j = lane_start;
    while j < lane_end {
        let lane = lanes_buf[j as usize];
        let outer_edge = accum + lane.width;

        if lane.marking_type != 0 {
            let boundary_dist = (abs_dist - outer_edge).abs();
            let mut vis = marking_visibility(boundary_dist);

            if is_dashed(lane.marking_type) {
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
        j += 1;
    }

    // Center line (double solid yellow)
    let center_gap = 0.06f32;
    let d1 = (signed_dist - center_gap).abs();
    let d2 = (signed_dist + center_gap).abs();
    let c1 = marking_visibility(d1);
    let c2 = marking_visibility(d2);
    let center_vis = c1.max(c2);

    if center_vis > best_marking_vis {
        best_marking_vis = center_vis;
        best_marking_color = YELLOW;
    }

    // Composite marking on top of base color
    color = mix_vec3(color, best_marking_color, best_marking_vis);

    *output = Vec4::new(color.x, color.y, color.z, alpha);
}

#[inline]
fn marking_visibility(dist_to_boundary: f32) -> f32 {
    1.0 - smoothstep(
        MARKING_HALF_WIDTH - MARKING_AA,
        MARKING_HALF_WIDTH + MARKING_AA,
        dist_to_boundary,
    )
}

#[inline]
fn marking_color(marking_type: u32) -> Vec3 {
    if marking_type >= 3 {
        YELLOW
    } else {
        WHITE
    }
}

#[inline]
fn is_dashed(marking_type: u32) -> bool {
    marking_type == 2 || marking_type == 4
}

#[inline]
fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

#[inline]
fn mix_vec3(a: Vec3, b: Vec3, t: f32) -> Vec3 {
    a * (1.0 - t) + b * t
}
