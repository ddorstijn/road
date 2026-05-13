use crate::road_eval::{closest_point_on_segment, world_to_segment_local};
use gpu_shared::{GpuSegment, TileHeader};
use spirv_std::glam::{IVec2, UVec3, Vec2};
use spirv_std::spirv;
use spirv_std::Image;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct SdfPushConstants {
    pub atlas_offset_x: u32,
    pub atlas_offset_y: u32,
    pub tile_world_x: f32,
    pub tile_world_y: f32,
    pub tile_size: f32,
    pub tile_resolution: u32,
    pub tile_index: u32,
    pub road_id_atlas_offset_x: u32,
    pub road_id_atlas_offset_y: u32,
    pub road_id_resolution: u32,
    pub tile_border: u32,
    pub _pad: u32,
}

#[spirv(compute(threads(16, 16)))]
pub fn sdf_generate_main(
    #[spirv(global_invocation_id)] id: UVec3,
    #[spirv(push_constant)] pc: &SdfPushConstants,
    #[spirv(descriptor_set = 0, binding = 0)] sdf_atlas: &Image!(
        2D,
        format = rg16f,
        sampled = false
    ),
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] segments: &[GpuSegment],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] tile_headers: &[TileHeader],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] tile_segment_indices: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] road_indices: &[u32],
    #[spirv(descriptor_set = 0, binding = 5)] road_id_atlas: &Image!(
        2D,
        format = r16f,
        sampled = false
    ),
) {
    // Total texels per tile slot = data + 2 * border
    let dispatch_size = pc.tile_resolution + 2 * pc.tile_border;
    if id.x >= dispatch_size || id.y >= dispatch_size {
        return;
    }

    let texel_size = pc.tile_size / pc.tile_resolution as f32;
    // Offset world position by -border so border texels cover the area
    // just outside the tile, enabling bilinear interpolation across tile edges.
    let world_x = pc.tile_world_x + (id.x as f32 - pc.tile_border as f32 + 0.5) * texel_size;
    let world_y = pc.tile_world_y + (id.y as f32 - pc.tile_border as f32 + 0.5) * texel_size;
    let world_pos = Vec2::new(world_x, world_y);

    let header = tile_headers[pc.tile_index as usize];

    let mut min_dist = 1e10f32;
    let mut best_s = 0.0f32;
    let mut best_road = -1.0f32;
    let mut best_signed_dist = 1e10f32;

    let mut i = 0u32;
    while i < header.count {
        let seg_idx = tile_segment_indices[(header.offset + i) as usize];
        let road_idx = road_indices[(header.offset + i) as usize];
        let seg = segments[seg_idx as usize];

        let origin = Vec2::new(seg.origin[0], seg.origin[1]);
        let local_pt = world_to_segment_local(world_pos, origin, seg.heading);

        let cp = closest_point_on_segment(
            seg.segment_type,
            local_pt,
            seg.length,
            seg.k_start,
            seg.k_end,
        );

        let abs_dist = cp.signed_dist.abs();
        if abs_dist < min_dist {
            min_dist = abs_dist;
            best_signed_dist = cp.signed_dist;
            best_s = seg.s_start + cp.s;
            best_road = road_idx as f32;
        }
        i += 1;
    }

    // Clamp road_id to -1 for texels far from any road to prevent
    // stray fragments at tile boundaries and road endpoints.
    // 20m is wider than any road + shoulder so legitimate road texels are unaffected.
    if min_dist > 20.0 {
        best_road = -1.0;
    }

    // Write SDF atlas (RG16F): signed_dist, s_coord
    let atlas_pos = IVec2::new(
        (pc.atlas_offset_x + id.x) as i32,
        (pc.atlas_offset_y + id.y) as i32,
    );

    unsafe {
        sdf_atlas.write(atlas_pos, Vec2::new(best_signed_dist, best_s));
    }

    // Write road ID atlas (R16_SFLOAT) at lower resolution (every Nth texel)
    // Only interior (non-border) SDF texels contribute to road_id.
    let interior_x = id.x.wrapping_sub(pc.tile_border);
    let interior_y = id.y.wrapping_sub(pc.tile_border);
    if interior_x < pc.tile_resolution && interior_y < pc.tile_resolution {
        let ratio = pc.tile_resolution / pc.road_id_resolution;
        if interior_x % ratio == 0 && interior_y % ratio == 0 {
            let rid_x = interior_x / ratio;
            let rid_y = interior_y / ratio;
            let road_id_pos = IVec2::new(
                (pc.road_id_atlas_offset_x + rid_x) as i32,
                (pc.road_id_atlas_offset_y + rid_y) as i32,
            );
            unsafe {
                road_id_atlas.write(road_id_pos, best_road);
            }
        }
    }
}
