//! GPU frustum culling compute shaders for cars and tiles.
//!
//! For the 2D orthographic camera, the frustum is a world-space axis-aligned
//! rectangle. Each invocation tests one car (or tile) and atomically appends
//! visible items to a compacted output buffer + indirect draw command.

use crate::road_eval::{compute_lane_offset, evaluate_road_at_s};
use gpu_shared::{GpuLane, GpuLaneSection, GpuRoad, GpuSegment, GpuTileInstance};
use spirv_std::glam::Vec2;
use spirv_std::num_traits::Float;
use spirv_std::spirv;

// ---------------------------------------------------------------------------
// Car frustum culling
// ---------------------------------------------------------------------------

#[repr(C)]
#[derive(Clone, Copy)]
pub struct CarCullPushConstants {
    /// World-space view bounds: (min_x, min_y, max_x, max_y)
    pub view_min_x: f32,
    pub view_min_y: f32,
    pub view_max_x: f32,
    pub view_max_y: f32,
    pub car_count: u32,
    pub _pad0: u32,
    pub _pad1: u32,
    pub _pad2: u32,
}

/// Margin around each car for conservative culling (half-diagonal of car bounding box).
const CAR_MARGIN: f32 = 3.5; // sqrt(4.5^2 + 2.0^2) ≈ 4.9, but 3.5 is enough for axis-aligned test

/// Compute shader: one invocation per car.
/// Tests if the car's world position is inside the view frustum (with margin).
/// If visible, atomically appends the car index to `visible_indices` and increments
/// the indirect draw command's instance_count.
#[spirv(compute(threads(256)))]
pub fn car_cull_main(
    #[spirv(global_invocation_id)] gid: spirv_std::glam::UVec3,
    #[spirv(push_constant)] pc: &CarCullPushConstants,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] car_road_id: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] car_s: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] car_lane: &[i32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] segments: &[GpuSegment],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] roads: &[GpuRoad],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 5)] lane_sections: &[GpuLaneSection],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 6)] lanes_buf: &[GpuLane],
    // Output: draw_indirect[0..3] = {vertex_count, instance_count, first_vertex, first_instance}
    #[spirv(storage_buffer, descriptor_set = 0, binding = 7)] draw_indirect: &mut [u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 8)] visible_indices: &mut [u32],
) {
    let idx = gid.x;
    if idx >= pc.car_count {
        return;
    }

    let road_id = car_road_id[idx as usize];
    let s = car_s[idx as usize];
    let lane = car_lane[idx as usize];

    // Evaluate world position
    let pose = evaluate_road_at_s(road_id, s, roads, segments);
    let lat_offset = compute_lane_offset(road_id, lane, s, roads, lane_sections, lanes_buf);
    let normal = Vec2::new(-pose.heading.sin(), pose.heading.cos());
    let center = pose.position + normal * lat_offset;

    // AABB test with margin
    if center.x + CAR_MARGIN < pc.view_min_x
        || center.x - CAR_MARGIN > pc.view_max_x
        || center.y + CAR_MARGIN < pc.view_min_y
        || center.y - CAR_MARGIN > pc.view_max_y
    {
        return;
    }

    // Visible: atomically append
    let slot = unsafe {
        spirv_std::arch::atomic_i_add::<
            u32,
            { spirv_std::memory::Scope::Device as u32 },
            { spirv_std::memory::Semantics::NONE.bits() },
        >(&mut draw_indirect[1], 1) // instance_count at offset 1
    };

    visible_indices[slot as usize] = idx;
}

// ---------------------------------------------------------------------------
// Tile frustum culling
// ---------------------------------------------------------------------------

#[repr(C)]
#[derive(Clone, Copy)]
pub struct TileCullPushConstants {
    /// World-space view bounds: (min_x, min_y, max_x, max_y)
    pub view_min_x: f32,
    pub view_min_y: f32,
    pub view_max_x: f32,
    pub view_max_y: f32,
    pub tile_count: u32,
    pub _pad0: u32,
    pub _pad1: u32,
    pub _pad2: u32,
}

/// Compute shader: one invocation per tile.
/// Tests if the tile's world-space quad overlaps the view frustum.
/// If visible, atomically appends the tile instance data to `visible_tiles`
/// and increments the indirect draw command's instance_count.
#[spirv(compute(threads(64)))]
pub fn tile_cull_main(
    #[spirv(global_invocation_id)] gid: spirv_std::glam::UVec3,
    #[spirv(push_constant)] pc: &TileCullPushConstants,
    // Input: all tile instances (pre-built on CPU)
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] all_tiles: &[GpuTileInstance],
    // Output: indirect draw command
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] draw_indirect: &mut [u32],
    // Output: compacted visible tile instances
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)]
    visible_tiles: &mut [GpuTileInstance],
) {
    let idx = gid.x;
    if idx >= pc.tile_count {
        return;
    }

    let tile = all_tiles[idx as usize];
    let tile_min_x = tile.tile_world_origin[0];
    let tile_min_y = tile.tile_world_origin[1];
    let tile_max_x = tile_min_x + tile.tile_world_size[0];
    let tile_max_y = tile_min_y + tile.tile_world_size[1];

    // AABB overlap test
    if tile_max_x < pc.view_min_x
        || tile_min_x > pc.view_max_x
        || tile_max_y < pc.view_min_y
        || tile_min_y > pc.view_max_y
    {
        return;
    }

    // Visible: atomically append
    let slot = unsafe {
        spirv_std::arch::atomic_i_add::<
            u32,
            { spirv_std::memory::Scope::Device as u32 },
            { spirv_std::memory::Semantics::NONE.bits() },
        >(&mut draw_indirect[1], 1) // instance_count at offset 1
    };

    visible_tiles[slot as usize] = tile;
}
