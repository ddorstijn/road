#import road::types::{GpuSegment, TileHeader}
#import road::eval::{world_to_segment_local, closest_point_on_segment}

// SDF tile generation compute shader.
//
// For each texel in the SDF atlas, computes the signed distance to the
// nearest road segment, along with the s-coordinate and road index.
//
// The shared types and road_eval modules are composed via naga_oil imports.
// They provide: GpuSegment, TileHeader, closest_point_on_segment,
//               world_to_segment_local, etc.

// --- Bindings ---

@group(0) @binding(0) var sdf_atlas: texture_storage_2d<rgba16float, write>;
@group(0) @binding(1) var<storage, read> segments: array<GpuSegment>;
@group(0) @binding(2) var<storage, read> tile_headers: array<TileHeader>;
@group(0) @binding(3) var<storage, read> tile_segment_indices: array<u32>;
@group(0) @binding(4) var<storage, read> road_indices: array<u32>;

// Push constants
struct SdfPushConstants {
    // Atlas offset for this tile (in texels)
    atlas_offset_x: u32,
    atlas_offset_y: u32,
    // Tile world-space origin (bottom-left corner)
    tile_world_x: f32,
    tile_world_y: f32,
    // Tile size in world units
    tile_size: f32,
    // Tile resolution in texels (e.g. 256)
    tile_resolution: u32,
    // Tile index into tile_headers
    tile_index: u32,
    _pad: u32,
}

var<immediate> pc: SdfPushConstants;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    // Check bounds (tile_resolution x tile_resolution)
    if id.x >= pc.tile_resolution || id.y >= pc.tile_resolution {
        return;
    }

    // Convert texel to world position (center of texel)
    let texel_size = pc.tile_size / f32(pc.tile_resolution);
    let world_x = pc.tile_world_x + (f32(id.x) + 0.5) * texel_size;
    let world_y = pc.tile_world_y + (f32(id.y) + 0.5) * texel_size;
    let world_pos = vec2<f32>(world_x, world_y);

    // Read tile header to find which segments overlap this tile
    let header = tile_headers[pc.tile_index];

    var min_dist = 1e10;
    var best_s = 0.0;
    var best_road: f32 = -1.0;
    var best_signed_dist = 1e10;

    // Iterate over segments that overlap this tile
    for (var i = 0u; i < header.count; i++) {
        let seg_idx = tile_segment_indices[header.offset + i];
        let road_idx = road_indices[header.offset + i];
        let seg = segments[seg_idx];

        // Transform world point into segment-local coordinates
        let local_pt = world_to_segment_local(world_pos, seg.origin, seg.heading);

        // Find closest point on this segment
        let cp = closest_point_on_segment(
            seg.segment_type,
            local_pt,
            seg.length,
            seg.k_start,
            seg.k_end,
        );

        let abs_dist = abs(cp.signed_dist);
        if abs_dist < min_dist {
            min_dist = abs_dist;
            best_signed_dist = cp.signed_dist;
            best_s = seg.s_start + cp.s;
            best_road = f32(road_idx);
        }
    }

    // Write to atlas: (signed_distance, s_coordinate, road_index, s_mod_period)
    // Alpha stores s % 6.0 (dash period) to preserve precision in fp16 atlases
    let s_mod = best_s % 6.0;
    let atlas_pos = vec2<i32>(
        i32(pc.atlas_offset_x + id.x),
        i32(pc.atlas_offset_y + id.y),
    );
    textureStore(sdf_atlas, atlas_pos, vec4<f32>(best_signed_dist, best_s, best_road, s_mod));
}
