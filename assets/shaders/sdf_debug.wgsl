// Debug visualization: render SDF tile quads on screen.
// Each tile is drawn as a world-space quad textured from the SDF atlas.
// Color encodes signed distance: gray bands show distance, green = on road.

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

@group(0) @binding(0) var sdf_tex: texture_2d<f32>;
@group(0) @binding(1) var sdf_sampler: sampler;

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let sdf_val = textureSample(sdf_tex, sdf_sampler, in.uv);
    let signed_dist = sdf_val.x;
    let road_idx = sdf_val.z;

    // No road nearby — transparent
    if road_idx < 0.0 {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }

    // Visualize: road surface = dark gray, distance = grayscale bands
    let abs_dist = abs(signed_dist);

    // Road surface (within ~14m = 4 lanes of 3.5m)
    let max_road_width = 14.0;
    if abs_dist < max_road_width {
        // Dark asphalt for road surface, lighter at edges
        let t = abs_dist / max_road_width;
        let gray = mix(0.15, 0.3, t);
        return vec4<f32>(gray, gray, gray, 0.9);
    }

    // Beyond road — fade out
    let fade = 1.0 - smoothstep(max_road_width, max_road_width + 10.0, abs_dist);
    if fade < 0.01 {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }

    return vec4<f32>(0.1, 0.1, 0.12, fade * 0.3);
}
