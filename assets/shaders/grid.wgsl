@group(0) @binding(0)
var output: texture_storage_2d<rgba16float, write>;

struct CameraParams {
    inv_vp: mat4x4<f32>,
}

var<immediate> camera: CameraParams;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = textureDimensions(output);
    if id.x >= size.x || id.y >= size.y {
        return;
    }

    // Pixel to NDC
    let uv = vec2<f32>(
        (f32(id.x) + 0.5) / f32(size.x) * 2.0 - 1.0,
        -((f32(id.y) + 0.5) / f32(size.y) * 2.0 - 1.0),
    );

    // NDC to world via inverse VP
    let world4 = camera.inv_vp * vec4<f32>(uv, 0.0, 1.0);
    let world = world4.xy;

    // Background color
    var color = vec3<f32>(0.08, 0.08, 0.12);

    // Grid parameters: choose spacing based on zoom
    // Major grid every 10 units, minor grid every 1 unit
    let minor_spacing = 1.0;
    let major_spacing = 10.0;

    // Compute pixel size in world units (approx) for anti-aliasing
    let pixel_world_size = 2.0 / (f32(size.y) * length(vec2<f32>(camera.inv_vp[1][0], camera.inv_vp[1][1])));

    // Minor grid lines
    let minor_x = abs(fract(world.x / minor_spacing + 0.5) - 0.5) * minor_spacing;
    let minor_y = abs(fract(world.y / minor_spacing + 0.5) - 0.5) * minor_spacing;
    let minor_dist = min(minor_x, minor_y);
    let minor_line = 1.0 - smoothstep(0.0, pixel_world_size * 1.5, minor_dist);
    color = mix(color, vec3<f32>(0.2, 0.2, 0.25), minor_line * 0.5);

    // Major grid lines
    let major_x = abs(fract(world.x / major_spacing + 0.5) - 0.5) * major_spacing;
    let major_y = abs(fract(world.y / major_spacing + 0.5) - 0.5) * major_spacing;
    let major_dist = min(major_x, major_y);
    let major_line = 1.0 - smoothstep(0.0, pixel_world_size * 2.0, major_dist);
    color = mix(color, vec3<f32>(0.4, 0.4, 0.5), major_line * 0.7);

    // Axis lines (X = red, Y = green)
    let axis_x = abs(world.y);
    let axis_y = abs(world.x);
    let axis_thickness = pixel_world_size * 2.0;
    let x_axis = 1.0 - smoothstep(0.0, axis_thickness, axis_x);
    let y_axis = 1.0 - smoothstep(0.0, axis_thickness, axis_y);
    color = mix(color, vec3<f32>(0.8, 0.2, 0.2), x_axis * 0.8);
    color = mix(color, vec3<f32>(0.2, 0.8, 0.2), y_axis * 0.8);

    textureStore(output, vec2<i32>(id.xy), vec4<f32>(color, 1.0));
}
