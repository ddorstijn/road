@group(0) @binding(0)
var output: texture_storage_2d<rgba16float, write>;

struct ClearParams {
    color: vec4<f32>,
}

var<immediate> params: ClearParams;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = textureDimensions(output);
    if id.x >= size.x || id.y >= size.y {
        return;
    }

    textureStore(output, vec2<i32>(id.xy), params.color);
}
