// Heightmap height resolution 1 = 1m, 10 = 1dm, 100 = 1cm
const HEIGHT_SCALE: f32 = 1000.0; 
// Heightmap width resolution 1 = 1m, 10 = 1dm, 100 = 1cm
const WIDTH_SCALE: f32 = 2.0; 

@group(0) @binding(0) var<uniform> u_view_proj : mat4x4<f32>;

@group(1) @binding(0) var u_heightmap           : texture_2d<u32>;
@group(1) @binding(1) var u_sampler             : sampler;

struct Instance {
    transform: mat4x4<f32>,
}
@group(2) @binding(0)
var<storage, read> instances: array<Instance>;

struct VSOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0)       world_pos: vec3<f32>,
};

@vertex
fn vs_main(
    @location(0) a_pos: vec2<f32>,
    @builtin(instance_index) instance_idx: u32
) -> VSOut {
    var out: VSOut;

    let instance_data = instances[instance_idx];
    // Reconstruct 4×4 model matrix
    let model = instance_data.transform;

    // 1) Lift 2D (X,Y,0,1)
    let local = vec4<f32>(a_pos.x, a_pos.y, 0.0, 1.0);

    // 2) Transform to world XY
    let world_xy4 = model * local;
    let world_xy = world_xy4.xy;

    // 3) Sample heightmap at integer XY
    let uv = world_xy / WIDTH_SCALE + vec2<f32>(textureDimensions(u_heightmap)) * 0.5;
    let height = f32(textureLoad(u_heightmap, vec2<i32>(uv), 0).r) / HEIGHT_SCALE;
    // 4) Assemble Z-up world position
    let world_pos3 = vec3<f32>(world_xy, height);

    // 5) Final clip‐space
    out.clip_pos = u_view_proj * vec4<f32>(world_pos3, 1.0);
    out.world_pos = world_pos3;

    return out;
}

@fragment
fn fs_main(in: VSOut) -> @location(0) vec4<f32> {
    // Simple height‐based shading
    let shade = vec3<f32>(in.world_pos.z / HEIGHT_SCALE);
    return vec4<f32>(shade, 1.0);
}