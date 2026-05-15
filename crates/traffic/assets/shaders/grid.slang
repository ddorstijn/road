// Grid background compute shader

struct CameraParams {
    float4x4 inv_vp;
    uint width;
    uint height;
    uint _pad0;
    uint _pad1;
};

[[vk::push_constant]] ConstantBuffer<CameraParams> camera;
[[vk::binding(0, 0), vk::image_format("rgba16f")]] RWTexture2D<float4> output;

float smoothstep_manual(float edge0, float edge1, float x) {
    float t = clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0);
    return t * t * (3.0 - 2.0 * t);
}

float3 mix_vec3(float3 a, float3 b, float t) {
    return a * (1.0 - t) + b * t;
}

[shader("compute")]
[numthreads(16, 16, 1)]
void grid_main(uint3 id : SV_DispatchThreadID) {
    uint2 size = uint2(camera.width, camera.height);
    if (id.x >= size.x || id.y >= size.y)
        return;

    // Pixel to NDC
    float2 uv = float2(
        (float(id.x) + 0.5) / float(size.x) * 2.0 - 1.0,
        (float(id.y) + 0.5) / float(size.y) * 2.0 - 1.0
    );

    // NDC to world via inverse VP
    float4 world4 = mul(camera.inv_vp, float4(uv.x, uv.y, 0.0, 1.0));
    float2 world = world4.xy;

    // Background color
    float3 color = float3(0.08, 0.08, 0.12);

    // Compute world-space size of one pixel.
    float half_view_h = length(float2(camera.inv_vp[1][0], camera.inv_vp[1][1]));
    float pixel_world_size = 2.0 * half_view_h / float(size.y);

    // Adaptive grid: choose spacing so lines are always visible.
    float min_world_gap = pixel_world_size * 8.0;

    // Pick power-of-10 spacings
    float log_gap = log2(min_world_gap) / log2(10.0);
    float minor_exp = ceil(log_gap);
    float minor_spacing = pow(10.0, minor_exp);
    float major_spacing = minor_spacing * 10.0;

    // Minor grid lines
    float minor_x = abs(fmod(fmod(world.x / minor_spacing, 1.0) + 1.0, 1.0) - 0.5) * minor_spacing;
    float minor_y = abs(fmod(fmod(world.y / minor_spacing, 1.0) + 1.0, 1.0) - 0.5) * minor_spacing;
    float minor_dist = min(minor_x, minor_y);
    float minor_line = 1.0 - smoothstep_manual(0.0, pixel_world_size * 1.5, minor_dist);
    color = mix_vec3(color, float3(0.2, 0.2, 0.25), minor_line * 0.5);

    // Major grid lines
    float major_x = abs(fmod(fmod(world.x / major_spacing, 1.0) + 1.0, 1.0) - 0.5) * major_spacing;
    float major_y = abs(fmod(fmod(world.y / major_spacing, 1.0) + 1.0, 1.0) - 0.5) * major_spacing;
    float major_dist = min(major_x, major_y);
    float major_line = 1.0 - smoothstep_manual(0.0, pixel_world_size * 2.0, major_dist);
    color = mix_vec3(color, float3(0.4, 0.4, 0.5), major_line * 0.7);

    // Axis lines
    float axis_thickness = pixel_world_size * 3.0;
    float x_axis = 1.0 - smoothstep_manual(0.0, axis_thickness, abs(world.y));
    float y_axis = 1.0 - smoothstep_manual(0.0, axis_thickness, abs(world.x));
    color = mix_vec3(color, float3(0.8, 0.2, 0.2), x_axis * 0.9);
    color = mix_vec3(color, float3(0.2, 0.8, 0.2), y_axis * 0.9);

    output[id.xy] = float4(color, 1.0);
}
