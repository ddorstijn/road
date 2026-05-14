// egui mesh rendering — vertex + fragment shaders.
// Vertex layout matches egui::epaint::Vertex:
//   pos:   float2  (offset 0,  8 bytes)
//   uv:    float2  (offset 8,  8 bytes)
//   color: rgba8   (offset 16, 4 bytes) — auto-unpacked to float4 via R8G8B8A8_UNORM
// Total stride: 20 bytes.

struct VsInput {
    float2 pos   : POSITION;
    float2 uv    : TEXCOORD0;
    float4 color : COLOR0;
};

struct VsOutput {
    float4 position : SV_Position;
    float2 uv       : TEXCOORD0;
    float4 color    : COLOR0;
};

struct PushConstants {
    float2 screen_size;
};

[[vk::push_constant]]
ConstantBuffer<PushConstants> pc;

// ---- Vertex Shader ----

[shader("vertex")]
VsOutput vs_main(VsInput input) {
    VsOutput output;
    output.position = float4(
        2.0 * input.pos.x / pc.screen_size.x - 1.0,
        2.0 * input.pos.y / pc.screen_size.y - 1.0,
        0.0,
        1.0
    );
    output.uv    = input.uv;
    output.color = input.color;
    return output;
}

// ---- Fragment Shader ----

[[vk::binding(0, 0)]]
Texture2D<float4>  egui_texture;

[[vk::binding(1, 0)]]
SamplerState       egui_sampler;

[shader("fragment")]
float4 fs_main(VsOutput input) : SV_Target {
    float4 tex_color = egui_texture.Sample(egui_sampler, input.uv);
    float4 out_color = tex_color * input.color;
    // Premultiply alpha for ONE / ONE_MINUS_SRC_ALPHA blending.
    out_color.rgb *= out_color.a;
    return out_color;
}
