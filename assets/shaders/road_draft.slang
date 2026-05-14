// Draft road preview — renders a semi-transparent highlight from the SDF

struct DraftPushConstants {
    float4x4 view_proj;
    float2 world_origin;
    float2 world_size;
    float road_half_width;
    float _pad0;
    float _pad1;
    float _pad2;
};

[[vk::push_constant]] ConstantBuffer<DraftPushConstants> pc;
[[vk::binding(0, 0)]] Texture2D<float4> sdf_tex;
[[vk::binding(1, 0)]] SamplerState sdf_sampler;

struct VsOutput {
    float4 position : SV_Position;
    float2 uv : TEXCOORD0;
};

[shader("vertex")]
VsOutput vs_main(uint vi : SV_VertexID) {
    float qx, qy;
    switch (vi) {
    case 0: qx = 0.0; qy = 0.0; break;
    case 1: qx = 1.0; qy = 0.0; break;
    case 2: qx = 1.0; qy = 1.0; break;
    case 3: qx = 0.0; qy = 0.0; break;
    case 4: qx = 1.0; qy = 1.0; break;
    default: qx = 0.0; qy = 1.0; break;
    }

    float2 local = float2(qx, qy);
    float2 world = pc.world_origin + local * pc.world_size;

    VsOutput output;
    output.position = mul(pc.view_proj, float4(world, 0.0, 1.0));
    output.uv = local;
    return output;
}

static const float3 HIGHLIGHT_COLOR = float3(0.3, 0.6, 1.0);
static const float HIGHLIGHT_ALPHA = 0.45;
static const float EDGE_AA = 0.3;

[shader("fragment")]
float4 fs_main(VsOutput input) : SV_Target {
    float2 sdf_val = sdf_tex.Sample(sdf_sampler, input.uv).xy;
    float signed_dist = sdf_val.x;
    float abs_dist = abs(signed_dist);

    float hw = pc.road_half_width;
    if (abs_dist > hw + EDGE_AA)
        discard;

    // Smooth edge falloff
    float edge = 1.0 - smoothstep(hw - EDGE_AA, hw + EDGE_AA, abs_dist);
    float alpha = HIGHLIGHT_ALPHA * edge;

    return float4(HIGHLIGHT_COLOR, alpha);
}
