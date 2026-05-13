// Road line rendering — simple vertex + fragment shader for line geometry.

struct PushConstants {
    float4x4 view_proj;
};

[[vk::push_constant]] ConstantBuffer<PushConstants> pc;

struct VsInput {
    float2 position : POSITION0;
    float4 color : COLOR0;
};

struct VsOutput {
    float4 position : SV_Position;
    float4 color : COLOR0;
};

[shader("vertex")]
VsOutput vs_main(VsInput input) {
    VsOutput output;
    output.position = mul(pc.view_proj, float4(input.position, 0.0, 1.0));
    output.color = input.color;
    return output;
}

[shader("fragment")]
float4 fs_main(VsOutput input) : SV_Target {
    return input.color;
}
