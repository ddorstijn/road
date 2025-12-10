#version 460

layout(location = 0) in vec2 v_uv;
layout(location = 0) out vec4 f_color;

layout(set = 0, binding = 0) uniform sampler2D sdf_texture;

void main() {
    // 1. Read the pre-calculated road data
    vec4 data = texture(sdf_texture, v_uv);

    float d = data.r; // Distance to center
    float s = data.g; // Distance along road
    float id = data.b; // Road ID

    // 2. Visualization
    vec3 col = vec3(0.1, 0.4, 0.15); // Grass Base

    float roadWidth = 3.5;

    // Check if we are on the road
    if (d < roadWidth + 2.0) {
        // Asphalt
        float roadMask = smoothstep(roadWidth, roadWidth - 0.1, d);
        col = mix(col, vec3(0.2), roadMask);

        // Edge Lines (Solid White)
        float edgeDist = abs(d - roadWidth + 0.2);
        float edgeMask = smoothstep(0.15, 0.05, edgeDist);
        col = mix(col, vec3(0.9), edgeMask * roadMask);

        // Dashed Center Line (Yellow)
        // We use the 's' coordinate to create the pattern
        float centerDist = abs(d);
        float dashPattern = sin(s * 0.5); // Adjust frequency of dashes here
        float centerMask = smoothstep(0.1, 0.05, centerDist);

        if (dashPattern > 0.0) {
            col = mix(col, vec3(1.0, 0.8, 0.0), centerMask * roadMask);
        }
    }

    // Optional: Visual Debug of Road ID
    // if (d < 1.0) col += vec3(id * 0.1, 0.0, 0.0);

    f_color = vec4(col, 1.0);
}
