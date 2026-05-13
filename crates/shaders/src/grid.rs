use spirv_std::glam::{Mat4, UVec2, UVec3, Vec2, Vec3, Vec4};
use spirv_std::num_traits::Float;
use spirv_std::spirv;
use spirv_std::Image;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct CameraParams {
    pub inv_vp: Mat4,
    pub width: u32,
    pub height: u32,
    pub _pad0: u32,
    pub _pad1: u32,
}

#[spirv(compute(threads(16, 16)))]
pub fn grid_main(
    #[spirv(global_invocation_id)] id: UVec3,
    #[spirv(push_constant)] camera: &CameraParams,
    #[spirv(descriptor_set = 0, binding = 0)] output: &Image!(
        2D,
        format = rgba16f,
        sampled = false
    ),
) {
    let size = UVec2::new(camera.width, camera.height);
    if id.x >= size.x || id.y >= size.y {
        return;
    }

    // Pixel to NDC (Vulkan convention: Y increases downward in screen space).
    let uv = Vec2::new(
        (id.x as f32 + 0.5) / size.x as f32 * 2.0 - 1.0,
        (id.y as f32 + 0.5) / size.y as f32 * 2.0 - 1.0,
    );

    // NDC to world via inverse VP
    let world4 = camera.inv_vp * Vec4::new(uv.x, uv.y, 0.0, 1.0);
    let world = world4.truncate().truncate();

    // Background color
    let mut color = Vec3::new(0.08, 0.08, 0.12);

    // Compute world-space size of one pixel.
    let half_view_h = Vec2::new(camera.inv_vp.y_axis.x, camera.inv_vp.y_axis.y).length();
    let pixel_world_size = 2.0 * half_view_h / size.y as f32;

    // Adaptive grid: choose spacing so lines are always visible.
    let min_world_gap = pixel_world_size * 8.0;

    // Pick power-of-10 spacings
    let log_gap = min_world_gap.log2() / 10.0f32.log2();
    let minor_exp = log_gap.ceil();
    let minor_spacing = 10.0f32.powf(minor_exp);
    let major_spacing = minor_spacing * 10.0;

    // Minor grid lines
    let minor_x = (((world.x / minor_spacing) % 1.0 + 1.0) % 1.0 - 0.5).abs() * minor_spacing;
    let minor_y = (((world.y / minor_spacing) % 1.0 + 1.0) % 1.0 - 0.5).abs() * minor_spacing;
    let minor_dist = minor_x.min(minor_y);
    let minor_line = 1.0 - smoothstep(0.0, pixel_world_size * 1.5, minor_dist);
    color = mix_vec3(color, Vec3::new(0.2, 0.2, 0.25), minor_line * 0.5);

    // Major grid lines
    let major_x = (((world.x / major_spacing) % 1.0 + 1.0) % 1.0 - 0.5).abs() * major_spacing;
    let major_y = (((world.y / major_spacing) % 1.0 + 1.0) % 1.0 - 0.5).abs() * major_spacing;
    let major_dist = major_x.min(major_y);
    let major_line = 1.0 - smoothstep(0.0, pixel_world_size * 2.0, major_dist);
    color = mix_vec3(color, Vec3::new(0.4, 0.4, 0.5), major_line * 0.7);

    // Axis lines
    let axis_thickness = pixel_world_size * 3.0;
    let x_axis = 1.0 - smoothstep(0.0, axis_thickness, world.y.abs());
    let y_axis = 1.0 - smoothstep(0.0, axis_thickness, world.x.abs());
    color = mix_vec3(color, Vec3::new(0.8, 0.2, 0.2), x_axis * 0.9);
    color = mix_vec3(color, Vec3::new(0.2, 0.8, 0.2), y_axis * 0.9);

    unsafe {
        output.write(id.truncate(), Vec4::new(color.x, color.y, color.z, 1.0));
    }
}

#[inline]
fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

#[inline]
fn mix_vec3(a: Vec3, b: Vec3, t: f32) -> Vec3 {
    a * (1.0 - t) + b * t
}
