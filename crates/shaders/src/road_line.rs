use spirv_std::glam::{Mat4, Vec2, Vec4};
use spirv_std::spirv;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct PushConstants {
    pub view_proj: Mat4,
}

#[spirv(vertex)]
pub fn vs_main(
    #[spirv(push_constant)] pc: &PushConstants,
    position: Vec2,
    color: Vec4,
    #[spirv(position)] out_pos: &mut Vec4,
    #[spirv(location = 0)] out_color: &mut Vec4,
) {
    *out_pos = pc.view_proj * Vec4::new(position.x, position.y, 0.0, 1.0);
    *out_color = color;
}

#[spirv(fragment)]
pub fn fs_main(#[spirv(location = 0)] color: Vec4, output: &mut Vec4) {
    *output = Vec4::new(color.x, color.y, color.z, color.w);
}
