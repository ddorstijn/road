use spirv_std::glam::UVec3;
use spirv_std::spirv;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct PushConstants {
    pub pass_id: u32,
    pub car_count: u32,
    pub num_workgroups: u32,
    pub _pad: u32,
}

const TILE_SIZE: u32 = 256;

#[spirv(compute(threads(256)))]
pub fn traffic_sort_histogram_main(
    #[spirv(local_invocation_id)] lid: UVec3,
    #[spirv(workgroup_id)] wg_id: UVec3,
    #[spirv(push_constant)] pc: &PushConstants,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] keys_in: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] histograms: &mut [u32],
) {
    let local_idx = lid.x;
    let group_id = wg_id.x;
    let global_idx = group_id * TILE_SIZE + local_idx;

    if global_idx < pc.car_count {
        let key = keys_in[global_idx as usize];
        let digit = (key >> (pc.pass_id * 8)) & 0xFF;
        // Note: In the original WGSL this used atomicAdd. In rust-gpu we use
        // spirv_std::arch::atomic_i_add for the same effect.
        let hist_idx = (digit * pc.num_workgroups + group_id) as usize;
        unsafe {
            spirv_std::arch::atomic_i_add::<
                u32,
                { spirv_std::memory::Scope::Device as u32 },
                { spirv_std::memory::Semantics::NONE.bits() },
            >(&mut histograms[hist_idx], 1);
        }
    }
}
