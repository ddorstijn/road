use glam::UVec3;
use spirv_std::spirv;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct PushConstants {
    pub num_workgroups: u32,
    pub _pad0: u32,
    pub _pad1: u32,
    pub _pad2: u32,
}

#[spirv(compute(threads(1)))]
pub fn traffic_sort_scan_main(
    #[spirv(global_invocation_id)] _gid: UVec3,
    #[spirv(push_constant)] pc: &PushConstants,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] histograms: &mut [u32],
) {
    let mut global_offset = 0u32;

    let mut digit = 0u32;
    while digit < 256 {
        let base = digit * pc.num_workgroups;
        let mut running_sum = global_offset;

        let mut i = 0u32;
        while i < pc.num_workgroups {
            let val = histograms[(base + i) as usize];
            histograms[(base + i) as usize] = running_sum;
            running_sum += val;
            i += 1;
        }
        global_offset = running_sum;
        digit += 1;
    }
}
