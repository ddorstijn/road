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

/// Exclusive prefix sum over the histogram table (256 digits * num_workgroups entries).
/// Single-threaded because the table is small relative to GPU overhead of synchronization.
#[spirv(compute(threads(1)))]
pub fn traffic_sort_scan_main(
    #[spirv(global_invocation_id)] _gid: UVec3,
    #[spirv(push_constant)] pc: &PushConstants,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] histograms: &mut [u32],
) {
    // Clamp num_workgroups to a safe maximum to avoid out-of-bounds
    let num_workgroups = pc.num_workgroups.min(1024);
    let total = 256u32 * num_workgroups;

    let mut sum = 0u32;
    let mut i = 0u32;
    while i < total {
        let val = histograms[i as usize];
        histograms[i as usize] = sum;
        sum += val;
        i += 1;
    }
}
