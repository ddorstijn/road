use glam::UVec3;
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
pub fn traffic_sort_scatter_main(
    #[spirv(local_invocation_id)] lid: UVec3,
    #[spirv(workgroup_id)] wg_id: UVec3,
    #[spirv(push_constant)] pc: &PushConstants,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] keys_in: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] vals_in: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] keys_out: &mut [u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] vals_out: &mut [u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] histograms: &[u32],
) {
    let local_idx = lid.x;
    let group_id = wg_id.x;
    let global_idx = group_id * TILE_SIZE + local_idx;

    // Use .min() to guard against out-of-bounds (also prevents spirv-opt from
    // incorrectly stripping this entry point — rust-gpu optimizer bug workaround)
    let car_count = pc.car_count.min(1 << 24);
    if global_idx >= car_count {
        return;
    }

    let key = keys_in[global_idx as usize];
    let val = vals_in[global_idx as usize];
    let digit = (key >> (pc.pass_id * 8)) & 0xFF;

    // Compute local rank: count how many preceding elements in this tile
    // have the same digit (for stable sort)
    let mut local_rank = 0u32;
    let tile_start = group_id * TILE_SIZE;
    let mut i = 0u32;
    while i < local_idx {
        let other_idx = tile_start + i;
        if other_idx < car_count {
            let other_key = keys_in[other_idx as usize];
            let other_digit = (other_key >> (pc.pass_id * 8)) & 0xFF;
            if other_digit == digit {
                local_rank += 1;
            }
        }
        i += 1;
    }

    // Output position = global prefix sum offset + local rank
    let global_offset = histograms[(digit * pc.num_workgroups + group_id) as usize];
    let dst = global_offset + local_rank;

    keys_out[dst as usize] = key;
    vals_out[dst as usize] = val;
}
