// Radix sort: per-workgroup histogram pass.
// Each workgroup processes TILE_SIZE elements and counts occurrences of each
// 8-bit digit (256 bins) for the current radix pass.
// Output: histograms[digit * num_workgroups + wg_id] = count
// Uses global atomics to avoid workgroup shared memory (naga ArrayStride issue).

const TILE_SIZE: u32 = 256u;

struct PushConstants {
    pass_id: u32,       // Which 8-bit pass (0-3)
    car_count: u32,     // Total number of elements
    num_workgroups: u32,
    _pad: u32,
}

var<immediate> pc: PushConstants;

@group(0) @binding(0) var<storage, read> keys_in: array<u32>;
@group(0) @binding(1) var<storage, read_write> histograms: array<atomic<u32>>;

@compute @workgroup_size(256)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>,
) {
    let local_idx = lid.x;
    let group_id = wg_id.x;
    let global_idx = group_id * TILE_SIZE + local_idx;

    if global_idx < pc.car_count {
        let key = keys_in[global_idx];
        let digit = (key >> (pc.pass_id * 8u)) & 0xFFu;
        atomicAdd(&histograms[digit * pc.num_workgroups + group_id], 1u);
    }
}
