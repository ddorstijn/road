// Radix sort: scatter pass.
// Each thread determines its output position from the prefix-summed histogram
// and its local rank (computed by counting preceding elements with same digit).
// Avoids workgroup shared memory (naga ArrayStride issue).

const TILE_SIZE: u32 = 256u;

struct PushConstants {
    pass_id: u32,
    car_count: u32,
    num_workgroups: u32,
    _pad: u32,
}

var<immediate> pc: PushConstants;

@group(0) @binding(0) var<storage, read> keys_in: array<u32>;
@group(0) @binding(1) var<storage, read> vals_in: array<u32>;
@group(0) @binding(2) var<storage, read_write> keys_out: array<u32>;
@group(0) @binding(3) var<storage, read_write> vals_out: array<u32>;
@group(0) @binding(4) var<storage, read> histograms: array<u32>;

@compute @workgroup_size(256)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>,
) {
    let local_idx = lid.x;
    let group_id = wg_id.x;
    let global_idx = group_id * TILE_SIZE + local_idx;

    if global_idx >= pc.car_count {
        return;
    }

    let key = keys_in[global_idx];
    let val = vals_in[global_idx];
    let digit = (key >> (pc.pass_id * 8u)) & 0xFFu;

    // Compute local rank: count how many preceding elements in this tile
    // have the same digit (for stable sort)
    var local_rank = 0u;
    let tile_start = group_id * TILE_SIZE;
    for (var i = 0u; i < local_idx; i++) {
        let other_idx = tile_start + i;
        if other_idx < pc.car_count {
            let other_key = keys_in[other_idx];
            let other_digit = (other_key >> (pc.pass_id * 8u)) & 0xFFu;
            if other_digit == digit {
                local_rank += 1u;
            }
        }
    }

    // Output position = global prefix sum offset + local rank
    let global_offset = histograms[digit * pc.num_workgroups + group_id];
    let dst = global_offset + local_rank;

    keys_out[dst] = key;
    vals_out[dst] = val;
}
