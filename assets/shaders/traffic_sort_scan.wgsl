// Radix sort: prefix sum (exclusive scan) over histogram rows.
// Dispatched as workgroup_size(1), 1 workgroup total.
// This single thread processes all 256 digit rows sequentially,
// computing a GLOBAL exclusive prefix sum so that each digit's scatter
// positions start after the previous digit's total count.

struct PushConstants {
    num_workgroups: u32,  // actual number of sort workgroups (row length)
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

var<immediate> pc: PushConstants;

@group(0) @binding(0) var<storage, read_write> histograms: array<u32>;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    // Phase 1: compute the total count for each digit row
    //          and store row totals in a running offset
    // We process all 256 rows sequentially, accumulating a global offset.
    // For each row, we replace the counts with exclusive-prefix-sum values
    // shifted by the global offset of that digit.

    var global_offset = 0u;

    for (var digit = 0u; digit < 256u; digit++) {
        let base = digit * pc.num_workgroups;

        // Sequential exclusive prefix sum over the row, offset by global_offset
        var running_sum = global_offset;
        for (var i = 0u; i < pc.num_workgroups; i++) {
            let val = histograms[base + i];
            histograms[base + i] = running_sum;
            running_sum += val;
        }
        // running_sum is now global_offset + total_count_for_this_digit
        global_offset = running_sum;
    }
}
