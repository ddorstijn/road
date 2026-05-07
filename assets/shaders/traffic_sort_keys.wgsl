// Build sort keys from car SoA data.
// Key encoding (32-bit): road_id[31:24] | (lane+128)[23:16] | s_quantized[15:0]
// After sorting by this key, cars are ordered by (road_id, lane, s).

struct PushConstants {
    car_count: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

var<immediate> pc: PushConstants;

@group(0) @binding(0) var<storage, read> car_road_id: array<u32>;
@group(0) @binding(1) var<storage, read> car_s: array<f32>;
@group(0) @binding(2) var<storage, read> car_lane: array<i32>;
@group(0) @binding(3) var<storage, read> road_total_lengths: array<f32>;
@group(0) @binding(4) var<storage, read_write> sort_keys: array<u32>;
@group(0) @binding(5) var<storage, read_write> sort_vals: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= pc.car_count {
        // Fill unused slots with max key so they sort to the end
        sort_keys[idx] = 0xFFFFFFFFu;
        sort_vals[idx] = idx;
        return;
    }

    let road_id = car_road_id[idx];
    let s = car_s[idx];
    let lane = car_lane[idx];
    let road_len = road_total_lengths[road_id];

    // Quantize s to 16 bits (0..65535)
    let s_norm = clamp(s / max(road_len, 0.001), 0.0, 1.0);
    let s_quant = u32(s_norm * 65535.0);

    // Map lane to unsigned: lane + 128 → 0..255
    let lane_u = u32(clamp(lane + 128, 0, 255));

    // Build key: road_id[31:24] | lane[23:16] | s[15:0]
    let key = (road_id << 24u) | (lane_u << 16u) | s_quant;

    sort_keys[idx] = key;
    sort_vals[idx] = idx;
}
