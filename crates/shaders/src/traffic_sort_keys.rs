use spirv_std::glam::UVec3;
use spirv_std::spirv;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct PushConstants {
    pub car_count: u32,
    pub _pad0: u32,
    pub _pad1: u32,
    pub _pad2: u32,
}

#[spirv(compute(threads(256)))]
pub fn traffic_sort_keys_main(
    #[spirv(global_invocation_id)] gid: UVec3,
    #[spirv(push_constant)] pc: &PushConstants,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] car_road_id: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] car_s: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] car_lane: &[i32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] road_total_lengths: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] sort_keys: &mut [u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 5)] sort_vals: &mut [u32],
) {
    let idx = gid.x;
    if idx >= pc.car_count {
        sort_keys[idx as usize] = 0xFFFFFFFF;
        sort_vals[idx as usize] = idx;
        return;
    }

    let road_id = car_road_id[idx as usize];
    let s = car_s[idx as usize];
    let lane = car_lane[idx as usize];
    let road_len = road_total_lengths[road_id as usize];

    // Quantize s to 16 bits (0..65535)
    let s_norm = (s / road_len.max(0.001)).clamp(0.0, 1.0);
    let s_quant = (s_norm * 65535.0) as u32;

    // Map lane to unsigned: lane + 128 → 0..255
    let lane_u = (lane + 128).clamp(0, 255) as u32;

    // Build key: road_id[31:24] | lane[23:16] | s[15:0]
    let key = (road_id << 24) | (lane_u << 16) | s_quant;

    sort_keys[idx as usize] = key;
    sort_vals[idx as usize] = idx;
}
