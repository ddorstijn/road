use bytemuck::{Pod, Zeroable};

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct GpuSegment {
    pub segment_type: u32,
    pub s_start: f32,
    pub origin: [f32; 2],
    pub heading: f32,
    pub length: f32,
    pub k_start: f32,
    pub k_end: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct GpuRoad {
    pub segment_offset: u32,
    pub segment_count: u32,
    pub lane_section_offset: u32,
    pub lane_section_count: u32,
    pub total_length: f32,
    pub _pad: [u32; 3],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct GpuLaneSection {
    pub s_start: f32,
    pub s_end: f32,
    pub lane_offset: u32,
    pub lane_count: u32,
    pub left_lane_count: u32,
    pub _pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct GpuLane {
    pub width: f32,
    pub lane_type: u32,
    pub marking_type: u32,
    pub _pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct TileHeader {
    pub offset: u32,
    pub count: u32,
}
