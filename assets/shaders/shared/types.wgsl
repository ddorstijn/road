// Shared GPU types — must match Rust-side GpuSegment, GpuRoad, etc.

struct GpuSegment {
    segment_type: u32,  // 0 = Line, 1 = Arc, 2 = Spiral
    s_start: f32,
    origin: vec2<f32>,
    heading: f32,
    length: f32,
    k_start: f32,
    k_end: f32,
}

struct GpuRoad {
    segment_offset: u32,
    segment_count: u32,
    lane_section_offset: u32,
    lane_section_count: u32,
    total_length: f32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

struct GpuLaneSection {
    s_start: f32,
    s_end: f32,
    lane_offset: u32,
    lane_count: u32,
    left_lane_count: u32,
    _pad: u32,
}

struct GpuLane {
    width: f32,
    lane_type: u32,
    marking_type: u32,
    _pad: u32,
}

// Tile header: where in tile_segment_indices this tile's segments start
struct TileHeader {
    offset: u32,
    count: u32,
}
