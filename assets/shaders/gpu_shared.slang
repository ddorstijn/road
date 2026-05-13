// GPU shared types — must match crates/gpu-shared/src/types.rs layout exactly.

struct GpuSegment {
    uint segment_type;
    float s_start;
    float2 origin;
    float heading;
    float length;
    float k_start;
    float k_end;
};

struct GpuRoad {
    uint segment_offset;
    uint segment_count;
    uint lane_section_offset;
    uint lane_section_count;
    float total_length;
    uint _pad0;
    uint _pad1;
    uint _pad2;
};

struct GpuLaneSection {
    float s_start;
    float s_end;
    uint lane_offset;
    uint lane_count;
    uint left_lane_count;
    uint _pad;
};

struct GpuLane {
    float width;
    uint lane_type;
    uint marking_type;
    uint _pad;
};

struct TileHeader {
    uint offset;
    uint count;
};

// Matches VkDrawIndirectCommand layout.
struct DrawIndirectCommand {
    uint vertex_count;
    uint instance_count;
    uint first_vertex;
    uint first_instance;
};

struct GpuTileInstance {
    float2 atlas_uv_offset;
    float2 atlas_uv_scale;
    float2 tile_world_origin;
    float2 tile_world_size;
    float2 road_id_uv_offset;
    float2 road_id_uv_scale;
};
