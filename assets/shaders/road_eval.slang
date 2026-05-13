// Road evaluation functions shared between GPU shaders.
// Replicates the math from crates/road/src/primitives.rs.

import gpu_shared;

struct PoseResult {
    float2 position;
    float heading;
};

struct ClosestResult {
    float s;
    float signed_dist;
};

// ---------------------------------------------------------------------------
// Segment evaluation: position and heading at parameter s (segment-local)
// ---------------------------------------------------------------------------

float2 eval_line(float s) {
    return float2(s, 0.0);
}

float heading_line() {
    return 0.0;
}

float2 eval_arc(float s, float curvature) {
    if (abs(curvature) < 1e-9)
        return float2(s, 0.0);
    float r = 1.0 / curvature;
    float theta = s * curvature;
    return float2(r * sin(theta), r * (1.0 - cos(theta)));
}

float heading_arc(float s, float curvature) {
    return s * curvature;
}

// Spiral (clothoid) evaluation via Simpson's rule (16 intervals).
float2 eval_spiral(float s, float k_start, float k_end, float total_length) {
    float dk = (total_length > 1e-9) ? (k_end - k_start) / total_length : 0.0;

    uint n = 16;
    float h = s / float(n);
    float sum_x = 0.0;
    float sum_y = 0.0;

    for (uint i = 0; i <= n; i++) {
        float t = float(i) * h;
        float theta = k_start * t + 0.5 * dk * t * t;
        float w;
        if (i == 0 || i == n)
            w = 1.0;
        else if (i % 2 == 1)
            w = 4.0;
        else
            w = 2.0;
        sum_x += w * cos(theta);
        sum_y += w * sin(theta);
    }

    return float2(sum_x * h / 3.0, sum_y * h / 3.0);
}

float heading_spiral(float s, float k_start, float k_end, float total_length) {
    float dk = (total_length > 1e-9) ? (k_end - k_start) / total_length : 0.0;
    return k_start * s + 0.5 * dk * s * s;
}

// ---------------------------------------------------------------------------
// Evaluate a segment at local parameter s -> (position, heading)
// ---------------------------------------------------------------------------

PoseResult eval_segment(uint seg_type, float s, float k_start, float k_end, float seg_length) {
    PoseResult result;
    switch (seg_type) {
    case 0:
        result.position = eval_line(s);
        result.heading = heading_line();
        break;
    case 1:
        result.position = eval_arc(s, k_start);
        result.heading = heading_arc(s, k_start);
        break;
    default:
        result.position = eval_spiral(s, k_start, k_end, seg_length);
        result.heading = heading_spiral(s, k_start, k_end, seg_length);
        break;
    }
    return result;
}

// Transform from segment-local coordinates to world coordinates.
float2 segment_local_to_world(float2 local_pos, float2 origin, float heading) {
    float c = cos(heading);
    float s = sin(heading);
    return origin + float2(c * local_pos.x - s * local_pos.y,
                           s * local_pos.x + c * local_pos.y);
}

// Transform from world coordinates to segment-local coordinates.
float2 world_to_segment_local(float2 world_pos, float2 origin, float heading) {
    float c = cos(heading);
    float s = sin(heading);
    float2 d = world_pos - origin;
    return float2(c * d.x + s * d.y, -s * d.x + c * d.y);
}

// ---------------------------------------------------------------------------
// Closest point on a segment (segment-local coordinates)
// ---------------------------------------------------------------------------

ClosestResult closest_point_line(float2 point, float seg_len) {
    float raw_s = point.x;
    float s = clamp(raw_s, 0.0, seg_len);
    bool clamped = raw_s < 0.0 || raw_s > seg_len;
    float2 closest = float2(s, 0.0);
    float2 diff = point - closest;
    float dist = length(diff);
    float signed_dist;
    if (clamped)
        signed_dist = dist;
    else if (point.y >= 0.0)
        signed_dist = dist;
    else
        signed_dist = -dist;

    ClosestResult cr;
    cr.s = s;
    cr.signed_dist = signed_dist;
    return cr;
}

ClosestResult closest_point_arc(float2 point, float seg_length, float curvature) {
    if (abs(curvature) < 1e-9)
        return closest_point_line(point, seg_length);

    float r = 1.0 / curvature;
    float2 center = float2(0.0, r);
    float2 to_point = point - center;

    float theta = atan2(curvature * to_point.x, -curvature * to_point.y);
    float unsigned_angle = theta * sign(curvature);

    float max_angle = abs(seg_length * curvature);
    float clamped_angle = clamp(unsigned_angle, 0.0, max_angle);
    bool endpoint_clamped = unsigned_angle < 0.0 || unsigned_angle > max_angle;
    float s = clamped_angle / abs(curvature);

    float2 closest_pt = eval_arc(s, curvature);
    float2 diff = point - closest_pt;
    float dist = length(diff);

    float hdg = heading_arc(s, curvature);
    float2 tangent = float2(cos(hdg), sin(hdg));
    float cross_val = tangent.x * diff.y - tangent.y * diff.x;

    float signed_dist;
    if (endpoint_clamped)
        signed_dist = dist;
    else if (cross_val >= 0.0)
        signed_dist = dist;
    else
        signed_dist = -dist;

    ClosestResult cr;
    cr.s = s;
    cr.signed_dist = signed_dist;
    return cr;
}

ClosestResult closest_point_spiral(float2 point, float seg_length, float k_start, float k_end) {
    float dk = (seg_length > 1e-9) ? (k_end - k_start) / seg_length : 0.0;

    // Coarse search: 8 samples
    float best_s = 0.0;
    float best_dist_sq = 1e30;
    for (uint i = 0; i <= 8; i++) {
        float t = seg_length * float(i) / 8.0;
        float2 pos = eval_spiral(t, k_start, k_end, seg_length);
        float d = dot(pos - point, pos - point);
        if (d < best_dist_sq) {
            best_dist_sq = d;
            best_s = t;
        }
    }

    // Newton-Raphson refinement: 4 iterations
    float s = best_s;
    for (uint iter = 0; iter < 4; iter++) {
        float2 pos = eval_spiral(s, k_start, k_end, seg_length);
        float2 diff = pos - point;
        float hdg = heading_spiral(s, k_start, k_end, seg_length);
        float2 tangent = float2(cos(hdg), sin(hdg));
        float f_val = dot(diff, tangent);
        float k_at_s = k_start + dk * s;
        float2 normal = float2(-sin(hdg), cos(hdg));
        float f_prime = 1.0 + dot(diff, normal) * k_at_s;
        if (abs(f_prime) > 1e-9)
            s -= f_val / f_prime;
        s = clamp(s, 0.0, seg_length);
    }

    float2 pose = eval_spiral(s, k_start, k_end, seg_length);
    float2 diff = point - pose;
    float dist = length(diff);
    float hdg = heading_spiral(s, k_start, k_end, seg_length);
    float2 tangent = float2(cos(hdg), sin(hdg));
    float cross_val = tangent.x * diff.y - tangent.y * diff.x;
    bool endpoint_clamped = s <= 0.001 || s >= seg_length - 0.001;

    float signed_dist;
    if (endpoint_clamped)
        signed_dist = dist;
    else if (cross_val >= 0.0)
        signed_dist = dist;
    else
        signed_dist = -dist;

    ClosestResult cr;
    cr.s = s;
    cr.signed_dist = signed_dist;
    return cr;
}

ClosestResult closest_point_on_segment(uint seg_type, float2 point, float seg_length, float k_start, float k_end) {
    switch (seg_type) {
    case 0: return closest_point_line(point, seg_length);
    case 1: return closest_point_arc(point, seg_length, k_start);
    default: return closest_point_spiral(point, seg_length, k_start, k_end);
    }
}

// ---------------------------------------------------------------------------
// High-level road evaluation helpers
// ---------------------------------------------------------------------------

// Evaluate a road at a given s-coordinate, returning world position + heading.
PoseResult evaluate_road_at_s(
    uint road_id, float s_global,
    StructuredBuffer<GpuRoad> roads,
    StructuredBuffer<GpuSegment> segments)
{
    GpuRoad road = roads[road_id];
    uint seg_idx = road.segment_offset;
    float local_s = s_global;

    for (uint i = 0; i < road.segment_count; i++) {
        GpuSegment seg = segments[road.segment_offset + i];
        if (s_global < seg.s_start + seg.length || i == road.segment_count - 1) {
            seg_idx = road.segment_offset + i;
            local_s = clamp(s_global - seg.s_start, 0.0, seg.length);
            break;
        }
    }

    GpuSegment seg = segments[seg_idx];
    PoseResult pose = eval_segment(seg.segment_type, local_s, seg.k_start, seg.k_end, seg.length);

    float2 origin = seg.origin;
    PoseResult result;
    result.position = segment_local_to_world(pose.position, origin, seg.heading);
    result.heading = seg.heading + pose.heading;
    return result;
}

// Compute the lateral offset for a given lane on a road at s.
float compute_lane_offset(
    uint road_id, int lane_idx, float s_global,
    StructuredBuffer<GpuRoad> roads,
    StructuredBuffer<GpuLaneSection> lane_sections,
    StructuredBuffer<GpuLane> lanes)
{
    GpuRoad road = roads[road_id];

    uint section_idx = road.lane_section_offset;
    for (uint i = 0; i < road.lane_section_count; i++) {
        GpuLaneSection ls = lane_sections[road.lane_section_offset + i];
        if (s_global >= ls.s_start && s_global < ls.s_end) {
            section_idx = road.lane_section_offset + i;
            break;
        }
        section_idx = road.lane_section_offset + i;
    }

    GpuLaneSection section = lane_sections[section_idx];
    int left_count = int(section.left_lane_count);
    float offset = 0.0;

    if (lane_idx >= 0) {
        uint right_start = uint(left_count);
        for (uint j = 0; j < uint(lane_idx); j++) {
            offset += lanes[section.lane_offset + right_start + j].width;
        }
        offset += lanes[section.lane_offset + right_start + uint(lane_idx)].width * 0.5;
        offset = -offset;
    } else {
        uint left_idx = uint(-lane_idx - 1);
        for (uint j = 0; j < left_idx; j++) {
            offset += lanes[section.lane_offset + j].width;
        }
        offset += lanes[section.lane_offset + left_idx].width * 0.5;
    }

    return offset;
}

// Return the curvature of the reference line at a given s-coordinate on a road.
float curvature_at_s(
    uint road_id, float s_global,
    StructuredBuffer<GpuRoad> roads,
    StructuredBuffer<GpuSegment> segments)
{
    GpuRoad road = roads[road_id];

    float local_s = s_global;
    uint seg_idx = road.segment_offset;

    for (uint i = 0; i < road.segment_count; i++) {
        GpuSegment seg = segments[road.segment_offset + i];
        if (s_global < seg.s_start + seg.length || i == road.segment_count - 1) {
            seg_idx = road.segment_offset + i;
            local_s = clamp(s_global - seg.s_start, 0.0, seg.length);
            break;
        }
    }

    GpuSegment seg = segments[seg_idx];
    switch (seg.segment_type) {
    case 0: return 0.0; // Line
    case 1: return seg.k_start; // Arc (constant curvature)
    default: {
        // Spiral: linearly interpolated curvature
        float dk = (seg.length > 1e-9) ? (seg.k_end - seg.k_start) / seg.length : 0.0;
        return seg.k_start + dk * local_s;
    }
    }
}
