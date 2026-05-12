//! Road evaluation functions shared between GPU shaders and CPU code.
//! Replicates the math from crates/road/src/primitives.rs.

use gpu_shared::{GpuLane, GpuLaneSection, GpuRoad, GpuSegment};
use spirv_std::glam::Vec2;
use spirv_std::num_traits::Float;

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct PoseResult {
    pub position: Vec2,
    pub heading: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct ClosestResult {
    pub s: f32,
    pub signed_dist: f32,
}

// ---------------------------------------------------------------------------
// Segment evaluation: position and heading at parameter s (segment-local)
// ---------------------------------------------------------------------------

pub fn eval_line(s: f32) -> Vec2 {
    Vec2::new(s, 0.0)
}

pub fn heading_line() -> f32 {
    0.0
}

pub fn eval_arc(s: f32, curvature: f32) -> Vec2 {
    if curvature.abs() < 1e-9 {
        return Vec2::new(s, 0.0);
    }
    let r = 1.0 / curvature;
    let theta = s * curvature;
    Vec2::new(r * theta.sin(), r * (1.0 - theta.cos()))
}

pub fn heading_arc(s: f32, curvature: f32) -> f32 {
    s * curvature
}

/// Spiral (clothoid) evaluation via Simpson's rule (16 intervals).
pub fn eval_spiral(s: f32, k_start: f32, k_end: f32, total_length: f32) -> Vec2 {
    let dk = if total_length > 1e-9 {
        (k_end - k_start) / total_length
    } else {
        0.0
    };

    let n = 16u32;
    let h = s / n as f32;
    let mut sum_x = 0.0f32;
    let mut sum_y = 0.0f32;

    let mut i = 0u32;
    while i <= n {
        let t = i as f32 * h;
        let theta = k_start * t + 0.5 * dk * t * t;
        let w = if i == 0 || i == n {
            1.0
        } else if i % 2 == 1 {
            4.0
        } else {
            2.0
        };
        sum_x += w * theta.cos();
        sum_y += w * theta.sin();
        i += 1;
    }

    Vec2::new(sum_x * h / 3.0, sum_y * h / 3.0)
}

pub fn heading_spiral(s: f32, k_start: f32, k_end: f32, total_length: f32) -> f32 {
    let dk = if total_length > 1e-9 {
        (k_end - k_start) / total_length
    } else {
        0.0
    };
    k_start * s + 0.5 * dk * s * s
}

// ---------------------------------------------------------------------------
// Evaluate a segment at local parameter s → (position, heading)
// ---------------------------------------------------------------------------

pub fn eval_segment(
    seg_type: u32,
    s: f32,
    k_start: f32,
    k_end: f32,
    seg_length: f32,
) -> PoseResult {
    match seg_type {
        0 => PoseResult {
            position: eval_line(s),
            heading: heading_line(),
        },
        1 => PoseResult {
            position: eval_arc(s, k_start),
            heading: heading_arc(s, k_start),
        },
        _ => PoseResult {
            position: eval_spiral(s, k_start, k_end, seg_length),
            heading: heading_spiral(s, k_start, k_end, seg_length),
        },
    }
}

/// Transform from segment-local coordinates to world coordinates.
pub fn segment_local_to_world(local_pos: Vec2, origin: Vec2, heading: f32) -> Vec2 {
    let c = heading.cos();
    let s = heading.sin();
    origin
        + Vec2::new(
            c * local_pos.x - s * local_pos.y,
            s * local_pos.x + c * local_pos.y,
        )
}

/// Transform from world coordinates to segment-local coordinates.
pub fn world_to_segment_local(world_pos: Vec2, origin: Vec2, heading: f32) -> Vec2 {
    let c = heading.cos();
    let s = heading.sin();
    let d = world_pos - origin;
    Vec2::new(c * d.x + s * d.y, -s * d.x + c * d.y)
}

// ---------------------------------------------------------------------------
// Closest point on a segment (segment-local coordinates)
// ---------------------------------------------------------------------------

pub fn closest_point_line(point: Vec2, seg_len: f32) -> ClosestResult {
    let raw_s = point.x;
    let s = raw_s.clamp(0.0, seg_len);
    let clamped = raw_s < 0.0 || raw_s > seg_len;
    let closest = Vec2::new(s, 0.0);
    let diff = point - closest;
    let dist = diff.length();
    // At endpoints (s clamped), use unsigned distance to avoid a sign
    // discontinuity along the tangent that creates spike artifacts
    // when bilinearly interpolated.
    let signed_dist = if clamped {
        dist
    } else if point.y >= 0.0 {
        dist
    } else {
        -dist
    };
    ClosestResult { s, signed_dist }
}

pub fn closest_point_arc(point: Vec2, seg_length: f32, curvature: f32) -> ClosestResult {
    if curvature.abs() < 1e-9 {
        return closest_point_line(point, seg_length);
    }

    let r = 1.0 / curvature;
    let center = Vec2::new(0.0, r);
    let to_point = point - center;

    let theta = (curvature * to_point.x).atan2(-curvature * to_point.y);
    let unsigned_angle = theta * curvature.signum();

    let max_angle = (seg_length * curvature).abs();
    let clamped_angle = unsigned_angle.clamp(0.0, max_angle);
    let endpoint_clamped = unsigned_angle < 0.0 || unsigned_angle > max_angle;
    let s = clamped_angle / curvature.abs();

    let closest_pt = eval_arc(s, curvature);
    let diff = point - closest_pt;
    let dist = diff.length();

    let hdg = heading_arc(s, curvature);
    let tangent = Vec2::new(hdg.cos(), hdg.sin());
    let cross = tangent.x * diff.y - tangent.y * diff.x;
    // At endpoints (angle clamped), use unsigned distance to avoid sign
    // discontinuity that creates spike artifacts when bilinearly interpolated.
    let signed_dist = if endpoint_clamped {
        dist
    } else if cross >= 0.0 {
        dist
    } else {
        -dist
    };

    ClosestResult { s, signed_dist }
}

pub fn closest_point_spiral(
    point: Vec2,
    seg_length: f32,
    k_start: f32,
    k_end: f32,
) -> ClosestResult {
    let dk = if seg_length > 1e-9 {
        (k_end - k_start) / seg_length
    } else {
        0.0
    };

    // Coarse search: 8 samples
    let mut best_s = 0.0f32;
    let mut best_dist_sq = 1e30f32;
    let mut i = 0u32;
    while i <= 8 {
        let t = seg_length * i as f32 / 8.0;
        let pos = eval_spiral(t, k_start, k_end, seg_length);
        let d = (pos - point).length_squared();
        if d < best_dist_sq {
            best_dist_sq = d;
            best_s = t;
        }
        i += 1;
    }

    // Newton-Raphson refinement: 4 iterations
    let mut s = best_s;
    let mut iter = 0u32;
    while iter < 4 {
        let pos = eval_spiral(s, k_start, k_end, seg_length);
        let diff = pos - point;
        let hdg = heading_spiral(s, k_start, k_end, seg_length);
        let tangent = Vec2::new(hdg.cos(), hdg.sin());
        let f_val = diff.dot(tangent);
        let k_at_s = k_start + dk * s;
        let normal = Vec2::new(-hdg.sin(), hdg.cos());
        let f_prime: f32 = 1.0 + diff.dot(normal) * k_at_s;
        if f_prime.abs() > 1e-9 {
            s -= f_val / f_prime;
        }
        s = s.clamp(0.0, seg_length);
        iter += 1;
    }

    let pose = eval_spiral(s, k_start, k_end, seg_length);
    let diff = point - pose;
    let dist = diff.length();
    let hdg = heading_spiral(s, k_start, k_end, seg_length);
    let tangent = Vec2::new(hdg.cos(), hdg.sin());
    let cross = tangent.x * diff.y - tangent.y * diff.x;
    // At endpoints (s clamped), use unsigned distance to avoid sign
    // discontinuity that creates spike artifacts when bilinearly interpolated.
    let endpoint_clamped = s <= 0.001 || s >= seg_length - 0.001;
    let signed_dist = if endpoint_clamped {
        dist
    } else if cross >= 0.0 {
        dist
    } else {
        -dist
    };

    ClosestResult { s, signed_dist }
}

pub fn closest_point_on_segment(
    seg_type: u32,
    point: Vec2,
    seg_length: f32,
    k_start: f32,
    k_end: f32,
) -> ClosestResult {
    match seg_type {
        0 => closest_point_line(point, seg_length),
        1 => closest_point_arc(point, seg_length, k_start),
        _ => closest_point_spiral(point, seg_length, k_start, k_end),
    }
}

// ---------------------------------------------------------------------------
// High-level road evaluation helpers (shared by car_render and frustum_cull)
// ---------------------------------------------------------------------------

/// Evaluate a road at a given s-coordinate, returning world position + heading.
pub fn evaluate_road_at_s(
    road_id: u32,
    s_global: f32,
    roads: &[GpuRoad],
    segments: &[GpuSegment],
) -> PoseResult {
    let road = roads[road_id as usize];
    let mut seg_idx = road.segment_offset;
    let mut local_s = s_global;

    let mut i = 0u32;
    while i < road.segment_count {
        let seg = segments[(road.segment_offset + i) as usize];
        if s_global < seg.s_start + seg.length || i == road.segment_count - 1 {
            seg_idx = road.segment_offset + i;
            local_s = (s_global - seg.s_start).clamp(0.0, seg.length);
            break;
        }
        i += 1;
    }

    let seg = segments[seg_idx as usize];
    let pose = eval_segment(
        seg.segment_type,
        local_s,
        seg.k_start,
        seg.k_end,
        seg.length,
    );

    let origin = Vec2::new(seg.origin[0], seg.origin[1]);
    PoseResult {
        position: segment_local_to_world(pose.position, origin, seg.heading),
        heading: seg.heading + pose.heading,
    }
}

/// Compute the lateral offset for a given lane on a road at s.
pub fn compute_lane_offset(
    road_id: u32,
    lane_idx: i32,
    s_global: f32,
    roads: &[GpuRoad],
    lane_sections: &[GpuLaneSection],
    lanes: &[GpuLane],
) -> f32 {
    let road = roads[road_id as usize];

    let mut section_idx = road.lane_section_offset;
    let mut i = 0u32;
    while i < road.lane_section_count {
        let ls = lane_sections[(road.lane_section_offset + i) as usize];
        if s_global >= ls.s_start && s_global < ls.s_end {
            section_idx = road.lane_section_offset + i;
            break;
        }
        section_idx = road.lane_section_offset + i;
        i += 1;
    }

    let section = lane_sections[section_idx as usize];
    let left_count = section.left_lane_count as i32;
    let mut offset = 0.0f32;

    if lane_idx >= 0 {
        let right_start = left_count as u32;
        let mut j = 0u32;
        while j < lane_idx as u32 {
            let l = lanes[(section.lane_offset + right_start + j) as usize];
            offset += l.width;
            j += 1;
        }
        let current = lanes[(section.lane_offset + right_start + lane_idx as u32) as usize];
        offset += current.width * 0.5;
        offset = -offset;
    } else {
        let left_idx = (-lane_idx - 1) as u32;
        let mut j = 0u32;
        while j < left_idx {
            let l = lanes[(section.lane_offset + j) as usize];
            offset += l.width;
            j += 1;
        }
        let current = lanes[(section.lane_offset + left_idx) as usize];
        offset += current.width * 0.5;
    }

    offset
}
