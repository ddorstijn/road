//! Road evaluation functions shared between GPU shaders and CPU code.
//! Replicates the math from crates/road/src/primitives.rs.

use glam::Vec2;
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

pub fn eval_segment(seg_type: u32, s: f32, k_start: f32, k_end: f32, seg_length: f32) -> PoseResult {
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
    origin + Vec2::new(c * local_pos.x - s * local_pos.y, s * local_pos.x + c * local_pos.y)
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
    let s = point.x.clamp(0.0, seg_len);
    let closest = Vec2::new(s, 0.0);
    let diff = point - closest;
    let dist = diff.length();
    let signed_dist = if point.y >= 0.0 { dist } else { -dist };
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
    let s = clamped_angle / curvature.abs();

    let closest_pt = eval_arc(s, curvature);
    let diff = point - closest_pt;
    let dist = diff.length();

    let hdg = heading_arc(s, curvature);
    let tangent = Vec2::new(hdg.cos(), hdg.sin());
    let cross = tangent.x * diff.y - tangent.y * diff.x;
    let signed_dist = if cross >= 0.0 { dist } else { -dist };

    ClosestResult { s, signed_dist }
}

pub fn closest_point_spiral(point: Vec2, seg_length: f32, k_start: f32, k_end: f32) -> ClosestResult {
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
    let signed_dist = if cross >= 0.0 { dist } else { -dist };

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
