use glam::Vec2;
use std::f32::consts::PI;

/// A road geometry segment (one piece of a reference line).
#[derive(Debug, Clone, Copy)]
pub enum Segment {
    Line {
        length: f32,
    },
    Arc {
        length: f32,
        curvature: f32,
    },
    Spiral {
        length: f32,
        k_start: f32,
        k_end: f32,
    },
}

/// Result of evaluating a segment at parameter s.
#[derive(Debug, Clone, Copy)]
pub struct PoseOnSegment {
    pub position: Vec2,
    pub heading: f32,
}

/// Result of a closest-point query.
#[derive(Debug, Clone, Copy)]
pub struct ClosestPointResult {
    pub s: f32,
    pub signed_distance: f32,
}

impl Segment {
    /// Total arc length of this segment.
    pub fn length(&self) -> f32 {
        match *self {
            Segment::Line { length } => length,
            Segment::Arc { length, .. } => length,
            Segment::Spiral { length, .. } => length,
        }
    }

    /// Evaluate position and heading at distance `s` along the segment.
    /// The segment starts at the local origin heading along +X (heading=0).
    pub fn evaluate(&self, s: f32) -> PoseOnSegment {
        match *self {
            Segment::Line { .. } => PoseOnSegment {
                position: Vec2::new(s, 0.0),
                heading: 0.0,
            },
            Segment::Arc { curvature, .. } => {
                if curvature.abs() < 1e-9 {
                    return PoseOnSegment {
                        position: Vec2::new(s, 0.0),
                        heading: 0.0,
                    };
                }
                let r = 1.0 / curvature;
                let theta = s * curvature;
                PoseOnSegment {
                    position: Vec2::new(r * theta.sin(), r * (1.0 - theta.cos())),
                    heading: theta,
                }
            }
            Segment::Spiral {
                k_start,
                k_end,
                length,
            } => eval_spiral(s, k_start, k_end, length),
        }
    }

    /// Find the closest point on this segment to a given point (in segment-local frame).
    pub fn closest_point(&self, point: Vec2) -> ClosestPointResult {
        match *self {
            Segment::Line { length } => closest_point_line(point, length),
            Segment::Arc { length, curvature } => closest_point_arc(point, length, curvature),
            Segment::Spiral {
                length,
                k_start,
                k_end,
            } => closest_point_spiral(point, length, k_start, k_end),
        }
    }
}

// ---------------------------------------------------------------------------
// Line closest point
// ---------------------------------------------------------------------------

fn closest_point_line(point: Vec2, length: f32) -> ClosestPointResult {
    let s = point.x.clamp(0.0, length);
    let closest = Vec2::new(s, 0.0);
    let dist = (point - closest).length();
    let signed_distance = dist.copysign(point.y);
    ClosestPointResult { s, signed_distance }
}

// ---------------------------------------------------------------------------
// Arc closest point
// ---------------------------------------------------------------------------

fn closest_point_arc(point: Vec2, length: f32, curvature: f32) -> ClosestPointResult {
    if curvature.abs() < 1e-9 {
        return closest_point_line(point, length);
    }

    let r = 1.0 / curvature;
    let center = Vec2::new(0.0, r);
    let to_point = point - center;

    // Angle from center to point (measured from start of arc)
    let angle = (-to_point.x).atan2(to_point.y * curvature.signum());
    let angle = if angle < 0.0 { angle + 2.0 * PI } else { angle };

    let max_angle = (length * curvature).abs();
    let clamped_angle = angle.clamp(0.0, max_angle);
    let s = clamped_angle / curvature.abs();

    // Evaluate arc at clamped s for the actual closest point
    let seg = Segment::Arc { length, curvature };
    let closest = seg.evaluate(s);
    let diff = point - closest.position;
    let dist = diff.length();

    // Sign via cross product of tangent and diff
    let tangent = Vec2::new(closest.heading.cos(), closest.heading.sin());
    let cross = tangent.x * diff.y - tangent.y * diff.x;
    let signed_distance = if cross >= 0.0 { dist } else { -dist };

    ClosestPointResult { s, signed_distance }
}

// ---------------------------------------------------------------------------
// Spiral (Clothoid) evaluation via Taylor series
// ---------------------------------------------------------------------------

/// Evaluate a clothoid (Euler spiral) at parameter s.
/// Curvature varies linearly from k_start to k_end over `total_length`.
fn eval_spiral(s: f32, k_start: f32, k_end: f32, total_length: f32) -> PoseOnSegment {
    // Curvature at parameter t: k(t) = k_start + (k_end - k_start) * t / total_length
    // Heading: theta(t) = k_start * t + 0.5 * (k_end - k_start) * t^2 / total_length
    // Position: integrate cos(theta), sin(theta) via Taylor series

    let dk = if total_length > 1e-9 {
        (k_end - k_start) / total_length
    } else {
        0.0
    };

    // Heading at s
    let heading = k_start * s + 0.5 * dk * s * s;

    // Numerical integration using Simpson's rule with 16 intervals
    let n = 16usize;
    let h = s / n as f32;
    let mut sum_x = 0.0f32;
    let mut sum_y = 0.0f32;

    for i in 0..=n {
        let t = i as f32 * h;
        let theta = k_start * t + 0.5 * dk * t * t;
        let weight = if i == 0 || i == n {
            1.0
        } else if i % 2 == 1 {
            4.0
        } else {
            2.0
        };
        sum_x += weight * theta.cos();
        sum_y += weight * theta.sin();
    }

    let position = Vec2::new(sum_x * h / 3.0, sum_y * h / 3.0);

    PoseOnSegment { position, heading }
}

// ---------------------------------------------------------------------------
// Spiral closest point (Newton-Raphson)
// ---------------------------------------------------------------------------

fn closest_point_spiral(point: Vec2, length: f32, k_start: f32, k_end: f32) -> ClosestPointResult {
    // Initial guess: sample a few points and pick the closest
    let samples = 8;
    let mut best_s = 0.0f32;
    let mut best_dist_sq = f32::MAX;

    for i in 0..=samples {
        let t = length * i as f32 / samples as f32;
        let pose = eval_spiral(t, k_start, k_end, length);
        let d = (pose.position - point).length_squared();
        if d < best_dist_sq {
            best_dist_sq = d;
            best_s = t;
        }
    }

    // Newton-Raphson refinement (4 iterations)
    let mut s = best_s;
    for _ in 0..4 {
        let pose = eval_spiral(s, k_start, k_end, length);
        let diff = pose.position - point;
        let tangent = Vec2::new(pose.heading.cos(), pose.heading.sin());

        // f(s) = dot(P(s) - point, T(s)) = 0 at closest point
        let f = diff.dot(tangent);
        // f'(s) ≈ 1 + dot(diff, dT/ds) where dT/ds = k(s) * N(s)
        let dk = if length > 1e-9 {
            (k_end - k_start) / length
        } else {
            0.0
        };
        let k_at_s = k_start + dk * s;
        let normal = Vec2::new(-pose.heading.sin(), pose.heading.cos());
        let f_prime = 1.0 + diff.dot(normal) * k_at_s;

        if f_prime.abs() > 1e-9 {
            s -= f / f_prime;
        }
        s = s.clamp(0.0, length);
    }

    let pose = eval_spiral(s, k_start, k_end, length);
    let diff = point - pose.position;
    let dist = diff.length();
    let tangent = Vec2::new(pose.heading.cos(), pose.heading.sin());
    let cross = tangent.x * diff.y - tangent.y * diff.x;
    let signed_distance = if cross >= 0.0 { dist } else { -dist };

    ClosestPointResult { s, signed_distance }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f32 = 1e-4;

    #[test]
    fn line_evaluate() {
        let seg = Segment::Line { length: 10.0 };
        let pose = seg.evaluate(5.0);
        assert!((pose.position.x - 5.0).abs() < EPSILON);
        assert!((pose.position.y).abs() < EPSILON);
        assert!((pose.heading).abs() < EPSILON);
    }

    #[test]
    fn line_closest_point() {
        let seg = Segment::Line { length: 10.0 };
        let res = seg.closest_point(Vec2::new(5.0, 3.0));
        assert!((res.s - 5.0).abs() < EPSILON);
        assert!((res.signed_distance - 3.0).abs() < EPSILON);
    }

    #[test]
    fn line_closest_point_clamped() {
        let seg = Segment::Line { length: 10.0 };
        let res = seg.closest_point(Vec2::new(-2.0, 1.0));
        assert!((res.s).abs() < EPSILON);
    }

    #[test]
    fn arc_evaluate_quarter_circle() {
        // Curvature = 1/10, length = pi/2 * 10 = quarter circle of radius 10
        let r = 10.0;
        let length = PI / 2.0 * r;
        let seg = Segment::Arc {
            length,
            curvature: 1.0 / r,
        };
        let pose = seg.evaluate(length);
        // End of quarter circle: (r, r) relative to center offset
        assert!((pose.position.x - r).abs() < 0.01);
        assert!((pose.position.y - r).abs() < 0.01);
        assert!((pose.heading - PI / 2.0).abs() < 0.01);
    }

    #[test]
    fn arc_evaluate_negative_curvature() {
        let r = 10.0;
        let length = PI / 2.0 * r;
        let seg = Segment::Arc {
            length,
            curvature: -1.0 / r,
        };
        let pose = seg.evaluate(length);
        assert!((pose.position.x - r).abs() < 0.01);
        assert!((pose.position.y + r).abs() < 0.01); // curves right
        assert!((pose.heading + PI / 2.0).abs() < 0.01);
    }

    #[test]
    fn arc_closest_point() {
        let r = 10.0;
        let seg = Segment::Arc {
            length: PI * r,
            curvature: 1.0 / r,
        };
        // Point at center of arc circle should be at distance -r from arc
        let res = seg.closest_point(Vec2::new(0.0, r));
        // On a left-curving arc with center at (0, r), a point at the center
        // is r away from the arc (inside the curve)
        assert!(res.signed_distance.abs() - r < 0.5);
    }

    #[test]
    fn spiral_straight_when_zero_curvature() {
        // Spiral with k_start=0, k_end=0 should be a straight line
        let seg = Segment::Spiral {
            length: 10.0,
            k_start: 0.0,
            k_end: 0.0,
        };
        let pose = seg.evaluate(5.0);
        assert!((pose.position.x - 5.0).abs() < 0.01);
        assert!((pose.position.y).abs() < 0.01);
        assert!((pose.heading).abs() < 0.01);
    }

    #[test]
    fn spiral_approaches_arc() {
        // Spiral with k_start=k_end=k should behave like an arc
        let k = 0.1;
        let length = 5.0;
        let spiral = Segment::Spiral {
            length,
            k_start: k,
            k_end: k,
        };
        let arc = Segment::Arc {
            length,
            curvature: k,
        };
        let p_spiral = spiral.evaluate(length);
        let p_arc = arc.evaluate(length);
        assert!((p_spiral.position - p_arc.position).length() < 0.01);
        assert!((p_spiral.heading - p_arc.heading).abs() < 0.01);
    }

    #[test]
    fn spiral_closest_point_on_curve() {
        let seg = Segment::Spiral {
            length: 20.0,
            k_start: 0.0,
            k_end: 0.1,
        };
        let pose = seg.evaluate(10.0);
        // Offset perpendicular to the curve (along the normal) at s=10
        let normal = Vec2::new(-pose.heading.sin(), pose.heading.cos());
        let offset_point = pose.position + normal * 2.0;
        let res = seg.closest_point(offset_point);
        assert!((res.s - 10.0).abs() < 0.5);
    }
}
