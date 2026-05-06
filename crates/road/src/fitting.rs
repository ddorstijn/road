use glam::Vec2;

use crate::primitives::{ClosestPointResult, PoseOnSegment, Segment};

/// A control point used to define a road path.
#[derive(Debug, Clone, Copy)]
pub struct ControlPoint {
    pub position: Vec2,
    /// Radius of the circular arc at this point (0 = sharp corner).
    pub turn_radius: f32,
    /// Length of the spiral (clothoid) transition at arc entry/exit.
    pub spiral_length: f32,
}

/// A fitted reference line composed of segments with known positions and headings.
#[derive(Debug, Clone)]
pub struct ReferenceLine {
    /// The ordered list of geometry segments.
    pub segments: Vec<Segment>,
    /// World-space origin of each segment's local frame.
    pub origins: Vec<Vec2>,
    /// World-space heading at the start of each segment (radians).
    pub headings: Vec<f32>,
    /// Cumulative arc-length at the start of each segment.
    pub s_offsets: Vec<f32>,
    /// Total length of the reference line.
    pub total_length: f32,
}

impl ReferenceLine {
    /// Fit a reference line through the given control points.
    ///
    /// Algorithm:
    /// - Between consecutive points, compute tangent directions.
    /// - At each interior point, insert an arc of given radius with spiral transitions.
    /// - Fill gaps between arcs/spirals with line segments.
    pub fn fit(points: &[ControlPoint]) -> Option<Self> {
        if points.len() < 2 {
            return None;
        }

        if points.len() == 2 {
            // Simple straight line between two points
            let dir = points[1].position - points[0].position;
            let length = dir.length();
            if length < 1e-6 {
                return None;
            }
            let heading = dir.y.atan2(dir.x);
            return Some(ReferenceLine {
                segments: vec![Segment::Line { length }],
                origins: vec![points[0].position],
                headings: vec![heading],
                s_offsets: vec![0.0],
                total_length: length,
            });
        }

        // For 3+ points, build segments through each pair with transitions at interior points
        let mut segments = Vec::new();
        let mut origins = Vec::new();
        let mut headings_out = Vec::new();
        let mut s_offsets = Vec::new();
        let mut current_s = 0.0f32;

        // Compute tangent directions between consecutive points
        let n = points.len();
        let mut tangents: Vec<Vec2> = Vec::with_capacity(n - 1);
        let mut lengths: Vec<f32> = Vec::with_capacity(n - 1);
        for i in 0..n - 1 {
            let d = points[i + 1].position - points[i].position;
            let l = d.length();
            lengths.push(l);
            tangents.push(if l > 1e-6 { d / l } else { Vec2::X });
        }

        // For each interior point, compute how much of the adjacent line segments
        // is consumed by the arc+spiral transitions
        struct TurnInfo {
            tangent_length: f32, // distance along tangent consumed (each side)
            #[allow(dead_code)]
            arc_angle: f32,
            arc_length: f32,
            spiral_length: f32,
            curvature: f32,
        }

        let mut turns: Vec<Option<TurnInfo>> = Vec::with_capacity(n);
        turns.push(None); // first point has no turn

        for i in 1..n - 1 {
            let cp = &points[i];
            if cp.turn_radius < 1e-6 {
                turns.push(None);
                continue;
            }

            let dir_in = tangents[i - 1];
            let dir_out = tangents[i];

            // Angle between directions
            let cos_a = dir_in.dot(dir_out).clamp(-1.0, 1.0);
            let half_angle = ((1.0 - cos_a) / 2.0).sqrt().asin(); // half the turn angle

            if half_angle.abs() < 1e-6 {
                turns.push(None);
                continue;
            }

            let full_angle = 2.0 * half_angle;
            let tangent_length = cp.turn_radius * half_angle.tan();
            let spiral_len = cp.spiral_length.min(tangent_length * 0.4); // don't let spiral exceed available space

            // Arc after removing spiral portions
            let spiral_angle = if cp.turn_radius > 1e-6 {
                spiral_len / (2.0 * cp.turn_radius) // approximate angle consumed by each spiral
            } else {
                0.0
            };
            let arc_angle = (full_angle - 2.0 * spiral_angle).max(0.0);
            let curvature = 1.0 / cp.turn_radius;
            let arc_length = arc_angle * cp.turn_radius;

            // Determine turn direction (left or right)
            let cross = dir_in.x * dir_out.y - dir_in.y * dir_out.x;
            let signed_curvature = if cross >= 0.0 { curvature } else { -curvature };

            turns.push(Some(TurnInfo {
                tangent_length: tangent_length + spiral_len,
                arc_angle,
                arc_length,
                spiral_length: spiral_len,
                curvature: signed_curvature,
            }));
        }
        turns.push(None); // last point has no turn

        // Now build the segment chain
        let mut current_pos = points[0].position;
        let mut current_heading = tangents[0].y.atan2(tangents[0].x);

        for i in 0..n - 1 {
            let _seg_heading = tangents[i].y.atan2(tangents[i].x);

            // How much of this line segment is available?
            let total_line_length = lengths[i];
            let consumed_start = if let Some(ref t) = turns[i] {
                t.tangent_length
            } else {
                0.0
            };
            let consumed_end = if let Some(ref t) = turns[i + 1] {
                t.tangent_length
            } else {
                0.0
            };

            let available = (total_line_length - consumed_start - consumed_end).max(0.0);

            // Emit line segment (if there's room)
            if available > 1e-4 {
                segments.push(Segment::Line { length: available });
                origins.push(current_pos);
                headings_out.push(current_heading);
                s_offsets.push(current_s);
                current_s += available;

                // Advance position
                let dir = Vec2::new(current_heading.cos(), current_heading.sin());
                current_pos += dir * available;
            }

            // Emit turn at point i+1 (if it's an interior point with a turn)
            if let Some(ref turn) = turns[i + 1] {
                // Entry spiral (curvature 0 → turn curvature)
                if turn.spiral_length > 1e-4 {
                    segments.push(Segment::Spiral {
                        length: turn.spiral_length,
                        k_start: 0.0,
                        k_end: turn.curvature,
                    });
                    origins.push(current_pos);
                    headings_out.push(current_heading);
                    s_offsets.push(current_s);

                    let pose = Segment::Spiral {
                        length: turn.spiral_length,
                        k_start: 0.0,
                        k_end: turn.curvature,
                    }
                    .evaluate(turn.spiral_length);
                    let (sin_h, cos_h) = current_heading.sin_cos();
                    current_pos += Vec2::new(
                        cos_h * pose.position.x - sin_h * pose.position.y,
                        sin_h * pose.position.x + cos_h * pose.position.y,
                    );
                    current_heading += pose.heading;
                    current_s += turn.spiral_length;
                }

                // Arc
                if turn.arc_length > 1e-4 {
                    segments.push(Segment::Arc {
                        length: turn.arc_length,
                        curvature: turn.curvature,
                    });
                    origins.push(current_pos);
                    headings_out.push(current_heading);
                    s_offsets.push(current_s);

                    let pose = Segment::Arc {
                        length: turn.arc_length,
                        curvature: turn.curvature,
                    }
                    .evaluate(turn.arc_length);
                    let (sin_h, cos_h) = current_heading.sin_cos();
                    current_pos += Vec2::new(
                        cos_h * pose.position.x - sin_h * pose.position.y,
                        sin_h * pose.position.x + cos_h * pose.position.y,
                    );
                    current_heading += pose.heading;
                    current_s += turn.arc_length;
                }

                // Exit spiral (turn curvature → 0)
                if turn.spiral_length > 1e-4 {
                    segments.push(Segment::Spiral {
                        length: turn.spiral_length,
                        k_start: turn.curvature,
                        k_end: 0.0,
                    });
                    origins.push(current_pos);
                    headings_out.push(current_heading);
                    s_offsets.push(current_s);

                    let pose = Segment::Spiral {
                        length: turn.spiral_length,
                        k_start: turn.curvature,
                        k_end: 0.0,
                    }
                    .evaluate(turn.spiral_length);
                    let (sin_h, cos_h) = current_heading.sin_cos();
                    current_pos += Vec2::new(
                        cos_h * pose.position.x - sin_h * pose.position.y,
                        sin_h * pose.position.x + cos_h * pose.position.y,
                    );
                    current_heading += pose.heading;
                    current_s += turn.spiral_length;
                }
            }
        }

        let total_length = current_s;

        Some(ReferenceLine {
            segments,
            origins,
            headings: headings_out,
            s_offsets,
            total_length,
        })
    }

    /// Evaluate the reference line at arc-length parameter `s`.
    /// Returns the world-space position and heading.
    pub fn evaluate(&self, s: f32) -> PoseOnSegment {
        let s = s.clamp(0.0, self.total_length);

        // Binary search for the segment containing s
        let idx = match self
            .s_offsets
            .binary_search_by(|off| off.partial_cmp(&s).unwrap())
        {
            Ok(i) => i,
            Err(i) => i.saturating_sub(1),
        };
        let idx = idx.min(self.segments.len() - 1);

        let local_s = s - self.s_offsets[idx];
        let seg_pose = self.segments[idx].evaluate(local_s);

        // Transform from segment-local to world space
        let heading = self.headings[idx];
        let (sin_h, cos_h) = heading.sin_cos();
        let world_pos = self.origins[idx]
            + Vec2::new(
                cos_h * seg_pose.position.x - sin_h * seg_pose.position.y,
                sin_h * seg_pose.position.x + cos_h * seg_pose.position.y,
            );
        let world_heading = heading + seg_pose.heading;

        PoseOnSegment {
            position: world_pos,
            heading: world_heading,
        }
    }

    /// Find the closest point on the reference line to a world-space point.
    pub fn closest_point(&self, point: Vec2) -> ClosestPointResult {
        let mut best = ClosestPointResult {
            s: 0.0,
            signed_distance: f32::MAX,
        };

        for (idx, seg) in self.segments.iter().enumerate() {
            // Transform point into segment-local frame
            let heading = self.headings[idx];
            let (sin_h, cos_h) = heading.sin_cos();
            let rel = point - self.origins[idx];
            let local = Vec2::new(
                cos_h * rel.x + sin_h * rel.y,
                -sin_h * rel.x + cos_h * rel.y,
            );

            let result = seg.closest_point(local);
            if result.signed_distance.abs() < best.signed_distance.abs() {
                best = ClosestPointResult {
                    s: self.s_offsets[idx] + result.s,
                    signed_distance: result.signed_distance,
                };
            }
        }

        best
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn fit_two_points_straight() {
        let points = vec![
            ControlPoint {
                position: Vec2::ZERO,
                turn_radius: 0.0,
                spiral_length: 0.0,
            },
            ControlPoint {
                position: Vec2::new(100.0, 0.0),
                turn_radius: 0.0,
                spiral_length: 0.0,
            },
        ];
        let rl = ReferenceLine::fit(&points).unwrap();
        assert!((rl.total_length - 100.0).abs() < 0.01);

        let pose = rl.evaluate(50.0);
        assert!((pose.position.x - 50.0).abs() < 0.01);
        assert!((pose.position.y).abs() < 0.01);
    }

    #[test]
    fn fit_three_points_with_turn() {
        let points = vec![
            ControlPoint {
                position: Vec2::new(0.0, 0.0),
                turn_radius: 0.0,
                spiral_length: 0.0,
            },
            ControlPoint {
                position: Vec2::new(50.0, 0.0),
                turn_radius: 20.0,
                spiral_length: 5.0,
            },
            ControlPoint {
                position: Vec2::new(50.0, 50.0),
                turn_radius: 0.0,
                spiral_length: 0.0,
            },
        ];
        let rl = ReferenceLine::fit(&points).unwrap();
        assert!(rl.total_length > 0.0);
        assert!(rl.segments.len() >= 3); // at least line + arc + line

        // Start should be near first point
        let start = rl.evaluate(0.0);
        assert!((start.position - points[0].position).length() < 0.01);

        // End should be near last point (some error due to arc/spiral geometry)
        let end = rl.evaluate(rl.total_length);
        assert!((end.position - points[2].position).length() < 5.0);
    }

    #[test]
    fn evaluate_and_closest_roundtrip() {
        let points = vec![
            ControlPoint {
                position: Vec2::ZERO,
                turn_radius: 0.0,
                spiral_length: 0.0,
            },
            ControlPoint {
                position: Vec2::new(100.0, 0.0),
                turn_radius: 0.0,
                spiral_length: 0.0,
            },
        ];
        let rl = ReferenceLine::fit(&points).unwrap();

        // Evaluate at s=30, offset by 5 to the left
        let pose = rl.evaluate(30.0);
        let normal = Vec2::new(-pose.heading.sin(), pose.heading.cos());
        let test_point = pose.position + normal * 5.0;

        let result = rl.closest_point(test_point);
        assert!((result.s - 30.0).abs() < 0.1);
        assert!((result.signed_distance - 5.0).abs() < 0.1);
    }

    #[test]
    fn fit_returns_none_for_insufficient_points() {
        assert!(ReferenceLine::fit(&[]).is_none());
        let single = vec![ControlPoint {
            position: Vec2::ZERO,
            turn_radius: 0.0,
            spiral_length: 0.0,
        }];
        assert!(ReferenceLine::fit(&single).is_none());
    }
}
