use glam::Vec2;

use crate::primitives::{ClosestPointResult, PoseOnSegment, Segment};

/// A control point used to define a road path.
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
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
        let mut tangent_headings: Vec<f32> = Vec::with_capacity(n - 1);
        let mut lengths: Vec<f32> = Vec::with_capacity(n - 1);
        for i in 0..n - 1 {
            let d = points[i + 1].position - points[i].position;
            let l = d.length();
            lengths.push(l);
            let t = if l > 1e-6 { d / l } else { Vec2::X };
            tangents.push(t);
            tangent_headings.push(t.y.atan2(t.x));
        }

        // For each interior point, compute the turn geometry
        struct TurnInfo {
            tangent_length: f32, // distance along tangent consumed on each side
            #[allow(dead_code)]
            turn_angle: f32, // signed turn angle
            arc_length: f32,
            spiral_length: f32,
            curvature: f32, // signed curvature
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

            // Signed turn angle via atan2 of cross and dot
            let cross = dir_in.x * dir_out.y - dir_in.y * dir_out.x;
            let dot = dir_in.dot(dir_out).clamp(-1.0, 1.0);
            let turn_angle = cross.atan2(dot); // positive = left turn

            if turn_angle.abs() < 1e-6 {
                turns.push(None);
                continue;
            }

            let half_abs = turn_angle.abs() / 2.0;
            let tan_half = half_abs.tan();

            // Clamp the effective turn radius so the tangent length fits within
            // the available space on both adjacent segments (use at most 45% of
            // each segment length to leave room for turns at neighboring points).
            let max_tangent = (lengths[i - 1].min(lengths[i])) * 0.45;
            let max_radius = if tan_half > 1e-6 {
                max_tangent / tan_half
            } else {
                cp.turn_radius
            };
            let radius = cp.turn_radius.min(max_radius).max(0.0);

            if radius < 1e-6 {
                turns.push(None);
                continue;
            }

            // Limit spiral length to fit within a reasonable fraction of the tangent distance
            let basic_tangent = radius * tan_half;
            let spiral_len = cp.spiral_length.min(basic_tangent * 0.4);

            // The spiral consumes some of the turn angle
            let spiral_angle = if radius > 1e-6 {
                spiral_len / (2.0 * radius)
            } else {
                0.0
            };

            // Compute spiral shift (p) and long tangent offset (k) from the
            // spiral endpoint, using the standard highway design formulas.
            // Evaluate the spiral to get its endpoint in local coordinates.
            let (tangent_length, arc_abs_angle) = if spiral_len > 1e-6 {
                let spiral_end = Segment::Spiral {
                    length: spiral_len,
                    k_start: 0.0,
                    k_end: 1.0 / radius, // unsigned curvature for computation
                }
                .evaluate(spiral_len);
                let x_s = spiral_end.position.x;
                let y_s = spiral_end.position.y;
                // p = lateral shift of spiral from the arc
                let p = y_s - radius * (1.0 - spiral_angle.cos());
                // k = longitudinal offset
                let k = x_s - radius * spiral_angle.sin();
                let t = (radius + p) * tan_half + k;
                let arc_angle = (turn_angle.abs() - 2.0 * spiral_angle).max(0.0);
                (t, arc_angle)
            } else {
                (basic_tangent, turn_angle.abs())
            };

            let arc_length = arc_abs_angle * radius;
            let signed_curvature = turn_angle.signum() / radius;

            turns.push(Some(TurnInfo {
                tangent_length,
                turn_angle,
                arc_length,
                spiral_length: spiral_len,
                curvature: signed_curvature,
            }));
        }
        turns.push(None); // last point has no turn

        // Build the segment chain.
        // After each turn, snap heading to the exact outgoing tangent direction
        // to avoid accumulated numerical drift. Line lengths are computed
        // dynamically from current_pos to absorb small position errors from spirals.
        let mut current_pos = points[0].position;
        let mut current_heading = tangent_headings[0];

        for i in 0..n - 1 {
            // Compute the target point for this line segment: the tangent point
            // on the incoming side of the next turn, or the final control point.
            let target = if let Some(ref turn) = turns[i + 1] {
                // Tangent point = control point minus tangent_length along incoming dir
                points[i + 1].position - tangents[i] * turn.tangent_length
            } else {
                points[i + 1].position
            };

            // Line length = projection of (target - current_pos) onto heading direction
            let heading_dir = Vec2::new(current_heading.cos(), current_heading.sin());
            let available = (target - current_pos).dot(heading_dir).max(0.0);

            // Emit line segment (if there's room)
            if available > 1e-4 {
                segments.push(Segment::Line { length: available });
                origins.push(current_pos);
                headings_out.push(current_heading);
                s_offsets.push(current_s);
                current_s += available;

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

                // Snap heading to exact outgoing tangent direction to prevent drift.
                // Position flows naturally from the geometry to avoid discontinuities.
                if i + 1 < n - 1 {
                    current_heading = tangent_headings[i + 1];
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

        // End should be near last point
        let end = rl.evaluate(rl.total_length);
        assert!((end.position - points[2].position).length() < 1.0);
    }

    #[test]
    fn fit_four_points_s_curve() {
        // S-curve: right then left
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
                turn_radius: 20.0,
                spiral_length: 5.0,
            },
            ControlPoint {
                position: Vec2::new(100.0, 50.0),
                turn_radius: 0.0,
                spiral_length: 0.0,
            },
        ];
        let rl = ReferenceLine::fit(&points).unwrap();
        assert!(rl.total_length > 0.0);

        let start = rl.evaluate(0.0);
        assert!((start.position - points[0].position).length() < 0.01);

        let end = rl.evaluate(rl.total_length);
        assert!(
            (end.position - points[3].position).length() < 1.0,
            "end {:?} far from target {:?}, dist={}",
            end.position,
            points[3].position,
            (end.position - points[3].position).length()
        );
    }

    #[test]
    fn tessellation_no_extreme_values() {
        // Verify that tessellation at 1m intervals produces no discontinuities
        let points = vec![
            ControlPoint {
                position: Vec2::new(-20.0, 5.0),
                turn_radius: 20.0,
                spiral_length: 5.0,
            },
            ControlPoint {
                position: Vec2::new(10.0, 3.0),
                turn_radius: 20.0,
                spiral_length: 5.0,
            },
            ControlPoint {
                position: Vec2::new(15.0, -25.0),
                turn_radius: 20.0,
                spiral_length: 5.0,
            },
        ];
        let rl = ReferenceLine::fit(&points).unwrap();

        let n_samples = (rl.total_length / 1.0).ceil() as usize + 1;
        let mut prev_pos: Option<Vec2> = None;
        for i in 0..n_samples {
            let s = (i as f32).min(rl.total_length);
            let pose = rl.evaluate(s);
            assert!(
                !pose.position.x.is_nan() && !pose.position.y.is_nan(),
                "NaN at s={}",
                s
            );
            assert!(
                pose.position.length() < 500.0,
                "extreme pos at s={}: {:?}",
                s,
                pose.position
            );
            // No large jumps between consecutive 1m samples
            if let Some(prev) = prev_pos {
                let jump = pose.position - prev;
                assert!(
                    jump.length() < 2.0,
                    "discontinuity at s={}: jump={}",
                    s,
                    jump.length()
                );
            }
            prev_pos = Some(pose.position);
        }
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
