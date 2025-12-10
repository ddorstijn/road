use engine::component::{AutoCommandBufferBuilder, GameComponent, PrimaryAutoCommandBuffer, Ui};
use glam::Vec2;

#[derive(Debug)]
pub enum GeometryType {
    Line {
        length: f32,
    },
    Spiral {
        length: f32,
        curv_start: f32,
        curv_end: f32,
    },
    Arc {
        length: f32,
        curvature: f32,
    },
}

#[derive(Debug)]
pub struct RoadSegment {
    pub start_x: f32,
    pub start_y: f32,
    pub start_hdg: f32,
    pub geometry: GeometryType,
}

// Helper to rotate a vector (x, y) by angle `a`
fn rotate_vector(x: f32, y: f32, a: f32) -> (f32, f32) {
    let c = a.cos();
    let s = a.sin();
    (x * c - y * s, x * s + y * c)
}

// 3rd Order Taylor Series Expansion for Euler Spiral (Same as Shader)
// Returns local (x, y) end position assuming start is (0,0) and hdg is 0
fn integrate_spiral_local(length: f32, curv_start: f32, curv_end: f32) -> (f32, f32) {
    let c_dot = (curv_end - curv_start) / length;

    // If starting curvature is not zero, we must use the "Virtual Start" method
    // (Calculating the segment as a slice of a larger spiral starting at 0)
    let s_start = curv_start / c_dot;
    let s_end = s_start + length;

    // Helper closure for the standard polynomial
    let odr_spiral = |s: f32| -> (f32, f32) {
        let s2 = s * s;
        let s3 = s2 * s;
        let s4 = s2 * s2;
        let s5 = s4 * s;
        let a = 0.5 * c_dot;
        let a2 = a * a;

        let x = s - (a2 * s5) / 10.0;
        let y = (a * s3) / 3.0 - (a2 * a * s4 * s3) / 42.0;
        (x, y)
    };

    let (x0, y0) = odr_spiral(s_start);
    let (x1, y1) = odr_spiral(s_end);

    // The vector (x1-x0, y1-y0) is in the coordinate system of the infinite spiral.
    // We need to rotate it so the start point (x0, y0) aligns with our local tangent (0 heading).
    // Heading at s_start on infinite spiral:
    let theta_start = 0.5 * c_dot * s_start * s_start;

    // Rotate BACK by theta_start to align with x-axis
    rotate_vector(x1 - x0, y1 - y0, -theta_start)
}

pub fn generate_road_geometry(points: &[Vec2], radius: f32, spiral_len: f32) -> Vec<RoadSegment> {
    let mut segments = Vec::new();
    let curvature = 1.0 / radius;

    if points.len() < 2 {
        return vec![];
    }

    // --- 1. CALCULATE CORNER DATA ---
    struct CornerData {
        tangent_dist: f32,
        turn_angle: f32,
        is_left: bool,
    }

    let mut corners = Vec::new();

    for i in 1..points.len() - 1 {
        let p_prev = points[i - 1];
        let p_curr = points[i];
        let p_next = points[i + 1];

        let v_in = (p_curr - p_prev).normalize();
        let v_out = (p_next - p_curr).normalize();

        let dot = v_in.dot(v_out).clamp(-1.0, 1.0);
        let turn_angle = dot.acos();
        let shift = (spiral_len * spiral_len) / (24.0 * radius);
        let tangent_dist = (radius + shift) * (turn_angle / 2.0).tan() + (spiral_len / 2.0);
        let is_left = v_in.perp_dot(v_out) > 0.0;

        corners.push(CornerData {
            tangent_dist,
            turn_angle,
            is_left,
        });
    }

    // --- 2. INITIALIZE CURSOR ---
    // Start at Vec2 0
    let mut curr_x = points[0].x;
    let mut curr_y = points[0].y;

    // Initial Heading is direction from P0 to P1
    let v_init = (points[1] - points[0]).normalize();
    let mut curr_hdg = v_init.y.atan2(v_init.x);

    // --- 3. GENERATE SEGMENTS ---
    for i in 0..points.len() - 1 {
        let p_start = points[i];
        let p_end = points[i + 1];
        let total_dist = (p_end - p_start).length();

        let trim_start = if i == 0 {
            0.0
        } else {
            corners[i - 1].tangent_dist
        };
        let trim_end = if i == points.len() - 2 {
            0.0
        } else {
            corners[i].tangent_dist
        };

        // A. LINE SEGMENT
        let line_len = total_dist - trim_start - trim_end;
        if line_len > 1e-3 {
            // 1. Push Segment
            segments.push(RoadSegment {
                start_x: curr_x,
                start_y: curr_y,
                start_hdg: curr_hdg,
                geometry: GeometryType::Line { length: line_len },
            });

            // 2. Update Cursor (Move Straight)
            curr_x += line_len * curr_hdg.cos();
            curr_y += line_len * curr_hdg.sin();
            // Heading does not change on a line
        }

        // B. CORNER SEGMENTS (Spiral -> Arc -> Spiral)
        if i < points.len() - 2 {
            let data = &corners[i];
            let k_target = if data.is_left { curvature } else { -curvature };

            // --- SPIRAL IN (0 -> k) ---
            segments.push(RoadSegment {
                start_x: curr_x,
                start_y: curr_y,
                start_hdg: curr_hdg,
                geometry: GeometryType::Spiral {
                    length: spiral_len,
                    curv_start: 0.0,
                    curv_end: k_target,
                },
            });

            // Update Cursor (Spiral Integration)
            let (dx, dy) = integrate_spiral_local(spiral_len, 0.0, k_target);
            let (rdx, rdy) = rotate_vector(dx, dy, curr_hdg);
            curr_x += rdx;
            curr_y += rdy;
            curr_hdg += (spiral_len * k_target) / 2.0; // Angle change = Area under curvature graph (triangle)

            // --- ARC ---
            let spiral_angle_consumed = (spiral_len * k_target.abs()) / 2.0;
            let arc_angle_required = data.turn_angle - (2.0 * spiral_angle_consumed);
            let arc_len = arc_angle_required / k_target.abs();

            if arc_len > 1e-3 {
                segments.push(RoadSegment {
                    start_x: curr_x,
                    start_y: curr_y,
                    start_hdg: curr_hdg,
                    geometry: GeometryType::Arc {
                        length: arc_len,
                        curvature: k_target,
                    },
                });

                // Update Cursor (Circular Motion)
                // Chord length = 2 * R * sin(theta/2) is simpler, but let's do standard integration style
                // Local Arc End: x = sin(theta)/k, y = (1-cos(theta))/k
                // (Note: This local formula assumes arc starts tangent to X-axis)
                let d_angle = arc_len * k_target;
                let local_x = d_angle.sin() / k_target;
                let local_y = (1.0 - d_angle.cos()) / k_target;

                let (rdx, rdy) = rotate_vector(local_x, local_y, curr_hdg);
                curr_x += rdx;
                curr_y += rdy;
                curr_hdg += d_angle;
            }

            // --- SPIRAL OUT (k -> 0) ---
            segments.push(RoadSegment {
                start_x: curr_x,
                start_y: curr_y,
                start_hdg: curr_hdg,
                geometry: GeometryType::Spiral {
                    length: spiral_len,
                    curv_start: k_target,
                    curv_end: 0.0,
                },
            });

            // Update Cursor
            let (dx, dy) = integrate_spiral_local(spiral_len, k_target, 0.0);
            let (rdx, rdy) = rotate_vector(dx, dy, curr_hdg);
            curr_x += rdx;
            curr_y += rdy;
            curr_hdg += (spiral_len * k_target) / 2.0;
        }
    }

    segments
}
