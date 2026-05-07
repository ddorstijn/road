// Shared road evaluation functions — used by SDF generation and car rendering.
// Replicates the math from crates/road/src/primitives.rs in WGSL.

// ---------------------------------------------------------------------------
// Segment evaluation: position and heading at parameter s (segment-local)
// ---------------------------------------------------------------------------

fn eval_line(s: f32) -> vec2<f32> {
    return vec2<f32>(s, 0.0);
}

fn heading_line() -> f32 {
    return 0.0;
}

fn eval_arc(s: f32, curvature: f32) -> vec2<f32> {
    if abs(curvature) < 1e-9 {
        return vec2<f32>(s, 0.0);
    }
    let r = 1.0 / curvature;
    let theta = s * curvature;
    return vec2<f32>(r * sin(theta), r * (1.0 - cos(theta)));
}

fn heading_arc(s: f32, curvature: f32) -> f32 {
    return s * curvature;
}

// Spiral (clothoid) evaluation via Simpson's rule (16 intervals).
fn eval_spiral(s: f32, k_start: f32, k_end: f32, total_length: f32) -> vec2<f32> {
    let dk = select(0.0, (k_end - k_start) / total_length, total_length > 1e-9);

    let n = 16u;
    let h = s / f32(n);
    var sum_x = 0.0;
    var sum_y = 0.0;

    for (var i = 0u; i <= n; i++) {
        let t = f32(i) * h;
        let theta = k_start * t + 0.5 * dk * t * t;
        var w = 1.0;
        if i == 0u || i == n {
            w = 1.0;
        } else if i % 2u == 1u {
            w = 4.0;
        } else {
            w = 2.0;
        }
        sum_x += w * cos(theta);
        sum_y += w * sin(theta);
    }

    return vec2<f32>(sum_x * h / 3.0, sum_y * h / 3.0);
}

fn heading_spiral(s: f32, k_start: f32, k_end: f32, total_length: f32) -> f32 {
    let dk = select(0.0, (k_end - k_start) / total_length, total_length > 1e-9);
    return k_start * s + 0.5 * dk * s * s;
}

// ---------------------------------------------------------------------------
// Evaluate a segment at local parameter s → (position, heading)
// ---------------------------------------------------------------------------

struct PoseResult {
    position: vec2<f32>,
    heading: f32,
}

fn eval_segment(seg_type: u32, s: f32, k_start: f32, k_end: f32, seg_length: f32) -> PoseResult {
    var result: PoseResult;
    if seg_type == 0u {
        // Line
        result.position = eval_line(s);
        result.heading = heading_line();
    } else if seg_type == 1u {
        // Arc
        result.position = eval_arc(s, k_start);
        result.heading = heading_arc(s, k_start);
    } else {
        // Spiral
        result.position = eval_spiral(s, k_start, k_end, seg_length);
        result.heading = heading_spiral(s, k_start, k_end, seg_length);
    }
    return result;
}

// Transform from segment-local coordinates to world coordinates
fn segment_local_to_world(local_pos: vec2<f32>, origin: vec2<f32>, heading: f32) -> vec2<f32> {
    let c = cos(heading);
    let s = sin(heading);
    return origin + vec2<f32>(
        c * local_pos.x - s * local_pos.y,
        s * local_pos.x + c * local_pos.y,
    );
}

fn world_to_segment_local(world_pos: vec2<f32>, origin: vec2<f32>, heading: f32) -> vec2<f32> {
    let c = cos(heading);
    let s = sin(heading);
    let d = world_pos - origin;
    return vec2<f32>(
        c * d.x + s * d.y,
        -s * d.x + c * d.y,
    );
}

// ---------------------------------------------------------------------------
// Closest point on a segment (segment-local coordinates)
// ---------------------------------------------------------------------------

struct ClosestResult {
    s: f32,
    signed_dist: f32,
}

fn closest_point_line(point: vec2<f32>, seg_len: f32) -> ClosestResult {
    var result: ClosestResult;
    result.s = clamp(point.x, 0.0, seg_len);
    let closest = vec2<f32>(result.s, 0.0);
    let diff = point - closest;
    let dist = length(diff);
    // Use y-sign: positive = left of line direction
    result.signed_dist = select(-dist, dist, point.y >= 0.0);
    return result;
}

fn closest_point_arc(point: vec2<f32>, seg_length: f32, curvature: f32) -> ClosestResult {
    var result: ClosestResult;
    if abs(curvature) < 1e-9 {
        return closest_point_line(point, seg_length);
    }

    let r = 1.0 / curvature;
    let center = vec2<f32>(0.0, r);
    let to_point = point - center;

    var angle = atan2(-to_point.x, to_point.y * sign(curvature));
    if angle < 0.0 {
        angle += 6.283185307;  // 2 * PI
    }

    let max_angle = abs(seg_length * curvature);
    let clamped_angle = clamp(angle, 0.0, max_angle);
    result.s = clamped_angle / abs(curvature);

    // Evaluate arc at clamped s for the actual closest point
    let closest_pt = eval_arc(result.s, curvature);
    let diff = point - closest_pt;
    let dist = length(diff);

    // Sign via cross product of tangent and diff
    let hdg = heading_arc(result.s, curvature);
    let tangent = vec2<f32>(cos(hdg), sin(hdg));
    let cross = tangent.x * diff.y - tangent.y * diff.x;
    result.signed_dist = select(-dist, dist, cross >= 0.0);
    return result;
}

fn closest_point_spiral(point: vec2<f32>, seg_length: f32, k_start: f32, k_end: f32) -> ClosestResult {
    var result: ClosestResult;

    // Coarse search: 8 samples
    var best_s = 0.0;
    var best_dist_sq = 1e30;
    for (var i = 0u; i <= 8u; i++) {
        let t = seg_length * f32(i) / 8.0;
        let pos = eval_spiral(t, k_start, k_end, seg_length);
        let d = dot(pos - point, pos - point);
        if d < best_dist_sq {
            best_dist_sq = d;
            best_s = t;
        }
    }

    // Newton-Raphson refinement: 4 iterations
    var s = best_s;
    let dk = select(0.0, (k_end - k_start) / seg_length, seg_length > 1e-9);
    for (var iter = 0u; iter < 4u; iter++) {
        let pos = eval_spiral(s, k_start, k_end, seg_length);
        let diff = pos - point;
        let hdg = heading_spiral(s, k_start, k_end, seg_length);
        let tangent = vec2<f32>(cos(hdg), sin(hdg));
        let f_val = dot(diff, tangent);
        let k_at_s = k_start + dk * s;
        let normal = vec2<f32>(-sin(hdg), cos(hdg));
        let f_prime = 1.0 + dot(diff, normal) * k_at_s;
        if abs(f_prime) > 1e-9 {
            s -= f_val / f_prime;
        }
        s = clamp(s, 0.0, seg_length);
    }

    let pose = eval_spiral(s, k_start, k_end, seg_length);
    let diff = point - pose;
    let dist = length(diff);
    let hdg = heading_spiral(s, k_start, k_end, seg_length);
    let tangent = vec2<f32>(cos(hdg), sin(hdg));
    let cross = tangent.x * diff.y - tangent.y * diff.x;
    result.s = s;
    // Sign via cross product of tangent and diff
    result.signed_dist = select(-dist, dist, cross >= 0.0);
    return result;
}

fn closest_point_on_segment(seg_type: u32, point: vec2<f32>, seg_length: f32, k_start: f32, k_end: f32) -> ClosestResult {
    if seg_type == 0u {
        return closest_point_line(point, seg_length);
    } else if seg_type == 1u {
        return closest_point_arc(point, seg_length, k_start);
    } else {
        return closest_point_spiral(point, seg_length, k_start, k_end);
    }
}
