use glam::{Mat4, Vec2};

/// 2D orthographic camera with pan and zoom.
#[derive(Clone, Debug)]
pub struct Camera2D {
    /// World-space position of the camera center.
    pub position: Vec2,
    /// Zoom level. Higher = more zoomed in (world units per screen unit decrease).
    pub zoom: f32,
}

impl Default for Camera2D {
    fn default() -> Self {
        Self {
            position: Vec2::ZERO,
            zoom: 1.0,
        }
    }
}

impl Camera2D {
    /// Compute the view-projection matrix for an orthographic projection.
    /// `aspect` = width / height.
    /// The projection maps `[-half_w, half_w] × [-half_h, half_h]` world units
    /// (centered on `position`) to clip space `[-1, 1]`.
    pub fn view_projection(&self, aspect: f32) -> Mat4 {
        let half_h = 1.0 / self.zoom;
        let half_w = half_h * aspect;

        let left = self.position.x - half_w;
        let right = self.position.x + half_w;
        let bottom = self.position.y - half_h;
        let top = self.position.y + half_h;

        Mat4::orthographic_rh(left, right, bottom, top, -1.0, 1.0)
    }

    /// The inverse view-projection: maps NDC → world coordinates.
    pub fn inverse_view_projection(&self, aspect: f32) -> Mat4 {
        self.view_projection(aspect).inverse()
    }

    /// Convert a screen-space pixel position to world coordinates.
    /// `screen_pos` is in pixels from the top-left corner.
    /// `window_size` is (width, height) in pixels.
    pub fn screen_to_world(&self, screen_pos: Vec2, window_size: Vec2) -> Vec2 {
        // Screen [0, w] × [0, h] → NDC [-1, 1] × [-1, 1] (y flipped)
        let ndc_x = (screen_pos.x / window_size.x) * 2.0 - 1.0;
        let ndc_y = -((screen_pos.y / window_size.y) * 2.0 - 1.0);

        let aspect = window_size.x / window_size.y;
        let half_h = 1.0 / self.zoom;
        let half_w = half_h * aspect;

        Vec2::new(
            self.position.x + ndc_x * half_w,
            self.position.y + ndc_y * half_h,
        )
    }

    /// Pan the camera by a screen-space pixel delta.
    /// Converts the pixel movement to world units based on current zoom and window size.
    pub fn pan_by_pixels(&mut self, dx: f64, dy: f64, window_width: u32, window_height: u32) {
        let half_h = 1.0 / self.zoom;
        let aspect = window_width as f32 / window_height as f32;
        let half_w = half_h * aspect;

        // Pixels to world: each pixel spans (2*half_w / window_width) world units
        let world_dx = -(dx as f32) * (2.0 * half_w) / window_width as f32;
        let world_dy = (dy as f32) * (2.0 * half_h) / window_height as f32;

        self.position.x += world_dx;
        self.position.y += world_dy;
    }

    /// Zoom toward/away from the cursor world position.
    /// `scroll_delta` > 0 means zoom in; < 0 means zoom out.
    /// `cursor_world` is the world position under the mouse cursor.
    pub fn zoom_at(&mut self, scroll_delta: f32, cursor_world: Vec2) {
        let factor = (scroll_delta * 0.1).exp();
        let new_zoom = (self.zoom * factor).clamp(0.001, 1000.0);

        // Adjust position so the world point under the cursor stays fixed.
        // Before zoom: cursor_world = position + ndc * (1/zoom) * ...
        // We want the same cursor_world to map to the same screen pixel after zoom.
        let zoom_ratio = 1.0 - self.zoom / new_zoom;
        self.position += (cursor_world - self.position) * zoom_ratio;
        self.zoom = new_zoom;
    }
}
