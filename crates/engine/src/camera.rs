use glam::{Mat4, Vec2, Vec4};

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
            zoom: 0.25,
        }
    }
}

impl Camera2D {
    /// Compute the view-projection matrix for an orthographic projection.
    /// `aspect` = width / height.
    ///
    /// Uses standard bottom < top convention. The vertex shader Y-flip is
    /// handled by naga's `ADJUST_COORDINATE_SPACE` flag.
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
    ///
    /// Note: naga's ADJUST_COORDINATE_SPACE flips Y in the vertex shader but
    /// NOT in compute shaders. The grid compute shader manually negates ndc_y.
    /// Here we also negate ndc_y to match the screen convention (Y=0 at top).
    pub fn screen_to_world(&self, screen_pos: Vec2, window_size: Vec2) -> Vec2 {
        let ndc_x = (screen_pos.x / window_size.x) * 2.0 - 1.0;
        let ndc_y = -((screen_pos.y / window_size.y) * 2.0 - 1.0);

        let aspect = window_size.x / window_size.y;
        let inv_vp = self.inverse_view_projection(aspect);
        let world4 = inv_vp * Vec4::new(ndc_x, ndc_y, 0.0, 1.0);

        Vec2::new(world4.x, world4.y)
    }

    /// Pan the camera by a screen-space pixel delta (grab-pan: content follows cursor).
    pub fn pan_by_pixels(&mut self, dx: f64, dy: f64, window_width: u32, window_height: u32) {
        let half_h = 1.0 / self.zoom;
        let aspect = window_width as f32 / window_height as f32;
        let half_w = half_h * aspect;

        // Grab-pan: drag right → camera moves left, drag down → camera moves up.
        // screen_to_world negates ndc_y, so world_y = cam_y - ndc_y * half_h.
        // For the cursor-under-point invariant:
        //   world_dx = -dx * 2*half_w / w
        //   world_dy = +dy * 2*half_h / h  (because of the Y negate in screen_to_world)
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

        let zoom_ratio = 1.0 - self.zoom / new_zoom;
        self.position += (cursor_world - self.position) * zoom_ratio;
        self.zoom = new_zoom;
    }
}
