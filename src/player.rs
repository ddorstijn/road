use wre_input::{InputState, KeyCode};

const SPEED_MODIFIER: f32 = 50.0;

pub struct Player {
    pub speed: f32,
}

impl Player {
    pub fn new(speed: f32) -> Self {
        Self { speed }
    }

    pub fn update(&self, camera: &mut Camera, input: &InputState, delta_time: std::time::Duration) {
        let forward = Vec3::Y;
        let right = Vec3::X;
        let up = Vec3::Z;

        let modifier = if self.is_shift_pressed {
            SPEED_MODIFIER
        } else {
            1.0
        };

        if input.is_key_pressed(KeyCode::KeyW) {
            camera.target -= forward * self.speed * modifier * delta_time.as_secs_f32();
            camera.eye -= forward * self.speed * modifier * delta_time.as_secs_f32();
        }
        if input.is_key_pressed(KeyCode::KeyS) {
            camera.target += forward * self.speed * modifier * delta_time.as_secs_f32();
            camera.eye += forward * self.speed * modifier * delta_time.as_secs_f32();
        }
        if input.is_key_pressed(KeyCode::KeyA) {
            camera.target -= right * self.speed * modifier * delta_time.as_secs_f32();
            camera.eye -= right * self.speed * modifier * delta_time.as_secs_f32();
        }
        if input.is_key_pressed(KeyCode::KeyD) {
            camera.target += right * self.speed * modifier * delta_time.as_secs_f32();
            camera.eye += right * self.speed * modifier * delta_time.as_secs_f32();
        }

        if input.is_key_pressed(KeyCode::Space) {
            camera.target += up * self.speed * modifier * 0.75 * delta_time.as_secs_f32();
            camera.eye += up * self.speed * modifier * 0.75 * delta_time.as_secs_f32();
        }
        if input.is_key_pressed(KeyCode::ControlLeft) {
            camera.target -= up * self.speed * modifier * 0.75 * delta_time.as_secs_f32();
            camera.eye -= up * self.speed * modifier * 0.75 * delta_time.as_secs_f32();
        }
    }
}
