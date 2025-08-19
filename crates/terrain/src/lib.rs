#![feature(const_index)]
#![feature(const_trait_impl)]

use std::path::PathBuf;

use crate::consts::{
    CROSS_MESH, FILLER_MESH, Mesh2d, N_LEVELS, ROTATIONS, SCALE_OFFSET, SEAM_MESH, TILE_MESH,
    TILE_RES, TRIM_MESH,
};
use glam::{Mat4, Quat, Vec2, Vec3Swizzles};
use wre_camera::Camera;

mod consts;

pub struct TerrainComponent {
    pub instances: Vec<InstanceData>,
    pub positions: Vec<Vec2>,
    pub indices: Vec<u32>,
}

impl From<Mesh2d> for TerrainComponent {
    fn from(mesh: Mesh2d) -> Self {
        Self {
            instances: Vec::new(),
            positions: mesh.positions.to_vec(),
            indices: mesh.indices.to_vec(),
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum TerrainError {
    #[error("Terrain [IO] error:")]
    TempError,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct InstanceData {
    transform: Mat4,
}

// In your main render state or engine structure
pub struct Terrain {
    #[allow(unused)]
    heightmap: PathBuf,
    tile: TerrainComponent,
    cross: TerrainComponent,
    fill: TerrainComponent,
    trim: TerrainComponent,
    seam: TerrainComponent,
}

impl Terrain {
    pub fn new(heightmap: PathBuf) -> Result<Self, TerrainError> {
        let tile = TerrainComponent::from(TILE_MESH);
        let cross = TerrainComponent::from(CROSS_MESH);
        let fill = TerrainComponent::from(FILLER_MESH);
        let trim = TerrainComponent::from(TRIM_MESH);
        let seam = TerrainComponent::from(SEAM_MESH);

        Ok(Self {
            heightmap,
            tile,
            cross,
            fill,
            trim,
            seam,
        })
    }

    pub fn update(&mut self, camera: &Camera) {
        // We'll accumulate instance data here
        self.tile.instances = Vec::new();
        self.fill.instances = Vec::new();
        self.trim.instances = Vec::new();
        self.cross.instances = Vec::new();
        self.seam.instances = Vec::new();

        let camera_position = camera.eye.xy();

        // The main 4×4 tile ring & filler/trim/seam per level
        for level in 0..N_LEVELS {
            let scale = (1u32 << level + SCALE_OFFSET) as f32;
            let tile_size = Vec2::splat((TILE_RES << level + SCALE_OFFSET) as f32);

            let v_scale = Vec2::splat(scale).extend(1.0);
            // snapped camera for this LOD
            let snapped_pos = (camera_position / scale).floor() * scale;
            // bottom‐left corner of 4×4 grid
            let base = snapped_pos - tile_size * 2.0;

            // --- Cross ---
            if level == 0 {
                let transform = Mat4::from_scale_rotation_translation(
                    v_scale,
                    Quat::IDENTITY,
                    snapped_pos.extend(0.0),
                );
                self.cross.instances.push(InstanceData { transform });
            }

            // --- 4×4 Tiles (skip middle 2×2 if not finest) ---
            for x in 0..4 {
                for y in 0..4 {
                    if level != 0 && (matches!(x, 1 | 2)) && (matches!(y, 1 | 2)) {
                        continue;
                    }

                    let pos = Vec2::new(x as f32, y as f32);
                    let fill = Vec2::new(
                        if x >= 2 { 1.0 } else { 0.0 },
                        if y >= 2 { 1.0 } else { 0.0 },
                    ) * scale;

                    let bl = base + pos * tile_size + fill;
                    let transform = Mat4::from_scale_rotation_translation(
                        v_scale,
                        Quat::IDENTITY,
                        bl.extend(0.0),
                    );
                    self.tile.instances.push(InstanceData { transform });
                }
            }

            // --- Filler ring ---
            {
                let transform = Mat4::from_scale_rotation_translation(
                    v_scale,
                    Quat::IDENTITY,
                    snapped_pos.extend(0.0),
                );
                self.fill.instances.push(InstanceData { transform });
            }

            // Trim and seam are not generated for the finest level
            if level < N_LEVELS - 1 {
                let next_scale = scale * 2.0;
                let next_snap = (camera_position / next_scale).floor() * next_scale;

                // --- Seam ---
                let next_base =
                    next_snap - Vec2::splat((TILE_RES << (level + SCALE_OFFSET + 1)) as f32);
                let transform = Mat4::from_scale_rotation_translation(
                    v_scale,
                    Quat::IDENTITY,
                    next_base.extend(0.0),
                );

                self.seam.instances.push(InstanceData { transform });

                // --- Trim ---
                let d = camera_position - next_snap;
                let r = (if d.x < scale { 2 } else { 0 }) | (if d.y < scale { 1 } else { 0 });

                let center = snapped_pos + 0.5 * v_scale.xy();
                let transform = Mat4::from_scale_rotation_translation(
                    v_scale,
                    ROTATIONS[r],
                    center.extend(0.0),
                );

                self.trim.instances.push(InstanceData { transform });
            }
        }
    }
}
