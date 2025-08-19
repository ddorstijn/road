use const_for::const_for;
use glam::{Quat, Vec2, quat};

pub(crate) const ROTATIONS: [Quat; 4] = [
    Quat::IDENTITY,
    quat(0.0, 0.0, 0.70710677, -0.70710677), // 270 degrees
    quat(0.0, 0.0, 0.70710677, 0.70710677),  // 90 degrees
    quat(0.0, 0.0, 1.0, 0.0),                // 180 degrees
];

pub(crate) const SCALE_OFFSET: usize = 5;
pub(crate) const N_LEVELS: usize = 10;
pub(crate) const N_TILES: usize = N_LEVELS * 16; // 4x4 tiles per level
pub(crate) const N_FILLERS: usize = N_LEVELS; // 1 filler per level
pub(crate) const N_TRIMS: usize = N_LEVELS - 1; // no trim for finest
pub(crate) const N_SEAMS: usize = N_LEVELS - 1; // no seam for finest
pub(crate) const N_CROSS: usize = 1; // 1 cross at camera position

// Const generated meshes
pub const TILE_RES: usize = 64;
const PATCH_RES: usize = TILE_RES + 1;
const CLIP_RES: usize = TILE_RES * 4 + 1;
const CLIP_VERT_RES: usize = CLIP_RES + 1;

const V_TILE: usize = PATCH_RES * PATCH_RES;
const I_TILE: usize = 6 * (TILE_RES * TILE_RES);

const V_FILL: usize = 8 * PATCH_RES;
const I_FILL: usize = 24 * TILE_RES;

const V_TRIM: usize = (CLIP_VERT_RES * 2 + 1) * 2;
const I_TRIM: usize = (CLIP_VERT_RES * 2 - 1) * 6;

const V_CROSS: usize = 8 * PATCH_RES;
const I_CROSS: usize = 24 * TILE_RES + 6;

const V_SEAM: usize = 4 * CLIP_VERT_RES;
const I_SEAM: usize = 6 * CLIP_VERT_RES;

// TILE always has the most vertices and indices. Means we are wasting some space for the others, but makes implementation much easier on our side.
const V_MAX: usize = V_TILE;
const I_MAX: usize = I_TILE;

#[derive(Debug)]
pub struct Mesh2d {
    pub label: &'static str,
    pub positions: [Vec2; V_MAX],
    pub indices: [u32; I_MAX],
    pub vertex_count: usize,
    pub index_count: usize,
    pub instance_count: usize,
}

const fn generate_tile_mesh() -> Mesh2d {
    let mut vertices = [Vec2::ZERO; V_MAX];
    const_for!(v in 0..V_TILE => {
        let x = (v % PATCH_RES) as f32;
        let y = (v / PATCH_RES) as f32;
        vertices[v] = Vec2::new(x, y);
    });

    let mut indices = [0u32; I_MAX];
    let mut idx = 0;
    const_for!(y in 0..TILE_RES => {
        const_for!(x in 0..TILE_RES => {
            let base = (y * PATCH_RES + x) as u32;
            let right = base + 1;
            let up = base + PATCH_RES as u32;
            let up_right = up + 1;

            // [base, up_right, up] and [base, right, up_right]
            let quad_indices = [base, up_right, up, base, right, up_right];
            indices[idx..idx + 6].copy_from_slice(&quad_indices);
            idx += 6;
        });
    });

    Mesh2d {
        label: "Tile",
        positions: vertices,
        indices,
        vertex_count: V_TILE,
        index_count: I_TILE,
        instance_count: N_TILES,
    }
}

const fn generate_filler_mesh() -> Mesh2d {
    let mut vertices = [Vec2::ZERO; V_MAX];
    let mut v = 0;
    let offset = TILE_RES as f32;

    // A single loop generates vertices for all 4 arms
    const_for!(arm in 0..4 => {
        const_for!(i in 0..PATCH_RES => {
            let i_f = i as f32;
            vertices[v..v + 2].copy_from_slice(&match arm {
                0 => {
                    // +X arm
                    let x = offset + i_f + 1.0;
                    [Vec2::new(x, 0.0), Vec2::new(x, 1.0)]
                }
                1 => {
                    // +Y arm
                    let y = offset + i_f + 1.0;
                    [Vec2::new(1.0, y), Vec2::new(0.0, y)]
                }
                2 => {
                    // -X arm
                    let x = -(offset + i_f);
                    [Vec2::new(x, 1.0), Vec2::new(x, 0.0)]
                }
                _ => {
                    // -Y arm
                    let y = -(offset + i_f);
                    [Vec2::new(0.0, y), Vec2::new(1.0, y)]
                }
            });
            v += 2;
        });
    });

    let mut indices = [0u32; I_MAX];
    let mut idx = 0;
    const_for!(a in 0..(TILE_RES * 4) => {
        let arm = a / TILE_RES;
        let local = a % TILE_RES;
        let arm_start = (arm * PATCH_RES * 2) as u32;

        let bl = arm_start + (local * 2) as u32;
        let br = bl + 1;
        let tl = bl + 2;
        let tr = tl + 1;

        // Apply correct winding order based on arm orientation
        let quad_indices = if arm % 2 == 0 {
            // Horizontal arms: [br,bl,tr], [bl,tl,tr]
            [br, bl, tr, bl, tl, tr]
        } else {
            // Vertical arms: [br,bl,tl], [br,tl,tr] - FIXED
            [br, bl, tl, br, tl, tr]
        };
        indices[idx..idx + 6].copy_from_slice(&quad_indices);
        idx += 6;
    });

    Mesh2d {
        label: "Filler",
        positions: vertices,
        indices,
        vertex_count: V_FILL,
        index_count: I_FILL,
        instance_count: N_FILLERS,
    }
}

const fn generate_trim_mesh() -> Mesh2d {
    let mut vertices = [Vec2::ZERO; V_MAX];
    let mut indices = [0u32; I_MAX];

    // precompute half the total extent (to center at origin)
    let extent_f = CLIP_VERT_RES as f32 + 1.0;
    let half = 0.5 * extent_f;

    // 1) build the L shape vertices (vertical then horizontal), already offset
    let mut v = 0;
    // vertical bar: from y = +R down to y = −(half−1)
    const_for!(i in 0..CLIP_VERT_RES + 1 => {
        let y = CLIP_VERT_RES as f32 - i as f32 - half;
        vertices[v] = Vec2::new(0.0 - half, y);
        vertices[v + 1] = Vec2::new(1.0 - half, y);
        v += 2;
    });
    // mark where the horizontal strip starts
    let start_h = v as u32;

    // horizontal bar: from x = 1−half up to x = R−half
    const_for!(i in 0..CLIP_VERT_RES => {
        let x = (i as f32 + 1.0) - half;
        vertices[v] = Vec2::new(x, 0.0 - half);
        vertices[v + 1] = Vec2::new(x, 1.0 - half);
        v += 2;
    });

    // 2) build indices for the two strips of quads (6 idx per segment)

    // vertical strip
    let mut idx = 0;
    const_for!(i in 0..CLIP_VERT_RES => {
        let base = (i as u32) * 2;
        // [br, bl, next_bl], [next_br, br, next_bl]
        indices[idx + 0] = base + 1;
        indices[idx + 1] = base + 0;
        indices[idx + 2] = base + 2;
        indices[idx + 3] = base + 3;
        indices[idx + 4] = base + 1;
        indices[idx + 5] = base + 2;
        idx += 6;
    });

    // horizontal strip
    const_for!(i in 0..(CLIP_VERT_RES - 1) => {
        let base = start_h + (i as u32) * 2;
        indices[idx + 0] = base + 1;
        indices[idx + 1] = base + 0;
        indices[idx + 2] = base + 2;
        indices[idx + 3] = base + 3;
        indices[idx + 4] = base + 1;
        indices[idx + 5] = base + 2;
        idx += 6;
    });

    Mesh2d {
        label: "Trim",
        positions: vertices,
        indices,
        vertex_count: V_TRIM,
        index_count: I_TRIM,
        instance_count: N_TRIMS,
    }
}

const fn generate_cross_mesh() -> Mesh2d {
    let mut vertices = [Vec2::ZERO; V_MAX];
    let mut indices = [0u32; I_MAX];

    // 1) horizontal bar vertices
    let tile_f = TILE_RES as f32;
    let mut v = 0;
    const_for!(i in 0..(PATCH_RES * 2) => {
        let x = i as f32 - tile_f;
        vertices[v] = Vec2::new(x, 0.0);
        vertices[v + 1] = Vec2::new(x, 1.0);
        v += 2;
    });

    // remember where the vertical bar starts
    let vert_base = v as u32;

    // 2) vertical bar vertices
    const_for!(i in 0..(PATCH_RES * 2) => {
        let y = i as f32 - tile_f;
        vertices[v] = Vec2::new(0.0, y);
        vertices[v + 1] = Vec2::new(1.0, y);
        v += 2;
    });

    // 3) horizontal‐strip indices
    let mut idx = 0;
    const_for!(i in 0..TILE_RES * 2 + 1 => {
        let bl = (i * 2) as u32;
        let br = bl + 1;
        let tl = bl + 2;
        let tr = br + 2;
        indices[idx + 0] = br;
        indices[idx + 1] = bl;
        indices[idx + 2] = tr;
        indices[idx + 3] = bl;
        indices[idx + 4] = tl;
        indices[idx + 5] = tr;
        idx += 6;
    });

    // 4) vertical‐strip indices (skip the center line at i == TILE_RES)
    const_for!(i in 0..TILE_RES * 2 + 1 => {
        if i != TILE_RES {
            let bl = (i * 2) as u32;
            let br = bl + 1;
            let tl = bl + 2;
            let tr = br + 2;
            indices[idx + 0] = vert_base + br;
            indices[idx + 1] = vert_base + tr;
            indices[idx + 2] = vert_base + bl;
            indices[idx + 3] = vert_base + bl;
            indices[idx + 4] = vert_base + tr;
            indices[idx + 5] = vert_base + tl;
            idx += 6;
        }
    });

    Mesh2d {
        label: "Cross",
        positions: vertices,
        indices,
        vertex_count: V_CROSS,
        index_count: I_CROSS,
        instance_count: N_CROSS,
    }
}

const fn generate_seam_mesh() -> Mesh2d {
    let mut vertices = [Vec2::ZERO; V_MAX];
    let mut indices = [0u32; I_MAX];

    // 1) ring of CLIP_VERT_RES verts on each side
    let res_f = CLIP_VERT_RES as f32;
    const_for!(i in 0..CLIP_VERT_RES => {
        let f = i as f32;
        vertices[i] = Vec2::new(f, 0.0);
        vertices[CLIP_VERT_RES + i] = Vec2::new(res_f, f);
        vertices[CLIP_VERT_RES * 2 + i] = Vec2::new(res_f - f, res_f);
        vertices[CLIP_VERT_RES * 3 + i] = Vec2::new(0.0, res_f - f);
    });

    // 2) triangle‐strip indices, wrapping via `% V_SEAM`
    let mut idx = 0;
    let vcount = V_SEAM as u32;
    const_for!(pair in 0..(V_SEAM / 2) => {
        let j = (pair * 2) as u32;
        indices[idx + 0] = j + 1;
        indices[idx + 1] = j;
        indices[idx + 2] = (j + 2) % vcount;
        idx += 3;
    });

    Mesh2d {
        label: "Seam",
        positions: vertices,
        indices,
        vertex_count: V_SEAM,
        index_count: I_SEAM,
        instance_count: N_SEAMS,
    }
}

pub const TILE_MESH: Mesh2d = generate_tile_mesh();
pub const FILLER_MESH: Mesh2d = generate_filler_mesh();
pub const TRIM_MESH: Mesh2d = generate_trim_mesh();
pub const CROSS_MESH: Mesh2d = generate_cross_mesh();
pub const SEAM_MESH: Mesh2d = generate_seam_mesh();
