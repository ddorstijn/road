use std::path::PathBuf;
use std::process::Command;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let out_dir = PathBuf::from(std::env::var("OUT_DIR")?);
    let shader_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../assets/shaders");
    let shader_dir_str = shader_dir.to_str().unwrap();

    // Each entry: (module_filename, entry_point_name, env_var_name)
    let shaders: &[(&str, &str, &str)] = &[
        ("grid", "grid_main", "SHADER_GRID"),
        ("road_line", "vs_main", "SHADER_ROAD_LINE_VS"),
        ("road_line", "fs_main", "SHADER_ROAD_LINE_FS"),
        ("sdf_generate", "sdf_generate_main", "SHADER_SDF_GENERATE"),
        ("road_render", "vs_main", "SHADER_ROAD_RENDER_VS"),
        ("road_render", "fs_main", "SHADER_ROAD_RENDER_FS"),
        ("frustum_cull", "car_cull_main", "SHADER_CAR_CULL"),
        ("tile_cull", "tile_cull_main", "SHADER_TILE_CULL"),
        ("car_render", "vs_main", "SHADER_CAR_RENDER_VS"),
        ("car_render", "fs_main", "SHADER_CAR_RENDER_FS"),
        (
            "traffic_sort_keys",
            "traffic_sort_keys_main",
            "SHADER_SORT_KEYS",
        ),
        (
            "traffic_sort_histogram",
            "traffic_sort_histogram_main",
            "SHADER_SORT_HISTOGRAM",
        ),
        (
            "traffic_sort_scan",
            "traffic_sort_scan_main",
            "SHADER_SORT_SCAN",
        ),
        (
            "traffic_sort_scatter",
            "traffic_sort_scatter_main",
            "SHADER_SORT_SCATTER",
        ),
        ("traffic_idm", "traffic_idm_main", "SHADER_TRAFFIC_IDM"),
        (
            "traffic_lane_change",
            "traffic_lane_change_main",
            "SHADER_TRAFFIC_LANE_CHANGE",
        ),
    ];

    for &(module_name, entry_name, env_name) in shaders {
        let source_path = shader_dir.join(format!("{}.slang", module_name));
        let out_path = out_dir.join(format!("{}_{}.spv", module_name, entry_name));

        let output = Command::new("slangc")
            .arg(source_path.to_str().unwrap())
            .args(["-target", "spirv"])
            .args(["-entry", entry_name])
            .args(["-profile", "glsl_450"])
            .args(["-I", shader_dir_str])
            .args(["-O3"])
            .args(["-o", out_path.to_str().unwrap()])
            .output()
            .expect("Failed to run slangc. Is the Vulkan SDK installed and slangc in PATH?");

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            let stdout = String::from_utf8_lossy(&output.stdout);
            panic!(
                "slangc failed for {}::{}\nstdout: {}\nstderr: {}",
                module_name, entry_name, stdout, stderr
            );
        }

        println!("cargo:rustc-env={}={}", env_name, out_path.display());
    }

    // Rerun if any shader source changes
    println!("cargo:rerun-if-changed={}", shader_dir_str);

    Ok(())
}
