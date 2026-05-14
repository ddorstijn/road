use std::path::PathBuf;
use std::process::Command;

/// Parse `crates/gpu-shared/src/types.rs` and generate an equivalent `gpu_shared.slang`.
/// This keeps the Rust types as the single source of truth.
fn generate_gpu_shared_slang(out_dir: &PathBuf) -> Result<(), Box<dyn std::error::Error>> {
    let types_rs = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../gpu-shared/src/types.rs");
    println!("cargo:rerun-if-changed={}", types_rs.display());

    let source = std::fs::read_to_string(&types_rs)?;
    let mut slang = String::from(
        "// AUTO-GENERATED from crates/gpu-shared/src/types.rs \u{2014} do not edit.\n\n",
    );

    let mut in_struct = false;

    for line in source.lines() {
        let trimmed = line.trim();

        if trimmed.starts_with("pub struct ") {
            let name = trimmed
                .strip_prefix("pub struct ")
                .unwrap()
                .trim_end_matches(" {")
                .trim();
            slang.push_str(&format!("struct {} {{\n", name));
            in_struct = true;
            continue;
        }

        if in_struct && trimmed == "}" {
            slang.push_str("};\n\n");
            in_struct = false;
            continue;
        }

        if in_struct && trimmed.starts_with("pub ") {
            let field = trimmed.strip_prefix("pub ").unwrap().trim_end_matches(',');
            if let Some((name, ty)) = field.split_once(": ") {
                let slang_ty = rust_type_to_slang(ty.trim());
                slang.push_str(&format!("    {} {};\n", slang_ty, name.trim()));
            }
            continue;
        }

        if in_struct && trimmed.starts_with("///") {
            let comment = trimmed.strip_prefix("///").unwrap_or("");
            slang.push_str(&format!("    //{}\n", comment));
        }
    }

    std::fs::write(out_dir.join("gpu_shared.slang"), slang)?;
    Ok(())
}

fn rust_type_to_slang(ty: &str) -> &str {
    match ty {
        "u32" => "uint",
        "i32" => "int",
        "f32" => "float",
        "[f32; 2]" => "float2",
        "[f32; 3]" => "float3",
        "[f32; 4]" => "float4",
        "[u32; 2]" => "uint2",
        "[u32; 3]" => "uint3",
        "[u32; 4]" => "uint4",
        _ => panic!(
            "Unsupported type in gpu-shared types.rs: `{ty}` \u{2014} add a mapping in build.rs"
        ),
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let out_dir = PathBuf::from(std::env::var("OUT_DIR")?);
    let shader_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../assets/shaders");
    let shader_dir_str = shader_dir.to_str().unwrap();

    // Generate gpu_shared.slang from Rust types (single source of truth)
    let generated_dir = shader_dir.join("generated");
    std::fs::create_dir_all(&generated_dir)?;
    generate_gpu_shared_slang(&generated_dir)?;

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
        ("road_draft", "vs_main", "SHADER_ROAD_DRAFT_VS"),
        ("road_draft", "fs_main", "SHADER_ROAD_DRAFT_FS"),
    ];

    // Use no optimization for dev builds, optimized for release.
    // Note: `-g` (SPIR-V debug info via SPV_KHR_non_semantic_info) is intentionally
    // omitted because AMD's shader compiler crashes on NonSemantic.Shader.DebugInfo
    // instructions. Use spirv-opt --strip-nonsemantic if debug info is needed on NVIDIA.
    let is_release = std::env::var("PROFILE").unwrap_or_default() == "release";

    for &(module_name, entry_name, env_name) in shaders {
        let source_path = shader_dir.join(format!("{}.slang", module_name));
        let out_path = out_dir.join(format!("{}_{}.spv", module_name, entry_name));

        let mut cmd = Command::new("slangc");
        cmd.arg(source_path.to_str().unwrap())
            .args(["-target", "spirv"])
            .args(["-entry", entry_name])
            .args(["-profile", "glsl_450"])
            .args(["-I", shader_dir_str]);

        if is_release {
            cmd.arg("-O3");
        } else {
            cmd.arg("-O0");
        }

        let output = cmd
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

        // Validate SPIR-V with spirv-val (catches issues that crash certain drivers)
        if let Ok(val_output) = Command::new("spirv-val")
            .arg(out_path.to_str().unwrap())
            .output()
        {
            if !val_output.status.success() {
                let stderr = String::from_utf8_lossy(&val_output.stderr);
                panic!(
                    "spirv-val failed for {}::{}\n{}",
                    module_name, entry_name, stderr
                );
            }
        }

        println!("cargo:rustc-env={}={}", env_name, out_path.display());
    }

    // Rerun if any shader source changes
    println!("cargo:rerun-if-changed={}", shader_dir_str);

    Ok(())
}
