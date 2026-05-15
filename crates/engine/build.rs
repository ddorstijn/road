use std::path::PathBuf;
use std::process::Command;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let out_dir = PathBuf::from(std::env::var("OUT_DIR")?);
    let shader_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("assets/shaders");
    let shader_dir_str = shader_dir.to_str().unwrap();

    let shaders: &[(&str, &str, &str)] = &[
        ("egui", "vs_main", "SHADER_EGUI_VS"),
        ("egui", "fs_main", "SHADER_EGUI_FS"),
    ];

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
            cmd.args(["-O0", "-g"]);
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

        println!("cargo:rustc-env={}={}", env_name, out_path.display());
    }

    println!("cargo:rerun-if-changed={}", shader_dir_str);

    Ok(())
}
