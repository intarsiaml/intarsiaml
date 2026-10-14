//! Compiles kernels/metal/*.metal (AOT) into one .metallib embedded by `src/lib.rs` via `include_bytes!`.
//!
//! Needs the Xcode Metal toolchain: `xcrun -sdk macosx metal`. `xcrun -f metal` only resolves with
//! `DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer` on this fleet; an existing `DEVELOPER_DIR` wins, else
//! that path is used when it exists. Without a toolchain (or off macOS) an empty metallib is written and the build
//! warns; `MetalBackend::new` then reports the missing library at run time, so `cargo build --workspace` still works.

use std::path::{Path, PathBuf};
use std::process::Command;

const XCODE_DEVELOPER_DIR: &str = "/Applications/Xcode.app/Contents/Developer";
/// bfloat needs MSL 3.1.
const METAL_STD: &str = "-std=metal3.1";

fn xcrun(developer_dir: &Option<String>) -> Command {
    let mut c = Command::new("xcrun");
    if let Some(d) = developer_dir {
        c.env("DEVELOPER_DIR", d);
    }
    c
}

fn run(mut c: Command, what: &str) -> Result<(), String> {
    let out = c.output().map_err(|e| format!("{what}: {e}"))?;
    if !out.status.success() {
        return Err(format!(
            "{what} failed:\n{}{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        ));
    }
    Ok(())
}

fn build(shaders: &[PathBuf], out_dir: &Path, developer_dir: &Option<String>) -> Result<(), String> {
    let mut airs = Vec::new();
    for s in shaders {
        let air = out_dir.join(s.file_stem().unwrap()).with_extension("air");
        let mut c = xcrun(developer_dir);
        c.args(["-sdk", "macosx", "metal", "-c", METAL_STD, "-O3", "-fno-fast-math", "-o"]).arg(&air).arg(s);
        run(c, &format!("metal -c {}", s.display()))?;
        airs.push(air);
    }
    let lib = out_dir.join("intarsia.metallib");
    let mut c = xcrun(developer_dir);
    c.args(["-sdk", "macosx", "metallib"]).args(&airs).arg("-o").arg(&lib);
    run(c, "metallib")
}

fn main() {
    println!("cargo:rustc-check-cfg=cfg(ik_metal)");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=../../kernels/metal");
    println!("cargo:rerun-if-env-changed=DEVELOPER_DIR");
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let lib = out_dir.join("intarsia.metallib");
    let _ = std::fs::write(&lib, b"");
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("macos") {
        return;
    }

    let shader_dir = Path::new(&std::env::var("CARGO_MANIFEST_DIR").unwrap()).join("../../kernels/metal");
    let mut shaders: Vec<PathBuf> = std::fs::read_dir(&shader_dir)
        .map(|rd| {
            rd.filter_map(|e| e.ok())
                .map(|e| e.path())
                .filter(|p| p.extension().is_some_and(|x| x == "metal"))
                .collect()
        })
        .unwrap_or_default();
    shaders.sort();
    for s in &shaders {
        println!("cargo:rerun-if-changed={}", s.display());
    }

    let developer_dir = std::env::var("DEVELOPER_DIR")
        .ok()
        .or_else(|| Path::new(XCODE_DEVELOPER_DIR).exists().then(|| XCODE_DEVELOPER_DIR.to_string()));
    match build(&shaders, &out_dir, &developer_dir) {
        Ok(_) => println!("cargo:rustc-cfg=ik_metal"),
        Err(e) => println!(
            "cargo:warning=intarsia-backend-metal: no metallib built ({e}); MetalBackend::new will fail at run time"
        ),
    }
}
