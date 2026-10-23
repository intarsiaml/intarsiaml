//! Compiles kernels/hip/*.hip with hipcc (AOT for one gfx arch, default gfx1201 = R9700) into a static
//! library and links it plus the HIP runtime. Without a ROCm toolchain (the Mac, CI) the crate still builds:
//! it compiles to a stub whose `create()` returns `Unsupported`, so `cargo check --workspace` works anywhere.
//!
//! Environment:
//!   ROCM_PATH / HIP_PATH   ROCm root (default: /opt/rocm, then hipcc on PATH)
//!   INTARSIA_HIP_ARCH      hipcc --offload-arch value, default `gfx1201` (the kernels need gfx12: wave32 + WMMA)
//!   INTARSIA_NO_HIP=1      force the stub

use std::path::{Path, PathBuf};
use std::process::Command;

/// Translation units compiled in CU mode (`-mcumode`); every other one keeps the WGP-mode default.
const CU_MODE_SOURCES: &[&str] = &["gdn.hip"];

fn find_hipcc() -> Option<PathBuf> {
    let mut roots: Vec<PathBuf> = Vec::new();
    for var in ["ROCM_PATH", "HIP_PATH"] {
        if let Some(v) = std::env::var_os(var) {
            roots.push(PathBuf::from(v));
        }
    }
    roots.push(PathBuf::from("/opt/rocm"));
    for r in roots {
        let p = r.join("bin").join("hipcc");
        if p.is_file() {
            return Some(p);
        }
    }
    if Command::new("hipcc").arg("--version").output().map(|o| o.status.success()).unwrap_or(false) {
        return Some(PathBuf::from("hipcc"));
    }
    None
}

/// Directories that may hold libamdhip64, derived from the hipcc location.
fn lib_dirs(hipcc: &Path) -> Vec<PathBuf> {
    let mut roots: Vec<PathBuf> = Vec::new();
    if let Some(r) = hipcc.canonicalize().ok().and_then(|p| p.parent().and_then(|b| b.parent()).map(Path::to_path_buf)) {
        roots.push(r);
    }
    if let Some(r) = hipcc.parent().and_then(|b| b.parent()) {
        roots.push(r.to_path_buf());
    }
    roots.push(PathBuf::from("/opt/rocm"));
    let mut out: Vec<PathBuf> = Vec::new();
    for root in roots {
        let d = root.join("lib");
        if d.join("libamdhip64.so").exists() && !out.contains(&d) {
            out.push(d);
        }
    }
    out
}

fn main() {
    println!("cargo:rustc-check-cfg=cfg(ik_hip)");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=../../kernels");
    for v in ["ROCM_PATH", "HIP_PATH", "INTARSIA_HIP_ARCH", "INTARSIA_NO_HIP", "INTARSIA_HIP_CHECK_ONLY"] {
        println!("cargo:rerun-if-env-changed={v}");
    }
    let arch = std::env::var("INTARSIA_HIP_ARCH").unwrap_or_else(|_| "gfx1201".into());
    println!("cargo:rustc-env=IK_HIP_ARCH={arch}");

    // Check-only build (CI): turn on `ik_hip` so `cargo check` type-checks the whole HIP Rust code and tests, but compile
    // and link nothing. The result cannot link or run; never set it for a build you will run.
    if std::env::var("INTARSIA_HIP_CHECK_ONLY").map(|v| v == "1").unwrap_or(false) {
        println!("cargo:warning=intarsia-backend-hip: INTARSIA_HIP_CHECK_ONLY=1, type-check only, no kernels built, will not link");
        println!("cargo:rustc-env=IK_HIP_COMPILER=check-only");
        println!("cargo:rustc-cfg=ik_hip");
        return;
    }
    if std::env::var("INTARSIA_NO_HIP").map(|v| v == "1").unwrap_or(false) {
        println!("cargo:warning=intarsia-backend-hip: INTARSIA_NO_HIP=1, building the stub");
        println!("cargo:rustc-env=IK_HIP_COMPILER=none");
        return;
    }
    if !cfg!(target_os = "linux") {
        println!("cargo:rustc-env=IK_HIP_COMPILER=none");
        return; // no ROCm on this host OS; stay quiet, the stub reports Unsupported at runtime
    }
    let Some(hipcc) = find_hipcc() else {
        println!("cargo:warning=intarsia-backend-hip: hipcc not found, building the stub (set ROCM_PATH)");
        println!("cargo:rustc-env=IK_HIP_COMPILER=none");
        return;
    };

    // Rebuild when the toolchain changes: cargo's target dir can outlive an image bump, and the cached
    // libintarsia_hip_kernels.a would then describe kernels built by the old compiler. Track the hipcc binary and the
    // ROCm version stamp, and bake the compiler version into the backend fingerprint (admission cache key).
    println!("cargo:rerun-if-changed={}", hipcc.display());
    if let Some(root) = hipcc.canonicalize().ok().and_then(|p| p.parent().and_then(|b| b.parent()).map(Path::to_path_buf)) {
        for f in [".info/version", ".info/version-dev"] {
            let p = root.join(f);
            if p.is_file() {
                println!("cargo:rerun-if-changed={}", p.display());
            }
        }
    }
    let compiler = Command::new(&hipcc)
        .arg("--version")
        .output()
        .ok()
        .map(|o| String::from_utf8_lossy(&o.stdout).lines().take(2).collect::<Vec<_>>().join(" ").replace('|', "/"))
        .unwrap_or_default();
    println!("cargo:rustc-env=IK_HIP_COMPILER={compiler}");

    let manifest = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let kdir = manifest.join("../../kernels/hip").canonicalize().expect("kernels/hip");
    let kroot = kdir.parent().unwrap().to_path_buf();
    let out = PathBuf::from(std::env::var("OUT_DIR").unwrap());

    let mut sources: Vec<PathBuf> = std::fs::read_dir(&kdir)
        .expect("read kernels/hip")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().map(|x| x == "hip").unwrap_or(false))
        .collect();
    sources.sort();
    assert!(!sources.is_empty(), "no .hip files in {}", kdir.display());

    // compile every translation unit in parallel
    let mut children = Vec::new();
    let mut objects = Vec::new();
    for src in &sources {
        let obj = out.join(src.file_stem().unwrap()).with_extension("o");
        // CU mode (lane hp, docs/results/2026-10-22-hp-hip-prefill-roofline.md) for the GatedDeltaNet kernels: their
        // one-wave blocks are LDS-bound, and in CU mode each CU serves its waves from its own LDS half instead of the two
        // CUs of a WGP sharing one (the delta-rule scan -13 % per token). It changes no instruction's arithmetic.
        let cu_mode = CU_MODE_SOURCES.iter().any(|f| src.file_name().map(|n| n == *f).unwrap_or(false));
        let child = Command::new(&hipcc)
            .args(["-c", "-O3", "-std=c++17", "-fPIC", "-Wall", "-Wno-unused-function"])
            .args(if cu_mode { &["-mcumode"][..] } else { &[][..] })
            .arg(format!("--offload-arch={arch}"))
            .arg("-I")
            .arg(&kroot)
            .arg("-o")
            .arg(&obj)
            .arg(src)
            .spawn()
            .unwrap_or_else(|e| panic!("failed to run {}: {e}", hipcc.display()));
        children.push((src.clone(), child));
        objects.push(obj);
    }
    for (src, mut child) in children {
        let st = child.wait().expect("wait for hipcc");
        assert!(st.success(), "hipcc failed on {}", src.display());
    }

    let lib = out.join("libintarsia_hip_kernels.a");
    let _ = std::fs::remove_file(&lib);
    let st = Command::new("ar").arg("rcs").arg(&lib).args(&objects).status().expect("ar");
    assert!(st.success(), "ar failed");

    println!("cargo:rustc-cfg=ik_hip");
    println!("cargo:rustc-link-search=native={}", out.display());
    println!("cargo:rustc-link-lib=static=intarsia_hip_kernels");
    for d in lib_dirs(&hipcc) {
        println!("cargo:rustc-link-search=native={}", d.display());
        // also usable at run time without LD_LIBRARY_PATH when the loader cache does not know ROCm
        println!("cargo:rustc-link-arg=-Wl,-rpath,{}", d.display());
    }
    println!("cargo:rustc-link-lib=dylib=amdhip64");
    println!("cargo:rustc-link-lib=dylib=stdc++");
}
