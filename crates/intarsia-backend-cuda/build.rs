//! Compiles kernels/cuda/*.cu with nvcc into a static library and links it (plus the CUDA
//! runtime and cuBLAS). Without a CUDA toolkit (the Mac, CI) the crate still builds: it compiles
//! to a stub whose `create()` returns `Unsupported`, so `cargo check --workspace` works anywhere.
//!
//! Environment:
//!   CUDA_PATH / CUDA_HOME / CUDA_ROOT   toolkit root (default: /usr/local/cuda, then nvcc on PATH)
//!   INTARSIA_CUDA_ARCH                   nvcc -arch value, default `sm_121` (GB10)
//!   INTARSIA_NO_CUDA=1                   force the stub

use std::path::{Path, PathBuf};
use std::process::Command;

fn find_nvcc() -> Option<PathBuf> {
    let mut roots: Vec<PathBuf> = Vec::new();
    for var in ["CUDA_PATH", "CUDA_HOME", "CUDA_ROOT"] {
        if let Some(v) = std::env::var_os(var) {
            roots.push(PathBuf::from(v));
        }
    }
    roots.push(PathBuf::from("/usr/local/cuda"));
    for r in roots {
        let p = r.join("bin").join("nvcc");
        if p.is_file() {
            return Some(p);
        }
    }
    // nvcc on PATH
    if Command::new("nvcc").arg("--version").output().map(|o| o.status.success()).unwrap_or(false) {
        return Some(PathBuf::from("nvcc"));
    }
    None
}

/// Directories that may hold libcudart / libcublas, derived from the nvcc location.
fn lib_dirs(nvcc: &Path) -> Vec<PathBuf> {
    let root = nvcc.parent().and_then(|b| b.parent()).map(Path::to_path_buf);
    let mut out = Vec::new();
    if let Some(root) = root {
        for rel in ["lib64", "lib", "targets/sbsa-linux/lib", "targets/aarch64-linux/lib", "targets/x86_64-linux/lib"] {
            let d = root.join(rel);
            if d.join("libcudart.so").exists() || d.join("libcudart.so.13").exists() {
                out.push(d);
            }
        }
    }
    out
}

fn main() {
    println!("cargo:rustc-check-cfg=cfg(ik_cuda)");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=../../kernels");
    for v in ["CUDA_PATH", "CUDA_HOME", "CUDA_ROOT", "INTARSIA_CUDA_ARCH", "INTARSIA_NO_CUDA"] {
        println!("cargo:rerun-if-env-changed={v}");
    }
    let arch = std::env::var("INTARSIA_CUDA_ARCH").unwrap_or_else(|_| "sm_121".into());
    println!("cargo:rustc-env=IK_CUDA_ARCH={arch}");
    // Files named `*_sm120a.cu` use block-scaled FP4 tensor-core instructions (`mma ... kind::mxf4nvf4.block_scale`),
    // which nvcc only accepts for the architecture-specific targets sm_120a / sm_121a (plain `sm_121` rejects them).
    // Only those files are built for the `a` target; every other kernel keeps `arch`. On any other arch the files are
    // built as stubs (-DIK_SM120A=0) and the Rust side does not register their kernels (cfg `ik_sm120a`).
    let sm120a_arch = match arch.as_str() {
        "sm_120" | "sm_121" => Some(format!("{arch}a")),
        "sm_120a" | "sm_121a" => Some(arch.clone()),
        _ => None,
    };
    println!("cargo:rustc-check-cfg=cfg(ik_sm120a)");

    if std::env::var("INTARSIA_NO_CUDA").map(|v| v == "1").unwrap_or(false) {
        println!("cargo:warning=intarsia-backend-cuda: INTARSIA_NO_CUDA=1, building the stub");
        return;
    }
    if !cfg!(target_os = "linux") {
        return; // no CUDA on this host OS; stay quiet, the stub reports Unsupported at runtime
    }
    let Some(nvcc) = find_nvcc() else {
        println!("cargo:warning=intarsia-backend-cuda: nvcc not found, building the stub (set CUDA_PATH)");
        return;
    };

    let manifest = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let kdir = manifest.join("../../kernels/cuda").canonicalize().expect("kernels/cuda");
    let kroot = kdir.parent().unwrap().to_path_buf();
    let out = PathBuf::from(std::env::var("OUT_DIR").unwrap());

    let mut sources: Vec<PathBuf> = std::fs::read_dir(&kdir)
        .expect("read kernels/cuda")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().map(|x| x == "cu").unwrap_or(false))
        .collect();
    sources.sort();
    assert!(!sources.is_empty(), "no .cu files in {}", kdir.display());

    // compile every translation unit in parallel
    let mut children = Vec::new();
    let mut objects = Vec::new();
    for src in &sources {
        let obj = out.join(src.file_stem().unwrap()).with_extension("o");
        let is_a = src.file_stem().unwrap().to_string_lossy().ends_with("_sm120a");
        // every other translation unit gets exactly the command line it had before
        // (`-arch=sm_121a` would also emit compute_121 PTX, which lacks the instruction and is rejected: gencode only the a target)
        let (arch_args, a_def) = match (&sm120a_arch, is_a) {
            (Some(a), true) => (
                vec![format!("-gencode=arch={},code={a}", a.replace("sm_", "compute_"))],
                Some("-DIK_SM120A=1"),
            ),
            (None, true) => (vec![format!("-arch={arch}")], Some("-DIK_SM120A=0")),
            _ => (vec![format!("-arch={arch}")], None),
        };
        let child = Command::new(&nvcc)
            .args(["-c", "-O3", "--std=c++17", "-lineinfo", "--expt-relaxed-constexpr"])
            .args(&arch_args)
            .args(a_def)
            .args(["-Xcompiler", "-fPIC,-Wall"])
            .arg("-I")
            .arg(&kroot)
            .arg("-o")
            .arg(&obj)
            .arg(src)
            .spawn()
            .unwrap_or_else(|e| panic!("failed to run {}: {e}", nvcc.display()));
        children.push((src.clone(), child));
        objects.push(obj);
    }
    for (src, mut child) in children {
        let st = child.wait().expect("wait for nvcc");
        assert!(st.success(), "nvcc failed on {}", src.display());
    }

    let lib = out.join("libintarsia_kernels.a");
    let st = Command::new(&nvcc).arg("-lib").arg("-o").arg(&lib).args(&objects).status().expect("nvcc -lib");
    assert!(st.success(), "nvcc -lib failed");

    println!("cargo:rustc-cfg=ik_cuda");
    if sm120a_arch.is_some() {
        println!("cargo:rustc-cfg=ik_sm120a");
    }
    println!("cargo:rustc-link-search=native={}", out.display());
    println!("cargo:rustc-link-lib=static=intarsia_kernels");
    for d in lib_dirs(&nvcc) {
        println!("cargo:rustc-link-search=native={}", d.display());
        // also usable at run time without LD_LIBRARY_PATH when the loader cache does not know CUDA
        println!("cargo:rustc-link-arg=-Wl,-rpath,{}", d.display());
    }
    println!("cargo:rustc-link-lib=dylib=cublas");
    println!("cargo:rustc-link-lib=dylib=cublasLt");
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=stdc++");
}
