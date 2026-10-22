//! Compiles kernels/cuda/*.cu with nvcc into a static library and links it (plus the CUDA
//! runtime and cuBLAS). Without a CUDA toolkit (the Mac, CI) the crate still builds: it compiles
//! to a stub whose `create()` returns `Unsupported`, so `cargo check --workspace` works anywhere.
//!
//! Environment:
//!   CUDA_PATH / CUDA_HOME / CUDA_ROOT   toolkit root (default: /usr/local/cuda, then nvcc on PATH)
//!   INTARSIA_CUDA_ARCH                   nvcc -arch value, default `sm_121` (GB10)
//!   INTARSIA_NO_CUDA=1                   force the stub
//!   INTARSIA_CUDA_NO_STAGE=1             compile the kernels in place (see "Reproducible objects"; the binary then depends on the checkout path)
//!
//! Reproducible objects: the same git sha must give the same `libintarsia_kernels.a` bytes in any checkout directory.
//! nvcc breaks that three ways: it names the anonymous-namespace symbols in every cubin `_GLOBAL__N__<hash>_...` with a
//! hash of the source file's ABSOLUTE path and mtime; `-lineinfo` records the source directory; and the host object
//! carries nvcc's temp file name (`tmpxft_<pid>_...cudafe1.cpp`) as an ELF file symbol. So the kernels are compiled from a
//! copy at a fixed, content-addressed path (`/tmp/intarsia-cuda-kernels-<hash>`, files stamped with a constant mtime), and
//! `objcopy --strip-unneeded` drops the file symbol and unreferenced local symbols from each object. Device code is
//! unchanged.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// Directories under `kernels/` the CUDA sources include from (`#include "../common/..."`, `"abi/..."`).
const STAGE_DIRS: [&str; 3] = ["cuda", "common", "abi"];
/// The mtime every staged file gets (nvcc mixes it into the anonymous-namespace hash).
const STAGE_MTIME_SECS: u64 = 1_000_000_000;

fn collect_files(dir: &Path, rel: &Path, out: &mut Vec<(PathBuf, PathBuf)>) {
    let mut entries: Vec<_> = std::fs::read_dir(dir).expect("read kernels dir").filter_map(|e| e.ok()).collect();
    entries.sort_by_key(|e| e.file_name());
    for e in entries {
        let (p, r) = (e.path(), rel.join(e.file_name()));
        if p.is_dir() {
            collect_files(&p, &r, out);
        } else {
            out.push((p, r));
        }
    }
}

/// FNV-1a 64 over every staged file's relative path and bytes (stable across Rust versions, unlike `DefaultHasher`).
fn content_hash(files: &[(PathBuf, PathBuf)]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    let mut eat = |bytes: &[u8]| {
        for b in bytes {
            h = (h ^ *b as u64).wrapping_mul(0x100000001b3);
        }
    };
    for (p, r) in files {
        eat(r.to_string_lossy().as_bytes());
        eat(&[0]);
        let data = std::fs::read(p).expect("read kernel source");
        eat(&(data.len() as u64).to_le_bytes());
        eat(&data);
    }
    h
}

fn set_mtime(path: &Path) {
    let t = UNIX_EPOCH + Duration::from_secs(STAGE_MTIME_SECS);
    let f = std::fs::File::open(path).expect("open for mtime");
    f.set_times(std::fs::FileTimes::new().set_accessed(t).set_modified(t)).expect("set mtime");
}

/// Copy the kernel sources to `/tmp/intarsia-cuda-kernels-<content hash>/` (once; later builds of the same sources reuse it)
/// and return that directory, or None to compile in place. A pre-existing directory is used only after its content hashes
/// to the expected value, so a stale or foreign directory cannot change what is compiled.
fn stage_kernels(kroot: &Path) -> Option<PathBuf> {
    if std::env::var("INTARSIA_CUDA_NO_STAGE").map(|v| v == "1").unwrap_or(false) {
        return None;
    }
    let mut files = Vec::new();
    for d in STAGE_DIRS {
        let dir = kroot.join(d);
        if dir.is_dir() {
            collect_files(&dir, Path::new(d), &mut files);
        }
    }
    let hash = content_hash(&files);
    let dest = PathBuf::from(format!("/tmp/intarsia-cuda-kernels-{hash:016x}"));
    let verified = |dir: &Path| {
        let mut staged = Vec::new();
        for d in STAGE_DIRS {
            if dir.join(d).is_dir() {
                collect_files(&dir.join(d), Path::new(d), &mut staged);
            }
        }
        content_hash(&staged) == hash
    };
    if dest.is_dir() {
        if verified(&dest) {
            return Some(dest);
        }
        println!("cargo:warning=intarsia-backend-cuda: {} exists with other content; compiling in place (the binary will depend on the checkout path)", dest.display());
        return None;
    }
    let tmp = PathBuf::from(format!("/tmp/.intarsia-cuda-stage-{}-{}", std::process::id(), SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or(0)));
    let built = (|| -> std::io::Result<()> {
        for (src, rel) in &files {
            let to = tmp.join(rel);
            std::fs::create_dir_all(to.parent().unwrap())?;
            std::fs::copy(src, &to)?;
            set_mtime(&to);
        }
        // directories last, deepest first: creating a file changes its directory's mtime
        let mut dirs: Vec<PathBuf> = files.iter().filter_map(|(_, r)| r.parent().map(|p| tmp.join(p))).collect();
        dirs.sort();
        dirs.dedup();
        for d in dirs.iter().rev() {
            set_mtime(d);
        }
        set_mtime(&tmp);
        std::fs::rename(&tmp, &dest)
    })();
    match built {
        Ok(()) => Some(dest),
        Err(_) if dest.is_dir() && verified(&dest) => {
            let _ = std::fs::remove_dir_all(&tmp); // another build created it first
            Some(dest)
        }
        Err(e) => {
            let _ = std::fs::remove_dir_all(&tmp);
            println!("cargo:warning=intarsia-backend-cuda: cannot stage the kernels under /tmp ({e}); compiling in place (the binary will depend on the checkout path)");
            None
        }
    }
}

/// `objcopy --strip-unneeded` on an object nvcc wrote: drops the `tmpxft_<pid>_...` file symbol and unreferenced local symbols.
/// Best effort: without objcopy the build still works, only the binary is not reproducible.
fn strip_object(obj: &Path) -> bool {
    Command::new("objcopy").arg("--strip-unneeded").arg(obj).status().map(|s| s.success()).unwrap_or(false)
}

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
    for v in ["CUDA_PATH", "CUDA_HOME", "CUDA_ROOT", "INTARSIA_CUDA_ARCH", "INTARSIA_NO_CUDA", "INTARSIA_CUDA_NO_STAGE", "INTARSIA_CUDA_CHECK_ONLY"] {
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

    // Check-only build (CI): turn on the real-GPU cfgs so `cargo check` type-checks the whole `ik_cuda` Rust code and
    // tests, but compile and link nothing. The result cannot link or run; never set it for a build you will run.
    if std::env::var("INTARSIA_CUDA_CHECK_ONLY").map(|v| v == "1").unwrap_or(false) {
        println!("cargo:warning=intarsia-backend-cuda: INTARSIA_CUDA_CHECK_ONLY=1, type-check only, no kernels built, will not link");
        println!("cargo:rustc-cfg=ik_cuda");
        println!("cargo:rustc-cfg=ik_sm120a");
        return;
    }
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

    // compile every translation unit in parallel, from the staged copy of the kernels (see "Reproducible objects" above)
    let workdir = stage_kernels(&kroot).unwrap_or_else(|| kroot.clone());
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
        // Run in the kernels root (staged: a fixed path) with a relative source path and `-I .`. The output path is not embedded.
        let rel = src.strip_prefix(&kroot).unwrap_or(src);
        let child = Command::new(&nvcc)
            .current_dir(&workdir)
            .args(["-c", "-O3", "--std=c++17", "-lineinfo", "--expt-relaxed-constexpr"])
            .args(&arch_args)
            .args(a_def)
            .args(["-Xcompiler", "-fPIC,-Wall"])
            .arg("-I")
            .arg(".")
            .arg("-o")
            .arg(&obj)
            .arg(rel)
            .spawn()
            .unwrap_or_else(|e| panic!("failed to run {}: {e}", nvcc.display()));
        children.push((src.clone(), child));
        objects.push(obj);
    }
    for (src, mut child) in children {
        let st = child.wait().expect("wait for nvcc");
        assert!(st.success(), "nvcc failed on {}", src.display());
    }
    // (see "Reproducible objects"); a missing objcopy only costs reproducibility
    if !objects.iter().all(|o| strip_object(o)) {
        println!("cargo:warning=intarsia-backend-cuda: objcopy failed or is missing; the kernels object files keep nvcc's temp file names (the binary hash will vary between builds)");
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
