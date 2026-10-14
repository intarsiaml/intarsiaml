//! `intarsia doctor`: host fingerprint and known-pitfall checks.
//!
//! Prints a checklist (or the full JSON with `--json`). The JSON is what `intarsia bench speed
//! --fingerprint-file` embeds in benchmark results so numbers can be traced to a machine state.

use crate::common::open_backend;
use anyhow::Result;
use intarsia_core::backend::{Backend, MemKind};
use serde_json::{json, Value};
use std::path::PathBuf;
use std::process::Command;
use std::sync::Arc;
use std::time::Instant;

#[derive(clap::Args, Debug)]
pub struct DoctorArgs {
    /// Backend to inspect: cpu, cuda, hip, metal, or auto (cuda / hip / metal when compiled in, else cpu).
    #[arg(long, default_value = "auto")]
    pub backend: String,
    /// Device ordinal for GPU backends.
    #[arg(long, default_value_t = 0)]
    pub device: usize,
    /// Print the full fingerprint as JSON instead of the checklist.
    #[arg(long)]
    pub json: bool,
    /// Also write the JSON to this file.
    #[arg(long)]
    pub out: Option<PathBuf>,
    /// Buffer size for the bandwidth measurement, MiB (0 skips it).
    #[arg(long, default_value_t = 256)]
    pub bw_mib: usize,
    /// Port `intarsia serve` is expected to use (checked for conflicts).
    #[arg(long, default_value_t = 8090)]
    pub port: u16,
}

fn out(cmd: &str, args: &[&str]) -> Option<String> {
    let o = Command::new(cmd).args(args).output().ok()?;
    if !o.status.success() {
        return None;
    }
    let s = String::from_utf8_lossy(&o.stdout).trim().to_string();
    (!s.is_empty()).then_some(s)
}

fn read(p: &str) -> Option<String> {
    std::fs::read_to_string(p).ok().map(|s| s.trim().to_string()).filter(|s| !s.is_empty())
}

fn check(id: &str, status: &str, detail: impl Into<String>, fix: Option<&str>) -> Value {
    json!({"id": id, "status": status, "detail": detail.into(), "fix": fix})
}

fn gbs(bytes: usize, secs: f64) -> f64 {
    ((bytes as f64 / secs / 1e9) * 10.0).round() / 10.0
}

/// Best-of-3 transfer rates. `d2d` counts read + write traffic of a device-to-device copy.
fn bandwidth(be: &dyn Backend, mib: usize) -> Result<Value> {
    let n = mib << 20;
    let a = be.alloc(n, MemKind::Device).map_err(|e| anyhow::anyhow!("{e}"))?;
    let b = be.alloc(n, MemKind::Device).map_err(|e| anyhow::anyhow!("{e}"))?;
    let host = vec![1u8; n];
    let mut back = vec![0u8; n];
    let (mut h2d, mut d2h, mut d2d) = (f64::MAX, f64::MAX, f64::MAX);
    let e = |e: intarsia_core::Error| anyhow::anyhow!("{e}");
    for _ in 0..4 {
        let t = Instant::now();
        be.upload(&a, 0, &host, 0).map_err(e)?;
        be.synchronize(0).map_err(e)?;
        h2d = h2d.min(t.elapsed().as_secs_f64());
        let t = Instant::now();
        be.copy(&b, 0, &a, 0, n, 0).map_err(e)?;
        be.synchronize(0).map_err(e)?;
        d2d = d2d.min(t.elapsed().as_secs_f64());
        let t = Instant::now();
        be.download(&b, 0, &mut back, 0).map_err(e)?;
        d2h = d2h.min(t.elapsed().as_secs_f64());
    }
    Ok(json!({
        "buffer_mib": mib,
        "host_to_device_gbs": gbs(n, h2d),
        "device_to_host_gbs": gbs(n, d2h),
        "device_copy_gbs": gbs(2 * n, d2d),
        "note": "device_copy_gbs counts read + write bytes; compare with the memory bandwidth you pass to `bench speed --bandwidth-gbs`",
    }))
}

fn df(path: &str) -> Option<(u64, u64)> {
    let o = out("df", &["-Pk", path])?;
    let l = o.lines().nth(1)?;
    let f: Vec<&str> = l.split_whitespace().collect();
    Some((f.get(1)?.parse::<u64>().ok()? * 1024, f.get(3)?.parse::<u64>().ok()? * 1024))
}

pub fn run(args: DoctorArgs) -> Result<()> {
    let mut checks: Vec<Value> = vec![];

    // --- host ---
    let mut sys = sysinfo::System::new();
    sys.refresh_memory();
    sys.refresh_cpu_all();
    let cpus = sys.cpus();
    let cpu_model = cpus.first().map(|c| c.brand().to_string()).filter(|s| !s.is_empty()).or_else(|| {
        read("/proc/cpuinfo").and_then(|t| t.lines().find(|l| l.starts_with("model name")).and_then(|l| l.split(':').nth(1)).map(|s| s.trim().to_string()))
    });
    let os_pretty = read("/etc/os-release")
        .and_then(|t| t.lines().find_map(|l| l.strip_prefix("PRETTY_NAME=").map(|v| v.trim_matches('"').to_string())))
        .or_else(|| out("sw_vers", &["-productVersion"]).map(|v| format!("macOS {v}")));
    let host = json!({
        "hostname": out("hostname", &[]),
        "os": os_pretty,
        "kernel": out("uname", &["-sr"]),
        "arch": std::env::consts::ARCH,
        "cpu_model": cpu_model,
        "logical_cores": cpus.len(),
        "physical_cores": sysinfo::System::physical_core_count(),
        "ram_total_bytes": sys.total_memory(),
        "ram_available_bytes": sys.available_memory(),
        "swap_total_bytes": sys.total_swap(),
    });

    // --- nvidia ---
    let smi = out("nvidia-smi", &["--query-gpu=name,driver_version,compute_cap,memory.total", "--format=csv,noheader"]);
    let smi_fields: Vec<String> = smi.as_deref().and_then(|s| s.lines().next()).map(|l| l.split(',').map(|x| x.trim().to_string()).collect()).unwrap_or_default();
    let nvcc = out("nvcc", &["--version"])
        .or_else(|| out("/usr/local/cuda/bin/nvcc", &["--version"]))
        .and_then(|t| t.lines().rev().find(|l| l.contains("release")).map(|l| l.trim().to_string()));
    let cuda_ver = read("/usr/local/cuda/version.json")
        .and_then(|t| serde_json::from_str::<Value>(&t).ok())
        .and_then(|v| v["cuda"]["version"].as_str().map(String::from));
    let gpu = json!({
        "nvidia_smi": if smi_fields.len() >= 3 { json!({
            "name": smi_fields[0], "driver_version": smi_fields[1], "compute_cap": smi_fields[2], "memory_total": smi_fields.get(3),
        }) } else { Value::Null },
        "proc_driver_nvidia": read("/proc/driver/nvidia/version").map(|t| t.lines().next().unwrap_or("").to_string()),
        "nvcc": nvcc,
        "cuda_toolkit_version": cuda_ver,
        "rocm_version": read("/opt/rocm/.info/version"),
        "hipcc": out("hipcc", &["--version"]).and_then(|t| t.lines().next().map(|l| l.trim().to_string())),
    });

    // --- backend ---
    let want = match args.backend.as_str() {
        "auto" => {
            if cfg!(feature = "cuda") {
                "cuda"
            } else if cfg!(feature = "hip") {
                "hip"
            } else if cfg!(all(feature = "metal", target_os = "macos")) {
                "metal"
            } else {
                "cpu"
            }
        }
        o => o,
    }
    .to_string();
    let backend: std::result::Result<Arc<dyn Backend>, String> = open_backend(&want, args.device).map_err(|e| e.to_string());
    let backend_json = match &backend {
        Ok(be) => {
            let info = be.info();
            let kernels = be.kernels();
            let mut by_arch: std::collections::BTreeMap<String, usize> = Default::default();
            for k in &kernels {
                for a in &k.archs {
                    *by_arch.entry(a.clone()).or_default() += 1;
                }
            }
            let bw = if args.bw_mib > 0 {
                match bandwidth(be.as_ref(), args.bw_mib) {
                    Ok(v) => v,
                    Err(e) => json!({"error": e.to_string()}),
                }
            } else {
                Value::Null
            };
            json!({"requested": want, "name": be.name(), "device": info, "kernel_count": kernels.len(), "kernels_by_arch": by_arch, "bandwidth": bw})
        }
        Err(e) => json!({"requested": want, "error": e}),
    };

    // --- checks ---
    let cc = smi_fields.get(2).cloned();
    match (&cc, &backend) {
        (Some(cc), Ok(be)) if cc == "12.1" && be.name() == "cuda" => {
            let n = backend_json["kernels_by_arch"]["sm_121"].as_u64().unwrap_or(0);
            if n > 0 {
                checks.push(check("sm_121_target", "ok", format!("GPU is compute 12.1 and {n} kernels are built for sm_121"), None));
            } else {
                checks.push(check("sm_121_target", "fail", "GPU is compute 12.1 (GB10) but no kernel is registered for sm_121", Some("rebuild kernels with nvcc -gencode arch=compute_121,code=sm_121 (the build uses CUDA_ARCHS)")));
            }
        }
        (Some(cc), Err(_)) => checks.push(check("sm_121_target", "warn", format!("GPU compute capability {cc}; cannot verify kernel targets because the cuda backend is unavailable"), Some("build with `cargo build -p intarsia-cli --release --features cuda`"))),
        (Some(cc), Ok(_)) => checks.push(check("sm_121_target", "info", format!("GPU compute capability {cc}; backend '{want}' selected"), None)),
        (None, _) => checks.push(check("sm_121_target", "info", "no NVIDIA GPU visible (nvidia-smi missing or failing)", None)),
    }
    if let (Ok(be), "hip") = (&backend, want.as_str()) {
        let arch = be.info().arch;
        let n = backend_json["kernels_by_arch"][arch.as_str()].as_u64().unwrap_or(0);
        if n > 0 {
            checks.push(check("hip_arch_target", "ok", format!("GPU is {arch} and {n} kernels are built for it"), None));
        } else {
            checks.push(check("hip_arch_target", "fail", format!("GPU is {arch} but no kernel is registered for it"), Some("rebuild with INTARSIA_HIP_ARCH=<gfx arch> (the kernels need gfx12: wave32 + WMMA)")));
        }
    }
    if let (Ok(_), "metal") = (&backend, want.as_str()) {
        let dev = &backend_json["device"];
        let gb = |k: &str| dev[k].as_f64().unwrap_or(0.0) / 1e9;
        let bw = dev["bandwidth_gbs"].as_f64().map_or("not measured".to_string(), |b| format!("{b:.0} GB/s measured copy"));
        checks.push(check(
            "metal_device",
            "info",
            format!(
                "{} ({}), {} memory, {:.1} GB recommended working set, {:.1} GB free, bandwidth {bw}",
                dev["name"].as_str().unwrap_or("?"),
                dev["arch"].as_str().unwrap_or("?"),
                if dev["unified"].as_bool().unwrap_or(false) { "unified" } else { "discrete" },
                gb("total_mem"),
                gb("free_mem"),
            ),
            None,
        ));
    }
    if cc.as_deref() == Some("12.1") {
        let ver = cuda_ver.clone().or_else(|| nvcc.clone()).unwrap_or_default();
        let major = ver.split(|c: char| !c.is_ascii_digit() && c != '.').find(|t| t.contains('.')).and_then(|t| t.split('.').next()).and_then(|m| m.parse::<u32>().ok());
        match major {
            Some(m) if m >= 13 || (m == 12 && ver.contains("12.9")) => checks.push(check("cuda_toolkit", "ok", format!("CUDA toolkit {ver} supports sm_121"), None)),
            Some(_) => checks.push(check("cuda_toolkit", "fail", format!("CUDA toolkit {ver} is too old for sm_121 (needs >= 12.9)"), Some("install CUDA 13 and put /usr/local/cuda/bin on PATH"))),
            None => checks.push(check("cuda_toolkit", "warn", "nvcc / CUDA version not found", Some("export PATH=/usr/local/cuda/bin:$PATH"))),
        }
    }
    if let Some(t) = read("/sys/kernel/mm/transparent_hugepage/enabled") {
        let sel = t.split('[').nth(1).and_then(|r| r.split(']').next()).unwrap_or("?").to_string();
        let st = if sel == "never" { "warn" } else { "ok" };
        checks.push(check("transparent_hugepages", st, format!("THP = {sel} (kernel: {t})"), (st == "warn").then_some("echo madvise | sudo tee /sys/kernel/mm/transparent_hugepage/enabled  (large mmap'd weight loads fault less with madvise/always)")));
    } else {
        checks.push(check("transparent_hugepages", "info", "THP setting not readable on this OS", None));
    }
    for p in [".", std::env::var("HOME").unwrap_or_default().as_str()] {
        if p.is_empty() {
            continue;
        }
        if let Some((tot, avail)) = df(p) {
            let pct = avail as f64 / tot.max(1) as f64 * 100.0;
            let st = if avail < 5 << 30 { "fail" } else if pct < 10.0 || avail < 20 << 30 { "warn" } else { "ok" };
            checks.push(check(
                &format!("disk_free:{p}"),
                st,
                format!("{:.1} GB free of {:.0} GB ({pct:.0}% free)", avail as f64 / 1e9, tot as f64 / 1e9),
                (st != "ok").then_some("free space before downloading models or building (target/ dirs are large); keep result files small"),
            ));
        }
    }
    let hip = std::env::var("GPU_MAX_HW_QUEUES").ok();
    // radiance measured ~20% on MoE decode with 4 hardware queues (docs/research/amd-r9700-lessons.md)
    let hw_ok = matches!(hip.as_deref(), Some("1") | Some("2"));
    let st = if want == "hip" && !hw_ok { "warn" } else { "info" };
    checks.push(check(
        "hip_gpu_max_hw_queues",
        st,
        format!("GPU_MAX_HW_QUEUES={} (ROCm: 1-2 avoids hardware-queue oversubscription slowdowns)", hip.as_deref().unwrap_or("unset")),
        (st == "warn").then_some("export GPU_MAX_HW_QUEUES=1 before starting the engine"),
    ));
    match std::net::TcpListener::bind(("127.0.0.1", args.port)) {
        Ok(_) => checks.push(check("serve_port", "ok", format!("port {} is free", args.port), None)),
        Err(e) => checks.push(check("serve_port", "warn", format!("port {} is in use ({e}); 8000 belongs to vLLM on spark1", args.port), Some("pick another with `intarsia serve --port`"))),
    }
    if let Err(e) = &backend {
        checks.push(check("backend", if want == "cpu" { "fail" } else { "warn" }, format!("backend '{want}' unavailable: {e}"), None));
    } else if let Some(bw) = backend_json["bandwidth"].as_object().filter(|b| b.contains_key("device_copy_gbs")) {
        checks.push(check("bandwidth", "info", format!("host->device {} GB/s, device->host {} GB/s, device copy {} GB/s (read+write)", bw["host_to_device_gbs"], bw["device_to_host_gbs"], bw["device_copy_gbs"]), None));
    }

    let ts = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
    let report = json!({
        "kind": "intarsia-doctor", "schema_version": 1, "timestamp_unix": ts,
        "intarsia_version": env!("CARGO_PKG_VERSION"), "cuda_feature": cfg!(feature = "cuda"), "hip_feature": cfg!(feature = "hip"), "metal_feature": cfg!(all(feature = "metal", target_os = "macos")),
        "host": host, "gpu": gpu, "backend": backend_json, "checks": checks,
    });
    if let Some(p) = &args.out {
        std::fs::write(p, serde_json::to_string_pretty(&report)? + "\n")?;
    }
    if args.json {
        println!("{}", serde_json::to_string_pretty(&report)?);
    } else {
        println!("intarsia doctor  ({} {}, backend {})", host["hostname"].as_str().unwrap_or("?"), host["arch"].as_str().unwrap_or(""), want);
        println!("  os      {}", host["os"].as_str().unwrap_or("?"));
        println!("  cpu     {} ({} threads)", host["cpu_model"].as_str().unwrap_or("?"), host["logical_cores"]);
        println!("  ram     {:.1} GB total, {:.1} GB available", sys.total_memory() as f64 / 1e9, sys.available_memory() as f64 / 1e9);
        if let Some(d) = backend_json.get("device").filter(|d| d.is_object()) {
            println!("  device  {} [{} {}] free {:.1} GB", d["name"].as_str().unwrap_or(""), d["backend"].as_str().unwrap_or(""), d["arch"].as_str().unwrap_or(""), d["free_mem"].as_f64().unwrap_or(0.0) / 1e9);
        }
        println!();
        for c in &checks {
            let tag = match c["status"].as_str().unwrap_or("") { "ok" => "[ ok ]", "warn" => "[warn]", "fail" => "[FAIL]", _ => "[info]" };
            println!("{tag} {}: {}", c["id"].as_str().unwrap_or(""), c["detail"].as_str().unwrap_or(""));
            if let Some(f) = c["fix"].as_str() {
                println!("         fix: {f}");
            }
        }
        if let Some(p) = &args.out {
            println!("\nfingerprint written to {}", p.display());
        }
    }
    Ok(())
}
