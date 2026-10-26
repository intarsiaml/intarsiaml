//! Runs the kernel admission sweep of every hip kernel against the cpu reference. Needs a GPU:
//! skips (passes) when no device is available. `INTARSIA_HIP_DEVICE` picks the ordinal.
#![cfg(ik_hip)]

use intarsia_backend_cpu::CpuBackend;
use intarsia_core::admission::{run_admission, AdmissionOpts};

fn device() -> Option<std::sync::Arc<dyn intarsia_core::Backend>> {
    let ord = std::env::var("INTARSIA_HIP_DEVICE").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
    match intarsia_backend_hip::create(ord) {
        Ok(b) => Some(b),
        Err(e) => {
            eprintln!("skipping: no usable HIP device ({e})");
            None
        }
    }
}

/// Admission sweep of every hip kernel against the cpu reference, in two passes of this one test. The opt-in WMMA family
/// (`INTARSIA_HIP_WMMA_ROWS=1`) rejects every shape with the knob unset, and with it set its GEMV twins reject theirs (the knob is read
/// once per process), so no single process can grade both. Pass 1 (knob unset): every kernel that ran a case must be admitted. Pass 2: this
/// test re-runs itself with the knob on (table `fp8a8=all,bf16=all`, as `quick_coverage_wmma.rs`) and grades the kernels pass 1 gave no case,
/// each of which must now run at least one case and be admitted. A kernel with no case in either pass fails, and so does a pass 2 that
/// graded nothing (no device in the child, a crash).
#[test]
fn every_hip_kernel_matches_the_reference() {
    let Some(hip) = device() else { return };
    let cpu = CpuBackend::arc();
    if let Ok(names) = std::env::var("INTARSIA_TEST_CHILD") {
        let names: Vec<&str> = names.split(',').collect();
        let mut bad = Vec::new();
        for name in &names {
            let opts = AdmissionOpts { sweep: None, only_ops: None, only_kernel: Some(name.to_string()) };
            let report = run_admission(&*cpu, &*hip, &opts).unwrap();
            let text = report.render(true);
            println!("{text}");
            // a kernel with no summary at all (arch filter, row table) is a failure too, not a skip
            if !matches!(report.kernels.as_slice(), [k] if k.admitted) {
                bad.push(format!("{name}\n{text}"));
            }
        }
        assert!(bad.is_empty(), "kernels not admitted with INTARSIA_HIP_WMMA_ROWS=1:\n{}", bad.join("\n"));
        println!("child graded {n}/{n}", n = names.len());
        return;
    }
    let report =
        run_admission(&*cpu, &*hip, &AdmissionOpts { sweep: None, only_ops: None, only_kernel: None }).unwrap();
    let text = report.render(true);
    println!("{text}");
    assert!(!report.kernels.is_empty(), "no kernels graded");
    let failed: Vec<_> = report.kernels.iter().filter(|k| k.cases > 0 && !k.admitted).map(|k| k.kernel.as_str()).collect();
    assert!(failed.is_empty(), "some hip kernels failed admission: {failed:?}\n{text}");
    let no_case: Vec<_> = report.kernels.iter().filter(|k| k.cases == 0).map(|k| k.kernel.as_str()).collect();
    if no_case.is_empty() {
        return;
    }
    let out = std::process::Command::new(std::env::current_exe().unwrap())
        .args(["--exact", "every_hip_kernel_matches_the_reference", "--nocapture"])
        .env("INTARSIA_HIP_WMMA_ROWS", "1")
        .env("INTARSIA_HIP_WMMA_TABLE", "fp8a8=all,bf16=all")
        .env("INTARSIA_TEST_CHILD", no_case.join(","))
        .output()
        .unwrap();
    let (so, se) = (String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr));
    println!("{so}");
    assert!(out.status.success(), "kernels with no case without the knob fail with INTARSIA_HIP_WMMA_ROWS=1: {no_case:?}\n{so}\n{se}");
    let marker = format!("child graded {n}/{n}", n = no_case.len());
    assert!(so.contains(&marker), "the knob-on pass did not grade {no_case:?} (no `{marker}` in its output)\n{so}\n{se}");
}

#[test]
fn info_reports_the_device() {
    let Some(hip) = device() else { return };
    let info = hip.info();
    assert_eq!(info.backend, "hip");
    assert!(info.total_mem > 0 && info.free_mem > 0, "{info:?}");
    assert!(info.arch.starts_with("gfx12"));
}

#[test]
fn backend_is_usable_from_other_threads() {
    let Some(hip) = device() else { return };
    let b = hip.alloc(1024, intarsia_core::MemKind::Device).unwrap();
    let data: Vec<u8> = (0..1024u32).map(|i| (i % 251) as u8).collect();
    std::thread::scope(|s| {
        s.spawn(|| hip.upload(&b, 0, &data, 0).unwrap());
    });
    let mut back = vec![0u8; 1024];
    std::thread::scope(|s| {
        s.spawn(|| hip.download(&b, 0, &mut back, 0).unwrap());
    });
    assert_eq!(back, data);
}

/// Kernels registered as `Deterministic` (or `Exact`) must give bit-identical results when the same
/// call is run twice.
#[test]
fn deterministic_kernels_repeat_bit_for_bit() {
    use intarsia_core::admission::{sweep, SweepConfig};
    use intarsia_core::exec::run_host;
    use intarsia_core::Exactness;
    let Some(hip) = device() else { return };
    let cfg = SweepConfig { quick: true, ..SweepConfig::default() };
    let mut checked = 0;
    for k in hip.kernels() {
        if k.exact == Exactness::Fast {
            continue;
        }
        for case in sweep(k.op, &cfg) {
            // only cases this kernel accepts: reuse the harness's own predicate through a dry run
            let a = run_host(&*hip, &k.name, &case.call);
            let b = run_host(&*hip, &k.name, &case.call);
            let (Ok(a), Ok(b)) = (a, b) else { continue }; // not accepted by this kernel (dtype / shape)
            for (x, y) in a.iter().zip(&b) {
                assert_eq!(x.data, y.data, "{} not repeatable on {}", k.name, case.label);
            }
            checked += 1;
        }
    }
    println!("{checked} repeat checks");
    assert!(checked >= 20, "only {checked} repeat checks ran");
}

#[test]
fn fingerprint_names_the_arch_and_runtime() {
    let Some(hip) = device() else { return };
    let fp = hip.fingerprint();
    // the admission cache key: a different gfx arch or HIP runtime must not reuse cached results
    assert!(fp.contains(&hip.info().arch) && fp.contains("gfx12"), "{fp}");
    assert!(fp.contains("rt") && fp.contains("drv"), "{fp}");
    // the compiler that built the kernels and the tuning env knobs are part of the key too
    assert!(fp.contains("|cc:") && fp.contains("|env:"), "{fp}");
    assert!(fp.contains(&format!("|env:{}", intarsia_backend_hip::kernels::env_knobs())), "{fp}");
}

/// `INTARSIA_HIP_GEMV` names a (rows per warp, loads in flight) variant. A pair that is not instantiated used to launch
/// nothing and report success (stale output, garbage logits). Such a value must now fall back to the default variant and
/// still compute the right answer; the knob is read once per process, so the test re-runs itself with the bad value.
#[test]
fn uninstantiated_gemv_knob_still_computes() {
    use intarsia_core::exec::{run_host, HostCall, OutSpec};
    use intarsia_core::rng::SplitMix64;
    use intarsia_core::{Backend as _, DType, HostTensor, Op, OpAttrs};
    let Some(hip) = device() else { return };
    if std::env::var_os("INTARSIA_TEST_CHILD").is_none() {
        let st = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "uninstantiated_gemv_knob_still_computes", "--nocapture"])
            .env("INTARSIA_HIP_GEMV", "2,16")
            .env("INTARSIA_TEST_CHILD", "1")
            .status()
            .unwrap();
        assert!(st.success(), "INTARSIA_HIP_GEMV=2,16 broke the GEMV");
        return;
    }
    let cpu = CpuBackend::arc();
    let mut rng = SplitMix64::new(9);
    let (n, k) = (1024usize, 4096usize);
    let hc = HostCall {
        op: Op::Matmul,
        inputs: vec![
            HostTensor::from_f32_as(DType::Bf16, &[1, k], &rng.normal_vec(k, 1.0)),
            HostTensor::from_f32_as(DType::Bf16, &[n, k], &rng.normal_vec(n * k, 0.05)),
        ],
        outputs: vec![OutSpec::new(DType::Bf16, &[1, n])],
        attrs: OpAttrs::default(),
    };
    let rname = cpu.kernels().into_iter().find(|c| c.op == Op::Matmul && c.is_reference).unwrap().name;
    let want = run_host(&*cpu, &rname, &hc).unwrap()[0].to_f32();
    let got = run_host(&*hip, "matmul_bf16_gemv", &hc).unwrap()[0].to_f32();
    let (mut num, mut den) = (0f64, 0f64);
    for (g, w) in got.iter().zip(&want) {
        num += ((g - w) as f64).powi(2);
        den += (*w as f64).powi(2);
    }
    let rel = (num / den).sqrt();
    assert!(rel < 1e-3, "gemv with INTARSIA_HIP_GEMV=2,16: rel_l2 {rel:e}");
}
