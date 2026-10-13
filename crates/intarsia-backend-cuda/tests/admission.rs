//! Runs the kernel admission sweep of every cuda kernel against the cpu reference. Needs a GPU:
//! skips (passes) when no device is available. `INTARSIA_CUDA_DEVICE` picks the ordinal.
#![cfg(ik_cuda)]

use intarsia_backend_cpu::CpuBackend;
use intarsia_core::admission::{run_admission, AdmissionOpts};

fn device() -> Option<std::sync::Arc<dyn intarsia_core::Backend>> {
    let ord = std::env::var("INTARSIA_CUDA_DEVICE").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
    match intarsia_backend_cuda::create(ord) {
        Ok(b) => Some(b),
        Err(e) => {
            eprintln!("skipping: no usable CUDA device ({e})");
            None
        }
    }
}

#[test]
fn every_cuda_kernel_matches_the_reference() {
    let Some(cuda) = device() else { return };
    let cpu = CpuBackend::arc();
    let report =
        run_admission(&*cpu, &*cuda, &AdmissionOpts { sweep: None, only_ops: None, only_kernel: None }).unwrap();
    let text = report.render(true);
    println!("{text}");
    assert!(report.all_pass(), "some cuda kernels failed admission:\n{text}");
    // every registered kernel must have been exercised by at least one case
    for k in &report.kernels {
        assert!(k.cases > 0, "kernel {} ran no admission cases", k.kernel);
    }
}

#[test]
fn info_reports_the_device() {
    let Some(cuda) = device() else { return };
    let info = cuda.info();
    assert_eq!(info.backend, "cuda");
    assert!(info.total_mem > 0 && info.free_mem > 0, "{info:?}");
    assert!(info.arch.starts_with("sm_"));
}

#[test]
fn backend_is_usable_from_other_threads() {
    let Some(cuda) = device() else { return };
    let b = cuda.alloc(1024, intarsia_core::MemKind::Device).unwrap();
    let data: Vec<u8> = (0..1024u32).map(|i| (i % 251) as u8).collect();
    std::thread::scope(|s| {
        s.spawn(|| cuda.upload(&b, 0, &data, 0).unwrap());
    });
    let mut back = vec![0u8; 1024];
    std::thread::scope(|s| {
        s.spawn(|| cuda.download(&b, 0, &mut back, 0).unwrap());
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
    let Some(cuda) = device() else { return };
    let cfg = SweepConfig { quick: true, ..SweepConfig::default() };
    let mut checked = 0;
    for k in cuda.kernels() {
        if k.exact == Exactness::Fast {
            continue;
        }
        for case in sweep(k.op, &cfg) {
            // only cases this kernel accepts: reuse the harness's own predicate through a dry run
            let a = run_host(&*cuda, &k.name, &case.call);
            let b = run_host(&*cuda, &k.name, &case.call);
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
