//! The admission harness grading the cpu backend against its own oracle.

use intarsia_backend_cpu::CpuBackend;
use intarsia_core::admission::{run_admission, AdmissionOpts, SweepConfig};
use intarsia_core::{Admission, Registry};

#[test]
fn cpu_fast_kernels_pass_admission_quick_sweep() {
    let b = CpuBackend::new();
    let opts = AdmissionOpts { sweep: Some(SweepConfig { quick: true, ..Default::default() }), ..Default::default() };
    let report = run_admission(&b, &b, &opts).unwrap();
    assert!(!report.kernels.is_empty());
    // the DeepSeek-V4.1 ops without a reference yet (lane 1b) are rejected with their "not implemented" error, never
    // admitted vacuously; every other kernel passes
    let stubs = intarsia_backend_cpu::dsv41_ref::STUBS;
    assert!(report.kernels.iter().filter(|k| !stubs.contains(&k.op)).all(|k| k.admitted), "\n{}", report.render(false));
    for op in stubs {
        assert!(report.kernels.iter().any(|k| k.op == op && !k.admitted && k.cases > 0), "{op}: stub must have cases and be rejected");
        assert!(
            report.rows.iter().any(|r| r.op == op && r.error.as_deref().is_some_and(|e| e.contains("not implemented (lane 1b)"))),
            "{op}: admission must report the stub error"
        );
    }
    // every other op has an admitted implementation with at least one case
    for op in intarsia_core::Op::ALL.into_iter().filter(|o| !stubs.contains(o)) {
        assert!(report.kernels.iter().any(|k| k.op == op && k.admitted && k.cases > 0), "{op} has no admitted kernel");
    }
    // applying the report marks kernels admitted in a registry
    let mut reg = Registry::new("cpu", &intarsia_core::Backend::info(&b), intarsia_core::Backend::kernels(&b));
    report.apply(&mut reg);
    assert!(reg.impls().any(|(_, a)| a == Admission::Admitted));
}

#[test]
fn broken_kernel_is_rejected() {
    // A candidate backend that wraps cpu but corrupts matmul must fail admission.
    use intarsia_core::backend::*;
    use intarsia_core::*;
    struct Broken(CpuBackend);
    impl Backend for Broken {
        fn name(&self) -> &str {
            "broken"
        }
        fn info(&self) -> DeviceInfo {
            self.0.info()
        }
        fn alloc(&self, b: usize, k: MemKind) -> Result<Buffer> {
            self.0.alloc(b, k)
        }
        fn upload(&self, d: &Buffer, o: usize, s: &[u8], st: StreamId) -> Result<()> {
            self.0.upload(d, o, s, st)
        }
        fn download(&self, s: &Buffer, o: usize, d: &mut [u8], st: StreamId) -> Result<()> {
            self.0.download(s, o, d, st)
        }
        fn copy(&self, d: &Buffer, d0: usize, s: &Buffer, s0: usize, n: usize, st: StreamId) -> Result<()> {
            self.0.copy(d, d0, s, s0, n, st)
        }
        fn kernels(&self) -> Vec<KernelImpl> {
            self.0
                .kernels()
                .into_iter()
                .filter(|k| !k.is_reference && k.op == Op::Matmul)
                .map(|mut k| {
                    k.backend = "broken".into();
                    k.name = "bad_matmul".into();
                    k
                })
                .collect()
        }
        fn launch(&self, _k: &str, call: &OpCall, st: StreamId) -> Result<()> {
            // run the real kernel, then clobber the output
            self.0.launch("cpu_f32_matmul", call, st)?;
            let out = &call.outputs[0];
            let junk = vec![0x3fu8; out.nbytes()];
            self.0.upload(&out.buf, out.offset, &junk, st)
        }
        fn synchronize(&self, s: StreamId) -> Result<()> {
            self.0.synchronize(s)
        }
    }
    let good = CpuBackend::new();
    let bad = Broken(CpuBackend::new());
    let opts = AdmissionOpts { sweep: Some(SweepConfig { quick: true, ..Default::default() }), ..Default::default() };
    let report = run_admission(&good, &bad, &opts).unwrap();
    assert!(!report.all_pass());
    assert!(report.kernels.iter().all(|k| !k.admitted));
}
