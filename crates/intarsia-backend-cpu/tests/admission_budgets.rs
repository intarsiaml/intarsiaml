//! Mutation check for the per-output-dtype admission budgets: a candidate that runs the real cpu
//! kernel and then perturbs its output by a known relative amount BETWEEN the tightened and the old
//! budget must be refused under the tightened `Tol` and admitted under the old one.

use intarsia_backend_cpu::CpuBackend;
use intarsia_core::admission::{run_admission, AdmissionOpts, AdmissionReport, KernelSummary, SweepConfig};
use intarsia_core::backend::*;
use intarsia_core::*;

const OLD: Tol = Tol::new(1.5e-2, 6e-3);
/// Tightened bf16-output budget (CUDA/HIP quantmoe `TOL`).
const NEW: Tol = Tol::new(1.5e-2, 1e-3);
/// f32-output budget (lm_head logits).
const F32_OUT: Tol = Tol::new(1e-3, 2e-4);

/// Wraps the cpu kernels of Matmul (bf16 out) and Logits (f32 out) and scales each output by
/// `1 + eps` (bf16 outputs: `eps_bf16`, f32 outputs: `eps_f32`).
struct Perturbed {
    cpu: CpuBackend,
    eps_bf16: f32,
    eps_f32: f32,
    tol: Tol,
    tol_f32: Option<Tol>,
}

fn bf16_to_f32(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}
fn f32_to_bf16(x: f32) -> u16 {
    let u = x.to_bits();
    (u.wrapping_add(0x7fff + ((u >> 16) & 1)) >> 16) as u16
}

impl Backend for Perturbed {
    fn name(&self) -> &str {
        "perturbed"
    }
    fn info(&self) -> DeviceInfo {
        self.cpu.info()
    }
    fn alloc(&self, b: usize, k: MemKind) -> Result<Buffer> {
        self.cpu.alloc(b, k)
    }
    fn upload(&self, d: &Buffer, o: usize, s: &[u8], st: StreamId) -> Result<()> {
        self.cpu.upload(d, o, s, st)
    }
    fn download(&self, s: &Buffer, o: usize, d: &mut [u8], st: StreamId) -> Result<()> {
        self.cpu.download(s, o, d, st)
    }
    fn copy(&self, d: &Buffer, d0: usize, s: &Buffer, s0: usize, n: usize, st: StreamId) -> Result<()> {
        self.cpu.copy(d, d0, s, s0, n, st)
    }
    fn kernels(&self) -> Vec<KernelImpl> {
        self.cpu
            .kernels()
            .into_iter()
            .filter(|k| !k.is_reference && matches!(k.op, Op::Matmul | Op::Logits))
            .map(|mut k| {
                k.backend = "perturbed".into();
                k.name = format!("pert_{}", k.op.as_str());
                k.exact = Exactness::Deterministic;
                k.tol = self.tol;
                k.tol_f32 = self.tol_f32;
                k
            })
            .collect()
    }
    fn launch(&self, k: &str, call: &OpCall, st: StreamId) -> Result<()> {
        let real = format!("cpu_f32_{}", k.strip_prefix("pert_").unwrap());
        self.cpu.launch(&real, call, st)?;
        let out = &call.outputs[0];
        let mut bytes = vec![0u8; out.nbytes()];
        self.cpu.download(&out.buf, out.offset, &mut bytes, st)?;
        match out.dtype {
            DType::F32 => {
                for c in bytes.chunks_exact_mut(4) {
                    let v = f32::from_le_bytes([c[0], c[1], c[2], c[3]]) * (1.0 + self.eps_f32);
                    c.copy_from_slice(&v.to_le_bytes());
                }
            }
            DType::Bf16 => {
                for c in bytes.chunks_exact_mut(2) {
                    let v = bf16_to_f32(u16::from_le_bytes([c[0], c[1]])) * (1.0 + self.eps_bf16);
                    c.copy_from_slice(&f32_to_bf16(v).to_le_bytes());
                }
            }
            d => panic!("unexpected output dtype {d:?}"),
        }
        self.cpu.upload(&out.buf, out.offset, &bytes, st)
    }
    fn synchronize(&self, s: StreamId) -> Result<()> {
        self.cpu.synchronize(s)
    }
}

fn admit(eps_bf16: f32, eps_f32: f32, tol: Tol, tol_f32: Option<Tol>) -> AdmissionReport {
    let cand = Perturbed { cpu: CpuBackend::new(), eps_bf16, eps_f32, tol, tol_f32 };
    let opts = AdmissionOpts { sweep: Some(SweepConfig { quick: true, ..Default::default() }), ..Default::default() };
    run_admission(&CpuBackend::new(), &cand, &opts).unwrap()
}

fn summary(r: &AdmissionReport, op: Op) -> &KernelSummary {
    let s = r.kernels.iter().find(|k| k.op == op).unwrap();
    assert!(s.cases > 0, "{op}: no cases ran");
    s
}

#[test]
fn unperturbed_kernels_are_admitted_at_the_tight_budgets() {
    let r = admit(0.0, 0.0, NEW, Some(F32_OUT));
    assert!(summary(&r, Op::Matmul).admitted, "\n{}", r.render(true));
    assert!(summary(&r, Op::Logits).admitted, "\n{}", r.render(true));
}

#[test]
fn bf16_error_between_new_and_old_budget_is_refused_by_new_and_admitted_by_old() {
    // a 3e-3 relative scale (plus bf16 rounding) is ~3x over the new 1e-3 and ~2x under the old 6e-3
    let tight = admit(3e-3, 0.0, NEW, None);
    let m = summary(&tight, Op::Matmul);
    assert!(!m.admitted, "3e-3 bf16 error must be refused at rel_l2 1e-3 (worst {:.2e})", m.worst_rel_l2);
    assert!(m.worst_rel_l2 > 1e-3 && m.worst_rel_l2 < 6e-3, "perturbation not between budgets: {:.2e}", m.worst_rel_l2);
    let loose = admit(3e-3, 0.0, OLD, None);
    assert!(summary(&loose, Op::Matmul).admitted, "\n{}", loose.render(true));
}

#[test]
fn f32_error_between_f32_budget_and_bf16_budget_is_refused_only_with_tol_f32() {
    // 5e-4 relative on the f32 logits: over the f32 budget (2e-4), under the bf16 budget (1e-3)
    let with = admit(0.0, 5e-4, NEW, Some(F32_OUT));
    let l = summary(&with, Op::Logits);
    assert!(!l.admitted, "5e-4 f32 error must be refused by tol_f32 (worst {:.2e})", l.worst_rel_l2);
    // the row reports the tolerance actually applied (the f32 one), not `tol`
    let row = with.rows.iter().find(|r| r.op == Op::Logits).unwrap();
    assert_eq!((row.tol_max_abs, row.tol_rel_l2), (F32_OUT.max_abs, F32_OUT.rel_l2));
    // bf16 matmul outputs of the same kernel table keep `tol`
    let row = with.rows.iter().find(|r| r.op == Op::Matmul).unwrap();
    assert_eq!((row.tol_max_abs, row.tol_rel_l2), (NEW.max_abs, NEW.rel_l2));
    // without tol_f32 the same error is admitted under the shared bf16 budget
    let without = admit(0.0, 5e-4, NEW, None);
    assert!(summary(&without, Op::Logits).admitted, "\n{}", without.render(true));
}
