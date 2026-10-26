//! Direct correctness check of tensor-parallel partial sums on the real HIP kernels and the real P2P all-reduce,
//! independent of KL: the tolerance is derived from the fp32 reduction-order bound, not tuned (derivation, what it does
//! not cover and the mutant table: `common/tp_partial.rs`; the same driver runs against emulated GEMVs on any machine in
//! `tp_partial_math.rs`).
//!
//! The sharded path is the model's own row-parallel projection (out_proj / down_proj, `ShardAxis::Cols`): tp=1 is the
//! registry-selected bf16 GEMV with f32 output over the full K; tp=2 gives each rank half of K (its `x` slice and
//! its `[n, K/2]` weight columns), each rank's GEMV writes an f32 partial, and `Comm::all_reduce_sum_into` adds them
//! into f32 and (a second call, same partials) into bf16. K = 2048 (35B hidden / 4096-wide o_proj halves) and
//! K = 5120 (27B hidden), n = 2048, m = 1 (decode GEMV), 8 (multi-row GEMV), 64 (WMMA tile kernels).
//!
//! Needs two GPUs that can reach each other: without them the test prints SKIPPED (and fails under CI=1 /
//! INTARSIA_REQUIRE_GPU=1). `INTARSIA_HIP_DEVICES=0,1` picks the devices. Run: `cargo test -p intarsia-backend-hip
//! --test tp_partial_sum -- --nocapture` (every number is printed before the assertion fires).
#![cfg(ik_hip)]

mod common;

use common::tp_partial::*;
use intarsia_core::backend::{Backend, MemKind, TensorRef};
use intarsia_core::Comm;
use intarsia_core::dtype::decode_to_f32;
use intarsia_core::quant::{HostQTensor, QTensor, BF16};
use intarsia_core::{DType, HostTensor, Registry, ShapeInfo};
use std::sync::{Arc, Mutex, MutexGuard};

/// One group at a time (see `comm.rs`: with GPU_MAX_HW_QUEUES=1 two groups in one process can deadlock).
static SERIAL: Mutex<()> = Mutex::new(());

fn serial() -> MutexGuard<'static, ()> {
    SERIAL.lock().unwrap_or_else(|e| e.into_inner())
}

fn group(test: &str) -> Option<(Vec<Arc<dyn Backend>>, Vec<Arc<dyn Comm>>)> {
    let ords: Vec<usize> = std::env::var("INTARSIA_HIP_DEVICES")
        .ok()
        .map(|v| v.split(',').filter_map(|x| x.trim().parse().ok()).collect())
        .unwrap_or_else(|| vec![0, 1]);
    match intarsia_backend_hip::create_group(&ords) {
        Ok(g) => Some(g),
        Err(e) => {
            common::skip(test, &format!("no usable 2-GPU HIP group ({e})"));
            None
        }
    }
}

fn upload(be: &dyn Backend, dt: DType, shape: &[usize], vals: &[f32]) -> TensorRef {
    let h = HostTensor::from_f32_as(dt, shape, vals);
    let buf = be.alloc(h.nbytes().max(16), MemKind::Device).unwrap();
    be.upload(&buf, 0, &h.data, 0).unwrap();
    TensorRef::new(buf, dt, shape)
}

fn zeros(be: &dyn Backend, dt: DType, shape: &[usize]) -> TensorRef {
    let n: usize = shape.iter().product();
    let buf = be.alloc((n * dt.size_bytes()).max(16), MemKind::Device).unwrap();
    be.upload(&buf, 0, &vec![0xA5u8; n * dt.size_bytes()], 0).unwrap();
    TensorRef::new(buf, dt, shape)
}

fn download(be: &dyn Backend, t: &TensorRef) -> Vec<f32> {
    be.synchronize(0).unwrap();
    let mut bytes = vec![0u8; t.nbytes()];
    be.download(&t.buf, t.offset, &mut bytes, 0).unwrap();
    let mut out = vec![0f32; t.numel()];
    decode_to_f32(t.dtype, &bytes, &mut out);
    out
}

fn bf16_weights(be: &dyn Backend, n: usize, k: usize, vals: &[f32]) -> QTensor {
    let part = HostTensor::from_f32_as(DType::Bf16, &[n, k], vals);
    HostQTensor { format: BF16, n, k, parts: vec![part], act_quant: false }.upload(be).unwrap()
}

struct Gpu {
    be: Vec<Arc<dyn Backend>>,
    comms: Vec<Arc<dyn Comm>>,
    regs: Vec<Registry>,
    w_full: Option<QTensor>,
    w_half: Vec<QTensor>,
    kernels: std::collections::BTreeSet<String>,
}

impl Gpu {
    fn new(be: Vec<Arc<dyn Backend>>, comms: Vec<Arc<dyn Comm>>) -> Gpu {
        let regs = be.iter().map(|b| Registry::new(b.name(), &b.info(), b.kernels())).collect();
        Gpu { be, comms, regs, w_full: None, w_half: vec![], kernels: Default::default() }
    }

    /// `x @ w^T` on rank `r`'s device, f32 `[m, n]`, through the kernel the registry picks (the model's own selection).
    fn gemv(&mut self, r: usize, w: &QTensor, x: &TensorRef) -> TensorRef {
        let (be, reg) = (&self.be[r], &self.regs[r]);
        let m = x.shape[0];
        let out = zeros(&**be, DType::F32, &[m, w.n]);
        let mut call = w.call(x.clone(), out.clone());
        call.check().unwrap();
        let k = reg.select(&call).unwrap_or_else(|e| panic!("no kernel for m={m} n={} k={}: {e}", w.n, w.k));
        let need = k.scratch_bytes(&ShapeInfo::from_call(&call));
        if need > 0 {
            let b = be.alloc(need.max(16), MemKind::Device).unwrap();
            call.scratch = Some(TensorRef::new(b, DType::U8, &[need]));
        }
        be.launch(&k.name, &call, 0).unwrap_or_else(|e| panic!("{}: {e}", k.name));
        be.synchronize(0).unwrap();
        self.kernels.insert(format!("m={m} k={}: {}", w.k, k.name));
        out
    }
}

impl Engine for Gpu {
    fn name(&self) -> String {
        format!("hip {}", self.comms[0].name())
    }

    fn prepare(&mut self, c: &Case) {
        self.w_full = Some(bf16_weights(&*self.be[0], c.n, c.k, &c.w));
        self.w_half = (0..2).map(|r| bf16_weights(&*self.be[r], c.n, c.k / 2, &c.w_half(r))).collect();
    }

    fn tp1(&mut self, c: &Case) -> Vec<f32> {
        let x = upload(&*self.be[0], DType::Bf16, &[c.m, c.k], &c.x);
        let w = self.w_full.clone().unwrap();
        let out = self.gemv(0, &w, &x);
        download(&*self.be[0], &out)
    }

    fn partials(&mut self, c: &Case, xh: [Vec<f32>; 2]) -> [Vec<f32>; 2] {
        [0, 1].map(|r| {
            let x = upload(&*self.be[r], DType::Bf16, &[c.m, c.k / 2], &xh[r]);
            let w = self.w_half[r].clone();
            let out = self.gemv(r, &w, &x);
            download(&*self.be[r], &out)
        })
    }

    fn reduce(&mut self, c: &Case, p: &[Vec<f32>; 2]) -> Reduced {
        let shape = [c.m, c.n];
        let parts: Vec<TensorRef> = (0..2).map(|r| upload(&*self.be[r], DType::F32, &shape, &p[r])).collect();
        let o32: Vec<TensorRef> = (0..2).map(|r| zeros(&*self.be[r], DType::F32, &shape)).collect();
        let o16: Vec<TensorRef> = (0..2).map(|r| zeros(&*self.be[r], DType::Bf16, &shape)).collect();
        std::thread::scope(|sc| {
            for r in 0..2 {
                let (cm, pt, a, b) = (&self.comms[r], &parts[r], &o32[r], &o16[r]);
                sc.spawn(move || {
                    cm.all_reduce_sum_into(pt, a, 0).unwrap();
                    cm.all_reduce_sum_into(pt, b, 0).unwrap();
                });
            }
        });
        Reduced {
            f32_out: [0, 1].map(|r| download(&*self.be[r], &o32[r])),
            bf16_out: [0, 1].map(|r| download(&*self.be[r], &o16[r])),
        }
    }
}

#[test]
fn tp2_partial_sums_match_tp1_within_the_derived_fp32_bound_and_mutants_do_not() {
    let _one = serial();
    let Some((be, comms)) = group("tp2_partial_sums_match_tp1_within_the_derived_fp32_bound_and_mutants_do_not") else { return };
    assert_eq!(comms[0].name(), "hip-p2p");
    let mut gpu = Gpu::new(be, comms);
    let mut failures = vec![];
    println!(
        "gamma_2048 {:.3e} gammaT_2048 {:.3e} gamma_5120 {:.3e} gammaT_5120 {:.3e} (u = 2^-24, lambda = {LAMBDA})",
        gamma(2048.0),
        gamma_tilde(2048.0),
        gamma(5120.0),
        gamma_tilde(5120.0)
    );
    for kind in [Kind::SameSign, Kind::RandomSign] {
        for (m, n, k) in [(1usize, 2048usize, 2048usize), (1, 2048, 5120), (8, 2048, 2048), (8, 2048, 5120), (64, 2048, 2048)] {
            let c = Case::new(kind, m, n, k, 1000 + (m * 7 + k) as u64);
            // a panic inside a kernel launch (no kernel for this shape, a launch error) becomes a failure line, so the
            // rest of the table still prints in this GPU window
            let rep = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| run_case(&mut gpu, &c))) {
                Ok(rep) => rep,
                Err(e) => {
                    let msg = e.downcast_ref::<String>().cloned().or_else(|| e.downcast_ref::<&str>().map(|s| s.to_string())).unwrap_or_default();
                    Report { lines: vec![], failures: vec![format!("{kind:?} m={m} n={n} k={k}: panicked: {msg}")] }
                }
            };
            for l in &rep.lines {
                println!("{l}");
            }
            failures.extend(rep.failures);
        }
    }
    println!("kernels selected: {:#?}", gpu.kernels);
    assert!(failures.is_empty(), "{} check(s) failed:\n{}", failures.len(), failures.join("\n"));
}
