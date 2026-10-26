//! Vocab-parallel lm_head (tensor parallelism), the bit-exactness gate of the split: the logits of head rows `[0, V/2)` and
//! `[V/2, V)` computed as two separate half-size calls (each rank's call: its own weight allocation, its own row count) are
//! BITWISE the two halves of the logits of one full-size call, for every row count a head group can have (1..=16, the decode
//! GEMV, the multi-row GEMV and the WMMA tile kernels), K of the 35B (2048), the 8B (4096) and the 27B (5120 is covered by the
//! selection test in `kernels.rs`), the dense bf16 head and the NVFP4 head (f32 logits, the shipped heads), and the tied-head
//! form where the second half is a row view into the first call's weight allocation. Same `Registry::select` the model uses.
//!
//! Needs a GPU: without one the tests print SKIPPED (and fail under CI=1 / INTARSIA_REQUIRE_GPU=1).
#![cfg(ik_hip)]

mod common;

use intarsia_core::backend::{Backend, MemKind, TensorRef};
use intarsia_core::quant::{nvfp4, HostQTensor, QTensor, BF16};
use intarsia_core::rng::SplitMix64;
use intarsia_core::{DType, HostTensor, Registry, ShapeInfo};
use std::sync::Arc;

fn device(test: &str) -> Option<Arc<dyn Backend>> {
    let ord = std::env::var("INTARSIA_HIP_DEVICE").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
    match intarsia_backend_hip::create(ord) {
        Ok(be) => Some(be),
        Err(e) => {
            common::skip(test, &format!("no usable HIP device ({e})"));
            None
        }
    }
}

/// bf16 `[n, k]` weights of small, finite, mixed-sign values (fast: random sign / exponent / mantissa).
fn bf16_weights(rng: &mut SplitMix64, n: usize, k: usize) -> HostQTensor {
    let data: Vec<u8> = (0..n * k)
        .flat_map(|_| {
            let r = rng.next_u64();
            let e = 0x70 + (r % 14) as u16;
            (((r >> 8) as u16 & 1) << 15 | e << 7 | (r >> 16) as u16 & 0x7f).to_le_bytes()
        })
        .collect();
    HostQTensor { format: BF16, n, k, parts: vec![HostTensor::new(DType::Bf16, vec![n, k], data).unwrap()], act_quant: false }
}

/// Rows `[r * n/world, (r + 1) * n/world)` of `q` (the parts with `n` rows are cut, the per-tensor scalars stay).
fn rows_of(q: &HostQTensor, r: usize, world: usize) -> HostQTensor {
    let nl = q.n / world;
    let parts = q
        .parts
        .iter()
        .map(|p| {
            if p.shape.first() == Some(&q.n) {
                let row = p.nbytes() / q.n;
                let mut shape = p.shape.clone();
                shape[0] = nl;
                HostTensor::new(p.dtype, shape, p.data[r * nl * row..(r + 1) * nl * row].to_vec()).unwrap()
            } else {
                p.clone()
            }
        })
        .collect();
    HostQTensor { format: q.format, n: nl, k: q.k, parts, act_quant: q.act_quant }
}

struct Rig {
    be: Arc<dyn Backend>,
    reg: Registry,
}

impl Rig {
    fn new(be: Arc<dyn Backend>) -> Rig {
        let reg = Registry::new(be.name(), &be.info(), be.kernels());
        Rig { be, reg }
    }

    /// `x @ W^T` as `[m, w.n]` f32 bytes, through the kernel the registry picks; also returns that kernel's name.
    fn logits(&self, w: &QTensor, x: &TensorRef) -> (Vec<u8>, String) {
        let m = x.shape[0];
        let mut init = HostTensor::zeros(DType::F32, &[m, w.n]);
        init.data.fill(0xA5);
        let buf = self.be.alloc(init.nbytes().max(16), MemKind::Device).unwrap();
        self.be.upload(&buf, 0, &init.data, 0).unwrap();
        let out = TensorRef::new(buf, DType::F32, &[m, w.n]);
        let mut call = w.call(x.clone(), out.clone());
        call.check().unwrap();
        let k = self.reg.select(&call).unwrap_or_else(|e| panic!("no kernel for m={m} n={}: {e}", w.n));
        let need = k.scratch_bytes(&ShapeInfo::from_call(&call));
        if need > 0 {
            let b = self.be.alloc(need.max(16), MemKind::Device).unwrap();
            call.scratch = Some(TensorRef::new(b, DType::U8, &[need]));
        }
        self.be.launch(&k.name, &call, 0).unwrap_or_else(|e| panic!("{}: {e}", k.name));
        self.be.synchronize(0).unwrap();
        let mut data = vec![0u8; out.nbytes()];
        self.be.download(&out.buf, out.offset, &mut data, 0).unwrap();
        (data, k.name.clone())
    }
}

/// `full` is the whole head; split it into halves like two ranks would load them and compare the logits of every row count.
fn check(rig: &Rig, full: &HostQTensor, tied_view: bool, rng: &mut SplitMix64) {
    let (n, k) = (full.n, full.k);
    let vh = n / 2;
    let w_full = full.upload(&*rig.be).unwrap();
    let halves: Vec<QTensor> = (0..2).map(|r| rows_of(full, r, 2).upload(&*rig.be).unwrap()).collect();
    // the tied form: rank 1's head is a row view of one allocation holding the whole table
    let view = |r: usize| {
        let mut q = w_full.clone();
        q.n = vh;
        q.parts[0] = q.parts[0].rows(r * vh, vh);
        q
    };
    let mut kernels = std::collections::BTreeSet::new();
    for m in 1..=16usize {
        let xs = rng.normal_vec(m * k, 1.0);
        let xh = HostTensor::from_f32_as(DType::Bf16, &[m, k], &xs);
        let buf = rig.be.alloc(xh.nbytes().max(16), MemKind::Device).unwrap();
        rig.be.upload(&buf, 0, &xh.data, 0).unwrap();
        let x = TensorRef::new(buf, DType::Bf16, &[m, k]);
        let (lf, kf) = rig.logits(&w_full, &x);
        kernels.insert(kf.clone());
        for r in 0..2 {
            let w = if tied_view { view(r) } else { halves[r].clone() };
            let (lh, kh) = rig.logits(&w, &x);
            assert_eq!(kf, kh, "m={m}: the split call runs another kernel than the full one");
            for i in 0..m {
                let (a, b) = (&lf[(i * n + r * vh) * 4..(i * n + (r + 1) * vh) * 4], &lh[i * vh * 4..(i + 1) * vh * 4]);
                assert!(a == b, "{kf} m={m} row {i} half {r} (n={n} k={k}, tied view {tied_view}): split logits differ from the full head's");
            }
        }
    }
    println!("vocab split n={n} k={k} tied_view={tied_view}: kernels {kernels:?}, m=1..16 bitwise");
}

#[test]
fn split_bf16_head_is_bitwise_the_full_head() {
    let Some(be) = device("split_bf16_head_is_bitwise_the_full_head") else { return };
    let rig = Rig::new(be);
    let mut rng = SplitMix64::new(61);
    for (n, k) in [(2 * 4100, 2048usize), (2 * 4100, 4096), (2 * 19_000, 4096), (2 * 37_700, 2048)] {
        let full = bf16_weights(&mut rng, n, k);
        check(&rig, &full, false, &mut rng);
        check(&rig, &full, true, &mut rng);
    }
}

#[test]
fn split_nvfp4_head_is_bitwise_the_full_head() {
    let Some(be) = device("split_nvfp4_head_is_bitwise_the_full_head") else { return };
    let rig = Rig::new(be);
    let mut rng = SplitMix64::new(62);
    for (n, k) in [(2 * 4100, 2048usize), (2 * 37_700, 2048), (2 * 19_000, 4096)] {
        let full = nvfp4::random_rows(&mut rng, n, k);
        check(&rig, &full, false, &mut rng);
    }
}
