//! G-B, the row-exactness gate of hybrid continuous batching on HIP (lane hb, stage R0): row r of an M-row call of a
//! `rows_exact` kernel is BITWISE equal to row 0 of the 1-row kernel the solo decode step runs (`matmul_fp8_gemv_v2`,
//! `matmul_nvfp4_gemv_v2`, `matmul_bf16_gemv`, `logits_bf16_gemv`, `moe_experts_bf16_v2`) on row r's input alone; for every M
//! from 1 to the cap (16), every row r, on the real Qwen3.6-35B-A3B and Qwen3-8B projection shapes and on odd edges. No
//! tolerance anywhere: outputs are compared as bytes.
//!
//! Inputs are hand built to be nasty: normal activations, an all-zero row, a row identical to row 0, a row spanning 40 binades,
//! rows that saturate e4m3 under the W8A8 input scale (and sit exactly at the clip boundary), a single huge element, tiny
//! magnitudes. W8A8 (input_scale > 0) and W8A16 (input_scale == 0) both. The rows of one call are also run in reversed order.
//! MoE: distinct routing, all tokens on the same experts, shared experts in different top-k order, invalid ids (-1, >= E), zero
//! routing weights.
//!
//! The WMMA / mma paths (fp8 M >= 5, nvfp4 M >= 9, bf16 M >= 5, MoE T >= 16) are not row-exact against the one-row GEMV; they
//! are what the ordinary registrations keep for prefill and what `rows_exact` kernels never take.
//!
//! Needs a GPU: without one the tests print SKIPPED (and fail under CI=1 / INTARSIA_REQUIRE_GPU=1).
#![cfg(ik_hip)]

mod common;

use intarsia_core::backend::{Backend, MemKind, TensorRef};
use intarsia_core::quant::cases::{random_experts, random_routing};
use intarsia_core::quant::fp8::scales_tensor;
use intarsia_core::quant::nvfp4;
use intarsia_core::rng::SplitMix64;
use intarsia_core::{DType, HostTensor, Op, OpAttrs, OpCall, ShapeInfo};
use std::collections::BTreeMap;
use std::sync::Arc;

const CAP: usize = 16;

struct Rig {
    be: Arc<dyn Backend>,
    /// (kernel, M) -> (shape cases, rows compared, bytes compared)
    seen: std::cell::RefCell<BTreeMap<(String, usize), (usize, usize, usize)>>,
}

impl Rig {
    fn new(test: &str) -> Option<Rig> {
        let ord = std::env::var("INTARSIA_HIP_DEVICE").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
        match intarsia_backend_hip::create(ord) {
            Ok(be) => Some(Rig { be, seen: Default::default() }),
            Err(e) => {
                common::skip(test, &format!("no usable HIP device ({e})"));
                None
            }
        }
    }

    fn put(&self, t: &HostTensor) -> TensorRef {
        let buf = self.be.alloc(t.nbytes().max(16), MemKind::Device).unwrap();
        self.be.zero(&buf, 0).unwrap();
        if t.nbytes() > 0 {
            self.be.upload(&buf, 0, &t.data, 0).unwrap();
        }
        TensorRef::new(buf, t.dtype, &t.shape)
    }

    /// A dense fp8 weight; with INTARSIA_WG_FRAG=1 rearranged into fragment order (`Layout::Fp8Frag`) when the shape allows it, so the
    /// row-invariance proof of the WMMA family runs on tagged weights too (lane wg).
    fn put_w(&self, t: &HostTensor) -> TensorRef {
        let w = self.put(t);
        if std::env::var("INTARSIA_WG_FRAG").as_deref() == Ok("1")
            && t.dtype == DType::Fp8E4m3
            && t.shape.len() == 2
            && intarsia_core::quant::fp8::frag_eligible(t.shape[0], t.shape[1])
        {
            self.be.relayout_fp8_frag(&w).unwrap();
            assert_eq!(w.layout(), intarsia_core::backend::Layout::Fp8Frag);
        }
        w
    }

    /// Launch `kernel` on already uploaded inputs; the outputs start as a 0xA5 pattern (a NaN in bf16 / f32), so a row the
    /// kernel does not write cannot equal a computed one.
    fn run(&self, kernel: &str, op: Op, inputs: &[TensorRef], outs: &[(DType, Vec<usize>)], attrs: &OpAttrs) -> Vec<HostTensor> {
        let mut outputs = Vec::new();
        for (dt, shape) in outs {
            let mut t = HostTensor::zeros(*dt, shape);
            t.data.fill(0xA5);
            outputs.push(self.put(&t));
        }
        let mut call = OpCall::new(op, inputs.to_vec(), outputs.clone(), attrs.clone());
        call.check().unwrap();
        if let Some(k) = self.be.kernels().iter().find(|k| k.name == kernel && k.op == op) {
            let need = k.scratch_bytes(&ShapeInfo::from_call(&call));
            if need > 0 {
                let buf = self.be.alloc(need.max(16), MemKind::Device).unwrap();
                call.scratch = Some(TensorRef::new(buf, DType::U8, &[need]));
            }
        }
        self.be.launch(kernel, &call, 0).unwrap_or_else(|e| panic!("{kernel}: {e}"));
        self.be.synchronize(0).unwrap();
        let mut res = Vec::new();
        for (o, (dt, shape)) in outputs.iter().zip(outs) {
            let mut data = vec![0u8; o.nbytes()];
            self.be.download(&o.buf, o.offset, &mut data, 0).unwrap();
            res.push(HostTensor::new(*dt, shape.clone(), data).unwrap());
        }
        res
    }

    fn note(&self, kernel: &str, m: usize, rows: usize, bytes: usize) {
        let mut s = self.seen.borrow_mut();
        let e = s.entry((kernel.to_string(), m)).or_default();
        e.0 += 1;
        e.1 += rows;
        e.2 += bytes;
    }

    fn report(&self, what: &str) {
        let mut line = String::new();
        let mut cur = String::new();
        for ((k, m), (cases, rows, bytes)) in self.seen.borrow().iter() {
            if *k != cur {
                if !line.is_empty() {
                    println!("{line}");
                }
                cur = k.clone();
                line = format!("G-B {what} {k}: M:cases/rows/bytes");
            }
            line.push_str(&format!(" {m}:{cases}/{rows}/{bytes}"));
        }
        println!("{line}");
    }
}

// ---- inputs ---------------------------------------------------------------------------------------------------------

fn bf16_row(vals: &[f32]) -> HostTensor {
    HostTensor::from_f32_as(DType::Bf16, &[1, vals.len()], vals)
}

/// Sixteen nasty activation rows of width `k`. `clip` is the value at which e4m3 saturates under the W8A8 input scale in use
/// (448 * input_scale; pass 448 for W8A16 / unit scale).
fn nasty_rows(rng: &mut SplitMix64, k: usize, clip: f32) -> Vec<Vec<f32>> {
    nasty_rows_n(rng, k, clip, CAP)
}

/// `n` (at least 8) nasty rows.
fn nasty_rows_n(rng: &mut SplitMix64, k: usize, clip: f32, n: usize) -> Vec<Vec<f32>> {
    let normal = |rng: &mut SplitMix64, s: f32| rng.normal_vec(k, s);
    let r0 = normal(rng, 1.0);
    let mut rows: Vec<Vec<f32>> = vec![r0.clone(), vec![0.0; k], r0.clone()];
    // 40 binades of dynamic range, both signs
    let wide: Vec<f32> = (0..k)
        .map(|_| {
            let e = (rng.next_u64() % 41) as i32 - 20;
            let v = rng.normal_vec(1, 1.0)[0];
            v * (2.0f32).powi(e)
        })
        .collect();
    rows.push(wide);
    // saturating: |x| up to 20x past the e4m3 clip for W8A8
    rows.push(rng.normal_vec(k, clip * 6.0));
    // exactly on the clip boundary and just past it (half-way rounding region), alternating sign
    rows.push((0..k).map(|i| if i % 2 == 0 { clip } else { -clip * 1.0625 }).collect());
    // one huge element (last), the rest zero
    let mut single = vec![0.0f32; k];
    single[k - 1] = clip * 1000.0;
    rows.push(single);
    // tiny magnitudes (underflow of the e4m3 quantiser)
    rows.push(rng.normal_vec(k, clip * 1e-6));
    // the rest: random scales over six decades, every other one with a +-5 binade spread per element (inside the e4m3 range
    // after quantisation, so the products span many binades and the sums round)
    while rows.len() < n {
        let s = (10.0f32).powf((rng.next_f32() - 0.5) * 6.0);
        if rows.len() % 2 == 0 {
            let spread: Vec<f32> = (0..k)
                .map(|_| {
                    let e = (rng.next_u64() % 11) as i32 - 5;
                    rng.normal_vec(1, 1.0)[0] * (clip / 64.0) * (2.0f32).powi(e)
                })
                .collect();
            rows.push(spread);
        } else {
            rows.push(normal(rng, s));
        }
    }
    rows
}

/// Weights whose products span many binades, so that the f32 sums of the GEMVs ROUND and their order matters (with the narrow
/// model-like data most partial sums are exact and a reordering would not show: the first negative control proved it).
fn wide_fp8_weights(rng: &mut SplitMix64, n: usize, k: usize) -> HostTensor {
    let data: Vec<u8> = (0..n * k)
        .map(|_| {
            let b = rng.next_u64() as u8;
            if b & 0x7F == 0x7F {
                b ^ 1 // no e4m3 NaN
            } else {
                b
            }
        })
        .collect();
    HostTensor::new(DType::Fp8E4m3, vec![n, k], data).unwrap()
}

/// NVFP4 with block scales over the whole finite positive e4m3 range (0x08 .. 0x7e: 2^-6 .. 448) instead of 0.5 .. 2.25.
fn wide_nvfp4(rng: &mut SplitMix64, n: usize, k: usize) -> Vec<HostTensor> {
    let mut q = nvfp4::random_rows(rng, n, k);
    for b in q.parts[1].data.iter_mut() {
        *b = 0x08 + (rng.next_u64() % 0x77) as u8;
    }
    q.parts[2] = HostTensor::from_f32(&[1], &[1.0 / (64.0 * (k as f32).sqrt())]);
    q.parts
}

/// bf16 weights, each scaled by a random power of two in 2^-8 .. 2^8.
fn wide_bf16(rng: &mut SplitMix64, n: usize, k: usize) -> HostTensor {
    let mut t = fast_bf16(rng, n, k);
    for c in t.data.chunks_mut(2) {
        let e = (rng.next_u64() % 17) as i32 - 8;
        let v = f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16) * (2.0f32).powi(e);
        c.copy_from_slice(&((v.to_bits() >> 16) as u16).to_le_bytes());
    }
    t
}

fn clip_of(scales: &HostTensor) -> f32 {
    let is = scales.to_f32()[1];
    if is > 0.0 {
        448.0 * is
    } else {
        448.0
    }
}

/// Negative controls (recorded in the results doc, not a gate): `INTARSIA_HIP_GB_CONTROL=matmul_fp8_gemv_rows=matmul_fp8_mma_f8,...`
/// runs the named ordinary kernel where the driver would run the `_rows` one; the driver must then FAIL, which shows the test can.
fn control(rows: &str) -> String {
    std::env::var("INTARSIA_HIP_GB_CONTROL")
        .ok()
        .and_then(|v| v.split(',').find_map(|p| p.split_once('=').filter(|(a, _)| *a == rows).map(|(_, b)| b.to_string())))
        .unwrap_or_else(|| rows.to_string())
}

// ---- the G-B driver for the matmul-like ops ---------------------------------------------------------------------------

/// `weights`: the inputs after x (w, scales ... as the op takes them). `solo`: the 1-row kernel, `rows`: the row-exact one.
#[allow(clippy::too_many_arguments)]
fn gb_matmul(
    rig: &Rig,
    op: Op,
    solo: &str,
    rows: &str,
    weights: &[TensorRef],
    x_rows: &[Vec<f32>],
    n: usize,
    out_dt: DType,
    label: &str,
    caps: usize,
) {
    let rows_s = control(rows);
    let rows = rows_s.as_str();
    let k = x_rows[0].len();
    let a = OpAttrs::default();
    let osz = out_dt.size_bytes();
    // the solo reference: every row alone through the 1-row kernel
    let mut want: Vec<Vec<u8>> = Vec::new();
    for r in 0..caps {
        let x = rig.put(&bf16_row(&x_rows[r]));
        let mut ins = vec![x];
        ins.extend_from_slice(weights);
        let o = rig.run(solo, op, &ins, &[(out_dt, vec![1, n])], &a);
        want.push(o[0].data.clone());
    }
    let all: Vec<f32> = x_rows[..caps].iter().flatten().copied().collect();
    let x_all = rig.put(&HostTensor::from_f32_as(DType::Bf16, &[caps, k], &all));
    for m in 1..=caps {
        let mut ins = vec![x_all.rows(0, m)];
        ins.extend_from_slice(weights);
        let o = rig.run(rows, op, &ins, &[(out_dt, vec![m, n])], &a);
        for r in 0..m {
            let got = &o[0].data[r * n * osz..(r + 1) * n * osz];
            if got != &want[r][..] {
                // diagnostics: are the two launches repeatable?
                let x1 = rig.put(&bf16_row(&x_rows[r]));
                let mut ins1 = vec![x1];
                ins1.extend_from_slice(weights);
                let again_solo = rig.run(solo, op, &ins1, &[(out_dt, vec![1, n])], &a)[0].data.clone();
                let mut ins2 = vec![x_all.rows(0, m)];
                ins2.extend_from_slice(weights);
                let again_rows = rig.run(rows, op, &ins2, &[(out_dt, vec![m, n])], &a)[0].data[r * n * osz..(r + 1) * n * osz].to_vec();
                eprintln!("DIAG solo repeatable: {}, rows repeatable: {}", again_solo == want[r], again_rows == got);
                let bad = got.chunks(osz).zip(want[r].chunks(osz)).enumerate().filter(|(_, (g, w))| g != w).count();
                let first = got.chunks(osz).zip(want[r].chunks(osz)).position(|(g, w)| g != w).unwrap();
                panic!(
                    "G-B FAIL {label}: {rows} M={m} row {r} differs from {solo} on that row alone in {bad} of {n} elements \
                     (first at col {first}: {:02x?} vs {:02x?})",
                    &got[first * osz..(first + 1) * osz],
                    &want[r][first * osz..(first + 1) * osz]
                );
            }
        }
        rig.note(rows, m, m, m * n * osz);
    }
    // the rows of one full call in reversed order: position must not matter
    let rev: Vec<f32> = x_rows[..caps].iter().rev().flatten().copied().collect();
    let x_rev = rig.put(&HostTensor::from_f32_as(DType::Bf16, &[caps, k], &rev));
    let mut ins = vec![x_rev];
    ins.extend_from_slice(weights);
    let o = rig.run(rows, op, &ins, &[(out_dt, vec![caps, n])], &a);
    for r in 0..caps {
        let got = &o[0].data[r * n * osz..(r + 1) * n * osz];
        assert!(got == &want[caps - 1 - r][..], "G-B FAIL {label}: {rows} reversed order, row {r}");
    }
}

// ---- fp8 -------------------------------------------------------------------------------------------------------------

#[test]
fn fp8_rows_are_bitwise_the_one_row_gemv() {
    let Some(rig) = Rig::new("fp8_rows_are_bitwise_the_one_row_gemv") else { return };
    let mut rng = SplitMix64::new(801);
    // (n, k): Qwen3.6-35B-A3B GDN in_qkv 8192x2048, in_z / attention q 4096x2048, k / v 512x2048, GDN out + o_proj 2048x4096,
    // plus edges: n not a multiple of the 8 warps, K = 16, K with a ragged lane tail (129 blocks of 16), one row, K spanning
    // several staging chunks at M = 16 (1024 elements each: 6144, and 16400 = 16 chunks + a tail of 16), a K just past one chunk
    let shapes = [
        (8192usize, 2048usize),
        (4096, 2048),
        (512, 2048),
        (2048, 4096),
        (300, 2048),
        (9, 16),
        (33, 2064),
        (1, 4096),
        (64, 6144),
        (16, 16400),
        (8, 1040),
        // --tp 2 shards of the 35B (lane tp2/batch: batched decode under tensor parallelism takes these kernels at the halved
        // shapes): GDN in_z / out_proj and o_proj at K / 2 (2048x2048), k / v per rank (256x2048)
        (2048, 2048),
        (256, 2048),
    ];
    for (n, k) in shapes {
        // (act quant, input scale): W8A8 with a few scales (incl. one that makes everything saturate / underflow), W8A16
        for (wide, aq, is) in [(false, true, 0.125f32), (true, true, 0.125), (true, true, 0.0117), (true, true, 3.0), (false, false, 0.0), (true, false, 0.0)] {
            let hc = intarsia_core::quant::cases::matmul_fp8_case(&mut rng, 1, n, k, aq, DType::Bf16, DType::Bf16);
            let ws = hc.inputs[2].to_f32()[0];
            let scales = scales_tensor(if wide { ws / 16.0 } else { ws }, if aq { is } else { 0.0 });
            let w = rig.put(&if wide { wide_fp8_weights(&mut rng, n, k) } else { hc.inputs[1].clone() });
            let sc = rig.put(&scales);
            let x_rows = nasty_rows(&mut rng, k, clip_of(&scales));
            for out_dt in [DType::Bf16, DType::F32] {
                let label = format!("fp8 n={n} k={k} wide={wide} aq={aq} is={is} out={out_dt:?}");
                gb_matmul(&rig, Op::MatmulFp8, "matmul_fp8_gemv_v2", "matmul_fp8_gemv_rows", &[w.clone(), sc.clone()], &x_rows, n, out_dt, &label, CAP);
            }
        }
    }
    rig.report("fp8");
}

/// Lane wg: the WMMA family (kernels/hip/ik_f8_rows.h). Inside the family row r of an M-row call is bitwise the one-row call on row r
/// alone, for every M = 1..=64, whatever the neighbouring rows hold (zeros, saturating, 40 binades, a huge element ...), in W8A8 and
/// W8A16, bf16 and f32 out, at K with a 16-tail (K % 32 == 16), K past one staging window, N not a multiple of 16. The reference is
/// the family's own M = 1 launch (`matmul_fp8_wmma`; the rows registration is the same launcher). Negative control:
/// `INTARSIA_HIP_GB_CONTROL=matmul_fp8_wmma_rows=matmul_fp8_mma_f8 cargo test ... wmma_fp8` must FAIL.
#[test]
fn wmma_fp8_rows_are_bitwise_the_one_row_call() {
    let Some(rig) = Rig::new("wmma_fp8_rows_are_bitwise_the_one_row_call") else { return };
    let mut rng = SplitMix64::new(811);
    let wmma_rows = 64;
    // 35B shapes, then the edges: N tails, K = 16 / 48 (tail only / full chunk + tail), K past a window at M = 64 (> 768) and at M = 16
    let shapes = [
        (8192usize, 2048usize),
        (512, 2048),
        (2048, 4096),
        (256, 2048),
        (300, 2048),
        (9, 16),
        (33, 48),
        (1, 4096),
        (64, 6144),
        (40, 16400),
        (24, 1040),
        (48, 2064),
    ];
    for (n, k) in shapes {
        for (wide, aq, is) in [(true, true, 0.125f32), (true, true, 0.0117), (false, true, 3.0), (true, false, 0.0), (false, false, 0.0)] {
            let hc = intarsia_core::quant::cases::matmul_fp8_case(&mut rng, 1, n, k, aq, DType::Bf16, DType::Bf16);
            let ws = hc.inputs[2].to_f32()[0];
            let scales = scales_tensor(if wide { ws / 16.0 } else { ws }, if aq { is } else { 0.0 });
            let w = rig.put_w(&if wide { wide_fp8_weights(&mut rng, n, k) } else { hc.inputs[1].clone() });
            let sc = rig.put(&scales);
            let x_rows = nasty_rows_n(&mut rng, k, clip_of(&scales), wmma_rows);
            for out_dt in [DType::Bf16, DType::F32] {
                let label = format!("wmma fp8 n={n} k={k} wide={wide} aq={aq} is={is} out={out_dt:?}");
                gb_matmul(&rig, Op::MatmulFp8, "matmul_fp8_wmma", "matmul_fp8_wmma_rows", &[w.clone(), sc.clone()], &x_rows, n, out_dt, &label, wmma_rows);
            }
        }
    }
    rig.report("wmma fp8");
}

/// The bf16 half of the family (router / gate / GDN a, b): row r of an M = 1..=64 call is bitwise the one-row call, with nasty neighbours,
/// wide weights, a 16-tail K and N tails, bf16 and f32 out. Control: `INTARSIA_HIP_GB_CONTROL=matmul_bf16_wmma_rows=matmul_bf16` must FAIL.
#[test]
fn wmma_bf16_rows_are_bitwise_the_one_row_call() {
    let Some(rig) = Rig::new("wmma_bf16_rows_are_bitwise_the_one_row_call") else { return };
    let mut rng = SplitMix64::new(813);
    let wmma_rows = 64;
    for (n, k) in [(256usize, 2048usize), (32, 2048), (1, 2048), (300, 4096), (9, 16), (33, 48), (64, 6144), (40, 16400), (130, 2064)] {
        for wide in [false, true] {
            let w = rig.put(&if wide { wide_bf16(&mut rng, n, k) } else { fast_bf16(&mut rng, n, k) });
            let x_rows = nasty_rows_n(&mut rng, k, 448.0, wmma_rows);
            let label = format!("wmma bf16 n={n} k={k} wide={wide}");
            gb_matmul(&rig, Op::Matmul, "matmul_bf16_wmma", "matmul_bf16_wmma_rows", &[w.clone()], &x_rows, n, DType::Bf16, &label, wmma_rows);
            gb_matmul(&rig, Op::Matmul, "matmul_bf16_wmma", "matmul_bf16_wmma_rows", &[w], &x_rows, n, DType::F32, &label, wmma_rows);
        }
    }
    rig.report("wmma bf16");
}

/// Lane wg, the dispatch-table boundary (`WmmaTable`, quantmoe.rs): a kind that switches kernels at M = lo (INTARSIA_HIP_WMMA_TABLE=fp8a8=lo-16)
/// serves rows 1..lo-1 from the GEMV and rows lo.. from the WMMA family. For fp8 W8A8 this only REPORTS whether a WMMA row equals the GEMV
/// one-row call on this data: window A (2026-10-25) found them bitwise equal here although the standalone bench shows rel_l2 ~ 1e-6 to 3e-5 on
/// random data (the products are exact in fp8 x fp8 and the gate's sums stay inside 24 bits), so this data cannot pin "not bitwise" and the
/// test must not assert it. The bf16 twin below does assert it (its products round).
#[test]
fn wmma_fp8_switch_from_the_gemv_reports_equality() {
    let Some(rig) = Rig::new("wmma_fp8_switch_from_the_gemv_reports_equality") else { return };
    let mut rng = SplitMix64::new(821);
    let (n, k) = (512usize, 2048usize);
    let hc = intarsia_core::quant::cases::matmul_fp8_case(&mut rng, 1, n, k, true, DType::Bf16, DType::Bf16);
    let scales = scales_tensor(hc.inputs[2].to_f32()[0] / 16.0, 0.0117);
    let w = rig.put(&wide_fp8_weights(&mut rng, n, k));
    let sc = rig.put(&scales);
    let x_rows = nasty_rows_n(&mut rng, k, clip_of(&scales), 16);
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        gb_matmul(&rig, Op::MatmulFp8, "matmul_fp8_gemv_v2", "matmul_fp8_wmma_rows", &[w, sc], &x_rows, n, DType::Bf16, "switch fp8 W8A8 gemv -> wmma", 16)
    }));
    match r {
        Ok(()) => eprintln!("REPORT fp8 W8A8: WMMA rows are bitwise the GEMV one-row call on this data (not a proof for other data)"),
        Err(p) => {
            let msg = p.downcast_ref::<String>().cloned().or_else(|| p.downcast_ref::<&str>().map(|s| s.to_string())).unwrap_or_default();
            assert!(msg.contains("G-B FAIL"), "the comparison must fail by differing bits, not die some other way: {msg}");
            eprintln!("REPORT fp8 W8A8: WMMA rows differ from the GEMV one-row call: {msg}");
        }
    }
}

/// The same for the bf16 router/gate kind (INTARSIA_HIP_WMMA_TABLE=bf16=lo-16).
#[test]
fn wmma_bf16_switch_from_the_gemv_is_not_bitwise() {
    let Some(rig) = Rig::new("wmma_bf16_switch_from_the_gemv_is_not_bitwise") else { return };
    let mut rng = SplitMix64::new(823);
    let (n, k) = (256usize, 2048usize);
    let w = rig.put(&wide_bf16(&mut rng, n, k));
    let x_rows = nasty_rows_n(&mut rng, k, 448.0, 16);
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        gb_matmul(&rig, Op::Matmul, "matmul_bf16_gemv", "matmul_bf16_wmma_rows", &[w], &x_rows, n, DType::Bf16, "switch bf16 gemv -> wmma", 16)
    }));
    let msg = match r {
        Ok(()) => panic!("the WMMA rows are bitwise the GEMV one-row call: the table switch is not a numerics change after all"),
        Err(p) => p.downcast_ref::<String>().cloned().or_else(|| p.downcast_ref::<&str>().map(|s| s.to_string())).unwrap_or_default(),
    };
    assert!(msg.contains("G-B FAIL"), "the switch must fail the bitwise comparison, not die some other way: {msg}");
}

/// The family agrees with the dequantised reference (the cpu `ref` arithmetic) as well as the GEMV does: rel_l2 against the f64 sum of
/// the exact products. Not a bitwise gate; it catches a wrong fragment layout or k permutation, which row invariance cannot.
#[test]
fn wmma_fp8_matches_the_reference_sum() {
    let Some(rig) = Rig::new("wmma_fp8_matches_the_reference_sum") else { return };
    let mut rng = SplitMix64::new(812);
    for (n, k) in [(256usize, 2048usize), (300, 1040), (48, 48), (64, 6144), (96, 16400)] {
        for (aq, is) in [(true, 0.125f32), (false, 0.0)] {
            let hc = intarsia_core::quant::cases::matmul_fp8_case(&mut rng, 1, n, k, aq, DType::Bf16, DType::Bf16);
            let ws = hc.inputs[2].to_f32()[0];
            let scales = scales_tensor(ws, if aq { is } else { 0.0 });
            let wh = hc.inputs[1].clone();
            let w = rig.put(&wh);
            let sc = rig.put(&scales);
            let m = 5;
            let x_rows = nasty_rows_n(&mut rng, k, clip_of(&scales), m);
            let flat: Vec<f32> = x_rows[..m].iter().flatten().copied().collect();
            let x = rig.put(&HostTensor::from_f32_as(DType::Bf16, &[m, k], &flat));
            let got = rig.run("matmul_fp8_wmma", Op::MatmulFp8, &[x.clone(), w, sc], &[(DType::F32, vec![m, n])], &OpAttrs::default())[0].to_f32();
            // reference: the cpu quantiser + exact f64 sum
            let xq: Vec<f32> = HostTensor::from_f32_as(DType::Bf16, &[m, k], &flat).to_f32();
            let wf = wh.to_f32();
            let (s_w, s_in) = (scales.to_f32()[0] as f64, scales.to_f32()[1] as f64);
            let mut num = 0f64;
            let mut den = 0f64;
            for r in 0..m {
                for c in 0..n {
                    let mut acc = 0f64;
                    for i in 0..k {
                        let xv = xq[r * k + i] as f64;
                        let xv = if s_in > 0.0 { intarsia_core::dtype::fp8_e4m3_to_f32(intarsia_core::dtype::f32_to_fp8_e4m3((xv / s_in) as f32)) as f64 } else { xv };
                        acc += xv * wf[c * k + i] as f64;
                    }
                    let want = acc * s_w * if s_in > 0.0 { s_in } else { 1.0 };
                    let d = got[r * n + c] as f64 - want;
                    num += d * d;
                    den += want * want;
                }
            }
            let rel = (num / den.max(1e-300)).sqrt();
            assert!(rel < 2e-4, "wmma fp8 n={n} k={k} aq={aq}: rel_l2 {rel:.3e} against the f64 reference");
        }
    }
}

// ---- nvfp4 -----------------------------------------------------------------------------------------------------------

#[test]
fn nvfp4_rows_are_bitwise_the_one_row_gemv() {
    let Some(rig) = Rig::new("nvfp4_rows_are_bitwise_the_one_row_gemv") else { return };
    let mut rng = SplitMix64::new(802);
    // (n, k, out): the 35B shared expert (512x2048, 2048x512), its lm_head (248320x2048, f32 logits; a thin-K full-vocab case
    // for the grid limit on N), odd edges (see tests/fused_decode.rs for the glu shapes)
    let shapes = [
        (512usize, 2048usize, DType::Bf16),
        (2048, 512, DType::Bf16),
        (248320, 2048, DType::F32),
        (248320, 64, DType::F32),
        (300, 4112, DType::Bf16),
        (9, 16, DType::F32),
        (1030, 160, DType::Bf16),
        (1, 16, DType::Bf16),
        // --tp 2 shards: the shared expert per rank (256x2048, 2048x256) and the vocab-parallel head's half (124160x2048)
        (256, 2048, DType::Bf16),
        (2048, 256, DType::Bf16),
        (124160, 2048, DType::F32),
    ];
    for (n, k, model_out) in shapes {
        for wide in [false, true] {
            let parts = if wide { wide_nvfp4(&mut rng, n, k) } else { nvfp4::random_rows(&mut rng, n, k).parts };
            let weights: Vec<TensorRef> = parts.iter().map(|p| rig.put(p)).collect();
            let x_rows = nasty_rows(&mut rng, k, 448.0);
            // the model's output dtype first, then the other one (f32 outputs expose a last-bit difference bf16 would round away)
            for out_dt in [model_out, if model_out == DType::F32 { DType::Bf16 } else { DType::F32 }] {
                gb_matmul(&rig, Op::MatmulNvfp4, "matmul_nvfp4_gemv_v2", "matmul_nvfp4_gemv_rows", &weights, &x_rows, n, out_dt, &format!("nvfp4 n={n} k={k} wide={wide} {out_dt:?}"), CAP);
            }
        }
    }
    rig.report("nvfp4");
}

// ---- mxfp4 (Qwen3.8-27B) ---------------------------------------------------------------------------------------------

/// MXFP4 rows (W4A4 dynamic and W4A16) are bitwise the one-row GEMV, M = 1..=16 (the `rows` kernel's cap, ROWS_MAX), on the 27B's
/// projection shapes and odd edges; the pair kernel too.
#[test]
fn mxfp4_rows_are_bitwise_the_one_row_gemv() {
    let Some(rig) = Rig::new("mxfp4_rows_are_bitwise_the_one_row_gemv") else { return };
    let mut rng = SplitMix64::new(803);
    // Qwen3.8-27B: attention q (gated) 12288x5120, k / v 1024x5120, o 5120x6144, MLP gate / up 17408x5120, down 5120x17408,
    // GDN projections; edges: n not a multiple of the 8 warps, K = 32, a ragged lane tail
    let shapes = [
        (12288usize, 5120usize),
        (1024, 5120),
        (5120, 6144),
        (17408, 5120),
        (5120, 17408),
        (10240, 5120),
        (6144, 5120),
        (300, 2048),
        (9, 32),
        (33, 2080),
        (1, 4096),
        (64, 6176),
        // --tp 2 shards of the 27B (lane tp2: w15 found a batched step of 4 rows that differs from the solo step at tp=2 only):
        // column-parallel halves of q (gated) / k,v / gate,up / the GDN in projections, row-parallel halves of o / down / the GDN
        // out projection (K = 3072, 8704: not multiples of the 512- and 1024-element lane strides), the vocab-parallel head's half
        (6144, 5120),
        (512, 5120),
        (5120, 3072),
        (8704, 5120),
        (5120, 8704),
        (5120, 5120),
        (3072, 5120),
        (124160, 5120),
    ];
    for (n, k) in shapes {
        for act in [intarsia_core::quant::Mxfp4Act::Dynamic, intarsia_core::quant::Mxfp4Act::None] {
            let q = intarsia_core::quant::mxfp4::random_rows(&mut rng, n, k, act);
            let weights: Vec<TensorRef> = q.parts.iter().map(|p| rig.put(p)).collect();
            let x_rows = nasty_rows(&mut rng, k, 448.0);
            for out_dt in [DType::Bf16, DType::F32] {
                gb_matmul(
                    &rig,
                    Op::MatmulMxfp4,
                    "matmul_mxfp4_gemv",
                    "matmul_mxfp4_gemv_rows",
                    &weights,
                    &x_rows,
                    n,
                    out_dt,
                    &format!("mxfp4 n={n} k={k} act={act:?} {out_dt:?}"),
                    CAP,
                );
            }
        }
    }
    rig.report("mxfp4");
}

// ---- bf16 matmul / logits -----------------------------------------------------------------------------------------

fn fast_bf16(rng: &mut SplitMix64, n: usize, k: usize) -> HostTensor {
    // uniform in (-1, 1) / sqrt(k): cheap enough for the 151936 x 4096 head (the normal generator is not)
    let s = 1.0 / (k as f32).sqrt();
    let mut data = Vec::with_capacity(n * k * 2);
    for _ in 0..n * k {
        let u = (rng.next_u64() >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0;
        let b = (u * s).to_bits() >> 16;
        data.extend_from_slice(&(b as u16).to_le_bytes());
    }
    HostTensor::new(DType::Bf16, vec![n, k], data).unwrap()
}

#[test]
fn bf16_matmul_and_logits_rows_are_bitwise_the_one_row_gemv() {
    let Some(rig) = Rig::new("bf16_matmul_and_logits_rows_are_bitwise_the_one_row_gemv") else { return };
    let mut rng = SplitMix64::new(803);
    // 35B: router 256x2048, GDN in_a / in_b 32x2048, shared-expert gate 1x2048. 8B: q / o 4096x4096, k / v 1024x4096,
    // gate / up 12288x4096, down 4096x12288. Edges: K % 8 != 0 (the scalar path), K = 8, a K that is not a multiple of 256
    for (n, k) in [
        (256usize, 2048usize),
        (32, 2048),
        (1, 2048),
        (4096, 4096),
        (1024, 4096),
        (12288, 4096),
        (4096, 12288),
        (64, 4100),
        (33, 7),
        (5, 8),
        (130, 2056),
        // --tp 2: GDN in_a / in_b per rank
        (16, 2048),
    ] {
        for wide in [false, true] {
            let w = rig.put(&if wide { wide_bf16(&mut rng, n, k) } else { fast_bf16(&mut rng, n, k) });
            let x_rows = nasty_rows(&mut rng, k, 448.0);
            gb_matmul(&rig, Op::Matmul, "matmul_bf16_gemv", "matmul_bf16_gemv_rows", &[w.clone()], &x_rows, n, DType::Bf16, &format!("matmul bf16 n={n} k={k} wide={wide}"), CAP);
            // the f32-output twin on the same weights: a last-bit difference survives there
            if n * k <= 1 << 24 {
                gb_matmul(&rig, Op::Logits, "logits_bf16_gemv", "logits_bf16_gemv_rows", &[w], &x_rows, n, DType::F32, &format!("logits bf16 n={n} k={k} wide={wide}"), CAP);
            }
        }
    }
    // the 27B head (248320 x 5120, 2.5 GB): the weights are a 2048-row wide pool tiled with a shift (cheap to generate), so
    // the rows still differ; M = 9..16 run the wide kernel, 1..8 the old one
    {
        let (n, k) = (248320usize, 5120usize);
        let pool = wide_bf16(&mut rng, 2048, k);
        let shift = 2 * 1237;
        let mut data = Vec::with_capacity(n * k * 2);
        while data.len() < n * k * 2 {
            let take = (n * k * 2 - data.len()).min(pool.data.len() - shift);
            data.extend_from_slice(&pool.data[shift..shift + take]);
        }
        let w = rig.put(&HostTensor::new(DType::Bf16, vec![n, k], data).unwrap());
        let x_rows = nasty_rows(&mut rng, k, 448.0);
        gb_matmul(&rig, Op::Logits, "logits_bf16_gemv", "logits_bf16_gemv_rows", &[w], &x_rows, n, DType::F32, &format!("logits bf16 n={n} k={k} (27B head)"), CAP);
    }
    // the 8B head and an odd one, f32 logits
    for (n, k) in [(151936usize, 4096usize), (129, 36), (50000, 2048), (24000, 5120), (9, 5128)] {
        let w = rig.put(&wide_bf16(&mut rng, n, k));
        let x_rows = nasty_rows(&mut rng, k, 448.0);
        gb_matmul(&rig, Op::Logits, "logits_bf16_gemv", "logits_bf16_gemv_rows", &[w], &x_rows, n, DType::F32, &format!("logits bf16 n={n} k={k}"), CAP);
    }
    rig.report("bf16");
}

// ---- MoE decode v2 --------------------------------------------------------------------------------------------------

/// Routing for 16 tokens over `e` experts, `k` per token, in the cases listed in the module doc. (ids, weights)
fn routings(rng: &mut SplitMix64, e: usize, k: usize) -> Vec<(&'static str, Vec<i32>, Vec<f32>)> {
    let t = CAP;
    let mut out = Vec::new();
    let (ids, ws) = random_routing(rng, t, e, k);
    out.push(("distinct", ids.to_i32().unwrap(), ws.to_f32()));
    // every token on the same experts, different weights
    let first: Vec<i32> = ids.to_i32().unwrap()[..k].to_vec();
    let mut same = Vec::new();
    for _ in 0..t {
        same.extend_from_slice(&first);
    }
    let (_, ws2) = random_routing(rng, t, e, k);
    out.push(("same experts", same, ws2.to_f32()));
    // tokens 2i, 2i + 1 share their experts in different top-k order; the next pair shares half
    let base = ids.to_i32().unwrap();
    let mut shared = base.clone();
    for p in 0..t / 2 {
        let a: Vec<i32> = base[2 * p * k..(2 * p + 1) * k].to_vec();
        let mut b = a.clone();
        b.reverse();
        if p % 2 == 1 {
            for j in 0..k / 2 {
                b[j] = base[((2 * p + 5) % t) * k + j]; // half foreign experts (distinct from `a` is not required)
            }
        }
        shared[2 * p * k..(2 * p + 1) * k].copy_from_slice(&a);
        shared[(2 * p + 1) * k..(2 * p + 2) * k].copy_from_slice(&b);
    }
    out.push(("shared pairs", shared, ws.to_f32()));
    // invalid ids (-1, e, e + 100, i32::MIN) in some slots; zero routing weights in others
    let mut bad = base.clone();
    let mut wz = ws.to_f32();
    for (i, v) in bad.iter_mut().enumerate() {
        match i % 11 {
            3 => *v = -1,
            5 => *v = e as i32,
            7 if i % 2 == 0 => *v = e as i32 + 100,
            9 if i % 3 == 0 => *v = i32::MIN,
            _ => {}
        }
        if i % 13 == 4 {
            wz[i] = 0.0;
        }
    }
    out.push(("invalid ids / zero weights", bad, wz));
    out
}

/// `random_experts` with block scales over the whole finite e4m3 range and per-expert global scales over 2^-6 .. 2^6.
fn wide_experts(rng: &mut SplitMix64, e: usize, h: usize, i: usize) -> intarsia_core::quant::cases::HostExperts {
    let mut ex = random_experts(rng, e, h, i);
    for b in ex.gate_up_s.data.iter_mut().chain(ex.down_s.data.iter_mut()) {
        *b = 0x08 + (rng.next_u64() % 0x77) as u8;
    }
    let gs: Vec<f32> = ex.gscales.to_f32().iter().map(|g| g * (2.0f32).powi((rng.next_u64() % 13) as i32 - 6)).collect();
    ex.gscales = HostTensor::from_f32(&[e, 3], &gs);
    ex
}

fn gb_moe(rig: &Rig, solo: &str, rows: &str, attrs: &OpAttrs, e: usize, topk: usize, h: usize, i: usize, seed: u64, wide: bool) {
    let rows_s = control(rows);
    let rows = rows_s.as_str();
    let mut rng = SplitMix64::new(seed);
    let ex = if wide { wide_experts(&mut rng, e, h, i) } else { random_experts(&mut rng, e, h, i) };
    let w: Vec<TensorRef> = [&ex.gate_up_w, &ex.gate_up_s, &ex.down_w, &ex.down_s, &ex.gscales].iter().map(|t| rig.put(t)).collect();
    let x_rows = nasty_rows(&mut rng, h, 4.0);
    let x_all: Vec<f32> = x_rows.iter().flatten().copied().collect();
    for (rname, ids, mut ws) in routings(&mut rng, e, topk) {
        if wide {
            // routing weights over 2^-14 .. 1: the per-token f32 sum of the K bf16 terms then rounds, and its order matters
            for w in ws.iter_mut() {
                *w *= (2.0f32).powi(-((rng.next_u64() % 15) as i32));
            }
        }
        let label = format!("moe e={e} top{topk} h={h} i={i} wide={wide} routing={rname} attrs.gelu={}", attrs.gelu);
        // solo: each token through the T = 1 kernel
        let mut want: Vec<Vec<u8>> = Vec::new();
        for t in 0..CAP {
            let ins = vec![
                rig.put(&HostTensor::from_f32_as(DType::Bf16, &[1, h], &x_rows[t])),
                rig.put(&HostTensor::from_i32(&[1, topk], &ids[t * topk..(t + 1) * topk])),
                rig.put(&HostTensor::from_f32(&[1, topk], &ws[t * topk..(t + 1) * topk])),
                w[0].clone(),
                w[1].clone(),
                w[2].clone(),
                w[3].clone(),
                w[4].clone(),
            ];
            let o = rig.run(solo, Op::MoeExperts, &ins, &[(DType::Bf16, vec![1, h])], attrs);
            want.push(o[0].data.clone());
        }
        for tcount in 1..=CAP {
            let ins = vec![
                rig.put(&HostTensor::from_f32_as(DType::Bf16, &[tcount, h], &x_all[..tcount * h])),
                rig.put(&HostTensor::from_i32(&[tcount, topk], &ids[..tcount * topk])),
                rig.put(&HostTensor::from_f32(&[tcount, topk], &ws[..tcount * topk])),
                w[0].clone(),
                w[1].clone(),
                w[2].clone(),
                w[3].clone(),
                w[4].clone(),
            ];
            let o = rig.run(rows, Op::MoeExperts, &ins, &[(DType::Bf16, vec![tcount, h])], attrs);
            for t in 0..tcount {
                let got = &o[0].data[t * h * 2..(t + 1) * h * 2];
                if got != &want[t][..] {
                    let bad = got.chunks(2).zip(want[t].chunks(2)).filter(|(g, w)| g != w).count();
                    let first = got.chunks(2).zip(want[t].chunks(2)).position(|(g, w)| g != w).unwrap();
                    panic!(
                        "G-B FAIL {label}: {rows} T={tcount} token {t} differs from {solo} on that token alone in {bad} of {h} elements \
                         (first at {first}: {:02x?} vs {:02x?})",
                        &got[first * 2..first * 2 + 2],
                        &want[t][first * 2..first * 2 + 2]
                    );
                }
            }
            rig.note(rows, tcount, tcount, tcount * h * 2);
        }
    }
}

#[test]
fn moe_decode_rows_are_bitwise_the_one_token_kernel() {
    let Some(rig) = Rig::new("moe_decode_rows_are_bitwise_the_one_token_kernel") else { return };
    let silu = OpAttrs { top_k: 8, ..OpAttrs::default() };
    // the 35B layer (hidden 2048, routed intermediate 512, top-8) over 64 of its 256 experts (the kernels only index by
    // expert id; 256 would be 400 MB of random weights per case), and a small shape with top-4
    for wide in [false, true] {
        gb_moe(&rig, "moe_experts_bf16_v2", "moe_experts_bf16_rows", &silu, 64, 8, 2048, 512, 901, wide);
        // --tp 2: every rank holds every expert at half the routed intermediate (256)
        gb_moe(&rig, "moe_experts_bf16_v2", "moe_experts_bf16_rows", &silu, 64, 8, 2048, 256, 903, wide);
        gb_moe(&rig, "moe_experts_bf16_v2", "moe_experts_bf16_rows", &OpAttrs { top_k: 4, ..OpAttrs::default() }, 8, 4, 128, 64, 902, wide);
    }
    rig.report("moe");
}

// ---- the ops that need no rows_exact kernel ---------------------------------------------------------------------------

use intarsia_core::exec::{HostCall, OutSpec};
use intarsia_core::quant::cases::{combine_case, gated_rmsnorm_case, topk_case};

fn rows_of(t: &HostTensor, rows: usize) -> Option<usize> {
    (t.shape.first() == Some(&rows)).then(|| t.data.len() / rows)
}

/// Row `r` of every tensor whose leading dimension is `rows` (the others, weights and tables, are kept).
fn slice_row(t: &HostTensor, r: usize, rows: usize) -> HostTensor {
    match rows_of(t, rows) {
        Some(rb) => {
            let mut shape = t.shape.clone();
            shape[0] = 1;
            HostTensor::new(t.dtype, shape, t.data[r * rb..(r + 1) * rb].to_vec()).unwrap()
        }
        None => t.clone(),
    }
}

/// Row-parallel ops keep their ordinary kernel under `exact_rows` (the registry only restricts an op that HAS a row-exact
/// kernel): the claim is that row r of the R-row call is bitwise the same kernel's 1-row call on row r. Checked for R = 16.
#[test]
fn row_parallel_ops_are_row_independent() {
    let Some(rig) = Rig::new("row_parallel_ops_are_row_independent") else { return };
    let mut rng = SplitMix64::new(804);
    let r = CAP;
    let bf = |rng: &mut SplitMix64, shape: &[usize], s: f32| HostTensor::from_f32_as(DType::Bf16, shape, &rng.normal_vec(shape.iter().product(), s));
    let attrs_eps = OpAttrs { eps: 1e-6, ..OpAttrs::default() };
    let ids: Vec<i32> = (0..r).map(|_| (rng.next_u64() % 1000) as i32).collect();
    let mut logits: Vec<f32> = rng.normal_vec(r * 5000, 3.0);
    for row in 0..r {
        logits[row * 5000 + (row * 37) % 5000] = 50.0; // a clear winner per row; ties are exercised below
    }
    let mut calls: Vec<(&str, HostCall)> = vec![
        ("rmsnorm_bf16", HostCall { op: Op::RmsNorm, inputs: vec![bf(&mut rng, &[r, 2048], 3.0), bf(&mut rng, &[2048], 1.0)], outputs: vec![OutSpec::new(DType::Bf16, &[r, 2048])], attrs: attrs_eps.clone() }),
        ("rmsnorm_add_bf16", HostCall { op: Op::RmsNormAdd, inputs: vec![bf(&mut rng, &[r, 2048], 3.0), bf(&mut rng, &[r, 2048], 3.0), bf(&mut rng, &[2048], 1.0)], outputs: vec![OutSpec::new(DType::Bf16, &[r, 2048]), OutSpec::new(DType::Bf16, &[r, 2048])], attrs: attrs_eps.clone() }),
        ("add_bf16", HostCall { op: Op::Add, inputs: vec![bf(&mut rng, &[r, 2048], 3.0), bf(&mut rng, &[r, 2048], 3.0)], outputs: vec![OutSpec::new(DType::Bf16, &[r, 2048])], attrs: OpAttrs::default() }),
        ("silu_glu_bf16", HostCall { op: Op::SiluGlu, inputs: vec![bf(&mut rng, &[r, 512], 3.0), bf(&mut rng, &[r, 512], 3.0)], outputs: vec![OutSpec::new(DType::Bf16, &[r, 512])], attrs: OpAttrs::default() }),
        ("sigmoid_mul_bf16", HostCall { op: Op::SigmoidMul, inputs: vec![bf(&mut rng, &[r, 4096], 3.0), bf(&mut rng, &[r, 4096], 3.0)], outputs: vec![OutSpec::new(DType::Bf16, &[r, 4096])], attrs: OpAttrs::default() }),
        ("embedding_bf16", HostCall { op: Op::Embedding, inputs: vec![HostTensor::from_i32(&[r], &ids), bf(&mut rng, &[1000, 2048], 1.0)], outputs: vec![OutSpec::new(DType::Bf16, &[r, 2048])], attrs: OpAttrs::default() }),
        ("argmax_f32", HostCall { op: Op::Argmax, inputs: vec![HostTensor::from_f32(&[r, 5000], &logits)], outputs: vec![OutSpec::new(DType::I32, &[r])], attrs: OpAttrs::default() }),
        ("moe_combine_bf16", combine_case(&mut rng, r, 2048, DType::Bf16)),
        ("gated_rmsnorm_bf16", gated_rmsnorm_case(&mut rng, r, 128, DType::Bf16)),
    ];
    for ties in [false, true] {
        for e in [256usize, 64] {
            let mut hc = topk_case(&mut rng, r, e, 8, ties, DType::Bf16, true);
            hc.attrs.top_k = 8;
            calls.push(("moe_topk_bf16_v2", hc.clone()));
            calls.push(("moe_topk_bf16", hc));
        }
    }
    for (kernel, hc) in calls {
        let ins: Vec<TensorRef> = hc.inputs.iter().map(|t| rig.put(t)).collect();
        let outs: Vec<(DType, Vec<usize>)> = hc.outputs.iter().map(|o| (o.dtype, o.shape.clone())).collect();
        let all = rig.run(kernel, hc.op, &ins, &outs, &hc.attrs);
        for row in 0..r {
            let ins1: Vec<TensorRef> = hc.inputs.iter().map(|t| rig.put(&slice_row(t, row, r))).collect();
            let outs1: Vec<(DType, Vec<usize>)> = hc.outputs.iter().map(|o| (o.dtype, std::iter::once(1).chain(o.shape[1..].iter().copied()).collect())).collect();
            let one = rig.run(kernel, hc.op, &ins1, &outs1, &hc.attrs);
            for (o, (a, b)) in all.iter().zip(&one).enumerate() {
                let rb = a.data.len() / r;
                assert!(a.data[row * rb..(row + 1) * rb] == b.data[..], "{kernel}: output {o} row {row} of the {r}-row call differs from the 1-row call");
            }
        }
        rig.note(kernel, r, r, 0);
    }
    rig.report("row-parallel");
}

// ---- micro-timing (not part of the gate) ---------------------------------------------------------------------------------

impl Rig {
    /// Microseconds per launch of `kernel` over `iters` back-to-back launches of one prepared call (host launch overhead
    /// included, hidden while the kernel is longer than a launch).
    fn time_us(&self, kernel: &str, op: Op, inputs: &[TensorRef], outs: &[(DType, Vec<usize>)], attrs: &OpAttrs, iters: usize) -> f64 {
        let outputs: Vec<TensorRef> = outs.iter().map(|(dt, s)| self.put(&HostTensor::zeros(*dt, s))).collect();
        let mut call = OpCall::new(op, inputs.to_vec(), outputs, attrs.clone());
        call.check().unwrap();
        if let Some(k) = self.be.kernels().iter().find(|k| k.name == kernel && k.op == op) {
            let need = k.scratch_bytes(&ShapeInfo::from_call(&call));
            if need > 0 {
                let buf = self.be.alloc(need.max(16), MemKind::Device).unwrap();
                call.scratch = Some(TensorRef::new(buf, DType::U8, &[need]));
            }
        }
        for _ in 0..30 {
            self.be.launch(kernel, &call, 0).unwrap();
        }
        self.be.synchronize(0).unwrap();
        let t0 = std::time::Instant::now();
        for _ in 0..iters {
            self.be.launch(kernel, &call, 0).unwrap();
        }
        self.be.synchronize(0).unwrap();
        t0.elapsed().as_secs_f64() * 1e6 / iters as f64
    }
}

/// `cargo test -p intarsia-backend-hip --release --test rows_exact -- --ignored --nocapture timing` (inside a fleetq slot that
/// is quiet on the GPU). Prints `kernel M us GB/s(weight bytes)`; kernels the build does not have are skipped, so the same file
/// times the tree before the rows kernels (the 1-row kernels' before/after).
#[test]
#[ignore = "micro-timing"]
fn timing_gemv_rows() {
    let Some(rig) = Rig::new("timing_gemv_rows") else { return };
    let have = |n: &str| rig.be.kernels().iter().any(|k| k.name == n);
    let mut rng = SplitMix64::new(77);
    let a = OpAttrs::default();
    let iters = 400;
    let mut line = |what: &str, kernel: &str, m: usize, us: f64, bytes: usize| {
        println!("TIME {what:<28} {kernel:<26} M={m:<2} {us:8.2} us  {:7.1} GB/s", bytes as f64 / us / 1e3);
    };
    for (n, k) in [(2048usize, 4096usize), (2048, 2048), (8192, 2048)] {
        let hc = intarsia_core::quant::cases::matmul_fp8_case(&mut rng, 1, n, k, true, DType::Bf16, DType::Bf16);
        let w = rig.put(&hc.inputs[1]);
        let sc = rig.put(&scales_tensor(hc.inputs[2].to_f32()[0], 0.125));
        let x = rig.put(&HostTensor::from_f32_as(DType::Bf16, &[CAP, k], &rng.normal_vec(CAP * k, 1.0)));
        for (kernel, ms) in [("matmul_fp8_gemv_v2", vec![1usize, 2, 4]), ("matmul_fp8_gemv_rows", vec![1, 2, 4, 8, 12, 16])] {
            if !have(kernel) {
                continue;
            }
            for m in ms {
                let us = rig.time_us(kernel, Op::MatmulFp8, &[x.rows(0, m), w.clone(), sc.clone()], &[(DType::Bf16, vec![m, n])], &a, iters);
                line(&format!("fp8 {n}x{k}"), kernel, m, us, n * k);
            }
        }
    }
    for (n, k, odt) in [(512usize, 2048usize, DType::Bf16), (2048, 512, DType::Bf16), (248320, 2048, DType::F32)] {
        let q = nvfp4::random_rows(&mut rng, n, k);
        let ws: Vec<TensorRef> = q.parts.iter().map(|p| rig.put(p)).collect();
        let x = rig.put(&HostTensor::from_f32_as(DType::Bf16, &[CAP, k], &rng.normal_vec(CAP * k, 1.0)));
        for (kernel, ms) in [("matmul_nvfp4_gemv_v2", vec![1usize, 2, 4, 8]), ("matmul_nvfp4_gemv_rows", vec![1, 2, 4, 8, 12, 16])] {
            if !have(kernel) {
                continue;
            }
            for m in ms {
                let mut ins = vec![x.rows(0, m)];
                ins.extend(ws.iter().cloned());
                let us = rig.time_us(kernel, Op::MatmulNvfp4, &ins, &[(odt, vec![m, n])], &a, iters);
                line(&format!("nvfp4 {n}x{k}"), kernel, m, us, n * k / 2 + n * k / 16);
            }
        }
    }
    for (n, k) in [(4096usize, 4096usize), (256, 2048)] {
        let w = rig.put(&fast_bf16(&mut rng, n, k));
        let x = rig.put(&HostTensor::from_f32_as(DType::Bf16, &[CAP, k], &rng.normal_vec(CAP * k, 1.0)));
        for (kernel, ms) in [("matmul_bf16_gemv", vec![1usize, 2, 4]), ("matmul_bf16_gemv_rows", vec![1, 2, 4, 8, 12, 16])] {
            if !have(kernel) {
                continue;
            }
            for m in ms {
                let us = rig.time_us(kernel, Op::Matmul, &[x.rows(0, m), w.clone()], &[(DType::Bf16, vec![m, n])], &a, iters);
                line(&format!("bf16 {n}x{k}"), kernel, m, us, n * k * 2);
            }
        }
    }
    // MoE: the 35B layer over 64 experts, T tokens with distinct routing; bytes = every pair's expert read once
    let (e, topk, h, i) = (64usize, 8usize, 2048usize, 512usize);
    let ex = random_experts(&mut rng, e, h, i);
    let w: Vec<TensorRef> = [&ex.gate_up_w, &ex.gate_up_s, &ex.down_w, &ex.down_s, &ex.gscales].iter().map(|t| rig.put(t)).collect();
    let (ids, ws) = random_routing(&mut rng, CAP, e, topk);
    let x = rig.put(&HostTensor::from_f32_as(DType::Bf16, &[CAP, h], &rng.normal_vec(CAP * h, 1.0)));
    let (ids, ws) = (rig.put(&ids), rig.put(&ws));
    let attrs = OpAttrs { top_k: topk, ..OpAttrs::default() };
    let pair_bytes = 2 * i * h / 2 + 2 * i * h / 16 + h * i / 2 + h * i / 16;
    for (kernel, ms) in [("moe_experts_bf16_v2", vec![1usize]), ("moe_experts_bf16_rows", vec![1, 2, 4, 8, 12, 16]), ("moe_experts_bf16_v2", vec![2, 4, 8])] {
        if !have(kernel) {
            continue;
        }
        for m in ms {
            let mut ins = vec![x.rows(0, m), ids.rows(0, m), ws.rows(0, m)];
            ins.extend(w.iter().cloned());
            let us = rig.time_us(kernel, Op::MoeExperts, &ins, &[(DType::Bf16, vec![m, h])], &attrs, iters / 2);
            line("moe 64e top8 2048/512", kernel, m, us, m * topk * pair_bytes);
        }
    }
}

/// Window timing of the 27B's verify GEMVs: what an M-row call costs against the one-row call (the verify pass of a speculative
/// step is M = k + 1 rows). Prints `TIME` lines like `timing_gemv_rows`; run in a fleetq slot quiet on the GPU.
#[test]
#[ignore = "micro-timing"]
fn timing_mxfp4_rows() {
    let Some(rig) = Rig::new("timing_mxfp4_rows") else { return };
    let mut rng = SplitMix64::new(78);
    let iters = 300;
    for (n, k) in [(17408usize, 5120usize), (5120, 17408), (12288, 5120), (5120, 6144), (1024, 5120)] {
        let q = intarsia_core::quant::mxfp4::random_rows(&mut rng, n, k, intarsia_core::quant::Mxfp4Act::Dynamic);
        let ws: Vec<TensorRef> = q.parts.iter().map(|p| rig.put(p)).collect();
        let x = rig.put(&HostTensor::from_f32_as(DType::Bf16, &[CAP, k], &rng.normal_vec(CAP * k, 1.0)));
        for kernel in ["matmul_mxfp4_gemv", "matmul_mxfp4_gemv_rows"] {
            let ms: &[usize] = if kernel.ends_with("_rows") { &[1, 2, 3, 4, 5, 6, 7, 8, 12, 16] } else { &[1, 2, 3, 4, 5, 8] };
            for &m in ms {
                let mut ins = vec![x.rows(0, m)];
                ins.extend(ws.iter().cloned());
                let us = rig.time_us(kernel, Op::MatmulMxfp4, &ins, &[(DType::Bf16, vec![m, n])], &OpAttrs::default(), iters);
                let bytes = n * k / 2 + n * k / 32;
                println!("TIME mxfp4 {n}x{k:<6} {kernel:<24} M={m:<2} {us:8.2} us  {:7.1} GB/s", bytes as f64 / us / 1e3);
            }
        }
    }
    // the lm_head sweep of the 27B (bf16 weights, f32 logits), one row and the verify rows
    let (n, k) = (248320usize, 5120usize);
    let w = rig.put(&fast_bf16(&mut rng, n, k));
    let x = rig.put(&HostTensor::from_f32_as(DType::Bf16, &[CAP, k], &rng.normal_vec(CAP * k, 1.0)));
    for kernel in ["logits_bf16_gemv", "logits_bf16_gemv_rows"] {
        let ms: &[usize] = if kernel.ends_with("_rows") { &[1, 2, 3, 4, 5, 6, 8, 12, 16] } else { &[1, 2, 3, 4] };
        for &m in ms {
            let us = rig.time_us(kernel, Op::Logits, &[x.rows(0, m), w.clone()], &[(DType::F32, vec![m, n])], &OpAttrs::default(), 100);
            println!("TIME lm_head {n}x{k} {kernel:<24} M={m:<2} {us:8.2} us  {:7.1} GB/s", (n * k * 2) as f64 / us / 1e3);
        }
    }
}
