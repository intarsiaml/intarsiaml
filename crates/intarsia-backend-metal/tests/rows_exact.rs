//! G-B for the hybrid Qwen3.6-35B-A3B on Metal (lane mb, docs/results/2026-10-23-mb-metal-hybrid-batching.md): every kernel
//! of the batched decode step gives, for each of M = 1..=8 activation rows, BITWISE the bytes of the M = 1 launch the solo
//! decode step runs for that row. Each test runs the op once on the `[M, ...]` batch (the row-exact kernel a batched step
//! selects) and M times on the 1-row slices of the same inputs (the solo step's launches), then compares every byte of the
//! output buffers, guards included.
//!
//!  * `matmul_fp8_gemv_rows`, `matmul_nvfp4_gemv_rows`, `moe_experts_bf16_rows`: the three new `rows_exact` kernels.
//!  * the existing `rows_exact` bf16 pair kernel at the 35B's router / shared-gate / GDN a-b pairs.
//!  * the kernels a batched step shares with the solo step because they are row-parallel (rmsnorm, rmsnorm_add, silu_glu,
//!    embedding, gate_split, sigmoid_mul, moe_topk, moe_combine, argmax): same kernel for M rows and for 1 row, so the claim
//!    is that no row's bytes depend on the row count.
//!
//! The synthetic tests use the GPU (some allocate 0.4 GB): on endor take a fleetq slot first.

#![cfg(target_os = "macos")]

mod common;
use common::*;
use intarsia_backend_metal::bench::{f32_to_bf16, Rng};
use intarsia_backend_metal::MetalBackend;
use intarsia_core::backend::{Backend, Buffer, MemKind, TensorRef};
use intarsia_core::quant::cases::{random_experts, random_routing};
use intarsia_core::rng::SplitMix64;
use intarsia_core::{DType, HostTensor, Op, OpAttrs, OpCall, ShapeInfo};
use std::sync::atomic::{AtomicUsize, Ordering};

/// Launches compared bitwise (M-row launch vs M one-row launches), per test, printed at the end of each.
static CASES: AtomicUsize = AtomicUsize::new(0);
static ROWS: AtomicUsize = AtomicUsize::new(0);

fn tally(rows: usize) {
    CASES.fetch_add(1, Ordering::Relaxed);
    ROWS.fetch_add(rows, Ordering::Relaxed);
}

fn report(name: &str) {
    eprintln!(
        "{name}: {} cases, {} rows compared bitwise",
        CASES.swap(0, Ordering::Relaxed),
        ROWS.swap(0, Ordering::Relaxed)
    );
}

fn put_bytes(be: &MetalBackend, off: usize, data: &[u8]) -> Buffer {
    let b = be.alloc(off + data.len() + 16, MemKind::Device).unwrap();
    be.upload(&b, 0, &vec![0xEEu8; off + data.len() + 16], 0).unwrap();
    be.upload(&b, off, data, 0).unwrap();
    b
}

fn guarded(be: &MetalBackend, off: usize, bytes: usize) -> Buffer {
    let total = off + bytes + 16;
    let b = be.alloc(total, MemKind::Device).unwrap();
    be.upload(&b, 0, &vec![0xEEu8; total], 0).unwrap();
    b
}

fn dl(be: &MetalBackend, b: &Buffer, n: usize) -> Vec<u8> {
    be.synchronize(0).unwrap();
    let mut raw = vec![0u8; n];
    be.download(b, 0, &mut raw, 0).unwrap();
    raw
}

fn t(buf: &Buffer, off: usize, dt: DType, shape: &[usize]) -> TensorRef {
    TensorRef { buf: buf.clone(), offset: off, dtype: dt, shape: shape.to_vec() }
}

fn same(a: &[u8], b: &[u8], what: &str) {
    assert_eq!(a.len(), b.len(), "{what}: length");
    if let Some(i) = a.iter().zip(b).position(|(x, y)| x != y) {
        panic!(
            "{what}: first differing byte {i} of {} (one-row launches {:#04x}, M-row launch {:#04x})",
            a.len(),
            a[i],
            b[i]
        );
    }
}

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn bf16_bytes(v: &[u16]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// Activation rows of mixed character, row r by r % 6: random in [-1, 1), wide range (magnitudes 2^-18..2^+18), saturating
/// (reaches past e4m3's 448 after the input scale), all zero, one nonzero element, small noise around 1e-3.
fn act_rows(m: usize, k: usize, seed: u64) -> Vec<u16> {
    let mut rng = Rng(seed | 1);
    let mut out = Vec::with_capacity(m * k);
    for r in 0..m {
        for j in 0..k {
            let v = match r % 6 {
                0 => rng.unit(),
                1 => {
                    let e = (rng.next_u64() % 37) as i32 - 18;
                    rng.unit() * 2f32.powi(e)
                }
                2 => rng.unit() * 3000.0,
                3 => 0.0,
                4 => {
                    if j == (r * 7919) % k {
                        1.5
                    } else {
                        0.0
                    }
                }
                _ => rng.unit() * 1e-3,
            };
            out.push(f32_to_bf16(v));
        }
    }
    out
}

fn rand_e4m3(n: usize, seed: u64, nan_every: usize) -> Vec<u8> {
    let mut r = Rng(seed | 1);
    (0..n)
        .map(|i| {
            let b = (r.next_u64() >> 32) as u8;
            if nan_every > 0 && i % nan_every == nan_every - 1 {
                0x7F
            } else if b & 0x7F == 0x7F {
                b & 0xFE
            } else {
                b
            }
        })
        .collect()
}

// ---- matmul_fp8 -----------------------------------------------------------------------------------------------------

/// (N, K) of the 35B's fp8 projections: GDN in_qkv (conv_dim 8192) / in_z (4096) / out (K = 4096), attention q+gate (8192) /
/// k, v (512) / o (K = 4096); plus odd shapes (N not a multiple of the rows per threadgroup, K = 48, 16).
const FP8_SHAPES: &[(usize, usize)] =
    &[(8192, 2048), (4096, 2048), (512, 2048), (2048, 4096), (130, 512), (33, 48), (65, 16), (4096, 4096)];

#[allow(clippy::too_many_arguments)]
fn fp8_case(
    be: &MetalBackend,
    w: &Buffer,
    w_off: usize,
    n: usize,
    k: usize,
    m: usize,
    is: f32,
    f32_out: bool,
    x_off: usize,
    seed: u64,
) {
    let x = act_rows(m, k, seed);
    let xb = put_bytes(be, x_off, &bf16_bytes(&x));
    let sb = put_bytes(be, 4, &f32_bytes(&[0.0123, is]));
    let (odt, esz) = if f32_out { (DType::F32, 4) } else { (DType::Bf16, 2) };
    let yo = 8usize;
    let (ys, ym) = (guarded(be, yo, m * n * esz), guarded(be, yo, m * n * esz));
    let xt = t(&xb, x_off, DType::Bf16, &[m, k]);
    let wt = t(w, w_off, DType::Fp8E4m3, &[n, k]);
    let st = t(&sb, 4, DType::F32, &[2]);
    let (yst, ymt) = (t(&ys, yo, odt, &[m, n]), t(&ym, yo, odt, &[m, n]));
    let call = |x: TensorRef, y: TensorRef| {
        OpCall::new(Op::MatmulFp8, vec![x, wt.clone(), st.clone()], vec![y], OpAttrs::default())
    };
    for r in 0..m {
        be.launch("matmul_fp8_gemv", &call(xt.rows(r, 1), yst.rows(r, 1)), 0).unwrap();
    }
    let rows = call(xt.clone(), ymt);
    be.launch("matmul_fp8_gemv_rows", &rows, 0).unwrap();
    let tot = yo + m * n * esz + 16;
    same(
        &dl(be, &ys, tot),
        &dl(be, &ym, tot),
        &format!("matmul_fp8 N={n} K={k} M={m} is={is} f32_out={f32_out} w_off={w_off} x_off={x_off}"),
    );
    tally(m);
}

#[test]
fn fp8_rows_are_bitwise_the_one_row_gemv() {
    let be = backend();
    for (si, &(n, k)) in FP8_SHAPES.iter().enumerate() {
        for (wi, w_off) in [8usize, 16].into_iter().enumerate() {
            // a NaN code in some weights on the second layout: the NaN path of the dot must agree too
            let w = rand_e4m3(n * k, 40 + si as u64, if wi == 1 { 997 } else { 0 });
            let wb = put_bytes(&be, w_off, &w);
            for m in 2..=16usize {
                for (ii, is) in [0.0f32, 0.37, 2.0].into_iter().enumerate() {
                    for f32_out in [false, true] {
                        let x_off = if (m + ii) % 2 == 0 { 0 } else { 2 };
                        fp8_case(&be, &wb, w_off, n, k, m, is, f32_out, x_off, 700 + (si * 100 + m * 10 + ii) as u64);
                    }
                }
            }
        }
    }
    report("fp8");
}

// ---- matmul_nvfp4 ---------------------------------------------------------------------------------------------------

/// (N, K): shared expert gate / up (512 x 2048) and down (2048 x 512), lm_head (248320 x 2048, f32 logits), odd shapes.
const NV_SHAPES: &[(usize, usize)] = &[(512, 2048), (2048, 512), (248320, 2048), (130, 512), (33, 48), (65, 16)];

fn rand_block_scales(n: usize, seed: u64) -> Vec<u8> {
    let mut r = Rng(seed | 1);
    (0..n)
        .map(|_| {
            let v = (r.next_u64() >> 32) as u8;
            (v & 0x87) | (((v >> 3) % 8 + 4) << 3)
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn nv_case(
    be: &MetalBackend,
    bufs: &(Buffer, Buffer, Buffer),
    w_off: usize,
    n: usize,
    k: usize,
    m: usize,
    f32_out: bool,
    x_off: usize,
    seed: u64,
) {
    let x = act_rows(m, k, seed);
    let xb = put_bytes(be, x_off, &bf16_bytes(&x));
    let (odt, esz) = if f32_out { (DType::F32, 4) } else { (DType::Bf16, 2) };
    let yo = 8usize;
    let (ys, ym) = (guarded(be, yo, m * n * esz), guarded(be, yo, m * n * esz));
    let xt = t(&xb, x_off, DType::Bf16, &[m, k]);
    let (wt, bt, gt) = (
        t(&bufs.0, w_off, DType::U8, &[n, k / 2]),
        t(&bufs.1, 3, DType::Fp8E4m3, &[n, k / 16]),
        t(&bufs.2, 4, DType::F32, &[1]),
    );
    let (yst, ymt) = (t(&ys, yo, odt, &[m, n]), t(&ym, yo, odt, &[m, n]));
    let call = |x: TensorRef, y: TensorRef| {
        OpCall::new(Op::MatmulNvfp4, vec![x, wt.clone(), bt.clone(), gt.clone()], vec![y], OpAttrs::default())
    };
    for r in 0..m {
        be.launch("matmul_nvfp4_gemv", &call(xt.rows(r, 1), yst.rows(r, 1)), 0).unwrap();
    }
    be.launch("matmul_nvfp4_gemv_rows", &call(xt.clone(), ymt), 0).unwrap();
    let tot = yo + m * n * esz + 16;
    same(
        &dl(be, &ys, tot),
        &dl(be, &ym, tot),
        &format!("matmul_nvfp4 N={n} K={k} M={m} f32_out={f32_out} w_off={w_off} x_off={x_off}"),
    );
    tally(m);
}

#[test]
fn nvfp4_rows_are_bitwise_the_one_row_gemv() {
    let be = backend();
    for (si, &(n, k)) in NV_SHAPES.iter().enumerate() {
        let mut r = Rng(900 + si as u64);
        let w: Vec<u8> = (0..n * k / 2).map(|_| (r.next_u64() >> 32) as u8).collect();
        let bs = rand_block_scales(n * k / 16, 910 + si as u64);
        for w_off in [8usize, 16] {
            let bufs =
                (put_bytes(&be, w_off, &w), put_bytes(&be, 3, &bs), put_bytes(&be, 4, &0.00071f32.to_le_bytes()));
            for m in 2..=16usize {
                for f32_out in [false, true] {
                    let x_off = if m % 2 == 0 { 0 } else { 2 };
                    nv_case(&be, &bufs, w_off, n, k, m, f32_out, x_off, 1700 + (si * 100 + m) as u64);
                }
            }
        }
    }
    report("nvfp4");
}

// ---- moe_experts ----------------------------------------------------------------------------------------------------

const E: usize = 256;
const K: usize = 8;
const H: usize = 2048;
const I: usize = 512;

fn put_host(be: &MetalBackend, h: &HostTensor, off: usize) -> TensorRef {
    let b = put_bytes(be, off, &h.data);
    TensorRef { buf: b, offset: off, dtype: h.dtype, shape: h.shape.clone() }
}

/// Routing scenarios for `t` tokens: (name, ids [t * K], weights [t * K]).
fn routings(rng: &mut SplitMix64, t: usize) -> Vec<(&'static str, Vec<i32>, Vec<f32>)> {
    let (ids, ws) = random_routing(rng, t, E, K);
    let (rids, rws) = (ids.to_i32().unwrap(), ws.to_f32());
    let mut out: Vec<(&'static str, Vec<i32>, Vec<f32>)> = vec![("random", rids.clone(), rws.clone())];
    // every token picks the same 8 experts: each expert has t pairs (the most one expert sees in a tile)
    let same8: Vec<i32> = (0..t).flat_map(|_| rids[..K].to_vec()).collect();
    out.push(("identical", same8, rws.clone()));
    // overlap: token i shares 4 experts with token 0 and has 4 of its own
    let mut ov = Vec::new();
    for i in 0..t {
        for j in 0..K {
            ov.push(if j < 4 { rids[j] } else { rids[(i * K + j) % rids.len()] });
        }
    }
    // keep the ids of one token distinct: replace repeats by the next free expert
    for tok in ov.chunks_mut(K) {
        for j in 0..K {
            while tok[..j].contains(&tok[j]) {
                tok[j] = (tok[j] + 1) % E as i32;
            }
        }
    }
    out.push(("overlap", ov, rws.clone()));
    // out-of-range ids (-1, E, i32::MIN) on some slots: the pair is dropped, the token still sums the others
    let mut bad = rids.clone();
    for (i, v) in bad.iter_mut().enumerate() {
        match i % 11 {
            3 => *v = -1,
            7 => *v = E as i32,
            9 => *v = i32::MIN,
            _ => {}
        }
    }
    out.push(("invalid", bad, rws.clone()));
    // one token that has no valid expert at all
    let mut none = rids.clone();
    for v in none[..K].iter_mut() {
        *v = -1;
    }
    out.push(("empty-token", none, rws.clone()));
    // the same expert on every slot of every other token (duplicates: eight pairs of one expert per token, several tiles)
    let dup: Vec<i32> = (0..t * K).map(|i| if i / K % 2 == 0 { 5 } else { rids[i] }).collect();
    out.push(("duplicates", dup, rws));
    out
}

#[test]
fn moe_experts_rows_are_bitwise_the_one_token_call() {
    let be = backend();
    let mut rng = SplitMix64::new(0x6b_0e_77);
    let ex = random_experts(&mut rng, E, H, I);
    let ks = be.kernels();
    let solo = ks.iter().find(|k| k.name == "moe_experts_bf16").unwrap();
    let rows_k = ks.iter().find(|k| k.name == "moe_experts_bf16_rows").unwrap();
    assert!(rows_k.rows_exact && !solo.rows_exact);
    for w_off in [0usize, 8] {
        let guw = put_host(&be, &ex.gate_up_w, w_off);
        let gus = put_host(&be, &ex.gate_up_s, 0);
        let dw = put_host(&be, &ex.down_w, w_off);
        let ds = put_host(&be, &ex.down_s, 0);
        let gs = put_host(&be, &ex.gscales, 0);
        for t_rows in 1..=16usize {
            let x = HostTensor::from_f32_as(DType::Bf16, &[t_rows, H], &rng.normal_vec(t_rows * H, 1.0));
            let xr = put_host(&be, &x, 0);
            for (name, ids, ws) in routings(&mut rng, t_rows) {
                let idt = put_host(&be, &HostTensor::from_i32(&[t_rows, K], &ids), 0);
                let wst = put_host(&be, &HostTensor::from_f32(&[t_rows, K], &ws), 0);
                for odt in [DType::Bf16, DType::F32] {
                    let esz = odt.size_bytes();
                    let (ys, ym) = (guarded(&be, 0, t_rows * H * esz), guarded(&be, 0, t_rows * H * esz));
                    let mk = |rows: usize, r0: usize, out: &Buffer| {
                        OpCall::new(
                            Op::MoeExperts,
                            vec![
                                xr.rows(r0, rows),
                                idt.rows(r0, rows),
                                wst.rows(r0, rows),
                                guw.clone(),
                                gus.clone(),
                                dw.clone(),
                                ds.clone(),
                                gs.clone(),
                            ],
                            vec![t(out, r0 * H * esz, odt, &[rows, H])],
                            OpAttrs { top_k: K, ..OpAttrs::default() },
                        )
                    };
                    let with_scratch = |mut c: OpCall, kern: &intarsia_core::KernelImpl| {
                        c.check().unwrap();
                        let need = kern.scratch_bytes(&ShapeInfo::from_call(&c));
                        let sb = be.alloc(need.max(16), MemKind::Device).unwrap();
                        c.scratch = Some(TensorRef::new(sb, DType::U8, &[need]));
                        c
                    };
                    for r in 0..t_rows {
                        be.launch("moe_experts_bf16", &with_scratch(mk(1, r, &ys), solo), 0).unwrap();
                    }
                    if t_rows >= 2 {
                        let c = with_scratch(mk(t_rows, 0, &ym), rows_k);
                        assert!(rows_k.accepts(&ShapeInfo::from_call(&c)));
                        be.launch("moe_experts_bf16_rows", &c, 0).unwrap();
                        let tot = t_rows * H * esz + 16;
                        same(
                            &dl(&be, &ys, tot),
                            &dl(&be, &ym, tot),
                            &format!("moe_experts T={t_rows} routing={name} out={odt} w_off={w_off}"),
                        );
                        tally(t_rows);
                    } else {
                        // M = 1 is the solo call itself: the rows kernel does not take it
                        let c = with_scratch(mk(1, 0, &ym), rows_k);
                        assert!(!rows_k.accepts(&ShapeInfo::from_call(&c)));
                    }
                }
            }
        }
    }
    report("moe_experts");
}

// ---- the bf16 pair kernel at the 35B's pairs ------------------------------------------------------------------------

#[test]
fn bf16_pair_rows_at_the_35b_pairs() {
    let be = backend();
    // (N0, N1, K): router + shared-expert gate logit (256, 1, 2048); GDN in_b + in_a (32, 32, 2048)
    for (si, &(n0, n1, k)) in [(256usize, 1usize, 2048usize), (32, 32, 2048)].iter().enumerate() {
        for w_off in [0usize, 8, 16] {
            let w0 = rand_bf16(n0 * k, 3000 + si as u64);
            let w1 = rand_bf16(n1 * k, 3100 + si as u64);
            let (w0b, w1b) = (put_bytes(&be, w_off, &bf16_bytes(&w0)), put_bytes(&be, w_off, &bf16_bytes(&w1)));
            for m in 2..=16usize {
                let x = act_rows(m, k, 3200 + m as u64);
                for x_off in [0usize, 8] {
                    let xb = put_bytes(&be, x_off, &bf16_bytes(&x));
                    let (s0, s1) = (guarded(&be, 0, m * n0 * 2), guarded(&be, 0, m * n1 * 2));
                    let (m0, m1) = (guarded(&be, 0, m * n0 * 2), guarded(&be, 0, m * n1 * 2));
                    let xt = t(&xb, x_off, DType::Bf16, &[m, k]);
                    let (w0t, w1t) = (t(&w0b, w_off, DType::Bf16, &[n0, k]), t(&w1b, w_off, DType::Bf16, &[n1, k]));
                    let call = |x: TensorRef, y0: TensorRef, y1: TensorRef| {
                        OpCall::new(Op::MatmulPair, vec![x, w0t.clone(), w1t.clone()], vec![y0, y1], OpAttrs::default())
                    };
                    let (s0t, s1t) = (t(&s0, 0, DType::Bf16, &[m, n0]), t(&s1, 0, DType::Bf16, &[m, n1]));
                    let (m0t, m1t) = (t(&m0, 0, DType::Bf16, &[m, n0]), t(&m1, 0, DType::Bf16, &[m, n1]));
                    for r in 0..m {
                        be.launch("matmul_pair_bf16_gemv", &call(xt.rows(r, 1), s0t.rows(r, 1), s1t.rows(r, 1)), 0)
                            .unwrap();
                    }
                    be.launch("matmul_pair_bf16_gemv_rows", &call(xt.clone(), m0t, m1t), 0).unwrap();
                    same(
                        &dl(&be, &s0, m * n0 * 2 + 16),
                        &dl(&be, &m0, m * n0 * 2 + 16),
                        &format!("pair y0 ({n0},{n1},{k}) M={m} w_off={w_off} x_off={x_off}"),
                    );
                    same(
                        &dl(&be, &s1, m * n1 * 2 + 16),
                        &dl(&be, &m1, m * n1 * 2 + 16),
                        &format!("pair y1 ({n0},{n1},{k}) M={m} w_off={w_off} x_off={x_off}"),
                    );
                    tally(m);
                }
            }
        }
    }
    report("bf16 pair");
}

// ---- row-parallel kernels (the same kernel for M rows and for 1 row) ------------------------------------------------

/// Output buffers `a` (one launch per row) and `b` (one M-row launch) of `outs` (elements per row, dtype); compare bytes.
fn serial_vs_rows(
    be: &MetalBackend,
    kernel: &str,
    m: usize,
    outs: &[(usize, DType)],
    mk: &dyn Fn(usize, usize, &[Buffer]) -> OpCall, // (rows, first row, output buffers)
    what: &str,
) {
    let alloc = || -> Vec<Buffer> { outs.iter().map(|&(per, dt)| guarded(be, 0, m * per * dt.size_bytes())).collect() };
    let (a, b) = (alloc(), alloc());
    for r in 0..m {
        be.launch(kernel, &mk(1, r, &a), 0).unwrap();
    }
    be.launch(kernel, &mk(m, 0, &b), 0).unwrap();
    for (i, &(per, dt)) in outs.iter().enumerate() {
        let n = m * per * dt.size_bytes() + 16;
        same(&dl(be, &a[i], n), &dl(be, &b[i], n), &format!("{kernel} output {i} M={m} {what}"));
    }
    tally(m);
}

fn bf_buf(be: &MetalBackend, n: usize, seed: u64) -> Buffer {
    put_bytes(be, 0, &bf16_bytes(&act_rows(n, 1, seed)))
}

#[test]
fn row_parallel_kernels_do_not_depend_on_the_row_count() {
    let be = backend();
    let eps = OpAttrs { eps: 1e-6, w_offset: 0.0, ..Default::default() };
    for m in 2..=16usize {
        // rmsnorm: hidden 2048 (block norms), [m * 32, 128] (the GDN gated norm), [m * 16, 256] (qk norm shape)
        for (rows_per, h) in [(1usize, 2048usize), (32, 128), (16, 256), (1, 4096)] {
            let n = m * rows_per;
            let x = put_bytes(&be, 0, &bf16_bytes(&rand_bf16(n * h, 11 + h as u64)));
            let w = put_bytes(&be, 0, &bf16_bytes(&rand_bf16(h, 12)));
            let mk = |rows: usize, r0: usize, o: &[Buffer]| {
                OpCall::new(
                    Op::RmsNorm,
                    vec![t(&x, r0 * rows_per * h * 2, DType::Bf16, &[rows * rows_per, h]), t(&w, 0, DType::Bf16, &[h])],
                    vec![t(&o[0], r0 * rows_per * h * 2, DType::Bf16, &[rows * rows_per, h])],
                    eps.clone(),
                )
            };
            serial_vs_rows(
                &be,
                "rmsnorm_bf16",
                m,
                &[(rows_per * h, DType::Bf16)],
                &mk,
                &format!("h={h} rows/seq={rows_per}"),
            );
        }
        // rmsnorm_add at hidden 2048 (the block fold)
        {
            let h = 2048;
            let x = put_bytes(&be, 0, &bf16_bytes(&rand_bf16(m * h, 21)));
            let r = put_bytes(&be, 0, &bf16_bytes(&rand_bf16(m * h, 22)));
            let w = put_bytes(&be, 0, &bf16_bytes(&rand_bf16(h, 23)));
            let mk = |rows: usize, r0: usize, o: &[Buffer]| {
                OpCall::new(
                    Op::RmsNormAdd,
                    vec![
                        t(&x, r0 * h * 2, DType::Bf16, &[rows, h]),
                        t(&r, r0 * h * 2, DType::Bf16, &[rows, h]),
                        t(&w, 0, DType::Bf16, &[h]),
                    ],
                    vec![t(&o[0], r0 * h * 2, DType::Bf16, &[rows, h]), t(&o[1], r0 * h * 2, DType::Bf16, &[rows, h])],
                    eps.clone(),
                )
            };
            serial_vs_rows(&be, "rmsnorm_add_bf16", m, &[(h, DType::Bf16), (h, DType::Bf16)], &mk, "h=2048");
        }
        // silu_glu: shared expert [m, 512], GDN gate [m * 32, 128]
        for per in [512usize, 4096] {
            let (a, b) = (bf_buf(&be, m * per, 31), bf_buf(&be, m * per, 32));
            let mk = |rows: usize, r0: usize, o: &[Buffer]| {
                OpCall::new(
                    Op::SiluGlu,
                    vec![
                        t(&a, r0 * per * 2, DType::Bf16, &[rows, per]),
                        t(&b, r0 * per * 2, DType::Bf16, &[rows, per]),
                    ],
                    vec![t(&o[0], r0 * per * 2, DType::Bf16, &[rows, per])],
                    OpAttrs::default(),
                )
            };
            serial_vs_rows(&be, "silu_glu_bf16", m, &[(per, DType::Bf16)], &mk, &format!("n={per}"));
        }
        // embedding: vocab 1000 x 2048 table
        {
            let (v, h) = (1000usize, 2048usize);
            let table = put_bytes(&be, 0, &bf16_bytes(&rand_bf16(v * h, 41)));
            let ids: Vec<u8> = (0..m).flat_map(|i| (((i * 397 + 11) % v) as i32).to_le_bytes()).collect();
            let idb = put_bytes(&be, 0, &ids);
            let mk = |rows: usize, r0: usize, o: &[Buffer]| {
                OpCall::new(
                    Op::Embedding,
                    vec![t(&idb, r0 * 4, DType::I32, &[rows]), t(&table, 0, DType::Bf16, &[v, h])],
                    vec![t(&o[0], r0 * h * 2, DType::Bf16, &[rows, h])],
                    OpAttrs::default(),
                )
            };
            serial_vs_rows(&be, "embedding_bf16", m, &[(h, DType::Bf16)], &mk, "h=2048");
        }
        // gate_split [m, 16, 512] -> q, gate [m, 16, 256]; sigmoid_mul [m, 4096]
        {
            let (nh, d) = (16usize, 256usize);
            let x = bf_buf(&be, m * nh * 2 * d, 51);
            let mk = |rows: usize, r0: usize, o: &[Buffer]| {
                OpCall::new(
                    Op::GateSplit,
                    vec![t(&x, r0 * nh * 2 * d * 2, DType::Bf16, &[rows, nh, 2 * d])],
                    vec![
                        t(&o[0], r0 * nh * d * 2, DType::Bf16, &[rows, nh, d]),
                        t(&o[1], r0 * nh * d * 2, DType::Bf16, &[rows, nh, d]),
                    ],
                    OpAttrs::default(),
                )
            };
            serial_vs_rows(&be, "gate_split_bf16", m, &[(nh * d, DType::Bf16), (nh * d, DType::Bf16)], &mk, "16 x 256");
            let n = nh * d;
            let (a, g) = (bf_buf(&be, m * n, 52), bf_buf(&be, m * n, 53));
            let mk = |rows: usize, r0: usize, o: &[Buffer]| {
                OpCall::new(
                    Op::SigmoidMul,
                    vec![t(&a, r0 * n * 2, DType::Bf16, &[rows, n]), t(&g, r0 * n * 2, DType::Bf16, &[rows, n])],
                    vec![t(&o[0], r0 * n * 2, DType::Bf16, &[rows, n])],
                    OpAttrs::default(),
                )
            };
            serial_vs_rows(&be, "sigmoid_mul_bf16", m, &[(n, DType::Bf16)], &mk, "n=4096");
        }
        // moe_topk: [m, 256] bf16 logits (with exact ties), top 8 normalised
        {
            let mut logits = rand_bf16(m * E, 61);
            for r in 0..m {
                logits[r * E + 5] = logits[r * E + 77]; // ties: the lower index must win in every row
                logits[r * E + 130] = logits[r * E + 5];
            }
            let lg = put_bytes(&be, 0, &bf16_bytes(&logits));
            let mk = |rows: usize, r0: usize, o: &[Buffer]| {
                OpCall::new(
                    Op::MoeTopk,
                    vec![t(&lg, r0 * E * 2, DType::Bf16, &[rows, E])],
                    vec![t(&o[0], r0 * K * 4, DType::F32, &[rows, K]), t(&o[1], r0 * K * 4, DType::I32, &[rows, K])],
                    OpAttrs { top_k: K, norm_topk: true, ..OpAttrs::default() },
                )
            };
            serial_vs_rows(&be, "moe_topk_bf16", m, &[(K, DType::F32), (K, DType::I32)], &mk, "E=256 K=8");
        }
        // moe_combine: routed + sigmoid(gate) * shared, [m, 2048], gate [m, 1]
        {
            let h = 2048;
            let (r_, s_, g_) = (bf_buf(&be, m * h, 71), bf_buf(&be, m * h, 72), bf_buf(&be, m, 73));
            let mk = |rows: usize, r0: usize, o: &[Buffer]| {
                OpCall::new(
                    Op::MoeCombine,
                    vec![
                        t(&r_, r0 * h * 2, DType::Bf16, &[rows, h]),
                        t(&s_, r0 * h * 2, DType::Bf16, &[rows, h]),
                        t(&g_, r0 * 2, DType::Bf16, &[rows, 1]),
                    ],
                    vec![t(&o[0], r0 * h * 2, DType::Bf16, &[rows, h])],
                    OpAttrs::default(),
                )
            };
            serial_vs_rows(&be, "moe_combine_bf16", m, &[(h, DType::Bf16)], &mk, "h=2048");
        }
        // argmax over the 35B's logits row (vocab 248320, f32 and bf16), with a tie of the maximum
        for dt in [DType::F32, DType::Bf16] {
            let v = 248320usize;
            let mut rng = Rng(81 + m as u64);
            let mut raw: Vec<u8> = Vec::new();
            for r in 0..m {
                let mut row: Vec<f32> = (0..v).map(|_| rng.unit()).collect();
                row[v / 3 + r] = 9.0;
                row[v - 2] = 9.0;
                for x in row {
                    if dt == DType::F32 {
                        raw.extend_from_slice(&x.to_le_bytes());
                    } else {
                        raw.extend_from_slice(&f32_to_bf16(x).to_le_bytes());
                    }
                }
            }
            let lb = put_bytes(&be, 0, &raw);
            let esz = dt.size_bytes();
            let mk = |rows: usize, r0: usize, o: &[Buffer]| {
                OpCall::new(
                    Op::Argmax,
                    vec![t(&lb, r0 * v * esz, dt, &[rows, v])],
                    vec![t(&o[0], r0 * 4, DType::I32, &[rows])],
                    OpAttrs::default(),
                )
            };
            let kernel = if dt == DType::F32 { "argmax_f32" } else { "argmax_bf16" };
            serial_vs_rows(&be, kernel, m, &[(1, DType::I32)], &mk, &format!("vocab={v}"));
        }
    }
    report("row-parallel");
}

// ---- registration pins ----------------------------------------------------------------------------------------------

/// The Metal kernels registered `rows_exact`, and the rows each takes: a new `rows_exact` kernel needs a G-B test above.
#[test]
fn rows_exact_registrations_are_pinned() {
    let be = backend();
    let mut names: Vec<String> = be.kernels().iter().filter(|k| k.rows_exact).map(|k| k.name.to_string()).collect();
    names.sort();
    assert_eq!(
        names,
        [
            "attn_prefill_bf16_splitkv_rows",
            "logits_bf16_gemv_rows",
            "matmul_bf16_gemv_rows",
            "matmul_fp8_gemv_rows",
            "matmul_nvfp4_gemv_rows",
            "matmul_pair_bf16_gemv_rows",
            "moe_experts_bf16_rows",
        ]
    );
    // the three new kernels take 2..=8 rows (the solo kernels keep M = 1)
    for (name, op, k, wdt) in [
        ("matmul_fp8_gemv_rows", Op::MatmulFp8, 2048usize, DType::Fp8E4m3),
        ("matmul_fp8_gemv_rows", Op::MatmulFp8, 4096, DType::Fp8E4m3),
        ("matmul_nvfp4_gemv_rows", Op::MatmulNvfp4, 2048, DType::U8),
        ("moe_experts_bf16_rows", Op::MoeExperts, 2048, DType::U8),
    ] {
        let kern = be.kernels().into_iter().find(|x| x.name == name).unwrap();
        assert_eq!(kern.op, op);
        for m in 1..=17usize {
            let s = ShapeInfo {
                m,
                k,
                n: 512,
                act: Some(DType::Bf16),
                weight: Some(wdt),
                experts: E,
                top_k: K,
                ..ShapeInfo::default()
            };
            assert_eq!(kern.accepts(&s), (2..=16).contains(&m), "{name} K={k} M={m}");
        }
    }
}

// ---- timing (not a gate) --------------------------------------------------------------------------------------------

/// Wall time per call of `f` (it enqueues; one synchronize per `reps` calls), best of 3 rounds, in microseconds.
fn time_us(be: &MetalBackend, reps: usize, mut f: impl FnMut()) -> f64 {
    let mut best = f64::MAX;
    for _ in 0..3 {
        be.synchronize(0).unwrap();
        let t0 = std::time::Instant::now();
        for _ in 0..reps {
            f();
        }
        be.synchronize(0).unwrap();
        best = best.min(t0.elapsed().as_secs_f64() * 1e6 / reps as f64);
    }
    best
}

/// `cargo test --release --test rows_exact -- --ignored --nocapture timing_rows`: per-kernel cost of M rows in one rows
/// launch against M one-row launches (what the batched step replaces), at the 35B's shapes. Run in a quiet fleetq slot.
#[test]
#[ignore]
fn timing_rows() {
    let be = backend();
    // fp8 8192 x 2048 (GDN in_qkv) and 2048 x 4096 (out projections); nvfp4 512 x 2048 (shared expert gate/up)
    for (n, k) in [(8192usize, 2048usize), (2048, 4096)] {
        let w = put_bytes(&be, 8, &rand_e4m3(n * k, 5, 0));
        let sb = put_bytes(&be, 4, &f32_bytes(&[0.0123, 0.37]));
        for m in [1usize, 2, 4, 8] {
            let xb = put_bytes(&be, 0, &bf16_bytes(&act_rows(m, k, 9)));
            let y = guarded(&be, 8, m * n * 2);
            let (xt, wt, st, yt) = (
                t(&xb, 0, DType::Bf16, &[m, k]),
                t(&w, 8, DType::Fp8E4m3, &[n, k]),
                t(&sb, 4, DType::F32, &[2]),
                t(&y, 8, DType::Bf16, &[m, n]),
            );
            let call = |x: TensorRef, y: TensorRef| {
                OpCall::new(Op::MatmulFp8, vec![x, wt.clone(), st.clone()], vec![y], OpAttrs::default())
            };
            let serial = time_us(&be, 200, || {
                for r in 0..m {
                    be.launch("matmul_fp8_gemv", &call(xt.rows(r, 1), yt.rows(r, 1)), 0).unwrap();
                }
            });
            let rows = if m >= 2 {
                time_us(&be, 200, || be.launch("matmul_fp8_gemv_rows", &call(xt.clone(), yt.clone()), 0).unwrap())
            } else {
                serial
            };
            eprintln!(
                "TIMING fp8 {n}x{k} M={m}: {m} one-row launches {serial:.1} us, rows launch {rows:.1} us ({:.2}x)",
                serial / rows
            );
        }
    }
    {
        let (n, k) = (512usize, 2048usize);
        let mut r = Rng(5);
        let w: Vec<u8> = (0..n * k / 2).map(|_| (r.next_u64() >> 32) as u8).collect();
        let (wb, bb, gb) = (
            put_bytes(&be, 8, &w),
            put_bytes(&be, 3, &rand_block_scales(n * k / 16, 6)),
            put_bytes(&be, 4, &0.00071f32.to_le_bytes()),
        );
        for m in [1usize, 2, 4, 8] {
            let xb = put_bytes(&be, 0, &bf16_bytes(&act_rows(m, k, 9)));
            let y = guarded(&be, 8, m * n * 2);
            let (xt, yt) = (t(&xb, 0, DType::Bf16, &[m, k]), t(&y, 8, DType::Bf16, &[m, n]));
            let (wt, bt, gt) = (
                t(&wb, 8, DType::U8, &[n, k / 2]),
                t(&bb, 3, DType::Fp8E4m3, &[n, k / 16]),
                t(&gb, 4, DType::F32, &[1]),
            );
            let call = |x: TensorRef, y: TensorRef| {
                OpCall::new(Op::MatmulNvfp4, vec![x, wt.clone(), bt.clone(), gt.clone()], vec![y], OpAttrs::default())
            };
            let serial = time_us(&be, 200, || {
                for r in 0..m {
                    be.launch("matmul_nvfp4_gemv", &call(xt.rows(r, 1), yt.rows(r, 1)), 0).unwrap();
                }
            });
            let rows = if m >= 2 {
                time_us(&be, 200, || be.launch("matmul_nvfp4_gemv_rows", &call(xt.clone(), yt.clone()), 0).unwrap())
            } else {
                serial
            };
            eprintln!(
                "TIMING nvfp4 {n}x{k} M={m}: {m} one-row launches {serial:.1} us, rows launch {rows:.1} us ({:.2}x)",
                serial / rows
            );
        }
    }
    // moe_experts at 35B shapes, random routing (so ~57 of 64 pairs are distinct experts at T = 8) and identical routing
    let mut rng = SplitMix64::new(0x71_3e);
    let ex = random_experts(&mut rng, E, H, I);
    let (guw, gus, dw, ds, gs) = (
        put_host(&be, &ex.gate_up_w, 8),
        put_host(&be, &ex.gate_up_s, 0),
        put_host(&be, &ex.down_w, 8),
        put_host(&be, &ex.down_s, 0),
        put_host(&be, &ex.gscales, 0),
    );
    let ks = be.kernels();
    let (solo, rows_k) = (
        ks.iter().find(|k| k.name == "moe_experts_bf16").unwrap(),
        ks.iter().find(|k| k.name == "moe_experts_bf16_rows").unwrap(),
    );
    // 24 routings per kind, cycled call by call: the weights each call reads are mostly not in the cache (a decode step
    // reads 40 layers x ~57 experts for 8 tokens, far more than the system cache holds)
    const VARIANTS: usize = 24;
    for m in [1usize, 2, 4, 8] {
        let x = HostTensor::from_f32_as(DType::Bf16, &[m, H], &rng.normal_vec(m * H, 1.0));
        let xr = put_host(&be, &x, 0);
        for (kind, name) in [(0usize, "random"), (1, "identical")] {
            let ys = guarded(&be, 0, m * H * 2);
            let mk = |idt: &TensorRef, wst: &TensorRef, rows: usize, r0: usize, kern: &intarsia_core::KernelImpl| {
                let mut c = OpCall::new(
                    Op::MoeExperts,
                    vec![
                        xr.rows(r0, rows),
                        idt.rows(r0, rows),
                        wst.rows(r0, rows),
                        guw.clone(),
                        gus.clone(),
                        dw.clone(),
                        ds.clone(),
                        gs.clone(),
                    ],
                    vec![t(&ys, r0 * H * 2, DType::Bf16, &[rows, H])],
                    OpAttrs { top_k: K, ..OpAttrs::default() },
                );
                let need = kern.scratch_bytes(&ShapeInfo::from_call(&c));
                c.scratch = Some(TensorRef::new(be.alloc(need.max(16), MemKind::Device).unwrap(), DType::U8, &[need]));
                c
            };
            let mut solo_sets: Vec<Vec<OpCall>> = Vec::new();
            let mut rows_calls: Vec<OpCall> = Vec::new();
            for _ in 0..VARIANTS {
                let (_, ids, ws) = routings(&mut rng, m).into_iter().nth(kind).unwrap();
                let idt = put_host(&be, &HostTensor::from_i32(&[m, K], &ids), 0);
                let wst = put_host(&be, &HostTensor::from_f32(&[m, K], &ws), 0);
                solo_sets.push((0..m).map(|r| mk(&idt, &wst, 1, r, solo)).collect());
                if m >= 2 {
                    rows_calls.push(mk(&idt, &wst, m, 0, rows_k));
                }
            }
            let mut i = 0usize;
            let serial = time_us(&be, 96, || {
                i += 1;
                for c in &solo_sets[i % VARIANTS] {
                    be.launch("moe_experts_bf16", c, 0).unwrap();
                }
            });
            let rows = if m >= 2 {
                let mut i = 0usize;
                time_us(&be, 96, || {
                    i += 1;
                    be.launch("moe_experts_bf16_rows", &rows_calls[i % VARIANTS], 0).unwrap()
                })
            } else {
                serial
            };
            eprintln!("TIMING moe_experts(cold) T={m} routing={name}: {m} one-token calls {serial:.1} us, rows call {rows:.1} us ({:.2}x)", serial / rows);
        }
    }
}
