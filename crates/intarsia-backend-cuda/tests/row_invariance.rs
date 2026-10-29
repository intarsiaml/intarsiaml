//! Lane rip (row-invariant prefill, docs/results/2026-10-29-rip-row-invariant-prefill.md): which CUDA prefill kernels give a row
//! bits that depend on how many rows share the call, or on where the call starts?
//!
//! Every kernel runs once over a whole 4096-row (attention: 4352-row) "one-pass prefill", then over windows of it: `(first row,
//! rows)`, the window's rows sliced from the same inputs and, for attention, `pos0` = the window's first row over the same KV
//! cache (a later chunk, or a prompt resumed from a prefix). Each window's rows must equal the same rows of the one-pass call
//! byte for byte. Windows: one row anywhere (a resumed prompt's last token), 2..17 rows (the split-KV / GEMV row counts), 255 /
//! 256 rows (busy chunks), 1000 and the rest of the pass.
//!
//! The kernels a row-invariant prefill pass selects (`row_invariant` in src/kernels.rs) are asserted. The kernels an ordinary
//! prefill selects by row count (cuBLAS `matmul_bf16`, the M <= 4 GEMV, split-KV attention for 2..16 rows, the split-KV decode
//! kernel for one row) are only reported, with the windows where they differ: that list is the lane's per-op finding.
//! Needs a GPU: without one the tests print SKIPPED (and fail under CI=1 / INTARSIA_REQUIRE_GPU=1).
#![cfg(ik_cuda)]

mod common;

use intarsia_core::exec::{run_host, HostCall, OutSpec};
use intarsia_core::rng::SplitMix64;
use intarsia_core::{Backend, DType, HostTensor, MemKind, Op, OpAttrs, OpCall, TensorRef};
use std::sync::Arc;
use std::time::Instant;

fn device(test: &str) -> Option<Arc<dyn Backend>> {
    let ord = std::env::var("INTARSIA_CUDA_DEVICE").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
    match intarsia_backend_cuda::create(ord) {
        Ok(b) => Some(b),
        Err(e) => {
            common::skip(test, &format!("no usable CUDA device ({e})"));
            None
        }
    }
}

fn rows(t: &HostTensor, a: usize, b: usize) -> HostTensor {
    let row = t.nbytes() / t.shape[0];
    let mut shape = t.shape.clone();
    shape[0] = b - a;
    HostTensor::new(t.dtype, shape, t.data[a * row..b * row].to_vec()).unwrap()
}

fn bf(rng: &mut SplitMix64, shape: &[usize], scale: f32) -> HostTensor {
    let n: usize = shape.iter().product();
    HostTensor::from_f32_as(DType::Bf16, shape, &rng.normal_vec(n, scale))
}

/// Windows of a `total`-row pass: (first row, rows).
fn windows(total: usize) -> Vec<(usize, usize)> {
    let mut w = vec![
        (0, 1),
        (0, 2),
        (0, 4),
        (0, 5),
        (0, 7),
        (0, 16),
        (0, 17),
        (0, 255),
        (0, 256),
        (0, 1000),
        (1, 1),
        (1, 7),
        (7, 1),
        (7, 9),
        (100, 3),
        (255, 1),
        (255, 256),
        (256, 1),
        (256, 256),
        (513, 17),
        (1000, 1),
        (1000, 255),
        (2047, 2),
        (4095, 1),
        (1, 4095),
    ];
    if total > 4096 {
        w.extend([(4096, 1), (4096, total - 4096), (4100, 5), (total - 1, 1), (3000, total - 3000)]);
    }
    w.retain(|&(lo, n)| lo + n <= total);
    w
}

/// Run `kernel` once over all rows and once per window; returns the windows whose rows differ, as (first row, rows, rows that
/// differ). `row_inputs`: the inputs whose leading dimension is the row count; every output is row-wise. `at` adjusts the
/// window's call (attention: `pos0`).
fn differ(
    be: &dyn Backend,
    kernel: &str,
    hc: &HostCall,
    row_inputs: &[usize],
    wins: &[(usize, usize)],
    at: &dyn Fn(&mut HostCall, usize),
) -> Vec<(usize, usize, usize)> {
    let total = hc.inputs[row_inputs[0]].shape[0];
    let full = run_host(be, kernel, hc).unwrap_or_else(|e| panic!("{kernel}: {e}"));
    let again = run_host(be, kernel, hc).unwrap();
    for (a, b) in full.iter().zip(&again) {
        assert_eq!(a.data, b.data, "{kernel}: not repeatable");
    }
    let mut bad = Vec::new();
    for &(lo, len) in wins {
        let mut one = hc.clone();
        for &i in row_inputs {
            one.inputs[i] = rows(&hc.inputs[i], lo, lo + len);
        }
        one.outputs = hc
            .outputs
            .iter()
            .map(|o| {
                let mut shape = o.shape.clone();
                shape[0] = len;
                OutSpec::new(o.dtype, &shape)
            })
            .collect();
        at(&mut one, lo);
        let single = run_host(be, kernel, &one).unwrap_or_else(|e| panic!("{kernel} window ({lo}, {len}): {e}"));
        let mut n_bad = 0;
        for (f, s) in full.iter().zip(&single) {
            let row = f.nbytes() / total;
            let want = &f.data[lo * row..(lo + len) * row];
            n_bad = n_bad.max((0..len).filter(|r| want[r * row..(r + 1) * row] != s.data[r * row..(r + 1) * row]).count());
        }
        if n_bad > 0 {
            bad.push((lo, len, n_bad));
        }
    }
    bad
}

fn show(bad: &[(usize, usize, usize)]) -> String {
    if bad.is_empty() {
        return "every window identical".into();
    }
    bad.iter().map(|(lo, n, b)| format!("({lo},{n}):{b}")).collect::<Vec<_>>().join(" ")
}

const NO_AT: &dyn Fn(&mut HostCall, usize) = &|_, _| {};

/// The 8B projections: q / o 4096x4096, k / v 1024x4096, gate / up 12288x4096, down 4096x12288.
const PROJ: [(usize, usize); 4] = [(4096, 4096), (1024, 4096), (12288, 4096), (4096, 12288)];

#[test]
fn gemm_rows_do_not_depend_on_m() {
    let Some(be) = device("gemm_rows_do_not_depend_on_m") else { return };
    let mut rng = SplitMix64::new(0x71b);
    let total = 4096;
    let wins = windows(total);
    let small: Vec<(usize, usize)> = wins.iter().copied().filter(|&(_, n)| n <= 4).collect();
    for (n, k) in PROJ {
        let x = bf(&mut rng, &[total, k], 1.0);
        let w = bf(&mut rng, &[n, k], 1.0 / (k as f32).sqrt());
        let hc = HostCall { op: Op::Matmul, inputs: vec![x, w], outputs: vec![OutSpec::new(DType::Bf16, &[total, n])], attrs: OpAttrs::default() };
        let inv = differ(&*be, "matmul_bf16_inv", &hc, &[0], &wins, NO_AT);
        let cublas = differ(&*be, "matmul_bf16", &hc, &[0], &wins, NO_AT);
        // the ordinary M <= 4 path, against the cuBLAS one-pass rows
        let gemv: Vec<_> = {
            let full = run_host(&*be, "matmul_bf16", &hc).unwrap();
            small
                .iter()
                .filter(|&&(lo, len)| {
                    let mut one = hc.clone();
                    one.inputs[0] = rows(&hc.inputs[0], lo, lo + len);
                    one.outputs = vec![OutSpec::new(DType::Bf16, &[len, n])];
                    let g = run_host(&*be, "matmul_bf16_gemv", &one).unwrap();
                    g[0].data[..] != full[0].data[lo * n * 2..(lo + len) * n * 2]
                })
                .copied()
                .collect()
        };
        eprintln!("FINDING matmul_bf16 (cuBLAS GemmEx heuristic) N{n} K{k}: {}", show(&cublas));
        eprintln!("FINDING matmul_bf16_gemv (M <= 4) vs cuBLAS one pass N{n} K{k}: {} of {} windows differ", gemv.len(), small.len());
        eprintln!("matmul_bf16_inv N{n} K{k}: {}", show(&inv));
        assert!(inv.is_empty(), "matmul_bf16_inv N{n} K{k}: rows depend on M: {}", show(&inv));
    }
}

/// Lane rip2: `matmul_bf16_inv` runs, per row count, the cuBLASLt heuristic's pick when a probe proves it bitwise equal to the
/// shape's primary algorithm. Every row count 1..=4352 of each 8B projection, against one 4352-row pass: rows 0, M / 2 and M - 1
/// of every call (all rows up to M = 64) byte for byte. An algorithm with another summation order would differ in nearly every
/// output, so three rows per call cover the call's algorithm.
#[test]
fn gemm_inv_rows_equal_at_every_m() {
    let Some(be) = device("gemm_inv_rows_equal_at_every_m") else { return };
    let mut rng = SplitMix64::new(0x72a);
    let total = 4352;
    let max_m = total;
    for (n, k) in PROJ {
        let t0 = Instant::now();
        let up = |h: &HostTensor| {
            let buf = be.alloc(h.nbytes(), MemKind::Device).unwrap();
            be.upload(&buf, 0, &h.data, 0).unwrap();
            buf
        };
        let xb = up(&bf(&mut rng, &[total, k], 1.0));
        let wt = TensorRef::new(up(&bf(&mut rng, &[n, k], 1.0 / (k as f32).sqrt())), DType::Bf16, &[n, k]);
        let ob = be.alloc(total * n * 2, MemKind::Device).unwrap();
        let call = |m: usize| {
            let c = OpCall::new(
                Op::Matmul,
                vec![TensorRef::new(xb.clone(), DType::Bf16, &[m, k]), wt.clone()],
                vec![TensorRef::new(ob.clone(), DType::Bf16, &[m, n])],
                OpAttrs::default(),
            );
            be.launch("matmul_bf16_inv", &c, 0).unwrap();
            be.synchronize(0).unwrap();
        };
        call(total);
        let row = n * 2;
        let mut full = vec![0u8; total * row];
        be.download(&ob, 0, &mut full, 0).unwrap();
        let mut bad = Vec::new();
        let mut got = vec![0u8; 64 * row];
        for m in 1..=max_m {
            call(m);
            let check: Vec<usize> = if m <= 64 { (0..m).collect() } else { vec![0, m / 2, m - 1] };
            if m <= 64 {
                be.download(&ob, 0, &mut got[..m * row], 0).unwrap();
            }
            for &r in &check {
                let mut one = vec![0u8; row];
                let have: &[u8] = if m <= 64 {
                    &got[r * row..(r + 1) * row]
                } else {
                    be.download(&ob, r * row, &mut one, 0).unwrap();
                    &one
                };
                if have != &full[r * row..(r + 1) * row] {
                    bad.push((m, r));
                    break;
                }
            }
        }
        eprintln!(
            "matmul_bf16_inv N{n} K{k}: M 1..={max_m}: {} row counts differ from the {total}-row pass ({:.1} s){}",
            bad.len(),
            t0.elapsed().as_secs_f64(),
            if bad.is_empty() { String::new() } else { format!(": first (M, row) {:?}", &bad[..bad.len().min(8)]) }
        );
        assert!(bad.is_empty(), "matmul_bf16_inv N{n} K{k}: rows depend on M at {} row counts", bad.len());
    }
}

/// The head of a prefill pass: one row (`LastLogits`); rows of an M <= 4 GEMV call run the one-row loop.
#[test]
fn head_gemv_rows_do_not_depend_on_m() {
    let Some(be) = device("head_gemv_rows_do_not_depend_on_m") else { return };
    let mut rng = SplitMix64::new(0x71c);
    let (n, k) = (4096, 4096);
    let x = bf(&mut rng, &[4, k], 1.0);
    let w = bf(&mut rng, &[n, k], 1.0 / (k as f32).sqrt());
    let hc = HostCall { op: Op::Logits, inputs: vec![x, w], outputs: vec![OutSpec::new(DType::F32, &[4, n])], attrs: OpAttrs::default() };
    let wins = [(0, 1), (1, 1), (3, 1), (0, 2), (1, 3), (2, 2), (0, 3)];
    let bad = differ(&*be, "logits_bf16_gemv", &hc, &[0], &wins, NO_AT);
    assert!(bad.is_empty(), "logits_bf16_gemv: {}", show(&bad));
}

fn attn_case(rng: &mut SplitMix64, total: usize, n: usize, nkv: usize, d: usize) -> HostCall {
    let cap = total + 8;
    HostCall {
        op: Op::AttnPrefill,
        inputs: vec![bf(rng, &[total, n, d], 1.0), bf(rng, &[nkv, cap, d], 1.0), bf(rng, &[nkv, cap, d], 1.0)],
        outputs: vec![OutSpec::new(DType::Bf16, &[total, n, d])],
        attrs: OpAttrs { pos0: 0, scale: 1.0 / (d as f32).sqrt(), n_heads: n, n_kv_heads: nkv, head_dim: d, kv_cap: cap, ..OpAttrs::default() },
    }
}

#[test]
fn flash_attention_rows_do_not_depend_on_the_split() {
    let Some(be) = device("flash_attention_rows_do_not_depend_on_the_split") else { return };
    let mut rng = SplitMix64::new(0x71d);
    let total = 4352;
    let wins = windows(total);
    let at: &dyn Fn(&mut HostCall, usize) = &|c, lo| c.attrs.pos0 = lo;
    // the 8B (32 / 8 x 128), and a one-to-one head layout
    for (n, nkv, d) in [(32usize, 8usize, 128usize), (8, 8, 128)] {
        let hc = attn_case(&mut rng, total, n, nkv, d);
        let flash = differ(&*be, "attn_prefill_bf16_flash", &hc, &[0], &wins, at);
        eprintln!("attn_prefill_bf16_flash {n}/{nkv}x{d}: {}", show(&flash));
        // the ordinary short-chunk / one-row paths, against the flash one-pass rows (finding only)
        let full = run_host(&*be, "attn_prefill_bf16_flash", &hc).unwrap();
        let row = n * d * 2;
        let mut sk = Vec::new();
        let mut dec = Vec::new();
        for &(lo, len) in &wins {
            let mut one = hc.clone();
            one.inputs[0] = rows(&hc.inputs[0], lo, lo + len);
            one.outputs = vec![OutSpec::new(DType::Bf16, &[len, n, d])];
            one.attrs.pos0 = lo;
            let want = &full[0].data[lo * row..(lo + len) * row];
            if (2..=16).contains(&len) {
                let got = run_host(&*be, "attn_prefill_bf16_splitkv", &one).unwrap();
                if got[0].data[..] != *want {
                    sk.push((lo, len, 0));
                }
            }
            if len == 1 {
                let mut d1 = one.clone();
                d1.op = Op::AttnDecode;
                let got = run_host(&*be, "attn_decode_bf16_splitkv", &d1).unwrap();
                if got[0].data[..] != *want {
                    dec.push((lo, len, 0));
                }
            }
        }
        eprintln!("FINDING attn_prefill_bf16_splitkv (2..16 rows) vs flash one pass {n}/{nkv}x{d}: {}", show(&sk));
        eprintln!("FINDING attn_decode_bf16_splitkv (1 row) vs flash one pass {n}/{nkv}x{d}: {}", show(&dec));
        assert!(flash.is_empty(), "attn_prefill_bf16_flash {n}/{nkv}x{d}: rows depend on the split: {}", show(&flash));
    }
}

#[test]
fn row_wise_ops_do_not_depend_on_m() {
    let Some(be) = device("row_wise_ops_do_not_depend_on_m") else { return };
    let mut rng = SplitMix64::new(0x71e);
    let total = 4096;
    let wins = windows(total);
    let (h, nh, nkv, d, i) = (4096usize, 32usize, 8usize, 128usize, 12288usize);
    let eps = OpAttrs { eps: 1e-6, ..OpAttrs::default() };
    let mut check = |kernel: &str, hc: HostCall, row_inputs: &[usize], at: &dyn Fn(&mut HostCall, usize)| {
        let bad = differ(&*be, kernel, &hc, row_inputs, &wins, at);
        eprintln!("{kernel}: {}", show(&bad));
        assert!(bad.is_empty(), "{kernel}: rows depend on M: {}", show(&bad));
    };
    let x = bf(&mut rng, &[total, h], 1.0);
    let res = bf(&mut rng, &[total, h], 1.0);
    let w = bf(&mut rng, &[h], 0.3);
    check(
        "rmsnorm_bf16",
        HostCall { op: Op::RmsNorm, inputs: vec![x.clone(), w.clone()], outputs: vec![OutSpec::new(DType::Bf16, &[total, h])], attrs: eps.clone() },
        &[0],
        NO_AT,
    );
    check(
        "rmsnorm_add_bf16",
        HostCall {
            op: Op::RmsNormAdd,
            inputs: vec![x.clone(), res.clone(), w],
            outputs: vec![OutSpec::new(DType::Bf16, &[total, h]), OutSpec::new(DType::Bf16, &[total, h])],
            attrs: eps.clone(),
        },
        &[0, 1],
        NO_AT,
    );
    check(
        "add_bf16",
        HostCall { op: Op::Add, inputs: vec![x, res], outputs: vec![OutSpec::new(DType::Bf16, &[total, h])], attrs: OpAttrs::default() },
        &[0, 1],
        NO_AT,
    );
    let g = bf(&mut rng, &[total, i], 1.0);
    let u = bf(&mut rng, &[total, i], 1.0);
    check(
        "silu_glu_bf16",
        HostCall { op: Op::SiluGlu, inputs: vec![g, u], outputs: vec![OutSpec::new(DType::Bf16, &[total, i])], attrs: OpAttrs::default() },
        &[0, 1],
        NO_AT,
    );
    for heads in [nh, nkv] {
        let q = bf(&mut rng, &[total, heads, d], 1.0);
        let qw = bf(&mut rng, &[d], 0.3);
        check(
            "qk_norm_bf16",
            HostCall { op: Op::QkNorm, inputs: vec![q.clone(), qw], outputs: vec![OutSpec::new(DType::Bf16, &[total, heads, d])], attrs: eps.clone() },
            &[0],
            NO_AT,
        );
        let at: &dyn Fn(&mut HostCall, usize) = &|c, lo| c.attrs.pos0 = lo;
        check(
            "rope_bf16",
            HostCall {
                op: Op::Rope,
                inputs: vec![q],
                outputs: vec![OutSpec::new(DType::Bf16, &[total, heads, d])],
                attrs: OpAttrs { pos0: 0, theta: 1e6, head_dim: d, rotary_dim: 0, ..OpAttrs::default() },
            },
            &[0],
            at,
        );
    }
}

/// Timing (not a gate; `--ignored`): the 8B's seven projection GEMMs of one layer (q, k, v, o, gate, up, down) at prefill row
/// counts, cuBLAS `matmul_bf16` vs the row-invariant `matmul_bf16_inv`; prints microseconds per layer and the ratio.
/// `INTARSIA_INV_REF_M` picks the row count `matmul_bf16_inv` chooses its one algorithm for (with `INTARSIA_LT_DEBUG=1`: which).
#[test]
#[ignore]
fn gemm_inv_timing_at_8b_shapes() {
    let Some(be) = device("gemm_inv_timing_at_8b_shapes") else { return };
    // random bf16 operands (zeros would change the power-capped GPU's clocks); outputs zeroed
    let rng = std::cell::RefCell::new(SplitMix64::new(0x71f));
    let t = |shape: &[usize], fill: bool| {
        let n: usize = shape.iter().product();
        let buf = be.alloc(n * 2, MemKind::Device).unwrap();
        if fill {
            let h = bf(&mut rng.borrow_mut(), shape, 1.0 / 64.0);
            be.upload(&buf, 0, &h.data, 0).unwrap();
        } else {
            be.zero(&buf, 0).unwrap();
        }
        TensorRef::new(buf, DType::Bf16, shape)
    };
    let layer: [(usize, usize); 7] = [(4096, 4096), (1024, 4096), (1024, 4096), (4096, 4096), (12288, 4096), (12288, 4096), (4096, 12288)];
    let refm = std::env::var("INTARSIA_INV_REF_M").unwrap_or_else(|_| "4096".into());
    for m in [16usize, 64, 256, 512, 991, 1024, 1536, 2048, 3000, 4096] {
        let calls: Vec<OpCall> = layer
            .iter()
            .map(|&(n, k)| OpCall::new(Op::Matmul, vec![t(&[m, k], true), t(&[n, k], true)], vec![t(&[m, n], false)], OpAttrs::default()))
            .collect();
        let mut us = Vec::new();
        for name in ["matmul_bf16", "matmul_bf16_inv"] {
            for c in &calls {
                be.launch(name, c, 0).unwrap();
            }
            be.synchronize(0).unwrap();
            let iters = 10;
            let t0 = Instant::now();
            for _ in 0..iters {
                for c in &calls {
                    be.launch(name, c, 0).unwrap();
                }
            }
            be.synchronize(0).unwrap();
            us.push(t0.elapsed().as_secs_f64() * 1e6 / iters as f64);
        }
        println!("TIMING ref_m {refm} M {m:>5}: cublas {:>8.1} us/layer  inv {:>8.1} us/layer  inv/cublas {:.3}", us[0], us[1], us[1] / us[0]);
    }
}
