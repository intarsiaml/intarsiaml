#![cfg(target_os = "macos")]
//! `GdnConv` / `GdnDeltaRule` (gdn_conv_bf16, gdn_delta_rule_bf16): chunked prefill (T = 37 split 1 + 20 + 16, and a
//! 1 + 70 + 29 split that crosses the conv's 32-token tile and takes its separate state dispatch) is bitwise one call,
//! and both match the f64 reference of intarsia_core::gdn (whole sequence); a 512-token delta-rule run stays within the
//! budget; conv taps K in {2, 4, 8} and delta-rule dk in {32, 64, 128, 256}; operands sit at a 2-byte offset (the leftover-element path).

mod common;
use common::*;
use intarsia_backend_metal::bench::{bf16_to_f32, f32_to_bf16};
use intarsia_backend_metal::kernels::{GATE_SPLIT_BF16, GDN_CONV_BF16, GDN_DELTA_RULE_BF16, SIGMOID_MUL_BF16};
use intarsia_backend_metal::MetalBackend;
use intarsia_core::backend::{Backend, Buffer, MemKind, TensorRef};
use intarsia_core::gdn::{self, DeltaDims};
use intarsia_core::{DType, Op, OpAttrs, OpCall};

/// Leading bf16 elements before every operand: byte offset 2, so `bind` hands the shader a skip of 1.
const OFF: usize = 2;

fn upload_off(be: &MetalBackend, v: &[u16]) -> Buffer {
    let mut p = vec![0u16; OFF / 2];
    p.extend_from_slice(v);
    p.extend_from_slice(&[0u16; 8]);
    upload(be, &p)
}

/// Rows `[lo, lo + n)` of a `[*, w]` bf16 tensor stored at `OFF` in `buf`, as a tensor of shape `[n, tail...]`.
fn rows(buf: &Buffer, w: usize, lo: usize, shape: &[usize]) -> TensorRef {
    tref(buf, OFF + lo * w * 2, shape)
}

fn f32_buf(be: &MetalBackend, v: &[f32]) -> Buffer {
    let b = be.alloc(v.len() * 4, MemKind::Device).unwrap();
    // SAFETY: f32 -> u8 view of the same memory.
    let bytes = unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) };
    be.upload(&b, 0, bytes, 0).unwrap();
    b
}

fn f32_tref(buf: &Buffer, shape: &[usize]) -> TensorRef {
    TensorRef { buf: buf.clone(), offset: 0, dtype: DType::F32, shape: shape.to_vec() }
}

fn read_bf16(be: &MetalBackend, buf: &Buffer, n: usize) -> Vec<f64> {
    let mut b = vec![0u8; n * 2];
    be.download(buf, OFF, &mut b, 0).unwrap();
    b.chunks(2).map(|c| bf16_to_f32(u16::from_le_bytes([c[0], c[1]])) as f64).collect()
}

fn read_f32(be: &MetalBackend, buf: &Buffer, n: usize) -> Vec<f32> {
    let mut b = vec![0u8; n * 4];
    be.download(buf, 0, &mut b, 0).unwrap();
    b.chunks(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

fn f64s(v: &[u16]) -> Vec<f64> {
    v.iter().map(|&b| bf16_to_f32(b) as f64).collect()
}

/// Relative L2 error of `got` against `want` rounded to bf16 (the output dtype).
fn rel_l2_bf16(got: &[f64], want: &[f64]) -> f64 {
    let r: Vec<f64> = want.iter().map(|&x| bf16_to_f32(f32_to_bf16(x as f32)) as f64).collect();
    rel_l2(got, &r)
}

fn rel_l2(got: &[f64], want: &[f64]) -> f64 {
    assert_eq!(got.len(), want.len());
    let (mut num, mut den) = (0.0f64, 0.0f64);
    for (g, w) in got.iter().zip(want) {
        num += (g - w) * (g - w);
        den += w * w;
    }
    (num / den.max(1e-30)).sqrt()
}

/// Runs the conv over `chunks` (token counts) and returns (y, final state).
fn run_conv(be: &MetalBackend, x: &[u16], w: &[u16], t: usize, c: usize, k: usize, chunks: &[usize]) -> (Vec<f64>, Vec<f32>) {
    assert_eq!(chunks.iter().sum::<usize>(), t);
    let (xb, wb) = (upload_off(be, x), upload_off(be, w));
    let yb = upload_off(be, &vec![0xEEEEu16; t * c]);
    let sb = f32_buf(be, &vec![0.0; c * (k - 1)]);
    let mut lo = 0;
    for &n in chunks {
        let call = OpCall::new(
            Op::GdnConv,
            vec![rows(&xb, c, lo, &[n, c]), tref(&wb, OFF, &[c, k])],
            vec![rows(&yb, c, lo, &[n, c]), f32_tref(&sb, &[c, k - 1])],
            OpAttrs::default(),
        );
        be.launch(GDN_CONV_BF16, &call, 0).unwrap();
        lo += n;
    }
    be.synchronize(0).unwrap();
    (read_bf16(be, &yb, t * c), read_f32(be, &sb, c * (k - 1)))
}

#[test]
fn conv_chunked_prefill_equals_one_call_and_the_reference() {
    let be = backend();
    let c = 300usize;
    // tap counts: 2 (shortest), 4 (the checkpoint's), 8 (longest registered)
    for (t, split, k) in [2usize, 4, 8].into_iter().flat_map(|k| {
        [(37usize, vec![1usize, 20, 16]), (100, vec![1, 70, 29]), (5, vec![1, 1, 1, 1, 1]), (2, vec![2])].into_iter().map(move |(t, s)| (t, s, k))
    }) {
        let x = rand_bf16(t * c, 11 + t as u64);
        let w = rand_bf16(c * k, 5);
        let (y1, s1) = run_conv(&be, &x, &w, t, c, k, &[t]);
        let (yc, sc) = run_conv(&be, &x, &w, t, c, k, &split);
        assert_eq!(y1, yc, "K={k} T={t} {split:?}: chunked y differs from one call");
        assert_eq!(s1, sc, "K={k} T={t} {split:?}: chunked state differs from one call");
        let mut st = vec![0.0f64; c * (k - 1)];
        let want = gdn::conv_silu(&f64s(&x), &f64s(&w), &mut st, t, c, k);
        let err = rel_l2_bf16(&y1, &want);
        assert!(err < 1e-3, "K={k} T={t}: y vs reference rel_l2 {err:e}");
        let st32: Vec<f64> = s1.iter().map(|&v| v as f64).collect();
        assert_eq!(st32, st, "K={k} T={t}: state vs reference"); // the state holds bf16 inputs: exact
    }
}

struct Delta {
    qkv: Vec<u16>,
    a: Vec<u16>,
    b: Vec<u16>,
    a_log: Vec<u16>,
    dt_bias: Vec<u16>,
}

fn delta_inputs(t: usize, d: &DeltaDims, seed: u64) -> Delta {
    let bf = |v: f32| f32_to_bf16(v);
    // A_log = ln(U(1, 16)), dt_bias ~ U(-1, 0.5) like the checkpoint's order of magnitude
    let a_log = rand_bf16(d.hv, seed + 3).iter().map(|&u| bf((8.5 + 7.5 * bf16_to_f32(u)).ln())).collect();
    let dt_bias = rand_bf16(d.hv, seed + 4).iter().map(|&u| bf(bf16_to_f32(u) * 0.75 - 0.25)).collect();
    Delta {
        qkv: rand_bf16(t * d.conv_dim(), seed),
        a: rand_bf16(t * d.hv, seed + 1),
        b: rand_bf16(t * d.hv, seed + 2),
        a_log,
        dt_bias,
    }
}

const EPS: f32 = 1e-6;

/// Runs the delta rule over `chunks` from a zero state; returns (o, final state).
fn run_delta(be: &MetalBackend, i: &Delta, d: &DeltaDims, chunks: &[usize]) -> (Vec<f64>, Vec<f32>) {
    let (t, c, hv, dv, dk) = (d.t, d.conv_dim(), d.hv, d.dv, d.dk);
    assert_eq!(chunks.iter().sum::<usize>(), t);
    let (qb, ab, bb) = (upload_off(be, &i.qkv), upload_off(be, &i.a), upload_off(be, &i.b));
    let (lb, db) = (upload_off(be, &i.a_log), upload_off(be, &i.dt_bias));
    let ob = upload_off(be, &vec![0xEEEEu16; t * hv * dv]);
    let sb = f32_buf(be, &vec![0.0; hv * dv * dk]);
    let mut lo = 0;
    for &n in chunks {
        let call = OpCall::new(
            Op::GdnDeltaRule,
            vec![
                rows(&qb, c, lo, &[n, c]),
                rows(&ab, hv, lo, &[n, hv]),
                rows(&bb, hv, lo, &[n, hv]),
                tref(&lb, OFF, &[hv]),
                tref(&db, OFF, &[hv]),
            ],
            vec![rows(&ob, hv * dv, lo, &[n, hv, dv]), f32_tref(&sb, &[hv, dv, dk])],
            OpAttrs { eps: EPS, ..OpAttrs::default() },
        );
        be.launch(GDN_DELTA_RULE_BF16, &call, 0).unwrap();
        lo += n;
    }
    be.synchronize(0).unwrap();
    (read_bf16(be, &ob, t * hv * dv), read_f32(be, &sb, hv * dv * dk))
}

fn reference_delta(i: &Delta, d: &DeltaDims) -> (Vec<f64>, Vec<f64>) {
    let mut st = vec![0.0f64; d.hv * d.dv * d.dk];
    let o = gdn::delta_rule(d, EPS as f64, &f64s(&i.qkv), &f64s(&i.a), &f64s(&i.b), &f64s(&i.a_log), &f64s(&i.dt_bias), &mut st);
    (o, st)
}

fn check_delta(be: &MetalBackend, dk: usize, t: usize, split: &[usize], o_tol: f64, st_tol: f64) {
    check_delta_heads(be, 2, 4, dk, t, split, o_tol, st_tol);
}

#[allow(clippy::too_many_arguments)]
fn check_delta_heads(be: &MetalBackend, hk: usize, hv: usize, dk: usize, t: usize, split: &[usize], o_tol: f64, st_tol: f64) {
    let d = DeltaDims { t, hk, hv, dk, dv: 128 };
    let i = delta_inputs(t, &d, 100 + t as u64);
    let (o, st) = run_delta(be, &i, &d, split);
    if split.len() > 1 {
        let (o1, st1) = run_delta(be, &i, &d, &[t]);
        assert_eq!(o1, o, "dk={dk} T={t} {split:?}: chunked o differs from one call");
        assert_eq!(st1, st, "dk={dk} T={t} {split:?}: chunked state differs from one call");
    }
    let (want, want_st) = reference_delta(&i, &d);
    let e = rel_l2_bf16(&o, &want);
    let es = rel_l2(&st.iter().map(|&v| v as f64).collect::<Vec<_>>(), &want_st);
    assert!(e < o_tol, "hk={hk} hv={hv} dk={dk} T={t} {split:?}: o vs reference rel_l2 {e:e}");
    assert!(es < st_tol, "hk={hk} hv={hv} dk={dk} T={t} {split:?}: state vs reference rel_l2 {es:e}");
}

#[test]
fn delta_rule_chunked_prefill_equals_one_call_and_the_reference() {
    let be = backend();
    for dk in [32, 64, 128, 256] {
        check_delta(&be, dk, 37, &[1, 20, 16], 1e-3, 1e-5);
        check_delta(&be, dk, 1, &[1], 1e-3, 1e-5);
    }
}

#[test]
fn delta_rule_long_sequence_stays_within_the_budget() {
    let be = backend();
    for dk in [32, 64, 128, 256] {
        check_delta(&be, dk, 512, &[512], 1e-3, 1e-4);
        check_delta(&be, dk, 512, &[1, 255, 256], 1e-3, 1e-4);
    }
}

/// The Qwen3.6-35B-A3B linear-attention layer shape (config.json: 16 key heads, 32 value heads, dk = dv = 128, so
/// conv_dim 8192): T = 300 split mid-sequence (1 + 127 + 172, so the carried state crosses chunk boundaries and the
/// conv's 32-token tile) and a single decode token, against the f64 reference.
#[test]
fn delta_rule_at_the_35b_layer_shape() {
    let be = backend();
    check_delta_heads(&be, 16, 32, 128, 300, &[1, 127, 172], 1e-3, 1e-4);
    check_delta_heads(&be, 16, 32, 128, 1, &[1], 1e-3, 1e-5);
    check_delta_heads(&be, 16, 32, 128, 1024, &[1024], 1e-3, 1e-4);
}

#[test]
fn conv_at_the_35b_channel_count() {
    let be = backend();
    let (c, k) = (8192usize, 4usize); // conv_dim = 2 * 16 * 128 + 32 * 128, linear_conv_kernel_dim 4
    for (t, split) in [(300usize, vec![1usize, 127, 172]), (1, vec![1]), (65, vec![33, 32])] {
        let x = rand_bf16(t * c, 21 + t as u64);
        let w = rand_bf16(c * k, 9);
        let (y1, s1) = run_conv(&be, &x, &w, t, c, k, &[t]);
        let (yc, sc) = run_conv(&be, &x, &w, t, c, k, &split);
        assert_eq!(y1, yc, "T={t} {split:?}: chunked y differs from one call");
        assert_eq!(s1, sc, "T={t} {split:?}: chunked state differs from one call");
        let mut st = vec![0.0f64; c * (k - 1)];
        let want = gdn::conv_silu(&f64s(&x), &f64s(&w), &mut st, t, c, k);
        let err = rel_l2_bf16(&y1, &want);
        assert!(err < 1e-3, "T={t}: y vs reference rel_l2 {err:e}");
        let st32: Vec<f64> = s1.iter().map(|&v| v as f64).collect();
        assert_eq!(st32, st, "T={t}: state vs reference");
    }
}

/// `GateSplit` at the full-attention layer's shape (16 heads, head_dim 256: q_proj emits [T, 16, 2 * 256] = q | gate per
/// head): both halves are exact copies of the input rows, with the input and both outputs at a 2-byte offset.
#[test]
fn gate_split_is_an_exact_per_head_split() {
    let be = backend();
    let (n, d) = (16usize, 256usize);
    for t in [1usize, 3, 300] {
        let x = rand_bf16(t * n * 2 * d, 31 + t as u64);
        let xb = upload_off(&be, &x);
        let qb = upload_off(&be, &vec![0xEEEEu16; t * n * d]);
        let gb = upload_off(&be, &vec![0xEEEEu16; t * n * d]);
        let call = OpCall::new(
            Op::GateSplit,
            vec![tref(&xb, OFF, &[t, n, 2 * d])],
            vec![tref(&qb, OFF, &[t, n, d]), tref(&gb, OFF, &[t, n, d])],
            OpAttrs::default(),
        );
        be.launch(GATE_SPLIT_BF16, &call, 0).unwrap();
        be.synchronize(0).unwrap();
        let (wq, wg) = gdn::gate_split(&f64s(&x), t, n, d);
        assert_eq!(read_bf16(&be, &qb, t * n * d), wq, "T={t}: q half");
        assert_eq!(read_bf16(&be, &gb, t * n * d), wg, "T={t}: gate half");
    }
}

/// `SigmoidMul` (a * sigmoid(g), f32 math, one bf16 rounding) at the attention output's size, with g spread over +-12 so the
/// sigmoid saturates at both ends; operands and output at a 2-byte offset. Within one bf16 ulp (2^-8 relative) of f64.
#[test]
fn sigmoid_mul_matches_the_reference() {
    let be = backend();
    let (n, d) = (16usize, 256usize);
    for t in [1usize, 3, 300] {
        let len = t * n * d;
        let a = rand_bf16(len, 41 + t as u64);
        let g: Vec<u16> = rand_bf16(len, 43 + t as u64).iter().map(|&u| f32_to_bf16(bf16_to_f32(u) * 12.0)).collect();
        let (ab, gb) = (upload_off(&be, &a), upload_off(&be, &g));
        let ob = upload_off(&be, &vec![0xEEEEu16; len]);
        let call = OpCall::new(
            Op::SigmoidMul,
            vec![tref(&ab, OFF, &[t, n * d]), tref(&gb, OFF, &[t, n * d])],
            vec![tref(&ob, OFF, &[t, n * d])],
            OpAttrs::default(),
        );
        be.launch(SIGMOID_MUL_BF16, &call, 0).unwrap();
        be.synchronize(0).unwrap();
        let got = read_bf16(&be, &ob, len);
        let want = gdn::sigmoid_mul(&f64s(&a), &f64s(&g));
        let mut worst = 0f64;
        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            let tol = w.abs() * 2f64.powi(-8) + 1e-30;
            assert!((g - w).abs() <= tol, "T={t} [{i}]: gpu {g} vs reference {w}");
            worst = worst.max((g - w).abs() / tol);
        }
        eprintln!("sigmoid_mul T={t}: worst error {worst:.3} of tolerance");
    }
}
