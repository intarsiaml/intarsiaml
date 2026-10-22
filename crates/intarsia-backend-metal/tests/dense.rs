//! The dense ops and the per-stream command-buffer encoding, on the GPU: GEMM against an f64 reference (including
//! odd element offsets), dependent launches in one encoder, host access with work pending, device-side positions. Attention has its
//! own file (attention.rs). The cpu-oracle comparison of every kernel is `intarsia kernels test --backend
//! metal`.

#![cfg(target_os = "macos")]

mod common;

use common::*;
use intarsia_backend_metal::bench::{bf16_to_f32, f32_to_bf16};
use intarsia_backend_metal::kernels::{ADD_BF16, LOGITS_BF16, LOGITS_BF16_GEMV, MATMUL_BF16, MATMUL_BF16_GEMV};
use intarsia_backend_metal::{MetalBackend, MAX_DISPATCHES};
use intarsia_core::backend::{Backend, Buffer, MemKind, TensorRef};
use intarsia_core::{DType, Op, OpAttrs, OpCall, ShapeInfo};

/// Upload `data` at byte offset `off` of a fresh buffer (the bytes before and after are 0xEE guards).
fn upload_at(be: &MetalBackend, off: usize, data: &[u16]) -> Buffer {
    let b = be.alloc(off + data.len() * 2 + 16, MemKind::Device).unwrap();
    be.upload(&b, 0, &vec![0xEEu8; off + data.len() * 2 + 16], 0).unwrap();
    be.upload(&b, off, as_bytes(data), 0).unwrap();
    b
}

fn t(buf: &Buffer, off: usize, dt: DType, shape: &[usize]) -> TensorRef {
    TensorRef { buf: buf.clone(), offset: off, dtype: dt, shape: shape.to_vec() }
}

fn read_u16(be: &MetalBackend, b: &Buffer, off: usize, n: usize) -> Vec<u16> {
    let mut raw = vec![0u8; n * 2];
    be.download(b, off, &mut raw, 0).unwrap();
    raw.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect()
}

fn read_f32(be: &MetalBackend, b: &Buffer, off: usize, n: usize) -> Vec<f32> {
    let mut raw = vec![0u8; n * 4];
    be.download(b, off, &mut raw, 0).unwrap();
    raw.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

/// (m, n, k, byte offset of x, w, y): tile edges in every dimension, odd element offsets (2 mod 4 bytes).
const GEMM_SHAPES: &[(usize, usize, usize, usize, usize, usize)] = &[
    (1, 1, 1, 0, 0, 0),
    (2, 3, 5, 2, 2, 2),
    (3, 33, 70, 0, 0, 0),
    (32, 32, 32, 0, 0, 0),
    (33, 65, 129, 2, 0, 2),
    (31, 31, 31, 0, 2, 0),
    (70, 100, 300, 2, 2, 2),
    (5, 1000, 4096, 0, 0, 0),
    (200, 130, 1024, 0, 0, 0),
];

#[test]
fn gemm_matches_f64_reference() {
    let be = backend();
    for (si, &(m, n, k, xo, wo, yo)) in GEMM_SHAPES.iter().enumerate() {
        let x = rand_bf16(m * k, 10 + si as u64);
        let w = rand_bf16(n * k, 99 + si as u64);
        let (xb, wb) = (upload_at(&be, xo, &x), upload_at(&be, wo, &w));
        let yb = be.alloc(yo + m * n * 2 + 16, MemKind::Device).unwrap();
        be.upload(&yb, 0, &vec![0xEEu8; yo + m * n * 2 + 16], 0).unwrap();
        let call = matmul_call(
            t(&xb, xo, DType::Bf16, &[m, k]),
            t(&wb, wo, DType::Bf16, &[n, k]),
            t(&yb, yo, DType::Bf16, &[m, n]),
        );
        be.launch(MATMUL_BF16, &call, 0).unwrap();
        be.synchronize(0).unwrap();
        let y = read_u16(&be, &yb, yo, m * n);
        for mi in 0..m {
            let refs = reference(&w, &x[mi * k..(mi + 1) * k], k, n);
            check(&y[mi * n..(mi + 1) * n], &refs, 0..n, &format!("matmul_bf16 m={m} n={n} k={k} row {mi}"));
        }
        // guards around y untouched
        let raw = read_u16(&be, &yb, 0, (yo + m * n * 2 + 16) / 2);
        assert!(
            raw[..yo / 2].iter().all(|v| *v == 0xEEEE) && raw[(yo / 2 + m * n)..].iter().all(|v| *v == 0xEEEE),
            "guard bytes overwritten"
        );
    }
}

#[test]
fn logits_gemm_and_gemv_match_f64_reference() {
    let be = backend();
    for (si, &(m, n, k, xo, wo, yo)) in [
        (1usize, 1000usize, 4096usize, 0usize, 0usize, 0usize),
        (1, 777, 123, 2, 2, 0),
        (1, 64, 64, 0, 0, 4),
        (4, 300, 200, 0, 2, 0),
        (33, 129, 130, 2, 0, 0),
    ]
    .iter()
    .enumerate()
    {
        let x = rand_bf16(m * k, 5 + si as u64);
        let w = rand_bf16(n * k, 77 + si as u64);
        let (xb, wb) = (upload_at(&be, xo, &x), upload_at(&be, wo, &w));
        let kernels: &[&str] = if m == 1 { &[LOGITS_BF16, LOGITS_BF16_GEMV] } else { &[LOGITS_BF16] };
        for &kn in kernels {
            let yb = be.alloc(yo + m * n * 4 + 16, MemKind::Device).unwrap();
            let call = OpCall::new(
                Op::Logits,
                vec![t(&xb, xo, DType::Bf16, &[m, k]), t(&wb, wo, DType::Bf16, &[n, k])],
                vec![t(&yb, yo, DType::F32, &[m, n])],
                OpAttrs::default(),
            );
            be.launch(kn, &call, 0).unwrap();
            be.synchronize(0).unwrap();
            let y = read_f32(&be, &yb, yo, m * n);
            for mi in 0..m {
                let refs = reference(&w, &x[mi * k..(mi + 1) * k], k, n);
                for (ni, &(r, l1)) in refs.iter().enumerate() {
                    let g = y[mi * n + ni] as f64;
                    assert!((g - r).abs() <= 2e-6 * l1 + 1e-9, "{kn} m={m} n={n} k={k} [{mi},{ni}]: gpu {g} ref {r}");
                }
            }
        }
    }
}

/// A chain of dependent adds in one stream, each launch reading the previous one's output, long enough to cross
/// MAX_DISPATCHES so a command buffer is committed mid-chain. A regression guard for the stream / flush logic. It does
/// not tell a serial encoder from a concurrent one: `alloc` buffers are hazard tracked, so Metal orders the dispatches
/// either way. The serial encoder keeps the ordering independent of the tracking mode (untracked buffers exist).
#[test]
fn dependent_launches_are_ordered_across_command_buffers() {
    let be = backend();
    let n = 1 << 16;
    let steps = MAX_DISPATCHES + 300;
    let a: Vec<u16> = (0..n).map(|i| f32_to_bf16((i % 7) as f32 * 0.25)).collect();
    let b: Vec<u16> = (0..n).map(|i| f32_to_bf16(((i % 5) as f32 - 2.0) * 0.125)).collect();
    let (ab, bb) = (upload_at(&be, 0, &a), upload_at(&be, 0, &b));
    let (p, q) = (be.alloc(n * 2, MemKind::Device).unwrap(), be.alloc(n * 2, MemKind::Device).unwrap());
    let call = |x: &Buffer, y: &Buffer, out: &Buffer| {
        OpCall::new(
            Op::Add,
            vec![t(x, 0, DType::Bf16, &[1, n]), t(y, 0, DType::Bf16, &[1, n])],
            vec![t(out, 0, DType::Bf16, &[1, n])],
            OpAttrs::default(),
        )
    };
    be.launch(ADD_BF16, &call(&ab, &bb, &p), 0).unwrap();
    let mut cur = &p;
    for _ in 1..steps {
        let next = if std::ptr::eq(cur, &p) { &q } else { &p };
        be.launch(ADD_BF16, &call(cur, &bb, next), 0).unwrap();
        cur = next;
    }
    // no synchronize: download must flush and wait by itself
    let got = read_u16(&be, cur, 0, n);
    let mut want = a.clone();
    for _ in 0..steps {
        for (w, bv) in want.iter_mut().zip(&b) {
            *w = f32_to_bf16(bf16_to_f32(*w) + bf16_to_f32(*bv));
        }
    }
    assert_eq!(got, want);
}

#[test]
fn upload_after_pending_launch_waits_for_it() {
    let be = backend();
    let n = 4096;
    let a = rand_bf16(n, 1);
    let ab = upload_at(&be, 0, &a);
    let out = be.alloc(n * 2, MemKind::Device).unwrap();
    let call = OpCall::new(
        Op::Add,
        vec![t(&ab, 0, DType::Bf16, &[1, n]), t(&ab, 0, DType::Bf16, &[1, n])],
        vec![t(&out, 0, DType::Bf16, &[1, n])],
        OpAttrs::default(),
    );
    be.launch(ADD_BF16, &call, 0).unwrap();
    // overwrite the input while the launch may still be pending: the launch must have seen the original data
    be.upload(&ab, 0, &vec![0u8; n * 2], 0).unwrap();
    let got = read_u16(&be, &out, 0, n);
    let want: Vec<u16> = a.iter().map(|v| f32_to_bf16(2.0 * bf16_to_f32(*v))).collect();
    assert_eq!(got, want);
    // zero() goes through the same rule
    be.zero(&out, 0).unwrap();
    assert!(read_u16(&be, &out, 0, n).iter().all(|v| *v == 0));
}

#[test]
fn elementwise_ops_with_odd_element_offsets() {
    let be = backend();
    let n = 1001;
    let a = rand_bf16(n, 3);
    let b = rand_bf16(n, 4);
    for (ao, bo, oo) in [(2usize, 0usize, 2usize), (0, 2, 0), (2, 2, 2)] {
        let (ab, bb) = (upload_at(&be, ao, &a), upload_at(&be, bo, &b));
        let out = be.alloc(oo + n * 2 + 16, MemKind::Device).unwrap();
        be.upload(&out, 0, &vec![0xEEu8; oo + n * 2 + 16], 0).unwrap();
        let call = OpCall::new(
            Op::Add,
            vec![t(&ab, ao, DType::Bf16, &[1, n]), t(&bb, bo, DType::Bf16, &[1, n])],
            vec![t(&out, oo, DType::Bf16, &[1, n])],
            OpAttrs::default(),
        );
        be.launch(ADD_BF16, &call, 0).unwrap();
        let got = read_u16(&be, &out, oo, n);
        let want: Vec<u16> = a.iter().zip(&b).map(|(x, y)| f32_to_bf16(bf16_to_f32(*x) + bf16_to_f32(*y))).collect();
        assert_eq!(got, want, "offsets {ao} {bo} {oo}");
        let raw = read_u16(&be, &out, 0, (oo + n * 2 + 16) / 2);
        assert!(raw[..oo / 2].iter().all(|v| *v == 0xEEEE) && raw[oo / 2 + n..].iter().all(|v| *v == 0xEEEE));
    }
}

/// Rope and KvAppend with `OpCall::pos` (a device i32 [1]) are bit-identical to the host `pos0` launch.
#[test]
fn device_side_position_matches_host_position() {
    let be = backend();
    let (tt, nh, d, cap) = (3usize, 4usize, 64usize, 40usize);
    let x = rand_bf16(tt * nh * d, 8);
    let xb = upload_at(&be, 0, &x);
    let pos0 = 17usize;
    let posb = be.alloc(4, MemKind::Device).unwrap();
    be.upload(&posb, 0, &(pos0 as i32).to_le_bytes(), 0).unwrap();
    let pos_t = TensorRef { buf: posb.clone(), offset: 0, dtype: DType::I32, shape: vec![1] };

    let rope = |dev: bool| {
        let out = be.alloc(x.len() * 2, MemKind::Device).unwrap();
        let mut call = OpCall::new(
            Op::Rope,
            vec![t(&xb, 0, DType::Bf16, &[tt, nh, d])],
            vec![t(&out, 0, DType::Bf16, &[tt, nh, d])],
            OpAttrs { pos0: if dev { 0 } else { pos0 }, theta: 1e6, head_dim: d, ..Default::default() },
        );
        if dev {
            call.pos = Some(pos_t.clone());
            call.attrs.pos0 = pos0; // bounds the call; the kernel reads the tensor
        }
        be.launch("rope_bf16", &call, 0).unwrap();
        read_u16(&be, &out, 0, x.len())
    };
    assert_eq!(rope(false), rope(true));

    let append = |dev: bool| {
        let cache = be.alloc(nh * cap * d * 2, MemKind::Device).unwrap();
        be.zero(&cache, 0).unwrap();
        let mut call = OpCall::new(
            Op::KvAppend,
            vec![t(&xb, 0, DType::Bf16, &[tt, nh, d])],
            vec![t(&cache, 0, DType::Bf16, &[nh, cap, d])],
            OpAttrs { pos0, kv_cap: cap, n_kv_heads: nh, head_dim: d, ..Default::default() },
        );
        if dev {
            call.pos = Some(pos_t.clone());
        }
        be.launch("kv_append_bf16", &call, 0).unwrap();
        read_u16(&be, &cache, 0, nh * cap * d)
    };
    let (h, dv) = (append(false), append(true));
    assert_eq!(h, dv);
    // rows outside [pos0, pos0+T) keep their (zero) bits; appended rows hold the new values
    for hh in 0..nh {
        for r in 0..cap {
            let row = &h[(hh * cap + r) * d..(hh * cap + r + 1) * d];
            if (pos0..pos0 + tt).contains(&r) {
                assert_eq!(row, &x[((r - pos0) * nh + hh) * d..][..d]);
            } else {
                assert!(row.iter().all(|v| *v == 0));
            }
        }
    }
}

#[test]
fn argmax_ties_nan_and_all_neg_inf() {
    let be = backend();
    let v = 1000usize;
    let mut rows: Vec<Vec<f32>> = vec![vec![0.5; v], vec![f32::NEG_INFINITY; v], vec![f32::NAN; v]];
    let mut r3 = vec![0.0f32; v];
    r3[10] = 3.0;
    r3[900] = 3.0;
    r3[500] = f32::NAN;
    rows.push(r3);
    let flat: Vec<f32> = rows.iter().flatten().copied().collect();
    let lb = be.alloc(flat.len() * 4, MemKind::Device).unwrap();
    // SAFETY: f32 -> u8 view of the same memory.
    be.upload(&lb, 0, unsafe { std::slice::from_raw_parts(flat.as_ptr() as *const u8, flat.len() * 4) }, 0).unwrap();
    let ob = be.alloc(rows.len() * 4, MemKind::Device).unwrap();
    let call = OpCall::new(
        Op::Argmax,
        vec![t(&lb, 0, DType::F32, &[rows.len(), v])],
        vec![t(&ob, 0, DType::I32, &[rows.len()])],
        OpAttrs::default(),
    );
    be.launch("argmax_f32", &call, 0).unwrap();
    let mut raw = vec![0u8; rows.len() * 4];
    be.download(&ob, 0, &mut raw, 0).unwrap();
    let got: Vec<i32> = raw.chunks_exact(4).map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    // the oracle's rule: lowest index among maxima, NaN never wins, nothing above -inf gives 0
    assert_eq!(got, vec![0, 0, 0, 10]);
}

#[test]
fn predicates_route_m1_to_gemv() {
    let be = backend();
    let b = be.alloc(1 << 12, MemKind::Device).unwrap();
    let mm = |m: usize| {
        OpCall::new(
            Op::Matmul,
            vec![t(&b, 0, DType::Bf16, &[m, 8]), t(&b, 0, DType::Bf16, &[4, 8])],
            vec![t(&b, 0, DType::Bf16, &[m, 4])],
            OpAttrs::default(),
        )
    };
    let ks = be.kernels();
    let accepts = |name: &str, c: &OpCall| ks.iter().any(|k| k.name == name && k.accepts(&ShapeInfo::from_call(c)));
    assert!(accepts(MATMUL_BF16_GEMV, &mm(1)) && !accepts(MATMUL_BF16_GEMV, &mm(2)));
    assert!(accepts(MATMUL_BF16, &mm(1)) && accepts(MATMUL_BF16, &mm(64)));
    let op_of = |name: &str| ks.iter().find(|k| k.name == name).unwrap().op;
    assert_eq!((op_of(LOGITS_BF16), op_of(LOGITS_BF16_GEMV)), (Op::Logits, Op::Logits));
}

/// `RopeScaled` (host and device position, partial rotary, odd offsets) against an f64 evaluation of the same formula,
/// with the real llama3 and yarn tables and YaRN's attention factor; the unscaled head passes through.
#[test]
fn rope_scaled_matches_f64_reference() {
    use intarsia_core::rope::RopeScaling;
    let be = backend();
    let (tt, nh, d, rd) = (3usize, 4usize, 128usize, 96usize);
    let tables = [
        RopeScaling::Llama3 { factor: 8.0, low_freq_factor: 1.0, high_freq_factor: 4.0, original_max: 8192.0 }
            .freqs(500000.0, rd)
            .unwrap(),
        RopeScaling::Yarn {
            factor: 4.0,
            original_max: 32768.0,
            beta_fast: 32.0,
            beta_slow: 1.0,
            truncate: true,
            attention_factor: None,
            mscale: None,
            mscale_all_dim: None,
        }
        .freqs(1000000.0, rd)
        .unwrap(),
    ];
    for (ti, freqs) in tables.iter().enumerate() {
        for (xo, oo, pos0, dev) in [(0usize, 0usize, 0usize, false), (2, 2, 9000, false), (0, 0, 70000, true)] {
            let x = rand_bf16(tt * nh * d, 20 + ti as u64);
            let xb = upload_at(&be, xo, &x);
            let out = be.alloc(oo + x.len() * 2 + 16, MemKind::Device).unwrap();
            be.upload(&out, 0, &vec![0xEEu8; oo + x.len() * 2 + 16], 0).unwrap();
            let tb = be.alloc(freqs.inv_freq.len() * 4, MemKind::Device).unwrap();
            be.upload(&tb, 0, &freqs.inv_freq.iter().flat_map(|f| f.to_le_bytes()).collect::<Vec<u8>>(), 0).unwrap();
            let posb = be.alloc(4, MemKind::Device).unwrap();
            be.upload(&posb, 0, &(pos0 as i32).to_le_bytes(), 0).unwrap();
            let mut call = OpCall::new(
                Op::RopeScaled,
                vec![t(&xb, xo, DType::Bf16, &[tt, nh, d]), t(&tb, 0, DType::F32, &[rd / 2])],
                vec![t(&out, oo, DType::Bf16, &[tt, nh, d])],
                OpAttrs { pos0, head_dim: d, rotary_dim: rd, rope_mscale: freqs.attention_factor, ..Default::default() },
            );
            if dev {
                call.pos = Some(TensorRef { buf: posb.clone(), offset: 0, dtype: DType::I32, shape: vec![1] });
            }
            be.launch("rope_scaled_bf16", &call, 0).unwrap();
            let got = read_u16(&be, &out, oo, x.len());
            for tok in 0..tt {
                for h in 0..nh {
                    let base = (tok * nh + h) * d;
                    for i in 0..d {
                        let xf = |j: usize| bf16_to_f32(x[base + j]) as f64;
                        let want = if i >= rd {
                            xf(i)
                        } else {
                            let half = rd / 2;
                            let k = i % half;
                            let ang = (pos0 + tok) as f64 * freqs.inv_freq[k] as f64;
                            let (s, c) = (ang.sin() * freqs.attention_factor as f64, ang.cos() * freqs.attention_factor as f64);
                            if i < half { xf(k) * c - xf(k + half) * s } else { xf(k + half) * c + xf(k) * s }
                        };
                        let g = bf16_to_f32(got[base + i]) as f64;
                        // f32 angle (positions up to 70000) and bf16 output: a few percent of the unit-scale input
                        assert!((g - want).abs() <= 0.02 * want.abs() + 0.02, "table {ti} pos0 {pos0} [{tok},{h},{i}]: {g} vs {want}");
                    }
                }
            }
            let raw = read_u16(&be, &out, 0, (oo + x.len() * 2 + 16) / 2);
            assert!(raw[..oo / 2].iter().all(|v| *v == 0xEEEE), "guard before the output was overwritten");
        }
    }
}

/// `AddBias`, including in place and at odd element offsets: x + bias with one bf16 rounding.
#[test]
fn add_bias_matches_reference_in_place_and_at_odd_offsets() {
    let be = backend();
    let (rows, cols) = (5usize, 301usize);
    let x = rand_bf16(rows * cols, 31);
    let bias = rand_bf16(cols, 32);
    let want: Vec<u16> = x
        .iter()
        .enumerate()
        .map(|(i, v)| f32_to_bf16(bf16_to_f32(*v) + bf16_to_f32(bias[i % cols])))
        .collect();
    for (xo, bo, oo, in_place) in [(0usize, 0usize, 0usize, false), (2, 2, 0, false), (2, 0, 2, true), (0, 2, 0, true)] {
        let xb = upload_at(&be, xo, &x);
        let bb = upload_at(&be, bo, &bias);
        let (out, out_off) = if in_place { (xb.clone(), xo) } else { (be.alloc(oo + x.len() * 2 + 16, MemKind::Device).unwrap(), oo) };
        let call = OpCall::new(
            Op::AddBias,
            vec![t(&xb, xo, DType::Bf16, &[rows, cols]), t(&bb, bo, DType::Bf16, &[cols])],
            vec![t(&out, out_off, DType::Bf16, &[rows, cols])],
            OpAttrs::default(),
        );
        be.launch("add_bias_bf16", &call, 0).unwrap();
        assert_eq!(read_u16(&be, &out, out_off, x.len()), want, "offsets {xo} {bo} {oo} in_place {in_place}");
    }
}

/// `AddBias` on an f32 input (the f32 output of a projection): `bf16(x + bias)`, one rounding, bit-exact against the host.
#[test]
fn add_bias_f32_input_rounds_once() {
    let be = backend();
    let (rows, cols) = (5usize, 301usize);
    let xs: Vec<f32> = rand_bf16(rows * cols, 33).iter().map(|v| bf16_to_f32(*v) * 1.0009765).collect();
    let bias = rand_bf16(cols, 34);
    let want: Vec<u16> = xs.iter().enumerate().map(|(i, v)| f32_to_bf16(v + bf16_to_f32(bias[i % cols]))).collect();
    // the pair that rounding the projection first would get wrong must exist in the data
    let twice: Vec<u16> = xs.iter().enumerate().map(|(i, v)| f32_to_bf16(bf16_to_f32(f32_to_bf16(*v)) + bf16_to_f32(bias[i % cols]))).collect();
    assert_ne!(want, twice, "the data does not separate one rounding from two");
    for (bo, oo) in [(0usize, 0usize), (2, 2), (0, 2)] {
        let xb = be.alloc(xs.len() * 4, MemKind::Device).unwrap();
        be.upload(&xb, 0, &xs.iter().flat_map(|f| f.to_le_bytes()).collect::<Vec<u8>>(), 0).unwrap();
        let bb = upload_at(&be, bo, &bias);
        let out = be.alloc(oo + xs.len() * 2 + 16, MemKind::Device).unwrap();
        let call = OpCall::new(
            Op::AddBias,
            vec![t(&xb, 0, DType::F32, &[rows, cols]), t(&bb, bo, DType::Bf16, &[cols])],
            vec![t(&out, oo, DType::Bf16, &[rows, cols])],
            OpAttrs::default(),
        );
        be.launch("add_bias_f32_bf16", &call, 0).unwrap();
        assert_eq!(read_u16(&be, &out, oo, xs.len()), want, "offsets bias {bo} out {oo}");
    }
}
