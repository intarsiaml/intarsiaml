#![cfg(target_os = "macos")]
//! The DSpine draft's code-bridge kernels (`bridge_norm_bf16`, `bridge_code_bf16`, `bridge_gate_bf16`, `bridge_write_bf16`,
//! `Op::BridgeNorm` / `BridgeCode` / `BridgeGate` / `BridgeWrite`, docs/plans/2026-11-07-dspine-port.md section 5) against an
//! f64-accumulating restatement of the op definitions (ops.rs) with the same bf16 rounding points, at the 27B draft's widths
//! (H 5120, code 1024, rank 512) and a tiny one, for T = 1 2 8 9 15 16 block rows, operands at a 2-byte offset.
//!
//! Gates: outputs within one bf16 ulp of the restatement and at least 99.9 % of them bitwise (the kernel sums in f32 in another
//! order, so a rare value sits on the other side of a rounding boundary); `rms` relative 1e-6; `BridgeWrite` row 0 bitwise the
//! raw row; `BridgeGate` row 0 zero; nothing written past the outputs. The model-level parity against Tencent's MLX reference is
//! `crates/intarsia-models/tests/dspine_cpu.rs` (cpu backend, tiny fixture) and `dspine_metal_real.rs` (this backend, real shapes).

mod common;
use common::*;
use intarsia_backend_metal::bench::{bf16_to_f32, f32_to_bf16};
use intarsia_backend_metal::MetalBackend;
use intarsia_core::backend::{Backend, Buffer, MemKind, TensorRef};
use intarsia_core::{DType, Op, OpAttrs, OpCall};

const OFF: usize = 2;
const TINY: f32 = 1.175_494_35e-38;

fn upload_off(be: &MetalBackend, v: &[u16]) -> Buffer {
    let mut p = vec![0u16; OFF / 2];
    p.extend_from_slice(v);
    p.extend_from_slice(&[0u16; 8]);
    upload(be, &p)
}

fn upload_f32(be: &MetalBackend, v: &[f32]) -> Buffer {
    let b = be.alloc(v.len() * 4 + 16, MemKind::Device).unwrap();
    let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
    be.upload(&b, 0, &bytes, 0).unwrap();
    b
}

fn read_bf16(be: &MetalBackend, buf: &Buffer, off: usize, n: usize) -> Vec<u16> {
    let mut b = vec![0u8; n * 2];
    be.download(buf, off, &mut b, 0).unwrap();
    b.chunks(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect()
}

fn read_f32(be: &MetalBackend, buf: &Buffer, n: usize) -> Vec<f32> {
    let mut b = vec![0u8; n * 4];
    be.download(buf, 0, &mut b, 0).unwrap();
    b.chunks(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect()
}

fn bf(x: f32) -> f32 {
    bf16_to_f32(f32_to_bf16(x))
}
fn f(v: u16) -> f32 {
    bf16_to_f32(v)
}
fn ss(row: &[f32]) -> f32 {
    row.iter().map(|&v| v as f64 * v as f64).sum::<f64>() as f32
}

fn tref_f32(buf: &Buffer, shape: &[usize]) -> TensorRef {
    TensorRef { buf: buf.clone(), offset: 0, dtype: DType::F32, shape: shape.to_vec() }
}

/// Compare bf16 outputs against the restatement: within one ulp, almost all bitwise.
fn compare(got: &[u16], want: &[f32], what: &str) {
    assert_eq!(got.len(), want.len());
    let (mut exact, mut worst) = (0usize, 0.0f64);
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        let wb = f32_to_bf16(w);
        if g == wb {
            exact += 1;
            continue;
        }
        // one bf16 ulp of the larger magnitude (neighbouring bit patterns of the same sign differ by one)
        let near = (g as i32 - wb as i32).abs() <= 1 && (g ^ wb) & 0x8000 == 0;
        let ulp = (f(g).abs().max(w.abs()) as f64) * 2f64.powi(-7);
        worst = worst.max((f(g) as f64 - w as f64).abs() / ulp.max(1e-38));
        assert!(near || (f(g) as f64 - w as f64).abs() <= ulp, "{what}: element {i}: got {} want {w}", f(g));
    }
    let frac = exact as f64 / got.len() as f64;
    eprintln!("{what}: {}/{} bitwise ({:.4} %), worst off-by-one {:.3} ulp", exact, got.len(), 100.0 * frac, worst);
    assert!(frac >= 0.999, "{what}: only {:.4} % bitwise", 100.0 * frac);
}

fn run(be: &MetalBackend, call: &OpCall, kernel: &str) {
    be.launch(kernel, call, 0).unwrap();
    be.synchronize(0).unwrap();
}

const SHAPES: [(usize, usize, usize); 2] = [(5120, 1024, 512), (48, 20, 12)];
const ROWS: [usize; 6] = [1, 2, 8, 9, 15, 16];

#[test]
fn bridge_norm_and_code_match_the_reference() {
    let be = backend();
    for &(h, c, _) in &SHAPES {
        for &t in &ROWS {
            // BridgeNorm over H, BridgeCode over C
            let x = rand_bf16(t * h, (h * 13 + t) as u64);
            let (xb, yb) = (upload_off(&be, &x), upload_off(&be, &vec![0xEEEEu16; t * h]));
            let rms = upload_f32(&be, &vec![-1.0f32; t + 2]);
            let eps = 1e-6f32;
            let call = OpCall::new(
                Op::BridgeNorm,
                vec![tref(&xb, OFF, &[t, h])],
                vec![tref(&yb, OFF, &[t, h]), tref_f32(&rms, &[t])],
                OpAttrs { eps, ..Default::default() },
            );
            run(&be, &call, "bridge_norm_bf16");
            let got = read_bf16(&be, &yb, OFF, t * h);
            let mut want = vec![0f32; t * h];
            let mut want_rms = vec![0f32; t];
            for r in 0..t {
                let row: Vec<f32> = x[r * h..(r + 1) * h].iter().map(|&v| f(v)).collect();
                let rr = ((ss(&row) as f64 / h as f64) as f32).max(TINY).sqrt();
                want_rms[r] = rr;
                for k in 0..h {
                    want[r * h + k] = row[k] / (rr + eps);
                }
            }
            compare(&got, &want, &format!("bridge_norm T={t} H={h}"));
            let grms = read_f32(&be, &rms, t + 2);
            for r in 0..t {
                assert!(((grms[r] - want_rms[r]) / want_rms[r]).abs() < 1e-6, "rms row {r}: {} vs {}", grms[r], want_rms[r]);
            }
            assert_eq!(grms[t], -1.0, "bridge_norm wrote past its rms output");
            assert_eq!(read_bf16(&be, &yb, OFF + t * h * 2, 1), [0u16], "bridge_norm wrote past its output");

            let x = rand_bf16(t * c, (c * 17 + t) as u64);
            let (xb, yb) = (upload_off(&be, &x), upload_off(&be, &vec![0xEEEEu16; t * c]));
            let scale = (c as f32).sqrt();
            let call = OpCall::new(Op::BridgeCode, vec![tref(&xb, OFF, &[t, c])], vec![tref(&yb, OFF, &[t, c])], OpAttrs { eps, scale, ..Default::default() });
            run(&be, &call, "bridge_code_bf16");
            let got = read_bf16(&be, &yb, OFF, t * c);
            let mut want = vec![0f32; t * c];
            for r in 0..t {
                let row: Vec<f32> = x[r * c..(r + 1) * c].iter().map(|&v| f(v)).collect();
                let n = ss(&row).sqrt();
                for k in 0..c {
                    want[r * c + k] = (scale * row[k]) / (n + eps);
                }
            }
            compare(&got, &want, &format!("bridge_code T={t} C={c}"));
            assert_eq!(read_bf16(&be, &yb, OFF + t * c * 2, 1), [0u16], "bridge_code wrote past its output");
        }
    }
}

#[test]
fn bridge_gate_matches_the_reference() {
    let be = backend();
    for &(_, _, r) in &SHAPES {
        for &t in &ROWS {
            let packed = rand_bf16(t * 3 * r, (r * 3 + t) as u64);
            let bias = rand_bf16(r, (r * 5) as u64);
            let s0 = rand_bf16(2 * r, (r * 7 + t) as u64);
            let (pb, bb, sb) = (upload_off(&be, &packed), upload_off(&be, &bias), upload_off(&be, &s0));
            let mb = upload_off(&be, &vec![0xEEEEu16; t * r]);
            let call = OpCall::new(
                Op::BridgeGate,
                vec![tref(&pb, OFF, &[t, 3 * r]), tref(&bb, OFF, &[r]), tref(&sb, OFF, &[2 * r])],
                vec![tref(&mb, OFF, &[t, r])],
                OpAttrs::default(),
            );
            run(&be, &call, "bridge_gate_bf16");
            let got = read_bf16(&be, &mb, OFF, t * r);
            let mut want = vec![0f32; t * r];
            for row in 1..t {
                for k in 0..r {
                    let q = f(packed[row * 3 * r + k]);
                    let (s, v) = if row == 1 { (f(s0[k]), f(s0[r + k])) } else { (f(packed[(row - 1) * 3 * r + r + k]), f(packed[(row - 1) * 3 * r + 2 * r + k])) };
                    let z = ((q + s) + f(bias[k])) as f64;
                    want[row * r + k] = ((1.0 / (1.0 + (-z).exp())) as f32) * v;
                }
            }
            assert!(got[..r].iter().all(|&v| v == 0), "bridge_gate row 0 must be zero");
            compare(&got, &want, &format!("bridge_gate T={t} R={r}"));
            assert_eq!(read_bf16(&be, &mb, OFF + t * r * 2, 1), [0u16], "bridge_gate wrote past its output");
        }
    }
}

#[test]
fn bridge_write_matches_the_reference() {
    let be = backend();
    for &(h, _, _) in &SHAPES {
        for &t in &ROWS {
            let raw = rand_bf16(t * h, (h * 19 + t) as u64);
            let delta = rand_bf16(t * h, (h * 23 + t) as u64);
            let w: Vec<u16> = rand_bf16(h, (h * 29) as u64).iter().map(|&v| f32_to_bf16(1.0 + 0.25 * f(v))).collect();
            let rms: Vec<f32> = rand_bf16(t, (t * 31) as u64).iter().map(|&v| 0.2 + f(v).abs()).collect();
            let eps = 1e-6f32;
            let (rb, db, wb) = (upload_off(&be, &raw), upload_off(&be, &delta), upload_off(&be, &w));
            let rmsb = upload_f32(&be, &rms);
            let (ob, nb) = (upload_off(&be, &vec![0xEEEEu16; t * h]), upload_off(&be, &vec![0xEEEEu16; t * h]));
            let call = OpCall::new(
                Op::BridgeWrite,
                vec![tref(&rb, OFF, &[t, h]), tref_f32(&rmsb, &[t]), tref(&db, OFF, &[t, h]), tref(&wb, OFF, &[h])],
                vec![tref(&ob, OFF, &[t, h]), tref(&nb, OFF, &[t, h])],
                OpAttrs { eps, ..Default::default() },
            );
            run(&be, &call, "bridge_write_bf16");
            let (got_o, got_n) = (read_bf16(&be, &ob, OFF, t * h), read_bf16(&be, &nb, OFF, t * h));
            let (mut want_o, mut want_n) = (vec![0f32; t * h], vec![0f32; t * h]);
            for r in 0..t {
                let na = if r == 0 { 0.0f32 } else { 1.0 };
                for k in 0..h {
                    want_o[r * h + k] = bf(f(raw[r * h + k]) + (rms[r] * f(delta[r * h + k])) * na);
                }
                let inv = 1.0 / (((ss(&want_o[r * h..(r + 1) * h]) as f64 / h as f64) as f32) + eps).sqrt();
                for k in 0..h {
                    want_n[r * h + k] = bf(bf(want_o[r * h + k] * inv) * f(w[k]));
                }
            }
            assert_eq!(got_o[..h], raw[..h], "bridge_write row 0 must be the raw row, bitwise");
            compare(&got_o, &want_o, &format!("bridge_write out T={t} H={h}"));
            // the normed rows are computed from the kernel's own `out` (a one-ulp flip of `out` moves them): restate from got_o
            let from_got: Vec<f32> = (0..t)
                .flat_map(|r| {
                    let row: Vec<f32> = got_o[r * h..(r + 1) * h].iter().map(|&v| f(v)).collect();
                    let inv = 1.0 / (((ss(&row) as f64 / h as f64) as f32) + eps).sqrt();
                    (0..h).map(|k| bf(bf(row[k] * inv) * f(w[k]))).collect::<Vec<_>>()
                })
                .collect();
            compare(&got_n, &from_got, &format!("bridge_write normed T={t} H={h}"));
            let _ = want_n;
            assert_eq!(read_bf16(&be, &ob, OFF + t * h * 2, 1), [0u16], "bridge_write wrote past out");
            assert_eq!(read_bf16(&be, &nb, OFF + t * h * 2, 1), [0u16], "bridge_write wrote past normed");
        }
    }
}
