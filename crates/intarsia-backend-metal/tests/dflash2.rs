#![cfg(target_os = "macos")]
//! Lane dm (DFlash2 on Metal, docs/results/2026-11-06-metal-dflash2.md): the draft's two Metal kernels.
//!
//! * `dyn_conv_bf16` (`Op::DynConv`) against an f64 reference of the op's definition (ops.rs) at the 27B draft's shape (H 5120,
//!   K 2 taps, G 320 groups) for T 1 / 3 / 8 / 16 block rows, and at an odd geometry (H 48, K 3, G 3); operands at a 2-byte offset.
//! * `attn_decode_bf16_window` (`AttnDecode` with `OpAttrs::window > 0`): bitwise the plain decode kernel run over a cache that
//!   holds only the window's keys (`max(0, p + 1 - w) ..= p` moved to rows `0 ..`), at the draft's heads (32 / 8, head_dim 128)
//!   on the split-KV path and at a shape that takes the single-pass path; and a window that covers the whole context is
//!   bitwise the plain call.

mod common;
use common::*;
use intarsia_backend_metal::bench::{bf16_to_f32, f32_to_bf16};
use intarsia_backend_metal::kernels::{ATTN_DECODE_BF16, ATTN_DECODE_BF16_SPLITKV};
use intarsia_backend_metal::MetalBackend;
use intarsia_core::backend::{Backend, Buffer, MemKind, TensorRef};
use intarsia_core::{DType, Op, OpAttrs, OpCall};

const DYN_CONV: &str = "dyn_conv_bf16";
const WINDOW: &str = "attn_decode_bf16_window";
/// Leading bf16 elements before every conv operand (byte offset 2: the shader gets a skip of 1).
const OFF: usize = 2;

fn upload_off(be: &MetalBackend, v: &[u16]) -> Buffer {
    let mut p = vec![0u16; OFF / 2];
    p.extend_from_slice(v);
    p.extend_from_slice(&[0u16; 8]);
    upload(be, &p)
}

fn read_bf16(be: &MetalBackend, buf: &Buffer, off: usize, n: usize) -> Vec<u16> {
    let mut b = vec![0u8; n * 2];
    be.download(buf, off, &mut b, 0).unwrap();
    b.chunks(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect()
}

fn bf(v: u16) -> f64 {
    bf16_to_f32(v) as f64
}

/// f64 `DynConv` and the L1 mass of each output (for the rounding budget).
fn conv_ref(x: &[u16], k: &[u16], base: &[u16], t: usize, h: usize, taps: usize, g: usize) -> Vec<(f64, f64)> {
    let mut out = vec![(0.0, 0.0); t * h];
    for r in 0..t {
        for c in 0..h {
            let grp = c / (h / g);
            let (mut acc, mut mass) = (0.0, 0.0);
            for o in 0..taps.min(r + 1) {
                let term = (bf(base[o * h + c]) + bf(k[r * taps * g + o * g + grp])) * bf(x[(r - o) * h + c]);
                acc += term;
                mass += term.abs();
            }
            out[r * h + c] = (acc, mass);
        }
    }
    out
}

#[test]
fn dyn_conv_matches_the_reference() {
    let be = backend();
    for &(h, taps, g) in &[(5120usize, 2usize, 320usize), (48, 3, 3)] {
        for &t in &[1usize, 3, 8, 16] {
            let seed = (h * 31 + t * 7 + taps) as u64;
            let x = rand_bf16(t * h, seed);
            let k = rand_bf16(t * taps * g, seed + 1);
            let base = rand_bf16(taps * h, seed + 2);
            let (xb, kb, bb) = (upload_off(&be, &x), upload_off(&be, &k), upload_off(&be, &base));
            let ob = upload_off(&be, &vec![0xEEEEu16; t * h]);
            let call = OpCall::new(
                Op::DynConv,
                vec![tref(&xb, OFF, &[t, h]), tref(&kb, OFF, &[t, taps * g]), tref(&bb, OFF, &[taps, h])],
                vec![tref(&ob, OFF, &[t, h])],
                OpAttrs::default(),
            );
            be.launch(DYN_CONV, &call, 0).unwrap();
            be.synchronize(0).unwrap();
            let got = read_bf16(&be, &ob, OFF, t * h);
            let want = conv_ref(&x, &k, &base, t, h, taps, g);
            let mut bad = 0;
            for (i, (&gv, &(w, m))) in got.iter().zip(&want).enumerate() {
                // one bf16 rounding of the f32 sum (2^-8 relative) plus f32 accumulation over <= K terms
                let tol = w.abs() * (1.0 / 256.0) + m * 1e-6 + 1e-30;
                if (bf(gv) - w).abs() > tol {
                    if bad < 5 {
                        eprintln!("T={t} H={h} i={i}: got {} want {w}", bf(gv));
                    }
                    bad += 1;
                }
            }
            assert_eq!(bad, 0, "dyn_conv T={t} H={h} K={taps} G={g}: {bad} outputs off");
            // nothing written past the output
            let tail = read_bf16(&be, &ob, OFF + t * h * 2, 1);
            assert_eq!(tail, [0u16], "dyn_conv wrote past its output");
        }
    }
}

struct Attn {
    n: usize,
    nkv: usize,
    d: usize,
    cap: usize,
}

fn scratch(be: &MetalBackend, a: &Attn) -> TensorRef {
    let bytes = a.n * 64 * (a.d + 2) * 4;
    let b = be.alloc(bytes, MemKind::Device).unwrap();
    TensorRef { buf: b, offset: 0, dtype: DType::U8, shape: vec![bytes] }
}

/// One decode call of query `q` at position `pos` (window `win`) over caches `kc` / `vc` of `cap` rows per kv head.
#[allow(clippy::too_many_arguments)]
fn decode(be: &MetalBackend, kernel: &str, a: &Attn, q: &Buffer, kc: &Buffer, vc: &Buffer, cap: usize, pos: usize, win: usize) -> Vec<u16> {
    let out = upload(be, &vec![0xEEEEu16; a.n * a.d]);
    let mut call = OpCall::new(
        Op::AttnDecode,
        vec![tref(q, 0, &[1, a.n, a.d]), tref(kc, 0, &[a.nkv, cap, a.d]), tref(vc, 0, &[a.nkv, cap, a.d])],
        vec![tref(&out, 0, &[1, a.n, a.d])],
        OpAttrs {
            pos0: pos,
            scale: 1.0 / (a.d as f32).sqrt(),
            n_heads: a.n,
            n_kv_heads: a.nkv,
            head_dim: a.d,
            kv_cap: cap,
            window: win,
            ..Default::default()
        },
    );
    call.scratch = Some(scratch(be, a));
    be.launch(kernel, &call, 0).unwrap();
    be.synchronize(0).unwrap();
    read_bf16(be, &out, 0, a.n * a.d)
}

/// Rows `s ..= p` of every kv head of a `[nkv, cap, d]` cache, as a `[nkv, cap, d]` cache with them at rows `0 ..`.
fn window_copy(c: &[u16], a: &Attn, s: usize, p: usize) -> Vec<u16> {
    let mut o = vec![0u16; c.len()];
    for h in 0..a.nkv {
        let src = &c[(h * a.cap + s) * a.d..(h * a.cap + p + 1) * a.d];
        o[h * a.cap * a.d..h * a.cap * a.d + src.len()].copy_from_slice(src);
    }
    o
}

#[test]
fn window_decode_is_bitwise_the_plain_decode_over_the_window() {
    let be = backend();
    // the 27B draft's heads (split-KV path) and a group of 6 at head_dim 72 (single-pass path: not a power-of-two head_dim)
    for (a, plain) in [
        (Attn { n: 32, nkv: 8, d: 128, cap: 4200 }, ATTN_DECODE_BF16_SPLITKV),
        (Attn { n: 12, nkv: 2, d: 72, cap: 700 }, ATTN_DECODE_BF16),
    ] {
        let kv_n = a.nkv * a.cap * a.d;
        // scale the keys so the softmax is not flat (a wrong window start must move the output)
        let k: Vec<u16> = rand_bf16(kv_n, 11).iter().map(|&v| f32_to_bf16(bf16_to_f32(v) * 4.0)).collect();
        let v = rand_bf16(kv_n, 12);
        let q = upload(&be, &rand_bf16(a.n * a.d, 13));
        let (kb, vb) = (upload(&be, &k), upload(&be, &v));
        let w0 = if a.cap > 4000 { 2048 } else { 300 };
        let cases: Vec<(usize, usize)> = [(w0 / 2, w0), (w0 - 1, w0), (w0, w0), (w0 + 7, w0 + 7), (w0 + 350, w0 + 3), (a.cap - 1, w0)]
            .into_iter()
            .filter(|&(p, _)| p < a.cap)
            .collect();
        for (p, w) in cases {
            let s = (p + 1).saturating_sub(w);
            let got = decode(&be, WINDOW, &a, &q, &kb, &vb, a.cap, p, w);
            let (kw, vw) = (upload(&be, &window_copy(&k, &a, s, p)), upload(&be, &window_copy(&v, &a, s, p)));
            let want = decode(&be, plain, &a, &q, &kw, &vw, a.cap, p - s, 0);
            assert_eq!(got, want, "n={} d={} pos={p} window={w}: not bitwise the plain decode over keys {s}..={p}", a.n, a.d);
            if s == 0 {
                let full = decode(&be, plain, &a, &q, &kb, &vb, a.cap, p, 0);
                assert_eq!(got, full, "n={} pos={p} window={w} covers the context: not bitwise the plain call", a.n);
            } else {
                let full = decode(&be, plain, &a, &q, &kb, &vb, a.cap, p, 0);
                assert_ne!(got, full, "n={} pos={p} window={w}: the window did not change the output (test too weak)", a.n);
            }
        }
    }
}

#[test]
fn window_decode_is_never_picked_by_plain_calls() {
    use intarsia_core::ShapeInfo;
    let ks = intarsia_backend_metal::kernels::all();
    let win = ks.iter().find(|k| k.name == WINDOW).expect("registered");
    assert!(win.ext && win.op == Op::AttnDecode);
    assert!(!win.rows_exact && !win.row_table);
    for k in ks.iter().filter(|k| k.op == Op::AttnDecode && k.name != WINDOW) {
        assert!(!k.ext, "{} must not take windowed calls", k.name);
        // verify-row and row-table kernels are selected only by their own calls; every other plain decode kernel outranks it
        if !k.rows_exact && !k.row_table {
            assert!(win.priority < k.priority, "a plain call must rank {} above the window kernel", k.name);
        }
    }
    let bf = Some(DType::Bf16);
    let s = ShapeInfo { m: 1, n_heads: 32, n_kv_heads: 8, head_dim: 128, k: 128, act: bf, out: bf, weight: bf, ..Default::default() };
    assert!((win.when)(&s));
}

