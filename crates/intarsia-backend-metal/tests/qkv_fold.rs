#![cfg(target_os = "macos")]
//! `QkNormRopeKv` (`qk_norm_rope_kv_bf16`): bitwise the unfused Metal sequence it replaces (`qk_norm_bf16` + `rope_bf16` for
//! q and for k, `kv_append_bf16` for k and v), one dispatch instead of six, and it touches nothing but the rows it writes
//! (the whole caches are compared, untouched rows included). Host and device positions, decode and prefill rows, both
//! zero-centred and plain norms, full and partial rotary, 2-byte-offset operands.

mod common;
use common::*;
use intarsia_backend_metal::kernels::{KV_APPEND_BF16, QK_NORM_BF16, QK_NORM_ROPE_KV_BF16, ROPE_BF16};
use intarsia_backend_metal::MetalBackend;
use intarsia_core::backend::{Backend, MemKind, TensorRef};
use intarsia_core::{DType, Op, OpAttrs, OpCall, ShapeInfo};

#[derive(Clone, Copy, Debug)]
struct Cfg {
    t: usize,
    n: usize,
    nkv: usize,
    d: usize,
    rotary_dim: usize,
    pos0: usize,
    cap: usize,
    dev_pos: bool,
    /// Byte offset of every bf16 operand (0 or 2: the leftover-element path).
    off: usize,
    w_offset: f32,
    theta: f32,
}

const DEC: Cfg = Cfg {
    t: 1,
    n: 32,
    nkv: 8,
    d: 128,
    rotary_dim: 0,
    pos0: 133,
    cap: 8192,
    dev_pos: false,
    off: 0,
    w_offset: 0.0,
    theta: 1_000_000.0,
};

struct Operands {
    q: TensorRef,
    k: TensorRef,
    v: TensorRef,
    wq: TensorRef,
    wk: TensorRef,
    qo: TensorRef,
    kc: TensorRef,
    vc: TensorRef,
    pos: Option<TensorRef>,
}

fn with_off(be: &MetalBackend, v: &[u16], off: usize, shape: &[usize]) -> TensorRef {
    let mut p = vec![0u16; off / 2];
    p.extend_from_slice(v);
    p.extend_from_slice(&[0u16; 8]);
    tref(&upload(be, &p), off, shape)
}

fn norm_weights(n: usize, seed: u64) -> Vec<u16> {
    // 1 +- 0.25 like a real norm weight: bf16 of (1 + 0.5 * (u - 0.5)), u from the unit-range generator
    rand_bf16(n, seed).iter().map(|&b| intarsia_backend_metal::bench::f32_to_bf16(1.0 + 0.25 * intarsia_backend_metal::bench::bf16_to_f32(b))).collect()
}

fn operands(be: &MetalBackend, c: &Cfg, seed: u64) -> Operands {
    let cache_len = c.nkv * c.cap * c.d;
    let pos = c.dev_pos.then(|| {
        let b = be.alloc(8, MemKind::Device).unwrap();
        be.upload(&b, 0, &(c.pos0 as i32).to_le_bytes(), 0).unwrap();
        TensorRef { buf: b, offset: 0, dtype: DType::I32, shape: vec![1] }
    });
    Operands {
        q: with_off(be, &rand_bf16(c.t * c.n * c.d, seed), c.off, &[c.t, c.n, c.d]),
        k: with_off(be, &rand_bf16(c.t * c.nkv * c.d, seed + 1), c.off, &[c.t, c.nkv, c.d]),
        v: with_off(be, &rand_bf16(c.t * c.nkv * c.d, seed + 2), c.off, &[c.t, c.nkv, c.d]),
        wq: with_off(be, &norm_weights(c.d, seed + 3), c.off, &[c.d]),
        wk: with_off(be, &norm_weights(c.d, seed + 4), c.off, &[c.d]),
        qo: with_off(be, &vec![0xEEEEu16; c.t * c.n * c.d], c.off, &[c.t, c.n, c.d]),
        kc: with_off(be, &rand_bf16(cache_len, seed + 5), c.off, &[c.nkv, c.cap, c.d]),
        vc: with_off(be, &rand_bf16(cache_len, seed + 6), c.off, &[c.nkv, c.cap, c.d]),
        pos,
    }
}

fn bytes(be: &MetalBackend, t: &TensorRef) -> Vec<u8> {
    let n = t.shape.iter().product::<usize>() * 2;
    let mut v = vec![0u8; n];
    be.download(&t.buf, t.offset, &mut v, 0).unwrap();
    v
}

fn attrs(c: &Cfg) -> OpAttrs {
    OpAttrs {
        pos0: if c.dev_pos { 0 } else { c.pos0 },
        theta: c.theta,
        eps: 1e-6,
        w_offset: c.w_offset,
        head_dim: c.d,
        rotary_dim: c.rotary_dim,
        n_heads: c.n,
        n_kv_heads: c.nkv,
        kv_cap: c.cap,
        ..Default::default()
    }
}

/// Run the fused kernel on one operand set and the unfused kernels on a second, identical one; assert equal bytes
/// (q_out and both whole caches) and the dispatch counts.
fn check_bitwise(c: Cfg, seed: u64) {
    let be = backend();
    let (f, u) = (operands(&be, &c, seed), operands(&be, &c, seed));
    let at = attrs(&c);

    let mut fused = OpCall::new(
        Op::QkNormRopeKv,
        vec![f.q.clone(), f.k.clone(), f.v.clone(), f.wq.clone(), f.wk.clone()],
        vec![f.qo.clone(), f.kc.clone(), f.vc.clone()],
        at.clone(),
    );
    fused.pos = f.pos.clone();
    fused.check().unwrap();
    let ks = be.kernels();
    let shape = ShapeInfo::from_call(&fused);
    assert!(
        ks.iter().any(|k| k.op == Op::QkNormRopeKv && k.name == QK_NORM_ROPE_KV_BF16 && k.accepts(&shape)),
        "registered and accepts {c:?}"
    );
    be.synchronize(0).unwrap(); // a fresh buffer: the count below cannot straddle a commit (FIRST_COMMIT / MAX_DISPATCHES)
    let before = be.pending_dispatches();
    be.launch(QK_NORM_ROPE_KV_BF16, &fused, 0).unwrap();
    assert_eq!(be.pending_dispatches() - before, 1, "one dispatch");

    // the unfused sequence, exactly as `Model::attn_block` emits it (intermediates in their own tensors)
    let tmp = |shape: &[usize]| with_off(&be, &vec![0xDDDDu16; shape.iter().product()], c.off, shape);
    let (q3, k3) = ([c.t, c.n, c.d], [c.t, c.nkv, c.d]);
    let (qn, kn, kr) = (tmp(&q3), tmp(&k3), tmp(&k3));
    let norm = |x: &TensorRef, w: &TensorRef, out: &TensorRef| {
        OpCall::new(Op::QkNorm, vec![x.clone(), w.clone()], vec![out.clone()], at.clone())
    };
    let rope = |x: &TensorRef, out: &TensorRef| {
        let mut r = OpCall::new(Op::Rope, vec![x.clone()], vec![out.clone()], at.clone());
        r.pos = u.pos.clone();
        r
    };
    let append = |x: &TensorRef, cache: &TensorRef| {
        let mut r = OpCall::new(Op::KvAppend, vec![x.clone()], vec![cache.clone()], at.clone());
        r.pos = u.pos.clone();
        r
    };
    be.launch(QK_NORM_BF16, &norm(&u.q, &u.wq, &qn), 0).unwrap();
    be.launch(ROPE_BF16, &rope(&qn, &u.qo), 0).unwrap();
    be.launch(QK_NORM_BF16, &norm(&u.k, &u.wk, &kn), 0).unwrap();
    be.launch(ROPE_BF16, &rope(&kn, &kr), 0).unwrap();
    be.launch(KV_APPEND_BF16, &append(&kr, &u.kc), 0).unwrap();
    be.launch(KV_APPEND_BF16, &append(&u.v, &u.vc), 0).unwrap();
    be.synchronize(0).unwrap();

    for (name, a, b) in [("q_out", &f.qo, &u.qo), ("k cache", &f.kc, &u.kc), ("v cache", &f.vc, &u.vc)] {
        let (x, y) = (bytes(&be, a), bytes(&be, b));
        if x != y {
            let at = x.iter().zip(&y).position(|(p, q)| p != q).unwrap();
            panic!("{name} differs from the unfused sequence at byte {at} of {} for {c:?}", x.len());
        }
    }
}

#[test]
fn decode_qwen3_8b_is_bitwise_the_unfused_sequence() {
    for (pos0, dev) in [(0, false), (133, false), (133, true), (4000, true), (8100, false), (8191, true)] {
        check_bitwise(Cfg { pos0, dev_pos: dev, ..DEC }, 100 + pos0 as u64);
    }
}

#[test]
fn prefill_rows_are_bitwise() {
    for (t, pos0) in [(5, 37), (64, 0), (256, 0), (256, 256), (300, 1000)] {
        check_bitwise(Cfg { t, pos0, cap: 2048, ..DEC }, 200 + t as u64);
    }
}

#[test]
fn head_dims_and_partial_rotary() {
    // D 64 / 96 / 128 / 256 (96 and 64 are not 128-wide threadgroups), rotary a quarter of the head, zero-centred norms
    for (n, nkv, d, rd) in [(8, 2, 64, 0), (12, 4, 96, 0), (16, 2, 256, 0), (16, 2, 256, 64), (16, 2, 256, 0), (4, 2, 128, 32)] {
        for w_offset in [0.0, 1.0] {
            check_bitwise(Cfg { n, nkv, d, rotary_dim: rd, w_offset, t: 3, pos0: 11, cap: 64, ..DEC }, 300 + d as u64);
        }
    }
    check_bitwise(Cfg { n: 16, nkv: 2, d: 256, rotary_dim: 64, w_offset: 1.0, theta: 10_000_000.0, ..DEC }, 310);
}

#[test]
fn operands_at_two_byte_offsets() {
    // every bf16 operand 2 bytes into its buffer (a weight inside a safetensors shard): the skip path
    check_bitwise(Cfg { off: 2, ..DEC }, 400);
    check_bitwise(Cfg { off: 2, t: 7, pos0: 5, cap: 40, dev_pos: true, ..DEC }, 401);
}

#[test]
fn a_device_position_past_the_cache_is_dropped_not_written() {
    // pos0 + t >= cap: k / v rows are not stored (kv_append_bf16 drops them); q_out is still produced
    check_bitwise(Cfg { t: 4, pos0: 62, cap: 64, dev_pos: true, ..DEC }, 500);
    check_bitwise(Cfg { t: 1, pos0: 64, cap: 64, dev_pos: true, ..DEC }, 501);
}

#[test]
fn prefill_form_head_counts_off_the_threadgroup_group() {
    // `qk_norm_rope_kv_bf16_pf` runs 12 heads (simdgroups) per threadgroup: head totals N + 2 Nkv of 8, 18, 24, 15, 7, 3 leave
    // partial last groups (and a v / k / q boundary inside a group); full rotation, D 128, T > 1
    for (n, nkv) in [(4, 2), (14, 2), (16, 4), (9, 3), (5, 1), (1, 1)] {
        for w_offset in [0.0, 1.0] {
            check_bitwise(Cfg { n, nkv, t: 9, pos0: 30_000, cap: 40_000, w_offset, ..DEC }, 600 + (n * 10 + nkv) as u64);
        }
    }
    // a device-resident position and the cache end inside the token block
    check_bitwise(Cfg { n: 14, nkv: 2, t: 6, pos0: 61, cap: 64, dev_pos: true, ..DEC }, 650);
}
