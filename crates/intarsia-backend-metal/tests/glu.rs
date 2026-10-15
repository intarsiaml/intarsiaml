#![cfg(target_os = "macos")]
//! `MatmulGlu` (`matmul_glu_bf16_gemv`) and `RmsNormAddMatmulGlu` (`rmsnorm_add_matmul_glu_bf16_gemv`): bitwise the unfused Metal
//! sequence they replace (`rmsnorm_add_bf16` + `gemv_pair_bf16_rt_*` + `silu_glu_bf16`), in one dispatch instead of three
//! (two for the plain form). 16-byte aligned operands (the common case) and 8-byte aligned ones (weights of a safetensors shard
//! whose header length is 8 mod 16) take the vector loads and must match to the bit; operands at a 2-byte offset or a K that
//! is not a multiple of 8 take the 2-byte-load variant, which is within bf16 tolerance of the unfused sequence.

mod common;
use common::*;
use intarsia_backend_metal::bench::{bf16_to_f32, f32_to_bf16};
use intarsia_backend_metal::kernels::{
    MATMUL_BF16_GEMV, MATMUL_GLU_BF16_GEMV, MATMUL_PAIR_BF16_GEMV, RMSNORM_ADD_BF16, RMSNORM_ADD_MATMUL_GLU_BF16_GEMV,
    SILU_GLU_BF16,
};
use intarsia_backend_metal::MetalBackend;
use intarsia_core::backend::{Backend, MemKind, TensorRef};
use intarsia_core::{Op, OpAttrs, OpCall, ShapeInfo};

#[derive(Clone, Copy, Debug)]
struct Cfg {
    k: usize,
    n: usize,
    /// Byte offsets of the weights and of every other operand.
    w_off: usize,
    o_off: usize,
    norm: bool,
    w_offset: f32,
    /// Whether the result must match to the bit (vector-load variants) or only within tolerance.
    bitwise: bool,
}

const QWEN: Cfg = Cfg { k: 4096, n: 12288, w_off: 0, o_off: 0, norm: false, w_offset: 0.0, bitwise: true };

fn with_off(be: &MetalBackend, v: &[u16], off: usize, shape: &[usize]) -> TensorRef {
    let mut p = vec![0u16; off / 2];
    p.extend_from_slice(v);
    p.extend_from_slice(&[0u16; 16]);
    tref(&upload(be, &p), off, shape)
}

fn bytes(be: &MetalBackend, t: &TensorRef) -> Vec<u8> {
    let n = t.shape.iter().product::<usize>() * 2;
    let mut v = vec![0u8; n];
    be.download(&t.buf, t.offset, &mut v, 0).unwrap();
    v
}

fn norm_weights(n: usize, seed: u64) -> Vec<u16> {
    rand_bf16(n, seed).iter().map(|&b| f32_to_bf16(1.0 + 0.25 * bf16_to_f32(b))).collect()
}

struct Ops {
    x: TensorRef,
    res: TensorRef,
    nw: TensorRef,
    wg: TensorRef,
    wu: TensorRef,
    a: TensorRef,
    sum: TensorRef,
}

fn operands(be: &MetalBackend, c: &Cfg, seed: u64) -> Ops {
    let (k, n) = (c.k, c.n);
    // weights scaled like a real layer (1/sqrt(K)) so g and u stay in a normal range
    let w = |s: u64| -> Vec<u16> {
        rand_bf16(n * k, s).iter().map(|&b| f32_to_bf16(bf16_to_f32(b) * 2.0 / (k as f32).sqrt())).collect()
    };
    Ops {
        x: with_off(be, &rand_bf16(k, seed), c.o_off, &[1, k]),
        res: with_off(be, &rand_bf16(k, seed + 1).iter().map(|&b| f32_to_bf16(2.0 * bf16_to_f32(b))).collect::<Vec<_>>(), c.o_off, &[1, k]),
        nw: with_off(be, &norm_weights(k, seed + 2), c.o_off, &[k]),
        wg: with_off(be, &w(seed + 3), c.w_off, &[n, k]),
        wu: with_off(be, &w(seed + 4), c.w_off, &[n, k]),
        a: with_off(be, &vec![0xEEEEu16; n], c.o_off, &[1, n]),
        sum: with_off(be, &vec![0xDDDDu16; k], c.o_off, &[1, k]),
    }
}

fn check_glu(c: Cfg, seed: u64) {
    let be = backend();
    let (f, u) = (operands(&be, &c, seed), operands(&be, &c, seed));
    let at = OpAttrs { eps: 1e-6, w_offset: c.w_offset, ..Default::default() };

    let (op, name, inputs, outputs) = if c.norm {
        (
            Op::RmsNormAddMatmulGlu,
            RMSNORM_ADD_MATMUL_GLU_BF16_GEMV,
            vec![f.x.clone(), f.res.clone(), f.nw.clone(), f.wg.clone(), f.wu.clone()],
            vec![f.a.clone(), f.sum.clone()],
        )
    } else {
        (Op::MatmulGlu, MATMUL_GLU_BF16_GEMV, vec![f.x.clone(), f.wg.clone(), f.wu.clone()], vec![f.a.clone()])
    };
    let fused = OpCall::new(op, inputs, outputs, at.clone());
    fused.check().unwrap();
    let shape = ShapeInfo::from_call(&fused);
    assert!(be.kernels().iter().any(|kn| kn.op == op && kn.name == name && kn.accepts(&shape)), "registered and accepts {c:?}");
    let before = be.pending_dispatches();
    be.launch(name, &fused, 0).unwrap();
    assert_eq!(be.pending_dispatches() - before, 1, "one dispatch for {c:?}");

    // the unfused sequence exactly as `Model::ffn_block` emits it
    let tmp = |shape: &[usize]| with_off(&be, &vec![0xCCCCu16; shape.iter().product()], c.o_off, shape);
    let (g, uu) = (tmp(&[1, c.n]), tmp(&[1, c.n]));
    let normed = tmp(&[1, c.k]);
    let xin = if c.norm {
        let norm = OpCall::new(
            Op::RmsNormAdd,
            vec![u.x.clone(), u.res.clone(), u.nw.clone()],
            vec![normed.clone(), u.sum.clone()],
            at.clone(),
        );
        be.launch(RMSNORM_ADD_BF16, &norm, 0).unwrap();
        normed.clone()
    } else {
        u.x.clone()
    };
    let pair = OpCall::new(
        Op::MatmulPair,
        vec![xin, u.wg.clone(), u.wu.clone()],
        vec![g.clone(), uu.clone()],
        OpAttrs::default(),
    );
    be.launch(MATMUL_PAIR_BF16_GEMV, &pair, 0).unwrap();
    let silu = OpCall::new(Op::SiluGlu, vec![g, uu], vec![u.a.clone()], OpAttrs::default());
    be.launch(SILU_GLU_BF16, &silu, 0).unwrap();
    be.synchronize(0).unwrap();

    let mut pairs = vec![("a", &f.a, &u.a)];
    if c.norm {
        pairs.push(("sum", &f.sum, &u.sum));
    }
    for (what, x, y) in pairs {
        let (bx, by) = (bytes(&be, x), bytes(&be, y));
        if c.bitwise || what == "sum" {
            if bx != by {
                let at = bx.iter().zip(&by).position(|(p, q)| p != q).unwrap();
                panic!("{what} differs from the unfused sequence at byte {at} of {} for {c:?}", bx.len());
            }
        } else {
            let (mut worst, mut mean) = (0.0f64, 0.0f64);
            for (p, q) in bx.chunks_exact(2).zip(by.chunks_exact(2)) {
                let (p, q) = (bf16_to_f32(u16::from_le_bytes([p[0], p[1]])) as f64, bf16_to_f32(u16::from_le_bytes([q[0], q[1]])) as f64);
                mean += q.abs() / (bx.len() / 2) as f64;
                worst = worst.max((p - q).abs() - q.abs() * 2f64.powi(-6));
            }
            assert!(worst <= 0.05 * mean + 1e-6, "{what}: |gpu - unfused| exceeds 2^-6 relative + 5% of the mean magnitude ({worst} vs {mean}) for {c:?}");
        }
    }

    // and the batch-1 GEMV the pair is bitwise of, for the plain form: g from `matmul_bf16_gemv` alone is what the
    // pair already tests; here the guard is that no output row was left unwritten (0xEEEE)
    let a = bytes(&be, &f.a);
    assert!(a.chunks_exact(2).all(|e| e != [0xEE, 0xEE]), "every row of a written for {c:?}");
    let _ = MATMUL_BF16_GEMV;
}

#[test]
fn plain_qwen3_8b_is_bitwise_the_unfused_sequence() {
    check_glu(QWEN, 1);
    check_glu(Cfg { n: 64, ..QWEN }, 2);
    check_glu(Cfg { n: 33, ..QWEN }, 3);
}

#[test]
fn plain_k_tails_are_bitwise() {
    // K = 4104: one full block of the 16-byte kernel and a tail chunk; K = 520 is below one block
    for k in [4104, 520, 8, 1032] {
        check_glu(Cfg { k, n: 129, ..QWEN }, 10 + k as u64);
    }
}

#[test]
fn norm_qwen3_8b_is_bitwise_the_unfused_sequence() {
    check_glu(Cfg { norm: true, ..QWEN }, 20);
    check_glu(Cfg { norm: true, n: 64, w_offset: 1.0, ..QWEN }, 21);
    check_glu(Cfg { norm: true, n: 33, ..QWEN }, 22);
}

#[test]
fn norm_hidden_sizes_are_bitwise() {
    // H below one thread's 16 elements (200), a partial last element round (2056, 4000), the widest (4096)
    for k in [200, 2056, 4000, 4096, 8] {
        check_glu(Cfg { k, n: 70, norm: true, ..QWEN }, 30 + k as u64);
    }
}

#[test]
fn eight_byte_aligned_weights_take_the_8_byte_loads_bitwise() {
    // weights 8 bytes into a 16-byte aligned buffer (the other operands stay 16-byte aligned: the decode stream's activations)
    for norm in [false, true] {
        check_glu(Cfg { w_off: 8, norm, ..QWEN }, 40);
        check_glu(Cfg { w_off: 8, n: 77, k: 2056, norm, ..QWEN }, 41);
    }
}

#[test]
fn two_byte_offsets_and_odd_k_stay_within_tolerance() {
    for norm in [false, true] {
        check_glu(Cfg { w_off: 2, o_off: 2, n: 129, k: 4096, norm, bitwise: false, ..QWEN }, 50);
        check_glu(Cfg { n: 129, k: 70, norm, bitwise: false, ..QWEN }, 51);
        check_glu(Cfg { w_off: 2, o_off: 6, n: 64, k: 1000, norm, bitwise: false, ..QWEN }, 52);
    }
}

#[test]
fn prefill_rows_are_not_accepted() {
    let be = backend();
    let c = QWEN;
    let o = operands(&be, &Cfg { n: 64, ..c }, 60);
    let x5 = tref(&be.alloc(5 * c.k * 2, MemKind::Device).unwrap(), 0, &[5, c.k]);
    let a5 = tref(&be.alloc(5 * 64 * 2, MemKind::Device).unwrap(), 0, &[5, 64]);
    let call = OpCall::new(Op::MatmulGlu, vec![x5, o.wg.clone(), o.wu.clone()], vec![a5], OpAttrs::default());
    call.check().unwrap();
    let shape = ShapeInfo::from_call(&call);
    assert!(!be.kernels().iter().any(|kn| kn.name == MATMUL_GLU_BF16_GEMV && kn.accepts(&shape)), "M = 5 is the GEMM's");
}

// ---- RmsNormAddMatmulQkv ------------------------------------------------------------------------------------------------

#[derive(Clone, Copy, Debug)]
struct QCfg {
    k: usize,
    nq: usize,
    nk: usize,
    w_off: usize,
    o_off: usize,
    w_offset: f32,
    bitwise: bool,
}

const QQ: QCfg = QCfg { k: 4096, nq: 4096, nk: 1024, w_off: 0, o_off: 0, w_offset: 0.0, bitwise: true };

fn check_qkv(c: QCfg, seed: u64) {
    use intarsia_backend_metal::kernels::RMSNORM_ADD_MATMUL_QKV_BF16_GEMV as NAME;
    let be = backend();
    let mk = |be: &MetalBackend| {
        let w = |s: u64, n: usize| -> TensorRef {
            let v: Vec<u16> =
                rand_bf16(n * c.k, s).iter().map(|&b| f32_to_bf16(bf16_to_f32(b) * 2.0 / (c.k as f32).sqrt())).collect();
            with_off(be, &v, c.w_off, &[n, c.k])
        };
        let o = |n: usize| with_off(be, &vec![0xEEEEu16; n], c.o_off, &[1, n]);
        (
            with_off(be, &rand_bf16(c.k, seed), c.o_off, &[1, c.k]),
            with_off(
                be,
                &rand_bf16(c.k, seed + 1).iter().map(|&b| f32_to_bf16(2.0 * bf16_to_f32(b))).collect::<Vec<_>>(),
                c.o_off,
                &[1, c.k],
            ),
            with_off(be, &norm_weights(c.k, seed + 2), c.o_off, &[c.k]),
            [w(seed + 3, c.nq), w(seed + 4, c.nk), w(seed + 5, c.nk)],
            [o(c.nq), o(c.nk), o(c.nk), o(c.k)],
        )
    };
    let (f, u) = (mk(&be), mk(&be));
    let at = OpAttrs { eps: 1e-6, w_offset: c.w_offset, ..Default::default() };
    let fused = OpCall::new(
        Op::RmsNormAddMatmulQkv,
        vec![f.0.clone(), f.1.clone(), f.2.clone(), f.3[0].clone(), f.3[1].clone(), f.3[2].clone()],
        f.4.to_vec(),
        at.clone(),
    );
    fused.check().unwrap();
    let shape = ShapeInfo::from_call(&fused);
    assert!(be.kernels().iter().any(|kn| kn.name == NAME && kn.accepts(&shape)), "registered and accepts {c:?}");
    let before = be.pending_dispatches();
    be.launch(NAME, &fused, 0).unwrap();
    assert_eq!(be.pending_dispatches() - before, 1, "one dispatch for {c:?}");

    // unfused: rmsnorm_add, q GEMV, k + v pair
    let normed = with_off(&be, &vec![0xCCCCu16; c.k], c.o_off, &[1, c.k]);
    be.launch(
        RMSNORM_ADD_BF16,
        &OpCall::new(Op::RmsNormAdd, vec![u.0.clone(), u.1.clone(), u.2.clone()], vec![normed.clone(), u.4[3].clone()], at.clone()),
        0,
    )
    .unwrap();
    be.launch(MATMUL_BF16_GEMV, &matmul_call(normed.clone(), u.3[0].clone(), u.4[0].clone()), 0).unwrap();
    be.launch(
        MATMUL_PAIR_BF16_GEMV,
        &OpCall::new(
            Op::MatmulPair,
            vec![normed, u.3[1].clone(), u.3[2].clone()],
            vec![u.4[1].clone(), u.4[2].clone()],
            OpAttrs::default(),
        ),
        0,
    )
    .unwrap();
    be.synchronize(0).unwrap();
    for (j, name) in ["q", "k", "v", "sum"].iter().enumerate() {
        let (bx, by) = (bytes(&be, &f.4[j]), bytes(&be, &u.4[j]));
        if c.bitwise || *name == "sum" {
            if bx != by {
                let p = bx.iter().zip(&by).position(|(a, b)| a != b).unwrap();
                panic!("{name} differs from the unfused sequence at byte {p} of {} for {c:?}", bx.len());
            }
        } else {
            let (mut worst, mut mean) = (0.0f64, 0.0f64);
            for (p, q) in bx.chunks_exact(2).zip(by.chunks_exact(2)) {
                let (p, q) = (
                    bf16_to_f32(u16::from_le_bytes([p[0], p[1]])) as f64,
                    bf16_to_f32(u16::from_le_bytes([q[0], q[1]])) as f64,
                );
                mean += q.abs() / (bx.len() / 2) as f64;
                worst = worst.max((p - q).abs() - q.abs() * 2f64.powi(-6));
            }
            assert!(worst <= 0.05 * mean + 1e-6, "{name}: {worst} vs {mean} for {c:?}");
        }
        assert!(bx.chunks_exact(2).all(|e| e != [0xEE, 0xEE]), "{name} fully written for {c:?}");
    }
}

#[test]
fn qkv_qwen3_8b_is_bitwise_the_unfused_sequence() {
    check_qkv(QQ, 70);
    check_qkv(QCfg { w_offset: 1.0, nq: 64, nk: 33, ..QQ }, 71);
}

#[test]
fn qkv_hidden_sizes_and_8_byte_weights() {
    for k in [200, 2056, 4000] {
        check_qkv(QCfg { k, nq: 70, nk: 21, ..QQ }, 72 + k as u64);
    }
    check_qkv(QCfg { w_off: 8, ..QQ }, 80);
    check_qkv(QCfg { w_off: 8, k: 2056, nq: 77, nk: 19, ..QQ }, 81);
}

#[test]
fn qkv_two_byte_offsets_and_odd_k_stay_within_tolerance() {
    check_qkv(QCfg { w_off: 2, o_off: 2, nq: 129, nk: 33, bitwise: false, ..QQ }, 90);
    check_qkv(QCfg { k: 70, nq: 129, nk: 33, bitwise: false, ..QQ }, 91);
}
