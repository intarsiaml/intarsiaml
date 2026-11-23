//! `ink_in_proj_f8_nax` (lane ink-fp8, brief 22; kernels/metal/gemm_nax_fp8.metal `ink_in_proj_f8_nax_t`): the e4m3
//! `InkInProj` prefill kernel is BITWISE the model's pack path, three `matmul_fp8_nax` launches with scales `(ws_j, 0)` on the
//! row views of the same e4m3 stack, each reading input plane `min(j, S-1)`. Real Inkling-2-Mini shapes (D 2304, stacks
//! [6552, 1056, 1056] and [6552, 1024, 1024]), S = 1 and 3, T = 9 (the selection boundary), 17, 100, 200, 256, 2048, the default
//! tile and every tile forced; x at a 2-byte (leftover element) offset too. Also: the registry picks it exactly where the
//! packs take `matmul_fp8_nax` (M > 8), and `matmul_fp8_nax` W8A16 with x at a leftover-element offset is bitwise the same
//! call on an aligned copy of x (the GEMM reads x in place since brief 22).

#![cfg(target_os = "macos")]

mod common;

use common::*;
use intarsia_backend_metal::bench::Rng;
use intarsia_backend_metal::gemm_nax::NAX_VARIANTS;
use intarsia_backend_metal::MetalBackend;
use intarsia_core::backend::{Backend, Buffer, MemKind, TensorRef};
use intarsia_core::{DType, Op, OpAttrs, OpCall, ShapeInfo};

const FUSED: &str = "ink_in_proj_f8_nax";
const PACK: &str = "matmul_fp8_nax";

fn buf(be: &MetalBackend, len: usize) -> Buffer {
    let b = be.alloc(len + 32, MemKind::Device).unwrap();
    be.upload(&b, 0, &vec![0xEEu8; len + 32], 0).unwrap();
    b
}

fn up(be: &MetalBackend, off: usize, data: &[u8]) -> Buffer {
    let b = buf(be, off + data.len());
    be.upload(&b, off, data, 0).unwrap();
    b
}

fn t(b: &Buffer, off: usize, dt: DType, shape: &[usize]) -> TensorRef {
    TensorRef { buf: b.clone(), offset: off, dtype: dt, shape: shape.to_vec() }
}

/// e4m3 bytes without NaN codes, weight-like magnitudes (all exponents)
fn e4m3(n: usize, seed: u64) -> Vec<u8> {
    let mut r = Rng(seed | 1);
    (0..n)
        .map(|_| {
            let b = (r.next_u64() >> 32) as u8;
            if b & 0x7F == 0x7F {
                b & 0xFE
            } else {
                b
            }
        })
        .collect()
}

fn with_scratch(be: &MetalBackend, kernel: &str, mut cl: OpCall) -> OpCall {
    let need = be.kernels().iter().find(|q| q.name == kernel).unwrap().scratch_bytes(&ShapeInfo::from_call(&cl));
    if need > 0 {
        let b = buf(be, need);
        cl.scratch = Some(TensorRef { buf: b, offset: 0, dtype: DType::U8, shape: vec![need] });
    }
    cl
}

fn run(be: &MetalBackend, kernel: &str, cl: &OpCall, tile: Option<usize>) {
    let go = || be.launch(kernel, cl, 0).unwrap_or_else(|e| panic!("{kernel}: {e}"));
    match tile {
        Some(i) => intarsia_backend_metal::fp8::with_nax_tile(i, go),
        None => go(),
    }
    be.synchronize(0).unwrap();
}

fn down(be: &MetalBackend, y: &TensorRef) -> Vec<u8> {
    let mut raw = vec![0u8; y.shape.iter().product::<usize>() * 2];
    be.download(&y.buf, y.offset, &mut raw, 0).unwrap();
    raw
}

/// One stack: e4m3 W `[n0+n1+n2, D]` at an 8-byte offset, scales [3], x `[S, T, D]` at `x_off` bytes.
struct Stack {
    d: usize,
    ns: [usize; 3],
    w: Buffer,
    sc: [f32; 3],
    s3: Buffer,
    packs: [Buffer; 3],
}

impl Stack {
    fn new(be: &MetalBackend, d: usize, ns: [usize; 3], seed: u64) -> Self {
        let n: usize = ns.iter().sum();
        let w = up(be, 8, &e4m3(n * d, seed));
        // per-pack scales of weight-like size, distinct, not powers of two (the epilogue multiply must match bit for bit)
        let sc = [0.00123f32, 0.000871, 0.00311];
        let fb = |v: &[f32]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
        let s3 = up(be, 4, &fb(&sc));
        let packs = [0, 1, 2].map(|j| up(be, 4, &fb(&[sc[j], 0.0])));
        Stack { d, ns, w, sc, s3, packs }
    }

    fn x(&self, be: &MetalBackend, s: usize, t_: usize, x_off: usize, seed: u64) -> TensorRef {
        let x = rand_bf16(s * t_ * self.d, seed);
        t(&up(be, x_off, as_bytes(&x)), x_off, DType::Bf16, &[s, t_, self.d])
    }

    fn outs(&self, be: &MetalBackend, t_: usize) -> Vec<TensorRef> {
        // outputs at a 2-byte offset (a leftover element on the store side too)
        self.ns.iter().map(|&n| t(&buf(be, 2 + t_ * n * 2), 2, DType::Bf16, &[t_, n])).collect()
    }

    fn fused(&self, be: &MetalBackend, x: &TensorRef, tile: Option<usize>) -> Vec<Vec<u8>> {
        let t_ = x.shape[1];
        let n: usize = self.ns.iter().sum();
        let ys = self.outs(be, t_);
        let cl = OpCall::new(
            Op::InkInProj,
            vec![x.clone(), t(&self.w, 8, DType::Fp8E4m3, &[n, self.d]), t(&self.s3, 4, DType::F32, &[3])],
            ys.clone(),
            OpAttrs::default(),
        );
        cl.check().unwrap();
        let cl = with_scratch(be, FUSED, cl);
        run(be, FUSED, &cl, tile);
        ys.iter().map(|y| down(be, y)).collect()
    }

    /// the model's pack path: pack j reads input plane min(j, S-1)
    fn packs(&self, be: &MetalBackend, x: &TensorRef, tile: Option<usize>) -> Vec<Vec<u8>> {
        let (s, t_, d) = (x.shape[0], x.shape[1], x.shape[2]);
        let ys = self.outs(be, t_);
        let mut r0 = 0;
        for j in 0..3 {
            let plane = j.min(s - 1);
            let xj = t(&x.buf, x.offset + plane * t_ * d * 2, DType::Bf16, &[t_, d]);
            let wj = t(&self.w, 8 + r0 * d, DType::Fp8E4m3, &[self.ns[j], d]);
            let cl = OpCall::new(Op::MatmulFp8, vec![xj, wj, t(&self.packs[j], 4, DType::F32, &[2])], vec![ys[j].clone()], OpAttrs::default());
            cl.check().unwrap();
            let cl = with_scratch(be, PACK, cl);
            run(be, PACK, &cl, tile);
            r0 += self.ns[j];
        }
        ys.iter().map(|y| down(be, y)).collect()
    }
}

fn same(a: &[Vec<u8>], b: &[Vec<u8>], what: &str) {
    for j in 0..3 {
        let (x, y) = (&a[j], &b[j]);
        assert_eq!(x.len(), y.len());
        let bad = x.chunks_exact(2).zip(y.chunks_exact(2)).filter(|(p, q)| p != q).count();
        assert_eq!(bad, 0, "{what}: segment {j}: {bad} of {} bf16 outputs differ from the pack path", x.len() / 2);
    }
}

const STACKS: [[usize; 3]; 2] = [[6552, 1056, 1056], [6552, 1024, 1024]];

/// Gate 1 of brief 22: bitwise the packs at the real shapes, T = 9 / 17 / 100 / 200 / 256 / 2048, S = 1 and 3, default tile.
#[test]
fn f8_nax_is_bitwise_the_pack_path() {
    let be = backend();
    for (si, ns) in STACKS.iter().enumerate() {
        let st = Stack::new(&be, 2304, *ns, 11 + si as u64);
        // 100 / 200: full and partial M tiles in one call (tail M x tail N on every tile height)
        for t_ in [9usize, 17, 100, 200, 256, 2048] {
            for s in [1usize, 3] {
                for x_off in [0usize, 2] {
                    if x_off == 2 && t_ == 2048 {
                        continue;
                    }
                    let x = st.x(&be, s, t_, x_off, 1000 + t_ as u64 + s as u64);
                    let what = format!("stack {ns:?} T={t_} S={s} x_off={x_off} scales {:?}", st.sc);
                    same(&st.fused(&be, &x, None), &st.packs(&be, &x, None), &what);
                }
            }
        }
    }
}

/// Every NAX tile, forced on both sides and on the fused side alone against the packs' own pick (the fused grid picks its
/// tile from the stacked N, each pack from its own N).
#[test]
fn f8_nax_tiles_give_the_pack_bits() {
    let be = backend();
    let st = Stack::new(&be, 2304, STACKS[0], 21);
    for t_ in [17usize, 256] {
        let x = st.x(&be, 3, t_, 0, 77 + t_ as u64);
        let want = st.packs(&be, &x, None);
        for ti in 0..NAX_VARIANTS.len() {
            let what = format!("T={t_} tile {}", NAX_VARIANTS[ti].shader);
            same(&st.fused(&be, &x, Some(ti)), &want, &what);
            same(&st.packs(&be, &x, Some(ti)), &want, &format!("{what} (packs forced)"));
        }
    }
}

/// Selection: the fused kernel is admitted for an ordinary e4m3 call exactly at M > 8 (where the packs take
/// `matmul_fp8_nax`), and never for a row-exact (verify) call.
#[test]
fn f8_nax_selection_matches_the_pack_rule() {
    let be = backend();
    let k = be.kernels().into_iter().find(|q| q.name == FUSED).expect("registered (metal4.0 library loaded)");
    let pk = be.kernels().into_iter().find(|q| q.name == PACK).expect("matmul_fp8_nax");
    assert!(!k.rows_exact);
    for m in [1usize, 2, 8, 9, 16, 17, 2048] {
        let s = ShapeInfo { m, n: 8664, k: 2304, ..Default::default() };
        let p = ShapeInfo { m, n: 6552, k: 2304, ..Default::default() };
        assert_eq!(k.accepts(&s), pk.accepts(&p), "m = {m}");
    }
}

/// `matmul_fp8_nax` W8A16 reads x in place (brief 22): x at a leftover-element offset gives the bits of the same call on an
/// aligned copy, and W8A8 (q(x) through the scratch) is unchanged across the two placements too.
#[test]
fn fp8_nax_reads_x_in_place_at_any_offset() {
    let be = backend();
    let (m, n, k) = (37usize, 1056usize, 2304usize);
    let x = rand_bf16(m * k, 5);
    let w = up(&be, 8, &e4m3(n * k, 6));
    for is in [0.0f32, 0.37] {
        let sb = up(&be, 4, &[0.0123f32, is].iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>());
        let mut got = Vec::new();
        for x_off in [0usize, 2] {
            let xt = t(&up(&be, x_off, as_bytes(&x)), x_off, DType::Bf16, &[m, k]);
            let y = t(&buf(&be, m * n * 2), 0, DType::Bf16, &[m, n]);
            let cl = OpCall::new(Op::MatmulFp8, vec![xt, t(&w, 8, DType::Fp8E4m3, &[n, k]), t(&sb, 4, DType::F32, &[2])], vec![y.clone()], OpAttrs::default());
            let cl = with_scratch(&be, PACK, cl);
            run(&be, PACK, &cl, None);
            got.push(down(&be, &y));
        }
        assert_eq!(got[0], got[1], "is = {is}: x at byte offset 2 vs 0");
    }
}
