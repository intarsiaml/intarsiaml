//! Device-position path of the DeepSeek-V4.1 positioned ops on a real GPU (contract: docs/design/dsv41-ops.md, "Device
//! position").
//!
//! * every positioned op (`rope_tail`, `window_kv`, `act_qdq`, `compress_pool`, `indexer_blocks`, `indexer_topk`,
//!   `attn_sparse`) launched at capacity shapes with its position in an `i32 [1]` tensor must give exactly the bits of
//!   the host call at `pos0 = P` on the leading rows (same GPU), with the documented untouched / -1 / 0 regions;
//! * a CUDA graph capturing the device calls replays at many positions (the position tensor rewritten between replays)
//!   and each replay equals the eager device calls at that position;
//! * `INTARSIA_DSV41_DPOS_BENCH=1`: a timing table of device vs host calls (indexer_topk, indexer_blocks, attn_sparse).
//!
//! Needs a GPU: without one the tests print SKIPPED (and fail under CI=1 / INTARSIA_REQUIRE_GPU=1).
#![cfg(ik_cuda)]

mod common;

use intarsia_core::dsv41::ops::{n_blocks, QdqFmt, QdqScale};
use intarsia_core::rng::SplitMix64;
use intarsia_core::{Backend, DType, HostTensor, MemKind, Op, OpAttrs, OpCall, ShapeInfo, TensorRef};
use std::sync::Arc;

fn device() -> Option<Arc<dyn Backend>> {
    let ord = std::env::var("INTARSIA_CUDA_DEVICE").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
    match intarsia_backend_cuda::create(ord) {
        Ok(b) => Some(b),
        Err(e) => {
            common::skip("dsv41_dpos", &format!("no usable CUDA device ({e})"));
            None
        }
    }
}

fn put(be: &dyn Backend, t: &HostTensor) -> TensorRef {
    let buf = be.alloc(t.nbytes().max(16), MemKind::Device).unwrap();
    be.zero(&buf, 0).unwrap();
    if t.nbytes() > 0 {
        be.upload(&buf, 0, &t.data, 0).unwrap();
    }
    TensorRef::new(buf, t.dtype, &t.shape)
}

fn get(be: &dyn Backend, t: &TensorRef) -> Vec<u8> {
    let mut d = vec![0u8; t.nbytes()];
    be.synchronize(0).unwrap();
    if !d.is_empty() {
        be.download(&t.buf, t.offset, &mut d, 0).unwrap();
    }
    d
}

fn set(be: &dyn Backend, t: &TensorRef, data: &[u8]) {
    be.synchronize(0).unwrap();
    be.upload(&t.buf, t.offset, data, 0).unwrap();
}

fn rnd(rng: &mut SplitMix64, dt: DType, shape: &[usize], scale: f32) -> HostTensor {
    HostTensor::from_f32_as(dt, shape, &rng.normal_vec(shape.iter().product(), scale))
}

fn bytes(rng: &mut SplitMix64, dt: DType, shape: &[usize], f: impl Fn(u64) -> u8) -> HostTensor {
    let n: usize = shape.iter().product::<usize>() * dt.size_bytes();
    HostTensor::new(dt, shape.to_vec(), (0..n).map(|_| f(rng.next_u64())).collect()).unwrap()
}

/// `[R, C]` row-major bytes: the leading `c` columns.
fn lead_cols(data: &[u8], rows: usize, cols: usize, c: usize, es: usize) -> Vec<u8> {
    (0..rows).flat_map(|r| data[r * cols * es..(r * cols + c) * es].to_vec()).collect()
}

/// `[R, c]` rows padded with `fill` (one element) to `[R, cols]`.
fn pad_cols(data: &[u8], rows: usize, c: usize, cols: usize, fill: &[u8]) -> Vec<u8> {
    let es = fill.len();
    let mut v = Vec::with_capacity(rows * cols * es);
    for r in 0..rows {
        v.extend_from_slice(&data[r * c * es..(r + 1) * c * es]);
        for _ in c..cols {
            v.extend_from_slice(fill);
        }
    }
    v
}

/// Scratch of `kernel` for `call` (allocated once, reused by eager launches and graph captures).
fn scratch_for(be: &dyn Backend, kernel: &str, call: &OpCall) -> Option<TensorRef> {
    let k = be.kernels().into_iter().find(|k| k.name == kernel).expect("registered kernel");
    let need = k.scratch_bytes(&ShapeInfo::from_call(call));
    (need > 0).then(|| TensorRef::new(be.alloc(need.max(16), MemKind::Device).unwrap(), DType::U8, &[need]))
}

/// A prepared launch: a host-position call (`pos = None`, `attrs.pos0 = p`) or a device-position one.
struct Launch {
    kernel: &'static str,
    call: OpCall,
}

impl Launch {
    fn new(be: &dyn Backend, kernel: &'static str, op: Op, inputs: Vec<TensorRef>, outputs: Vec<TensorRef>, attrs: &OpAttrs, pos: Option<&TensorRef>) -> Launch {
        let mut call = OpCall::new(op, inputs, outputs, attrs.clone());
        call.pos = pos.cloned();
        if pos.is_some() {
            call.attrs.pos0 = 0; // ignored
        }
        call.check().unwrap_or_else(|e| panic!("{kernel}: {e}"));
        call.scratch = scratch_for(be, kernel, &call);
        Launch { kernel, call }
    }
    fn run(&self, be: &dyn Backend) {
        be.launch(self.kernel, &self.call, 0).unwrap_or_else(|e| panic!("{}: {e}", self.kernel));
    }
}

fn pos_t(be: &dyn Backend, p: usize) -> TensorRef {
    put(be, &HostTensor::from_i32(&[1], &[p as i32]))
}

fn assert_bits(what: &str, got: &[u8], want: &[u8]) {
    if got != want {
        let es = 1;
        let first = got.iter().zip(want).position(|(a, b)| a != b).unwrap_or(got.len().min(want.len()));
        panic!("{what}: device-position call differs from the host call (len {} vs {}, first differing byte {})", got.len(), want.len(), first / es);
    }
}

// ---- one op at one position: device result vs the host-equivalent result ------------------------------------------------

/// Bitwise check of every op at one configuration; returns how many comparisons ran.
mod check {
    use super::*;

    pub fn rope(be: &dyn Backend, rng: &mut SplitMix64, dt: DType, t: usize, ratio: usize, stride: usize, inverse: bool, p: usize) {
        let x = put(be, &rnd(rng, dt, &[t, 4, 128], 1.0));
        let f: Vec<f32> = (0..32).map(|j| 10000f32.powf(-(2 * j) as f32 / 64.0)).collect();
        let f = put(be, &HostTensor::from_f32(&[32], &f));
        let mut a = OpAttrs::default();
        (a.dsv.ratio, a.dsv.pos_stride, a.dsv.inverse) = (ratio, stride, inverse);
        let (od, oh) = (put(be, &HostTensor::zeros(dt, &[t, 4, 128])), put(be, &HostTensor::zeros(dt, &[t, 4, 128])));
        let pos = pos_t(be, p);
        Launch::new(be, "rope_tail", Op::RopeTail, vec![x.clone(), f.clone()], vec![od.clone()], &a, Some(&pos)).run(be);
        let ah = OpAttrs { pos0: p / ratio * ratio, ..a };
        Launch::new(be, "rope_tail", Op::RopeTail, vec![x, f], vec![oh.clone()], &ah, None).run(be);
        assert_bits(&format!("rope_tail P={p} ratio={ratio} T={t} {dt}"), &get(be, &od), &get(be, &oh));
    }

    /// In place (`x` is the output): what the graph's attention `q` and `o` ropes launch with `GraphTp::rope_inplace`. The in-place
    /// kernel visits the rope dims only and must leave exactly the bits of the copy kernel in the whole row, for a host position
    /// and for a device position, at the model's head width (512) and at two narrower ones (128; 64 = only rope dims).
    pub fn rope_in_place(be: &dyn Backend, rng: &mut SplitMix64, dt: DType, t: usize, ratio: usize, stride: usize, inverse: bool, p: usize) {
        let f: Vec<f32> = (0..32).map(|j| 10000f32.powf(-(2 * j) as f32 / 64.0)).collect();
        let f = put(be, &HostTensor::from_f32(&[32], &f));
        let mut a = OpAttrs::default();
        (a.dsv.ratio, a.dsv.pos_stride, a.dsv.inverse) = (ratio, stride, inverse);
        let ah = OpAttrs { pos0: p / ratio * ratio, ..a.clone() };
        let pos = pos_t(be, p);
        for dh in [512usize, 128, 64] {
            let xh = rnd(rng, dt, &[t, 4, dh], 1.0);
            let (x, want) = (put(be, &xh), put(be, &HostTensor::zeros(dt, &[t, 4, dh])));
            Launch::new(be, "rope_tail", Op::RopeTail, vec![x.clone(), f.clone()], vec![want.clone()], &ah, None).run(be);
            for dev in [false, true] {
                let xi = put(be, &xh);
                let (at, ps) = if dev { (&a, Some(&pos)) } else { (&ah, None) };
                Launch::new(be, "rope_tail", Op::RopeTail, vec![xi.clone(), f.clone()], vec![xi.clone()], at, ps).run(be);
                assert_bits(&format!("rope_tail in place (dev {dev}) P={p} ratio={ratio} stride={stride} inverse={inverse} T={t} Dh={dh} {dt}"), &get(be, &xi), &get(be, &want));
            }
        }
    }

    pub fn window(be: &dyn Backend, rng: &mut SplitMix64, dt: DType, t: usize, p: usize) {
        let (w, wc, d) = (128, 144, 512);
        let kv = put(be, &rnd(rng, dt, &[t, d], 1.0));
        let ring0 = rnd(rng, dt, &[wc, d], 1.0);
        let a = OpAttrs { window: w, ..OpAttrs::default() };
        let run = |dev: bool| {
            let (win, ring) = (put(be, &HostTensor::zeros(dt, &[w + t, d])), put(be, &ring0));
            let pos = pos_t(be, p);
            let ah = OpAttrs { pos0: p, ..a.clone() };
            Launch::new(be, "window_kv", Op::WindowKv, vec![kv.clone()], vec![win.clone(), ring.clone()], &ah, dev.then_some(&pos)).run(be);
            (get(be, &win), get(be, &ring))
        };
        let (d_, h_) = (run(true), run(false));
        assert_bits(&format!("window_kv win P={p} T={t}"), &d_.0, &h_.0);
        assert_bits(&format!("window_kv ring P={p} T={t}"), &d_.1, &h_.1);
    }

    pub fn qdq(be: &dyn Backend, rng: &mut SplitMix64, dt: DType, g: usize, ry: usize, n: usize, fmt: (QdqFmt, usize, QdqScale), ratio: usize, p: usize) {
        let x = put(be, &rnd(rng, dt, &[g, n], 3.0));
        let y0 = rnd(rng, dt, &[ry, n], 1.0);
        let mut a = OpAttrs::default();
        (a.dsv.qdq_fmt, a.dsv.qdq_block, a.dsv.qdq_scale, a.dsv.ratio) = (fmt.0, fmt.1, fmt.2, ratio);
        let (yd, yh) = (put(be, &y0), put(be, &y0));
        let pos = pos_t(be, p);
        Launch::new(be, "act_qdq", Op::ActQdq, vec![x.clone()], vec![yd.clone()], &a, Some(&pos)).run(be);
        let off = p / ratio;
        let rows = g.min(ry.saturating_sub(off));
        if rows > 0 {
            Launch::new(be, "act_qdq", Op::ActQdq, vec![x.rows(0, rows)], vec![yh.rows(off, rows)], &a, None).run(be);
        }
        assert_bits(&format!("act_qdq P={p} ratio={ratio} G={g} Ry={ry} {dt}"), &get(be, &yd), &get(be, &yh));
    }

    pub fn pool(be: &dyn Backend, rng: &mut SplitMix64, dt: DType, t: usize, ratio: usize, gmax: usize, p: usize) {
        let d = 512;
        let rc = ratio * 9;
        let (kv, sc) = (put(be, &rnd(rng, DType::F32, &[t, d], 1.0)), put(be, &rnd(rng, DType::F32, &[t, d], 1.0)));
        let (r0, r1, out0) = (rnd(rng, DType::F32, &[rc, d], 1.0), rnd(rng, DType::F32, &[rc, d], 1.0), rnd(rng, dt, &[gmax, d], 1.0));
        let mut a = OpAttrs::default();
        a.dsv.ratio = ratio;
        let g = (p + t) / ratio - p / ratio;
        let (od, rkd, rsd) = (put(be, &out0), put(be, &r0), put(be, &r1));
        let pos = pos_t(be, p);
        Launch::new(be, "compress_pool", Op::CompressPool, vec![kv.clone(), sc.clone()], vec![od.clone(), rkd.clone(), rsd.clone()], &a, Some(&pos)).run(be);
        let (oh, rkh, rsh) = (put(be, &HostTensor::zeros(dt, &[g, d])), put(be, &r0), put(be, &r1));
        let ah = OpAttrs { pos0: p, ..a };
        Launch::new(be, "compress_pool", Op::CompressPool, vec![kv, sc], vec![oh.clone(), rkh.clone(), rsh.clone()], &ah, None).run(be);
        let es = dt.size_bytes();
        let mut want = out0.data.clone();
        let k = g.min(gmax) * d * es;
        want[..k].copy_from_slice(&get(be, &oh)[..k]);
        let what = format!("compress_pool P={p} ratio={ratio} T={t} G={g} Gmax={gmax} {dt}");
        assert_bits(&format!("{what} out"), &get(be, &od), &want);
        assert_bits(&format!("{what} kv ring"), &get(be, &rkd), &get(be, &rkh));
        assert_bits(&format!("{what} score ring"), &get(be, &rsd), &get(be, &rsh));
    }

    /// Indexer inputs at capacity: q [T,Hi,Di], k [Ncap,Di], w [T,Hi], candidates [T, ceil(Ncap/B)].
    pub struct Idx {
        pub q: TensorRef,
        pub k: TensorRef,
        pub w: TensorRef,
        pub cand: HostTensor,
        pub t: usize,
        pub ncap: usize,
        pub b: usize,
        pub ratio: usize,
        pub attrs: OpAttrs,
    }

    #[allow(clippy::too_many_arguments)]
    pub fn idx_inputs(be: &dyn Backend, rng: &mut SplitMix64, dt: DType, t: usize, hi: usize, di: usize, b: usize, ncap: usize, ratio: usize) -> Idx {
        let mut attrs = OpAttrs { scale: (di as f32).powf(-0.5) * (hi as f32).powf(-0.5), ..OpAttrs::default() };
        (attrs.dsv.ratio, attrs.dsv.block) = (ratio, b);
        // fp4-like inputs (exactly representable products): the selection is on well-separated scores
        let q = put(be, &rnd(rng, dt, &[t, hi, di], 1.0));
        let k = put(be, &rnd(rng, dt, &[ncap, di], 1.0));
        let w = put(be, &rnd(rng, dt, &[t, hi], 1.0));
        let cand = bytes(rng, DType::U8, &[t, n_blocks(ncap, b)], |v| (v % 4 != 0) as u8);
        Idx { q, k, w, cand, t, ncap, b, ratio, attrs }
    }

    pub fn blocks(be: &dyn Backend, x: &Idx, keep: usize, p: usize) {
        let n = ((p + x.t) / x.ratio).min(x.ncap);
        let (nbc, nbe) = (n_blocks(x.ncap, x.b), n_blocks(n, x.b));
        let a = OpAttrs { top_k: keep, ..x.attrs.clone() };
        let md = put(be, &HostTensor::new(DType::U8, vec![x.t, nbc], vec![7u8; x.t * nbc]).unwrap());
        let pos = pos_t(be, p);
        Launch::new(be, "indexer_blocks", Op::IndexerBlocks, vec![x.q.clone(), x.k.clone(), x.w.clone()], vec![md.clone()], &a, Some(&pos)).run(be);
        let want = if n == 0 {
            vec![0u8; x.t * nbc]
        } else {
            let mh = put(be, &HostTensor::zeros(DType::U8, &[x.t, nbe]));
            let ah = OpAttrs { pos0: p, ..a };
            Launch::new(be, "indexer_blocks", Op::IndexerBlocks, vec![x.q.clone(), x.k.rows(0, n), x.w.clone()], vec![mh.clone()], &ah, None).run(be);
            pad_cols(&get(be, &mh), x.t, nbe, nbc, &[0])
        };
        assert_bits(&format!("indexer_blocks P={p} ratio={} T={} Ncap={} N={n}", x.ratio, x.t, x.ncap), &get(be, &md), &want);
    }

    pub fn topk(be: &dyn Backend, x: &Idx, top_k: usize, with_cand: bool, p: usize) {
        let n = ((p + x.t) / x.ratio).min(x.ncap);
        let (nbc, nbe) = (n_blocks(x.ncap, x.b), n_blocks(n, x.b));
        let (kw, kh) = (top_k.min(x.ncap), top_k.min(n));
        let a = OpAttrs { top_k, ..x.attrs.clone() };
        let none = HostTensor::zeros(DType::U8, &[0]);
        let cand_d = put(be, if with_cand { &x.cand } else { &none });
        let idd = put(be, &HostTensor::from_i32(&[x.t, kw], &vec![12345; x.t * kw]));
        let pos = pos_t(be, p);
        Launch::new(be, "indexer_topk", Op::IndexerTopk, vec![x.q.clone(), x.k.clone(), x.w.clone(), cand_d], vec![idd.clone()], &a, Some(&pos)).run(be);
        let want = if n == 0 {
            [0xffu8; 4].repeat(x.t * kw)
        } else {
            let ch = if with_cand { HostTensor::new(DType::U8, vec![x.t, nbe], lead_cols(&x.cand.data, x.t, nbc, nbe, 1)).unwrap() } else { none.clone() };
            let idh = put(be, &HostTensor::zeros(DType::I32, &[x.t, kh]));
            let ah = OpAttrs { pos0: p, ..a };
            Launch::new(be, "indexer_topk", Op::IndexerTopk, vec![x.q.clone(), x.k.rows(0, n), x.w.clone(), put(be, &ch)], vec![idh.clone()], &ah, None).run(be);
            pad_cols(&get(be, &idh), x.t, kh, kw, &[0xff; 4])
        };
        assert_bits(&format!("indexer_topk P={p} ratio={} T={} Ncap={} N={n} cand={with_cand}", x.ratio, x.t, x.ncap), &get(be, &idd), &want);
    }

    /// Sparse attention at capacity: cmp [Ncap,D], idx [T,Kw] whose first Kc = min(Kw, N) columns are -1 or visible rows
    /// and the rest garbage (never read).
    #[allow(clippy::too_many_arguments)]
    pub fn attn(be: &dyn Backend, rng: &mut SplitMix64, dt: DType, t: usize, h: usize, d: usize, w: usize, ncap: usize, kw: usize, ratio: usize, p: usize) {
        let n = ((p + t) / ratio).min(ncap);
        let kc = if ncap > 0 { kw.min(n) } else { kw };
        let q = put(be, &rnd(rng, dt, &[t, h, d], 1.0));
        let win = put(be, &rnd(rng, dt, &[w + t, d], 1.0));
        let cmp = put(be, &rnd(rng, dt, &[ncap, d], 1.0));
        let sink = put(be, &rnd(rng, DType::F32, &[h], 1.0));
        let iv: Vec<i32> =
            (0..t * kw).map(|e| if e % kw >= kc { 1 << 30 } else if n == 0 { -1 } else { ((rng.next_u64() % (n as u64 + 2)) as i32 - 2).max(-1) }).collect();
        let idx = HostTensor::from_i32(&[t, kw], &iv);
        let mut a = OpAttrs { window: w, scale: (d as f32).powf(-0.5), ..OpAttrs::default() };
        a.dsv.ratio = ratio;
        let (od, oh) = (put(be, &HostTensor::zeros(dt, &[t, h, d])), put(be, &HostTensor::zeros(dt, &[t, h, d])));
        let pos = pos_t(be, p);
        Launch::new(be, "attn_sparse", Op::AttnSparse, vec![q.clone(), win.clone(), cmp.clone(), put(be, &idx), sink.clone()], vec![od.clone()], &a, Some(&pos))
            .run(be);
        let ih = HostTensor::new(DType::I32, vec![t, kc], lead_cols(&idx.data, t, kw, kc, 4)).unwrap();
        let ah = OpAttrs { pos0: p, ..a };
        Launch::new(be, "attn_sparse", Op::AttnSparse, vec![q, win, cmp.rows(0, n), put(be, &ih), sink], vec![oh.clone()], &ah, None).run(be);
        assert_bits(&format!("attn_sparse P={p} ratio={ratio} T={t} H={h} D={d} W={w} Ncap={ncap} N={n} Kw={kw} Kc={kc} {dt}"), &get(be, &od), &get(be, &oh));
    }
}

#[test]
fn rope_window_qdq_pool_device_position_is_bitwise_the_host_call() {
    let Some(be) = device() else { return };
    let be = &*be;
    let mut rng = SplitMix64::new(41);
    let r = &mut rng;
    for dt in [DType::Bf16, DType::F32] {
        for t in [1usize, 4, 5] {
            for (ratio, stride, inverse) in [(1, 1, false), (1, 1, true), (4, 4, false), (2, 2, false)] {
                for p in [0usize, 1, 3, 4, 7, 127, 4093, 30000] {
                    check::rope(be, r, dt, t, ratio, stride, inverse, p);
                    check::rope_in_place(be, r, dt, t, ratio, stride, inverse, p);
                }
            }
            for p in [0usize, 1, 100, 127, 128, 129, 143, 144, 4095, 30001] {
                check::window(be, r, dt, t, p);
            }
            for (fmt, ratio) in [((QdqFmt::Fp8E4m3, 512, QdqScale::Ue8m0), 1), ((QdqFmt::Fp4E2m1, 32, QdqScale::Ue8m0), 1), ((QdqFmt::Fp4E2m1, 16, QdqScale::E4m3), 2)] {
                let n = if fmt.1 == 512 { 512 } else { 128 };
                for p in [0usize, 1, 2, 3, 30, 61, 62, 63, 64, 70] {
                    check::qdq(be, r, dt, t, 32, n, fmt, ratio, p);
                }
            }
            for (ratio, gmax) in [(2usize, 3usize), (4, 2), (2, 1)] {
                for p in [0usize, 1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 1000, 1001, 4095] {
                    check::pool(be, r, dt, t, ratio, gmax, p);
                }
            }
        }
    }
}

#[test]
fn indexer_device_position_is_bitwise_the_host_call() {
    let Some(be) = device() else { return };
    let be = &*be;
    let mut rng = SplitMix64::new(43);
    let r = &mut rng;
    // (dtype, Hi, Di, B): bf16 32x128 B8 is the model's shape (tensor-core score); 4x32 B4 and f32 take the SIMT score
    for (dt, hi, di, b) in [(DType::Bf16, 32usize, 128usize, 8usize), (DType::Bf16, 4, 32, 4), (DType::F32, 32, 128, 8)] {
        for t in [1usize, 4] {
            // small capacity: one-block select on both sides; N from 0 to past top_k
            for ratio in [1usize, 2] {
                let x = check::idx_inputs(be, r, dt, t, hi, di, b, 300, ratio);
                for p in [0usize, 1, 2, 5, 7, 8, 63, 64, 100, 299 * ratio - t, 300 * ratio - t] {
                    check::blocks(be, &x, 12, p);
                    check::topk(be, &x, 64, false, p);
                    check::topk(be, &x, 64, true, p);
                }
            }
            // capacity past the fast-select bound (16384): the device call launches the multi-block select's kernels and
            // falls back to the one-block select inside them below 16384 rows, as the host call does by its shape
            if dt == DType::Bf16 && hi == 32 {
                for ratio in [1usize, 2] {
                    let ncap = 20000;
                    let x = check::idx_inputs(be, r, dt, t, hi, di, b, ncap, ratio);
                    for p in [0usize, 3, 100, 600, 1000, 4100, 16383 * ratio, 16384 * ratio, 19990 * ratio] {
                        check::blocks(be, &x, 2048, p);
                        check::topk(be, &x, 512, false, p);
                        check::topk(be, &x, 512, true, p);
                    }
                }
            }
        }
    }
}

#[test]
fn attn_sparse_device_position_is_bitwise_the_host_call() {
    let Some(be) = device() else { return };
    let be = &*be;
    let mut rng = SplitMix64::new(47);
    let r = &mut rng;
    // bf16 H 32 D 512: tensor-core kernel; H 4 / f32: SIMT kernel
    for (dt, h, d, w) in [(DType::Bf16, 32usize, 512usize, 128usize), (DType::Bf16, 4, 64, 8), (DType::F32, 32, 512, 128), (DType::F32, 4, 64, 8)] {
        for t in [1usize, 4, 17] {
            // Ncap 0 (window-only layer), small / large capacity at ratio 1 / 2, Kw = min(512, Ncap)
            for (ncap, ratio) in [(0usize, 1usize), (40, 1), (40, 2), (2000, 1), (2000, 2)] {
                let kw = if ncap == 0 { 16 } else { 512.min(ncap) };
                for p in [0usize, 1, 2, 7, 8, 9, 30, 100, 511, 600, 1000, 1999 * ratio - t.min(1999 * ratio)] {
                    check::attn(be, r, dt, t, h, d, w, ncap, kw, ratio, p);
                }
            }
        }
    }
}

// ---- graph capture / replay ------------------------------------------------------------------------------------------

#[test]
fn graph_of_device_calls_replays_at_every_position() {
    let Some(be) = device() else { return };
    assert!(be.supports_graphs());
    let be = &*be;
    let mut rng = SplitMix64::new(53);
    let r = &mut rng;
    let dt = DType::Bf16;
    // one decode step of a ratio-2 compressing layer (T = 1, then a T = 4 verify) at capacity shapes
    for t in [1usize, 4] {
        let (ratio, d, w, wc, hi, di, b, ncap, h) = (2usize, 512usize, 128usize, 144usize, 32usize, 128usize, 8usize, 20000usize, 32usize);
        let rc = ratio * 9;
        let pos = pos_t(be, 0);
        let f: Vec<f32> = (0..32).map(|j| 10000f32.powf(-(2 * j) as f32 / 64.0)).collect();
        let fr = put(be, &HostTensor::from_f32(&[32], &f));
        // inputs (fixed) and state / outputs (reset before each run)
        let x = put(be, &rnd(r, dt, &[t, 1, d], 1.0));
        let kv = put(be, &rnd(r, dt, &[t, d], 1.0));
        let (pkv, psc) = (put(be, &rnd(r, DType::F32, &[t, d], 1.0)), put(be, &rnd(r, DType::F32, &[t, d], 1.0)));
        let ix = check::idx_inputs(be, r, dt, t, hi, di, b, ncap, ratio);
        let q = put(be, &rnd(r, dt, &[t, h, d], 1.0));
        let sink = put(be, &rnd(r, DType::F32, &[h], 1.0));
        let cmp_src = put(be, &rnd(r, dt, &[ncap, d], 1.0));
        let gmax = t.div_ceil(ratio) + 1;
        let state0: Vec<HostTensor> = vec![
            rnd(r, dt, &[wc, d], 1.0),                // window ring
            rnd(r, DType::F32, &[rc, d], 1.0),         // compressor kv ring
            rnd(r, DType::F32, &[rc, d], 1.0),         // compressor score ring
            rnd(r, dt, &[gmax, d], 1.0),               // pooled rows
            rnd(r, dt, &[ncap, d], 1.0),               // shared compressed KV (written by act_qdq)
        ];
        let state: Vec<TensorRef> = state0.iter().map(|s| put(be, s)).collect();
        let rope_o = put(be, &HostTensor::zeros(dt, &[t, 1, d]));
        let win = put(be, &HostTensor::zeros(dt, &[w + t, d]));
        let mask = put(be, &HostTensor::zeros(DType::U8, &[t, n_blocks(ncap, b)]));
        let idx = put(be, &HostTensor::zeros(DType::I32, &[t, 512]));
        let out = put(be, &HostTensor::zeros(dt, &[t, h, d]));
        let cand = put(be, &ix.cand);
        let mut ar = OpAttrs::default();
        (ar.dsv.ratio, ar.dsv.pos_stride) = (ratio, ratio);
        let mut aq = OpAttrs::default();
        (aq.dsv.qdq_fmt, aq.dsv.qdq_block, aq.dsv.qdq_scale, aq.dsv.ratio) = (QdqFmt::Fp4E2m1, 16, QdqScale::E4m3, ratio);
        let mut ap = OpAttrs::default();
        ap.dsv.ratio = ratio;
        let mut aa = OpAttrs { window: w, scale: (d as f32).powf(-0.5), ..OpAttrs::default() };
        aa.dsv.ratio = ratio;
        let steps = vec![
            Launch::new(be, "rope_tail", Op::RopeTail, vec![x.clone(), fr.clone()], vec![rope_o.clone()], &ar, Some(&pos)),
            Launch::new(be, "window_kv", Op::WindowKv, vec![kv.clone()], vec![win.clone(), state[0].clone()], &OpAttrs { window: w, ..OpAttrs::default() }, Some(&pos)),
            Launch::new(be, "compress_pool", Op::CompressPool, vec![pkv.clone(), psc.clone()], vec![state[3].clone(), state[1].clone(), state[2].clone()], &ap, Some(&pos)),
            Launch::new(be, "act_qdq", Op::ActQdq, vec![state[3].rows(0, gmax)], vec![state[4].clone()], &aq, Some(&pos)),
            Launch::new(be, "indexer_blocks", Op::IndexerBlocks, vec![ix.q.clone(), ix.k.clone(), ix.w.clone()], vec![mask.clone()], &OpAttrs { top_k: 2048, ..ix.attrs.clone() }, Some(&pos)),
            Launch::new(be, "indexer_topk", Op::IndexerTopk, vec![ix.q.clone(), ix.k.clone(), ix.w.clone(), cand.clone()], vec![idx.clone()], &OpAttrs { top_k: 512, ..ix.attrs.clone() }, Some(&pos)),
            Launch::new(be, "attn_sparse", Op::AttnSparse, vec![q.clone(), win.clone(), cmp_src.clone(), idx.clone(), sink.clone()], vec![out.clone()], &aa, Some(&pos)),
        ];
        let outs: Vec<TensorRef> = [rope_o.clone(), win.clone(), mask.clone(), idx.clone(), out.clone()].into_iter().chain(state.iter().cloned()).collect();
        let reset = || {
            for (s, s0) in state.iter().zip(&state0) {
                set(be, s, &s0.data);
            }
        };
        let snap = || outs.iter().map(|o| get(be, o)).collect::<Vec<_>>();
        // warm up every kernel outside a capture (process-wide flags, function attributes)
        reset();
        steps.iter().for_each(|s| s.run(be));
        be.synchronize(0).unwrap();
        let graph = be
            .graph_capture(0, &mut || {
                for s in &steps {
                    be.launch(s.kernel, &s.call, 0)?;
                }
                Ok(())
            })
            .unwrap();
        let positions = [0usize, 1, 2, 3, 100, 1001, 4096, 16000, 30001, 39990 - t, 7];
        for &p in &positions {
            set(be, &pos, &(p as i32).to_le_bytes());
            reset();
            be.graph_replay(&graph, 0).unwrap();
            let got = snap();
            reset();
            steps.iter().for_each(|s| s.run(be));
            let want = snap();
            for (j, (g, w_)) in got.iter().zip(&want).enumerate() {
                assert_bits(&format!("graph replay T={t} P={p} output {j}"), g, w_);
            }
        }
        // the eager device calls themselves are the host calls (spot check at one position through the op checks above)
        check::topk(be, &ix, 512, true, 1001);
        check::attn(be, r, dt, t, h, d, w, ncap, 512, ratio, 1001);
    }
}

// ---- timing ------------------------------------------------------------------------------------------------------------

/// (eager, graph) microseconds per call: back-to-back eager launches (host launch cost included), and replays of a graph
/// holding 50 launches (about the GPU time per call, as inside the captured decode step).
fn time_us(be: &dyn Backend, l: &Launch, iters: usize) -> (f64, f64) {
    for _ in 0..5 {
        l.run(be);
    }
    be.synchronize(0).unwrap();
    let t0 = std::time::Instant::now();
    for _ in 0..iters {
        l.run(be);
    }
    be.synchronize(0).unwrap();
    let eager = t0.elapsed().as_secs_f64() * 1e6 / iters as f64;
    const PER: usize = 50;
    let g = be
        .graph_capture(0, &mut || {
            for _ in 0..PER {
                be.launch(l.kernel, &l.call, 0)?;
            }
            Ok(())
        })
        .unwrap();
    be.graph_replay(&g, 0).unwrap();
    be.synchronize(0).unwrap();
    let reps = 10;
    let t0 = std::time::Instant::now();
    for _ in 0..reps {
        be.graph_replay(&g, 0).unwrap();
    }
    be.synchronize(0).unwrap();
    (eager, t0.elapsed().as_secs_f64() * 1e6 / (reps * PER) as f64)
}

/// Device-position vs host-position cost at the model's shapes (capacity 32768 tokens): `INTARSIA_DSV41_DPOS_BENCH=1`.
#[test]
fn device_position_timing() {
    if std::env::var("INTARSIA_DSV41_DPOS_BENCH").ok().as_deref() != Some("1") {
        return;
    }
    let Some(be) = device() else { return };
    let be = &*be;
    let mut rng = SplitMix64::new(59);
    let r = &mut rng;
    let dt = DType::Bf16;
    let (hi, di, b, h, d, w, cap) = (32usize, 128usize, 8usize, 32usize, 512usize, 128usize, 32768usize);
    println!("TIMING op ratio T P | eager host us | eager device us | delta | graph host us | graph device us | delta");
    for ratio in [1usize, 2] {
        let ncap = cap / ratio;
        for t in [1usize, 4] {
            let x = check::idx_inputs(be, r, dt, t, hi, di, b, ncap, ratio);
            let q = put(be, &rnd(r, dt, &[t, h, d], 1.0));
            let win = put(be, &rnd(r, dt, &[w + t, d], 1.0));
            let cmp = put(be, &rnd(r, dt, &[ncap, d], 1.0));
            let sink = put(be, &rnd(r, DType::F32, &[h], 1.0));
            let cand_full = put(be, &x.cand);
            for p in [100usize, 1000, 8000, 30000] {
                let n = ((p + t) / ratio).min(ncap);
                let pos = pos_t(be, p);
                let (nbc, nbe) = (n_blocks(ncap, b), n_blocks(n, b));
                let (kw, kh) = (512.min(ncap), 512.min(n));
                let host = OpAttrs { pos0: p, ..x.attrs.clone() };
                let row = |name: &str, h: (f64, f64), d: (f64, f64)| {
                    println!("TIMING {name} {ratio} {t} {p} | {:8.2} | {:8.2} | {:+.2} | {:8.2} | {:8.2} | {:+.2}", h.0, d.0, d.0 - h.0, h.1, d.1, d.1 - h.1)
                };
                // indexer_blocks (keep 2048)
                let md = put(be, &HostTensor::zeros(DType::U8, &[t, nbc]));
                let mh = put(be, &HostTensor::zeros(DType::U8, &[t, nbe]));
                let ld = Launch::new(be, "indexer_blocks", Op::IndexerBlocks, vec![x.q.clone(), x.k.clone(), x.w.clone()], vec![md], &OpAttrs { top_k: 2048, ..x.attrs.clone() }, Some(&pos));
                let lh = Launch::new(be, "indexer_blocks", Op::IndexerBlocks, vec![x.q.clone(), x.k.rows(0, n), x.w.clone()], vec![mh], &OpAttrs { top_k: 2048, ..host.clone() }, None);
                row("indexer_blocks", time_us(be, &lh, 200), time_us(be, &ld, 200));
                // indexer_topk (top 512, with candidates)
                let ch = put(be, &HostTensor::new(DType::U8, vec![t, nbe], lead_cols(&x.cand.data, t, nbc, nbe, 1)).unwrap());
                let id = put(be, &HostTensor::zeros(DType::I32, &[t, kw]));
                let ih = put(be, &HostTensor::zeros(DType::I32, &[t, kh]));
                let ld = Launch::new(be, "indexer_topk", Op::IndexerTopk, vec![x.q.clone(), x.k.clone(), x.w.clone(), cand_full.clone()], vec![id.clone()], &OpAttrs { top_k: 512, ..x.attrs.clone() }, Some(&pos));
                let lh = Launch::new(be, "indexer_topk", Op::IndexerTopk, vec![x.q.clone(), x.k.rows(0, n), x.w.clone(), ch], vec![ih.clone()], &OpAttrs { top_k: 512, ..host.clone() }, None);
                row("indexer_topk", time_us(be, &lh, 200), time_us(be, &ld, 200));
                // attn_sparse on the picks
                let mut aa = OpAttrs { window: w, scale: (d as f32).powf(-0.5), ..OpAttrs::default() };
                aa.dsv.ratio = ratio;
                let od = put(be, &HostTensor::zeros(dt, &[t, h, d]));
                let ld = Launch::new(be, "attn_sparse", Op::AttnSparse, vec![q.clone(), win.clone(), cmp.clone(), id, sink.clone()], vec![od.clone()], &aa, Some(&pos));
                let lh = Launch::new(be, "attn_sparse", Op::AttnSparse, vec![q.clone(), win.clone(), cmp.rows(0, n), ih, sink.clone()], vec![od], &OpAttrs { pos0: p, ..aa.clone() }, None);
                row("attn_sparse", time_us(be, &lh, 200), time_us(be, &ld, 200));
            }
        }
    }
}
