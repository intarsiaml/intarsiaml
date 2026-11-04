//! DeepSeek-V4.1 kernels of lane cuda-attn (src/dsv41.rs, kernels/cuda/dsv41_*.cu) at the model's per-rank shapes (TP=2:
//! 32 of 64 heads; indexer and hc_* replicated) against the cpu reference, and an ignored decode / prefill microbenchmark:
//! `INTARSIA_REQUIRE_GPU=1 cargo test --release -p intarsia-backend-cuda --test dsv41 -- --ignored --nocapture`.
//! The admission sweep (`intarsia kernels test --backend cuda --op <op>`) is the gate; these add the real shapes.
//! Needs a GPU: without one the tests print SKIPPED (and fail under CI=1 / INTARSIA_REQUIRE_GPU=1).
#![cfg(ik_cuda)]

mod common;

use intarsia_backend_cpu::CpuBackend;
use intarsia_core::admission::judge;
use intarsia_core::backend::{Backend, MemKind, TensorRef};
use intarsia_core::dsv41::ops::{QdqFmt, QdqScale};
use intarsia_core::exec::{run_host, HostCall, OutSpec};
use intarsia_core::ops::OpAttrs;
use intarsia_core::rng::SplitMix64;
use intarsia_core::{DType, HostTensor, Op, OpCall, ShapeInfo};
use std::sync::Arc;
use std::time::Instant;

fn device() -> Option<Arc<dyn Backend>> {
    let ord = std::env::var("INTARSIA_CUDA_DEVICE").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
    match intarsia_backend_cuda::create(ord) {
        Ok(b) => Some(b),
        Err(e) => {
            common::skip("dsv41", &format!("no usable CUDA device ({e})"));
            None
        }
    }
}

fn rand_t(rng: &mut SplitMix64, dt: DType, shape: &[usize], scale: f32) -> HostTensor {
    let n: usize = shape.iter().product();
    HostTensor::from_f32_as(dt, shape, &rng.normal_vec(n, scale))
}

fn out(dt: DType, shape: &[usize]) -> OutSpec {
    OutSpec::new(dt, shape)
}

fn inout(t: HostTensor) -> OutSpec {
    OutSpec { dtype: t.dtype, shape: t.shape.clone(), init: Some(t) }
}

// DeepSeek-V4.1 Flash per rank at TP=2 (config.json of the Mia checkpoint)
const D: usize = 5120;
const S: usize = 4;
const H: usize = 32; // 64 heads / 2 ranks
const HD: usize = 512;
const RD: usize = 64;
const W: usize = 128;
const HI: usize = 32; // indexer heads (replicated)
const DI: usize = 128;
const TOPK: usize = 512;
const E: usize = 384;
const BF: DType = DType::Bf16;

fn inv_freq() -> HostTensor {
    let f: Vec<f32> = (0..RD / 2).map(|j| 10000f32.powf(-((2 * j) as f32) / RD as f32)).collect();
    HostTensor::from_f32(&[RD / 2], &f)
}

/// (label, kernel, call) at decode (T = 1, position `pos`) and a few multi-row shapes.
fn cases(t: usize, pos: usize, nk_ratio1: usize) -> Vec<(String, &'static str, HostCall)> {
    let mut rng = SplitMix64::new(0x5eed ^ (t as u64) ^ ((pos as u64) << 20));
    let r = &mut rng;
    let mut v = Vec::new();
    let a = OpAttrs::default();
    let m = (2 + S) * S;
    {
        let mut at = OpAttrs { eps: 1e-20, ..a.clone() };
        (at.dsv.iters, at.dsv.hc_eps) = (20, 1e-6);
        v.push((
            format!("hc_mixes T={t}"),
            "hc_mixes",
            HostCall {
                op: Op::HcMixes,
                inputs: vec![rand_t(r, BF, &[t, S, D], 1.0), rand_t(r, DType::F32, &[m, S * D], 0.01), rand_t(r, DType::F32, &[3], 0.5), rand_t(r, DType::F32, &[m], 0.5)],
                outputs: vec![out(DType::F32, &[t, S]), out(DType::F32, &[t, S]), out(DType::F32, &[t, S, S])],
                attrs: at,
            },
        ));
    }
    v.push((format!("hc_expand T={t}"), "hc_expand", HostCall { op: Op::HcExpand, inputs: vec![rand_t(r, BF, &[t, D], 1.0)], outputs: vec![out(BF, &[t, S, D])], attrs: a.clone() }));
    v.push((
        format!("hc_pre T={t}"),
        "hc_pre",
        HostCall { op: Op::HcPre, inputs: vec![rand_t(r, BF, &[t, S, D], 1.0), rand_t(r, DType::F32, &[t, S], 0.7)], outputs: vec![out(BF, &[t, D])], attrs: a.clone() },
    ));
    v.push((
        format!("hc_post T={t}"),
        "hc_post",
        HostCall {
            op: Op::HcPost,
            inputs: vec![rand_t(r, BF, &[t, D], 1.0), rand_t(r, BF, &[t, S, D], 1.0), rand_t(r, DType::F32, &[t, S], 1.0), rand_t(r, DType::F32, &[t, S, S], 0.5)],
            outputs: vec![out(BF, &[t, S, D])],
            attrs: a.clone(),
        },
    ));
    for fold in [false, true] {
        let mut at = OpAttrs { eps: 1e-20, ..a.clone() };
        (at.dsv.iters, at.dsv.hc_eps) = (20, 1e-6);
        let mut inputs = vec![
            rand_t(r, BF, &[t, S, D], 1.0),
            rand_t(r, DType::F32, &[m, S * D], 0.01),
            rand_t(r, DType::F32, &[3], 0.5),
            rand_t(r, DType::F32, &[m], 0.5),
            rand_t(r, DType::F32, &[t, S], 0.7),
            rand_t(r, BF, &[D], 1.0),
        ];
        let h_out = if fold {
            inputs.extend([rand_t(r, BF, &[t, D], 1.0), rand_t(r, DType::F32, &[t, S], 1.0), rand_t(r, DType::F32, &[t, S, S], 0.5)]);
            out(BF, &[t, S, D])
        } else {
            inputs.extend([HostTensor::zeros(BF, &[0]), HostTensor::zeros(DType::F32, &[0]), HostTensor::zeros(DType::F32, &[0])]);
            out(BF, &[0])
        };
        v.push((
            format!("hc_sublayer fold={fold} T={t}"),
            "hc_sublayer",
            HostCall {
                op: Op::HcSublayer,
                inputs,
                outputs: vec![out(BF, &[t, D]), out(DType::F32, &[t, S]), out(DType::F32, &[t, S]), out(DType::F32, &[t, S, S]), h_out],
                attrs: at,
            },
        ));
    }
    v.push((
        format!("rmsnorm (separate path) T={t}"),
        "rmsnorm_bf16",
        HostCall { op: Op::RmsNorm, inputs: vec![rand_t(r, BF, &[t, D], 1.0), rand_t(r, BF, &[D], 1.0)], outputs: vec![out(BF, &[t, D])], attrs: OpAttrs { eps: 1e-20, ..a.clone() } },
    ));
    v.push((
        format!("moe_route T={t} E={E} K=6"),
        "moe_route",
        HostCall {
            op: Op::MoeRoute,
            inputs: vec![rand_t(r, DType::F32, &[t, E], 2.0), rand_t(r, DType::F32, &[E], 0.1)],
            outputs: vec![out(DType::F32, &[t, 6]), out(DType::I32, &[t, 6])],
            attrs: OpAttrs { top_k: 6, scale: 1.5, norm_topk: true, ..a.clone() },
        },
    ));
    {
        let mut at = a.clone();
        at.dsv.limit = 10.0;
        v.push((
            format!("silu_glu_clamp T={t} I=1152"),
            "silu_glu_clamp",
            HostCall { op: Op::SiluGluClamp, inputs: vec![rand_t(r, BF, &[t, 1152], 8.0), rand_t(r, BF, &[t, 1152], 8.0)], outputs: vec![out(BF, &[t, 1152])], attrs: at },
        ));
    }
    for (what, n, fmt, block, sc) in [
        ("kv e4m3/32", HD, QdqFmt::Fp8E4m3, 32, QdqScale::Ue8m0),
        ("o e4m3/32", H * HD, QdqFmt::Fp8E4m3, 32, QdqScale::Ue8m0),
        ("index_q e2m1/32", HI * DI, QdqFmt::Fp4E2m1, 32, QdqScale::Ue8m0),
        ("latent e2m1/16 e4m3", HD, QdqFmt::Fp4E2m1, 16, QdqScale::E4m3),
    ] {
        let mut at = a.clone();
        (at.dsv.qdq_fmt, at.dsv.qdq_block, at.dsv.qdq_scale) = (fmt, block, sc);
        v.push((format!("act_qdq {what} T={t}"), "act_qdq", HostCall { op: Op::ActQdq, inputs: vec![rand_t(r, BF, &[t, n], 3.0)], outputs: vec![out(BF, &[t, n])], attrs: at }));
    }
    for (what, n, dh, inverse) in [("q", H, HD, false), ("o inverse", H, HD, true), ("kv", 1, HD, false), ("index_q", HI, DI, false)] {
        let mut at = OpAttrs { pos0: pos, ..a.clone() };
        (at.dsv.pos_stride, at.dsv.inverse) = (1, inverse);
        v.push((
            format!("rope_tail {what} T={t}"),
            "rope_tail",
            HostCall { op: Op::RopeTail, inputs: vec![rand_t(r, BF, &[t, n, dh], 1.0), inv_freq()], outputs: vec![out(BF, &[t, n, dh])], attrs: at },
        ));
    }
    v.push((
        format!("window_kv T={t} pos0={pos}"),
        "window_kv",
        HostCall {
            op: Op::WindowKv,
            inputs: vec![rand_t(r, BF, &[t, HD], 1.0)],
            outputs: vec![out(BF, &[W + t, HD]), inout(rand_t(r, BF, &[W, HD], 1.0))],
            attrs: OpAttrs { window: W, pos0: pos, ..a.clone() },
        },
    ));
    {
        // compressed rows of a ratio-2 source at this position, top-512 indices with some -1
        let nc = (pos + t) / 2;
        let kc = TOPK.min(nc);
        let idx: Vec<i32> = (0..t * kc).map(|_| if r.next_u64() % 16 == 0 { -1 } else { (r.next_u64() % nc as u64) as i32 }).collect();
        v.push((
            format!("attn_sparse T={t} H={H} D={HD} Nc={nc} Kc={kc} pos0={pos}"),
            "attn_sparse",
            HostCall {
                op: Op::AttnSparse,
                inputs: vec![
                    rand_t(r, BF, &[t, H, HD], 1.0),
                    rand_t(r, BF, &[W + t, HD], 1.0),
                    rand_t(r, BF, &[nc, HD], 1.0),
                    HostTensor::from_i32(&[t, kc], &idx),
                    rand_t(r, DType::F32, &[H], 1.0),
                ],
                outputs: vec![out(BF, &[t, H, HD])],
                attrs: OpAttrs { window: W, pos0: pos, scale: (HD as f32).powf(-0.5), ..a.clone() },
            },
        ));
    }
    {
        let ratio = 2;
        let g = (pos + t) / ratio - pos / ratio;
        let mut at = OpAttrs { pos0: pos, ..a.clone() };
        at.dsv.ratio = ratio;
        v.push((
            format!("compress_pool ratio=2 T={t} pos0={pos} G={g}"),
            "compress_pool",
            HostCall {
                op: Op::CompressPool,
                inputs: vec![rand_t(r, DType::F32, &[t, HD], 1.0), rand_t(r, DType::F32, &[t, HD], 1.0)],
                outputs: vec![out(BF, &[g, HD]), inout(rand_t(r, DType::F32, &[ratio, HD], 1.0)), inout(rand_t(r, DType::F32, &[ratio, HD], 1.0))],
                attrs: at,
            },
        ));
    }
    {
        // indexer of a ratio-1 source (layer 20) at `nk_ratio1` compressed rows: candidate blocks, plain top-512, top-512 in candidates
        let nk = nk_ratio1;
        let pos0 = nk - t;
        let nb = nk.div_ceil(8);
        let scale = (DI as f32).powf(-0.5) * (HI as f32).powf(-0.5);
        let q = rand_t(r, BF, &[t, HI, DI], 1.0);
        let k = rand_t(r, BF, &[nk, DI], 1.0);
        let w = rand_t(r, BF, &[t, HI], 1.0);
        let mut at = OpAttrs { pos0, scale, ..a.clone() };
        (at.dsv.ratio, at.dsv.block) = (1, 8);
        v.push((
            format!("indexer_blocks T={t} Nk={nk} keep=2048"),
            "indexer_blocks",
            HostCall {
                op: Op::IndexerBlocks,
                inputs: vec![q.clone(), k.clone(), w.clone()],
                outputs: vec![out(DType::U8, &[t, nb])],
                attrs: OpAttrs { top_k: 2048, ..at.clone() },
            },
        ));
        let kk = TOPK.min(nk);
        v.push((
            format!("indexer_topk T={t} Nk={nk} top{TOPK}"),
            "indexer_topk",
            HostCall {
                op: Op::IndexerTopk,
                inputs: vec![q.clone(), k.clone(), w.clone(), HostTensor::zeros(DType::U8, &[0])],
                outputs: vec![out(DType::I32, &[t, kk])],
                attrs: OpAttrs { top_k: TOPK, ..at.clone() },
            },
        ));
        let keep = 2048.min(nb);
        let cand: Vec<u8> = (0..t * nb).map(|j| ((j % nb) < keep / 2 || r.next_u64() % (nb as u64) < (keep / 2) as u64) as u8).collect();
        v.push((
            format!("indexer_topk T={t} Nk={nk} top{TOPK} in candidates"),
            "indexer_topk",
            HostCall {
                op: Op::IndexerTopk,
                inputs: vec![q, k, w, HostTensor::new(DType::U8, vec![t, nb], cand).unwrap()],
                outputs: vec![out(DType::I32, &[t, kk])],
                attrs: OpAttrs { top_k: TOPK, ..at },
            },
        ));
    }
    v
}

// ---- E5 bit guard ---------------------------------------------------------------------------------------------------------

/// `std::mt19937` (the bench's generator).
struct Mt19937 {
    mt: [u32; 624],
    i: usize,
}

impl Mt19937 {
    fn new(seed: u32) -> Self {
        let mut mt = [0u32; 624];
        mt[0] = seed;
        for i in 1..624 {
            mt[i] = 1812433253u32.wrapping_mul(mt[i - 1] ^ (mt[i - 1] >> 30)).wrapping_add(i as u32);
        }
        Mt19937 { mt, i: 624 }
    }
    fn next(&mut self) -> u32 {
        if self.i >= 624 {
            for k in 0..624 {
                let y = (self.mt[k] & 0x8000_0000) | (self.mt[(k + 1) % 624] & 0x7fff_ffff);
                self.mt[k] = self.mt[(k + 397) % 624] ^ (y >> 1) ^ if y & 1 != 0 { 0x9908_b0df } else { 0 };
            }
            self.i = 0;
        }
        let mut y = self.mt[self.i];
        self.i += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c_5680;
        y ^= (y << 15) & 0xefc6_0000;
        y ^ (y >> 18)
    }
    /// libstdc++'s `std::uniform_real_distribution<float>(lo, hi)` over a 32-bit engine: `generate_canonical<float, 24>` takes one draw,
    /// `float(u) / 2^32` (capped below 1), then `v * (hi - lo) + lo`. For the bench's symmetric ranges `hi - lo` is a power of two, so
    /// the product is exact and fused or not makes no difference.
    fn uniform(&mut self, lo: f32, hi: f32) -> f32 {
        let mut v = self.next() as f32 / 4294967296.0f32;
        if v >= 1.0 {
            v = f32::from_bits(1.0f32.to_bits() - 1);
        }
        v * (hi - lo) + lo
    }
}

fn bf16_rne(f: f32) -> u16 {
    let b = f.to_bits();
    ((b + 0x7fff + ((b >> 16) & 1)) >> 16) as u16
}

fn fnv64(d: &[u8]) -> u64 {
    d.iter().fold(1469598103934665603u64, |h, b| (h ^ *b as u64).wrapping_mul(1099511628211))
}

fn bench_bf16(n: usize, scale: f32, seed: u32) -> HostTensor {
    let mut g = Mt19937::new(seed);
    let data: Vec<u8> = (0..n).flat_map(|_| bf16_rne(g.uniform(-scale, scale)).to_le_bytes()).collect();
    HostTensor::new(BF, vec![n], data).unwrap()
}

/// E5 (the interleaved four-row softmax of `attn_sparse_mma`) is bit-identical to the old per-row form: the five attention shapes of
/// fleetq #8282 (`docs/results/2026-11-03-dsv41-attn-tfblock/raw/bench/attn_ab_bench.cu`, same generators, seeds and shapes) must hash to
/// what the unmodified kernel gave there (`expected-hashes.txt`, all six base pairs agreed). The sweep above judges by tolerance, so it
/// cannot see a one-ulp change; this can. Control: `INTARSIA_DSV41_ATTN_SIMT=1` runs the SIMT kernel, whose bits differ, and must fail it.
#[test]
fn attn_sparse_mma_is_bitwise_the_8282_base_kernel() {
    let Some(cuda) = device() else { return };
    let (h, d, w) = (32usize, 512usize, 128usize);
    // (name, T, Nc, Kw, pos0, ratio, hash of the unmodified kernel)
    let cases: [(&str, usize, usize, usize, usize, usize, u64); 5] = [
        ("CSA T512 Nc1024 Kw512", 512, 1024, 512, 3072, 4, 0x4ee9a847ca24437f),
        ("CSA T2048 Nc1024 Kw512", 2048, 1024, 512, 2048, 4, 0xaa254a578fedfa25),
        ("HCA T512 Nc32 Kw32", 512, 32, 32, 3072, 128, 0x90ad1d7fcca0b866),
        ("CSA T37 Nc1024 Kw512", 37, 1024, 512, 4000, 4, 0x21b89af74c9af5a2),
        ("CSA T11 Nc1024 Kw512", 11, 1024, 512, 4000, 4, 0xe450cadc14aad612),
    ];
    let mut bad = Vec::new();
    for (name, t, nc, kw, pos0, ratio, want) in cases {
        let with_shape = |x: HostTensor, shape: &[usize]| HostTensor::new(x.dtype, shape.to_vec(), x.data).unwrap();
        let q = with_shape(bench_bf16(t * h * d, 1.0, 41), &[t, h, d]);
        let win = with_shape(bench_bf16((w + t) * d, 1.0, 42), &[w + t, d]);
        let cmp = with_shape(bench_bf16(nc * d, 1.0, 43), &[nc, d]);
        let mut g = Mt19937::new(44);
        let idx: Vec<i32> = (0..t * kw).map(|_| { let v = (g.next() as i64 % (nc + nc / 16) as i64) as i32; if v >= nc as i32 { -1 } else { v } }).collect();
        let mut gs = Mt19937::new(45);
        let sink: Vec<f32> = (0..h).map(|_| gs.uniform(-1.0, 1.0)).collect();
        let mut attrs = OpAttrs { window: w, pos0, scale: 1.0 / (d as f32).sqrt(), ..OpAttrs::default() };
        attrs.dsv.ratio = ratio;
        let call = HostCall {
            op: Op::AttnSparse,
            inputs: vec![q, win, cmp, HostTensor::from_i32(&[t, kw], &idx), HostTensor::from_f32(&[h], &sink)],
            outputs: vec![out(BF, &[t, h, d])],
            attrs,
        };
        let got = fnv64(&run_host(&*cuda, "attn_sparse", &call).unwrap()[0].data);
        println!("{} {name:<24} hash {got:016x} want {want:016x}", if got == want { "ok  " } else { "FAIL" });
        if got != want {
            bad.push(name);
        }
    }
    assert!(bad.is_empty(), "attn_sparse differs from the unmodified kernel's recorded bits on: {bad:?}");
}

/// Every lane kernel at the model's shapes against the cpu reference, judged by the kernel's own tolerance.
#[test]
fn real_shapes_against_the_cpu_reference() {
    let Some(cuda) = device() else { return };
    let cpu = CpuBackend::new();
    let kernels = cuda.kernels();
    let mut failed = Vec::new();
    for (t, pos, nk) in [(1usize, 4095usize, 16384usize), (3, 4094, 4096), (1, 131071, 131072)] {
        for (label, kernel, call) in cases(t, pos, nk) {
            if pos == 131071 && !(kernel.starts_with("indexer") || kernel == "attn_sparse") {
                continue;
            }
            let want = run_host(&cpu, &format!("ref_{}", call.op.as_str()), &call).unwrap();
            let got = run_host(&*cuda, kernel, &call).unwrap();
            let kern = kernels.iter().find(|k| k.name == kernel).unwrap();
            let (ok, ma, rl) = judge(kern, &got, &want);
            println!("{:<4} {label:<58} max_abs={ma:.3e} rel_l2={rl:.3e}", if ok { "ok" } else { "FAIL" });
            if !ok {
                failed.push(label);
            }
        }
    }
    assert!(failed.is_empty(), "failed: {failed:?}");
}

/// attn_sparse prefill rows (T > 16: one block per query and head group, the direct-write branch, no split merge) and the
/// split / direct boundary, at the model's head shape, against the cpu reference.
#[test]
fn attn_sparse_prefill_rows_against_the_cpu_reference() {
    let Some(cuda) = device() else { return };
    let cpu = CpuBackend::new();
    let kern = cuda.kernels().into_iter().find(|k| k.name == "attn_sparse").unwrap();
    let mut failed = Vec::new();
    for (t, pos0, nc, kc) in [(16usize, 300usize, 150usize, 64usize), (17, 300, 150, 64), (32, 0, 16, 16), (64, 1000, 520, 512)] {
        let mut rng = SplitMix64::new(0xa77 ^ t as u64);
        let r = &mut rng;
        let idx: Vec<i32> = (0..t * kc).map(|_| if r.next_u64() % 8 == 0 { -1 } else { (r.next_u64() % nc as u64) as i32 }).collect();
        let call = HostCall {
            op: Op::AttnSparse,
            inputs: vec![
                rand_t(r, BF, &[t, H, HD], 1.0),
                rand_t(r, BF, &[W + t, HD], 1.0),
                rand_t(r, BF, &[nc, HD], 1.0),
                HostTensor::from_i32(&[t, kc], &idx),
                rand_t(r, DType::F32, &[H], 1.0),
            ],
            outputs: vec![out(BF, &[t, H, HD])],
            attrs: OpAttrs { window: W, pos0, scale: (HD as f32).powf(-0.5), ..OpAttrs::default() },
        };
        let want = run_host(&cpu, "ref_attn_sparse", &call).unwrap();
        let got = run_host(&*cuda, "attn_sparse", &call).unwrap();
        let (ok, ma, rl) = judge(&kern, &got, &want);
        println!("{:<4} attn_sparse T={t} pos0={pos0} Nc={nc} Kc={kc}  max_abs={ma:.3e} rel_l2={rl:.3e}", if ok { "ok" } else { "FAIL" });
        if !ok {
            failed.push(t);
        }
    }
    assert!(failed.is_empty(), "failed T: {failed:?}");
}

/// The indexer's query tiling: a scratch that holds only `rows` score rows makes the launcher run the chunk tile by tile
/// (t0 > 0 in every kernel); the result must equal the cpu reference exactly.
#[test]
fn indexer_tiles_against_the_cpu_reference() {
    let Some(cuda) = device() else { return };
    let cpu = CpuBackend::new();
    let mut failed = Vec::new();
    for (t, nk, rows, ratio) in [(40usize, 4096usize, 7usize, 1usize), (33, 2048, 5, 2), (9, 1024, 1, 1)] {
        let pos0 = nk * ratio - t;
        let nb = nk.div_ceil(8);
        let mut rng = SplitMix64::new(0x711e ^ t as u64);
        let r = &mut rng;
        let q = rand_t(r, BF, &[t, HI, DI], 1.0);
        let k = rand_t(r, BF, &[nk, DI], 1.0);
        let w = rand_t(r, BF, &[t, HI], 1.0);
        let cand: Vec<u8> = (0..t * nb).map(|_| (r.next_u64() % 3 != 0) as u8).collect();
        let mut at = OpAttrs { pos0, scale: 0.01, ..OpAttrs::default() };
        (at.dsv.ratio, at.dsv.block) = (ratio, 8);
        let kk = TOPK.min(nk);
        for (kernel, call) in [
            (
                "indexer_blocks",
                HostCall {
                    op: Op::IndexerBlocks,
                    inputs: vec![q.clone(), k.clone(), w.clone()],
                    outputs: vec![out(DType::U8, &[t, nb])],
                    attrs: OpAttrs { top_k: 64, ..at.clone() },
                },
            ),
            (
                "indexer_topk",
                HostCall {
                    op: Op::IndexerTopk,
                    inputs: vec![q.clone(), k.clone(), w.clone(), HostTensor::new(DType::U8, vec![t, nb], cand.clone()).unwrap()],
                    outputs: vec![out(DType::I32, &[t, kk])],
                    attrs: OpAttrs { top_k: TOPK, ..at.clone() },
                },
            ),
        ] {
            let want = run_host(&cpu, &format!("ref_{}", call.op.as_str()), &call).unwrap();
            let (mut dc, _) = device_call(&*cuda, kernel, &call);
            let len = if kernel == "indexer_blocks" { nb } else { nk };
            let bytes = rows * len * 4;
            let buf = cuda.alloc(bytes, MemKind::Device).unwrap();
            dc.scratch = Some(TensorRef::new(buf, DType::U8, &[bytes]));
            cuda.launch(kernel, &dc, 0).unwrap();
            cuda.synchronize(0).unwrap();
            let o = &dc.outputs[0];
            let mut data = vec![0u8; o.nbytes()];
            cuda.download(&o.buf, o.offset, &mut data, 0).unwrap();
            cuda.synchronize(0).unwrap();
            let same = data == want[0].data;
            println!("{:<4} {kernel} T={t} Nk={nk} ratio={ratio} tile={rows} rows", if same { "ok" } else { "FAIL" });
            if !same {
                failed.push(format!("{kernel} T={t}"));
            }
        }
    }
    assert!(failed.is_empty(), "failed: {failed:?}");
}

/// indexer_topk's multi-block threshold select (T <= 16, Nk >= 16384) on every branch: candidates within the cap, the
/// fallback to the one-block select when the k-th key's histogram bin is too full (-inf fill-ins when fewer than top_k rows
/// are inside the candidates; all-equal scores), against the cpu reference exactly.
#[test]
fn indexer_select_paths_against_the_cpu_reference() {
    let Some(cuda) = device() else { return };
    let cpu = CpuBackend::new();
    let mut failed = Vec::new();
    for t in [1usize, 3] {
        for what in ["random", "few candidates", "all ties"] {
            let nk = 32768;
            let pos0 = nk - t;
            let nb = nk.div_ceil(8);
            let mut rng = SplitMix64::new(0x5e1 ^ t as u64 ^ what.len() as u64);
            let r = &mut rng;
            let q = if what == "all ties" { HostTensor::zeros(BF, &[t, HI, DI]) } else { rand_t(r, BF, &[t, HI, DI], 1.0) };
            let k = rand_t(r, BF, &[nk, DI], 1.0);
            let w = rand_t(r, BF, &[t, HI], 1.0);
            let cand = if what == "few candidates" {
                let c: Vec<u8> = (0..t * nb).map(|j| ((j % nb) % 2000 == 7) as u8).collect();
                HostTensor::new(DType::U8, vec![t, nb], c).unwrap()
            } else {
                HostTensor::zeros(DType::U8, &[0])
            };
            let mut at = OpAttrs { pos0, scale: 0.01, top_k: TOPK, ..OpAttrs::default() };
            (at.dsv.ratio, at.dsv.block) = (1, 8);
            let call = HostCall { op: Op::IndexerTopk, inputs: vec![q, k, w, cand], outputs: vec![out(DType::I32, &[t, TOPK])], attrs: at };
            let want = run_host(&cpu, "ref_indexer_topk", &call).unwrap();
            let got = run_host(&*cuda, "indexer_topk", &call).unwrap();
            let same = got[0].data == want[0].data;
            println!("{:<4} indexer_topk select T={t} Nk={nk} {what}", if same { "ok" } else { "FAIL" });
            if !same {
                failed.push(format!("T={t} {what}"));
            }
        }
    }
    assert!(failed.is_empty(), "failed: {failed:?}");
}

/// Upload a host call once (inputs, outputs, scratch) for repeated launches.
fn device_call(b: &dyn Backend, kernel: &str, hc: &HostCall) -> (OpCall, usize) {
    let put = |t: &HostTensor| {
        let buf = b.alloc(t.data.len().max(16), MemKind::Device).unwrap();
        b.upload(&buf, 0, &t.data, 0).unwrap();
        TensorRef::new(buf, t.dtype, &t.shape)
    };
    let inputs = hc.inputs.iter().map(put).collect();
    let outputs = hc.outputs.iter().map(|o| put(&o.init.clone().unwrap_or_else(|| HostTensor::zeros(o.dtype, &o.shape)))).collect();
    let mut call = OpCall::new(hc.op, inputs, outputs, hc.attrs.clone());
    call.check().unwrap();
    let k = b.kernels().into_iter().find(|k| k.name == kernel).unwrap();
    let need = k.scratch_bytes(&ShapeInfo::from_call(&call));
    if need > 0 {
        let buf = b.alloc(need.max(16), MemKind::Device).unwrap();
        call.scratch = Some(TensorRef::new(buf, DType::U8, &[need]));
    }
    (call, need)
}

fn time_us(b: &dyn Backend, kernel: &str, call: &OpCall, iters: usize) -> f64 {
    for _ in 0..3 {
        b.launch(kernel, call, 0).unwrap();
    }
    b.synchronize(0).unwrap();
    let t0 = Instant::now();
    for _ in 0..iters {
        b.launch(kernel, call, 0).unwrap();
    }
    b.synchronize(0).unwrap();
    t0.elapsed().as_secs_f64() * 1e6 / iters as f64
}

/// Decode (T = 1) at 4K and 128K context and a 4096-row prefill chunk: us per call per rank, and each kernel's workspace.
#[test]
#[ignore]
fn bench_decode_and_prefill() {
    let Some(cuda) = device() else { return };
    for (t, pos, nk, iters) in [(1usize, 4095usize, 4096usize, 200usize), (1, 131071, 131072, 50), (4096, 0, 4096, 3), (4096, 126976, 131072, 2)] {
        println!("--- T={t} pos0={pos} (indexer Nk={nk}, ratio 1)");
        for (label, kernel, call) in cases(t, pos, nk) {
            let (dc, scratch) = device_call(&*cuda, kernel, &call);
            let us = time_us(&*cuda, kernel, &dc, iters);
            println!("{label:<62} {us:>10.1} us   scratch {:>10.1} KiB", scratch as f64 / 1024.0);
        }
    }
}
