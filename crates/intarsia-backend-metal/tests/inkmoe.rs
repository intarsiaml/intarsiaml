//! Lane ink-m-moe: the Inkling-2-Mini experts (`moe_experts_situ_bf16[_rows]`, NVFP4 with SiTU; `moe_experts_mxfp8_bf16[_rows]`,
//! silu / gelu / situ). Claim: a T-row call is bitwise the T one-row calls, under every decode policy (union-grouped or
//! per-pair gate+up, fused down + token sum or by-pair down + gather), with 16-byte and 8-byte aligned weights and aligned
//! and odd-element activations. Admission checks the values against the f64 reference; this checks the rows-exact rule.
//!
//! * `ink_moe_rows_are_bitwise_the_one_row_calls`: T 2..=16, routings random / identical / pooled / invalid ids /
//!   duplicates, bf16 and f32 out, real H and I (2304, 1152) over 32 experts, top-8.
//! * `ink_moe_negative_control`: the compare is live (one changed activation of the last token changes its bytes only).
//!
//! Uses the GPU (about 1.5 GB): on endor take a fleetq slot first.

#![cfg(target_os = "macos")]

mod common;
use common::*;
use intarsia_backend_metal::inkmoe::{set_policy, Policy};
use intarsia_backend_metal::MetalBackend;
use intarsia_core::backend::{Backend, Buffer, MemKind, TensorRef};
use intarsia_core::quant::cases::{random_experts, random_routing};
use intarsia_core::rng::SplitMix64;
use intarsia_core::{DType, HostTensor, KernelImpl, Op, OpAttrs, OpCall, ShapeInfo};

const E: usize = 32;
const K: usize = 8;
const H: usize = 2304;
const I: usize = 1152;

fn put_bytes(be: &MetalBackend, off: usize, data: &[u8]) -> Buffer {
    let b = be.alloc(off + data.len() + 16, MemKind::Device).unwrap();
    be.upload(&b, 0, &vec![0xEEu8; off + data.len() + 16], 0).unwrap();
    be.upload(&b, off, data, 0).unwrap();
    b
}

fn put_host(be: &MetalBackend, h: &HostTensor, off: usize) -> TensorRef {
    let b = put_bytes(be, off, &h.data);
    TensorRef { buf: b, offset: off, dtype: h.dtype, shape: h.shape.clone() }
}

fn guarded(be: &MetalBackend, bytes: usize) -> Buffer {
    let b = be.alloc(bytes + 16, MemKind::Device).unwrap();
    be.upload(&b, 0, &vec![0xEEu8; bytes + 16], 0).unwrap();
    b
}

fn dl(be: &MetalBackend, b: &Buffer, n: usize) -> Vec<u8> {
    be.synchronize(0).unwrap();
    let mut raw = vec![0u8; n];
    be.download(b, 0, &mut raw, 0).unwrap();
    raw
}

fn differing(a: &[u8], b: &[u8]) -> usize {
    a.iter().zip(b).filter(|(x, y)| x != y).count()
}

#[derive(Clone, Copy, PartialEq, Debug)]
enum Fmt {
    Nv,
    Mx,
}

/// e4m3 bytes with exponents 4..=10, random sign and mantissa (never NaN), as inkling2::cases.
fn fp8(rng: &mut SplitMix64, shape: &[usize]) -> HostTensor {
    let n: usize = shape.iter().product();
    let data = (0..n)
        .map(|_| {
            let r = rng.next_u64();
            ((r & 0x80) as u8) | ((4 + (r >> 8) % 7) as u8) << 3 | ((r >> 16) & 7) as u8
        })
        .collect();
    HostTensor::new(DType::Fp8E4m3, shape.to_vec(), data).unwrap()
}

fn e8m0(rng: &mut SplitMix64, shape: &[usize]) -> HostTensor {
    let n: usize = shape.iter().product();
    HostTensor::new(DType::U8, shape.to_vec(), (0..n).map(|_| 116 + (rng.next_u64() % 4) as u8).collect()).unwrap()
}

fn weights(be: &MetalBackend, rng: &mut SplitMix64, f: Fmt, w_off: usize) -> Vec<TensorRef> {
    match f {
        Fmt::Nv => {
            let ex = random_experts(rng, E, H, I);
            vec![
                put_host(be, &ex.gate_up_w, w_off),
                put_host(be, &ex.gate_up_s, 0),
                put_host(be, &ex.down_w, w_off),
                put_host(be, &ex.down_s, 0),
                put_host(be, &ex.gscales, 0),
            ]
        }
        Fmt::Mx => vec![
            put_host(be, &fp8(rng, &[E, 2 * I, H]), w_off),
            put_host(be, &e8m0(rng, &[E, 2 * I, H / 32]), 1),
            put_host(be, &fp8(rng, &[E, H, I]), w_off),
            put_host(be, &e8m0(rng, &[E, H, I / 32]), 3),
        ],
    }
}

struct Rig {
    be: MetalBackend,
    solo: KernelImpl,
    rows: KernelImpl,
    f: Fmt,
}

impl Rig {
    fn new(f: Fmt) -> Self {
        let be = backend();
        let ks = be.kernels();
        let (s, r) = match f {
            Fmt::Nv => ("moe_experts_situ_bf16", "moe_experts_situ_bf16_rows"),
            Fmt::Mx => ("moe_experts_mxfp8_bf16", "moe_experts_mxfp8_bf16_rows"),
        };
        let solo = ks.iter().find(|k| k.name == s).unwrap().clone();
        let rows = ks.iter().find(|k| k.name == r).unwrap().clone();
        Rig { be, solo, rows, f }
    }

    /// An expert call of `n` tokens from row `r0`, writing `out` at row `r0`, with its scratch. `epi`: 0 silu, 1 gelu, 2 situ.
    #[allow(clippy::too_many_arguments)]
    fn call(
        &self,
        w: &[TensorRef],
        x: &TensorRef,
        ids: &TensorRef,
        ws: &TensorRef,
        out: &Buffer,
        odt: DType,
        r0: usize,
        n: usize,
        epi: u8,
        rows: bool,
    ) -> OpCall {
        let esz = odt.size_bytes();
        let mut ins = vec![x.rows(r0, n), ids.rows(r0, n), ws.rows(r0, n)];
        ins.extend(w.iter().cloned());
        let mut a = OpAttrs { top_k: K, gelu: epi == 1, ..OpAttrs::default() };
        if epi == 2 {
            (a.ink.situ, a.ink.gate_cap, a.ink.up_cap) = (true, 4.0, 25.0);
        }
        let op = if self.f == Fmt::Nv { Op::MoeExperts } else { Op::MoeExpertsMxfp8 };
        let mut c = OpCall::new(
            op,
            ins,
            vec![TensorRef { buf: out.clone(), offset: r0 * H * esz, dtype: odt, shape: vec![n, H] }],
            a,
        );
        c.check().unwrap();
        let kern = if rows { &self.rows } else { &self.solo };
        assert!(kern.accepts(&ShapeInfo::from_call(&c)), "{} does not accept the call", kern.name);
        let need = kern.scratch_bytes(&ShapeInfo::from_call(&c));
        c.scratch = Some(TensorRef::new(self.be.alloc(need.max(16), MemKind::Device).unwrap(), DType::U8, &[need]));
        c
    }

    fn name(&self, rows: bool) -> String {
        if rows {
            self.rows.name.clone()
        } else {
            self.solo.name.clone()
        }
    }
}

/// Each token picks K distinct experts out of a pool of `pool`.
fn pooled(rng: &mut SplitMix64, t: usize, pool: usize) -> Vec<i32> {
    let mut all: Vec<i32> = (0..E as i32).collect();
    for i in 0..pool {
        let j = i + (rng.next_u64() as usize) % (E - i);
        all.swap(i, j);
    }
    let mut ids = Vec::with_capacity(t * K);
    for _ in 0..t {
        let mut p = all[..pool].to_vec();
        for k in 0..K {
            let j = k + (rng.next_u64() as usize) % (pool - k);
            p.swap(k, j);
            ids.push(p[k]);
        }
    }
    ids
}

fn routings(rng: &mut SplitMix64, t: usize) -> Vec<(&'static str, Vec<i32>, Vec<f32>)> {
    let (ids, ws) = random_routing(rng, t, E, K);
    let (rids, rws) = (ids.to_i32().unwrap(), ws.to_f32().iter().map(|v| v * 8.0).collect::<Vec<f32>>());
    let mut out = vec![("random", rids.clone(), rws.clone())];
    out.push(("identical", (0..t).flat_map(|_| rids[..K].to_vec()).collect(), rws.clone()));
    out.push(("pool12", pooled(rng, t, 12), rws.clone()));
    let mut bad = rids.clone();
    for (i, v) in bad.iter_mut().enumerate() {
        match i % 11 {
            3 => *v = -1,
            7 => *v = E as i32,
            9 => *v = i32::MIN,
            _ => {}
        }
    }
    out.push(("invalid", bad, rws.clone()));
    out.push(("duplicates", (0..t * K).map(|i| if i / K % 2 == 0 { 5 } else { rids[i] }).collect(), rws));
    out
}

const POLICIES: [Policy; 5] = [Policy::Auto, Policy::Union, Policy::Pair, Policy::Fused, Policy::UnionFused];

fn rows_exact(f: Fmt) {
    let rig = Rig::new(f);
    let be = &rig.be;
    let mut rng = SplitMix64::new(0x1a4b_e0e0 ^ f as u64);
    let (mut cases, mut rows_cmp) = (0usize, 0usize);
    let epis: &[u8] = if f == Fmt::Nv { &[2] } else { &[0, 1, 2] };
    for w_off in [0usize, 8] {
        let w = weights(be, &mut rng, f, w_off);
        for (ti, &tr) in [2usize, 3, 4, 7, 16].iter().enumerate() {
            let mut xv = rng.normal_vec(tr * H, 1.0);
            for (i, v) in xv.iter_mut().enumerate() {
                match (i / H) % 3 {
                    1 => *v *= 1e-3,
                    2 => *v *= 30.0,
                    _ => {}
                }
            }
            // odd-element activations (the unaligned x loads) on every other T
            let x_off = if ti % 2 == 1 { 2 } else { 0 };
            let x = put_host(be, &HostTensor::from_f32_as(DType::Bf16, &[tr, H], &xv), x_off);
            for (name, ids, ws) in routings(&mut rng, tr) {
                let idt = put_host(be, &HostTensor::from_i32(&[tr, K], &ids), 0);
                let wst = put_host(be, &HostTensor::from_f32(&[tr, K], &ws), 0);
                for &epi in epis {
                    for odt in [DType::Bf16, DType::F32] {
                        let tot = tr * H * odt.size_bytes() + 16;
                        // the reference: T one-row calls under the default policy
                        set_policy(Policy::Auto);
                        let ys = guarded(be, tot - 16);
                        for r in 0..tr {
                            be.launch(&rig.name(false), &rig.call(&w, &x, &idt, &wst, &ys, odt, r, 1, epi, false), 0)
                                .unwrap();
                        }
                        let a = dl(be, &ys, tot);
                        assert!(a[..tot - 16].iter().any(|&v| v != 0 && v != 0xEE), "output looks unwritten");
                        for pol in POLICIES {
                            set_policy(pol);
                            for rows in [true, false] {
                                let yb = guarded(be, tot - 16);
                                be.launch(
                                    &rig.name(rows),
                                    &rig.call(&w, &x, &idt, &wst, &yb, odt, 0, tr, epi, rows),
                                    0,
                                )
                                .unwrap();
                                let b = dl(be, &yb, tot);
                                let what = format!("{f:?} T={tr} routing={name} epi={epi} out={odt} w_off={w_off} x_off={x_off} {pol:?} rows={rows}");
                                assert_eq!(differing(&a, &b), 0, "T-row call vs one-row calls, {what}");
                                cases += 1;
                            }
                            // one-row calls under this policy too
                            let yb = guarded(be, tot - 16);
                            for r in 0..tr {
                                be.launch(
                                    &rig.name(false),
                                    &rig.call(&w, &x, &idt, &wst, &yb, odt, r, 1, epi, false),
                                    0,
                                )
                                .unwrap();
                            }
                            assert_eq!(differing(&a, &dl(be, &yb, tot)), 0, "{f:?} one-row calls under {pol:?}");
                        }
                        rows_cmp += tr;
                    }
                }
            }
        }
    }
    set_policy(Policy::Auto);
    eprintln!("INKMOE {f:?} bitwise: {cases} multi-row calls over {rows_cmp} token rows equal the one-row calls");
}

#[test]
fn ink_moe_rows_are_bitwise_the_one_row_calls() {
    rows_exact(Fmt::Nv);
    rows_exact(Fmt::Mx);
}

/// The compare can see a difference: one changed activation of the last token changes its output bytes, and only those.
#[test]
fn ink_moe_negative_control() {
    for f in [Fmt::Nv, Fmt::Mx] {
        let rig = Rig::new(f);
        let be = &rig.be;
        let mut rng = SplitMix64::new(0x0e9a_1a4b);
        let w = weights(be, &mut rng, f, 0);
        for tr in [2usize, 16] {
            let xv = rng.normal_vec(tr * H, 1.0);
            let mut xp = xv.clone();
            xp[(tr - 1) * H + 37] += 0.5;
            let x = put_host(be, &HostTensor::from_f32_as(DType::Bf16, &[tr, H], &xv), 0);
            let xq = put_host(be, &HostTensor::from_f32_as(DType::Bf16, &[tr, H], &xp), 0);
            let ids = pooled(&mut rng, tr, 12);
            let ws: Vec<f32> = (0..tr * K).map(|i| 0.5 + (i % 7) as f32 * 0.2).collect();
            let idt = put_host(be, &HostTensor::from_i32(&[tr, K], &ids), 0);
            let wst = put_host(be, &HostTensor::from_f32(&[tr, K], &ws), 0);
            let tot = tr * H * 2 + 16;
            let (ya, yb) = (guarded(be, tot - 16), guarded(be, tot - 16));
            be.launch(&rig.name(true), &rig.call(&w, &x, &idt, &wst, &ya, DType::Bf16, 0, tr, 2, true), 0).unwrap();
            be.launch(&rig.name(true), &rig.call(&w, &xq, &idt, &wst, &yb, DType::Bf16, 0, tr, 2, true), 0).unwrap();
            let (a, b) = (dl(be, &ya, tot), dl(be, &yb, tot));
            let first = a.iter().zip(&b).position(|(p, q)| p != q).unwrap_or(usize::MAX);
            let d = differing(&a, &b);
            let amax = a[..tot - 16]
                .chunks_exact(2)
                .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16).abs())
                .fold(0f32, f32::max);
            eprintln!(
                "INKMOE {f:?} negctl T={tr}: {d} bytes differ, first at {first} (last token at {}), max |out| {amax}",
                (tr - 1) * H * 2
            );
            assert!(d > 0, "{f:?} T={tr}: the changed activation changed nothing");
            assert!(first >= (tr - 1) * H * 2, "{f:?} T={tr}: a byte before the last token changed");
        }
    }
}
