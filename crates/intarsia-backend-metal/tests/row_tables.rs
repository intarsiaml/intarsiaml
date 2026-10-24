//! G-B for the row-table kernels of batched decode on Metal (lane mr3, `OpCall::rows`): row r of one row-table launch is BITWISE
//! equal to the one-row (T = 1) launch of the solo decode step on sequence r alone, for `gdn_conv_bf16_rows`,
//! `gdn_delta_rule_bf16_rows`, `kv_append_bf16_rows`, `qk_norm_rope_kv_bf16_rows`, `attn_decode_bf16_splitkv_rows`, and `rope_bf16`
//! with one position per token. Every row count 1..=16, the Qwen3.6-35B-A3B and Qwen3-8B shapes (and the dk = 64 delta rule),
//! states, caches and positions that differ per row. States and caches are compared as bytes after the launch (the in-place update
//! is part of the contract); no tolerance.
//!
//! Negative controls (each must differ, or the test cannot see what it claims): attention rows at the other simdgroup rule (the
//! solo launch with a DEVICE position always uses 4 simdgroups where the host-position launch uses 8 up to 1024 keys), and two
//! rows' table entries swapped.
//!
//! Synthetic, small allocations: no model needed, but take a fleetq slot on endor first.
#![cfg(target_os = "macos")]

use intarsia_backend_metal::MetalBackend;
use intarsia_core::backend::{Backend, MemKind, TensorRef};
use intarsia_core::ops::RowTab;
use intarsia_core::rng::SplitMix64;
use intarsia_core::{DType, HostTensor, Op, OpAttrs, OpCall, ShapeInfo};

const CAP: usize = 1200;

struct Rig {
    be: MetalBackend,
}

impl Rig {
    fn new() -> Rig {
        Rig { be: MetalBackend::new().expect("Metal backend") }
    }

    fn put_bytes(&self, data: &[u8], dtype: DType, shape: &[usize]) -> TensorRef {
        let buf = self.be.alloc(data.len().max(16), MemKind::Device).unwrap();
        self.be.upload(&buf, 0, &vec![0u8; data.len().max(16)], 0).unwrap();
        if !data.is_empty() {
            self.be.upload(&buf, 0, data, 0).unwrap();
        }
        TensorRef::new(buf, dtype, shape)
    }

    fn put(&self, t: &HostTensor) -> TensorRef {
        self.put_bytes(&t.data, t.dtype, &t.shape)
    }

    fn bf16(&self, vals: &[f32], shape: &[usize]) -> TensorRef {
        self.put(&HostTensor::from_f32_as(DType::Bf16, shape, vals))
    }

    fn f32s(&self, vals: &[f32], shape: &[usize]) -> TensorRef {
        self.put(&HostTensor::from_f32_as(DType::F32, shape, vals))
    }

    fn i32s(&self, vals: &[usize]) -> TensorRef {
        let b: Vec<u8> = vals.iter().flat_map(|p| (*p as i32).to_le_bytes()).collect();
        self.put_bytes(&b, DType::I32, &[vals.len()])
    }

    /// An output buffer of `bytes` filled with 0xA5 (a NaN in bf16 / f32).
    fn poison(&self, bytes: usize, dtype: DType, shape: &[usize]) -> TensorRef {
        self.put_bytes(&vec![0xA5u8; bytes], dtype, shape)
    }

    fn down(&self, t: &TensorRef) -> Vec<u8> {
        self.be.synchronize(0).unwrap();
        let mut d = vec![0u8; t.nbytes()];
        self.be.download(&t.buf, t.offset, &mut d, 0).unwrap();
        d
    }

    fn launch(&self, kernel: &str, mut call: OpCall) {
        call.check().unwrap_or_else(|e| panic!("{kernel}: check: {e}"));
        let k = self.be.kernels().into_iter().find(|k| k.name == kernel && k.op == call.op).unwrap_or_else(|| panic!("{kernel} not registered"));
        let need = k.scratch_bytes(&ShapeInfo::from_call(&call));
        if need > 0 {
            let buf = self.be.alloc(need.max(16), MemKind::Device).unwrap();
            self.be.upload(&buf, 0, &vec![0u8; need.max(16)], 0).unwrap();
            call.scratch = Some(TensorRef::new(buf, DType::U8, &[need]));
        }
        self.be.launch(kernel, &call, 0).unwrap_or_else(|e| panic!("{kernel}: {e}"));
        self.be.synchronize(0).unwrap();
    }

    /// The row table of `rows` rows: `entries[r]` are the buffers of row r (one per column), `stride` columns per row.
    fn table(&self, entries: &[Vec<&TensorRef>], stride: usize) -> TensorRef {
        let mut bytes = vec![0u8; entries.len() * stride * 8];
        for (r, row) in entries.iter().enumerate() {
            for (j, t) in row.iter().enumerate() {
                let a = self.be.device_address(&t.buf).expect("device address");
                bytes[(r * stride + j) * 8..(r * stride + j) * 8 + 8].copy_from_slice(&a.to_le_bytes());
            }
        }
        self.put_bytes(&bytes, DType::U8, &[entries.len(), stride, 8])
    }

    fn tab(&self, ptrs: TensorRef, index: usize, byte_off: usize) -> RowTab {
        let (rows, stride) = (ptrs.shape[0], ptrs.shape[1]);
        RowTab { ptrs, rows, stride, index, byte_off }
    }
}

/// A decoy for the unused column of a table: a buffer of `like`'s size filled with 0xA5 (NaN as bf16 / f32), which the kernel must
/// neither read (a NaN would reach the output) nor write; `still_poison` checks the second.
fn decoy(rig: &Rig, like: &TensorRef) -> TensorRef {
    rig.poison(like.nbytes(), DType::U8, &[like.nbytes()])
}

fn still_poison(rig: &Rig, d: &TensorRef, what: &str) -> usize {
    let b = rig.down(d);
    assert!(b.iter().all(|&x| x == 0xA5), "{what}: the decoy column buffer was written");
    b.len()
}

fn rows_of(t: &TensorRef, r: usize) -> TensorRef {
    t.rows(r, 1)
}

/// Random-with-spread activations: mostly normal, with a few large and tiny magnitudes so that sums round.
fn nasty(rng: &mut SplitMix64, n: usize, scale: f32) -> Vec<f32> {
    rng.normal_vec(n, scale)
        .into_iter()
        .enumerate()
        .map(|(i, v)| match i % 17 {
            0 => v * 64.0,
            5 => v * 1e-3,
            _ => v,
        })
        .collect()
}

/// Number of bytes compared; panics on the first difference.
fn same(what: &str, got: &[u8], want: &[u8]) -> usize {
    assert_eq!(got.len(), want.len(), "{what}: length");
    if let Some(i) = got.iter().zip(want).position(|(a, b)| a != b) {
        panic!("{what}: first differing byte {i} of {} (row-table launch {:#04x} vs solo {:#04x})", got.len(), got[i], want[i]);
    }
    got.len()
}

fn differs(got: &[u8], want: &[u8]) -> bool {
    got.iter().zip(want).any(|(a, b)| a != b)
}

#[test]
fn gdn_conv_rows_are_the_solo_conv() {
    let rig = Rig::new();
    let mut rng = SplitMix64::new(11);
    let mut total = 0;
    for (c, ks) in [(8192usize, 4usize), (6144, 4), (130, 3), (257, 8)] {
        for m in 1..=16 {
            let x = nasty(&mut rng, m * c, 1.0);
            let w = rng.normal_vec(c * ks, 0.5);
            let states: Vec<Vec<f32>> = (0..m).map(|_| nasty(&mut rng, c * (ks - 1), 1.0)).collect();
            let (xt, wt) = (rig.bf16(&x, &[m, c]), rig.bf16(&w, &[c, ks]));
            let mut want_y = Vec::new();
            let mut want_s = Vec::new();
            for r in 0..m {
                let st = rig.f32s(&states[r], &[c, ks - 1]);
                let y = rig.poison(c * 2, DType::Bf16, &[1, c]);
                rig.launch(
                    "gdn_conv_bf16",
                    OpCall::new(Op::GdnConv, vec![rows_of(&xt, r), wt.clone()], vec![y.clone(), st.clone()], OpAttrs::default()),
                );
                want_y.extend(rig.down(&y));
                want_s.push(rig.down(&st));
            }
            let sts: Vec<TensorRef> = (0..m).map(|r| rig.f32s(&states[r], &[c, ks - 1])).collect();
            let decoys: Vec<TensorRef> = sts.iter().map(|s| decoy(&rig, s)).collect();
            let entries: Vec<Vec<&TensorRef>> = sts.iter().zip(&decoys).map(|(s, d)| vec![d, s]).collect();
            let ptrs = rig.table(&entries, 2);
            let y = rig.poison(m * c * 2, DType::Bf16, &[m, c]);
            let mut call = OpCall::new(Op::GdnConv, vec![xt.clone(), wt.clone()], vec![y.clone(), sts[0].clone()], OpAttrs::default());
            call.rows = Some(rig.tab(ptrs, 1, 0));
            rig.launch("gdn_conv_bf16_rows", call);
            total += same(&format!("conv c={c} k={ks} m={m} y"), &rig.down(&y), &want_y);
            for r in 0..m {
                total += same(&format!("conv c={c} k={ks} m={m} state {r}"), &rig.down(&sts[r]), &want_s[r]);
                total += still_poison(&rig, &decoys[r], &format!("conv c={c} k={ks} m={m} row {r}"));
            }
        }
    }
    eprintln!("G-B gdn_conv_bf16_rows: {total} bytes bitwise equal (M 1..=16, 4 shapes)");
}

#[test]
fn gdn_delta_rows_are_the_solo_delta_rule() {
    let rig = Rig::new();
    let mut rng = SplitMix64::new(12);
    let mut total = 0;
    // (hk, hv, dk, dv): the 35B's 16 / 32 / 128 / 128, dk = 64, and a small GQA-free shape
    for (hk, hv, dk, dv) in [(16usize, 32usize, 128usize, 128usize), (4, 8, 64, 64), (2, 2, 32, 32)] {
        let c = 2 * hk * dk + hv * dv;
        for m in 1..=16 {
            let qkv = nasty(&mut rng, m * c, 1.0);
            let a = rng.normal_vec(m * hv, 1.0);
            let b = rng.normal_vec(m * hv, 1.0);
            let a_log = rng.normal_vec(hv, 0.5);
            let dt = rng.normal_vec(hv, 0.5);
            let (qt, at, bt, alt, dtt) = (
                rig.bf16(&qkv, &[m, c]),
                rig.bf16(&a, &[m, hv]),
                rig.bf16(&b, &[m, hv]),
                rig.bf16(&a_log, &[hv]),
                rig.bf16(&dt, &[hv]),
            );
            // each row's recurrent buffer: 256 bytes of "conv state" in front, then the f32 delta state (the real layout)
            let off = 256usize;
            let init: Vec<Vec<u8>> = (0..m)
                .map(|_| {
                    let mut v = vec![0x11u8; off];
                    v.extend(HostTensor::from_f32_as(DType::F32, &[hv * dv * dk], &rng.normal_vec(hv * dv * dk, 0.3)).data);
                    v
                })
                .collect();
            let attrs = OpAttrs { eps: 1e-6, ..Default::default() };
            let view = |t: &TensorRef| {
                let mut v = TensorRef::new(t.buf.clone(), DType::F32, &[hv, dv, dk]);
                v.offset = off;
                v
            };
            let mut want_o = Vec::new();
            let mut want_s = Vec::new();
            for r in 0..m {
                let st = rig.put_bytes(&init[r], DType::U8, &[init[r].len()]);
                let o = rig.poison(hv * dv * 2, DType::Bf16, &[1, hv, dv]);
                rig.launch(
                    "gdn_delta_rule_bf16",
                    OpCall::new(
                        Op::GdnDeltaRule,
                        vec![rows_of(&qt, r), rows_of(&at, r), rows_of(&bt, r), alt.clone(), dtt.clone()],
                        vec![o.clone(), view(&st)],
                        attrs.clone(),
                    ),
                );
                want_o.extend(rig.down(&o));
                want_s.push(rig.down(&st));
            }
            let sts: Vec<TensorRef> = (0..m).map(|r| rig.put_bytes(&init[r], DType::U8, &[init[r].len()])).collect();
            let decoys: Vec<TensorRef> = sts.iter().map(|s| decoy(&rig, s)).collect();
            let entries: Vec<Vec<&TensorRef>> = sts.iter().zip(&decoys).map(|(s, d)| vec![s, d]).collect();
            let ptrs = rig.table(&entries, 2);
            let o = rig.poison(m * hv * dv * 2, DType::Bf16, &[m, hv, dv]);
            let mut call = OpCall::new(
                Op::GdnDeltaRule,
                vec![qt.clone(), at.clone(), bt.clone(), alt.clone(), dtt.clone()],
                vec![o.clone(), view(&sts[0])],
                attrs.clone(),
            );
            call.rows = Some(rig.tab(ptrs, 0, off));
            rig.launch("gdn_delta_rule_bf16_rows", call);
            total += same(&format!("delta dk={dk} m={m} out"), &rig.down(&o), &want_o);
            for r in 0..m {
                total += same(&format!("delta dk={dk} m={m} state {r}"), &rig.down(&sts[r]), &want_s[r]);
                total += still_poison(&rig, &decoys[r], &format!("delta dk={dk} m={m} row {r}"));
            }
        }
    }
    eprintln!("G-B gdn_delta_rule_bf16_rows: {total} bytes bitwise equal (M 1..=16, dk 128 / 64 / 32)");
}

#[test]
fn kv_append_rows_are_the_solo_append() {
    let rig = Rig::new();
    let mut rng = SplitMix64::new(13);
    let mut total = 0;
    for (nkv, d) in [(2usize, 256usize), (8, 128), (1, 8)] {
        for m in 1..=16 {
            let newkv = nasty(&mut rng, m * nkv * d, 1.0);
            let nt = rig.bf16(&newkv, &[m, nkv, d]);
            // positions: the start, a middle, the last slot, one past the cache (dropped), and the rest spread
            let pos: Vec<usize> = (0..m).map(|r| [0usize, 1, 63, 64, 79, CAP - 1, 40, CAP, 17, 900, 2, 511, 512, 1023, 1024, 5][r % 16]).collect();
            let init: Vec<Vec<f32>> = (0..m).map(|_| nasty(&mut rng, nkv * CAP * d, 1.0)).collect();
            let mut want = Vec::new();
            for r in 0..m {
                let cache = rig.bf16(&init[r], &[nkv, CAP, d]);
                let attrs = OpAttrs { pos0: pos[r].min(CAP - 1), kv_cap: CAP, n_kv_heads: nkv, head_dim: d, ..Default::default() };
                // a position past the cache is not a valid host call: the solo step never makes one; skip the solo launch
                if pos[r] < CAP {
                    rig.launch("kv_append_bf16", OpCall::new(Op::KvAppend, vec![rows_of(&nt, r)], vec![cache.clone()], attrs));
                }
                want.push(rig.down(&cache));
            }
            let caches: Vec<TensorRef> = (0..m).map(|r| rig.bf16(&init[r], &[nkv, CAP, d])).collect();
            let decoys: Vec<TensorRef> = caches.iter().map(|s| decoy(&rig, s)).collect();
            let entries: Vec<Vec<&TensorRef>> = caches.iter().zip(&decoys).map(|(s, d)| vec![d, s]).collect();
            let ptrs = rig.table(&entries, 2);
            let post = rig.i32s(&pos);
            let attrs = OpAttrs { kv_cap: CAP, n_kv_heads: nkv, head_dim: d, ..Default::default() };
            let mut call = OpCall::new(Op::KvAppend, vec![nt.clone()], vec![caches[0].clone()], attrs);
            call.rows = Some(rig.tab(ptrs, 1, 0));
            call.pos = Some(post);
            rig.launch("kv_append_bf16_rows", call);
            for r in 0..m {
                total += same(&format!("kv_append nkv={nkv} d={d} m={m} cache {r}"), &rig.down(&caches[r]), &want[r]);
                total += still_poison(&rig, &decoys[r], &format!("kv_append nkv={nkv} d={d} m={m} row {r}"));
            }
        }
    }
    eprintln!("G-B kv_append_bf16_rows: {total} bytes bitwise equal (M 1..=16, 3 shapes, includes a dropped out-of-range position)");
}

#[test]
fn qk_norm_rope_kv_rows_are_the_solo_prologue() {
    let rig = Rig::new();
    let mut rng = SplitMix64::new(14);
    let mut total = 0;
    // (n, nkv, d, rd): the 35B's gated attention (16 q heads, 2 kv heads of 256, partial rotary 64), Qwen3-8B (32, 8, 128, 128)
    for (n, nkv, d, rd) in [(16usize, 2usize, 256usize, 64usize), (32, 8, 128, 128), (4, 2, 8, 8)] {
        for m in 1..=16 {
            let q = nasty(&mut rng, m * n * d, 1.0);
            let k = nasty(&mut rng, m * nkv * d, 1.0);
            let v = nasty(&mut rng, m * nkv * d, 1.0);
            let (qw, kw) = (rng.normal_vec(d, 0.3), rng.normal_vec(d, 0.3));
            let (qt, kt, vt) = (rig.bf16(&q, &[m, n, d]), rig.bf16(&k, &[m, nkv, d]), rig.bf16(&v, &[m, nkv, d]));
            let (qwt, kwt) = (rig.bf16(&qw, &[d]), rig.bf16(&kw, &[d]));
            let pos: Vec<usize> = (0..m).map(|r| [0usize, 1, 977, 31, 1023, 1024, 7, 500, 1199, 63, 64, 65, 2, 300, 900, 128][(r * 7 + m) % 16]).collect();
            let init: Vec<(Vec<f32>, Vec<f32>)> = (0..m).map(|_| (nasty(&mut rng, nkv * CAP * d, 1.0), nasty(&mut rng, nkv * CAP * d, 1.0))).collect();
            let attrs = |p: usize| OpAttrs {
                eps: 1e-6,
                w_offset: 1.0,
                pos0: p,
                theta: 1.0e7,
                head_dim: d,
                rotary_dim: rd,
                n_heads: n,
                n_kv_heads: nkv,
                kv_cap: CAP,
                ..Default::default()
            };
            let mut want_q = Vec::new();
            let mut want_c = Vec::new();
            for r in 0..m {
                let (kc, vc) = (rig.bf16(&init[r].0, &[nkv, CAP, d]), rig.bf16(&init[r].1, &[nkv, CAP, d]));
                let qo = rig.poison(n * d * 2, DType::Bf16, &[1, n, d]);
                rig.launch(
                    "qk_norm_rope_kv_bf16",
                    OpCall::new(
                        Op::QkNormRopeKv,
                        vec![rows_of(&qt, r), rows_of(&kt, r), rows_of(&vt, r), qwt.clone(), kwt.clone()],
                        vec![qo.clone(), kc.clone(), vc.clone()],
                        attrs(pos[r]),
                    ),
                );
                want_q.extend(rig.down(&qo));
                want_c.push((rig.down(&kc), rig.down(&vc)));
            }
            let kcs: Vec<TensorRef> = (0..m).map(|r| rig.bf16(&init[r].0, &[nkv, CAP, d])).collect();
            let vcs: Vec<TensorRef> = (0..m).map(|r| rig.bf16(&init[r].1, &[nkv, CAP, d])).collect();
            let entries: Vec<Vec<&TensorRef>> = (0..m).map(|r| vec![&kcs[r], &vcs[r]]).collect();
            let ptrs = rig.table(&entries, 2);
            let post = rig.i32s(&pos);
            let qo = rig.poison(m * n * d * 2, DType::Bf16, &[m, n, d]);
            let mut call = OpCall::new(
                Op::QkNormRopeKv,
                vec![qt.clone(), kt.clone(), vt.clone(), qwt.clone(), kwt.clone()],
                vec![qo.clone(), kcs[0].clone(), vcs[0].clone()],
                attrs(0),
            );
            call.rows = Some(rig.tab(ptrs, 0, 0));
            call.pos = Some(post);
            rig.launch("qk_norm_rope_kv_bf16_rows", call);
            total += same(&format!("prologue n={n} d={d} m={m} q"), &rig.down(&qo), &want_q);
            for r in 0..m {
                total += same(&format!("prologue n={n} d={d} m={m} k cache {r}"), &rig.down(&kcs[r]), &want_c[r].0);
                total += same(&format!("prologue n={n} d={d} m={m} v cache {r}"), &rig.down(&vcs[r]), &want_c[r].1);
            }
        }
    }
    eprintln!("G-B qk_norm_rope_kv_bf16_rows: {total} bytes bitwise equal (M 1..=16, 35B / 8B / tiny shapes)");
}

/// The attention rows with the given (nh, nkv, d), positions per row; returns (row-table output, solo-host-position outputs,
/// solo-device-position outputs).
fn attn_case(rig: &Rig, rng: &mut SplitMix64, nh: usize, nkv: usize, d: usize, pos: &[usize], cancel: bool) -> (Vec<u8>, Vec<u8>, Vec<u8>, usize) {
    let m = pos.len();
    // `cancel`: q = 0 (every key weighs the same) and V alternating +-1e6 per key with unit noise: the output is a sum with
    // heavy cancellation, so the accumulation order (the simdgroup count) shows in the bf16 result
    let q = if cancel { vec![0.0; m * nh * d] } else { nasty(rng, m * nh * d, 1.0) };
    let qt = rig.bf16(&q, &[m, nh, d]);
    let kv: Vec<(Vec<f32>, Vec<f32>)> = (0..m)
        .map(|_| {
            let k = nasty(rng, nkv * CAP * d, 1.0);
            let v = if cancel {
                // key k mod 3: +2^20, -2^20, or a small value: the bigs cancel exactly when the visible count is a multiple of
                // 3, and every small value is rounded at the granularity of the running big sum, whatever the order
                let noise = rng.normal_vec(nkv * CAP * d, 0.1);
                noise.iter().enumerate().map(|(i, x)| match (i / d) % 3 { 0 => 1048576.0, 1 => -1048576.0, _ => *x }).collect()
            } else {
                nasty(rng, nkv * CAP * d, 1.0)
            };
            (k, v)
        })
        .collect();
    let scale = 1.0 / (d as f32).sqrt();
    let mk = |p: usize| OpAttrs { pos0: p, scale, n_heads: nh, n_kv_heads: nkv, head_dim: d, kv_cap: CAP, ..Default::default() };
    let mut want_host = Vec::new();
    let mut want_dev = Vec::new();
    for r in 0..m {
        let (kc, vc) = (rig.bf16(&kv[r].0, &[nkv, CAP, d]), rig.bf16(&kv[r].1, &[nkv, CAP, d]));
        let o = rig.poison(nh * d * 2, DType::Bf16, &[1, nh, d]);
        rig.launch(
            "attn_decode_bf16_splitkv",
            OpCall::new(Op::AttnDecode, vec![rows_of(&qt, r), kc.clone(), vc.clone()], vec![o.clone()], mk(pos[r])),
        );
        want_host.extend(rig.down(&o));
        // the same row with a DEVICE position: the other simdgroup rule (negative control)
        let od = rig.poison(nh * d * 2, DType::Bf16, &[1, nh, d]);
        let mut call = OpCall::new(Op::AttnDecode, vec![rows_of(&qt, r), kc, vc], vec![od.clone()], mk(0));
        call.pos = Some(rig.i32s(&[pos[r]]));
        rig.launch("attn_decode_bf16_splitkv", call);
        want_dev.extend(rig.down(&od));
    }
    let kcs: Vec<TensorRef> = (0..m).map(|r| rig.bf16(&kv[r].0, &[nkv, CAP, d])).collect();
    let vcs: Vec<TensorRef> = (0..m).map(|r| rig.bf16(&kv[r].1, &[nkv, CAP, d])).collect();
    let entries: Vec<Vec<&TensorRef>> = (0..m).map(|r| vec![&kcs[r], &vcs[r]]).collect();
    let ptrs = rig.table(&entries, 2);
    let o = rig.poison(m * nh * d * 2, DType::Bf16, &[m, nh, d]);
    let mut call = OpCall::new(Op::AttnDecode, vec![qt.clone(), kcs[0].clone(), vcs[0].clone()], vec![o.clone()], mk(0));
    call.rows = Some(rig.tab(ptrs, 0, 0));
    call.pos = Some(rig.i32s(pos));
    rig.launch("attn_decode_bf16_splitkv_rows", call);
    let got = rig.down(&o);
    let n = got.len();
    (got, want_host, want_dev, n)
}

#[test]
fn attn_decode_rows_are_the_solo_decode() {
    let rig = Rig::new();
    let mut rng = SplitMix64::new(15);
    let mut total = 0;
    // positions cover: 0, one split, many splits, the 1024 / 1025 key boundary of the simdgroup rule (host position: 8
    // simdgroups while Nkv * nsplit <= 64), the end of the cache
    let ps = [0usize, 1, 63, 64, 65, 255, 511, 700, 1023, 1024, 1199, 128, 2, 300, 900, 31];
    let mut dev_differs = 0usize;
    for (nh, nkv, d) in [(16usize, 2usize, 256usize), (32, 8, 128), (8, 2, 64)] {
        for m in 1..=16 {
            let pos: Vec<usize> = (0..m).map(|r| ps[(r * 7 + m) % ps.len()]).collect();
            let (got, want, dev, _) = attn_case(&rig, &mut rng, nh, nkv, d, &pos, false);
            total += same(&format!("attn nh={nh} nkv={nkv} d={d} m={m} pos={pos:?}"), &got, &want);
            if differs(&got, &dev) {
                dev_differs += 1;
            }
        }
    }
    // the boundary rows in one launch, on cancelling data (the negative control: the host-position rows match the host-position
    // solo launches bit for bit, and the device-position launch, which has 4 simdgroups where the host rule has 8 up to 1024
    // keys, does not)
    for (nh, nkv, d) in [(16usize, 2usize, 256usize), (32, 8, 128)] {
        let pos = [1022usize, 1019, 1001, 998, 599, 299, 98, 1166, 1022, 2];
        let (got, want, dev, _) = attn_case(&rig, &mut rng, nh, nkv, d, &pos, true);
        total += same(&format!("attn cancelling data nh={nh} boundary rows"), &got, &want);
        let n_diff = got.iter().zip(&dev).filter(|(a, b)| a != b).count();
        eprintln!("negative control nh={nh}: {n_diff} of {} output bytes differ from the device-position launch", got.len());
        dev_differs += (n_diff > 0) as usize;
    }
    assert!(dev_differs > 0, "negative control: no case differed from the device-position launch");
    eprintln!("G-B attn_decode_bf16_splitkv_rows: {total} bytes bitwise equal (M 1..=16, 3 shapes; negative control: {dev_differs} cases differ from the device-position launch)");
}

#[test]
fn a_swapped_table_entry_is_seen() {
    // negative control for addressing: rows 0 and 1 swap their caches in the table, so row 0 attends row 1's cache
    let rig = Rig::new();
    let mut rng = SplitMix64::new(16);
    let (nh, nkv, d, m) = (16usize, 2usize, 256usize, 2usize);
    let pos = [700usize, 701];
    let q = nasty(&mut rng, m * nh * d, 1.0);
    let qt = rig.bf16(&q, &[m, nh, d]);
    let kv: Vec<(Vec<f32>, Vec<f32>)> = (0..m).map(|_| (nasty(&mut rng, nkv * CAP * d, 1.0), nasty(&mut rng, nkv * CAP * d, 1.0))).collect();
    let kcs: Vec<TensorRef> = (0..m).map(|r| rig.bf16(&kv[r].0, &[nkv, CAP, d])).collect();
    let vcs: Vec<TensorRef> = (0..m).map(|r| rig.bf16(&kv[r].1, &[nkv, CAP, d])).collect();
    let run = |order: [usize; 2]| {
        let entries: Vec<Vec<&TensorRef>> = order.iter().map(|&r| vec![&kcs[r], &vcs[r]]).collect();
        let ptrs = rig.table(&entries, 2);
        let o = rig.poison(m * nh * d * 2, DType::Bf16, &[m, nh, d]);
        let attrs = OpAttrs { scale: 1.0 / (d as f32).sqrt(), n_heads: nh, n_kv_heads: nkv, head_dim: d, kv_cap: CAP, ..Default::default() };
        let mut call = OpCall::new(Op::AttnDecode, vec![qt.clone(), kcs[0].clone(), vcs[0].clone()], vec![o.clone()], attrs);
        call.rows = Some(rig.tab(ptrs, 0, 0));
        call.pos = Some(rig.i32s(&pos));
        rig.launch("attn_decode_bf16_splitkv_rows", call);
        rig.down(&o)
    };
    assert!(differs(&run([0, 1]), &run([1, 0])), "negative control: swapping the caches in the table changed nothing");
    eprintln!("G-B negative control: a swapped table entry changes the output");
}

#[test]
fn rope_with_one_position_per_token_is_the_solo_rope() {
    let rig = Rig::new();
    let mut rng = SplitMix64::new(17);
    let mut total = 0;
    for (n, d, rd) in [(16usize, 256usize, 64usize), (32, 128, 128), (2, 256, 64)] {
        for m in 2..=16 {
            let x = nasty(&mut rng, m * n * d, 1.0);
            let xt = rig.bf16(&x, &[m, n, d]);
            let pos: Vec<usize> = (0..m).map(|r| [0usize, 1, 977, 4096, 31, 12345, 7, 262143][(r + m) % 8]).collect();
            let mut want = Vec::new();
            for r in 0..m {
                let o = rig.poison(n * d * 2, DType::Bf16, &[1, n, d]);
                let attrs = OpAttrs { pos0: pos[r], theta: 1.0e7, head_dim: d, rotary_dim: rd, ..Default::default() };
                rig.launch("rope_bf16", OpCall::new(Op::Rope, vec![rows_of(&xt, r)], vec![o.clone()], attrs));
                want.extend(rig.down(&o));
            }
            let post = rig.i32s(&pos);
            let o = rig.poison(m * n * d * 2, DType::Bf16, &[m, n, d]);
            let attrs = OpAttrs { theta: 1.0e7, head_dim: d, rotary_dim: rd, ..Default::default() };
            let mut call = OpCall::new(Op::Rope, vec![xt.clone()], vec![o.clone()], attrs);
            call.pos = Some(post);
            rig.launch("rope_bf16", call);
            total += same(&format!("rope n={n} d={d} rd={rd} m={m}"), &rig.down(&o), &want);
        }
    }
    eprintln!("G-B rope_bf16 per-token positions: {total} bytes bitwise equal (M 2..=16, 3 shapes)");
}

#[test]
fn swapped_state_entries_are_seen() {
    // negative controls for the stateful kernels: rows 0 and 1 swap their buffers in the table, so each row updates the other
    // row's state / cache; the result must differ from the solo one (otherwise the tests above could not tell a wrong table)
    let rig = Rig::new();
    let mut rng = SplitMix64::new(18);
    // conv
    {
        let (c, ks, m) = (1024usize, 4usize, 2usize);
        let x = nasty(&mut rng, m * c, 1.0);
        let w = rng.normal_vec(c * ks, 0.5);
        let states: Vec<Vec<f32>> = (0..m).map(|_| nasty(&mut rng, c * (ks - 1), 1.0)).collect();
        let (xt, wt) = (rig.bf16(&x, &[m, c]), rig.bf16(&w, &[c, ks]));
        let solo: Vec<Vec<u8>> = (0..m)
            .map(|r| {
                let st = rig.f32s(&states[r], &[c, ks - 1]);
                let y = rig.poison(c * 2, DType::Bf16, &[1, c]);
                rig.launch("gdn_conv_bf16", OpCall::new(Op::GdnConv, vec![rows_of(&xt, r), wt.clone()], vec![y, st.clone()], OpAttrs::default()));
                rig.down(&st)
            })
            .collect();
        let run = |order: [usize; 2]| {
            let sts: Vec<TensorRef> = (0..m).map(|r| rig.f32s(&states[r], &[c, ks - 1])).collect();
            let entries: Vec<Vec<&TensorRef>> = order.iter().map(|&r| vec![&sts[r], &sts[r]]).collect();
            let y = rig.poison(m * c * 2, DType::Bf16, &[m, c]);
            let mut call = OpCall::new(Op::GdnConv, vec![xt.clone(), wt.clone()], vec![y, sts[0].clone()], OpAttrs::default());
            call.rows = Some(rig.tab(rig.table(&entries, 2), 1, 0));
            rig.launch("gdn_conv_bf16_rows", call);
            (0..m).map(|r| rig.down(&sts[r])).collect::<Vec<_>>()
        };
        let ok = run([0, 1]);
        assert!(ok.iter().zip(&solo).all(|(a, b)| a == b));
        let bad = run([1, 0]);
        assert!(bad.iter().zip(&solo).any(|(a, b)| differs(a, b)), "conv: swapping the table entries changed nothing");
    }
    // KV append
    {
        let (nkv, d, m) = (2usize, 128usize, 2usize);
        let newkv = nasty(&mut rng, m * nkv * d, 1.0);
        let nt = rig.bf16(&newkv, &[m, nkv, d]);
        let init: Vec<Vec<f32>> = (0..m).map(|_| nasty(&mut rng, nkv * CAP * d, 1.0)).collect();
        let pos = [10usize, 500];
        let run = |order: [usize; 2]| {
            let caches: Vec<TensorRef> = (0..m).map(|r| rig.bf16(&init[r], &[nkv, CAP, d])).collect();
            let entries: Vec<Vec<&TensorRef>> = order.iter().map(|&r| vec![&caches[r], &caches[r]]).collect();
            let attrs = OpAttrs { kv_cap: CAP, n_kv_heads: nkv, head_dim: d, ..Default::default() };
            let mut call = OpCall::new(Op::KvAppend, vec![nt.clone()], vec![caches[0].clone()], attrs);
            call.rows = Some(rig.tab(rig.table(&entries, 2), 1, 0));
            call.pos = Some(rig.i32s(&pos));
            rig.launch("kv_append_bf16_rows", call);
            (0..m).map(|r| rig.down(&caches[r])).collect::<Vec<_>>()
        };
        let (ok, bad) = (run([0, 1]), run([1, 0]));
        assert!(ok.iter().zip(&bad).any(|(a, b)| differs(a, b)), "kv_append: swapping the table entries changed nothing");
    }
    // delta rule
    {
        let (hk, hv, dk, dv, m) = (2usize, 4usize, 64usize, 64usize, 2usize);
        let c = 2 * hk * dk + hv * dv;
        let qkv = nasty(&mut rng, m * c, 1.0);
        let (a, b) = (rng.normal_vec(m * hv, 1.0), rng.normal_vec(m * hv, 1.0));
        let (al, dt) = (rng.normal_vec(hv, 0.5), rng.normal_vec(hv, 0.5));
        let (qt, at, bt, alt, dtt) = (rig.bf16(&qkv, &[m, c]), rig.bf16(&a, &[m, hv]), rig.bf16(&b, &[m, hv]), rig.bf16(&al, &[hv]), rig.bf16(&dt, &[hv]));
        let init: Vec<Vec<f32>> = (0..m).map(|_| rng.normal_vec(hv * dv * dk, 0.3)).collect();
        let run = |order: [usize; 2]| {
            let sts: Vec<TensorRef> = (0..m).map(|r| rig.f32s(&init[r], &[hv, dv, dk])).collect();
            let entries: Vec<Vec<&TensorRef>> = order.iter().map(|&r| vec![&sts[r], &sts[r]]).collect();
            let o = rig.poison(m * hv * dv * 2, DType::Bf16, &[m, hv, dv]);
            let mut call = OpCall::new(
                Op::GdnDeltaRule,
                vec![qt.clone(), at.clone(), bt.clone(), alt.clone(), dtt.clone()],
                vec![o, sts[0].clone()],
                OpAttrs { eps: 1e-6, ..Default::default() },
            );
            call.rows = Some(rig.tab(rig.table(&entries, 2), 0, 0));
            rig.launch("gdn_delta_rule_bf16_rows", call);
            (0..m).map(|r| rig.down(&sts[r])).collect::<Vec<_>>()
        };
        let (ok, bad) = (run([0, 1]), run([1, 0]));
        assert!(ok.iter().zip(&bad).any(|(a, b)| differs(a, b)), "delta rule: swapping the table entries changed nothing");
    }
    eprintln!("G-B negative controls: swapped table entries change the conv state, the KV cache and the delta state");
}
