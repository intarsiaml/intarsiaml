//! G-B for the row-table kernels of batched decode on HIP (lane hr3, `OpCall::rows`): row r of one row-table launch is BITWISE equal to
//! the one-row (T = 1) launch of the solo decode step on sequence r alone, for `gdn_conv_bf16_rows`, `gdn_delta_rule_bf16_v2_rows`,
//! `kv_append_bf16_rows`, `attn_decode_bf16_splitkv_rows`, and `rope_bf16` with one position per token. Every row count from 1
//! to 16, on the Qwen3.6-35B-A3B and Qwen3-8B shapes (the delta rule is dk = 128 only here), with states, caches and positions that differ
//! per row. States and caches are compared as bytes after the launch (the in-place update is part of the contract); no tolerance.
//!
//! Needs a GPU: without one the tests print SKIPPED (and fail under CI=1 / INTARSIA_REQUIRE_GPU=1).
#![cfg(ik_hip)]

mod common;

use intarsia_core::backend::{Backend, MemKind, TensorRef};
use intarsia_core::ops::RowTab;
use intarsia_core::rng::SplitMix64;
use intarsia_core::{DType, HostTensor, Op, OpAttrs, OpCall, ShapeInfo};
use std::sync::Arc;

const CAP: usize = 16;

struct Rig {
    be: Arc<dyn Backend>,
}

impl Rig {
    fn new(test: &str) -> Option<Rig> {
        let ord = std::env::var("INTARSIA_HIP_DEVICE").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
        match intarsia_backend_hip::create(ord) {
            Ok(be) => Some(Rig { be }),
            Err(e) => {
                common::skip(test, &format!("no usable HIP device ({e})"));
                None
            }
        }
    }

    fn put_bytes(&self, data: &[u8], dtype: DType, shape: &[usize]) -> TensorRef {
        let buf = self.be.alloc(data.len().max(16), MemKind::Device).unwrap();
        self.be.zero(&buf, 0).unwrap();
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

    /// An output buffer of `bytes` filled with 0xA5 (a NaN in bf16 / f32).
    fn poison(&self, bytes: usize, dtype: DType, shape: &[usize]) -> TensorRef {
        self.put_bytes(&vec![0xA5u8; bytes], dtype, shape)
    }

    fn down(&self, t: &TensorRef) -> Vec<u8> {
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
            self.be.zero(&buf, 0).unwrap();
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

fn rows_of(t: &TensorRef, r: usize) -> TensorRef {
    t.rows(r, 1)
}

/// Random-with-spread activations: mostly normal, with a few large and tiny magnitudes so that sums round.
fn nasty(rng: &mut SplitMix64, n: usize, scale: f32) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let v = rng.normal_vec(1, 1.0)[0] * scale;
            match i % 17 {
                0 => v * 64.0,
                5 => v * 1e-3,
                _ => v,
            }
        })
        .collect()
}

fn same(what: &str, got: &[u8], want: &[u8]) -> usize {
    assert_eq!(got.len(), want.len(), "{what}: length");
    if let Some(i) = got.iter().zip(want).position(|(a, b)| a != b) {
        panic!("{what}: first differing byte {i} of {} (row-table launch {:#04x} vs solo {:#04x})", got.len(), got[i], want[i]);
    }
    got.len()
}

#[test]
fn gdn_conv_rows_are_the_solo_conv() {
    let Some(rig) = Rig::new("gdn_conv_rows") else { return };
    let mut rng = SplitMix64::new(11);
    let mut total = 0;
    for (c, ks) in [(8192usize, 4usize), (6144, 4), (130, 3)] {
        for m in 1..=CAP {
            let x = nasty(&mut rng, m * c, 1.0);
            let w = rng.normal_vec(c * ks, 0.5);
            let states: Vec<Vec<f32>> = (0..m).map(|_| nasty(&mut rng, c * (ks - 1), 1.0)).collect();
            let (xt, wt) = (rig.bf16(&x, &[m, c]), rig.bf16(&w, &[c, ks]));
            // solo: one T = 1 call per row on its own state
            let mut want_y = Vec::new();
            let mut want_s = Vec::new();
            for r in 0..m {
                let st = rig.f32s(&states[r], &[c, ks - 1]);
                let y = rig.poison(c * 2, DType::Bf16, &[1, c]);
                rig.launch("gdn_conv_bf16", OpCall::new(Op::GdnConv, vec![rows_of(&xt, r), wt.clone()], vec![y.clone(), st.clone()], OpAttrs::default()));
                want_y.extend(rig.down(&y));
                want_s.push(rig.down(&st));
            }
            // row table: the same initial states
            let sts: Vec<TensorRef> = (0..m).map(|r| rig.f32s(&states[r], &[c, ks - 1])).collect();
            let entries: Vec<Vec<&TensorRef>> = sts.iter().map(|s| vec![s, s]).collect();
            let ptrs = rig.table(&entries, 2);
            let y = rig.poison(m * c * 2, DType::Bf16, &[m, c]);
            let mut call = OpCall::new(Op::GdnConv, vec![xt.clone(), wt.clone()], vec![y.clone(), sts[0].clone()], OpAttrs::default());
            call.rows = Some(rig.tab(ptrs, 1, 0));
            rig.launch("gdn_conv_bf16_rows", call);
            total += same(&format!("conv c={c} k={ks} m={m} y"), &rig.down(&y), &want_y);
            for r in 0..m {
                total += same(&format!("conv c={c} k={ks} m={m} state {r}"), &rig.down(&sts[r]), &want_s[r]);
            }
        }
    }
    println!("G-B gdn_conv_bf16_rows: {total} bytes identical to the solo launches (M 1..=16, 3 shapes)");
}

#[test]
fn gdn_delta_rows_are_the_solo_delta_rule() {
    let Some(rig) = Rig::new("gdn_delta_rows") else { return };
    let mut rng = SplitMix64::new(12);
    let mut total = 0;
    // (Hk, Hv, dk, dv): the 35B (16, 32, 128, 128) and a smaller dk = 128 head layout
    for (hk, hv, dk, dv) in [(16usize, 32usize, 128usize, 128usize), (4, 8, 128, 64)] {
        let c = 2 * hk * dk + hv * dv;
        let sbytes = hv * dv * dk * 4;
        let off = 4096usize; // the delta state sits behind the conv state in a layer's buffer
        for m in 1..=CAP {
            let qkv = nasty(&mut rng, m * c, 1.0);
            let a = rng.normal_vec(m * hv, 1.0);
            let b = rng.normal_vec(m * hv, 1.0);
            let a_log = rng.normal_vec(hv, 0.5);
            let dt = rng.normal_vec(hv, 0.5);
            let (qt, at, bt, alt, dtt) =
                (rig.bf16(&qkv, &[m, c]), rig.bf16(&a, &[m, hv]), rig.bf16(&b, &[m, hv]), rig.bf16(&a_log, &[hv]), rig.bf16(&dt, &[hv]));
            let init: Vec<Vec<u8>> = (0..m)
                .map(|_| {
                    let mut v = vec![0u8; off];
                    v.extend(HostTensor::from_f32_as(DType::F32, &[hv * dv * dk], &rng.normal_vec(hv * dv * dk, 0.3)).data);
                    v
                })
                .collect();
            let attrs = OpAttrs { eps: 1e-6, ..Default::default() };
            let view = |buf: &TensorRef| {
                let mut s = TensorRef::new(buf.buf.clone(), DType::F32, &[hv, dv, dk]);
                s.offset = off;
                s
            };
            let mut want_o = Vec::new();
            let mut want_s = Vec::new();
            for r in 0..m {
                let st = rig.put_bytes(&init[r], DType::U8, &[init[r].len()]);
                let o = rig.poison(hv * dv * 2, DType::Bf16, &[1, hv, dv]);
                rig.launch(
                    "gdn_delta_rule_bf16_v2",
                    OpCall::new(Op::GdnDeltaRule, vec![rows_of(&qt, r), rows_of(&at, r), rows_of(&bt, r), alt.clone(), dtt.clone()], vec![o.clone(), view(&st)], attrs.clone()),
                );
                want_o.extend(rig.down(&o));
                want_s.push(rig.down(&st));
            }
            let sts: Vec<TensorRef> = (0..m).map(|r| rig.put_bytes(&init[r], DType::U8, &[init[r].len()])).collect();
            let entries: Vec<Vec<&TensorRef>> = sts.iter().map(|s| vec![s, s]).collect();
            let ptrs = rig.table(&entries, 2);
            let o = rig.poison(m * hv * dv * 2, DType::Bf16, &[m, hv, dv]);
            let mut call = OpCall::new(Op::GdnDeltaRule, vec![qt.clone(), at.clone(), bt.clone(), alt.clone(), dtt.clone()], vec![o.clone(), view(&sts[0])], attrs);
            call.rows = Some(rig.tab(ptrs, 0, off));
            rig.launch("gdn_delta_rule_bf16_v2_rows", call);
            total += same(&format!("delta dk={dk} m={m} out"), &rig.down(&o), &want_o);
            for r in 0..m {
                total += same(&format!("delta dk={dk} m={m} state {r}"), &rig.down(&sts[r]), &want_s[r]);
                assert_eq!(rig.down(&sts[r]).len(), off + sbytes);
            }
        }
    }
    println!("G-B gdn_delta_rule_bf16_v2_rows: {total} bytes identical to the solo launches (M 1..=16, 2 shapes)");
}

#[test]
fn kv_append_rows_are_the_solo_append() {
    let Some(rig) = Rig::new("kv_append_rows") else { return };
    let mut rng = SplitMix64::new(13);
    let cap = 80usize;
    let mut total = 0;
    for (nkv, d) in [(2usize, 256usize), (8, 128)] {
        for m in 1..=CAP {
            let newkv = nasty(&mut rng, m * nkv * d, 1.0);
            let nt = rig.bf16(&newkv, &[m, nkv, d]);
            let pos: Vec<i32> = (0..m).map(|r| [0, 1, 63, 64, 79, 17, 40, 2][r % 8] as i32).collect();
            let init: Vec<Vec<f32>> = (0..m).map(|_| nasty(&mut rng, nkv * cap * d, 1.0)).collect();
            let mut want = Vec::new();
            for r in 0..m {
                let cache = rig.bf16(&init[r], &[nkv, cap, d]);
                let attrs = OpAttrs { pos0: pos[r] as usize, kv_cap: cap, n_kv_heads: nkv, head_dim: d, ..Default::default() };
                rig.launch("kv_append_bf16", OpCall::new(Op::KvAppend, vec![rows_of(&nt, r)], vec![cache.clone()], attrs));
                want.push(rig.down(&cache));
            }
            let caches: Vec<TensorRef> = (0..m).map(|r| rig.bf16(&init[r], &[nkv, cap, d])).collect();
            let entries: Vec<Vec<&TensorRef>> = caches.iter().map(|s| vec![s, s]).collect();
            let ptrs = rig.table(&entries, 2);
            let pt: Vec<u8> = pos.iter().flat_map(|p| p.to_le_bytes()).collect();
            let post = rig.put_bytes(&pt, DType::I32, &[m]);
            let attrs = OpAttrs { kv_cap: cap, n_kv_heads: nkv, head_dim: d, ..Default::default() };
            let mut call = OpCall::new(Op::KvAppend, vec![nt.clone()], vec![caches[0].clone()], attrs);
            call.rows = Some(rig.tab(ptrs, 1, 0));
            call.pos = Some(post);
            rig.launch("kv_append_bf16_rows", call);
            for r in 0..m {
                total += same(&format!("kv_append nkv={nkv} d={d} m={m} cache {r}"), &rig.down(&caches[r]), &want[r]);
            }
        }
    }
    println!("G-B kv_append_bf16_rows: {total} bytes identical to the solo launches (M 1..=16, 2 shapes)");
}

#[test]
fn attn_decode_rows_are_the_solo_decode() {
    let Some(rig) = Rig::new("attn_decode_rows") else { return };
    let mut rng = SplitMix64::new(14);
    let cap = 1200usize;
    let mut total = 0;
    // 35B attention (16 q heads, 2 kv heads, 256), Qwen3-8B (32, 8, 128)
    for (nh, nkv, d) in [(16usize, 2usize, 256usize), (32, 8, 128)] {
        let ps = [0usize, 1, 63, 64, 65, 255, 511, 700, 1023, 1199, 128, 2, 300, 900, 31, 1100];
        for m in 1..=CAP {
            let q = nasty(&mut rng, m * nh * d, 1.0);
            let qt = rig.bf16(&q, &[m, nh, d]);
            let pos: Vec<usize> = (0..m).map(|r| ps[(r * 7 + m) % ps.len()]).collect();
            let kv: Vec<(Vec<f32>, Vec<f32>)> = (0..m).map(|_| (nasty(&mut rng, nkv * cap * d, 1.0), nasty(&mut rng, nkv * cap * d, 1.0))).collect();
            let scale = 1.0 / (d as f32).sqrt();
            let mut want = Vec::new();
            for r in 0..m {
                let (kc, vc) = (rig.bf16(&kv[r].0, &[nkv, cap, d]), rig.bf16(&kv[r].1, &[nkv, cap, d]));
                let o = rig.poison(nh * d * 2, DType::Bf16, &[1, nh, d]);
                let attrs = OpAttrs { pos0: pos[r], scale, n_heads: nh, n_kv_heads: nkv, head_dim: d, kv_cap: cap, ..Default::default() };
                // the solo decode step passes its position on the device (the graph is recorded once): so does the reference
                let mut solo = OpCall::new(Op::AttnDecode, vec![rows_of(&qt, r), kc, vc], vec![o.clone()], attrs);
                solo.pos = Some(rig.put_bytes(&(pos[r] as i32).to_le_bytes(), DType::I32, &[1]));
                rig.launch("attn_decode_bf16_splitkv", solo);
                want.extend(rig.down(&o));
            }
            let kcs: Vec<TensorRef> = (0..m).map(|r| rig.bf16(&kv[r].0, &[nkv, cap, d])).collect();
            let vcs: Vec<TensorRef> = (0..m).map(|r| rig.bf16(&kv[r].1, &[nkv, cap, d])).collect();
            let entries: Vec<Vec<&TensorRef>> = (0..m).map(|r| vec![&kcs[r], &vcs[r]]).collect();
            let ptrs = rig.table(&entries, 2);
            let pt: Vec<u8> = pos.iter().flat_map(|p| (*p as i32).to_le_bytes()).collect();
            let post = rig.put_bytes(&pt, DType::I32, &[m]);
            let o = rig.poison(m * nh * d * 2, DType::Bf16, &[m, nh, d]);
            let attrs = OpAttrs { scale, n_heads: nh, n_kv_heads: nkv, head_dim: d, kv_cap: cap, ..Default::default() };
            let mut call = OpCall::new(Op::AttnDecode, vec![qt.clone(), kcs[0].clone(), vcs[0].clone()], vec![o.clone()], attrs);
            call.rows = Some(rig.tab(ptrs, 0, 0));
            call.pos = Some(post);
            rig.launch("attn_decode_bf16_splitkv_rows", call);
            total += same(&format!("attn nh={nh} nkv={nkv} d={d} m={m}"), &rig.down(&o), &want);
        }
    }
    println!("G-B attn_decode_bf16_splitkv_rows: {total} bytes identical to the solo launches (M 1..=16, 2 shapes, positions 0..1199)");
}

#[test]
fn rope_with_one_position_per_row_is_the_solo_rope() {
    let Some(rig) = Rig::new("rope_rows") else { return };
    let mut rng = SplitMix64::new(15);
    let mut total = 0;
    for (n, d, rd) in [(16usize, 256usize, 64usize), (32, 128, 128), (2, 256, 64)] {
        for m in 2..=CAP {
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
            let pt: Vec<u8> = pos.iter().flat_map(|p| (*p as i32).to_le_bytes()).collect();
            let post = rig.put_bytes(&pt, DType::I32, &[m]);
            let o = rig.poison(m * n * d * 2, DType::Bf16, &[m, n, d]);
            let attrs = OpAttrs { theta: 1.0e7, head_dim: d, rotary_dim: rd, ..Default::default() };
            let mut call = OpCall::new(Op::Rope, vec![xt.clone()], vec![o.clone()], attrs);
            call.pos = Some(post);
            rig.launch("rope_bf16", call);
            total += same(&format!("rope n={n} d={d} rd={rd} m={m}"), &rig.down(&o), &want);
        }
    }
    println!("G-B rope_bf16 (one position per row): {total} bytes identical to the solo launches (M 2..=16, 3 shapes)");
}


/// The solo decode step runs the GatedDeltaNet conv and delta rule as the one fused `gdn_conv_delta_bf16`: the row-table pair is
/// bitwise that kernel on each row (the fusion table's proof, `fusion_p2.rs`, composed with the two row-table tests above, checked
/// directly), conv state and delta state included.
#[test]
fn gdn_row_tables_are_the_solo_fused_conv_delta() {
    let Some(rig) = Rig::new("gdn_fused") else { return };
    let mut rng = SplitMix64::new(17);
    let (hk, hv, dk, dv, ks) = (16usize, 32usize, 128usize, 128usize, 4usize);
    let c = 2 * hk * dk + hv * dv;
    let off = c * (ks - 1) * 4; // the delta state sits behind the conv state in a layer's buffer
    let mut total = 0;
    for m in 1..=CAP {
        let qkv = nasty(&mut rng, m * c, 1.0);
        let w = rng.normal_vec(c * ks, 0.5);
        let a = rng.normal_vec(m * hv, 1.0);
        let b = rng.normal_vec(m * hv, 1.0);
        let a_log = rng.normal_vec(hv, 0.5);
        let dt = rng.normal_vec(hv, 0.5);
        let (qt, wt, at, bt, alt, dtt) = (
            rig.bf16(&qkv, &[m, c]),
            rig.bf16(&w, &[c, ks]),
            rig.bf16(&a, &[m, hv]),
            rig.bf16(&b, &[m, hv]),
            rig.bf16(&a_log, &[hv]),
            rig.bf16(&dt, &[hv]),
        );
        // one recurrent buffer per sequence: [conv state f32 [c, ks - 1] | pad to `off` | delta state f32 [hv, dv, dk]]
        let init: Vec<Vec<u8>> = (0..m)
            .map(|_| {
                let mut v = HostTensor::from_f32_as(DType::F32, &[c * (ks - 1)], &nasty(&mut rng, c * (ks - 1), 1.0)).data;
                v.resize(off, 0);
                v.extend(HostTensor::from_f32_as(DType::F32, &[hv * dv * dk], &rng.normal_vec(hv * dv * dk, 0.3)).data);
                v
            })
            .collect();
        let attrs = OpAttrs { eps: 1e-6, ..Default::default() };
        let conv_view = |buf: &TensorRef| TensorRef::new(buf.buf.clone(), DType::F32, &[c, ks - 1]);
        let delta_view = |buf: &TensorRef| {
            let mut s = TensorRef::new(buf.buf.clone(), DType::F32, &[hv, dv, dk]);
            s.offset = off;
            s
        };
        let mut want_o = Vec::new();
        let mut want_s = Vec::new();
        for r in 0..m {
            let st = rig.put_bytes(&init[r], DType::U8, &[init[r].len()]);
            let o = rig.poison(hv * dv * 2, DType::Bf16, &[1, hv, dv]);
            rig.launch(
                "gdn_conv_delta_bf16",
                OpCall::new(
                    Op::GdnConvDelta,
                    vec![rows_of(&qt, r), wt.clone(), rows_of(&at, r), rows_of(&bt, r), alt.clone(), dtt.clone()],
                    vec![o.clone(), delta_view(&st), conv_view(&st)],
                    attrs.clone(),
                ),
            );
            want_o.extend(rig.down(&o));
            want_s.push(rig.down(&st));
        }
        let sts: Vec<TensorRef> = (0..m).map(|r| rig.put_bytes(&init[r], DType::U8, &[init[r].len()])).collect();
        let entries: Vec<Vec<&TensorRef>> = sts.iter().map(|s| vec![s, s]).collect();
        let conv = rig.poison(m * c * 2, DType::Bf16, &[m, c]);
        let mut cc = OpCall::new(Op::GdnConv, vec![qt.clone(), wt.clone()], vec![conv.clone(), conv_view(&sts[0])], OpAttrs::default());
        cc.rows = Some(rig.tab(rig.table(&entries, 2), 0, 0));
        rig.launch("gdn_conv_bf16_rows", cc);
        let o = rig.poison(m * hv * dv * 2, DType::Bf16, &[m, hv, dv]);
        let mut dc = OpCall::new(
            Op::GdnDeltaRule,
            vec![conv.clone(), at.clone(), bt.clone(), alt.clone(), dtt.clone()],
            vec![o.clone(), delta_view(&sts[0])],
            attrs,
        );
        dc.rows = Some(rig.tab(rig.table(&entries, 2), 0, off));
        rig.launch("gdn_delta_rule_bf16_v2_rows", dc);
        total += same(&format!("fused m={m} out"), &rig.down(&o), &want_o);
        for r in 0..m {
            total += same(&format!("fused m={m} state {r} (conv + delta)"), &rig.down(&sts[r]), &want_s[r]);
        }
    }
    println!("G-B gdn_conv_bf16_rows + gdn_delta_rule_bf16_v2_rows: {total} bytes identical to the solo fused gdn_conv_delta_bf16 (M 1..=16)");
}

/// Negative control: the comparison can fail. Two rows' table entries swapped (the kernel then updates the wrong sequences'
/// state / caches) must differ from the solo launches.
#[test]
fn control_swapped_table_entries_are_detected() {
    let Some(rig) = Rig::new("control") else { return };
    let mut rng = SplitMix64::new(18);
    // conv
    let (c, ks, m) = (8192usize, 4usize, 3usize);
    let x = nasty(&mut rng, m * c, 1.0);
    let w = rng.normal_vec(c * ks, 0.5);
    let states: Vec<Vec<f32>> = (0..m).map(|_| nasty(&mut rng, c * (ks - 1), 1.0)).collect();
    let (xt, wt) = (rig.bf16(&x, &[m, c]), rig.bf16(&w, &[c, ks]));
    let mut want_y = Vec::new();
    for r in 0..m {
        let st = rig.f32s(&states[r], &[c, ks - 1]);
        let y = rig.poison(c * 2, DType::Bf16, &[1, c]);
        rig.launch("gdn_conv_bf16", OpCall::new(Op::GdnConv, vec![rows_of(&xt, r), wt.clone()], vec![y.clone(), st], OpAttrs::default()));
        want_y.extend(rig.down(&y));
    }
    let sts: Vec<TensorRef> = (0..m).map(|r| rig.f32s(&states[r], &[c, ks - 1])).collect();
    let order = [1usize, 0, 2];
    let entries: Vec<Vec<&TensorRef>> = order.iter().map(|&r| vec![&sts[r], &sts[r]]).collect();
    let y = rig.poison(m * c * 2, DType::Bf16, &[m, c]);
    let mut call = OpCall::new(Op::GdnConv, vec![xt.clone(), wt.clone()], vec![y.clone(), sts[0].clone()], OpAttrs::default());
    call.rows = Some(rig.tab(rig.table(&entries, 2), 1, 0));
    rig.launch("gdn_conv_bf16_rows", call);
    assert_ne!(rig.down(&y), want_y, "swapped state entries must change the conv output");
    // attention: the two rows' positions swapped
    let (nh, nkv, d, cap) = (16usize, 2usize, 256usize, 300usize);
    let q = nasty(&mut rng, 2 * nh * d, 1.0);
    let qt = rig.bf16(&q, &[2, nh, d]);
    let kv: Vec<(Vec<f32>, Vec<f32>)> = (0..2).map(|_| (nasty(&mut rng, nkv * cap * d, 1.0), nasty(&mut rng, nkv * cap * d, 1.0))).collect();
    let kcs: Vec<TensorRef> = (0..2).map(|r| rig.bf16(&kv[r].0, &[nkv, cap, d])).collect();
    let vcs: Vec<TensorRef> = (0..2).map(|r| rig.bf16(&kv[r].1, &[nkv, cap, d])).collect();
    let scale = 1.0 / (d as f32).sqrt();
    let run = |pos: [i32; 2]| {
        let entries: Vec<Vec<&TensorRef>> = (0..2).map(|r| vec![&kcs[r], &vcs[r]]).collect();
        let pt: Vec<u8> = pos.iter().flat_map(|p| p.to_le_bytes()).collect();
        let o = rig.poison(2 * nh * d * 2, DType::Bf16, &[2, nh, d]);
        let attrs = OpAttrs { scale, n_heads: nh, n_kv_heads: nkv, head_dim: d, kv_cap: cap, ..Default::default() };
        let mut call = OpCall::new(Op::AttnDecode, vec![qt.clone(), kcs[0].clone(), vcs[0].clone()], vec![o.clone()], attrs);
        call.rows = Some(rig.tab(rig.table(&entries, 2), 0, 0));
        call.pos = Some(rig.put_bytes(&pt, DType::I32, &[2]));
        rig.launch("attn_decode_bf16_splitkv_rows", call);
        rig.down(&o)
    };
    assert_ne!(run([5, 250]), run([250, 5]), "swapped positions must change the attention output");
}
