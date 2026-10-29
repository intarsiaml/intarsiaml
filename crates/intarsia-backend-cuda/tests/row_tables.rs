//! G-B for the row-table kernels of batched decode (R3a, `OpCall::rows`): row r of one row-table launch is BITWISE equal to the
//! one-row (T = 1) launch of the solo decode step on sequence r alone, for `gdn_conv_bf16_rows`, `gdn_delta_rule_bf16_split_rows`,
//! `kv_append_bf16_rows`, `attn_decode_bf16_splitkv_rows`, and `rope_bf16` with one position per token. Every row count from 1
//! to 16, on the Qwen3.6-35B-A3B and Qwen3-8B shapes (and the dk = 64 delta rule), with states, caches and positions that differ
//! per row. States and caches are compared as bytes after the launch (the in-place update is part of the contract); no tolerance.
//!
//! Needs a GPU: without one the tests print SKIPPED (and fail under CI=1 / INTARSIA_REQUIRE_GPU=1).
#![cfg(ik_cuda)]

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
        let ord = std::env::var("INTARSIA_CUDA_DEVICE").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
        match intarsia_backend_cuda::create(ord) {
            Ok(be) => Some(Rig { be }),
            Err(e) => {
                common::skip(test, &format!("no usable CUDA device ({e})"));
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
    // (Hk, Hv, dk, dv): the 35B (16, 32, 128, 128) and a dk = 64 head
    for (hk, hv, dk, dv) in [(16usize, 32usize, 128usize, 128usize), (4, 8, 64, 64)] {
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
                    "gdn_delta_rule_bf16_split",
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
            rig.launch("gdn_delta_rule_bf16_split_rows", call);
            total += same(&format!("delta dk={dk} m={m} out"), &rig.down(&o), &want_o);
            for r in 0..m {
                total += same(&format!("delta dk={dk} m={m} state {r}"), &rig.down(&sts[r]), &want_s[r]);
                assert_eq!(rig.down(&sts[r]).len(), off + sbytes);
            }
        }
    }
    println!("G-B gdn_delta_rule_bf16_split_rows: {total} bytes identical to the solo launches (M 1..=16, dk 128 and 64)");
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
    // 35B attention (16 q heads, 2 kv heads, 256), Qwen3-8B (32, 8, 128), and the other head dims / group sizes the kernel's
    // `when` admits (D = 32 / 64, a MHA group of 1, non power of two groups 3 and 5, the largest group 8), which no model here runs
    // but the registry would select
    for (nh, nkv, d) in [(16usize, 2usize, 256usize), (32, 8, 128), (4, 4, 64), (6, 2, 32), (12, 4, 128), (10, 2, 64), (8, 1, 128)] {
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
                rig.launch("attn_decode_bf16_splitkv", OpCall::new(Op::AttnDecode, vec![rows_of(&qt, r), kc, vc], vec![o.clone()], attrs));
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
    println!("G-B attn_decode_bf16_splitkv_rows: {total} bytes identical to the solo launches (M 1..=16, 7 shapes, positions 0..1199)");
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

/// R3b: the device top-k is the host sampler's candidate list (value descending, ties by ascending index, -0.0 == +0.0): ids and
/// values equal to a host sort, for rows with many bf16-grid ties, signed zeros and -inf, k from 1 to 64, R 1..=16, V = 248320 (the
/// 35B vocabulary) and odd sizes; a row with a NaN is flagged.
#[test]
fn device_topk_is_the_host_candidate_order() {
    let Some(rig) = Rig::new("topk") else { return };
    let mut rng = SplitMix64::new(16);
    let mut checked = 0usize;
    for v in [248320usize, 151936, 1000, 70] {
        for (rows, k) in [(1usize, 1usize), (3, 20), (16, 40), (16, 64), (5, 64.min(v - 1))] {
            let mut data = Vec::new();
            for r in 0..rows {
                // bf16-grid logits (many exact ties), skewed like a real row; some special values
                let mut row: Vec<f32> = (0..v)
                    .map(|i| {
                        let x = rng.normal_vec(1, 3.0)[0] + if i % 97 == 0 { 9.0 } else { 0.0 };
                        f32::from_bits(x.to_bits() & 0xffff_0000)
                    })
                    .collect();
                if r % 3 == 1 {
                    for (i, x) in row.iter_mut().enumerate() {
                        match i % 11 {
                            1 => *x = 0.0,
                            2 => *x = -0.0,
                            3 => *x = f32::NEG_INFINITY,
                            _ => {}
                        }
                    }
                }
                if r % 5 == 4 {
                    row.iter_mut().for_each(|x| *x = 2.0); // all tied: the k lowest indices
                }
                data.extend(row);
            }
            let logits = rig.f32s(&data, &[rows, v]);
            let ids = rig.poison(rows * k * 4, DType::I32, &[rows, k]);
            let vals = rig.poison(rows * k * 4, DType::F32, &[rows, k]);
            let nan = rig.poison(rows * 4, DType::I32, &[rows]);
            let sbuf = rig.put_bytes(&vec![0u8; rows * 32 * k * 8], DType::U8, &[rows * 32 * k * 8]);
            rig.be.topk_rows(&logits, k, &ids, &vals, &nan, &sbuf, 0).unwrap();
            rig.be.synchronize(0).unwrap();
            let (gi, gv, gn) = (rig.down(&ids), rig.down(&vals), rig.down(&nan));
            for r in 0..rows {
                let row = &data[r * v..(r + 1) * v];
                let mut order: Vec<u32> = (0..v as u32).collect();
                let cmp = |a: &u32, b: &u32| row[*b as usize].partial_cmp(&row[*a as usize]).unwrap().then(a.cmp(b));
                order.select_nth_unstable_by(k - 1, cmp);
                order.truncate(k);
                order.sort_unstable_by(cmp);
                assert_eq!(i32::from_le_bytes(gn[r * 4..r * 4 + 4].try_into().unwrap()), 0, "no NaN in row {r}");
                for j in 0..k {
                    let id = i32::from_le_bytes(gi[(r * k + j) * 4..(r * k + j) * 4 + 4].try_into().unwrap());
                    let val = f32::from_le_bytes(gv[(r * k + j) * 4..(r * k + j) * 4 + 4].try_into().unwrap());
                    assert_eq!(id as u32, order[j], "v {v} rows {rows} k {k} row {r} rank {j}");
                    assert_eq!(val.to_bits(), row[order[j] as usize].to_bits(), "v {v} rows {rows} k {k} row {r} rank {j} value");
                    checked += 1;
                }
            }
        }
    }
    // a NaN anywhere in a row flags that row
    let mut data = vec![1.0f32; 2 * 500];
    data[500 + 321] = f32::NAN;
    let logits = rig.f32s(&data, &[2, 500]);
    let ids = rig.poison(2 * 8 * 4, DType::I32, &[2, 8]);
    let vals = rig.poison(2 * 8 * 4, DType::F32, &[2, 8]);
    let nan = rig.poison(8, DType::I32, &[2]);
    let sbuf = rig.put_bytes(&vec![0u8; 2 * 32 * 8 * 8], DType::U8, &[2 * 32 * 8 * 8]);
    rig.be.topk_rows(&logits, 8, &ids, &vals, &nan, &sbuf, 0).unwrap();
    rig.be.synchronize(0).unwrap();
    let gn = rig.down(&nan);
    assert_eq!((i32::from_le_bytes(gn[0..4].try_into().unwrap()), i32::from_le_bytes(gn[4..8].try_into().unwrap())), (0, 1));
    println!("G-B topk_f32: {checked} (id, value) pairs equal to the host order (V 248320 / 151936 / 1000 / 70, R <= 16, k <= 64), NaN row flagged");
}

/// Timing (ignored; `--ignored --nocapture timing_gdn_rows`): the batched-decode GDN launches at the 35B shape for M rows,
/// cycling over 30 layers' own recurrent buffers (conv state, then the delta state) so the states come from DRAM as in the
/// decode step. Prints `GDNT kernel M us/call`. Env knobs of the launchers (INTARSIA_GDN_DEC, INTARSIA_GDN_DEC_WPB, INTARSIA_GDN_SPLIT_LPR)
/// apply as in the engine.
#[test]
#[ignore]
fn timing_gdn_rows() {
    let Some(rig) = Rig::new("timing_gdn_rows") else { return };
    let mut rng = SplitMix64::new(14);
    let (hk, hv, dk, dv, c, ks, layers) = (16usize, 32usize, 128usize, 128usize, 8192usize, 4usize, 30usize);
    let qc = 2 * hk * dk + hv * dv;
    let off = c * (ks - 1) * 4;
    let bytes = off + hv * dv * dk * 4;
    let ms: Vec<usize> = std::env::var("GDNT_M").ok().map(|v| v.split(',').map(|x| x.parse().unwrap()).collect()).unwrap_or(vec![1, 4, 8, 16]);
    let cap = *ms.iter().max().unwrap();
    // one buffer per (layer, row), small random states
    let init = HostTensor::from_f32_as(DType::F32, &[bytes / 4], &rng.normal_vec(bytes / 4, 0.1)).data;
    let bufs: Vec<Vec<TensorRef>> = (0..layers).map(|_| (0..cap).map(|_| rig.put_bytes(&init, DType::U8, &[bytes])).collect()).collect();
    let attrs = OpAttrs { eps: 1e-6, ..Default::default() };
    for &m in &ms {
        let qkv = rig.bf16(&rng.normal_vec(m * qc, 1.0), &[m, qc]);
        let x = rig.bf16(&rng.normal_vec(m * c, 1.0), &[m, c]);
        let w = rig.bf16(&rng.normal_vec(c * ks, 0.5), &[c, ks]);
        let y = rig.poison(m * c * 2, DType::Bf16, &[m, c]);
        let (at, bt) = (rig.bf16(&rng.normal_vec(m * hv, 1.0), &[m, hv]), rig.bf16(&rng.normal_vec(m * hv, 1.0), &[m, hv]));
        let (alt, dtt) = (rig.bf16(&rng.normal_vec(hv, 0.5), &[hv]), rig.bf16(&rng.normal_vec(hv, 0.5), &[hv]));
        let o = rig.poison(m * hv * dv * 2, DType::Bf16, &[m, hv, dv]);
        let mut convs = Vec::new();
        let mut deltas = Vec::new();
        for l in 0..layers {
            let entries: Vec<Vec<&TensorRef>> = bufs[l][..m].iter().map(|s| vec![s]).collect();
            let ptrs = rig.table(&entries, 1);
            let cs = TensorRef::new(bufs[l][0].buf.clone(), DType::F32, &[c, ks - 1]);
            let mut call = OpCall::new(Op::GdnConv, vec![x.clone(), w.clone()], vec![y.clone(), cs], OpAttrs::default());
            call.rows = Some(rig.tab(ptrs.clone(), 0, 0));
            call.check().unwrap();
            convs.push(call);
            let mut ds = TensorRef::new(bufs[l][0].buf.clone(), DType::F32, &[hv, dv, dk]);
            ds.offset = off;
            let mut call = OpCall::new(Op::GdnDeltaRule, vec![qkv.clone(), at.clone(), bt.clone(), alt.clone(), dtt.clone()], vec![o.clone(), ds], attrs.clone());
            call.rows = Some(rig.tab(ptrs, 0, off));
            call.check().unwrap();
            deltas.push(call);
        }
        for (name, calls) in [("gdn_conv_bf16_rows", &convs), ("gdn_delta_rule_bf16_split_rows", &deltas)] {
            let mut best = f64::MAX;
            for rep in 0..16 {
                rig.be.synchronize(0).unwrap();
                let t0 = std::time::Instant::now();
                for call in calls.iter() {
                    rig.be.launch(name, call, 0).unwrap();
                }
                rig.be.synchronize(0).unwrap();
                let us = t0.elapsed().as_secs_f64() * 1e6 / calls.len() as f64;
                if rep > 0 {
                    best = best.min(us);
                }
            }
            println!("GDNT {name} M={m} {best:.1} us/call");
        }
    }
}

/// Timing (ignored; `--ignored --nocapture timing_attn_rows`): the batched-decode attention launch (`attn_decode_bf16_splitkv_rows`,
/// stage 1 + combine) at the 35B shape (16 q heads, 2 kv heads of 256) for M rows, cycling over 10 layers' own K / V caches per
/// row so the caches come from DRAM as in the decode step. Row r attends ATTNT_POS + 37 * r % 200 keys (default 1100, the
/// served ~1k-token prompts). Prints `ATTNT M us/call GB/s` (K + V bytes read). INTARSIA_ATTN_ROWS_PERSIST applies as in the engine.
#[test]
#[ignore]
fn timing_attn_rows() {
    let Some(rig) = Rig::new("timing_attn_rows") else { return };
    let mut rng = SplitMix64::new(15);
    let (nh, nkv, d, layers, cap) = (16usize, 2usize, 256usize, 10usize, 1400usize);
    let ms: Vec<usize> = std::env::var("ATTNT_M").ok().map(|v| v.split(',').map(|x| x.parse().unwrap()).collect()).unwrap_or(vec![1, 4, 8, 16]);
    let base: usize = std::env::var("ATTNT_POS").ok().map(|v| v.parse().unwrap()).unwrap_or(1100);
    let mmax = *ms.iter().max().unwrap();
    let init = HostTensor::from_f32_as(DType::Bf16, &[nkv * cap * d], &rng.normal_vec(nkv * cap * d, 1.0)).data;
    let caches: Vec<Vec<(TensorRef, TensorRef)>> = (0..layers)
        .map(|_| (0..mmax).map(|_| (rig.put_bytes(&init, DType::Bf16, &[nkv, cap, d]), rig.put_bytes(&init, DType::Bf16, &[nkv, cap, d]))).collect())
        .collect();
    let scale = 1.0 / (d as f32).sqrt();
    let k = rig.be.kernels().into_iter().find(|k| k.name == "attn_decode_bf16_splitkv_rows" && k.op == Op::AttnDecode).expect("registered");
    for &m in &ms {
        let pos: Vec<usize> = (0..m).map(|r| (base + 37 * r % 200).min(cap - 1)).collect();
        let pt: Vec<u8> = pos.iter().flat_map(|p| (*p as i32).to_le_bytes()).collect();
        let post = rig.put_bytes(&pt, DType::I32, &[m]);
        let q = rig.bf16(&rng.normal_vec(m * nh * d, 1.0), &[m, nh, d]);
        let o = rig.poison(m * nh * d * 2, DType::Bf16, &[m, nh, d]);
        let attrs = OpAttrs { scale, n_heads: nh, n_kv_heads: nkv, head_dim: d, kv_cap: cap, ..Default::default() };
        let mut calls = Vec::new();
        for l in 0..layers {
            let entries: Vec<Vec<&TensorRef>> = (0..m).map(|r| vec![&caches[l][r].0, &caches[l][r].1]).collect();
            let ptrs = rig.table(&entries, 2);
            let mut call = OpCall::new(Op::AttnDecode, vec![q.clone(), caches[l][0].0.clone(), caches[l][0].1.clone()], vec![o.clone()], attrs.clone());
            call.rows = Some(rig.tab(ptrs, 0, 0));
            call.pos = Some(post.clone());
            call.check().unwrap();
            let need = k.scratch_bytes(&ShapeInfo::from_call(&call));
            let buf = rig.be.alloc(need.max(16), MemKind::Device).unwrap();
            call.scratch = Some(TensorRef::new(buf, DType::U8, &[need]));
            calls.push(call);
        }
        let mut best = f64::MAX;
        for rep in 0..16 {
            rig.be.synchronize(0).unwrap();
            let t0 = std::time::Instant::now();
            for call in calls.iter() {
                rig.be.launch("attn_decode_bf16_splitkv_rows", call, 0).unwrap();
            }
            rig.be.synchronize(0).unwrap();
            let us = t0.elapsed().as_secs_f64() * 1e6 / calls.len() as f64;
            if rep > 0 {
                best = best.min(us);
            }
        }
        let bytes: usize = pos.iter().map(|p| 2 * nkv * (p + 1) * d * 2).sum();
        println!("ATTNT attn_decode_bf16_splitkv_rows M={m} pos~{base} {best:.1} us/call {:.0} GB/s", bytes as f64 / best / 1e3);
    }
}
