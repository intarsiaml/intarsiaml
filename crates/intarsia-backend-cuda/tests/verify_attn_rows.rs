//! Lane cvat (CUDA): the speculative verify pass's attention rows as ONE row-table launch (`Model::verify_attn_rows`) are BYTE-equal
//! to the per-row `attn_decode_bf16_splitkv` launches they replace (row `r` at position `pos0 + r`, the cache rows already appended).
//! Two position sources, as the model has them: a host-built position vector (an eager pass: `verify_rows_pos_bytes`, uploaded once per
//! `(pos0, t)`) and a captured pass's device position table (`ws.pos`, row `r` reads entry `r`; the per-row launches read entry `r` as a
//! one-element tensor), the latter inside a CUDA graph that is replayed at several positions against two different caches (two "states").
//! The K / V address tables are the model's own builder (`verify_rows_tab_bytes`).
//!
//! `INTARSIA_CVAT_TIMING=1` also prints how long a graph of 10 attention layers takes per-row against row-table (35B shapes, t = 2..4,
//! several positions): `cargo test -p intarsia-backend-cuda --test verify_attn_rows --release -- --nocapture`.
//!
//! Needs a GPU: without one the tests print SKIPPED (and fail under CI=1 / INTARSIA_REQUIRE_GPU=1).
#![cfg(ik_cuda)]

mod common;

use intarsia_core::backend::{Backend, MemKind, TensorRef};
use intarsia_core::model::{verify_rows_pos_bytes, verify_rows_pos_stale, verify_rows_tab_bytes};
use intarsia_core::ops::RowTab;
use intarsia_core::rng::SplitMix64;
use intarsia_core::{DType, HostTensor, Op, OpAttrs, OpCall, ShapeInfo};
use std::sync::Arc;

const POS_BYTES: usize = 256;
const REGION_BYTES: usize = 16 * 2 * 8;

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

    fn bf16(&self, vals: &[f32], shape: &[usize]) -> TensorRef {
        let t = HostTensor::from_f32_as(DType::Bf16, shape, vals);
        self.put_bytes(&t.data, t.dtype, &t.shape)
    }

    fn poison(&self, bytes: usize, shape: &[usize]) -> TensorRef {
        self.put_bytes(&vec![0xA5u8; bytes], DType::Bf16, shape)
    }

    fn down(&self, t: &TensorRef) -> Vec<u8> {
        let mut d = vec![0u8; t.nbytes()];
        self.be.synchronize(0).unwrap();
        self.be.download(&t.buf, t.offset, &mut d, 0).unwrap();
        d
    }

    /// Scratch the kernel needs for `call` (allocated once; the caller keeps it alive across a graph's replays).
    fn scratch(&self, kernel: &str, call: &OpCall) -> Option<TensorRef> {
        let k = self.be.kernels().into_iter().find(|k| k.name == kernel && k.op == call.op).unwrap_or_else(|| panic!("{kernel} not registered"));
        let need = k.scratch_bytes(&ShapeInfo::from_call(call));
        (need > 0).then(|| {
            let buf = self.be.alloc(need.max(16), MemKind::Device).unwrap();
            self.be.zero(&buf, 0).unwrap();
            TensorRef::new(buf, DType::U8, &[need])
        })
    }

    /// Launch without waiting (also what a graph capture records).
    fn enqueue(&self, kernel: &str, call: &OpCall) {
        call.check().unwrap_or_else(|e| panic!("{kernel}: check: {e}"));
        self.be.launch(kernel, call, 0).unwrap_or_else(|e| panic!("{kernel}: {e}"));
    }

    fn launch(&self, kernel: &str, mut call: OpCall) {
        call.scratch = self.scratch(kernel, &call);
        self.enqueue(kernel, &call);
        self.be.synchronize(0).unwrap();
    }
}

fn nasty(rng: &mut SplitMix64, n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let v = rng.normal_vec(1, 1.0)[0];
            match i % 17 {
                0 => v * 64.0,
                5 => v * 1e-3,
                _ => v,
            }
        })
        .collect()
}

/// Keys with scores of order one, so that every key of the prefix carries weight (a row that sees one key too few or too many differs).
fn kgen(rng: &mut SplitMix64, n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let v = rng.normal_vec(1, 1.0)[0];
            match i % 17 {
                0 => v * 4.0,
                5 => v * 1e-3,
                _ => v,
            }
        })
        .collect()
}

fn attrs(nh: usize, nkv: usize, d: usize, cap: usize, pos0: usize) -> OpAttrs {
    OpAttrs { pos0, scale: 1.0 / (d as f32).sqrt(), n_heads: nh, n_kv_heads: nkv, head_dim: d, kv_cap: cap, ..Default::default() }
}

/// The 35B's attention (16 q / 2 kv heads, head_dim 256), the 27B's per-rank shape (12 / 2 / 256) and the 8B (32 / 8 / 128).
const SHAPES: [(usize, usize, usize); 3] = [(16, 2, 256), (12, 2, 256), (32, 8, 128)];

#[test]
fn verify_rows_launch_is_the_per_row_launches() {
    let Some(rig) = Rig::new("verify_attn_rows") else { return };
    let mut rng = SplitMix64::new(28);
    let cap = 1500usize;
    let mut total = 0usize;
    for (nh, nkv, d) in SHAPES {
        let kc = rig.bf16(&kgen(&mut rng, nkv * cap * d), &[nkv, cap, d]);
        let vc = rig.bf16(&nasty(&mut rng, nkv * cap * d), &[nkv, cap, d]);
        let (ka, va) = (rig.be.device_address(&kc.buf).unwrap(), rig.be.device_address(&vc.buf).unwrap());
        // the model's buffer: positions at 0, layer 0's region after them (layer 3's region is used here, to cover the offset)
        let slot = 3usize;
        let buf = rig.be.alloc(POS_BYTES + 5 * REGION_BYTES, MemKind::Device).unwrap();
        rig.be.zero(&buf, 0).unwrap();
        let mut filled: Option<(u64, u64)> = None;
        let mut pos_key: Option<(usize, usize)> = None;
        // passes of different row counts and start positions, in the order a spec run produces them (t shrinks and grows at the same pos0)
        let passes: [(usize, usize); 14] = [
            (100, 3), (100, 8), (101, 2), (400, 8), (400, 16), (400, 5), (63, 8), (0, 8), (1, 2), (1000, 8), (1100, 16), (1100, 3), (1483, 16), (700, 7),
        ];
        for (pos0, t) in passes {
            if filled != Some((ka, va)) {
                rig.be.upload(&buf, POS_BYTES + slot * REGION_BYTES, &verify_rows_tab_bytes(ka, va, 16), 0).unwrap();
                filled = Some((ka, va));
            }
            if verify_rows_pos_stale(pos_key, pos0, t) {
                rig.be.upload(&buf, 0, &verify_rows_pos_bytes(pos0, t), 0).unwrap();
                pos_key = Some((pos0, t));
            }
            let q = rig.bf16(&nasty(&mut rng, t * nh * d), &[t, nh, d]);
            // the per-row launches the verify pass made before: row r alone at host position pos0 + r
            let mut want = Vec::new();
            for r in 0..t {
                let o = rig.poison(nh * d * 2, &[1, nh, d]);
                let solo = OpCall::new(Op::AttnDecode, vec![q.rows(r, 1), kc.clone(), vc.clone()], vec![o.clone()], attrs(nh, nkv, d, cap, pos0 + r));
                rig.launch("attn_decode_bf16_splitkv", solo);
                want.extend(rig.down(&o));
            }
            // the one row-table launch, built as `Model::verify_attn_rows` builds it
            let mut ptrs = TensorRef::new(buf.clone(), DType::U8, &[t, 2, 8]);
            ptrs.offset = POS_BYTES + slot * REGION_BYTES;
            let pos = TensorRef::new(buf.clone(), DType::I32, &[t]);
            let o = rig.poison(t * nh * d * 2, &[t, nh, d]);
            let mut call = OpCall::new(Op::AttnDecode, vec![q.clone(), kc.clone(), vc.clone()], vec![o.clone()], attrs(nh, nkv, d, cap, 0));
            call.pos = Some(pos);
            call.rows = Some(RowTab { ptrs, rows: t, stride: 2, index: 0, byte_off: 0 });
            rig.launch("attn_decode_bf16_splitkv_rows", call);
            let got = rig.down(&o);
            assert_eq!(got.len(), want.len());
            if let Some(i) = got.iter().zip(&want).position(|(a, b)| a != b) {
                panic!("nh={nh} nkv={nkv} d={d} pos0={pos0} t={t}: first differing byte {i} of {} (row-table {:#04x} vs per-row {:#04x})", got.len(), got[i], want[i]);
            }
            total += got.len();
        }
    }
    println!("cvat verify_attn_rows (host positions): {total} bytes identical to the per-row launches (3 shapes, 14 passes each, t 2..=16)");
}

/// One captured "layer": the per-row launches (row r reads device position entry r) or the one row-table launch, on `q` / `o` of a state.
struct Layer {
    q: TensorRef,
    o_rows: TensorRef,
    o_solo: TensorRef,
    kc: TensorRef,
    vc: TensorRef,
    tab: TensorRef,
}

#[test]
fn captured_rows_launch_is_the_captured_per_row_launches() {
    let Some(rig) = Rig::new("verify_attn_rows") else { return };
    let mut rng = SplitMix64::new(29);
    let cap = 1500usize;
    let t = 3usize;
    let mut total = 0usize;
    for (nh, nkv, d) in SHAPES {
        // two "states": own caches, own K / V address regions, one shared device position table (as `ws.pos`, here at byte offset 64, with
        // other entries before and after it that must not be read)
        let pos_buf = rig.be.alloc(256, MemKind::Device).unwrap();
        rig.be.zero(&pos_buf, 0).unwrap();
        let mut pos = TensorRef::new(pos_buf.clone(), DType::I32, &[t]);
        pos.offset = 64;
        let tabs = rig.be.alloc(2 * REGION_BYTES, MemKind::Device).unwrap();
        let mut layers = Vec::new();
        for st in 0..2usize {
            let kc = rig.bf16(&kgen(&mut rng, nkv * cap * d), &[nkv, cap, d]);
            let vc = rig.bf16(&nasty(&mut rng, nkv * cap * d), &[nkv, cap, d]);
            let (ka, va) = (rig.be.device_address(&kc.buf).unwrap(), rig.be.device_address(&vc.buf).unwrap());
            rig.be.upload(&tabs, st * REGION_BYTES, &verify_rows_tab_bytes(ka, va, 16), 0).unwrap();
            let mut tab = TensorRef::new(tabs.clone(), DType::U8, &[t, 2, 8]);
            tab.offset = st * REGION_BYTES;
            layers.push(Layer {
                q: rig.bf16(&nasty(&mut rng, t * nh * d), &[t, nh, d]),
                o_rows: rig.poison(t * nh * d * 2, &[t, nh, d]),
                o_solo: rig.poison(t * nh * d * 2, &[t, nh, d]),
                kc,
                vc,
                tab,
            });
        }
        let row_pos = |r: usize| {
            let mut v = TensorRef::new(pos_buf.clone(), DType::I32, &[1]);
            v.offset = 64 + 4 * r;
            v
        };
        // graphs: per state, the per-row launches and the row-table launch (both read the device positions)
        let mut graphs = Vec::new();
        // the calls hold the scratch buffers the graphs address: they must outlive the replays
        let mut keep: Vec<OpCall> = Vec::new();
        for (st, l) in layers.iter().enumerate() {
            let mut solo_calls = Vec::new();
            for r in 0..t {
                let mut c = OpCall::new(Op::AttnDecode, vec![l.q.rows(r, 1), l.kc.clone(), l.vc.clone()], vec![l.o_solo.rows(r, 1)], attrs(nh, nkv, d, cap, 0));
                c.pos = Some(row_pos(r));
                c.scratch = rig.scratch("attn_decode_bf16_splitkv", &c);
                solo_calls.push(c);
            }
            let mut rc = OpCall::new(Op::AttnDecode, vec![l.q.clone(), l.kc.clone(), l.vc.clone()], vec![l.o_rows.clone()], attrs(nh, nkv, d, cap, 0));
            rc.pos = Some(pos.clone());
            rc.rows = Some(RowTab { ptrs: l.tab.clone(), rows: t, stride: 2, index: 0, byte_off: 0 });
            rc.scratch = rig.scratch("attn_decode_bf16_splitkv_rows", &rc);
            let g_solo = rig.be.graph_capture(0, &mut || {
                for c in &solo_calls {
                    rig.enqueue("attn_decode_bf16_splitkv", c);
                }
                Ok(())
            });
            let g_rows = rig.be.graph_capture(0, &mut || {
                rig.enqueue("attn_decode_bf16_splitkv_rows", &rc);
                Ok(())
            });
            keep.extend(solo_calls);
            keep.push(rc);
            graphs.push((g_solo.unwrap_or_else(|e| panic!("state {st}: capture per-row: {e}")), g_rows.unwrap_or_else(|e| panic!("state {st}: capture rows: {e}"))));
        }
        // replays: positions change between replays, the two states alternate (a later state must not disturb an earlier state's graph)
        for (round, pos0) in [5usize, 100, 400, 401, 1000, 1495].into_iter().enumerate() {
            for (st, l) in layers.iter().enumerate().rev().chain(layers.iter().enumerate()) {
                let ps: Vec<u8> = (0..t).flat_map(|r| ((pos0 + r) as i32).to_le_bytes()).collect();
                rig.be.upload(&pos_buf, 64, &ps, 0).unwrap();
                for (o, g) in [(&l.o_solo, &graphs[st].0), (&l.o_rows, &graphs[st].1)] {
                    rig.be.upload(&o.buf, o.offset, &vec![0xA5u8; o.nbytes()], 0).unwrap();
                    rig.be.graph_replay(g, 0).unwrap();
                }
                let (want, got) = (rig.down(&l.o_solo), rig.down(&l.o_rows));
                if let Some(i) = got.iter().zip(&want).position(|(a, b)| a != b) {
                    panic!("nh={nh} nkv={nkv} d={d} state {st} round {round} pos0={pos0}: first differing byte {i} (row-table {:#04x} vs per-row {:#04x})", got[i], want[i]);
                }
                total += got.len();
            }
        }
    }
    println!("cvat verify_attn_rows (captured, device positions, two states): {total} bytes identical (3 shapes, 12 replays each)");
}

/// Timing of what a captured pass spends in attention: 10 layers (the 35B's full-attention layers, own caches) as per-row launches or as one
/// row-table launch each, replayed as graphs. Prints only with `INTARSIA_CVAT_TIMING=1`.
#[test]
fn timing_captured_attention_layers() {
    if std::env::var("INTARSIA_CVAT_TIMING").map_or(true, |v| v == "0") {
        return;
    }
    let Some(rig) = Rig::new("verify_attn_rows") else { return };
    let mut rng = SplitMix64::new(30);
    let (nh, nkv, d, cap, nl) = (16usize, 2usize, 256usize, 8192usize, 10usize);
    let kcs: Vec<(TensorRef, TensorRef)> = (0..nl)
        .map(|_| (rig.bf16(&kgen(&mut rng, nkv * cap * d), &[nkv, cap, d]), rig.bf16(&nasty(&mut rng, nkv * cap * d), &[nkv, cap, d])))
        .collect();
    let tabs = rig.be.alloc(nl * REGION_BYTES, MemKind::Device).unwrap();
    for (i, (k, v)) in kcs.iter().enumerate() {
        let (ka, va) = (rig.be.device_address(&k.buf).unwrap(), rig.be.device_address(&v.buf).unwrap());
        rig.be.upload(&tabs, i * REGION_BYTES, &verify_rows_tab_bytes(ka, va, 16), 0).unwrap();
    }
    let pos_buf = rig.be.alloc(256, MemKind::Device).unwrap();
    for t in [2usize, 3, 4, 8] {
        let q = rig.bf16(&nasty(&mut rng, t * nh * d), &[t, nh, d]);
        let o = rig.poison(t * nh * d * 2, &[t, nh, d]);
        let mut pos = TensorRef::new(pos_buf.clone(), DType::I32, &[t]);
        pos.offset = 0;
        let row_pos = |r: usize| {
            let mut v = TensorRef::new(pos_buf.clone(), DType::I32, &[1]);
            v.offset = 4 * r;
            v
        };
        let mut solo = Vec::new();
        let mut rows = Vec::new();
        for (i, (k, v)) in kcs.iter().enumerate() {
            for r in 0..t {
                let mut c = OpCall::new(Op::AttnDecode, vec![q.rows(r, 1), k.clone(), v.clone()], vec![o.rows(r, 1)], attrs(nh, nkv, d, cap, 0));
                c.pos = Some(row_pos(r));
                c.scratch = rig.scratch("attn_decode_bf16_splitkv", &c);
                solo.push(c);
            }
            let mut tab = TensorRef::new(tabs.clone(), DType::U8, &[t, 2, 8]);
            tab.offset = i * REGION_BYTES;
            let mut c = OpCall::new(Op::AttnDecode, vec![q.clone(), k.clone(), v.clone()], vec![o.clone()], attrs(nh, nkv, d, cap, 0));
            c.pos = Some(pos.clone());
            c.rows = Some(RowTab { ptrs: tab, rows: t, stride: 2, index: 0, byte_off: 0 });
            c.scratch = rig.scratch("attn_decode_bf16_splitkv_rows", &c);
            rows.push(c);
        }
        let g_solo = rig.be.graph_capture(0, &mut || solo.iter().try_for_each(|c| Ok(rig.enqueue("attn_decode_bf16_splitkv", c)))).unwrap();
        let g_rows = rig.be.graph_capture(0, &mut || rows.iter().try_for_each(|c| Ok(rig.enqueue("attn_decode_bf16_splitkv_rows", c)))).unwrap();
        for pos0 in [200usize, 600, 1500, 3000, 6000] {
            let ps: Vec<u8> = (0..t).flat_map(|r| ((pos0 + r) as i32).to_le_bytes()).collect();
            rig.be.upload(&pos_buf, 0, &ps, 0).unwrap();
            let time = |g: &intarsia_core::backend::GraphHandle| {
                let mut best = f64::MAX;
                for _ in 0..5 {
                    for _ in 0..5 {
                        rig.be.graph_replay(g, 0).unwrap();
                    }
                    rig.be.synchronize(0).unwrap();
                    let t0 = std::time::Instant::now();
                    for _ in 0..40 {
                        rig.be.graph_replay(g, 0).unwrap();
                    }
                    rig.be.synchronize(0).unwrap();
                    best = best.min(t0.elapsed().as_secs_f64() / 40.0);
                }
                best * 1e6
            };
            let (a, b) = (time(&g_solo), time(&g_rows));
            println!("cvat timing t={t} pos0={pos0}: {nl} layers per-row {a:.0} us, row-table {b:.0} us, saved {:.0} us ({:.1} %)", a - b, 100.0 * (a - b) / a);
        }
    }
}
