//! Lane vat (HIP): the speculative verify pass's attention rows as ONE row-table launch (`Model::verify_attn_rows`) are BYTE-equal to the
//! per-row `attn_decode_bf16_splitkv` launches they replace (row `r` at position `pos0 + r`, host position, the cache rows already
//! appended). The tables are built with the model's own builders (`verify_rows_tab_bytes`, `verify_rows_pos_bytes`,
//! `verify_rows_pos_stale`): one shared K / V cache named by every row of a layer's region, a position vector shared across layers
//! and rewritten only when `(pos0, t)` changes, passes of different row counts one after the other against the same device buffer.
//!
//! Planted faults (`INTARSIA_HIP_VAT_MUT` = 1..4, `verify_attn.rs:mutant`) must make this test fail; `docs/results/2026-10-28-vat-verify-attn-rows.md`
//! lists the run.
//!
//! Needs a GPU: without one the test prints SKIPPED (and fails under CI=1 / INTARSIA_REQUIRE_GPU=1).
#![cfg(ik_hip)]

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

    fn bf16(&self, vals: &[f32], shape: &[usize]) -> TensorRef {
        let t = HostTensor::from_f32_as(DType::Bf16, shape, vals);
        self.put_bytes(&t.data, t.dtype, &t.shape)
    }

    fn poison(&self, bytes: usize, shape: &[usize]) -> TensorRef {
        self.put_bytes(&vec![0xA5u8; bytes], DType::Bf16, shape)
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

#[test]
fn verify_rows_launch_is_the_per_row_launches() {
    let Some(rig) = Rig::new("verify_attn_rows") else { return };
    let mut rng = SplitMix64::new(28);
    let cap = 1500usize;
    let mut total = 0usize;
    // 27B-like per-rank shapes (12 q / 2 kv heads and 24 / 4, head_dim 256), the 8B (32, 8, 128)
    for (nh, nkv, d) in [(12usize, 2usize, 256usize), (24, 4, 256), (32, 8, 128)] {
        let kc = rig.bf16(&nasty(&mut rng, nkv * cap * d), &[nkv, cap, d]);
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
            let scale = 1.0 / (d as f32).sqrt();
            // the per-row launches the verify pass made before: row r alone at host position pos0 + r
            let mut want = Vec::new();
            for r in 0..t {
                let o = rig.poison(nh * d * 2, &[1, nh, d]);
                let attrs = OpAttrs { pos0: pos0 + r, scale, n_heads: nh, n_kv_heads: nkv, head_dim: d, kv_cap: cap, ..Default::default() };
                let solo = OpCall::new(Op::AttnDecode, vec![q.rows(r, 1), kc.clone(), vc.clone()], vec![o.clone()], attrs);
                rig.launch("attn_decode_bf16_splitkv", solo);
                want.extend(rig.down(&o));
            }
            // the one row-table launch, built as `Model::verify_attn_rows` builds it
            let mut ptrs = TensorRef::new(buf.clone(), DType::U8, &[t, 2, 8]);
            ptrs.offset = POS_BYTES + slot * REGION_BYTES;
            let pos = TensorRef::new(buf.clone(), DType::I32, &[t]);
            let o = rig.poison(t * nh * d * 2, &[t, nh, d]);
            let attrs = OpAttrs { pos0: pos0 + t - 1, scale, n_heads: nh, n_kv_heads: nkv, head_dim: d, kv_cap: cap, ..Default::default() };
            let mut call = OpCall::new(Op::AttnDecode, vec![q.clone(), kc.clone(), vc.clone()], vec![o.clone()], attrs);
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
    println!("vat verify_attn_rows: {total} bytes identical to the per-row launches (3 shapes, 14 passes each, t 2..=16)");
}
