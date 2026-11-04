//! The bitwise head split (`INTARSIA_METAL_ATTN_HSB=<n>`, attn.rs `head_plan`): for the Qwen3.8-27B full attention (24 query
//! heads over 4 kv heads, head_dim 256, a group of 6) the group runs in n threadgroups of 6 / n heads with the work partition
//! of the unsplit launch, so the output is bit for bit the unsplit launch's, for the 1-row decode launch (host and device
//! position), the multi-row verify launch and the row-table launch of batched decode (`attn_decode_bf16_splitkv_rows`, rows on
//! different caches and positions, also bit for bit the solo launch of each row). Other shapes ignore the setting. One test
//! function: the override is process-wide.

#![cfg(target_os = "macos")]

mod common;

use common::*;
use intarsia_backend_metal::kernels::{ATTN_DECODE_BF16_SPLITKV, ATTN_PREFILL_BF16_SPLITKV_ROWS};
use intarsia_backend_metal::{head_plan_for_test, set_hsb_for_test, MetalBackend};
use intarsia_core::backend::{Backend, Buffer, MemKind, TensorRef};
use intarsia_core::ops::RowTab;
use intarsia_core::{DType, Op, OpAttrs, OpCall, ShapeInfo};

const ATTN_DECODE_ROWTAB: &str = "attn_decode_bf16_splitkv_rows";

fn tr(b: &Buffer, off: usize, shape: &[usize]) -> TensorRef {
    TensorRef { buf: b.clone(), offset: off, dtype: DType::Bf16, shape: shape.to_vec() }
}

fn read_u16(be: &MetalBackend, b: &Buffer, n: usize) -> Vec<u16> {
    let mut raw = vec![0u8; n * 2];
    be.download(b, 0, &mut raw, 0).unwrap();
    raw.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect()
}

fn scratch_for(be: &MetalBackend, kernel: &str, cl: &OpCall) -> Option<TensorRef> {
    let need = be.kernels().iter().find(|k| k.name == kernel).map_or(0, |k| k.scratch_bytes(&ShapeInfo::from_call(cl)));
    (need > 0).then(|| {
        let b = be.alloc(need, MemKind::Device).unwrap();
        be.upload(&b, 0, &vec![0xEEu8; need], 0).unwrap();
        TensorRef { buf: b, offset: 0, dtype: DType::U8, shape: vec![need] }
    })
}

struct Case {
    be: MetalBackend,
    n: usize,
    nkv: usize,
    d: usize,
    cap: usize,
    kb: Buffer,
    vb: Buffer,
    /// further caches (other data) for the rows of a row-table launch; row r uses cache `r % (1 + more.len())`
    more: Vec<(Buffer, Buffer)>,
    qb: Buffer,
}

impl Case {
    fn new(n: usize, nkv: usize, d: usize, cap: usize) -> Case {
        let be = backend();
        let kb = upload(&be, &rand_bf16(nkv * cap * d, 11 + d as u64));
        let vb = upload(&be, &rand_bf16(nkv * cap * d, 12 + d as u64));
        let qb = upload(&be, &rand_bf16(16 * n * d, 13 + n as u64));
        let more = (1..3u64)
            .map(|i| (upload(&be, &rand_bf16(nkv * cap * d, 21 + i + d as u64)), upload(&be, &rand_bf16(nkv * cap * d, 31 + i + d as u64))))
            .collect();
        Case { be, n, nkv, d, cap, kb, vb, more, qb }
    }
    fn attrs(&self, pos0: usize) -> OpAttrs {
        OpAttrs {
            pos0,
            scale: 1.0 / (self.d as f32).sqrt(),
            n_heads: self.n,
            n_kv_heads: self.nkv,
            head_dim: self.d,
            kv_cap: self.cap,
            ..Default::default()
        }
    }
    fn kv(&self) -> Vec<TensorRef> {
        self.kv_of(0)
    }
    /// K / V of cache `ci % 3` (0 = the main one).
    fn kv_of(&self, ci: usize) -> Vec<TensorRef> {
        let ci = ci % (1 + self.more.len());
        let (k, v) = if ci == 0 { (&self.kb, &self.vb) } else { (&self.more[ci - 1].0, &self.more[ci - 1].1) };
        vec![tr(k, 0, &[self.nkv, self.cap, self.d]), tr(v, 0, &[self.nkv, self.cap, self.d])]
    }
    fn dev_pos(&self, v: i32) -> TensorRef {
        let pb = self.be.alloc(4, MemKind::Device).unwrap();
        self.be.upload(&pb, 0, &v.to_le_bytes(), 0).unwrap();
        TensorRef { buf: pb, offset: 0, dtype: DType::I32, shape: vec![1] }
    }
    /// 1-row decode launch at position `pos` (host position, or the device position tensor when `dev`).
    fn one(&self, pos: usize, dev: bool) -> Vec<u16> {
        self.one_on(0, 0, pos, dev)
    }
    /// The same on cache `ci` with query row `qr` of `qb`.
    fn one_on(&self, ci: usize, qr: usize, pos: usize, dev: bool) -> Vec<u16> {
        let len = self.n * self.d;
        let out = self.be.alloc(len * 2, MemKind::Device).unwrap();
        let mut inputs = vec![tr(&self.qb, qr * self.n * self.d * 2, &[1, self.n, self.d])];
        inputs.extend(self.kv_of(ci));
        let mut cl = OpCall::new(Op::AttnDecode, inputs, vec![tr(&out, 0, &[1, self.n, self.d])], self.attrs(if dev { 0 } else { pos }));
        cl.scratch = scratch_for(&self.be, ATTN_DECODE_BF16_SPLITKV, &cl);
        if dev {
            cl.pos = Some(self.dev_pos(pos as i32));
        }
        self.be.launch(ATTN_DECODE_BF16_SPLITKV, &cl, 0).unwrap();
        self.be.synchronize(0).unwrap();
        read_u16(&self.be, &out, len)
    }
    /// Row-table launch (batched decode): row r decodes at position `pos[r]` on cache `r % 3`, query row r of `qb`.
    fn rowtab(&self, pos: &[usize]) -> Vec<u16> {
        let rows = pos.len();
        let len = rows * self.n * self.d;
        let out = self.be.alloc(len * 2, MemKind::Device).unwrap();
        let mut bytes = vec![0u8; rows * 2 * 8];
        for r in 0..rows {
            for (j, t) in self.kv_of(r).iter().enumerate() {
                let a = self.be.device_address(&t.buf).expect("device address");
                bytes[(r * 2 + j) * 8..(r * 2 + j) * 8 + 8].copy_from_slice(&a.to_le_bytes());
            }
        }
        let tb = self.be.alloc(bytes.len(), MemKind::Device).unwrap();
        self.be.upload(&tb, 0, &bytes, 0).unwrap();
        let ptrs = TensorRef { buf: tb, offset: 0, dtype: DType::U8, shape: vec![rows, 2, 8] };
        let kv0 = self.kv_of(0);
        let mut cl = OpCall::new(
            Op::AttnDecode,
            vec![tr(&self.qb, 0, &[rows, self.n, self.d]), kv0[0].clone(), kv0[1].clone()],
            vec![tr(&out, 0, &[rows, self.n, self.d])],
            self.attrs(0),
        );
        cl.rows = Some(RowTab { ptrs, rows, stride: 2, index: 0, byte_off: 0 });
        cl.pos = Some({
            let pb = self.be.alloc(rows * 4, MemKind::Device).unwrap();
            let v: Vec<u8> = pos.iter().flat_map(|p| (*p as i32).to_le_bytes()).collect();
            self.be.upload(&pb, 0, &v, 0).unwrap();
            TensorRef { buf: pb, offset: 0, dtype: DType::I32, shape: vec![rows] }
        });
        cl.scratch = scratch_for(&self.be, ATTN_DECODE_ROWTAB, &cl);
        cl.check().unwrap();
        self.be.launch(ATTN_DECODE_ROWTAB, &cl, 0).unwrap();
        self.be.synchronize(0).unwrap();
        read_u16(&self.be, &out, len)
    }
    /// k-row verify launch at `pos0` (host position, or the device position tensor when `dev`).
    fn rows(&self, k: usize, pos0: usize, dev: bool) -> Vec<u16> {
        let len = k * self.n * self.d;
        let out = self.be.alloc(len * 2, MemKind::Device).unwrap();
        let mut inputs = vec![tr(&self.qb, 0, &[k, self.n, self.d])];
        inputs.extend(self.kv());
        let mut cl = OpCall::new(Op::AttnPrefill, inputs, vec![tr(&out, 0, &[k, self.n, self.d])], self.attrs(if dev { 0 } else { pos0 }));
        cl.scratch = scratch_for(&self.be, ATTN_PREFILL_BF16_SPLITKV_ROWS, &cl);
        if dev {
            cl.pos = Some(self.dev_pos(pos0 as i32));
            self.be.launch_attn_rows(&cl).unwrap();
        } else {
            self.be.launch(ATTN_PREFILL_BF16_SPLITKV_ROWS, &cl, 0).unwrap();
        }
        self.be.synchronize(0).unwrap();
        read_u16(&self.be, &out, len)
    }
}

/// Positions around every change of the partition at 4 kv heads (nsplit = min(ceil(v / 128), 30): a new split every 128 keys
/// up to 3840 visible, then a constant 30 with the chunk growing), the 1-row simdgroup rule (the few-threadgroups branch
/// is irrelevant for the group of 6: the unsplit launch is limited to 2 simdgroups either way) and the cache ends.
fn positions(cap: usize, k: usize) -> Vec<usize> {
    let mut v: Vec<usize> = vec![0, 1, 7, 31, 32, 33, 126, 127, 128, 129, 255, 256, 1000, 1023, 1024, 1025];
    for b in [3712usize, 3840, 3968, 4096, 4097, 5000, 8191, 8192, 12000, 16000] {
        v.extend([b - 1, b, b + 1]);
    }
    v.extend([cap / 2, cap - k, cap - k - 1]);
    v.retain(|p| *p + k <= cap);
    v.sort();
    v.dedup();
    v
}

#[test]
fn bitwise_head_split_is_bit_identical_to_the_unsplit_launch() {
    // Qwen3.8-27B: 24 / 4 / 256, a group of 6; the split factors the group allows (2: GP 4, 3: GP 2, 6: GP 1)
    let c = Case::new(24, 4, 256, 16384 + 8);
    let mut windows = 0;
    for dev in [false, true] {
        for pos in positions(c.cap, 1) {
            set_hsb_for_test(Some(0));
            let want = c.one(pos, dev);
            for hsb in [2usize, 3, 6] {
                set_hsb_for_test(Some(hsb));
                assert!(c.one(pos, dev) == want, "1-row dev={dev} pos {pos} HSB={hsb}: differs from the unsplit launch");
                windows += 1;
            }
        }
    }
    for dev in [false, true] {
        for k in [2usize, 5, 8] {
            for pos in positions(c.cap, k) {
                set_hsb_for_test(Some(0));
                let want = c.rows(k, pos, dev);
                for hsb in [2usize, 3, 6] {
                    set_hsb_for_test(Some(hsb));
                    assert!(c.rows(k, pos, dev) == want, "rows k {k} dev={dev} pos {pos} HSB={hsb}: differs from the unsplit launch");
                    windows += 1;
                }
            }
        }
    }
    // the row-table launch of batched decode: R rows, row r at position `ps[(w + 7 r) % len]` on cache r % 3 (so the rows of one
    // launch differ in length, split count and data), compared with the unsplit row-table launch, and the unsplit launch with
    // each row's solo 1-row launch (unsplit too; the rows_exact contract, so the split launch is also the solo launch's bits)
    let ps = positions(c.cap, 1);
    let mut rt_windows = 0;
    for r in [1usize, 2, 3, 4, 16] {
        for w in 0..ps.len() {
            let pos: Vec<usize> = (0..r).map(|i| ps[(w + 7 * i) % ps.len()]).collect();
            set_hsb_for_test(Some(0));
            let want = c.rowtab(&pos);
            let rl = c.n * c.d;
            for (i, p) in pos.iter().enumerate() {
                assert!(want[i * rl..(i + 1) * rl] == c.one_on(i, i, *p, false)[..], "rowtab R={r} row {i} pos {p}: unsplit row-table row differs from its solo launch");
            }
            for hsb in [2usize, 3, 6] {
                set_hsb_for_test(Some(hsb));
                assert!(c.rowtab(&pos) == want, "rowtab R={r} pos {pos:?} HSB={hsb}: differs from the unsplit row-table launch");
                rt_windows += 1;
            }
        }
    }
    eprintln!("hs_bitwise: {rt_windows} row-table (rows, positions, HSB) windows bit-identical to the unsplit row-table launch");

    // the split is live for this shape (not silently ignored), and only for it
    for (hsb, want) in [(0usize, (1usize, 0usize)), (2, (2, 8)), (3, (3, 8)), (6, (6, 8)), (4, (1, 0))] {
        set_hsb_for_test(Some(hsb));
        assert_eq!(head_plan_for_test(6, 256), want, "HSB={hsb}");
        assert_eq!(head_plan_for_test(8, 256), (4, 0), "the 35B shape keeps its head split (HSB={hsb})");
        assert_eq!(head_plan_for_test(4, 128), (1, 0), "the 8B shape (HSB={hsb})");
        assert_eq!(head_plan_for_test(6, 128), (1, 0), "group 6 at head_dim 128 (HSB={hsb})");
    }
    // the shipped default (no override, variable unset): the bitwise split is on for the 27B shape; Some(0) above is the opt-out
    // and gave the unsplit launch the comparisons were made against
    set_hsb_for_test(None);
    if std::env::var_os("INTARSIA_METAL_ATTN_HSB").is_none() && std::env::var_os("INTARSIA_METAL_ATTN_HS").is_none() {
        assert_eq!(head_plan_for_test(6, 256), (3, 8), "default");
    }
    eprintln!("hs_bitwise: {windows} (pos, HSB) windows bit-identical to the unsplit 24/4/256 launch");

    // every other shape ignores the setting: the 8B (32 / 8 / 128), the 35B group of 8 (16 / 2 / 256, HS 4 by default), a
    // group of 6 at head_dim 128
    for (n, nkv, d) in [(32usize, 8usize, 128usize), (16, 2, 256), (12, 2, 128)] {
        let c = Case::new(n, nkv, d, 4104);
        for pos in [0usize, 130, 1500, 4000] {
            set_hsb_for_test(Some(0));
            let (w1, wr) = (c.one(pos, false), c.rows(4, pos, false));
            set_hsb_for_test(Some(3));
            assert!(c.one(pos, false) == w1, "{n}/{nkv}/{d} 1-row pos {pos}: HSB changed a non-27B shape");
            assert!(c.rows(4, pos, false) == wr, "{n}/{nkv}/{d} rows pos {pos}: HSB changed a non-27B shape");
        }
        for pos in [[0usize, 130, 1500, 4000], [4000, 1, 2000, 129], [127, 128, 3999, 600]] {
            set_hsb_for_test(Some(0));
            let wt = c.rowtab(&pos);
            set_hsb_for_test(Some(3));
            assert!(c.rowtab(&pos) == wt, "{n}/{nkv}/{d} row-table {pos:?}: HSB changed a non-27B shape");
        }
    }
    set_hsb_for_test(None);
}
