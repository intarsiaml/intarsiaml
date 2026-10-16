//! G0b: the multi-row verification attention (`attn_prefill_bf16_splitkv_rows`, `AttnPrefill` with T = 2..=8) is bitwise the
//! 1-row serving decode attention (`attn_decode_bf16_splitkv` with the host position `pos0 + r`, what the eager Metal decode
//! step launches) per row, on the same KV cache. Random data fills the whole cache, positions past `pos0 + r` included, so a
//! row that reads a key it must not see changes bits.

#![cfg(target_os = "macos")]

mod common;

use common::*;
use intarsia_backend_metal::kernels::{ATTN_DECODE_BF16, ATTN_DECODE_BF16_SPLITKV, ATTN_PREFILL_BF16_SPLITKV_ROWS};
use intarsia_backend_metal::MetalBackend;
use intarsia_core::backend::{Backend, Buffer, MemKind, TensorRef};
use intarsia_core::{DType, Op, OpAttrs, OpCall, ShapeInfo};

fn upload_at(be: &MetalBackend, off: usize, data: &[u16]) -> Buffer {
    let b = be.alloc(off + data.len() * 2 + 16, MemKind::Device).unwrap();
    be.upload(&b, 0, &vec![0xEEu8; off + data.len() * 2 + 16], 0).unwrap();
    be.upload(&b, off, as_bytes(data), 0).unwrap();
    b
}

fn read_u16(be: &MetalBackend, b: &Buffer, off: usize, n: usize) -> Vec<u16> {
    let mut raw = vec![0u8; n * 2];
    be.download(b, off, &mut raw, 0).unwrap();
    raw.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect()
}

fn tr(b: &Buffer, off: usize, shape: &[usize]) -> TensorRef {
    TensorRef { buf: b.clone(), offset: off, dtype: DType::Bf16, shape: shape.to_vec() }
}

fn scratch_for(be: &MetalBackend, kernel: &str, cl: &OpCall) -> Option<TensorRef> {
    let need = be.kernels().iter().find(|k| k.name == kernel).map_or(0, |k| k.scratch_bytes(&ShapeInfo::from_call(cl)));
    (need > 0).then(|| {
        let b = be.alloc(need, MemKind::Device).unwrap();
        be.upload(&b, 0, &vec![0xEEu8; need], 0).unwrap();
        TensorRef { buf: b, offset: 0, dtype: DType::U8, shape: vec![need] }
    })
}

struct Cache {
    be: MetalBackend,
    n: usize,
    nkv: usize,
    d: usize,
    cap: usize,
    off: usize,
    kb: Buffer,
    vb: Buffer,
}

impl Cache {
    fn new(n: usize, nkv: usize, d: usize, cap: usize, off: usize, seed: u64) -> Cache {
        let be = backend();
        let k = rand_bf16(nkv * cap * d, seed);
        let v = rand_bf16(nkv * cap * d, seed + 1);
        let (kb, vb) = (upload_at(&be, off, &k), upload_at(&be, off, &v));
        Cache { be, n, nkv, d, cap, off, kb, vb }
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
        vec![tr(&self.kb, self.off, &[self.nkv, self.cap, self.d]), tr(&self.vb, self.off, &[self.nkv, self.cap, self.d])]
    }

    /// Multi-row call on `k` rows of `q` (q buffer at byte offset `qoff`), output rows as u16.
    fn rows(&self, q: &Buffer, qoff: usize, k: usize, pos0: usize) -> Vec<u16> {
        let len = k * self.n * self.d;
        let out = self.be.alloc(len * 2 + 64, MemKind::Device).unwrap();
        self.be.upload(&out, 0, &vec![0xEEu8; len * 2 + 64], 0).unwrap();
        let mut inputs = vec![tr(q, qoff, &[k, self.n, self.d])];
        inputs.extend(self.kv());
        let mut cl = OpCall::new(Op::AttnPrefill, inputs, vec![tr(&out, 16, &[k, self.n, self.d])], self.attrs(pos0));
        cl.scratch = scratch_for(&self.be, ATTN_PREFILL_BF16_SPLITKV_ROWS, &cl);
        self.be.launch(ATTN_PREFILL_BF16_SPLITKV_ROWS, &cl, 0).unwrap();
        self.be.synchronize(0).unwrap();
        let got = read_u16(&self.be, &out, 16, len);
        // guard: the 16 bytes before and after the output stay untouched
        let head = read_u16(&self.be, &out, 0, 8);
        let tail = read_u16(&self.be, &out, 16 + len * 2, 24);
        assert!(head.iter().all(|v| *v == 0xEEEE) && tail[..24 - 8].iter().all(|v| *v == 0xEEEE), "guard overwritten");
        got
    }

    /// The serving 1-row launch at host position `pos`, query row `r` of `q`.
    fn one_row(&self, q: &Buffer, qoff: usize, r: usize, pos: usize) -> Vec<u16> {
        let len = self.n * self.d;
        let out = self.be.alloc(len * 2, MemKind::Device).unwrap();
        let mut inputs = vec![tr(q, qoff + r * len * 2, &[1, self.n, self.d])];
        inputs.extend(self.kv());
        let mut cl = OpCall::new(Op::AttnDecode, inputs, vec![tr(&out, 0, &[1, self.n, self.d])], self.attrs(pos));
        cl.scratch = scratch_for(&self.be, ATTN_DECODE_BF16_SPLITKV, &cl);
        self.be.launch(ATTN_DECODE_BF16_SPLITKV, &cl, 0).unwrap();
        self.be.synchronize(0).unwrap();
        read_u16(&self.be, &out, 0, len)
    }

    fn check(&self, q: &Buffer, qoff: usize, k: usize, pos0: usize) {
        let got = self.rows(q, qoff, k, pos0);
        let len = self.n * self.d;
        for r in 0..k {
            let want = self.one_row(q, qoff, r, pos0 + r);
            assert!(
                got[r * len..(r + 1) * len] == want[..],
                "shape {}/{}/{} cap {} pos0 {pos0} k {k}: row {r} (position {}) differs from the 1-row launch",
                self.n,
                self.nkv,
                self.d,
                self.cap,
                pos0 + r
            );
        }
    }
}

/// Positions whose row windows straddle every change of the partition: split count (every 128 keys up to 15 splits at 1793),
/// the simdgroup count (visible 1024 -> 1025), the chunk (roundup32(ceil(v / nsplit)) once nsplit = 15, every 480 keys), the
/// first / last cache row, plus a spread.
fn positions(cap: usize, k: usize) -> Vec<usize> {
    let mut v: Vec<usize> = vec![0, 1, 2, 7, 31, 32, 33];
    for b in (128usize..=2048).step_by(128).chain([1024, 1025, 1792, 1793, 480, 960, 1440, 1920, 2400, 2880, 3360, 3840, 4320, 4800, 8160, 8640]) {
        for dlt in 0..=k + 1 {
            v.push(b.saturating_sub(dlt));
        }
        v.push(b);
    }
    v.extend([cap / 3, cap / 2, cap - k, cap - k - 1, cap - 2 * k]);
    v.retain(|p| *p + k <= cap);
    v.sort();
    v.dedup();
    v
}

fn sweep(n: usize, nkv: usize, d: usize, cap: usize, ks: &[usize], off: usize, qoff: usize, stride: usize) {
    let c = Cache::new(n, nkv, d, cap, off, 4242 + (n + d + cap) as u64);
    let q = rand_bf16(8 * n * d, 99 + d as u64);
    let qb = upload_at(&c.be, qoff, &q);
    let mut checked = 0;
    for &k in ks {
        for (i, pos0) in positions(cap, k).into_iter().enumerate() {
            if i % stride != 0 {
                continue;
            }
            c.check(&qb, qoff, k, pos0);
            checked += 1;
        }
    }
    eprintln!("g0b attn: {n}/{nkv}/{d} cap {cap}: {checked} (k, pos0) windows bitwise equal to the 1-row launches");
}

#[test]
fn rows_bitwise_qwen3_8b_all_k() {
    // 32 q heads / 8 kv heads / head_dim 128, k = 2..=8, every boundary window of a 4096-key cache
    sweep(32, 8, 128, 4096, &[2, 3, 4, 5, 6, 7, 8], 0, 0, 1);
}

#[test]
fn rows_bitwise_qwen3_8b_long_context() {
    // 16k cache: 15 splits, chunk changes, the last rows of the cache
    sweep(32, 8, 128, 16384, &[2, 4, 6, 8], 0, 0, 1);
}

#[test]
fn rows_bitwise_other_shapes() {
    // group 1, group 2, group 8 (head_dim 256: 2 simdgroups), group 3 (GP 4 with an idle head), small head dims
    for (n, nkv, d) in [(8, 8, 64), (8, 2, 64), (16, 2, 256), (12, 4, 32), (6, 2, 128), (8, 1, 16)] {
        sweep(n, nkv, d, 2304, &[2, 3, 5, 8], 0, 0, 3);
    }
}

#[test]
fn rows_bitwise_with_cache_and_q_offsets() {
    // 16-byte aligned non-zero offsets of the tensors (the layouts a state slice produces)
    sweep(32, 8, 128, 2048, &[2, 4, 8], 32, 48, 4);
}

#[test]
fn rows_misaligned_operands_fall_back_to_the_single_pass_rows() {
    // q at a 2-byte offset: the 1-row launcher runs the single-pass kernel, so do the rows
    let c = Cache::new(32, 8, 128, 1024, 0, 31);
    let q = rand_bf16(4 * 32 * 128, 5);
    let qb = upload_at(&c.be, 2, &q);
    for pos0 in [0usize, 100, 600, 1019] {
        let k = 4;
        let got = c.rows(&qb, 2, k, pos0);
        let len = 32 * 128;
        for r in 0..k {
            let out = c.be.alloc(len * 2, MemKind::Device).unwrap();
            let mut inputs = vec![tr(&qb, 2 + r * len * 2, &[1, 32, 128])];
            inputs.extend(c.kv());
            let mut cl = OpCall::new(Op::AttnDecode, inputs, vec![tr(&out, 0, &[1, 32, 128])], c.attrs(pos0 + r));
            cl.scratch = scratch_for(&c.be, ATTN_DECODE_BF16_SPLITKV, &cl);
            c.be.launch(ATTN_DECODE_BF16_SPLITKV, &cl, 0).unwrap();
            c.be.synchronize(0).unwrap();
            assert_eq!(got[r * len..(r + 1) * len], read_u16(&c.be, &out, 0, len)[..], "pos0 {pos0} row {r}");
        }
    }
}

#[test]
fn rows_device_position_is_the_device_position_one_row_launch() {
    // The core cannot carry a device position on AttnPrefill (OpCall::check), so this goes through the launcher directly:
    // with a position tensor, row r is bitwise the 1-row launch with the same tensor value + r (4 simdgroups at every
    // context, the device-position rule of the 1-row launcher)
    let c = Cache::new(32, 8, 128, 4096, 0, 77);
    let q = rand_bf16(8 * 32 * 128, 8);
    let qb = upload_at(&c.be, 0, &q);
    let len = 32 * 128;
    let mk_pos = |v: i32| {
        let pb = c.be.alloc(4, MemKind::Device).unwrap();
        c.be.upload(&pb, 0, &v.to_le_bytes(), 0).unwrap();
        TensorRef { buf: pb, offset: 0, dtype: DType::I32, shape: vec![1] }
    };
    for k in [2usize, 5, 8] {
        for pos in [0usize, 126, 127, 700, 1023, 1024, 1900, 3000, 4096 - k] {
            let out = c.be.alloc(k * len * 2, MemKind::Device).unwrap();
            let mut inputs = vec![tr(&qb, 0, &[k, 32, 128])];
            inputs.extend(c.kv());
            let mut cl = OpCall::new(Op::AttnPrefill, inputs, vec![tr(&out, 0, &[k, 32, 128])], c.attrs(0));
            cl.scratch = scratch_for(&c.be, ATTN_PREFILL_BF16_SPLITKV_ROWS, &cl);
            cl.pos = Some(mk_pos(pos as i32));
            c.be.launch_attn_rows(&cl).unwrap();
            c.be.synchronize(0).unwrap();
            let got = read_u16(&c.be, &out, 0, k * len);
            for r in 0..k {
                let o1 = c.be.alloc(len * 2, MemKind::Device).unwrap();
                let mut inputs = vec![tr(&qb, r * len * 2, &[1, 32, 128])];
                inputs.extend(c.kv());
                let mut c1 = OpCall::new(Op::AttnDecode, inputs, vec![tr(&o1, 0, &[1, 32, 128])], c.attrs(0));
                c1.scratch = scratch_for(&c.be, ATTN_DECODE_BF16_SPLITKV, &c1);
                c1.pos = Some(mk_pos((pos + r) as i32));
                c.be.launch(ATTN_DECODE_BF16_SPLITKV, &c1, 0).unwrap();
                c.be.synchronize(0).unwrap();
                assert_eq!(got[r * len..(r + 1) * len], read_u16(&c.be, &o1, 0, len)[..], "k {k} pos {pos} row {r}");
            }
        }
    }
}

#[test]
fn rows_registration_is_below_serving() {
    let be = backend();
    let ks = be.kernels();
    let k = ks.iter().find(|k| k.name == ATTN_PREFILL_BF16_SPLITKV_ROWS).unwrap();
    let prefill = ks.iter().find(|k| k.name == intarsia_backend_metal::kernels::ATTN_PREFILL_BF16).unwrap();
    assert_eq!(k.op, Op::AttnPrefill);
    assert!(k.priority < prefill.priority && k.priority < 0, "never selected by serving");
    let shape = |m: usize, n: usize, nkv: usize, d: usize| ShapeInfo {
        m,
        n_heads: n,
        n_kv_heads: nkv,
        head_dim: d,
        act: Some(DType::Bf16),
        weight: Some(DType::Bf16),
        ..Default::default()
    };
    for m in 2..=8 {
        assert!(k.accepts(&shape(m, 32, 8, 128)), "m {m}");
    }
    for (s, why) in [(shape(1, 32, 8, 128), "one row"), (shape(9, 32, 8, 128), "nine rows"), (shape(4, 32, 8, 96), "D = 96")] {
        assert!(!k.accepts(&s), "{why}");
    }
    assert_eq!(k.scratch_bytes(&shape(4, 32, 8, 128)), 4 * 32 * 64 * 130 * 4);
    let _ = ATTN_DECODE_BF16;
}
