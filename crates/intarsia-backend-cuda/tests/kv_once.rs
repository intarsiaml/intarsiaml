//! Lane kvonce (CUDA): `INTARSIA_CUDA_ATTN_KV_ONCE=1` renumbers the items of the row-table decode attention's persistent kernel
//! (`splitkv_rows_persist_kernel`'s ONCE, `kernels/cuda/attention_splitkv.cu`) so that the rows of one sequence read each K / V line
//! together. The launch with the knob on is BYTE-equal to the launch with it off, and every row is its solo one-row launch.
//!
//!  * row tables of M = 1..=16 rows of ONE cache at consecutive positions (a verify pass), started around every change of the
//!    split count (every 64 keys up to 4096), of the chunk (every 4096 keys above) and of the 16-key iteration, so that the rows of
//!    a pass straddle them;
//!  * mixed tables: two and three sequences of several rows, a cache that comes back after another, rows of one cache at
//!    unrelated positions, 16 rows over three caches;
//!  * a captured CUDA graph (device positions) replayed at many positions, knob on against knob off;
//!  * the fp8_e4m3-cache twin on the 35B shape;
//!  * mode 2 (the same numbering with streaming loads);
//!  * a planted fault (every row of a group gets the group leader's end of keys) that must be seen at every position tried.
//!
//! Shapes: Qwen3.6-35B-A3B (16 / 2 / 256) and Qwen3.8-27B (24 / 4 / 256), caches of 66000 keys (so 64k contexts are covered).
//! One test function: the knob override is process-wide. Uses about 2 GB of GPU memory.
//!
//! Needs a GPU: without one the test prints SKIPPED (and fails under CI=1 / INTARSIA_REQUIRE_GPU=1).
#![cfg(ik_cuda)]

mod common;

use intarsia_backend_cuda::set_attn_kv_once_for_test;
use intarsia_core::backend::{Backend, MemKind, TensorRef};
use intarsia_core::ops::RowTab;
use intarsia_core::{DType, Op, OpAttrs, OpCall, ShapeInfo};
use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::Arc;

/// Fast deterministic bits (xorshift64*).
struct Bits(u64);
impl Bits {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
}

/// `n` bf16 values of magnitude 2^-7 .. 2^2 with random signs and mantissas (scores of order one, so every key carries weight).
fn bf16_bytes(n: usize, seed: u64) -> Vec<u8> {
    let mut b = Bits(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);
    let mut out = Vec::with_capacity(n * 2);
    let mut w = 0u64;
    for i in 0..n {
        if i % 2 == 0 {
            w = b.next();
        }
        let r = (w >> (32 * (i % 2))) as u32;
        let sign = (r & 1) << 15;
        let exp = 120 + (r >> 1) % 10; // 2^-7 .. 2^2
        let man = (r >> 8) & 0x7F;
        let v = (sign | (exp << 7) | man) as u16;
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

/// `n` e4m3 bytes of magnitude 2^-3 .. 2^2 (never NaN).
fn e4m3_bytes(n: usize, seed: u64) -> Vec<u8> {
    let mut b = Bits(seed.wrapping_mul(0xD1B5_4A32_D192_ED03) | 1);
    let mut out = Vec::with_capacity(n);
    let mut w = 0u64;
    for i in 0..n {
        if i % 4 == 0 {
            w = b.next();
        }
        let r = (w >> (16 * (i % 4))) as u32;
        let sign = (r & 1) << 7;
        let exp = 4 + (r >> 1) % 6; // bias 7: 2^-3 .. 2^2
        let man = (r >> 8) & 0x7;
        out.push((sign | (exp << 3) | man) as u8);
    }
    out
}

const MAX_ROWS: usize = 16;

struct Case {
    be: Arc<dyn Backend>,
    nh: usize,
    nkv: usize,
    d: usize,
    cap: usize,
    kvt: DType,
    /// `ncache` caches (K, V) of different data (Cases built alike get the same data in their own buffers)
    kv: Vec<(TensorRef, TensorRef)>,
    q: TensorRef,
    scratch: RefCell<HashMap<usize, TensorRef>>,
}

impl Case {
    fn new(be: &Arc<dyn Backend>, nh: usize, nkv: usize, d: usize, cap: usize, kvt: DType, ncache: u64) -> Case {
        let put = |bytes: Vec<u8>, dt: DType, shape: &[usize]| {
            let buf = be.alloc(bytes.len().max(16), MemKind::Device).unwrap();
            be.upload(&buf, 0, &bytes, 0).unwrap();
            TensorRef::new(buf, dt, shape)
        };
        let n = nkv * cap * d;
        let gen = |seed: u64| if kvt == DType::Fp8E4m3 { e4m3_bytes(n, seed) } else { bf16_bytes(n, seed) };
        let kv = (0..ncache).map(|i| (put(gen(100 + 2 * i + d as u64), kvt, &[nkv, cap, d]), put(gen(101 + 2 * i + d as u64), kvt, &[nkv, cap, d]))).collect();
        let q = put(bf16_bytes(MAX_ROWS * nh * d, 7 + nh as u64), DType::Bf16, &[MAX_ROWS, nh, d]);
        Case { be: be.clone(), nh, nkv, d, cap, kvt, kv, q, scratch: RefCell::new(HashMap::new()) }
    }
    fn kernels(&self) -> (&'static str, &'static str) {
        if self.kvt == DType::Fp8E4m3 {
            ("attn_decode_f8kv_splitkv", "attn_decode_f8kv_splitkv_rows")
        } else {
            ("attn_decode_bf16_splitkv", "attn_decode_bf16_splitkv_rows")
        }
    }
    fn attrs(&self, pos0: usize) -> OpAttrs {
        OpAttrs { pos0, scale: 1.0 / (self.d as f32).sqrt(), n_heads: self.nh, n_kv_heads: self.nkv, head_dim: self.d, kv_cap: self.cap, ..Default::default() }
    }
    fn bytes(&self, data: &[u8], dt: DType, shape: &[usize]) -> TensorRef {
        let buf = self.be.alloc(data.len().max(16), MemKind::Device).unwrap();
        self.be.upload(&buf, 0, data, 0).unwrap();
        TensorRef::new(buf, dt, shape)
    }
    fn poison(&self, rows: usize) -> TensorRef {
        self.bytes(&vec![0xA5u8; rows * self.nh * self.d * 2], DType::Bf16, &[rows, self.nh, self.d])
    }
    fn down(&self, t: &TensorRef) -> Vec<u8> {
        let mut d = vec![0u8; t.nbytes()];
        self.be.synchronize(0).unwrap();
        self.be.download(&t.buf, t.offset, &mut d, 0).unwrap();
        d
    }
    fn with_scratch(&self, kernel: &str, call: &mut OpCall) {
        let k = self.be.kernels().into_iter().find(|k| k.name == kernel && k.op == call.op).unwrap_or_else(|| panic!("{kernel} not registered"));
        let need = k.scratch_bytes(&ShapeInfo::from_call(call));
        if need > 0 {
            let t = self
                .scratch
                .borrow_mut()
                .entry(need)
                .or_insert_with(|| {
                    let buf = self.be.alloc(need, MemKind::Device).unwrap();
                    self.be.zero(&buf, 0).unwrap();
                    TensorRef::new(buf, DType::U8, &[need])
                })
                .clone();
            call.scratch = Some(t);
        }
    }
    fn launch(&self, kernel: &str, mut call: OpCall) {
        self.with_scratch(kernel, &mut call);
        call.check().unwrap_or_else(|e| panic!("{kernel}: check: {e}"));
        self.be.launch(kernel, &call, 0).unwrap_or_else(|e| panic!("{kernel}: {e}"));
    }
    /// The solo one-row launch: query row `qr` on cache `ci` at host position `pos`.
    fn solo(&self, ci: usize, qr: usize, pos: usize) -> Vec<u8> {
        let (k, v) = &self.kv[ci];
        let o = self.poison(1);
        self.launch(self.kernels().0, OpCall::new(Op::AttnDecode, vec![self.q.rows(qr, 1), k.clone(), v.clone()], vec![o.clone()], self.attrs(pos)));
        self.down(&o)
    }
    /// The K / V address table of rows on caches `caches[r]`.
    fn table(&self, caches: &[usize]) -> TensorRef {
        let mut bytes = vec![0u8; caches.len() * 2 * 8];
        for (r, &ci) in caches.iter().enumerate() {
            for (j, t) in [&self.kv[ci].0, &self.kv[ci].1].into_iter().enumerate() {
                let a = self.be.device_address(&t.buf).expect("device address") + t.offset as u64;
                bytes[(r * 2 + j) * 8..(r * 2 + j) * 8 + 8].copy_from_slice(&a.to_le_bytes());
            }
        }
        self.bytes(&bytes, DType::U8, &[caches.len(), 2, 8])
    }
    fn rows_call(&self, ptrs: TensorRef, pos: TensorRef, o: &TensorRef, m: usize) -> OpCall {
        let (k, v) = &self.kv[0];
        let mut call = OpCall::new(Op::AttnDecode, vec![self.q.rows(0, m), k.clone(), v.clone()], vec![o.clone()], self.attrs(0));
        call.rows = Some(RowTab { ptrs, rows: m, stride: 2, index: 0, byte_off: 0 });
        call.pos = Some(pos);
        call
    }
    /// One row-table launch: row r (query row r) at position `pos[r]` on cache `caches[r]`.
    fn rowtab(&self, pos: &[usize], caches: &[usize]) -> Vec<u8> {
        let m = pos.len();
        let pt: Vec<u8> = pos.iter().flat_map(|p| (*p as i32).to_le_bytes()).collect();
        let post = self.bytes(&pt, DType::I32, &[m]);
        let o = self.poison(m);
        self.launch(self.kernels().1, self.rows_call(self.table(caches), post, &o, m));
        self.down(&o)
    }
    fn row_bytes(&self) -> usize {
        self.nh * self.d * 2
    }
}

fn same(a: &[u8], b: &[u8], what: &str) {
    assert_eq!(a.len(), b.len(), "{what}: length");
    if let Some(i) = a.iter().zip(b).position(|(x, y)| x != y) {
        panic!("{what}: first differing byte {i} of {} (knob off {:#04x}, on {:#04x})", a.len(), a[i], b[i]);
    }
}

/// `f` with the knob off, then on (mode `on`).
fn off_on<T>(on: u8, f: impl Fn() -> T) -> (T, T) {
    set_attn_kv_once_for_test(Some(0), false);
    let a = f();
    set_attn_kv_once_for_test(Some(on), false);
    let b = f();
    (a, b)
}

/// Visible lengths where the work partition of one row changes (CUDA's policy, `splitkv_count` / `splitkv_chunk` for one query
/// row: nsplit = min(ceil(kv / 64), 64) for these shapes, chunk = ceil(kv / nsplit) rounded up to 64 keys, 16 keys per iteration):
/// every split-count change sampled up to 4096, every chunk change (each 4096 keys) up to the cache, iteration and chunk edges.
fn boundaries(cap: usize) -> Vec<usize> {
    let mut v: Vec<usize> = vec![1, 2, 3, 15, 16, 17, 33];
    for s in [1usize, 2, 3, 7, 8, 15, 16, 31, 32, 33, 47, 62, 63, 64] {
        v.extend([64 * s - 1, 64 * s, 64 * s + 1, 64 * s + 16]);
    }
    let mut b = 4096;
    while b + 64 <= cap {
        v.extend([b - 1, b, b + 1, b + 2, b + 16, b + 17]);
        b += 4096;
    }
    // inside a 64-split partition: the end of split 63's chunk (chunk 1024 at 65536 keys), mid-chunk iteration edges
    v.extend([20000, 20001, 20016, 32768, 32769, 49152, 49153, 65535, 65536, 65537, 65552]);
    v.retain(|&x| x >= 1 && x <= cap);
    v.sort();
    v.dedup();
    v
}

/// First positions of an M-row pass so that its rows start at, straddle and end at each boundary.
fn starts(cap: usize, m: usize) -> Vec<usize> {
    let mut v = Vec::new();
    for b in boundaries(cap) {
        for p0 in [b as isize - 1, b as isize - 1 - (m as isize / 2), b as isize - m as isize] {
            if p0 >= 0 && p0 as usize + m <= cap {
                v.push(p0 as usize);
            }
        }
    }
    v.sort();
    v.dedup();
    v
}

#[test]
fn kv_once_launches_are_bit_identical_to_the_default_launches() {
    let ord = std::env::var("INTARSIA_CUDA_DEVICE").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
    let be = match intarsia_backend_cuda::create(ord) {
        Ok(be) => be,
        Err(e) => {
            common::skip("kv_once", &format!("no usable CUDA device ({e})"));
            return;
        }
    };
    let cap = 66000usize;
    let shapes: [(usize, usize, usize, DType); 3] = [(16, 2, 256, DType::Bf16), (24, 4, 256, DType::Bf16), (16, 2, 256, DType::Fp8E4m3)];
    for (nh, nkv, d, kvt) in shapes {
        let c = Case::new(&be, nh, nkv, d, cap, kvt, 3);
        let fp8 = kvt == DType::Fp8E4m3;
        let tag = format!("{nh}/{nkv}/{d} {kvt}");
        let rb = c.row_bytes();
        let (mut n_one, mut n_mixed, mut n_solo, mut n_graph) = (0usize, 0usize, 0usize, 0usize);

        // 1. one cache, M rows at consecutive positions (the verify pass), M = 1..=16
        for m in 1..=MAX_ROWS {
            let ps = starts(cap, m);
            for (i, &p0) in ps.iter().enumerate() {
                // every start for the MTP passes (2..=4 rows) on the bf16 shapes, a sample otherwise
                let every = if fp8 { 6 } else if (2..=4).contains(&m) { 1 } else { 3 };
                if i % every != m % every {
                    continue;
                }
                let pos: Vec<usize> = (p0..p0 + m).collect();
                let caches = vec![1usize; m];
                let (a, b) = off_on(1, || c.rowtab(&pos, &caches));
                same(&a, &b, &format!("{tag} one cache M={m} pos0 {p0}"));
                n_one += 1;
                if i % 23 == 0 {
                    for (r, &p) in pos.iter().enumerate() {
                        let s = c.solo(1, r, p);
                        assert!(b[r * rb..(r + 1) * rb] == s[..], "{tag} one cache M={m} row {r} pos {p}: differs from the solo launch");
                        n_solo += 1;
                    }
                }
            }
        }

        // 2. mixed tables: (cache, first position, rows) per sequence
        let mixed: Vec<Vec<(usize, usize, usize)>> = vec![
            vec![(0, 4093, 3), (2, 65530, 3)],                     // two verify streams, one straddling a chunk change
            vec![(0, 300, 3), (1, 300, 3), (2, 300, 3)],           // three sequences at the same positions
            vec![(0, 8190, 3), (1, 77, 1), (0, 8190, 2)],          // cache 0 comes back after another
            vec![(1, 1000, 2), (1, 40000, 2), (1, 63, 2)],         // one cache at unrelated positions (one group or several)
            vec![(2, 62, 4), (0, 4094, 4), (1, 12287, 4), (2, 30000, 4)], // 16 rows: split / chunk changes inside every sequence
            vec![(0, 0, 1), (0, 1, 1), (0, 2, 1), (1, 65000, 5)],  // the first keys of a cache, one row per entry
            vec![(2, 49150, 16)],
        ];
        for (li, layout) in mixed.iter().enumerate() {
            let (mut pos, mut caches) = (Vec::new(), Vec::new());
            for &(ci, p0, n) in layout {
                pos.extend(p0..p0 + n);
                caches.extend(vec![ci; n]);
            }
            let (a, b) = off_on(1, || c.rowtab(&pos, &caches));
            same(&a, &b, &format!("{tag} mixed layout {li} {layout:?}"));
            for (r, (&p, &ci)) in pos.iter().zip(&caches).enumerate() {
                let s = c.solo(ci, r, p);
                assert!(b[r * rb..(r + 1) * rb] == s[..], "{tag} mixed layout {li} row {r} pos {p}: differs from the solo launch");
                n_solo += 1;
            }
            n_mixed += 1;
        }

        // 3. mode 2 (streaming loads): a sample of verify passes
        for (m, p0) in [(3usize, 4095usize), (3, 65530), (8, 127), (16, 20000), (2, 63)] {
            let pos: Vec<usize> = (p0..p0 + m).collect();
            let (a, b) = off_on(2, || c.rowtab(&pos, &vec![0; m]));
            same(&a, &b, &format!("{tag} mode 2 M={m} pos0 {p0}"));
        }

        // 4. captured graphs (device positions): the launch recorded with the knob on against the one recorded with it off
        if !fp8 {
            for m in [3usize, 16] {
                let pos_buf = c.be.alloc(256, MemKind::Device).unwrap();
                c.be.zero(&pos_buf, 0).unwrap();
                let mut post = TensorRef::new(pos_buf.clone(), DType::I32, &[m]);
                post.offset = 64; // entries before and after it must not be read
                let ptrs = c.table(&vec![2usize; m]);
                let mut calls = Vec::new();
                let mut graphs = Vec::new();
                for mode in [0u8, 1] {
                    let o = c.poison(m);
                    let mut call = c.rows_call(ptrs.clone(), post.clone(), &o, m);
                    c.with_scratch(c.kernels().1, &mut call);
                    call.check().unwrap();
                    set_attn_kv_once_for_test(Some(mode), false);
                    let g = c.be.graph_capture(0, &mut || c.be.launch(c.kernels().1, &call, 0)).unwrap_or_else(|e| panic!("{tag} capture mode {mode}: {e}"));
                    graphs.push((g, o));
                    calls.push(call); // the graph addresses the call's scratch
                }
                set_attn_kv_once_for_test(Some(0), false);
                for p0 in [0usize, 5, 62, 63, 100, 4093, 4094, 4095, 8190, 30000, 49151, 65534, cap - m] {
                    let ps: Vec<u8> = (0..m).flat_map(|r| ((p0 + r) as i32).to_le_bytes()).collect();
                    c.be.upload(&pos_buf, 64, &ps, 0).unwrap();
                    let mut outs = Vec::new();
                    for (g, o) in &graphs {
                        c.be.upload(&o.buf, o.offset, &vec![0xA5u8; o.nbytes()], 0).unwrap();
                        c.be.graph_replay(g, 0).unwrap();
                        outs.push(c.down(o));
                    }
                    same(&outs[0], &outs[1], &format!("{tag} captured M={m} pos0 {p0}"));
                    n_graph += 1;
                }
                drop(calls);
            }
        }

        // 5. planted fault: every row of a group gets the leader's end of keys; must be seen at every position
        let fps: Vec<usize> = [300usize, 1023, 1500, 4000, 4094, 20000, 65530].into_iter().filter(|p| p + 3 <= cap).collect();
        let mut caught = 0;
        for &p0 in &fps {
            let pos = [p0, p0 + 1, p0 + 2];
            set_attn_kv_once_for_test(Some(0), false);
            let good = c.rowtab(&pos, &[0, 0, 0]);
            set_attn_kv_once_for_test(Some(1), true);
            let bad = c.rowtab(&pos, &[0, 0, 0]);
            caught += (good != bad) as usize;
            // the leader row itself is untouched by the fault
            assert!(good[..rb] == bad[..rb], "{tag} fault pos0 {p0}: the leader row changed");
        }
        set_attn_kv_once_for_test(Some(0), false);
        assert_eq!(caught, fps.len(), "{tag}: the planted fault was seen at {caught} of {} positions", fps.len());
        println!(
            "kv_once {tag}: knob on bit-identical to off: {n_one} one-cache verify tables (M 1..=16), {n_mixed} mixed tables, \
             {n_graph} captured replays, mode 2 x5; {n_solo} rows equal to their solo launch; planted fault seen {caught} of {}",
            fps.len()
        );
    }
    set_attn_kv_once_for_test(None, false);
}

/// Kernel time of a captured pass's verify attention: 10 layers (the 35B's full-attention layers, own caches of 66000 keys) as one
/// row-table launch each, replayed as a graph, knob off / on / mode 2, t = 1..=4 rows at 4k / 32k / 64k. Prints only with
/// `KVONCE_TIMING=1`: `cargo test -p intarsia-backend-cuda --test kv_once --release -- --nocapture timing`.
#[test]
fn timing_kv_once_layers() {
    if std::env::var("KVONCE_TIMING").map_or(true, |v| v == "0") {
        return;
    }
    let ord = std::env::var("INTARSIA_CUDA_DEVICE").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
    let Ok(be) = intarsia_backend_cuda::create(ord) else {
        common::skip("kv_once timing", "no usable CUDA device");
        return;
    };
    let (nh, nkv, d, cap, nl) = (16usize, 2usize, 256usize, 66000usize, 10usize);
    let layers: Vec<Case> = (0..nl).map(|_| Case::new(&be, nh, nkv, d, cap, DType::Bf16, 1)).collect(); // one cache per layer
    let pos_buf = be.alloc(256, MemKind::Device).unwrap();
    be.zero(&pos_buf, 0).unwrap();
    for t in 1..=4usize {
        let post = TensorRef::new(pos_buf.clone(), DType::I32, &[t]);
        let mut graphs = Vec::new();
        let mut keep = Vec::new();
        for mode in [0u8, 1, 2] {
            let mut calls = Vec::new();
            for c in &layers {
                let o = c.poison(t);
                let mut call = c.rows_call(c.table(&vec![0usize; t]), post.clone(), &o, t);
                c.with_scratch(c.kernels().1, &mut call);
                calls.push(call);
            }
            set_attn_kv_once_for_test(Some(mode), false);
            let g = be.graph_capture(0, &mut || calls.iter().try_for_each(|cl| be.launch("attn_decode_bf16_splitkv_rows", cl, 0))).unwrap();
            graphs.push((mode, g));
            keep.push(calls);
        }
        set_attn_kv_once_for_test(None, false);
        for p_end in [4096usize, 32768, 65536] {
            let p0 = p_end - t;
            let ps: Vec<u8> = (0..t).flat_map(|r| ((p0 + r) as i32).to_le_bytes()).collect();
            be.upload(&pos_buf, 0, &ps, 0).unwrap();
            let mut line = format!("kvonce timing t={t} ctx={p_end}:");
            for (mode, g) in &graphs {
                let mut best = f64::MAX;
                for _ in 0..5 {
                    for _ in 0..3 {
                        be.graph_replay(g, 0).unwrap();
                    }
                    be.synchronize(0).unwrap();
                    let t0 = std::time::Instant::now();
                    for _ in 0..20 {
                        be.graph_replay(g, 0).unwrap();
                    }
                    be.synchronize(0).unwrap();
                    best = best.min(t0.elapsed().as_secs_f64() / 20.0);
                }
                // unique K + V bytes of the pass: every layer reads keys 0 .. p_end once
                let gbs = (nl * 2 * nkv * p_end * d * 2) as f64 / best / 1e9;
                line += &format!("  mode {mode} {:.0} us ({gbs:.0} GB/s unique)", best * 1e6);
            }
            println!("{line}");
        }
        drop(keep);
    }
}
