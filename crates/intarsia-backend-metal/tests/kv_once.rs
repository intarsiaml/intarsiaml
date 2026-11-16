//! Packed threadgroups of the decode attention (`INTARSIA_METAL_ATTN_KV_ONCE=1`, lane mka, attn.rs `pack_plan`,
//! kernels/metal/attention_splitkv.metal "Packed threadgroups"): the three launches that take them are bit for bit the
//! unpacked launches.
//!
//!  * the 1-row decode launch (`attn_decode_bf16_splitkv`), host and device position;
//!  * the G0b verify rows (`attn_prefill_bf16_splitkv_rows`, M = 2..=8), host and device position;
//!  * the row table (`attn_decode_bf16_splitkv_rows`, M = 1..=16), rows on different caches and positions (a batched step)
//!    and rows on ONE cache at consecutive positions (a verify pass); each packed row is also its solo 1-row launch.
//!
//! Shapes: the 35B (16 / 2 / 256, head split 4), the 27B (24 / 4 / 256, bitwise head split 3), the 8B (32 / 8 / 128), and two
//! small ones. Visible lengths: around every change of the split count, of the simdgroup rule and of the chunk rounding up to
//! 40000 keys. Caches and q at byte offsets inside their buffers. A planted fault (one step of one stream skipped) must be
//! seen. One test function: the knob override is process-wide.
//!
//! Uses the GPU and about 1 GB: on endor take a fleetq slot first.

#![cfg(target_os = "macos")]

mod common;

use common::*;
use intarsia_backend_metal::kernels::{ATTN_DECODE_BF16_SPLITKV, ATTN_PREFILL_BF16_SPLITKV_ROWS};
use intarsia_backend_metal::{set_kv_once_for_test, set_kvo_fault_for_test, MetalBackend};
use intarsia_core::backend::{Backend, Buffer, MemKind, TensorRef};
use intarsia_core::ops::RowTab;
use intarsia_core::{DType, Op, OpAttrs, OpCall, ShapeInfo};
use std::cell::RefCell;
use std::collections::HashMap;

const ATTN_DECODE_ROWTAB: &str = "attn_decode_bf16_splitkv_rows";
/// Byte offset of every cache and of q inside its buffer (16-byte aligned, as the kernels need).
const OFF: usize = 48;

fn tr(b: &Buffer, off: usize, shape: &[usize]) -> TensorRef {
    TensorRef { buf: b.clone(), offset: off, dtype: DType::Bf16, shape: shape.to_vec() }
}

fn upload_at(be: &MetalBackend, off: usize, data: &[u16]) -> Buffer {
    let b = be.alloc(off + data.len() * 2 + 16, MemKind::Device).unwrap();
    be.upload(&b, 0, &vec![0xEEu8; off + data.len() * 2 + 16], 0).unwrap();
    be.upload(&b, off, as_bytes(data), 0).unwrap();
    b
}

/// The scratch workspace of `kernel` for `cl`, one buffer per size (filled once: the kernels never read what they did not write).
fn scratch_for(be: &MetalBackend, kernel: &str, cl: &OpCall, cache: &RefCell<HashMap<usize, TensorRef>>) -> Option<TensorRef> {
    let need = be
        .kernels()
        .iter()
        .find(|k| k.name == kernel && k.op == cl.op)
        .map_or(0, |k| k.scratch_bytes(&ShapeInfo::from_call(cl)));
    (need > 0).then(|| {
        cache
            .borrow_mut()
            .entry(need)
            .or_insert_with(|| {
                let b = be.alloc(need, MemKind::Device).unwrap();
                be.upload(&b, 0, &vec![0xEEu8; need], 0).unwrap();
                TensorRef { buf: b, offset: 0, dtype: DType::U8, shape: vec![need] }
            })
            .clone()
    })
}

struct Case {
    be: MetalBackend,
    n: usize,
    nkv: usize,
    d: usize,
    cap: usize,
    /// three caches (K, V) of different data; row r of a row table uses cache `r % 3`
    kv: Vec<(Buffer, Buffer)>,
    qb: Buffer,
    scratch: RefCell<HashMap<usize, TensorRef>>,
}

impl Case {
    fn new(n: usize, nkv: usize, d: usize, cap: usize) -> Case {
        let be = backend();
        let kv = (0..3u64)
            .map(|i| {
                (
                    upload_at(&be, OFF, &rand_bf16(nkv * cap * d, 11 + 10 * i + d as u64)),
                    upload_at(&be, OFF, &rand_bf16(nkv * cap * d, 12 + 10 * i + d as u64)),
                )
            })
            .collect();
        let qb = upload_at(&be, OFF, &rand_bf16(16 * n * d, 13 + n as u64));
        Case { be, n, nkv, d, cap, kv, qb, scratch: RefCell::new(HashMap::new()) }
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
    fn kv_of(&self, ci: usize) -> Vec<TensorRef> {
        let (k, v) = &self.kv[ci % self.kv.len()];
        vec![tr(k, OFF, &[self.nkv, self.cap, self.d]), tr(v, OFF, &[self.nkv, self.cap, self.d])]
    }
    fn q(&self, row0: usize, rows: usize) -> TensorRef {
        tr(&self.qb, OFF + row0 * self.n * self.d * 2, &[rows, self.n, self.d])
    }
    fn i32s(&self, v: &[usize]) -> TensorRef {
        let pb = self.be.alloc((v.len() * 4).max(16), MemKind::Device).unwrap();
        let bytes: Vec<u8> = v.iter().flat_map(|p| (*p as i32).to_le_bytes()).collect();
        self.be.upload(&pb, 0, &bytes, 0).unwrap();
        TensorRef { buf: pb, offset: 0, dtype: DType::I32, shape: vec![v.len()] }
    }
    /// An output of `len` elements between guards; returns the buffer bytes, guards included.
    fn run(&self, kernel: &str, mut cl: OpCall, out: &Buffer, len: usize, direct_rows: bool) -> Vec<u8> {
        cl.scratch = scratch_for(&self.be, kernel, &cl, &self.scratch);
        if direct_rows {
            self.be.launch_attn_rows(&cl).unwrap();
        } else {
            cl.check().unwrap();
            self.be.launch(kernel, &cl, 0).unwrap();
        }
        self.be.synchronize(0).unwrap();
        let mut raw = vec![0u8; OFF + len * 2 + 16];
        self.be.download(out, 0, &mut raw, 0).unwrap();
        raw
    }
    fn out(&self, len: usize) -> Buffer {
        upload_at(&self.be, OFF, &vec![0xA5A5u16; len])
    }
    /// 1-row decode launch on cache `ci` with query row `qr` at position `pos`.
    fn one(&self, ci: usize, qr: usize, pos: usize, dev: bool) -> Vec<u8> {
        let len = self.n * self.d;
        let out = self.out(len);
        let mut inputs = vec![self.q(qr, 1)];
        inputs.extend(self.kv_of(ci));
        let mut cl = OpCall::new(
            Op::AttnDecode,
            inputs,
            vec![tr(&out, OFF, &[1, self.n, self.d])],
            self.attrs(if dev { 0 } else { pos }),
        );
        if dev {
            cl.pos = Some(self.i32s(&[pos]));
        }
        self.run(ATTN_DECODE_BF16_SPLITKV, cl, &out, len, false)
    }
    /// G0b verify rows: k rows of cache 0 at positions `pos0 ..`.
    fn rows(&self, k: usize, pos0: usize, dev: bool) -> Vec<u8> {
        let len = k * self.n * self.d;
        let out = self.out(len);
        let mut inputs = vec![self.q(0, k)];
        inputs.extend(self.kv_of(0));
        let mut cl = OpCall::new(
            Op::AttnPrefill,
            inputs,
            vec![tr(&out, OFF, &[k, self.n, self.d])],
            self.attrs(if dev { 0 } else { pos0 }),
        );
        if dev {
            cl.pos = Some(self.i32s(&[pos0]));
        }
        self.run(ATTN_PREFILL_BF16_SPLITKV_ROWS, cl, &out, len, dev)
    }
    /// Row table: row r at position `pos[r]` on cache `cache(r)`.
    fn rowtab(&self, pos: &[usize], cache: impl Fn(usize) -> usize) -> Vec<u8> {
        let rows = pos.len();
        let len = rows * self.n * self.d;
        let out = self.out(len);
        let mut bytes = vec![0u8; rows * 2 * 8];
        for r in 0..rows {
            for (j, t) in self.kv_of(cache(r)).iter().enumerate() {
                let a = self.be.device_address(&t.buf).expect("device address") + OFF as u64;
                bytes[(r * 2 + j) * 8..(r * 2 + j) * 8 + 8].copy_from_slice(&a.to_le_bytes());
            }
        }
        let tb = self.be.alloc(bytes.len(), MemKind::Device).unwrap();
        self.be.upload(&tb, 0, &bytes, 0).unwrap();
        let ptrs = TensorRef { buf: tb, offset: 0, dtype: DType::U8, shape: vec![rows, 2, 8] };
        let kv0 = self.kv_of(0);
        let mut cl = OpCall::new(
            Op::AttnDecode,
            vec![self.q(0, rows), kv0[0].clone(), kv0[1].clone()],
            vec![tr(&out, OFF, &[rows, self.n, self.d])],
            self.attrs(0),
        );
        cl.rows = Some(RowTab { ptrs, rows, stride: 2, index: 0, byte_off: 0 });
        cl.pos = Some(self.i32s(pos));
        self.run(ATTN_DECODE_ROWTAB, cl, &out, len, false)
    }
}

/// The payload rows of a guarded output buffer.
fn row_of(raw: &[u8], r: usize, row_elems: usize) -> &[u8] {
    &raw[OFF + r * row_elems * 2..OFF + (r + 1) * row_elems * 2]
}

/// Positions whose visible length `p + 1` is around every change of the work partition for `npol` policy kv heads: the
/// split count (a new split every 128 keys until `ceil(120 / npol)`), the simdgroup rule (`npol * nsplit <= 64`), the
/// chunk rounding (the chunk grows by 32 keys every `32 * nsplit` visible keys; sampled along the whole range), the cache
/// end. `k` rows must fit.
fn positions(cap: usize, k: usize, npol: usize) -> Vec<usize> {
    let smax = 120usize.div_ceil(npol).min(64);
    let mut vis: Vec<usize> = vec![1, 2, 7, 8, 31, 32, 33, 63, 64, 65, 127, 128, 129];
    for s in 1..=smax + 1 {
        vis.extend([128 * s - 1, 128 * s, 128 * s + 1, 128 * s + 2]);
    }
    let step = 32 * smax;
    for target in [4096usize, 6000, 8192, 12000, 16384, 20000, 24000, 32768, 34000, 39000] {
        let b = target / step * step;
        vis.extend([b - 1, b, b + 1, b + 2, b + 31, b + 32, b + 33, target, target + 1]);
    }
    vis.extend([cap / 2, cap - 1, cap]);
    let mut v: Vec<usize> = vis.into_iter().filter(|&x| x >= 1 && x + k - 1 <= cap).map(|x| x - 1).collect();
    v.sort();
    v.dedup();
    v
}

fn same(a: &[u8], b: &[u8], what: &str) {
    assert_eq!(a.len(), b.len(), "{what}: length");
    if let Some(i) = a.iter().zip(b).position(|(x, y)| x != y) {
        panic!("{what}: first differing byte {i} of {} (unpacked {:#04x}, packed {:#04x})", a.len(), a[i], b[i]);
    }
}

fn off_on<T>(f: impl Fn() -> T) -> (T, T) {
    set_kv_once_for_test(Some(false));
    let a = f();
    set_kv_once_for_test(Some(true));
    let b = f();
    (a, b)
}

#[test]
fn packed_launches_are_bit_identical_to_the_unpacked_launches() {
    // (n, nkv, d, policy kv heads of the default head plan, cap, long)
    let shapes: &[(usize, usize, usize, usize, usize)] = &[
        (16, 2, 256, 8, 40008),  // Qwen3.6-35B-A3B: head split 4
        (24, 4, 256, 4, 40008),  // Qwen3.8-27B: bitwise head split 3
        (32, 8, 128, 8, 40008),  // Qwen3-8B
        (8, 2, 64, 2, 4104),     // group 4 at head_dim 64 (4 keys per simdgroup step)
        (12, 2, 128, 2, 4104),   // group 6 at head_dim 128 (GP 8, 2 heads idle)
    ];
    // The shipped policy packs only launches whose threadgroup has fewer than 4 simdgroups (the 27B). The test packs EVERY shape
    // (`INTARSIA_METAL_ATTN_KVO_BELOW=16`, read once by the launcher) unless `KVO_TEST_SHIPPED_POLICY=1`; run it again with
    // `INTARSIA_METAL_ATTN_KVO_IPT=1|2|3` and `INTARSIA_METAL_ATTN_KVO_TRIM=0` for the other layouts.
    let shipped = std::env::var("KVO_TEST_SHIPPED_POLICY").is_ok_and(|v| v == "1");
    if !shipped && std::env::var_os("INTARSIA_METAL_ATTN_KVO_BELOW").is_none() {
        std::env::set_var("INTARSIA_METAL_ATTN_KVO_BELOW", "16");
    }
    let only: Option<usize> = std::env::var("KVO_SHAPE").ok().and_then(|v| v.parse().ok());
    for (si, &(n, nkv, d, npol, cap)) in shapes.iter().enumerate() {
        if only.is_some_and(|o| o != si) {
            continue;
        }
        let c = Case::new(n, nkv, d, cap);
        let rl = n * d;
        let (mut n1, mut nr, mut nt, mut nv) = (0usize, 0usize, 0usize, 0usize);

        // 1-row decode launch
        let ps1 = positions(cap, 1, npol);
        for dev in [false, true] {
            for &pos in &ps1 {
                let (a, b) = off_on(|| c.one(0, 0, pos, dev));
                same(&a, &b, &format!("{n}/{nkv}/{d} 1-row dev={dev} pos {pos}"));
                n1 += 1;
            }
        }

        // G0b verify rows
        for dev in [false, true] {
            for k in [2usize, 3, 4, 5, 8] {
                let ps = positions(cap, k, npol);
                // every position for 3 rows (the MTP verify pass), every third otherwise
                for (i, &pos) in ps.iter().enumerate() {
                    if k != 3 && i % 3 != k % 3 {
                        continue;
                    }
                    let (a, b) = off_on(|| c.rows(k, pos, dev));
                    same(&a, &b, &format!("{n}/{nkv}/{d} rows k={k} dev={dev} pos {pos}"));
                    nr += 1;
                }
            }
        }

        // row table, rows on different caches and positions; every packed row is also its solo unpacked launch
        for r in [1usize, 2, 3, 4, 5, 8, 16] {
            for w in (0..ps1.len()).step_by(if r <= 3 { 1 } else { 4 }) {
                let pos: Vec<usize> = (0..r).map(|i| ps1[(w + 7 * i) % ps1.len()]).collect();
                let (a, b) = off_on(|| c.rowtab(&pos, |i| i));
                same(&a, &b, &format!("{n}/{nkv}/{d} row table R={r} pos {pos:?}"));
                if w % 16 == 0 {
                    set_kv_once_for_test(Some(false));
                    for (i, p) in pos.iter().enumerate() {
                        let solo = c.one(i, i, *p, false);
                        assert!(
                            row_of(&b, i, rl) == row_of(&solo, 0, rl),
                            "{n}/{nkv}/{d} row table R={r} row {i} pos {p}: packed row differs from its solo launch"
                        );
                    }
                }
                nt += 1;
            }
        }

        // row table, M rows of ONE cache at consecutive positions (the verify pass as a table), M = 2..=16
        for m in [2usize, 3, 4, 6, 8, 11, 16] {
            let ps = positions(cap, m, npol);
            for (i, &p0) in ps.iter().enumerate() {
                if m != 3 && i % 4 != m % 4 {
                    continue;
                }
                let pos: Vec<usize> = (0..m).map(|r| p0 + r).collect();
                let (a, b) = off_on(|| c.rowtab(&pos, |_| 0));
                same(&a, &b, &format!("{n}/{nkv}/{d} one-cache table M={m} pos0 {p0}"));
                if i % 8 == 0 {
                    // each row of the packed multi-row launch is the packed AND the unpacked one-row launch
                    for (r, p) in pos.iter().enumerate() {
                        let (s0, s1) = off_on(|| c.one(0, r, *p, false));
                        assert!(
                            row_of(&b, r, rl) == row_of(&s0, 0, rl) && row_of(&b, r, rl) == row_of(&s1, 0, rl),
                            "{n}/{nkv}/{d} one-cache table M={m} row {r} pos {p}: differs from the one-row launch"
                        );
                    }
                }
                nv += 1;
            }
        }

        // planted fault: the first step of stream 0 of split 0 skipped in the packed launches
        set_kv_once_for_test(Some(true));
        let mut caught = [0usize; 3];
        let fps: Vec<usize> = [300usize, 1023, 1500, 4000, cap - 20].into_iter().filter(|p| p + 8 <= cap).collect();
        for &pos in &fps {
            set_kvo_fault_for_test(false);
            let (g1, gr, gt) = (c.one(0, 0, pos, false), c.rows(3, pos, false), c.rowtab(&[pos, pos + 1, pos + 2], |_| 0));
            set_kvo_fault_for_test(true);
            let (f1, fr, ft) = (c.one(0, 0, pos, false), c.rows(3, pos, false), c.rowtab(&[pos, pos + 1, pos + 2], |_| 0));
            set_kvo_fault_for_test(false);
            caught[0] += (g1 != f1) as usize;
            caught[1] += (gr != fr) as usize;
            caught[2] += (gt != ft) as usize;
        }
        // (under the shipped policy only the 27B shape is packed, so only there the fault can show)
        assert!(
            (shipped && si != 1) || caught.iter().all(|&x| x == fps.len()),
            "{n}/{nkv}/{d}: the planted fault was seen in {caught:?} of {} positions per launch kind",
            fps.len()
        );
        eprintln!(
            "kv_once {n}/{nkv}/{d}: bit-identical packed vs unpacked: {n1} 1-row, {nr} verify-rows, {nt} row-table, {nv} \
             one-cache-table launches (guards included); planted fault seen {caught:?} of {}",
            fps.len()
        );
    }
    set_kv_once_for_test(None);
}
