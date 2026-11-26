//! G-B of the Inkling-2 row-table kernels on HIP (lane ink-hip-batch, batched decode, `OpCall::rows`): row r of one row-table
//! launch of `conv_mix_bf16_rows` / `ink_attn_decode_bf16_rows` is BITWISE the solo T = 1 launch of its base kernel
//! (`conv_mix_bf16`, `ink_attn_decode_bf16`, device position: the captured decode step's form) on sequence r alone. Compared:
//! every output row, and every byte of every row's state buffers after the call (rings, caches and the pads around them).
//! * row counts 1, 2, 3, 5, 16, every row at its own position: before / at / past the conv-ring wrap, the SWA window and ring,
//!   2048 keys (above it the solo launcher takes op 75's split path, the rows launcher always the fused kernel), the tau floor;
//! * state planes at nonzero offsets in their buffers, both planes of a ring behind one table entry (as `upload_row_table` writes
//!   an Inkling-2 ring slot), the other table entries pointing at a poisoned buffer;
//! * ConvMix: the admission sweep's mix forms (conv alone, mix with / without norm, blend with 1 and 3 streams) and epilogues,
//!   widths 64 and 2304 (the real model), under the mix masks 3 (default) and 0 (first forms);
//! * both under `INTARSIA_HIP_INK_DEC3` default and 0 (`set_dec3`).
//!
//! Needs a GPU: without one the tests print SKIPPED (and fail under CI=1 / INTARSIA_REQUIRE_GPU=1).
#![cfg(ik_hip)]

mod common;

use intarsia_core::backend::{Backend, Buffer, MemKind, TensorRef};
use intarsia_core::inkling2::ops::nq;
use intarsia_core::ops::RowTab;
use intarsia_core::rng::SplitMix64;
use intarsia_core::{DType, HostTensor, Op, OpAttrs, OpCall, ShapeInfo};
use std::sync::{Arc, Mutex};

/// The process-wide kernel-library switches (mix mode, dec3) are flipped by both tests: one at a time.
static SWITCHES: Mutex<()> = Mutex::new(());

const STRIDE: usize = 8;
const INDEX: usize = 2;

fn device(test: &str) -> Option<Arc<dyn Backend>> {
    let ord = std::env::var("INTARSIA_HIP_DEVICE").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
    match intarsia_backend_hip::create(ord) {
        Ok(b) => Some(b),
        Err(e) => {
            common::skip(test, &format!("no usable HIP device ({e})"));
            None
        }
    }
}

fn tr(b: &Buffer, off: usize, dt: DType, shape: &[usize]) -> TensorRef {
    TensorRef { buf: b.clone(), offset: off, dtype: dt, shape: shape.to_vec() }
}

fn normal(rng: &mut SplitMix64, dt: DType, n: usize, scale: f32) -> Vec<u8> {
    HostTensor::from_f32_as(dt, &[n], &rng.normal_vec(n, scale)).data
}

fn uniform(rng: &mut SplitMix64, n: usize, lo: f32, hi: f32) -> Vec<u8> {
    let v: Vec<f32> = (0..n).map(|_| lo + (hi - lo) * rng.next_f32()).collect();
    HostTensor::from_f32_as(DType::Bf16, &[n], &v).data
}

struct Dev<'a> {
    be: &'a dyn Backend,
    /// a poisoned buffer the unused table entries point at (a wrong entry reads garbage instead of faulting)
    junk: Buffer,
}

impl<'a> Dev<'a> {
    fn new(be: &'a dyn Backend) -> Dev<'a> {
        let none = be.alloc(16, MemKind::Device).unwrap();
        be.zero(&none, 0).unwrap();
        set_none(none);
        let junk = be.alloc(64 << 20, MemKind::Device).unwrap();
        be.upload(&junk, 0, &vec![0xA5u8; 64 << 20], 0).unwrap();
        Dev { be, junk }
    }

    fn buf(&self, data: &[u8]) -> Buffer {
        let b = self.be.alloc(data.len().max(16), MemKind::Device).unwrap();
        self.be.zero(&b, 0).unwrap();
        if !data.is_empty() {
            self.be.upload(&b, 0, data, 0).unwrap();
        }
        b
    }

    fn poison(&self, n: usize) -> Buffer {
        self.buf(&vec![0xA5u8; n])
    }

    fn get(&self, b: &Buffer, n: usize) -> Vec<u8> {
        self.be.synchronize(0).unwrap();
        let mut d = vec![0u8; n];
        self.be.download(b, 0, &mut d, 0).unwrap();
        d
    }

    fn addr(&self, b: &Buffer) -> u64 {
        self.be.device_address(b).expect("hip gives device addresses")
    }

    /// The row table of `rows` rows: entries INDEX.. of row r are `ents[r]`, every other entry the junk buffer.
    fn table(&self, ents: &[Vec<u64>]) -> TensorRef {
        let mut t = vec![0u8; ents.len() * STRIDE * 8];
        for (r, e) in ents.iter().enumerate() {
            for j in 0..STRIDE {
                let a = if j >= INDEX && j - INDEX < e.len() { e[j - INDEX] } else { self.addr(&self.junk) };
                t[(r * STRIDE + j) * 8..(r * STRIDE + j + 1) * 8].copy_from_slice(&a.to_le_bytes());
            }
        }
        tr(&self.buf(&t), 0, DType::U8, &[ents.len(), STRIDE, 8])
    }

    fn pos(&self, p: &[usize]) -> TensorRef {
        let b: Vec<u8> = p.iter().flat_map(|&x| (x as i32).to_le_bytes()).collect();
        tr(&self.buf(&b), 0, DType::I32, &[p.len()])
    }

    fn launch(&self, kernel: &str, mut call: OpCall) {
        call.check().unwrap_or_else(|e| panic!("{kernel}: check: {e}"));
        let k = self.be.kernels().into_iter().find(|k| k.name == kernel && k.op == call.op).unwrap_or_else(|| panic!("{kernel} not registered"));
        let need = k.scratch_bytes(&ShapeInfo::from_call(&call));
        if need > 0 {
            call.scratch = Some(tr(&self.poison(need), 0, DType::U8, &[need]));
        }
        self.be.launch(kernel, &call, 0).unwrap_or_else(|e| panic!("{kernel}: {e}"));
        self.be.synchronize(0).unwrap();
    }
}

// the absent operand (`[0]`) of a call: a live buffer of this test's device
thread_local!(static NONE_BUF: std::cell::RefCell<Option<Buffer>> = const { std::cell::RefCell::new(None) });

fn set_none(b: Buffer) {
    NONE_BUF.with(|n| *n.borrow_mut() = Some(b));
}

fn none_t(dt: DType) -> TensorRef {
    let b = NONE_BUF.with(|n| n.borrow().clone()).expect("Dev::new first");
    tr(&b, 0, dt, &[0])
}

fn bf16_rows(v: &[u8], row: usize, width: usize) -> &[u8] {
    &v[row * width * 2..(row + 1) * width * 2]
}

/// Row r of every plane of a `[n, rows, width]` tensor of `esz`-byte elements, as `[n, 1, width]` bytes.
fn plane_rows(v: &[u8], n: usize, rows: usize, width: usize, esz: usize, r: usize) -> Vec<u8> {
    (0..n).flat_map(|p| v[((p * rows + r) * width) * esz..((p * rows + r + 1) * width) * esz].to_vec()).collect()
}

fn diff(a: &[u8], b: &[u8]) -> usize {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).filter(|(x, y)| x != y).count()
}

/// Row counts of the batched calls, and the positions the rows take (cycled per row with a per-case offset).
const ROWS: [usize; 5] = [1, 2, 3, 5, 16];

fn positions(list: &[usize], b: usize, case: usize) -> Vec<usize> {
    (0..b).map(|r| list[(case * 7 + r * 3) % list.len()]).collect()
}

// ------------------------------------------------------------------------------------------------------------------------
// ConvMix
// ------------------------------------------------------------------------------------------------------------------------

const K: usize = 4;
const RR: usize = K - 1 + 16;
const G: usize = 8;
const RING_PAD: usize = 512;

/// One ConvMix case: width, mix form (N sources, Q queries, norm, S blend streams) and epilogue (res, addend, sum).
#[derive(Clone, Copy, Debug)]
struct Mix {
    c: usize,
    n: usize,
    q: usize,
    norm: bool,
    s: usize,
    res: bool,
    add: bool,
    sum: bool,
}

#[allow(clippy::too_many_arguments)]
fn conv_mix_case(d: &Dev, m: Mix, b: usize, poss: &[usize], plane: usize, seed: u64) -> usize {
    let mut rng = SplitMix64::new(seed);
    let c = m.c;
    let bf = DType::Bf16;
    // inputs of the B-row call (host bytes)
    let x = normal(&mut rng, bf, b * c, 1.0);
    let taps = normal(&mut rng, bf, K * c, 0.5);
    let res = m.res.then(|| normal(&mut rng, bf, b * c, 1.0));
    let add = m.add.then(|| normal(&mut rng, bf, b * c, 1.0));
    let src = normal(&mut rng, bf, m.n * b * c, 1.5);
    let qs = normal(&mut rng, DType::F32, m.q * c, 0.2);
    let nrows = if m.s > 0 { m.s } else { m.q };
    let norm = (m.norm || m.s > 0).then(|| uniform(&mut rng, nrows * c, 0.5, 1.5));
    let msg = (m.s > 0).then(|| normal(&mut rng, DType::F32, m.s * b * c, 1.0));
    let down = (m.s > 0).then(|| normal(&mut rng, bf, m.s * G * 2 * c, (2.0 * c as f32).powf(-0.5)));
    let gain = (m.s > 0).then(|| normal(&mut rng, bf, m.s * c * G, 0.7));
    let forget = (m.s > 0).then(|| normal(&mut rng, bf, m.s * c * G, 0.7));
    // each sequence's ring slot: a pad, then two planes [RR, c]; the call uses plane `plane`
    let ring_bytes = RING_PAD + 2 * RR * c * 2;
    let roff = RING_PAD + plane * RR * c * 2;
    let rings0: Vec<Vec<u8>> = (0..b).map(|_| normal(&mut rng, bf, ring_bytes / 2, 1.0)).collect();
    // the mix output: S planes with a blend, Q with a norm (bf16), Q f32 without, none for Q = 0
    let (mdt, mplanes) = if m.q == 0 {
        (bf, 0)
    } else if m.s > 0 {
        (bf, m.s)
    } else if m.norm {
        (bf, m.q)
    } else {
        (DType::F32, m.q)
    };
    let a = OpAttrs { pos0: 0, eps: 1e-6, ..OpAttrs::default() };
    let ws = |v: &Option<Vec<u8>>, dt: DType, shape: &[usize]| v.as_ref().map_or_else(|| none_t(dt), |v| tr(&d.buf(v), 0, dt, shape));
    let shared = |dd: &Dev| {
        (
            tr(&dd.buf(&taps), 0, bf, &[K, c]),
            tr(&dd.buf(&qs), 0, DType::F32, &[m.q, c]),
            ws(&norm, bf, &[nrows, c]),
            ws(&down, bf, &[m.s, G, 2 * c]),
            ws(&gain, bf, &[m.s, c, G]),
            ws(&forget, bf, &[m.s, c, G]),
        )
    };
    let (taps_t, qs_t, norm_t, down_t, gain_t, forget_t) = shared(d);

    // the batched call
    let rings: Vec<Buffer> = rings0.iter().map(|v| d.buf(v)).collect();
    let y = d.poison(b * c * 2);
    let sum = m.sum.then(|| d.poison(b * c * 2));
    let mesz = if mdt == DType::F32 { 4 } else { 2 };
    let mix = (mplanes > 0).then(|| d.poison(mplanes * b * c * mesz));
    let mut call = OpCall::new(
        Op::ConvMix,
        vec![
            tr(&d.buf(&x), 0, bf, &[b, c]),
            taps_t.clone(),
            ws(&res, bf, &[b, c]),
            ws(&add, bf, &[b, c]),
            tr(&d.buf(&src), 0, bf, &[m.n, b, c]),
            qs_t.clone(),
            norm_t.clone(),
            ws(&msg, DType::F32, &[m.s, b, c]),
            down_t.clone(),
            gain_t.clone(),
            forget_t.clone(),
        ],
        vec![
            tr(&y, 0, bf, &[b, c]),
            tr(&rings[0], roff, bf, &[RR, c]),
            sum.as_ref().map_or_else(|| none_t(bf), |s| tr(s, 0, bf, &[b, c])),
            mix.as_ref().map_or_else(|| none_t(bf), |x| tr(x, 0, mdt, &[mplanes, b, c])),
        ],
        a.clone(),
    );
    call.rows = Some(RowTab { ptrs: d.table(&rings.iter().map(|r| vec![d.addr(r)]).collect::<Vec<_>>()), rows: b, stride: STRIDE, index: INDEX, byte_off: 0 });
    call.pos = Some(d.pos(poss));
    d.launch("conv_mix_bf16_rows", call);
    let (yb, sumb) = (d.get(&y, b * c * 2), sum.as_ref().map(|s| d.get(s, b * c * 2)));
    let mixb = mix.as_ref().map(|x| d.get(x, mplanes * b * c * mesz));
    let ringsb: Vec<Vec<u8>> = rings.iter().map(|r| d.get(r, ring_bytes)).collect();

    // solo T = 1 calls, one per sequence, on a fresh copy of its ring
    let mut bad = 0;
    for r in 0..b {
        let ring = d.buf(&rings0[r]);
        let yr = d.poison(c * 2);
        let sr = m.sum.then(|| d.poison(c * 2));
        let mr = (mplanes > 0).then(|| d.poison(mplanes * c * mesz));
        let row = |v: &Option<Vec<u8>>| v.as_ref().map_or_else(|| none_t(bf), |v| tr(&d.buf(bf16_rows(v, r, c)), 0, bf, &[1, c]));
        let msg_r = msg.as_ref().map_or_else(|| none_t(DType::F32), |v| tr(&d.buf(&plane_rows(v, m.s, b, c, 4, r)), 0, DType::F32, &[m.s, 1, c]));
        let mut call = OpCall::new(
            Op::ConvMix,
            vec![
                tr(&d.buf(bf16_rows(&x, r, c)), 0, bf, &[1, c]),
                taps_t.clone(),
                row(&res),
                row(&add),
                tr(&d.buf(&plane_rows(&src, m.n, b, c, 2, r)), 0, bf, &[m.n, 1, c]),
                qs_t.clone(),
                norm_t.clone(),
                msg_r,
                down_t.clone(),
                gain_t.clone(),
                forget_t.clone(),
            ],
            vec![
                tr(&yr, 0, bf, &[1, c]),
                tr(&ring, roff, bf, &[RR, c]),
                sr.as_ref().map_or_else(|| none_t(bf), |s| tr(s, 0, bf, &[1, c])),
                mr.as_ref().map_or_else(|| none_t(bf), |x| tr(x, 0, mdt, &[mplanes, 1, c])),
            ],
            a.clone(),
        );
        call.pos = Some(d.pos(&poss[r..r + 1]));
        d.launch("conv_mix_bf16", call);
        let what = format!("{m:?} B={b} row {r} pos {} plane {plane}", poss[r]);
        let n = diff(bf16_rows(&yb, r, c), &d.get(&yr, c * 2));
        assert_eq!(n, 0, "{what}: y differs in {n} bytes");
        if let (Some(sb), Some(sr)) = (&sumb, &sr) {
            let n = diff(bf16_rows(sb, r, c), &d.get(sr, c * 2));
            assert_eq!(n, 0, "{what}: sum differs in {n} bytes");
        }
        if let (Some(mb), Some(mr)) = (&mixb, &mr) {
            let n = diff(&plane_rows(mb, mplanes, b, c, mesz, r), &d.get(mr, mplanes * c * mesz));
            assert_eq!(n, 0, "{what}: mix differs in {n} bytes");
        }
        let n = diff(&ringsb[r], &d.get(&ring, ring_bytes));
        assert_eq!(n, 0, "{what}: the ring buffer differs in {n} bytes");
        // the outputs are not trivially equal (the poison never survives a written row)
        assert!(bf16_rows(&yb, r, c).iter().any(|&v| v != 0xA5), "{what}: y not written");
        bad += n;
    }
    bad
}

#[test]
fn conv_mix_rows_bitwise_solo() {
    let Some(be) = device("conv_mix_rows_bitwise_solo") else { return };
    let _g = SWITCHES.lock().unwrap_or_else(|e| e.into_inner());
    let d = Dev::new(be.as_ref());
    // positions: before the conv history fills (0..K-1), around the ring wrap (RR), far past it
    let pos_list = [0usize, 1, 2, 3, 4, 17, 18, 19, 20, 37, 38, 100, 1023, 1024, 4228, 9001];
    let epis = [(false, false, false), (true, false, false), (true, false, true), (true, true, true)];
    let small: &[(usize, usize, bool, usize)] = &[(0, 0, false, 0), (2, 1, true, 0), (3, 3, true, 0), (1, 1, false, 0), (2, 1, false, 1), (3, 1, false, 3), (3, 3, false, 3)];
    let real: &[(usize, usize, bool, usize)] = &[(0, 0, false, 0), (4, 3, true, 0), (3, 1, true, 0), (3, 1, false, 3), (3, 3, false, 3)];
    let mut cases = 0usize;
    for (mode, dec3) in [(3, -1), (0, -1), (3, 0)] {
        intarsia_backend_hip::inkmix::set_mix_mode(mode);
        intarsia_backend_hip::set_dec3(dec3);
        let sw = intarsia_backend_hip::inkmix::rows_switches();
        eprintln!("conv_mix_bf16_rows: mix mode {mode}, dec3 {dec3}: the row-table TU reads switches {sw} (mask {})", sw & 255);
        assert_eq!(sw & 255, mode, "set_mix_mode({mode}) does not reach the row-table TU (its switches {sw})");
        for (c, mixes) in [(64usize, small), (2304, real)] {
            for &(n, q, norm, s) in mixes {
                for (ei, &(res, add, sum)) in epis.iter().enumerate() {
                    if c > 64 && ei == 1 {
                        continue;
                    }
                    let m = Mix { c, n, q, norm, s, res, add, sum };
                    for &b in &ROWS {
                        let poss = positions(&pos_list, b, cases);
                        conv_mix_case(&d, m, b, &poss, cases % 2, 1000 + cases as u64);
                        cases += 1;
                    }
                }
            }
        }
    }
    intarsia_backend_hip::inkmix::set_mix_mode(3);
    intarsia_backend_hip::set_dec3(-1);
    // the other switch reaches the row-table TU too (one shared copy, lane ink-hip-batch): the fast norm tail off / on / back to the env
    for (dec, on) in [(0, false), (1, true)] {
        intarsia_backend_hip::inkmix::set_mix_dec(dec);
        let sw = intarsia_backend_hip::inkmix::rows_switches();
        assert_eq!(sw & 256 != 0, on, "set_mix_dec({dec}) does not reach the row-table TU (its switches {sw})");
    }
    intarsia_backend_hip::inkmix::set_mix_dec(-1);
    eprintln!("conv_mix_bf16_rows: {cases} row-table calls bitwise the solo calls (every row, every ring byte)");
}

// ------------------------------------------------------------------------------------------------------------------------
// InkAttnDecode
// ------------------------------------------------------------------------------------------------------------------------

const H: usize = 24;
const HK: usize = 8;
const DH: usize = 128;
const DR: usize = 16;
const E: usize = 1024;
const CKV: usize = HK * DH;
const KV_PAD: usize = 256;
const CONV_PAD: usize = 128;

struct Layer {
    window: usize,
    ring: bool,
    cr: usize,
    dynamic: bool,
}

fn attn_attrs(l: &Layer) -> OpAttrs {
    let mut a = OpAttrs { pos0: 0, n_heads: H, n_kv_heads: HK, head_dim: DH, window: l.window, scale: (DH as f32).powf(-0.5), eps: 1e-6, ..OpAttrs::default() };
    a.ink.qk_clip = 6.0;
    a.ink.rel_clip = 1.5;
    a.ink.d_rel = DR;
    a.ink.kv_ring = l.ring;
    a.ink.dyn_conv = l.dynamic;
    if l.dynamic {
        (a.ink.tau_floor, a.ink.tau_alpha) = (1024, 0.1);
    }
    a
}

/// One sequence's attention state: the KV slot (a ring: one buffer, k / v planes after a pad; linear: two buffers) and the
/// K / V conv ring slot (one buffer, two planes after a pad), as host bytes.
struct AttnState {
    kv: Vec<Vec<u8>>,
    conv: Vec<u8>,
}

/// (k cache, v cache, k ring, v ring) carriers of a state's device buffers, and the table entries of the state.
fn carriers(d: &Dev, l: &Layer, kv: &[Buffer], conv: &Buffer) -> (Vec<TensorRef>, Vec<u64>) {
    let plane = HK * l.cr * DH * 2;
    let bf = DType::Bf16;
    let (kc, vc, e0, e1) = if l.ring {
        (tr(&kv[0], KV_PAD, bf, &[HK, l.cr, DH]), tr(&kv[0], KV_PAD + plane, bf, &[HK, l.cr, DH]), d.addr(&kv[0]), d.addr(&kv[0]))
    } else {
        (tr(&kv[0], 0, bf, &[HK, l.cr, DH]), tr(&kv[1], 0, bf, &[HK, l.cr, DH]), d.addr(&kv[0]), d.addr(&kv[1]))
    };
    let kr = tr(conv, CONV_PAD, bf, &[RR, CKV]);
    let vr = tr(conv, CONV_PAD + RR * CKV * 2, bf, &[RR, CKV]);
    (vec![kc, vc, kr, vr], vec![e0, e1, d.addr(conv), d.addr(conv)])
}

fn attn_case(d: &Dev, l: &Layer, b: usize, poss: &[usize], seed: u64) {
    let mut rng = SplitMix64::new(seed);
    let bf = DType::Bf16;
    let nqw = nq(H, DH, DR);
    let pw = if l.dynamic { CKV + HK * K } else { CKV };
    let pq = normal(&mut rng, bf, b * nqw, 1.0);
    let pk = normal(&mut rng, bf, b * pw, 1.0);
    let pv = normal(&mut rng, bf, b * pw, 1.0);
    let w = [
        tr(&d.buf(&normal(&mut rng, bf, K * CKV, 0.5)), 0, bf, &[K, CKV]),
        tr(&d.buf(&normal(&mut rng, bf, K * CKV, 0.5)), 0, bf, &[K, CKV]),
        tr(&d.buf(&uniform(&mut rng, H * DH, 0.5, 6.0)), 0, bf, &[H, DH]),
        tr(&d.buf(&uniform(&mut rng, H * DR, 0.5, 3.0)), 0, bf, &[H, DR]),
        tr(&d.buf(&normal(&mut rng, bf, DR * E, 0.25)), 0, bf, &[DR, E]),
    ];
    let plane = HK * l.cr * DH * 2;
    let conv_bytes = CONV_PAD + 2 * RR * CKV * 2;
    let states: Vec<AttnState> = (0..b)
        .map(|_| AttnState {
            kv: if l.ring { vec![normal(&mut rng, bf, (KV_PAD + 2 * plane) / 2, 1.2)] } else { vec![normal(&mut rng, bf, plane / 2, 1.5), normal(&mut rng, bf, plane / 2, 1.0)] },
            conv: normal(&mut rng, bf, conv_bytes / 2, 1.0),
        })
        .collect();
    let a = attn_attrs(l);

    // the batched call
    let kvs: Vec<Vec<Buffer>> = states.iter().map(|s| s.kv.iter().map(|v| d.buf(v)).collect()).collect();
    let convs: Vec<Buffer> = states.iter().map(|s| d.buf(&s.conv)).collect();
    let o = d.poison(b * H * DH * 2);
    let (car0, _) = carriers(d, l, &kvs[0], &convs[0]);
    let ents: Vec<Vec<u64>> = (0..b).map(|r| carriers(d, l, &kvs[r], &convs[r]).1).collect();
    let (pqb, pkb, pvb) = (d.buf(&pq), d.buf(&pk), d.buf(&pv));
    let mut inputs = vec![tr(&pqb, 0, bf, &[b, nqw]), tr(&pkb, 0, bf, &[b, pw]), tr(&pvb, 0, bf, &[b, pw])];
    inputs.extend(w.iter().cloned());
    inputs.extend([car0[0].clone(), car0[1].clone()]);
    let mut outputs = vec![tr(&o, 0, bf, &[b, H * DH])];
    outputs.extend(car0.iter().cloned());
    let mut call = OpCall::new(Op::InkAttnDecode, inputs, outputs, a.clone());
    call.rows = Some(RowTab { ptrs: d.table(&ents), rows: b, stride: STRIDE, index: INDEX, byte_off: 0 });
    call.pos = Some(d.pos(poss));
    d.launch("ink_attn_decode_bf16_rows", call);
    let ob = d.get(&o, b * H * DH * 2);
    let kv_after: Vec<Vec<Vec<u8>>> = kvs.iter().zip(&states).map(|(bs, s)| bs.iter().zip(&s.kv).map(|(x, h)| d.get(x, h.len())).collect()).collect();
    let conv_after: Vec<Vec<u8>> = convs.iter().map(|x| d.get(x, conv_bytes)).collect();

    // solo T = 1 calls at a device position, one per sequence, on fresh copies of its state
    for r in 0..b {
        let kv: Vec<Buffer> = states[r].kv.iter().map(|v| d.buf(v)).collect();
        let conv = d.buf(&states[r].conv);
        let (car, _) = carriers(d, l, &kv, &conv);
        let or = d.poison(H * DH * 2);
        let mut inputs = vec![tr(&pqb, r * nqw * 2, bf, &[1, nqw]), tr(&pkb, r * pw * 2, bf, &[1, pw]), tr(&pvb, r * pw * 2, bf, &[1, pw])];
        inputs.extend(w.iter().cloned());
        inputs.extend([car[0].clone(), car[1].clone()]);
        let mut outputs = vec![tr(&or, 0, bf, &[1, H * DH])];
        outputs.extend(car.iter().cloned());
        let mut call = OpCall::new(Op::InkAttnDecode, inputs, outputs, a.clone());
        call.pos = Some(d.pos(&poss[r..r + 1]));
        d.launch("ink_attn_decode_bf16", call);
        let what = format!("window {} ring {} cr {} B={b} row {r} pos {}", l.window, l.ring, l.cr, poss[r]);
        let n = diff(bf16_rows(&ob, r, H * DH), &d.get(&or, H * DH * 2));
        assert_eq!(n, 0, "{what}: o differs in {n} bytes");
        assert!(bf16_rows(&ob, r, H * DH).iter().any(|&v| v != 0xA5), "{what}: o not written");
        for (j, (x, h)) in kv.iter().zip(&states[r].kv).enumerate() {
            let n = diff(&kv_after[r][j], &d.get(x, h.len()));
            assert_eq!(n, 0, "{what}: KV buffer {j} differs in {n} bytes");
        }
        let n = diff(&conv_after[r], &d.get(&conv, conv_bytes));
        assert_eq!(n, 0, "{what}: the K / V conv ring buffer differs in {n} bytes");
    }
}

#[test]
fn attn_decode_rows_bitwise_solo() {
    let Some(be) = device("attn_decode_rows_bitwise_solo") else { return };
    let _g = SWITCHES.lock().unwrap_or_else(|e| e.into_inner());
    let d = Dev::new(be.as_ref());
    let swa = Layer { window: 1024, ring: true, cr: 1040, dynamic: false };
    // full: a capacity above INTARSIA_INK_DEC_FUSED_KEYS (2048), so the solo device-position launch takes the split path
    let full = Layer { window: 0, ring: false, cr: 2600, dynamic: true };
    let swa_pos = [0usize, 1, 2, 3, 18, 19, 63, 64, 1022, 1023, 1024, 1039, 1040, 1041, 2079, 2080, 4228, 9000];
    let full_pos = [0usize, 3, 64, 1018, 1023, 1024, 1025, 1500, 2047, 2048, 2049, 2500, 2598, 2599];
    let mut cases = 0usize;
    for dec3 in [-1, 0] {
        intarsia_backend_hip::set_dec3(dec3);
        for (l, list) in [(&swa, &swa_pos[..]), (&full, &full_pos[..])] {
            for &b in &ROWS {
                for rep in 0..2 {
                    let poss = positions(list, b, cases + rep);
                    attn_case(&d, l, b, &poss, 77 + cases as u64);
                    cases += 1;
                }
            }
        }
    }
    intarsia_backend_hip::set_dec3(-1);
    eprintln!("ink_attn_decode_bf16_rows: {cases} row-table calls bitwise the solo calls (every row, every cache / ring byte)");
}
