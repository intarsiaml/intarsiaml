//! Chunked == whole: the exactness gate that chunked prefill and speculative decode rely on. A layer-2-shaped attention
//! path (RoPE'd window KV in a ring, ratio-`r` compressor with partial-group state, a second compressor for the index keys,
//! indexer top-k (optionally over candidate blocks), sparse attention, output de-rotation) is run on the same token stream
//! in one chunk, in many chunk schedules, as prefill + one-token decode, and after rolled-back speculation. The rings are
//! `SlotSpec::Ring` rows (`p % rows`, in place) and the compressed rows `SlotSpec::Shared` rows (valid up to
//! `len / ratio`, stale rows past a truncate are overwritten, never cleared). Every result must be bit-identical: the fixtures
//! show the official code is (`attn4k_L2_bf16_report.json`), and a kernel set that is not cannot verify drafts.
mod dsv41_common;

use dsv41_common::*;
use intarsia_core::exec::OutSpec;
use intarsia_core::rng::SplitMix64;
use intarsia_core::{DType, HostTensor, Op, OpAttrs};

#[derive(Clone)]
struct Cfg {
    w: usize,
    /// ring rows of the window / compressor tails (>= w, >= ratio, multiple of ratio)
    wc: usize,
    rc: usize,
    r: usize,
    d: usize,
    h: usize,
    hi: usize,
    di: usize,
    topk: usize,
    rd: usize,
    /// candidate blocks: (block length, blocks kept), as layers 24+ (`candidate_source_layer`)
    cand: Option<(usize, usize)>,
    dt: DType,
}

struct Tokens {
    kv: Vec<f32>,
    ckv: Vec<f32>,
    csc: Vec<f32>,
    ikv: Vec<f32>,
    isc: Vec<f32>,
    q: Vec<f32>,
    iq: Vec<f32>,
    iw: Vec<f32>,
}

fn tokens(c: &Cfg, n: usize, seed: u64) -> Tokens {
    let mut rng = SplitMix64::new(seed);
    let mut v = |len: usize, sc: f32| -> Vec<f32> { HostTensor::from_f32_as(c.dt, &[len], &rng.normal_vec(len, sc)).to_f32() };
    Tokens { kv: v(n * c.d, 1.0), ckv: v(n * c.d, 1.0), csc: v(n * c.d, 1.5), ikv: v(n * c.di, 1.0), isc: v(n * c.di, 1.5), q: v(n * c.h * c.d, 1.0), iq: v(n * c.hi * c.di, 1.0), iw: v(n * c.hi, 0.5) }
}

/// The state one layer keeps for a sequence.
#[derive(Clone)]
struct State {
    win_ring: HostTensor,
    ckv_ring: HostTensor,
    csc_ring: HostTensor,
    ikv_ring: HostTensor,
    isc_ring: HostTensor,
    /// `Shared` slots: compressed KV rows and index-key rows (row `g` valid for `g < len / ratio`)
    cmp: Vec<f32>,
    ik: Vec<f32>,
}

fn st_new(c: &Cfg) -> State {
    let z = |dt, rows, w| HostTensor::zeros(dt, &[rows, w]);
    State { win_ring: z(c.dt, c.wc, c.d), ckv_ring: z(DType::F32, c.rc, c.d), csc_ring: z(DType::F32, c.rc, c.d), ikv_ring: z(DType::F32, c.rc, c.di), isc_ring: z(DType::F32, c.rc, c.di), cmp: vec![], ik: vec![] }
}

fn inv_freq(rd: usize) -> Vec<f32> {
    (0..rd / 2).map(|j| 10000f32.powf(-((2 * j) as f32) / rd as f32)).collect()
}

fn rope(x: &HostTensor, c: &Cfg, pos0: usize, stride: usize, inverse: bool, heads: usize, width: usize) -> HostTensor {
    let mut a = OpAttrs { pos0, ..Default::default() };
    (a.dsv.pos_stride, a.dsv.inverse) = (stride, inverse);
    let t = x.numel() / (heads * width);
    let hc = call(Op::RopeTail, vec![x.clone().reshaped(&[t, heads, width]), f32t(&[c.rd / 2], &inv_freq(c.rd))], vec![OutSpec::new(x.dtype, &[t, heads, width])], a);
    go(&hc).remove(0)
}

/// Pool `kv`/`score` rows of the chunk into the rings; the new groups' rows (empty when none complete).
fn compress(c: &Cfg, kv: &[f32], sc: &[f32], w: usize, kv_ring: &mut HostTensor, sc_ring: &mut HostTensor, pos0: usize, t: usize) -> Vec<f32> {
    if c.r == 1 {
        return kv.to_vec();
    }
    let g = (pos0 + t) / c.r - pos0 / c.r;
    let mut a = OpAttrs { pos0, ..Default::default() };
    a.dsv.ratio = c.r;
    let hc = call(
        Op::CompressPool,
        vec![f32t(&[t, w], kv), f32t(&[t, w], sc)],
        vec![OutSpec::new(c.dt, &[g, w]), OutSpec { dtype: DType::F32, shape: vec![c.rc, w], init: Some(kv_ring.clone()) }, OutSpec { dtype: DType::F32, shape: vec![c.rc, w], init: Some(sc_ring.clone()) }],
        a,
    );
    let o = go(&hc);
    *kv_ring = o[1].clone();
    *sc_ring = o[2].clone();
    o[0].to_f32()
}

fn put_rows(store: &mut Vec<f32>, row0: usize, w: usize, rows: &[f32]) {
    store.truncate(row0 * w); // rows past a truncate are stale: overwritten from `row0`
    store.extend_from_slice(rows);
}

/// One chunk of `t` tokens at `pos0`: returns the attention output `[t, h, d]` after the de-rotation.
fn step(c: &Cfg, s: &mut State, tk: &Tokens, pos0: usize, t: usize) -> Vec<f32> {
    let (d, h, hi, di) = (c.d, c.h, c.hi, c.di);
    let sl = |v: &[f32], w: usize| v[pos0 * w..(pos0 + t) * w].to_vec();
    // window KV: rotated rows into the ring, `win` = the last W rows before the chunk + the chunk
    let kv = HostTensor::from_f32_as(c.dt, &[t, 1, d], &sl(&tk.kv, d));
    let kv = rope(&kv, c, pos0, 1, false, 1, d).reshaped(&[t, d]);
    let mut a = OpAttrs { pos0, window: c.w, ..Default::default() };
    a.dsv.ratio = c.r;
    let hc = call(Op::WindowKv, vec![kv], vec![OutSpec::new(c.dt, &[c.w + t, d]), OutSpec { dtype: c.dt, shape: vec![c.wc, d], init: Some(s.win_ring.clone()) }], a);
    let o = go(&hc);
    let win = o[0].clone();
    s.win_ring = o[1].clone();
    // compressed KV (rotated at the group's first position) and index keys
    let g0 = pos0 / c.r;
    let rows = compress(c, &sl(&tk.ckv, d), &sl(&tk.csc, d), d, &mut s.ckv_ring, &mut s.csc_ring, pos0, t);
    let ng = rows.len() / d;
    let rows = if ng > 0 { rope(&HostTensor::from_f32_as(c.dt, &[ng, 1, d], &rows), c, g0 * c.r, c.r, false, 1, d).to_f32() } else { rows };
    put_rows(&mut s.cmp, g0, d, &rows);
    let irows = compress(c, &sl(&tk.ikv, di), &sl(&tk.isc, di), di, &mut s.ikv_ring, &mut s.isc_ring, pos0, t);
    put_rows(&mut s.ik, g0, di, &HostTensor::from_f32_as(c.dt, &[irows.len()], &irows).to_f32());
    // indexer over the compressed rows visible to the chunk
    let nk = (pos0 + t) / c.r;
    let idx = if nk == 0 {
        HostTensor::from_i32(&[t, 1], &vec![-1; t])
    } else {
        let mut a = OpAttrs { pos0, scale: 0.25, top_k: c.topk, ..Default::default() };
        a.dsv.ratio = c.r;
        let (q, k, w) = (HostTensor::from_f32_as(c.dt, &[t, hi, di], &sl(&tk.iq, hi * di)), HostTensor::from_f32_as(c.dt, &[nk, di], &s.ik[..nk * di]), HostTensor::from_f32_as(c.dt, &[t, hi], &sl(&tk.iw, hi)));
        let kk = c.topk.min(nk);
        let cand = match c.cand {
            Some((b, keep)) => {
                let mut ab = a.clone();
                (ab.top_k, ab.dsv.block) = (keep, b);
                let nb = nk.div_ceil(b);
                go(&call(Op::IndexerBlocks, vec![q.clone(), k.clone(), w.clone()], vec![OutSpec::new(DType::U8, &[t, nb])], ab)).remove(0)
            }
            None => HostTensor::zeros(DType::U8, &[0]),
        };
        a.dsv.block = c.cand.map_or(0, |x| x.0);
        go(&call(Op::IndexerTopk, vec![q, k, w, cand], vec![OutSpec::new(DType::I32, &[t, kk])], a)).remove(0)
    };
    // sparse attention
    let ncmp = nk.max(1);
    let mut cmp = s.cmp[..nk * d].to_vec();
    cmp.resize(ncmp * d, 0.0);
    let a = OpAttrs { window: c.w, pos0, scale: (d as f32).powf(-0.5), ..Default::default() };
    let q = HostTensor::from_f32_as(c.dt, &[t, h, d], &sl(&tk.q, h * d));
    let sink = HostTensor::from_f32(&[h], &(0..h).map(|i| 0.1 * i as f32 - 0.2).collect::<Vec<_>>());
    let hc = call(Op::AttnSparse, vec![q, win, HostTensor::from_f32_as(c.dt, &[ncmp, d], &cmp), idx, sink], vec![OutSpec::new(c.dt, &[t, h, d])], a);
    let o = go(&hc).remove(0);
    rope(&o, c, pos0, 1, true, h, d).to_f32()
}

fn run(c: &Cfg, tk: &Tokens, schedule: &[usize]) -> (Vec<f32>, State) {
    let mut s = st_new(c);
    let (mut pos, mut out) = (0, vec![]);
    for &t in schedule {
        out.extend(step(c, &mut s, tk, pos, t));
        pos += t;
    }
    (out, s)
}

fn base(r: usize, dt: DType, cand: Option<(usize, usize)>) -> Cfg {
    let (w, margin) = (16, 16);
    Cfg { w, wc: w + margin, rc: r * (1 + margin.div_ceil(r)), r, d: 32, h: 4, hi: 4, di: 16, topk: 8, rd: 8, cand, dt }
}

const N: usize = 211;

fn schedules() -> Vec<(&'static str, Vec<usize>)> {
    let ones = vec![1; N];
    let mut odd = vec![];
    let (mut left, mut k) = (N, 0);
    while left > 0 {
        let t = [7usize, 13, 1, 100, 2, 3, 40, 1][k % 8].min(left);
        odd.push(t);
        left -= t;
        k += 1;
    }
    vec![("512-style equal chunks of 64", [vec![64; 3], vec![N - 192]].concat()), ("odd sizes 7,13,1,100,2,3,40,1,..", odd), ("prefill N-3 + 3 decode steps", vec![N - 3, 1, 1, 1]), ("prefill 100 + one-token decode", [vec![100], vec![1; N - 100]].concat()), ("all one-token (pure decode from an empty state)", ones), ("chunks larger than the window ring (T > Wc)", vec![50, 61, 100])]
}

fn same_state(a: &State, b: &State, what: &str) {
    assert_eq!(a.win_ring.data, b.win_ring.data, "{what}: window ring");
    assert_eq!(a.cmp, b.cmp, "{what}: compressed KV rows");
    assert_eq!(a.ik, b.ik, "{what}: index key rows");
    assert_eq!(a.ckv_ring.data, b.ckv_ring.data, "{what}: compressor kv tail");
    assert_eq!(a.csc_ring.data, b.csc_ring.data, "{what}: compressor score tail");
    assert_eq!(a.ikv_ring.data, b.ikv_ring.data, "{what}: index compressor kv tail");
    assert_eq!(a.isc_ring.data, b.isc_ring.data, "{what}: index compressor score tail");
}

fn chunked_equals_whole(c: &Cfg, label: &str) {
    let tk = tokens(c, N + 40, 0xC0FFEE ^ c.r as u64);
    let (whole, ws) = run(c, &tk, &[N]);
    assert!(whole.iter().any(|v| *v != 0.0) && whole.iter().all(|v| v.is_finite()));
    for (name, sch) in schedules() {
        assert_eq!(sch.iter().sum::<usize>(), N, "{name}");
        let (out, st) = run(c, &tk, &sch);
        assert!(out == whole, "{label}: schedule `{name}` differs from the single-chunk run (first diff at {:?})", out.iter().zip(&whole).position(|(a, b)| a != b));
        same_state(&st, &ws, &format!("{label} / {name}"));
        println!("PARITY chunked==whole {label:<34} {name:<48} bit-identical output ({} values) and state", whole.len());
    }
}

#[test]
fn chunked_equals_whole_ratio2_bf16() {
    chunked_equals_whole(&base(2, DType::Bf16, None), "ratio 2 bf16 (layers 2-19)");
}

#[test]
fn chunked_equals_whole_ratio2_f32_and_odd_ratio() {
    chunked_equals_whole(&base(2, DType::F32, None), "ratio 2 f32");
    // ratio 3 puts group boundaries at odd offsets of the chunk schedules (the model only uses 1 and 2)
    chunked_equals_whole(&base(3, DType::Bf16, None), "ratio 3 bf16 (stress)");
}

#[test]
fn chunked_equals_whole_ratio1_and_candidates() {
    chunked_equals_whole(&base(1, DType::Bf16, None), "ratio 1 (layer 20..)");
    chunked_equals_whole(&base(1, DType::Bf16, Some((4, 5))), "ratio 1 + candidate blocks (24..)");
    chunked_equals_whole(&base(2, DType::Bf16, Some((4, 5))), "ratio 2 + candidate blocks");
}

/// Speculative decode: draft `k <= margin` tokens past the accepted length (different tokens than the real continuation),
/// roll back (`SequenceState::truncate`: the rings keep their stale rows, the shared rows are cut by length), and continue
/// with the real tokens. The result must equal the run that never drafted. A ring without the rollback margin cannot do it.
fn rollback_run(c: &Cfg, accepted: usize, draft: usize) -> (Vec<f32>, State) {
    let real = tokens(c, N + 40, 0xC0FFEE ^ c.r as u64);
    let junk = tokens(c, N + 40, 0xBAD5EED);
    // the drafted tokens replace the real ones at positions accepted .. accepted + draft
    let mut mix = tokens(c, N + 40, 0xC0FFEE ^ c.r as u64);
    let cp = |dst: &mut Vec<f32>, src: &Vec<f32>, w: usize| dst[accepted * w..(accepted + draft) * w].copy_from_slice(&src[accepted * w..(accepted + draft) * w]);
    cp(&mut mix.kv, &junk.kv, c.d);
    cp(&mut mix.ckv, &junk.ckv, c.d);
    cp(&mut mix.csc, &junk.csc, c.d);
    cp(&mut mix.ikv, &junk.ikv, c.di);
    cp(&mut mix.isc, &junk.isc, c.di);
    cp(&mut mix.q, &junk.q, c.h * c.d);
    cp(&mut mix.iq, &junk.iq, c.hi * c.di);
    cp(&mut mix.iw, &junk.iw, c.hi);
    let mut s = st_new(c);
    let mut pos = 0;
    for t in [accepted - 1, 1] {
        step(c, &mut s, &mix, pos, t);
        pos += t;
    }
    // verify pass of `draft` tokens, then roll back to `accepted`
    step(c, &mut s, &mix, accepted, draft);
    // continue with the real tokens from `accepted`
    let (mut out, mut pos) = (vec![], accepted);
    for t in [3usize, 1, N - accepted - 4] {
        out.extend(step(c, &mut s, &real, pos, t));
        pos += t;
    }
    (out, s)
}

#[test]
fn rollback_within_the_margin_equals_never_drafting() {
    for (r, accepted, draft) in [(2usize, 101usize, 16usize), (2, 100, 9), (2, 77, 1), (1, 90, 16), (3, 61, 16)] {
        let c = base(r, DType::Bf16, None);
        let real = tokens(&c, N + 40, 0xC0FFEE ^ r as u64);
        let (want, ws) = run(&c, &real, &[accepted, 3, 1, N - accepted - 4]);
        let (got, st) = rollback_run(&c, accepted, draft);
        // outputs for positions accepted .. N
        let skip = accepted * c.h * c.d;
        assert!(got == want[skip..], "ratio {r}, accepted {accepted}, draft {draft}: rolled-back run differs");
        same_state(&st, &ws, &format!("rollback r={r} accepted={accepted} draft={draft}"));
        println!("PARITY rollback ratio {r}: drafted {draft} tokens past {accepted}, truncated, continued: bit-identical to never drafting");
    }
}

#[test]
fn rollback_beyond_the_margin_is_detected_as_unsafe() {
    // negative control: window ring of exactly W rows (no margin) loses rows a rollback needs
    let mut c = base(2, DType::Bf16, None);
    (c.wc, c.rc) = (c.w, c.r);
    let real = tokens(&c, N + 40, 0xC0FFEE ^ 2);
    let (want, _) = run(&c, &real, &[101, 3, 1, N - 105]);
    let (got, _) = rollback_run(&c, 101, 4);
    assert!(got != want[101 * c.h * c.d..], "a ring without margin must not survive a rollback (this control checks the test can tell)");
}
