#![cfg(target_os = "macos")]
//! The Inkling-2 controller-fusion kernels (inkctrl.rs, lane ink-dfuse) on Metal:
//! - `ink_route_bf16` (op 76) is bitwise the Metal pair it replaces (`Logits` on `logits_bf16_gemv` / `_rows`, then
//!   `moe_route_f32`) for rows 1..=16 and a 40-row call, both GEMV forms (16-byte and 2-byte operands), with and without the
//!   logits output, at the real router shape (D 2304, 128 experts, top-8) and small ones;
//! - an R-row call is bitwise R one-row calls (rows-exact);
//! - a stress run of many calls in one command buffer across all shapes and row counts (the hand-off counters reset in-kernel
//!   and a watchdog fails the test instead of hanging the suite). `INK_CTRL_STRESS=<n>` sets the call count (default 2000;
//!   the lane's gate runs 12000).
//! - the fused blend tail (`ink_blend_tail_g*`, one dispatch for the h GEMV, gate and norm of a blend chunk of up to 16 rows) is
//!   bitwise the three dispatches it replaces, for LoopBlend and ConvMix with a blend, and survives the same stress run.
//!
//! Numerics against the f64 reference are `intarsia kernels test --backend metal --op ink_route` (the admission sweep).

use intarsia_backend_metal::bench::{f32_to_bf16, Rng};
use intarsia_backend_metal::MetalBackend;
use intarsia_core::backend::{Backend, MemKind, TensorRef};
use intarsia_core::dsv41::ops::ScoreFunc;
use intarsia_core::{DType, Op, OpAttrs, OpCall};
use std::sync::mpsc;
use std::time::Duration;

const FUSED: &str = intarsia_backend_metal::inkctrl::INK_ROUTE_BF16;

fn be() -> Option<MetalBackend> {
    let b = MetalBackend::new().expect("Metal backend");
    if !b.has_metal4() {
        eprintln!("skip: no metal4.0 library (ink_route_bf16 is not registered)");
        return None;
    }
    Some(b)
}

/// A device tensor of `dt` with `shape`, filled from `vals` (bf16 or f32 / i32 bytes), at byte `off` of its buffer.
fn dev(be: &MetalBackend, dt: DType, shape: &[usize], bytes: &[u8], off: usize) -> TensorRef {
    let b = be.alloc((off + bytes.len()).max(16), MemKind::Device).unwrap();
    if !bytes.is_empty() {
        be.upload(&b, off, bytes, 0).unwrap();
    }
    TensorRef { buf: b, offset: off, dtype: dt, shape: shape.to_vec() }
}

fn bf16_bytes(n: usize, scale: f32, rng: &mut Rng) -> Vec<u8> {
    (0..n).flat_map(|_| f32_to_bf16(rng.unit() * scale).to_le_bytes()).collect()
}

fn f32_bytes(n: usize, scale: f32, rng: &mut Rng) -> Vec<u8> {
    (0..n).flat_map(|_| (rng.unit() * scale).to_le_bytes()).collect()
}

fn down(be: &MetalBackend, t: &TensorRef) -> Vec<u8> {
    let mut v = vec![0u8; t.numel() * t.dtype.size_bytes()];
    if !v.is_empty() {
        be.download(&t.buf, t.offset, &mut v, 0).unwrap();
    }
    v
}

#[derive(Clone, Copy, Debug)]
struct Shape {
    d: usize,
    e: usize,
    k: usize,
    scale: f32,
    sf: ScoreFunc,
    norm: bool,
    temp: f32,
    /// byte offset of x / w in their buffers (2 = the 2-byte GEMV form)
    off: usize,
}

fn shapes() -> Vec<Shape> {
    let s = |d, e, k, scale, sf, norm, temp, off| Shape { d, e, k, scale, sf, norm, temp, off };
    vec![
        s(2304, 128, 8, 8.0, ScoreFunc::Sigmoid, true, 1.0, 0),
        s(2304, 128, 8, 8.0, ScoreFunc::Sigmoid, true, 1.0, 2),
        s(64, 16, 4, 1.5, ScoreFunc::Softmax, true, 1.0, 0),
        s(60, 40, 6, 1.0, ScoreFunc::SqrtSoftplus, true, 1.0, 0),
        s(96, 33, 5, 2.0, ScoreFunc::Sigmoid, false, 0.7, 2),
    ]
}

struct Case {
    x: TensorRef,
    w: TensorRef,
    bias: TensorRef,
    attrs: OpAttrs,
    t: usize,
    e: usize,
    k: usize,
}

fn case(be: &MetalBackend, sh: Shape, t: usize, rng: &mut Rng) -> Case {
    let x = dev(be, DType::Bf16, &[t, sh.d], &bf16_bytes(t * sh.d, 1.0, rng), sh.off);
    let w = dev(be, DType::Bf16, &[sh.e, sh.d], &bf16_bytes(sh.e * sh.d, (sh.d as f32).powf(-0.5), rng), sh.off);
    let bias = dev(be, DType::F32, &[sh.e], &f32_bytes(sh.e, 0.1, rng), 0);
    let mut attrs = OpAttrs { top_k: sh.k, scale: sh.scale, norm_topk: sh.norm, temperature: sh.temp, ..OpAttrs::default() };
    attrs.dsv.score_func = sh.sf;
    Case { x, w, bias, attrs, t, e: sh.e, k: sh.k }
}

/// The fused call on rows `r0..r0 + n` of the case; `logits` = whether the call takes the logits output. Returns
/// (weights, ids, logits) bytes; encodes without waiting when `wait` is false.
fn fused(be: &MetalBackend, c: &Case, r0: usize, n: usize, logits: bool, wait: bool) -> Option<[TensorRef; 3]> {
    let wo = dev(be, DType::F32, &[n, c.k], &[], 0);
    let ids = dev(be, DType::I32, &[n, c.k], &[], 0);
    let lg = if logits { dev(be, DType::F32, &[n, c.e], &[], 0) } else { dev(be, DType::F32, &[0], &[], 0) };
    let mut call = OpCall::new(Op::InkRoute, vec![c.x.rows(r0, n), c.w.clone(), c.bias.clone()], vec![wo.clone(), ids.clone(), lg.clone()], c.attrs.clone());
    call.check().unwrap();
    if !logits {
        let sb = be.alloc(n * c.e * 4 + 16, MemKind::Device).unwrap();
        call.scratch = Some(TensorRef { buf: sb, offset: 0, dtype: DType::U8, shape: vec![n * c.e * 4] });
    }
    be.launch(FUSED, &call, 0).unwrap();
    if wait {
        be.synchronize(0).unwrap();
    }
    Some([wo, ids, lg])
}

/// The Metal pair on the case's rows: `Logits` (the rows kernel for 2..=16 rows under exact_rows, as the graph runs it) then
/// `moe_route_f32`.
fn pair(be: &MetalBackend, c: &Case) -> [Vec<u8>; 3] {
    let t = c.t;
    let lg = dev(be, DType::F32, &[t, c.e], &[], 0);
    let mut lc = OpCall::new(Op::Logits, vec![c.x.clone(), c.w.clone()], vec![lg.clone()], OpAttrs::default());
    lc.exact_rows = (2..=16).contains(&t);
    lc.check().unwrap();
    let kern = if t == 1 { "logits_bf16_gemv" } else if t <= 16 { "logits_bf16_gemv_rows" } else { "" };
    if kern.is_empty() {
        for r in 0..t {
            let one = OpCall::new(Op::Logits, vec![c.x.rows(r, 1), c.w.clone()], vec![lg.rows(r, 1)], OpAttrs::default());
            be.launch("logits_bf16_gemv", &one, 0).unwrap();
        }
    } else {
        be.launch(kern, &lc, 0).unwrap();
    }
    let wo = dev(be, DType::F32, &[t, c.k], &[], 0);
    let ids = dev(be, DType::I32, &[t, c.k], &[], 0);
    let rc = OpCall::new(Op::MoeRoute, vec![lg.clone(), c.bias.clone()], vec![wo.clone(), ids.clone()], c.attrs.clone());
    rc.check().unwrap();
    be.launch("moe_route_f32", &rc, 0).unwrap();
    be.synchronize(0).unwrap();
    [down(be, &wo), down(be, &ids), down(be, &lg)]
}

#[test]
fn ink_route_is_the_metal_pair_bitwise_and_rows_exact() {
    let Some(be) = be() else { return };
    let mut rng = Rng(0x5eed_76);
    let mut n = 0;
    for sh in shapes() {
        for t in [1usize, 2, 3, 4, 7, 16, 40] {
            let c = case(&be, sh, t, &mut rng);
            let want = pair(&be, &c);
            for logits in [true, false] {
                let got = fused(&be, &c, 0, t, logits, true).unwrap();
                let what = format!("{sh:?} T={t} logits={logits}");
                assert_eq!(down(&be, &got[0]), want[0], "{what}: weights");
                assert_eq!(down(&be, &got[1]), want[1], "{what}: ids");
                if logits {
                    assert_eq!(down(&be, &got[2]), want[2], "{what}: logits");
                }
            }
            // rows-exact: row r of the T-row call == the 1-row call on row r
            let all = fused(&be, &c, 0, t, true, true).unwrap();
            let (wa, ia) = (down(&be, &all[0]), down(&be, &all[1]));
            for r in 0..t {
                let one = fused(&be, &c, r, 1, true, true).unwrap();
                let kb = c.k * 4;
                assert_eq!(down(&be, &one[0]), wa[r * kb..(r + 1) * kb], "{sh:?} T={t} row {r}: weights");
                assert_eq!(down(&be, &one[1]), ia[r * kb..(r + 1) * kb], "{sh:?} T={t} row {r}: ids");
            }
            n += 1;
        }
    }
    eprintln!("{n} cases bitwise the pair and rows-exact");
}

/// Many fused calls back to back in one stream (no wait between them), every shape and row count, then each result checked
/// against its own single call. A hand-off that failed to reset a counter would make a later call's last threadgroup never
/// fire (stale outputs: caught by the comparison) or fire early (wrong outputs); a hang trips the watchdog.
#[test]
fn ink_route_stress_no_hang_no_stale_counter() {
    let Some(be) = be() else { return };
    let calls: usize = std::env::var("INK_CTRL_STRESS").ok().and_then(|v| v.parse().ok()).unwrap_or(2000);
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        let be = be;
        let mut rng = Rng(0x57_5e55);
        let rows = [1usize, 2, 3, 4, 5, 8, 16, 40];
        let cases: Vec<(Case, [Vec<u8>; 3])> = shapes()
            .into_iter()
            .flat_map(|sh| rows.iter().map(move |&t| (sh, t)))
            .map(|(sh, t)| {
                let c = case(&be, sh, t, &mut rng);
                let want = pair(&be, &c);
                (c, want)
            })
            .collect();
        let mut done = 0;
        let mut bad = 0;
        let mut worst = Duration::ZERO;
        while done < calls {
            let batch = 200.min(calls - done);
            let t0 = std::time::Instant::now();
            let mut outs = Vec::with_capacity(batch);
            for j in 0..batch {
                let ci = (done + j) * 7 % cases.len();
                let logits = (done + j) % 3 != 0;
                outs.push((ci, logits, fused(&be, &cases[ci].0, 0, cases[ci].0.t, logits, false).unwrap()));
            }
            be.synchronize(0).unwrap();
            worst = worst.max(t0.elapsed());
            for (ci, logits, o) in outs {
                let want = &cases[ci].1;
                if down(&be, &o[0]) != want[0] || down(&be, &o[1]) != want[1] || (logits && down(&be, &o[2]) != want[2]) {
                    bad += 1;
                }
            }
            done += batch;
        }
        let _ = tx.send((done, bad, worst));
    });
    let budget = Duration::from_secs(30 + calls as u64 / 50);
    match rx.recv_timeout(budget) {
        Ok((done, bad, worst)) => {
            assert_eq!(bad, 0, "{bad} of {done} stress calls differ from their single call");
            eprintln!("STRESS ink_route: {done} calls, all bitwise; slowest batch of 200 (encode + GPU + wait) {:.2} ms (watchdog {budget:?})", worst.as_secs_f64() * 1e3);
        }
        Err(_) => panic!("ink_route stress: no result within {budget:?} (GPU hang or stall)"),
    }
}

// ---- the fused blend tail (ink_blend_tail) == the h GEMV + gate + norm dispatches, bitwise -----------------------------------

fn bf_pos(n: usize, rng: &mut Rng) -> Vec<u8> {
    (0..n).flat_map(|_| f32_to_bf16(1.0 + 0.5 * rng.unit()).to_le_bytes()).collect()
}

fn blend_scratch_for(be: &MetalBackend, call: &OpCall) -> TensorRef {
    let n = intarsia_backend_metal::inkmix::blend_scratch(&intarsia_core::ShapeInfo::from_call(call));
    let b = be.alloc(n.max(16), MemKind::Device).unwrap();
    TensorRef { buf: b, offset: 0, dtype: DType::U8, shape: vec![n] }
}

/// Run `kernel` on `op` (fresh outputs from `outs` bytes) with the tail forced `tail`; returns every output's bytes.
/// The blend tests flip the process-wide tail switch: one at a time.
static BLEND_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn run_blend(be: &MetalBackend, kernel: &str, op: Op, ins: &[(DType, Vec<usize>, Vec<u8>)], outs: &[(DType, Vec<usize>, Vec<u8>)], attrs: &OpAttrs, tail: bool) -> Vec<Vec<u8>> {
    intarsia_backend_metal::inkctrl::set_blend_tail(Some(tail));
    let it: Vec<TensorRef> = ins.iter().map(|(dt, sh, b)| dev(be, *dt, sh, b, 0)).collect();
    let ot: Vec<TensorRef> = outs.iter().map(|(dt, sh, b)| dev(be, *dt, sh, b, 0)).collect();
    let mut call = OpCall::new(op, it, ot.clone(), attrs.clone());
    call.exact_rows = ins[0].1.len() >= 2 && ins[0].1[0] > 1;
    call.check().unwrap();
    call.scratch = Some(blend_scratch_for(be, &call));
    be.launch(kernel, &call, 0).unwrap();
    be.synchronize(0).unwrap();
    intarsia_backend_metal::inkctrl::set_blend_tail(None);
    ot.iter().map(|t| down(be, t)).collect()
}

#[test]
fn blend_tail_is_the_three_dispatches_bitwise() {
    let Some(be) = be() else { return };
    let _g = BLEND_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let mut rng = Rng(0x7a11);
    let bf = DType::Bf16;
    let f = DType::F32;
    let mut n = 0;
    // (D, G): the model's, a D that is not a multiple of the 256-channel items, the gate's other register widths
    for &(d, g) in &[(2304usize, 64usize), (200, 64), (256, 8), (320, 128)] {
        for &r in &[1usize, 2, 3, 4, 7, 16] {
            for &(s, sd) in &[(3usize, 1usize), (3, 3), (1, 1)] {
                // LoopBlend: dest, msg, down, gain, forget, norm -> out
                let ins = vec![
                    (f, vec![sd, r, d], f32_bytes(sd * r * d, 1.0, &mut rng)),
                    (f, vec![s, r, d], f32_bytes(s * r * d, 1.0, &mut rng)),
                    (bf, vec![s, g, 2 * d], bf16_bytes(s * g * 2 * d, (2.0 * d as f32).powf(-0.5), &mut rng)),
                    (bf, vec![s, d, g], bf16_bytes(s * d * g, 0.25, &mut rng)),
                    (bf, vec![s, d, g], bf16_bytes(s * d * g, 0.25, &mut rng)),
                    (bf, vec![s, d], bf_pos(s * d, &mut rng)),
                ];
                let outs = vec![(bf, vec![s, r, d], vec![0u8; s * r * d * 2])];
                let a = OpAttrs { eps: 1e-6, ..OpAttrs::default() };
                let off = run_blend(&be, "loop_blend_bf16", Op::LoopBlend, &ins, &outs, &a, false);
                let on = run_blend(&be, "loop_blend_bf16", Op::LoopBlend, &ins, &outs, &a, true);
                assert_eq!(on, off, "LoopBlend D={d} G={g} R={r} S={s} Sd={sd}");
                n += 1;
            }
            // ConvMix with the blend: x, taps, res, add, src, queries (Q 1), norm [S, D], msg, down, gain, forget ->
            // y, ring, sum, mix [S, T, D]
            for &(s, nsrc) in &[(3usize, 3usize), (1, 2)] {
                const K: usize = 4;
                let rr = K - 1 + 16;
                let ins = vec![
                    (bf, vec![r, d], bf16_bytes(r * d, 1.0, &mut rng)),
                    (bf, vec![K, d], bf16_bytes(K * d, 0.5, &mut rng)),
                    (bf, vec![r, d], bf16_bytes(r * d, 1.0, &mut rng)),
                    (bf, vec![r, d], bf16_bytes(r * d, 1.0, &mut rng)),
                    (bf, vec![nsrc, r, d], bf16_bytes(nsrc * r * d, 1.5, &mut rng)),
                    (f, vec![1, d], f32_bytes(d, 0.2, &mut rng)),
                    (bf, vec![s, d], bf_pos(s * d, &mut rng)),
                    (f, vec![s, r, d], f32_bytes(s * r * d, 1.0, &mut rng)),
                    (bf, vec![s, g, 2 * d], bf16_bytes(s * g * 2 * d, (2.0 * d as f32).powf(-0.5), &mut rng)),
                    (bf, vec![s, d, g], bf16_bytes(s * d * g, 0.25, &mut rng)),
                    (bf, vec![s, d, g], bf16_bytes(s * d * g, 0.25, &mut rng)),
                ];
                let outs = vec![
                    (bf, vec![r, d], vec![0u8; r * d * 2]),
                    (bf, vec![rr, d], bf16_bytes(rr * d, 1.0, &mut rng)),
                    (bf, vec![r, d], vec![0u8; r * d * 2]),
                    (bf, vec![s, r, d], vec![0u8; s * r * d * 2]),
                ];
                let a = OpAttrs { pos0: 5, eps: 1e-6, ..OpAttrs::default() };
                let off = run_blend(&be, "conv_mix_bf16", Op::ConvMix, &ins, &outs, &a, false);
                let on = run_blend(&be, "conv_mix_bf16", Op::ConvMix, &ins, &outs, &a, true);
                assert_eq!(on, off, "ConvMix blend D={d} G={g} R={r} S={s} N={nsrc}");
                n += 1;
            }
        }
    }
    eprintln!("{n} blend cases: the fused tail is bitwise the three dispatches");
}

/// Many fused-tail LoopBlend calls back to back in one stream (no wait between them) over several shapes and row counts, each
/// checked against the three-dispatch result: a ticket or counter left non-zero would misassign roles or release a gate item
/// early (wrong bits), a lost wake-up hangs (the watchdog). `INK_CTRL_STRESS` as above.
#[test]
fn blend_tail_stress_no_hang_no_stale_counter() {
    let Some(be) = be() else { return };
    let _g = BLEND_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let calls: usize = std::env::var("INK_CTRL_STRESS").ok().and_then(|v| v.parse().ok()).unwrap_or(2000);
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        let be = be;
        let mut rng = Rng(0xb1e7d);
        let (bf, f) = (DType::Bf16, DType::F32);
        let mut cases = Vec::new();
        for &(d, g) in &[(2304usize, 64usize), (200, 64), (320, 128)] {
            for &r in &[1usize, 2, 3, 4, 8, 16] {
                for &s in &[1usize, 3] {
                    let ins = vec![
                        (f, vec![1, r, d], f32_bytes(r * d, 1.0, &mut rng)),
                        (f, vec![s, r, d], f32_bytes(s * r * d, 1.0, &mut rng)),
                        (bf, vec![s, g, 2 * d], bf16_bytes(s * g * 2 * d, (2.0 * d as f32).powf(-0.5), &mut rng)),
                        (bf, vec![s, d, g], bf16_bytes(s * d * g, 0.25, &mut rng)),
                        (bf, vec![s, d, g], bf16_bytes(s * d * g, 0.25, &mut rng)),
                        (bf, vec![s, d], bf_pos(s * d, &mut rng)),
                    ];
                    let outs = vec![(bf, vec![s, r, d], vec![0u8; s * r * d * 2])];
                    let a = OpAttrs { eps: 1e-6, ..OpAttrs::default() };
                    let want = run_blend(&be, "loop_blend_bf16", Op::LoopBlend, &ins, &outs, &a, false);
                    let it: Vec<TensorRef> = ins.iter().map(|(dt, sh, b)| dev(&be, *dt, sh, b, 0)).collect();
                    let ot: Vec<TensorRef> = outs.iter().map(|(dt, sh, b)| dev(&be, *dt, sh, b, 0)).collect();
                    let probe = OpCall::new(Op::LoopBlend, it.clone(), ot, a.clone());
                    let sc = blend_scratch_for(&be, &probe);
                    cases.push((it, outs, a, want, sc));
                }
            }
        }
        intarsia_backend_metal::inkctrl::set_blend_tail(Some(true));
        let (mut done, mut bad) = (0, 0);
        let mut worst = Duration::ZERO;
        while done < calls {
            let batch = 200.min(calls - done);
            let t0 = std::time::Instant::now();
            let mut pend = Vec::with_capacity(batch);
            for j in 0..batch {
                let ci = (done + j) * 5 % cases.len();
                let (it, outs, a, _, sc) = &cases[ci];
                let ot: Vec<TensorRef> = outs.iter().map(|(dt, sh, b)| dev(&be, *dt, sh, b, 0)).collect();
                let mut call = OpCall::new(Op::LoopBlend, it.clone(), ot.clone(), a.clone());
                call.exact_rows = it[0].shape[1] > 1;
                call.scratch = Some(sc.clone());
                be.launch("loop_blend_bf16", &call, 0).unwrap();
                pend.push((ci, ot));
            }
            be.synchronize(0).unwrap();
            worst = worst.max(t0.elapsed());
            for (ci, ot) in pend {
                if down(&be, &ot[0]) != cases[ci].3[0] {
                    bad += 1;
                }
            }
            done += batch;
        }
        intarsia_backend_metal::inkctrl::set_blend_tail(None);
        let _ = tx.send((done, bad, worst));
    });
    let budget = Duration::from_secs(60 + calls as u64 / 20);
    match rx.recv_timeout(budget) {
        Ok((done, bad, worst)) => {
            assert_eq!(bad, 0, "{bad} of {done} blend-tail stress calls differ from the three-dispatch result");
            eprintln!("STRESS blend_tail: {done} calls, all bitwise; slowest batch of 200 (encode + GPU + wait) {:.2} ms (watchdog {budget:?})", worst.as_secs_f64() * 1e3);
        }
        Err(_) => panic!("blend tail stress: no result within {budget:?} (GPU hang or stall)"),
    }
}
