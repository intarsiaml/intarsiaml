//! The fused two-rank all-reduce + residual add + rmsnorm (`kernels/hip/ar_norm.hip`, `Comm::all_reduce_add_rmsnorm`) on real
//! GPUs: bit for bit the unfused pair (`all_reduce_sum_into`, then the `rmsnorm_add_bf16` kernel on its result), on wide
//! dynamic range data where every rounding point matters, for f32 and bf16 partials, the hidden sizes of the 35B (2048), the
//! 8B (4096) and the 27B (5120), 1..=16 rows and more, rank-asymmetric inputs, in place and not, and mixed with plain
//! collectives eagerly and inside a replayed graph (the per-block counters must stay in lockstep). Needs two HIP devices
//! that can reach each other (skips otherwise); `INTARSIA_HIP_DEVICES=0,1` picks them.
//!
//! The comparison is against the unfused pair, and also against host arithmetic for the two roundings that precede the
//! norm (so a rounding both kernels shared could not hide). Mutants of the kernel are in `scripts/tp2-hip/stages-arnorm.sh`.

#![cfg(ik_hip)]

use intarsia_core::comm::ArNorm;
use intarsia_core::dtype::{decode_to_f32, encode_from_f32};
use intarsia_core::rng::SplitMix64;
use intarsia_core::{Backend, Comm, DType, MemKind, Op, OpAttrs, OpCall, TensorRef};
use std::sync::{Arc, Mutex, MutexGuard};

/// One group at a time (see tests/comm.rs: two groups interleaving on GPU_MAX_HW_QUEUES=1 queues can trap).
static SERIAL: Mutex<()> = Mutex::new(());

fn serial() -> MutexGuard<'static, ()> {
    SERIAL.lock().unwrap_or_else(|e| e.into_inner())
}

fn group() -> Option<(Vec<Arc<dyn Backend>>, Vec<Arc<dyn Comm>>)> {
    let ords: Vec<usize> = std::env::var("INTARSIA_HIP_DEVICES")
        .ok()
        .map(|v| v.split(',').filter_map(|x| x.trim().parse().ok()).collect())
        .unwrap_or_else(|| vec![0, 1]);
    match intarsia_backend_hip::create_group(&ords) {
        Ok(g) => Some(g),
        Err(e) => {
            eprintln!("skipping: no usable 2-GPU HIP group ({e})");
            None
        }
    }
}

fn put(be: &dyn Backend, dt: DType, shape: &[usize], vals: &[f32]) -> TensorRef {
    let mut bytes = vec![0u8; vals.len() * dt.size_bytes()];
    encode_from_f32(dt, vals, &mut bytes);
    let buf = be.alloc(bytes.len().max(16), MemKind::Device).unwrap();
    be.upload(&buf, 0, &bytes, 0).unwrap();
    TensorRef::new(buf, dt, shape)
}

fn raw(be: &dyn Backend, t: &TensorRef) -> Vec<u8> {
    be.synchronize(0).unwrap();
    let mut bytes = vec![0u8; t.nbytes()];
    be.download(&t.buf, t.offset, &mut bytes, 0).unwrap();
    bytes
}

fn vals(be: &dyn Backend, t: &TensorRef) -> Vec<f32> {
    let mut out = vec![0f32; t.numel()];
    decode_to_f32(t.dtype, &raw(be, t), &mut out);
    out
}

/// `a + b` in f32 rounded once to bf16, as the kernels do (and as the model's partials are reduced).
fn host_round_sum(a: &[f32], b: &[f32]) -> Vec<f32> {
    let s: Vec<f32> = a.iter().zip(b).map(|(x, y)| x + y).collect();
    let mut bytes = vec![0u8; s.len() * 2];
    encode_from_f32(DType::Bf16, &s, &mut bytes);
    let mut out = vec![0f32; s.len()];
    decode_to_f32(DType::Bf16, &bytes, &mut out);
    out
}

/// Values over a wide dynamic range: a normal draw times 2^e with e uniform in -6..=6, so the f32 sums, the bf16 roundings
/// and the sum of squares all lose different bits (a unit-variance draw hides a missing or doubled rounding).
fn wide(rng: &mut SplitMix64, n: usize, sigma: f32) -> Vec<f32> {
    rng.normal_vec(n, sigma).into_iter().map(|x| x * 2f32.powi((rng.next_u64() % 13) as i32 - 6)).collect()
}

fn call_unfused(be: &dyn Backend, x: &TensorRef, res: &TensorRef, w: &TensorRef, nrm: &TensorRef, sum: &TensorRef, eps: f32, woff: f32) {
    let call = OpCall::new(
        Op::RmsNormAdd,
        vec![x.clone(), res.clone(), w.clone()],
        vec![nrm.clone(), sum.clone()],
        OpAttrs { eps, w_offset: woff, ..Default::default() },
    );
    be.launch("rmsnorm_add_bf16", &call, 0).unwrap();
}

fn ar_norm<'a>(p: &'a TensorRef, out: &'a TensorRef, res: &'a TensorRef, w: &'a TensorRef, nrm: &'a TensorRef, sum: &'a TensorRef, eps: f32, woff: f32) -> ArNorm<'a> {
    ArNorm { partial: p, out, residual: res, weight: w, norm: nrm, sum, eps, w_offset: woff }
}

/// One trial: random operands, the fused collective on both ranks, then the unfused pair, bit-compared on both ranks.
#[allow(clippy::too_many_arguments)]
fn trial(be: &[Arc<dyn Backend>], comms: &[Arc<dyn Comm>], rng: &mut SplitMix64, dt: DType, rows: usize, h: usize, woff: f32, in_place: bool, mode: usize) {
    let n = rows * h;
    let sh = [rows, h];
    // rank-asymmetric partials: mode 0 same scale, 1 rank 0 large and rank 1 tiny, 2 rank 1 nearly cancels rank 0
    let a = wide(rng, n, 4.0);
    let b = match mode {
        0 => wide(rng, n, 4.0),
        1 => wide(rng, n, 0.01),
        _ => a.iter().map(|x| -x * (1.0 + (rng.next_f32() - 0.5) * 1e-3)).collect(),
    };
    let res = wide(rng, n, 2.0);
    let w = rng.normal_vec(h, 1.0);
    let zero_b = vec![0f32; n];
    let parts = [a, b];
    let mut p = Vec::new();
    let mut out_f = Vec::new();
    let mut nrm_f = Vec::new();
    let mut sum_f = Vec::new();
    let mut resv = Vec::new();
    let mut wv = Vec::new();
    let mut out_u = Vec::new();
    let mut nrm_u = Vec::new();
    let mut sum_u = Vec::new();
    let mut p_u = Vec::new();
    for r in 0..2 {
        let d = &*be[r];
        let pp = put(d, dt, &sh, &parts[r]);
        let of = if in_place { pp.clone() } else { put(d, DType::Bf16, &sh, &zero_b) };
        let (nf, sf) = (put(d, DType::Bf16, &sh, &zero_b), put(d, DType::Bf16, &sh, &zero_b));
        // the unfused reference reduces a copy of the same partial (in place: it overwrites its input)
        let pu = put(&*be[r], dt, &sh, &parts[r]);
        p.push(pp);
        out_f.push(of);
        nrm_f.push(nf);
        sum_f.push(sf);
        resv.push(put(&*be[r], DType::Bf16, &sh, &res));
        wv.push(put(&*be[r], DType::Bf16, &[h], &w));
        out_u.push(if in_place { pu.clone() } else { put(&*be[r], DType::Bf16, &sh, &zero_b) });
        p_u.push(pu);
        nrm_u.push(put(&*be[r], DType::Bf16, &sh, &zero_b));
        sum_u.push(put(&*be[r], DType::Bf16, &sh, &zero_b));
    }
    let eps = 1e-6f32;
    // what the partials are once held in `dt` (the f32 ones exactly, the bf16 ones rounded)
    let held: Vec<Vec<f32>> = (0..2).map(|r| vals(&*be[r], &p[r])).collect();
    for r in 0..2 {
        let a = ar_norm(&p[r], &out_f[r], &resv[r], &wv[r], &nrm_f[r], &sum_f[r], eps, woff);
        assert!(comms[r].ar_norm_supported(&a), "rank {r}: {dt:?} {rows}x{h} not supported");
    }
    std::thread::scope(|sc| {
        for r in 0..2 {
            let (c, d) = (&comms[r], &be[r]);
            let (p, out_f, res, w, nrm_f, sum_f) = (&p[r], &out_f[r], &resv[r], &wv[r], &nrm_f[r], &sum_f[r]);
            let (p_u, out_u, nrm_u, sum_u) = (&p_u[r], &out_u[r], &nrm_u[r], &sum_u[r]);
            sc.spawn(move || {
                c.all_reduce_add_rmsnorm(&ar_norm(p, out_f, res, w, nrm_f, sum_f, eps, woff), 0).unwrap();
                c.all_reduce_sum_into(p_u, out_u, 0).unwrap();
                call_unfused(&**d, out_u, res, w, nrm_u, sum_u, eps, woff);
            });
        }
    });
    // host arithmetic for the first two roundings
    let r1 = host_round_sum(&held[0], &held[1]);
    let s = host_round_sum(&r1, &vals(&*be[0], &resv[0]));
    let tag = format!("{dt:?} rows {rows} h {h} woff {woff} in_place {in_place} mode {mode}");
    for r in 0..2 {
        let d = &*be[r];
        assert_eq!(vals(d, &out_f[r]), r1, "{tag} rank {r}: all-reduce result vs host");
        assert_eq!(vals(d, &sum_f[r]), s, "{tag} rank {r}: residual stream vs host");
        assert_eq!(raw(d, &out_f[r]), raw(d, &out_u[r]), "{tag} rank {r}: out vs unfused");
        assert_eq!(raw(d, &sum_f[r]), raw(d, &sum_u[r]), "{tag} rank {r}: sum vs unfused");
        assert_eq!(raw(d, &nrm_f[r]), raw(d, &nrm_u[r]), "{tag} rank {r}: norm vs unfused");
        if !in_place {
            // the partials are untouched
            assert_eq!(vals(d, &p[r]), held[r], "{tag} rank {r}: partial changed");
        }
    }
    assert_eq!(raw(&*be[0], &nrm_f[0]), raw(&*be[1], &nrm_f[1]), "{tag}: ranks disagree on the norm");
    assert_eq!(raw(&*be[0], &sum_f[0]), raw(&*be[1], &sum_f[1]), "{tag}: ranks disagree on the residual");
    // not a vacuous comparison: the norm output is not constant and not all zero
    let nv = vals(&*be[0], &nrm_f[0]);
    assert!(nv.iter().any(|x| *x != 0.0) && nv.iter().any(|x| *x != nv[0]), "{tag}: degenerate output");
}

#[test]
fn fused_is_bitwise_the_unfused_pair_on_wide_range_data() {
    let _one = serial();
    let Some((be, comms)) = group() else { return };
    assert!(comms.iter().all(|c| c.fuses_ar_norm()));
    let mut rng = SplitMix64::new(101);
    let mut trials = 0;
    // the 35B (2048), the 8B (4096), the 27B (5120), and small / odd item counts (8: one item, 40: five items, bd 32)
    for h in [2048usize, 4096, 5120, 8, 40] {
        for dt in [DType::F32, DType::Bf16] {
            // decode rows 1..=16 (one row per block at most: what the comm accepts)
            for rows in 1..=16usize {
                for (woff, mode) in [(0.0f32, 0usize), (1.0, 1), (0.0, 2)] {
                    trial(&be, &comms, &mut rng, dt, rows, h, woff, false, mode);
                    trials += 1;
                }
            }
        }
    }
    eprintln!("{trials} trials");
}

#[test]
fn bf16_partials_reduced_in_place_like_the_moe_block_output() {
    let _one = serial();
    let Some((be, comms)) = group() else { return };
    let mut rng = SplitMix64::new(102);
    for h in [2048usize, 4096, 5120] {
        for rows in [1usize, 2, 3, 7, 16] {
            for (woff, mode) in [(0.0f32, 0usize), (1.0, 2)] {
                trial(&be, &comms, &mut rng, DType::Bf16, rows, h, woff, true, mode);
            }
        }
    }
}

/// 16 rows (one per block) is the limit, 17 is refused (the model then runs the pair), and so are the operands the kernel
/// cannot take.
#[test]
fn sixteen_rows_is_the_limit_and_unsupported_operands_are_refused() {
    let _one = serial();
    let Some((be, comms)) = group() else { return };
    let mut rng = SplitMix64::new(103);
    trial(&be, &comms, &mut rng, DType::F32, 16, 5120, 0.0, false, 0);
    let d = &*be[0];
    let mk = |dt: DType, shape: &[usize]| put(d, dt, shape, &vec![0.5; shape.iter().product()]);
    let set = |rows: usize, h: usize, pdt: DType| {
        (mk(pdt, &[rows, h]), mk(DType::Bf16, &[rows, h]), mk(DType::Bf16, &[rows, h]), mk(DType::Bf16, &[h]), mk(DType::Bf16, &[rows, h]), mk(DType::Bf16, &[rows, h]))
    };
    let (p, o, r, w, n, s) = set(17, 2048, DType::F32);
    assert!(!comms[0].ar_norm_supported(&ar_norm(&p, &o, &r, &w, &n, &s, 1e-6, 0.0)), "17 rows");
    let (p, o, r, w, n, s) = set(16, 2048, DType::F32);
    assert!(comms[0].ar_norm_supported(&ar_norm(&p, &o, &r, &w, &n, &s, 1e-6, 0.0)), "16 rows");
    // hidden size not a multiple of 8, an f16 output, an offset that is not 16-byte aligned
    let (p, o, r, w, n, s) = set(2, 36, DType::F32);
    assert!(!comms[0].ar_norm_supported(&ar_norm(&p, &o, &r, &w, &n, &s, 1e-6, 0.0)), "h 36");
    let (p, o, r, w, n, s) = set(2, 64, DType::F32);
    assert!(comms[0].ar_norm_supported(&ar_norm(&p, &o, &r, &w, &n, &s, 1e-6, 0.0)));
    let o16 = mk(DType::F16, &[2, 64]);
    assert!(!comms[0].ar_norm_supported(&ar_norm(&p, &o16, &r, &w, &n, &s, 1e-6, 0.0)), "f16 out");
    let mut shifted = w.clone();
    shifted.offset = 2;
    assert!(!comms[0].ar_norm_supported(&ar_norm(&p, &o, &r, &shifted, &n, &s, 1e-6, 0.0)), "unaligned weight");
}

/// Plain and fused collectives alternate with no host synchronisation (the ranks run ahead of each other), eagerly and as
/// one replayed graph: every launch advances every block's counter once, so the scratch parity and the flag values stay
/// in step whichever kernel a launch is. Each round's results are checked against the unfused pair.
#[test]
fn fused_and_plain_collectives_interleave_eagerly_and_in_a_replayed_graph() {
    let _one = serial();
    let Some((be, comms)) = group() else { return };
    let mut rng = SplitMix64::new(104);
    let (rows, h) = (1usize, 2048usize); // the 35B's decode row
    let n = rows * h;
    let sh = [rows, h];
    let z = vec![0f32; n];
    struct B {
        pf: TensorRef, // f32 partial for the fused call
        of: TensorRef,
        nf: TensorRef,
        sf: TensorRef,
        pg: TensorRef, // bf16 partial for the second fused call (in place)
        ng: TensorRef,
        sg: TensorRef,
        pb: TensorRef, // bf16 in place plain all-reduce
        res: TensorRef,
        w: TensorRef,
    }
    let mk = |r: usize| {
        let d = &*be[r];
        B {
            pf: put(d, DType::F32, &sh, &z),
            of: put(d, DType::Bf16, &sh, &z),
            nf: put(d, DType::Bf16, &sh, &z),
            sf: put(d, DType::Bf16, &sh, &z),
            pg: put(d, DType::Bf16, &sh, &z),
            ng: put(d, DType::Bf16, &sh, &z),
            sg: put(d, DType::Bf16, &sh, &z),
            pb: put(d, DType::Bf16, &sh, &z),
            res: put(d, DType::Bf16, &sh, &z),
            w: put(d, DType::Bf16, &[h], &z),
        }
    };
    let bufs = [mk(0), mk(1)];
    let eps = 1e-6f32;
    let body = |r: usize| -> intarsia_core::Result<()> {
        let b = &bufs[r];
        let c = &comms[r];
        c.all_reduce_add_rmsnorm(&ar_norm(&b.pf, &b.of, &b.res, &b.w, &b.nf, &b.sf, eps, 0.0), 0)?;
        c.all_reduce_add_rmsnorm(&ar_norm(&b.pg, &b.pg, &b.res, &b.w, &b.ng, &b.sg, eps, 1.0), 0)?;
        c.all_reduce_sum(&b.pb, 0)
    };
    // three launches per graph: each replay starts on the other scratch parity than the one before
    let graphs: Vec<_> = (0..2).map(|r| be[r].graph_capture(0, &mut || body(r)).unwrap()).collect();
    let upload = |r: usize, t: &TensorRef, v: &[f32]| {
        let mut bytes = vec![0u8; v.len() * t.dtype.size_bytes()];
        encode_from_f32(t.dtype, v, &mut bytes);
        be[r].upload(&t.buf, t.offset, &bytes, 0).unwrap();
    };
    let w = rng.normal_vec(h, 1.0);
    for round in 0..120 {
        let res = wide(&mut rng, n, 2.0);
        for r in 0..2 {
            let b = &bufs[r];
            for t in [&b.pf, &b.pg, &b.pb] {
                upload(r, t, &wide(&mut rng, n, 3.0));
            }
            upload(r, &b.res, &res);
            upload(r, &b.w, &w);
        }
        // references first, on scratch tensors of their own, through the plain collective and the norm kernel
        let held: Vec<Vec<Vec<f32>>> = (0..2).map(|r| [&bufs[r].pf, &bufs[r].pg, &bufs[r].pb].iter().map(|t| vals(&*be[r], t)).collect()).collect();
        let want_of = host_round_sum(&held[0][0], &held[1][0]);
        let want_og = host_round_sum(&held[0][1], &held[1][1]);
        let want_ob = host_round_sum(&held[0][2], &held[1][2]);
        let res_h = vals(&*be[0], &bufs[0].res);
        let want_sf = host_round_sum(&want_of, &res_h);
        let want_sg = host_round_sum(&want_og, &res_h);
        let ref_n: Vec<(TensorRef, TensorRef)> = (0..2).map(|r| (put(&*be[r], DType::Bf16, &sh, &z), put(&*be[r], DType::Bf16, &sh, &z))).collect();
        let ref_g: Vec<(TensorRef, TensorRef)> = (0..2).map(|r| (put(&*be[r], DType::Bf16, &sh, &z), put(&*be[r], DType::Bf16, &sh, &z))).collect();
        let (of_t, og_t): (Vec<TensorRef>, Vec<TensorRef>) = (0..2).map(|r| (put(&*be[r], DType::Bf16, &sh, &want_of), put(&*be[r], DType::Bf16, &sh, &want_og))).unzip();
        for r in 0..2 {
            call_unfused(&*be[r], &of_t[r], &bufs[r].res, &bufs[r].w, &ref_n[r].0, &ref_n[r].1, eps, 0.0);
            call_unfused(&*be[r], &og_t[r], &bufs[r].res, &bufs[r].w, &ref_g[r].0, &ref_g[r].1, eps, 1.0);
        }
        let eager = round % 4 == 3;
        std::thread::scope(|sc| {
            for r in 0..2 {
                let (b, g) = (&be[r], &graphs[r]);
                let body = &body;
                sc.spawn(move || if eager { body(r).unwrap() } else { b.graph_replay(g, 0).unwrap() });
            }
        });
        for r in 0..2 {
            let (d, b) = (&*be[r], &bufs[r]);
            let tag = format!("round {round} ({}) rank {r}", if eager { "eager" } else { "replay" });
            assert_eq!(vals(d, &b.of), want_of, "{tag}: fused out");
            assert_eq!(vals(d, &b.sf), want_sf, "{tag}: fused sum");
            assert_eq!(raw(d, &b.nf), raw(d, &ref_n[r].0), "{tag}: fused norm vs unfused");
            assert_eq!(vals(d, &b.pg), want_og, "{tag}: fused in place out");
            assert_eq!(vals(d, &b.sg), want_sg, "{tag}: fused in place sum");
            assert_eq!(raw(d, &b.ng), raw(d, &ref_g[r].0), "{tag}: fused in place norm vs unfused");
            assert_eq!(vals(d, &b.pb), want_ob, "{tag}: plain in place");
        }
    }
}
