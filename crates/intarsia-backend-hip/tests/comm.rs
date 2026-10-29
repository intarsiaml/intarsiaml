//! The two-rank peer-to-peer all-reduce on real GPUs. Needs two HIP devices that can reach each other (skips
//! otherwise); `INTARSIA_HIP_DEVICES=0,1` picks them. The rate test is `--ignored --nocapture`.

#![cfg(ik_hip)]

use intarsia_core::dtype::{decode_to_f32, encode_from_f32};
use intarsia_core::rng::SplitMix64;
use intarsia_core::{Backend, Comm, DType, MemKind, TensorRef};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Instant;

/// One group at a time. The tests run in parallel threads of one process, each with its own two-rank group, and with
/// GPU_MAX_HW_QUEUES=1 (deploy/tatooine/dev.sh) every stream of a device feeds one hardware queue: group A's rank-0
/// collective can then spin on GPU 0 while GPU 1's queue holds group B's collective ahead of A's, which waits for B's
/// rank 0, queued behind A on GPU 0. The bounded wait ends that in a trap (lane H1, window 3). The engine runs one
/// group per process, whose ranks issue the same collectives in the same order, so it cannot interleave like this.
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

fn put(be: &dyn Backend, dt: DType, vals: &[f32]) -> TensorRef {
    let mut bytes = vec![0u8; vals.len() * dt.size_bytes()];
    encode_from_f32(dt, vals, &mut bytes);
    let buf = be.alloc(bytes.len().max(16), MemKind::Device).unwrap();
    be.upload(&buf, 0, &bytes, 0).unwrap();
    TensorRef::new(buf, dt, &[vals.len()])
}

fn get(be: &dyn Backend, t: &TensorRef) -> Vec<f32> {
    be.synchronize(0).unwrap();
    let mut bytes = vec![0u8; t.nbytes()];
    be.download(&t.buf, t.offset, &mut bytes, 0).unwrap();
    let mut out = vec![0f32; t.numel()];
    decode_to_f32(t.dtype, &bytes, &mut out);
    out
}

/// Round `a + b` the way the kernel does (f32 add, one round to nearest even into `dt`).
fn host_sum(dt: DType, a: &[f32], b: &[f32]) -> Vec<f32> {
    let s: Vec<f32> = a.iter().zip(b).map(|(x, y)| x + y).collect();
    let mut bytes = vec![0u8; s.len() * dt.size_bytes()];
    encode_from_f32(dt, &s, &mut bytes);
    let mut out = vec![0f32; s.len()];
    decode_to_f32(dt, &bytes, &mut out);
    out
}

fn run_pair(comms: &[Arc<dyn Comm>], x: &[TensorRef]) {
    std::thread::scope(|sc| {
        for r in 0..2 {
            let (c, t) = (&comms[r], &x[r]);
            sc.spawn(move || c.all_reduce_sum(t, 0).unwrap());
        }
    });
}

#[test]
fn all_reduce_is_exact_and_identical_on_both_ranks() {
    let _one = serial();
    let Some((be, comms)) = group() else { return };
    assert_eq!(comms[0].name(), "hip-p2p");
    let mut rng = SplitMix64::new(7);
    // decode-sized, odd-block-count, prefill-sized, and larger than one scratch slot (split into several launches)
    for (dt, n) in [(DType::Bf16, 4096), (DType::Bf16, 8), (DType::Bf16, 3 * 4096), (DType::Bf16, 256 * 4096), (DType::F32, 4096), (DType::Bf16, 3 * 1024 * 1024)] {
        for _ in 0..3 {
            let a: Vec<f32> = rng.normal_vec(n, 3.0);
            let b: Vec<f32> = rng.normal_vec(n, 3.0);
            let (ta, tb) = (put(&*be[0], dt, &a), put(&*be[1], dt, &b));
            // what each rank holds is the rounded input
            let (ra, rb) = (get(&*be[0], &ta), get(&*be[1], &tb));
            run_pair(&comms, &[ta.clone(), tb.clone()]);
            let want = host_sum(dt, &ra, &rb);
            let (ga, gb) = (get(&*be[0], &ta), get(&*be[1], &tb));
            assert_eq!(ga, want, "rank 0, {dt:?} n={n}");
            assert_eq!(gb, want, "rank 1, {dt:?} n={n}");
        }
    }
}

#[test]
fn f32_partials_reduce_into_bf16_with_one_rounding() {
    let _one = serial();
    let Some((be, comms)) = group() else { return };
    let mut rng = SplitMix64::new(21);
    for n in [4096usize, 3 * 4096, 256 * 4096, 2 * 1024 * 1024] {
        let a: Vec<f32> = rng.normal_vec(n, 5.0);
        let b: Vec<f32> = rng.normal_vec(n, 5.0);
        let (pa, pb) = (put(&*be[0], DType::F32, &a), put(&*be[1], DType::F32, &b));
        let (oa, ob) = (put(&*be[0], DType::Bf16, &vec![0.0; n]), put(&*be[1], DType::Bf16, &vec![0.0; n]));
        std::thread::scope(|sc| {
            for (r, (p, o)) in [(&pa, &oa), (&pb, &ob)].into_iter().enumerate() {
                let c = &comms[r];
                sc.spawn(move || c.all_reduce_sum_into(p, o, 0).unwrap());
            }
        });
        let want = host_sum(DType::Bf16, &a, &b);
        assert_eq!(get(&*be[0], &oa), want, "rank 0 n={n}");
        assert_eq!(get(&*be[1], &ob), want, "rank 1 n={n}");
        // the partials are untouched
        assert_eq!(get(&*be[0], &pa), a);
        assert_eq!(get(&*be[1], &pb), b);
    }
}

#[test]
fn many_back_to_back_launches_keep_both_scratch_parities_straight() {
    let _one = serial();
    let Some((be, comms)) = group() else { return };
    let mut rng = SplitMix64::new(11);
    let (k, n) = (64usize, 4096usize);
    for round in 0..150 {
        let mut inputs = Vec::new();
        let mut tens: [Vec<TensorRef>; 2] = [Vec::new(), Vec::new()];
        for i in 0..k {
            let a: Vec<f32> = rng.normal_vec(n, 2.0);
            let b: Vec<f32> = rng.normal_vec(n, 2.0);
            tens[0].push(put(&*be[0], DType::Bf16, &a));
            tens[1].push(put(&*be[1], DType::Bf16, &b));
            inputs.push((get(&*be[0], &tens[0][i]), get(&*be[1], &tens[1][i])));
        }
        // no synchronisation between the launches: the ranks run ahead of each other
        std::thread::scope(|sc| {
            for r in 0..2 {
                let (c, ts) = (&comms[r], &tens[r]);
                sc.spawn(move || {
                    for t in ts {
                        c.all_reduce_sum(t, 0).unwrap();
                    }
                });
            }
        });
        for i in 0..k {
            let want = host_sum(DType::Bf16, &inputs[i].0, &inputs[i].1);
            assert_eq!(get(&*be[0], &tens[0][i]), want, "round {round} buffer {i} rank 0");
            assert_eq!(get(&*be[1], &tens[1][i]), want, "round {round} buffer {i} rank 1");
        }
    }
}

/// Lane H1: the collectives recorded into a hipGraph on each rank (what the tp=2 decode graph does) give the exact sum
/// at every replay, with new data in the same buffers each time, while the ranks run unsynchronised and eager
/// collectives (prefill) interleave with the replays. Three collectives per graph, so each replay starts on the other
/// scratch parity than the one before: the parity comes from the device counter, not from the recording.
#[test]
fn collectives_captured_into_a_graph_replay_exactly_and_interleave_with_eager_ones() {
    let _one = serial();
    let Some((be, comms)) = group() else { return };
    assert!(comms.iter().all(|c| c.capture_safe()), "hip-p2p says capture_safe");
    let mut rng = SplitMix64::new(31);
    let n = 2048usize; // the 35B's hidden size: one decode row
    let zero = vec![0f32; n];
    // per rank: [f32 partial -> bf16 out] x 2 and one bf16 in-place, fixed buffers (a graph keeps its addresses)
    let mk = |r: usize| {
        (
            [put(&*be[r], DType::F32, &zero), put(&*be[r], DType::F32, &zero)],
            [put(&*be[r], DType::Bf16, &zero), put(&*be[r], DType::Bf16, &zero)],
            put(&*be[r], DType::Bf16, &zero),
        )
    };
    let bufs = [mk(0), mk(1)];
    let body = |r: usize| {
        let (p, o, x) = &bufs[r];
        comms[r].all_reduce_sum_into(&p[0], &o[0], 0)?;
        comms[r].all_reduce_sum_into(&p[1], &o[1], 0)?;
        comms[r].all_reduce_sum(x, 0)
    };
    // capture executes nothing, so the ranks need not capture together
    let graphs: Vec<_> = (0..2).map(|r| be[r].graph_capture(0, &mut || body(r)).unwrap()).collect();
    let upload = |r: usize, t: &TensorRef, v: &[f32]| {
        let mut bytes = vec![0u8; v.len() * t.dtype.size_bytes()];
        encode_from_f32(t.dtype, v, &mut bytes);
        be[r].upload(&t.buf, t.offset, &bytes, 0).unwrap();
    };
    for round in 0..200 {
        // new inputs in the recorded buffers, then replay (or, every fifth round, the same calls eagerly)
        let mut want = Vec::new();
        let mut held = [Vec::new(), Vec::new()];
        for r in 0..2 {
            let (p, _, x) = &bufs[r];
            for t in [&p[0], &p[1], x] {
                let v = rng.normal_vec(n, 3.0);
                upload(r, t, &v);
                held[r].push(get(&*be[r], t));
            }
        }
        for i in 0..3 {
            want.push(host_sum(DType::Bf16, &held[0][i], &held[1][i]));
        }
        let eager = round % 5 == 4;
        std::thread::scope(|sc| {
            for r in 0..2 {
                let (b, g) = (&be[r], &graphs[r]);
                let body = &body;
                sc.spawn(move || if eager { body(r).unwrap() } else { b.graph_replay(g, 0).unwrap() });
            }
        });
        for r in 0..2 {
            let (_, o, x) = &bufs[r];
            for (i, t) in [&o[0], &o[1], x].into_iter().enumerate() {
                assert_eq!(get(&*be[r], t), want[i], "round {round} ({}) collective {i} rank {r}", if eager { "eager" } else { "replay" });
            }
        }
    }
}

#[test]
fn an_aborted_group_refuses_collectives_instead_of_waiting() {
    let _one = serial();
    let Some((be, comms)) = group() else { return };
    comms[0].abort();
    let t = put(&*be[0], DType::Bf16, &[1.0; 8]);
    assert!(comms[0].all_reduce_sum(&t, 0).is_err());
}

/// `cargo test --release -p intarsia-backend-hip --test comm -- --ignored --nocapture`
#[test]
#[ignore]
fn all_reduce_rate() {
    let _one = serial();
    let Some((be, comms)) = group() else { return };
    println!("{:>10} {:>12} {:>12} {:>10}", "bytes", "us/call", "GB/s (msg)", "launches");
    for n in [4096usize, 2 * 4096, 8 * 4096, 64 * 4096, 256 * 4096] {
        // zeros: the same buffer is reduced thousands of times, which would double real data out of range
        let z = vec![0f32; n];
        let t = [put(&*be[0], DType::Bf16, &z), put(&*be[1], DType::Bf16, &z)];
        let iters = if n <= 8 * 4096 { 4000 } else { 400 };
        for _ in 0..50 {
            run_pair(&comms, &t);
        }
        be[0].synchronize(0).unwrap();
        be[1].synchronize(0).unwrap();
        let t0 = Instant::now();
        std::thread::scope(|sc| {
            for r in 0..2 {
                let (c, x, b) = (&comms[r], &t[r], &be[r]);
                sc.spawn(move || {
                    for _ in 0..iters {
                        c.all_reduce_sum(x, 0).unwrap();
                    }
                    b.synchronize(0).unwrap();
                });
            }
        });
        let us = t0.elapsed().as_secs_f64() * 1e6 / iters as f64;
        let bytes = n * 2;
        println!("{:>10} {:>12.1} {:>12.2} {:>10}", bytes, us, bytes as f64 / us / 1e3, iters);
    }
}

fn upload_f32(be: &dyn Backend, t: &TensorRef, v: &[f32]) {
    let mut bytes = vec![0u8; v.len() * t.dtype.size_bytes()];
    encode_from_f32(t.dtype, v, &mut bytes);
    be.upload(&t.buf, t.offset, &bytes, 0).unwrap();
}

/// Vocab-parallel head: the all-gather is rank-major (`recv[r * n ..]` is rank r's `send`), bit exact (f32 words are copied,
/// never added), with `send` disjoint from `recv` or this rank's own segment of it (in place), at decode-sized messages (one
/// logits row half of the 35B: 124160 f32), odd sizes, and messages above one scratch slot (several launches).
#[test]
fn all_gather_is_rank_major_exact_in_place_and_disjoint() {
    let _one = serial();
    let Some((be, comms)) = group() else { return };
    let mut rng = SplitMix64::new(41);
    for n in [4usize, 4096, 124_160, 3 * 1024 * 1024 + 4, 5 * 1024 * 1024] {
        for in_place in [false, true] {
            let a: Vec<f32> = rng.normal_vec(n, 3.0);
            let b: Vec<f32> = rng.normal_vec(n, 3.0);
            let vals = [&a, &b];
            let recv: Vec<TensorRef> = (0..2).map(|r| put(&*be[r], DType::F32, &vec![f32::from_bits(0x7fc0_dead); 2 * n]).reshaped(&[2, n])).collect();
            let send: Vec<TensorRef> = (0..2)
                .map(|r| {
                    if in_place {
                        upload_f32(&*be[r], &recv[r].rows(r, 1), vals[r]);
                        recv[r].rows(r, 1)
                    } else {
                        put(&*be[r], DType::F32, vals[r])
                    }
                })
                .collect();
            std::thread::scope(|sc| {
                for r in 0..2 {
                    let (c, s, d) = (&comms[r], &send[r], &recv[r]);
                    sc.spawn(move || c.all_gather(s, d, 0).unwrap());
                }
            });
            let want: Vec<u32> = a.iter().chain(b.iter()).map(|x| x.to_bits()).collect();
            for r in 0..2 {
                let got: Vec<u32> = get(&*be[r], &recv[r]).iter().map(|x| x.to_bits()).collect();
                assert!(got == want, "rank {r} n={n} in_place={in_place}: gathered halves differ");
            }
        }
    }
}

/// The gather recorded into a hipGraph on each rank, between two all-reduces: replays (and eager rounds) with new inputs in the
/// recorded buffers give the exact bits, and the per-block counters stay in lockstep through the mixed sequence.
#[test]
fn all_gather_captured_in_a_graph_replays_exactly_between_all_reduces() {
    let _one = serial();
    let Some((be, comms)) = group() else { return };
    assert!(comms.iter().all(|c| c.capture_safe()));
    let mut rng = SplitMix64::new(43);
    let (nx, ng) = (2048usize, 124_160usize);
    let bufs: Vec<(TensorRef, TensorRef)> = (0..2)
        .map(|r| (put(&*be[r], DType::Bf16, &vec![0f32; nx]), put(&*be[r], DType::F32, &vec![0f32; 2 * ng]).reshaped(&[2, ng])))
        .collect();
    let body = |r: usize| {
        let (x, g) = &bufs[r];
        comms[r].all_reduce_sum(x, 0)?;
        comms[r].all_gather(&g.rows(r, 1), g, 0)?;
        comms[r].all_reduce_sum(x, 0)?;
        comms[r].all_gather(&g.rows(r, 1), g, 0)
    };
    let graphs: Vec<_> = (0..2).map(|r| be[r].graph_capture(0, &mut || body(r)).unwrap()).collect();
    for round in 0..120 {
        let mut held = Vec::new();
        let mut gv: Vec<Vec<f32>> = Vec::new();
        for r in 0..2 {
            let (x, g) = &bufs[r];
            upload_f32(&*be[r], x, &rng.normal_vec(nx, 3.0));
            held.push(get(&*be[r], x));
            let v: Vec<f32> = rng.normal_vec(ng, 3.0);
            upload_f32(&*be[r], &g.rows(r, 1), &v);
            upload_f32(&*be[r], &g.rows(1 - r, 1), &vec![0f32; ng]); // the other segment must be written by the gather
            gv.push(v);
        }
        let once = host_sum(DType::Bf16, &held[0], &held[1]);
        let twice = host_sum(DType::Bf16, &once, &once);
        let want: Vec<u32> = gv[0].iter().chain(gv[1].iter()).map(|x| x.to_bits()).collect();
        let eager = round % 4 == 3;
        std::thread::scope(|sc| {
            for r in 0..2 {
                let (b, g) = (&be[r], &graphs[r]);
                let body = &body;
                sc.spawn(move || if eager { body(r).unwrap() } else { b.graph_replay(g, 0).unwrap() });
            }
        });
        for r in 0..2 {
            let (x, g) = &bufs[r];
            let tag = if eager { "eager" } else { "replay" };
            assert_eq!(get(&*be[r], x), twice, "round {round} ({tag}) all-reduce rank {r}");
            let got: Vec<u32> = get(&*be[r], g).iter().map(|v| v.to_bits()).collect();
            assert!(got == want, "round {round} ({tag}) gather rank {r}");
        }
    }
}

// ---- lane ars: the tagged-line (LL) handshake, INTARSIA_HIP_AR_DRAIN=7, against the fence mode (1) ----
//
// The tests above run under whatever INTARSIA_HIP_AR_DRAIN the environment sets (`INTARSIA_HIP_AR_DRAIN=7` and
// `INTARSIA_HIP_AR_LL_MAX=2097152` run them all on the LL path). These build their groups with an explicit mode.

fn group_mode(drain: i32, ll_max: usize) -> Option<(Vec<Arc<dyn Backend>>, Vec<Arc<dyn Comm>>)> {
    let ords: Vec<usize> = std::env::var("INTARSIA_HIP_DEVICES")
        .ok()
        .map(|v| v.split(',').filter_map(|x| x.trim().parse().ok()).collect())
        .unwrap_or_else(|| vec![0, 1]);
    match intarsia_backend_hip::create_group_mode(&ords, drain, ll_max) {
        Ok(g) => Some(g),
        Err(e) => {
            eprintln!("skipping: no usable 2-GPU HIP group ({e})");
            None
        }
    }
}

fn raw_bits(be: &dyn Backend, t: &TensorRef) -> Vec<u8> {
    be.synchronize(0).unwrap();
    let mut bytes = vec![0u8; t.nbytes()];
    be.download(&t.buf, t.offset, &mut bytes, 0).unwrap();
    bytes
}

/// Every collective kind at decode, boundary and prefill sizes, one fixed input stream: the bytes both ranks end up with.
fn ll_suite(drain: i32, ll_max: usize) -> Option<Vec<Vec<u8>>> {
    use intarsia_core::comm::ArNorm;
    let (be, comms) = group_mode(drain, ll_max)?;
    let mut rng = SplitMix64::new(2026);
    let mut out: Vec<Vec<u8>> = Vec::new();
    let run2 = |f: &(dyn Fn(usize) + Sync)| {
        std::thread::scope(|sc| {
            for r in 0..2 {
                sc.spawn(move || f(r));
            }
        });
    };
    // in-place bf16 and f32, and f32 -> bf16: below, at and above the default 64 KiB LL limit, odd word counts, fewer words than blocks
    for (dt, n) in [
        (DType::Bf16, 8usize), (DType::Bf16, 16), (DType::Bf16, 24), (DType::Bf16, 4096), (DType::Bf16, 5120), (DType::Bf16, 3 * 5120),
        (DType::Bf16, 8 * 5120), (DType::Bf16, 32768), (DType::Bf16, 32776), (DType::Bf16, 262_144), (DType::Bf16, 1_048_576),
        (DType::F32, 4), (DType::F32, 4096), (DType::F32, 3 * 5120), (DType::F32, 16384), (DType::F32, 16388), (DType::F32, 300_000),
    ] {
        let a: Vec<f32> = rng.normal_vec(n, 3.0);
        let b: Vec<f32> = rng.normal_vec(n, 3.0);
        let t = [put(&*be[0], dt, &a), put(&*be[1], dt, &b)];
        run2(&|r| comms[r].all_reduce_sum(&t[r], 0).unwrap());
        out.push(raw_bits(&*be[0], &t[0]));
        out.push(raw_bits(&*be[1], &t[1]));
        if dt == DType::F32 {
            let p = [put(&*be[0], dt, &a), put(&*be[1], dt, &b)];
            let o = [put(&*be[0], DType::Bf16, &vec![0.0; n]), put(&*be[1], DType::Bf16, &vec![0.0; n])];
            run2(&|r| comms[r].all_reduce_sum_into(&p[r], &o[r], 0).unwrap());
            out.push(raw_bits(&*be[0], &o[0]));
            out.push(raw_bits(&*be[1], &o[1]));
        }
    }
    // all-gather, in place and disjoint
    for n in [4usize, 4096, 16384, 16388, 124_160] {
        for in_place in [false, true] {
            let a: Vec<f32> = rng.normal_vec(n, 3.0);
            let b: Vec<f32> = rng.normal_vec(n, 3.0);
            let vals = [&a, &b];
            let recv: Vec<TensorRef> = (0..2).map(|r| put(&*be[r], DType::F32, &vec![f32::from_bits(0x7fc0_dead); 2 * n]).reshaped(&[2, n])).collect();
            let send: Vec<TensorRef> = (0..2)
                .map(|r| {
                    if in_place {
                        upload_f32(&*be[r], &recv[r].rows(r, 1), vals[r]);
                        recv[r].rows(r, 1)
                    } else {
                        put(&*be[r], DType::F32, vals[r])
                    }
                })
                .collect();
            run2(&|r| comms[r].all_gather(&send[r], &recv[r], 0).unwrap());
            out.push(raw_bits(&*be[0], &recv[0]));
            out.push(raw_bits(&*be[1], &recv[1]));
        }
    }
    // the fused all-reduce + residual add + rmsnorm: 1, 3 and 16 rows of the 35B's and the 27B's hidden size, f32 and bf16 partials
    for (h, rows, dt) in [(2048usize, 1usize, DType::F32), (5120, 1, DType::F32), (5120, 3, DType::Bf16), (5120, 3, DType::F32), (2048, 16, DType::Bf16), (5120, 16, DType::F32)] {
        let n = rows * h;
        let parts = [rng.normal_vec(n, 2.0), rng.normal_vec(n, 2.0)];
        let res = rng.normal_vec(n, 1.0);
        let wv = rng.normal_vec(h, 1.0);
        let z = vec![0f32; n];
        let mk = |r: usize| {
            (
                put(&*be[r], dt, &parts[r]),
                put(&*be[r], DType::Bf16, &z),
                put(&*be[r], DType::Bf16, &res),
                put(&*be[r], DType::Bf16, &wv),
                put(&*be[r], DType::Bf16, &z),
                put(&*be[r], DType::Bf16, &z),
            )
        };
        let bufs = [mk(0), mk(1)];
        run2(&|r| {
            let (p, o, rs, w, nr, s) = &bufs[r];
            let a = ArNorm { partial: p, out: o, residual: rs, weight: w, norm: nr, sum: s, eps: 1e-6, w_offset: 0.0 };
            assert!(comms[r].ar_norm_supported(&a));
            comms[r].all_reduce_add_rmsnorm(&a, 0).unwrap();
        });
        for r in 0..2 {
            let (_, o, _, _, nr, s) = &bufs[r];
            out.push(raw_bits(&*be[r], o));
            out.push(raw_bits(&*be[r], nr));
            out.push(raw_bits(&*be[r], s));
        }
    }
    Some(out)
}

#[test]
fn ll_handshake_is_bitwise_the_fence_mode_across_ops_and_sizes() {
    let _one = serial();
    let Some(fence) = ll_suite(1, 0) else { return };
    for (ll_max, name) in [(64 << 10, "LL up to 64 KiB, fence above"), (2 << 20, "LL up to 2 MiB")] {
        let ll = ll_suite(7, ll_max).unwrap();
        assert_eq!(fence.len(), ll.len());
        for (i, (f, l)) in fence.iter().zip(&ll).enumerate() {
            assert!(f == l, "{name}: result {i} differs from mode 1");
        }
    }
}

/// One rank (alternating by round) is slow: random host sleeps before its launches, so the other waits in the handshake with its
/// data already pushed, or pushes while the late one has not started. Long unsynchronised chains of mixed collectives (both scratch
/// parities, several launches in flight), every result checked against the host sum of the inputs.
fn ll_stress(drain: i32, ll_max: usize, rounds: usize) -> usize {
    let Some((be, comms)) = group_mode(drain, ll_max) else { return 0 };
    let mut rng = SplitMix64::new(77);
    let mut wrong = 0;
    for round in 0..rounds {
        let k = 24usize;
        let sizes = [8usize, 2048, 4096, 5120, 3 * 5120];
        let mut ins: Vec<(Vec<f32>, Vec<f32>)> = Vec::new();
        let mut tens: [Vec<TensorRef>; 2] = [Vec::new(), Vec::new()];
        for i in 0..k {
            let n = sizes[i % sizes.len()];
            let a: Vec<f32> = rng.normal_vec(n, 2.0);
            let b: Vec<f32> = rng.normal_vec(n, 2.0);
            tens[0].push(put(&*be[0], DType::Bf16, &a));
            tens[1].push(put(&*be[1], DType::Bf16, &b));
            ins.push((get(&*be[0], &tens[0][i]), get(&*be[1], &tens[1][i])));
        }
        let slow = round % 2; // the delayed rank
        let delays: Vec<u64> = (0..k).map(|_| rng.next_u64() % 400).collect();
        std::thread::scope(|sc| {
            for r in 0..2 {
                let (c, ts, d) = (&comms[r], &tens[r], &delays);
                sc.spawn(move || {
                    for (i, t) in ts.iter().enumerate() {
                        if r == slow && i % 3 == 0 {
                            std::thread::sleep(std::time::Duration::from_micros(d[i]));
                        }
                        c.all_reduce_sum(t, 0).unwrap();
                    }
                });
            }
        });
        for i in 0..k {
            let want = host_sum(DType::Bf16, &ins[i].0, &ins[i].1);
            for r in 0..2 {
                if get(&*be[r], &tens[r][i]) != want {
                    wrong += 1;
                }
            }
        }
    }
    wrong
}

#[test]
fn ll_handshake_survives_a_delayed_rank() {
    let _one = serial();
    assert_eq!(ll_stress(7, 64 << 10, 120), 0, "wrong sums with the LL handshake while one rank was delayed");
}

/// Negative control: drain 8 is the LL kernel without the tag check. It reads whatever is in its scratch the moment its own push is
/// done, so with a delayed peer it must produce wrong sums; if it did not, the stress test above would prove nothing.
#[test]
fn ll_without_the_tag_check_is_caught_by_the_stress_test() {
    let _one = serial();
    let wrong = ll_stress(8, 64 << 10, 30);
    assert!(wrong > 0, "the tag check was removed and nothing went wrong: the stress test cannot see a stale read");
    eprintln!("negative control: {wrong} wrong results without the tag check");
}

/// `cargo test --release -p intarsia-backend-hip --test comm ar_norm_rate -- --ignored --nocapture`: us per fused all-reduce + add + rmsnorm
/// call, back to back on both ranks, at the decode shapes (35B hidden 2048, 27B hidden 5120, one row and a 3-row verify step), whatever
/// `INTARSIA_HIP_AR_DRAIN` says. Lane ars: the all-reduce rate test above does not touch this kernel.
#[test]
#[ignore]
fn ar_norm_rate() {
    use intarsia_core::comm::ArNorm;
    let _one = serial();
    let Some((be, comms)) = group() else { return };
    println!("{:>6} {:>5} {:>5} {:>10} {:>10}", "H", "rows", "part", "us/call", "launches");
    for (h, rows, dt) in [(2048usize, 1usize, DType::F32), (2048, 1, DType::Bf16), (5120, 1, DType::Bf16), (5120, 1, DType::F32), (5120, 3, DType::Bf16), (5120, 3, DType::F32)] {
        let n = rows * h;
        let z = vec![0f32; n];
        let w = vec![0f32; h];
        let mk = |r: usize| {
            (
                put(&*be[r], dt, &z),
                put(&*be[r], DType::Bf16, &z),
                put(&*be[r], DType::Bf16, &z),
                put(&*be[r], DType::Bf16, &w),
                put(&*be[r], DType::Bf16, &z),
                put(&*be[r], DType::Bf16, &z),
            )
        };
        let bufs = [mk(0), mk(1)];
        let iters = 4000;
        let run = |count: usize| {
            std::thread::scope(|sc| {
                for r in 0..2 {
                    let (c, b, bufs) = (&comms[r], &be[r], &bufs);
                    sc.spawn(move || {
                        let (p, o, rs, w, nr, s) = &bufs[r];
                        let a = ArNorm { partial: p, out: o, residual: rs, weight: w, norm: nr, sum: s, eps: 1e-6, w_offset: 0.0 };
                        for _ in 0..count {
                            c.all_reduce_add_rmsnorm(&a, 0).unwrap();
                        }
                        b.synchronize(0).unwrap();
                    });
                }
            });
        };
        run(100);
        let t0 = Instant::now();
        run(iters);
        let us = t0.elapsed().as_secs_f64() * 1e6 / iters as f64;
        println!("{:>6} {:>5} {:>5} {:>10.1} {:>10}", h, rows, if dt == DType::F32 { "f32" } else { "bf16" }, us, iters);
    }
}

/// `cargo test --release -p intarsia-backend-hip --test comm ar_graph_rate -- --ignored --nocapture`: us per all-reduce with 64 of them
/// recorded back to back in one hipGraph per rank and replayed (what the tp=2 decode graph does), whatever `INTARSIA_HIP_AR_DRAIN` says.
/// The eager rate tests above are limited by the host's launch rate (about 5 us per call), which hides a kernel that is shorter than that;
/// a replayed graph has no launch gaps, so this is the kernel's own critical path. Lane ars.
#[test]
#[ignore]
fn ar_graph_rate() {
    let _one = serial();
    let Some((be, comms)) = group() else { return };
    println!("{:>8} {:>6} {:>10}", "bytes", "dtype", "us/call");
    for (dt, n) in [(DType::Bf16, 2048usize), (DType::Bf16, 5120), (DType::Bf16, 3 * 5120), (DType::Bf16, 16 * 5120), (DType::F32, 5120)] {
        let z = vec![0f32; n];
        let t = [put(&*be[0], dt, &z), put(&*be[1], dt, &z)];
        const K: usize = 64;
        let graphs: Vec<_> = (0..2)
            .map(|r| {
                be[r]
                    .graph_capture(0, &mut || {
                        for _ in 0..K {
                            comms[r].all_reduce_sum(&t[r], 0)?;
                        }
                        Ok(())
                    })
                    .unwrap()
            })
            .collect();
        let run = |reps: usize| {
            std::thread::scope(|sc| {
                for r in 0..2 {
                    let (b, g) = (&be[r], &graphs[r]);
                    sc.spawn(move || {
                        for _ in 0..reps {
                            b.graph_replay(g, 0).unwrap();
                        }
                        b.synchronize(0).unwrap();
                    });
                }
            });
        };
        run(5);
        let reps = 200;
        let t0 = Instant::now();
        run(reps);
        let us = t0.elapsed().as_secs_f64() * 1e6 / (reps * K) as f64;
        println!("{:>8} {:>6} {:>10.2}", n * dt.size_bytes(), if dt == DType::F32 { "f32" } else { "bf16" }, us);
    }
}
