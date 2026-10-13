//! The two-rank peer-to-peer all-reduce on real GPUs. Needs two HIP devices that can reach each other (skips
//! otherwise); `INTARSIA_HIP_DEVICES=0,1` picks them. The rate test is `--ignored --nocapture`.

#![cfg(ik_hip)]

use intarsia_core::dtype::{decode_to_f32, encode_from_f32};
use intarsia_core::rng::SplitMix64;
use intarsia_core::{Backend, Comm, DType, MemKind, TensorRef};
use std::sync::Arc;
use std::time::Instant;

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

#[test]
fn an_aborted_group_refuses_collectives_instead_of_waiting() {
    let Some((be, comms)) = group() else { return };
    comms[0].abort();
    let t = put(&*be[0], DType::Bf16, &[1.0; 8]);
    assert!(comms[0].all_reduce_sum(&t, 0).is_err());
}

/// `cargo test --release -p intarsia-backend-hip --test comm -- --ignored --nocapture`
#[test]
#[ignore]
fn all_reduce_rate() {
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
