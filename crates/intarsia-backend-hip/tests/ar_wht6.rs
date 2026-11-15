//! The lossy wht6 all-reduce wire (lane wht6-ar, `kernels/hip/allreduce_wht6.hip`, `INTARSIA_HIP_AR_WIRE=wht6`) on real GPUs,
//! against the exact all-reduce. Needs two HIP devices that can reach each other (skips otherwise); `INTARSIA_HIP_DEVICES=0,1`
//! picks them. The rate test is `--ignored --nocapture`.
#![cfg(ik_hip)]

use intarsia_core::dtype::{decode_to_f32, encode_from_f32};
use intarsia_core::rng::SplitMix64;
use intarsia_core::{Backend, Comm, DType, MemKind, TensorRef};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Instant;

/// One group at a time (see `tests/comm.rs`).
static SERIAL: Mutex<()> = Mutex::new(());
fn serial() -> MutexGuard<'static, ()> {
    SERIAL.lock().unwrap_or_else(|e| e.into_inner())
}

type Grp = (Vec<Arc<dyn Backend>>, Vec<Arc<dyn Comm>>);
/// The default group (LL up to 128 KiB, wide path 64 blocks / 20 MiB) with the wire given: `Some(min_bytes)` = wht6.
fn group(drain: i32, wire: Option<usize>) -> Option<Grp> {
    let ords: Vec<usize> = std::env::var("INTARSIA_HIP_DEVICES")
        .ok()
        .map(|v| v.split(',').filter_map(|x| x.trim().parse().ok()).collect())
        .unwrap_or_else(|| vec![0, 1]);
    match intarsia_backend_hip::create_group_wire(&ords, drain, 128 << 10, Some((64, 20 << 20)), false, wire) {
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
fn upload(be: &dyn Backend, t: &TensorRef, vals: &[f32]) {
    let mut bytes = vec![0u8; vals.len() * t.dtype.size_bytes()];
    encode_from_f32(t.dtype, vals, &mut bytes);
    be.upload(&t.buf, 0, &bytes, 0).unwrap();
}
fn raw(be: &dyn Backend, t: &TensorRef) -> Vec<u8> {
    be.synchronize(0).unwrap();
    let mut bytes = vec![0u8; t.nbytes()];
    be.download(&t.buf, t.offset, &mut bytes, 0).unwrap();
    bytes
}
fn floats(dt: DType, bytes: &[u8]) -> Vec<f32> {
    let mut out = vec![0f32; bytes.len() / dt.size_bytes()];
    decode_to_f32(dt, bytes, &mut out);
    out
}
fn enc(dt: DType, v: &[f32]) -> Vec<u8> {
    let mut bytes = vec![0u8; v.len() * dt.size_bytes()];
    encode_from_f32(dt, v, &mut bytes);
    bytes
}
fn bf16_round(v: &[f32]) -> Vec<f32> {
    floats(DType::Bf16, &enc(DType::Bf16, v))
}
/// The exact all-reduce's result: f32 add, one round to nearest even into `dt`.
fn host_sum(dt: DType, a: &[f32], b: &[f32]) -> Vec<u8> {
    let s: Vec<f32> = a.iter().zip(b).map(|(x, y)| x + y).collect();
    enc(dt, &s)
}
fn run2(f: &(dyn Fn(usize) + Sync)) {
    std::thread::scope(|sc| {
        for r in 0..2 {
            sc.spawn(move || f(r));
        }
    });
}

/// Activations like a residual stream's partial: normal, a few channels (of a hidden size of 5120) far larger, some rows near zero,
/// the first group or two all zero.
fn partial(rng: &mut SplitMix64, n: usize) -> Vec<f32> {
    let mut v = rng.normal_vec(n, 1.5);
    for (i, x) in v.iter_mut().enumerate() {
        let (row, col) = (i / 5120, i % 5120);
        if col == 7 || col == 1300 || col == 4099 {
            *x *= 60.0;
        }
        if row % 11 == 3 {
            *x *= 1e-3;
        }
    }
    // two all-zero groups (the scale's zero guard), the second one only on the rank that draws an even first value
    let k = if v.len() >= 128 && v[200 % v.len()].to_bits() & 1 == 0 { 128 } else { 64 };
    for x in v.iter_mut().take(k.min(n)) {
        *x = 0.0;
    }
    v
}

/// Orthonormal 64-point Walsh-Hadamard transform of one group, in f64.
fn wht64(g: &[f64]) -> [f64; 64] {
    let mut v = [0f64; 64];
    v.copy_from_slice(g);
    let mut h = 1;
    while h < 64 {
        for i in 0..64 {
            if i & h == 0 {
                let (x, y) = (v[i], v[i | h]);
                v[i] = x + y;
                v[i | h] = x - y;
            }
        }
        h <<= 1;
    }
    v.map(|x| x / 8.0)
}

/// The codec's error bound for a reduced f32 result `got` of the partials `a`, `b`: per group, every element of the ROTATED error is
/// at most half a quantiser step of each rank (step = amax of the rank's rotated group / 31, shipped as a bf16: 0.4 % slack) plus
/// f32 round-off. Returns (worst rotated error / bound, rms error / rms of the exact sum).
fn check_bound(a: &[f32], b: &[f32], got: &[f32]) -> (f64, f64) {
    assert_eq!(a.len() % 64, 0);
    let (mut worst, mut se, mut ss) = (0f64, 0f64, 0f64);
    for g in 0..a.len() / 64 {
        let r = g * 64..(g + 1) * 64;
        let fa: Vec<f64> = a[r.clone()].iter().map(|&x| x as f64).collect();
        let fb: Vec<f64> = b[r.clone()].iter().map(|&x| x as f64).collect();
        let err: Vec<f64> = (0..64).map(|i| got[g * 64 + i] as f64 - (fa[i] + fb[i])).collect();
        let amax = |v: &[f64; 64]| v.iter().fold(0f64, |m, x| m.max(x.abs()));
        let (a0, a1) = (amax(&wht64(&fa)), amax(&wht64(&fb)));
        let bound = (0.5 * 1.004 / 31.0 + 2e-5) * (a0 + a1);
        let e = amax(&wht64(&err));
        if bound > 0.0 {
            worst = worst.max(e / bound);
        } else {
            assert_eq!(e, 0.0, "group {g}: both partials are zero, the sum is not");
        }
        se += err.iter().map(|x| x * x).sum::<f64>();
        ss += (0..64).map(|i| (fa[i] + fb[i]).powi(2)).sum::<f64>();
    }
    (worst, (se / ss.max(1e-300)).sqrt())
}

const KIB128: usize = 128 << 10;
/// f32 element counts: below / at the 128 KiB floor (exact), just above it (lossy), not multiples of 64 (exact), a last chunk that
/// is not full, a single chunk more than the blocks, a 2048-row chunk of the 27B (hidden 5120), and one above a slot's capacity
/// of 26 214 400 elements (two launches).
const SIZES: [usize; 10] =
    [KIB128 / 4 - 64, KIB128 / 4, KIB128 / 4 + 64, KIB128 / 4 + 32, 40_000, 100_008, 65 * 2048, 300 * 2048 + 64, 2048 * 5120, 26_214_400 + 64 * 1000];

/// Is an all-reduce of `n` elements of `dt` lossy at a floor of `min` bytes (128 KiB LL limit: the wide path starts above it)?
fn lossy(n: usize, dt: DType, min: usize) -> bool {
    let bytes = n * dt.size_bytes();
    bytes > KIB128 && bytes >= min && n % 64 == 0
}

#[test]
fn wht6_wire_is_rank_identical_within_its_bound_and_exact_below_the_floor() {
    let _one = serial();
    let Some((be, comms)) = group(7, Some(KIB128)) else { return };
    let mut rng = SplitMix64::new(606);
    let (mut worst_all, mut rms_all) = (0f64, 0f64);
    for n in SIZES {
        let (a, b) = (partial(&mut rng, n), partial(&mut rng, n));
        // f32 in place
        let t = [put(&*be[0], DType::F32, &a), put(&*be[1], DType::F32, &b)];
        run2(&|r| comms[r].all_reduce_sum(&t[r], 0).unwrap());
        let ff = [raw(&*be[0], &t[0]), raw(&*be[1], &t[1])];
        assert!(ff[0] == ff[1], "n {n} f32: the ranks' reduced outputs differ");
        // f32 -> bf16, the partial left alone
        let p = [put(&*be[0], DType::F32, &a), put(&*be[1], DType::F32, &b)];
        let o = [put(&*be[0], DType::Bf16, &vec![0.0; n]), put(&*be[1], DType::Bf16, &vec![0.0; n])];
        run2(&|r| comms[r].all_reduce_sum_into(&p[r], &o[r], 0).unwrap());
        let fb = [raw(&*be[0], &o[0]), raw(&*be[1], &o[1])];
        assert!(fb[0] == fb[1], "n {n} f32 -> bf16: the ranks' reduced outputs differ");
        assert!(raw(&*be[0], &p[0]) == enc(DType::F32, &a), "n {n} f32 -> bf16: the partial was changed");
        if lossy(n, DType::F32, KIB128) {
            let got = floats(DType::F32, &ff[0]);
            let (worst, rms) = check_bound(&a, &b, &got);
            eprintln!("wht6 f32 n {n:>9}: worst rotated error {worst:.3} of the bound, rms error {:.3} % of the sum's rms", rms * 100.0);
            assert!(worst <= 1.0, "n {n}: a rotated error is {worst:.3} x the codec's bound");
            assert!(rms < 0.1, "n {n}: rms error {rms}");
            assert!(ff[0] != host_sum(DType::F32, &a, &b), "n {n}: expected the lossy wire, got the exact sum");
            // the bf16 output is the f32 output rounded once
            assert!(fb[0] == enc(DType::Bf16, &got), "n {n}: f32 -> bf16 is not the f32 result rounded to bf16");
            worst_all = worst_all.max(worst);
            rms_all = rms_all.max(rms);
        } else {
            assert!(ff[0] == host_sum(DType::F32, &a, &b), "n {n} f32: below the floor or not whole groups, but not the exact sum");
            assert!(fb[0] == host_sum(DType::Bf16, &a, &b), "n {n} f32 -> bf16: below the floor or not whole groups, but not the exact sum");
        }
        // bf16 in place: the same code as f32 partials holding the same (bf16) values, or the exact sum
        let (ab, bb) = (bf16_round(&a), bf16_round(&b));
        let t = [put(&*be[0], DType::Bf16, &ab), put(&*be[1], DType::Bf16, &bb)];
        run2(&|r| comms[r].all_reduce_sum(&t[r], 0).unwrap());
        let bbits = [raw(&*be[0], &t[0]), raw(&*be[1], &t[1])];
        assert!(bbits[0] == bbits[1], "n {n} bf16: the ranks' reduced outputs differ");
        if lossy(n, DType::Bf16, KIB128) {
            assert!(lossy(n, DType::F32, KIB128));
            let p = [put(&*be[0], DType::F32, &ab), put(&*be[1], DType::F32, &bb)];
            let o = [put(&*be[0], DType::Bf16, &vec![0.0; n]), put(&*be[1], DType::Bf16, &vec![0.0; n])];
            run2(&|r| comms[r].all_reduce_sum_into(&p[r], &o[r], 0).unwrap());
            assert!(bbits[0] == raw(&*be[0], &o[0]), "n {n}: the bf16 partial's code differs from the f32 partial's of the same values");
            assert!(bbits[0] != host_sum(DType::Bf16, &ab, &bb), "n {n} bf16: expected the lossy wire, got the exact sum");
        } else {
            assert!(bbits[0] == host_sum(DType::Bf16, &ab, &bb), "n {n} bf16: below the floor or not whole groups, but not the exact sum");
        }
    }
    eprintln!("wht6: worst rotated error {worst_all:.3} of the bound, worst rms error {:.3} %", rms_all * 100.0);
    let st = comms[0].debug_state().unwrap_or_default();
    assert!(st.contains("wht6 launches"), "{st}");
}

/// A higher floor keeps more exact; with the wire off nothing is lossy.
#[test]
fn wht6_floor_and_the_exact_default() {
    let _one = serial();
    let mut rng = SplitMix64::new(77);
    let n = 2048 * 256; // 2 MiB of f32
    let (a, b) = (partial(&mut rng, n), partial(&mut rng, n));
    for (wire, want_lossy) in [(None, false), (Some(4 << 20), false), (Some(2 << 20), true)] {
        let Some((be, comms)) = group(7, wire) else { return };
        let t = [put(&*be[0], DType::F32, &a), put(&*be[1], DType::F32, &b)];
        run2(&|r| comms[r].all_reduce_sum(&t[r], 0).unwrap());
        let got = [raw(&*be[0], &t[0]), raw(&*be[1], &t[1])];
        assert!(got[0] == got[1]);
        assert_eq!(got[0] != host_sum(DType::F32, &a, &b), want_lossy, "wire {wire:?}");
    }
}

/// Chains of mixed collectives (small LL ones, exact wide ones, lossy ones, bf16 and f32) while one rank is delayed at random:
/// every result must be the bits of the undelayed first round (and the exact ones the host sum). Returns the wrong results.
fn stress(drain: i32, rounds: usize) -> usize {
    let Some((be, comms)) = group(drain, Some(KIB128)) else { return 0 };
    let mut rng = SplitMix64::new(92);
    let shapes: [(DType, usize); 10] = [
        (DType::Bf16, 8),
        (DType::F32, 2048 * 640),
        (DType::Bf16, 2048 * 5120),
        (DType::F32, 5120),
        (DType::F32, 100_008),
        (DType::Bf16, 300 * 2048 + 64),
        (DType::F32, 2048 * 5120),
        (DType::Bf16, 40_000),
        (DType::F32, 65 * 2048),
        (DType::Bf16, 3 * 5120),
    ];
    let mut ins: Vec<[Vec<f32>; 2]> = Vec::new();
    let mut tens: [Vec<TensorRef>; 2] = [Vec::new(), Vec::new()];
    for (dt, n) in shapes {
        let ab = [bf16_round(&partial(&mut rng, n)), bf16_round(&partial(&mut rng, n))];
        for r in 0..2 {
            tens[r].push(put(&*be[r], dt, &ab[r]));
        }
        ins.push(ab);
    }
    let mut want: Vec<Vec<u8>> = Vec::new();
    let mut wrong = 0;
    for round in 0..rounds {
        if round > 0 {
            for (i, ab) in ins.iter().enumerate() {
                for r in 0..2 {
                    upload(&*be[r], &tens[r][i], &ab[r]);
                }
            }
            be[0].synchronize(0).unwrap();
            be[1].synchronize(0).unwrap();
        }
        let slow = round % 2;
        let delays: Vec<u64> = (0..shapes.len()).map(|_| rng.next_u64() % 2000).collect();
        std::thread::scope(|sc| {
            for r in 0..2 {
                let (c, ts, d) = (&comms[r], &tens[r], &delays);
                sc.spawn(move || {
                    for (i, t) in ts.iter().enumerate() {
                        if round > 0 && r == slow && i % 3 != 1 {
                            std::thread::sleep(std::time::Duration::from_micros(d[i]));
                        }
                        c.all_reduce_sum(t, 0).unwrap();
                    }
                });
            }
        });
        for (i, (dt, n)) in shapes.iter().enumerate() {
            let got = [raw(&*be[0], &tens[0][i]), raw(&*be[1], &tens[1][i])];
            if round == 0 {
                // the reference: checked on its own merits
                if lossy(*n, *dt, KIB128) {
                    let (worst, rms) = check_bound(&ins[i][0], &ins[i][1], &floats(*dt, &got[0]));
                    // a bf16 output adds its own rounding to the rotated error: only the rms is checked for it
                    if (*dt == DType::F32 && worst > 1.0) || rms > 0.1 || got[0] != got[1] {
                        wrong += 1;
                    }
                } else if got[0] != host_sum(*dt, &ins[i][0], &ins[i][1]) || got[0] != got[1] {
                    wrong += 1;
                }
                want.push(got[0].clone());
            } else {
                wrong += got.iter().filter(|g| **g != want[i]).count();
            }
        }
    }
    wrong
}

#[test]
fn wht6_wire_survives_a_delayed_rank_between_exact_collectives() {
    let _one = serial();
    assert_eq!(stress(7, 30), 0, "wrong or unequal results on the wht6 wire while one rank was delayed");
}

/// Negative control: drain 9 is the handshake without its wait; with a delayed peer the stress test must see wrong results.
#[test]
fn wht6_wire_without_the_wait_is_caught_by_the_stress_test() {
    let _one = serial();
    let wrong = stress(9, 8);
    assert!(wrong > 0, "the wht6 path's wait was removed and nothing went wrong: the stress test cannot see a stale read");
    eprintln!("negative control (wht6, no wait): {wrong} wrong results");
}

/// `cargo test --release -p intarsia-backend-hip --test ar_wht6 wht6_rate -- --ignored --nocapture`: us per all-reduce (both ranks,
/// back to back, the slower rank) for the exact wide path and the wht6 wire at prefill sizes of hidden 2048 (35B) and 5120 (27B).
#[test]
#[ignore]
fn wht6_rate() {
    let _one = serial();
    let iters = 200;
    for (label, wire) in [("exact", None), ("wht6 ", Some(KIB128))] {
        let Some((be, comms)) = group(7, wire) else { return };
        let mut rng = SplitMix64::new(5);
        for hidden in [2048usize, 5120] {
            for rows in [64usize, 256, 1024, 2048] {
                for dt in [DType::Bf16, DType::F32] {
                    let n = rows * hidden;
                    let v = rng.normal_vec(n, 1.0);
                    let t = [put(&*be[0], dt, &v), put(&*be[1], dt, &v)];
                    let o = [put(&*be[0], DType::Bf16, &v), put(&*be[1], DType::Bf16, &v)];
                    let time = |k: usize| {
                        let bar = std::sync::Barrier::new(2);
                        let us = Mutex::new(0f64);
                        std::thread::scope(|sc| {
                            for r in 0..2 {
                                let (c, t, o, b, bar, us) = (&comms[r], &t[r], &o[r], &be[r], &bar, &us);
                                sc.spawn(move || {
                                    b.synchronize(0).unwrap();
                                    bar.wait();
                                    let t0 = Instant::now();
                                    for _ in 0..k {
                                        c.all_reduce_sum_into(t, o, 0).unwrap();
                                    }
                                    b.synchronize(0).unwrap();
                                    let d = t0.elapsed().as_secs_f64() * 1e6 / k as f64;
                                    let mut u = us.lock().unwrap();
                                    *u = u.max(d);
                                });
                            }
                        });
                        let u = *us.lock().unwrap();
                        u
                    };
                    time(10);
                    let us = time(iters);
                    eprintln!(
                        "AR {label} {:>4} rows x {hidden} {:<4} -> bf16 {:>9} B {us:>8.1} us/call {:>6.2} GB/s of partial",
                        rows,
                        dt.name(),
                        n * dt.size_bytes(),
                        (n * dt.size_bytes()) as f64 / us / 1e3
                    );
                }
            }
        }
    }
}
