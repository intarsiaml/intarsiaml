//! Round 9 (docs/results/2026-10-19-hip-round9.md): small uploads and downloads go through page-locked staging
//! (device.rs `Staging`): an upload is queued from one of 8 ring slots without waiting for the stream, a download goes
//! through its own slot. Every copy must still land the caller's bytes, in stream order, although the caller rewrites its
//! source right after each call and the ring wraps while the stream is still busy.
//!
//! The stream is kept busy by a kernel spinning ~300 ms on the wall clock, so every staged copy is still pending when the
//! host moves on. Phase A: 24 uploads of at most one slot (3 laps of the ring) to distinct destinations, each from the same
//! host vector, rewritten right after the call. Phase B: 16 more, mixed with uploads larger than a slot (the blocking
//! path). Every destination must hold its own bytes. In the staged modes the ring's first lap must return while the stream
//! is busy (no wait), and the ninth upload (slot 0 again) must wait for the first one's copy, i.e. for the spin. Then a
//! download right behind a pending staged upload of the same bytes.
//!
//! INTARSIA_HIP_STAGE (read at every copy): default (uploads and downloads staged), `up` (uploads only), `0` (neither).
//! Its own test binary with a single test: it sets an environment variable, which no other test may race with.
#![cfg(ik_hip)]

use intarsia_core::backend::MemKind;
use std::time::{Duration, Instant};

const SMALL: [usize; 8] = [4, 8, 1000, 4096, 65536, 3, 12345, 64];
const MIXED: [usize; 8] = [65537, 4, 70000, 4096, 65536, 100000, 8, 1];

fn size(k: usize) -> usize {
    if k < 24 {
        SMALL[k % 8]
    } else {
        MIXED[k % 8]
    }
}

fn pattern(k: usize, n: usize) -> Vec<u8> {
    (0..n).map(|j| (k.wrapping_mul(131) ^ j.wrapping_mul(7) ^ (j >> 8)) as u8).collect()
}

#[test]
fn staged_copies_land_in_order() {
    let ord = std::env::var("INTARSIA_HIP_DEVICE").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
    let (b, spin) = match intarsia_backend_hip::create_with_spin(ord) {
        Ok(x) => x,
        Err(e) => {
            eprintln!("skipping: no usable HIP device ({e})");
            return;
        }
    };
    let n_up = 40usize;
    let region = 100_000usize;
    let dst = b.alloc(n_up * region, MemKind::Device).unwrap();
    for mode in ["", "up", "0"] {
        if mode.is_empty() {
            std::env::remove_var("INTARSIA_HIP_STAGE");
        } else {
            std::env::set_var("INTARSIA_HIP_STAGE", mode);
        }
        let staged = mode != "0";
        b.zero(&dst, 0).unwrap();
        b.synchronize(0).unwrap();
        spin(300_000).unwrap();
        let t0 = Instant::now();
        let mut host = vec![0u8; region];
        let (mut first_lap, mut ninth) = (Duration::ZERO, Duration::ZERO);
        for k in 0..n_up {
            let n = size(k);
            host[..n].copy_from_slice(&pattern(k, n));
            let t = Instant::now();
            b.upload(&dst, k * region, &host[..n], 0).unwrap();
            if k == 8 {
                ninth = t.elapsed();
            }
            // the caller may reuse its buffer at once: overwrite it before the queued copy can have run
            host.iter_mut().for_each(|x| *x = 0xa5);
            if k == 7 {
                first_lap = t0.elapsed();
            }
        }
        // the downloads wait for the stream (the spin and every upload queued before them)
        let mut bad = 0;
        for k in 0..n_up {
            let n = size(k);
            let mut got = vec![0u8; n];
            b.download(&dst, k * region, &mut got, 0).unwrap();
            let want = pattern(k, n);
            if got != want {
                bad += 1;
                let at = got.iter().zip(&want).position(|(a, b)| a != b).unwrap();
                eprintln!("mode {mode:?}: upload {k} ({n} bytes) differs at byte {at}: {} != {}", got[at], want[at]);
            }
        }
        eprintln!("mode {mode:?}: {} of {n_up} uploads bitwise; first lap of 8 returned after {first_lap:?}, the 9th took {ninth:?}", n_up - bad);
        assert_eq!(bad, 0, "mode {mode:?}: {bad} of {n_up} uploads differ");
        if staged {
            assert!(first_lap < Duration::from_millis(100), "mode {mode:?}: the first 8 staged uploads waited ({first_lap:?})");
            assert!(ninth >= Duration::from_millis(100), "mode {mode:?}: the 9th upload reused slot 0 without waiting ({ninth:?})");
        } else {
            assert!(first_lap >= Duration::from_millis(200), "mode \"0\": the blocking uploads did not wait ({first_lap:?})");
        }
        // a download right behind a pending staged upload of the same bytes (stream order, no host wait in between)
        spin(50_000).unwrap();
        let v = pattern(99, 4096);
        let t = Instant::now();
        b.upload(&dst, 0, &v, 0).unwrap();
        let mut got = vec![0u8; 4096];
        b.download(&dst, 0, &mut got, 0).unwrap();
        let waited = t.elapsed();
        assert_eq!(got, v, "mode {mode:?}: download behind a pending staged upload");
        assert!(waited >= Duration::from_millis(30), "mode {mode:?}: the download returned after {waited:?}, before the spin ended");
    }
    std::env::remove_var("INTARSIA_HIP_STAGE");
}
