//! Lane arx: the row-parallel GEMM that pushes its partial to the peer as its tiles finish (`INTARSIA_HIP_AR_PUSH`, `Comm::prepush`,
//! kernels/hip/ar_push.h) on two real GPUs. Needs two HIP devices that reach each other (skips otherwise; `INTARSIA_HIP_DEVICES=0,1`).
//!
//! 1. `pushed_all_reduce_is_bitwise_the_plain_pair`: `prepush`, `matmul_mxfp4_a8_mma`, all-reduce at the 27B tp2 prefill shapes (2048- and
//!    1810-row chunks of hidden 5120) and every case the push must decline (the staged kernel at M <= 64, an N that is not a multiple of 8,
//!    a partial larger than the wide slot): bitwise the same GEMM + all-reduce without the push, both ranks equal, and the push taken
//!    exactly where it should be.
//! 2. `pushed_all_reduce_survives_a_delayed_rank`: the same chains with one rank delayed (host sleeps before a third of its GEMMs),
//!    interleaved with plain wide all-reduces on the same buffers, every round against the reference bits.
//! 3. Negative control: drain 9 (the reduce without its wait) under the same stress must give wrong sums, or test 2 could not see a
//!    missing handshake.

#![cfg(ik_hip)]

use intarsia_core::backend::{MemKind, TensorRef};
use intarsia_core::exec::HostCall;
use intarsia_core::quant::cases::matmul_mxfp4_case;
use intarsia_core::quant::Mxfp4Act;
use intarsia_core::rng::SplitMix64;
use intarsia_core::{Backend, Comm, DType, HostTensor, OpCall, ShapeInfo};
use std::sync::Arc;

type Group = (Vec<Arc<dyn Backend>>, Vec<Arc<dyn Comm>>);

/// One group at a time (tests/comm.rs's SERIAL: two groups' collectives interleaved on one hardware queue would deadlock).
static SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn serial() -> std::sync::MutexGuard<'static, ()> {
    SERIAL.lock().unwrap_or_else(|e| e.into_inner())
}

fn group(drain: i32, push: bool) -> Option<Group> {
    let ords: Vec<usize> = std::env::var("INTARSIA_HIP_DEVICES")
        .ok()
        .map(|v| v.split(',').filter_map(|x| x.trim().parse().ok()).collect())
        .unwrap_or_else(|| vec![0, 1]);
    match intarsia_backend_hip::create_group_push(&ords, drain, 64 << 10, Some((64, 20 << 20)), push) {
        Ok(g) => Some(g),
        Err(e) => {
            eprintln!("skipping: no usable 2-GPU HIP group ({e})");
            None
        }
    }
}

/// (M, N, K, pushed): the 27B's row-parallel shapes at prefill and the calls the push must decline
const SHAPES: [(usize, usize, usize, bool); 9] = [
    (2048, 5120, 3072, true),  // a full 2048-row chunk (20 MiB: exactly the wide slot)
    (1810, 5120, 3072, true),  // the 4k prompt's second chunk (a part-filled last token block)
    (1024, 5120, 1024, true),
    (256, 5120, 512, true),    // the 256-token direct tile
    (130, 5120, 512, true),    // the smallest direct call above the LL limit
    (64, 5120, 512, false),    // the staged kernel (M <= 64): the all-reduce pushes itself
    (600, 777, 512, false),    // N % 8 != 0: no vector store, no push
    (4096, 5120, 512, false),  // 40 MiB: larger than one wide slot, prepush declines
    (2048, 3072, 1024, true),
];

fn put(be: &dyn Backend, t: &HostTensor) -> TensorRef {
    let buf = be.alloc(t.nbytes().max(16), MemKind::Device).unwrap();
    be.upload(&buf, 0, &t.data, 0).unwrap();
    TensorRef::new(buf, t.dtype, &t.shape)
}

/// One GEMM call per rank (each rank its own weights and activations, as a row-parallel shard), with its scratch.
fn calls(be: &[Arc<dyn Backend>], seed: u64) -> Vec<[OpCall; 2]> {
    let mut rng = SplitMix64::new(seed);
    SHAPES
        .iter()
        .map(|&(m, n, k, _)| {
            let mk = |r: usize, hc: &HostCall| {
                let b = &*be[r];
                let ins: Vec<_> = hc.inputs.iter().map(|t| put(b, t)).collect();
                let outs: Vec<_> = hc.outputs.iter().map(|o| put(b, &HostTensor::zeros(o.dtype, &o.shape))).collect();
                let mut call = OpCall::new(hc.op, ins, outs, hc.attrs.clone());
                let kern = b.kernels().into_iter().find(|x| x.name == "matmul_mxfp4_a8_mma").unwrap();
                let need = kern.scratch_bytes(&ShapeInfo::from_call(&call)).max(16);
                call.scratch = Some(TensorRef::new(b.alloc(need, MemKind::Device).unwrap(), DType::U8, &[need]));
                call
            };
            let h0 = matmul_mxfp4_case(&mut rng, m, n, k, Mxfp4Act::Fp8, DType::Bf16, DType::Bf16);
            let h1 = matmul_mxfp4_case(&mut rng, m, n, k, Mxfp4Act::Fp8, DType::Bf16, DType::Bf16);
            [mk(0, &h0), mk(1, &h1)]
        })
        .collect()
}

fn bits(be: &dyn Backend, t: &TensorRef) -> Vec<u8> {
    be.synchronize(0).unwrap();
    let mut v = vec![0u8; t.nbytes()];
    be.download(&t.buf, t.offset, &mut v, 0).unwrap();
    v
}

fn pushed(c: &Arc<dyn Comm>) -> u64 {
    let s = c.debug_state().unwrap_or_default();
    s.split("pushed ").nth(1).and_then(|t| t.split(|ch: char| !ch.is_ascii_digit()).next()).and_then(|n| n.parse().ok()).unwrap_or(0)
}

/// Every case on both ranks: [prepush], GEMM, in-place all-reduce of its output, then a plain wide all-reduce of a second buffer (the
/// push and the old path share the wide buffers and counters). `delay`: the slow rank and its host sleeps (us) per case.
fn run(g: &Group, cs: &[[OpCall; 2]], extra: &[TensorRef; 2], prepush: bool, delay: Option<(usize, &[u64])>) {
    let (be, comms) = g;
    std::thread::scope(|sc| {
        for r in 0..2 {
            let (b, c) = (&be[r], &comms[r]);
            sc.spawn(move || {
                for (i, call) in cs.iter().enumerate() {
                    if let Some((slow, d)) = delay {
                        if slow == r && i % 3 == 0 {
                            std::thread::sleep(std::time::Duration::from_micros(d[i % d.len()]));
                        }
                    }
                    let out = &call[r].outputs[0];
                    if prepush {
                        c.prepush(out).unwrap();
                    }
                    b.launch("matmul_mxfp4_a8_mma", &call[r], 0).unwrap();
                    c.all_reduce_sum(out, 0).unwrap();
                    c.all_reduce_sum(&extra[r], 0).unwrap();
                }
            });
        }
    });
}

fn outputs(g: &Group, cs: &[[OpCall; 2]]) -> Vec<[Vec<u8>; 2]> {
    cs.iter().map(|c| [bits(&*g.0[0], &c[0].outputs[0]), bits(&*g.0[1], &c[1].outputs[0])]).collect()
}

fn extra(be: &[Arc<dyn Backend>]) -> [TensorRef; 2] {
    // zeros: reduced again every case, so the values stay put; the call only advances the wide counters between the pushes
    let z = HostTensor::zeros(DType::Bf16, &[3 << 20]);
    [put(&*be[0], &z), put(&*be[1], &z)]
}

/// The reference: the plain GEMM + all-reduce (push off), both ranks equal.
fn reference(seed: u64) -> Option<Vec<[Vec<u8>; 2]>> {
    let g = group(7, false)?;
    let cs = calls(&g.0, seed);
    run(&g, &cs, &extra(&g.0), false, None);
    let r = outputs(&g, &cs);
    for (i, o) in r.iter().enumerate() {
        assert!(o[0] == o[1], "reference: case {i} differs between the ranks");
    }
    Some(r)
}

#[test]
fn pushed_all_reduce_is_bitwise_the_plain_pair() {
    let _one = serial();
    let Some(want) = reference(7) else { return };
    let g = group(7, true).unwrap();
    let cs = calls(&g.0, 7);
    let ex = extra(&g.0);
    // the push group without prepush is the plain pair too
    run(&g, &cs, &ex, false, None);
    assert_eq!(pushed(&g.1[0]), 0);
    for (i, (o, w)) in outputs(&g, &cs).iter().zip(&want).enumerate() {
        assert!(o[0] == w[0] && o[1] == w[1], "push group, no prepush: case {i} {:?} differs from the reference", SHAPES[i]);
    }
    run(&g, &cs, &ex, true, None);
    let n = SHAPES.iter().filter(|s| s.3).count() as u64;
    assert_eq!(pushed(&g.1[0]), 2 * n, "the GEMM pushed for a different set of calls than expected ({:?})", g.1[0].debug_state());
    for (i, (o, w)) in outputs(&g, &cs).iter().zip(&want).enumerate() {
        assert!(o[0] == o[1], "pushed: case {i} {:?} differs between the ranks", SHAPES[i]);
        assert!(o[0] == w[0], "pushed: case {i} {:?} differs from the plain GEMM + all-reduce", SHAPES[i]);
    }
}

/// Rounds of the chain with one rank (alternating) delayed by up to `max_us` before a third of its GEMMs; the number of wrong outputs
/// (rank x case x round) against the reference.
fn stress(drain: i32, rounds: usize, max_us: u64) -> usize {
    let Some(want) = reference(11) else { return 0 };
    let g = group(drain, true).unwrap();
    let cs = calls(&g.0, 11);
    let ex = extra(&g.0);
    let mut rng = SplitMix64::new(5);
    let mut wrong = 0;
    for round in 0..rounds {
        let d: Vec<u64> = (0..SHAPES.len()).map(|_| rng.next_u64() % max_us).collect();
        run(&g, &cs, &ex, true, Some((round % 2, &d)));
        for (o, w) in outputs(&g, &cs).iter().zip(&want) {
            wrong += (o[0] != w[0]) as usize + (o[1] != w[1]) as usize;
        }
    }
    wrong
}

#[test]
fn pushed_all_reduce_survives_a_delayed_rank() {
    let _one = serial();
    assert_eq!(stress(7, 30, 3000), 0, "wrong sums with the GEMM push while one rank was delayed");
}

#[test]
fn pushed_all_reduce_without_the_wait_is_caught_by_the_stress_test() {
    let _one = serial();
    let wrong = stress(9, 6, 3000);
    assert!(wrong > 0, "the pushed reduce's wait was removed and nothing went wrong: the stress test cannot see a missing handshake");
    eprintln!("negative control (pushed, no wait): {wrong} wrong results");
}
