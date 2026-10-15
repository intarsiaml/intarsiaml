//! G0b: the non-GEMV, non-attention decode ops with `k = 2..=8` rows are bitwise `k` 1-row launches, per row: the existing
//! kernels are row-parallel (one threadgroup / thread range per row, no cross-row reduction), so no k-row variant is
//! needed. Each test runs the op once on a `[k, ...]` batch (the registry kernel the model would pick) and `k` times on the
//! 1-row slices of the same inputs (the decode step's launches) and compares every output byte.

#![cfg(target_os = "macos")]

mod common;

use common::*;
use intarsia_backend_metal::bench::{f32_to_bf16, Rng};
use intarsia_backend_metal::kernels::*;
use intarsia_backend_metal::MetalBackend;
use intarsia_core::backend::{Backend, Buffer, MemKind, TensorRef};
use intarsia_core::{DType, Op, OpAttrs, OpCall};

const KS: [usize; 7] = [2, 3, 4, 5, 6, 7, 8];

fn bytes_of(be: &MetalBackend, b: &Buffer, n: usize) -> Vec<u8> {
    let mut raw = vec![0u8; n];
    be.download(b, 0, &mut raw, 0).unwrap();
    raw
}

fn bf(be: &MetalBackend, n: usize, seed: u64) -> Buffer {
    let b = upload(be, &rand_bf16(n, seed));
    b
}

fn t(b: &Buffer, off: usize, dt: DType, shape: &[usize]) -> TensorRef {
    TensorRef { buf: b.clone(), offset: off, dtype: dt, shape: shape.to_vec() }
}

fn run(be: &MetalBackend, kernel: &str, call: &OpCall) {
    be.launch(kernel, call, 0).unwrap();
}

fn eq(be: &MetalBackend, a: &Buffer, b: &Buffer, n: usize, what: &str) {
    be.synchronize(0).unwrap();
    assert!(bytes_of(be, a, n) == bytes_of(be, b, n), "{what}: k-row output differs from the 1-row launches");
}

fn zeros(be: &MetalBackend, n: usize) -> Buffer {
    let b = be.alloc(n, MemKind::Device).unwrap();
    be.upload(&b, 0, &vec![0u8; n], 0).unwrap();
    b
}

#[test]
fn rmsnorm_rows() {
    let be = backend();
    for h in [4096usize, 2560, 1024, 128, 5120] {
        for off in [0usize, 16] {
            for k in KS {
                let (x, w) = (bf(&be, off + k * h, 1 + h as u64), bf(&be, off + h, 2));
                let (o1, o2) = (zeros(&be, (off + k * h) * 2), zeros(&be, (off + k * h) * 2));
                let at = OpAttrs { eps: 1e-6, w_offset: 0.0, ..Default::default() };
                let mk = |rows: usize, r0: usize, o: &Buffer| {
                    OpCall::new(
                        Op::RmsNorm,
                        vec![t(&x, (off + r0 * h) * 2, DType::Bf16, &[rows, h]), t(&w, off * 2, DType::Bf16, &[h])],
                        vec![t(o, (off + r0 * h) * 2, DType::Bf16, &[rows, h])],
                        at.clone(),
                    )
                };
                run(&be, RMSNORM_BF16, &mk(k, 0, &o1));
                for r in 0..k {
                    run(&be, RMSNORM_BF16, &mk(1, r, &o2));
                }
                eq(&be, &o1, &o2, (off + k * h) * 2, &format!("rmsnorm h {h} k {k} off {off}"));
            }
        }
    }
}

#[test]
fn rmsnorm_add_rows() {
    let be = backend();
    for h in [4096usize, 2560, 128] {
        for k in KS {
            let (x, r, w) = (bf(&be, k * h, 3), bf(&be, k * h, 4), bf(&be, h, 5));
            let (n1, s1, n2, s2) =
                (zeros(&be, k * h * 2), zeros(&be, k * h * 2), zeros(&be, k * h * 2), zeros(&be, k * h * 2));
            let at = OpAttrs { eps: 1e-6, w_offset: 0.0, ..Default::default() };
            let mk = |rows: usize, r0: usize, n: &Buffer, s: &Buffer| {
                OpCall::new(
                    Op::RmsNormAdd,
                    vec![
                        t(&x, r0 * h * 2, DType::Bf16, &[rows, h]),
                        t(&r, r0 * h * 2, DType::Bf16, &[rows, h]),
                        t(&w, 0, DType::Bf16, &[h]),
                    ],
                    vec![t(n, r0 * h * 2, DType::Bf16, &[rows, h]), t(s, r0 * h * 2, DType::Bf16, &[rows, h])],
                    at.clone(),
                )
            };
            run(&be, RMSNORM_ADD_BF16, &mk(k, 0, &n1, &s1));
            for r in 0..k {
                run(&be, RMSNORM_ADD_BF16, &mk(1, r, &n2, &s2));
            }
            eq(&be, &n1, &n2, k * h * 2, &format!("rmsnorm_add normed h {h} k {k}"));
            eq(&be, &s1, &s2, k * h * 2, &format!("rmsnorm_add sum h {h} k {k}"));
        }
    }
}

#[test]
fn add_and_silu_glu_rows() {
    let be = backend();
    for (op, kernel, n) in [(Op::Add, ADD_BF16, 4096usize), (Op::SiluGlu, SILU_GLU_BF16, 12288)] {
        for k in KS {
            let (a, b) = (bf(&be, k * n, 6), bf(&be, k * n, 7));
            let (o1, o2) = (zeros(&be, k * n * 2), zeros(&be, k * n * 2));
            let mk = |rows: usize, r0: usize, o: &Buffer| {
                OpCall::new(
                    op,
                    vec![t(&a, r0 * n * 2, DType::Bf16, &[rows, n]), t(&b, r0 * n * 2, DType::Bf16, &[rows, n])],
                    vec![t(o, r0 * n * 2, DType::Bf16, &[rows, n])],
                    OpAttrs::default(),
                )
            };
            run(&be, kernel, &mk(k, 0, &o1));
            for r in 0..k {
                run(&be, kernel, &mk(1, r, &o2));
            }
            eq(&be, &o1, &o2, k * n * 2, &format!("{kernel} k {k}"));
        }
    }
}

#[test]
fn embedding_rows() {
    let be = backend();
    let (v, h) = (1000usize, 4096usize);
    let table = bf(&be, v * h, 8);
    for k in KS {
        let ids: Vec<u8> = (0..k).flat_map(|i| (((i * 397 + 11) % v) as i32).to_le_bytes()).collect();
        let idb = be.alloc(ids.len(), MemKind::Device).unwrap();
        be.upload(&idb, 0, &ids, 0).unwrap();
        let (o1, o2) = (zeros(&be, k * h * 2), zeros(&be, k * h * 2));
        let mk = |rows: usize, r0: usize, o: &Buffer| {
            OpCall::new(
                Op::Embedding,
                vec![t(&idb, r0 * 4, DType::I32, &[rows]), t(&table, 0, DType::Bf16, &[v, h])],
                vec![t(o, r0 * h * 2, DType::Bf16, &[rows, h])],
                OpAttrs::default(),
            )
        };
        run(&be, EMBEDDING_BF16, &mk(k, 0, &o1));
        for r in 0..k {
            run(&be, EMBEDDING_BF16, &mk(1, r, &o2));
        }
        eq(&be, &o1, &o2, k * h * 2, &format!("embedding k {k}"));
    }
}

#[test]
fn argmax_rows() {
    let be = backend();
    let v = 151936usize;
    for dt in [DType::F32, DType::Bf16] {
        for k in KS {
            let mut rng = Rng(77 + k as u64);
            // ties and the max at assorted places (first / last / duplicated) as well as random rows
            let mut vals: Vec<f32> = (0..k * v).map(|_| rng.unit() * 20.0 - 10.0).collect();
            for r in 0..k {
                vals[r * v + (r * 18869) % v] = 25.0;
                vals[r * v + (r * 18869 + 40001) % v] = 25.0; // tie: the lowest index wins
            }
            vals[v - 1] = 26.0;
            vals[(k - 1) * v] = 26.0;
            let (l, kernel) = if dt == DType::F32 {
                let b = be.alloc(k * v * 4, MemKind::Device).unwrap();
                be.upload(&b, 0, &vals.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>(), 0).unwrap();
                (b, ARGMAX_F32)
            } else {
                let b = be.alloc(k * v * 2, MemKind::Device).unwrap();
                let raw: Vec<u8> = vals.iter().flat_map(|x| f32_to_bf16(*x).to_le_bytes()).collect();
                be.upload(&b, 0, &raw, 0).unwrap();
                (b, ARGMAX_BF16)
            };
            let esz = dt.size_bytes();
            let (o1, o2) = (zeros(&be, k * 4), zeros(&be, k * 4));
            let mk = |rows: usize, r0: usize, o: &Buffer| {
                OpCall::new(
                    Op::Argmax,
                    vec![t(&l, r0 * v * esz, dt, &[rows, v])],
                    vec![t(o, r0 * 4, DType::I32, &[rows])],
                    OpAttrs::default(),
                )
            };
            run(&be, kernel, &mk(k, 0, &o1));
            for r in 0..k {
                run(&be, kernel, &mk(1, r, &o2));
            }
            eq(&be, &o1, &o2, k * 4, &format!("{kernel} k {k}"));
        }
    }
}

#[test]
fn kv_append_rows() {
    let be = backend();
    let (nkv, d, cap) = (8usize, 128usize, 512usize);
    for k in KS {
        for pos0 in [0usize, 5, 100, 512 - k] {
            let src = bf(&be, k * nkv * d, 9);
            let (c1, c2) = (bf(&be, nkv * cap * d, 10), bf(&be, nkv * cap * d, 10));
            let mk = |rows: usize, r0: usize, c: &Buffer, p: usize| {
                OpCall::new(
                    Op::KvAppend,
                    vec![t(&src, r0 * nkv * d * 2, DType::Bf16, &[rows, nkv, d])],
                    vec![t(c, 0, DType::Bf16, &[nkv, cap, d])],
                    OpAttrs { pos0: p, ..Default::default() },
                )
            };
            run(&be, KV_APPEND_BF16, &mk(k, 0, &c1, pos0));
            for r in 0..k {
                run(&be, KV_APPEND_BF16, &mk(1, r, &c2, pos0 + r));
            }
            eq(&be, &c1, &c2, nkv * cap * d * 2, &format!("kv_append k {k} pos0 {pos0}"));
        }
    }
}

/// The fused attention prologue: T = k rows (the `_pf` form for head_dim 128) against k 1-row decode launches at positions
/// pos0 + r (host position, what the eager decode step does): q_out rows and both caches.
#[test]
fn qk_norm_rope_kv_rows() {
    let be = backend();
    for (n, nkv, d, rd) in [(32usize, 8usize, 128usize, 128usize), (16, 4, 64, 64), (8, 2, 128, 64), (6, 2, 32, 32)] {
        let cap = 2048usize;
        for k in KS {
            for pos0 in [0usize, 1, 127, 128, 1000, 1500, cap - k] {
                let (q, kk, vv) = (bf(&be, k * n * d, 11), bf(&be, k * nkv * d, 12), bf(&be, k * nkv * d, 13));
                let (wq, wk) = (bf(&be, d, 14), bf(&be, d, 15));
                let (qo1, qo2) = (zeros(&be, k * n * d * 2), zeros(&be, k * n * d * 2));
                let (kc1, kc2, vc1, vc2) = (
                    bf(&be, nkv * cap * d, 16),
                    bf(&be, nkv * cap * d, 16),
                    bf(&be, nkv * cap * d, 17),
                    bf(&be, nkv * cap * d, 17),
                );
                let at = |p: usize| OpAttrs {
                    pos0: p,
                    theta: 1_000_000.0,
                    eps: 1e-6,
                    w_offset: 0.0,
                    head_dim: d,
                    rotary_dim: rd,
                    n_heads: n,
                    n_kv_heads: nkv,
                    kv_cap: cap,
                    ..Default::default()
                };
                let mk = |rows: usize, r0: usize, qo: &Buffer, kc: &Buffer, vc: &Buffer, p: usize| {
                    OpCall::new(
                        Op::QkNormRopeKv,
                        vec![
                            t(&q, r0 * n * d * 2, DType::Bf16, &[rows, n, d]),
                            t(&kk, r0 * nkv * d * 2, DType::Bf16, &[rows, nkv, d]),
                            t(&vv, r0 * nkv * d * 2, DType::Bf16, &[rows, nkv, d]),
                            t(&wq, 0, DType::Bf16, &[d]),
                            t(&wk, 0, DType::Bf16, &[d]),
                        ],
                        vec![
                            t(qo, r0 * n * d * 2, DType::Bf16, &[rows, n, d]),
                            t(kc, 0, DType::Bf16, &[nkv, cap, d]),
                            t(vc, 0, DType::Bf16, &[nkv, cap, d]),
                        ],
                        at(p),
                    )
                };
                run(&be, QK_NORM_ROPE_KV_BF16, &mk(k, 0, &qo1, &kc1, &vc1, pos0));
                for r in 0..k {
                    run(&be, QK_NORM_ROPE_KV_BF16, &mk(1, r, &qo2, &kc2, &vc2, pos0 + r));
                }
                let w = format!("qk_norm_rope_kv {n}/{nkv}/{d}/{rd} k {k} pos0 {pos0}");
                eq(&be, &qo1, &qo2, k * n * d * 2, &format!("{w} q_out"));
                eq(&be, &kc1, &kc2, nkv * cap * d * 2, &format!("{w} k cache"));
                eq(&be, &vc1, &vc2, nkv * cap * d * 2, &format!("{w} v cache"));
            }
        }
    }
}
