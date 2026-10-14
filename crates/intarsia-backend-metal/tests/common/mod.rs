#![allow(dead_code)]

use intarsia_backend_metal::bench::{bf16_to_f32, f32_to_bf16, Rng};
use intarsia_backend_metal::MetalBackend;
use intarsia_core::backend::{Backend, Buffer, MemKind, TensorRef};
use intarsia_core::{DType, Op, OpAttrs, OpCall};

pub fn backend() -> MetalBackend {
    MetalBackend::new().expect("Metal backend")
}

pub fn rand_bf16(n: usize, seed: u64) -> Vec<u16> {
    let mut r = Rng(seed | 1);
    (0..n).map(|_| f32_to_bf16(r.unit())).collect()
}

pub fn as_bytes(v: &[u16]) -> &[u8] {
    // SAFETY: u16 -> u8 view of the same memory.
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 2) }
}

pub fn upload(be: &MetalBackend, data: &[u16]) -> Buffer {
    let b = be.alloc(data.len() * 2, MemKind::Device).unwrap();
    be.upload(&b, 0, as_bytes(data), 0).unwrap();
    b
}

/// f64 reference y[i] = sum_k w[i*k + j] * x[j] for the rows in `rows`; also the L1 mass sum |w x| for the tolerance.
pub fn reference(w: &[u16], x: &[u16], k: usize, n: usize) -> Vec<(f64, f64)> {
    let mut out = vec![(0.0, 0.0); n];
    let threads = std::thread::available_parallelism().map_or(4, |p| p.get());
    let chunk = n.div_ceil(threads).max(1);
    std::thread::scope(|s| {
        for (ci, oc) in out.chunks_mut(chunk).enumerate() {
            s.spawn(move || {
                for (i, o) in oc.iter_mut().enumerate() {
                    let row = &w[(ci * chunk + i) * k..][..k];
                    let (mut acc, mut l1) = (0.0f64, 0.0f64);
                    for (a, b) in row.iter().zip(x) {
                        let p = bf16_to_f32(*a) as f64 * bf16_to_f32(*b) as f64;
                        acc += p;
                        l1 += p.abs();
                    }
                    *o = (acc, l1);
                }
            });
        }
    });
    out
}

pub fn matmul_call(x: TensorRef, w: TensorRef, y: TensorRef) -> OpCall {
    OpCall::new(Op::Matmul, vec![x, w], vec![y], OpAttrs::default())
}

pub fn tref(buf: &Buffer, offset: usize, shape: &[usize]) -> TensorRef {
    TensorRef { buf: buf.clone(), offset, dtype: DType::Bf16, shape: shape.to_vec() }
}

/// Compare GPU bf16 output rows against the reference; returns (max |err|/tol-unit, rows checked). Panics on a miss.
/// Allowed error: half a bf16 ulp of the result (2^-8 relative, 2% slack) + f32 accumulation noise
/// (1e-6 of the L1 mass).
pub fn check(gpu: &[u16], refs: &[(f64, f64)], rows: impl Iterator<Item = usize>, what: &str) -> usize {
    let (mut n, mut worst) = (0, 0.0f64);
    for i in rows {
        let (r, l1) = refs[i];
        let g = bf16_to_f32(gpu[i]) as f64;
        let tol = r.abs() * 2f64.powi(-8) * 1.02 + 2e-6 * l1 + 1e-30;
        assert!((g - r).abs() <= tol, "{what}: row {i}: gpu {g} ref {r} (|err| {} > tol {tol})", (g - r).abs());
        worst = worst.max((g - r).abs() / tol);
        n += 1;
    }
    eprintln!("{what}: {n} rows, worst error {:.3} of tolerance", worst);
    n
}
