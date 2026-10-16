#![cfg(target_os = "macos")]
//! Phase B2b low-latency kernels: row norms that issue all loads before the reduction (`rmsnorm_*_ex*`) and
//! `argmax_f32_v4`. Norms are checked against an f64 reference and bitwise against the 256-thread kernels they replace,
//! over the sizes where the launcher takes the fast path and where it falls back (more than 8192 elements).

mod common;
use common::*;
use intarsia_backend_metal::bench::{bf16_to_f32, f32_to_bf16, Rng};
use intarsia_backend_metal::kernels::{ARGMAX_F32, QK_NORM_BF16, RMSNORM_ADD_BF16, RMSNORM_BF16};
use intarsia_backend_metal::{set_bytes, MetalBackend, MetalBuffer};
use intarsia_core::backend::{Backend, Buffer, MemKind, TensorRef};
use intarsia_core::{DType, Op, OpAttrs, OpCall};
use objc2_metal::{MTLComputeCommandEncoder, MTLSize};

fn t(buf: &Buffer, off: usize, dt: DType, shape: &[usize]) -> TensorRef {
    TensorRef { buf: buf.clone(), offset: off, dtype: dt, shape: shape.to_vec() }
}

/// `data` at byte offset `off` of a fresh buffer; the rest is 0xEE.
fn upload_at(be: &MetalBackend, off: usize, data: &[u16]) -> Buffer {
    let b = be.alloc(off + data.len() * 2 + 16, MemKind::Device).unwrap();
    be.upload(&b, 0, &vec![0xEEu8; off + data.len() * 2 + 16], 0).unwrap();
    be.upload(&b, off, as_bytes(data), 0).unwrap();
    b
}

fn read_u16(be: &MetalBackend, b: &Buffer, off: usize, n: usize) -> Vec<u16> {
    let mut raw = vec![0u8; n * 2];
    be.download(b, off, &mut raw, 0).unwrap();
    raw.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect()
}

fn out_buf(be: &MetalBackend, off: usize, n: usize) -> Buffer {
    let b = be.alloc(off + n * 2 + 16, MemKind::Device).unwrap();
    be.upload(&b, 0, &vec![0xEEu8; off + n * 2 + 16], 0).unwrap();
    b
}

fn assert_guards(be: &MetalBackend, b: &Buffer, off: usize, n: usize, what: &str) {
    let raw = read_u16(be, b, 0, (off + n * 2 + 16) / 2);
    assert!(raw[..off / 2].iter().all(|v| *v == 0xEEEE), "{what}: stray store before");
    assert!(raw[off / 2 + n..].iter().all(|v| *v == 0xEEEE), "{what}: stray store after");
}

/// f64 rmsnorm of one row of already-bf16-rounded values, with the oracle's bf16 rounding of the result.
fn want_row(x: &[f32], w: &[u16], eps: f32, w_off: f32) -> Vec<f32> {
    let ss: f64 = x.iter().map(|v| (*v as f64) * (*v as f64)).sum();
    let r = 1.0 / (ss / x.len() as f64 + eps as f64).sqrt();
    x.iter().zip(w).map(|(v, wv)| (*v as f64 * r * (bf16_to_f32(*wv) as f64 + w_off as f64)) as f32).collect()
}

fn close(got: u16, want: f32, what: &str) {
    let g = bf16_to_f32(got);
    // half a bf16 ulp of the result plus a hair for the f32 evaluation of the same expression
    assert!((g - want).abs() <= want.abs() * 2f32.powi(-8) * 1.02 + 1e-30, "{what}: got {g} want {want}");
}

/// Offsets (bytes) of x, w, out: aligned (vector path), x 2 mod 4 (skip, scalar path), 8 mod 16 (scalar path).
const OFFS: [(usize, usize, usize); 4] = [(0, 0, 0), (0, 2, 0), (2, 0, 0), (8, 6, 8)];

fn run_rmsnorm(h: usize, rows: usize, qk: bool, seed: u64) {
    let be = backend();
    for (xo, wo, oo) in OFFS {
        let x = rand_bf16(rows * h, seed);
        let w = rand_bf16(h, seed + 1);
        let (xb, wb) = (upload_at(&be, xo, &x), upload_at(&be, wo, &w));
        let out = out_buf(&be, oo, rows * h);
        let (op, kernel, shape) =
            if qk { (Op::QkNorm, QK_NORM_BF16, vec![1, rows, h]) } else { (Op::RmsNorm, RMSNORM_BF16, vec![rows, h]) };
        let call = OpCall::new(
            op,
            vec![t(&xb, xo, DType::Bf16, &shape), t(&wb, wo, DType::Bf16, &[h])],
            vec![t(&out, oo, DType::Bf16, &shape)],
            OpAttrs { eps: 1e-6, w_offset: 0.0, ..Default::default() },
        );
        be.launch(kernel, &call, 0).unwrap();
        let got = read_u16(&be, &out, oo, rows * h);
        assert_guards(&be, &out, oo, rows * h, "rmsnorm");
        for r in 0..rows {
            let xr: Vec<f32> = x[r * h..(r + 1) * h].iter().map(|v| bf16_to_f32(*v)).collect();
            let want = want_row(&xr, &w, 1e-6, 0.0);
            for i in 0..h {
                close(got[r * h + i], want[i], &format!("rmsnorm H={h} rows={rows} offs=({xo},{wo},{oo}) [{r},{i}]"));
            }
        }
    }
}

fn run_rmsnorm_add(h: usize, rows: usize, seed: u64) {
    let be = backend();
    for (xo, wo, oo) in OFFS {
        let x = rand_bf16(rows * h, seed);
        let res = rand_bf16(rows * h, seed + 2);
        let w = rand_bf16(h, seed + 1);
        let (xb, rb, wb) = (upload_at(&be, xo, &x), upload_at(&be, xo, &res), upload_at(&be, wo, &w));
        let (normed, sum) = (out_buf(&be, oo, rows * h), out_buf(&be, oo, rows * h));
        let call = OpCall::new(
            Op::RmsNormAdd,
            vec![
                t(&xb, xo, DType::Bf16, &[rows, h]),
                t(&rb, xo, DType::Bf16, &[rows, h]),
                t(&wb, wo, DType::Bf16, &[h]),
            ],
            vec![t(&normed, oo, DType::Bf16, &[rows, h]), t(&sum, oo, DType::Bf16, &[rows, h])],
            OpAttrs { eps: 1e-6, w_offset: 0.0, ..Default::default() },
        );
        be.launch(RMSNORM_ADD_BF16, &call, 0).unwrap();
        let (gn, gs) = (read_u16(&be, &normed, oo, rows * h), read_u16(&be, &sum, oo, rows * h));
        assert_guards(&be, &normed, oo, rows * h, "normed");
        assert_guards(&be, &sum, oo, rows * h, "sum");
        for r in 0..rows {
            let s: Vec<u16> =
                (0..h).map(|i| f32_to_bf16(bf16_to_f32(x[r * h + i]) + bf16_to_f32(res[r * h + i]))).collect();
            assert_eq!(&gs[r * h..(r + 1) * h], &s[..], "sum H={h} offs=({xo},{wo},{oo}) row {r}");
            let sf: Vec<f32> = s.iter().map(|v| bf16_to_f32(*v)).collect();
            let want = want_row(&sf, &w, 1e-6, 0.0);
            for i in 0..h {
                close(
                    gn[r * h + i],
                    want[i],
                    &format!("rmsnorm_add H={h} rows={rows} offs=({xo},{wo},{oo}) [{r},{i}]"),
                );
            }
        }
    }
}

/// Hidden / head sizes: 128 (qk-norm), 4096 (Qwen3-8B), 5120, 2560, 4104, 12288 (> 8192: the 256-thread kernel), 32768,
/// 40960, 100 and 4097 (odd).
const NORM_H: [usize; 10] = [128, 4096, 5120, 2560, 4104, 12288, 32768, 40960, 100, 4097];

#[test]
fn rmsnorm_vector_and_scalar_paths_match_reference() {
    for (i, h) in NORM_H.into_iter().enumerate() {
        run_rmsnorm(h, 3, false, 100 + i as u64);
    }
}

#[test]
fn qk_norm_vector_path_matches_reference() {
    for (i, h) in [64usize, 128, 256].into_iter().enumerate() {
        run_rmsnorm(h, 40, true, 200 + i as u64);
    }
}

#[test]
fn rmsnorm_add_vector_and_scalar_paths_match_reference() {
    for (i, h) in NORM_H.into_iter().enumerate() {
        run_rmsnorm_add(h, 2, 300 + i as u64);
    }
}

#[test]
fn argmax_vector_path_matches_host() {
    let be = backend();
    let mut rng = Rng(99);
    // 151936 = Qwen3 vocab (vector path, 1024 threads); 1000 and 4 small; 1001 and an offset input take the scalar kernel
    for (rows, v, off) in
        [(1usize, 151936usize, 0usize), (2, 151936, 0), (3, 1000, 0), (1, 4, 0), (1, 1001, 0), (1, 1000, 4)]
    {
        let mut flat: Vec<f32> = (0..rows * v).map(|_| rng.unit()).collect();
        // a tie of the maximum in the last row, at positions straddling thread and chunk boundaries
        let (a, b) = (v / 3, v - 2);
        flat[(rows - 1) * v + a] = 9.0;
        flat[(rows - 1) * v + b] = 9.0;
        let lb = be.alloc(off + flat.len() * 4, MemKind::Device).unwrap();
        // SAFETY: f32 -> u8 view of the same memory.
        be.upload(&lb, off, unsafe { std::slice::from_raw_parts(flat.as_ptr() as *const u8, flat.len() * 4) }, 0)
            .unwrap();
        let ob = be.alloc(rows * 4, MemKind::Device).unwrap();
        let call = OpCall::new(
            Op::Argmax,
            vec![t(&lb, off, DType::F32, &[rows, v])],
            vec![t(&ob, 0, DType::I32, &[rows])],
            OpAttrs::default(),
        );
        be.launch(ARGMAX_F32, &call, 0).unwrap();
        let mut raw = vec![0u8; rows * 4];
        be.download(&ob, 0, &mut raw, 0).unwrap();
        let got: Vec<i32> = raw.chunks_exact(4).map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
        let want: Vec<i32> = flat
            .chunks(v)
            .map(|r| {
                let m = r.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                r.iter().position(|x| *x == m).unwrap() as i32
            })
            .collect();
        assert_eq!(got, want, "rows={rows} v={v} off={off}");
    }
}

/// OpP of kernels/metal/common.h (64 bytes).
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct OpP {
    dims: [u32; 8],
    skips: [u32; 5],
    f0: f32,
    f1: f32,
    pad: u32,
}

/// Dispatch shader `name` by hand: one threadgroup of 256 threads per row, buffers at offset 0, then `p`.
fn dispatch_rows(be: &MetalBackend, name: &str, bufs: &[&Buffer], p: &OpP, rows: usize) {
    let pso = be.pso(name).unwrap();
    let mtls: Vec<&_> = bufs.iter().map(|b| b.downcast::<MetalBuffer>().unwrap().mtl()).collect();
    be.with_encoder(|enc| {
        enc.setComputePipelineState(pso);
        for (i, m) in mtls.iter().enumerate() {
            // SAFETY: the buffers cover the tensors.
            unsafe { enc.setBuffer_offset_atIndex(Some(*m), 0, i) };
        }
        set_bytes(enc, p, bufs.len());
        enc.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize { width: rows, height: 1, depth: 1 },
            MTLSize { width: 256, height: 1, depth: 1 },
        );
    })
    .unwrap();
}

/// The low-latency norms keep the 256-thread kernels' element-to-thread assignment and reduction order, so their
/// results are bitwise the same (what keeps greedy decoding token-identical to the previous kernels).
#[test]
fn low_latency_norms_are_bitwise_the_256_thread_kernels() {
    let be = backend();
    for (i, h) in [1usize, 31, 100, 128, 255, 256, 257, 1000, 4096, 5120, 8192, 8200, 12288].into_iter().enumerate() {
        let rows = 3;
        let (x, res, w) = (rand_bf16(rows * h, 500 + i as u64), rand_bf16(rows * h, 600 + i as u64), rand_bf16(h, 700));
        let (xb, rb, wb) = (upload_at(&be, 0, &x), upload_at(&be, 0, &res), upload_at(&be, 0, &w));
        let p = OpP { dims: [h as u32, 0, 0, 0, 0, 0, 0, 0], f0: 1e-6, f1: 0.0, ..Default::default() };
        // RmsNorm
        let (got, want) = (out_buf(&be, 0, rows * h), out_buf(&be, 0, rows * h));
        let call = OpCall::new(
            Op::RmsNorm,
            vec![t(&xb, 0, DType::Bf16, &[rows, h]), t(&wb, 0, DType::Bf16, &[h])],
            vec![t(&got, 0, DType::Bf16, &[rows, h])],
            OpAttrs { eps: 1e-6, ..Default::default() },
        );
        be.launch(RMSNORM_BF16, &call, 0).unwrap();
        dispatch_rows(&be, "rmsnorm_bf16", &[&xb, &wb, &want], &p, rows);
        assert_eq!(read_u16(&be, &got, 0, rows * h), read_u16(&be, &want, 0, rows * h), "rmsnorm H={h}");
        // RmsNormAdd
        let (gn, gs, wn, ws) = (
            out_buf(&be, 0, rows * h),
            out_buf(&be, 0, rows * h),
            out_buf(&be, 0, rows * h),
            out_buf(&be, 0, rows * h),
        );
        let call = OpCall::new(
            Op::RmsNormAdd,
            vec![t(&xb, 0, DType::Bf16, &[rows, h]), t(&rb, 0, DType::Bf16, &[rows, h]), t(&wb, 0, DType::Bf16, &[h])],
            vec![t(&gn, 0, DType::Bf16, &[rows, h]), t(&gs, 0, DType::Bf16, &[rows, h])],
            OpAttrs { eps: 1e-6, ..Default::default() },
        );
        be.launch(RMSNORM_ADD_BF16, &call, 0).unwrap();
        dispatch_rows(&be, "rmsnorm_add_bf16", &[&xb, &rb, &wb, &wn, &ws], &p, rows);
        assert_eq!(read_u16(&be, &gn, 0, rows * h), read_u16(&be, &wn, 0, rows * h), "rmsnorm_add normed H={h}");
        assert_eq!(read_u16(&be, &gs, 0, rows * h), read_u16(&be, &ws, 0, rows * h), "rmsnorm_add sum H={h}");
    }
}
