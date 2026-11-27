//! Kernel table and device dispatch of the int8 per-row GEMVs on HIP (lane ink-hip-i8, docs/results/2026-11-26-inkling2-hip-i8.md):
//! `MatmulI8` and the i8 form of `InkInProj` (i8 codes as the weight, scales `[N, 1]` f32 as the third input). Opt-in and lossy
//! (`INTARSIA_INK_ATTN_QUANT=int8_row` / `INTARSIA_LMHEAD_QUANT=int8_row`). Kernels in kernels/hip/ink_i8.hip, ABI
//! kernels/hip/ik_hip_abi_inki8.h.
//!
//! * `matmul_i8_gemv` / `matmul_i8_logits_gemv` (bf16 / f32 out): ordinary calls of 1..=4 rows;
//! * `_rows`: the `rows_exact` registrations, 1..=16 rows;
//! * `ink_in_proj_i8_gemv[_rows]`: the in-projection, 1..=16 rows.
//!
//! Calls with more rows (prefill chunks) have no i8 kernel: the model keeps the bf16 weight `bf16(q * scale)` for them.
//!
//! An op of `OPS` counts as ported once `TABLE` lists it (`kernels::unimplemented`).

use crate::kernels::ARCH;
use intarsia_core::quant::i8row::ROWS_MAX;
use intarsia_core::{DType, Exactness, KernelImpl, Op, ShapeInfo, Tol};

pub const TABLE: &[(&str, Op)] = &[
    ("matmul_i8_gemv", Op::MatmulI8),
    ("matmul_i8_gemv_rows", Op::MatmulI8),
    ("matmul_i8_logits_gemv", Op::MatmulI8),
    ("matmul_i8_logits_gemv_rows", Op::MatmulI8),
    ("ink_in_proj_i8_gemv", Op::InkInProj),
    ("ink_in_proj_i8_gemv_rows", Op::InkInProj),
];

/// The ops this module is the home of (`InkInProj` is inkmix's; this module adds its i8 form).
pub const OPS: &[Op] = &[Op::MatmulI8];

const BF16_TOL: Tol = Tol::new(1.5e-2, 6e-3);
const F32_TOL: Tol = Tol::new(1e-3, 2e-4);

/// 1..=16 activation rows, K a multiple of 16, at most 65536.
fn rows_ok(s: &ShapeInfo) -> bool {
    (1..=ROWS_MAX).contains(&s.m) && s.k % 16 == 0 && s.k <= 65536
}

pub fn all() -> Vec<KernelImpl> {
    TABLE
        .iter()
        .map(|(name, op)| {
            let k = KernelImpl::new(*op, *name, "hip").arch([ARCH]).acts([DType::Bf16]).weights([DType::I8]).exact(Exactness::Deterministic);
            match *name {
                "matmul_i8_gemv" => k.when(|s| rows_ok(s) && s.m <= 4 && s.out == Some(DType::Bf16)).tol(BF16_TOL).priority(10),
                "matmul_i8_gemv_rows" => k.when(|s| rows_ok(s) && s.out == Some(DType::Bf16)).tol(BF16_TOL).priority(-1).rows_exact(),
                "matmul_i8_logits_gemv" => k.when(|s| rows_ok(s) && s.m <= 4 && s.out == Some(DType::F32)).tol(F32_TOL).priority(10),
                "matmul_i8_logits_gemv_rows" => k.when(|s| rows_ok(s) && s.out == Some(DType::F32)).tol(F32_TOL).priority(-1).rows_exact(),
                "ink_in_proj_i8_gemv" => k.when(rows_ok).tol(BF16_TOL).priority(10),
                "ink_in_proj_i8_gemv_rows" => k.when(rows_ok).tol(BF16_TOL).priority(-1).rows_exact(),
                other => unreachable!("inki8 kernel {other} has no registration"),
            }
        })
        .collect()
}

/// Tests / benches: the launch shape (`ik_hip_i8_set`). `gemv` = (rows per wave, chunk loads in flight per lane) of the `MatmulI8`
/// kernel, `None` the default; `in_proj_u` the in-projection's chunk loads in flight (1..=5), <= 0 the default.
pub fn set_i8_shape(gemv: Option<(i32, i32)>, in_proj_u: i32) {
    #[cfg(ik_hip)]
    {
        extern "C" {
            fn ik_hip_i8_set(r: std::ffi::c_int, u: std::ffi::c_int, ip: std::ffi::c_int);
        }
        let (r, u) = gemv.unwrap_or((-1, -1));
        // SAFETY: sets three plain ints of the kernel library.
        unsafe { ik_hip_i8_set(r, u, in_proj_u) }
    }
    #[cfg(not(ik_hip))]
    let _ = (gemv, in_proj_u);
}

#[cfg(ik_hip)]
pub(crate) mod launch {
    use crate::device::InkArgs;
    use crate::ffi::IkTensor;
    use std::ffi::{c_int, c_void};
    type T = *const IkTensor;
    type TM = *mut IkTensor;
    type S = *mut c_void;
    extern "C" {
        fn ik_matmul_i8_gemv(x: T, q: T, sc: T, out: TM, s: S) -> c_int;
        fn ik_matmul_i8_gemv_rows(x: T, q: T, sc: T, out: TM, s: S) -> c_int;
        fn ik_matmul_i8_logits_gemv(x: T, q: T, sc: T, out: TM, s: S) -> c_int;
        fn ik_matmul_i8_logits_gemv_rows(x: T, q: T, sc: T, out: TM, s: S) -> c_int;
        fn ik_ink_in_proj_i8_gemv(x: T, q: T, sc: T, pq: TM, pk: TM, pv: TM, s: S) -> c_int;
        fn ik_ink_in_proj_i8_gemv_rows(x: T, q: T, sc: T, pq: TM, pk: TM, pv: TM, s: S) -> c_int;
    }

    /// The launcher status when `kernel` is this module's, `None` otherwise (called from `HipBackend::launch` like the other
    /// Inkling-2 lane modules').
    ///
    /// # Safety
    /// As the launchers of `ffi`: the tensors are bounds-checked views of live device buffers held by `a.call`; the launchers
    /// validate dtype / rank / shape and return an error status instead of faulting.
    pub(crate) unsafe fn launch(kernel: &str, a: &InkArgs) -> Option<c_int> {
        Some(match kernel {
            "matmul_i8_gemv" => ik_matmul_i8_gemv(a.i(0), a.i(1), a.i(2), a.o(0), a.s),
            "matmul_i8_gemv_rows" => ik_matmul_i8_gemv_rows(a.i(0), a.i(1), a.i(2), a.o(0), a.s),
            "matmul_i8_logits_gemv" => ik_matmul_i8_logits_gemv(a.i(0), a.i(1), a.i(2), a.o(0), a.s),
            "matmul_i8_logits_gemv_rows" => ik_matmul_i8_logits_gemv_rows(a.i(0), a.i(1), a.i(2), a.o(0), a.s),
            "ink_in_proj_i8_gemv" => ik_ink_in_proj_i8_gemv(a.i(0), a.i(1), a.i(2), a.o(0), a.o(1), a.o(2), a.s),
            "ink_in_proj_i8_gemv_rows" => ik_ink_in_proj_i8_gemv_rows(a.i(0), a.i(1), a.i(2), a.o(0), a.o(1), a.o(2), a.s),
            _ => return None,
        })
    }
}
