//! Kernel table and device dispatch of the P12 (lossless 12-bit bf16, `intarsia_core::quant::p12`) GEMVs on HIP (lane ink-hip-p12,
//! docs/results/2026-11-26-inkling2-hip-p12.md): `MatmulP12` and the P12 form of `InkInProj` (u8 codes as the weight). Kernels in
//! kernels/hip/ink_p12.hip, ABI kernels/hip/ik_hip_abi_inkp12.h. Each is its HIP bf16 twin's arithmetic on the decoded weight, bit
//! for bit (tests/ink_p12.rs), and its registration mirrors the twin's, so P12 on selects a P12 kernel exactly where P12 off runs
//! the twin:
//!
//! * `matmul_p12_gemv` / `matmul_p12_logits_gemv` (bf16 / f32 out) == `matmul_bf16_gemv` / `logits_bf16_gemv` (kernels.rs), ordinary
//!   calls of 1..=4 rows. The bf16 Matmul twins step aside when the WMMA table takes bf16 at that row count (`wmma_takes`), so
//!   these do too.
//! * `_rows`: the `rows_exact` registrations, == `matmul_bf16_gemv_rows` (1..=16 rows) / `logits_bf16_gemv_rows` (1..=8 rows:
//!   `LOGITS_ROWS_MAX`; a 9..=16-row head call keeps the bf16 kernel, faster there).
//! * `ink_in_proj_p12_gemv[_rows]` == `ink_in_proj_bf16_gemv[_rows]` (inkmix.rs), 1..=16 rows.
//!
//! The model makes the P12 copies when `p12::decide` says so for "hip" and uses them where these kernels are selected.
//!
//! An op of `OPS` counts as ported once `TABLE` lists it (`kernels::unimplemented`).

use crate::kernels::ARCH;
use crate::quantmoe::{wmma_takes, WmmaKind};
use intarsia_core::quant::p12::ROWS_MAX;
use intarsia_core::{DType, Exactness, KernelImpl, Op, ShapeInfo, Tol};

pub const TABLE: &[(&str, Op)] = &[
    ("matmul_p12_gemv", Op::MatmulP12),
    ("matmul_p12_gemv_rows", Op::MatmulP12),
    ("matmul_p12_logits_gemv", Op::MatmulP12),
    ("matmul_p12_logits_gemv_rows", Op::MatmulP12),
    ("ink_in_proj_p12_gemv", Op::InkInProj),
    ("ink_in_proj_p12_gemv_rows", Op::InkInProj),
];

/// The ops this module is the home of (`InkInProj` is inkmix's; this module adds its P12 form).
pub const OPS: &[Op] = &[Op::MatmulP12];

const BF16_TOL: Tol = Tol::new(1.5e-2, 6e-3);
const F32_TOL: Tol = Tol::new(1e-3, 2e-4);

/// Most rows of `matmul_p12_logits_gemv_rows` (the unembed's rows twin).
pub const LOGITS_ROWS_MAX: usize = 8;

/// 1..=16 activation rows, K a multiple of 16 (the launchers' 8-element chunks in pairs), at most 65536 (the format).
fn rows_ok(s: &ShapeInfo) -> bool {
    (1..=ROWS_MAX).contains(&s.m) && s.k % 16 == 0 && s.k <= 65536
}

/// Where the bf16 Matmul GEMVs run (kernels.rs `matmul_bf16_gemv[_rows]`): not where the WMMA table takes bf16.
fn bf16_gemv_row(s: &ShapeInfo) -> bool {
    !wmma_takes(WmmaKind::Bf16, s.m)
}

pub fn all() -> Vec<KernelImpl> {
    TABLE
        .iter()
        .map(|(name, op)| {
            let k = KernelImpl::new(*op, *name, "hip").arch([ARCH]).acts([DType::Bf16]).weights([DType::U8]).exact(Exactness::Deterministic);
            match *name {
                "matmul_p12_gemv" => k.when(|s| rows_ok(s) && s.m <= 4 && bf16_gemv_row(s) && s.out == Some(DType::Bf16)).tol(BF16_TOL).priority(10),
                "matmul_p12_gemv_rows" => {
                    k.when(|s| rows_ok(s) && bf16_gemv_row(s) && s.out == Some(DType::Bf16)).tol(BF16_TOL).priority(-1).rows_exact()
                }
                "matmul_p12_logits_gemv" => k.when(|s| rows_ok(s) && s.m <= 4 && s.out == Some(DType::F32)).tol(F32_TOL).priority(10),
                // up to LOGITS_ROWS_MAX rows: at 16 rows the unembed's P12 kernel is slower than the bf16 one (window 1: 2109 against
                // 1884 us, ALU bound), so 9..=16-row head calls keep the bf16 rows kernel (the same bits either way)
                "matmul_p12_logits_gemv_rows" => {
                    k.when(|s| rows_ok(s) && s.m <= LOGITS_ROWS_MAX && s.out == Some(DType::F32)).tol(F32_TOL).priority(-1).rows_exact()
                }
                "ink_in_proj_p12_gemv" => k.when(rows_ok).tol(BF16_TOL).priority(10),
                "ink_in_proj_p12_gemv_rows" => k.when(rows_ok).tol(BF16_TOL).priority(-1).rows_exact(),
                other => unreachable!("inkp12 kernel {other} has no registration"),
            }
        })
        .collect()
}

/// Tests / benches: the launch shape of the P12 GEMVs (`ik_hip_p12_set`; every value gives the same bits). `gemv` = (rows per wave,
/// chunk loads in flight per lane) of the 1..=8-row MatmulP12 kernel, `None` the default (1, 16 at one row, 1, 4 at 2..=4, 2, 4 at
/// 5..=8); `in_proj_u` 0 = the bf16 twin's stride loop, 4 / 8 / 9 = that many chunk loads in flight per lane, -1 the default (9 at
/// 1..=4 rows when K / 8 is a multiple of 288, else the stride loop).
pub fn set_p12_shape(gemv: Option<(i32, i32)>, in_proj_u: i32) {
    #[cfg(ik_hip)]
    {
        extern "C" {
            fn ik_hip_p12_set(r: std::ffi::c_int, u: std::ffi::c_int, ip: std::ffi::c_int);
        }
        let (r, u) = gemv.unwrap_or((-1, -1));
        // SAFETY: sets three plain ints of the kernel library.
        unsafe { ik_hip_p12_set(r, u, in_proj_u) }
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
        fn ik_matmul_p12_gemv(x: T, codes: T, aux: T, out: TM, s: S) -> c_int;
        fn ik_matmul_p12_gemv_rows(x: T, codes: T, aux: T, out: TM, s: S) -> c_int;
        fn ik_matmul_p12_logits_gemv(x: T, codes: T, aux: T, out: TM, s: S) -> c_int;
        fn ik_matmul_p12_logits_gemv_rows(x: T, codes: T, aux: T, out: TM, s: S) -> c_int;
        fn ik_ink_in_proj_p12_gemv(x: T, codes: T, aux: T, pq: TM, pk: TM, pv: TM, s: S) -> c_int;
        fn ik_ink_in_proj_p12_gemv_rows(x: T, codes: T, aux: T, pq: TM, pk: TM, pv: TM, s: S) -> c_int;
    }

    /// The launcher status when `kernel` is this module's, `None` otherwise. Called from `HipBackend::launch` like the other
    /// Inkling-2 lane modules' (`a.i(n)` input n, `a.o(n)` output n, `a.s` stream).
    ///
    /// # Safety
    /// As the launchers of `ffi`: the tensors are bounds-checked views of live device buffers held by `a.call`; the launchers
    /// validate dtype / rank / shape and return an error status instead of faulting.
    pub(crate) unsafe fn launch(kernel: &str, a: &InkArgs) -> Option<c_int> {
        Some(match kernel {
            "matmul_p12_gemv" => ik_matmul_p12_gemv(a.i(0), a.i(1), a.i(2), a.o(0), a.s),
            "matmul_p12_gemv_rows" => ik_matmul_p12_gemv_rows(a.i(0), a.i(1), a.i(2), a.o(0), a.s),
            "matmul_p12_logits_gemv" => ik_matmul_p12_logits_gemv(a.i(0), a.i(1), a.i(2), a.o(0), a.s),
            "matmul_p12_logits_gemv_rows" => ik_matmul_p12_logits_gemv_rows(a.i(0), a.i(1), a.i(2), a.o(0), a.s),
            "ink_in_proj_p12_gemv" => ik_ink_in_proj_p12_gemv(a.i(0), a.i(1), a.i(2), a.o(0), a.o(1), a.o(2), a.s),
            "ink_in_proj_p12_gemv_rows" => ik_ink_in_proj_p12_gemv_rows(a.i(0), a.i(1), a.i(2), a.o(0), a.o(1), a.o(2), a.s),
            _ => return None,
        })
    }
}
