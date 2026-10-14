//! The kernel table: which Metal kernel implements which op, for which shapes, and how closely it must match the
//! reference. Mirrors `intarsia-backend-hip/src/kernels.rs`. The shader names are the `host_name`s in
//! kernels/metal/*.metal.

use intarsia_core::{DType, Exactness, KernelImpl, Op, Tol};

/// bf16-rounded outputs: a one-ulp flip is allowed (same budget as the cpu and hip backends).
const BF16_TOL: Tol = Tol::new(1.5e-2, 6e-3);

/// `KernelImpl::name` of the batch-1 bf16 GEMV (the HIP backend calls the same thing `matmul_bf16_gemv`).
pub const MATMUL_BF16_GEMV: &str = "matmul_bf16_gemv";

/// A vectorised GEMV shader variant (an instantiation of `gemv_bf16_t<ROWS, UNROLL, SG>`).
#[derive(Clone, Copy, Debug)]
pub struct GemvVariant {
    pub name: &'static str,
    /// Output rows per simdgroup.
    pub rows: u32,
    /// 16-byte weight chunks per row each lane loads per step.
    pub unroll: u32,
    /// Simdgroups (32 lanes) per threadgroup.
    pub sg: u32,
}

impl GemvVariant {
    /// Output rows per threadgroup.
    pub const fn rows_per_tg(&self) -> u32 {
        self.rows * self.sg
    }
    pub const fn threads_per_tg(&self) -> u32 {
        self.sg * 32
    }
}

macro_rules! variant {
    ($r:literal, $u:literal, $s:literal) => {
        GemvVariant { name: concat!("gemv_bf16_r", $r, "u", $u, "s", $s), rows: $r, unroll: $u, sg: $s }
    };
}

/// Every vectorised variant compiled into the metallib (needs K % 8 == 0 and 16-byte aligned W, x, y).
pub const GEMV_VARIANTS: &[GemvVariant] = &[
    variant!(1, 1, 8),
    variant!(1, 2, 8),
    variant!(1, 4, 4),
    variant!(1, 4, 8),
    variant!(1, 4, 16),
    variant!(1, 8, 8),
    variant!(2, 1, 4),
    variant!(2, 2, 4),
    variant!(2, 4, 4),
    variant!(4, 1, 4),
    variant!(4, 2, 4),
    variant!(8, 1, 2),
    variant!(8, 2, 2),
];

/// The variant `matmul_bf16_gemv` launches when the operands allow vector loads (chosen from the spike sweep, raw log
/// docs/results/raw/2026-10-14-metal-spike/gemv_bw.log: one row per simdgroup, four 16-byte loads per lane in flight).
pub const DEFAULT_GEMV: GemvVariant = variant!(1, 4, 4);

/// 8-byte-load variants (`gemv_bf16_a8_t<UNROLL, SG>`, one row per simdgroup) for K % 4 == 0 and operands that are
/// 8-byte but not 16-byte aligned, e.g. every weight of a safetensors shard whose header length is 8 mod 16.
pub const GEMV_A8_VARIANTS: &[GemvVariant] = &[
    GemvVariant { name: "gemv_bf16_a8u4s4", rows: 1, unroll: 4, sg: 4 },
    GemvVariant { name: "gemv_bf16_a8u8s4", rows: 1, unroll: 8, sg: 4 },
    GemvVariant { name: "gemv_bf16_a8u8s8", rows: 1, unroll: 8, sg: 8 },
];
pub const DEFAULT_GEMV_A8: GemvVariant = GEMV_A8_VARIANTS[1];

/// Fallback for any K and any 2-byte alignment: one simdgroup per row, 8 rows per threadgroup, 2-byte loads.
pub const GEMV_SCALAR: GemvVariant = GemvVariant { name: "gemv_bf16_scalar", rows: 1, unroll: 1, sg: 8 };

pub const COPY_U4: &str = "copy_u4";
pub const READ_XOR_U4: &str = "read_xor_u4";
/// 16-byte vectors each copy/read thread handles (`PER_THREAD` in copy.metal).
pub const COPY_PER_THREAD: usize = 4;
/// Same for `read_xor_u4` (`READ_PER_THREAD`).
pub const READ_PER_THREAD: usize = 16;

/// Every shader entry point the backend builds a pipeline for.
pub fn shader_names() -> Vec<&'static str> {
    let mut v: Vec<&'static str> = GEMV_VARIANTS.iter().map(|g| g.name).collect();
    v.extend(GEMV_A8_VARIANTS.iter().map(|g| g.name));
    v.extend([GEMV_SCALAR.name, COPY_U4, READ_XOR_U4]);
    v
}

/// Every kernel implementation this backend registers.
pub fn all() -> Vec<KernelImpl> {
    vec![KernelImpl::new(Op::Matmul, MATMUL_BF16_GEMV, "metal")
        .acts([DType::Bf16])
        .weights([DType::Bf16])
        .when(|s| s.m == 1 && s.out.is_none_or(|d| d == DType::Bf16))
        .exact(Exactness::Deterministic)
        .tol(BF16_TOL)
        .priority(10)]
}
