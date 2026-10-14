//! NAX (M5 neural accelerator) bf16 GEMM launcher: `Y[M,N] = X[M,K] . W[N,K]^T`, kernels/metal/gemm_nax_bf16.metal,
//! compiled `-std=metal4.0` into `intarsia_metal4.metallib` (build.rs `METAL4_SOURCES`).
//!
//! Tile configuration is a property of the shader entry point and the same for every M (a row's result must not
//! depend on the call it arrives in). The precision mode (`relaxed_precision` of `matmul2d_descriptor`) is part of the
//! entry point too.

use crate::{set_bytes, MetalBackend};
use intarsia_core::{DType, Error, OpCall, Result};
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLComputeCommandEncoder, MTLComputePipelineState, MTLSize};

/// One compiled configuration of `gemm_nax_t`.
#[derive(Clone, Copy, Debug)]
pub struct NaxVariant {
    /// Shader entry point (`kernels::metal4_shaders()`).
    pub shader: &'static str,
    /// Output tile per threadgroup.
    pub bm: u32,
    pub bn: u32,
    /// Simdgroups per threadgroup.
    pub sg: u32,
    /// `matmul2d_descriptor::relaxed_precision`.
    pub relaxed: bool,
}

impl NaxVariant {
    pub const fn threads(&self) -> u32 {
        self.sg * 32
    }
}

macro_rules! v {
    ($name:literal, $bm:literal, $bn:literal, $sg:literal, $r:literal) => {
        NaxVariant { shader: $name, bm: $bm, bn: $bn, sg: $sg, relaxed: $r }
    };
}

/// Every `gemm_nax_t` instantiation (the `NAX_T` lines of gemm_nax_bf16.metal). All of them give bitwise the same
/// output for the same operands (the K reduction order does not depend on the tile), strict and relaxed alike.
pub const NAX_VARIANTS: &[NaxVariant] = &[
    v!("gemm_nax_t64x64w4", 64, 64, 4, false),
    v!("gemm_nax_t128x64w4", 128, 64, 4, false),
    v!("gemm_nax_t64x128w4", 64, 128, 4, false),
    v!("gemm_nax_t128x128w8", 128, 128, 8, false),
    v!("gemm_nax_t64x64w4_r", 64, 64, 4, true),
    v!("gemm_nax_t128x64w4_r", 128, 64, 4, true),
    v!("gemm_nax_t64x128w4_r", 64, 128, 4, true),
    v!("gemm_nax_t128x128w8_r", 128, 128, 8, true),
];

/// Kernel arguments (`NaxP` in gemm_nax_bf16.metal; 32 bytes).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct NaxParams {
    pub m: u32,
    pub n: u32,
    pub k: u32,
    /// Threadgroup-index swizzle: `1 << swz_log` consecutive threadgroups along M share an N tile.
    pub swz_log: u32,
    /// Leading bf16 elements (0 or 1) of A, W, C past their 4-byte aligned binding.
    pub s0: u32,
    pub s1: u32,
    pub s2: u32,
    pub pad: u32,
}

/// Threadgroups `(x, y)` for `v` over an `m` by `n` output.
pub fn grid(v: &NaxVariant, m: u32, n: u32, swz_log: u32) -> (usize, usize) {
    let tiles_m = m.div_ceil(v.bm);
    let tiles_n = n.div_ceil(v.bn);
    ((tiles_n << swz_log) as usize, tiles_m.div_ceil(1 << swz_log) as usize)
}

/// Encode one dispatch. Each operand is `(buffer, 4-byte aligned byte offset)`; `p.s0/s1/s2` carry the leftover element.
pub fn encode_nax(
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    pso: &ProtocolObject<dyn MTLComputePipelineState>,
    v: &NaxVariant,
    a: (&ProtocolObject<dyn MTLBuffer>, usize),
    w: (&ProtocolObject<dyn MTLBuffer>, usize),
    c: (&ProtocolObject<dyn MTLBuffer>, usize),
    p: &NaxParams,
) {
    enc.setComputePipelineState(pso);
    // SAFETY: offsets are inside the buffers (the callers check) and 4-byte aligned.
    unsafe {
        enc.setBuffer_offset_atIndex(Some(a.0), a.1, 0);
        enc.setBuffer_offset_atIndex(Some(w.0), w.1, 1);
        enc.setBuffer_offset_atIndex(Some(c.0), c.1, 2);
    }
    set_bytes(enc, p, 3);
    let (gx, gy) = grid(v, p.m, p.n, p.swz_log);
    enc.dispatchThreadgroups_threadsPerThreadgroup(
        MTLSize { width: gx, height: gy, depth: 1 },
        MTLSize { width: v.threads() as usize, height: 1, depth: 1 },
    );
}

/// Variant a registered kernel launches.
pub fn variant_for(kernel: &str) -> Option<&'static NaxVariant> {
    let shader = match kernel {
        crate::kernels::MATMUL_BF16_NAX => crate::kernels::NAX_DEFAULT,
        crate::kernels::MATMUL_BF16_NAX_RELAXED => crate::kernels::NAX_DEFAULT_RELAXED,
        _ => return None,
    };
    NAX_VARIANTS.iter().find(|v| v.shader == shader)
}

impl MetalBackend {
    /// Launch `matmul_bf16_nax[_relaxed]` for an `Op::Matmul` call (x `[M,K]`, w `[N,K]`, y `[M,N]`, all bf16).
    pub(crate) fn launch_nax(&self, kernel: &str, call: &OpCall) -> Result<()> {
        let v = variant_for(kernel).ok_or_else(|| Error::NoKernel(format!("metal kernel '{kernel}'")))?;
        let pso = self
            .pso(v.shader)
            .ok_or_else(|| Error::backend(format!("metal: no pipeline '{}' (metal4.0 library not loaded)", v.shader)))?;
        let (i, o) = (&call.inputs, &call.outputs);
        let (m, k, n) = (i[0].shape[0], i[0].shape[1], i[1].shape[0]);
        if m * n == 0 {
            return Ok(());
        }
        if i[1].shape[1] != k || o[0].shape != [m, n] {
            return Err(Error::invalid("matmul_bf16_nax: shape mismatch"));
        }
        let bf = DType::Bf16;
        let (x, w, y) = (
            self.bind(&i[0], bf, false, "x")?,
            self.bind(&i[1], bf, false, "w")?,
            self.bind(&o[0], bf, true, "y")?,
        );
        let dim = |v: usize, what: &str| {
            u32::try_from(v).map_err(|_| Error::invalid(format!("matmul_bf16_nax: {what} = {v} exceeds u32")))
        };
        let p = NaxParams {
            m: dim(m, "M")?,
            n: dim(n, "N")?,
            k: dim(k, "K")?,
            swz_log: NAX_SWZ_LOG,
            s0: x.skip,
            s1: w.skip,
            s2: y.skip,
            pad: 0,
        };
        self.with_encoder(|enc| {
            encode_nax(enc, pso, v, (&x.mtl, x.off), (&w.mtl, w.off), (&y.mtl, y.off), &p);
        })
    }
}

/// Default threadgroup swizzle of the registered kernels.
pub const NAX_SWZ_LOG: u32 = 0;
