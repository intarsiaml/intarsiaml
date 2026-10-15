//! NAX (M5 neural accelerator) bf16 GEMM launchers: `Y[M,N] = X[M,K] . W[N,K]^T` for `Op::Matmul` and the two-weight
//! `Op::MatmulPair`, kernels/metal/gemm_nax_bf16.metal, compiled `-std=metal4.0` into `intarsia_metal4.metallib`
//! (build.rs `METAL4_SOURCES`).
//!
//! The tile is a property of the shader entry point. All tiles give bitwise the same output for the same operands (the
//! K reduction order does not depend on the tile, tests/gemm_nax.rs), so [`pick_variant`] may choose by shape without a
//! row's result depending on the call it arrives in. Precision is strict (`relaxed_precision = false`); the relaxed
//! mode was bitwise equal and no faster in the P1 spike, so it is not built.

use crate::kernels::{MATMUL_BF16_NAX, MATMUL_PAIR_BF16_NAX};
use crate::{set_bytes, MetalBackend};
use intarsia_core::{DType, Error, OpCall, Result};
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLComputeCommandEncoder, MTLComputePipelineState, MTLSize};

/// One compiled configuration of `gemm_nax_t` / `gemm_nax_pair_t`.
#[derive(Clone, Copy, Debug)]
pub struct NaxVariant {
    /// `Matmul` entry point (`kernels::metal4_shaders`).
    pub shader: &'static str,
    /// `MatmulPair` entry point of the same tile.
    pub pair_shader: &'static str,
    /// Output tile per threadgroup.
    pub bm: u32,
    pub bn: u32,
    /// Simdgroups per threadgroup.
    pub sg: u32,
}

impl NaxVariant {
    pub const fn threads(&self) -> u32 {
        self.sg * 32
    }
}

/// Every `NAX_T` instantiation of gemm_nax_bf16.metal.
pub const NAX_VARIANTS: &[NaxVariant] = &[
    NaxVariant { shader: "gemm_nax_t64x64w4", pair_shader: "gemm_nax_pair_t64x64w4", bm: 64, bn: 64, sg: 4 },
    NaxVariant { shader: "gemm_nax_t128x64w4", pair_shader: "gemm_nax_pair_t128x64w4", bm: 128, bn: 64, sg: 4 },
    NaxVariant { shader: "gemm_nax_t32x32w1", pair_shader: "gemm_nax_pair_t32x32w1", bm: 32, bn: 32, sg: 1 },
    NaxVariant { shader: "gemm_nax_t32x128w2", pair_shader: "gemm_nax_pair_t32x128w2", bm: 32, bn: 128, sg: 2 },
    NaxVariant { shader: "gemm_nax_t64x64w2", pair_shader: "gemm_nax_pair_t64x64w2", bm: 64, bn: 64, sg: 2 },
];

/// Smallest M the NAX kernels are registered for (M = 1 is the decode GEMV's). Tails of 2..15 rows are mostly padding in a
/// 16-row accelerator tile but the call is weight-bandwidth bound, so it is far cheaper than the tiled GEMM (which streams
/// the weights at about a tenth of the bandwidth) and keeps a row's bits independent of the chunk it lands in.
pub const NAX_MIN_M: usize = 2;
/// The 128x64 tile is used for M >= `NAX_BIG_TILE_MIN_M` when it still leaves at least this many threadgroups (40 GPU
/// cores; with fewer, the 64x64 tile's finer grid wins; for M < 128 the second 64 rows of a 128-row tile are padding).
/// Measured per shape and M in docs/results/raw/2026-10-14-metal-prefill-p1/full/bench-pick.txt.
pub const NAX_BIG_TILE_MIN_TGS: usize = 192;
pub const NAX_BIG_TILE_MIN_M: usize = 128;
/// Rows up to which the chunk-sized calls (the 256-row default chunk and its tail) take the P4/P5 tiles of
/// [`pick_variant`]; above it the P1 rule above applies again.
pub const NAX_CHUNK_MAX_M: usize = 512;
/// Output elements (`m * n`) below which the one-simdgroup 32x32 tile is the fastest (the k + v pair at M = 256: 2048
/// columns, 64 x 8 tiles).
pub const NAX_FINE_ELEMS: usize = 700_000;
/// Calls with fewer columns than this (q / o, down: 4096) take the 32-row tile at an M tail (`m % 64` in 1..=32); the
/// 24576-column gate + up pair does not gain from it.
pub const NAX_TAIL_MAX_N: usize = 8192;

fn variant(shader: &str) -> &'static NaxVariant {
    NAX_VARIANTS.iter().find(|v| v.shader == shader).expect("NAX variant table")
}

/// The tile for an `m` by `n` output (`n` of a pair is `n0 + n1`). Any choice gives the same bits; this rule is the faster
/// one measured per shape and M inside the engine (docs/results/2026-10-15-metal-prefill-p45.md, raw/.../engine-sweep).
/// M < 128 and M > `NAX_CHUNK_MAX_M`: the P1 pick (64x64 four simdgroups, or 128x64 when it leaves enough threadgroups).
/// Chunk-sized M: the k + v pair on 32x32 / one simdgroup; q / o and down at an M tail on 32x128 / two simdgroups (a
/// 64-row tile would be mostly padding); everything else 64x64 / two simdgroups.
pub fn pick_variant(m: usize, n: usize) -> &'static NaxVariant {
    // measurement knob: INTARSIA_NAX_TILE=<shader name, e.g. gemm_nax_t64x64w2> forces one tile for every call
    static FORCED: std::sync::OnceLock<Option<&'static NaxVariant>> = std::sync::OnceLock::new();
    if let Some(v) = FORCED.get_or_init(|| {
        let want = std::env::var("INTARSIA_NAX_TILE").ok()?;
        NAX_VARIANTS.iter().find(|v| v.shader == want || v.shader.ends_with(&want))
    }) {
        return v;
    }
    if (NAX_BIG_TILE_MIN_M..=NAX_CHUNK_MAX_M).contains(&m) {
        if m * n < NAX_FINE_ELEMS {
            variant("gemm_nax_t32x32w1")
        } else if (1..=32).contains(&(m % 64)) && n < NAX_TAIL_MAX_N {
            variant("gemm_nax_t32x128w2")
        } else {
            variant("gemm_nax_t64x64w2")
        }
    } else {
        let big = variant("gemm_nax_t128x64w4");
        if m >= NAX_BIG_TILE_MIN_M && m.div_ceil(big.bm as usize) * n.div_ceil(big.bn as usize) >= NAX_BIG_TILE_MIN_TGS {
            big
        } else {
            variant("gemm_nax_t64x64w4")
        }
    }
}

/// Kernel arguments of `gemm_nax_t` (`NaxP` in gemm_nax_bf16.metal; 32 bytes).
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

/// Kernel arguments of `gemm_nax_pair_t` (`NaxPairP`; 48 bytes).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct NaxPairParams {
    pub m: u32,
    pub n0: u32,
    pub n1: u32,
    pub k: u32,
    pub s_a: u32,
    pub s_w0: u32,
    pub s_w1: u32,
    pub s_c0: u32,
    pub s_c1: u32,
    pub pad: [u32; 3],
}

/// Threadgroups `(x, y)` for `v` over an `m` by `n` output.
pub fn grid(v: &NaxVariant, m: u32, n: u32, swz_log: u32) -> (usize, usize) {
    let tiles_m = m.div_ceil(v.bm);
    let tiles_n = n.div_ceil(v.bn);
    ((tiles_n << swz_log) as usize, tiles_m.div_ceil(1 << swz_log) as usize)
}

type MtlBuf<'a> = (&'a ProtocolObject<dyn MTLBuffer>, usize);

/// Encode one dispatch. Each operand is `(buffer, 4-byte aligned byte offset)`; `p.s0/s1/s2` carry the leftover element.
pub fn encode_nax(
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    pso: &ProtocolObject<dyn MTLComputePipelineState>,
    v: &NaxVariant,
    a: MtlBuf,
    w: MtlBuf,
    c: MtlBuf,
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

/// Encode one `gemm_nax_pair_t` dispatch: `c0 = a w0^T`, `c1 = a w1^T`.
#[allow(clippy::too_many_arguments)]
pub fn encode_nax_pair(
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    pso: &ProtocolObject<dyn MTLComputePipelineState>,
    v: &NaxVariant,
    a: MtlBuf,
    w0: MtlBuf,
    w1: MtlBuf,
    c0: MtlBuf,
    c1: MtlBuf,
    p: &NaxPairParams,
) {
    enc.setComputePipelineState(pso);
    // SAFETY: as `encode_nax`.
    unsafe {
        enc.setBuffer_offset_atIndex(Some(a.0), a.1, 0);
        enc.setBuffer_offset_atIndex(Some(w0.0), w0.1, 1);
        enc.setBuffer_offset_atIndex(Some(w1.0), w1.1, 2);
        enc.setBuffer_offset_atIndex(Some(c0.0), c0.1, 3);
        enc.setBuffer_offset_atIndex(Some(c1.0), c1.1, 4);
    }
    set_bytes(enc, p, 5);
    enc.dispatchThreadgroups_threadsPerThreadgroup(
        MTLSize { width: (p.n0.div_ceil(v.bn) + p.n1.div_ceil(v.bn)) as usize, height: p.m.div_ceil(v.bm) as usize, depth: 1 },
        MTLSize { width: v.threads() as usize, height: 1, depth: 1 },
    );
}

fn dim(v: usize, what: &str) -> Result<u32> {
    u32::try_from(v).map_err(|_| Error::invalid(format!("metal nax: {what} = {v} exceeds u32")))
}

impl MetalBackend {
    /// Launch `matmul_bf16_nax` for an `Op::Matmul` call (x `[M,K]`, w `[N,K]`, y `[M,N]`, all bf16).
    pub(crate) fn launch_nax(&self, kernel: &str, call: &OpCall) -> Result<()> {
        debug_assert_eq!(kernel, MATMUL_BF16_NAX);
        let (i, o) = (&call.inputs, &call.outputs);
        let (m, k, n) = (i[0].shape[0], i[0].shape[1], i[1].shape[0]);
        if m * n == 0 {
            return Ok(());
        }
        if i[1].shape[1] != k || o[0].shape != [m, n] {
            return Err(Error::invalid("matmul_bf16_nax: shape mismatch"));
        }
        let v = pick_variant(m, n);
        let pso = self
            .pso(v.shader)
            .ok_or_else(|| Error::backend(format!("metal: no pipeline '{}' (metal4.0 library not loaded)", v.shader)))?;
        let bf = DType::Bf16;
        let (x, w, y) = (
            self.bind(&i[0], bf, false, "x")?,
            self.bind(&i[1], bf, false, "w")?,
            self.bind(&o[0], bf, true, "y")?,
        );
        let p = NaxParams {
            m: dim(m, "M")?,
            n: dim(n, "N")?,
            k: dim(k, "K")?,
            swz_log: 0,
            s0: x.skip,
            s1: w.skip,
            s2: y.skip,
            pad: 0,
        };
        self.with_encoder(|enc| {
            encode_nax(enc, pso, v, (&x.mtl, x.off), (&w.mtl, w.off), (&y.mtl, y.off), &p);
        })
    }

    /// Launch `matmul_pair_bf16_nax` for an `Op::MatmulPair` call with M >= 16 (x `[M,K]`, w0 `[N0,K]`, w1 `[N1,K]`,
    /// y0 `[M,N0]`, y1 `[M,N1]`): bitwise two `matmul_bf16_nax` launches, in one dispatch.
    pub(crate) fn launch_nax_pair(&self, kernel: &str, call: &OpCall) -> Result<()> {
        debug_assert_eq!(kernel, MATMUL_PAIR_BF16_NAX);
        let (i, o) = (&call.inputs, &call.outputs);
        if i.len() != 3 || o.len() != 2 {
            return Err(Error::invalid("matmul_pair_bf16_nax: expected 3 inputs / 2 outputs"));
        }
        let (m, k, n0, n1) = (i[0].shape[0], i[0].shape[1], i[1].shape[0], i[2].shape[0]);
        if m == 0 {
            return Ok(());
        }
        if i[1].shape[1] != k || i[2].shape[1] != k || o[0].shape != [m, n0] || o[1].shape != [m, n1] {
            return Err(Error::invalid("matmul_pair_bf16_nax: shape mismatch"));
        }
        let v = pick_variant(m, n0 + n1);
        let pso = self.pso(v.pair_shader).ok_or_else(|| {
            Error::backend(format!("metal: no pipeline '{}' (metal4.0 library not loaded)", v.pair_shader))
        })?;
        let bf = DType::Bf16;
        let (x, w0, w1, y0, y1) = (
            self.bind(&i[0], bf, false, "x")?,
            self.bind(&i[1], bf, false, "w0")?,
            self.bind(&i[2], bf, false, "w1")?,
            self.bind(&o[0], bf, true, "y0")?,
            self.bind(&o[1], bf, true, "y1")?,
        );
        let p = NaxPairParams {
            m: dim(m, "M")?,
            n0: dim(n0, "N0")?,
            n1: dim(n1, "N1")?,
            k: dim(k, "K")?,
            s_a: x.skip,
            s_w0: w0.skip,
            s_w1: w1.skip,
            s_c0: y0.skip,
            s_c1: y1.skip,
            pad: [0; 3],
        };
        self.with_encoder(|enc| {
            encode_nax_pair(
                enc,
                pso,
                v,
                (&x.mtl, x.off),
                (&w0.mtl, w0.off),
                (&w1.mtl, w1.off),
                (&y0.mtl, y0.off),
                (&y1.mtl, y1.off),
                &p,
            );
        })
    }
}
