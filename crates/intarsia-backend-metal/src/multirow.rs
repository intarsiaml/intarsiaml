//! G0 prototype: bit-exact multi-row (`R` activation rows, one weight pass) decode GEMVs for speculative-decoding
//! verification (docs/research/metal-next-levers.md K1, docs/results/2026-10-15-metal-g0-multirow.md).
//!
//! Each shader in kernels/metal/gemv_multirow.metal mirrors one serving 1-row decode kernel and keeps its per-row
//! reduction order, so output row `r` is bitwise what the 1-row launch produces for activation row `r`:
//!
//! | op | 1-row serving kernel | multi-row shader |
//! |---|---|---|
//! | `Matmul`, 16-byte aligned | `gemv_bf16_rt_u2s8` | `gemv_bf16_rt_u2s8_mr<R>` |
//! | `Matmul`, 8-byte aligned (header 8 mod 16 shards) | `gemv_bf16_rt_a8u4s8` | `gemv_bf16_rt_a8u4s8_mr<R>` |
//! | `MatmulPair`, 16-byte / 8-byte | `gemv_pair_bf16_rt_u2s8` / `_a8u4s8` | `gemv_pair_bf16_rt_*_mr<R>` |
//! | `Logits`, 16-byte | `gemv_f32o_v` | `gemv_f32o_v_mr<R>` |
//!
//! Layouts: `x [R, K]`, `y [R, N]` row-major. Any other alignment (the 1-row scalar kernels) falls back to `R` ordinary
//! 1-row launches, which is bit-exact by construction.
//!
//! The registry entries are NOT used by serving: `Matmul` / `Logits` rank below the tiled GEMM (priority -1), and the
//! `MatmulPair` entry is hidden unless `INTARSIA_METAL_MULTIROW=1` (it is the only pair kernel that would accept
//! `M = 2..8`, so without the gate the model would start fusing short prefill chunks with it).

use crate::{encode_gemv, set_bytes, GemvParams, MetalBackend};
use intarsia_core::backend::TensorRef;
use intarsia_core::{DType, Error, Op, OpAttrs, OpCall, Result};
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLComputeCommandEncoder, MTLComputePipelineState, MTLSize};

pub const MATMUL_BF16_GEMV_ROWS: &str = "matmul_bf16_gemv_rows";
pub const MATMUL_PAIR_BF16_GEMV_ROWS: &str = "matmul_pair_bf16_gemv_rows";
pub const LOGITS_BF16_GEMV_ROWS: &str = "logits_bf16_gemv_rows";

/// Row counts compiled into the metallib (8 is the largest the registry accepts; 16 is for the benchmark only).
pub const MR_ROWS: [usize; 9] = [1, 2, 3, 4, 5, 6, 7, 8, 16];
/// Largest `M` the registry entries accept.
pub const MR_MAX_REGISTERED: usize = 8;

macro_rules! names {
    ($p:literal) => {
        [
            concat!($p, "1"),
            concat!($p, "2"),
            concat!($p, "3"),
            concat!($p, "4"),
            concat!($p, "5"),
            concat!($p, "6"),
            concat!($p, "7"),
            concat!($p, "8"),
            concat!($p, "16"),
        ]
    };
}

pub const MATMUL_V16: [&str; 9] = names!("gemv_bf16_rt_u2s8_mr");
pub const MATMUL_A8: [&str; 9] = names!("gemv_bf16_rt_a8u4s8_mr");
pub const PAIR_V16: [&str; 9] = names!("gemv_pair_bf16_rt_u2s8_mr");
pub const PAIR_A8: [&str; 9] = names!("gemv_pair_bf16_rt_a8u4s8_mr");
pub const LOGITS_V16: [&str; 9] = names!("gemv_f32o_v_mr");

/// Threadgroup size of the rt shaders (8 simdgroups, one output row per threadgroup).
pub const RT_THREADS: u32 = 256;
/// Output rows per threadgroup and threads of the f32-out logits shaders.
pub const LOGITS_ROWS_PER_TG: u32 = 4;
pub const LOGITS_THREADS: u32 = 128;

/// Host names of every multi-row shader (for the pipeline table).
pub fn shader_names() -> Vec<&'static str> {
    let mut v = Vec::new();
    for set in [&MATMUL_V16, &MATMUL_A8, &PAIR_V16, &PAIR_A8, &LOGITS_V16] {
        v.extend(set.iter().copied());
    }
    v
}

/// The shader of `set` for `rows` activation rows.
pub fn shader(set: &'static [&'static str; 9], rows: usize) -> Option<&'static str> {
    MR_ROWS.iter().position(|&r| r == rows).map(|i| set[i])
}

/// Whether the (default-off) `MatmulPair` registry entry is visible: `INTARSIA_METAL_MULTIROW=1`.
pub fn pair_entry_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("INTARSIA_METAL_MULTIROW").is_some_and(|v| v != "0"))
}

/// Kernel arguments of the pair shaders (`PairParams` in the shader; 32 bytes).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct PairParams {
    pub k: u32,
    pub n0: u32,
    pub n1: u32,
    pub y0_skip: u32,
    pub y1_skip: u32,
    pub pad: [u32; 3],
}

type Buf<'a> = (&'a ProtocolObject<dyn MTLBuffer>, usize);

/// Encode a multi-row `Matmul` GEMV (`y [R,N] = x [R,K] W^T`): one threadgroup per output row.
pub fn encode_matmul_rows(
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    pso: &ProtocolObject<dyn MTLComputePipelineState>,
    w: Buf,
    x: Buf,
    y: Buf,
    p: &GemvParams,
) {
    encode_gemv(enc, pso, w, x, y, p, p.n, RT_THREADS);
}

/// Encode a multi-row `Logits` GEMV (f32 out).
pub fn encode_logits_rows(
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    pso: &ProtocolObject<dyn MTLComputePipelineState>,
    w: Buf,
    x: Buf,
    y: Buf,
    p: &GemvParams,
) {
    encode_gemv(enc, pso, w, x, y, p, p.n.div_ceil(LOGITS_ROWS_PER_TG), LOGITS_THREADS);
}

/// Encode a multi-row `MatmulPair` GEMV (one dispatch over `n0 + n1` threadgroups).
#[allow(clippy::too_many_arguments)]
pub fn encode_pair_rows(
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    pso: &ProtocolObject<dyn MTLComputePipelineState>,
    w0: Buf,
    w1: Buf,
    x: Buf,
    y0: Buf,
    y1: Buf,
    p: &PairParams,
) {
    enc.setComputePipelineState(pso);
    // SAFETY: the caller keeps the buffers alive and the offsets inside them.
    unsafe {
        enc.setBuffer_offset_atIndex(Some(w0.0), w0.1, 0);
        enc.setBuffer_offset_atIndex(Some(w1.0), w1.1, 1);
        enc.setBuffer_offset_atIndex(Some(x.0), x.1, 2);
        enc.setBuffer_offset_atIndex(Some(y0.0), y0.1, 3);
        enc.setBuffer_offset_atIndex(Some(y1.0), y1.1, 4);
    }
    set_bytes(enc, p, 5);
    enc.dispatchThreadgroups_threadsPerThreadgroup(
        MTLSize { width: (p.n0 + p.n1) as usize, height: 1, depth: 1 },
        MTLSize { width: RT_THREADS as usize, height: 1, depth: 1 },
    );
}

/// Encode the tiled bf16 GEMM (`gemm_bf16_o_bf16` / `gemm_bf16_o_f32`, the serving fallback for `M = 2..15`):
/// `y [M,N] = x [M,K] W^T`. Test / benchmark helper (aligned operands, no skips).
#[allow(clippy::too_many_arguments)]
pub fn encode_tiled_gemm(
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    pso: &ProtocolObject<dyn MTLComputePipelineState>,
    x: Buf,
    w: Buf,
    y: Buf,
    m: u32,
    n: u32,
    k: u32,
) {
    let p = crate::dense::OpP { dims: [m, n, k, 0, 0, 0, 0, 0], skips: [0; 5], f0: 0.0, f1: 0.0, pad: 0 };
    enc.setComputePipelineState(pso);
    // SAFETY: as above.
    unsafe {
        enc.setBuffer_offset_atIndex(Some(x.0), x.1, 0);
        enc.setBuffer_offset_atIndex(Some(w.0), w.1, 1);
        enc.setBuffer_offset_atIndex(Some(y.0), y.1, 2);
    }
    set_bytes(enc, &p, 3);
    enc.dispatchThreadgroups_threadsPerThreadgroup(
        MTLSize { width: (n as usize).div_ceil(32), height: (m as usize).div_ceil(32), depth: 1 },
        MTLSize { width: 128, height: 1, depth: 1 },
    );
}

fn single(x: &TensorRef, w: &TensorRef, y: &TensorRef) -> OpCall {
    OpCall::new(Op::Matmul, vec![x.clone(), w.clone()], vec![y.clone()], OpAttrs::default())
}

impl MetalBackend {
    fn rows_shader(&self, set: &'static [&'static str; 9], rows: usize, what: &str) -> Result<&'static str> {
        shader(set, rows).ok_or_else(|| Error::invalid(format!("{what}: {rows} rows (compiled: {MR_ROWS:?})")))
    }

    /// `Matmul` with `M = 1..=16` rows (see [`MR_ROWS`]) on the multi-row GEMV.
    pub(crate) fn launch_matmul_rows(&self, call: &OpCall) -> Result<()> {
        let (x, w, y) = match (call.inputs.as_slice(), call.outputs.as_slice()) {
            ([x, w], [y]) => (x, w, y),
            _ => return Err(Error::invalid("matmul_bf16_gemv_rows: expected 2 inputs / 1 output")),
        };
        if x.shape.len() != 2 || w.shape.len() != 2 || y.shape.len() != 2 {
            return Err(Error::invalid("matmul_bf16_gemv_rows: rank 2 operands"));
        }
        let (m, k, n) = (x.shape[0], w.shape[1], w.shape[0]);
        if x.shape[1] != k || y.shape != [m, n] {
            return Err(Error::invalid("matmul_bf16_gemv_rows: shape mismatch"));
        }
        if [x.dtype, w.dtype, y.dtype] != [DType::Bf16; 3] {
            return Err(Error::invalid("matmul_bf16_gemv_rows: bf16 only"));
        }
        if m == 0 || n == 0 {
            return Ok(());
        }
        if k > u32::MAX as usize || n > u32::MAX as usize {
            return Err(Error::invalid("matmul_bf16_gemv_rows: dimension exceeds u32"));
        }
        let (wb, xb, yb) = (self.mbuf(&w.buf)?, self.mbuf(&x.buf)?, self.mbuf(&y.buf)?);
        let (wo, xo, yo) = (wb.base + w.offset, xb.base + x.offset, yb.base + y.offset);
        if yb.read_only {
            return Err(Error::backend("metal: output buffer is read-only (wrapped host memory)"));
        }
        let set = if k % 8 == 0 && wo % 16 == 0 && xo % 16 == 0 {
            Some(&MATMUL_V16)
        } else if k % 4 == 0 && wo % 8 == 0 && xo % 8 == 0 {
            Some(&MATMUL_A8)
        } else {
            None
        };
        let Some(set) = set.filter(|_| shader(&MATMUL_V16, m).is_some()) else {
            // scalar kernels / unsupported row count: R ordinary 1-row launches (bit-exact by construction)
            for r in 0..m {
                let plan = self.plan_gemv(&single(&x.rows(r, 1), w, &y.rows(r, 1)))?;
                let pso =
                    self.pso(plan.name).ok_or_else(|| Error::backend(format!("metal: no pipeline '{}'", plan.name)))?;
                self.with_encoder(|enc| {
                    encode_gemv(
                        enc,
                        pso,
                        (&plan.w.0, plan.w.1),
                        (&plan.x.0, plan.x.1),
                        (&plan.y.0, plan.y.1),
                        &plan.params,
                        plan.groups,
                        plan.threads,
                    )
                })?;
            }
            return Ok(());
        };
        let name = self.rows_shader(set, m, "matmul_bf16_gemv_rows")?;
        let pso = self.pso(name).ok_or_else(|| Error::backend(format!("metal: no pipeline '{name}'")))?;
        let params = GemvParams { k: k as u32, n: n as u32, y_skip: ((yo & 3) / 2) as u32, ..Default::default() };
        self.with_encoder(|enc| encode_matmul_rows(enc, pso, (&wb.mtl, wo), (&xb.mtl, xo), (&yb.mtl, yo & !3), &params))
    }

    /// `MatmulPair` with `M` rows on the multi-row pair GEMV (16-byte or 8-byte operands, else per-row launches).
    pub(crate) fn launch_pair_rows(&self, call: &OpCall) -> Result<()> {
        let (x, w0, w1, y0, y1) = match (call.inputs.as_slice(), call.outputs.as_slice()) {
            ([x, w0, w1], [y0, y1]) => (x, w0, w1, y0, y1),
            _ => return Err(Error::invalid("matmul_pair_bf16_gemv_rows: expected 3 inputs / 2 outputs")),
        };
        if x.shape.len() != 2 || w0.shape.len() != 2 || w1.shape.len() != 2 {
            return Err(Error::invalid("matmul_pair_bf16_gemv_rows: rank 2 operands"));
        }
        let (m, k, n0, n1) = (x.shape[0], x.shape[1], w0.shape[0], w1.shape[0]);
        if w0.shape[1] != k || w1.shape[1] != k || y0.shape != [m, n0] || y1.shape != [m, n1] {
            return Err(Error::invalid("matmul_pair_bf16_gemv_rows: shape mismatch"));
        }
        if [x.dtype, w0.dtype, w1.dtype, y0.dtype, y1.dtype] != [DType::Bf16; 5] {
            return Err(Error::invalid("matmul_pair_bf16_gemv_rows: bf16 only"));
        }
        if m == 0 || n0 + n1 == 0 {
            return Ok(());
        }
        let (w0b, w1b, xb, y0b, y1b) =
            (self.mbuf(&w0.buf)?, self.mbuf(&w1.buf)?, self.mbuf(&x.buf)?, self.mbuf(&y0.buf)?, self.mbuf(&y1.buf)?);
        if y0b.read_only || y1b.read_only {
            return Err(Error::backend("metal: output buffer is read-only (wrapped host memory)"));
        }
        let (w0o, w1o, xo, y0o, y1o) = (
            w0b.base + w0.offset,
            w1b.base + w1.offset,
            xb.base + x.offset,
            y0b.base + y0.offset,
            y1b.base + y1.offset,
        );
        let v16 = k % 8 == 0 && w0o % 16 == 0 && w1o % 16 == 0 && xo % 16 == 0;
        let a8 = k % 4 == 0 && w0o % 8 == 0 && w1o % 8 == 0 && xo % 8 == 0;
        let set = if v16 {
            Some(&PAIR_V16)
        } else if a8 {
            Some(&PAIR_A8)
        } else {
            None
        };
        let Some(set) = set.filter(|_| shader(&PAIR_V16, m).is_some()) else {
            for r in 0..m {
                // two ordinary GEMVs per row (what the model runs when no pair kernel is available)
                self.launch_matmul_pair(&OpCall::new(
                    Op::MatmulPair,
                    vec![x.rows(r, 1), w0.clone(), w1.clone()],
                    vec![y0.rows(r, 1), y1.rows(r, 1)],
                    OpAttrs::default(),
                ))?;
            }
            return Ok(());
        };
        let name = self.rows_shader(set, m, "matmul_pair_bf16_gemv_rows")?;
        let pso = self.pso(name).ok_or_else(|| Error::backend(format!("metal: no pipeline '{name}'")))?;
        let p = PairParams {
            k: k as u32,
            n0: n0 as u32,
            n1: n1 as u32,
            y0_skip: ((y0o & 3) / 2) as u32,
            y1_skip: ((y1o & 3) / 2) as u32,
            pad: [0; 3],
        };
        self.with_encoder(|enc| {
            encode_pair_rows(
                enc,
                pso,
                (&w0b.mtl, w0o),
                (&w1b.mtl, w1o),
                (&xb.mtl, xo),
                (&y0b.mtl, y0o & !3),
                (&y1b.mtl, y1o & !3),
                &p,
            )
        })
    }

    /// `Logits` (f32 out) with `M` rows on the multi-row f32-out GEMV (16-byte operands, else per-row launches).
    pub(crate) fn launch_logits_rows(&self, call: &OpCall) -> Result<()> {
        let (x, w, y) = match (call.inputs.as_slice(), call.outputs.as_slice()) {
            ([x, w], [y]) => (x, w, y),
            _ => return Err(Error::invalid("logits_bf16_gemv_rows: expected 2 inputs / 1 output")),
        };
        if x.shape.len() != 2 || w.shape.len() != 2 {
            return Err(Error::invalid("logits_bf16_gemv_rows: rank 2 operands"));
        }
        let (m, k, n) = (x.shape[0], x.shape[1], w.shape[0]);
        if w.shape[1] != k || y.shape != [m, n] || y.dtype != DType::F32 {
            return Err(Error::invalid("logits_bf16_gemv_rows: shape mismatch / f32 output"));
        }
        if m == 0 || n == 0 {
            return Ok(());
        }
        let (wb, xb, yb) = (self.mbuf(&w.buf)?, self.mbuf(&x.buf)?, self.mbuf(&y.buf)?);
        if yb.read_only {
            return Err(Error::backend("metal: output buffer is read-only (wrapped host memory)"));
        }
        let (wo, xo, yo) = (wb.base + w.offset, xb.base + x.offset, yb.base + y.offset);
        if !(k % 8 == 0 && wo % 16 == 0 && xo % 16 == 0 && yo % 4 == 0) || shader(&LOGITS_V16, m).is_none() {
            for r in 0..m {
                self.launch_dense(
                    crate::kernels::LOGITS_BF16_GEMV,
                    &OpCall::new(Op::Logits, vec![x.rows(r, 1), w.clone()], vec![y.rows(r, 1)], OpAttrs::default()),
                )?;
            }
            return Ok(());
        }
        let name = self.rows_shader(&LOGITS_V16, m, "logits_bf16_gemv_rows")?;
        let pso = self.pso(name).ok_or_else(|| Error::backend(format!("metal: no pipeline '{name}'")))?;
        let params = GemvParams { k: k as u32, n: n as u32, ..Default::default() };
        self.with_encoder(|enc| encode_logits_rows(enc, pso, (&wb.mtl, wo), (&xb.mtl, xo), (&yb.mtl, yo), &params))
    }
}
