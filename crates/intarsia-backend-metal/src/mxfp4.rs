//! Launcher of `matmul_mxfp4_gemv` (`Op::MatmulMxfp4`, kernels/metal/mxfp4.metal; the port of kernels/hip/mxfp4.hip).
//!
//! `out[M,N] = q(x)[M,K] @ W^T`, W = packed e2m1 `u8 [N,K/2]` with one e8m0 scale per 32 along K (`u8 [N,K/32]`), `act
//! [1] f32` picking W4A16 (0, x as is) or W4A4 (1, x QDQ'd to MXFP4 per group of 32, Quark "even" scale). Two dispatches
//! in one launch: the activations are QDQ'd (or copied) once into the plan-reserved scratch (`bf16 [M,K]`, exact), then a
//! simdgroup-per-output-row GEMV reads them. Any M: 1 row uses the `_m1` shader, more take 8 activation rows per
//! threadgroup row of the grid.

use crate::dense::OpP;
use crate::{set_bytes, MetalBackend};
use intarsia_core::{DType, Error, OpCall, Result, ShapeInfo};
use objc2_metal::{MTLComputeCommandEncoder, MTLComputePipelineState, MTLSize};

/// `KernelImpl::name` of the kernel.
pub const MATMUL_MXFP4_GEMV: &str = "matmul_mxfp4_gemv";

const XPREP: &str = "mxfp4_xprep_bf16";
/// Shader entry points (kernels/metal/mxfp4.metal).
pub const SHADERS: [&str; 5] =
    [XPREP, "mxfp4_gemv_bf16_m1", "mxfp4_gemv_bf16_m8", "mxfp4_gemv_f32_m1", "mxfp4_gemv_f32_m8"];

/// Output rows (= simdgroups) per threadgroup (`MX_SG` in the shader) and activation rows per grid row of `_m8`.
const ROWS_PER_TG: usize = 4;
const M_BLOCK: usize = 8;

/// Scratch: the QDQ'd activations as bf16 `[M,K]` (same bound as the hip entry).
pub fn scratch(s: &ShapeInfo) -> usize {
    s.m * s.k * 2 + 16
}

fn u32_of(v: usize, what: &str) -> Result<u32> {
    u32::try_from(v).map_err(|_| Error::invalid(format!("metal matmul_mxfp4_gemv: {what} = {v} exceeds u32")))
}

impl MetalBackend {
    pub(crate) fn launch_mxfp4(&self, call: &OpCall) -> Result<()> {
        let (x, w, sc, act, out) = match (call.inputs.as_slice(), call.outputs.as_slice()) {
            ([x, w, sc, act], [out]) => (x, w, sc, act, out),
            _ => return Err(Error::invalid("metal matmul_mxfp4_gemv: expected 4 inputs / 1 output")),
        };
        if x.shape.len() != 2 || w.shape.len() != 2 || sc.shape.len() != 2 || out.shape.len() != 2 {
            return Err(Error::invalid("metal matmul_mxfp4_gemv: x, w, scale and out must be 2-D"));
        }
        let (m, k, n) = (x.shape[0], x.shape[1], w.shape[0]);
        if k % 32 != 0
            || w.shape[1] != k / 2
            || sc.shape[0] != n
            || sc.shape[1] != k / 32
            || act.numel() != 1
            || out.shape[0] != m
            || out.shape[1] != n
        {
            return Err(Error::invalid("metal matmul_mxfp4_gemv: shape mismatch"));
        }
        if m == 0 || n == 0 {
            return Ok(());
        }
        let f32_out = match out.dtype {
            DType::Bf16 => false,
            DType::F32 => true,
            d => return Err(Error::invalid(format!("metal matmul_mxfp4_gemv: output dtype {d}"))),
        };
        // the shaders index with 32-bit math in places
        u32_of(m * k, "M*K")?;
        u32_of(n * (k / 2), "N*K/2")?;
        u32_of(m * n, "M*N")?;
        if k == 0 {
            return Err(Error::invalid("metal matmul_mxfp4_gemv: K = 0"));
        }
        let xb = self.bind(x, DType::Bf16, false, "x")?;
        let wb = self.bind(w, DType::U8, false, "w")?;
        let sb = self.bind(sc, DType::U8, false, "scale")?;
        let ab = self.bind(act, DType::F32, false, "act")?;
        let ob = self.bind(out, if f32_out { DType::F32 } else { DType::Bf16 }, true, "out")?;
        if wb.skip != 0 {
            return Err(Error::invalid("metal matmul_mxfp4_gemv: the weights must be 4-byte aligned"));
        }
        let need = m * k * 2;
        let scratch = call
            .scratch
            .as_ref()
            .ok_or_else(|| Error::invalid("metal matmul_mxfp4_gemv: the scratch workspace (OpCall::scratch) is missing"))?;
        if scratch.numel() * scratch.dtype.size_bytes() < need {
            return Err(Error::invalid(format!(
                "metal matmul_mxfp4_gemv: scratch of {} bytes, need {need}",
                scratch.numel() * scratch.dtype.size_bytes()
            )));
        }
        let xs = self.bind(scratch, DType::U8, true, "scratch")?;
        if xs.skip != 0 {
            return Err(Error::invalid("metal matmul_mxfp4_gemv: the scratch must be 4-byte aligned"));
        }

        let gemv = match (f32_out, m == 1) {
            (false, true) => SHADERS[1],
            (false, false) => SHADERS[2],
            (true, true) => SHADERS[3],
            (true, false) => SHADERS[4],
        };
        let pso_x = self.pso(XPREP).ok_or_else(|| Error::backend(format!("metal: no pipeline '{XPREP}'")))?;
        let pso_g = self.pso(gemv).ok_or_else(|| Error::backend(format!("metal: no pipeline '{gemv}'")))?;
        let px = OpP { dims: [u32_of(m, "M")?, u32_of(n, "N")?, u32_of(k, "K")?, 0, 0, 0, 0, 0], skips: [xb.skip, 0, 0, 0, 0], ..Default::default() };
        let pg = OpP { skips: [sb.skip, ob.skip, 0, 0, 0], ..px };
        let groups = m * (k / 32);
        let tx = pso_x.maxTotalThreadsPerThreadgroup().min(256);
        let m_blocks = if m == 1 { 1 } else { m.div_ceil(M_BLOCK) };
        self.with_encoder(|enc| {
            // SAFETY: offsets lie inside the buffers (TensorRef bounds were checked by `OpCall::check`, the scratch size above).
            unsafe {
                enc.setComputePipelineState(pso_x);
                for (i, b) in [&xb, &ab, &xs].into_iter().enumerate() {
                    enc.setBuffer_offset_atIndex(Some(&b.mtl), b.off, i);
                }
            }
            set_bytes(enc, &px, 3);
            enc.dispatchThreads_threadsPerThreadgroup(
                MTLSize { width: groups, height: 1, depth: 1 },
                MTLSize { width: tx, height: 1, depth: 1 },
            );
            // SAFETY: as above.
            unsafe {
                enc.setComputePipelineState(pso_g);
                for (i, b) in [&xs, &wb, &sb, &ob].into_iter().enumerate() {
                    enc.setBuffer_offset_atIndex(Some(&b.mtl), b.off, i);
                }
            }
            set_bytes(enc, &pg, 4);
            enc.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize { width: n.div_ceil(ROWS_PER_TG), height: m_blocks, depth: 1 },
                MTLSize { width: ROWS_PER_TG * 32, height: 1, depth: 1 },
            );
        })
    }
}
