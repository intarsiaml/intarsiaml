//! Launchers of `matmul_nvfp4_gemv` / `matmul_nvfp4_mma` (`Op::MatmulNvfp4`, kernels/metal/nvfp4.metal; the port of
//! kernels/hip/nvfp4.hip).
//!
//! `out[M,N] = x[M,K] W^T`, W = e2m1 `u8 [N,K/2]` * e4m3 block scales `[N,K/16]` * global scale `[1] f32` (read on the GPU).
//! The decode GEMV (M <= 8) is one dispatch without scratch (M = 1: split-K over the simdgroups of a threadgroup); the GEMM (any M) is a tiled `simdgroup_matrix` bf16 GEMM that
//! dequantises the weight tiles (code * block scale, exact in bf16) and applies the global scale in the epilogue.

use crate::dense::OpP;
use crate::{set_bytes, MetalBackend};
use intarsia_core::{DType, Error, OpCall, Result, ShapeInfo};
use objc2_metal::{MTLComputeCommandEncoder, MTLComputePipelineState, MTLSize};

/// `KernelImpl::name` of the kernels.
pub const MATMUL_NVFP4_GEMV: &str = "matmul_nvfp4_gemv";
pub const MATMUL_NVFP4_MMA: &str = "matmul_nvfp4_mma";

/// Shader entry points (kernels/metal/nvfp4.metal).
pub const SHADERS: [&str; 4] = ["nv_gemv_bf16_m8", "nv_gemv_f32_m8", "nv_mma_bf16", "nv_mma_f32"];

/// Split-K M = 1 GEMV entry points `nv_gv_{bf16,f32}_s<SG>r<R>`, index `(sg_i * 2 + f32) * 3 + r_i` over `GV_SG` / `GV_R`.
pub const GV_SHADERS: [&str; 24] = [
    "nv_gv_bf16_s1r1",
    "nv_gv_bf16_s1r2",
    "nv_gv_bf16_s1r4",
    "nv_gv_f32_s1r1",
    "nv_gv_f32_s1r2",
    "nv_gv_f32_s1r4",
    "nv_gv_bf16_s2r1",
    "nv_gv_bf16_s2r2",
    "nv_gv_bf16_s2r4",
    "nv_gv_f32_s2r1",
    "nv_gv_f32_s2r2",
    "nv_gv_f32_s2r4",
    "nv_gv_bf16_s4r1",
    "nv_gv_bf16_s4r2",
    "nv_gv_bf16_s4r4",
    "nv_gv_f32_s4r1",
    "nv_gv_f32_s4r2",
    "nv_gv_f32_s4r4",
    "nv_gv_bf16_s8r1",
    "nv_gv_bf16_s8r2",
    "nv_gv_bf16_s8r4",
    "nv_gv_f32_s8r1",
    "nv_gv_f32_s8r2",
    "nv_gv_f32_s8r4",
];
const GV_SG: [usize; 4] = [1, 2, 4, 8];
const GV_R: [usize; 3] = [1, 2, 4];
/// Threads the split-K GEMV launches at least (when N allows): enough to fill the GPU.
const GV_THREADS: usize = 32768;

/// `_m8` GEMV (M = 2..=8): simdgroups per threadgroup (`NV_SG`), output rows per simdgroup (`RS`), activation rows (`MB`).
const GEMV_SG: usize = 8;
const GEMV_RS: usize = 2;
const GEMV_MB: usize = 8;
/// GEMM tile (`NV_BM` / `NV_BN`) and threads per threadgroup.
const TILE: usize = 64;
const MMA_THREADS: usize = 128;

/// The decode GEMV serves 1..=8 rows.
pub fn gemv_ok(s: &ShapeInfo) -> bool {
    (1..=GEMV_MB).contains(&s.m) && s.k % 16 == 0
}

/// (simdgroups per threadgroup, rows per threadgroup) of the split-K GEMV: one 16-element block of a row per thread (SG from K),
/// and the most rows per threadgroup that still launches `GV_THREADS`.
fn gv_pick(n: usize, k: usize) -> (usize, usize) {
    let sg = GV_SG.into_iter().rev().find(|sg| k / 16 >= 32 * sg).unwrap_or(1);
    let r = GV_R.into_iter().rev().find(|r| n.div_ceil(*r) * sg * 32 >= GV_THREADS).unwrap_or(1);
    (sg, r)
}

fn u32_of(v: usize, what: &str) -> Result<u32> {
    u32::try_from(v).map_err(|_| Error::invalid(format!("metal matmul_nvfp4: {what} = {v} exceeds u32")))
}

impl MetalBackend {
    pub(crate) fn launch_nvfp4(&self, kernel: &str, call: &OpCall) -> Result<()> {
        let (x, w, bs, gs, out) = match (call.inputs.as_slice(), call.outputs.as_slice()) {
            ([x, w, bs, gs], [out]) => (x, w, bs, gs, out),
            _ => return Err(Error::invalid("metal matmul_nvfp4: expected 4 inputs / 1 output")),
        };
        if x.shape.len() != 2 || w.shape.len() != 2 || bs.shape.len() != 2 || out.shape.len() != 2 {
            return Err(Error::invalid("metal matmul_nvfp4: x, w, block scales and out must be 2-D"));
        }
        let (m, k, n) = (x.shape[0], x.shape[1], w.shape[0]);
        if k % 16 != 0
            || w.shape[1] != k / 2
            || bs.shape[0] != n
            || bs.shape[1] != k / 16
            || gs.numel() != 1
            || out.shape[0] != m
            || out.shape[1] != n
        {
            return Err(Error::invalid("metal matmul_nvfp4: shape mismatch (K % 16 == 0)"));
        }
        if m == 0 || n == 0 || k == 0 {
            return Ok(());
        }
        let f32_out = match out.dtype {
            DType::Bf16 => false,
            DType::F32 => true,
            d => return Err(Error::invalid(format!("metal matmul_nvfp4: output dtype {d}"))),
        };
        // the shaders index with 32-bit math in places
        u32_of(m * k, "M*K")?;
        u32_of(n * (k / 2), "N*K/2")?;
        u32_of(m * n, "M*N")?;
        let xb = self.bind(x, DType::Bf16, false, "x")?;
        let wb = self.bind(w, DType::U8, false, "w")?;
        let sb = self.bind(bs, DType::Fp8E4m3, false, "block scales")?;
        let gb = self.bind(gs, DType::F32, false, "global scale")?;
        let ob = self.bind(out, if f32_out { DType::F32 } else { DType::Bf16 }, true, "out")?;
        // rows are K / 2 = 8 * (K / 16) bytes, so an 8-byte aligned base keeps every 8-byte weight load aligned
        if wb.skip != 0 || wb.off % 8 != 0 {
            return Err(Error::invalid(format!(
                "metal matmul_nvfp4: the weights must be 8-byte aligned (bound at byte {}, skip {})",
                wb.off, wb.skip
            )));
        }
        let p = OpP {
            dims: [u32_of(m, "M")?, u32_of(n, "N")?, u32_of(k, "K")?, 0, 0, 0, 0, 0],
            skips: [xb.skip, sb.skip, ob.skip, 0, 0],
            ..Default::default()
        };
        let (pso, grid, threads) = match kernel {
            MATMUL_NVFP4_GEMV => {
                if m > GEMV_MB {
                    return Err(Error::invalid(format!("metal matmul_nvfp4_gemv: M = {m} > {GEMV_MB}")));
                }
                // split-K for M = 1, a simdgroup per 2 rows with the activations re-read per row block for M = 2..=8
                let (name, grid, threads) = if m == 1 {
                    let (sg, r) = gv_pick(n, k);
                    let si = GV_SG.iter().position(|&g| g == sg).unwrap_or(0);
                    let ri = GV_R.iter().position(|&g| g == r).unwrap_or(0);
                    (GV_SHADERS[(si * 2 + f32_out as usize) * 3 + ri], (n.div_ceil(r), 1), sg * 32)
                } else {
                    let name = if f32_out { SHADERS[1] } else { SHADERS[0] };
                    (name, (n.div_ceil(GEMV_SG * GEMV_RS), 1), GEMV_SG * 32)
                };
                let pso = self.pso(name).ok_or_else(|| Error::backend(format!("metal: no pipeline '{name}'")))?;
                (pso, grid, threads)
            }
            MATMUL_NVFP4_MMA => {
                let name = if f32_out { SHADERS[3] } else { SHADERS[2] };
                let pso = self.pso(name).ok_or_else(|| Error::backend(format!("metal: no pipeline '{name}'")))?;
                (pso, (n.div_ceil(TILE), m.div_ceil(TILE)), MMA_THREADS)
            }
            _ => return Err(Error::NoKernel(format!("metal kernel '{kernel}' for op 'matmul_nvfp4'"))),
        };
        if pso.maxTotalThreadsPerThreadgroup() < threads {
            return Err(Error::backend(format!("metal: '{kernel}' cannot run {threads} threads")));
        }
        self.with_encoder(|enc| {
            // SAFETY: offsets lie inside the buffers (TensorRef bounds were checked by `OpCall::check`).
            unsafe {
                enc.setComputePipelineState(pso);
                for (i, b) in [&xb, &wb, &sb, &gb, &ob].into_iter().enumerate() {
                    enc.setBuffer_offset_atIndex(Some(&b.mtl), b.off, i);
                }
            }
            set_bytes(enc, &p, 5);
            enc.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize { width: grid.0, height: grid.1, depth: 1 },
                MTLSize { width: threads, height: 1, depth: 1 },
            );
        })
    }
}
