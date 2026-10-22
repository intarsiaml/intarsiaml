//! Launchers of `matmul_fp8_gemv` / `matmul_fp8_mma` (`Op::MatmulFp8`, kernels/metal/fp8.metal; the port of kernels/hip/fp8.hip).
//!
//! `out[M,N] = q(x)[M,K] W^T * ws * is`, W = e4m3 bytes `[N,K]`, `scales [2] f32 = (ws, is)` read on the GPU: `is > 0` is W8A8
//! static (q(x) = e4m3(x / is)), else W8A16. The decode GEMV (M <= 8) is one dispatch (M = 1: split-K, each thread
//! quantises its own activations; M >= 2: q(x) staged in threadgroup memory); the GEMM (any M) first QDQs x once into the
//! plan-reserved scratch (bf16 `[M,K]`, exact) and then runs a tiled
//! `simdgroup_matrix` bf16 GEMM that dequantises the weight tiles.

use crate::dense::OpP;
use crate::{set_bytes, MetalBackend};
use intarsia_core::{DType, Error, OpCall, Result, ShapeInfo};
use objc2_metal::{MTLComputeCommandEncoder, MTLComputePipelineState, MTLSize};

/// `KernelImpl::name` of the kernels.
pub const MATMUL_FP8_GEMV: &str = "matmul_fp8_gemv";
pub const MATMUL_FP8_MMA: &str = "matmul_fp8_mma";

const XPREP: &str = "fp8_xprep_bf16";
/// Shader entry points (kernels/metal/fp8.metal).
pub const SHADERS: [&str; 5] = [XPREP, "fp8_gemv_bf16_m8", "fp8_gemv_f32_m8", "fp8_mma_bf16", "fp8_mma_f32"];

/// Split-K M = 1 GEMV entry points `fp8_gv_{bf16,f32}_s<SG>r<R>`, index `(sg_i * 2 + f32) * 4 + r_i` over `GV_SG` / `GV_R`.
pub const GV_SHADERS: [&str; 32] = [
    "fp8_gv_bf16_s1r1",
    "fp8_gv_bf16_s1r2",
    "fp8_gv_bf16_s1r4",
    "fp8_gv_bf16_s1r8",
    "fp8_gv_f32_s1r1",
    "fp8_gv_f32_s1r2",
    "fp8_gv_f32_s1r4",
    "fp8_gv_f32_s1r8",
    "fp8_gv_bf16_s2r1",
    "fp8_gv_bf16_s2r2",
    "fp8_gv_bf16_s2r4",
    "fp8_gv_bf16_s2r8",
    "fp8_gv_f32_s2r1",
    "fp8_gv_f32_s2r2",
    "fp8_gv_f32_s2r4",
    "fp8_gv_f32_s2r8",
    "fp8_gv_bf16_s4r1",
    "fp8_gv_bf16_s4r2",
    "fp8_gv_bf16_s4r4",
    "fp8_gv_bf16_s4r8",
    "fp8_gv_f32_s4r1",
    "fp8_gv_f32_s4r2",
    "fp8_gv_f32_s4r4",
    "fp8_gv_f32_s4r8",
    "fp8_gv_bf16_s8r1",
    "fp8_gv_bf16_s8r2",
    "fp8_gv_bf16_s8r4",
    "fp8_gv_bf16_s8r8",
    "fp8_gv_f32_s8r1",
    "fp8_gv_f32_s8r2",
    "fp8_gv_f32_s8r4",
    "fp8_gv_f32_s8r8",
];
const GV_SG: [usize; 4] = [1, 2, 4, 8];
const GV_R: [usize; 4] = [1, 2, 4, 8];
/// Threads the split-K GEMV launches at least (when N allows): enough to fill the GPU.
const GV_THREADS: usize = 32768;

/// `_m8` GEMV (M = 2..=8): simdgroups per threadgroup (`FP8_SG`), output rows per simdgroup (`RS`), activation rows (`MB`).
const GEMV_SG: usize = 8;
const GEMV_RS: usize = 2;
const GEMV_MB: usize = 8;
/// Largest threadgroup memory a GEMV may stage its activations in (Metal's per-threadgroup limit).
pub const GEMV_TG_BYTES: usize = 32768;
/// GEMM tile (`FP8_BM` / `FP8_BN`) and threads per threadgroup.
const TILE: usize = 64;
const MMA_THREADS: usize = 128;

/// Largest M of the decode GEMV for K: the staged activations (`M * K` bf16) must fit threadgroup memory.
pub fn gemv_ok(s: &ShapeInfo) -> bool {
    (1..=GEMV_MB).contains(&s.m) && s.k % 16 == 0 && s.m * s.k * 2 <= GEMV_TG_BYTES
}

/// GEMM scratch: the QDQ'd activations as bf16 `[M,K]`.
pub fn scratch(s: &ShapeInfo) -> usize {
    s.m * s.k * 2 + 16
}

/// (simdgroups per threadgroup, rows per threadgroup) of the split-K GEMV: one 8-byte chunk of a row per thread (SG from K),
/// and the most rows per threadgroup (each threadgroup quantises the activations once) that still launches `GV_THREADS`.
fn gv_pick(n: usize, k: usize) -> (usize, usize) {
    let sg = GV_SG.into_iter().rev().find(|sg| k / 8 >= 32 * sg).unwrap_or(1);
    let r = GV_R.into_iter().rev().find(|r| n.div_ceil(*r) * sg * 32 >= GV_THREADS).unwrap_or(1);
    (sg, r)
}

fn u32_of(v: usize, what: &str) -> Result<u32> {
    u32::try_from(v).map_err(|_| Error::invalid(format!("metal matmul_fp8: {what} = {v} exceeds u32")))
}

impl MetalBackend {
    pub(crate) fn launch_fp8(&self, kernel: &str, call: &OpCall) -> Result<()> {
        let (x, w, sc, out) = match (call.inputs.as_slice(), call.outputs.as_slice()) {
            ([x, w, sc], [out]) => (x, w, sc, out),
            _ => return Err(Error::invalid("metal matmul_fp8: expected 3 inputs / 1 output")),
        };
        if x.shape.len() != 2 || w.shape.len() != 2 || out.shape.len() != 2 {
            return Err(Error::invalid("metal matmul_fp8: x, w and out must be 2-D"));
        }
        let (m, k, n) = (x.shape[0], x.shape[1], w.shape[0]);
        if k % 16 != 0 || w.shape[1] != k || sc.numel() != 2 || out.shape[0] != m || out.shape[1] != n {
            return Err(Error::invalid("metal matmul_fp8: shape mismatch (K % 16 == 0, scales [2])"));
        }
        if m == 0 || n == 0 {
            return Ok(());
        }
        let f32_out = match out.dtype {
            DType::Bf16 => false,
            DType::F32 => true,
            d => return Err(Error::invalid(format!("metal matmul_fp8: output dtype {d}"))),
        };
        // the shaders index with 32-bit math in places
        u32_of(m * k, "M*K")?;
        u32_of(n * k, "N*K")?;
        u32_of(m * n, "M*N")?;
        let xb = self.bind(x, DType::Bf16, false, "x")?;
        let wb = self.bind(w, DType::Fp8E4m3, false, "w")?;
        let sb = self.bind(sc, DType::F32, false, "scales")?;
        let ob = self.bind(out, if f32_out { DType::F32 } else { DType::Bf16 }, true, "out")?;
        if wb.skip != 0 || wb.off % 8 != 0 {
            return Err(Error::invalid(format!(
                "metal matmul_fp8: the weights must be 8-byte aligned (bound at byte {}, skip {})",
                wb.off, wb.skip
            )));
        }
        let pdims = [u32_of(m, "M")?, u32_of(n, "N")?, u32_of(k, "K")?, 0, 0, 0, 0, 0];
        if kernel == MATMUL_FP8_GEMV {
            if m > GEMV_MB || m * k * 2 > GEMV_TG_BYTES {
                return Err(Error::invalid(format!("metal matmul_fp8_gemv: M = {m}, K = {k} does not fit")));
            }
            let p = OpP { dims: pdims, skips: [xb.skip, ob.skip, 0, 0, 0], ..Default::default() };
            // split-K for M = 1; M = 2..=8 stage the quantised rows in threadgroup memory
            let (name, grid, threads, tg_bytes) = if m == 1 {
                let (sg, r) = gv_pick(n, k);
                let si = GV_SG.iter().position(|&g| g == sg).unwrap_or(0);
                let ri = GV_R.iter().position(|&g| g == r).unwrap_or(0);
                (GV_SHADERS[(si * 2 + f32_out as usize) * 4 + ri], n.div_ceil(r), sg * 32, 0)
            } else {
                let name = if f32_out { SHADERS[2] } else { SHADERS[1] };
                (name, n.div_ceil(GEMV_SG * GEMV_RS), GEMV_SG * 32, (m * k * 2).next_multiple_of(16))
            };
            let pso = self.pso(name).ok_or_else(|| Error::backend(format!("metal: no pipeline '{name}'")))?;
            if pso.maxTotalThreadsPerThreadgroup() < threads {
                return Err(Error::backend(format!("metal: '{name}' cannot run {threads} threads")));
            }
            return self.with_encoder(|enc| {
                // SAFETY: offsets lie inside the buffers (TensorRef bounds were checked by `OpCall::check`).
                unsafe {
                    enc.setComputePipelineState(pso);
                    for (i, b) in [&xb, &wb, &sb, &ob].into_iter().enumerate() {
                        enc.setBuffer_offset_atIndex(Some(&b.mtl), b.off, i);
                    }
                    if tg_bytes > 0 {
                        enc.setThreadgroupMemoryLength_atIndex(tg_bytes, 0);
                    }
                }
                set_bytes(enc, &p, 4);
                enc.dispatchThreadgroups_threadsPerThreadgroup(
                    MTLSize { width: grid, height: 1, depth: 1 },
                    MTLSize { width: threads, height: 1, depth: 1 },
                );
            });
        }
        if kernel != MATMUL_FP8_MMA {
            return Err(Error::NoKernel(format!("metal kernel '{kernel}' for op 'matmul_fp8'")));
        }
        let need = m * k * 2;
        let scratch = call
            .scratch
            .as_ref()
            .ok_or_else(|| Error::invalid("metal matmul_fp8_mma: the scratch workspace (OpCall::scratch) is missing"))?;
        if scratch.numel() * scratch.dtype.size_bytes() < need {
            return Err(Error::invalid(format!(
                "metal matmul_fp8_mma: scratch of {} bytes, need {need}",
                scratch.numel() * scratch.dtype.size_bytes()
            )));
        }
        let xs = self.bind(scratch, DType::U8, true, "scratch")?;
        if xs.skip != 0 {
            return Err(Error::invalid("metal matmul_fp8_mma: the scratch must be 4-byte aligned"));
        }
        let gemm = if f32_out { SHADERS[4] } else { SHADERS[3] };
        let pso_x = self.pso(XPREP).ok_or_else(|| Error::backend(format!("metal: no pipeline '{XPREP}'")))?;
        let pso_g = self.pso(gemm).ok_or_else(|| Error::backend(format!("metal: no pipeline '{gemm}'")))?;
        if pso_g.maxTotalThreadsPerThreadgroup() < MMA_THREADS {
            return Err(Error::backend(format!("metal: '{gemm}' cannot run {MMA_THREADS} threads")));
        }
        let px = OpP { dims: pdims, skips: [xb.skip, 0, 0, 0, 0], ..Default::default() };
        let pg = OpP { skips: [0, ob.skip, 0, 0, 0], ..px };
        let tx = pso_x.maxTotalThreadsPerThreadgroup().min(256);
        self.with_encoder(|enc| {
            // SAFETY: offsets lie inside the buffers (TensorRef bounds were checked by `OpCall::check`, the scratch size above).
            unsafe {
                enc.setComputePipelineState(pso_x);
                for (i, b) in [&xb, &sb, &xs].into_iter().enumerate() {
                    enc.setBuffer_offset_atIndex(Some(&b.mtl), b.off, i);
                }
            }
            set_bytes(enc, &px, 3);
            enc.dispatchThreads_threadsPerThreadgroup(
                MTLSize { width: (m * k).div_ceil(8), height: 1, depth: 1 },
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
                MTLSize { width: n.div_ceil(TILE), height: m.div_ceil(TILE), depth: 1 },
                MTLSize { width: MMA_THREADS, height: 1, depth: 1 },
            );
        })
    }
}
