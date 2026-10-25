//! Launchers of `matmul_nvfp4_gemv` / `matmul_nvfp4_mma` (`Op::MatmulNvfp4`, kernels/metal/nvfp4.metal; the port of
//! kernels/hip/nvfp4.hip).
//!
//! `out[M,N] = x[M,K] W^T`, W = e2m1 `u8 [N,K/2]` * e4m3 block scales `[N,K/16]` * global scale `[1] f32` (read on the GPU).
//! The decode GEMV (M <= 8) is one dispatch without scratch; the GEMM (any M) is a tiled `simdgroup_matrix` bf16 GEMM that
//! dequantises the weight tiles (code * block scale, exact in bf16) and applies the global scale in the epilogue.

use crate::dense::OpP;
use crate::{set_bytes, MetalBackend};
use intarsia_core::{DType, Error, OpCall, Result, ShapeInfo};
use objc2_metal::{MTLComputeCommandEncoder, MTLComputePipelineState, MTLSize};

/// `KernelImpl::name` of the kernels.
pub const MATMUL_NVFP4_GEMV: &str = "matmul_nvfp4_gemv";
pub const MATMUL_NVFP4_MMA: &str = "matmul_nvfp4_mma";
/// `rows_exact` multi-row GEMV (continuous batching of the hybrid 35B, docs/results/2026-10-23-mb-metal-hybrid-batching.md):
/// row r of an M-row launch (M = 2..=8) is bitwise the M = 1 `matmul_nvfp4_gemv` launch on row r alone (tests/rows_exact.rs).
pub const MATMUL_NVFP4_GEMV_ROWS: &str = "matmul_nvfp4_gemv_rows";

/// Shader entry points (kernels/metal/nvfp4.metal).
pub const SHADERS: [&str; 24] = [
    "nv_gemv_bf16_m1",
    "nv_gemv_bf16_m8",
    "nv_gemv_f32_m1",
    "nv_gemv_f32_m8",
    "nv_mma_bf16",
    "nv_mma_f32",
    "nvx_bf16_m2_r1",
    "nvx_f32_m2_r1",
    "nvx_bf16_m2_r2",
    "nvx_f32_m2_r2",
    "nvx_bf16_m2_r4",
    "nvx_f32_m2_r4",
    "nvx_bf16_m4_r1",
    "nvx_f32_m4_r1",
    "nvx_bf16_m4_r2",
    "nvx_f32_m4_r2",
    "nvx_bf16_m4_r4",
    "nvx_f32_m4_r4",
    "nvx_bf16_m8_r1",
    "nvx_f32_m8_r1",
    "nvx_bf16_m8_r2",
    "nvx_f32_m8_r2",
    "nvx_bf16_m8_r4",
    "nvx_f32_m8_r4",
];

/// The widest matrix whose multi-row GEMV runs one row per threadgroup row (lane mm): an N-row matrix with N at most this
/// gets the one-row shader over a grid of M threadgroup rows (each row is then the one-row launch itself; the weights come
/// again from cache), wider ones the 8-row shader. The 35B's shared-expert matrices (512 / 2048 rows) give the 8-row shader
/// only 32 / 128 threadgroups. `INTARSIA_METAL_NV_ROWS_SPLIT=0` keeps the 8-row shader everywhere.
const NARROW_N: usize = 4096;

fn nv_rows_mb(n: usize) -> usize {
    static SPLIT: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let split = *SPLIT.get_or_init(|| std::env::var("INTARSIA_METAL_NV_ROWS_SPLIT").map_or(true, |v| v != "0"));
    if split && n <= NARROW_N {
        1
    } else {
        GEMV_MB
    }
}

/// Simdgroups per GEMV threadgroup (`NV_SG`), output rows per simdgroup of `_m1` / `_m8` (`RS`), activation rows of `_m8`.
const GEMV_SG: usize = 8;
const GEMV_RS: [usize; 2] = [4, 2];
const GEMV_MB: usize = 8;
/// GEMM tile (`NV_BM` / `NV_BN`) and threads per threadgroup.
const TILE: usize = 64;
const MMA_THREADS: usize = 128;

/// The decode GEMV serves 1..=8 rows.
pub fn gemv_ok(s: &ShapeInfo) -> bool {
    (1..=GEMV_MB).contains(&s.m) && s.k % 16 == 0
}

/// The row-exact multi-row GEMV serves 2..=8 rows.
pub fn rows_ok(s: &ShapeInfo) -> bool {
    (2..=GEMV_MB).contains(&s.m) && s.k % 16 == 0
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
            MATMUL_NVFP4_GEMV | MATMUL_NVFP4_GEMV_ROWS => {
                if m > GEMV_MB || (kernel == MATMUL_NVFP4_GEMV_ROWS && m < 2) {
                    return Err(Error::invalid(format!("metal {kernel}: M = {m} outside the kernel's rows")));
                }
                let mb = if m == 1 {
                    1
                } else if kernel == MATMUL_NVFP4_GEMV_ROWS {
                    nv_rows_mb(n)
                } else {
                    GEMV_MB
                };
                let (name, rs) = match (f32_out, mb) {
                    (false, 1) => (SHADERS[0], GEMV_RS[0]),
                    (false, _) => (SHADERS[1], GEMV_RS[1]),
                    (true, 1) => (SHADERS[2], GEMV_RS[0]),
                    (true, _) => (SHADERS[3], GEMV_RS[1]),
                };
                let exp = (kernel == MATMUL_NVFP4_GEMV_ROWS && m > 1).then(nv_exp).flatten();
                let (name, rs, mb) = match exp {
                    Some((emb, ers)) => {
                        let n = SHADERS.iter().find(|s| **s == format!("nvx_{}_m{emb}_r{ers}", if f32_out { "f32" } else { "bf16" }));
                        (*n.unwrap_or(&name), ers, emb)
                    }
                    None => (name, rs, mb),
                };
                let pso = self.pso(name).ok_or_else(|| Error::backend(format!("metal: no pipeline '{name}'")))?;
                (pso, (n.div_ceil(GEMV_SG * rs), m.div_ceil(mb)), GEMV_SG * 32)
            }
            MATMUL_NVFP4_MMA => {
                let name = if f32_out { SHADERS[5] } else { SHADERS[4] };
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

/// Experiments of lane mm step 2: `INTARSIA_METAL_NV_EXP=MB,RS` picks `nvx_{bf16|f32}_m{MB}_r{RS}` for 2+ rows.
fn nv_exp() -> Option<(usize, usize)> {
    static V: std::sync::OnceLock<Option<(usize, usize)>> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        let v = std::env::var("INTARSIA_METAL_NV_EXP").ok()?;
        let p: Vec<usize> = v.split(',').filter_map(|t| t.trim().parse().ok()).collect();
        (p.len() == 2).then(|| (p[0], p[1]))
    })
}
