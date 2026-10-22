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
/// Fused decode GEMVs (M = 1), bitwise the unfused chains: `Op::MatmulNvfp4Glu` (gate + up GEMV + `silu_glu_bf16`) and
/// `Op::SharedDownCombine` (down GEMV + `moe_combine_bf16`).
pub const MATMUL_NVFP4_GLU_GEMV: &str = "matmul_nvfp4_glu_gemv";
pub const SHARED_DOWN_COMBINE_GEMV: &str = "shared_down_combine_gemv";

/// Shader entry points (kernels/metal/nvfp4.metal).
pub const SHADERS: [&str; 8] = [
    "nv_gemv_bf16_m1",
    "nv_gemv_bf16_m8",
    "nv_gemv_f32_m1",
    "nv_gemv_f32_m8",
    "nv_mma_bf16",
    "nv_mma_f32",
    "nv_glu_gemv_bf16",
    "nv_down_combine_gemv_bf16",
];

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

/// The fused decode GEMVs serve 1..=8 rows (one threadgroup row per activation row, 4 weight rows per simdgroup).
pub fn fused_ok(s: &ShapeInfo) -> bool {
    (1..=GEMV_MB).contains(&s.m) && s.k % 16 == 0
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
                let (name, rs, mb) = match (f32_out, m == 1) {
                    (false, true) => (SHADERS[0], GEMV_RS[0], 1),
                    (false, false) => (SHADERS[1], GEMV_RS[1], GEMV_MB),
                    (true, true) => (SHADERS[2], GEMV_RS[0], 1),
                    (true, false) => (SHADERS[3], GEMV_RS[1], GEMV_MB),
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

    /// `Op::MatmulNvfp4Glu` (`nv_glu_gemv_bf16_m1`) and `Op::SharedDownCombine` (`nv_down_combine_gemv_bf16_m1`): the unfused
    /// GEMV(s) plus their consumer in one dispatch, M <= 8, bf16 everywhere (bitwise the unfused launches).
    pub(crate) fn launch_nvfp4_fused(&self, kernel: &str, call: &OpCall) -> Result<()> {
        let glu = kernel == MATMUL_NVFP4_GLU_GEMV;
        let what = if glu { "matmul_nvfp4_glu" } else { "shared_down_combine" };
        let bad = |m: &str| Error::invalid(format!("metal {what}: {m}"));
        let (x, parts, extra, out) = match (glu, call.inputs.as_slice(), call.outputs.as_slice()) {
            (true, [x, gw, gbs, ggs, uw, ubs, ugs], [out]) => (x, vec![(gw, gbs, ggs), (uw, ubs, ugs)], None, out),
            (false, [x, w, bs, gs, routed, logit], [out]) => (x, vec![(w, bs, gs)], Some((routed, logit)), out),
            _ => return Err(bad("unexpected input / output count")),
        };
        if x.shape.len() != 2 || out.shape.len() != 2 {
            return Err(bad("x and out must be 2-D"));
        }
        let (m, k, n) = (x.shape[0], x.shape[1], parts[0].0.shape[0]);
        if m == 0 || m > GEMV_MB {
            return Err(bad(&format!("M = {m} (the fused GEMV serves 1..={GEMV_MB} rows)")));
        }
        if k % 16 != 0 || out.shape[0] != m || out.shape[1] != n {
            return Err(bad("shape mismatch (K % 16 == 0)"));
        }
        for (w, bs, gs) in &parts {
            if w.shape.len() != 2 || bs.shape.len() != 2 {
                return Err(bad("w and block scales must be 2-D"));
            }
            if w.shape[0] != n || w.shape[1] != k / 2 || bs.shape[0] != n || bs.shape[1] != k / 16 || gs.numel() != 1 {
                return Err(bad("shape mismatch (w [N,K/2], block scales [N,K/16], global scale [1])"));
            }
        }
        if let Some((routed, logit)) = extra {
            if routed.shape != [m, n] || logit.shape != [m, 1] {
                return Err(bad("routed must be [M,N] and the gate logit [M,1]"));
            }
        }
        if n == 0 || k == 0 {
            return Ok(());
        }
        // the shaders index with 32-bit math in places
        u32_of(m * k, "M*K")?;
        u32_of(n * (k / 2), "N*K/2")?;
        u32_of(m * n, "M*N")?;
        let xb = self.bind(x, DType::Bf16, false, "x")?;
        let mut skips = [xb.skip, 0, 0, 0, 0];
        let mut binds = vec![xb];
        for (j, (w, bs, gs)) in parts.iter().enumerate() {
            let wb = self.bind(w, DType::U8, false, "w")?;
            let sb = self.bind(bs, DType::Fp8E4m3, false, "block scales")?;
            let gb = self.bind(gs, DType::F32, false, "global scale")?;
            // rows are K / 2 = 8 * (K / 16) bytes, so an 8-byte aligned base keeps every 8-byte weight load aligned
            if wb.skip != 0 || wb.off % 8 != 0 {
                return Err(bad(&format!(
                    "the weights must be 8-byte aligned (bound at byte {}, skip {})",
                    wb.off, wb.skip
                )));
            }
            skips[1 + j] = sb.skip;
            binds.extend([wb, sb, gb]);
        }
        if let Some((routed, logit)) = extra {
            let rb = self.bind(routed, DType::Bf16, false, "routed")?;
            let lb = self.bind(logit, DType::Bf16, false, "gate logit")?;
            skips[2] = rb.skip;
            skips[3] = lb.skip;
            binds.extend([rb, lb]);
        }
        let ob = self.bind(out, DType::Bf16, true, "out")?;
        // skips: glu s0 x, s1 gate block scales, s2 up block scales, s3 out; down s0 a, s1 block scales, s2 routed, s3 logit, s4 out
        skips[if glu { 3 } else { 4 }] = ob.skip;
        binds.push(ob);
        let p = OpP {
            dims: [u32_of(m, "M")?, u32_of(n, "N")?, u32_of(k, "K")?, 0, 0, 0, 0, 0],
            skips,
            ..Default::default()
        };
        let name = if glu { SHADERS[6] } else { SHADERS[7] };
        let pso = self.pso(name).ok_or_else(|| Error::backend(format!("metal: no pipeline '{name}'")))?;
        let threads = GEMV_SG * 32;
        if pso.maxTotalThreadsPerThreadgroup() < threads {
            return Err(Error::backend(format!("metal: '{kernel}' cannot run {threads} threads")));
        }
        let groups = n.div_ceil(GEMV_SG * GEMV_RS[0]);
        let pi = binds.len();
        self.with_encoder(|enc| {
            // SAFETY: offsets lie inside the buffers (TensorRef bounds were checked by `OpCall::check`).
            unsafe {
                enc.setComputePipelineState(pso);
                for (i, b) in binds.iter().enumerate() {
                    enc.setBuffer_offset_atIndex(Some(&b.mtl), b.off, i);
                }
            }
            set_bytes(enc, &p, pi);
            enc.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize { width: groups, height: m, depth: 1 },
                MTLSize { width: threads, height: 1, depth: 1 },
            );
        })
    }
}
