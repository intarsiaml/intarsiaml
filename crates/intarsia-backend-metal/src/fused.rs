//! Launchers of the optional fused decode ops (`MatmulPair`). The model runs them when the registry has a kernel for the
//! exact call and falls back to the ops they replace otherwise (`Model::try_fused`).
//!
//! `MatmulPair`: `y0 = x W0^T`, `y1 = x W1^T` of one `x [1,K]`. When both GEMVs would take the same one of the two
//! default kernels (16-byte vector loads, or 8-byte loads for operands that are only 8-byte aligned, which is every weight
//! of a safetensors shard whose header length is 8 mod 16) it is one dispatch of `gemv_pair_bf16` / `gemv_pair_bf16_a8`
//! (kernels/metal/gemv_bf16.metal), bitwise the two `matmul_bf16_gemv` launches; anything else is two ordinary GEMV
//! dispatches (correct, not faster).

use crate::kernels::{
    DEFAULT_GEMV, DEFAULT_GEMV_A8, GEMV_PAIR_BF16, GEMV_PAIR_BF16_A8, GEMV_PAIR_ROWS_PER_TG, GEMV_PAIR_THREADS,
    RMSNORM_ADD_BF16, RMSNORM_ADD_GEMV_PAIR_BF16, RMSNORM_ADD_GEMV_PAIR_BF16_A8,
};
use crate::{encode_gemv, set_bytes, MetalBackend};
use intarsia_core::backend::TensorRef;
use intarsia_core::{Error, Op, OpAttrs, OpCall, Result};
use objc2_metal::{MTLComputeCommandEncoder, MTLSize};

/// Kernel arguments of `gemv_pair_bf16` (`PairParams` in gemv_bf16.metal; 32 bytes).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
struct PairParams {
    k: u32,
    n0: u32,
    n1: u32,
    y0_skip: u32,
    y1_skip: u32,
    pad: [u32; 3],
}

/// Kernel arguments of `rmsnorm_add_gemv_pair_bf16_rt_*` (`NormPairParams` in gemv_bf16.metal; 64 bytes).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
struct NormPairParams {
    k: u32,
    n0: u32,
    n1: u32,
    y0_skip: u32,
    y1_skip: u32,
    x_skip: u32,
    res_skip: u32,
    nw_skip: u32,
    normed_skip: u32,
    sum_skip: u32,
    pad0: u32,
    pad1: u32,
    eps: f32,
    w_offset: f32,
    pad2: f32,
    pad3: f32,
}

/// Largest K the norm-folded pair stages in threadgroup memory (K bf16 plus the shader's two 8-float partial arrays).
pub const NORM_PAIR_MAX_K: usize = (32768 - 256) / 2;

impl MetalBackend {
    /// `Op::RmsNormAddMatmulPair`: `[x, res, norm_w, w0, w1] -> [normed, sum, y0, y1]` for one row. When both GEMVs would take
    /// the same default kernel (as in `launch_matmul_pair`) it is one dispatch of `rmsnorm_add_gemv_pair_bf16_rt_*`, bitwise
    /// `rmsnorm_add_bf16` followed by the pair; otherwise those launches run one after the other. `normed` / `sum` must not
    /// overlap `x` / `res` (threadgroup 0 stores them while the others still read).
    pub(crate) fn launch_rmsnorm_add_matmul_pair(&self, call: &OpCall) -> Result<()> {
        let (x, res, nw, w0, w1, normed, sum, y0, y1) = match (call.inputs.as_slice(), call.outputs.as_slice()) {
            ([x, res, nw, w0, w1], [normed, sum, y0, y1]) => (x, res, nw, w0, w1, normed, sum, y0, y1),
            _ => return Err(Error::invalid("rmsnorm_add_matmul_pair: expected 5 inputs / 4 outputs")),
        };
        let bad = |what: &str| Error::invalid(format!("metal rmsnorm_add_matmul_pair: {what}"));
        if x.shape.len() != 2 || x.shape[0] != 1 || w0.shape.len() != 2 || w1.shape.len() != 2 {
            return Err(bad("one row of x and 2-D weights expected"));
        }
        let (k, n0, n1) = (x.shape[1], w0.shape[0], w1.shape[0]);
        if k == 0
            || k % 8 != 0
            || k > NORM_PAIR_MAX_K
            || w0.shape[1] != k
            || w1.shape[1] != k
            || res.shape != x.shape
            || nw.shape != [k]
            || normed.shape != x.shape
            || sum.shape != x.shape
            || y0.shape != [1, n0]
            || y1.shape != [1, n1]
        {
            return Err(bad("shape mismatch (K % 8 == 0, K <= 16256)"));
        }
        for o in [normed, sum] {
            if [x, res].into_iter().any(|i| crate::fp8::overlap(o, i)) {
                return Err(bad("normed / sum overlap x / res"));
            }
        }
        if n0 == 0 || n1 == 0 {
            return Err(bad("empty weight"));
        }
        // each half is planned exactly like the stand-alone `Matmul` on `normed` that the unfused chain runs
        let single = |w: &TensorRef, y: &TensorRef| {
            OpCall::new(Op::Matmul, vec![normed.clone(), w.clone()], vec![y.clone()], OpAttrs::default())
        };
        let (p0, p1) = (self.plan_gemv(&single(w0, y0))?, self.plan_gemv(&single(w1, y1))?);
        let shader = match (p0.name, p1.name) {
            (a, b) if a == DEFAULT_GEMV.name && b == a => Some(RMSNORM_ADD_GEMV_PAIR_BF16),
            (a, b) if a == DEFAULT_GEMV_A8.name && b == a => Some(RMSNORM_ADD_GEMV_PAIR_BF16_A8),
            _ => None,
        };
        let Some(shader) = shader else {
            let norm = OpCall::new(
                Op::RmsNormAdd,
                vec![x.clone(), res.clone(), nw.clone()],
                vec![normed.clone(), sum.clone()],
                call.attrs.clone(),
            );
            self.launch_dense(RMSNORM_ADD_BF16, &norm)?;
            let pair = OpCall::new(
                Op::MatmulPair,
                vec![normed.clone(), w0.clone(), w1.clone()],
                vec![y0.clone(), y1.clone()],
                OpAttrs::default(),
            );
            return self.launch_matmul_pair(&pair);
        };
        let bf = intarsia_core::DType::Bf16;
        let xb = self.bind(x, bf, false, "x")?;
        let rb = self.bind(res, bf, false, "res")?;
        let nb = self.bind(nw, bf, false, "norm w")?;
        let nrm = self.bind(normed, bf, true, "normed")?;
        let smb = self.bind(sum, bf, true, "sum")?;
        let pso = self.pso(shader).ok_or_else(|| Error::backend(format!("metal: no pipeline '{shader}'")))?;
        let params = NormPairParams {
            k: p0.params.k,
            n0: p0.params.n,
            n1: p1.params.n,
            y0_skip: p0.params.y_skip,
            y1_skip: p1.params.y_skip,
            x_skip: xb.skip,
            res_skip: rb.skip,
            nw_skip: nb.skip,
            normed_skip: nrm.skip,
            sum_skip: smb.skip,
            eps: call.attrs.eps,
            w_offset: call.attrs.w_offset,
            ..Default::default()
        };
        let groups = p0.params.n.div_ceil(GEMV_PAIR_ROWS_PER_TG) + p1.params.n.div_ceil(GEMV_PAIR_ROWS_PER_TG);
        let tg_bytes = (k * 2).next_multiple_of(16);
        self.with_encoder(|enc| {
            enc.setComputePipelineState(pso);
            // SAFETY: offsets are inside the buffers (`plan_gemv` / `bind` checked them) and aligned for the loads.
            unsafe {
                enc.setBuffer_offset_atIndex(Some(&p0.w.0), p0.w.1, 0);
                enc.setBuffer_offset_atIndex(Some(&p1.w.0), p1.w.1, 1);
                for (i, b) in [&xb, &rb, &nb, &nrm, &smb].into_iter().enumerate() {
                    enc.setBuffer_offset_atIndex(Some(&b.mtl), b.off, 2 + i);
                }
                enc.setBuffer_offset_atIndex(Some(&p0.y.0), p0.y.1, 7);
                enc.setBuffer_offset_atIndex(Some(&p1.y.0), p1.y.1, 8);
                enc.setThreadgroupMemoryLength_atIndex(tg_bytes, 0);
            }
            set_bytes(enc, &params, 9);
            enc.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize { width: groups as usize, height: 1, depth: 1 },
                MTLSize { width: GEMV_PAIR_THREADS as usize, height: 1, depth: 1 },
            );
        })
    }

    pub(crate) fn launch_matmul_pair(&self, call: &OpCall) -> Result<()> {
        let (x, w0, w1, y0, y1) = match (call.inputs.as_slice(), call.outputs.as_slice()) {
            ([x, w0, w1], [y0, y1]) => (x, w0, w1, y0, y1),
            _ => return Err(Error::invalid("matmul_pair: expected 3 inputs / 2 outputs")),
        };
        // each half is validated and planned exactly like a stand-alone `Matmul`
        let single = |w: &TensorRef, y: &TensorRef| {
            OpCall::new(Op::Matmul, vec![x.clone(), w.clone()], vec![y.clone()], OpAttrs::default())
        };
        let (p0, p1) = (self.plan_gemv(&single(w0, y0))?, self.plan_gemv(&single(w1, y1))?);
        let pso = |name: &str| self.pso(name).ok_or_else(|| Error::backend(format!("metal: no pipeline '{name}'")));
        // both halves on the same GEMV kernel and that kernel has a pair form: one dispatch
        let shader = match (p0.name, p1.name) {
            (a, b) if a == DEFAULT_GEMV.name && b == a => Some(GEMV_PAIR_BF16),
            (a, b) if a == DEFAULT_GEMV_A8.name && b == a => Some(GEMV_PAIR_BF16_A8),
            _ => None,
        };
        if let Some(shader) = shader {
            let pair = pso(shader)?;
            let params = PairParams {
                k: p0.params.k,
                n0: p0.params.n,
                n1: p1.params.n,
                y0_skip: p0.params.y_skip,
                y1_skip: p1.params.y_skip,
                pad: [0; 3],
            };
            let groups = p0.params.n.div_ceil(GEMV_PAIR_ROWS_PER_TG) + p1.params.n.div_ceil(GEMV_PAIR_ROWS_PER_TG);
            return self.with_encoder(|enc| {
                enc.setComputePipelineState(pair);
                // SAFETY: offsets are inside the buffers (`plan_gemv` bound them) and 16-byte aligned for the loads.
                unsafe {
                    enc.setBuffer_offset_atIndex(Some(&p0.w.0), p0.w.1, 0);
                    enc.setBuffer_offset_atIndex(Some(&p1.w.0), p1.w.1, 1);
                    enc.setBuffer_offset_atIndex(Some(&p0.x.0), p0.x.1, 2);
                    enc.setBuffer_offset_atIndex(Some(&p0.y.0), p0.y.1, 3);
                    enc.setBuffer_offset_atIndex(Some(&p1.y.0), p1.y.1, 4);
                }
                set_bytes(enc, &params, 5);
                enc.dispatchThreadgroups_threadsPerThreadgroup(
                    MTLSize { width: groups as usize, height: 1, depth: 1 },
                    MTLSize { width: GEMV_PAIR_THREADS as usize, height: 1, depth: 1 },
                );
            });
        }
        for p in [&p0, &p1] {
            let pso = pso(p.name)?;
            self.with_encoder(|enc| {
                encode_gemv(enc, pso, (&p.w.0, p.w.1), (&p.x.0, p.x.1), (&p.y.0, p.y.1), &p.params, p.groups, p.threads)
            })?;
        }
        Ok(())
    }
}
