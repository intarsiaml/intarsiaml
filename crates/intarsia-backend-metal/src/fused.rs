//! Launchers of the optional fused decode ops (`MatmulPair`). The model runs them when the registry has a kernel for the
//! exact call and falls back to the ops they replace otherwise (`Model::try_fused`).
//!
//! `MatmulPair`: `y0 = x W0^T`, `y1 = x W1^T` of one `x [1,K]`. When both GEMVs would take the default 16-byte vector
//! kernel it is one dispatch of `gemv_pair_bf16` (kernels/metal/gemv_bf16.metal), bitwise the two `matmul_bf16_gemv`
//! launches; any other operand alignment is two ordinary GEMV dispatches (correct, not faster).

use crate::kernels::{DEFAULT_GEMV, GEMV_PAIR_BF16, GEMV_PAIR_ROWS_PER_TG, GEMV_PAIR_THREADS};
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

impl MetalBackend {
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
        if p0.name == DEFAULT_GEMV.name && p1.name == DEFAULT_GEMV.name {
            let pair = pso(GEMV_PAIR_BF16)?;
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
