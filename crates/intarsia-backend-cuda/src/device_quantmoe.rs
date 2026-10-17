//! Device dispatch of the quantised-weight and MoE launchers (see `quantmoe.rs` for the table).

use crate::device::CudaBackend;
use crate::ffi::{status_name, IkTensor};
use intarsia_core::backend::StreamId;
use intarsia_core::{Error, OpCall, Result};
use std::ffi::{c_int, c_void};

/// Mirror of `ik_scratch` in kernels/abi/ik_abi.h.
#[repr(C)]
struct IkScratch {
    data: *mut c_void,
    bytes: i64,
}

type T = *const IkTensor;
type TM = *mut IkTensor;
type S = *mut c_void;

extern "C" {
    fn ik_matmul_fp8_gemv(x: T, w: T, scales: T, out: TM, s: S) -> c_int;
    fn ik_matmul_fp8_mma(x: T, w: T, scales: T, out: TM, s: S) -> c_int;
    fn ik_matmul_fp8_gemm(x: T, w: T, scales: T, out: TM, scratch: *const IkScratch, s: S) -> c_int;
    fn ik_matmul_fp8_gemm_f8(x: T, w: T, scales: T, out: TM, scratch: *const IkScratch, s: S) -> c_int;
    fn ik_matmul_nvfp4_gemv(x: T, w: T, bs: T, gs: T, out: TM, s: S) -> c_int;
    fn ik_matmul_nvfp4_mma(x: T, w: T, bs: T, gs: T, out: TM, s: S) -> c_int;
    fn ik_matmul_nvfp4_gemm(x: T, w: T, bs: T, gs: T, out: TM, s: S) -> c_int;
    fn ik_moe_topk_bf16(logits: T, top_k: c_int, norm: c_int, weights: TM, ids: TM, s: S) -> c_int;
    #[allow(clippy::too_many_arguments)]
    fn ik_moe_experts_bf16(
        x: T,
        ids: T,
        weights: T,
        gate_up_w: T,
        gate_up_s: T,
        down_w: T,
        down_s: T,
        gscales: T,
        out: TM,
        scratch: *const IkScratch,
        s: S,
    ) -> c_int;
    #[allow(clippy::too_many_arguments)]
    fn ik_moe_experts_mma_bf16(
        x: T,
        ids: T,
        weights: T,
        gate_up_w: T,
        gate_up_s: T,
        down_w: T,
        down_s: T,
        gscales: T,
        out: TM,
        scratch: *const IkScratch,
        s: S,
    ) -> c_int;
    #[allow(clippy::too_many_arguments)]
    fn ik_moe_experts_nvfp4_mma_bf16(
        x: T,
        ids: T,
        weights: T,
        gate_up_w: T,
        gate_up_s: T,
        down_w: T,
        down_s: T,
        gscales: T,
        out: TM,
        scratch: *const IkScratch,
        s: S,
    ) -> c_int;
    #[allow(clippy::too_many_arguments)]
    fn ik_moe_experts_decode_bf16(
        x: T,
        ids: T,
        weights: T,
        gate_up_w: T,
        gate_up_s: T,
        down_w: T,
        down_s: T,
        gscales: T,
        out: TM,
        scratch: *const IkScratch,
        s: S,
    ) -> c_int;
    #[allow(clippy::too_many_arguments)]
    fn ik_moe_experts_decode_v2_bf16(
        x: T,
        ids: T,
        weights: T,
        gate_up_w: T,
        gate_up_s: T,
        down_w: T,
        down_s: T,
        gscales: T,
        out: TM,
        scratch: *const IkScratch,
        s: S,
    ) -> c_int;
    fn ik_moe_combine_bf16(routed: T, shared: T, gate: T, out: TM, s: S) -> c_int;
    fn ik_matmul_pair_bf16_gemv(x: T, w0: T, w1: T, out0: TM, out1: TM, s: S) -> c_int;
    fn ik_matmul_fp8_pair_gemv(x: T, w0: T, sc0: T, w1: T, sc1: T, out0: TM, out1: TM, s: S) -> c_int;
    #[allow(clippy::too_many_arguments)]
    fn ik_matmul_nvfp4_glu_gemv(x: T, wg: T, bsg: T, gsg: T, wu: T, bsu: T, gsu: T, out: TM, s: S) -> c_int;
    fn ik_gated_rmsnorm_bf16(x: T, z: T, w: T, eps: f32, w_offset: f32, out: TM, s: S) -> c_int;
    fn ik_gated_rmsnorm_matmul_fp8_gemv(x: T, z: T, nw: T, eps: f32, w_offset: f32, w: T, scales: T, out: TM, s: S) -> c_int;
    #[allow(clippy::too_many_arguments)]
    fn ik_rmsnorm_add_matmul_pair_bf16_gemv(
        x: T,
        res: T,
        w: T,
        eps: f32,
        w_offset: f32,
        w0: T,
        w1: T,
        normed: TM,
        sum: TM,
        out0: TM,
        out1: TM,
        s: S,
    ) -> c_int;
}

impl CudaBackend {
    /// `Some(result)` when `kernel` is one of the quantised/MoE launchers, `None` otherwise.
    pub(crate) fn launch_quantmoe(&self, kernel: &str, call: &OpCall, stream: StreamId) -> Option<Result<()>> {
        let (_, op) = crate::quantmoe::TABLE.iter().find(|(n, _)| *n == kernel)?;
        Some(self.launch_quantmoe_inner(kernel, *op, call, stream))
    }

    fn launch_quantmoe_inner(&self, kernel: &str, op: intarsia_core::Op, call: &OpCall, stream: StreamId) -> Result<()> {
        if op != call.op {
            return Err(Error::invalid(format!("kernel '{kernel}' implements {op}, not {}", call.op)));
        }
        self.bind()?;
        let s = self.stream(stream)?;
        let inp: Vec<IkTensor> = call.inputs.iter().map(Self::tensor).collect::<Result<_>>()?;
        let mut out: Vec<IkTensor> = call.outputs.iter().map(Self::tensor).collect::<Result<_>>()?;
        let a = &call.attrs;
        let i = |n: usize| &inp[n] as *const IkTensor;
        let o = out.as_mut_ptr();
        let scratch = match &call.scratch {
            Some(t) => {
                let t = Self::tensor(t)?;
                Some(IkScratch { data: t.data, bytes: t.shape[0] })
            }
            None => None,
        };
        // SAFETY: tensors are bounds-checked views of live device buffers (held by `call`); the launchers validate
        // dtype/rank/stride/alignment and return an error status instead of faulting, and never synchronise or allocate.
        let rc = unsafe {
            match kernel {
                "matmul_fp8_gemv" => ik_matmul_fp8_gemv(i(0), i(1), i(2), o, s),
                "matmul_fp8_mma" => ik_matmul_fp8_mma(i(0), i(1), i(2), o, s),
                "matmul_fp8_gemm" => ik_matmul_fp8_gemm(
                    i(0),
                    i(1),
                    i(2),
                    o,
                    scratch.as_ref().map_or(std::ptr::null(), |p| p as *const IkScratch),
                    s,
                ),
                "matmul_fp8_gemm_f8" => ik_matmul_fp8_gemm_f8(
                    i(0),
                    i(1),
                    i(2),
                    o,
                    scratch.as_ref().map_or(std::ptr::null(), |p| p as *const IkScratch),
                    s,
                ),
                "matmul_nvfp4_gemv" => ik_matmul_nvfp4_gemv(i(0), i(1), i(2), i(3), o, s),
                "matmul_nvfp4_mma" => ik_matmul_nvfp4_mma(i(0), i(1), i(2), i(3), o, s),
                "matmul_nvfp4_gemm" => ik_matmul_nvfp4_gemm(i(0), i(1), i(2), i(3), o, s),
                "moe_topk_bf16" => ik_moe_topk_bf16(i(0), a.top_k as c_int, a.norm_topk as c_int, o, o.add(1), s),
                "moe_experts_bf16" => ik_moe_experts_bf16(
                    i(0),
                    i(1),
                    i(2),
                    i(3),
                    i(4),
                    i(5),
                    i(6),
                    i(7),
                    o,
                    scratch.as_ref().map_or(std::ptr::null(), |p| p as *const IkScratch),
                    s,
                ),
                "moe_experts_mma_bf16" => ik_moe_experts_mma_bf16(
                    i(0),
                    i(1),
                    i(2),
                    i(3),
                    i(4),
                    i(5),
                    i(6),
                    i(7),
                    o,
                    scratch.as_ref().map_or(std::ptr::null(), |p| p as *const IkScratch),
                    s,
                ),
                "moe_experts_nvfp4_mma_bf16" => ik_moe_experts_nvfp4_mma_bf16(
                    i(0),
                    i(1),
                    i(2),
                    i(3),
                    i(4),
                    i(5),
                    i(6),
                    i(7),
                    o,
                    scratch.as_ref().map_or(std::ptr::null(), |p| p as *const IkScratch),
                    s,
                ),
                "moe_experts_decode_bf16" => ik_moe_experts_decode_bf16(
                    i(0),
                    i(1),
                    i(2),
                    i(3),
                    i(4),
                    i(5),
                    i(6),
                    i(7),
                    o,
                    scratch.as_ref().map_or(std::ptr::null(), |p| p as *const IkScratch),
                    s,
                ),
                "moe_experts_decode_v2_bf16" => ik_moe_experts_decode_v2_bf16(
                    i(0),
                    i(1),
                    i(2),
                    i(3),
                    i(4),
                    i(5),
                    i(6),
                    i(7),
                    o,
                    scratch.as_ref().map_or(std::ptr::null(), |p| p as *const IkScratch),
                    s,
                ),
                "moe_combine_bf16" => ik_moe_combine_bf16(i(0), i(1), i(2), o, s),
                "matmul_pair_bf16_gemv" => ik_matmul_pair_bf16_gemv(i(0), i(1), i(2), o, o.add(1), s),
                "matmul_fp8_pair_gemv" => ik_matmul_fp8_pair_gemv(i(0), i(1), i(2), i(3), i(4), o, o.add(1), s),
                "matmul_nvfp4_glu_gemv" => ik_matmul_nvfp4_glu_gemv(i(0), i(1), i(2), i(3), i(4), i(5), i(6), o, s),
                "gated_rmsnorm_bf16" => ik_gated_rmsnorm_bf16(i(0), i(1), i(2), a.eps, a.w_offset, o, s),
                "gated_rmsnorm_matmul_fp8_gemv" => {
                    ik_gated_rmsnorm_matmul_fp8_gemv(i(0), i(1), i(2), a.eps, a.w_offset, i(3), i(4), o, s)
                }
                "rmsnorm_add_matmul_pair_bf16_gemv" => ik_rmsnorm_add_matmul_pair_bf16_gemv(
                    i(0),
                    i(1),
                    i(2),
                    a.eps,
                    a.w_offset,
                    i(3),
                    i(4),
                    o,
                    o.add(1),
                    o.add(2),
                    o.add(3),
                    s,
                ),
                other => return Err(Error::NoKernel(format!("cuda kernel '{other}' has no dispatch entry"))),
            }
        };
        if rc != 0 {
            let dts: Vec<_> = call.inputs.iter().map(|t| t.dtype).collect();
            return Err(Error::backend(format!(
                "ik_{kernel} returned {} ({rc}); inputs {dts:?} shapes {:?}",
                status_name(rc),
                call.inputs.iter().map(|t| t.shape.clone()).collect::<Vec<_>>()
            )));
        }
        Ok(())
    }
}
