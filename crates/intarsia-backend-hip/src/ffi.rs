//! `extern "C"` bindings to the hipcc-built kernel library (kernels/hip/*.hip).
//! Only compiled when build.rs found a ROCm toolchain (`cfg(ik_hip)`).

use std::ffi::{c_char, c_double, c_int, c_void};

pub const IK_MAX_DIMS: usize = 6;

/// Mirror of `ik_tensor` in kernels/abi/ik_abi.h.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct IkTensor {
    pub data: *mut c_void,
    pub dtype: i32,
    pub ndim: i32,
    pub shape: [i64; IK_MAX_DIMS],
    pub stride: [i64; IK_MAX_DIMS],
}

/// Mirror of `ik_scratch` in kernels/abi/ik_abi.h.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct IkScratch {
    pub data: *mut c_void,
    pub bytes: i64,
}

/// Mirror of `ik_device_info` in kernels/hip/ik_runtime.h.
#[repr(C)]
pub struct IkDeviceInfo {
    pub name: [c_char; 256],
    pub arch: [c_char; 64],
    pub major: i32,
    pub minor: i32,
    pub integrated: i32,
    pub sm_count: i32,
    pub mem_clock_khz: i32,
    pub bus_width_bits: i32,
    pub warp_size: i32,
    pub runtime_version: i32,
    pub driver_version: i32,
    pub _pad: i32,
    pub total_bytes: u64,
    pub free_bytes: u64,
}

type T = *const IkTensor;
type TM = *mut IkTensor;
type SC = *const IkScratch;
type S = *mut c_void;

extern "C" {
    // ---- runtime wrappers (kernels/hip/ik_runtime.h); every one returns a hipError_t ----
    pub fn ik_rt_device_count(n: *mut c_int) -> c_int;
    pub fn ik_rt_init_device(ordinal: c_int, info: *mut IkDeviceInfo) -> c_int;
    pub fn ik_rt_set_device(ordinal: c_int) -> c_int;
    pub fn ik_rt_mem_info(free: *mut u64, total: *mut u64) -> c_int;
    pub fn ik_rt_malloc(p: *mut *mut c_void, bytes: usize) -> c_int;
    pub fn ik_rt_free(p: *mut c_void) -> c_int;
    pub fn ik_rt_memcpy_h2d(dst: *mut c_void, src: *const c_void, n: usize, stream: S) -> c_int;
    pub fn ik_rt_memcpy_d2h(dst: *mut c_void, src: *const c_void, n: usize, stream: S) -> c_int;
    pub fn ik_rt_memcpy_d2d(dst: *mut c_void, src: *const c_void, n: usize, stream: S) -> c_int;
    pub fn ik_rt_memset(dst: *mut c_void, value: c_int, n: usize, stream: S) -> c_int;
    pub fn ik_rt_stream_create(stream: *mut S) -> c_int;
    pub fn ik_rt_stream_sync(stream: S) -> c_int;
    pub fn ik_rt_event_create(ev: *mut S) -> c_int;
    pub fn ik_rt_event_record(ev: S, stream: S) -> c_int;
    pub fn ik_rt_event_elapsed_ms(ms: *mut f32, a: S, b: S) -> c_int;
    pub fn ik_rt_event_destroy(ev: S) -> c_int;
    pub fn ik_rt_measure_bandwidth(gbs: *mut c_double) -> c_int;
    pub fn ik_rt_error_string(code: c_int) -> *const c_char;

    // ---- launchers (kernels/abi/ik_abi.h) ----
    pub fn ik_embedding_bf16(ids: T, table: T, out: TM, s: S) -> c_int;
    pub fn ik_rmsnorm_bf16(x: T, w: T, eps: f32, w_offset: f32, out: TM, s: S) -> c_int;
    pub fn ik_rmsnorm_add_bf16(x: T, res: T, w: T, eps: f32, w_offset: f32, out: TM, sum: TM, s: S) -> c_int;
    pub fn ik_add_bf16(a: T, b: T, out: TM, s: S) -> c_int;
    pub fn ik_qk_norm_bf16(x: T, w: T, eps: f32, w_offset: f32, out: TM, s: S) -> c_int;
    pub fn ik_rope_bf16(x: T, pos0: i64, theta: f32, rotary_dim: i64, out: TM, pos: T, s: S) -> c_int;
    pub fn ik_gate_split_bf16(x: T, q: TM, gate: TM, s: S) -> c_int;
    pub fn ik_sigmoid_mul_bf16(a: T, g: T, out: TM, s: S) -> c_int;
    pub fn ik_matmul_bf16(x: T, w: T, out: TM, s: S) -> c_int;
    pub fn ik_matmul_bf16_gemv(x: T, w: T, out: TM, s: S) -> c_int;
    pub fn ik_logits_bf16(x: T, w: T, out: TM, s: S) -> c_int;
    pub fn ik_logits_bf16_gemv(x: T, w: T, out: TM, s: S) -> c_int;
    pub fn ik_kv_append_bf16(new_kv: T, pos0: i64, cache: TM, pos: T, s: S) -> c_int;
    pub fn ik_attn_prefill_bf16(q: T, kc: T, vc: T, pos0: i64, scale: f32, out: TM, s: S) -> c_int;
    pub fn ik_attn_decode_bf16(q: T, kc: T, vc: T, pos0: i64, scale: f32, out: TM, pos: T, s: S) -> c_int;
    pub fn ik_attn_decode_bf16_splitkv(q: T, kc: T, vc: T, pos0: i64, scale: f32, out: TM, scratch: SC, pos: T, s: S) -> c_int;
    pub fn ik_attn_prefill_bf16_splitkv(q: T, kc: T, vc: T, pos0: i64, scale: f32, out: TM, scratch: SC, s: S) -> c_int;
    pub fn ik_attn_prefill_bf16_flash(q: T, kc: T, vc: T, pos0: i64, scale: f32, out: TM, s: S) -> c_int;
    pub fn ik_silu_glu_bf16(gate: T, up: T, out: TM, s: S) -> c_int;
    pub fn ik_argmax_f32(logits: T, out: TM, s: S) -> c_int;
    pub fn ik_argmax_bf16(logits: T, out: TM, s: S) -> c_int;
    pub fn ik_gdn_conv_bf16(x: T, taps: T, y: TM, state: TM, s: S) -> c_int;
    #[allow(clippy::too_many_arguments)]
    pub fn ik_gdn_delta_rule_bf16(
        qkv: T,
        a: T,
        b: T,
        a_log: T,
        dt_bias: T,
        eps: f32,
        o: TM,
        state: TM,
        s: S,
    ) -> c_int;
    pub fn ik_matmul_fp8_gemv(x: T, w: T, scales: T, out: TM, s: S) -> c_int;
    pub fn ik_matmul_fp8_mma(x: T, w: T, scales: T, out: TM, s: S) -> c_int;
    pub fn ik_matmul_nvfp4_gemv(x: T, w: T, bs: T, gs: T, out: TM, s: S) -> c_int;
    pub fn ik_matmul_nvfp4_mma(x: T, w: T, bs: T, gs: T, out: TM, s: S) -> c_int;
    pub fn ik_moe_topk_bf16(logits: T, top_k: c_int, norm: c_int, weights: TM, ids: TM, s: S) -> c_int;
    #[allow(clippy::too_many_arguments)]
    pub fn ik_moe_experts_bf16(
        x: T,
        ids: T,
        weights: T,
        gate_up_w: T,
        gate_up_s: T,
        down_w: T,
        down_s: T,
        gscales: T,
        out: TM,
        scratch: SC,
        s: S,
    ) -> c_int;
    pub fn ik_moe_combine_bf16(routed: T, shared: T, gate: T, out: TM, s: S) -> c_int;
}

/// Name of an `enum ik_status` value.
pub fn status_name(code: i32) -> &'static str {
    match code {
        0 => "IK_OK",
        1 => "IK_ERR_INVALID_ARG",
        2 => "IK_ERR_DTYPE",
        3 => "IK_ERR_SHAPE",
        4 => "IK_ERR_STRIDE",
        5 => "IK_ERR_UNSUPPORTED",
        6 => "IK_ERR_LAUNCH",
        _ => "IK_ERR_?",
    }
}
