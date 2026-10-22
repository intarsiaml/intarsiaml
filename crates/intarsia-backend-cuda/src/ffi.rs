//! `extern "C"` bindings to the nvcc-built kernel library (kernels/cuda/*.cu).
//! Only compiled when build.rs found a CUDA toolkit (`cfg(ik_cuda)`).

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

/// Mirror of `ik_device_info` in kernels/cuda/ik_runtime.h.
#[repr(C)]
pub struct IkDeviceInfo {
    pub name: [c_char; 256],
    pub major: i32,
    pub minor: i32,
    pub integrated: i32,
    pub sm_count: i32,
    pub mem_clock_khz: i32,
    pub bus_width_bits: i32,
    pub total_bytes: u64,
    pub free_bytes: u64,
}

type T = *const IkTensor;
type TM = *mut IkTensor;
type SC = *const IkScratch;
type S = *mut c_void;

extern "C" {
    // ---- runtime wrappers (kernels/cuda/ik_runtime.h); every one returns a cudaError_t ----
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
    pub fn ik_rt_f32_round(src: *const c_void, dst: *mut c_void, n: usize, dst_dtype: c_int, stream: S) -> c_int;
    pub fn ik_rt_measure_bandwidth(gbs: *mut c_double) -> c_int;
    pub fn ik_rt_error_string(code: c_int) -> *const c_char;
    pub fn ik_rt_graph_begin(stream: S) -> c_int;
    pub fn ik_rt_graph_end(stream: S, exec: *mut *mut c_void) -> c_int;
    pub fn ik_rt_graph_launch(exec: *mut c_void, stream: S) -> c_int;
    pub fn ik_rt_graph_destroy(exec: *mut c_void) -> c_int;

    // ---- launchers (kernels/abi/ik_abi.h) ----
    pub fn ik_embedding_bf16(ids: T, table: T, out: TM, s: S) -> c_int;
    pub fn ik_rmsnorm_bf16(x: T, w: T, eps: f32, w_offset: f32, out: TM, s: S) -> c_int;
    pub fn ik_rmsnorm_add_bf16(x: T, res: T, w: T, eps: f32, w_offset: f32, out: TM, sum: TM, s: S) -> c_int;
    pub fn ik_add_bf16(a: T, b: T, out: TM, s: S) -> c_int;
    pub fn ik_qk_norm_bf16(x: T, w: T, eps: f32, w_offset: f32, out: TM, s: S) -> c_int;
    pub fn ik_rope_bf16(x: T, pos0: i64, theta: f32, rotary_dim: i64, out: TM, pos: T, s: S) -> c_int;
    pub fn ik_gate_split_bf16(x: T, q: TM, gate: TM, s: S) -> c_int;
    pub fn ik_sigmoid_mul_bf16(a: T, g: T, out: TM, s: S) -> c_int;
    pub fn ik_gdn_conv_bf16(x: T, taps: T, y: TM, state: TM, s: S) -> c_int;
    pub fn ik_gdn_delta_rule_bf16(qkv: T, a: T, b: T, a_log: T, dt_bias: T, eps: f32, o: TM, state: TM, s: S) -> c_int;
    pub fn ik_gdn_delta_rule_bf16_lanes(qkv: T, a: T, b: T, a_log: T, dt_bias: T, eps: f32, o: TM, state: TM, s: S) -> c_int;
    pub fn ik_gdn_delta_rule_bf16_split(qkv: T, a: T, b: T, a_log: T, dt_bias: T, eps: f32, o: TM, state: TM, s: S) -> c_int;
    pub fn ik_matmul_bf16(x: T, w: T, out: TM, s: S) -> c_int;
    pub fn ik_matmul_bf16_gemv(x: T, w: T, out: TM, s: S) -> c_int;
    pub fn ik_logits_bf16(x: T, w: T, out: TM, s: S) -> c_int;
    pub fn ik_logits_bf16_gemv(x: T, w: T, out: TM, s: S) -> c_int;
    pub fn ik_kv_append_bf16(new_kv: T, pos0: i64, cache: TM, pos: T, s: S) -> c_int;
    pub fn ik_attn_prefill_bf16(q: T, kc: T, vc: T, pos0: i64, scale: f32, out: TM, s: S) -> c_int;
    pub fn ik_attn_decode_bf16(q: T, kc: T, vc: T, pos0: i64, scale: f32, out: TM, pos: T, s: S) -> c_int;
    pub fn ik_attn_decode_bf16_splitkv(q: T, kc: T, vc: T, pos0: i64, scale: f32, out: TM, scratch: SC, pos: T, s: S) -> c_int;
    pub fn ik_attn_prefill_bf16_splitkv(q: T, kc: T, vc: T, pos0: i64, scale: f32, out: TM, scratch: SC, s: S) -> c_int;
    pub fn ik_attn_decode_f8kv_splitkv(q: T, kc: T, vc: T, pos0: i64, scale: f32, out: TM, scratch: SC, pos: T, s: S) -> c_int;
    pub fn ik_attn_prefill_f8kv_splitkv(q: T, kc: T, vc: T, pos0: i64, scale: f32, out: TM, scratch: SC, s: S) -> c_int;
    pub fn ik_attn_prefill_f8kv_flash(q: T, kc: T, vc: T, pos0: i64, scale: f32, out: TM, s: S) -> c_int;
    pub fn ik_attn_prefill_bf16_flash(q: T, kc: T, vc: T, pos0: i64, scale: f32, out: TM, s: S) -> c_int;
    pub fn ik_silu_glu_bf16(gate: T, up: T, out: TM, s: S) -> c_int;
    pub fn ik_argmax_f32(logits: T, out: TM, s: S) -> c_int;
    pub fn ik_argmax_bf16(logits: T, out: TM, s: S) -> c_int;

    // ---- g4 (Gemma 4) extended-attribute launchers (kernels/cuda/ik_cuda_abi_g4.h) ----
    pub fn ik_gelu_glu_bf16(gate: T, up: T, out: TM, s: S) -> c_int;
    pub fn ik_rope_prop_bf16(x: T, pos0: i64, theta: f32, rotary_dim: i64, out: TM, pos: T, s: S) -> c_int;
    pub fn ik_add_scaled_bf16(a: T, b: T, out: TM, scale: f32, s: S) -> c_int;
    pub fn ik_embedding_scaled_bf16(ids: T, table: T, out: TM, scale: f32, s: S) -> c_int;
    pub fn ik_attn_prefill_bf16_ext(q: T, kc: T, vc: T, pos0: i64, scale: f32, window: i64, out: TM, s: S) -> c_int;
    pub fn ik_attn_decode_bf16_splitkv_ext(q: T, kc: T, vc: T, pos0: i64, scale: f32, window: i64, out: TM, scratch: SC, pos: T, s: S) -> c_int;
    pub fn ik_attn_prefill_bf16_splitkv_ext(q: T, kc: T, vc: T, pos0: i64, scale: f32, window: i64, out: TM, scratch: SC, s: S) -> c_int;
    pub fn ik_attn_decode_bf16_ext(q: T, kc: T, vc: T, pos0: i64, scale: f32, window: i64, out: TM, pos: T, s: S) -> c_int;
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
