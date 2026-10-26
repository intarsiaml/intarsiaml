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
    pub fn ik_rt_graph_begin(stream: S) -> c_int;
    pub fn ik_rt_graph_end(stream: S, exec: *mut S) -> c_int;
    pub fn ik_rt_graph_launch(exec: S, stream: S) -> c_int;
    pub fn ik_rt_graph_destroy(exec: S) -> c_int;
    pub fn ik_rt_device_sync() -> c_int;
    pub fn ik_rt_sync_capture_on() -> c_int;
    pub fn ik_rt_can_access_peer(device: c_int, peer: c_int, ok: *mut c_int) -> c_int;
    pub fn ik_rt_enable_peer_access(device: c_int, peer: c_int) -> c_int;
    pub fn ik_rt_malloc_finegrained(p: *mut *mut c_void, bytes: usize) -> c_int;
    #[allow(clippy::too_many_arguments)]
    pub fn ik_ar2_oneshot(
        x: *mut c_void,
        out: *mut c_void,
        dtype: c_int,
        out_dtype: c_int,
        n_elem: i64,
        peer_scratch: *mut c_void,
        my_scratch: *mut c_void,
        peer_flags: *mut u32,
        my_flags: *mut u32,
        ctrs: *mut u32,
        slot_bytes: i64,
        nblocks: c_int,
        drain: c_int,
        stream: S,
    ) -> c_int;
    #[allow(clippy::too_many_arguments)]
    pub fn ik_ag2_gather(
        x: *mut c_void,
        out: *mut c_void,
        bytes: i64,
        out_stride_bytes: i64,
        peer_scratch: *mut c_void,
        my_scratch: *mut c_void,
        peer_flags: *mut u32,
        my_flags: *mut u32,
        ctrs: *mut u32,
        slot_bytes: i64,
        nblocks: c_int,
        rank: c_int,
        drain: c_int,
        stream: S,
    ) -> c_int;
    pub fn ik_rt_measure_bandwidth(gbs: *mut c_double) -> c_int;
    pub fn ik_rt_error_string(code: c_int) -> *const c_char;
    /// INTARSIA_HIP_STAMP diagnostics (ik_runtime.h): write the wall-clock stamps to the file; no-op when unset
    pub fn ik_rt_stamp_dump() -> c_int;
    /// round 9 pinned staging (device.rs `Staging`)
    pub fn ik_rt_memcpy_h2d_pinned(dst: *mut c_void, src: *const c_void, n: usize, stream: S) -> c_int;
    pub fn ik_rt_memcpy_d2h_pinned(dst: *mut c_void, src: *const c_void, n: usize, stream: S) -> c_int;
    pub fn ik_rt_host_alloc(p: *mut *mut c_void, bytes: usize) -> c_int;
    pub fn ik_rt_host_free(p: *mut c_void) -> c_int;
    pub fn ik_rt_event_create_sync(ev: *mut S) -> c_int;
    pub fn ik_rt_event_sync(ev: S) -> c_int;
    pub fn ik_rt_spin(stream: S, us: u32) -> c_int;

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
    pub fn ik_matmul_bf16_gemv_rows(x: T, w: T, out: TM, s: S) -> c_int;
    pub fn ik_logits_bf16_gemv_rows(x: T, w: T, out: TM, s: S) -> c_int;
    pub fn ik_kv_append_bf16(new_kv: T, pos0: i64, cache: TM, pos: T, s: S) -> c_int;
    // row-table launches (batched decode, `OpCall::rows`; kernels/hip/ik_hip_abi.h)
    pub fn ik_gdn_conv_bf16_rows(x: T, taps: T, y: TM, state0: T, tab: T, stride: i64, index: i64, s: S) -> c_int;
    pub fn ik_gdn_delta_rule_bf16_v2_rows(
        qkv: T,
        a: T,
        b: T,
        a_log: T,
        dt_bias: T,
        eps: f32,
        o: TM,
        state0: T,
        tab: T,
        stride: i64,
        index: i64,
        byte_off: i64,
        s: S,
    ) -> c_int;
    pub fn ik_kv_append_bf16_rows(new_kv: T, cache0: T, tab: T, stride: i64, index: i64, pos: T, s: S) -> c_int;
    pub fn ik_attn_decode_bf16_splitkv_rows(
        q: T,
        kc0: T,
        vc0: T,
        scale: f32,
        out: TM,
        scratch: SC,
        tab: T,
        stride: i64,
        index: i64,
        pos: T,
        s: S,
    ) -> c_int;
    pub fn ik_attn_prefill_bf16(q: T, kc: T, vc: T, pos0: i64, scale: f32, out: TM, s: S) -> c_int;
    pub fn ik_attn_decode_bf16(q: T, kc: T, vc: T, pos0: i64, scale: f32, out: TM, pos: T, s: S) -> c_int;
    pub fn ik_attn_decode_bf16_window(q: T, kc: T, vc: T, pos0: i64, window: i64, scale: f32, out: TM, pos: T, s: S) -> c_int;
    pub fn ik_dyn_conv_bf16(x: T, k: T, base: T, out: TM, s: S) -> c_int;
    pub fn ik_attn_decode_bf16_splitkv(q: T, kc: T, vc: T, pos0: i64, scale: f32, out: TM, scratch: SC, pos: T, s: S) -> c_int;
    pub fn ik_attn_prefill_bf16_splitkv(q: T, kc: T, vc: T, pos0: i64, scale: f32, out: TM, scratch: SC, s: S) -> c_int;
    pub fn ik_attn_prefill_bf16_flash(q: T, kc: T, vc: T, pos0: i64, scale: f32, out: TM, s: S) -> c_int;
    pub fn ik_attn_prefill_bf16_flash_v2(q: T, kc: T, vc: T, pos0: i64, scale: f32, out: TM, s: S) -> c_int;
    pub fn ik_hip_flash_hpb() -> c_int;
    pub fn ik_hip_flash_vp() -> c_int;
    pub fn ik_hip_moe_mw(pairs: c_int) -> c_int;
    pub fn ik_hip_moe_mw_down(pairs: c_int) -> c_int;
    pub fn ik_hip_split_pieces(est_us: c_double, max_pieces: i64) -> c_int;
    pub fn ik_hip_splitkv_occ() -> c_int;
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
    #[allow(clippy::too_many_arguments)]
    pub fn ik_gdn_delta_rule_bf16_v2(
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
    #[allow(clippy::too_many_arguments)]
    pub fn ik_gdn_delta_rule_bf16_v3(
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
    #[allow(clippy::too_many_arguments)]
    pub fn ik_qk_norm_rope_kv_bf16(
        q: T,
        k: T,
        v: T,
        qw: T,
        kw: T,
        eps: f32,
        w_offset: f32,
        pos0: i64,
        theta: f32,
        rotary_dim: i64,
        q_out: TM,
        kcache: TM,
        vcache: TM,
        pos: T,
        s: S,
    ) -> c_int;
    pub fn ik_matmul_fp8_gemv(x: T, w: T, scales: T, out: TM, s: S) -> c_int;
    pub fn ik_matmul_fp8_gemv_v2(x: T, w: T, scales: T, out: TM, s: S) -> c_int;
    pub fn ik_matmul_fp8_gemv_rows(x: T, w: T, scales: T, out: TM, s: S) -> c_int;
    pub fn ik_matmul_bf16_wmma(x: T, w: T, out: TM, s: S) -> c_int;
    pub fn ik_matmul_bf16_wmma_rows(x: T, w: T, out: TM, s: S) -> c_int;
    pub fn ik_matmul_pair_bf16_wmma(x: T, w0: T, w1: T, out0: TM, out1: TM, s: S) -> c_int;
    #[allow(clippy::too_many_arguments)]
    pub fn ik_rmsnorm_add_matmul_pair_bf16_wmma(
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
    pub fn ik_fp8_relayout_frag(src: T, dst: TM, s: S) -> c_int;
    pub fn ik_fp8_unrelayout_frag(src: T, dst: TM, s: S) -> c_int;
    pub fn ik_matmul_fp8_wmma(x: T, w: T, scales: T, out: TM, scratch: SC, s: S) -> c_int;
    pub fn ik_matmul_fp8_wmma_rows(x: T, w: T, scales: T, out: TM, scratch: SC, s: S) -> c_int;
    pub fn ik_matmul_fp8_pair_wmma(x: T, w0: T, sc0: T, w1: T, sc1: T, out0: TM, out1: TM, s: S) -> c_int;
    pub fn ik_gated_rmsnorm_matmul_fp8_wmma(x: T, z: T, nw: T, eps: f32, w_offset: f32, w: T, scales: T, out: TM, s: S) -> c_int;
    pub fn ik_sigmoid_mul_matmul_fp8_wmma(a: T, gate: T, w: T, scales: T, out: TM, s: S) -> c_int;
    pub fn ik_matmul_fp8_mma(x: T, w: T, scales: T, out: TM, s: S) -> c_int;
    pub fn ik_matmul_fp8_mma_f8(x: T, w: T, scales: T, out: TM, scratch: SC, s: S) -> c_int;
    pub fn ik_matmul_fp8_mma_v2(x: T, w: T, scales: T, out: TM, scratch: SC, s: S) -> c_int;
    pub fn ik_matmul_nvfp4_gemv(x: T, w: T, bs: T, gs: T, out: TM, s: S) -> c_int;
    pub fn ik_matmul_nvfp4_gemv_v2(x: T, w: T, bs: T, gs: T, out: TM, s: S) -> c_int;
    pub fn ik_matmul_nvfp4_gemv_rows(x: T, w: T, bs: T, gs: T, out: TM, s: S) -> c_int;
    pub fn ik_matmul_nvfp4_mma(x: T, w: T, bs: T, gs: T, out: TM, s: S) -> c_int;
    pub fn ik_matmul_mxfp4_gemv(x: T, w: T, scales: T, act: T, out: TM, scratch: SC, s: S) -> c_int;
    pub fn ik_matmul_mxfp4_mma(x: T, w: T, scales: T, act: T, out: TM, scratch: SC, s: S) -> c_int;
    pub fn ik_matmul_mxfp4_mma_v1(x: T, w: T, scales: T, act: T, out: TM, scratch: SC, s: S) -> c_int;
    pub fn ik_matmul_mxfp4_mma_fold(x: T, w: T, scales: T, act: T, out: TM, scratch: SC, s: S) -> c_int;
    pub fn ik_matmul_mxfp4_pair_gemv(x: T, w0: T, s0: T, w1: T, s1: T, act: T, out0: TM, out1: TM, scratch: SC, s: S) -> c_int;
    pub fn ik_matmul_mxfp4_gemv_rows(x: T, w: T, scales: T, act: T, out: TM, scratch: SC, s: S) -> c_int;
    pub fn ik_matmul_mxfp4_pair_gemv_rows(x: T, w0: T, s0: T, w1: T, s1: T, act: T, out0: TM, out1: TM, scratch: SC, s: S) -> c_int;
    pub fn ik_matmul_mxfp4_gemv_rows_bf16(x: T, w: T, scales: T, act: T, out: TM, scratch: SC, s: S) -> c_int;
    pub fn ik_matmul_mxfp4_pair_gemv_rows_bf16(x: T, w0: T, s0: T, w1: T, s1: T, act: T, out0: TM, out1: TM, scratch: SC, s: S) -> c_int;
    pub fn ik_moe_topk_bf16(logits: T, top_k: c_int, norm: c_int, weights: TM, ids: TM, s: S) -> c_int;
    pub fn ik_moe_topk_bf16_v2(logits: T, top_k: c_int, norm: c_int, weights: TM, ids: TM, s: S) -> c_int;
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
    #[allow(clippy::too_many_arguments)]
    pub fn ik_moe_experts_bf16_v2(
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
    #[allow(clippy::too_many_arguments)]
    pub fn ik_moe_experts_bf16_rows(
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
    #[allow(clippy::too_many_arguments)]
    pub fn ik_moe_experts_bf16_v3(
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
    pub fn ik_matmul_pair_bf16_gemv(x: T, w0: T, w1: T, out0: TM, out1: TM, s: S) -> c_int;
    pub fn ik_matmul_fp8_pair_gemv(x: T, w0: T, sc0: T, w1: T, sc1: T, out0: TM, out1: TM, s: S) -> c_int;
    pub fn ik_matmul_nvfp4_glu_gemv(x: T, wg: T, bsg: T, gsg: T, wu: T, bsu: T, gsu: T, out: TM, s: S) -> c_int;
    pub fn ik_gated_rmsnorm_bf16(x: T, z: T, w: T, eps: f32, w_offset: f32, out: TM, s: S) -> c_int;
    pub fn ik_gated_rmsnorm_matmul_fp8_gemv(x: T, z: T, nw: T, eps: f32, w_offset: f32, w: T, scales: T, out: TM, s: S) -> c_int;
    #[allow(clippy::too_many_arguments)]
    pub fn ik_rmsnorm_add_matmul_pair_bf16_gemv(
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
    #[allow(clippy::too_many_arguments)]
    pub fn ik_rmsnorm_add_matmul_fp8_pair_wmma(
        x: T,
        res: T,
        w: T,
        eps: f32,
        w_offset: f32,
        w0: T,
        sc0: T,
        w1: T,
        sc1: T,
        normed: TM,
        sum: TM,
        out0: TM,
        out1: TM,
        s: S,
    ) -> c_int;
    // core-op fusion phase 1 (kernels/hip/ik_hip_abi.h)
    #[allow(clippy::too_many_arguments)]
    pub fn ik_rmsnorm_add_matmul_fp8_pair_gemv(
        x: T,
        res: T,
        w: T,
        eps: f32,
        w_offset: f32,
        w0: T,
        sc0: T,
        w1: T,
        sc1: T,
        normed: TM,
        sum: TM,
        out0: TM,
        out1: TM,
        s: S,
    ) -> c_int;
    pub fn ik_sigmoid_mul_matmul_fp8_gemv(a: T, gate: T, w: T, scales: T, out: TM, s: S) -> c_int;
    pub fn ik_shared_down_combine_nvfp4_gemv(x: T, w: T, bs: T, gs: T, routed: T, logit: T, out: TM, s: S) -> c_int;
    /// P2' (kernels/hip/moe_block.hip): `inputs` points at the 17 `Op::MoeBlock` inputs, contiguous and in order.
    pub fn ik_moe_block_bf16(inputs: T, top_k: c_int, norm: c_int, out: TM, scratch: SC, s: S) -> c_int;
    pub fn ik_moe_block_bf16_rows(inputs: T, top_k: c_int, norm: c_int, out: TM, scratch: SC, s: S) -> c_int;
    pub fn ik_moe_block_bf16_grouped(inputs: T, top_k: c_int, norm: c_int, out: TM, scratch: SC, s: S) -> c_int;
    #[allow(clippy::too_many_arguments)]
    pub fn ik_gate_split_qk_norm_rope_kv_bf16(
        qg: T,
        k: T,
        v: T,
        qw: T,
        kw: T,
        eps: f32,
        w_offset: f32,
        pos0: i64,
        theta: f32,
        rotary_dim: i64,
        q_out: TM,
        gate: TM,
        kcache: TM,
        vcache: TM,
        pos: T,
        s: S,
    ) -> c_int;
    #[allow(clippy::too_many_arguments)]
    pub fn ik_gdn_conv_delta_bf16(x: T, taps: T, a: T, b: T, a_log: T, dt_bias: T, eps: f32, o: TM, state: TM, cstate: TM, s: S) -> c_int;

    /// `kernels/hip/ar_norm.hip`: `ik_ar2_oneshot` fused with the residual add + rmsnorm after it
    #[allow(clippy::too_many_arguments)]
    pub fn ik_ar2_add_rmsnorm(
        x: *mut c_void,
        out: *mut c_void,
        res: *const c_void,
        w: *const c_void,
        nrm: *mut c_void,
        sum: *mut c_void,
        dtype: c_int,
        rows: i64,
        h: i64,
        eps: f32,
        w_offset: f32,
        peer_scratch: *mut c_void,
        my_scratch: *mut c_void,
        peer_flags: *mut u32,
        my_flags: *mut u32,
        ctrs: *mut u32,
        slot_bytes: i64,
        nblocks: c_int,
        drain: c_int,
        stream: S,
    ) -> c_int;
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
