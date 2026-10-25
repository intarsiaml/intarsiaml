/* HIP (gfx12 / RDNA4, wave32) device helpers and launcher plumbing for the .hip files in this directory.
 *
 * This is the HIP-side counterpart of kernels/common/ik_portable.h + kernels/cuda/ik_launch.h. It is kept
 * separate (not shared) for now: ik_portable.h hard-codes IK_WARP 64 for HIP, which is wrong for gfx12
 * (wave32), and uses the *_sync shuffles that ROCm 7.2 only accepts with a 64-bit mask. Merging the two
 * headers is later work (see docs/backends/hip.md). Only gfx12 is supported: the WMMA kernels use the
 * __builtin_amdgcn_wmma_*_w32_gfx12 builtins and every reduction assumes 32 lanes.
 */
#ifndef IK_HIP_H
#define IK_HIP_H

#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include "../abi/ik_abi.h"
#include "ik_runtime.h"  // ik_rt_stamp (INTARSIA_HIP_STAMP diagnostics around the long prefill dispatches)

#if defined(__AMDGCN_WAVEFRONT_SIZE__) && __AMDGCN_WAVEFRONT_SIZE__ != 32
#error "kernels/hip requires wave32 (gfx12); do not pass -mwavefrontsize64"
#endif

#define IK_WARP 32
#define IK_DEVI static __device__ __forceinline__

/* ---- bf16 as a raw 16-bit pattern (no dependence on the hip_bf16.h type, which changed between ROCm 6 and 7) */
struct ik_bf16 {
    uint16_t bits;
};

IK_DEVI float ik_b2f(ik_bf16 v) { return __uint_as_float((uint32_t)v.bits << 16); }

/* round-to-nearest-even, NaN stays NaN (same as __float2bfloat16_rn) */
IK_DEVI uint16_t ik_f2bits(float f) {
    uint32_t u = __float_as_uint(f);
    if ((u & 0x7fffffffu) > 0x7f800000u) return (uint16_t)((u >> 16) | 0x40u);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (uint16_t)(u >> 16);
}
IK_DEVI ik_bf16 ik_f2b(float v) {
    ik_bf16 r;
    r.bits = ik_f2bits(v);
    return r;
}

/* 8 x bf16 = 16 bytes, moved with one vector load/store */
IK_DEVI void ik_unpack8(const uint4& u, float* f) {
    f[0] = __uint_as_float(u.x << 16);
    f[1] = __uint_as_float(u.x & 0xffff0000u);
    f[2] = __uint_as_float(u.y << 16);
    f[3] = __uint_as_float(u.y & 0xffff0000u);
    f[4] = __uint_as_float(u.z << 16);
    f[5] = __uint_as_float(u.z & 0xffff0000u);
    f[6] = __uint_as_float(u.w << 16);
    f[7] = __uint_as_float(u.w & 0xffff0000u);
}

IK_DEVI uint32_t ik_pack2(float lo, float hi) { return (uint32_t)ik_f2bits(lo) | ((uint32_t)ik_f2bits(hi) << 16); }

IK_DEVI uint4 ik_pack8(const float* f) {
    uint4 u;
    u.x = ik_pack2(f[0], f[1]);
    u.y = ik_pack2(f[2], f[3]);
    u.z = ik_pack2(f[4], f[5]);
    u.w = ik_pack2(f[6], f[7]);
    return u;
}

/* silu(g) = g * sigmoid(g). The expression is the one silu_glu uses (elementwise.hip): the fused kernels that fold a
 * silu_glu into their epilogue must produce its bits. */
IK_DEVI float ik_silu(float g) { return g / (1.f + expf(-g)); }

/* streaming 16 B load of data that is read exactly once (weights): do not pollute the caches */
typedef unsigned int ik_u32x4 __attribute__((ext_vector_type(4)));
IK_DEVI uint4 ik_ldcs(const uint4* p) {
    const ik_u32x4 v = __builtin_nontemporal_load(reinterpret_cast<const ik_u32x4*>(p));
    return make_uint4(v[0], v[1], v[2], v[3]);
}

IK_DEVI uint4 ik_ldg(const uint4* p) { return *p; }

/* ---- wave32 reductions (xor shuffles; width 32 keeps them inside the wave) ---------------------------------- */
IK_DEVI float ik_warp_sum(float v) {
#pragma unroll
    for (int o = IK_WARP / 2; o > 0; o >>= 1) v += __shfl_xor(v, o, IK_WARP);
    return v;
}

IK_DEVI float ik_warp_max(float v) {
#pragma unroll
    for (int o = IK_WARP / 2; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor(v, o, IK_WARP));
    return v;
}

/* Block-wide sum / max, result broadcast to every thread. `red` is a shared array of at least
 * blockDim.x / IK_WARP floats. Every thread must call it. */
IK_DEVI float ik_block_sum(float v, float* red) {
    const int lane = threadIdx.x % IK_WARP, w = threadIdx.x / IK_WARP, nw = (blockDim.x + IK_WARP - 1) / IK_WARP;
    v = ik_warp_sum(v);
    __syncthreads();  // `red` may still be in use by a previous reduction
    if (lane == 0) red[w] = v;
    __syncthreads();
    float r = 0.f;
    for (int i = 0; i < nw; ++i) r += red[i];  // fixed order: deterministic
    return r;
}

IK_DEVI float ik_block_max(float v, float* red) {
    const int lane = threadIdx.x % IK_WARP, w = threadIdx.x / IK_WARP, nw = (blockDim.x + IK_WARP - 1) / IK_WARP;
    v = ik_warp_max(v);
    __syncthreads();
    if (lane == 0) red[w] = v;
    __syncthreads();
    float r = red[0];
    for (int i = 1; i < nw; ++i) r = fmaxf(r, red[i]);
    return r;
}

/* ---- host-side helpers shared by the launchers ---------------------------------------------------------------- */
static inline int64_t ik_numel(const ik_tensor* t) {
    int64_t n = 1;
    for (int i = 0; i < t->ndim; ++i) n *= t->shape[i];
    return n;
}

static inline bool ik_dense(const ik_tensor* t) {
    int64_t expect = 1;
    for (int i = t->ndim - 1; i >= 0; --i) {
        if (t->shape[i] > 1 && t->stride[i] != expect) return false;
        expect *= t->shape[i];
    }
    return true;
}

/* null / dtype / rank / dense-row-major check; IK_OK or an IK_ERR_* */
static inline int ik_check(const ik_tensor* t, int dtype, int ndim) {
    if (!t || (!t->data && ik_numel(t) > 0)) return IK_ERR_INVALID_ARG;
    if (t->ndim < 0 || t->ndim > IK_MAX_DIMS) return IK_ERR_INVALID_ARG;
    if (t->dtype != dtype) return IK_ERR_DTYPE;
    if (t->ndim != ndim) return IK_ERR_SHAPE;
    if (!ik_dense(t)) return IK_ERR_STRIDE;
    return IK_OK;
}

/* optional device-side position: null, or an i32 tensor with one element (the launcher's host `pos0` is then only a
 * bound for validation and grid sizing; the kernel reads the position from the tensor when it runs) */
static inline int ik_check_pos(const ik_tensor* t) {
    if (!t) return IK_OK;
    if (!t->data || t->dtype != IK_DTYPE_I32 || ik_numel(t) != 1) return IK_ERR_INVALID_ARG;
    return IK_OK;
}

/* Row table of a batched decode call (`OpCall::rows`): u8 [rows, stride, 8] little-endian device addresses, 8-byte aligned; `index` is
 * this call's entry in a row. */
static inline int ik_check_tab(const ik_tensor* tab, int64_t rows, int64_t stride, int64_t index) {
    if (!tab || !tab->data || tab->dtype != IK_DTYPE_U8 || tab->ndim != 3 || tab->shape[0] != rows ||
        tab->shape[1] != stride || tab->shape[2] != 8 || index < 0 || index >= stride)
        return IK_ERR_INVALID_ARG;
    if (reinterpret_cast<uintptr_t>(tab->data) & 7u) return IK_ERR_INVALID_ARG;
    return IK_OK;
}

#define IK_TRY(expr)                      \
    do {                                  \
        int _ik_s = (expr);               \
        if (_ik_s != IK_OK) return _ik_s; \
    } while (0)

/* Map a failed kernel launch to IK_ERR_LAUNCH (does not synchronise). */
static inline int ik_launch_status() { return hipGetLastError() == hipSuccess ? IK_OK : IK_ERR_LAUNCH; }

/* Sub-millisecond launches (round 8, docs/results/2026-10-19-hip-round8.md). On the R9700 (gfx1201, ROCm 7.2) every dispatch
 * that runs longer than about 1 ms is followed by ~275 us in which the GPU starts nothing, although the next dispatch is
 * long queued (the host is 10+ ms ahead); shorter dispatches are followed by the usual ~3 us. The prefill's long kernels
 * (MoE down at 64 pairs, the GDN delta rule; gate_up is not split, moe.hip) are therefore launched as pieces, each estimated below
 * INTARSIA_HIP_SPLIT_US microseconds (default 600; 0 = never split). A piece computes exactly what its part of the single
 * launch computed (independent output tiles, or a token range that continues the in-place recurrent state), so every
 * value gives the same bits. Read at every launch (the bitwise tests switch it in one process); a small value forces the
 * most pieces. Returns the number of pieces, 1 .. max_pieces.
 *
 * Round 9 (docs/results/2026-10-19-hip-round9.md): a launch is only split when its estimate exceeds
 * INTARSIA_HIP_SPLIT_MIN_US (default 950; 0 = round 8's rule, split whenever the estimate exceeds the piece budget). A
 * launch of 600 us to ~1 ms does not stall, so its pieces were pure overhead (chunks of roughly 780 to 1230 tokens). Pieces
 * are still sized by INTARSIA_HIP_SPLIT_US. Without a profiler the stall shows one dispatch later than under rocprofv3
 * (round 9 wall-clock stamps), ~256 us after every >= 1 ms dispatch. */
static inline int ik_split_pieces(double est_us, int64_t max_pieces) {
    const char* e = getenv("INTARSIA_HIP_SPLIT_US");
    const double budget = e && *e ? atof(e) : 600.0;
    const char* m = getenv("INTARSIA_HIP_SPLIT_MIN_US");
    const double trigger = m && *m ? atof(m) : 950.0;
    if (!(budget > 0.0) || est_us <= budget || est_us <= trigger || max_pieces <= 1) return 1;
    const double n = est_us / budget;
    const int64_t p = n >= (double)max_pieces ? max_pieces : (int64_t)n + 1;
    return (int)(p < 1 ? 1 : p);
}

static inline bool ik_aligned16(const void* p) { return ((uintptr_t)p & 15u) == 0; }

static inline unsigned ik_round_up(unsigned v, unsigned m) { return (v + m - 1) / m * m; }

#define IK_STREAM(s) (reinterpret_cast<hipStream_t>(s))

#endif /* IK_HIP_H */
