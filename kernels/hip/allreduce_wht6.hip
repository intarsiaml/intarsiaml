/* Two-rank one-shot all-reduce with a LOSSY wire: each rank pushes a rotated 6-bit code of its partial instead of the partial
 * (6.25 bits per element: 5.1x fewer bytes than f32, 2.6x fewer than bf16). Opt-in (`INTARSIA_HIP_AR_WIRE=wht6`, comm.rs), for
 * large (prefill) messages only, where the PCIe wire time dominates an all-reduce; small messages keep allreduce.hip.
 *
 * Design after radiance's libr4d r4d_ar_oneshot_2rank_wht6 (Apache-2.0, https://codeberg.org/StillDeadcode/radiance, tag v2.0.0,
 * commit ddcb9c6; see kernels/hip/NOTICE-radiance). The code (rotation, scale, symbols, chunk packing) follows
 * r4d_ar_oneshot_2rank_wht6.h; the handshake, counters and slot parity are allreduce.hip's (ar2_kernel), on the wide path's
 * buffers, so a wht6 launch and an exact wide launch may follow each other in any order as long as both ranks launch the same
 * sequence. Not ported: radiance's push check (r4d_archk_*), its host mailbox mirror and its fused gated-residual writes.
 *
 *  - A group is 64 consecutive elements. It is rotated by an orthonormal 64-point Walsh-Hadamard transform (which spreads an
 *    outlier channel over the group), scaled by the group's own amax / 31 (shipped as a bf16) and stored as 6-bit uniform
 *    symbols. The rotation is its own inverse.
 *  - Push: rotate, quantise, pack straight into the peer's scratch, and keep a local copy of the same words. Reduce: unpack both
 *    halves, dequantise and add in f32, rotate back, round once to the output dtype.
 *  - Both ranks reduce the same two codes: this rank's `mine * my_scale + theirs * their_scale` is the other's sum with the
 *    operands swapped, which is the same f32 as long as no product is contracted into an FMA. Hence the pragma below and
 *    -ffp-contract=off for this file (build.rs): the reduced outputs of the two ranks are bit-identical.
 *  - The partial is encoded from the dtype it has (f32 partials are NOT rounded to bf16 first: the rotation runs in f32 either
 *    way, so the f32 partial costs only its wider load and loses nothing before the 6-bit quantiser).
 *  - Layout of a slot: [packed symbols, chunk-interleaved] ... [at scale_off: one u16 (bf16) scale per group]. A chunk is 32
 *    groups = 2048 elements, owned by one wave: every lane holds 8 elements of a group (8 lanes span a group, 4 groups per
 *    wave-iteration, 8 iterations), so a lane emits 64 symbols = 384 bits = three 16-byte words and lane L's u-th word lands
 *    at chunk * 96 + u * 32 + L: one coalesced 512-byte store per wave.
 *  - n_groups need not fill the last chunk: its missing groups quantise zeros and are skipped on output. Every cross-lane
 *    exchange of the rotation stays inside a group's 8 lanes, so no exchange is split by that guard.
 *  - The error (see docs/results/2026-11-14-hip-wht6-ar.md): per rank a uniform quantiser of step amax(rotated group) / 31 in
 *    the rotated domain, so the reduced group is off by at most (step_0 + step_1) / 2 per rotated element (a little more when
 *    the bf16 scale rounds down and the extreme symbol clamps). A non-finite element makes its whole group non-finite.
 */
#include "ik_hip.h"

#pragma clang fp contract(off)

#define W6_SPIN_MAX 480000000u /* as AR_SPIN_MAX (allreduce.hip) */
#define W6_GROUP 64            /* elements per rotation + scale group */
#define W6_BITS 6
#define W6_EPL 8                               /* elements per lane per group */
#define W6_LPG (W6_GROUP / W6_EPL)             /* 8 lanes span a group */
#define W6_GPW (IK_WARP * W6_EPL / W6_GROUP)   /* 4 groups per wave-iteration */
#define W6_CG 32                               /* groups per chunk */
#define W6_ITERS (W6_CG / W6_GPW)              /* 8 wave-iterations per chunk */
#define W6_NW (W6_ITERS * W6_EPL * W6_BITS / 32) /* 12 packed dwords per lane per chunk */
#define W6_NU (W6_NW / 4)                      /* = 3 uint4 */
#define W6_CHUNK_ELEMS (W6_CG * W6_GROUP)      /* 2048 */
#define W6_CHUNK_BYTES (W6_NU * IK_WARP * 16)  /* 1536 */

IK_DEVI void w6_store_rel(unsigned int* p, unsigned int v) { __hip_atomic_store(p, v, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM); }
IK_DEVI unsigned int w6_load_acq(const unsigned int* p) { return __hip_atomic_load(p, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM); }

/* the value of lane (L ^ N), N in {1, 2, 4}: DPP row_xmask, one VALU op, no LDS. Every lane of the group must be active. */
template <int N>
IK_DEVI float w6_xor_lane(float v) {
    return __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(v), 0x160 | N, 0xf, 0xf, true));
}

/* round to nearest even to bf16 and back: the scale ships as its top 16 bits, and the sender's local copy must be the float
 * the receiver rebuilds */
IK_DEVI float w6_bf16_round(float v) {
    unsigned u = __float_as_uint(v);
    u += 0x7fffu + ((u >> 16) & 1u);
    return __uint_as_float(u & 0xffff0000u);
}

/* `lo ? v + o : o - v` as `o + (lo ? v : -v)`: the sign flip is one XOR with a stage-uniform mask */
template <int M>
IK_DEVI void w6_xstage(float* v, int lig) {
    const unsigned int sm = (lig & M) ? 0x80000000u : 0u;
#pragma unroll
    for (int i = 0; i < W6_EPL; ++i) {
        const float o = w6_xor_lane<M>(v[i]);
        v[i] = __int_as_float(__float_as_int(v[i]) ^ sm) + o;
    }
}

/* orthonormal Walsh-Hadamard over one group (8 lanes x 8 elements); its own inverse */
IK_DEVI void w6_wht(float* v, int lig) {
#pragma unroll
    for (int h = 1; h < W6_EPL; h <<= 1)
#pragma unroll
        for (int i = 0; i < W6_EPL; ++i)
            if ((i & h) == 0) {
                const float x = v[i], y = v[i ^ h];
                v[i] = x + y;
                v[i ^ h] = x - y;
            }
    w6_xstage<1>(v, lig);
    w6_xstage<2>(v, lig);
    w6_xstage<4>(v, lig);
#pragma unroll
    for (int i = 0; i < W6_EPL; ++i) v[i] *= 0.125f; /* 1 / sqrt(64), exact */
}

/* 8 consecutive elements at element offset `off` (a multiple of 8; the base is 16-byte aligned) */
template <int F32>
IK_DEVI void w6_load(const void* x, long off, float* v) {
    if (F32) {
        const float4* p = (const float4*)x + (off >> 2);
        const float4 a = p[0], b = p[1];
        v[0] = a.x; v[1] = a.y; v[2] = a.z; v[3] = a.w;
        v[4] = b.x; v[5] = b.y; v[6] = b.z; v[7] = b.w;
    } else {
        ik_unpack8(((const uint4*)x)[off >> 3], v);
    }
}
template <int F32>
IK_DEVI void w6_store(void* out, long off, const float* v) {
    if (F32) {
        float4* p = (float4*)out + (off >> 2);
        p[0] = make_float4(v[0], v[1], v[2], v[3]);
        p[1] = make_float4(v[4], v[5], v[6], v[7]);
    } else {
        ((uint4*)out)[off >> 3] = ik_pack8(v);
    }
}

/* push one chunk: rotate, amax, quantise, pack into the peer's slot and the local copy */
template <int F32>
IK_DEVI void w6_push_chunk(const void* x, int c, int n_groups, int lane, uint4* p_pay, unsigned short* p_sc, uint4* l_pay, float* l_sc) {
    const int lig = lane & (W6_LPG - 1), gw = lane / W6_LPG;
    const int qmin = -(1 << (W6_BITS - 1)), qmax = (1 << (W6_BITS - 1)) - 1;
    unsigned int acc[W6_NW];
#pragma unroll
    for (int k = 0; k < W6_NW; ++k) acc[k] = 0u;
#pragma unroll
    for (int t = 0; t < W6_ITERS; ++t) {
        const int g = c * W6_CG + t * W6_GPW + gw;
        const bool live = g < n_groups; /* uniform over a group's lanes */
        float v[W6_EPL];
        if (live) {
            w6_load<F32>(x, (long)g * W6_GROUP + (long)lig * W6_EPL, v);
        } else {
#pragma unroll
            for (int i = 0; i < W6_EPL; ++i) v[i] = 0.f;
        }
        w6_wht(v, lig);
        float a = 0.f;
#pragma unroll
        for (int i = 0; i < W6_EPL; ++i) a = fmaxf(a, fabsf(v[i]));
        a = fmaxf(a, w6_xor_lane<1>(a));
        a = fmaxf(a, w6_xor_lane<2>(a));
        a = fmaxf(a, w6_xor_lane<4>(a));
        float sc = (a > 0.f) ? w6_bf16_round(a * (1.0f / (float)qmax)) : 1.0f;
        if (!(sc > 0.f)) sc = 1.0f;
        if (lig == 0 && live) {
            p_sc[g] = (unsigned short)(__float_as_uint(sc) >> 16);
            l_sc[g] = sc;
        }
        const float inv = 1.0f / sc;
#pragma unroll
        for (int i = 0; i < W6_EPL; ++i) {
            const int q = min(max((int)rintf(v[i] * inv), qmin), qmax) - qmin;
            const unsigned int sym = (unsigned int)q;
            const int bit = (t * W6_EPL + i) * W6_BITS, wi = bit >> 5, bo = bit & 31;
            acc[wi] |= sym << bo;
            if (bo + W6_BITS > 32) acc[wi + 1] |= sym >> (32 - bo);
        }
    }
#pragma unroll
    for (int u = 0; u < W6_NU; ++u) {
        const uint4 q = make_uint4(acc[4 * u], acc[4 * u + 1], acc[4 * u + 2], acc[4 * u + 3]);
        const long o = (long)c * (W6_NU * IK_WARP) + (long)u * IK_WARP + lane;
        p_pay[o] = q;
        l_pay[o] = q;
    }
}

/* reduce one chunk: unpack both halves, add in f32, rotate back, store */
template <int F32OUT>
IK_DEVI void w6_reduce_chunk(void* out, int c, int n_groups, int lane, const uint4* m_pay, const unsigned short* m_sc, const uint4* l_pay,
                             const float* l_sc) {
    const int lig = lane & (W6_LPG - 1), gw = lane / W6_LPG;
    const int qmin = -(1 << (W6_BITS - 1));
    const unsigned int MASK = (1u << W6_BITS) - 1u;
    unsigned int pac[W6_NW], lac[W6_NW];
#pragma unroll
    for (int u = 0; u < W6_NU; ++u) {
        const long o = (long)c * (W6_NU * IK_WARP) + (long)u * IK_WARP + lane;
        const uint4 pq = m_pay[o], lq = l_pay[o];
        pac[4 * u] = pq.x; pac[4 * u + 1] = pq.y; pac[4 * u + 2] = pq.z; pac[4 * u + 3] = pq.w;
        lac[4 * u] = lq.x; lac[4 * u + 1] = lq.y; lac[4 * u + 2] = lq.z; lac[4 * u + 3] = lq.w;
    }
#pragma unroll
    for (int t = 0; t < W6_ITERS; ++t) {
        const int g = c * W6_CG + t * W6_GPW + gw;
        if (g >= n_groups) continue; /* a whole group: no exchange is split */
        const float psc = __uint_as_float((unsigned int)m_sc[g] << 16);
        const float lsc = l_sc[g];
        float v[W6_EPL];
#pragma unroll
        for (int i = 0; i < W6_EPL; ++i) {
            const int bit = (t * W6_EPL + i) * W6_BITS, wi = bit >> 5, bo = bit & 31;
            unsigned int ps = pac[wi] >> bo, ls = lac[wi] >> bo;
            if (bo + W6_BITS > 32) {
                ps |= pac[wi + 1] << (32 - bo);
                ls |= lac[wi + 1] << (32 - bo);
            }
            /* two products and one add, no FMA: the peer computes the same sum with the operands swapped */
            v[i] = (float)((int)(ps & MASK) + qmin) * psc + (float)((int)(ls & MASK) + qmin) * lsc;
        }
        w6_wht(v, lig);
        w6_store<F32OUT>(out, (long)g * W6_GROUP + (long)lig * W6_EPL, v);
    }
}

/* x: this rank's partial (n_groups * 64 elements; read only in the push, so out may be x). loc: this rank's own packed half
 * (ordinary device memory, laid out like a slot but with f32 scales at scale_off). The handshake is ar2_kernel's: per-block
 * counter -> seq, slot seq & 1, the flag after the pushes are fenced, a bounded wait for the peer's flag. Block b owns the
 * chunks [b * cpb, (b + 1) * cpb); a wave takes every nwave-th of them. A block without chunks only does the handshake (the
 * block count is the wide path's, fixed, so the counters of both ranks stay in lockstep). */
template <int F32, int F32OUT>
__global__ void ar2w6_kernel(const void* x, void* out, char* peer_scratch, const char* my_scratch, char* loc, unsigned int* peer_flags,
                             unsigned int* my_flags, unsigned int* ctrs, int n_groups, int n_chunks, long slot_stride, long scale_off,
                             int drain) {
    const int b = blockIdx.x, nb = gridDim.x, tid = threadIdx.x, nt = blockDim.x;
    const int lane = tid & (IK_WARP - 1), wid = tid / IK_WARP, nwave = nt / IK_WARP;
    __shared__ unsigned int s_seq;
    if (tid == 0) s_seq = atomicAdd(&ctrs[b], 1u) + 1u;
    __syncthreads();
    const unsigned int seq = s_seq;
    const size_t slot = seq & 1u;
    char* pbase = peer_scratch + slot * (size_t)slot_stride;
    const char* mbase = my_scratch + slot * (size_t)slot_stride;
    uint4* p_pay = (uint4*)pbase;
    unsigned short* p_sc = (unsigned short*)(pbase + scale_off);
    const uint4* m_pay = (const uint4*)mbase;
    const unsigned short* m_sc = (const unsigned short*)(mbase + scale_off);
    uint4* l_pay = (uint4*)loc;
    float* l_sc = (float*)(loc + scale_off);

    const int cpb = (n_chunks + nb - 1) / nb;
    const int c0 = min(b * cpb, n_chunks), c1 = min(c0 + cpb, n_chunks);

    for (int c = c0 + wid; c < c1; c += nwave) w6_push_chunk<F32>(x, c, n_groups, lane, p_pay, p_sc, l_pay, l_sc);

    /* the data must be visible at the peer before the flag is (allreduce.hip: the fence is the mode that never reads stale rows) */
    if (drain >= 3 && drain <= 5)
        asm volatile("s_wait_storecnt 0x0" ::: "memory");
    else
        __threadfence_system();
    __syncthreads();
    if (tid == 0) {
        w6_store_rel(&peer_flags[b], seq);
        unsigned int spins = 0;
        /* drain 9 (tests only): no wait, the negative control */
        while (drain != 9 && (int)(w6_load_acq(&my_flags[b]) - seq) < 0) {
            __builtin_amdgcn_s_sleep(1);
            if (++spins > W6_SPIN_MAX) __builtin_trap();
        }
    }
    __syncthreads();

    for (int c = c0 + wid; c < c1; c += nwave) w6_reduce_chunk<F32OUT>(out, c, n_groups, lane, m_pay, m_sc, l_pay, l_sc);
}

extern "C" {

/* Bytes of a slot's payload region for a slot of slot_bytes: the scales (2 bytes per group of a chunk-rounded message) start
 * there. 15/16 of the slot, a whole number of chunks. */
int64_t ik_ar2_wht6_scale_off(int64_t slot_bytes) { return slot_bytes / 16 * 15 / W6_CHUNK_BYTES * W6_CHUNK_BYTES; }

/* Elements one launch can carry in a slot of slot_bytes (a multiple of 2048). */
int64_t ik_ar2_wht6_capacity(int64_t slot_bytes) {
    const int64_t so = ik_ar2_wht6_scale_off(slot_bytes);
    int64_t chunks = so / W6_CHUNK_BYTES;
    const int64_t by_scales = (slot_bytes - so) / (2 * W6_CG);
    if (chunks > by_scales) chunks = by_scales;
    return chunks * W6_CHUNK_ELEMS;
}

/* Bytes of the local pack buffer (`loc`, ordinary device memory, one per rank) for a slot of slot_bytes. */
int64_t ik_ar2_wht6_loc_bytes(int64_t slot_bytes) {
    return ik_ar2_wht6_scale_off(slot_bytes) + ik_ar2_wht6_capacity(slot_bytes) / W6_GROUP * 4;
}

/* out = x + the peer's x through the wht6 wire (lossy; bit-identical on both ranks). (dtype, out_dtype) as ik_ar2_oneshot:
 * (bf16, bf16), (f32, f32), (f32, bf16); out may be x for the first two. n_elem: a multiple of 64, at most
 * ik_ar2_wht6_capacity(slot_bytes). Scratch (2 x slot_bytes), flags and counters (nblocks each) are a fence path's (the wide
 * path's, comm.rs), shared with ik_ar2_oneshot on the same buffers; nblocks must be that path's block count. loc:
 * ik_ar2_wht6_loc_bytes(slot_bytes) of this rank's device memory, used by this launch only. drain as ik_ar2_oneshot (7 / 8 -> 1). */
int ik_ar2_wht6(void* x, void* out, int dtype, int out_dtype, int64_t n_elem, void* peer_scratch, void* my_scratch, void* loc,
                unsigned int* peer_flags, unsigned int* my_flags, unsigned int* ctrs, int64_t slot_bytes, int nblocks, int drain, void* stream) {
    if (!x || !out || !peer_scratch || !my_scratch || !loc || !peer_flags || !my_flags || !ctrs || nblocks < 1 || nblocks > 1024) return IK_ERR_INVALID_ARG;
    int kind;
    if (dtype == IK_DTYPE_BF16 && out_dtype == IK_DTYPE_BF16) kind = 0;
    else if (dtype == IK_DTYPE_F32 && out_dtype == IK_DTYPE_F32) kind = 1;
    else if (dtype == IK_DTYPE_F32 && out_dtype == IK_DTYPE_BF16) kind = 2;
    else return IK_ERR_DTYPE;
    if (n_elem < 0 || n_elem % W6_GROUP != 0 || ((uintptr_t)x & 15) != 0 || ((uintptr_t)out & 15) != 0) return IK_ERR_SHAPE;
    if (slot_bytes % 16 != 0 || n_elem > ik_ar2_wht6_capacity(slot_bytes)) return IK_ERR_SHAPE;
    if (n_elem == 0) return IK_OK;
    hipStream_t s = IK_STREAM(stream);
    const int n_groups = (int)(n_elem / W6_GROUP), n_chunks = (n_groups + W6_CG - 1) / W6_CG;
    const long so = (long)ik_ar2_wht6_scale_off(slot_bytes);
    if (drain == 7 || drain == 8) drain = 1;
#define W6_LAUNCH(F, FO)                                                                                                                 \
    ar2w6_kernel<F, FO><<<nblocks, 256, 0, s>>>(x, out, (char*)peer_scratch, (const char*)my_scratch, (char*)loc, peer_flags, my_flags, \
                                                ctrs, n_groups, n_chunks, (long)slot_bytes, so, drain)
    if (kind == 0) W6_LAUNCH(0, 0);
    else if (kind == 1) W6_LAUNCH(1, 1);
    else W6_LAUNCH(1, 0);
#undef W6_LAUNCH
    return ik_launch_status();
}

} /* extern "C" */
