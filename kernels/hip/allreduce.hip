/* Two-rank one-shot all-reduce over PCIe peer-to-peer, for one process driving both GPUs (tensor parallelism).
 *
 * Design after radiance's libr4d r4d_ar_oneshot_2rank_exact (Apache-2.0, https://codeberg.org/StillDeadcode/radiance,
 * tag v1.4.0; see kernels/hip/NOTICE-radiance). Rewritten for this ABI, not copied:
 *
 *  - PUSH model, one-shot: each rank writes its whole input into the PEER's scratch (peer-mapped device memory),
 *    raises a flag in the peer, waits for the peer's flag, then reduces its own input with what the peer pushed into
 *    its own scratch. Only scratch and flags are shared; inputs and outputs stay local, so the reduce is in place.
 *    No host synchronisation, no events, no copy engine: a collective is one kernel launch per rank, enqueued on the
 *    rank's own stream by its own thread. Both ranks must launch the same sequence of collectives.
 *  - Multi-block with per-block flags and no grid barrier: block b owns a contiguous chunk of 16-byte words, pushes it,
 *    sets peer_flags[b], spins on my_flags[b], reduces its chunk. The block count is fixed (every launch advances
 *    every block's counter by one), so the per-block counters of both ranks stay in lockstep.
 *  - The scratch is double buffered by the launch counter's parity. A slot is rewritten two launches later; by then the
 *    peer has passed this launch's flag handshake of the launch in between, so its reduce of the old contents is done
 *    (kernel boundary on its stream).
 *  - Scratch and flags are fine-grained (uncached) device memory, so peer stores are posted straight to the fabric and
 *    the system-scope release / acquire on the flags need no L2 write-back.
 *  - fp32 accumulation, one rounding to bf16: a + b commutes, so both ranks produce the same bits.
 *  - The all-gather (ag2_kernel, below) is the same push handshake with a copy in place of the sum: the vocab-parallel lm_head
 *    gathers each rank's half of the logits with it. It shares the scratch, flags and counters, so the two kernels may follow
 *    each other in any order as long as both ranks launch the same sequence.
 *  - A rank that never arrives (its thread died) would leave the other spinning: the wait gives up after a bounded
 *    number of polls and traps, which kills the process rather than hanging the GPU.
 */
#include "ik_hip.h"

#define AR_SPIN_MAX 480000000u /* polls (w27 diagnostic: 8x the old 60M, which trapped after ~6 s) */

IK_DEVI void ar_store_rel(unsigned int* p, unsigned int v) { __hip_atomic_store(p, v, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM); }
IK_DEVI unsigned int ar_load_acq(const unsigned int* p) { return __hip_atomic_load(p, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM); }

template <int KIND>
IK_DEVI int4 ar_add16(const int4& a, const int4& b);

/* KIND 0: 8 x bf16 */
template <>
IK_DEVI int4 ar_add16<0>(const int4& a, const int4& b) {
    uint4 ua = *reinterpret_cast<const uint4*>(&a), ub = *reinterpret_cast<const uint4*>(&b);
    float fa[8], fb[8];
    ik_unpack8(ua, fa);
    ik_unpack8(ub, fb);
#pragma unroll
    for (int i = 0; i < 8; ++i) fa[i] += fb[i];
    uint4 r = ik_pack8(fa);
    return *reinterpret_cast<int4*>(&r);
}

/* KIND 1: 4 x f32 */
template <>
IK_DEVI int4 ar_add16<1>(const int4& a, const int4& b) {
    int4 r;
    float* pr = reinterpret_cast<float*>(&r);
    const float* pa = reinterpret_cast<const float*>(&a);
    const float* pb = reinterpret_cast<const float*>(&b);
#pragma unroll
    for (int i = 0; i < 4; ++i) pr[i] = pa[i] + pb[i];
    return r;
}

/* KIND 2: 4 x f32 in, 4 x bf16 out (8 bytes) */
IK_DEVI uint2 ar_add16_to_bf16(const int4& a, const int4& b) {
    const float* pa = reinterpret_cast<const float*>(&a);
    const float* pb = reinterpret_cast<const float*>(&b);
    uint2 r;
    r.x = ik_pack2(pa[0] + pb[0], pa[1] + pb[1]);
    r.y = ik_pack2(pa[2] + pb[2], pa[3] + pb[3]);
    return r;
}

/* x4: this rank's partial (read-only after the push); out: where the sum goes (== x4 for KIND 0 / 1, in place;
 * a bf16 array of the same element count for KIND 2) */
template <int KIND>
__global__ void ar2_kernel(const int4* x4, void* out, int4* peer_scratch, int4* my_scratch, unsigned int* peer_flags,
                           unsigned int* my_flags, unsigned int* ctrs, int n16, int slot_stride16, int drain) {
    const int b = blockIdx.x, nb = gridDim.x, tid = threadIdx.x, nt = blockDim.x;
    __shared__ unsigned int s_seq;
    if (tid == 0) s_seq = atomicAdd(&ctrs[b], 1u) + 1u;
    __syncthreads();
    const unsigned int seq = s_seq;
    const size_t slot = seq & 1u;
    int4* ps = peer_scratch + slot * (size_t)slot_stride16;
    const int4* ms = my_scratch + slot * (size_t)slot_stride16;
    const int w = (n16 + nb - 1) / nb;
    const int start = min(b * w, n16), end = min(start + w, n16);

    /* push my chunk into the peer's scratch (4x unrolled: many posted writes in flight) */
    int k = start + tid;
    for (; k + 3 * nt < end; k += 4 * nt) {
        int4 v0 = x4[k], v1 = x4[k + nt], v2 = x4[k + 2 * nt], v3 = x4[k + 3 * nt];
        ps[k] = v0;
        ps[k + nt] = v1;
        ps[k + 2 * nt] = v2;
        ps[k + 3 * nt] = v3;
    }
    for (; k < end; k += nt) ps[k] = x4[k];
    /* the data must be visible at the peer before the flag is */
    if (drain == 1)
        __threadfence_system();
    else if (drain >= 3)
        asm volatile("s_wait_storecnt 0x0" ::: "memory");
    __syncthreads();
    if (tid == 0) {
        ar_store_rel(&peer_flags[b], seq);
        unsigned int spins = 0;
        while ((int)(ar_load_acq(&my_flags[b]) - seq) < 0) {
            __builtin_amdgcn_s_sleep(1);
            if (drain == 4) ar_store_rel(&peer_flags[b], seq); /* INTARSIA_HIP_AR_DRAIN=4 (diagnostic): re-send the flag while waiting; hang gone = a flag store was lost */
            if (++spins > AR_SPIN_MAX) __builtin_trap();
        }
        if (drain == 5) /* INTARSIA_HIP_AR_DRAIN=5 (diagnostic): ~25 us before the data is read; failures gone = the data landed after the flag */
            for (int i = 0; i < 100; ++i) __builtin_amdgcn_s_sleep(10);
    }
    __syncthreads();

    if (KIND == 2) {
        uint2* o2 = (uint2*)out;
        for (k = start + tid; k < end; k += nt) o2[k] = ar_add16_to_bf16(x4[k], ms[k]);
    } else {
        int4* o4 = (int4*)out;
        for (k = start + tid; k < end; k += nt) o4[k] = ar_add16<KIND>(x4[k], ms[k]);
    }
}

/* Two-rank all-gather, the same push handshake as ar2_kernel (same scratch, flags, counters and parity, so the per-block
 * counters of both ranks stay in lockstep and an all-gather and an all-reduce launch can follow each other in any order).
 * x: this rank's n16 words; out: rank-major, rank r's words at out[r * out_stride16 ..]. Both ranks wait for the peer's flag
 * (the slot-reuse argument of ar2_kernel needs that of every launch). x may be this rank's own segment of out (in place:
 * the self copy is skipped); nothing else may overlap out. Bytes are copied as they are: any dtype. */
__global__ void ag2_kernel(const int4* x4, int4* out4, int4* peer_scratch, int4* my_scratch, unsigned int* peer_flags,
                           unsigned int* my_flags, unsigned int* ctrs, int n16, int slot_stride16, int out_stride16, int rank,
                           int drain) {
    const int b = blockIdx.x, nb = gridDim.x, tid = threadIdx.x, nt = blockDim.x;
    __shared__ unsigned int s_seq;
    if (tid == 0) s_seq = atomicAdd(&ctrs[b], 1u) + 1u;
    __syncthreads();
    const unsigned int seq = s_seq;
    const size_t slot = seq & 1u;
    int4* ps = peer_scratch + slot * (size_t)slot_stride16;
    const int4* ms = my_scratch + slot * (size_t)slot_stride16;
    const int w = (n16 + nb - 1) / nb;
    const int start = min(b * w, n16), end = min(start + w, n16);

    int k = start + tid;
    for (; k + 3 * nt < end; k += 4 * nt) {
        int4 v0 = x4[k], v1 = x4[k + nt], v2 = x4[k + 2 * nt], v3 = x4[k + 3 * nt];
        ps[k] = v0;
        ps[k + nt] = v1;
        ps[k + 2 * nt] = v2;
        ps[k + 3 * nt] = v3;
    }
    for (; k < end; k += nt) ps[k] = x4[k];
    if (drain == 1)
        __threadfence_system();
    else if (drain >= 3)
        asm volatile("s_wait_storecnt 0x0" ::: "memory");
    __syncthreads();
    if (tid == 0) {
        ar_store_rel(&peer_flags[b], seq);
        unsigned int spins = 0;
        while ((int)(ar_load_acq(&my_flags[b]) - seq) < 0) {
            __builtin_amdgcn_s_sleep(1);
            if (drain == 4) ar_store_rel(&peer_flags[b], seq); /* INTARSIA_HIP_AR_DRAIN=4 (diagnostic): re-send the flag while waiting; hang gone = a flag store was lost */
            if (++spins > AR_SPIN_MAX) __builtin_trap();
        }
        if (drain == 5) /* INTARSIA_HIP_AR_DRAIN=5 (diagnostic): ~25 us before the data is read; failures gone = the data landed after the flag */
            for (int i = 0; i < 100; ++i) __builtin_amdgcn_s_sleep(10);
    }
    __syncthreads();

    int4* mine = out4 + (size_t)rank * out_stride16;
    int4* theirs = out4 + (size_t)(1 - rank) * out_stride16;
    const bool in_place = (const int4*)mine == x4;
    for (k = start + tid; k < end; k += nt) {
        theirs[k] = ms[k];
        if (!in_place) mine[k] = x4[k];
    }
}

extern "C" {

/* out = sum with the same tensor on the peer rank; x: this rank's partial (bf16 or f32, n_elem elements, 16-byte
 * aligned, a whole number of 16-byte words), left unchanged. (dtype, out_dtype) is (bf16, bf16), (f32, f32) or
 * (f32, bf16); out may be x for the first two (in place). peer_scratch / my_scratch: 2 x slot_bytes each; flags /
 * ctrs: nblocks u32 each (zero at start; ctrs is ordinary device memory). */
int ik_ar2_oneshot(void* x, void* out, int dtype, int out_dtype, int64_t n_elem, void* peer_scratch, void* my_scratch,
                   unsigned int* peer_flags, unsigned int* my_flags, unsigned int* ctrs, int64_t slot_bytes,
                   int nblocks, int drain, void* stream) {
    if (!x || !out || !peer_scratch || !my_scratch || !peer_flags || !my_flags || !ctrs || nblocks < 1 || nblocks > 1024) return IK_ERR_INVALID_ARG;
    int kind;
    if (dtype == IK_DTYPE_BF16 && out_dtype == IK_DTYPE_BF16) kind = 0;
    else if (dtype == IK_DTYPE_F32 && out_dtype == IK_DTYPE_F32) kind = 1;
    else if (dtype == IK_DTYPE_F32 && out_dtype == IK_DTYPE_BF16) kind = 2;
    else return IK_ERR_DTYPE;
    const int64_t esz = dtype == IK_DTYPE_BF16 ? 2 : 4;
    const int64_t bytes = n_elem * esz;
    if (bytes % 16 != 0 || ((uintptr_t)x & 15) != 0) return IK_ERR_SHAPE;
    if (((uintptr_t)out & 15) != 0) return IK_ERR_SHAPE;
    if (bytes > slot_bytes || slot_bytes % 16 != 0) return IK_ERR_SHAPE;
    if (n_elem == 0) return IK_OK;
    hipStream_t s = IK_STREAM(stream);
    const int n16 = (int)(bytes / 16), stride16 = (int)(slot_bytes / 16);
    if (kind == 0)
        ar2_kernel<0><<<nblocks, 256, 0, s>>>((const int4*)x, out, (int4*)peer_scratch, (int4*)my_scratch, peer_flags, my_flags, ctrs, n16, stride16, drain);
    else if (kind == 1)
        ar2_kernel<1><<<nblocks, 256, 0, s>>>((const int4*)x, out, (int4*)peer_scratch, (int4*)my_scratch, peer_flags, my_flags, ctrs, n16, stride16, drain);
    else
        ar2_kernel<2><<<nblocks, 256, 0, s>>>((const int4*)x, out, (int4*)peer_scratch, (int4*)my_scratch, peer_flags, my_flags, ctrs, n16, stride16, drain);
    return ik_launch_status();
}

/* all-gather of `bytes` bytes per rank (a multiple of 16, 16-byte aligned x / out) into out (rank-major, `out_stride_bytes`
 * between the two ranks' segments, 16-byte multiple, >= bytes); `rank` is this rank (0 or 1). x may be this rank's segment of
 * out. Scratch, flags and counters as ik_ar2_oneshot (the two share them). */
int ik_ag2_gather(void* x, void* out, int64_t bytes, int64_t out_stride_bytes, void* peer_scratch, void* my_scratch,
                  unsigned int* peer_flags, unsigned int* my_flags, unsigned int* ctrs, int64_t slot_bytes, int nblocks,
                  int rank, int drain, void* stream) {
    if (!x || !out || !peer_scratch || !my_scratch || !peer_flags || !my_flags || !ctrs || nblocks < 1 || nblocks > 1024) return IK_ERR_INVALID_ARG;
    if (rank != 0 && rank != 1) return IK_ERR_INVALID_ARG;
    if (bytes % 16 != 0 || out_stride_bytes % 16 != 0 || out_stride_bytes < bytes) return IK_ERR_SHAPE;
    if (((uintptr_t)x & 15) != 0 || ((uintptr_t)out & 15) != 0) return IK_ERR_SHAPE;
    if (bytes > slot_bytes || slot_bytes % 16 != 0) return IK_ERR_SHAPE;
    if (bytes == 0) return IK_OK;
    hipStream_t s = IK_STREAM(stream);
    const int n16 = (int)(bytes / 16), stride16 = (int)(slot_bytes / 16), ostride16 = (int)(out_stride_bytes / 16);
    ag2_kernel<<<nblocks, 256, 0, s>>>((const int4*)x, (int4*)out, (int4*)peer_scratch, (int4*)my_scratch, peer_flags, my_flags, ctrs, n16, stride16, ostride16, rank, drain);
    return ik_launch_status();
}

} /* extern "C" */
