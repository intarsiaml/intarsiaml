/* Two-rank one-shot all-reduce fused with the residual add and the rmsnorm after it, for one process driving both GPUs
 * (tensor parallelism): out = round_bf16(a + b), sum = round_bf16(out + res), norm = rmsnorm(sum) * (w + woff), where
 * a, b are the two ranks' partials (f32 or bf16). Bit for bit the pair `ik_ar2_oneshot` + `ik_rmsnorm_add_bf16`, in one launch
 * and without writing and re-reading the all-reduce result between them.
 *
 * Design after radiance's libr4d r4d_ar_ln_had_quant_i8.hip (Apache-2.0, https://codeberg.org/StillDeadcode/radiance, tag
 * v1.12.0; see kernels/hip/NOTICE-radiance): the all-reduce is folded into the norm's load, one handshake per block, no
 * grid barrier. Rewritten for this ABI and for exact bf16 (radiance's version quantises to int8 behind a Hadamard
 * transform; this one keeps the wire and the arithmetic exact). The push / flag / double-buffered scratch protocol is
 * allreduce.hip's, line for line, so the two kernels share one scratch, one set of flags and one set of per-block counters:
 *
 *  - every block advances its counter once per launch on both ranks (a captured graph replays the same sequence of
 *    launches whichever kernel they are), takes the scratch slot of the counter's parity, pushes, raises the peer's flag,
 *    waits for its own;
 *  - the partition differs: block b owns a run of WHOLE ROWS (the norm needs the whole row), rows b*rpb .. (b+1)*rpb with
 *    rpb = ceil(rows / nblocks); the Rust side only launches it with rows <= nblocks (one row per block, as the unfused norm
 *    runs one block per row; more rows would serialise within a block). Both ranks use the same partition within one launch, which is all the protocol needs: a
 *    slot is rewritten two launches later and the peer's flag of the launch in between says its previous launch is over
 *    (kernel boundary on its stream), whichever blocks read it. Blocks with no row still push nothing, raise and wait.
 *
 * Exactness (the unfused pair's rounding points and order, see kernels/hip/norm.hip rmsnorm_kernel<true, true>):
 *  1. y = bf16(pa + pb) per element: the f32 add of the rank's own partial and the peer's (+ commutes, both ranks agree),
 *     one rounding (KIND 2 of allreduce.hip for f32 partials; KIND 0, unpack both and add, for bf16 partials).
 *  2. s = bf16(y + res): the norm reads the ROUNDED y (it is stored to `out` and re-read in the unfused pair), adds the
 *     residual in f32 and rounds again; s is the residual stream the next layer carries (stored to `sum`).
 *  3. ss = sum of s*s: thread t handles 8-element items i = t, t + bd, ... in order, 8 products per item in element order,
 *     then ik_block_sum over bd threads, with bd = round_up(min(H / 8, 256), 32): the norm kernel's block size and loops.
 *  4. r = rsqrtf(ss / H + eps); norm = bf16(s * r * (w + woff)) with s re-read from `sum`.
 * The statements are the norm kernel's, so the compiler contracts the same multiplies and adds.
 */
#include "ik_hip.h"
#include "ar_ll.h"

#define ARN_SPIN_MAX 480000000u /* polls (w27 diagnostic: 8x the old 60M, which trapped after ~6 s) */

IK_DEVI void arn_store_rel(unsigned int* p, unsigned int v) { __hip_atomic_store(p, v, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM); }
IK_DEVI unsigned int arn_load_acq(const unsigned int* p) { return __hip_atomic_load(p, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM); }

/* F32P: the partials are f32 (4 per 16 bytes, 2 x 16 bytes per 8-element item), else bf16 (8 per 16 bytes) */
template <bool F32P>
__global__ void ar2_norm_kernel(const int4* x4, ik_bf16* out, const ik_bf16* res, const ik_bf16* w, ik_bf16* nrm, ik_bf16* sum,
                                int4* peer_scratch, int4* my_scratch, unsigned int* peer_flags, unsigned int* my_flags,
                                unsigned int* ctrs, int rows, int H, int slot_stride16, float eps, float woff, int drain) {
    const int b = blockIdx.x, nb = gridDim.x, tid = threadIdx.x, nt = blockDim.x;
    __shared__ unsigned int s_seq;
    __shared__ float red[32];
    if (tid == 0) s_seq = atomicAdd(&ctrs[b], 1u) + 1u;
    __syncthreads();
    const unsigned int seq = s_seq;
    const size_t slot = seq & 1u;
    int4* ps = peer_scratch + slot * (size_t)slot_stride16;
    const int4* ms = my_scratch + slot * (size_t)slot_stride16;
    const int rpb = (rows + nb - 1) / nb;
    const int r0 = min(b * rpb, rows), r1 = min(r0 + rpb, rows);
    const int wpr = (F32P ? H * 4 : H * 2) / 16; /* 16-byte words per row */
    const int start = r0 * wpr, end = r1 * wpr;

    /* push my rows into the peer's scratch (allreduce.hip's loop) */
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
        arn_store_rel(&peer_flags[b], seq);
        unsigned int spins = 0;
        while ((int)(arn_load_acq(&my_flags[b]) - seq) < 0) {
            __builtin_amdgcn_s_sleep(1);
            if (drain == 4) arn_store_rel(&peer_flags[b], seq); /* INTARSIA_HIP_AR_DRAIN=4 (diagnostic): re-send the flag while waiting; hang gone = a flag store was lost */
            if (++spins > ARN_SPIN_MAX) __builtin_trap();
        }
        if (drain == 5) /* INTARSIA_HIP_AR_DRAIN=5 (diagnostic): ~25 us before the data is read; failures gone = the data landed after the flag */
            for (int i = 0; i < 100; ++i) __builtin_amdgcn_s_sleep(10);
    }
    __syncthreads();

    const int nv = H / 8;
    for (int row = r0; row < r1; ++row) {
        const size_t e0 = (size_t)row * H;
        ik_bf16* orow = out + e0;
        const ik_bf16* rr = res + e0;
        ik_bf16* sr = sum + e0;
        float ss = 0.f;
        for (int i = tid; i < nv; i += nt) {
            float f[8], g[8];
            if (F32P) {
                const float4* pa = reinterpret_cast<const float4*>(reinterpret_cast<const float*>(x4) + e0) + 2 * i;
                const float4* pb = reinterpret_cast<const float4*>(reinterpret_cast<const float*>(ms) + e0) + 2 * i;
                float4 a0 = pa[0], a1 = pa[1], b0 = pb[0], b1 = pb[1];
                f[0] = a0.x + b0.x; f[1] = a0.y + b0.y; f[2] = a0.z + b0.z; f[3] = a0.w + b0.w;
                f[4] = a1.x + b1.x; f[5] = a1.y + b1.y; f[6] = a1.z + b1.z; f[7] = a1.w + b1.w;
            } else {
                float fb[8];
                ik_unpack8(reinterpret_cast<const uint4*>(reinterpret_cast<const uint16_t*>(x4) + e0)[i], f);
                ik_unpack8(reinterpret_cast<const uint4*>(reinterpret_cast<const uint16_t*>(ms) + e0)[i], fb);
#pragma unroll
                for (int q = 0; q < 8; ++q) f[q] += fb[q];
            }
            const uint4 y = ik_pack8(f); /* the all-reduce result, rounded: what the unfused pair stores to `out` */
            reinterpret_cast<uint4*>(orow)[i] = y;
            ik_unpack8(y, f);
            ik_unpack8(reinterpret_cast<const uint4*>(rr)[i], g);
#pragma unroll
            for (int q = 0; q < 8; ++q) f[q] += g[q];
            const uint4 p = ik_pack8(f); /* the new residual stream */
            reinterpret_cast<uint4*>(sr)[i] = p;
            ik_unpack8(p, f); /* continue from the bf16-rounded sum */
#pragma unroll
            for (int q = 0; q < 8; ++q) ss += f[q] * f[q];
        }
        ss = ik_block_sum(ss, red);
        const float r = rsqrtf(ss / (float)H + eps);
        for (int i = tid; i < nv; i += nt) {
            float f[8], g[8];
            ik_unpack8(reinterpret_cast<const uint4*>(sr)[i], f);
            ik_unpack8(reinterpret_cast<const uint4*>(w)[i], g);
#pragma unroll
            for (int q = 0; q < 8; ++q) f[q] = f[q] * r * (g[q] + woff);
            reinterpret_cast<uint4*>(nrm + e0)[i] = ik_pack8(f);
        }
    }
}


/* The same kernel on allreduce.hip's LL handshake (drain 7 / 8, small messages): no flag, no block barrier, no fence. A thread pushes
 * exactly the words it reduces later (its 8-element items), so it has read its partial before it can overwrite it (`out` may be the
 * bf16 partial) without a barrier, and reads each peer word through ll_get (validated by the lines' tags) where the fence kernel reads
 * the scratch. The arithmetic is the fence kernel's, statement for statement. */
template <bool F32P, bool CHECK>
__global__ void ar2_norm_ll_kernel(const int4* x4, ik_bf16* out, const ik_bf16* res, const ik_bf16* w, ik_bf16* nrm, ik_bf16* sum,
                                   unsigned long long* peer_ll, const unsigned long long* my_ll, unsigned int* ctrs, int rows, int H,
                                   int ll_stride64, float eps, float woff) {
    __shared__ unsigned int s_seq;
    __shared__ float red[32];
    const unsigned int seq = ll_seq(ctrs, &s_seq);
    const int b = blockIdx.x, nb = gridDim.x, tid = threadIdx.x, nt = blockDim.x;
    unsigned long long* pl = peer_ll + (size_t)(seq & 1u) * ll_stride64;
    const unsigned long long* ml = my_ll + (size_t)(seq & 1u) * ll_stride64;
    const int rpb = (rows + nb - 1) / nb;
    const int r0 = min(b * rpb, rows), r1 = min(r0 + rpb, rows);
    const int wpr = (F32P ? H * 4 : H * 2) / 16; /* 16-byte words per row */
    const int n16 = rows * wpr;
    const int nv = H / 8;

    /* items are handled JB at a time per thread: all the loads of a batch first, then the pushes; all the peer words of a batch in one
     * round of polling, then the arithmetic item by item in ascending order (the fence kernel's order of summation) */
    constexpr int JB = 4, W = F32P ? 2 : 1;
    for (int row = r0; row < r1; ++row)
        for (int base = 0; base < nv; base += JB * nt) {
            int4 v[JB * W];
#pragma unroll
            for (int j = 0; j < JB; ++j) {
                const int i = base + j * nt + tid;
                if (i < nv)
#pragma unroll
                    for (int q = 0; q < W; ++q) v[j * W + q] = x4[row * wpr + W * i + q];
            }
#pragma unroll
            for (int j = 0; j < JB; ++j) {
                const int i = base + j * nt + tid;
                if (i < nv)
#pragma unroll
                    for (int q = 0; q < W; ++q) ll_put(pl, n16, row * wpr + W * i + q, seq, v[j * W + q]);
            }
        }

    for (int row = r0; row < r1; ++row) {
        const size_t e0 = (size_t)row * H;
        ik_bf16* orow = out + e0;
        const ik_bf16* rr = res + e0;
        ik_bf16* sr = sum + e0;
        float ss = 0.f;
        for (int base = 0; base < nv; base += JB * nt) {
            int ks[JB * W];
            int cnt = 0;
#pragma unroll
            for (int j = 0; j < JB; ++j) {
                const int i = base + j * nt + tid;
                if (i < nv) {
#pragma unroll
                    for (int q = 0; q < W; ++q) ks[j * W + q] = row * wpr + W * i + q;
                    cnt = (j + 1) * W;
                }
            }
            int4 pq[JB * W];
            ll_get_n<CHECK, JB * W>(ml, n16, ks, cnt, seq, pq);
#pragma unroll
            for (int j = 0; j < JB; ++j) {
                const int i = base + j * nt + tid;
                if (i >= nv) break;
                float f[8], g[8];
                if (F32P) {
                    const float4* pa = reinterpret_cast<const float4*>(reinterpret_cast<const float*>(x4) + e0) + 2 * i;
                    const float4 b0 = *reinterpret_cast<const float4*>(&pq[j * W]), b1 = *reinterpret_cast<const float4*>(&pq[j * W + 1]);
                    float4 a0 = pa[0], a1 = pa[1];
                    f[0] = a0.x + b0.x; f[1] = a0.y + b0.y; f[2] = a0.z + b0.z; f[3] = a0.w + b0.w;
                    f[4] = a1.x + b1.x; f[5] = a1.y + b1.y; f[6] = a1.z + b1.z; f[7] = a1.w + b1.w;
                } else {
                    float fb[8];
                    ik_unpack8(reinterpret_cast<const uint4*>(reinterpret_cast<const uint16_t*>(x4) + e0)[i], f);
                    ik_unpack8(*reinterpret_cast<const uint4*>(&pq[j]), fb);
#pragma unroll
                    for (int q2 = 0; q2 < 8; ++q2) f[q2] += fb[q2];
                }
                const uint4 y = ik_pack8(f); /* the all-reduce result, rounded: what the unfused pair stores to `out` */
                reinterpret_cast<uint4*>(orow)[i] = y;
                ik_unpack8(y, f);
                ik_unpack8(reinterpret_cast<const uint4*>(rr)[i], g);
#pragma unroll
                for (int q = 0; q < 8; ++q) f[q] += g[q];
                const uint4 p = ik_pack8(f); /* the new residual stream */
                reinterpret_cast<uint4*>(sr)[i] = p;
                ik_unpack8(p, f); /* continue from the bf16-rounded sum */
#pragma unroll
                for (int q = 0; q < 8; ++q) ss += f[q] * f[q];
            }
        }
        ss = ik_block_sum(ss, red);
        const float r = rsqrtf(ss / (float)H + eps);
        for (int i = tid; i < nv; i += nt) {
            float f[8], g[8];
            ik_unpack8(reinterpret_cast<const uint4*>(sr)[i], f);
            ik_unpack8(reinterpret_cast<const uint4*>(w)[i], g);
#pragma unroll
            for (int q = 0; q < 8; ++q) f[q] = f[q] * r * (g[q] + woff);
            reinterpret_cast<uint4*>(nrm + e0)[i] = ik_pack8(f);
        }
    }
}

extern "C" {

/* x: this rank's partial, `rows` x `H` (f32 or bf16 per `dtype`), left unchanged. out: bf16 [rows, H], receives the
 * all-reduce result (the x of the rmsnorm_add). res / w: the residual stream before the add [rows, H] and the norm weight [H]
 * (bf16). sum: bf16 [rows, H] the residual after the add. nrm: bf16 [rows, H] the normed rows. Everything 16-byte aligned,
 * H a multiple of 8, rows * row bytes within one scratch slot. The kernel and the scratch / flags / counters are
 * ik_ar2_oneshot's (same group, same nblocks, same drain, same LL regions). */
int ik_ar2_add_rmsnorm(void* x, void* out, const void* res, const void* w, void* nrm, void* sum, int dtype, int64_t rows,
                       int64_t H, float eps, float w_offset, void* peer_scratch, void* my_scratch, unsigned int* peer_flags,
                       unsigned int* my_flags, unsigned int* ctrs, int64_t slot_bytes, int nblocks, int drain, void* peer_ll, void* my_ll, int64_t ll_slot_bytes, void* stream) {
    if (!x || !out || !res || !w || !nrm || !sum || !peer_scratch || !my_scratch || !peer_flags || !my_flags || !ctrs || nblocks < 1 ||
        nblocks > 1024)
        return IK_ERR_INVALID_ARG;
    if (dtype != IK_DTYPE_BF16 && dtype != IK_DTYPE_F32) return IK_ERR_DTYPE;
    if (rows < 1 || H < 8 || H % 8 != 0) return IK_ERR_SHAPE;
    const int64_t esz = dtype == IK_DTYPE_BF16 ? 2 : 4;
    const int64_t bytes = rows * H * esz;
    if (bytes > slot_bytes || slot_bytes % 16 != 0 || rows * H > 0x7fffffffLL) return IK_ERR_SHAPE;
    if (!ik_aligned16(x) || !ik_aligned16(out) || !ik_aligned16(res) || !ik_aligned16(w) || !ik_aligned16(nrm) || !ik_aligned16(sum))
        return IK_ERR_SHAPE;
    hipStream_t s = IK_STREAM(stream);
    const int64_t items = H / 8;
    const unsigned bd = ik_round_up((unsigned)(items < 256 ? items : 256), 32); /* rmsnorm_kernel's block size */
    const int stride16 = (int)(slot_bytes / 16);
    if (drain >= 7) {
        if (ll_fits(drain, bytes, peer_ll, my_ll, ll_slot_bytes)) {
            const int ls = (int)(ll_slot_bytes / 8);
            auto pl = (unsigned long long*)peer_ll;
            auto ml = (const unsigned long long*)my_ll;
#define ARN_LL(F32P, CHECK) \
    ar2_norm_ll_kernel<F32P, CHECK><<<nblocks, bd, 0, s>>>((const int4*)x, (ik_bf16*)out, (const ik_bf16*)res, (const ik_bf16*)w, (ik_bf16*)nrm, \
                                                           (ik_bf16*)sum, pl, ml, ctrs, (int)rows, (int)H, ls, eps, w_offset)
            if (dtype == IK_DTYPE_F32) {
                if (drain == 7) ARN_LL(true, true); else ARN_LL(true, false);
            } else {
                if (drain == 7) ARN_LL(false, true); else ARN_LL(false, false);
            }
#undef ARN_LL
            return ik_launch_status();
        }
        drain = 1;
    }
    if (dtype == IK_DTYPE_F32)
        ar2_norm_kernel<true><<<nblocks, bd, 0, s>>>((const int4*)x, (ik_bf16*)out, (const ik_bf16*)res, (const ik_bf16*)w, (ik_bf16*)nrm,
                                                    (ik_bf16*)sum, (int4*)peer_scratch, (int4*)my_scratch, peer_flags, my_flags, ctrs,
                                                    (int)rows, (int)H, stride16, eps, w_offset, drain);
    else
        ar2_norm_kernel<false><<<nblocks, bd, 0, s>>>((const int4*)x, (ik_bf16*)out, (const ik_bf16*)res, (const ik_bf16*)w, (ik_bf16*)nrm,
                                                     (ik_bf16*)sum, (int4*)peer_scratch, (int4*)my_scratch, peer_flags, my_flags, ctrs,
                                                     (int)rows, (int)H, stride16, eps, w_offset, drain);
    return ik_launch_status();
}

} /* extern "C" */
