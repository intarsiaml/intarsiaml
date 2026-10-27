// Flash-style tiled causal GQA prefill attention on tensor cores for Gemma 4 (mma.sync m16n8k16, bf16 in, f32 accumulate) over a
// contiguous bf16 KV cache [Nkv, cap, D]: head_dim 256 and 512, scale as given (Gemma: 1.0), and an optional sliding window
// (the query at absolute position p attends keys max(0, p + 1 - window)..=p; window 0 = all keys).
//
// This is attention_flash.cu's kernel (read its header first: block = BQ/16 warps of 16 query rows of one q head, K and V tiles of BK
// keys through shared memory with cp.async, S = Q K^T and O += P V on register fragments, P never leaves registers, online softmax in
// the log2 domain) with three changes:
//   * window: the key loop starts at the tile holding the block's first visible key, and a tile that reaches below a row's first
//     visible key gets a per-element lower mask next to the causal one. A row whose window misses a tile sees all -inf there (its
//     running max stays -inf and the correction factor is exp2(-inf) = 0 on a still-zero O: nothing changes).
//   * D = 512: the f32 O tile of a warp (16 x 512) does not fit the register file, so the output dims are split over blockIdx.z = 2
//     blocks of DO = 256 dims each. Both blocks compute the full S = Q K^T (a 512-long contraction, the same bits in both: the
//     per-element arithmetic of S and of the softmax does not depend on which half the block owns) and each multiplies P by its
//     half of V. Q is not staged in shared memory (it would not fit next to the K tile) but read as A fragments straight from global
//     memory (L1 hits after the first tile).
//   * the V tile holds DO dims, the K tile D.
// The arithmetic per output element is the same sequence of operations as attention_flash.cu's (same tile size BK = 32, same
// order of the mma accumulations), so D = 256 with window 0 gives the same bits as attn_prefill_bf16_flash at scale 1/sqrt(D) = the
// scale passed (the admission sweep and tests/attention.rs compare it with the cpu reference).
#include "ik_launch.h"
#include <stdlib.h>
#include "ik_quant.cuh"

namespace {

constexpr float kLog2e = 1.4426950408889634f;

__device__ __forceinline__ uint32_t smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }

__device__ __forceinline__ void cp_async16(void* dst, const void* src, bool valid) {
    const int sz = valid ? 16 : 0;  // src-size 0 zero-fills the 16 bytes
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(smem_u32(dst)), "l"(src), "r"(sz));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void cp_wait() {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

__device__ __forceinline__ void ldsm_x4(uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
                 : "r"(smem_u32(p)));
}
__device__ __forceinline__ void ldsm_x4_t(uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
                 : "r"(smem_u32(p)));
}

__device__ __forceinline__ void mma_bf16(float* c, const uint32_t* a, uint32_t b0, uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ uint32_t pack_bf16(float lo, float hi) {
    __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
    return *reinterpret_cast<uint32_t*>(&v);
}

// D: head_dim (contraction of S), DO: output dims of this block (D, or D / 2 with blockIdx.z the half), BK: keys per tile,
// BQ: query rows per block (BQ / 16 warps).
template <int D, int DO, int BK, int BQ>
constexpr size_t flash_ext_smem_bytes() {
    return (size_t)(BK * (D + 8) + BK * (DO + 8) + (D == 256 ? BQ * (D + 8) : 0)) * 2;
}

template <int D, int DO, int BK, int BQ>
__global__ void __launch_bounds__(2 * BQ)
flash_ext_kernel(const ik_bf16* __restrict__ q, const ik_bf16* __restrict__ kc, const ik_bf16* __restrict__ vc,
                 ik_bf16* __restrict__ out, int T, int N, int G, int64_t cap, int64_t pos0, float sl2, int64_t window) {
    constexpr bool QS = D == 256;    // Q staged in shared memory (D = 512: A fragments read from global memory)
    constexpr int NT = 2 * BQ;       // threads
    constexpr int LDK = D + 8;       // padded smem rows (elements): conflict-free ldmatrix / 16 B aligned rows
    constexpr int LDV = DO + 8;
    constexpr int NKS = D / 16;      // k steps of Q K^T
    constexpr int NNB = BK / 8;      // key n-blocks of S
    constexpr int NDB = DO / 8;      // d n-blocks of O
    constexpr int CHK = D / 8;       // 16 B chunks per K row
    constexpr int CHV = DO / 8;      // ... per V row

    extern __shared__ __align__(16) unsigned char smem_raw[];
    ik_bf16* Ks = reinterpret_cast<ik_bf16*>(smem_raw);
    ik_bf16* Vs = Ks + BK * LDK;
    ik_bf16* Qs = Vs + BK * LDV;  // only when QS

    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int g = lane >> 2, c = lane & 3;
    const int qt = (int)gridDim.x - 1 - (int)blockIdx.x;  // longest blocks first
    const int h = blockIdx.y, kvh = h / G;
    const int doff = (int)blockIdx.z * DO;                // first output dim of this block
    const int q0 = qt * BQ;
    const int row_lo = q0 + warp * 16 + g, row_hi = row_lo + 8;

    const ik_bf16* kb = kc + (int64_t)kvh * cap * D;
    const ik_bf16* vb = vc + (int64_t)kvh * cap * D + doff;

    const int q_last = (q0 + BQ < T ? q0 + BQ : T) - 1;
    const int64_t kv_end = pos0 + q_last + 1;               // keys visible to the last row of the block
    const int ntiles = (int)((kv_end + BK - 1) / BK);
    const int64_t mask_from = pos0 + q0 + 1;                // tiles ending beyond this need the causal mask
    // window: first key the first row sees (the block's lowest), and the first key the last row sees (tiles starting below it
    // need the lower mask)
    const int64_t lo_min = (window > 0 && pos0 + q0 + 1 > window) ? pos0 + q0 + 1 - window : 0;
    const int64_t lo_max = (window > 0 && kv_end > window) ? kv_end - window : 0;
    const int kt0 = (int)(lo_min / BK);

    auto load_tile = [&](ik_bf16* dst, const ik_bf16* src, int tile, int ld, int chunks) {
        for (int idx = tid; idx < BK * chunks; idx += NT) {
            const int r = idx / chunks, ch = idx % chunks;
            const int64_t key = (int64_t)tile * BK + r;
            const bool valid = key < kv_end;
            cp_async16(dst + r * ld + ch * 8, valid ? src + key * D + ch * 8 : src, valid);
        }
    };

    // ---- Q ----
    if constexpr (QS) {
        for (int idx = tid; idx < BQ * CHK; idx += NT) {
            const int r = idx / CHK, ch = idx % CHK;
            const bool valid = q0 + r < T;
            cp_async16(Qs + r * LDK + ch * 8, valid ? q + ((int64_t)(q0 + r) * N + h) * D + ch * 8 : q, valid);
        }
    }
    const ik_bf16* qg_lo = q + ((int64_t)row_lo * N + h) * D + 2 * c;  // D = 512: this lane's A fragment source
    const ik_bf16* qg_hi = q + ((int64_t)row_hi * N + h) * D + 2 * c;

    load_tile(Ks, kb, kt0, LDK, CHK);
    cp_commit();

    float o[NDB][4];
#pragma unroll
    for (int i = 0; i < NDB; ++i) o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0.f;
    float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.f, l1 = 0.f;

    const int lm = lane >> 3, lr = lane & 7;  // ldmatrix lane addressing

    for (int kt = kt0; kt < ntiles; ++kt) {
        cp_wait<0>();     // K[kt] (and Q) landed
        __syncthreads();  // ... for every thread; also: PV of the previous tile is done with Vs
        load_tile(Vs, vb, kt, LDV, CHV);
        cp_commit();

        // ---- S = Q K^T ----
        float s[NNB][4];
#pragma unroll
        for (int i = 0; i < NNB; ++i) s[i][0] = s[i][1] = s[i][2] = s[i][3] = 0.f;
#pragma unroll
        for (int ks = 0; ks < NKS; ++ks) {
            uint32_t a[4];
            if constexpr (QS) {
                ldsm_x4(a[0], a[1], a[2], a[3], Qs + (warp * 16 + (lm & 1) * 8 + lr) * LDK + ks * 16 + (lm >> 1) * 8);
            } else {
                a[0] = row_lo < T ? *reinterpret_cast<const uint32_t*>(qg_lo + ks * 16) : 0u;
                a[1] = row_hi < T ? *reinterpret_cast<const uint32_t*>(qg_hi + ks * 16) : 0u;
                a[2] = row_lo < T ? *reinterpret_cast<const uint32_t*>(qg_lo + ks * 16 + 8) : 0u;
                a[3] = row_hi < T ? *reinterpret_cast<const uint32_t*>(qg_hi + ks * 16 + 8) : 0u;
            }
#pragma unroll
            for (int np = 0; np < NNB / 2; ++np) {
                uint32_t b0, b1, b2, b3;
                ldsm_x4(b0, b1, b2, b3, Ks + ((np * 2 + (lm >> 1)) * 8 + lr) * LDK + ks * 16 + (lm & 1) * 8);
                mma_bf16(s[np * 2], a, b0, b1);
                mma_bf16(s[np * 2 + 1], a, b2, b3);
            }
        }
        __syncthreads();  // everyone is done reading Ks
        if (kt + 1 < ntiles) load_tile(Ks, kb, kt + 1, LDK, CHK);
        cp_commit();

        // ---- online softmax (log2 domain) ----
        const bool need_mask = (int64_t)(kt + 1) * BK > mask_from || (int64_t)kt * BK < lo_max;
        float mx0 = -INFINITY, mx1 = -INFINITY;
#pragma unroll
        for (int nb = 0; nb < NNB; ++nb) {
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                float v = s[nb][e] * sl2;
                if (need_mask) {
                    const int64_t key = (int64_t)kt * BK + nb * 8 + 2 * c + (e & 1);
                    const int64_t lim = pos0 + (e < 2 ? row_lo : row_hi) + 1;  // exclusive
                    if (key >= lim || (window > 0 && key < lim - window)) v = -INFINITY;
                }
                s[nb][e] = v;
                if (e < 2) mx0 = fmaxf(mx0, v); else mx1 = fmaxf(mx1, v);
            }
        }
        mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 1));
        mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 2));
        mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 1));
        mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 2));
        const float mn0 = fmaxf(m0, mx0), mn1 = fmaxf(m1, mx1);
        const float mu0 = mn0 == -INFINITY ? 0.f : mn0, mu1 = mn1 == -INFINITY ? 0.f : mn1;
        const float cr0 = exp2f(m0 - mu0), cr1 = exp2f(m1 - mu1);
        m0 = mn0;
        m1 = mn1;
        float ps0 = 0.f, ps1 = 0.f;
        uint32_t pa[NNB][2];  // P as bf16 pairs: [nb][0] row g, [nb][1] row g+8
#pragma unroll
        for (int nb = 0; nb < NNB; ++nb) {
            const float p0 = exp2f(s[nb][0] - mu0), p1 = exp2f(s[nb][1] - mu0);
            const float p2 = exp2f(s[nb][2] - mu1), p3 = exp2f(s[nb][3] - mu1);
            ps0 += p0 + p1;
            ps1 += p2 + p3;
            pa[nb][0] = pack_bf16(p0, p1);
            pa[nb][1] = pack_bf16(p2, p3);
        }
        l0 = l0 * cr0 + ps0;
        l1 = l1 * cr1 + ps1;
#pragma unroll
        for (int i = 0; i < NDB; ++i) {
            o[i][0] *= cr0;
            o[i][1] *= cr0;
            o[i][2] *= cr1;
            o[i][3] *= cr1;
        }

        cp_wait<1>();     // V[kt] landed (the K[kt+1] group may still be in flight)
        __syncthreads();

        // ---- O += P V ----
#pragma unroll
        for (int j = 0; j < BK / 16; ++j) {
            uint32_t a[4] = {pa[2 * j][0], pa[2 * j][1], pa[2 * j + 1][0], pa[2 * j + 1][1]};
#pragma unroll
            for (int dp = 0; dp < NDB / 2; ++dp) {
                uint32_t b0, b1, b2, b3;
                ldsm_x4_t(b0, b1, b2, b3, Vs + (j * 16 + (lm & 1) * 8 + lr) * LDV + (dp * 2 + (lm >> 1)) * 8);
                mma_bf16(o[dp * 2], a, b0, b1);
                mma_bf16(o[dp * 2 + 1], a, b2, b3);
            }
        }
    }

    // ---- epilogue ----
    l0 += __shfl_xor_sync(0xffffffffu, l0, 1);
    l0 += __shfl_xor_sync(0xffffffffu, l0, 2);
    l1 += __shfl_xor_sync(0xffffffffu, l1, 1);
    l1 += __shfl_xor_sync(0xffffffffu, l1, 2);
    const float i0 = 1.f / l0, i1 = 1.f / l1;
#pragma unroll
    for (int nb = 0; nb < NDB; ++nb) {
        if (row_lo < T)
            *reinterpret_cast<uint32_t*>(out + ((int64_t)row_lo * N + h) * D + doff + nb * 8 + 2 * c) =
                pack_bf16(o[nb][0] * i0, o[nb][1] * i0);
        if (row_hi < T)
            *reinterpret_cast<uint32_t*>(out + ((int64_t)row_hi * N + h) * D + doff + nb * 8 + 2 * c) =
                pack_bf16(o[nb][2] * i1, o[nb][3] * i1);
    }
}

template <int D, int DO, int BK, int BQ>
int launch_flash_ext(const ik_bf16* q, const ik_bf16* kc, const ik_bf16* vc, ik_bf16* out, int T, int N, int NKV,
                     int64_t cap, int64_t pos0, float scale, int64_t window, cudaStream_t s) {
    constexpr size_t smem = flash_ext_smem_bytes<D, DO, BK, BQ>();
    // the dynamic-smem opt-in is per function and per device context; set it once per device
    static bool configured[16] = {};
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess) return IK_ERR_LAUNCH;
    if (dev < 0 || dev >= 16) return IK_ERR_UNSUPPORTED;
    if (!configured[dev]) {
        if (cudaFuncSetAttribute(flash_ext_kernel<D, DO, BK, BQ>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem) !=
            cudaSuccess)
            return IK_ERR_LAUNCH;
        configured[dev] = true;
    }
    dim3 grid((T + BQ - 1) / BQ, N, D / DO);
    flash_ext_kernel<D, DO, BK, BQ><<<grid, 2 * BQ, smem, s>>>(q, kc, vc, out, T, N, N / NKV, cap, pos0, scale * kLog2e, window);
    return ik_launch_status();
}

}  // namespace

extern "C" int ik_attn_prefill_bf16_flash_ext(const ik_tensor* q, const ik_tensor* kc, const ik_tensor* vc, int64_t pos0,
                                              float scale, int64_t window, ik_tensor* out, void* stream) {
    IK_TRY(ik_check(q, IK_DTYPE_BF16, 3));
    IK_TRY(ik_check(kc, IK_DTYPE_BF16, 3));
    IK_TRY(ik_check(vc, IK_DTYPE_BF16, 3));
    IK_TRY(ik_check(out, IK_DTYPE_BF16, 3));
    const int64_t T = q->shape[0], N = q->shape[1], D = q->shape[2];
    const int64_t NKV = kc->shape[0], cap = kc->shape[1];
    if (vc->shape[0] != NKV || vc->shape[1] != cap || vc->shape[2] != D || kc->shape[2] != D) return IK_ERR_SHAPE;
    if (out->shape[0] != T || out->shape[1] != N || out->shape[2] != D) return IK_ERR_SHAPE;
    if (NKV <= 0 || N % NKV != 0 || pos0 < 0 || window < 0 || pos0 + T > cap) return IK_ERR_SHAPE;
    if (D != 256 && D != 512) return IK_ERR_SHAPE;
    if (T == 0 || N == 0) return IK_OK;
    if (T > 0x7fffffffLL / 2 || N > 65535) return IK_ERR_SHAPE;
    if (!ik_aligned16(kc->data) || !ik_aligned16(vc->data) || (reinterpret_cast<uintptr_t>(q->data) & 3) ||
        (reinterpret_cast<uintptr_t>(out->data) & 3))
        return IK_ERR_STRIDE;
    if (D == 256 && !ik_aligned16(q->data)) return IK_ERR_STRIDE;  // Q is staged with 16 B cp.async
    cudaStream_t s = IK_STREAM(stream);
    const ik_bf16* qp = (const ik_bf16*)q->data;
    const ik_bf16* kp = (const ik_bf16*)kc->data;
    const ik_bf16* vp = (const ik_bf16*)vc->data;
    ik_bf16* op = (ik_bf16*)out->data;
    if (D == 256) return launch_flash_ext<256, 256, 32, 128>(qp, kp, vp, op, (int)T, (int)N, (int)NKV, cap, pos0, scale, window, s);
    return launch_flash_ext<512, 256, 32, 64>(qp, kp, vp, op, (int)T, (int)N, (int)NKV, cap, pos0, scale, window, s);
}
