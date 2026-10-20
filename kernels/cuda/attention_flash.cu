// Flash-style tiled causal GQA prefill attention on tensor cores (mma.sync m16n8k16, bf16 in, f32 accumulate)
// over a contiguous bf16 KV cache [Nkv, cap, D].
//
// Block = 4 warps = 64 query rows of one q head; each warp owns 16 rows. K and V tiles of BK keys are
// streamed through shared memory with cp.async (K for tile i+1 is loaded while the softmax / PV of tile i
// runs, V for tile i while Q K^T runs). S = Q K^T and O += P V use the register fragments of mma.sync directly
// (the accumulator layout of S is the A-operand layout of P, so P never goes through memory). Q lives in
// registers (D <= 128) or in shared memory (D == 256). Online softmax in the log2 domain, f32 state.
//
// Causality / chunking: query row r of the call sees keys 0 .. pos0 + r. Tiles that lie fully below the
// first row of the block are processed without a mask. Blocks are issued longest-first.
//
// fp8 KV cache (attn_prefill_f8kv_flash, opt-in through an fp8_e4m3 cache): KT = e4m3 bytes. A K / V tile's bytes are
// streamed with cp.async into the upper BK x D bytes of the very bf16 tile they will become, issued at the same points as
// the bf16 kernel's copies (when that tile is free); where the bf16 kernel waits for the tile, each thread reads the bytes
// it copied itself into registers, a barrier lets every thread do so, and each thread writes its bytes converted exactly to
// bf16 (every e4m3 value is a bf16 value) into the tile, before the bf16 kernel's own __syncthreads. No extra shared memory
// (same occupancy). The rest is the bf16 kernel's code, so the output is bitwise the bf16 kernel's on the cache upcast to
// bf16 (tests/attn_f8kv.rs).
#include "ik_launch.h"
#include <stdlib.h>
#include "ik_quant.cuh"

namespace {

constexpr int kBQ = 64;
constexpr int kThreads = 128;
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

template <int D, int BK, int BQ = kBQ>
constexpr size_t flash_smem_bytes() {
    return (size_t)(2 * BK + (D > 128 ? BQ : 0)) * (D + 8) * 2;
}

// D: head_dim, BK: keys per tile. QREG: Q fragments in registers (else shared memory).
// 8 e4m3 bytes -> 8 bf16 (exact), as one 16-byte store
__device__ __forceinline__ uint4 e4m3x8_to_bf16x8(uint32_t lo, uint32_t hi) {
    const uint32_t w[2] = {lo, hi};
    uint32_t o[4];
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const float2 a = ik_e4m3x2_to_f2(static_cast<unsigned short>(w[i] & 0xFFFFu));
        const float2 b = ik_e4m3x2_to_f2(static_cast<unsigned short>(w[i] >> 16));
        o[2 * i] = pack_bf16(a.x, a.y);
        o[2 * i + 1] = pack_bf16(b.x, b.y);
    }
    return make_uint4(o[0], o[1], o[2], o[3]);
}

// BQ: query rows per block (BQ / 16 warps). The rows of a block share its K / V tiles; every row's arithmetic is the same
// for any BQ (a warp's tiles past its own rows' last key are fully masked: the running max, sum and O are multiplied by
// exp2f(0) = 1 and P = 0 adds nothing), so BQ = 128 gives the same bits as 64 (tests/attention.rs) with twice the warps per
// SM and half the K / V tile traffic per row. D = 256 at BQ = 128 uses 101376 bytes, the opt-in maximum.
template <int D, int BK, typename KT = ik_bf16, int BQ = kBQ>
__global__ void __launch_bounds__(2 * BQ)
flash_kernel(const ik_bf16* __restrict__ q, const KT* __restrict__ kc, const KT* __restrict__ vc,
             ik_bf16* __restrict__ out, int T, int N, int G, int64_t cap, int64_t pos0, float sl2) {
    constexpr bool QREG = D <= 128;
    constexpr int NT = 2 * BQ;     // threads
    constexpr int LD = D + 8;     // padded smem row (elements): conflict-free ldmatrix / 16 B aligned rows
    constexpr int NKS = D / 16;   // k steps of Q K^T
    constexpr int NNB = BK / 8;   // key n-blocks of S
    constexpr int NDB = D / 8;    // d n-blocks of O
    constexpr int CHUNKS = D / 8; // 16 B chunks per row

    extern __shared__ __align__(16) unsigned char smem_raw[];
    ik_bf16* Ks = reinterpret_cast<ik_bf16*>(smem_raw);
    ik_bf16* Vs = Ks + BK * LD;
    ik_bf16* Qs = Vs + BK * LD;  // only when !QREG

    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int g = lane >> 2, c = lane & 3;
    const int qt = (int)gridDim.x - 1 - (int)blockIdx.x;  // longest blocks first
    const int h = blockIdx.y, kvh = h / G;
    const int q0 = qt * BQ;
    const int row_lo = q0 + warp * 16 + g, row_hi = row_lo + 8;

    const KT* kb = kc + (int64_t)kvh * cap * D;
    const KT* vb = vc + (int64_t)kvh * cap * D;

    const int q_last = (q0 + BQ < T ? q0 + BQ : T) - 1;
    const int64_t kv_end = pos0 + q_last + 1;                // keys visible to the last row of the block
    const int ntiles = (int)((kv_end + BK - 1) / BK);
    const int64_t mask_from = pos0 + q0 + 1;                 // tiles ending beyond this need the causal mask

    auto load_tile = [&](ik_bf16* dst, const KT* src, int tile) {
        for (int idx = tid; idx < BK * CHUNKS; idx += NT) {
            const int r = idx / CHUNKS, ch = idx % CHUNKS;
            const int64_t key = (int64_t)tile * BK + r;
            const bool valid = key < kv_end;
            cp_async16(dst + r * LD + ch * 8, valid ? src + key * D + ch * 8 : src, valid);
        }
    };
    // fp8: 16 bytes (16 dims) per copy into the staging bytes `st` [BK][D] (zero-filled past the end: e4m3 0 = bf16 0), the
    // upper BK x D bytes of the tile (BK * LD bf16) the bytes become
    constexpr int CH8 = D / 16;
    constexpr int NPT = BK * CH8 / NT;  // 16-byte groups per thread per tile
    static_assert(BK * CH8 % NT == 0, "whole groups per thread");
    constexpr int ST_OFF = BK * LD * 2 - BK * D;  // byte offset of the staging bytes in a tile
    unsigned char* Kst = reinterpret_cast<unsigned char*>(Ks) + ST_OFF;
    unsigned char* Vst = reinterpret_cast<unsigned char*>(Vs) + ST_OFF;
    auto issue8 = [&](unsigned char* st, const KT* src, int tile) {
        for (int idx = tid; idx < BK * CH8; idx += NT) {
            const int r = idx / CH8, ch = idx % CH8;
            const int64_t key = (int64_t)tile * BK + r;
            const bool valid = key < kv_end;
            cp_async16(st + r * D + ch * 16, valid ? src + key * D + ch * 16 : src, valid);
        }
    };
    // ... and, once landed (cp_wait: this thread's copies), the bytes this thread copied (same idx mapping) -> registers;
    // a barrier (every thread has read its bytes: the tile may be overwritten) -> bf16 into the tile
    auto land8 = [&](ik_bf16* dst, const unsigned char* st) {
        uint4 v[NPT];
#pragma unroll
        for (int i = 0; i < NPT; ++i) {
            const int idx = tid + i * NT, r = idx / CH8, ch = idx % CH8;
            v[i] = *reinterpret_cast<const uint4*>(st + r * D + ch * 16);
        }
        __syncthreads();  // f8: every thread holds its staged bytes
#pragma unroll
        for (int i = 0; i < NPT; ++i) {
            const int idx = tid + i * NT, r = idx / CH8, ch = idx % CH8;
            *reinterpret_cast<uint4*>(dst + r * LD + ch * 16) = e4m3x8_to_bf16x8(v[i].x, v[i].y);
            *reinterpret_cast<uint4*>(dst + r * LD + ch * 16 + 8) = e4m3x8_to_bf16x8(v[i].z, v[i].w);
        }
    };
    constexpr bool F8 = sizeof(KT) == 1;

    // ---- Q ----
    uint32_t qa[QREG ? NKS : 1][4];
    if (QREG) {
#pragma unroll
        for (int ks = 0; ks < (QREG ? NKS : 1); ++ks) {
            const ik_bf16* p0 = q + ((int64_t)row_lo * N + h) * D + ks * 16 + 2 * c;
            const ik_bf16* p1 = q + ((int64_t)row_hi * N + h) * D + ks * 16 + 2 * c;
            qa[ks][0] = row_lo < T ? *reinterpret_cast<const uint32_t*>(p0) : 0u;
            qa[ks][1] = row_hi < T ? *reinterpret_cast<const uint32_t*>(p1) : 0u;
            qa[ks][2] = row_lo < T ? *reinterpret_cast<const uint32_t*>(p0 + 8) : 0u;
            qa[ks][3] = row_hi < T ? *reinterpret_cast<const uint32_t*>(p1 + 8) : 0u;
        }
    } else {
        for (int idx = tid; idx < BQ * CHUNKS; idx += NT) {
            const int r = idx / CHUNKS, ch = idx % CHUNKS;
            const bool valid = q0 + r < T;
            cp_async16(Qs + r * LD + ch * 8, valid ? q + ((int64_t)(q0 + r) * N + h) * D + ch * 8 : q, valid);
        }
    }

    if constexpr (F8) issue8(Kst, kb, 0);
    else load_tile(Ks, kb, 0);
    cp_commit();

    float o[NDB][4];
#pragma unroll
    for (int i = 0; i < NDB; ++i) o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0.f;
    float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.f, l1 = 0.f;

    // ldmatrix lane addressing
    const int lm = lane >> 3, lr = lane & 7;

    for (int kt = 0; kt < ntiles; ++kt) {
        cp_wait<0>();     // K[kt] (and Q) landed
        if constexpr (F8) land8(Ks, Kst);
        __syncthreads();  // ... for every thread; also: PV of the previous tile is done with Vs
        if constexpr (F8) issue8(Vst, vb, kt);
        else load_tile(Vs, vb, kt);
        cp_commit();

        // ---- S = Q K^T ----
        float s[NNB][4];
#pragma unroll
        for (int i = 0; i < NNB; ++i) s[i][0] = s[i][1] = s[i][2] = s[i][3] = 0.f;
#pragma unroll
        for (int ks = 0; ks < NKS; ++ks) {
            uint32_t a[4];
            if (QREG) {
                a[0] = qa[ks][0]; a[1] = qa[ks][1]; a[2] = qa[ks][2]; a[3] = qa[ks][3];
            } else {
                ldsm_x4(a[0], a[1], a[2], a[3], Qs + (warp * 16 + (lm & 1) * 8 + lr) * LD + ks * 16 + (lm >> 1) * 8);
            }
#pragma unroll
            for (int np = 0; np < NNB / 2; ++np) {
                uint32_t b0, b1, b2, b3;
                ldsm_x4(b0, b1, b2, b3, Ks + ((np * 2 + (lm >> 1)) * 8 + lr) * LD + ks * 16 + (lm & 1) * 8);
                mma_bf16(s[np * 2], a, b0, b1);
                mma_bf16(s[np * 2 + 1], a, b2, b3);
            }
        }
        __syncthreads();  // everyone is done reading Ks
        if (kt + 1 < ntiles) {
            if constexpr (F8) issue8(Kst, kb, kt + 1);
            else load_tile(Ks, kb, kt + 1);
        }
        cp_commit();

        // ---- online softmax (log2 domain) ----
        const bool need_mask = (int64_t)(kt + 1) * BK > mask_from;
        float mx0 = -INFINITY, mx1 = -INFINITY;
#pragma unroll
        for (int nb = 0; nb < NNB; ++nb) {
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                float v = s[nb][e] * sl2;
                if (need_mask) {
                    const int64_t key = (int64_t)kt * BK + nb * 8 + 2 * c + (e & 1);
                    const int64_t lim = pos0 + (e < 2 ? row_lo : row_hi) + 1;  // exclusive
                    if (key >= lim) v = -INFINITY;
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
        if constexpr (F8) land8(Vs, Vst);
        __syncthreads();

        // ---- O += P V ----
#pragma unroll
        for (int j = 0; j < BK / 16; ++j) {
            uint32_t a[4] = {pa[2 * j][0], pa[2 * j][1], pa[2 * j + 1][0], pa[2 * j + 1][1]};
#pragma unroll
            for (int dp = 0; dp < NDB / 2; ++dp) {
                uint32_t b0, b1, b2, b3;
                ldsm_x4_t(b0, b1, b2, b3, Vs + (j * 16 + (lm & 1) * 8 + lr) * LD + (dp * 2 + (lm >> 1)) * 8);
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
            *reinterpret_cast<uint32_t*>(out + ((int64_t)row_lo * N + h) * D + nb * 8 + 2 * c) =
                pack_bf16(o[nb][0] * i0, o[nb][1] * i0);
        if (row_hi < T)
            *reinterpret_cast<uint32_t*>(out + ((int64_t)row_hi * N + h) * D + nb * 8 + 2 * c) =
                pack_bf16(o[nb][2] * i1, o[nb][3] * i1);
    }
}

template <int D, int BK, typename KT = ik_bf16, int BQ = kBQ>
int launch_flash(const ik_bf16* q, const KT* kc, const KT* vc, ik_bf16* out, int T, int N, int NKV,
                 int64_t cap, int64_t pos0, float scale, cudaStream_t s) {
    constexpr size_t smem = flash_smem_bytes<D, BK, BQ>();
    // the dynamic-smem opt-in is per function and per device context; set it once per device
    static bool configured[16] = {};
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess) return IK_ERR_LAUNCH;
    if (dev < 0 || dev >= 16) return IK_ERR_UNSUPPORTED;
    if (!configured[dev]) {
        if (cudaFuncSetAttribute(flash_kernel<D, BK, KT, BQ>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem) !=
            cudaSuccess)
            return IK_ERR_LAUNCH;
        configured[dev] = true;
    }
    dim3 grid((T + BQ - 1) / BQ, N);
    flash_kernel<D, BK, KT, BQ><<<grid, 2 * BQ, smem, s>>>(q, kc, vc, out, T, N, N / NKV, cap, pos0, scale * kLog2e);
    return ik_launch_status();
}

}  // namespace

namespace {
int flash_launch(const ik_tensor* q, const ik_tensor* kc, const ik_tensor* vc, int64_t pos0, float scale, ik_tensor* out,
                 void* stream, bool f8kv) {
    IK_TRY(ik_check(q, IK_DTYPE_BF16, 3));
    IK_TRY(ik_check(kc, f8kv ? IK_DTYPE_FP8_E4M3 : IK_DTYPE_BF16, 3));
    IK_TRY(ik_check(vc, f8kv ? IK_DTYPE_FP8_E4M3 : IK_DTYPE_BF16, 3));
    IK_TRY(ik_check(out, IK_DTYPE_BF16, 3));
    const int64_t T = q->shape[0], N = q->shape[1], D = q->shape[2];
    const int64_t NKV = kc->shape[0], cap = kc->shape[1];
    if (vc->shape[0] != NKV || vc->shape[1] != cap || vc->shape[2] != D || kc->shape[2] != D) return IK_ERR_SHAPE;
    if (out->shape[0] != T || out->shape[1] != N || out->shape[2] != D) return IK_ERR_SHAPE;
    if (NKV <= 0 || N % NKV != 0 || pos0 < 0 || pos0 + T > cap) return IK_ERR_SHAPE;
    if (D != 64 && D != 128 && D != 256) return IK_ERR_SHAPE;
    if (T == 0 || N == 0) return IK_OK;
    if (T > 0x7fffffffLL / 2 || N > 65535 || (T + kBQ - 1) / kBQ > 0x7fffffffLL) return IK_ERR_SHAPE;
    if (!ik_aligned16(kc->data) || !ik_aligned16(vc->data) || (reinterpret_cast<uintptr_t>(q->data) & 3) ||
        (reinterpret_cast<uintptr_t>(out->data) & 3))
        return IK_ERR_STRIDE;
    if (D == 256 && !ik_aligned16(q->data)) return IK_ERR_STRIDE;  // Q is staged with 16 B cp.async
    cudaStream_t s = IK_STREAM(stream);
    const ik_bf16* qp = (const ik_bf16*)q->data;
    ik_bf16* op = (ik_bf16*)out->data;
    if (f8kv) {
        const uint8_t* kp = (const uint8_t*)kc->data;
        const uint8_t* vp = (const uint8_t*)vc->data;
        switch (D) {
            case 64: return launch_flash<64, 64>(qp, kp, vp, op, (int)T, (int)N, (int)NKV, cap, pos0, scale, s);
            case 128: return launch_flash<128, 64>(qp, kp, vp, op, (int)T, (int)N, (int)NKV, cap, pos0, scale, s);
            default: return launch_flash<256, 32>(qp, kp, vp, op, (int)T, (int)N, (int)NKV, cap, pos0, scale, s);
        }
    }
    const ik_bf16* kp = (const ik_bf16*)kc->data;
    const ik_bf16* vp = (const ik_bf16*)vc->data;
    // D = 256: 128 query rows per block (8 warps; the same bits as 64). INTARSIA_CUDA_FLASH_BQ=64 restores 64 (A/B, and
    // tests/flash_bq.rs, which flips it between calls: read per launch, a few hundred ns against a launch of ms).
    const char* bqv = D == 256 ? getenv("INTARSIA_CUDA_FLASH_BQ") : nullptr;
    const bool bq64 = bqv && atoi(bqv) == 64;
    switch (D) {
        case 64: return launch_flash<64, 64>(qp, kp, vp, op, (int)T, (int)N, (int)NKV, cap, pos0, scale, s);
        case 128: return launch_flash<128, 64>(qp, kp, vp, op, (int)T, (int)N, (int)NKV, cap, pos0, scale, s);
        default:
            if (bq64) return launch_flash<256, 32>(qp, kp, vp, op, (int)T, (int)N, (int)NKV, cap, pos0, scale, s);
            return launch_flash<256, 32, ik_bf16, 128>(qp, kp, vp, op, (int)T, (int)N, (int)NKV, cap, pos0, scale, s);
    }
}
}  // namespace

extern "C" int ik_attn_prefill_bf16_flash(const ik_tensor* q, const ik_tensor* kc, const ik_tensor* vc, int64_t pos0,
                                          float scale, ik_tensor* out, void* stream) {
    return flash_launch(q, kc, vc, pos0, scale, out, stream, false);
}

// fp8_e4m3 KV cache, bf16 q / out
extern "C" int ik_attn_prefill_f8kv_flash(const ik_tensor* q, const ik_tensor* kc, const ik_tensor* vc, int64_t pos0,
                                          float scale, ik_tensor* out, void* stream) {
    return flash_launch(q, kc, vc, pos0, scale, out, stream, true);
}
