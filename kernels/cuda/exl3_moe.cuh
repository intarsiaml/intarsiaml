// EXL3 routed experts (Op::MoeExperts, EXL3 encoding: crates/intarsia-core/src/quant/exl3_moe.rs): device code shared by
// exl3_moe.cu (grouping, input rotation, epilogues, combine, the ik_* entry points), exl3_moe_cp.cu (the decode-window
// grouped GEMV) and exl3_moe_rows.cu (the prompt-chunk grouped kernel).
//
// Ported from TensorFold's grouped EXL3 expert kernels as BertholomusAI's DeepSeek-V4.1 fork extends them:
//   https://github.com/bertholomus/TensorFold, branch deepseek-v41-tp2 @ af26d42b1f618e9e653903a6c043448b52729c64
//   (files last changed in 508bfb34), src/tensorfold/cuda/exl3/{experts_grouped.cuh, experts.cu}
//   Copyright 2026 TensorFold contributors; fork changes Copyright 2026 BertholomusAI (Albert Lee).
//   Licensed under the Apache License, Version 2.0 (LICENSES/Apache-2.0-TensorFold.txt, kernels/cuda/NOTICE-tensorfold).
//   The pinned DeepSeek-V4.1 family (branch deepseek-v41-zig @ 629c524a0ea57512a5342abb498d1b7c25353343,
//   zig/src/families/dsv41/exl3_experts.zig) launches the cubin built from these sources (tensorfold_exl3_experts_v19)
//   and holds no copy of them; the launch sequence ported here is the one it documents (decode window: grouping and
//   rotated rows, grouped_cp_kernel for gate / up, grouped_cp_kernel for down; prompt chunk: work list, grouped kernels,
//   gate / up epilogue, down combine).
//   The decode follows ExLlamaV3 (MIT, Copyright (c) 2025 Turboderp; LICENSES/exllamav3-MIT.txt).
// Changes made here (Apache-2.0 section 4(b)): torch-free (ik ABI); experts addressed through one i16 arena with an
// [E, 8] i32 table of 16-byte offsets and widths and one [E, 3H + 3I] f16 scales row per expert, instead of int64 pointer
// arrays and stacked suh/svh tensors; members as one list with per-place starts and counts (CSR) instead of a
// [places, maxm] table; the codebook is read on the device and dispatched with the width inside each
// program (no host-side codebook / width range); an offset outside the arena or a width that is no EXL3 width gives NaN
// instead of __trap; the ring of grouped_cp_kernel keeps 3 stages up to 5 bits and 2 above (static shared memory sized
// for 8 bits, no other change to the arithmetic); no programmatic dependent launch, readiness flags or L2 discards; the
// combine writes bf16 / f16 / f32.
#pragma once

#include "ik_launch.h"
#include "ik_quant.cuh"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include <type_traits>

namespace ik_exl3x {

constexpr float HAD_SCALE = 0.08838834764831845f;  // 1 / sqrt(128)
constexpr int META_COLS = 8;
constexpr int kGroupRows = 64;  // members of a prompt work-list group: one decode of a weight tile feeds kGroupRows / 16 member tiles

// One call's experts (the op's inputs 3..7).
struct Experts {
    const uint32_t* arena;  // the i16 arena as 32-bit words
    long long words;        // its length in 32-bit words
    const half* sc;         // scales [E, C]: suh_g [H] | suh_u [H] | svh_g [I] | svh_u [I] | suh_d [I] | svh_d [H]
    int C;
    const int* meta;  // [E, 8]: gate, up, down 16-byte offsets, gate, up, down k2
    const int* cbp;   // codebook [1]
    int E, H, I;
    __device__ __forceinline__ const half* suh_g(int e) const { return sc + (size_t)e * C; }
    __device__ __forceinline__ const half* suh_u(int e) const { return sc + (size_t)e * C + H; }
    __device__ __forceinline__ const half* svh_g(int e) const { return sc + (size_t)e * C + 2 * H; }
    __device__ __forceinline__ const half* svh_u(int e) const { return sc + (size_t)e * C + 2 * H + I; }
    __device__ __forceinline__ const half* suh_d(int e) const { return sc + (size_t)e * C + 2 * H + 2 * I; }
    __device__ __forceinline__ const half* svh_d(int e) const { return sc + (size_t)e * C + 2 * H + 3 * I; }
};

__host__ __device__ constexpr bool valid_k2(int k2, int cb) {
    return (k2 % 2 == 0 && k2 >= 2 && k2 <= 16) || ((k2 == 3 || k2 == 5 || k2 == 7) && cb == 2);
}

// Matrix j (0 gate, 1 up, 2 down) of expert e, [K, N]: its trellis words and width, or nullptr when the offset is outside
// the arena or the width is no EXL3 width with the call's codebook.
__device__ __forceinline__ const uint32_t* mat_words(const Experts& x, int e, int j, int K, int N, int cb, int& k2) {
    const int off = x.meta[e * META_COLS + j];
    k2 = x.meta[e * META_COLS + 3 + j];
    if (off < 0 || !valid_k2(k2, cb)) return nullptr;
    const long long need = (long long)(K >> 4) * (N >> 4) * 4 * k2;
    if ((long long)off * 4 + need > x.words) return nullptr;
    return x.arena + (size_t)off * 4;
}

// Two codebook values (CB 0 3inst, 1 mcg, 2 mul1) as a half2, bit-identical to ExLlamaV3's decode_3inst_2<cb>.
template <int CB>
__device__ __forceinline__ uint32_t cb_pair(uint32_t s0, uint32_t s1) {
    if constexpr (CB == 2) {
        const uint32_t x0 = s0 * 0x83DCD12Du, x1 = s1 * 0x83DCD12Du;
        const uint32_t sum0 = __dp4a(x0, 0x01010101u, 0x6400u);
        const uint32_t sum1 = __dp4a(x1, 0x01010101u, 0x6400u);
        const uint32_t hv = __byte_perm(sum0, sum1, 0x5410);
        half2 h = *reinterpret_cast<const half2*>(&hv);
        half2 r = __hfma2(h, __half2half2(__ushort_as_half(0x1eee)), __half2half2(__ushort_as_half(0xc931)));
        return *reinterpret_cast<uint32_t*>(&r);
    } else {
        uint32_t x0, x1;
        if constexpr (CB == 1) {
            x0 = s0 * 0xCBAC1FEDu;
            x1 = s1 * 0xCBAC1FEDu;
        } else {
            x0 = s0 * 89226354u + 64248484u;
            x1 = s1 * 89226354u + 64248484u;
        }
        x0 = (x0 & 0x8FFF8FFFu) ^ 0x3B603B60u;
        x1 = (x1 & 0x8FFF8FFFu) ^ 0x3B603B60u;
        uint32_t lo = __byte_perm(x0, x1, 0x5410);
        uint32_t hi = __byte_perm(x0, x1, 0x7632);
        half2 r = __hadd2(*reinterpret_cast<half2*>(&lo), *reinterpret_cast<half2*>(&hi));
        return *reinterpret_cast<uint32_t*>(&r);
    }
}

// K2 half-bits a value: a tile is 4 * K2 words; a lane's eight windows fall in NG runs of GV within two words.
template <int K2>
struct Fmt {
    static constexpr int TW = 4 * K2;
    static constexpr int LW = (TW + 31) / 32;
    static constexpr int GV = (K2 >= 13) ? 2 : ((K2 == 7 || (K2 >= 9 && K2 <= 12) || K2 == 16) ? 4 : 8);
    static constexpr int NG = 8 / GV;
    __host__ __device__ static constexpr int end(int p) { return (p >> 1) * K2 + ((p & 1) ? K2 : (K2 >> 1)); }
    __host__ __device__ static constexpr int off(int j) { return end(GV - 1) - end(j); }
};

template <int K2>
struct LaneMap {
    int hi[Fmt<K2>::NG], lo[Fmt<K2>::NG], sh[Fmt<K2>::NG];
    __device__ __forceinline__ explicit LaneMap(int lane) {
        constexpr int TW = Fmt<K2>::TW, GV = Fmt<K2>::GV;
#pragma unroll
        for (int g = 0; g < Fmt<K2>::NG; ++g) {
            const int last_end = Fmt<K2>::end(8 * lane + g * GV + GV - 1) + 128 * K2;
            const int hr = (last_end - 1) >> 5;
            hi[g] = hr % TW;
            lo[g] = (hr + TW - 1) % TW;
            sh[g] = (hr + 1) * 32 - last_end;
        }
    }
};

template <int LW>
__device__ __forceinline__ uint32_t fetch(const uint32_t (&w)[LW], int idx) {
    if constexpr (LW == 1) {
        return __shfl_sync(0xffffffffu, w[0], idx);
    } else {
        const uint32_t a = __shfl_sync(0xffffffffu, w[0], idx & 31);
        const uint32_t b = __shfl_sync(0xffffffffu, w[1], idx & 31);
        return idx < 32 ? a : b;
    }
}

// This lane's eight values of a tile as the B fragments of its two n8 halves.
template <int CB, int K2>
__device__ __forceinline__ void decode_tile(const uint32_t (&w)[Fmt<K2>::LW], const LaneMap<K2>& m, int lane,
                                            uint32_t (&b0)[2], uint32_t (&b1)[2]) {
    uint32_t st[8];
    if constexpr (K2 == 8) {
        const uint32_t p = __shfl_sync(0xffffffffu, w[0], (lane + 31) & 31);
        const uint32_t s = __funnelshift_r(w[0], p, 20);
        st[0] = (s >> 8) & 0xffffu;
        st[1] = (s >> 4) & 0xffffu;
        st[2] = s & 0xffffu;
        st[3] = w[0] >> 16;
        st[4] = (w[0] >> 12) & 0xffffu;
        st[5] = (w[0] >> 8) & 0xffffu;
        st[6] = (w[0] >> 4) & 0xffffu;
        st[7] = w[0] & 0xffffu;
    } else {
        constexpr int GV = Fmt<K2>::GV, NG = Fmt<K2>::NG;
#pragma unroll
        for (int g = 0; g < NG; ++g) {
            const uint32_t whi = fetch<Fmt<K2>::LW>(w, m.hi[g]);
            const uint32_t wlo = fetch<Fmt<K2>::LW>(w, m.lo[g]);
            const uint64_t mm = ((((uint64_t)wlo) << 32) | whi) >> m.sh[g];
#pragma unroll
            for (int j = 0; j < GV; ++j) st[g * GV + j] = (uint32_t)(mm >> Fmt<K2>::off(j)) & 0xffffu;
        }
    }
    b0[0] = cb_pair<CB>(st[0], st[1]);
    b0[1] = cb_pair<CB>(st[2], st[3]);
    b1[0] = cb_pair<CB>(st[4], st[5]);
    b1[1] = cb_pair<CB>(st[6], st[7]);
}

__device__ __forceinline__ void mma16816(float (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

__device__ __forceinline__ uint32_t load_pair(const half* x, bool ok) {
    return ok ? *reinterpret_cast<const uint32_t*>(x) : 0u;
}

template <int K2>
__device__ __forceinline__ void load_words(uint32_t (&dst)[Fmt<K2>::LW], const uint32_t* p, int lane) {
    constexpr int TW = Fmt<K2>::TW;
#pragma unroll
    for (int l = 0; l < Fmt<K2>::LW; ++l) {
        if constexpr ((TW % 32) == 0)
            dst[l] = __ldg(p + l * 32);
        else
            dst[l] = (l * 32 + lane < TW) ? __ldg(p + l * 32) : 0u;
    }
}

__device__ __forceinline__ void cp_async16(void* smem, const void* gmem, bool ok) {
    const unsigned s = (unsigned)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(s), "l"(gmem), "r"(ok ? 16 : 0));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N_>
__device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N_)); }

// Run f(integral_constant<CB>, integral_constant<K2>) for the call's codebook and a matrix's width; false (f not run) when
// the pair is no EXL3 width (the caller then writes NaN). The widths ExLlamaV3 writes: 1..8 bits with every codebook,
// 1.5 / 2.5 / 3.5 with mul1.
template <class F>
__device__ __forceinline__ bool with_width(int cb, int k2, F&& f) {
#define IKX_W(CB_, K2_)                                                                     \
    case K2_:                                                                               \
        f(std::integral_constant<int, CB_>{}, std::integral_constant<int, K2_>{});          \
        return true;
#define IKX_EVEN(CB_) IKX_W(CB_, 2) IKX_W(CB_, 4) IKX_W(CB_, 6) IKX_W(CB_, 8) IKX_W(CB_, 10) IKX_W(CB_, 12) IKX_W(CB_, 14) IKX_W(CB_, 16)
    switch (cb) {
        case 0:
            switch (k2) { IKX_EVEN(0) }
            break;
        case 1:
            switch (k2) { IKX_EVEN(1) }
            break;
        case 2:
            switch (k2) { IKX_EVEN(2) IKX_W(2, 3) IKX_W(2, 5) IKX_W(2, 7) }
            break;
        default:
            break;
    }
    return false;
#undef IKX_EVEN
#undef IKX_W
}

// Walsh-Hadamard transform of 128 values, 4 a lane, fixed butterfly order (strides 1, 2 in registers, 4..64 across lanes).
__device__ __forceinline__ void fwht128(float (&v)[4], int lane) {
    float a = v[0] + v[1], b = v[0] - v[1], c = v[2] + v[3], d = v[2] - v[3];
    v[0] = a + c;
    v[1] = b + d;
    v[2] = a - c;
    v[3] = b - d;
#pragma unroll
    for (int m = 1; m < 32; m <<= 1) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            float o = __shfl_xor_sync(0xffffffffu, v[j], m);
            v[j] = (lane & m) ? o - v[j] : v[j] + o;
        }
    }
}

// Launch arguments of the grouped kernels.
struct GroupedArgs {
    const half* x0;  // rotated rows of matrix 0 (gate, or down's xd)
    const half* x1;  // ... of matrix 1 (up; = x0 for down)
    Experts ex;
    int j0;  // the meta column of matrix 0 (0 gate/up: matrices 0 and 1; 2 down)
    const int* uids;    // [places] expert of each place (distinct routed experts in id order)
    const int* ucount;  // [1] places used
    const int* mstart;  // [places] first member of each place in `members`
    const int* mcount;  // [places] members of each place
    const int* members; // row * 32 + slot of each member, place by place, rows in order
    float* z;
    int K, N, P, SK, maxm, slots, mats;  // maxm: a host bound of any place's members (grid size of the decode kernel)
    int nexp;         // grid.x expert places (decode) or the work list's length (prompt)
    const int* work;  // prompt: (place, member group) pairs; nullptr: decode
};

// exl3_moe_cp.cu / exl3_moe_rows.cu
int launch_cp(const GroupedArgs& a, cudaStream_t s);
int launch_dense_cp(const half* xh, const uint32_t* T, float* part, const int* cbp, int k2, int M, int K, int N, int S,
                    cudaStream_t s);
int launch_rows(const GroupedArgs& a, bool fold, cudaStream_t s);

}  // namespace ik_exl3x
