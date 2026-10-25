// Lane l8 candidates (experimental kernels live here until the winner is moved into kernels/hip/ik_f8_gemm.h).
// Every variant keeps the reference's per-output k order: for each 64-byte stage, for each group g of 32 k bytes, the "lo"
// WMMA (bytes 0..7 of each lane half's 16) then the "hi" WMMA (bytes 8..15) on the same accumulator. Only the LDS layout,
// the order in which independent accumulators are issued, the global-load distance and the tile shape change.
#pragma once
#include "ik_f8_gemm.h"

// SWZ: 64-byte rows with the 16-byte chunk XOR-swizzled by (row >> 2) & 3 instead of an 80-byte padded stride.
// PF: global-load distance in stages (1 = as the reference, 2 = two register sets, loads issued two stages ahead).
// ORD: 0 = lo+hi back to back per accumulator (reference), 1 = all lo WMMAs of a group, then all hi.
// LB: 0 = __syncthreads (workgroup fence over all address spaces: global_inv + load-counter drain on gfx12),
//     1 = LDS-only fence + s_barrier (global loads stay in flight, L0/L1 lines are kept).
// RD: 0 = fragments of one 32-k group at a time, 1 = both groups' fragments read before the first WMMA.
template <int WN, int WM, int TN, int TM, bool SWZ, int PF, int ORD, int RD, int LB, typename OutT>
__global__ void __launch_bounds__(WN * WM * 32)
f8v_kernel(const uint8_t* __restrict__ xq, const uint8_t* __restrict__ w, const float* __restrict__ sc,
           OutT* __restrict__ out, int M, int64_t N, int K, int vec_out) {
    if (!(sc[1] > 0.f)) return;
    constexpr int NT = WN * WM * IK_WARP;
    constexpr int BN = WN * TN * 16, BM = WM * TM * 16;
    constexpr int CHW = BN * 4, CHX = BM * 4;
    constexpr int CW = (CHW + NT - 1) / NT, CX = (CHX + NT - 1) / NT;
    constexpr int LD = SWZ ? 64 : 80;
    __shared__ __align__(16) uint8_t sw[2][BN * LD];
    __shared__ __align__(16) uint8_t sx[2][BM * LD];

    const int tid = threadIdx.x, wave = tid >> 5, lane = tid & 31;
    const int wn = wave % WN, wm = wave / WN;
    const int m_blk = blockIdx.x * BM;
    const int64_t n_blk = (int64_t)blockIdx.y * BN;
    const int nk = (K + kF8BK - 1) / kF8BK;

    auto soff = [](int row, int ch) -> int {
        return SWZ ? row * 64 + ((ch ^ ((row >> 2) & 3)) << 4) : row * 80 + ch * 16;
    };
    auto gload = [&](int k0, uint4(&rw)[CW], uint4(&rx)[CX]) {
#pragma unroll
        for (int c = 0; c < CW; ++c) {
            const int idx = tid + c * NT, r = idx >> 2, ch = idx & 3;
            const int64_t n = n_blk + r;
            const int k = k0 + ch * 16;
            rw[c] = (idx < CHW && n < N && k < K) ? *reinterpret_cast<const uint4*>(w + n * K + k) : make_uint4(0, 0, 0, 0);
        }
#pragma unroll
        for (int c = 0; c < CX; ++c) {
            const int idx = tid + c * NT, r = idx >> 2, ch = idx & 3;
            const int m = m_blk + r;
            const int k = k0 + ch * 16;
            rx[c] = (idx < CHX && m < M && k < K) ? *reinterpret_cast<const uint4*>(xq + (int64_t)m * K + k)
                                                   : make_uint4(0, 0, 0, 0);
        }
    };
    auto sstore = [&](int buf, const uint4(&rw)[CW], const uint4(&rx)[CX]) {
#pragma unroll
        for (int c = 0; c < CW; ++c) {
            const int idx = tid + c * NT;
            if (idx < CHW) *reinterpret_cast<uint4*>(&sw[buf][soff(idx >> 2, idx & 3)]) = rw[c];
        }
#pragma unroll
        for (int c = 0; c < CX; ++c) {
            const int idx = tid + c * NT;
            if (idx < CHX) *reinterpret_cast<uint4*>(&sx[buf][soff(idx >> 2, idx & 3)]) = rx[c];
        }
    };

    ik_dq_v8f acc[TN][TM];
#pragma unroll
    for (int i = 0; i < TN; ++i)
#pragma unroll
        for (int j = 0; j < TM; ++j)
#pragma unroll
            for (int e = 0; e < 8; ++e) acc[i][j][e] = 0.f;

    const int lr = lane & 15, lhc = lane >> 4;  // lane half: chunk offset 0 / 1 inside a 32-byte group
    // per-lane LDS read offsets (bytes) of fragment row i for group g
    int aoff[TN][2], boff[TM][2];
#pragma unroll
    for (int i = 0; i < TN; ++i)
#pragma unroll
        for (int g = 0; g < 2; ++g) aoff[i][g] = soff(wn * TN * 16 + i * 16 + lr, g * 2 + lhc);
#pragma unroll
    for (int j = 0; j < TM; ++j)
#pragma unroll
        for (int g = 0; g < 2; ++g) boff[j][g] = soff(wm * TM * 16 + j * 16 + lr, g * 2 + lhc);

    auto mma_group = [&](const uint4(&af)[TN], const uint4(&bf)[TM]) {
        if constexpr (ORD == 2) {  // NEGATIVE CONTROL: hi before lo on the same accumulator (must change bits)
#pragma unroll
            for (int i = 0; i < TN; ++i)
#pragma unroll
                for (int j = 0; j < TM; ++j) {
                    acc[i][j] = ik_f8_wmma(af[i].z, af[i].w, bf[j].z, bf[j].w, acc[i][j]);
                    acc[i][j] = ik_f8_wmma(af[i].x, af[i].y, bf[j].x, bf[j].y, acc[i][j]);
                }
        } else if constexpr (ORD == 0) {
#pragma unroll
            for (int i = 0; i < TN; ++i)
#pragma unroll
                for (int j = 0; j < TM; ++j) {
                    acc[i][j] = ik_f8_wmma(af[i].x, af[i].y, bf[j].x, bf[j].y, acc[i][j]);
                    acc[i][j] = ik_f8_wmma(af[i].z, af[i].w, bf[j].z, bf[j].w, acc[i][j]);
                }
        } else {
#pragma unroll
            for (int i = 0; i < TN; ++i)
#pragma unroll
                for (int j = 0; j < TM; ++j) acc[i][j] = ik_f8_wmma(af[i].x, af[i].y, bf[j].x, bf[j].y, acc[i][j]);
#pragma unroll
            for (int i = 0; i < TN; ++i)
#pragma unroll
                for (int j = 0; j < TM; ++j) acc[i][j] = ik_f8_wmma(af[i].z, af[i].w, bf[j].z, bf[j].w, acc[i][j]);
        }
    };
    auto compute = [&](int buf) {
        if constexpr (RD == 0) {
#pragma unroll
            for (int g = 0; g < 2; ++g) {
                uint4 af[TN], bf[TM];
#pragma unroll
                for (int i = 0; i < TN; ++i) af[i] = *reinterpret_cast<const uint4*>(&sw[buf][aoff[i][g]]);
#pragma unroll
                for (int j = 0; j < TM; ++j) bf[j] = *reinterpret_cast<const uint4*>(&sx[buf][boff[j][g]]);
                mma_group(af, bf);
            }
        } else {
            uint4 af[2][TN], bf[2][TM];
#pragma unroll
            for (int g = 0; g < 2; ++g) {
#pragma unroll
                for (int i = 0; i < TN; ++i) af[g][i] = *reinterpret_cast<const uint4*>(&sw[buf][aoff[i][g]]);
#pragma unroll
                for (int j = 0; j < TM; ++j) bf[g][j] = *reinterpret_cast<const uint4*>(&sx[buf][boff[j][g]]);
            }
#pragma unroll
            for (int g = 0; g < 2; ++g) mma_group(af[g], bf[g]);
        }
    };

    auto bar = [&]() {
        if constexpr (LB) {
            __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");
            __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");
        } else {
            __syncthreads();
        }
    };
    uint4 rwA[CW], rxA[CX], rwB[CW], rxB[CX];
    if constexpr (PF == 1) {
        gload(0, rwA, rxA);
        sstore(0, rwA, rxA);
        bar();
        for (int kt = 0; kt < nk; ++kt) {
            const int buf = kt & 1;
            if (kt + 1 < nk) gload((kt + 1) * kF8BK, rwA, rxA);
            compute(buf);
            if (kt + 1 < nk) sstore(buf ^ 1, rwA, rxA);
            bar();
        }
    } else {
        gload(0, rwA, rxA);
        sstore(0, rwA, rxA);
        if (nk > 1) gload(kF8BK, rwA, rxA);
        if (nk > 2) gload(2 * kF8BK, rwB, rxB);
        bar();
        for (int kt = 0; kt < nk; kt += 2) {
            // even stage: regs A hold stage kt + 1, regs B stage kt + 2 (both loads already in flight)
            if (kt + 1 < nk) sstore(1, rwA, rxA);
            if (kt + 3 < nk) gload((kt + 3) * kF8BK, rwA, rxA);
            compute(0);
            bar();
            if (kt + 1 < nk) {
                if (kt + 2 < nk) sstore(0, rwB, rxB);
                if (kt + 4 < nk) gload((kt + 4) * kF8BK, rwB, rxB);
                compute(1);
                bar();
            }
        }
    }

    const float so = sc[0] * sc[1];
#pragma unroll
    for (int i = 0; i < TN; ++i) {
#pragma unroll
        for (int j = 0; j < TM; ++j) {
            const int m = m_blk + wm * TM * 16 + j * 16 + lr;
            const int64_t n = n_blk + wn * TN * 16 + i * 16 + (lane >> 4) * 8;
            if (m >= M || n >= N) continue;
            float v[8];
#pragma unroll
            for (int e = 0; e < 8; ++e) v[e] = acc[i][j][e] * so;
            OutT* o = out + (int64_t)m * N + n;
            if (vec_out) {
                if (sizeof(OutT) == 2) {
                    *reinterpret_cast<uint4*>(o) = ik_pack8(v);
                } else {
                    reinterpret_cast<float4*>(o)[0] = make_float4(v[0], v[1], v[2], v[3]);
                    reinterpret_cast<float4*>(o)[1] = make_float4(v[4], v[5], v[6], v[7]);
                }
            } else {
                for (int e = 0; e < 8 && n + e < N; ++e) ik_store(o + e, v[e]);
            }
        }
    }
}

template <int WN, int WM, int TN, int TM, bool SWZ, int PF, int ORD, int RD, int LB = 0>
static void lv(const uint8_t* xq, const uint8_t* w, const float* sc, ik_bf16* out, int M, int64_t N, int K, hipStream_t s) {
    constexpr int BN = WN * TN * 16, BM = WM * TM * 16;
    dim3 grid((unsigned)(((int64_t)M + BM - 1) / BM), (unsigned)((N + BN - 1) / BN));
    f8v_kernel<WN, WM, TN, TM, SWZ, PF, ORD, RD, LB, ik_bf16><<<grid, WN * WM * 32, 0, s>>>(xq, w, sc, out, M, N, K, 1);
}

// name encodes: WN,WM,TN,TM / s = swizzled 64 B rows (p = 80 B padded) / pf (loads in flight) / o (issue order) / r (frag reads) / lb
#define L8_VARIANTS(X)                                                           \
    X("NEG 4442 hi-before-lo", (lv<4, 4, 4, 2, false, 1, 2, 0, 1>))             \
    X("4442 p pf1 o0 r0 lb0", (lv<4, 4, 4, 2, false, 1, 0, 0, 0>))             \
    X("4442 p pf1 o0 r0 lb1", (lv<4, 4, 4, 2, false, 1, 0, 0, 1>))             \
    X("4442 s pf1 o0 r0 lb1", (lv<4, 4, 4, 2, true, 1, 0, 0, 1>))              \
    X("4442 p pf2 o0 r0 lb1", (lv<4, 4, 4, 2, false, 2, 0, 0, 1>))             \
    X("4442 s pf2 o0 r0 lb1", (lv<4, 4, 4, 2, true, 2, 0, 0, 1>))              \
    X("4442 p pf1 o1 r0 lb1", (lv<4, 4, 4, 2, false, 1, 1, 0, 1>))             \
    X("4442 p pf1 o0 r1 lb1", (lv<4, 4, 4, 2, false, 1, 0, 1, 1>))             \
    X("4442 s pf2 o1 r1 lb1", (lv<4, 4, 4, 2, true, 2, 1, 1, 1>))              \
    X("4424 p pf1 o0 r0 lb0", (lv<4, 4, 2, 4, false, 1, 0, 0, 0>))             \
    X("4424 p pf1 o0 r0 lb1", (lv<4, 4, 2, 4, false, 1, 0, 0, 1>))             \
    X("4424 p pf2 o0 r0 lb1", (lv<4, 4, 2, 4, false, 2, 0, 0, 1>))             \
    X("4424 s pf2 o1 r1 lb1", (lv<4, 4, 2, 4, true, 2, 1, 1, 1>))              \
    X("4224 p pf1 o0 r0 lb1", (lv<4, 2, 2, 4, false, 1, 0, 0, 1>))             \
    X("4224 p pf2 o0 r0 lb1", (lv<4, 2, 2, 4, false, 2, 0, 0, 1>))             \
    X("4444 s pf1 o0 r0 lb1", (lv<4, 4, 4, 4, true, 1, 0, 0, 1>))              \
    X("4444 s pf2 o0 r0 lb1", (lv<4, 4, 4, 4, true, 2, 0, 0, 1>))              \
    X("4444 s pf2 o1 r0 lb1", (lv<4, 4, 4, 4, true, 2, 1, 0, 1>))              \
    X("4244 s pf2 o1 r0 lb1", (lv<4, 2, 4, 4, true, 2, 1, 0, 1>))
