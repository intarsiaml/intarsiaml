// matmul_mxfp4_a8_mma (lane W4A8): MXFP4 weights x dynamic per-token fp8 (e4m3) activations, the prefill / large-M GEMM on the
// gfx12 fp8 x fp8 WMMA (v_wmma_f32_16x16x16_fp8_fp8). Any M >= 1 (used for M > 16).
//   w     u8  [N, K/2]   two e2m1 codes per byte, element 2i in the low nibble
//   scale u8  [N, K/32]  e8m0 (2^(b - 127)), plain row-major (checkpoint layout)
//   x     bf16 [M, K]    quantised per token by ik_mx_a8.h's pre-pass: xs[m] = max(amax_m / 448, 1 / (448 * 512)),
//                        xq[m, k] = e4m3(clamp(x / xs, +-448)) RNE
//   out[m, n] = round(xs[m] * sum_g 2^(sw[n, g] - 127) * sum_{e < 32} e2m1(w[n, 32g + e]) * xq[m, 32g + e])
// Weights reach the matrix core EXACTLY: every e2m1 magnitude {0, .5, 1, 1.5, 2, 3, 4, 6} is an e4m3 value. Two paths, same
// math, picked per call on the device (no host sync) by a pre-pass over the scales:
//   fold      (the normal case)  every row's scales span <= 14 octaves (max - min <= 14, no 255): a code of group scale s becomes
//             e4m3(mag * 2^t), t = s - rowmax + 6 in [-8, 6], which is EXACT, so the group scales are folded into the weight
//             bytes and one fp8 GEMM runs over the whole K (the structure of mxfp4.hip's mx_fold_gemm_kernel, which also folds the
//             activations; here the activations are the plain fp8 ones). out = acc * xs[m] * 2^(rowmax - 133).
//   per-group (a row spans more, or a scale byte is 255 = NaN)  the weight bytes are the bare e2m1 values (scale 1); each
//             group of 32 k runs its two WMMAs into a zeroed accumulator tile and acc += group * 2^(sw - 127) per output
//             feature (per accumulator element of the C fragment), then out = acc * xs[m]. Always exact, ~25 % more VALU.
// INTARSIA_HIP_MXFP4_A8_PERGROUP=1 forces the per-group path (tests: real checkpoints never leave the fold path).
// INTARSIA_HIP_MXFP4_A8_TILE=WN,WM,TN,TM forces one tile (sweeps; the four tiles of the table are accepted).
// Block: WN x WM waves, each wave TN x TM fragments of 16 x 16 (weights x tokens); 64 k per stage, double-buffered through
// registers, LDS rows padded (ik_f8_gemm.h's layout). Fragments / k layout as ik_f8_gemm.h (both operands permute k alike).
// K % 32 == 0 (a K % 64 == 32 tail stage is zero padded). Deterministic, fixed reduction order, independent of M.
#include "ik_dq_gemm.h"
#include "ik_mx_a8.h"

#include <stdio.h>

namespace {

constexpr int kA8G = 32;            // elements per e8m0 scale
constexpr int kA8BK = 64;           // k per stage of the shipped tiles: two groups (a tile may take 32: one group, see a8_gemm_kernel's BK)
constexpr int kA8LD = kA8BK + 16;   // LDS row stride in bytes of a 64-k stage (ik_f8_gemm.h's kF8LD)

typedef int a8_v2i __attribute__((ext_vector_type(2)));
IK_DEVI ik_dq_v8f a8_wmma(uint32_t alo, uint32_t ahi, uint32_t blo, uint32_t bhi, ik_dq_v8f c) {
    const uint2 a = make_uint2(alo, ahi), b = make_uint2(blo, bhi);
    return __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(__builtin_bit_cast(a8_v2i, a), __builtin_bit_cast(a8_v2i, b), c);
}

// e8m0 byte -> 2^(b - 127); 0 -> 2^-127 (f32 subnormal), 255 -> NaN (mxfp4.hip's mx_e8m0)
IK_DEVI float a8_e8m0(uint32_t b) {
    return b == 0u ? __uint_as_float(0x00400000u) : (b == 255u ? __uint_as_float(0x7FC00000u) : __uint_as_float(b << 23));
}

// byte m of row t + 8 = e4m3(mag[m] * 2^t), t = -8 .. 6 (exact; mxfp4.hip's kMxFoldTab, checked by scripts/mx/fold_table.py)
__constant__ uint32_t kA8FoldTab[15][2] = {
    {0x03020100u, 0x0c080604u}, {0x06040200u, 0x14100c08u}, {0x0c080400u, 0x1c181410u}, {0x14100800u, 0x24201c18u},
    {0x1c181000u, 0x2c282420u}, {0x24201800u, 0x34302c28u}, {0x2c282000u, 0x3c383430u}, {0x34302800u, 0x44403c38u},
    {0x3c383000u, 0x4c484440u}, {0x44403800u, 0x54504c48u}, {0x4c484000u, 0x5c585450u}, {0x54504800u, 0x64605c58u},
    {0x5c585000u, 0x6c686460u}, {0x64605800u, 0x74706c68u}, {0x6c686000u, 0x7c787470u},
};
constexpr uint32_t kA8Tab0 = 0x3c383000u, kA8Tab1 = 0x4c484440u;  // the t = 0 row: bytes 00 30 38 3c 40 44 48 4c

// 8 e2m1 codes (element j = nibble j) -> 8 e4m3 bytes (element j = byte j) through the magnitude table, sign bit 3 -> 7
// (mxfp4.hip's mx_fold8)
IK_DEVI uint2 a8_fold8(uint32_t w, uint32_t t0, uint32_t t1) {
    const uint32_t ev = w & 0x0F0F0F0Fu, od = (w >> 4) & 0x0F0F0F0Fu;
    const uint32_t be = __builtin_amdgcn_perm(t1, t0, ev & 0x07070707u) | ((ev & 0x08080808u) << 4);
    const uint32_t bo = __builtin_amdgcn_perm(t1, t0, od & 0x07070707u) | ((od & 0x08080808u) << 4);
    return make_uint2(__builtin_amdgcn_perm(bo, be, 0x05010400u), __builtin_amdgcn_perm(bo, be, 0x07030602u));
}

// Block barrier that waits only for this wave's LDS operations (the fence of __syncthreads also waits for every global load in flight, which
// ends the prefetch of the next stages at each barrier). The kernel's global operands are read-only inputs, nothing else needs ordering.
IK_DEVI void a8_lds_barrier() { asm volatile("s_wait_dscnt 0x0\n\ts_barrier_signal -1\n\ts_barrier_wait -1" ::: "memory"); }

// element i (0..3) of four values, selected by a lane-dependent i with v_cndmask (no register-array indexing)
IK_DEVI uint32_t a8_pick4(int i, uint32_t a, uint32_t b, uint32_t c, uint32_t d) { return i == 0 ? a : (i == 1 ? b : (i == 2 ? c : d)); }

// Pre-pass over the weight scales, one warp per row: wref[n] = largest scale byte of row n; flag |= 1 when some row cannot be
// folded exactly (span > 14 octaves, or a 255 = NaN scale). flag is zeroed before (hipMemsetAsync).
__global__ void __launch_bounds__(256) a8_wspan_kernel(const uint8_t* __restrict__ ws, uint8_t* __restrict__ wref,
                                                       int* __restrict__ flag, int64_t N, int KG) {
    const int64_t n = (int64_t)blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (n >= N) return;  // whole warp
    const int lane = threadIdx.x & 31;
    const uint8_t* p = ws + n * KG;
    int mx = 0, mn = 255;
    for (int g = lane; g < KG; g += 32) {
        const int b = p[g];
        mx = max(mx, b);
        mn = min(mn, b);
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        mx = max(mx, __shfl_xor(mx, o, 32));
        mn = min(mn, __shfl_xor(mn, o, 32));
    }
    if (lane == 0) {
        wref[n] = (uint8_t)mx;
        if (mx == 255 || mx - mn > 14) atomicOr(flag, 1);
    }
}

struct A8Args {
    const uint8_t* xq;
    const float* xs;
    const uint8_t* w;
    const uint8_t* ws;
    const uint8_t* wref;
    const int* flag;
    int M;
    int64_t N;
    int K;
    int vec_out;
};

// FOLD: runs when flag == 0 (every row foldable); !FOLD: runs when flag != 0. The other launch exits at once.
// PD: global-load prefetch distance in stages (1: the next stage's data is in flight during this stage's WMMAs; 2: the one after, two register sets).
// BK: k per stage, 64 (two groups) or 32 (one group); DIAG: 0 shipped, 1 = diagnostic build without the weight decode, 2 = without the WMMAs
// (wrong results, timing only). The k order of every accumulator is the same for any BK: stages run in ascending k, a stage's WMMAs in k order.
template <bool FOLD, int WN, int WM, int TN, int TM, int BK, int DIAG, int PD, int ORD, int BAR, int WB, typename OutT>
__global__ void __launch_bounds__(WN * WM * 32) a8_gemm_kernel(A8Args a, OutT* __restrict__ out) {
    if ((*a.flag == 0) != FOLD) return;
    constexpr int NT = WN * WM * IK_WARP;
    constexpr int BN = WN * TN * 16, BM = WM * TM * 16;
    constexpr int GPK = BK / kA8G, XPK = BK / 16;  // weight groups / 16-byte x chunks per row and stage
    constexpr int kLD = BK + 16;                   // LDS row stride in bytes (ik_f8_gemm.h's kF8LD for BK 64)
    constexpr int CHW = BN * GPK, CHX = BM * XPK;  // 16-byte chunks of one stage: weights one group (32 codes), x 16 bytes
    constexpr int CW = (CHW + NT - 1) / NT, CX = (CHX + NT - 1) / NT;
    constexpr int CW2 = (CHW * 2 + NT - 1) / NT;  // WB: half-group (16 codes, 8 B) chunks per thread
    static_assert(!WB || (FOLD && BK == 64), "WB: fold path, 64-k stages");
    constexpr int SS = FOLD ? 4 : 2 * GPK * BN;  // per-group path: f32 scales [buf][group][n]
    __shared__ __align__(16) uint8_t sw[2][BN * kLD];
    __shared__ __align__(16) uint8_t sx[2][BM * kLD];
    __shared__ __align__(16) float sS[SS];

    const uint8_t* __restrict__ xq = a.xq;
    const uint8_t* __restrict__ w = a.w;
    const uint8_t* __restrict__ ws = a.ws;
    const int M = a.M, K = a.K;
    const int64_t N = a.N;
    const int tid = threadIdx.x, wave = tid >> 5, lane = tid & 31;
    const int wn = wave % WN, wm = wave / WN;
    const int m_blk = blockIdx.x * BM;
    const int64_t n_blk = (int64_t)blockIdx.y * BN;
    const int nk = (K + BK - 1) / BK, KG = K / kA8G;

    struct Stage {
        uint4 rw[WB ? 1 : CW], rx[CX];
        int rt[WB ? 1 : CW];  // FOLD: fold shift of the chunk's group; else: its e8m0 byte
        uint2 rw2[WB ? CW2 : 1];  // WB: 8 B (16 codes) per chunk
        int rt2[WB ? CW2 : 1];
    };
    Stage st[PD];  // register sets of the stages in flight: stage k's data is in st[k % PD]
    int wr[CW];  // FOLD: row reference (largest scale byte)
#pragma unroll
    for (int c = 0; c < CW; ++c) {
        wr[c] = 127;
        if constexpr (FOLD && !WB) {
            const int64_t n = n_blk + ((tid + c * NT) / GPK);
            if (n < N) wr[c] = (int)a.wref[n];
        }
    }
    int wr2[WB ? CW2 : 1];
    if constexpr (WB) {
#pragma unroll
        for (int c = 0; c < CW2; ++c) {
            wr2[c] = 127;
            const int64_t n = n_blk + ((tid + c * NT) / (GPK * 2));
            if (n < N) wr2[c] = (int)a.wref[n];
        }
    }
    auto gload = [&](Stage& st, int k0) {
        if constexpr (DIAG == 3) k0 = 0;  // every stage re-reads the first one: the loads always hit (timing only)
        if constexpr (WB) {
#pragma unroll
            for (int c = 0; c < CW2; ++c) {
                const int idx = tid + c * NT, r = idx / (GPK * 2), hh = idx % (GPK * 2);
                const int64_t n = n_blk + r;
                const int kg = k0 + (hh >> 1) * kA8G;
                if (idx < CHW * 2 && n < N && kg < K) {
                    st.rw2[c] = *reinterpret_cast<const uint2*>(w + n * (K / 2) + kg / 2 + (hh & 1) * 8);
                    st.rt2[c] = (int)ws[n * KG + kg / kA8G] - wr2[c] + 6;
                } else {
                    st.rw2[c] = make_uint2(0, 0);
                    st.rt2[c] = 0;
                }
            }
        }
#pragma unroll
        for (int c = 0; c < (WB ? 0 : CW); ++c) {
            const int idx = tid + c * NT, r = idx / GPK, ch = idx % GPK;
            const int64_t n = n_blk + r;
            const int kg = k0 + ch * kA8G;
            if (idx < CHW && n < N && kg < K) {
                st.rw[c] = *reinterpret_cast<const uint4*>(w + n * (K / 2) + kg / 2);
                const int s = (int)ws[n * KG + kg / kA8G];
                if constexpr (FOLD) st.rt[c] = s - wr[c] + 6;
                else st.rt[c] = s;
            } else {
                st.rw[c] = make_uint4(0, 0, 0, 0);
                st.rt[c] = FOLD ? 0 : 127;  // zero codes; a finite scale so 0 * scale stays 0
            }
        }
#pragma unroll
        for (int c = 0; c < CX; ++c) {
            const int idx = tid + c * NT, r = idx / XPK, ch = idx % XPK;
            const int m = m_blk + r;
            const int k = k0 + ch * 16;
            st.rx[c] = (idx < CHX && m < M && k < K) ? *reinterpret_cast<const uint4*>(xq + (int64_t)m * K + k) : make_uint4(0, 0, 0, 0);
        }
    };
    auto sstore = [&](const Stage& st, int buf) {
        if constexpr (WB) {
#pragma unroll
            for (int c = 0; c < CW2; ++c) {
                const int idx = tid + c * NT;
                if (idx < CHW * 2) {
                    const int r = idx / (GPK * 2), hh = idx % (GPK * 2);
                    const int ti = min(max(st.rt2[c], -8), 6) + 8;  // always in range when the flag is 0
                    const uint32_t t0 = kA8FoldTab[ti][0], t1 = kA8FoldTab[ti][1];
                    const uint2 p0 = a8_fold8(st.rw2[c].x, t0, t1), p1 = a8_fold8(st.rw2[c].y, t0, t1);
                    *reinterpret_cast<uint4*>(&sw[buf][r * kLD + (hh >> 1) * kA8G + (hh & 1) * 16]) = make_uint4(p0.x, p0.y, p1.x, p1.y);
                }
            }
        }
#pragma unroll
        for (int c = 0; c < (WB ? 0 : CW); ++c) {
            const int idx = tid + c * NT;
            if (idx < CHW) {
                const int r = idx / GPK, ch = idx % GPK;
                uint32_t t0 = kA8Tab0, t1 = kA8Tab1;
                if constexpr (FOLD) {
                    const int ti = min(max(st.rt[c], -8), 6) + 8;  // always in range when the flag is 0
                    t0 = kA8FoldTab[ti][0];
                    t1 = kA8FoldTab[ti][1];
                } else {
                    sS[(buf * GPK + ch) * BN + r] = a8_e8m0((uint32_t)st.rt[c]);
                }
                uint2 p0, p1, p2, p3;
                if constexpr (DIAG == 1) {  // no decode: the raw code bytes
                    p0 = make_uint2(st.rw[c].x, st.rw[c].y), p1 = make_uint2(st.rw[c].z, st.rw[c].w), p2 = p0, p3 = p1;
                } else {
                    p0 = a8_fold8(st.rw[c].x, t0, t1), p1 = a8_fold8(st.rw[c].y, t0, t1);
                    p2 = a8_fold8(st.rw[c].z, t0, t1), p3 = a8_fold8(st.rw[c].w, t0, t1);
                }
                uint8_t* p = &sw[buf][r * kLD + ch * kA8G];
                *reinterpret_cast<uint4*>(p) = make_uint4(p0.x, p0.y, p1.x, p1.y);
                *reinterpret_cast<uint4*>(p + 16) = make_uint4(p2.x, p2.y, p3.x, p3.y);
            }
        }
#pragma unroll
        for (int c = 0; c < CX; ++c) {
            const int idx = tid + c * NT;
            if (idx < CHX) *reinterpret_cast<uint4*>(&sx[buf][(idx / XPK) * kLD + (idx % XPK) * 16]) = st.rx[c];
        }
    };

    ik_dq_v8f acc[TN][TM];
#pragma unroll
    for (int i = 0; i < TN; ++i)
#pragma unroll
        for (int j = 0; j < TM; ++j)
#pragma unroll
            for (int e = 0; e < 8; ++e) acc[i][j][e] = 0.f;

    gload(st[0], 0);
    sstore(st[0], 0);
#pragma unroll
    for (int p = 1; p < PD; ++p)
        if (p < nk) gload(st[p], p * BK);
    if constexpr (BAR) a8_lds_barrier();
    else __syncthreads();
    const int lr = lane & 15, lh = (lane >> 4) * 16;
    const int ln = (lane >> 4) * 8;  // C fragment: lane holds token lr and features ln + e, e = 0..7
    uint4 f0a[TN], f0b[TM];  // ORD >= 2: the fragments of the current stage's first 32 k
    if constexpr (ORD >= 2) {
#pragma unroll
        for (int i = 0; i < TN; ++i) f0a[i] = *reinterpret_cast<const uint4*>(&sw[0][(wn * TN * 16 + i * 16 + lr) * kLD + lh]);
#pragma unroll
        for (int j = 0; j < TM; ++j) f0b[j] = *reinterpret_cast<const uint4*>(&sx[0][(wm * TM * 16 + j * 16 + lr) * kLD + lh]);
    }
    // one stage: `nx` holds the data of stage kt + 1 (stored into the other LDS buffer after the WMMAs), `ld` is the register set of stage kt
    // (already stored) and receives the loads of stage kt + PD, issued before the WMMAs (PD 1: nx and ld are the same set)
    auto step = [&](int kt, Stage& nx, Stage& ld) {
        const int buf = kt & 1;
        if (kt + PD < nk) gload(ld, (kt + PD) * BK);
        if constexpr (ORD >= 2) {
            // ORD 2: the fragments are software pipelined across the barrier. f0 holds the fragments of this stage's first 32 k (read before the
            // barrier of the last iteration); the second half's are read now, the WMMAs of the first half run while they arrive, then the next stage is
            // decoded and stored, the barrier makes it visible and the next stage's first-half fragments are read BEFORE the second half's WMMAs, so the
            // LDS reads of every half overlap the matrix work of the other half. ORD 3 adds s_setprio around the matrix work.
            static_assert(FOLD && BK == 64 && DIAG == 0 && PD >= 2, "ORD 2: fold path, 64-k stages, PD >= 2");
            uint4 f1a[TN], f1b[TM];
#pragma unroll
            for (int i = 0; i < TN; ++i) f1a[i] = *reinterpret_cast<const uint4*>(&sw[buf][(wn * TN * 16 + i * 16 + lr) * kLD + 32 + lh]);
#pragma unroll
            for (int j = 0; j < TM; ++j) f1b[j] = *reinterpret_cast<const uint4*>(&sx[buf][(wm * TM * 16 + j * 16 + lr) * kLD + 32 + lh]);
            if constexpr (ORD == 3) __builtin_amdgcn_s_setprio(2);
#pragma unroll
            for (int i = 0; i < TN; ++i)
#pragma unroll
                for (int j = 0; j < TM; ++j) {
                    acc[i][j] = a8_wmma(f0a[i].x, f0a[i].y, f0b[j].x, f0b[j].y, acc[i][j]);
                    acc[i][j] = a8_wmma(f0a[i].z, f0a[i].w, f0b[j].z, f0b[j].w, acc[i][j]);
                }
            if constexpr (ORD == 3) __builtin_amdgcn_s_setprio(0);
            if (kt + 1 < nk) sstore(nx, buf ^ 1);
            if constexpr (BAR) a8_lds_barrier();
            else __syncthreads();
            if (kt + 1 < nk) {
#pragma unroll
                for (int i = 0; i < TN; ++i) f0a[i] = *reinterpret_cast<const uint4*>(&sw[buf ^ 1][(wn * TN * 16 + i * 16 + lr) * kLD + lh]);
#pragma unroll
                for (int j = 0; j < TM; ++j) f0b[j] = *reinterpret_cast<const uint4*>(&sx[buf ^ 1][(wm * TM * 16 + j * 16 + lr) * kLD + lh]);
            }
            if constexpr (ORD == 3) __builtin_amdgcn_s_setprio(2);
#pragma unroll
            for (int i = 0; i < TN; ++i)
#pragma unroll
                for (int j = 0; j < TM; ++j) {
                    acc[i][j] = a8_wmma(f1a[i].x, f1a[i].y, f1b[j].x, f1b[j].y, acc[i][j]);
                    acc[i][j] = a8_wmma(f1a[i].z, f1a[i].w, f1b[j].z, f1b[j].w, acc[i][j]);
                }
            if constexpr (ORD == 3) __builtin_amdgcn_s_setprio(0);
            return;
        }
        if constexpr (ORD == 1) {
            // ORD 1: both 32-k halves' fragments are read first, then the NEXT stage is decoded and stored (it was loaded one iteration earlier,
            // PD >= 2), then the WMMAs run: the stores and the decode fill the LDS read latency, the reads of the half 1 overlap the WMMAs of the half 0.
            static_assert(FOLD && BK == 64 && DIAG == 0 && PD >= 2, "ORD 1: fold path, 64-k stages, PD >= 2");
            uint4 af[2][TN], bf[2][TM];
#pragma unroll
            for (int h = 0; h < 2; ++h) {
#pragma unroll
                for (int i = 0; i < TN; ++i)
                    af[h][i] = *reinterpret_cast<const uint4*>(&sw[buf][(wn * TN * 16 + i * 16 + lr) * kLD + h * 32 + lh]);
#pragma unroll
                for (int j = 0; j < TM; ++j)
                    bf[h][j] = *reinterpret_cast<const uint4*>(&sx[buf][(wm * TM * 16 + j * 16 + lr) * kLD + h * 32 + lh]);
            }
            if (kt + 1 < nk) sstore(nx, buf ^ 1);
#pragma unroll
            for (int h = 0; h < 2; ++h)
#pragma unroll
                for (int i = 0; i < TN; ++i)
#pragma unroll
                    for (int j = 0; j < TM; ++j) {
                        acc[i][j] = a8_wmma(af[h][i].x, af[h][i].y, bf[h][j].x, bf[h][j].y, acc[i][j]);
                        acc[i][j] = a8_wmma(af[h][i].z, af[h][i].w, bf[h][j].z, bf[h][j].w, acc[i][j]);
                    }
            if constexpr (BAR) a8_lds_barrier();
            else __syncthreads();
            return;
        }
#pragma unroll
        for (int kk = 0; kk < BK; kk += 32) {
            uint4 af[TN], bf[TM];
#pragma unroll
            for (int i = 0; i < TN; ++i) {
                if constexpr (DIAG == 5) af[i] = make_uint4(kt + i, lr, kk, lh);  // no fragment reads from LDS (timing only)
                else af[i] = *reinterpret_cast<const uint4*>(&sw[buf][(wn * TN * 16 + i * 16 + lr) * kLD + kk + lh]);
            }
#pragma unroll
            for (int j = 0; j < TM; ++j) {
                if constexpr (DIAG == 5) bf[j] = make_uint4(kt ^ j, lh, kk, lr);
                else bf[j] = *reinterpret_cast<const uint4*>(&sx[buf][(wm * TM * 16 + j * 16 + lr) * kLD + kk + lh]);
            }
            if constexpr (FOLD) {
#pragma unroll
                for (int i = 0; i < TN; ++i)
#pragma unroll
                    for (int j = 0; j < TM; ++j) {
                        if constexpr (DIAG == 2) {  // no WMMA: keep the LDS reads alive
                            acc[i][j][0] += __uint_as_float(af[i].x ^ af[i].z ^ bf[j].y ^ bf[j].w);
                        } else {
                            acc[i][j] = a8_wmma(af[i].x, af[i].y, bf[j].x, bf[j].y, acc[i][j]);
                            acc[i][j] = a8_wmma(af[i].z, af[i].w, bf[j].z, bf[j].w, acc[i][j]);
                        }
                    }
            } else {
                float sc[TN][8];  // 2^(sw - 127) of the 8 features this lane holds, per fragment row block
#pragma unroll
                for (int i = 0; i < TN; ++i) {
                    const float* sp = &sS[(buf * GPK + (kk >> 5)) * BN + wn * TN * 16 + i * 16 + ln];
                    const float4 s0 = *reinterpret_cast<const float4*>(sp), s1 = *reinterpret_cast<const float4*>(sp + 4);
                    sc[i][0] = s0.x;
                    sc[i][1] = s0.y;
                    sc[i][2] = s0.z;
                    sc[i][3] = s0.w;
                    sc[i][4] = s1.x;
                    sc[i][5] = s1.y;
                    sc[i][6] = s1.z;
                    sc[i][7] = s1.w;
                }
#pragma unroll
                for (int i = 0; i < TN; ++i)
#pragma unroll
                    for (int j = 0; j < TM; ++j) {
                        ik_dq_v8f g;
#pragma unroll
                        for (int e = 0; e < 8; ++e) g[e] = 0.f;
                        g = a8_wmma(af[i].x, af[i].y, bf[j].x, bf[j].y, g);
                        g = a8_wmma(af[i].z, af[i].w, bf[j].z, bf[j].w, g);
#pragma unroll
                        for (int e = 0; e < 8; ++e) acc[i][j][e] = fmaf(g[e], sc[i][e], acc[i][j][e]);
                    }
            }
        }
        if constexpr (DIAG != 6) {
            if (kt + 1 < nk) sstore(nx, buf ^ 1);  // DIAG 6: no stores to LDS (timing only)
        }
        if constexpr (DIAG != 4) {  // DIAG 4: no barrier (timing only)
            if constexpr (BAR) a8_lds_barrier();
            else __syncthreads();
        }
    };
    for (int kt0 = 0; kt0 < nk; kt0 += PD) {
#pragma unroll
        for (int u = 0; u < PD; ++u)
            if (kt0 + u < nk) step(kt0 + u, st[(u + 1) % PD], st[u]);
    }

#pragma unroll
    for (int i = 0; i < TN; ++i) {
#pragma unroll
        for (int j = 0; j < TM; ++j) {
            const int m = m_blk + wm * TM * 16 + j * 16 + lr;
            const int64_t n = n_blk + wn * TN * 16 + i * 16 + ln;
            if (m >= M || n >= N) continue;
            const float xsm = a.xs[m];
            float v[8];
            if constexpr (FOLD) {
                // wref is padded to a multiple of 16 bytes; n is a multiple of 8; bytes past N are never stored
                const uint2 wq = *reinterpret_cast<const uint2*>(a.wref + n);
#pragma unroll
                for (int e = 0; e < 8; ++e) {
                    const uint32_t b = ((e < 4 ? wq.x : wq.y) >> (8 * (e & 3))) & 0xffu;
                    v[e] = ldexpf(acc[i][j][e] * xsm, (int)b - 133);  // folded weights carry 2^(6 - (wref - 127))
                }
            } else {
#pragma unroll
                for (int e = 0; e < 8; ++e) v[e] = acc[i][j][e] * xsm;
            }
            OutT* o = out + (int64_t)m * N + n;
            if (a.vec_out) {
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

// Line-load fold GEMM (lane GLR). The same tile, the same LDS layout, the same WMMA loop and k order as a8_gemm_kernel's fold path (so the
// output is bitwise the same), but the global loads are whole 128 B lines per row: 8 lanes read one row's 128 B in one load instruction
// (the weights: 256 k = four stages = 8 scale groups of 16 B; the activations: 128 k = two stages). The register set of a quad (weights) or
// pair (activations) is refilled once its last stage has reached LDS. Lane p of a row's 8 takes group (p + 2c) % 8 of load c (weights) /
// chunk (p + 4c) % 8 (activations), so every thread owns exactly one weight chunk and one (or two) activation chunks per stage and the
// decode runs on the same lanes as before (waves 0 .. ND/32 - 1, all lanes); the chunk of the current stage is picked with a lane-dependent
// select over the register set (no dynamic register indexing).
template <int WN, int WM, int TN, int TM, typename OutT>
__global__ void __launch_bounds__(WN * WM * 32, (WN * WM >= 16) ? 8 : 6) a8_line_kernel(A8Args a, OutT* __restrict__ out) {
    if (*a.flag != 0) return;  // the per-group path runs a8_gemm_kernel<false>
    constexpr int NT = WN * WM * IK_WARP;
    constexpr int BN = WN * TN * 16, BM = WM * TM * 16;
    constexpr int ND = BN * 2;  // weight threads: 4 chunks of 16 B each per 256 k
    constexpr int RW = ND / 8;  // weight rows one chunk load of the ND threads covers (8 lanes per row)
    constexpr int RX = NT / 8;  // activation rows one chunk load of the NT threads covers
    constexpr int CXL = BM / RX;  // activation chunk loads per thread per pair of stages
    static_assert(ND <= NT && ND % 32 == 0 && RW * 4 == BN, "weight loads: four chunks per thread");
    static_assert(CXL * RX == BM && CXL % 2 == 0, "activation loads: an even number of chunks per thread");
    __shared__ __align__(16) uint8_t sw[2][BN * kA8LD];
    __shared__ __align__(16) uint8_t sx[2][BM * kA8LD];

    const uint8_t* __restrict__ xq = a.xq;
    const uint8_t* __restrict__ w = a.w;
    const uint8_t* __restrict__ ws = a.ws;
    const int M = a.M, K = a.K;
    const int64_t N = a.N;
    const int tid = threadIdx.x, wave = tid >> 5, lane = tid & 31;
    const int wn = wave % WN, wm = wave / WN;
    const int m_blk = blockIdx.x * BM;
    const int64_t n_blk = (int64_t)blockIdx.y * BN;
    const int nk = (K + kA8BK - 1) / kA8BK, KG = K / kA8G;
    const bool wthr = tid < ND;  // wave-uniform
    const int p = tid & 7, q = p >> 1, q2 = p >> 2, row0 = tid >> 3;

    // scalars, not arrays: a select over an array element lets the compiler turn it into a dynamically indexed (scratch) array
    uint4 rw0, rw1, rw2, rw3, rx0, rx1, rx2, rx3;
    uint32_t rtp = 0;  // fold table index (0..14) of the four chunks' groups, one nibble each
    uint32_t wrp = 0;  // row reference (largest scale byte) of the four chunks' rows, one byte each
#pragma unroll
    for (int c = 0; c < 4; ++c) {
        const int64_t n = n_blk + row0 + RW * c;
        wrp |= (uint32_t)((wthr && n < N) ? (int)a.wref[n] : 127) << (8 * c);
    }
    // weights of the stages 4S .. 4S + 3: row row0 + RW * c, group g = (p + 2c) % 8 of the 256 k line; returns the chunk's fold table index
    auto wl = [&](int S, int c, uint4& dst) -> uint32_t {
        const int g = (p + 2 * c) & 7, gi = 8 * S + g;
        int64_t n = n_blk + row0 + RW * c;
        asm volatile("" : "+v"(n));  // keep the row offsets n * (K / 2) from being hoisted out of the loop into registers
        if (wthr && n < N && gi < KG) {
            dst = *reinterpret_cast<const uint4*>(w + n * (K / 2) + (int64_t)128 * S + 16 * g);
            const int t = (int)ws[n * KG + gi] - (int)((wrp >> (8 * c)) & 0xffu) + 6;
            return (uint32_t)(min(max(t, -8), 6) + 8) << (4 * c);  // always in range when the flag is 0
        }
        dst = make_uint4(0, 0, 0, 0);
        return 8u << (4 * c);  // t = 0: zero codes decode to zero whatever the shift
    };
    auto wload = [&](int S) { rtp = wl(S, 0, rw0) | wl(S, 1, rw1) | wl(S, 2, rw2) | wl(S, 3, rw3); };
    // activations of the stages 2P, 2P + 1: row row0 + RX * c, 16 B chunk (p + 4c) % 8 of the 128 k line
    auto xl = [&](int P, int c, uint4& dst) {
        const int ch = (p + 4 * c) & 7, k = 128 * P + 16 * ch;
        int m = m_blk + row0 + RX * c;
        asm volatile("" : "+v"(m));  // as in wl
        dst = (m < M && k < K) ? *reinterpret_cast<const uint4*>(xq + (int64_t)m * K + k) : make_uint4(0, 0, 0, 0);
    };
    auto xload = [&](int P) {
        xl(P, 0, rx0);
        xl(P, 1, rx1);
        if constexpr (CXL > 2) {
            xl(P, 2, rx2);
            xl(P, 3, rx3);
        }
    };
    // decode and store stage kt + 1: ws_ = its quad slot (0..3), xs_ = its pair slot (0..1)
    auto sstore = [&](int ws_, int xs_, int buf) {
        if (wthr) {
            const int cs = (ws_ - q) & 3;  // the load whose group is in this stage
            const uint4 v = make_uint4(a8_pick4(cs, rw0.x, rw1.x, rw2.x, rw3.x), a8_pick4(cs, rw0.y, rw1.y, rw2.y, rw3.y),
                                       a8_pick4(cs, rw0.z, rw1.z, rw2.z, rw3.z), a8_pick4(cs, rw0.w, rw1.w, rw2.w, rw3.w));
            const uint32_t ti = (rtp >> (4 * cs)) & 15u;
            const uint32_t t0 = kA8FoldTab[ti][0], t1 = kA8FoldTab[ti][1];
            const uint2 p0 = a8_fold8(v.x, t0, t1), p1 = a8_fold8(v.y, t0, t1);
            const uint2 p2 = a8_fold8(v.z, t0, t1), p3 = a8_fold8(v.w, t0, t1);
            uint8_t* pp = &sw[buf][(row0 + RW * cs) * kA8LD + (p & 1) * kA8G];
            *reinterpret_cast<uint4*>(pp) = make_uint4(p0.x, p0.y, p1.x, p1.y);
            *reinterpret_cast<uint4*>(pp + 16) = make_uint4(p2.x, p2.y, p3.x, p3.y);
        }
        const int odd = (xs_ ^ q2) & 1;  // of the chunk loads 2j, 2j + 1, the one in this stage
        *reinterpret_cast<uint4*>(&sx[buf][(row0 + RX * odd) * kA8LD + (p & 3) * 16]) =
            make_uint4(odd ? rx1.x : rx0.x, odd ? rx1.y : rx0.y, odd ? rx1.z : rx0.z, odd ? rx1.w : rx0.w);
        if constexpr (CXL > 2)
            *reinterpret_cast<uint4*>(&sx[buf][(row0 + RX * (2 + odd)) * kA8LD + (p & 3) * 16]) =
                make_uint4(odd ? rx3.x : rx2.x, odd ? rx3.y : rx2.y, odd ? rx3.z : rx2.z, odd ? rx3.w : rx2.w);
    };

    ik_dq_v8f acc[TN][TM];
#pragma unroll
    for (int i = 0; i < TN; ++i)
#pragma unroll
        for (int j = 0; j < TM; ++j)
#pragma unroll
            for (int e = 0; e < 8; ++e) acc[i][j][e] = 0.f;

    wload(0);
    xload(0);
    sstore(0, 0, 0);
    __syncthreads();
    const int lr = lane & 15, lh = (lane >> 4) * 16;
    const int ln = (lane >> 4) * 8;  // C fragment: lane holds token lr and features ln + e, e = 0..7
    for (int kt = 0; kt < nk; ++kt) {
        const int buf = kt & 1;
        if (kt + 1 < nk) {  // the register sets are free once their last stage is in LDS: refill them for the next quad / pair
            if ((kt & 3) == 3) wload((kt + 1) >> 2);
            if ((kt & 1) == 1) xload((kt + 1) >> 1);
        }
#pragma unroll
        for (int kk = 0; kk < kA8BK; kk += 32) {
            uint4 af[TN], bf[TM];
#pragma unroll
            for (int i = 0; i < TN; ++i)
                af[i] = *reinterpret_cast<const uint4*>(&sw[buf][(wn * TN * 16 + i * 16 + lr) * kA8LD + kk + lh]);
#pragma unroll
            for (int j = 0; j < TM; ++j)
                bf[j] = *reinterpret_cast<const uint4*>(&sx[buf][(wm * TM * 16 + j * 16 + lr) * kA8LD + kk + lh]);
#pragma unroll
            for (int i = 0; i < TN; ++i)
#pragma unroll
                for (int j = 0; j < TM; ++j) {
                    acc[i][j] = a8_wmma(af[i].x, af[i].y, bf[j].x, bf[j].y, acc[i][j]);
                    acc[i][j] = a8_wmma(af[i].z, af[i].w, bf[j].z, bf[j].w, acc[i][j]);
                }
        }
        if (kt + 1 < nk) sstore((kt + 1) & 3, (kt + 1) & 1, buf ^ 1);
        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < TN; ++i) {
#pragma unroll
        for (int j = 0; j < TM; ++j) {
            const int m = m_blk + wm * TM * 16 + j * 16 + lr;
            const int64_t n = n_blk + wn * TN * 16 + i * 16 + ln;
            if (m >= M || n >= N) continue;
            const float xsm = a.xs[m];
            float v[8];
            // wref is padded to a multiple of 16 bytes; n is a multiple of 8; bytes past N are never stored
            const uint2 wq = *reinterpret_cast<const uint2*>(a.wref + n);
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                const uint32_t b = ((e < 4 ? wq.x : wq.y) >> (8 * (e & 3))) & 0xffu;
                v[e] = ldexpf(acc[i][j][e] * xsm, (int)b - 133);  // folded weights carry 2^(6 - (wref - 127))
            }
            OutT* o = out + (int64_t)m * N + n;
            if (a.vec_out) {
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

template <int WN, int WM, int TN, int TM, typename OutT>
void a8_enqueue_line(const A8Args& a, OutT* out, hipStream_t s) {
    dim3 grid((unsigned)(((int64_t)a.M + WM * TM * 16 - 1) / (WM * TM * 16)), (unsigned)((a.N + WN * TN * 16 - 1) / (WN * TN * 16)));
    a8_line_kernel<WN, WM, TN, TM, OutT><<<grid, WN * WM * 32, 0, s>>>(a, out);
}


template <bool FOLD, int WN, int WM, int TN, int TM, int BK = 64, int DIAG = 0, int PD = 1, int ORD = 0, int BAR = 0, int WB = 0, typename OutT>
void a8_enqueue(const A8Args& a, OutT* out, hipStream_t s) {
    dim3 grid((unsigned)(((int64_t)a.M + WM * TM * 16 - 1) / (WM * TM * 16)), (unsigned)((a.N + WN * TN * 16 - 1) / (WN * TN * 16)));
    a8_gemm_kernel<FOLD, WN, WM, TN, TM, BK, DIAG, PD, ORD, BAR, WB, OutT><<<grid, WN * WM * 32, 0, s>>>(a, out);
}

// INTARSIA_HIP_MXFP4_A8_TILE=WN,WM,TN,TM[,BK[,DIAG[,PD]]] and INTARSIA_HIP_MXFP4_A8_PERGROUP=1, read once
struct A8Cfg {
    int tile[11] = {0, 0, 0, 0, 64, 0, 1, 0, 0, 0, 0};
    int pergroup = 0;
    int line = 0;  // INTARSIA_HIP_MXFP4_A8_LINES=1: the fold path loads whole 128 B lines (a8_line_kernel)
};
const A8Cfg& a8_cfg() {
    static const A8Cfg cfg = [] {
        A8Cfg c;
        const char* e = getenv("INTARSIA_HIP_MXFP4_A8_TILE");
        if (e && sscanf(e, "%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d", &c.tile[0], &c.tile[1], &c.tile[2], &c.tile[3], &c.tile[4], &c.tile[5], &c.tile[6], &c.tile[7], &c.tile[8], &c.tile[9], &c.tile[10]) < 4) c.tile[0] = 0;
        const char* g = getenv("INTARSIA_HIP_MXFP4_A8_PERGROUP");
        c.pergroup = (g && g[0] == '1') ? 1 : 0;
        const char* l = getenv("INTARSIA_HIP_MXFP4_A8_LINES");
        c.line = (l && l[0] == '1') ? 1 : 0;
        return c;
    }();
    return cfg;
}

// Both paths of a tile are enqueued; the one the flag does not select exits at once. The per-group path's LDS (4 * BN floats
// more) would put the 256 x 128 tile exactly at 64 KB, so that path takes 128 x 256 there.
template <typename OutT>
void a8_run(const A8Args& a, OutT* out, hipStream_t s) {
#define IK_A8_TILE(WN, WM, TN, TM)                      \
    {                                                   \
        a8_enqueue<true, WN, WM, TN, TM>(a, out, s);    \
        a8_enqueue<false, WN, WM, TN, TM>(a, out, s);   \
        return;                                         \
    }
#define IK_A8_TILE_V(WN, WM, TN, TM, BK, DIAG, PD)                  \
    {                                                                \
        a8_enqueue<true, WN, WM, TN, TM, BK, DIAG, PD>(a, out, s);   \
        a8_enqueue<false, WN, WM, TN, TM, BK, DIAG, PD>(a, out, s);  \
        return;                                                      \
    }
    const int* t = a8_cfg().tile;
    const bool line = a8_cfg().line || t[7] == 1;
    // fold launch: the line-load kernel or the stage-load one; the per-group launch is always the stage-load kernel
#define IK_A8_FOLD(WN, WM, TN, TM)                                                           \
    do {                                                                                     \
        if (line) a8_enqueue_line<WN, WM, TN, TM>(a, out, s);                                \
        else a8_enqueue<true, WN, WM, TN, TM>(a, out, s);                                    \
    } while (0)
#define IK_A8_LTILE(WN, WM, TN, TM, NWN, NWM, NTN, NTM) \
    {                                                   \
        IK_A8_FOLD(WN, WM, TN, TM);                     \
        a8_enqueue<false, NWN, NWM, NTN, NTM>(a, out, s); \
        return;                                         \
    }
    if (t[0] > 0) {
        const int key = t[0] * 1000 + t[1] * 100 + t[2] * 10 + t[3];
        if (t[8] == 0 && t[9] == 0 && t[10] == 0 && t[4] == 64 && t[5] == 0 && t[6] == 1) {
            if (key == 4442) IK_A8_LTILE(4, 4, 4, 2, 4, 4, 2, 4)
            if (key == 4424) IK_A8_LTILE(4, 4, 2, 4, 4, 4, 2, 4)
            if (key == 4224) IK_A8_LTILE(4, 2, 2, 4, 4, 2, 2, 4)
            if (key == 2244) IK_A8_TILE(2, 2, 4, 4)
        }
        if (t[8] == 0 && t[9] == 0 && t[10] == 0 && t[4] == 64 && t[5] == 0 && t[6] == 2) {
            if (key == 4442) {
                a8_enqueue<true, 4, 4, 4, 2, 64, 0, 2>(a, out, s);
                a8_enqueue<false, 4, 4, 2, 4, 64, 0, 2>(a, out, s);
                return;
            }
            if (key == 4424) IK_A8_TILE_V(4, 4, 2, 4, 64, 0, 2)
            if (key == 4224) IK_A8_TILE_V(4, 2, 2, 4, 64, 0, 2)
        }
        if (t[8] == 0 && t[9] == 0 && t[10] == 0 && t[4] == 64 && t[5] == 0 && t[6] == 3) {
            if (key == 4442) {
                a8_enqueue<true, 4, 4, 4, 2, 64, 0, 3>(a, out, s);
                a8_enqueue<false, 4, 4, 2, 4, 64, 0, 3>(a, out, s);
                return;
            }
            if (key == 4424) IK_A8_TILE_V(4, 4, 2, 4, 64, 0, 3)
        }
        // 8-wave tiles with 4 x 4 fragments per wave (experiments): fold launch of the tile at prefetch distance PD, per-group launch 4 x 4 x 2 x 4
#define IK_A8_FT(WN, WM, TN, TM, PD)                                   \
    {                                                                    \
        a8_enqueue<true, WN, WM, TN, TM, 64, 0, PD>(a, out, s);          \
        a8_enqueue<false, 4, 4, 2, 4>(a, out, s);                        \
        return;                                                          \
    }
        if (t[8] == 0 && t[9] == 0 && t[10] == 0 && t[4] == 64 && t[5] == 0 && t[6] <= 2) {
            if (key == 2444 && t[6] == 1) IK_A8_FT(2, 4, 4, 4, 1)
            if (key == 2444 && t[6] == 2) IK_A8_FT(2, 4, 4, 4, 2)
            if (key == 4244 && t[6] == 1) IK_A8_FT(4, 2, 4, 4, 1)
            if (key == 4244 && t[6] == 2) IK_A8_FT(4, 2, 4, 4, 2)
        }
#undef IK_A8_FT
        // ORD 1 (early store of the next stage), PD 2 or 3: tiles 4442 / 4424 / 4224 and the 8-wave 4 x 4 ones
#define IK_A8_O(WN, WM, TN, TM, PD)                                       \
    {                                                                    \
        a8_enqueue<true, WN, WM, TN, TM, 64, 0, PD, 1>(a, out, s);       \
        a8_enqueue<false, 4, 4, 2, 4>(a, out, s);                        \
        return;                                                          \
    }
        // schedule variants of the stage-load fold kernel (tile 4442 / 4424 / 4224; PD = t[6], ORD = t[8], BAR = t[9]): fold launch of the variant,
        // per-group launch 4 x 4 x 2 x 4
        if ((t[8] >= 1 || t[9] >= 1 || t[10] >= 1) && t[4] == 64 && t[5] == 0) {
#define IK_A8_V(WN, WM, TN, TM, PD, OR, BA, WBV)                            \
    if (t[6] == PD && t[8] == OR && t[9] == BA && t[10] == WBV) {           \
        a8_enqueue<true, WN, WM, TN, TM, 64, 0, PD, OR, BA, WBV>(a, out, s); \
        a8_enqueue<false, 4, 4, 2, 4>(a, out, s);                           \
        return;                                                             \
    }
#define IK_A8_VT(WN, WM, TN, TM)                                            \
    IK_A8_V(WN, WM, TN, TM, 1, 0, 1, 0)                                     \
    IK_A8_V(WN, WM, TN, TM, 2, 0, 1, 0)                                     \
    IK_A8_V(WN, WM, TN, TM, 2, 2, 0, 0)                                     \
    IK_A8_V(WN, WM, TN, TM, 2, 2, 1, 0)                                     \
    IK_A8_V(WN, WM, TN, TM, 1, 0, 0, 1)                                     \
    IK_A8_V(WN, WM, TN, TM, 2, 0, 0, 1)                                     \
    IK_A8_V(WN, WM, TN, TM, 2, 0, 1, 1)                                     \
    IK_A8_V(WN, WM, TN, TM, 2, 2, 0, 1)
            if (key == 4424) { IK_A8_VT(4, 4, 2, 4) }
            if (key == 4442) { IK_A8_VT(4, 4, 4, 2) }
            if (key == 4224) { IK_A8_VT(4, 2, 2, 4) }
#undef IK_A8_VT
#undef IK_A8_V
        }
#undef IK_A8_O
        if (t[8] == 0 && t[9] == 0 && t[10] == 0 && t[5] >= 1 && t[4] == 64 && t[6] == 1 && key == 4424) {
            if (t[5] == 1) IK_A8_TILE_V(4, 4, 2, 4, 64, 1, 1)
            if (t[5] == 2) IK_A8_TILE_V(4, 4, 2, 4, 64, 2, 1)
            if (t[5] == 3) IK_A8_TILE_V(4, 4, 2, 4, 64, 3, 1)
            if (t[5] == 4) IK_A8_TILE_V(4, 4, 2, 4, 64, 4, 1)
            if (t[5] == 5) IK_A8_TILE_V(4, 4, 2, 4, 64, 5, 1)
            if (t[5] == 6) IK_A8_TILE_V(4, 4, 2, 4, 64, 6, 1)
        }
    }
    // ik_f8_gemm.h's f8_gemm_run table (weights x tokens), as the fold launch
    if (a.M <= 64) IK_A8_TILE(4, 1, 2, 2)
    if (a.M >= 2048 && a.N >= 8192) IK_A8_LTILE(4, 4, 4, 2, 4, 4, 2, 4)
    if (a.M >= 2048 && ((a.N + 127) / 128) * (((int64_t)a.M + 255) / 256) >= 64) IK_A8_LTILE(4, 4, 2, 4, 4, 4, 2, 4)
    IK_A8_LTILE(4, 2, 2, 4, 4, 2, 2, 4)
#undef IK_A8_LTILE
#undef IK_A8_FOLD
#undef IK_A8_TILE
#undef IK_A8_TILE_V
}

size_t a8_round16(size_t v) { return (v + 15) & ~(size_t)15; }

// scratch: mx_a8_scratch_bytes(M, K) (xq, xs; ik_mx_a8.h), padded to 16, then wref [round16(N)] and the fold flag [16]
size_t a8_scratch_total(int64_t M, int64_t K, int64_t N) {
    return a8_round16(mx_a8_scratch_bytes(M, K)) + a8_round16((size_t)N) + 16;
}

int launch_a8(const ik_tensor* x, const ik_tensor* w, const ik_tensor* sc, ik_tensor* out, const ik_scratch* scratch,
              void* stream) {
    IK_TRY(ik_check_a(x, IK_DTYPE_BF16, 2, 16));
    IK_TRY(ik_check_a(w, IK_DTYPE_U8, 2, 16));
    IK_TRY(ik_check_a(sc, IK_DTYPE_U8, 2, 1));
    if (!out || (out->dtype != IK_DTYPE_BF16 && out->dtype != IK_DTYPE_F32)) return IK_ERR_DTYPE;
    IK_TRY(ik_check(out, out->dtype, 2));
    const int64_t M = x->shape[0], K = x->shape[1], N = w->shape[0];
    if (K % kA8G != 0) return IK_ERR_SHAPE;
    if (w->shape[1] != K / 2 || sc->shape[0] != N || sc->shape[1] != K / kA8G || out->shape[0] != M || out->shape[1] != N)
        return IK_ERR_SHAPE;
    if (M > 0x7fffffffLL || K > 0x7fffffffLL) return IK_ERR_SHAPE;
    if (M == 0 || N == 0) return IK_OK;
    hipStream_t s = IK_STREAM(stream);
    if (K == 0)
        return hipMemsetAsync(out->data, 0, ik_numel(out) * (out->dtype == IK_DTYPE_F32 ? 4 : 2), s) == hipSuccess ? IK_OK
                                                                                                                      : IK_ERR_LAUNCH;
    if (!scratch || !scratch->data || scratch->bytes < (int64_t)a8_scratch_total(M, K, N)) return IK_ERR_INVALID_ARG;
    if (reinterpret_cast<uintptr_t>(scratch->data) & 15u) return IK_ERR_INVALID_ARG;
    uint8_t* base = (uint8_t*)scratch->data;
    const size_t off = a8_round16(mx_a8_scratch_bytes(M, K));
    uint8_t* wref = base + off;
    int* flag = reinterpret_cast<int*>(wref + a8_round16((size_t)N));
    const uint8_t* wp = (const uint8_t*)w->data;
    const uint8_t* sp = (const uint8_t*)sc->data;

    IK_TRY(mx_a8_quant_launch((const ik_bf16*)x->data, base, M, K, s));
    if (a8_cfg().pergroup) {
        if (hipMemsetAsync(flag, 1, sizeof(int), s) != hipSuccess) return IK_ERR_LAUNCH;  // flag = 0x01010101: never folds
    } else {
        if (hipMemsetAsync(flag, 0, sizeof(int), s) != hipSuccess) return IK_ERR_LAUNCH;
        a8_wspan_kernel<<<(unsigned)((N + 7) / 8), 256, 0, s>>>(sp, wref, flag, N, (int)(K / kA8G));
        IK_TRY(ik_launch_status());
    }
    A8Args a;
    a.xq = base;
    a.xs = reinterpret_cast<const float*>(base + mx_a8_xs_offset(M, K));
    a.w = wp;
    a.ws = sp;
    a.wref = wref;
    a.flag = flag;
    a.M = (int)M;
    a.N = N;
    a.K = (int)K;
    a.vec_out = (N % 8 == 0) && ik_aligned16(out->data);
    if (out->dtype == IK_DTYPE_F32) a8_run<float>(a, (float*)out->data, s);
    else a8_run<ik_bf16>(a, (ik_bf16*)out->data, s);
    return ik_launch_status();
}

}  // namespace

extern "C" {

// matmul_mxfp4_a8_mma (lane W4A8): see the header comment. `act` is accepted for the matmul_mxfp4 signature and not read.
// Scratch: mx_a8_scratch_bytes(M, K) rounded up to 16, + round16(N) + 16 bytes, 16-byte aligned base.
int ik_matmul_mxfp4_a8_mma(const ik_tensor* x, const ik_tensor* w, const ik_tensor* scales, const ik_tensor* act,
                           ik_tensor* out, const ik_scratch* scratch, void* stream) {
    (void)act;
    return launch_a8(x, w, scales, out, scratch, stream);
}

}  // extern "C"
