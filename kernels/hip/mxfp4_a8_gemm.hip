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
// INTARSIA_HIP_MXFP4_A8_TILE=WN,WM,TN,TM[,PD] forces one tile (sweeps; the tiles of the table are accepted, PD 1 or 2, default 1).
// INTARSIA_HIP_MXFP4_A8_PD=1 restores the previous kernel choice (global-load prefetch distance 1, the previous tile table).
// M > 64 runs the direct kernel (a8_direct_kernel, lane PF27: tokens straight from global memory in fragment order, weights alone
// through LDS; bitwise the same output); INTARSIA_HIP_MXFP4_A8_DIRECT=0 restores the staged kernel below, _DTILE forces a direct tile.
// Block: WN x WM waves, each wave TN x TM fragments of 16 x 16 (weights x tokens); 64 k per stage, double-buffered through
// registers (prefetch distance 2 by default: two register sets, the loads of stage k + 2 are in flight during stage k), LDS rows padded (ik_f8_gemm.h's layout). Fragments / k layout as ik_f8_gemm.h (both operands permute k alike).
// K % 32 == 0 (a K % 64 == 32 tail stage is zero padded). Deterministic, fixed reduction order, independent of M.
#include "ik_dq_gemm.h"
#include "ik_mx_a8.h"
#include "ar_push.h"

#include <stdio.h>

namespace {

constexpr int kA8G = 32;            // elements per e8m0 scale
constexpr int kA8BK = 64;           // k per stage: two groups
constexpr int kA8LD = kA8BK + 16;   // LDS row stride in bytes (ik_f8_gemm.h's kF8LD)

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

// Pre-pass over the weight scales, one warp per row: wref[n] = largest scale byte of row n; flag |= 1 when some row cannot be
// folded exactly (span > 14 octaves, or a 255 = NaN scale). flag is zeroed before (hipMemsetAsync).
__global__ void __launch_bounds__(256) a8_wspan_kernel(const uint8_t* __restrict__ ws, uint8_t* __restrict__ wref,
                                                       int* __restrict__ flag, int64_t N, int KG, const uint8_t* __restrict__ ws1,
                                                       uint8_t* __restrict__ wref1, int* __restrict__ flag1, int64_t N1) {
    int64_t n = (int64_t)blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    // lane pf27b: rows [N, N + N1) are a pair's second weight (its own wref and flag); N1 = 0 for one GEMM
    if (n >= N) {  // whole warp
        n -= N;
        if (n >= N1) return;
        ws = ws1;
        wref = wref1;
        flag = flag1;
    }
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
// PD: global-load prefetch distance in stages. 1: the next stage's data is in flight while this stage's WMMAs run. 2: the stage after that as well
// (two register sets): the loads have two stages of latency to hide instead of one, at +14 VGPRs (still 8 waves/SIMD on the 4 x 4 tiles). The k order
// of every accumulator is the same for any PD (stages in ascending k, a stage's WMMAs in k order), so the output is bitwise the same.
template <bool FOLD, int WN, int WM, int TN, int TM, int PD, typename OutT>
__global__ void __launch_bounds__(WN * WM * 32) a8_gemm_kernel(A8Args a, OutT* __restrict__ out) {
    if ((*a.flag == 0) != FOLD) return;
    constexpr int NT = WN * WM * IK_WARP;
    constexpr int BN = WN * TN * 16, BM = WM * TM * 16;
    constexpr int CHW = BN * 2, CHX = BM * 4;  // 16-byte chunks of one stage: weights one group (32 codes), x 16 bytes
    constexpr int CW = (CHW + NT - 1) / NT, CX = (CHX + NT - 1) / NT;
    constexpr int SS = FOLD ? 4 : 4 * BN;  // per-group path: f32 scales [buf][group][n]
    __shared__ __align__(16) uint8_t sw[2][BN * kA8LD];
    __shared__ __align__(16) uint8_t sx[2][BM * kA8LD];
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
    const int nk = (K + kA8BK - 1) / kA8BK, KG = K / kA8G;

    struct Stage {
        uint4 rw[CW], rx[CX];
        int rt[CW];  // FOLD: fold shift of the chunk's group; else: its e8m0 byte
    };
    Stage st[PD];  // register sets of the stages in flight: stage k's data is in st[k % PD]
    int wr[CW];  // FOLD: row reference (largest scale byte)
#pragma unroll
    for (int c = 0; c < CW; ++c) {
        wr[c] = 127;
        if constexpr (FOLD) {
            const int64_t n = n_blk + ((tid + c * NT) >> 1);
            if (n < N) wr[c] = (int)a.wref[n];
        }
    }
    auto gload = [&](Stage& st, int k0) {
#pragma unroll
        for (int c = 0; c < CW; ++c) {
            const int idx = tid + c * NT, r = idx >> 1, ch = idx & 1;
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
            const int idx = tid + c * NT, r = idx >> 2, ch = idx & 3;
            const int m = m_blk + r;
            const int k = k0 + ch * 16;
            st.rx[c] = (idx < CHX && m < M && k < K) ? *reinterpret_cast<const uint4*>(xq + (int64_t)m * K + k) : make_uint4(0, 0, 0, 0);
        }
    };
    auto sstore = [&](const Stage& st, int buf) {
#pragma unroll
        for (int c = 0; c < CW; ++c) {
            const int idx = tid + c * NT;
            if (idx < CHW) {
                const int r = idx >> 1, ch = idx & 1;
                uint32_t t0 = kA8Tab0, t1 = kA8Tab1;
                if constexpr (FOLD) {
                    const int ti = min(max(st.rt[c], -8), 6) + 8;  // always in range when the flag is 0
                    t0 = kA8FoldTab[ti][0];
                    t1 = kA8FoldTab[ti][1];
                } else {
                    sS[(buf * 2 + ch) * BN + r] = a8_e8m0((uint32_t)st.rt[c]);
                }
                const uint2 p0 = a8_fold8(st.rw[c].x, t0, t1), p1 = a8_fold8(st.rw[c].y, t0, t1);
                const uint2 p2 = a8_fold8(st.rw[c].z, t0, t1), p3 = a8_fold8(st.rw[c].w, t0, t1);
                uint8_t* p = &sw[buf][r * kA8LD + ch * kA8G];
                *reinterpret_cast<uint4*>(p) = make_uint4(p0.x, p0.y, p1.x, p1.y);
                *reinterpret_cast<uint4*>(p + 16) = make_uint4(p2.x, p2.y, p3.x, p3.y);
            }
        }
#pragma unroll
        for (int c = 0; c < CX; ++c) {
            const int idx = tid + c * NT;
            if (idx < CHX) *reinterpret_cast<uint4*>(&sx[buf][(idx >> 2) * kA8LD + (idx & 3) * 16]) = st.rx[c];
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
        if (p < nk) gload(st[p], p * kA8BK);
    __syncthreads();
    const int lr = lane & 15, lh = (lane >> 4) * 16;
    const int ln = (lane >> 4) * 8;  // C fragment: lane holds token lr and features ln + e, e = 0..7
    // one stage: `nx` holds the data of stage kt + 1 (stored into the other LDS buffer after the WMMAs), `ld` is the register set of stage kt
    // (already stored) and receives the loads of stage kt + PD, issued before the WMMAs (PD 1: nx and ld are the same set)
    auto step = [&](int kt, Stage& nx, Stage& ld) {
        const int buf = kt & 1;
        if (kt + PD < nk) gload(ld, (kt + PD) * kA8BK);
#pragma unroll
        for (int kk = 0; kk < kA8BK; kk += 32) {
            uint4 af[TN], bf[TM];
#pragma unroll
            for (int i = 0; i < TN; ++i)
                af[i] = *reinterpret_cast<const uint4*>(&sw[buf][(wn * TN * 16 + i * 16 + lr) * kA8LD + kk + lh]);
#pragma unroll
            for (int j = 0; j < TM; ++j)
                bf[j] = *reinterpret_cast<const uint4*>(&sx[buf][(wm * TM * 16 + j * 16 + lr) * kA8LD + kk + lh]);
            if constexpr (FOLD) {
#pragma unroll
                for (int i = 0; i < TN; ++i)
#pragma unroll
                    for (int j = 0; j < TM; ++j) {
                        acc[i][j] = a8_wmma(af[i].x, af[i].y, bf[j].x, bf[j].y, acc[i][j]);
                        acc[i][j] = a8_wmma(af[i].z, af[i].w, bf[j].z, bf[j].w, acc[i][j]);
                    }
            } else {
                float sc[TN][8];  // 2^(sw - 127) of the 8 features this lane holds, per fragment row block
#pragma unroll
                for (int i = 0; i < TN; ++i) {
                    const float* sp = &sS[(buf * 2 + (kk >> 5)) * BN + wn * TN * 16 + i * 16 + ln];
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
        if (kt + 1 < nk) sstore(nx, buf ^ 1);
        __syncthreads();
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

template <bool FOLD, int WN, int WM, int TN, int TM, int PD = 1, typename OutT>
void a8_enqueue(const A8Args& a, OutT* out, hipStream_t s) {
    dim3 grid((unsigned)(((int64_t)a.M + WM * TM * 16 - 1) / (WM * TM * 16)), (unsigned)((a.N + WN * TN * 16 - 1) / (WN * TN * 16)));
    a8_gemm_kernel<FOLD, WN, WM, TN, TM, PD, OutT><<<grid, WN * WM * 32, 0, s>>>(a, out);
}

// LDS-only block barrier: orders the block's LDS accesses and nothing else. __syncthreads() is a workgroup-scope release / acquire
// of every address space, so on gfx12 it waits for every outstanding global load too (s_wait_loadcnt 0), which would drain the
// token-fragment loads the direct kernel keeps in flight across the stage barrier. (The same construction as Radiance's
// mxt_lds_barrier, libr4d/r4d_gemm_mxfp4a8_tiled.hip, Apache-2.0; written here from the AMDGPU memory model, no code copied.)
IK_DEVI void a8_lds_barrier() {
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");
}

// Lane PF27, the direct kernel: the tokens skip LDS. The pre-pass (mx_a8_quant_tiled_kernel, ik_mx_a8.h) writes xq in fragment order,
// so each wave loads its token fragments straight from global memory into registers (one coalesced 512-byte uint4 load per fragment
// and 32-k group), one stage ahead; only the weights go through LDS (decode + store as a8_gemm_kernel, prefetch distance 2). The
// stage's LDS traffic drops from the weight AND token fragments (+ both stores) to the weight fragments and the weight store, which
// was the bound of a8_gemm_kernel (lane GLR's DIAG 5: no LDS fragment reads, -43 %). Every WMMA gets the same operand bytes in the
// same lanes, the accumulators the same WMMAs in the same k order, the epilogue the same math: the output is bitwise a8_gemm_kernel's.
// The stage barrier is a8_lds_barrier, so the token loads stay in flight across it.
// PUSH (lane arx, bf16 out with vec_out only): every output word also goes to `peer` (the peer rank's all-reduce slot) in the fragment order of
// ar_push.h; a8_direct_kernel fences each thread's pushes at system scope before it ends, so they have landed when the next launch on this
// stream raises the all-reduce flag. The stores to `out` and the arithmetic are the !PUSH kernel's.
template <bool FOLD, int WN, int WM, int TN, int TM, typename OutT, bool PUSH = false>
__device__ __forceinline__ void a8_direct_tile(const A8Args& a, OutT* __restrict__ out, uint4* __restrict__ peer, int bx, int by, int gx,
                                               uint4 (&hold)[TN * TM], int pbx = -1, int pby = 0, bool push_now = true) {
    // one BM x BN output tile (bx, by); gx tiles along M. PUSH: (pbx, pby) is this block's previous tile (pbx < 0: none), whose output words
    // this thread pushes spread over this tile's k stages from `hold` (registers: the previous epilogue's packed words), so the link takes
    // them while the WMMAs run instead of the wave stalling on a full store queue at the epilogue; push_now: push this tile's words in the
    // epilogue (the block's last), else keep them in `hold`. (Reading them back from `out` instead gave stale words for some tiles: lane arx
    // m13, the partial tiles pushed one tile late.)
    constexpr int NT = WN * WM * IK_WARP;
    constexpr int BN = WN * TN * 16, BM = WM * TM * 16;
    constexpr int CHW = BN * 2;  // 16-byte weight chunks of one stage (one group, 32 codes, each)
    constexpr int CW = (CHW + NT - 1) / NT;
    constexpr int SS = FOLD ? 4 : 4 * BN;  // per-group path: f32 scales [buf][group][n]
    __shared__ __align__(16) uint8_t sw[2][BN * kA8LD];
    __shared__ __align__(16) float sS[SS];

    const uint8_t* __restrict__ xt = a.xq;  // fragment-ordered tokens
    const uint8_t* __restrict__ w = a.w;
    const uint8_t* __restrict__ ws = a.ws;
    const int M = a.M, K = a.K;
    const int64_t N = a.N;
    const int tid = threadIdx.x, lane = tid & 31;
    const int wave = __builtin_amdgcn_readfirstlane(tid >> 5);  // wave-uniform: the token addresses below stay scalar
    const int wn = wave % WN, wm = wave / WN;
    const int m_blk = bx * BM;
    const int64_t n_blk = (int64_t)by * BN;
    const int nk = (K + kA8BK - 1) / kA8BK, KG = K / kA8G;
    const int MT = (M + 15) >> 4;  // 16-token tiles the pre-pass wrote
    const int KGP = nk * 2;        // groups per tile in the tiled xq: K padded to the stage (the pre-pass zeroes a K % 64 == 32 tail group)

    struct Stage {
        uint4 rw[CW];
        int rt[CW];  // FOLD: fold shift of the chunk's group; else: its e8m0 byte
    };
    Stage st[2];  // weights: stage k's registers in st[k % 2] (prefetch distance 2, as a8_gemm_kernel's default)
    uint4 xr[2][2][TM];  // tokens: stage k's fragments in xr[k % 2][group][j] (prefetch distance 1)
    int wr[CW];
#pragma unroll
    for (int c = 0; c < CW; ++c) {
        wr[c] = 127;
        if constexpr (FOLD) {
            const int64_t n = n_blk + ((tid + c * NT) >> 1);
            if (n < N) wr[c] = (int)a.wref[n];
        }
    }
    // the wave's token tiles; a tile past the last one is clamped onto it (its columns are never stored, and a WMMA column only
    // reaches its own outputs)
    // wave-uniform byte offset of tile j's first group (SGPRs); each lane adds lane * 16, so every token load is a scalar base + one
    // 32-bit lane offset (no per-load 64-bit address VALU)
    int64_t xb[TM];
#pragma unroll
    for (int j = 0; j < TM; ++j) xb[j] = (int64_t)min((m_blk >> 4) + wm * TM + j, MT - 1) * KGP * 512;
    const int loff = lane * 16;
    auto gload = [&](Stage& st, int k0) {
#pragma unroll
        for (int c = 0; c < CW; ++c) {
            const int idx = tid + c * NT, r = idx >> 1, ch = idx & 1;
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
    };
    auto xload = [&](uint4 (&xs)[2][TM], int k0) {
#pragma unroll
        for (int g = 0; g < 2; ++g) {
            const int kg = (k0 >> 5) + g;  // wave-uniform; the K % 64 == 32 tail stage's second group is the pre-pass's zero pad group
#pragma unroll
            for (int j = 0; j < TM; ++j) xs[g][j] = *reinterpret_cast<const uint4*>(xt + (xb[j] + (int64_t)kg * 512) + loff);
        }
    };
    auto sstore = [&](const Stage& st, int buf) {
#pragma unroll
        for (int c = 0; c < CW; ++c) {
            const int idx = tid + c * NT;
            if (idx < CHW) {
                const int r = idx >> 1, ch = idx & 1;
                uint32_t t0 = kA8Tab0, t1 = kA8Tab1;
                if constexpr (FOLD) {
                    const int ti = min(max(st.rt[c], -8), 6) + 8;
                    t0 = kA8FoldTab[ti][0];
                    t1 = kA8FoldTab[ti][1];
                } else {
                    sS[(buf * 2 + ch) * BN + r] = a8_e8m0((uint32_t)st.rt[c]);
                }
                const uint2 p0 = a8_fold8(st.rw[c].x, t0, t1), p1 = a8_fold8(st.rw[c].y, t0, t1);
                const uint2 p2 = a8_fold8(st.rw[c].z, t0, t1), p3 = a8_fold8(st.rw[c].w, t0, t1);
                uint8_t* p = &sw[buf][r * kA8LD + ch * kA8G];
                *reinterpret_cast<uint4*>(p) = make_uint4(p0.x, p0.y, p1.x, p1.y);
                *reinterpret_cast<uint4*>(p + 16) = make_uint4(p2.x, p2.y, p3.x, p3.y);
            }
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
    xload(xr[0], 0);
    sstore(st[0], 0);
    if (1 < nk) gload(st[1], kA8BK);
    a8_lds_barrier();
    const int lr = lane & 15, lh = (lane >> 4) * 16;
    const int ln = (lane >> 4) * 8;  // C fragment: lane holds token lr and features ln + e, e = 0..7
    int pq = 0;  // PUSH: the previous tile's words this thread pushed so far (q = i * TM + j)
    // word q of the previous tile (q = i * TM + j); the unrolled select keeps `hold` in registers (a runtime index would put it in scratch)
    auto push_prev = [&](int q) {
#pragma unroll
        for (int qq = 0; qq < TN * TM; ++qq) {
            if (qq != q) continue;
            const int i = qq / TM, j = qq % TM;
            const int m = pbx * BM + wm * TM * 16 + j * 16 + lr;
            const int64_t n = (int64_t)pby * BN + wn * TN * 16 + i * 16 + ln;
            if (m < M && n < N) {
                const int64_t blk = (int64_t)pby * gx + pbx;
                peer[(((blk * (WN * WM) + wave) * TN + i) * TM + j) * 32 + lane] = hold[qq];
            }
        }
    };
    const int pstride = nk / (TN * TM) > 0 ? nk / (TN * TM) : 1;
    // stage kt (u = kt % 2): st[u] (already stored) takes the weight loads of stage kt + 2, xr[u ^ 1] the token loads of stage kt + 1;
    // the WMMAs read the weights from LDS buffer u and the tokens from xr[u]; then st[u ^ 1] (stage kt + 1) is stored into buffer u ^ 1
    auto step = [&](int kt, uint4 (&xc)[2][TM], uint4 (&xn)[2][TM], Stage& nx, Stage& ld) {
        const int buf = kt & 1;
        if (kt + 2 < nk) gload(ld, (kt + 2) * kA8BK);
        if (kt + 1 < nk) xload(xn, (kt + 1) * kA8BK);
#pragma unroll
        for (int g = 0; g < 2; ++g) {
            const int kk = g * 32;
            uint4 af[TN];
#pragma unroll
            for (int i = 0; i < TN; ++i)
                af[i] = *reinterpret_cast<const uint4*>(&sw[buf][(wn * TN * 16 + i * 16 + lr) * kA8LD + kk + lh]);
            if constexpr (FOLD) {
#pragma unroll
                for (int i = 0; i < TN; ++i)
#pragma unroll
                    for (int j = 0; j < TM; ++j) {
                        acc[i][j] = a8_wmma(af[i].x, af[i].y, xc[g][j].x, xc[g][j].y, acc[i][j]);
                        acc[i][j] = a8_wmma(af[i].z, af[i].w, xc[g][j].z, xc[g][j].w, acc[i][j]);
                    }
            } else {
                float sc[TN][8];
#pragma unroll
                for (int i = 0; i < TN; ++i) {
                    const float* sp = &sS[(buf * 2 + g) * BN + wn * TN * 16 + i * 16 + ln];
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
                        ik_dq_v8f gq;
#pragma unroll
                        for (int e = 0; e < 8; ++e) gq[e] = 0.f;
                        gq = a8_wmma(af[i].x, af[i].y, xc[g][j].x, xc[g][j].y, gq);
                        gq = a8_wmma(af[i].z, af[i].w, xc[g][j].z, xc[g][j].w, gq);
#pragma unroll
                        for (int e = 0; e < 8; ++e) acc[i][j][e] = fmaf(gq[e], sc[i][e], acc[i][j][e]);
                    }
            }
        }
        if constexpr (PUSH) {
            if (pbx >= 0 && pq < TN * TM && kt % pstride == pstride - 1) push_prev(pq++);
        }
        if (kt + 1 < nk) sstore(nx, buf ^ 1);
        a8_lds_barrier();
    };
    for (int kt0 = 0; kt0 < nk; kt0 += 2) {
        step(kt0, xr[0], xr[1], st[1], st[0]);
        if (kt0 + 1 < nk) step(kt0 + 1, xr[1], xr[0], st[0], st[1]);
    }
    if constexpr (PUSH) {
        if (pbx >= 0)
            while (pq < TN * TM) push_prev(pq++);
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
                const uint2 wq = *reinterpret_cast<const uint2*>(a.wref + n);
#pragma unroll
                for (int e = 0; e < 8; ++e) {
                    const uint32_t b = ((e < 4 ? wq.x : wq.y) >> (8 * (e & 3))) & 0xffu;
                    v[e] = ldexpf(acc[i][j][e] * xsm, (int)b - 133);
                }
            } else {
#pragma unroll
                for (int e = 0; e < 8; ++e) v[e] = acc[i][j][e] * xsm;
            }
            OutT* o = out + (int64_t)m * N + n;
            if (a.vec_out) {
                if (sizeof(OutT) == 2) {
                    const uint4 pk = ik_pack8(v);
                    *reinterpret_cast<uint4*>(o) = pk;
                    if constexpr (PUSH) {
                        if (push_now) {
                            const int64_t blk = (int64_t)by * gx + bx;
                            peer[(((blk * (WN * WM) + wave) * TN + i) * TM + j) * 32 + lane] = pk;
                        } else {
                            hold[i * TM + j] = pk;
                        }
                    }
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

// PUSH: a persistent grid (a8_direct_enqueue: as many blocks as fit on the card at once) walks the tiles bx-fastest, and each tile's words
// are pushed during the block's next tile (a8_direct_tile), so the link drains them while the WMMAs run; the system fence comes once, after
// the last tile. (A fence per block held the CU until the link took the tile, and a push in each epilogue stalled the waves on the store
// queue: the push added to the GEMM instead of hiding behind it, lane arx m2 / m4.)
template <bool FOLD, int WN, int WM, int TN, int TM, typename OutT, bool PUSH>
__global__ void __launch_bounds__(WN * WM * 32) a8_direct_kernel(A8Args a, OutT* __restrict__ out, uint4* __restrict__ peer, int gx, int gy) {
    if ((*a.flag == 0) != FOLD) return;
    if constexpr (PUSH) {
        int prev = -1;
        uint4 hold[TN * TM];
        for (int t = blockIdx.x; t < gx * gy; t += gridDim.x) {
            const bool last = t + (int)gridDim.x >= gx * gy;
            a8_direct_tile<FOLD, WN, WM, TN, TM, OutT, PUSH>(a, out, peer, t % gx, t / gx, gx, hold, prev < 0 ? -1 : prev % gx, prev < 0 ? 0 : prev / gx, last);
            prev = t;
        }
        __threadfence_system();
    } else {
        uint4 hold[TN * TM];
        a8_direct_tile<FOLD, WN, WM, TN, TM, OutT, PUSH>(a, out, peer, blockIdx.x, blockIdx.y, gx, hold);
    }
}

// peer non-null: the PUSH kernel (bf16 out only; a8_gemm checks), and `lay` gets the tile
template <bool FOLD, int WN, int WM, int TN, int TM, typename OutT>
void a8_direct_enqueue(const A8Args& a, OutT* out, hipStream_t s, uint4* peer = nullptr, ik_ar_push* lay = nullptr) {
    dim3 grid((unsigned)(((int64_t)a.M + WM * TM * 16 - 1) / (WM * TM * 16)), (unsigned)((a.N + WN * TN * 16 - 1) / (WN * TN * 16)));
    if (peer) {
        // the persistent grid: every block resident at once (blocks per CU from the occupancy API, once per instance)
        static const int per_grid = [] {
            int dev = 0, cus = 0, per_cu = 0;
            (void)hipGetDevice(&dev);
            (void)hipDeviceGetAttribute(&cus, hipDeviceAttributeMultiprocessorCount, dev);
            (void)hipOccupancyMaxActiveBlocksPerMultiprocessor(&per_cu, a8_direct_kernel<FOLD, WN, WM, TN, TM, OutT, true>, WN * WM * 32, 0);
            return cus * (per_cu > 0 ? per_cu : 1);
        }();
        const int64_t tiles = (int64_t)grid.x * grid.y;
        const unsigned nb = (unsigned)(tiles < per_grid ? tiles : per_grid);
        a8_direct_kernel<FOLD, WN, WM, TN, TM, OutT, true><<<nb, WN * WM * 32, 0, s>>>(a, out, peer, (int)grid.x, (int)grid.y);
        if (lay) {
            lay->wn = WN;
            lay->tn = TN;
            lay->tm = TM;
            lay->nw = WN * WM;
            lay->gx = (int32_t)grid.x;
            lay->words = (int64_t)grid.x * grid.y * WN * WM * TN * TM * 32;
        }
    } else {
        a8_direct_kernel<FOLD, WN, WM, TN, TM, OutT, false><<<grid, WN * WM * 32, 0, s>>>(a, out, nullptr, (int)grid.x, (int)grid.y);
    }
}

// INTARSIA_HIP_MXFP4_A8_TILE=WN,WM,TN,TM[,PD] forces one tile (PD 1 or 2, default 1), INTARSIA_HIP_MXFP4_A8_PERGROUP=1 forces the per-group path,
// INTARSIA_HIP_MXFP4_A8_PD=1 restores the previous kernel choice (prefetch distance 1 and the previous tile table). Read once.
// INTARSIA_HIP_MXFP4_A8_DIRECT=0 turns the direct kernel off (tokens through LDS again, a8_gemm_kernel), INTARSIA_HIP_MXFP4_A8_DTILE=WN,WM,TN,TM
// forces one direct tile (sweeps; the tiles of a8_direct_run only). Both read once.
struct A8Cfg {
    int tile[5] = {0, 0, 0, 0, 1};
    int pergroup = 0;
    int pd = 2;
    int direct = 1;
    int dtile[4] = {0, 0, 0, 0};
};
const A8Cfg& a8_cfg() {
    static const A8Cfg cfg = [] {
        A8Cfg c;
        const char* e = getenv("INTARSIA_HIP_MXFP4_A8_TILE");
        if (e && sscanf(e, "%d,%d,%d,%d,%d", &c.tile[0], &c.tile[1], &c.tile[2], &c.tile[3], &c.tile[4]) < 4) c.tile[0] = 0;
        const char* g = getenv("INTARSIA_HIP_MXFP4_A8_PERGROUP");
        c.pergroup = (g && g[0] == '1') ? 1 : 0;
        const char* p = getenv("INTARSIA_HIP_MXFP4_A8_PD");
        c.pd = (p && p[0] == '1') ? 1 : 2;
        const char* d = getenv("INTARSIA_HIP_MXFP4_A8_DIRECT");
        c.direct = (d && d[0] == '0') ? 0 : 1;
        const char* dt = getenv("INTARSIA_HIP_MXFP4_A8_DTILE");
        if (dt && sscanf(dt, "%d,%d,%d,%d", &c.dtile[0], &c.dtile[1], &c.dtile[2], &c.dtile[3]) < 4) c.dtile[0] = 0;
        return c;
    }();
    return cfg;
}

// Both paths of a tile are enqueued; the one the flag does not select exits at once. The per-group path (always prefetch distance 1) has 4 * BN
// floats more LDS, which would put the 256 x 128 tile exactly at 64 KB, so that path takes 128 x 256 there.
template <typename OutT>
void a8_run(const A8Args& a, OutT* out, hipStream_t s) {
#define IK_A8_TILE(WN, WM, TN, TM, PD)                      \
    {                                                       \
        a8_enqueue<true, WN, WM, TN, TM, PD>(a, out, s);    \
        a8_enqueue<false, WN, WM, TN, TM>(a, out, s);       \
        return;                                             \
    }
    const int* t = a8_cfg().tile;
    if (t[0] > 0) {
        const int key = t[0] * 1000 + t[1] * 100 + t[2] * 10 + t[3];
        if (key == 4442 && t[4] == 1) {
            a8_enqueue<true, 4, 4, 4, 2>(a, out, s);
            a8_enqueue<false, 4, 4, 2, 4>(a, out, s);
            return;
        }
        if (key == 4442 && t[4] == 2) {
            a8_enqueue<true, 4, 4, 4, 2, 2>(a, out, s);
            a8_enqueue<false, 4, 4, 2, 4>(a, out, s);
            return;
        }
        if (key == 4424 && t[4] == 1) IK_A8_TILE(4, 4, 2, 4, 1)
        if (key == 4424 && t[4] == 2) IK_A8_TILE(4, 4, 2, 4, 2)
        if (key == 4224 && t[4] == 1) IK_A8_TILE(4, 2, 2, 4, 1)
        if (key == 4224 && t[4] == 2) IK_A8_TILE(4, 2, 2, 4, 2)
        if (key == 2244 && t[4] == 1) IK_A8_TILE(2, 2, 4, 4, 1)
    }
    // ik_f8_gemm.h's f8_gemm_run table (weights x tokens), as the fold launch
    if (a.M <= 64) IK_A8_TILE(4, 1, 2, 2, 1)
    const bool blocks64 = ((a.N + 127) / 128) * (((int64_t)a.M + 255) / 256) >= 64;  // 128 weights x 256 tokens blocks: at least one per CU
    if (a8_cfg().pd == 1) {  // the previous choice
        if (a.M >= 2048 && a.N >= 8192) {
            a8_enqueue<true, 4, 4, 4, 2>(a, out, s);
            a8_enqueue<false, 4, 4, 2, 4>(a, out, s);
            return;
        }
        if (a.M >= 2048 && blocks64) IK_A8_TILE(4, 4, 2, 4, 1)
        IK_A8_TILE(4, 2, 2, 4, 1)
    }
    // prefetch distance 2; the 128 x 256 tile for every M with a block per CU (it also wins below M = 2048 and for gate + up, where the 256 x 128 tile
    // was the choice), the 128 x 128 one for the small grids
    if (blocks64) IK_A8_TILE(4, 4, 2, 4, 2)
    IK_A8_TILE(4, 2, 2, 4, 2)
#undef IK_A8_TILE
}

// The direct kernel's tiles (both paths enqueued, the flag picks one, as a8_run).
template <typename OutT>
void a8_direct_run(const A8Args& a, OutT* out, hipStream_t s, uint4* peer = nullptr, ik_ar_push* lay = nullptr) {
#define IK_A8_DTILE(WN, WM, TN, TM)                                       \
    {                                                                     \
        a8_direct_enqueue<true, WN, WM, TN, TM>(a, out, s, peer, lay);    \
        a8_direct_enqueue<false, WN, WM, TN, TM>(a, out, s, peer, lay);   \
        return;                                                           \
    }
    const int* t = a8_cfg().dtile;
    if (t[0] > 0) {
        const int key = t[0] * 1000 + t[1] * 100 + t[2] * 10 + t[3];
        if (key == 2824) IK_A8_DTILE(2, 8, 2, 4)
        if (key == 2424) IK_A8_DTILE(2, 4, 2, 4)
        if (key == 2224) IK_A8_DTILE(2, 2, 2, 4)
        if (key == 4424) IK_A8_DTILE(4, 4, 2, 4)
        if (key == 4224) IK_A8_DTILE(4, 2, 2, 4)
    }
    // 64 features x 512 tokens (16 waves) from M = 768, else 64 x 256 (8 waves): lane PF27's sweep (raw/m5) over the 27B tp2 shapes at
    // M = 130 .. 2048 (the 512-token tile ties at M = 512 and wins from M = 1000; below it pads too many tokens)
    // lane pf27b: one 64-feature tile column (N <= 64: the 27B's GDN b / a gates, 24 rows per rank) takes the 128-token tile, 16 blocks at
    // M = 2048 instead of 4 (call level 92 -> 79 us; every tile gives the same bits, the a8_mma_hashes of each DTILE agree)
    if (a.N <= 64) IK_A8_DTILE(2, 2, 2, 4)
    if (a.M >= 768) IK_A8_DTILE(2, 8, 2, 4)
    IK_A8_DTILE(2, 4, 2, 4)
#undef IK_A8_DTILE
}

size_t a8_round16(size_t v) { return (v + 15) & ~(size_t)15; }

// scratch: mx_a8_scratch_bytes(round16(M), round64(K)) (xq, xs; ik_mx_a8.h; the direct kernel's tiled xq covers the 16-padded M and the
// 64-padded K), padded to 16,
// then wref [round16(N)] and the fold flag [16]. quantmoe.rs's matmul_mxfp4_a8_mma scratch is the same size.
size_t a8_scratch_total(int64_t M, int64_t K, int64_t N) {
    return a8_round16(mx_a8_scratch_bytes((M + 15) & ~(int64_t)15, (K + 63) & ~(int64_t)63)) + a8_round16((size_t)N) + 16;
}

// the direct kernel for this call? (M > 64: the staged kernel's 64-row tile keeps the small calls)
bool a8_use_direct(int64_t M) {
    const A8Cfg& c = a8_cfg();
    return c.direct && M > 64 && c.tile[0] == 0 && c.pd == 2;
}

// the shape / dtype checks of one GEMM of launch_a8 (x [M, K] bf16, w [N, K/2], sc [N, K/32], out [M, N] bf16 or f32)
int a8_check(const ik_tensor* x, const ik_tensor* w, const ik_tensor* sc, ik_tensor* out) {
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
    return IK_OK;
}

// the activation quant pre-pass into the scratch base (the direct kernel's fragment-ordered tokens when `direct`); the direct path's pre-pass
// also zeroes `nzero` fold flags at `flags` (lane pf27b; the staged path's GEMM zeroes its flag with a memset as before)
int a8_quant(const ik_tensor* x, uint8_t* base, int64_t M, int64_t K, bool direct, int* flags, int nzero, hipStream_t s) {
    const int64_t Mp = (M + 15) & ~(int64_t)15, Kp = (K + 63) & ~(int64_t)63;
    if (direct) {
        if (reinterpret_cast<uintptr_t>(x->data) & 15u) return IK_ERR_INVALID_ARG;
        mx_a8_quant_tiled_kernel<<<(unsigned)Mp, 256, 0, s>>>((const ik_bf16*)x->data, base, reinterpret_cast<float*>(base + mx_a8_xs_offset(Mp, Kp)),
                                                            (int)M, (int)K, flags, nzero);
        return ik_launch_status();
    }
    return mx_a8_quant_launch((const ik_bf16*)x->data, base, M, K, s);
}

// the fold flags and weight-span passes of one GEMM (w1 null) or a pair (both weights in one launch, each its own wref and flag): the
// flags are zero on entry when `zeroed` (the direct pre-pass did it), else memset here; the per-group knob sets them to 0x01010101.
int a8_spans(const ik_tensor* w0, const ik_tensor* s0, uint8_t* wref0, int* flag0, const ik_tensor* w1, const ik_tensor* s1, uint8_t* wref1,
             int* flag1, int64_t K, bool zeroed, hipStream_t s) {
    const int64_t N0 = w0->shape[0], N1 = w1 ? w1->shape[0] : 0;
    if (a8_cfg().pergroup) {  // flag = 0x01010101: never folds
        if (hipMemsetAsync(flag0, 1, sizeof(int), s) != hipSuccess) return IK_ERR_LAUNCH;
        if (w1 && hipMemsetAsync(flag1, 1, sizeof(int), s) != hipSuccess) return IK_ERR_LAUNCH;
        return IK_OK;
    }
    if (!zeroed) {
        if (hipMemsetAsync(flag0, 0, sizeof(int), s) != hipSuccess) return IK_ERR_LAUNCH;
        if (w1 && hipMemsetAsync(flag1, 0, sizeof(int), s) != hipSuccess) return IK_ERR_LAUNCH;
    }
    a8_wspan_kernel<<<(unsigned)((N0 + N1 + 7) / 8), 256, 0, s>>>((const uint8_t*)s0->data, wref0, flag0, N0, (int)(K / kA8G),
                                                                 w1 ? (const uint8_t*)s1->data : nullptr, wref1, flag1, N1);
    return ik_launch_status();
}

// one GEMM on the quantised activations in the scratch base, after its span pass (wref, flag). lay non-null (lane arx): also push the output
// into `peer` (ar_push.h) when this call takes the direct kernel with a bf16, 8-aligned output whose layout fits lay->cap; lay->pushed says
// whether it did (otherwise the GEMM is the ordinary one and the all-reduce pushes as before)
int a8_gemm(const ik_tensor* w, const ik_tensor* sc, ik_tensor* out, uint8_t* base, const uint8_t* wref, const int* flag, int64_t M, int64_t K,
            bool direct, hipStream_t s, void* peer = nullptr, ik_ar_push* lay = nullptr) {
    const int64_t N = w->shape[0];
    const int64_t Mp = (M + 15) & ~(int64_t)15, Kp = (K + 63) & ~(int64_t)63;
    A8Args a;
    a.xq = base;
    a.xs = reinterpret_cast<const float*>(base + (direct ? mx_a8_xs_offset(Mp, Kp) : mx_a8_xs_offset(M, K)));
    a.w = (const uint8_t*)w->data;
    a.ws = (const uint8_t*)sc->data;
    a.wref = wref;
    a.flag = flag;
    a.M = (int)M;
    a.N = N;
    a.K = (int)K;
    a.vec_out = (N % 8 == 0) && ik_aligned16(out->data);
    if (lay) {
        lay->pushed = 0;
        // every direct tile is at most 512 tokens x 128 features: the layout of any of them fits this bound
        const int64_t bound = ((M + 511) / 512 * 512) * ((N + 127) / 128 * 128) * 2;
        if (peer && direct && out->dtype == IK_DTYPE_BF16 && a.vec_out && bound <= lay->cap) {
            lay->M = (int32_t)M;
            lay->N = N;
            a8_direct_run<ik_bf16>(a, (ik_bf16*)out->data, s, (uint4*)peer, lay);
            lay->pushed = 1;
            return ik_launch_status();
        }
    }
    if (direct) {
        if (out->dtype == IK_DTYPE_F32) a8_direct_run<float>(a, (float*)out->data, s);
        else a8_direct_run<ik_bf16>(a, (ik_bf16*)out->data, s);
    } else if (out->dtype == IK_DTYPE_F32) {
        a8_run<float>(a, (float*)out->data, s);
    } else {
        a8_run<ik_bf16>(a, (ik_bf16*)out->data, s);
    }
    return ik_launch_status();
}

int launch_a8(const ik_tensor* x, const ik_tensor* w, const ik_tensor* sc, ik_tensor* out, const ik_scratch* scratch,
              void* stream, void* peer = nullptr, ik_ar_push* lay = nullptr) {
    if (lay) lay->pushed = 0;
    IK_TRY(a8_check(x, w, sc, out));
    const int64_t M = x->shape[0], K = x->shape[1], N = w->shape[0];
    if (M == 0 || N == 0) return IK_OK;
    hipStream_t s = IK_STREAM(stream);
    if (K == 0)
        return hipMemsetAsync(out->data, 0, ik_numel(out) * (out->dtype == IK_DTYPE_F32 ? 4 : 2), s) == hipSuccess ? IK_OK
                                                                                                                      : IK_ERR_LAUNCH;
    if (!scratch || !scratch->data || scratch->bytes < (int64_t)a8_scratch_total(M, K, N)) return IK_ERR_INVALID_ARG;
    if (reinterpret_cast<uintptr_t>(scratch->data) & 15u) return IK_ERR_INVALID_ARG;
    uint8_t* base = (uint8_t*)scratch->data;
    const int64_t Mp = (M + 15) & ~(int64_t)15, Kp = (K + 63) & ~(int64_t)63;
    uint8_t* wref = base + a8_round16(mx_a8_scratch_bytes(Mp, Kp));
    int* flag = reinterpret_cast<int*>(wref + a8_round16((size_t)N));
    const bool direct = a8_use_direct(M);
    IK_TRY(a8_quant(x, base, M, K, direct, flag, 1, s));
    IK_TRY(a8_spans(w, sc, wref, flag, nullptr, nullptr, nullptr, nullptr, K, direct, s));
    return a8_gemm(w, sc, out, base, wref, flag, M, K, direct, s, peer, lay);
}

// Lane pf27b: two projections of one activation (GDN qkv | z and b | a, attention k | v, FFN gate | up) on ONE quant pre-pass and one span
// launch. Each GEMM is exactly launch_a8's (its own weight-span result and fold flag, the same kernel and tile for this M), and the pre-pass
// is a pure function of x, so both outputs are bitwise two launch_a8 calls. Scratch: a8_scratch_total(M, K, round16(N0) + N1) (the
// matmul_mxfp4_pair_a8_mma registration in quantmoe.rs): wref0 [round16(N0)], wref1 [round16(N1)], then the two flags.
int launch_a8_pair(const ik_tensor* x, const ik_tensor* w0, const ik_tensor* s0, const ik_tensor* w1, const ik_tensor* s1, ik_tensor* out0,
                   ik_tensor* out1, const ik_scratch* scratch, void* stream) {
    IK_TRY(a8_check(x, w0, s0, out0));
    IK_TRY(a8_check(x, w1, s1, out1));
    const int64_t M = x->shape[0], K = x->shape[1], N0 = w0->shape[0], N1 = w1->shape[0];
    if (M == 0 || K == 0 || N0 == 0 || N1 == 0) {
        IK_TRY(launch_a8(x, w0, s0, out0, scratch, stream));
        return launch_a8(x, w1, s1, out1, scratch, stream);
    }
    hipStream_t s = IK_STREAM(stream);
    const int64_t NN = (int64_t)a8_round16((size_t)N0) + N1;
    if (!scratch || !scratch->data || scratch->bytes < (int64_t)a8_scratch_total(M, K, NN)) return IK_ERR_INVALID_ARG;
    if (reinterpret_cast<uintptr_t>(scratch->data) & 15u) return IK_ERR_INVALID_ARG;
    uint8_t* base = (uint8_t*)scratch->data;
    const int64_t Mp = (M + 15) & ~(int64_t)15, Kp = (K + 63) & ~(int64_t)63;
    uint8_t* wref0 = base + a8_round16(mx_a8_scratch_bytes(Mp, Kp));
    uint8_t* wref1 = wref0 + a8_round16((size_t)N0);
    int* flag0 = reinterpret_cast<int*>(wref0 + a8_round16((size_t)NN));
    int* flag1 = flag0 + 1;
    const bool direct = a8_use_direct(M);
    IK_TRY(a8_quant(x, base, M, K, direct, flag0, 2, s));
    IK_TRY(a8_spans(w0, s0, wref0, flag0, w1, s1, wref1, flag1, K, direct, s));
    IK_TRY(a8_gemm(w0, s0, out0, base, wref0, flag0, M, K, direct, s));
    return a8_gemm(w1, s1, out1, base, wref1, flag1, M, K, direct, s);
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

// lane arx: ik_matmul_mxfp4_a8_mma whose output also goes into the peer rank's all-reduce slot `peer` (ar_push.h) when the call takes the
// direct kernel; lay->cap is the slot's bytes, lay->pushed and the layout fields come back. The output is the plain call's, bit for bit.
int ik_matmul_mxfp4_a8_mma_push(const ik_tensor* x, const ik_tensor* w, const ik_tensor* scales, const ik_tensor* act, ik_tensor* out,
                                const ik_scratch* scratch, void* peer, ik_ar_push* lay, void* stream) {
    (void)act;
    if (!lay) return IK_ERR_INVALID_ARG;
    return launch_a8(x, w, scales, out, scratch, stream, peer, lay);
}

// matmul_mxfp4_pair_a8_mma (lane pf27b): MatmulMxfp4Pair at prefill M, one quant pre-pass for both GEMMs (launch_a8_pair). `act` is not read.
// Scratch: a8_scratch_total(M, K, round16(N0) + N1).
int ik_matmul_mxfp4_pair_a8_mma(const ik_tensor* x, const ik_tensor* w0, const ik_tensor* s0, const ik_tensor* w1, const ik_tensor* s1,
                                const ik_tensor* act, ik_tensor* out0, ik_tensor* out1, const ik_scratch* scratch, void* stream) {
    (void)act;
    return launch_a8_pair(x, w0, s0, w1, s1, out0, out1, scratch, stream);
}

}  // extern "C"
