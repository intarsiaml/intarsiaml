/* Lane l5 (docs/results/2026-10-25-l5-moe-fp8-wmma.md): grouped NVFP4 expert projections on the fp8 WMMA, W4A8, for prefill-sized calls.
 * OPT-IN (INTARSIA_HIP_MOE_FP8_WMMA=1, moe.hip): the activations are quantised to e4m3, which changes the numerics of the MoE block.
 *
 *   gate_up : h[pair, f] = bf16( round(silu(round(g))) * round(u) ),  g = gscale0 * a_row * sum_k xq[row, k] w_gate[e, f, k]
 *   down    : y[pair, f] = bf16( gscale2 * a_row * sum_k hq[pair, k] w_down[e, f, k] )
 *
 * with xq = e4m3(x / a_row), a_row = max|x_row| / 448 (per row = per token; for down per pair), and w = e2m1 * e4m3 block scale (one scale per
 * 16 k of a weight row). Every e2m1 value is exact in e4m3 and every e4m3 x e4m3 product is exact in f32; the block scale is applied to the
 * f32 partial sum of ONE fp8 WMMA, because one NVFP4 block (16 k) is exactly the k of one v_wmma_f32_16x16x16_fp8_fp8:
 *
 *       part = wmma(x_frag, w_frag, 0);   acc += part * scale_block        (8 FMAs per WMMA, per lane)
 *
 * so the sum over k is exact up to f32 additions. Weights are the B operand (columns = features) and the activations the A operand (rows =
 * pairs): the C fragment of a lane is then ONE feature column and 8 pairs (lane l: feature l % 16, pairs (l / 16) * 8 + i), i.e. the block
 * scale is one scalar per lane -- the scale of the weight row the lane itself loaded. (moe_wmma.h has the weights as A, where a lane's 8
 * accumulators are 8 features with 8 different scales; and it permutes k so that one WMMA step covers two scale blocks. Neither works here.)
 *
 * k layout. A 64-k block is 4 WMMA steps t = 0..3, step t = elements 16t .. 16t+15 = one scale block; lane half h (lane / 16) takes elements
 * 16t + 8h .. +7 = dword 2t + h of the row's 32 packed bytes. A lane loads dwords 4h .. 4h+3 (one 16-byte load, as in moe_wmma_r7.h) and
 * trades two dwords with the lane 16 away (v_permlanex16) to hold {h, 2+h, 4+h, 6+h}. The 8 nibbles of a dword become two e4m3 dwords with
 * ik_fp4x8_to_e4m3 (value * 2^-6, exact): even elements (0, 2, 4, 6) in the first, odd (1, 3, 5, 7) in the second. That pair IS the B fragment, so
 * the k slots of a lane are ordered (0, 2, 4, 6, 1, 3, 5, 7) inside its group of 8; the activation quantiser stores its bytes in that order, and
 * the A fragment is one 8-byte LDS read. The 2^6 is applied once in the epilogue (a_row * 64).
 *
 * Block = 8 waves, grid (active expert slot, output-feature tile, pair tile of PAIRS) exactly as moe_wmma_kernel_r7: gate_up 64 features
 * (waves 0-3 gate, 4-7 up, the up accumulators go to the gate waves through LDS), down 128 features. Every output accumulates the same WMMAs in
 * the same k order whatever else is in the tile, so a pair's result does not depend on the grouping.
 *
 * Structure and the e2m1 -> e4m3 byte map follow moe_wmma_r7.h; the idea (fp8 WMMA experts, per-row e4m3 activations) is radiance's
 * radiance-moe-w4a8-2 (Apache-2.0, kernels/hip/NOTICE-radiance), whose MXFP4 exponent fold does not port to e4m3 block scales.
 */
#ifndef IK_MOE_WMMA_FP8_H
#define IK_MOE_WMMA_FP8_H

#include "moe_wmma_r7.h"

typedef int ik_mf8_v2i __attribute__((ext_vector_type(2)));

constexpr int kMf8KB = 64;          // k per block of the loop
constexpr int kMf8LD = kMf8KB + 16;  // LDS row of the activation tile (bytes): 80
constexpr int kMf8Skip = 1, kMf8Deep = 2, kMf8ZFast = 4;  // variant bits V
/* kMf8ZFast: the grid is (pair tile, feature tile, expert slot) instead of (slot, feature tile, pair tile), so the blocks of one expert's
 * pair tiles are launched next to each other and read the expert's weights from the L2 together (launch geometry only: same bits). */
#ifdef IK_F8_ABLATE  /* scripts/l5/f8bench.hip only: timing ablations (the results are wrong by design) */
constexpr int kMf8NoW = 16, kMf8NoFma = 32, kMf8NoLds = 64, kMf8NoDec = 128;
#endif

IK_DEVI ik_mw_v8f ik_mf8_wmma(const uint2& a, const uint2& b, const ik_mw_v8f& c) {
    return __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(__builtin_bit_cast(ik_mf8_v2i, a), __builtin_bit_cast(ik_mf8_v2i, b), c);
}

/* the dword of lane ^ 16 (the other 16-lane row) */
IK_DEVI uint32_t ik_mf8_xor16(uint32_t v) {
    return __builtin_amdgcn_permlanex16(v, v, 0x76543210u, 0xfedcba98u, false, false);
}

/* e4m3 byte `i` (0..3, a compile-time constant after unrolling) of `w` as f32, exact; the block scales are never NaN */
#if __has_builtin(__builtin_amdgcn_cvt_f32_fp8)
IK_DEVI float ik_mf8_byte(uint32_t w, int i) {
    return i == 0 ? __builtin_amdgcn_cvt_f32_fp8((int)w, 0)
         : i == 1 ? __builtin_amdgcn_cvt_f32_fp8((int)w, 1)
         : i == 2 ? __builtin_amdgcn_cvt_f32_fp8((int)w, 2)
                  : __builtin_amdgcn_cvt_f32_fp8((int)w, 3);
}
#else
IK_DEVI float ik_mf8_byte(uint32_t w, int i) { return ik_e4m3_to_f((w >> (8 * i)) & 0xffu); }
#endif

/* ---- activation quantiser ------------------------------------------------------------------------------------------------ */

/* x[rows, K] bf16 -> q[rows, K] e4m3 bytes (x / s, RNE, saturating at +-448), s[rows] = max(amax / 448, 1e-20). Within every group of 8
 * elements the bytes are stored in the order (0, 2, 4, 6, 1, 3, 5, 7) (see the top). One wave per row; K = 256 * NCH. */
template <int NCH>
__global__ void __launch_bounds__(256) moe_f8_quant_kernel(const ik_bf16* __restrict__ x, uint8_t* __restrict__ q, float* __restrict__ sc, int rows) {
    constexpr int K = 256 * NCH;
    const int lane = threadIdx.x & 31, row = blockIdx.x * 8 + (threadIdx.x >> 5);
    if (row >= rows) return;
    const uint4* xr = reinterpret_cast<const uint4*>(x + (int64_t)row * K);
    uint4 v[NCH];
    float amax = 0.f;
#pragma unroll
    for (int c = 0; c < NCH; ++c) {
        v[c] = xr[c * 32 + lane];
        float f[8];
        ik_unpack8(v[c], f);
#pragma unroll
        for (int i = 0; i < 8; ++i) amax = fmaxf(amax, fabsf(f[i]));
    }
    amax = ik_warp_max(amax);
    const float s = fmaxf(__fdiv_rn(amax, 448.f), 1e-20f);
    if (lane == 0) sc[row] = s;
    uint2* qr = reinterpret_cast<uint2*>(q + (int64_t)row * K);
#pragma unroll
    for (int c = 0; c < NCH; ++c) {
        float f[8];
        ik_unpack8(v[c], f);
        float y[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) y[i] = fminf(fmaxf(__fdiv_rn(f[i], s), -448.f), 448.f);
#if __has_builtin(__builtin_amdgcn_cvt_pk_fp8_f32)
        int w0 = __builtin_amdgcn_cvt_pk_fp8_f32(y[0], y[2], 0, false);
        w0 = __builtin_amdgcn_cvt_pk_fp8_f32(y[4], y[6], w0, true);
        int w1 = __builtin_amdgcn_cvt_pk_fp8_f32(y[1], y[3], 0, false);
        w1 = __builtin_amdgcn_cvt_pk_fp8_f32(y[5], y[7], w1, true);
        qr[c * 32 + lane] = make_uint2((uint32_t)w0, (uint32_t)w1);
#else
        uint32_t w0 = 0, w1 = 0;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            w0 |= (uint32_t)ik_f_to_e4m3(y[2 * i]) << (8 * i);
            w1 |= (uint32_t)ik_f_to_e4m3(y[2 * i + 1]) << (8 * i);
        }
        qr[c * 32 + lane] = make_uint2(w0, w1);
#endif
    }
}

/* ---- the grouped projection ---------------------------------------------------------------------------------------------- */

struct Mf8W {  // one 64-k block of weights of this lane: its 16 code bytes (dwords 4h .. 4h+3) and the 4 block scales of its row
    uint4 c;
    uint32_t s;
};

constexpr int mf8_feat(bool gateup) { return gateup ? kMwFeatGateUp : kMwFeatDown; }

template <bool GATEUP, int PAIRS, int V>
__global__ void __launch_bounds__(kMwThreads)
moe_f8_kernel(const uint8_t* __restrict__ xq, const float* __restrict__ xs, const uint8_t* __restrict__ w, const uint8_t* __restrict__ ws,
              const float* __restrict__ gscales, const int* __restrict__ pairs, const int* __restrict__ offsets,
              const int* __restrict__ active, ik_bf16* __restrict__ dst, int E, int Kd, int Nd, int topk, int y0) {
    constexpr bool SKIP = (V & kMf8Skip) != 0, ZFAST = (V & kMf8ZFast) != 0;
#ifdef IK_F8_ABLATE
    constexpr bool NOW = (V & kMf8NoW) != 0, NOFMA = (V & kMf8NoFma) != 0, NOLDS = (V & kMf8NoLds) != 0, NODEC = (V & kMf8NoDec) != 0;
#endif
    constexpr int RING = (V & kMf8Deep) ? 6 : 3;  // weight blocks in flight per lane
    constexpr int FEAT = mf8_feat(GATEUP);
    constexpr int J = PAIRS / 16;   // 16-pair A fragments per weight fragment
    constexpr int XR = PAIRS / 64;  // activation rows (of 64 bytes per k block) each thread stages
    static_assert(PAIRS % 64 == 0, "pair tile");
    __shared__ __align__(16) uint8_t sx[2][PAIRS * kMf8LD];
    __shared__ __align__(16) float sred[GATEUP ? 4 * 32 * J * 8 : 1];  // up accumulators -> gate waves
    __shared__ int spair[PAIRS];     // pair index of every row of the tile (-1: none)
    __shared__ float srow[PAIRS];    // 64 * a_row of every row of the tile

    const int slot = ZFAST ? (int)blockIdx.z : (int)blockIdx.x;
    if (slot >= active[E]) return;  // block-uniform
    const int e = active[slot];
    const int begin = offsets[e], cnt = offsets[e + 1] - begin;
    const int n0 = ((int)blockIdx.y + y0) * FEAT;
    const int pt0 = ZFAST ? (int)blockIdx.x : (int)blockIdx.z, ptn = ZFAST ? (int)gridDim.x : (int)gridDim.z;  // first pair tile of this block, stride
    const int tid = threadIdx.x, wave = tid >> 5, lane = tid & 31;
    const int lr = lane & 15, h = lane >> 4;
    const bool hi = h != 0;
    const int part = GATEUP ? (wave >> 2) : 0;  // gate_up: 0 = gate rows, 1 = up rows
    const int fbase = n0 + (GATEUP ? (wave & 3) : wave) * 16;
    const float g0 = gscales[e * 3 + (GATEUP ? 0 : 2)];
    const float g1 = GATEUP ? gscales[e * 3 + 1] : 0.f;
    const int64_t kb2 = Kd / 2, kb16 = Kd / 16;  // bytes of packed codes / block scales per weight row
    const int nkb = Kd / kMf8KB;

    // this lane's weight row (feature n), pointing at its 16 bytes of the k block
    const int n = fbase + lr;
    const int64_t wrow = GATEUP ? ((int64_t)e * 2 + part) * Nd + n : (int64_t)e * Nd + n;
    const uint8_t* wcode = n < Nd ? w + wrow * kb2 + h * 16 : nullptr;
    const uint8_t* wsc = n < Nd ? ws + wrow * kb16 : nullptr;
    auto loadW = [&](int kb) {
        Mf8W r;
#ifdef IK_F8_ABLATE
        if (NOW) {
            r.c = make_uint4(0x12345678u * (kb + 1), 0x9abcdef0u + kb, 0x0fedcba9u ^ kb, 0x87654321u + lane);
            r.s = 0x38383838u;
            return r;
        }
#endif
        if (wcode) {
            r.c = *reinterpret_cast<const uint4*>(wcode + (int64_t)kb * 32);
            r.s = *reinterpret_cast<const uint32_t*>(wsc + (int64_t)kb * 4);
        } else {
            r.c = make_uint4(0, 0, 0, 0);
            r.s = 0;
        }
        return r;
    };

    for (int pt = pt0; pt * PAIRS < cnt; pt += ptn) {
        // this thread's activation rows (rows xr + 64 * r of the tile) and its 16-byte column chunk of the 64-byte k block
        const int xr = tid >> 2, xch = tid & 3;
        const uint8_t* xrow[XR];
#pragma unroll
        for (int r = 0; r < XR; ++r) {
            xrow[r] = nullptr;
            const int p_idx = pt * PAIRS + xr + 64 * r;
            if (p_idx < cnt) {
                const int pair = pairs[begin + p_idx];
                xrow[r] = xq + (int64_t)(GATEUP ? pair / topk : pair) * Kd;  // token row (gate_up) / own row (down)
            }
        }
        uint4 rx[XR];
        auto loadX = [&](int kb) {
#pragma unroll
            for (int r = 0; r < XR; ++r) rx[r] = xrow[r] ? *reinterpret_cast<const uint4*>(xrow[r] + kb * kMf8KB + xch * 16) : make_uint4(0, 0, 0, 0);
        };
        auto storeX = [&](int buf) {
#pragma unroll
            for (int r = 0; r < XR; ++r) *reinterpret_cast<uint4*>(&sx[buf][(xr + 64 * r) * kMf8LD + xch * 16]) = rx[r];
        };

        ik_mw_v8f acc[J];
#pragma unroll
        for (int j = 0; j < J; ++j)
#pragma unroll
            for (int i = 0; i < 8; ++i) acc[j][i] = 0.f;

        auto kloop = [&](auto jn_c) {
            constexpr int JN = decltype(jn_c)::value;
            auto compute = [&](const Mf8W& W, int buf) {
                // dwords {h, 2+h, 4+h, 6+h} of the row's 8: swap two dwords with the lane 16 away (6 cndmask + 2 permlanex16)
                const uint32_t send1 = hi ? W.c.x : W.c.y, send2 = hi ? W.c.z : W.c.w;
                const uint32_t r1 = ik_mf8_xor16(send1), r2 = ik_mf8_xor16(send2);
                const uint32_t sw[4] = {hi ? r1 : W.c.x, hi ? r2 : W.c.z, hi ? W.c.y : r1, hi ? W.c.w : r2};
                const ik_mw_v8f zero = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#pragma unroll
                for (int t = 0; t < 4; ++t) {
                    uint32_t ev, od;
                    ik_fp4x8_to_e4m3(sw[t], ev, od);
#ifdef IK_F8_ABLATE
                    if (NODEC) { ev = sw[t]; od = sw[t] ^ 0x01010101u; }
#endif
                    const uint2 bw = make_uint2(ev, od);
                    const float sc = ik_mf8_byte(W.s, t);
#pragma unroll
                    for (int j = 0; j < JN; ++j) {
                        uint2 af = *reinterpret_cast<const uint2*>(&sx[buf][(j * 16 + lr) * kMf8LD + t * 16 + h * 8]);
#ifdef IK_F8_ABLATE
                        if (NOLDS) af = make_uint2(0x38383838u + t * 0x01010101u + j, 0x30303030u + lane);
                        if (NOFMA) {
                            acc[j] = ik_mf8_wmma(af, bw, acc[j]);
                            continue;
                        }
#endif
                        const ik_mw_v8f p = ik_mf8_wmma(af, bw, zero);
#pragma unroll
                        for (int i = 0; i < 8; ++i) acc[j][i] = __builtin_fmaf(p[i], sc, acc[j][i]);
                    }
                }
            };

            __syncthreads();  // the previous pair tile's last reads of sx / sred / spair / srow are done
            if (tid < PAIRS) {
                const int p_idx = pt * PAIRS + tid;
                int pair = -1;
                float a = 0.f;
                if (p_idx < cnt) {
                    pair = pairs[begin + p_idx];
                    a = xs[GATEUP ? pair / topk : pair] * 64.f;
                }
                spair[tid] = pair;
                srow[tid] = a;
            }
            Mf8W W[RING];
            loadX(0);
#pragma unroll
            for (int i = 0; i < RING; ++i)
                if (i < nkb) W[i] = loadW(i);
            storeX(0);
            if (nkb > 1) loadX(1);
            __syncthreads();
            for (int kb0 = 0; kb0 < nkb; kb0 += RING) {
#pragma unroll
                for (int i = 0; i < RING; ++i) {
                    const int kb = kb0 + i;
                    if (kb >= nkb) break;
                    compute(W[i], kb & 1);
                    if (kb + RING < nkb) W[i] = loadW(kb + RING);
                    if (kb + 1 < nkb) {
                        storeX((kb + 1) & 1);
                        if (kb + 2 < nkb) loadX(kb + 2);
                    }
                    __syncthreads();
                }
            }
        };
        if constexpr (SKIP && J > 1) {
            const int rem = cnt - pt * PAIRS;  // > 0, block-uniform
            const int jn = rem >= PAIRS ? J : (rem + 15) >> 4;
            if (jn >= J) kloop(std::integral_constant<int, J>{});
            else if (jn == 1) kloop(std::integral_constant<int, 1>{});
            else if (jn == 2) kloop(std::integral_constant<int, (J > 2 ? 2 : 1)>{});
            else if (jn == 3) kloop(std::integral_constant<int, (J > 3 ? 3 : 1)>{});
            else kloop(std::integral_constant<int, J>{});
        } else {
            kloop(std::integral_constant<int, J>{});
        }

        if constexpr (GATEUP) {
            if (part == 1) {
#pragma unroll
                for (int j = 0; j < J; ++j)
#pragma unroll
                    for (int i = 0; i < 8; ++i) sred[((wave & 3) * 32 + lane) * (J * 8) + j * 8 + i] = acc[j][i];
            }
            __syncthreads();
            if (part == 1) continue;
        }
        if (n >= Nd) continue;
#pragma unroll
        for (int j = 0; j < J; ++j) {
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int pi = j * 16 + h * 8 + i;
                const int pair = spair[pi];
                if (pair < 0) continue;
                const float a = srow[pi];
                float v;
                if constexpr (GATEUP) {
                    const float g = ik_round_bf16((acc[j][i] * a) * g0);
                    const float u = ik_round_bf16((sred[((wave & 3) * 32 + lane) * (J * 8) + j * 8 + i] * a) * g1);
                    const float s = ik_round_bf16(g / (1.f + expf(-g)));
                    v = s * u;
                } else {
                    v = (acc[j][i] * a) * g0;
                }
                dst[(int64_t)pair * Nd + n] = ik_f2b(v);
            }
        }
    }
}

#endif /* IK_MOE_WMMA_FP8_H */
