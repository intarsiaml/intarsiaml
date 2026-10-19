/* Grouped bf16-WMMA expert projections for the sparse-MoE block (prefill and any larger token count).
 *
 * The GEMV-style projection kernels in moe.hip stream every expert row once per pass over <= 8 pairs on the CUDA
 * cores; at a few hundred tokens per chunk (every expert active, ~8 pairs each) that is ALU-bound and dominates
 * prefill. These kernels instead run bf16 WMMA with the expert weights (e2m1 * e4m3 block scale, exact in bf16)
 * dequantised straight into the A fragments in registers: no weight tile goes through shared memory, so only the
 * small activation tile needs a barrier, the kernel keeps many waves resident, and each lane has three 64-k blocks
 * of weights in flight (an LDS-staged version was limited to a few blocks per CU and ran at ~150 GB/s).
 *
 *   gate_up : h[pair, f] = bf16( round(silu(round(g))) * round(u) ), g = gate_scale * (Wg[e,f] . x[token]),
 *             u = up_scale * (Wu[e,f] . x[token]); both accumulators of a feature live in the same lane.
 *   down    : y[pair, f] = bf16(down_scale * (Wd[e,f] . h[pair]))
 *
 * Fragment layout (gfx12 wave32, 16x16x16): A (weights, 16 rows x 16 k) and B (activations, 16 k x 16 pairs): lane l
 * holds row/col l % 16 and k slots (l / 16) * 8 .. +7; C lane l holds pair l % 16 and features (l / 16) * 8 + i.
 * The k slots are permuted identically for A and B so that each lane's weight bytes are contiguous in memory: per
 * 64-element k block, lane (row, h) takes elements h * 32 + s * 8 .. +7 for WMMA step s = 0..3, i.e. 16 consecutive
 * code bytes and two block-scale bytes per row and block (one NVFP4 scale block is 16 elements; under this permutation the two lane halves h = 0, 1 read different
 * scale blocks, which is why each lane loads the two scale bytes of its half, at block offset kb * 4 + h * 2).
 *
 * Blocks have 8 waves. gate_up: waves 0-3 take the gate rows and waves 4-7 the up rows of the same 64 features (the
 * up accumulators go through shared memory to the gate waves, which apply the activation); down: 8 waves, 128
 * features. Each wave holds one A fragment row per step, which keeps the register count low enough for many waves.
 *
 * PAIRS (32 or 64) pairs per block: each dequantised weight fragment feeds PAIRS / 16 WMMAs, so a 64-pair tile halves the
 * decode ALU work per WMMA (the e2m1 lookups, scale multiplies and permutes) when the experts are busy enough to fill it.
 *
 * Grid: (active expert slot, output-feature tile, pair tile of PAIRS). Pair tiles beyond the expert's pair count
 * exit; a block steps through further pair tiles by gridDim.z when an expert has more pairs than the grid covers.
 * Every output element accumulates over k in the same order whatever else is in the tile, so a pair's result does
 * not depend on the grouping (deterministic run to run). Requires the reduction dimension to be a multiple of 64.
 */
#ifndef IK_MOE_WMMA_H
#define IK_MOE_WMMA_H

#include <type_traits>

#include "ik_quant.h"

typedef __bf16 ik_mw_v8bf __attribute__((ext_vector_type(8)));
typedef float ik_mw_v8f __attribute__((ext_vector_type(8)));

IK_DEVI ik_mw_v8f ik_mw_wmma(const uint4& a, const uint4& b, ik_mw_v8f c) {
    return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(__builtin_bit_cast(ik_mw_v8bf, a),
                                                             __builtin_bit_cast(ik_mw_v8bf, b), c);
}

/* 8 fp4 codes (one 32-bit word) times a block scale -> 8 bf16. lut value * scale is exact in f32 and in bf16, so the
 * f32 -> bf16 conversion is a plain truncation (one byte permute per pair). */
IK_DEVI uint4 ik_fp4x8_scaled_bf16(uint32_t c, float sc, const float* lut) {
    uint32_t u[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) u[j] = __float_as_uint(lut[(c >> (4 * j)) & 15u] * sc);
    uint4 r;
    r.x = __builtin_amdgcn_perm(u[1], u[0], 0x07060302u);
    r.y = __builtin_amdgcn_perm(u[3], u[2], 0x07060302u);
    r.z = __builtin_amdgcn_perm(u[5], u[4], 0x07060302u);
    r.w = __builtin_amdgcn_perm(u[7], u[6], 0x07060302u);
    return r;
}

constexpr int kMwKB = 64;            // k elements per block
constexpr int kMwLD = kMwKB + 8;     // padded LDS row of the activation tile (elements): 144 B
constexpr int kMwPairs = 32;         // pairs per block (the PAIRS template argument: 32, or 64 for busy experts, moe.hip)
constexpr int kMwThreads = 256;      // 8 waves
constexpr int kMwRing = 3;           // weight blocks in flight per lane
constexpr int kMwFeatGateUp = 64;    // output features per block
constexpr int kMwFeatDown = 128;

struct MwW {  // one 64-k block of weights of this lane: 16 code bytes and 2 scale bytes of its row
    uint4 c;
    uint32_t s;
};

template <bool GATEUP, int PAIRS>
__global__ void __launch_bounds__(kMwThreads)
moe_wmma_kernel(const ik_bf16* __restrict__ src, const uint8_t* __restrict__ w, const uint8_t* __restrict__ ws,
                const float* __restrict__ gscales, const int* __restrict__ pairs, const int* __restrict__ offsets,
                const int* __restrict__ active, ik_bf16* __restrict__ dst, int E, int Kd, int Nd, int topk) {
    constexpr int FEAT = GATEUP ? kMwFeatGateUp : kMwFeatDown;
    constexpr int J = PAIRS / 16;   // 16-pair B fragments per weight fragment (each dequantised A fragment feeds J WMMAs)
    constexpr int XR = PAIRS / 32;  // activation rows each thread stages
    static_assert(PAIRS % 32 == 0, "pair tile");
    __shared__ __align__(16) uint16_t sx[2][PAIRS * kMwLD];
    __shared__ __align__(16) float sred[GATEUP ? 4 * 32 * J * 8 : 1];  // up accumulators -> gate waves
    IK_FP4_LUT(lut);

    const int slot = blockIdx.x;
    if (slot >= active[E]) return;  // block-uniform
    const int e = active[slot];
    const int begin = offsets[e], cnt = offsets[e + 1] - begin;
    const int n0 = blockIdx.y * FEAT;
    const int tid = threadIdx.x, wave = tid >> 5, lane = tid & 31;
    const int lr = lane & 15, h = lane >> 4;
    const int part = GATEUP ? (wave >> 2) : 0;  // gate_up: 0 = gate rows, 1 = up rows
    const int fw = GATEUP ? (wave & 3) : wave;  // 16-feature slice of the tile
    const float g0 = gscales[e * 3 + (GATEUP ? 0 : 2)];
    const float g1 = GATEUP ? gscales[e * 3 + 1] : 0.f;
    const int64_t kb2 = Kd / 2, kb16 = Kd / 16;  // bytes of packed codes / block scales per weight row
    const int nkb = Kd / kMwKB;

    // this lane's weight row (feature n of the gate/up or down matrix), pointing at its half of the k block
    const int n = n0 + fw * 16 + lr;
    const int64_t wrow = GATEUP ? ((int64_t)e * 2 + part) * Nd + n : (int64_t)e * Nd + n;
    const uint8_t* wcode = n < Nd ? w + wrow * kb2 + h * 16 : nullptr;
    const uint8_t* wsc = n < Nd ? ws + wrow * kb16 + h * 2 : nullptr;
    auto loadW = [&](int kb) {
        MwW r;
        if (wcode) {
            r.c = *reinterpret_cast<const uint4*>(wcode + (int64_t)kb * 32);
            r.s = *reinterpret_cast<const uint16_t*>(wsc + (int64_t)kb * 4);
        } else {
            r.c = make_uint4(0, 0, 0, 0);
            r.s = 0;
        }
        return r;
    };

    for (int pt = blockIdx.z; pt * PAIRS < cnt; pt += gridDim.z) {
        // this thread's activation rows (pairs xr + 32 * r) and 8-element column chunk of the 64-wide tile
        const int xr = tid >> 3, xch = tid & 7;
        const ik_bf16* xrow[XR];
#pragma unroll
        for (int r = 0; r < XR; ++r) {
            xrow[r] = nullptr;
            const int p_idx = pt * PAIRS + xr + 32 * r;
            if (p_idx < cnt) {
                const int pair = pairs[begin + p_idx];
                xrow[r] = src + (int64_t)(GATEUP ? pair / topk : pair) * Kd;  // token row (gate_up) / own row (down)
            }
        }
        uint4 rx[XR];
        auto loadX = [&](int kb) {
#pragma unroll
            for (int r = 0; r < XR; ++r)
                rx[r] = xrow[r] ? *reinterpret_cast<const uint4*>(xrow[r] + kb * kMwKB + xch * 8) : make_uint4(0, 0, 0, 0);
        };
        auto storeX = [&](int buf) {
#pragma unroll
            for (int r = 0; r < XR; ++r) *reinterpret_cast<uint4*>(&sx[buf][(xr + 32 * r) * kMwLD + xch * 8]) = rx[r];
        };

        ik_mw_v8f acc[J];
#pragma unroll
        for (int j = 0; j < J; ++j)
#pragma unroll
            for (int i = 0; i < 8; ++i) acc[j][i] = 0.f;

        auto compute = [&](const MwW& W, int buf) {
            const uint32_t cw[4] = {W.c.x, W.c.y, W.c.z, W.c.w};
#pragma unroll
            for (int s = 0; s < 4; ++s) {
                const float sc = ik_e4m3_to_f((W.s >> (8 * (s >> 1))) & 0xffu);
                const uint4 af = ik_fp4x8_scaled_bf16(cw[s], sc, lut);
#pragma unroll
                for (int j = 0; j < J; ++j) {
                    const uint4 bf = *reinterpret_cast<const uint4*>(&sx[buf][(j * 16 + lr) * kMwLD + h * 32 + s * 8]);
                    acc[j] = ik_mw_wmma(af, bf, acc[j]);
                }
            }
        };

        __syncthreads();  // the previous pair tile's last reads of sx / sred are done
        MwW W[kMwRing];
        loadX(0);
#pragma unroll
        for (int i = 0; i < kMwRing; ++i)
            if (i < nkb) W[i] = loadW(i);
        storeX(0);
        if (nkb > 1) loadX(1);
        __syncthreads();
        for (int kb0 = 0; kb0 < nkb; kb0 += kMwRing) {
#pragma unroll
            for (int i = 0; i < kMwRing; ++i) {
                const int kb = kb0 + i;
                if (kb >= nkb) break;
                compute(W[i], kb & 1);
                if (kb + kMwRing < nkb) W[i] = loadW(kb + kMwRing);
                if (kb + 1 < nkb) {
                    storeX((kb + 1) & 1);
                    if (kb + 2 < nkb) loadX(kb + 2);
                }
                __syncthreads();
            }
        }

        if (GATEUP) {
            if (part == 1) {
#pragma unroll
                for (int j = 0; j < J; ++j)
#pragma unroll
                    for (int i = 0; i < 8; ++i) sred[(fw * 32 + lane) * (J * 8) + j * 8 + i] = acc[j][i];
            }
            __syncthreads();
            if (part == 1) continue;
        }
        const int f0 = n0 + fw * 16 + h * 8;
#pragma unroll
        for (int j = 0; j < J; ++j) {
            const int pi = pt * PAIRS + j * 16 + lr;
            if (pi >= cnt || f0 >= Nd) continue;
            const int pair = pairs[begin + pi];
            float v[8];
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                if (GATEUP) {
                    const float g = ik_round_bf16(acc[j][i] * g0);
                    const float u = ik_round_bf16(sred[(fw * 32 + lane) * (J * 8) + j * 8 + i] * g1);
                    const float a = ik_round_bf16(g / (1.f + expf(-g)));
                    v[i] = a * u;
                } else {
                    v[i] = acc[j][i] * g0;
                }
            }
            *reinterpret_cast<uint4*>(dst + (int64_t)pair * Nd + f0) = ik_pack8(v);
        }
    }
}

/* moe_wmma_kernel_v: moe_wmma_kernel with variant bits V (INTARSIA_HIP_MOE_MW, moe.hip; V = 0 runs moe_wmma_kernel
 * itself). Every variant runs the same float operations in the same order for each output, so all of them give the same
 * bits (tests/moe_pairs.rs):
 *   kMwSkip  a pair tile's 16-pair B fragments that hold no pair (the last tile of an expert) skip their LDS reads and
 *            WMMAs: the tile runs ceil(remaining pairs / 16) fragments instead of J (the skipped accumulators stay 0 and
 *            are never stored);
 *   kMwOcc   the up accumulators reuse the activation tiles' LDS (free after the k loop; the k loop's last barrier orders
 *            it) and the kernel is compiled for 8 waves per SIMD (<= 192 VGPRs), so a WGP can hold 4 gate_up blocks
 *            instead of 3.
 * At the 35B's 4408-token prefill (docs/results/2026-10-19-hip-round6.md, same-window profiles) both bits together take
 * gate_up<64> from 105.1 to 96.6 ms and down<64> from 53.3 to 51.9 ms. The 32-pair tile keeps moe_wmma_kernel (its variants
 * were not A/B'd on their own; it is the short tail chunk). A byte-indexed fp4 table (one LDS read per two weights) made
 * gate_up<64> slower (116.7 ms with kMwSkip) and was dropped. */
constexpr int kMwSkip = 1, kMwOcc = 4;

template <bool GATEUP, int PAIRS, int V>
__global__ void __launch_bounds__(kMwThreads) __attribute__((amdgpu_waves_per_eu((V & kMwOcc) ? 8 : 1)))
moe_wmma_kernel_v(const ik_bf16* __restrict__ src, const uint8_t* __restrict__ w, const uint8_t* __restrict__ ws,
                  const float* __restrict__ gscales, const int* __restrict__ pairs, const int* __restrict__ offsets,
                  const int* __restrict__ active, ik_bf16* __restrict__ dst, int E, int Kd, int Nd, int topk) {
    constexpr int FEAT = GATEUP ? kMwFeatGateUp : kMwFeatDown;
    constexpr int J = PAIRS / 16;   // 16-pair B fragments per weight fragment (each dequantised A fragment feeds J WMMAs)
    constexpr int XR = PAIRS / 32;  // activation rows each thread stages
    constexpr bool SKIP = (V & kMwSkip) != 0, ALIAS = GATEUP && (V & kMwOcc) != 0;
    static_assert(PAIRS % 32 == 0, "pair tile");
    constexpr int kRed = 4 * 32 * J * 8;  // up accumulators of one tile (floats)
    static_assert(2 * PAIRS * kMwLD * 2 >= kRed * 4, "the up accumulators fit in the activation tiles");
    __shared__ __align__(16) uint16_t sx[2][PAIRS * kMwLD];
    __shared__ __align__(16) float sred_own[(GATEUP && !ALIAS) ? kRed : 1];  // up accumulators -> gate waves
    float* const sred = ALIAS ? reinterpret_cast<float*>(&sx[0][0]) : sred_own;
    IK_FP4_LUT(lut);

    const int slot = blockIdx.x;
    if (slot >= active[E]) return;  // block-uniform
    const int e = active[slot];
    const int begin = offsets[e], cnt = offsets[e + 1] - begin;
    const int n0 = blockIdx.y * FEAT;
    const int tid = threadIdx.x, wave = tid >> 5, lane = tid & 31;
    const int lr = lane & 15, h = lane >> 4;
    const int part = GATEUP ? (wave >> 2) : 0;  // gate_up: 0 = gate rows, 1 = up rows
    const int fw = GATEUP ? (wave & 3) : wave;  // 16-feature slice of the tile
    const float g0 = gscales[e * 3 + (GATEUP ? 0 : 2)];
    const float g1 = GATEUP ? gscales[e * 3 + 1] : 0.f;
    const int64_t kb2 = Kd / 2, kb16 = Kd / 16;  // bytes of packed codes / block scales per weight row
    const int nkb = Kd / kMwKB;

    // this lane's weight row (feature n of the gate/up or down matrix), pointing at its half of the k block
    const int n = n0 + fw * 16 + lr;
    const int64_t wrow = GATEUP ? ((int64_t)e * 2 + part) * Nd + n : (int64_t)e * Nd + n;
    const uint8_t* wcode = n < Nd ? w + wrow * kb2 + h * 16 : nullptr;
    const uint8_t* wsc = n < Nd ? ws + wrow * kb16 + h * 2 : nullptr;
    auto loadW = [&](int kb) {
        MwW r;
        if (wcode) {
            r.c = *reinterpret_cast<const uint4*>(wcode + (int64_t)kb * 32);
            r.s = *reinterpret_cast<const uint16_t*>(wsc + (int64_t)kb * 4);
        } else {
            r.c = make_uint4(0, 0, 0, 0);
            r.s = 0;
        }
        return r;
    };

    for (int pt = blockIdx.z; pt * PAIRS < cnt; pt += gridDim.z) {
        // this thread's activation rows (pairs xr + 32 * r) and 8-element column chunk of the 64-wide tile
        const int xr = tid >> 3, xch = tid & 7;
        const ik_bf16* xrow[XR];
#pragma unroll
        for (int r = 0; r < XR; ++r) {
            xrow[r] = nullptr;
            const int p_idx = pt * PAIRS + xr + 32 * r;
            if (p_idx < cnt) {
                const int pair = pairs[begin + p_idx];
                xrow[r] = src + (int64_t)(GATEUP ? pair / topk : pair) * Kd;  // token row (gate_up) / own row (down)
            }
        }
        uint4 rx[XR];
        auto loadX = [&](int kb) {
#pragma unroll
            for (int r = 0; r < XR; ++r)
                rx[r] = xrow[r] ? *reinterpret_cast<const uint4*>(xrow[r] + kb * kMwKB + xch * 8) : make_uint4(0, 0, 0, 0);
        };
        auto storeX = [&](int buf) {
#pragma unroll
            for (int r = 0; r < XR; ++r) *reinterpret_cast<uint4*>(&sx[buf][(xr + 32 * r) * kMwLD + xch * 8]) = rx[r];
        };

        ik_mw_v8f acc[J];
#pragma unroll
        for (int j = 0; j < J; ++j)
#pragma unroll
            for (int i = 0; i < 8; ++i) acc[j][i] = 0.f;

        // the k loop over JN of the J fragments (JN < J only for a tile whose last fragments hold no pair, kMwSkip)
        auto kloop = [&](auto jn_c) {
            constexpr int JN = decltype(jn_c)::value;
            auto compute = [&](const MwW& W, int buf) {
                const uint32_t cw[4] = {W.c.x, W.c.y, W.c.z, W.c.w};
#pragma unroll
                for (int s = 0; s < 4; ++s) {
                    const float sc = ik_e4m3_to_f((W.s >> (8 * (s >> 1))) & 0xffu);
                    const uint4 af = ik_fp4x8_scaled_bf16(cw[s], sc, lut);
#pragma unroll
                    for (int j = 0; j < JN; ++j) {
                        const uint4 bf = *reinterpret_cast<const uint4*>(&sx[buf][(j * 16 + lr) * kMwLD + h * 32 + s * 8]);
                        acc[j] = ik_mw_wmma(af, bf, acc[j]);
                    }
                }
            };

            __syncthreads();  // the previous pair tile's last reads of sx / sred are done
            MwW W[kMwRing];
            loadX(0);
#pragma unroll
            for (int i = 0; i < kMwRing; ++i)
                if (i < nkb) W[i] = loadW(i);
            storeX(0);
            if (nkb > 1) loadX(1);
            __syncthreads();
            for (int kb0 = 0; kb0 < nkb; kb0 += kMwRing) {
#pragma unroll
                for (int i = 0; i < kMwRing; ++i) {
                    const int kb = kb0 + i;
                    if (kb >= nkb) break;
                    compute(W[i], kb & 1);
                    if (kb + kMwRing < nkb) W[i] = loadW(kb + kMwRing);
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
            else if (J == 4 && jn == 2) kloop(std::integral_constant<int, (J == 4 ? 2 : 1)>{});
            else kloop(std::integral_constant<int, (J == 4 ? 3 : 1)>{});
        } else {
            kloop(std::integral_constant<int, J>{});
        }

        if (GATEUP) {
            if (part == 1) {
#pragma unroll
                for (int j = 0; j < J; ++j)
#pragma unroll
                    for (int i = 0; i < 8; ++i) sred[(fw * 32 + lane) * (J * 8) + j * 8 + i] = acc[j][i];
            }
            __syncthreads();
            if (part == 1) continue;
        }
        const int f0 = n0 + fw * 16 + h * 8;
#pragma unroll
        for (int j = 0; j < J; ++j) {
            const int pi = pt * PAIRS + j * 16 + lr;
            if (pi >= cnt || f0 >= Nd) continue;
            const int pair = pairs[begin + pi];
            float v[8];
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                if (GATEUP) {
                    const float g = ik_round_bf16(acc[j][i] * g0);
                    const float u = ik_round_bf16(sred[(fw * 32 + lane) * (J * 8) + j * 8 + i] * g1);
                    const float a = ik_round_bf16(g / (1.f + expf(-g)));
                    v[i] = a * u;
                } else {
                    v[i] = acc[j][i] * g0;
                }
            }
            *reinterpret_cast<uint4*>(dst + (int64_t)pair * Nd + f0) = ik_pack8(v);
        }
    }
}

#endif /* IK_MOE_WMMA_H */
