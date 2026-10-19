/* Round 7 (docs/results/2026-10-19-hip-round7.md): the grouped bf16-WMMA expert projections of moe_wmma.h with more variant
 * bits (INTARSIA_HIP_MOE_MW, moe.hip). moe_wmma_kernel_r7 is moe_wmma_kernel_v (moe_wmma.h, left untouched as the bitwise
 * reference and the rollback) plus:
 *
 *   kMwCvt  the e2m1 weights are decoded without the shared-memory table: each nibble's sign and 3 magnitude bits are moved
 *           into an e4m3 byte, (s << 7) | (mag << 2), whose value is exactly the e2m1 value * 2^-6 (mag 1 = 0x04 is the e4m3
 *           subnormal 2^-7 = 0.5 * 2^-6; mag 2..7 are normals 2^(e-7) * (1 + m/2) = 2^(e-1) * (1 + m/2) * 2^-6; the largest
 *           byte is 0x9c, never the NaN 0x7f / 0xff), and the hardware fp8 conversion (v_cvt_pk_f32_fp8, two per
 *           instruction) turns them into f32. The block scale is folded with the 2^6: sc64 = e4m3(scale) * 64 (exact). Then
 *           (value * 2^-6) * sc64 is the exact product value * sc, the same f32 bits as the table path's lut[c] * sc (both are
 *           the exact, representable product, sign of zero included: the sign rule of the multiply is the same), so the bf16
 *           fragments and every WMMA are bit for bit those of moe_wmma_kernel_v. Per 8 weights: 7 integer ops, 4 conversions,
 *           8 multiplies and 4 permutes, no LDS read (the table path: 8 LDS reads and ~2 index ops per weight).
 *   kMwTwo  down only: each wave holds two 16-row A fragments (256 features per block), so every B (activation) fragment
 *           read from LDS feeds two WMMAs. Every output still accumulates the same WMMAs over k in the same order. (A gate_up
 *           with the gate and the up fragment of the same features in one wave was built and measured slower, round-7 doc.)
 *   kMwDeep six weight blocks in flight per lane instead of three (kMwRing).
 *
 * kMwSkip and kMwOcc mean what they mean in moe_wmma_kernel_v; kMwOcc does not cap the two-fragment down's registers.
 */
#ifndef IK_MOE_WMMA_R7_H
#define IK_MOE_WMMA_R7_H

#include "moe_wmma.h"

constexpr int kMwCvt = 8, kMwTwo = 16, kMwDeep = 32;

/* kMwTwo is a down-projection bit (gate_up ignores it) */
constexpr bool mw_r7_two(bool gateup, int v) { return !gateup && (v & kMwTwo) != 0; }
/* features per block of moe_wmma_kernel_r7 (grid y of the launch) */
constexpr int mw_r7_feat(bool gateup, int v) {
    return gateup ? kMwFeatGateUp : (mw_r7_two(gateup, v) ? 2 * kMwFeatDown : kMwFeatDown);
}
/* waves per SIMD the kernel is compiled for: kMwOcc's 8, except for the two-fragment down (it spills at 192 VGPRs) */
constexpr int mw_r7_waves(bool gateup, int v) { return ((v & kMwOcc) && !mw_r7_two(gateup, v)) ? 8 : 1; }

/* e2m1 codes of one 32-bit word (element j = nibble j) -> the e4m3 bytes of value * 2^-6, even elements (0, 2, 4, 6) in
 * `ev`, odd ones (1, 3, 5, 7) in `od`, byte i holding element 2i (+1) */
IK_DEVI void ik_fp4x8_to_e4m3(uint32_t c, uint32_t& ev, uint32_t& od) {
    ev = ((c << 2) & 0x1C1C1C1Cu) | ((c << 4) & 0x80808080u);
    od = ((c >> 2) & 0x1C1C1C1Cu) | (c & 0x80808080u);
}

/* 8 fp4 codes times a block scale -> 8 bf16, bitwise ik_fp4x8_scaled_bf16(c, sc, lut) for sc64 = sc * 64 (see the top). */
typedef float ik_mw7_v2f __attribute__((ext_vector_type(2)));
#if __has_builtin(__builtin_amdgcn_cvt_pk_f32_fp8)
IK_DEVI uint4 ik_fp4x8_scaled_bf16_cvt(uint32_t c, float sc64) {
    uint32_t ev, od;
    ik_fp4x8_to_e4m3(c, ev, od);
    const ik_mw7_v2f e02 = __builtin_amdgcn_cvt_pk_f32_fp8((int)ev, false), e46 = __builtin_amdgcn_cvt_pk_f32_fp8((int)ev, true);
    const ik_mw7_v2f o13 = __builtin_amdgcn_cvt_pk_f32_fp8((int)od, false), o57 = __builtin_amdgcn_cvt_pk_f32_fp8((int)od, true);
    uint4 r;
    r.x = __builtin_amdgcn_perm(__float_as_uint(o13[0] * sc64), __float_as_uint(e02[0] * sc64), 0x07060302u);
    r.y = __builtin_amdgcn_perm(__float_as_uint(o13[1] * sc64), __float_as_uint(e02[1] * sc64), 0x07060302u);
    r.z = __builtin_amdgcn_perm(__float_as_uint(o57[0] * sc64), __float_as_uint(e46[0] * sc64), 0x07060302u);
    r.w = __builtin_amdgcn_perm(__float_as_uint(o57[1] * sc64), __float_as_uint(e46[1] * sc64), 0x07060302u);
    return r;
}
#else  // no fp8 conversion builtin: the software e4m3 decode of the same bytes (exact, so the same bits)
IK_DEVI uint4 ik_fp4x8_scaled_bf16_cvt(uint32_t c, float sc64) {
    uint32_t ev, od;
    ik_fp4x8_to_e4m3(c, ev, od);
    uint32_t u[8];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        u[2 * i] = __float_as_uint(ik_e4m3_to_f((ev >> (8 * i)) & 0xffu) * sc64);
        u[2 * i + 1] = __float_as_uint(ik_e4m3_to_f((od >> (8 * i)) & 0xffu) * sc64);
    }
    uint4 r;
    r.x = __builtin_amdgcn_perm(u[1], u[0], 0x07060302u);
    r.y = __builtin_amdgcn_perm(u[3], u[2], 0x07060302u);
    r.z = __builtin_amdgcn_perm(u[5], u[4], 0x07060302u);
    r.w = __builtin_amdgcn_perm(u[7], u[6], 0x07060302u);
    return r;
}
#endif

template <bool GATEUP, int PAIRS, int V>
__global__ void __launch_bounds__(kMwThreads) __attribute__((amdgpu_waves_per_eu(mw_r7_waves(GATEUP, V))))
moe_wmma_kernel_r7(const ik_bf16* __restrict__ src, const uint8_t* __restrict__ w, const uint8_t* __restrict__ ws,
                   const float* __restrict__ gscales, const int* __restrict__ pairs, const int* __restrict__ offsets,
                   const int* __restrict__ active, ik_bf16* __restrict__ dst, int E, int Kd, int Nd, int topk) {
    constexpr bool SKIP = (V & kMwSkip) != 0, CVT = (V & kMwCvt) != 0, TWO = mw_r7_two(GATEUP, V);
    constexpr int RING = (V & kMwDeep) ? 6 : kMwRing;  // weight blocks in flight per lane
    constexpr bool ALIAS = GATEUP && (V & kMwOcc) != 0;  // up accumulators in the activation tiles' LDS (kMwOcc)
    constexpr int RF = TWO ? 2 : 1;                    // A fragments per wave
    constexpr int FEAT = mw_r7_feat(GATEUP, V);
    constexpr int J = PAIRS / 16;   // 16-pair B fragments per weight fragment (each dequantised A fragment feeds J WMMAs)
    constexpr int XR = PAIRS / 32;  // activation rows each thread stages
    static_assert(PAIRS % 32 == 0, "pair tile");
    constexpr int kRed = 4 * 32 * J * 8;  // up accumulators of one tile (floats)
    static_assert(2 * PAIRS * kMwLD * 2 >= kRed * 4, "the up accumulators fit in the activation tiles");
    __shared__ __align__(16) uint16_t sx[2][PAIRS * kMwLD];
    __shared__ __align__(16) float sred_own[(GATEUP && !ALIAS) ? kRed : 1];  // up accumulators -> gate waves
    float* const sred = ALIAS ? reinterpret_cast<float*>(&sx[0][0]) : sred_own;
    __shared__ float lut[16];
    if constexpr (!CVT) {  // IK_FP4_LUT, before any early return (it has a barrier)
        if (threadIdx.x < 16) lut[threadIdx.x] = ik_fp4_value(threadIdx.x);
        __syncthreads();
    }

    const int slot = blockIdx.x;
    if (slot >= active[E]) return;  // block-uniform
    const int e = active[slot];
    const int begin = offsets[e], cnt = offsets[e + 1] - begin;
    const int n0 = blockIdx.y * FEAT;
    const int tid = threadIdx.x, wave = tid >> 5, lane = tid & 31;
    const int lr = lane & 15, h = lane >> 4;
    const int part = GATEUP ? (wave >> 2) : 0;  // gate_up: 0 = gate rows, 1 = up rows
    // first feature of this wave's fragment f (16 features)
    auto fbase = [&](int f) {
        if constexpr (GATEUP) return n0 + (wave & 3) * 16;
        else return n0 + (wave * RF + f) * 16;
    };
    const float g0 = gscales[e * 3 + (GATEUP ? 0 : 2)];
    const float g1 = GATEUP ? gscales[e * 3 + 1] : 0.f;
    const int64_t kb2 = Kd / 2, kb16 = Kd / 16;  // bytes of packed codes / block scales per weight row
    const int nkb = Kd / kMwKB;

    // this lane's weight rows (one per fragment), pointing at its half of the k block
    const uint8_t* wcode[RF];
    const uint8_t* wsc[RF];
#pragma unroll
    for (int f = 0; f < RF; ++f) {
        const int n = fbase(f) + lr;
        const int64_t wrow = GATEUP ? ((int64_t)e * 2 + part) * Nd + n : (int64_t)e * Nd + n;
        wcode[f] = n < Nd ? w + wrow * kb2 + h * 16 : nullptr;
        wsc[f] = n < Nd ? ws + wrow * kb16 + h * 2 : nullptr;
    }
    auto loadW = [&](int f, int kb) {
        MwW r;
        if (wcode[f]) {
            r.c = *reinterpret_cast<const uint4*>(wcode[f] + (int64_t)kb * 32);
            r.s = *reinterpret_cast<const uint16_t*>(wsc[f] + (int64_t)kb * 4);
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

        ik_mw_v8f acc[RF][J];
#pragma unroll
        for (int f = 0; f < RF; ++f)
#pragma unroll
            for (int j = 0; j < J; ++j)
#pragma unroll
                for (int i = 0; i < 8; ++i) acc[f][j][i] = 0.f;

        // the k loop over JN of the J fragments (JN < J only for a tile whose last fragments hold no pair, kMwSkip)
        auto kloop = [&](auto jn_c) {
            constexpr int JN = decltype(jn_c)::value;
            auto compute = [&](const MwW* Wi, int buf) {  // Wi[f]: fragment f's weights of this k block
#pragma unroll
                for (int s = 0; s < 4; ++s) {
                    uint4 af[RF];
#pragma unroll
                    for (int f = 0; f < RF; ++f) {
                        const uint32_t cw[4] = {Wi[f].c.x, Wi[f].c.y, Wi[f].c.z, Wi[f].c.w};
                        const float sc = ik_e4m3_to_f((Wi[f].s >> (8 * (s >> 1))) & 0xffu);
                        if constexpr (CVT) af[f] = ik_fp4x8_scaled_bf16_cvt(cw[s], sc * 64.f);
                        else af[f] = ik_fp4x8_scaled_bf16(cw[s], sc, lut);
                    }
#pragma unroll
                    for (int j = 0; j < JN; ++j) {
                        const uint4 bf = *reinterpret_cast<const uint4*>(&sx[buf][(j * 16 + lr) * kMwLD + h * 32 + s * 8]);
#pragma unroll
                        for (int f = 0; f < RF; ++f) acc[f][j] = ik_mw_wmma(af[f], bf, acc[f][j]);
                    }
                }
            };

            __syncthreads();  // the previous pair tile's last reads of sx / sred are done
            MwW W[RING][RF];
            loadX(0);
#pragma unroll
            for (int i = 0; i < RING; ++i)
                if (i < nkb)
#pragma unroll
                    for (int f = 0; f < RF; ++f) W[i][f] = loadW(f, i);
            storeX(0);
            if (nkb > 1) loadX(1);
            __syncthreads();
            for (int kb0 = 0; kb0 < nkb; kb0 += RING) {
#pragma unroll
                for (int i = 0; i < RING; ++i) {
                    const int kb = kb0 + i;
                    if (kb >= nkb) break;
                    compute(W[i], kb & 1);
                    if (kb + RING < nkb)
#pragma unroll
                        for (int f = 0; f < RF; ++f) W[i][f] = loadW(f, kb + RING);
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

        if constexpr (GATEUP) {
            if (part == 1) {
#pragma unroll
                for (int j = 0; j < J; ++j)
#pragma unroll
                    for (int i = 0; i < 8; ++i) sred[((wave & 3) * 32 + lane) * (J * 8) + j * 8 + i] = acc[0][j][i];
            }
            __syncthreads();
            if (part == 1) continue;
        }
#pragma unroll
        for (int f = 0; f < RF; ++f) {
            const int f0 = fbase(f) + h * 8;
#pragma unroll
            for (int j = 0; j < J; ++j) {
                const int pi = pt * PAIRS + j * 16 + lr;
                if (pi >= cnt || f0 >= Nd) continue;
                const int pair = pairs[begin + pi];
                float v[8];
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    if constexpr (GATEUP) {
                        const float g = ik_round_bf16(acc[0][j][i] * g0);
                        const float u = ik_round_bf16(sred[((wave & 3) * 32 + lane) * (J * 8) + j * 8 + i] * g1);
                        const float a = ik_round_bf16(g / (1.f + expf(-g)));
                        v[i] = a * u;
                    } else {
                        v[i] = acc[f][j][i] * g0;
                    }
                }
                *reinterpret_cast<uint4*>(dst + (int64_t)pair * Nd + f0) = ik_pack8(v);
            }
        }
    }
}

#endif /* IK_MOE_WMMA_R7_H */
