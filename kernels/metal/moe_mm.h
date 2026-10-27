// Helpers shared by the grouped (prefill) MoE kernels: moe.metal (simdgroup_matrix, metal3.1) and moe_nax.metal
// (matmul2d, metal4.0). Moved out of moe.metal unchanged.
#pragma once
#include "common.h"

inline float moe_rb(float v) { return bf2f(f2bf(v)); }

// tile `ty` of `tm` rows -> (expert, first slot, rows); false past the last tile
inline bool moe_tile(device const int* off, device const int* tiles, uint E, uint tm, uint ty, thread uint& e,
                     thread uint& s0, thread uint& rows) {
    if ((int)ty >= tiles[E]) return false;
    uint lo = 0u, hi = E;  // largest e with tiles[e] <= ty (tiles[e + 1] > ty then)
    while (hi - lo > 1u) {
        const uint mid = (lo + hi) / 2u;
        if ((uint)tiles[mid] <= ty) lo = mid;
        else hi = mid;
    }
    e = lo;
    s0 = (uint)off[e] + (ty - (uint)tiles[e]) * tm;
    rows = min(tm, (uint)off[e + 1] - s0);
    return true;
}

// NVFP4 -> bf16 as kernels/metal/nvfp4.metal: two codes (nibbles at bits 0..3, 16..19) -> value / 2^14 via half bits,
// an e4m3 scale byte -> value / 256 (NaN code -> NaN); the 2^22 is folded back into the scale, the product is exact
inline float2 moe_x2(uint t) {
    const uint h = ((t & 0x00070007u) << 9) | ((t & 0x00080008u) << 12);
    return float2(as_type<half2>(h));
}
inline float moe_sc(uint b) {
    if ((b & 0x7Fu) == 0x7Fu) return as_type<float>(0x7FC00000u);
    return float(as_type<half>((ushort)(((b & 0x7Fu) << 7) | ((b & 0x80u) << 8)))) * 4194304.0f;
}
inline uint moe_pack(float lo, float hi) { return (as_type<uint>(lo) >> 16) | (as_type<uint>(hi) & 0xFFFF0000u); }

// one 16-element block (2 words, scale s from moe_sc) -> 16 bf16 at dst (16-byte aligned)
inline void moe_blk_bf16(uint2 v, float s, threadgroup bfloat* dst) {
    uint o[8];
    for (int i = 0; i < 2; ++i) {
        float2 c[4];  // c[k]: elements k and k + 4 of the word
        for (int k = 0; k < 4; ++k) c[k] = moe_x2((v[i] >> (4 * k)) & 0x000F000Fu) * s;
        o[4 * i] = moe_pack(c[0].x, c[1].x);
        o[4 * i + 1] = moe_pack(c[2].x, c[3].x);
        o[4 * i + 2] = moe_pack(c[0].y, c[1].y);
        o[4 * i + 3] = moe_pack(c[2].y, c[3].y);
    }
    threadgroup uint4* d = (threadgroup uint4*)dst;
    d[0] = uint4(o[0], o[1], o[2], o[3]);
    d[1] = uint4(o[4], o[5], o[6], o[7]);
}

constant constexpr uint MOE_WR = 64;            // weight rows per tile: 32 gate + 32 up columns (gu), 64 columns (dn)
constant constexpr uint MOE_LC = MOE_WR + 4;    // floats per epilogue row

// raw weight words and scale bytes of blocks bq .. bq + NB - 1 of the K step at k0 (zero past the matrix)
template <uint NB>
inline void moe_fetch(device const uchar* wp, device const uchar* sp, bool bok, uint bq, uint k0, uint KD,
                      thread uint2* wv, thread uint* sv) {
    for (uint q = 0; q < NB; ++q) {
        const uint gk = k0 + (bq + q) * 16u;
        const bool ok = bok && gk < KD;
        wv[q] = ok ? uint2(*(device const packed_uint2*)(wp + gk / 2)) : uint2(0u);
        sv[q] = ok ? (uint)sp[gk / 16] : 0u;
    }
}
