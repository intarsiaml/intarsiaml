// Lossless 12-bit bf16 weights (P12, lane ink-pack12): GEMVs for 1..=16 activation rows that read the packed codes and are
// BITWISE their bf16 twins. Format: crates/intarsia-core/src/quant/p12.rs (module doc); launchers
// crates/intarsia-backend-metal/src/p12.rs.
//
// Every kernel decodes each weight element to its exact bf16 bits in registers (the in-place 12-bit code, then the row's
// escapes patched in place), so the uint4 it hands to dot8 is the uint4 the bf16 kernel loads from the bf16 weight. The
// lane -> chunk mapping, the fma chain (dot8), the per-lane chunk order and the reduction tree are those of the mirrored
// kernel, so the outputs are the same bits:
//
//   p12_in_proj_gemv<R,P,U,RB>  == ink_in_proj_gemv<R,P,8,*,RB>    (InkInProj, the s8 family: 256 lanes per row)
//   p12_gemv_rt<R,P,U,RB>       == gemv_bf16_rt_u2s8(_mr)          (Matmul, DEFAULT_GEMV: 256 lanes per row)
//   p12_gemv_f32o<R,P,U,RB>     == gemv_f32o_v(_mr)                (Logits: 32 lanes per row, 4 simdgroups per threadgroup)
//
// UNROLL (U) and P only change how many chunks are in flight and how many weight rows share an x load: lane L still adds
// chunks L, L + lanes, L + 2 lanes, ... in that order, so neither changes a bit. R only adds accumulators.

#include <metal_stdlib>
using namespace metal;

inline float pk_lo(uint u) { return as_type<float>(u << 16); }
inline float pk_hi(uint u) { return as_type<float>(u & 0xffff0000u); }

// The fma chain of gemv_multirow.metal dot8 / inkling2_mix.metal ip_dot8.
inline float pk_dot8(uint4 w, uint4 x) {
    float a = pk_lo(w.x) * pk_lo(x.x);
    a = fma(pk_hi(w.x), pk_hi(x.x), a);
    a = fma(pk_lo(w.y), pk_lo(x.y), a);
    a = fma(pk_hi(w.y), pk_hi(x.y), a);
    a = fma(pk_lo(w.z), pk_lo(x.z), a);
    a = fma(pk_hi(w.z), pk_hi(x.z), a);
    a = fma(pk_lo(w.w), pk_lo(x.w), a);
    a = fma(pk_hi(w.w), pk_hi(x.w), a);
    return a;
}

inline ushort pk_bf16(float v) { return as_type<ushort>(bfloat(v)); }

// One packed weight row: plane A (uint2 per 8-element chunk), plane B (uint per chunk), the base term, its escapes.
struct P12Row {
    device const uint2* a;
    device const uint* b;
    uint bb;                  // base * 0x00800080
    device const uint* side;  // this row's escapes (col << 16 | bf16)
    uint ns;                  // how many
};

// Row `r` of a P12 weight with `nrows` rows of K = 8 * k8 elements (codes row stride 12 * k8 bytes).
inline P12Row p12_row(device const uchar* codes, device const uint* aux, uint nrows, uint r, uint k8) {
    P12Row w;
    device const uchar* base = codes + (ulong)r * (12u * k8);
    w.a = (device const uint2*)base;
    w.b = (device const uint*)(base + 8u * k8);
    const uint m0 = aux[r], m1 = aux[r + 1];
    w.bb = (m0 & 0xFFu) * 0x00800080u;
    w.side = aux + nrows + 1u + (m0 >> 8);
    w.ns = (m1 >> 8) - (m0 >> 8);
    return w;
}

// Chunk c of row w as the bf16 pair words a bf16 weight stores (escapes not yet patched).
inline uint4 p12_chunk(thread const P12Row& w, uint c) {
    const uint2 a = w.a[c];
    const uint b = w.b[c];
    const uint q3 = ((a.x >> 11) & 0x000F000Fu) | ((a.y >> 7) & 0x00F000F0u) | ((b >> 3) & 0x07000700u) | ((b << 1) & 0x80008000u);
    return uint4((a.x & 0x87FF87FFu) + w.bb, (a.y & 0x87FF87FFu) + w.bb, (b & 0x87FF87FFu) + w.bb, q3 + w.bb);
}

// Overwrite the escapes of chunk c (rare: a fifth of the rows have any, 2-9 at most per row on the checkpoint).
inline void p12_patch(thread const P12Row& w, uint c, thread uint4& v) {
    for (uint e = 0; e < w.ns; ++e) {
        const uint ent = w.side[e];
        const uint col = ent >> 16;
        if ((col >> 3) == c) {
            const uint j = (col >> 1) & 3u, sh = (col & 1u) * 16u;
            const uint m = 0xFFFFu << sh;
            v[j] = (v[j] & ~m) | ((ent & 0xFFFFu) << sh);
        }
    }
}

inline uint4 p12_load(thread const P12Row& w, uint c) {
    uint4 v = p12_chunk(w, c);
    if (w.ns != 0) p12_patch(w, c, v);
    return v;
}

// ================================================================================================================
// The 256-lane row body (SG = 8): rt_rows16<R,P,UNROLL,8> of gemv_multirow.metal == ip_rows<RB,P,U,8> of inkling2_mix.metal,
// with the weight loads replaced by p12_load. P weight rows (global rows rbase + min(row0 + q, nvalid - 1)) x RB activation
// rows (X, row stride k8). `part` holds P * RB * 8 floats; totals land in out[] on every thread.
// ================================================================================================================
template <int RB, int P, int U>
inline void p12_rows256(device const uchar* codes, device const uint* aux, uint nrows, uint rbase, uint row0, uint nvalid,
                        device const uint4* X, uint k8, uint sg, uint lane, threadgroup float* part, thread float* out) {
    constexpr uint SG = 8;
    constexpr uint lanes = 32 * SG;
    constexpr uint block = lanes * U;
    P12Row wr[P];
    for (int q = 0; q < P; ++q) wr[q] = p12_row(codes, aux, nrows, rbase + min(row0 + q, nvalid - 1), k8);
    const uint nblk = k8 / block;
    uint i = sg * 32 + lane;
    float acc[P * RB];
    for (int j = 0; j < P * RB; ++j) acc[j] = 0.0f;
    for (uint b = 0; b < nblk; ++b, i += block) {
        uint4 wv[P][U];
        for (int q = 0; q < P; ++q)
            for (int u = 0; u < U; ++u) wv[q][u] = p12_load(wr[q], i + lanes * u);
        for (int r = 0; r < RB; ++r) {
            uint4 xv[U];
            for (int u = 0; u < U; ++u) xv[u] = X[(ulong)r * k8 + i + lanes * u];
            for (int q = 0; q < P; ++q)
                for (int u = 0; u < U; ++u) acc[q * RB + r] += pk_dot8(wv[q][u], xv[u]);
        }
    }
    for (i = nblk * block + sg * 32 + lane; i < k8; i += lanes) {
        uint4 wv[P];
        for (int q = 0; q < P; ++q) wv[q] = p12_load(wr[q], i);
        for (int r = 0; r < RB; ++r) {
            const uint4 xv = X[(ulong)r * k8 + i];
            for (int q = 0; q < P; ++q) acc[q * RB + r] += pk_dot8(wv[q], xv);
        }
    }
    for (int j = 0; j < P * RB; ++j) {
        const float s = simd_sum(acc[j]);
        if (lane == 0) part[j * SG + sg] = s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int j = 0; j < P * RB; ++j) {
        float t = part[j * SG];
        for (uint q = 1; q < SG; ++q) t += part[j * SG + q];
        out[j] = t;
    }
}

// ---- InkInProj over P12 codes (mirrors ink_in_proj_gemv, the s8 family) -------------------------------------------------
struct P12ProjP {
    uint k;              // D (K % 16 == 0)
    uint n0, n1, n2;     // rows per segment (pq, pk, pv)
    uint xo1, xo2;       // element offset of the x block of segments 1 and 2 (min(j, S-1) * T * D)
    uint ys0, ys1, ys2;  // leading bf16 elements to skip in Y0 / Y1 / Y2 (4-byte aligned binding)
    uint g0, g1;         // threadgroups of segment 0 and 1 (ceil(n_j / P))
    uint gtot;           // all virtual threadgroups (g0 + g1 + g2); the grid may be smaller and loop over them
};

// bind: 0 codes [n0+n1+n2, 3K/2] (8-byte aligned), 1 X [S, T, K] (16-byte aligned), 2 Y0 [R, n0], 3 Y1 [R, n1], 4 Y2 [R, n2],
// 5 aux, 6 P12ProjP. Grid: g0 + g1 + ceil(n2 / P) threadgroups (or fewer, grid-stride) of 256 threads.
template <int R, int P, int U, int RB>
kernel void p12_in_proj_gemv(device const uchar* codes [[buffer(0)]],
                             device const uint4* X [[buffer(1)]],
                             device ushort* Y0 [[buffer(2)]],
                             device ushort* Y1 [[buffer(3)]],
                             device ushort* Y2 [[buffer(4)]],
                             device const uint* aux [[buffer(5)]],
                             constant P12ProjP& p [[buffer(6)]],
                             uint tg [[threadgroup_position_in_grid]],
                             uint ngrid [[threadgroups_per_grid]],
                             uint sg [[simdgroup_index_in_threadgroup]],
                             uint lane [[thread_index_in_simdgroup]]) {
    static_assert(R % RB == 0, "R must be a multiple of RB");
    threadgroup float part[P * R * 8];
    const uint k8 = p.k / 8;
    const uint nrows = p.n0 + p.n1 + p.n2;
    for (uint vt = tg; vt < p.gtot; vt += ngrid) {
        uint lt, n, wrow, xo, ys;
        device ushort* Y;
        if (vt < p.g0) {
            lt = vt; n = p.n0; wrow = 0; xo = 0; Y = Y0; ys = p.ys0;
        } else if (vt < p.g0 + p.g1) {
            lt = vt - p.g0; n = p.n1; wrow = p.n0; xo = p.xo1; Y = Y1; ys = p.ys1;
        } else {
            lt = vt - p.g0 - p.g1; n = p.n2; wrow = p.n0 + p.n1; xo = p.xo2; Y = Y2; ys = p.ys2;
        }
        const uint row0 = lt * P;
        device const uint4* Xs = X + xo / 8;
        float t[P * R];
        for (int rb = 0; rb < R; rb += RB) {
            float tb[P * RB];
            p12_rows256<RB, P, U>(codes, aux, nrows, wrow, row0, n, Xs + (ulong)rb * k8, k8, sg, lane, part + rb * P * 8, tb);
            for (int q = 0; q < P; ++q)
                for (int r = 0; r < RB; ++r) t[q * R + rb + r] = tb[q * RB + r];
        }
        if (sg == 0 && lane == 0) {
            for (int q = 0; q < P; ++q)
                if (row0 + q < n)
                    for (int r = 0; r < R; ++r) Y[ys + (ulong)r * n + row0 + q] = pk_bf16(t[q * R + r]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);  // `part` is rewritten by the next virtual threadgroup
    }
}

// ---- Matmul (mirrors gemv_bf16_rt_u2s8 / gemv_bf16_rt_u2s8_mr) ------------------------------------------------------------
struct P12GemvP {
    uint k;       // K (K % 16 == 0)
    uint n;       // N
    uint y_skip;  // leading output elements to skip (4-byte aligned binding)
    uint pad;
};

// bind: 0 codes [N, 3K/2], 1 X [R, K], 2 Y [R, N] bf16, 3 aux, 4 P12GemvP. Grid: ceil(N / P) threadgroups of 256 threads.
template <int R, int P, int U, int RB>
kernel void p12_gemv_rt(device const uchar* codes [[buffer(0)]],
                        device const uint4* X [[buffer(1)]],
                        device ushort* Y [[buffer(2)]],
                        device const uint* aux [[buffer(3)]],
                        constant P12GemvP& p [[buffer(4)]],
                        uint tg [[threadgroup_position_in_grid]],
                        uint sg [[simdgroup_index_in_threadgroup]],
                        uint lane [[thread_index_in_simdgroup]]) {
    static_assert(R % RB == 0, "R must be a multiple of RB");
    threadgroup float part[P * R * 8];
    const uint k8 = p.k / 8;
    const uint row0 = tg * P;
    float t[P * R];
    for (int rb = 0; rb < R; rb += RB) {
        float tb[P * RB];
        p12_rows256<RB, P, U>(codes, aux, p.n, 0, row0, p.n, X + (ulong)rb * k8, k8, sg, lane, part + rb * P * 8, tb);
        for (int q = 0; q < P; ++q)
            for (int r = 0; r < RB; ++r) t[q * R + rb + r] = tb[q * RB + r];
    }
    if (sg == 0 && lane == 0) {
        for (int q = 0; q < P; ++q)
            if (row0 + q < p.n)
                for (int r = 0; r < R; ++r) Y[p.y_skip + (ulong)r * p.n + row0 + q] = pk_bf16(t[q * R + r]);
    }
}

// ---- Logits, f32 out (mirrors gemv_f32o_v / gemv_f32o_v_mr: one simdgroup per P rows, 4 per threadgroup) -----------------
// bind: 0 codes [N, 3K/2], 1 X [R, K], 2 Y [R, N] f32, 3 aux, 4 P12GemvP. Grid: ceil(N / (4 P)) threadgroups of 128 threads.
template <int R, int P, int U, int RB>
kernel void p12_gemv_f32o(device const uchar* codes [[buffer(0)]],
                          device const uint4* X [[buffer(1)]],
                          device float* Y [[buffer(2)]],
                          device const uint* aux [[buffer(3)]],
                          constant P12GemvP& p [[buffer(4)]],
                          uint tg [[threadgroup_position_in_grid]],
                          uint sg [[simdgroup_index_in_threadgroup]],
                          uint lane [[thread_index_in_simdgroup]]) {
    static_assert(R % RB == 0, "R must be a multiple of RB");
    const uint k8 = p.k / 8;
    const uint row0 = (tg * 4 + sg) * P;
    P12Row wr[P];
    for (int q = 0; q < P; ++q) wr[q] = p12_row(codes, aux, p.n, min(row0 + q, p.n - 1), k8);  // rows past N: valid, not stored
    for (int rb = 0; rb < R; rb += RB) {
        device const uint4* Xb = X + (ulong)rb * k8;
        float acc[P * RB];
        for (int j = 0; j < P * RB; ++j) acc[j] = 0.0f;
        constexpr uint block = 32 * U;
        const uint nblk = k8 / block;
        uint i = lane;
        for (uint b = 0; b < nblk; ++b, i += block) {
            uint4 wv[P][U];
            for (int q = 0; q < P; ++q)
                for (int u = 0; u < U; ++u) wv[q][u] = p12_load(wr[q], i + 32 * u);
            for (int r = 0; r < RB; ++r) {
                uint4 xv[U];
                for (int u = 0; u < U; ++u) xv[u] = Xb[(ulong)r * k8 + i + 32 * u];
                for (int q = 0; q < P; ++q)
                    for (int u = 0; u < U; ++u) acc[q * RB + r] += pk_dot8(wv[q][u], xv[u]);
            }
        }
        for (i = nblk * block + lane; i < k8; i += 32) {
            uint4 wv[P];
            for (int q = 0; q < P; ++q) wv[q] = p12_load(wr[q], i);
            for (int r = 0; r < RB; ++r) {
                const uint4 xv = Xb[(ulong)r * k8 + i];
                for (int q = 0; q < P; ++q) acc[q * RB + r] += pk_dot8(wv[q], xv);
            }
        }
        for (int q = 0; q < P; ++q) {
            for (int r = 0; r < RB; ++r) {
                const float s = simd_sum(acc[q * RB + r]);
                if (lane == 0 && row0 + q < p.n) Y[p.y_skip + (ulong)(rb + r) * p.n + row0 + q] = s;
            }
        }
    }
}

// ---- instances: host name <kernel>_r<R>p<P>u<U>b<RB> (generated; the launcher's table INSTS in p12.rs lists the same) --------
// every row count 1..=16 (RB = R up to 9, then a divisor), P in {1, 2, 4} at 1 row and {1, 2} at 2..=4 rows, three unrolls
#define P12_PROJ_ARGS device const uchar*, device const uint4*, device ushort*, device ushort*, device ushort*, \
    device const uint*, constant P12ProjP&, uint, uint, uint, uint
#define P12_RT_ARGS device const uchar*, device const uint4*, device ushort*, device const uint*, constant P12GemvP&, uint, uint, uint
#define P12_F32_ARGS device const uchar*, device const uint4*, device float*, device const uint*, constant P12GemvP&, uint, uint, uint

template [[host_name("p12_in_proj_gemv_r1p1u1b1")]] kernel void p12_in_proj_gemv<1, 1, 1, 1>(P12_PROJ_ARGS);
template [[host_name("p12_in_proj_gemv_r1p2u1b1")]] kernel void p12_in_proj_gemv<1, 2, 1, 1>(P12_PROJ_ARGS);
template [[host_name("p12_in_proj_gemv_r1p4u1b1")]] kernel void p12_in_proj_gemv<1, 4, 1, 1>(P12_PROJ_ARGS);
template [[host_name("p12_in_proj_gemv_r2p1u1b2")]] kernel void p12_in_proj_gemv<2, 1, 1, 2>(P12_PROJ_ARGS);
template [[host_name("p12_in_proj_gemv_r2p2u1b2")]] kernel void p12_in_proj_gemv<2, 2, 1, 2>(P12_PROJ_ARGS);
template [[host_name("p12_in_proj_gemv_r3p1u1b3")]] kernel void p12_in_proj_gemv<3, 1, 1, 3>(P12_PROJ_ARGS);
template [[host_name("p12_in_proj_gemv_r3p2u1b3")]] kernel void p12_in_proj_gemv<3, 2, 1, 3>(P12_PROJ_ARGS);
template [[host_name("p12_in_proj_gemv_r4p1u1b4")]] kernel void p12_in_proj_gemv<4, 1, 1, 4>(P12_PROJ_ARGS);
template [[host_name("p12_in_proj_gemv_r4p2u1b4")]] kernel void p12_in_proj_gemv<4, 2, 1, 4>(P12_PROJ_ARGS);
template [[host_name("p12_in_proj_gemv_r5p1u1b5")]] kernel void p12_in_proj_gemv<5, 1, 1, 5>(P12_PROJ_ARGS);
template [[host_name("p12_in_proj_gemv_r6p1u1b6")]] kernel void p12_in_proj_gemv<6, 1, 1, 6>(P12_PROJ_ARGS);
template [[host_name("p12_in_proj_gemv_r7p1u1b7")]] kernel void p12_in_proj_gemv<7, 1, 1, 7>(P12_PROJ_ARGS);
template [[host_name("p12_in_proj_gemv_r8p1u1b8")]] kernel void p12_in_proj_gemv<8, 1, 1, 8>(P12_PROJ_ARGS);
template [[host_name("p12_in_proj_gemv_r9p1u1b9")]] kernel void p12_in_proj_gemv<9, 1, 1, 9>(P12_PROJ_ARGS);
template [[host_name("p12_in_proj_gemv_r10p1u1b5")]] kernel void p12_in_proj_gemv<10, 1, 1, 5>(P12_PROJ_ARGS);
template [[host_name("p12_in_proj_gemv_r11p1u1b11")]] kernel void p12_in_proj_gemv<11, 1, 1, 11>(P12_PROJ_ARGS);
template [[host_name("p12_in_proj_gemv_r12p1u1b6")]] kernel void p12_in_proj_gemv<12, 1, 1, 6>(P12_PROJ_ARGS);
template [[host_name("p12_in_proj_gemv_r13p1u1b13")]] kernel void p12_in_proj_gemv<13, 1, 1, 13>(P12_PROJ_ARGS);
template [[host_name("p12_in_proj_gemv_r14p1u1b7")]] kernel void p12_in_proj_gemv<14, 1, 1, 7>(P12_PROJ_ARGS);
template [[host_name("p12_in_proj_gemv_r15p1u1b5")]] kernel void p12_in_proj_gemv<15, 1, 1, 5>(P12_PROJ_ARGS);
template [[host_name("p12_in_proj_gemv_r16p1u1b8")]] kernel void p12_in_proj_gemv<16, 1, 1, 8>(P12_PROJ_ARGS);
template [[host_name("p12_in_proj_gemv_r1p1u2b1")]] kernel void p12_in_proj_gemv<1, 1, 2, 1>(P12_PROJ_ARGS);
template [[host_name("p12_in_proj_gemv_r1p2u2b1")]] kernel void p12_in_proj_gemv<1, 2, 2, 1>(P12_PROJ_ARGS);
template [[host_name("p12_in_proj_gemv_r1p4u2b1")]] kernel void p12_in_proj_gemv<1, 4, 2, 1>(P12_PROJ_ARGS);
template [[host_name("p12_in_proj_gemv_r2p1u2b2")]] kernel void p12_in_proj_gemv<2, 1, 2, 2>(P12_PROJ_ARGS);
template [[host_name("p12_in_proj_gemv_r2p2u2b2")]] kernel void p12_in_proj_gemv<2, 2, 2, 2>(P12_PROJ_ARGS);
template [[host_name("p12_in_proj_gemv_r3p1u2b3")]] kernel void p12_in_proj_gemv<3, 1, 2, 3>(P12_PROJ_ARGS);
template [[host_name("p12_in_proj_gemv_r3p2u2b3")]] kernel void p12_in_proj_gemv<3, 2, 2, 3>(P12_PROJ_ARGS);
template [[host_name("p12_in_proj_gemv_r4p1u2b4")]] kernel void p12_in_proj_gemv<4, 1, 2, 4>(P12_PROJ_ARGS);
template [[host_name("p12_in_proj_gemv_r4p2u2b4")]] kernel void p12_in_proj_gemv<4, 2, 2, 4>(P12_PROJ_ARGS);
template [[host_name("p12_in_proj_gemv_r5p1u2b5")]] kernel void p12_in_proj_gemv<5, 1, 2, 5>(P12_PROJ_ARGS);
template [[host_name("p12_in_proj_gemv_r6p1u2b6")]] kernel void p12_in_proj_gemv<6, 1, 2, 6>(P12_PROJ_ARGS);
template [[host_name("p12_in_proj_gemv_r7p1u2b7")]] kernel void p12_in_proj_gemv<7, 1, 2, 7>(P12_PROJ_ARGS);
template [[host_name("p12_in_proj_gemv_r8p1u2b8")]] kernel void p12_in_proj_gemv<8, 1, 2, 8>(P12_PROJ_ARGS);
template [[host_name("p12_in_proj_gemv_r9p1u2b9")]] kernel void p12_in_proj_gemv<9, 1, 2, 9>(P12_PROJ_ARGS);
template [[host_name("p12_in_proj_gemv_r10p1u2b5")]] kernel void p12_in_proj_gemv<10, 1, 2, 5>(P12_PROJ_ARGS);
template [[host_name("p12_in_proj_gemv_r11p1u2b11")]] kernel void p12_in_proj_gemv<11, 1, 2, 11>(P12_PROJ_ARGS);
template [[host_name("p12_in_proj_gemv_r12p1u2b6")]] kernel void p12_in_proj_gemv<12, 1, 2, 6>(P12_PROJ_ARGS);
template [[host_name("p12_in_proj_gemv_r13p1u2b13")]] kernel void p12_in_proj_gemv<13, 1, 2, 13>(P12_PROJ_ARGS);
template [[host_name("p12_in_proj_gemv_r14p1u2b7")]] kernel void p12_in_proj_gemv<14, 1, 2, 7>(P12_PROJ_ARGS);
template [[host_name("p12_in_proj_gemv_r15p1u2b5")]] kernel void p12_in_proj_gemv<15, 1, 2, 5>(P12_PROJ_ARGS);
template [[host_name("p12_in_proj_gemv_r16p1u2b8")]] kernel void p12_in_proj_gemv<16, 1, 2, 8>(P12_PROJ_ARGS);
template [[host_name("p12_in_proj_gemv_r1p1u4b1")]] kernel void p12_in_proj_gemv<1, 1, 4, 1>(P12_PROJ_ARGS);
template [[host_name("p12_in_proj_gemv_r1p2u4b1")]] kernel void p12_in_proj_gemv<1, 2, 4, 1>(P12_PROJ_ARGS);
template [[host_name("p12_in_proj_gemv_r1p4u4b1")]] kernel void p12_in_proj_gemv<1, 4, 4, 1>(P12_PROJ_ARGS);
template [[host_name("p12_in_proj_gemv_r2p1u4b2")]] kernel void p12_in_proj_gemv<2, 1, 4, 2>(P12_PROJ_ARGS);
template [[host_name("p12_in_proj_gemv_r2p2u4b2")]] kernel void p12_in_proj_gemv<2, 2, 4, 2>(P12_PROJ_ARGS);
template [[host_name("p12_in_proj_gemv_r3p1u4b3")]] kernel void p12_in_proj_gemv<3, 1, 4, 3>(P12_PROJ_ARGS);
template [[host_name("p12_in_proj_gemv_r3p2u4b3")]] kernel void p12_in_proj_gemv<3, 2, 4, 3>(P12_PROJ_ARGS);
template [[host_name("p12_in_proj_gemv_r4p1u4b4")]] kernel void p12_in_proj_gemv<4, 1, 4, 4>(P12_PROJ_ARGS);
template [[host_name("p12_in_proj_gemv_r4p2u4b4")]] kernel void p12_in_proj_gemv<4, 2, 4, 4>(P12_PROJ_ARGS);
template [[host_name("p12_in_proj_gemv_r5p1u4b5")]] kernel void p12_in_proj_gemv<5, 1, 4, 5>(P12_PROJ_ARGS);
template [[host_name("p12_in_proj_gemv_r6p1u4b6")]] kernel void p12_in_proj_gemv<6, 1, 4, 6>(P12_PROJ_ARGS);
template [[host_name("p12_in_proj_gemv_r7p1u4b7")]] kernel void p12_in_proj_gemv<7, 1, 4, 7>(P12_PROJ_ARGS);
template [[host_name("p12_in_proj_gemv_r8p1u4b8")]] kernel void p12_in_proj_gemv<8, 1, 4, 8>(P12_PROJ_ARGS);
template [[host_name("p12_in_proj_gemv_r9p1u4b9")]] kernel void p12_in_proj_gemv<9, 1, 4, 9>(P12_PROJ_ARGS);
template [[host_name("p12_in_proj_gemv_r10p1u4b5")]] kernel void p12_in_proj_gemv<10, 1, 4, 5>(P12_PROJ_ARGS);
template [[host_name("p12_in_proj_gemv_r11p1u4b11")]] kernel void p12_in_proj_gemv<11, 1, 4, 11>(P12_PROJ_ARGS);
template [[host_name("p12_in_proj_gemv_r12p1u4b6")]] kernel void p12_in_proj_gemv<12, 1, 4, 6>(P12_PROJ_ARGS);
template [[host_name("p12_in_proj_gemv_r13p1u4b13")]] kernel void p12_in_proj_gemv<13, 1, 4, 13>(P12_PROJ_ARGS);
template [[host_name("p12_in_proj_gemv_r14p1u4b7")]] kernel void p12_in_proj_gemv<14, 1, 4, 7>(P12_PROJ_ARGS);
template [[host_name("p12_in_proj_gemv_r15p1u4b5")]] kernel void p12_in_proj_gemv<15, 1, 4, 5>(P12_PROJ_ARGS);
template [[host_name("p12_in_proj_gemv_r16p1u4b8")]] kernel void p12_in_proj_gemv<16, 1, 4, 8>(P12_PROJ_ARGS);

template [[host_name("p12_gemv_rt_r1p1u1b1")]] kernel void p12_gemv_rt<1, 1, 1, 1>(P12_RT_ARGS);
template [[host_name("p12_gemv_rt_r1p2u1b1")]] kernel void p12_gemv_rt<1, 2, 1, 1>(P12_RT_ARGS);
template [[host_name("p12_gemv_rt_r1p4u1b1")]] kernel void p12_gemv_rt<1, 4, 1, 1>(P12_RT_ARGS);
template [[host_name("p12_gemv_rt_r2p1u1b2")]] kernel void p12_gemv_rt<2, 1, 1, 2>(P12_RT_ARGS);
template [[host_name("p12_gemv_rt_r2p2u1b2")]] kernel void p12_gemv_rt<2, 2, 1, 2>(P12_RT_ARGS);
template [[host_name("p12_gemv_rt_r3p1u1b3")]] kernel void p12_gemv_rt<3, 1, 1, 3>(P12_RT_ARGS);
template [[host_name("p12_gemv_rt_r3p2u1b3")]] kernel void p12_gemv_rt<3, 2, 1, 3>(P12_RT_ARGS);
template [[host_name("p12_gemv_rt_r4p1u1b4")]] kernel void p12_gemv_rt<4, 1, 1, 4>(P12_RT_ARGS);
template [[host_name("p12_gemv_rt_r4p2u1b4")]] kernel void p12_gemv_rt<4, 2, 1, 4>(P12_RT_ARGS);
template [[host_name("p12_gemv_rt_r5p1u1b5")]] kernel void p12_gemv_rt<5, 1, 1, 5>(P12_RT_ARGS);
template [[host_name("p12_gemv_rt_r6p1u1b6")]] kernel void p12_gemv_rt<6, 1, 1, 6>(P12_RT_ARGS);
template [[host_name("p12_gemv_rt_r7p1u1b7")]] kernel void p12_gemv_rt<7, 1, 1, 7>(P12_RT_ARGS);
template [[host_name("p12_gemv_rt_r8p1u1b8")]] kernel void p12_gemv_rt<8, 1, 1, 8>(P12_RT_ARGS);
template [[host_name("p12_gemv_rt_r9p1u1b9")]] kernel void p12_gemv_rt<9, 1, 1, 9>(P12_RT_ARGS);
template [[host_name("p12_gemv_rt_r10p1u1b5")]] kernel void p12_gemv_rt<10, 1, 1, 5>(P12_RT_ARGS);
template [[host_name("p12_gemv_rt_r11p1u1b11")]] kernel void p12_gemv_rt<11, 1, 1, 11>(P12_RT_ARGS);
template [[host_name("p12_gemv_rt_r12p1u1b6")]] kernel void p12_gemv_rt<12, 1, 1, 6>(P12_RT_ARGS);
template [[host_name("p12_gemv_rt_r13p1u1b13")]] kernel void p12_gemv_rt<13, 1, 1, 13>(P12_RT_ARGS);
template [[host_name("p12_gemv_rt_r14p1u1b7")]] kernel void p12_gemv_rt<14, 1, 1, 7>(P12_RT_ARGS);
template [[host_name("p12_gemv_rt_r15p1u1b5")]] kernel void p12_gemv_rt<15, 1, 1, 5>(P12_RT_ARGS);
template [[host_name("p12_gemv_rt_r16p1u1b8")]] kernel void p12_gemv_rt<16, 1, 1, 8>(P12_RT_ARGS);
template [[host_name("p12_gemv_rt_r1p1u2b1")]] kernel void p12_gemv_rt<1, 1, 2, 1>(P12_RT_ARGS);
template [[host_name("p12_gemv_rt_r1p2u2b1")]] kernel void p12_gemv_rt<1, 2, 2, 1>(P12_RT_ARGS);
template [[host_name("p12_gemv_rt_r1p4u2b1")]] kernel void p12_gemv_rt<1, 4, 2, 1>(P12_RT_ARGS);
template [[host_name("p12_gemv_rt_r2p1u2b2")]] kernel void p12_gemv_rt<2, 1, 2, 2>(P12_RT_ARGS);
template [[host_name("p12_gemv_rt_r2p2u2b2")]] kernel void p12_gemv_rt<2, 2, 2, 2>(P12_RT_ARGS);
template [[host_name("p12_gemv_rt_r3p1u2b3")]] kernel void p12_gemv_rt<3, 1, 2, 3>(P12_RT_ARGS);
template [[host_name("p12_gemv_rt_r3p2u2b3")]] kernel void p12_gemv_rt<3, 2, 2, 3>(P12_RT_ARGS);
template [[host_name("p12_gemv_rt_r4p1u2b4")]] kernel void p12_gemv_rt<4, 1, 2, 4>(P12_RT_ARGS);
template [[host_name("p12_gemv_rt_r4p2u2b4")]] kernel void p12_gemv_rt<4, 2, 2, 4>(P12_RT_ARGS);
template [[host_name("p12_gemv_rt_r5p1u2b5")]] kernel void p12_gemv_rt<5, 1, 2, 5>(P12_RT_ARGS);
template [[host_name("p12_gemv_rt_r6p1u2b6")]] kernel void p12_gemv_rt<6, 1, 2, 6>(P12_RT_ARGS);
template [[host_name("p12_gemv_rt_r7p1u2b7")]] kernel void p12_gemv_rt<7, 1, 2, 7>(P12_RT_ARGS);
template [[host_name("p12_gemv_rt_r8p1u2b8")]] kernel void p12_gemv_rt<8, 1, 2, 8>(P12_RT_ARGS);
template [[host_name("p12_gemv_rt_r9p1u2b9")]] kernel void p12_gemv_rt<9, 1, 2, 9>(P12_RT_ARGS);
template [[host_name("p12_gemv_rt_r10p1u2b5")]] kernel void p12_gemv_rt<10, 1, 2, 5>(P12_RT_ARGS);
template [[host_name("p12_gemv_rt_r11p1u2b11")]] kernel void p12_gemv_rt<11, 1, 2, 11>(P12_RT_ARGS);
template [[host_name("p12_gemv_rt_r12p1u2b6")]] kernel void p12_gemv_rt<12, 1, 2, 6>(P12_RT_ARGS);
template [[host_name("p12_gemv_rt_r13p1u2b13")]] kernel void p12_gemv_rt<13, 1, 2, 13>(P12_RT_ARGS);
template [[host_name("p12_gemv_rt_r14p1u2b7")]] kernel void p12_gemv_rt<14, 1, 2, 7>(P12_RT_ARGS);
template [[host_name("p12_gemv_rt_r15p1u2b5")]] kernel void p12_gemv_rt<15, 1, 2, 5>(P12_RT_ARGS);
template [[host_name("p12_gemv_rt_r16p1u2b8")]] kernel void p12_gemv_rt<16, 1, 2, 8>(P12_RT_ARGS);
template [[host_name("p12_gemv_rt_r1p1u4b1")]] kernel void p12_gemv_rt<1, 1, 4, 1>(P12_RT_ARGS);
template [[host_name("p12_gemv_rt_r1p2u4b1")]] kernel void p12_gemv_rt<1, 2, 4, 1>(P12_RT_ARGS);
template [[host_name("p12_gemv_rt_r1p4u4b1")]] kernel void p12_gemv_rt<1, 4, 4, 1>(P12_RT_ARGS);
template [[host_name("p12_gemv_rt_r2p1u4b2")]] kernel void p12_gemv_rt<2, 1, 4, 2>(P12_RT_ARGS);
template [[host_name("p12_gemv_rt_r2p2u4b2")]] kernel void p12_gemv_rt<2, 2, 4, 2>(P12_RT_ARGS);
template [[host_name("p12_gemv_rt_r3p1u4b3")]] kernel void p12_gemv_rt<3, 1, 4, 3>(P12_RT_ARGS);
template [[host_name("p12_gemv_rt_r3p2u4b3")]] kernel void p12_gemv_rt<3, 2, 4, 3>(P12_RT_ARGS);
template [[host_name("p12_gemv_rt_r4p1u4b4")]] kernel void p12_gemv_rt<4, 1, 4, 4>(P12_RT_ARGS);
template [[host_name("p12_gemv_rt_r4p2u4b4")]] kernel void p12_gemv_rt<4, 2, 4, 4>(P12_RT_ARGS);
template [[host_name("p12_gemv_rt_r5p1u4b5")]] kernel void p12_gemv_rt<5, 1, 4, 5>(P12_RT_ARGS);
template [[host_name("p12_gemv_rt_r6p1u4b6")]] kernel void p12_gemv_rt<6, 1, 4, 6>(P12_RT_ARGS);
template [[host_name("p12_gemv_rt_r7p1u4b7")]] kernel void p12_gemv_rt<7, 1, 4, 7>(P12_RT_ARGS);
template [[host_name("p12_gemv_rt_r8p1u4b8")]] kernel void p12_gemv_rt<8, 1, 4, 8>(P12_RT_ARGS);
template [[host_name("p12_gemv_rt_r9p1u4b9")]] kernel void p12_gemv_rt<9, 1, 4, 9>(P12_RT_ARGS);
template [[host_name("p12_gemv_rt_r10p1u4b5")]] kernel void p12_gemv_rt<10, 1, 4, 5>(P12_RT_ARGS);
template [[host_name("p12_gemv_rt_r11p1u4b11")]] kernel void p12_gemv_rt<11, 1, 4, 11>(P12_RT_ARGS);
template [[host_name("p12_gemv_rt_r12p1u4b6")]] kernel void p12_gemv_rt<12, 1, 4, 6>(P12_RT_ARGS);
template [[host_name("p12_gemv_rt_r13p1u4b13")]] kernel void p12_gemv_rt<13, 1, 4, 13>(P12_RT_ARGS);
template [[host_name("p12_gemv_rt_r14p1u4b7")]] kernel void p12_gemv_rt<14, 1, 4, 7>(P12_RT_ARGS);
template [[host_name("p12_gemv_rt_r15p1u4b5")]] kernel void p12_gemv_rt<15, 1, 4, 5>(P12_RT_ARGS);
template [[host_name("p12_gemv_rt_r16p1u4b8")]] kernel void p12_gemv_rt<16, 1, 4, 8>(P12_RT_ARGS);

template [[host_name("p12_gemv_f32o_r1p1u2b1")]] kernel void p12_gemv_f32o<1, 1, 2, 1>(P12_F32_ARGS);
template [[host_name("p12_gemv_f32o_r1p2u2b1")]] kernel void p12_gemv_f32o<1, 2, 2, 1>(P12_F32_ARGS);
template [[host_name("p12_gemv_f32o_r1p4u2b1")]] kernel void p12_gemv_f32o<1, 4, 2, 1>(P12_F32_ARGS);
template [[host_name("p12_gemv_f32o_r2p1u2b2")]] kernel void p12_gemv_f32o<2, 1, 2, 2>(P12_F32_ARGS);
template [[host_name("p12_gemv_f32o_r2p2u2b2")]] kernel void p12_gemv_f32o<2, 2, 2, 2>(P12_F32_ARGS);
template [[host_name("p12_gemv_f32o_r3p1u2b3")]] kernel void p12_gemv_f32o<3, 1, 2, 3>(P12_F32_ARGS);
template [[host_name("p12_gemv_f32o_r3p2u2b3")]] kernel void p12_gemv_f32o<3, 2, 2, 3>(P12_F32_ARGS);
template [[host_name("p12_gemv_f32o_r4p1u2b4")]] kernel void p12_gemv_f32o<4, 1, 2, 4>(P12_F32_ARGS);
template [[host_name("p12_gemv_f32o_r4p2u2b4")]] kernel void p12_gemv_f32o<4, 2, 2, 4>(P12_F32_ARGS);
template [[host_name("p12_gemv_f32o_r5p1u2b5")]] kernel void p12_gemv_f32o<5, 1, 2, 5>(P12_F32_ARGS);
template [[host_name("p12_gemv_f32o_r6p1u2b6")]] kernel void p12_gemv_f32o<6, 1, 2, 6>(P12_F32_ARGS);
template [[host_name("p12_gemv_f32o_r7p1u2b7")]] kernel void p12_gemv_f32o<7, 1, 2, 7>(P12_F32_ARGS);
template [[host_name("p12_gemv_f32o_r8p1u2b8")]] kernel void p12_gemv_f32o<8, 1, 2, 8>(P12_F32_ARGS);
template [[host_name("p12_gemv_f32o_r9p1u2b9")]] kernel void p12_gemv_f32o<9, 1, 2, 9>(P12_F32_ARGS);
template [[host_name("p12_gemv_f32o_r10p1u2b5")]] kernel void p12_gemv_f32o<10, 1, 2, 5>(P12_F32_ARGS);
template [[host_name("p12_gemv_f32o_r11p1u2b11")]] kernel void p12_gemv_f32o<11, 1, 2, 11>(P12_F32_ARGS);
template [[host_name("p12_gemv_f32o_r12p1u2b6")]] kernel void p12_gemv_f32o<12, 1, 2, 6>(P12_F32_ARGS);
template [[host_name("p12_gemv_f32o_r13p1u2b13")]] kernel void p12_gemv_f32o<13, 1, 2, 13>(P12_F32_ARGS);
template [[host_name("p12_gemv_f32o_r14p1u2b7")]] kernel void p12_gemv_f32o<14, 1, 2, 7>(P12_F32_ARGS);
template [[host_name("p12_gemv_f32o_r15p1u2b5")]] kernel void p12_gemv_f32o<15, 1, 2, 5>(P12_F32_ARGS);
template [[host_name("p12_gemv_f32o_r16p1u2b8")]] kernel void p12_gemv_f32o<16, 1, 2, 8>(P12_F32_ARGS);
template [[host_name("p12_gemv_f32o_r1p1u4b1")]] kernel void p12_gemv_f32o<1, 1, 4, 1>(P12_F32_ARGS);
template [[host_name("p12_gemv_f32o_r1p2u4b1")]] kernel void p12_gemv_f32o<1, 2, 4, 1>(P12_F32_ARGS);
template [[host_name("p12_gemv_f32o_r1p4u4b1")]] kernel void p12_gemv_f32o<1, 4, 4, 1>(P12_F32_ARGS);
template [[host_name("p12_gemv_f32o_r2p1u4b2")]] kernel void p12_gemv_f32o<2, 1, 4, 2>(P12_F32_ARGS);
template [[host_name("p12_gemv_f32o_r2p2u4b2")]] kernel void p12_gemv_f32o<2, 2, 4, 2>(P12_F32_ARGS);
template [[host_name("p12_gemv_f32o_r3p1u4b3")]] kernel void p12_gemv_f32o<3, 1, 4, 3>(P12_F32_ARGS);
template [[host_name("p12_gemv_f32o_r3p2u4b3")]] kernel void p12_gemv_f32o<3, 2, 4, 3>(P12_F32_ARGS);
template [[host_name("p12_gemv_f32o_r4p1u4b4")]] kernel void p12_gemv_f32o<4, 1, 4, 4>(P12_F32_ARGS);
template [[host_name("p12_gemv_f32o_r4p2u4b4")]] kernel void p12_gemv_f32o<4, 2, 4, 4>(P12_F32_ARGS);
template [[host_name("p12_gemv_f32o_r5p1u4b5")]] kernel void p12_gemv_f32o<5, 1, 4, 5>(P12_F32_ARGS);
template [[host_name("p12_gemv_f32o_r6p1u4b6")]] kernel void p12_gemv_f32o<6, 1, 4, 6>(P12_F32_ARGS);
template [[host_name("p12_gemv_f32o_r7p1u4b7")]] kernel void p12_gemv_f32o<7, 1, 4, 7>(P12_F32_ARGS);
template [[host_name("p12_gemv_f32o_r8p1u4b8")]] kernel void p12_gemv_f32o<8, 1, 4, 8>(P12_F32_ARGS);
template [[host_name("p12_gemv_f32o_r9p1u4b9")]] kernel void p12_gemv_f32o<9, 1, 4, 9>(P12_F32_ARGS);
template [[host_name("p12_gemv_f32o_r10p1u4b5")]] kernel void p12_gemv_f32o<10, 1, 4, 5>(P12_F32_ARGS);
template [[host_name("p12_gemv_f32o_r11p1u4b11")]] kernel void p12_gemv_f32o<11, 1, 4, 11>(P12_F32_ARGS);
template [[host_name("p12_gemv_f32o_r12p1u4b6")]] kernel void p12_gemv_f32o<12, 1, 4, 6>(P12_F32_ARGS);
template [[host_name("p12_gemv_f32o_r13p1u4b13")]] kernel void p12_gemv_f32o<13, 1, 4, 13>(P12_F32_ARGS);
template [[host_name("p12_gemv_f32o_r14p1u4b7")]] kernel void p12_gemv_f32o<14, 1, 4, 7>(P12_F32_ARGS);
template [[host_name("p12_gemv_f32o_r15p1u4b5")]] kernel void p12_gemv_f32o<15, 1, 4, 5>(P12_F32_ARGS);
template [[host_name("p12_gemv_f32o_r16p1u4b8")]] kernel void p12_gemv_f32o<16, 1, 4, 8>(P12_F32_ARGS);
template [[host_name("p12_gemv_f32o_r1p1u8b1")]] kernel void p12_gemv_f32o<1, 1, 8, 1>(P12_F32_ARGS);
template [[host_name("p12_gemv_f32o_r1p2u8b1")]] kernel void p12_gemv_f32o<1, 2, 8, 1>(P12_F32_ARGS);
template [[host_name("p12_gemv_f32o_r1p4u8b1")]] kernel void p12_gemv_f32o<1, 4, 8, 1>(P12_F32_ARGS);
template [[host_name("p12_gemv_f32o_r2p1u8b2")]] kernel void p12_gemv_f32o<2, 1, 8, 2>(P12_F32_ARGS);
template [[host_name("p12_gemv_f32o_r2p2u8b2")]] kernel void p12_gemv_f32o<2, 2, 8, 2>(P12_F32_ARGS);
template [[host_name("p12_gemv_f32o_r3p1u8b3")]] kernel void p12_gemv_f32o<3, 1, 8, 3>(P12_F32_ARGS);
template [[host_name("p12_gemv_f32o_r3p2u8b3")]] kernel void p12_gemv_f32o<3, 2, 8, 3>(P12_F32_ARGS);
template [[host_name("p12_gemv_f32o_r4p1u8b4")]] kernel void p12_gemv_f32o<4, 1, 8, 4>(P12_F32_ARGS);
template [[host_name("p12_gemv_f32o_r4p2u8b4")]] kernel void p12_gemv_f32o<4, 2, 8, 4>(P12_F32_ARGS);
template [[host_name("p12_gemv_f32o_r5p1u8b5")]] kernel void p12_gemv_f32o<5, 1, 8, 5>(P12_F32_ARGS);
template [[host_name("p12_gemv_f32o_r6p1u8b6")]] kernel void p12_gemv_f32o<6, 1, 8, 6>(P12_F32_ARGS);
template [[host_name("p12_gemv_f32o_r7p1u8b7")]] kernel void p12_gemv_f32o<7, 1, 8, 7>(P12_F32_ARGS);
template [[host_name("p12_gemv_f32o_r8p1u8b8")]] kernel void p12_gemv_f32o<8, 1, 8, 8>(P12_F32_ARGS);
template [[host_name("p12_gemv_f32o_r9p1u8b9")]] kernel void p12_gemv_f32o<9, 1, 8, 9>(P12_F32_ARGS);
template [[host_name("p12_gemv_f32o_r10p1u8b5")]] kernel void p12_gemv_f32o<10, 1, 8, 5>(P12_F32_ARGS);
template [[host_name("p12_gemv_f32o_r11p1u8b11")]] kernel void p12_gemv_f32o<11, 1, 8, 11>(P12_F32_ARGS);
template [[host_name("p12_gemv_f32o_r12p1u8b6")]] kernel void p12_gemv_f32o<12, 1, 8, 6>(P12_F32_ARGS);
template [[host_name("p12_gemv_f32o_r13p1u8b13")]] kernel void p12_gemv_f32o<13, 1, 8, 13>(P12_F32_ARGS);
template [[host_name("p12_gemv_f32o_r14p1u8b7")]] kernel void p12_gemv_f32o<14, 1, 8, 7>(P12_F32_ARGS);
template [[host_name("p12_gemv_f32o_r15p1u8b5")]] kernel void p12_gemv_f32o<15, 1, 8, 5>(P12_F32_ARGS);
template [[host_name("p12_gemv_f32o_r16p1u8b8")]] kernel void p12_gemv_f32o<16, 1, 8, 8>(P12_F32_ARGS);
