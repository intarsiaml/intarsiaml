// matmul_mxfp4_naxr: out[M,N] = x[M,K] @ W^T for OCP MXFP4 weights on the M5 neural accelerators, the 4-bit weights decoded
// per tile inside the kernel (lane mnr, opt-in `INTARSIA_METAL_NAX_ROWS=1`; launcher crates/intarsia-backend-metal/src/mxfp4.rs).
// Compiled -std=metal4.0 into intarsia_metal4.metallib (build.rs METAL4_SOURCES).
//
// The row form of `matmul_mxfp4_nax`: that one decodes the whole weight to bf16 scratch per call (about 200 ms per pass through
// the 27B whatever M); here a threadgroup owns WR weight rows, walks K in steps of BK and per step
//   * stages the activations of its TM = 16 rows (rows past M are zero for good) as bf16 [TM][BK] in threadgroup memory,
//   * decodes its WR x BK weights (16 code bytes + 1 e8m0 scale byte per 32) to bf16 [WR][BK] in threadgroup memory (e2m1 *
//     2^(scale - 127): exact in bf16, the same values `mxfp4_w_to_bf16` writes),
//   * runs mpp::tensor_ops::matmul2d (multiply-accumulate) into a float cooperative tensor per simdgroup,
// as moe_nax.metal does for the grouped experts. Epilogue: the float tile through threadgroup memory, stored as f32 or rounded to
// bf16 (nearest even, `f2bf`).
//
// Numerics: the accelerator's K order, so NOT the bits of `matmul_mxfp4_gemv`. What holds instead (tests/mxfp4_naxr.rs): a row's
// bits depend on neither M, the row's index, the other rows, the form (WR) nor buffer offsets. Every M takes the same TM = 16 tile
// and the same BK, so a one-row call and row r of a 16-row call run the same accumulation.
//
// bind: 0 x [M,K] bf16 (the call's x, or the QDQ'd copy of `mxfp4_xprep_bf16` for W4A4), 1 W u8 [N,K/2] (4-byte aligned),
// 2 scale u8 [N,K/32], 3 out [M,N] bf16 / f32, 4 OpP: a = M, b = N, c = K (K % 32 == 0); s0 = scale skip, s1 = out skip,
// s2 = x skip (elements). Grid: (ceil(N / WR), ceil(M / 16)) threadgroups of 128 threads.
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#include "common.h"
using namespace metal;
using namespace mpp::tensor_ops;

constant constexpr uint NR_TM = 16;  // activation rows of the accelerator tile

constant float kNrFp4[16] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f, -0.f, -0.5f, -1.f, -1.5f, -2.f, -3.f, -4.f, -6.f};

// The 8 codes of a weight word (element i in nibble i) with e8m0 scale byte sb -> 8 bf16, element 2j in the low half of
// component j / 2 .. : returned as two uints per pair, x = elements 0..7 packed as uint4 (2 bf16 per uint, memory order).
// bf16 bits of e2m1(code) * 2^(sb - 127) built with integer math: code = sign, e (2 bits), m (1 bit); nonzero magnitudes are
// 2^(e - 1) * (1 + m / 2) for e > 0 and 0.5 for (e, m) = (0, 1), so the biased exponent is sb - 1 + e and the top mantissa bit
// is m when e > 0. Valid while that exponent stays a normal one (2 <= sb <= 252); other scale bytes (subnormal, overflow, NaN)
// take the float decode of `mxfp4_w_to_bf16`.
inline uint4 nr_dec8(uint v, uint sb) {
    const uint4 w = uint4(v);
    const uint4 t = ((w >> uint4(0u, 8u, 16u, 24u)) & 0xFu) | (((w >> uint4(4u, 12u, 20u, 28u)) & 0xFu) << 16u);
    if (sb - 2u > 250u) {
        const float s = sb == 0u ? as_type<float>(0x00400000u) : (sb == 255u ? as_type<float>(0x7FC00000u) : as_type<float>(sb << 23));
        uint4 o;
        for (int j = 0; j < 4; ++j) {
            const float lo = kNrFp4[t[j] & 0xFu] * s, hi = kNrFp4[t[j] >> 16] * s;
            o[j] = (as_type<uint>(lo) >> 16) | (as_type<uint>(hi) & 0xFFFF0000u);
        }
        return o;
    }
    const uint4 e = (t >> 1u) & 0x00030003u;
    const uint4 m = t & 0x00010001u;
    const uint4 ge2 = (e | (e >> 1u)) & 0x00010001u;
    const uint4 nz = ge2 | m;
    return (nz * ((sb - 1u) << 7)) + (e << 7u) + ((m & ge2) << 6u) + ((t & 0x00080008u) << 12u);
}

constant uint kNrLut[256] = {0x00000000u, 0x00000001u, 0x00000081u, 0x000000C1u, 0x00000101u, 0x00000141u, 0x00000181u, 0x000001C1u, 0x00008000u, 0x00008001u, 0x00008081u, 0x000080C1u, 0x00008101u, 0x00008141u, 0x00008181u, 0x000081C1u, 0x00010000u, 0x00010001u, 0x00010081u, 0x000100C1u, 0x00010101u, 0x00010141u, 0x00010181u, 0x000101C1u, 0x00018000u, 0x00018001u, 0x00018081u, 0x000180C1u, 0x00018101u, 0x00018141u, 0x00018181u, 0x000181C1u, 0x00810000u, 0x00810001u, 0x00810081u, 0x008100C1u, 0x00810101u, 0x00810141u, 0x00810181u, 0x008101C1u, 0x00818000u, 0x00818001u, 0x00818081u, 0x008180C1u, 0x00818101u, 0x00818141u, 0x00818181u, 0x008181C1u, 0x00C10000u, 0x00C10001u, 0x00C10081u, 0x00C100C1u, 0x00C10101u, 0x00C10141u, 0x00C10181u, 0x00C101C1u, 0x00C18000u, 0x00C18001u, 0x00C18081u, 0x00C180C1u, 0x00C18101u, 0x00C18141u, 0x00C18181u, 0x00C181C1u, 0x01010000u, 0x01010001u, 0x01010081u, 0x010100C1u, 0x01010101u, 0x01010141u, 0x01010181u, 0x010101C1u, 0x01018000u, 0x01018001u, 0x01018081u, 0x010180C1u, 0x01018101u, 0x01018141u, 0x01018181u, 0x010181C1u, 0x01410000u, 0x01410001u, 0x01410081u, 0x014100C1u, 0x01410101u, 0x01410141u, 0x01410181u, 0x014101C1u, 0x01418000u, 0x01418001u, 0x01418081u, 0x014180C1u, 0x01418101u, 0x01418141u, 0x01418181u, 0x014181C1u, 0x01810000u, 0x01810001u, 0x01810081u, 0x018100C1u, 0x01810101u, 0x01810141u, 0x01810181u, 0x018101C1u, 0x01818000u, 0x01818001u, 0x01818081u, 0x018180C1u, 0x01818101u, 0x01818141u, 0x01818181u, 0x018181C1u, 0x01C10000u, 0x01C10001u, 0x01C10081u, 0x01C100C1u, 0x01C10101u, 0x01C10141u, 0x01C10181u, 0x01C101C1u, 0x01C18000u, 0x01C18001u, 0x01C18081u, 0x01C180C1u, 0x01C18101u, 0x01C18141u, 0x01C18181u, 0x01C181C1u, 0x80000000u, 0x80000001u, 0x80000081u, 0x800000C1u, 0x80000101u, 0x80000141u, 0x80000181u, 0x800001C1u, 0x80008000u, 0x80008001u, 0x80008081u, 0x800080C1u, 0x80008101u, 0x80008141u, 0x80008181u, 0x800081C1u, 0x80010000u, 0x80010001u, 0x80010081u, 0x800100C1u, 0x80010101u, 0x80010141u, 0x80010181u, 0x800101C1u, 0x80018000u, 0x80018001u, 0x80018081u, 0x800180C1u, 0x80018101u, 0x80018141u, 0x80018181u, 0x800181C1u, 0x80810000u, 0x80810001u, 0x80810081u, 0x808100C1u, 0x80810101u, 0x80810141u, 0x80810181u, 0x808101C1u, 0x80818000u, 0x80818001u, 0x80818081u, 0x808180C1u, 0x80818101u, 0x80818141u, 0x80818181u, 0x808181C1u, 0x80C10000u, 0x80C10001u, 0x80C10081u, 0x80C100C1u, 0x80C10101u, 0x80C10141u, 0x80C10181u, 0x80C101C1u, 0x80C18000u, 0x80C18001u, 0x80C18081u, 0x80C180C1u, 0x80C18101u, 0x80C18141u, 0x80C18181u, 0x80C181C1u, 0x81010000u, 0x81010001u, 0x81010081u, 0x810100C1u, 0x81010101u, 0x81010141u, 0x81010181u, 0x810101C1u, 0x81018000u, 0x81018001u, 0x81018081u, 0x810180C1u, 0x81018101u, 0x81018141u, 0x81018181u, 0x810181C1u, 0x81410000u, 0x81410001u, 0x81410081u, 0x814100C1u, 0x81410101u, 0x81410141u, 0x81410181u, 0x814101C1u, 0x81418000u, 0x81418001u, 0x81418081u, 0x814180C1u, 0x81418101u, 0x81418141u, 0x81418181u, 0x814181C1u, 0x81810000u, 0x81810001u, 0x81810081u, 0x818100C1u, 0x81810101u, 0x81810141u, 0x81810181u, 0x818101C1u, 0x81818000u, 0x81818001u, 0x81818081u, 0x818180C1u, 0x81818101u, 0x81818141u, 0x81818181u, 0x818181C1u, 0x81C10000u, 0x81C10001u, 0x81C10081u, 0x81C100C1u, 0x81C10101u, 0x81C10141u, 0x81C10181u, 0x81C101C1u, 0x81C18000u, 0x81C18001u, 0x81C18081u, 0x81C180C1u, 0x81C18101u, 0x81C18141u, 0x81C18181u, 0x81C181C1u};

// byte (two codes) -> two bf16 through the table: field = sign, e << 7, m << 6 and bit 0 = nonzero; adding (base - 1) per nonzero
// field turns the flag into the exponent base ((sb - 1) << 7)
inline uint4 nr_dec8_lut(uint v, uint sb) {
    if (sb - 2u > 250u) return nr_dec8(v, sb);
    const uint bm1 = ((sb - 1u) << 7) - 1u;
    uint4 o;
    for (int j = 0; j < 4; ++j) {
        const uint q = kNrLut[(v >> (8 * j)) & 0xFFu];
        o[j] = q + (q & 0x00010001u) * bm1;
    }
    return o;
}

// WR weight rows per threadgroup (128 threads, 4 simdgroups side by side along the weight rows), K step BK.
template <uint WR, uint BK, typename OutT>
inline void mxfp4_naxr_impl(device const ushort* x, device const uchar* w, device const uchar* sc, device OutT* out,
                            constant OpP& p, threadgroup float* smem, uint2 tg, uint tid, uint sg) {
    constexpr uint TM = NR_TM;
    constexpr uint SN = WR / 4;               // weight rows per simdgroup
    constexpr uint LD = BK + 8;               // bf16 per staged row (a 16-byte multiple)
    constexpr uint LC = WR + 4;               // floats per epilogue row
    constexpr uint AC = BK / 8;               // 8-element activation chunks per row and step
    constexpr uint NA = (TM * AC + 127) / 128;
    constexpr uint TPR = 128 / WR;            // threads per weight row
    constexpr uint NB = BK / 32 / TPR;        // 32-weight groups per thread and step
    const uint M = p.a, N = p.b, K = p.c;
    const uint m0 = tg.y * TM, n0 = tg.x * WR;
    const uint rows = min(TM, M - m0);
    threadgroup bfloat* As = (threadgroup bfloat*)smem;
    threadgroup bfloat* Bs = As + TM * LD;
    // rows past M: zero for good (the staging writes rows 0 .. rows - 1 only)
    for (uint i = rows * LD / 8 + tid; i < TM * LD / 8; i += 128u) *(threadgroup uint4*)(As + i * 8u) = uint4(0u);

    // activation chunks of this thread
    device const ushort* ap[NA];
    for (uint j = 0; j < NA; ++j) {
        const uint c = tid + 128u * j, r = c / AC;
        ap[j] = (c < TM * AC && r < rows) ? x + p.s2 + (ulong)(m0 + r) * K + (c % AC) * 8u : nullptr;
    }
    // weights of this thread: row br, groups bq .. bq + NB - 1 of each step
    const uint br = tid / TPR, bq = (tid % TPR) * NB;
    const bool bok = n0 + br < N;
    const ulong wrow = bok ? n0 + br : 0;
    device const uchar* wp = w + wrow * (K / 2);
    device const uchar* sp = sc + p.s0 + wrow * (K / 32);
    uint4 wv[NB], wn[NB];
    uint sv[NB], sn[NB];
    for (uint q = 0; q < NB; ++q) {
        const uint g = bq + q;
        const bool ok = bok && g * 32u < K;
        wv[q] = ok ? uint4(*(device const packed_uint4*)(wp + g * 16u)) : uint4(0u);
        sv[q] = ok ? (uint)sp[g] : 127u;
        const uint g1 = BK / 32u + bq + q;
        const bool ok1 = bok && g1 * 32u < K;
        wn[q] = ok1 ? uint4(*(device const packed_uint4*)(wp + (ulong)g1 * 16u)) : uint4(0u);
        sn[q] = ok1 ? (uint)sp[g1] : 127u;
    }
    const uint scol = sg * SN;

    constexpr auto desc = matmul2d_descriptor(TM, SN, BK, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroup> op;
    using ext_a = extents<int, BK, TM>;
    using ext_b = extents<int, BK, SN>;
    using ten_a = tensor<threadgroup bfloat, ext_a, tensor_inline>;
    using ten_b = tensor<threadgroup bfloat, ext_b, tensor_inline>;
    auto ct = op.template get_destination_cooperative_tensor<ten_a, ten_b, float>();
#pragma clang loop unroll(full)
    for (uint16_t i = 0; i < ct.get_capacity(); ++i) ct[i] = 0.f;

    for (uint k0 = 0; k0 < K; k0 += BK) {
        for (uint j = 0; j < NA; ++j) {  // activations (x may sit at an odd element: 2-byte loads)
            const uint c = tid + 128u * j;
            if (ap[j] == nullptr || (p.d & 8u)) continue;
            uint4 v = uint4(0u);
            if (k0 + (c % AC) * 8u < K) {
                device const packed_ushort4* q4 = (device const packed_ushort4*)(ap[j] + k0);
                const ushort4 a = ushort4(q4[0]), b = ushort4(q4[1]);
                v = uint4(a.x | ((uint)a.y << 16), a.z | ((uint)a.w << 16), b.x | ((uint)b.y << 16), b.z | ((uint)b.w << 16));
            }
            *(threadgroup uint4*)(As + (c / AC) * LD + (c % AC) * 8u) = v;
        }
        for (uint q = 0; q < NB; ++q) {
            threadgroup uint4* d = (threadgroup uint4*)(Bs + br * LD + (bq + q) * 32u);
            if (p.d & 2u) { if (k0 == 0) for (uint i = 0; i < 4; ++i) d[i] = nr_dec8(wv[q][i], sv[q]); }
            else if (p.d & 16u) for (uint i = 0; i < 4; ++i) d[i] = nr_dec8_lut(wv[q][i], sv[q]);
            else for (uint i = 0; i < 4; ++i) d[i] = nr_dec8(wv[q][i], sv[q]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // the words of the step after next are fetched while the accelerator runs (two steps of slack for the memory)
        for (uint q = 0; q < NB; ++q) {
            wv[q] = wn[q];
            sv[q] = sn[q];
            const uint g = (k0 + 2u * BK) / 32u + bq + q;
            const bool ok = bok && g * 32u < K;
            if (p.d & 32u) {
                const uint g1 = (k0 + BK) / 32u + bq + q;
                const bool ok1 = bok && g1 * 32u < K;
                wv[q] = ok1 ? uint4(*(device const packed_uint4*)(wp + (ulong)g1 * 16u)) : uint4(0u);
                sv[q] = ok1 ? (uint)sp[g1] : 127u;
            } else {
                wn[q] = ok ? uint4(*(device const packed_uint4*)(wp + (ulong)g * 16u)) : uint4(0u);
                sn[q] = ok ? (uint)sp[g] : 127u;
            }
        }
        ext_a ea;
        ext_b eb;
        array<int, 2> st = {1, int(LD)};
        ten_a tA(As, ea, st);                // activations [TM rows][BK]
        ten_b tB(Bs + scol * LD, eb, st);    // weights [SN rows][BK] (transpose_right)
        if (p.d & 8u) {
            tensor<device bfloat, dextents<int, 2>, tensor_inline> dA((device bfloat*)(x + p.s2), dextents<int, 2>(int(K), int(M)));
            auto sA = dA.template slice<BK, TM>(int(k0), int(m0));
            op.run(sA, tB, ct);
        } else if (!(p.d & 1u)) op.run(tA, tB, ct);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    {
        extents<int, SN, TM> ec;
        array<int, 2> st = {1, int(LC)};
        tensor<threadgroup float, extents<int, SN, TM>, tensor_inline> tC(smem + scol, ec, st);
        ct.store(tC);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint q = tid; q < rows * WR; q += 128u) {
        const uint r = q / WR, c = q % WR;
        if (n0 + c >= N) continue;
        store_out(out + p.s1, (ulong)(m0 + r) * N + n0 + c, smem[r * LC + c]);
    }
}

// threadgroup memory: the staged activation + weight tiles, reused by the f32 epilogue
#define NR_SMEM_N(WR, BK) \
    (((NR_TM + (WR)) * ((BK) + 8) * 2 > NR_TM * ((WR) + 4) * 4 ? (NR_TM + (WR)) * ((BK) + 8) * 2 : NR_TM * ((WR) + 4) * 4) / 16)

#define MX_NAXR(NAME, WR, BK, OUT)                                                                                   \
    kernel void NAME(device const ushort* x [[buffer(0)]], device const uchar* w [[buffer(1)]],                      \
                     device const uchar* sc [[buffer(2)]], device OUT* out [[buffer(3)]],                            \
                     constant OpP& p [[buffer(4)]], uint2 tg [[threadgroup_position_in_grid]],                       \
                     uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {         \
        threadgroup float4 smem[NR_SMEM_N(WR, BK)];                                                                  \
        mxfp4_naxr_impl<WR, BK, OUT>(x, w, sc, out, p, (threadgroup float*)smem, tg, tid, sg);                       \
    }

// Keep in step with NAXR_FORMS in src/mxfp4.rs.
MX_NAXR(mxfp4_naxr_bf16_w64k128, 64, 128, ushort)
MX_NAXR(mxfp4_naxr_f32_w64k128, 64, 128, float)
MX_NAXR(mxfp4_naxr_bf16_w32k128, 32, 128, ushort)
MX_NAXR(mxfp4_naxr_f32_w32k128, 32, 128, float)
MX_NAXR(mxfp4_naxr_bf16_w32k256, 32, 256, ushort)
MX_NAXR(mxfp4_naxr_f32_w32k256, 32, 256, float)

// ---- register-fed form (no threadgroup memory, no barriers) ----------------------------------------------------------------
// A simdgroup owns 32 weight rows and runs matmul2d 16 x 32 x 16 on register-resident cooperative tensors (the 16 x 16 fragment
// layout of attention_flash_nax.metal: lane = 4 * qid + r holds rows fm and fm + 8, fm = (qid & 4) | ((lane >> 1) & 3), and
// columns 4 * cq .. 4 * cq + 3, cq = (qid & 2) | (lane & 1); element i * 4 + j is (row fm + 8 i, column 4 cq + j)). Per K run of
// 16 a lane loads the 4 codes (2 bytes) it holds of each of its 4 weight rows and decodes them into the B fragments, and 4
// activations (8 bytes) of each of its 2 activation rows into the A fragment. Threadgroup: 4 simdgroups = 128 weight rows.
inline bfloat4 nq_dec4(uint v, uint sb) {  // 4 codes (16 bits) -> 4 bf16
    uint2 o;
    if (sb - 2u > 250u) {
        const uint4 d = nr_dec8(v, sb);
        o = uint2(d.x, d.y);
    } else {
        const uint bm1 = ((sb - 1u) << 7) - 1u;
        const uint a = kNrLut[v & 0xFFu], b = kNrLut[(v >> 8) & 0xFFu];
        o = uint2(a + (a & 0x00010001u) * bm1, b + (b & 0x00010001u) * bm1);
    }
    return as_type<bfloat4>(o);
}

template <typename OutT>
inline void mxfp4_naxq_impl(device const ushort* x, device const uchar* w, device const uchar* sc, device OutT* out,
                            constant OpP& p, uint2 tg, uint sg, uint lane) {
    const uint M = p.a, N = p.b, K = p.c;
    const uint m0 = tg.y * 16u, n0 = (tg.x * 4u + sg) * 32u;
    if (n0 >= N) return;  // whole simdgroup exits together
    const uint qid = lane >> 2;
    const uint fm = (qid & 4u) | ((lane >> 1) & 3u);
    const uint cq = (qid & 2u) | (lane & 1u);
    // activation rows fm, fm + 8 (null past M: zeros); weight rows 16 f + fm + 8 i of the simdgroup (null past N)
    device const ushort* ap[2];
    for (uint i = 0; i < 2; ++i) {
        const uint r = m0 + fm + 8u * i;
        ap[i] = r < M ? x + p.s2 + (ulong)r * K + 4u * cq : nullptr;
    }
    device const uchar* wp[4];
    device const uchar* sp[4];
    for (uint q = 0; q < 4; ++q) {
        const uint n = n0 + 16u * (q >> 1) + fm + 8u * (q & 1u);
        wp[q] = n < N ? w + (ulong)n * (K / 2) + 2u * cq : nullptr;
        sp[q] = sc + p.s0 + (ulong)(n < N ? n : 0u) * (K / 32);
    }
    constexpr auto desc = matmul2d_descriptor(16, 32, 16, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroup> op;
    auto ct_a = op.template get_left_input_cooperative_tensor<bfloat, bfloat, float>();
    auto ct_b = op.template get_right_input_cooperative_tensor<bfloat, bfloat, float>();
    auto ct_c = op.template get_destination_cooperative_tensor<decltype(ct_a), decltype(ct_b), float>();
#pragma clang loop unroll(full)
    for (uint16_t i = 0; i < 16; ++i) ct_c[i] = 0.f;
    for (uint g = 0; g < K / 32u; ++g) {
        uint sb[4];
        for (uint q = 0; q < 4; ++q) sb[q] = sp[q][g];
#pragma clang loop unroll(full)
        for (uint h = 0; h < 2; ++h) {
            const uint kb = g * 32u + h * 16u;
#pragma clang loop unroll(full)
            for (uint i = 0; i < 2; ++i) {
                bfloat4 a = bfloat4(bfloat(0.f));
                if (ap[i] != nullptr) a = as_type<bfloat4>(ushort4(*(device const packed_ushort4*)(ap[i] + kb)));
                ct_a[i * 4] = a.x;
                ct_a[i * 4 + 1] = a.y;
                ct_a[i * 4 + 2] = a.z;
                ct_a[i * 4 + 3] = a.w;
            }
#pragma clang loop unroll(full)
            for (uint q = 0; q < 4; ++q) {
                bfloat4 b = bfloat4(bfloat(0.f));
                if (wp[q] != nullptr) b = nq_dec4((uint)(*(device const ushort*)(wp[q] + kb / 2u)), sb[q]);
                ct_b[q * 4] = b.x;
                ct_b[q * 4 + 1] = b.y;
                ct_b[q * 4 + 2] = b.z;
                ct_b[q * 4 + 3] = b.w;
            }
            op.run(ct_a, ct_b, ct_c);
        }
    }
#pragma clang loop unroll(full)
    for (uint q = 0; q < 4; ++q) {
        const uint r = m0 + fm + 8u * (q & 1u);
        if (r >= M) continue;
        for (uint j = 0; j < 4; ++j) {
            const uint n = n0 + 16u * (q >> 1) + 4u * cq + j;
            if (n < N) store_out(out + p.s1, (ulong)r * N + n, ct_c[q * 4 + j]);
        }
    }
}

#define MX_NAXQ(NAME, OUT)                                                                                           \
    kernel void NAME(device const ushort* x [[buffer(0)]], device const uchar* w [[buffer(1)]],                      \
                     device const uchar* sc [[buffer(2)]], device OUT* out [[buffer(3)]],                            \
                     constant OpP& p [[buffer(4)]], uint2 tg [[threadgroup_position_in_grid]],                       \
                     uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {          \
        mxfp4_naxq_impl<OUT>(x, w, sc, out, p, tg, sg, lane);                                                        \
    }
MX_NAXQ(mxfp4_naxq_bf16, ushort)
MX_NAXQ(mxfp4_naxq_f32, float)
