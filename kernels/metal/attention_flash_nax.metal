// Flash-style causal prefill attention on the M5 neural accelerators (Metal 4, mpp::tensor_ops::matmul2d on
// simdgroup-scope cooperative tensors), GQA-shared work split. Compiled -std=metal4.0 into intarsia_metal4.metallib
// (build.rs METAL4_SOURCES). Launcher: crates/intarsia-backend-metal/src/flash_nax.rs. Same bindings and OpP fields as
// `attn_prefill_bf16_flash` (attention_flash.metal): 0 q [T,N,D], 1 k cache [Nkv,cap,D], 2 v cache, 3 out [T,N,D],
// 4 OpP (a = T, b = N, c = Nkv, d = D, e = cap, f = pos0, f0 = scale); operands 16-byte aligned (the launcher falls back
// to the P2 kernel otherwise). Query rows are positions pos0 .. pos0 + T, keys 0 .. pos0 + T are in the cache, row i sees
// keys 0 ..= pos0 + i. head_dim 128.
//
// Structure after MLX's NAX attention (mlx/backend/metal/kernels/steel/attn/kernels/steel_attention_nax.h and
// steel/attn/nax.h): 16x16 fragments with 8 elements per lane, matmul2d 16x32x16 on register-resident cooperative
// tensors, S = Q.K^T with a transposed B operand, O += P.V, and the S fragments feed the P.V product as its A operand
// (the C and A fragment layouts coincide). A simdgroup owns 16 query rows of one query head; a threadgroup is NSG
// simdgroups = G heads of a kv head times NSG / G consecutive 16-row slabs. There is no threadgroup staging: Q fragments
// stay in registers, K and V fragments come straight from device memory, where the simdgroups that share a kv head read
// the same lines.
//
// What differs from MLX is how the K / V fragments get into registers. MLX gathers them with scalar lane loads. Here a
// full key tile goes through the primitive's own `cooperative_tensor::load` from a device tensor slice, the path the
// NAX GEMM (gemm_nax_bf16.metal) takes to 55-65 TFLOPS. That was 3.3x faster than 16-byte lane loads from device memory
// (12.6 TFLOPS, 8 rows x 64 bytes per load instruction) and 2.8x faster than staging the tiles through threadgroup memory
// as attention_flash.metal does (14.7), see docs/results/2026-10-15-metal-prefill-f1-nax-attn.md. The tensor slice is not
// bounds-checked by `load`, so only tiles that lie entirely inside the visible keys use it; the last, partial tile uses
// bounds-checked lane loads (rows past the visible keys are zeros: the cache there may hold anything, and 0 * NaN would
// poison P.V). Both paths run the same matmul on the same bf16 values.
//
// Lane layout of a fragment (MLX BaseNAXFrag::get_coord): lane = 4 * qid + r; rows fm = (qid & 4) | ((lane >> 1) & 3) and
// fm + 8 (element i * 4 + j is row fm + 8 i); columns 4 * cq + j with cq = (qid & 2) | (lane & 1). A row lives on the four
// lanes that differ in bits 0 and 3 of the lane id (the xor-1 / xor-8 shuffles).
//
// Online softmax in float, log2 domain. P is rounded to bf16 for the P.V product, and the l sum is taken over those same
// rounded weights, so the weights P.V applies sum to l (up to float accumulation) and the row is a proper weighted
// average of V: a constant V comes out bitwise unchanged, which the unrounded-sum variant of F1 (and of
// attn_prefill_bf16_flash) does not give. Masking assigns -inf, so a fully masked tile leaves a row's max, sum and accumulator
// bitwise unchanged: block membership and extra tiles do not change a row's result (the matmul rows are independent).
// Key tiles start at absolute key 0, never at the query offset, so how a prompt is cut into chunks does not change a
// row's arithmetic.
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#include "common.h"
using namespace metal;

#define FN_UNROLL _Pragma("clang loop unroll(full)")

template <typename T>
using frag_t = vec<T, 8>;

// C (two 16x16 fragments, a 16x32 tile) += A (16x16) . B (16x32 as two 16x16 fragments); TB: B fragments hold [n, k].
// Strict precision (relaxed_precision = false, as the GEMM).
template <bool TB, typename AT, typename BT>
METAL_FUNC void nax_mma(thread frag_t<float>& C0, thread frag_t<float>& C1, thread const frag_t<AT>& A,
                        thread const frag_t<BT>& B0, thread const frag_t<BT>& B1) {
    constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
        16, 32, 16, false, TB, false, mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    mpp::tensor_ops::matmul2d<desc, execution_simdgroup> op;
    auto ct_a = op.template get_left_input_cooperative_tensor<AT, BT, float>();
    auto ct_b = op.template get_right_input_cooperative_tensor<AT, BT, float>();
    auto ct_c = op.template get_destination_cooperative_tensor<decltype(ct_a), decltype(ct_b), float>();
    FN_UNROLL
    for (short i = 0; i < 8; ++i) {
        ct_a[i] = A[i];
        ct_b[i] = B0[i];
        ct_b[8 + i] = B1[i];
        ct_c[i] = C0[i];
        ct_c[8 + i] = C1[i];
    }
    op.run(ct_a, ct_b, ct_c);
    FN_UNROLL
    for (short i = 0; i < 8; ++i) {
        C0[i] = ct_c[i];
        C1[i] = ct_c[8 + i];
    }
}

// the same product with the B fragments loaded by the primitive from a tensor slice
template <bool TB, typename AT, typename TS>
METAL_FUNC void nax_mma_ct(thread frag_t<float>& C0, thread frag_t<float>& C1, thread const frag_t<AT>& A, TS src) {
    constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
        16, 32, 16, false, TB, false, mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    mpp::tensor_ops::matmul2d<desc, execution_simdgroup> op;
    auto ct_a = op.template get_left_input_cooperative_tensor<AT, bfloat, float>();
    auto ct_b = op.template get_right_input_cooperative_tensor<AT, bfloat, float>();
    auto ct_c = op.template get_destination_cooperative_tensor<decltype(ct_a), decltype(ct_b), float>();
    ct_b.load(src);
    FN_UNROLL
    for (short i = 0; i < 8; ++i) {
        ct_a[i] = A[i];
        ct_c[i] = C0[i];
        ct_c[8 + i] = C1[i];
    }
    op.run(ct_a, ct_b, ct_c);
    FN_UNROLL
    for (short i = 0; i < 8; ++i) {
        C0[i] = ct_c[i];
        C1[i] = ct_c[8 + i];
    }
}

// 4 bf16 (8 bytes) of each of two rows into one fragment (elements 0..3 row r0, 4..7 row r1); zeros when !ok
METAL_FUNC frag_t<bfloat> ld_frag(device const ushort* r0, bool ok0, device const ushort* r1, bool ok1) {
    const uint2 a = ok0 ? *(device const uint2*)r0 : uint2(0);
    const uint2 b = ok1 ? *(device const uint2*)r1 : uint2(0);
    const bfloat4 al = as_type<bfloat4>(a), bl = as_type<bfloat4>(b);
    frag_t<bfloat> f;
    FN_UNROLL
    for (short j = 0; j < 4; ++j) {
        f[j] = al[j];
        f[4 + j] = bl[j];
    }
    return f;
}

template <int D, int BK, int NSG>
[[kernel, max_total_threads_per_threadgroup(NSG * 32)]] void attn_nax(
    device const ushort* q [[buffer(0)]],
    device const ushort* kc [[buffer(1)]],
    device const ushort* vc [[buffer(2)]],
    device ushort* out [[buffer(3)]],
    constant OpP& p [[buffer(4)]],
    uint2 tg [[threadgroup_position_in_grid]],
    uint2 grid [[threadgroups_per_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    constexpr int TD = D / 16;
    constexpr int TK = BK / 16;
    static_assert(BK % 32 == 0 && D % 32 == 0, "tile shape");

    const uint T = p.a, N = p.b, Nkv = p.c;
    const ulong cap = p.e, pos0 = p.f;
    const uint G = N / Nkv;
    const uint PQ = NSG / G;
    const uint BQ = 16 * PQ;
    const uint qb = grid.x - 1 - tg.x;
    const uint kvh = tg.y;
    const uint hg = sg / PQ, pq = sg % PQ;
    const uint head = kvh * G + hg;
    const uint qbase = qb * BQ + pq * 16;
    const ulong kv_len = min(pos0 + T, cap);
    const ulong kmax = min(kv_len, pos0 + min((ulong)T, (ulong)(qb + 1) * BQ));
    const uint nkt = (uint)((kmax + BK - 1) / BK);

    const uint qid = lane / 4;
    const uint fm = (qid & 4) + ((lane / 2) % 4);
    const uint cq = (qid & 2) + (lane & 1);

    device bfloat* kb = (device bfloat*)(kc + (ulong)kvh * cap * D);
    device bfloat* vb = (device bfloat*)(vc + (ulong)kvh * cap * D);
    tensor<device bfloat, dextents<int, 2>, tensor_inline> tK(kb, dextents<int, 2>(D, (int)kv_len));
    tensor<device bfloat, dextents<int, 2>, tensor_inline> tV(vb, dextents<int, 2>(D, (int)kv_len));

    const uint pos0r = qbase + fm, pos1r = qbase + fm + 8;
    frag_t<bfloat> Qf[TD];
    FN_UNROLL
    for (int id = 0; id < TD; ++id) {
        const uint2 a = pos0r < T ? *(device const uint2*)(q + ((ulong)pos0r * N + head) * D + 16 * id + 4 * cq) : uint2(0);
        const uint2 b = pos1r < T ? *(device const uint2*)(q + ((ulong)pos1r * N + head) * D + 16 * id + 4 * cq) : uint2(0);
        const bfloat4 al = as_type<bfloat4>(a), bl = as_type<bfloat4>(b);
        FN_UNROLL
        for (short j = 0; j < 4; ++j) {
            Qf[id][j] = al[j];
            Qf[id][4 + j] = bl[j];
        }
    }

    frag_t<float> O[TD];
    FN_UNROLL
    for (int i = 0; i < TD; ++i) O[i] = frag_t<float>(0.0f);
    float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.0f, l1 = 0.0f;
    const float c = p.f0 * 1.44269504088896340736f;
    const ulong row0 = pos0 + pos0r, row1 = pos0 + pos1r;
    const ulong first_row = pos0 + qbase;

    for (uint kt = 0; kt < nkt; ++kt) {
        const int key0 = (int)((ulong)kt * BK);

        frag_t<float> S[TK];
        FN_UNROLL
        for (int j = 0; j < TK; ++j) S[j] = frag_t<float>(0.0f);
        // full tiles go through cooperative_tensor::load (no bounds check on the tensor slice: it must lie inside the
        // cache); the last, partial tile through bounds-checked lane loads (rows past kv_len are zeros)
        const bool full = (ulong)key0 + BK <= kv_len;
        if (full) {
            FN_UNROLL
            for (int kp = 0; kp < TK / 2; ++kp) {
                FN_UNROLL
                for (int id = 0; id < TD; ++id) {
                    auto sk = tK.template slice<16, 32>(16 * id, key0 + 32 * kp);
                    nax_mma_ct<true>(S[2 * kp], S[2 * kp + 1], Qf[id], sk);
                }
            }
        } else {
            FN_UNROLL
            for (int kp = 0; kp < TK / 2; ++kp) {
                const ulong ka = (ulong)key0 + 32 * kp + fm, kb2 = ka + 16;
                FN_UNROLL
                for (int id = 0; id < TD; ++id) {
                    const uint co = 16 * id + 4 * cq;
                    const frag_t<bfloat> b0 = ld_frag((device const ushort*)kb + ka * D + co, ka < kv_len,
                                                      (device const ushort*)kb + (ka + 8) * D + co, ka + 8 < kv_len);
                    const frag_t<bfloat> b1 = ld_frag((device const ushort*)kb + kb2 * D + co, kb2 < kv_len,
                                                      (device const ushort*)kb + (kb2 + 8) * D + co, kb2 + 8 < kv_len);
                    nax_mma<true>(S[2 * kp], S[2 * kp + 1], Qf[id], b0, b1);
                }
            }
        }

        const bool masked = (ulong)key0 + BK - 1 > first_row;
        float mx0 = -INFINITY, mx1 = -INFINITY;
        FN_UNROLL
        for (int f = 0; f < TK; ++f) {
            FN_UNROLL
            for (int e = 0; e < 8; ++e) {
                float s = S[f][e] * c;
                const ulong key = (ulong)key0 + 16 * f + 4 * cq + (e & 3);
                if (masked && key > ((e & 4) ? row1 : row0)) s = -INFINITY;
                S[f][e] = s;
                if (e & 4) mx1 = max(mx1, s); else mx0 = max(mx0, s);
            }
        }
        mx0 = max(mx0, simd_shuffle_xor(mx0, ushort(1)));
        mx0 = max(mx0, simd_shuffle_xor(mx0, ushort(8)));
        mx1 = max(mx1, simd_shuffle_xor(mx1, ushort(1)));
        mx1 = max(mx1, simd_shuffle_xor(mx1, ushort(8)));
        const float mn0 = max(m0, mx0), mn1 = max(m1, mx1);
        const float mu0 = mn0 == -INFINITY ? 0.0f : mn0, mu1 = mn1 == -INFINITY ? 0.0f : mn1;
        const float f0 = fast::exp2(m0 - mu0), f1 = fast::exp2(m1 - mu1);
        m0 = mn0;
        m1 = mn1;
        float ls0 = 0.0f, ls1 = 0.0f;
        frag_t<bfloat> Pf[TK];
        FN_UNROLL
        for (int f = 0; f < TK; ++f) {
            FN_UNROLL
            for (int e = 0; e < 8; ++e) {
                Pf[f][e] = bfloat(fast::exp2(S[f][e] - ((e & 4) ? mu1 : mu0)));
                const float pr = float(Pf[f][e]);  // the weight P.V applies
                if (e & 4) ls1 += pr; else ls0 += pr;
            }
        }
        l0 = l0 * f0 + ls0;
        l1 = l1 * f1 + ls1;
        FN_UNROLL
        for (int i = 0; i < TD; ++i) {
            FN_UNROLL
            for (int e = 0; e < 8; ++e) O[i][e] *= (e & 4) ? f1 : f0;
        }

        if (full) {
            FN_UNROLL
            for (int kk = 0; kk < TK; ++kk) {
                FN_UNROLL
                for (int idp = 0; idp < TD / 2; ++idp) {
                    auto sv = tV.template slice<32, 16>(32 * idp, key0 + 16 * kk);
                    nax_mma_ct<false>(O[2 * idp], O[2 * idp + 1], Pf[kk], sv);
                }
            }
        } else {
            FN_UNROLL
            for (int kk = 0; kk < TK; ++kk) {
                const ulong va = (ulong)key0 + 16 * kk + fm;
                FN_UNROLL
                for (int idp = 0; idp < TD / 2; ++idp) {
                    const uint co = 32 * idp + 4 * cq;
                    const frag_t<bfloat> v0 = ld_frag((device const ushort*)vb + va * D + co, va < kv_len,
                                                      (device const ushort*)vb + (va + 8) * D + co, va + 8 < kv_len);
                    const frag_t<bfloat> v1 = ld_frag((device const ushort*)vb + va * D + co + 16, va < kv_len,
                                                      (device const ushort*)vb + (va + 8) * D + co + 16, va + 8 < kv_len);
                    nax_mma<false>(O[2 * idp], O[2 * idp + 1], Pf[kk], v0, v1);
                }
            }
        }
    }

    l0 += simd_shuffle_xor(l0, ushort(1));
    l0 += simd_shuffle_xor(l0, ushort(8));
    l1 += simd_shuffle_xor(l1, ushort(1));
    l1 += simd_shuffle_xor(l1, ushort(8));
    const float inv0 = 1.0f / l0, inv1 = 1.0f / l1;
    FN_UNROLL
    for (int i = 0; i < 2; ++i) {
        const uint pos = i ? pos1r : pos0r;
        if (pos >= T) continue;
        const float inv = i ? inv1 : inv0;
        device uint2* dst = (device uint2*)(out + ((ulong)pos * N + head) * D + 4 * cq);
        FN_UNROLL
        for (int id = 0; id < TD; ++id) {
            const ushort a = f2bf(O[id][4 * i] * inv), b = f2bf(O[id][4 * i + 1] * inv);
            const ushort c2 = f2bf(O[id][4 * i + 2] * inv), d2 = f2bf(O[id][4 * i + 3] * inv);
            dst[id * 4] = uint2((uint)a | ((uint)b << 16), (uint)c2 | ((uint)d2 << 16));
        }
    }
}

#define NAX_FLASH_INST(NAME, D, BK, NSG)                                                                              \
    template [[host_name(NAME)]] kernel void attn_nax<D, BK, NSG>(                                                    \
        device const ushort*, device const ushort*, device const ushort*, device ushort*, constant OpP&, uint2, uint2, \
        uint, uint);

// Keep in step with NAX_FLASH_VARIANTS in src/flash_nax.rs. NSG = 4 serves groups 1, 2, 4; NSG = 8 serves group 8.
NAX_FLASH_INST("attn_nax_d128_s4", 128, 32, 4)
NAX_FLASH_INST("attn_nax_d128_s8", 128, 32, 8)
