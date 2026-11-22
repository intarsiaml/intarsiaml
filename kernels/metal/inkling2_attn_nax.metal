// Inkling-2-Mini RelSinkAttn prefill on the M5 neural accelerators (lane ink-m-attn; Metal 4, compiled -std=metal4.0 into
// intarsia_metal4.metallib, build.rs METAL4_SOURCES; launcher crates/intarsia-backend-metal/src/inkattn.rs). Contract:
// crates/intarsia-core/src/inkling2/ops.rs (RelSinkAttn).
//
// The structure is attention_flash_nax.metal's attn_nax<128, 32, NSG> (16x16 fragments, matmul2d 16x32x16 on simdgroup
// cooperative tensors, full key tiles through cooperative_tensor::load, the partial tile through bounds-checked lane loads,
// bf16 P with l summed over the rounded weights, key tiles at absolute multiples of 32), with the Inkling additions:
//
// * keys come from one contiguous source per kv head, `src [Hk, S, D]` whose row r is absolute key kbase + r (kbase a
//   multiple of 32): for a linear cache the cache itself, after the launcher wrote the chunk's rows into it (rows pos0 .. are
//   not history, so writing them first changes nothing the op reads); for a ring the scratch staging rows that
//   ink_kv_stage_bf16 fills with [zeros below the first row's window | ring tail | chunk];
// * the window (keys p - window < j <= p) and causal mask, -inf;
// * the rel bias rel[t, h, p - j] for p - j < E, gathered per score element (log2 domain, not scaled by `scale`);
// * the online softmax starts from the sink: m = sink * log2(e), l = 1;
// * the epilogue multiplies by sigmoid(gate).
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#include "common.h"
using namespace metal;

#define IN_UNROLL _Pragma("clang loop unroll(full)")

template <typename T>
using ifrag_t = vec<T, 8>;

// mirrored by `InkP` in inkattn.rs; this kernel reads rr as kbase and cr as S (rows of the key source)
struct InkP {
    uint t, h, hk, d;
    uint dr, e, k, rr;
    uint cr, pos0, use_dev, flags;
    uint tau_floor, window, nq, pw;
    float qk_clip, rel_clip, eps, tau_alpha;
    float scale, cscale, f2, f3;
    uint sk[16];
};

constant constexpr float INX_LOG2E = 1.4426950408889634f;

template <bool TB, typename AT, typename BT>
METAL_FUNC void inx_mma(thread ifrag_t<float>& C0, thread ifrag_t<float>& C1, thread const ifrag_t<AT>& A,
                        thread const ifrag_t<BT>& B0, thread const ifrag_t<BT>& B1) {
    constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
        16, 32, 16, false, TB, false, mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    mpp::tensor_ops::matmul2d<desc, execution_simdgroup> op;
    auto ct_a = op.template get_left_input_cooperative_tensor<AT, BT, float>();
    auto ct_b = op.template get_right_input_cooperative_tensor<AT, BT, float>();
    auto ct_c = op.template get_destination_cooperative_tensor<decltype(ct_a), decltype(ct_b), float>();
    IN_UNROLL
    for (short i = 0; i < 8; ++i) {
        ct_a[i] = A[i];
        ct_b[i] = B0[i];
        ct_b[8 + i] = B1[i];
        ct_c[i] = C0[i];
        ct_c[8 + i] = C1[i];
    }
    op.run(ct_a, ct_b, ct_c);
    IN_UNROLL
    for (short i = 0; i < 8; ++i) {
        C0[i] = ct_c[i];
        C1[i] = ct_c[8 + i];
    }
}

template <bool TB, typename AT, typename TS>
METAL_FUNC void inx_mma_ct(thread ifrag_t<float>& C0, thread ifrag_t<float>& C1, thread const ifrag_t<AT>& A, TS src) {
    constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
        16, 32, 16, false, TB, false, mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    mpp::tensor_ops::matmul2d<desc, execution_simdgroup> op;
    auto ct_a = op.template get_left_input_cooperative_tensor<AT, bfloat, float>();
    auto ct_b = op.template get_right_input_cooperative_tensor<AT, bfloat, float>();
    auto ct_c = op.template get_destination_cooperative_tensor<decltype(ct_a), decltype(ct_b), float>();
    ct_b.load(src);
    IN_UNROLL
    for (short i = 0; i < 8; ++i) {
        ct_a[i] = A[i];
        ct_c[i] = C0[i];
        ct_c[8 + i] = C1[i];
    }
    op.run(ct_a, ct_b, ct_c);
    IN_UNROLL
    for (short i = 0; i < 8; ++i) {
        C0[i] = ct_c[i];
        C1[i] = ct_c[8 + i];
    }
}

METAL_FUNC ifrag_t<bfloat> inx_ld(device const ushort* r0, bool ok0, device const ushort* r1, bool ok1) {
    const uint2 a = ok0 ? *(device const uint2*)r0 : uint2(0);
    const uint2 b = ok1 ? *(device const uint2*)r1 : uint2(0);
    const bfloat4 al = as_type<bfloat4>(a), bl = as_type<bfloat4>(b);
    ifrag_t<bfloat> f;
    IN_UNROLL
    for (short j = 0; j < 4; ++j) {
        f[j] = al[j];
        f[4 + j] = bl[j];
    }
    return f;
}

// bind: 0 q [T,H,128], 1 k source [Hk,S,128], 2 v source, 3 rel [T,H,E], 4 pq, 5 out [T,H*128], 6 InkP.
// Grid (ceil(T / BQ), Hk), NSG simdgroups = G heads x NSG / G slabs of 16 rows. q / out 8-byte, sources 16-byte aligned.
template <int NSG>
[[kernel, max_total_threads_per_threadgroup(NSG * 32)]] void ink_attn_nax(
    device const ushort* q [[buffer(0)]],
    device const ushort* ks [[buffer(1)]],
    device const ushort* vs [[buffer(2)]],
    device const ushort* rel_ [[buffer(3)]],
    device const ushort* pq_ [[buffer(4)]],
    device ushort* out [[buffer(5)]],
    constant InkP& p [[buffer(6)]],
    uint2 tg [[threadgroup_position_in_grid]],
    uint2 grid [[threadgroups_per_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    constexpr int D = 128, BK = 32;
    constexpr int TD = D / 16;
    constexpr int TK = BK / 16;
    device const ushort* rel = rel_ + p.sk[3];
    device const ushort* pq = pq_ + p.sk[4];

    const uint T = p.t, H = p.h, Hk = p.hk, E = p.e;
    const uint S = p.cr, kbase = p.rr, pos0 = p.pos0, window = p.window;
    const uint G = H / Hk;
    const uint PQ = NSG / G;
    const uint BQ = 16 * PQ;
    const uint qb = grid.x - 1 - tg.x;
    const uint kvh = tg.y;
    const uint hg = sg / PQ, pqs = sg % PQ;
    const uint head = kvh * G + hg;
    const bool live = hg < G;
    const uint qbase = qb * BQ + pqs * 16;
    const uint kv_len = pos0 + T - kbase;  // valid source rows
    // the threadgroup's query positions and the keys any of them sees
    const uint first_row = pos0 + qb * BQ;
    const uint last_row = pos0 + min(T, (qb + 1) * BQ) - 1;
    const uint klo = (window > 0 && first_row + 1 > window) ? first_row + 1 - window : 0u;
    const uint kt0 = (max(klo, kbase) - kbase) / BK;
    const uint kt1 = (last_row + 1 - kbase + BK - 1) / BK;

    const uint qid = lane / 4;
    const uint fm = (qid & 4) + ((lane / 2) % 4);
    const uint cq = (qid & 2) + (lane & 1);

    device bfloat* kb = (device bfloat*)(ks + (ulong)kvh * S * D);
    device bfloat* vb = (device bfloat*)(vs + (ulong)kvh * S * D);
    tensor<device bfloat, dextents<int, 2>, tensor_inline> tK(kb, dextents<int, 2>(D, (int)kv_len));
    tensor<device bfloat, dextents<int, 2>, tensor_inline> tV(vb, dextents<int, 2>(D, (int)kv_len));

    const uint r0i = qbase + fm, r1i = qbase + fm + 8;  // chunk rows of the lane's two fragment rows
    const bool ok0 = r0i < T && live, ok1 = r1i < T && live;
    ifrag_t<bfloat> Qf[TD];
    IN_UNROLL
    for (int id = 0; id < TD; ++id) {
        const uint2 a = ok0 ? *(device const uint2*)(q + ((ulong)r0i * H + head) * D + 16 * id + 4 * cq) : uint2(0);
        const uint2 b = ok1 ? *(device const uint2*)(q + ((ulong)r1i * H + head) * D + 16 * id + 4 * cq) : uint2(0);
        const bfloat4 al = as_type<bfloat4>(a), bl = as_type<bfloat4>(b);
        IN_UNROLL
        for (short j = 0; j < 4; ++j) {
            Qf[id][j] = al[j];
            Qf[id][4 + j] = bl[j];
        }
    }

    ifrag_t<float> O[TD];
    IN_UNROLL
    for (int i = 0; i < TD; ++i) O[i] = ifrag_t<float>(0.0f);
    const uint sink_off = 2 * H * D + H * p.dr + head, gate_off = H * D + H * p.dr + head * D;
    // the online softmax starts from the sink logit: m = sink, l = e^0
    float m0 = ok0 ? bf2f(pq[(ulong)r0i * p.nq + sink_off]) * INX_LOG2E : 0.0f;
    float m1 = ok1 ? bf2f(pq[(ulong)r1i * p.nq + sink_off]) * INX_LOG2E : 0.0f;
    // a row's l is the sum of its four lanes' partials (lanes differing in bits 0 and 3): the sink's e^0 goes in one of them
    const float l_init = (lane & 9u) == 0 ? 1.0f : 0.0f;
    float l0 = l_init, l1 = l_init;
    const float c = p.scale * INX_LOG2E;
    const uint row0 = pos0 + r0i, row1 = pos0 + r1i;
    device const ushort* rel0 = rel + ((ulong)r0i * H + head) * E;
    device const ushort* rel1 = rel + ((ulong)r1i * H + head) * E;

    for (uint kt = kt0; kt < kt1; ++kt) {
        const int key0 = (int)(kt * BK);  // source row of the tile
        ifrag_t<float> Sf[TK];
        IN_UNROLL
        for (int j = 0; j < TK; ++j) Sf[j] = ifrag_t<float>(0.0f);
        const bool full = (uint)key0 + BK <= kv_len;
        if (full) {
            IN_UNROLL
            for (int kp = 0; kp < TK / 2; ++kp) {
                IN_UNROLL
                for (int id = 0; id < TD; ++id) {
                    auto sk = tK.template slice<16, 32>(16 * id, key0 + 32 * kp);
                    inx_mma_ct<true>(Sf[2 * kp], Sf[2 * kp + 1], Qf[id], sk);
                }
            }
        } else {
            IN_UNROLL
            for (int kp = 0; kp < TK / 2; ++kp) {
                const uint ka = (uint)key0 + 32 * kp + fm, kb2 = ka + 16;
                IN_UNROLL
                for (int id = 0; id < TD; ++id) {
                    const uint co = 16 * id + 4 * cq;
                    const ifrag_t<bfloat> b0 = inx_ld((device const ushort*)kb + (ulong)ka * D + co, ka < kv_len,
                                                      (device const ushort*)kb + (ulong)(ka + 8) * D + co, ka + 8 < kv_len);
                    const ifrag_t<bfloat> b1 = inx_ld((device const ushort*)kb + (ulong)kb2 * D + co, kb2 < kv_len,
                                                      (device const ushort*)kb + (ulong)(kb2 + 8) * D + co, kb2 + 8 < kv_len);
                    inx_mma<true>(Sf[2 * kp], Sf[2 * kp + 1], Qf[id], b0, b1);
                }
            }
        }

        float mx0 = -INFINITY, mx1 = -INFINITY;
        IN_UNROLL
        for (int f = 0; f < TK; ++f) {
            IN_UNROLL
            for (int e = 0; e < 8; ++e) {
                const bool hi = (e & 4) != 0;
                const uint row = hi ? row1 : row0;
                const bool rok = hi ? ok1 : ok0;
                const uint key = kbase + (uint)key0 + 16 * f + 4 * cq + (e & 3);
                const uint dist = row - key;  // wraps when key > row (masked)
                const bool vis = rok && key <= row && (window == 0 || dist < window);
                float s = -INFINITY;
                if (vis) {
                    s = Sf[f][e] * c;
                    if (dist < E) s += bf2f((hi ? rel1 : rel0)[dist]) * INX_LOG2E;
                }
                Sf[f][e] = s;
                if (hi) mx1 = max(mx1, s); else mx0 = max(mx0, s);
            }
        }
        mx0 = max(mx0, simd_shuffle_xor(mx0, ushort(1)));
        mx0 = max(mx0, simd_shuffle_xor(mx0, ushort(8)));
        mx1 = max(mx1, simd_shuffle_xor(mx1, ushort(1)));
        mx1 = max(mx1, simd_shuffle_xor(mx1, ushort(8)));
        const float mn0 = max(m0, mx0), mn1 = max(m1, mx1);
        const float f0 = fast::exp2(m0 - mn0), f1 = fast::exp2(m1 - mn1);
        m0 = mn0;
        m1 = mn1;
        float ls0 = 0.0f, ls1 = 0.0f;
        ifrag_t<bfloat> Pf[TK];
        IN_UNROLL
        for (int f = 0; f < TK; ++f) {
            IN_UNROLL
            for (int e = 0; e < 8; ++e) {
                Pf[f][e] = bfloat(fast::exp2(Sf[f][e] - ((e & 4) ? mn1 : mn0)));
                const float pr = float(Pf[f][e]);
                if (e & 4) ls1 += pr; else ls0 += pr;
            }
        }
        l0 = l0 * f0 + ls0;
        l1 = l1 * f1 + ls1;
        IN_UNROLL
        for (int i = 0; i < TD; ++i) {
            IN_UNROLL
            for (int e = 0; e < 8; ++e) O[i][e] *= (e & 4) ? f1 : f0;
        }

        if (full) {
            IN_UNROLL
            for (int kk = 0; kk < TK; ++kk) {
                IN_UNROLL
                for (int idp = 0; idp < TD / 2; ++idp) {
                    auto sv = tV.template slice<32, 16>(32 * idp, key0 + 16 * kk);
                    inx_mma_ct<false>(O[2 * idp], O[2 * idp + 1], Pf[kk], sv);
                }
            }
        } else {
            IN_UNROLL
            for (int kk = 0; kk < TK; ++kk) {
                const uint va = (uint)key0 + 16 * kk + fm;
                IN_UNROLL
                for (int idp = 0; idp < TD / 2; ++idp) {
                    const uint co = 32 * idp + 4 * cq;
                    const ifrag_t<bfloat> v0 = inx_ld((device const ushort*)vb + (ulong)va * D + co, va < kv_len,
                                                      (device const ushort*)vb + (ulong)(va + 8) * D + co, va + 8 < kv_len);
                    const ifrag_t<bfloat> v1 = inx_ld((device const ushort*)vb + (ulong)va * D + co + 16, va < kv_len,
                                                      (device const ushort*)vb + (ulong)(va + 8) * D + co + 16, va + 8 < kv_len);
                    inx_mma<false>(O[2 * idp], O[2 * idp + 1], Pf[kk], v0, v1);
                }
            }
        }
    }

    l0 += simd_shuffle_xor(l0, ushort(1));
    l0 += simd_shuffle_xor(l0, ushort(8));
    l1 += simd_shuffle_xor(l1, ushort(1));
    l1 += simd_shuffle_xor(l1, ushort(8));
    const float inv0 = 1.0f / l0, inv1 = 1.0f / l1;
    IN_UNROLL
    for (int i = 0; i < 2; ++i) {
        const uint ri = i ? r1i : r0i;
        if (!(i ? ok1 : ok0)) continue;
        const float inv = i ? inv1 : inv0;
        device const ushort* gate = pq + (ulong)ri * p.nq + gate_off + 4 * cq;
        device uint2* dst = (device uint2*)(out + ((ulong)ri * H + head) * D + 4 * cq);
        IN_UNROLL
        for (int id = 0; id < TD; ++id) {
            ushort o4[4];
            for (int x = 0; x < 4; ++x) {
                const float gz = bf2f(gate[16 * id + x]);
                const float sg2 = 1.0f / (1.0f + precise::exp(-gz));
                o4[x] = f2bf(O[id][4 * i + x] * inv * sg2);
            }
            dst[id * 4] = uint2((uint)o4[0] | ((uint)o4[1] << 16), (uint)o4[2] | ((uint)o4[3] << 16));
        }
    }
}

#define INX_INST(NAME, NSG)                                                                                           \
    template [[host_name(NAME)]] kernel void ink_attn_nax<NSG>(device const ushort*, device const ushort*,           \
                                                               device const ushort*, device const ushort*,           \
                                                               device const ushort*, device ushort*, constant InkP&, \
                                                               uint2, uint2, uint, uint);
INX_INST("ink_attn_nax_s6", 6)
INX_INST("ink_attn_nax_s12", 12)

// staging of a ring prefill's keys: row r of [Hk, S, D] is absolute key kbase + r: zero below lo (the first row's window),
// the ring row for lo <= j < pos0, the chunk row for j >= pos0. bind: 0 k chunk, 1 v chunk, 2 k cache, 3 v cache, 4 k stage,
// 5 v stage, 6 InkP (rr = kbase, f2 bits = lo, cr = ring rows, k = S). One thread per (kv head, row, 8 elements).
kernel void ink_kv_stage_bf16(device const ushort* kn [[buffer(0)]],
                              device const ushort* vn [[buffer(1)]],
                              device const ushort* kc [[buffer(2)]],
                              device const ushort* vc [[buffer(3)]],
                              device ushort* kst [[buffer(4)]],
                              device ushort* vst [[buffer(5)]],
                              constant InkP& p [[buffer(6)]],
                              uint gid [[thread_position_in_grid]]) {
    const uint D = p.d, S = p.k, Cr = p.cr, Hk = p.hk;
    const uint per = D / 8;
    if (gid >= Hk * S * per) return;
    const uint kvh = gid / (S * per), r = (gid / per) % S, c8 = gid % per;
    const uint j = p.rr + r;
    const uint lo = as_type<uint>(p.f2);
    uint4 kw = uint4(0), vw = uint4(0);
    if (j >= p.pos0) {
        const ulong o = ((ulong)(j - p.pos0) * Hk + kvh) * D + c8 * 8;
        kw = *(device const uint4*)(kn + o);
        vw = *(device const uint4*)(vn + o);
    } else if (j >= lo) {
        const ulong o = ((ulong)kvh * Cr + j % Cr) * D + c8 * 8;
        kw = *(device const uint4*)(kc + o);
        vw = *(device const uint4*)(vc + o);
    }
    const ulong d = ((ulong)kvh * S + r) * D + c8 * 8;
    *(device uint4*)(kst + d) = kw;
    *(device uint4*)(vst + d) = vw;
}
