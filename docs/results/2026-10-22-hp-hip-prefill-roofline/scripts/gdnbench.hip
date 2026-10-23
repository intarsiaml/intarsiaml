// Lane hp: GatedDeltaNet delta-rule scan variants against gdn_delta_rows_v2_kernel (kernels/hip/gdn.hip, included below),
// at the 35B's head layout (Hk 16, Hv 32, dv = dk = 128). Every variant must give v2's bits (output rows and final state);
// each is timed per token at T = 683 (one SPLIT_US piece of a 4096-token chunk) and 4096.
// Build (dev image): hipcc -O3 --offload-arch=gfx1201 -I kernels/hip -I kernels gdnbench.hip kernels/hip/ik_runtime.hip -o gdnbench
#include "gdn.hip"
#include <stdio.h>
#include <string.h>
#include <vector>

namespace {

// ---------------------------------------------------------------------------------------------------------------------
// Variant A ("v3i"): v2's lane map (one wave = 16 value rows, two lanes per row, 64 state elements per lane) and v2's float
// operations in v2's order, scheduled differently:
//  * the output dot of token t and the k dot of token t + 1 read the same state (after t's update, then decayed by t + 1's
//    gate), so they run in one loop over the state: two independent dependency chains instead of two loops in series;
//  * q / k of a token are normalised and staged into LDS two tokens ahead, in a ring of four buffers, so the warp sums,
//    rsqrt and LDS round trip of a token are off the state's critical path; the block is one wave and LDS operations of
//    one wave are performed in order, so no barrier orders the ring (a buffer is rewritten two tokens after its last read);
//  * raw inputs are loaded four tokens ahead of their staging (ring of four).
// ---------------------------------------------------------------------------------------------------------------------
template <int DK, bool NOLDS = false>  // NOLDS: q / k of the dots from registers (timing probe, wrong numbers)
__global__ void __launch_bounds__(32) gdn_v3i_kernel(const ik_bf16* __restrict__ qkv, const ik_bf16* __restrict__ a,
                                                      const ik_bf16* __restrict__ b, const ik_bf16* __restrict__ a_log,
                                                      const ik_bf16* __restrict__ dt_bias, float eps, ik_bf16* __restrict__ o,
                                                      float* __restrict__ state, int64_t T, int Hk, int Hv, int dv) {
    constexpr int SPLIT = 2, PL = DK / 32, EL = DK / SPLIT, ROWS = 32 / SPLIT, NB = 4;
    __shared__ __align__(16) float sq[NB][DK];
    __shared__ __align__(16) float sk[NB][DK];
    const int lane = threadIdx.x;
    const int part = lane % SPLIT;
    const int j = blockIdx.x * ROWS + lane / SPLIT;
    const bool valid = j < dv;
    const int jl = valid ? j : dv - 1;
    const int h = blockIdx.y;
    const int kh = h / (Hv / Hk);
    const int C = 2 * Hk * DK + Hv * dv;
    const float scale = rsqrtf((float)DK);
    const float neg_a = -expf(ik_b2f(a_log[h]));
    const float dtb = ik_b2f(dt_bias[h]);
    const int64_t last = T - 1;

    float s[EL];
    float4* sv = reinterpret_cast<float4*>(s);
    if (valid) {
        const float4* src = reinterpret_cast<const float4*>(state + ((int64_t)h * dv + j) * DK + part * EL);
#pragma unroll
        for (int i = 0; i < EL / 4; ++i) sv[i] = src[i];
    }
    auto load = [&](GdnRaw<PL>& r, int64_t t) {
        const ik_bf16* row = qkv + (t < last ? t : last) * C;
#pragma unroll
        for (int i = 0; i < PL; ++i) {
            r.q[i] = row[kh * DK + lane + 32 * i];
            r.k[i] = row[Hk * DK + kh * DK + lane + 32 * i];
        }
        r.v = row[2 * Hk * DK + h * dv + jl];
    };
    ik_bf16 ga, gb;
    auto load_gates = [&](int64_t g0) {
        const int64_t t = g0 + lane < last ? g0 + lane : last;
        ga = a[t * Hv + h];
        gb = b[t * Hv + h];
    };
    float cbeta, cdecay, nbeta, ndecay;  // gates of the current 32-token batch and of the next one (lane l: token l)
    auto gate_vals = [&](float& beta, float& decay) {
        const float ar = ik_b2f(ga), br = ik_b2f(gb);
        beta = sigmoidf(br);
        decay = expf(neg_a * softplus(ar + dtb));
    };

    GdnRaw<PL> ring[NB];
    float vr[NB];
    // stage token u (raw slot u & 3) into LDS buffer u & 3, keep its v, refill the slot with token u + 4
    auto stage = [&](int slot, int64_t u) {
        GdnRaw<PL>& r = ring[slot];
        float qr[PL], kr[PL];
#pragma unroll
        for (int i = 0; i < PL; ++i) {
            qr[i] = ik_b2f(r.q[i]);
            kr[i] = ik_b2f(r.k[i]);
        }
        vr[slot] = valid ? ik_b2f(r.v) : 0.f;
        load(r, u + NB);
        float qq = 0.f, kk = 0.f;
#pragma unroll
        for (int i = 0; i < PL; ++i) {
            qq += qr[i] * qr[i];
            kk += kr[i] * kr[i];
        }
        qq = gdn_warp_sum(qq);
        kk = gdn_warp_sum(kk);
        const float qn = rsqrtf(qq + eps) * scale, kn = rsqrtf(kk + eps);
#pragma unroll
        for (int i = 0; i < PL; ++i) {
            sq[slot][lane + 32 * i] = qr[i] * qn;
            sk[slot][lane + 32 * i] = kr[i] * kn;
        }
    };

#pragma unroll
    for (int d = 0; d < NB; ++d) load(ring[d], d);
    load_gates(0);
    gate_vals(cbeta, cdecay);
    load_gates(32);
    gate_vals(nbeta, ndecay);
    load_gates(64);

    float4 kreg[EL / 4];
    float kv = 0.f;
    stage(0, 0);
    stage(1, 1);
    {  // token 0's decay and k dot
        const float decay = __int_as_float(__builtin_amdgcn_readlane(__float_as_int(cdecay), 0));
        const float4* k4 = reinterpret_cast<const float4*>(sk[0]) + part * (EL / 4);
#pragma unroll
        for (int i = 0; i < EL / 4; ++i) {
            const float4 kk4 = k4[i];
            kreg[i] = kk4;
            sv[i].x *= decay; sv[i].y *= decay; sv[i].z *= decay; sv[i].w *= decay;
            kv += gdn_dot4(sv[i], kk4);
        }
    }

    // token t (slot t & 3): stage t + 2, finish t's k dot, update, output dot; with NEXT, t + 1's decay and k dot
    auto step = [&](int slot, int64_t t, auto next_c) {
        constexpr bool NEXT = decltype(next_c)::value;
        stage((slot + 2) & 3, t + 2);
        const int gl = (int)(t & 31);
        const float beta = __int_as_float(__builtin_amdgcn_readlane(__float_as_int(cbeta), gl));
        const int gn = (int)((t + 1) & 31);
        const float decn = gn == 0 ? __int_as_float(__builtin_amdgcn_readlane(__float_as_int(ndecay), 0))
                                   : __int_as_float(__builtin_amdgcn_readlane(__float_as_int(cdecay), gn));
        kv += gdn_xorrow<1>(kv);
        const float delta = (vr[slot] - kv) * beta;
        float out = 0.f, kvn = 0.f;
        const float4* q4 = reinterpret_cast<const float4*>(sq[slot]) + part * (EL / 4);
        const float4* k4n = reinterpret_cast<const float4*>(sk[(slot + 1) & 3]) + part * (EL / 4);
#pragma unroll
        for (int i = 0; i < EL / 4; ++i) {
            const float4 kk4 = kreg[i], qq4 = NOLDS ? make_float4(delta, kv, decn, beta) : q4[i];
            sv[i].x = __builtin_fmaf(kk4.x, delta, sv[i].x);
            sv[i].y = __builtin_fmaf(kk4.y, delta, sv[i].y);
            sv[i].z = __builtin_fmaf(kk4.z, delta, sv[i].z);
            sv[i].w = __builtin_fmaf(kk4.w, delta, sv[i].w);
            out += gdn_dot4(sv[i], qq4);
            if constexpr (NEXT) {
                const float4 kn4 = NOLDS ? make_float4(beta, decn, delta, kv) : k4n[i];
                kreg[i] = kn4;
                sv[i].x *= decn; sv[i].y *= decn; sv[i].z *= decn; sv[i].w *= decn;
                kvn += gdn_dot4(sv[i], kn4);
            }
        }
        out += gdn_xorrow<1>(out);
        if (valid && part == 0) o[(t * Hv + h) * dv + j] = ik_f2b(out);
        kv = kvn;
    };
    auto batch = [&](int64_t t0) {  // at a 32-token batch start (t0 > 0): shift the gates by one batch
        if ((t0 & 31) == 0 && t0 > 0) {
            cbeta = nbeta;
            cdecay = ndecay;
            gate_vals(nbeta, ndecay);
            load_gates(t0 + 64);
        }
    };
    const int64_t full = ((T - 1) / NB) * NB;  // tokens [0, full) all have a next token
    for (int64_t t0 = 0; t0 < full; t0 += NB) {
        batch(t0);
#pragma unroll
        for (int d = 0; d < NB; ++d) step(d, t0 + d, std::integral_constant<bool, true>{});
    }
    batch(full);
#pragma unroll
    for (int d = 0; d < NB; ++d) {
        const int64_t t = full + d;
        if (t + 1 < T) step(d, t, std::integral_constant<bool, true>{});
        else if (t < T) step(d, t, std::integral_constant<bool, false>{});
    }
    if (valid) {
        float4* dst = reinterpret_cast<float4*>(state + ((int64_t)h * dv + j) * DK + part * EL);
#pragma unroll
        for (int i = 0; i < EL / 4; ++i) dst[i] = sv[i];
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// Variant B ("v3r"): one lane per value row, all 128 state elements in its registers (32 rows per wave, 128 waves). The
// v2 pair sum (lane-local 64-element dot, then + the partner's) is a + b of the two 64-element dots, commutative, so
// summing the low and the high half as two chains and adding them gives v2's bits. Four independent chains per token
// (low/high of the output dot and of the next k dot); q / k reads are wave-uniform (every lane the same LDS address).
// ---------------------------------------------------------------------------------------------------------------------
template <int DK>
__global__ void __launch_bounds__(32) gdn_v3r_kernel(const ik_bf16* __restrict__ qkv, const ik_bf16* __restrict__ a,
                                                      const ik_bf16* __restrict__ b, const ik_bf16* __restrict__ a_log,
                                                      const ik_bf16* __restrict__ dt_bias, float eps, ik_bf16* __restrict__ o,
                                                      float* __restrict__ state, int64_t T, int Hk, int Hv, int dv) {
    constexpr int PL = DK / 32, EL = DK, HG = DK / 8, NB = 4;  // HG: float4 groups per half
    __shared__ __align__(16) float sq[NB][DK];
    __shared__ __align__(16) float sk[NB][DK];
    const int lane = threadIdx.x;
    const int j = blockIdx.x * 32 + lane;
    const bool valid = j < dv;
    const int jl = valid ? j : dv - 1;
    const int h = blockIdx.y;
    const int kh = h / (Hv / Hk);
    const int C = 2 * Hk * DK + Hv * dv;
    const float scale = rsqrtf((float)DK);
    const float neg_a = -expf(ik_b2f(a_log[h]));
    const float dtb = ik_b2f(dt_bias[h]);
    const int64_t last = T - 1;

    float s[EL];
    float4* sv = reinterpret_cast<float4*>(s);
    if (valid) {
        const float4* src = reinterpret_cast<const float4*>(state + ((int64_t)h * dv + j) * DK);
#pragma unroll
        for (int i = 0; i < EL / 4; ++i) sv[i] = src[i];
    }
    auto load = [&](GdnRaw<PL>& r, int64_t t) {
        const ik_bf16* row = qkv + (t < last ? t : last) * C;
#pragma unroll
        for (int i = 0; i < PL; ++i) {
            r.q[i] = row[kh * DK + lane + 32 * i];
            r.k[i] = row[Hk * DK + kh * DK + lane + 32 * i];
        }
        r.v = row[2 * Hk * DK + h * dv + jl];
    };
    ik_bf16 ga, gb;
    auto load_gates = [&](int64_t g0) {
        const int64_t t = g0 + lane < last ? g0 + lane : last;
        ga = a[t * Hv + h];
        gb = b[t * Hv + h];
    };
    float cbeta, cdecay, nbeta, ndecay;
    auto gate_vals = [&](float& beta, float& decay) {
        const float ar = ik_b2f(ga), br = ik_b2f(gb);
        beta = sigmoidf(br);
        decay = expf(neg_a * softplus(ar + dtb));
    };
    GdnRaw<PL> ring[NB];
    float vr[NB];
    auto stage = [&](int slot, int64_t u) {
        GdnRaw<PL>& r = ring[slot];
        float qr[PL], kr[PL];
#pragma unroll
        for (int i = 0; i < PL; ++i) {
            qr[i] = ik_b2f(r.q[i]);
            kr[i] = ik_b2f(r.k[i]);
        }
        vr[slot] = valid ? ik_b2f(r.v) : 0.f;
        load(r, u + NB);
        float qq = 0.f, kk = 0.f;
#pragma unroll
        for (int i = 0; i < PL; ++i) {
            qq += qr[i] * qr[i];
            kk += kr[i] * kr[i];
        }
        qq = gdn_warp_sum(qq);
        kk = gdn_warp_sum(kk);
        const float qn = rsqrtf(qq + eps) * scale, kn = rsqrtf(kk + eps);
#pragma unroll
        for (int i = 0; i < PL; ++i) {
            sq[slot][lane + 32 * i] = qr[i] * qn;
            sk[slot][lane + 32 * i] = kr[i] * kn;
        }
    };
#pragma unroll
    for (int d = 0; d < NB; ++d) load(ring[d], d);
    load_gates(0);
    gate_vals(cbeta, cdecay);
    load_gates(32);
    gate_vals(nbeta, ndecay);
    load_gates(64);

    float kvl = 0.f, kvh = 0.f;
    stage(0, 0);
    stage(1, 1);
    {
        const float decay = __int_as_float(__builtin_amdgcn_readlane(__float_as_int(cdecay), 0));
        const float4* k4 = reinterpret_cast<const float4*>(sk[0]);
#pragma unroll
        for (int i = 0; i < HG; ++i) {
            const float4 kl = k4[i], khh = k4[HG + i];
            sv[i].x *= decay; sv[i].y *= decay; sv[i].z *= decay; sv[i].w *= decay;
            sv[HG + i].x *= decay; sv[HG + i].y *= decay; sv[HG + i].z *= decay; sv[HG + i].w *= decay;
            kvl += gdn_dot4(sv[i], kl);
            kvh += gdn_dot4(sv[HG + i], khh);
        }
    }
    auto step = [&](int slot, int64_t t, auto next_c) {
        constexpr bool NEXT = decltype(next_c)::value;
        stage((slot + 2) & 3, t + 2);
        const int gl = (int)(t & 31);
        const float beta = __int_as_float(__builtin_amdgcn_readlane(__float_as_int(cbeta), gl));
        const int gn = (int)((t + 1) & 31);
        const float decn = gn == 0 ? __int_as_float(__builtin_amdgcn_readlane(__float_as_int(ndecay), 0))
                                   : __int_as_float(__builtin_amdgcn_readlane(__float_as_int(cdecay), gn));
        const float kv = kvl + kvh;
        const float delta = (vr[slot] - kv) * beta;
        float ol = 0.f, oh = 0.f, nl = 0.f, nh = 0.f;
        const float4* k4 = reinterpret_cast<const float4*>(sk[slot]);
        const float4* q4 = reinterpret_cast<const float4*>(sq[slot]);
        const float4* k4n = reinterpret_cast<const float4*>(sk[(slot + 1) & 3]);
#pragma unroll
        for (int i = 0; i < HG; ++i) {
#pragma unroll
            for (int half = 0; half < 2; ++half) {
                const int g = half * HG + i;
                const float4 kk4 = k4[g], qq4 = q4[g];
                sv[g].x = __builtin_fmaf(kk4.x, delta, sv[g].x);
                sv[g].y = __builtin_fmaf(kk4.y, delta, sv[g].y);
                sv[g].z = __builtin_fmaf(kk4.z, delta, sv[g].z);
                sv[g].w = __builtin_fmaf(kk4.w, delta, sv[g].w);
                (half ? oh : ol) += gdn_dot4(sv[g], qq4);
                if constexpr (NEXT) {
                    const float4 kn4 = k4n[g];
                    sv[g].x *= decn; sv[g].y *= decn; sv[g].z *= decn; sv[g].w *= decn;
                    (half ? nh : nl) += gdn_dot4(sv[g], kn4);
                }
            }
        }
        const float out = ol + oh;
        if (valid) o[(t * Hv + h) * dv + j] = ik_f2b(out);
        kvl = nl;
        kvh = nh;
    };
    auto batch = [&](int64_t t0) {
        if ((t0 & 31) == 0 && t0 > 0) {
            cbeta = nbeta;
            cdecay = ndecay;
            gate_vals(nbeta, ndecay);
            load_gates(t0 + 64);
        }
    };
    const int64_t full = ((T - 1) / NB) * NB;
    for (int64_t t0 = 0; t0 < full; t0 += NB) {
        batch(t0);
#pragma unroll
        for (int d = 0; d < NB; ++d) step(d, t0 + d, std::integral_constant<bool, true>{});
    }
    batch(full);
#pragma unroll
    for (int d = 0; d < NB; ++d) {
        const int64_t t = full + d;
        if (t + 1 < T) step(d, t, std::integral_constant<bool, true>{});
        else if (t < T) step(d, t, std::integral_constant<bool, false>{});
    }
    if (valid) {
        float4* dst = reinterpret_cast<float4*>(state + ((int64_t)h * dv + j) * DK);
#pragma unroll
        for (int i = 0; i < EL / 4; ++i) dst[i] = sv[i];
    }
}


// ---------------------------------------------------------------------------------------------------------------------
// Variant C ("v4"): v3r's lane map (one lane per value row, its 128 state elements in registers, 32 rows per wave, the low
// and high 64-element dots as separate chains: v2's pair sum is a + b of those two, commutative) without LDS. A token's
// normalised q / k vector lives in 8 registers per lane, lane l holding elements 8 * (l % 16) .. + 7 (both 16-lane rows the
// same), and every product takes its element straight from the owning lane with DPP row_share (the 16-lane broadcast), so
// no LDS traffic and no extra copy when the compiler folds the DPP move into the multiply. The norms are v2's: the same
// cooperative sum (lane L: elements L, L + 32, L + 64, L + 96, then the xor 16, 8, 4, 2, 1 butterfly) on a second, v2-layout
// load of the raw q / k; the products qr * qn and kr * kn are v2's operands, only computed in another lane.
// ---------------------------------------------------------------------------------------------------------------------
template <int E>
IK_DEVI float gdn_bc(const float (&vec)[8]) {  // element E of a token vector, in every lane
    static_assert(E >= 0 && E < 128, "element");
    return __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(vec[E & 7]), 0x150 + ((E >> 3) & 15), 0xf, 0xf, false));
}
template <int G>
IK_DEVI float4 gdn_bc4(const float (&vec)[8]) {
    return make_float4(gdn_bc<4 * G>(vec), gdn_bc<4 * G + 1>(vec), gdn_bc<4 * G + 2>(vec), gdn_bc<4 * G + 3>(vec));
}
struct GdnRaw4 {  // one token for one lane: v2-layout q / k (norms), this lane's 8-element q / k chunk, its row's v
    ik_bf16 q[4], k[4], v;
    uint4 qc, kc;
};
template <int G, int NG, bool NEXT>
IK_DEVI void gdn_v4_loop(float4* sv, const float (&kt)[8], const float (&qt)[8], const float (&kn)[8], float delta, float decn,
                         float& ol, float& oh, float& nl, float& nh) {
    if constexpr (G < NG) {
        const float4 kk4 = gdn_bc4<G>(kt), qq4 = gdn_bc4<G>(qt);
        sv[G].x = __builtin_fmaf(kk4.x, delta, sv[G].x);
        sv[G].y = __builtin_fmaf(kk4.y, delta, sv[G].y);
        sv[G].z = __builtin_fmaf(kk4.z, delta, sv[G].z);
        sv[G].w = __builtin_fmaf(kk4.w, delta, sv[G].w);
        (G < NG / 2 ? ol : oh) += gdn_dot4(sv[G], qq4);
        if constexpr (NEXT) {
            const float4 kn4 = gdn_bc4<G>(kn);
            sv[G].x *= decn; sv[G].y *= decn; sv[G].z *= decn; sv[G].w *= decn;
            (G < NG / 2 ? nl : nh) += gdn_dot4(sv[G], kn4);
        }
        gdn_v4_loop<G + 1, NG, NEXT>(sv, kt, qt, kn, delta, decn, ol, oh, nl, nh);
    }
}
template <int G, int NG>
IK_DEVI void gdn_v4_first(float4* sv, const float (&k0)[8], float decay, float& nl, float& nh) {
    if constexpr (G < NG) {
        const float4 kk4 = gdn_bc4<G>(k0);
        sv[G].x *= decay; sv[G].y *= decay; sv[G].z *= decay; sv[G].w *= decay;
        (G < NG / 2 ? nl : nh) += gdn_dot4(sv[G], kk4);
        gdn_v4_first<G + 1, NG>(sv, k0, decay, nl, nh);
    }
}

__global__ void __launch_bounds__(32) gdn_v4_kernel(const ik_bf16* __restrict__ qkv, const ik_bf16* __restrict__ a,
                                                     const ik_bf16* __restrict__ b, const ik_bf16* __restrict__ a_log,
                                                     const ik_bf16* __restrict__ dt_bias, float eps, ik_bf16* __restrict__ o,
                                                     float* __restrict__ state, int64_t T, int Hk, int Hv, int dv) {
    constexpr int DK = 128, PL = 4, NG = DK / 4, NR = 2;  // NR: raw ring (tokens loaded ahead of their staging)
    const int lane = threadIdx.x;
    const int j = blockIdx.x * 32 + lane;
    const bool valid = j < dv;
    const int jl = valid ? j : dv - 1;
    const int h = blockIdx.y;
    const int kh = h / (Hv / Hk);
    const int C = 2 * Hk * DK + Hv * dv;
    const float scale = rsqrtf((float)DK);
    const float neg_a = -expf(ik_b2f(a_log[h]));
    const float dtb = ik_b2f(dt_bias[h]);
    const int64_t last = T - 1;
    const int c8 = 8 * (lane & 15);

    float s[DK];
    float4* sv = reinterpret_cast<float4*>(s);
    if (valid) {
        const float4* src = reinterpret_cast<const float4*>(state + ((int64_t)h * dv + j) * DK);
#pragma unroll
        for (int i = 0; i < NG; ++i) sv[i] = src[i];
    }
    auto load = [&](GdnRaw4& r, int64_t t) {
        const ik_bf16* row = qkv + (t < last ? t : last) * C;
#pragma unroll
        for (int i = 0; i < PL; ++i) {
            r.q[i] = row[kh * DK + lane + 32 * i];
            r.k[i] = row[Hk * DK + kh * DK + lane + 32 * i];
        }
        r.qc = *reinterpret_cast<const uint4*>(row + kh * DK + c8);
        r.kc = *reinterpret_cast<const uint4*>(row + Hk * DK + kh * DK + c8);
        r.v = row[2 * Hk * DK + h * dv + jl];
    };
    ik_bf16 ga, gb;
    auto load_gates = [&](int64_t g0) {
        const int64_t t = g0 + lane < last ? g0 + lane : last;
        ga = a[t * Hv + h];
        gb = b[t * Hv + h];
    };
    float cbeta, cdecay, nbeta, ndecay;
    auto gate_vals = [&](float& beta, float& decay) {
        const float ar = ik_b2f(ga), br = ik_b2f(gb);
        beta = sigmoidf(br);
        decay = expf(neg_a * softplus(ar + dtb));
    };
    // normalise raw slot r into q / k vectors (8 elements each) and v; refill the slot with token u + NR
    auto stage = [&](GdnRaw4& r, int64_t u, float (&qv)[8], float (&kv8)[8], float& v) {
        float qr[PL], kr[PL];
#pragma unroll
        for (int i = 0; i < PL; ++i) {
            qr[i] = ik_b2f(r.q[i]);
            kr[i] = ik_b2f(r.k[i]);
        }
        float qcf[8], kcf[8];
        ik_unpack8(r.qc, qcf);
        ik_unpack8(r.kc, kcf);
        v = valid ? ik_b2f(r.v) : 0.f;
        load(r, u + NR);
        float qq = 0.f, kk = 0.f;
#pragma unroll
        for (int i = 0; i < PL; ++i) {
            qq += qr[i] * qr[i];
            kk += kr[i] * kr[i];
        }
        qq = gdn_warp_sum(qq);
        kk = gdn_warp_sum(kk);
        const float qn = rsqrtf(qq + eps) * scale, kn = rsqrtf(kk + eps);
#pragma unroll
        for (int m = 0; m < 8; ++m) {
            qv[m] = qcf[m] * qn;
            kv8[m] = kcf[m] * kn;
        }
    };

    GdnRaw4 ring[NR];
#pragma unroll
    for (int d = 0; d < NR; ++d) load(ring[d], d);
    load_gates(0);
    gate_vals(cbeta, cdecay);
    load_gates(32);
    gate_vals(nbeta, ndecay);
    load_gates(64);

    // vectors: q of tokens t and t + 1, k of tokens t and t + 1 (two-token ring, index t & 1), v likewise
    float qv[2][8], kvv[2][8], vv[2];
    stage(ring[0], 0, qv[0], kvv[0], vv[0]);
    stage(ring[1], 1, qv[1], kvv[1], vv[1]);
    float kvl = 0.f, kvh = 0.f;
    gdn_v4_first<0, NG>(sv, kvv[0], __int_as_float(__builtin_amdgcn_readlane(__float_as_int(cdecay), 0)), kvl, kvh);

    auto step = [&](int sl, int64_t t, auto next_c) {  // sl = t & 1
        constexpr bool NEXT = decltype(next_c)::value;
        const int gl = (int)(t & 31);
        const float beta = __int_as_float(__builtin_amdgcn_readlane(__float_as_int(cbeta), gl));
        const int gn = (int)((t + 1) & 31);
        const float decn = gn == 0 ? __int_as_float(__builtin_amdgcn_readlane(__float_as_int(ndecay), 0))
                                   : __int_as_float(__builtin_amdgcn_readlane(__float_as_int(cdecay), gn));
        const float kv = kvl + kvh;
        const float delta = (vv[sl] - kv) * beta;
        float ol = 0.f, oh = 0.f, nl = 0.f, nh = 0.f;
        if (sl == 0) gdn_v4_loop<0, NG, NEXT>(sv, kvv[0], qv[0], kvv[1], delta, decn, ol, oh, nl, nh);
        else gdn_v4_loop<0, NG, NEXT>(sv, kvv[1], qv[1], kvv[0], delta, decn, ol, oh, nl, nh);
        const float out = ol + oh;
        if (valid) o[(t * Hv + h) * dv + j] = ik_f2b(out);
        kvl = nl;
        kvh = nh;
        // token t + 2 replaces token t in the vector ring (its raw slot is t & 1 too: NR = 2)
        stage(ring[sl], t + 2, qv[sl], kvv[sl], vv[sl]);
    };
    auto batch = [&](int64_t t0) {
        if ((t0 & 31) == 0 && t0 > 0) {
            cbeta = nbeta;
            cdecay = ndecay;
            gate_vals(nbeta, ndecay);
            load_gates(t0 + 64);
        }
    };
    const int64_t full = ((T - 1) / 2) * 2;
    for (int64_t t0 = 0; t0 < full; t0 += 2) {
        batch(t0);
        step(0, t0, std::integral_constant<bool, true>{});
        step(1, t0 + 1, std::integral_constant<bool, true>{});
    }
    batch(full);
    if (full + 1 < T) {
        step(0, full, std::integral_constant<bool, true>{});
        step(1, full + 1, std::integral_constant<bool, false>{});
    } else {
        step(0, full, std::integral_constant<bool, false>{});
    }
    if (valid) {
        float4* dst = reinterpret_cast<float4*>(state + ((int64_t)h * dv + j) * DK);
#pragma unroll
        for (int i = 0; i < NG; ++i) dst[i] = sv[i];
    }
}


// ---------------------------------------------------------------------------------------------------------------------
// Variant E ("v6"): v4 (one lane per value row, token vectors in 8 registers per lane, elements broadcast with DPP
// row_share) with the broadcast inside the arithmetic instruction itself (v_fmac_f32_dpp / v_mul_f32_dpp, inline asm:
// the compiler folds a DPP move into v_mul / v_add but not into v_fmac). fmac computes S0 * S1 + D with one rounding:
// fma(k, x, d) = fma(x, k, d), so every value is v2's.
// ---------------------------------------------------------------------------------------------------------------------
template <int N>
IK_DEVI float gdn_fmac_bc(float acc, float k, float x) {  // acc + k[lane N of the row] * x, one rounding
    asm("v_fmac_f32_dpp %0, %1, %2 row_share:%3 row_mask:0xf bank_mask:0xf" : "+v"(acc) : "v"(k), "v"(x), "n"(N));
    return acc;
}
template <int N>
IK_DEVI float gdn_mul_bc(float k, float x) {  // k[lane N of the row] * x
    float d;
    asm("v_mul_f32_dpp %0, %1, %2 row_share:%3 row_mask:0xf bank_mask:0xf" : "=v"(d) : "v"(k), "v"(x), "n"(N));
    return d;
}
// gdn_dot4(s, vec[4G .. 4G + 3]) with the elements broadcast from their lanes
template <int G>
IK_DEVI float gdn_dot4_bc(const float4& s, const float (&vec)[8]) {
    constexpr int E = 4 * G, L = (E >> 3) & 15, R = E & 7;  // the 4 elements sit in one lane, registers R .. R + 3
    float t = gdn_mul_bc<L>(vec[R + 1], s.y);
    t = gdn_fmac_bc<L>(t, vec[R], s.x);
    t = gdn_fmac_bc<L>(t, vec[R + 2], s.z);
    return gdn_fmac_bc<L>(t, vec[R + 3], s.w);
}
template <int G, int NG, bool NEXT>
IK_DEVI void gdn_v6_loop(float4* sv, const float (&kt)[8], const float (&qt)[8], const float (&kn)[8], float delta, float decn,
                         float& ol, float& oh, float& nl, float& nh) {
    if constexpr (G < NG) {
        constexpr int L = ((4 * G) >> 3) & 15, R = (4 * G) & 7;
        sv[G].x = gdn_fmac_bc<L>(sv[G].x, kt[R], delta);
        sv[G].y = gdn_fmac_bc<L>(sv[G].y, kt[R + 1], delta);
        sv[G].z = gdn_fmac_bc<L>(sv[G].z, kt[R + 2], delta);
        sv[G].w = gdn_fmac_bc<L>(sv[G].w, kt[R + 3], delta);
        (G < NG / 2 ? ol : oh) += gdn_dot4_bc<G>(sv[G], qt);
        if constexpr (NEXT) {
            sv[G].x *= decn; sv[G].y *= decn; sv[G].z *= decn; sv[G].w *= decn;
            (G < NG / 2 ? nl : nh) += gdn_dot4_bc<G>(sv[G], kn);
        }
        gdn_v6_loop<G + 1, NG, NEXT>(sv, kt, qt, kn, delta, decn, ol, oh, nl, nh);
    }
}
template <int G, int NG>
IK_DEVI void gdn_v6_first(float4* sv, const float (&k0)[8], float decay, float& nl, float& nh) {
    if constexpr (G < NG) {
        sv[G].x *= decay; sv[G].y *= decay; sv[G].z *= decay; sv[G].w *= decay;
        (G < NG / 2 ? nl : nh) += gdn_dot4_bc<G>(sv[G], k0);
        gdn_v6_first<G + 1, NG>(sv, k0, decay, nl, nh);
    }
}
__global__ void __launch_bounds__(32) gdn_v6_kernel(const ik_bf16* __restrict__ qkv, const ik_bf16* __restrict__ a,
                                                     const ik_bf16* __restrict__ b, const ik_bf16* __restrict__ a_log,
                                                     const ik_bf16* __restrict__ dt_bias, float eps, ik_bf16* __restrict__ o,
                                                     float* __restrict__ state, int64_t T, int Hk, int Hv, int dv) {
    constexpr int DK = 128, PL = 4, NG = DK / 4, NR = 2;  // NR: raw ring (tokens loaded ahead of their staging)
    const int lane = threadIdx.x;
    const int j = blockIdx.x * 32 + lane;
    const bool valid = j < dv;
    const int jl = valid ? j : dv - 1;
    const int h = blockIdx.y;
    const int kh = h / (Hv / Hk);
    const int C = 2 * Hk * DK + Hv * dv;
    const float scale = rsqrtf((float)DK);
    const float neg_a = -expf(ik_b2f(a_log[h]));
    const float dtb = ik_b2f(dt_bias[h]);
    const int64_t last = T - 1;
    const int c8 = 8 * (lane & 15);

    float s[DK];
    float4* sv = reinterpret_cast<float4*>(s);
    if (valid) {
        const float4* src = reinterpret_cast<const float4*>(state + ((int64_t)h * dv + j) * DK);
#pragma unroll
        for (int i = 0; i < NG; ++i) sv[i] = src[i];
    }
    auto load = [&](GdnRaw4& r, int64_t t) {
        const ik_bf16* row = qkv + (t < last ? t : last) * C;
#pragma unroll
        for (int i = 0; i < PL; ++i) {
            r.q[i] = row[kh * DK + lane + 32 * i];
            r.k[i] = row[Hk * DK + kh * DK + lane + 32 * i];
        }
        r.qc = *reinterpret_cast<const uint4*>(row + kh * DK + c8);
        r.kc = *reinterpret_cast<const uint4*>(row + Hk * DK + kh * DK + c8);
        r.v = row[2 * Hk * DK + h * dv + jl];
    };
    ik_bf16 ga, gb;
    auto load_gates = [&](int64_t g0) {
        const int64_t t = g0 + lane < last ? g0 + lane : last;
        ga = a[t * Hv + h];
        gb = b[t * Hv + h];
    };
    float cbeta, cdecay, nbeta, ndecay;
    auto gate_vals = [&](float& beta, float& decay) {
        const float ar = ik_b2f(ga), br = ik_b2f(gb);
        beta = sigmoidf(br);
        decay = expf(neg_a * softplus(ar + dtb));
    };
    // normalise raw slot r into q / k vectors (8 elements each) and v; refill the slot with token u + NR
    auto stage = [&](GdnRaw4& r, int64_t u, float (&qv)[8], float (&kv8)[8], float& v) {
        float qr[PL], kr[PL];
#pragma unroll
        for (int i = 0; i < PL; ++i) {
            qr[i] = ik_b2f(r.q[i]);
            kr[i] = ik_b2f(r.k[i]);
        }
        float qcf[8], kcf[8];
        ik_unpack8(r.qc, qcf);
        ik_unpack8(r.kc, kcf);
        v = valid ? ik_b2f(r.v) : 0.f;
        load(r, u + NR);
        float qq = 0.f, kk = 0.f;
#pragma unroll
        for (int i = 0; i < PL; ++i) {
            qq += qr[i] * qr[i];
            kk += kr[i] * kr[i];
        }
        qq = gdn_warp_sum(qq);
        kk = gdn_warp_sum(kk);
        const float qn = rsqrtf(qq + eps) * scale, kn = rsqrtf(kk + eps);
#pragma unroll
        for (int m = 0; m < 8; ++m) {
            qv[m] = qcf[m] * qn;
            kv8[m] = kcf[m] * kn;
        }
    };

    GdnRaw4 ring[NR];
#pragma unroll
    for (int d = 0; d < NR; ++d) load(ring[d], d);
    load_gates(0);
    gate_vals(cbeta, cdecay);
    load_gates(32);
    gate_vals(nbeta, ndecay);
    load_gates(64);

    // vectors: q of tokens t and t + 1, k of tokens t and t + 1 (two-token ring, index t & 1), v likewise
    float qv[2][8], kvv[2][8], vv[2];
    stage(ring[0], 0, qv[0], kvv[0], vv[0]);
    stage(ring[1], 1, qv[1], kvv[1], vv[1]);
    float kvl = 0.f, kvh = 0.f;
    gdn_v6_first<0, NG>(sv, kvv[0], __int_as_float(__builtin_amdgcn_readlane(__float_as_int(cdecay), 0)), kvl, kvh);

    auto step = [&](int sl, int64_t t, auto next_c) {  // sl = t & 1
        constexpr bool NEXT = decltype(next_c)::value;
        const int gl = (int)(t & 31);
        const float beta = __int_as_float(__builtin_amdgcn_readlane(__float_as_int(cbeta), gl));
        const int gn = (int)((t + 1) & 31);
        const float decn = gn == 0 ? __int_as_float(__builtin_amdgcn_readlane(__float_as_int(ndecay), 0))
                                   : __int_as_float(__builtin_amdgcn_readlane(__float_as_int(cdecay), gn));
        const float kv = kvl + kvh;
        const float delta = (vv[sl] - kv) * beta;
        float ol = 0.f, oh = 0.f, nl = 0.f, nh = 0.f;
        if (sl == 0) gdn_v6_loop<0, NG, NEXT>(sv, kvv[0], qv[0], kvv[1], delta, decn, ol, oh, nl, nh);
        else gdn_v6_loop<0, NG, NEXT>(sv, kvv[1], qv[1], kvv[0], delta, decn, ol, oh, nl, nh);
        const float out = ol + oh;
        if (valid) o[(t * Hv + h) * dv + j] = ik_f2b(out);
        kvl = nl;
        kvh = nh;
        // token t + 2 replaces token t in the vector ring (its raw slot is t & 1 too: NR = 2)
        stage(ring[sl], t + 2, qv[sl], kvv[sl], vv[sl]);
    };
    auto batch = [&](int64_t t0) {
        if ((t0 & 31) == 0 && t0 > 0) {
            cbeta = nbeta;
            cdecay = ndecay;
            gate_vals(nbeta, ndecay);
            load_gates(t0 + 64);
        }
    };
    const int64_t full = ((T - 1) / 2) * 2;
    for (int64_t t0 = 0; t0 < full; t0 += 2) {
        batch(t0);
        step(0, t0, std::integral_constant<bool, true>{});
        step(1, t0 + 1, std::integral_constant<bool, true>{});
    }
    batch(full);
    if (full + 1 < T) {
        step(0, full, std::integral_constant<bool, true>{});
        step(1, full + 1, std::integral_constant<bool, false>{});
    } else {
        step(0, full, std::integral_constant<bool, false>{});
    }
    if (valid) {
        float4* dst = reinterpret_cast<float4*>(state + ((int64_t)h * dv + j) * DK);
#pragma unroll
        for (int i = 0; i < NG; ++i) dst[i] = sv[i];
    }
}


// v7: v6 with the low and high halves interleaved (group i and i + 16 together) and the per-group work in phases, so four
// dot chains and four accumulator chains are independent at every point (v6 ran the low half's 16 groups before the high's)
template <int I, int NG, bool NEXT>
IK_DEVI void gdn_v7_loop(float4* sv, const float (&kt)[8], const float (&qt)[8], const float (&kn)[8], float delta, float decn,
                         float& ol, float& oh, float& nl, float& nh) {
    if constexpr (I < NG / 2) {
        constexpr int A = I, B = I + NG / 2;
        constexpr int LA = ((4 * A) >> 3) & 15, RA = (4 * A) & 7, LB = ((4 * B) >> 3) & 15, RB = (4 * B) & 7;
        sv[A].x = gdn_fmac_bc<LA>(sv[A].x, kt[RA], delta);
        sv[B].x = gdn_fmac_bc<LB>(sv[B].x, kt[RB], delta);
        sv[A].y = gdn_fmac_bc<LA>(sv[A].y, kt[RA + 1], delta);
        sv[B].y = gdn_fmac_bc<LB>(sv[B].y, kt[RB + 1], delta);
        sv[A].z = gdn_fmac_bc<LA>(sv[A].z, kt[RA + 2], delta);
        sv[B].z = gdn_fmac_bc<LB>(sv[B].z, kt[RB + 2], delta);
        sv[A].w = gdn_fmac_bc<LA>(sv[A].w, kt[RA + 3], delta);
        sv[B].w = gdn_fmac_bc<LB>(sv[B].w, kt[RB + 3], delta);
        float qa = gdn_mul_bc<LA>(qt[RA + 1], sv[A].y), qb = gdn_mul_bc<LB>(qt[RB + 1], sv[B].y);
        float4 da, db;
        if constexpr (NEXT) {
            da = make_float4(sv[A].x * decn, sv[A].y * decn, sv[A].z * decn, sv[A].w * decn);
            db = make_float4(sv[B].x * decn, sv[B].y * decn, sv[B].z * decn, sv[B].w * decn);
        }
        qa = gdn_fmac_bc<LA>(qa, qt[RA], sv[A].x);
        qb = gdn_fmac_bc<LB>(qb, qt[RB], sv[B].x);
        float ka = 0.f, kb = 0.f;
        if constexpr (NEXT) {
            ka = gdn_mul_bc<LA>(kn[RA + 1], da.y);
            kb = gdn_mul_bc<LB>(kn[RB + 1], db.y);
        }
        qa = gdn_fmac_bc<LA>(qa, qt[RA + 2], sv[A].z);
        qb = gdn_fmac_bc<LB>(qb, qt[RB + 2], sv[B].z);
        if constexpr (NEXT) {
            ka = gdn_fmac_bc<LA>(ka, kn[RA], da.x);
            kb = gdn_fmac_bc<LB>(kb, kn[RB], db.x);
        }
        qa = gdn_fmac_bc<LA>(qa, qt[RA + 3], sv[A].w);
        qb = gdn_fmac_bc<LB>(qb, qt[RB + 3], sv[B].w);
        if constexpr (NEXT) {
            ka = gdn_fmac_bc<LA>(ka, kn[RA + 2], da.z);
            kb = gdn_fmac_bc<LB>(kb, kn[RB + 2], db.z);
            ka = gdn_fmac_bc<LA>(ka, kn[RA + 3], da.w);
            kb = gdn_fmac_bc<LB>(kb, kn[RB + 3], db.w);
            sv[A] = da;
            sv[B] = db;
        }
        ol += qa;
        oh += qb;
        if constexpr (NEXT) {
            nl += ka;
            nh += kb;
        }
        gdn_v7_loop<I + 1, NG, NEXT>(sv, kt, qt, kn, delta, decn, ol, oh, nl, nh);
    }
}
__global__ void __launch_bounds__(32) gdn_v7_kernel(const ik_bf16* __restrict__ qkv, const ik_bf16* __restrict__ a,
                                                     const ik_bf16* __restrict__ b, const ik_bf16* __restrict__ a_log,
                                                     const ik_bf16* __restrict__ dt_bias, float eps, ik_bf16* __restrict__ o,
                                                     float* __restrict__ state, int64_t T, int Hk, int Hv, int dv) {
    constexpr int DK = 128, PL = 4, NG = DK / 4, NR = 2;  // NR: raw ring (tokens loaded ahead of their staging)
    const int lane = threadIdx.x;
    const int j = blockIdx.x * 32 + lane;
    const bool valid = j < dv;
    const int jl = valid ? j : dv - 1;
    const int h = blockIdx.y;
    const int kh = h / (Hv / Hk);
    const int C = 2 * Hk * DK + Hv * dv;
    const float scale = rsqrtf((float)DK);
    const float neg_a = -expf(ik_b2f(a_log[h]));
    const float dtb = ik_b2f(dt_bias[h]);
    const int64_t last = T - 1;
    const int c8 = 8 * (lane & 15);

    float s[DK];
    float4* sv = reinterpret_cast<float4*>(s);
    if (valid) {
        const float4* src = reinterpret_cast<const float4*>(state + ((int64_t)h * dv + j) * DK);
#pragma unroll
        for (int i = 0; i < NG; ++i) sv[i] = src[i];
    }
    auto load = [&](GdnRaw4& r, int64_t t) {
        const ik_bf16* row = qkv + (t < last ? t : last) * C;
#pragma unroll
        for (int i = 0; i < PL; ++i) {
            r.q[i] = row[kh * DK + lane + 32 * i];
            r.k[i] = row[Hk * DK + kh * DK + lane + 32 * i];
        }
        r.qc = *reinterpret_cast<const uint4*>(row + kh * DK + c8);
        r.kc = *reinterpret_cast<const uint4*>(row + Hk * DK + kh * DK + c8);
        r.v = row[2 * Hk * DK + h * dv + jl];
    };
    ik_bf16 ga, gb;
    auto load_gates = [&](int64_t g0) {
        const int64_t t = g0 + lane < last ? g0 + lane : last;
        ga = a[t * Hv + h];
        gb = b[t * Hv + h];
    };
    float cbeta, cdecay, nbeta, ndecay;
    auto gate_vals = [&](float& beta, float& decay) {
        const float ar = ik_b2f(ga), br = ik_b2f(gb);
        beta = sigmoidf(br);
        decay = expf(neg_a * softplus(ar + dtb));
    };
    // normalise raw slot r into q / k vectors (8 elements each) and v; refill the slot with token u + NR
    auto stage = [&](GdnRaw4& r, int64_t u, float (&qv)[8], float (&kv8)[8], float& v) {
        float qr[PL], kr[PL];
#pragma unroll
        for (int i = 0; i < PL; ++i) {
            qr[i] = ik_b2f(r.q[i]);
            kr[i] = ik_b2f(r.k[i]);
        }
        float qcf[8], kcf[8];
        ik_unpack8(r.qc, qcf);
        ik_unpack8(r.kc, kcf);
        v = valid ? ik_b2f(r.v) : 0.f;
        load(r, u + NR);
        float qq = 0.f, kk = 0.f;
#pragma unroll
        for (int i = 0; i < PL; ++i) {
            qq += qr[i] * qr[i];
            kk += kr[i] * kr[i];
        }
        qq = gdn_warp_sum(qq);
        kk = gdn_warp_sum(kk);
        const float qn = rsqrtf(qq + eps) * scale, kn = rsqrtf(kk + eps);
#pragma unroll
        for (int m = 0; m < 8; ++m) {
            qv[m] = qcf[m] * qn;
            kv8[m] = kcf[m] * kn;
        }
    };

    GdnRaw4 ring[NR];
#pragma unroll
    for (int d = 0; d < NR; ++d) load(ring[d], d);
    load_gates(0);
    gate_vals(cbeta, cdecay);
    load_gates(32);
    gate_vals(nbeta, ndecay);
    load_gates(64);

    // vectors: q of tokens t and t + 1, k of tokens t and t + 1 (two-token ring, index t & 1), v likewise
    float qv[2][8], kvv[2][8], vv[2];
    stage(ring[0], 0, qv[0], kvv[0], vv[0]);
    stage(ring[1], 1, qv[1], kvv[1], vv[1]);
    float kvl = 0.f, kvh = 0.f;
    gdn_v6_first<0, NG>(sv, kvv[0], __int_as_float(__builtin_amdgcn_readlane(__float_as_int(cdecay), 0)), kvl, kvh);

    auto step = [&](int sl, int64_t t, auto next_c) {  // sl = t & 1
        constexpr bool NEXT = decltype(next_c)::value;
        const int gl = (int)(t & 31);
        const float beta = __int_as_float(__builtin_amdgcn_readlane(__float_as_int(cbeta), gl));
        const int gn = (int)((t + 1) & 31);
        const float decn = gn == 0 ? __int_as_float(__builtin_amdgcn_readlane(__float_as_int(ndecay), 0))
                                   : __int_as_float(__builtin_amdgcn_readlane(__float_as_int(cdecay), gn));
        const float kv = kvl + kvh;
        const float delta = (vv[sl] - kv) * beta;
        float ol = 0.f, oh = 0.f, nl = 0.f, nh = 0.f;
        if (sl == 0) gdn_v7_loop<0, NG, NEXT>(sv, kvv[0], qv[0], kvv[1], delta, decn, ol, oh, nl, nh);
        else gdn_v7_loop<0, NG, NEXT>(sv, kvv[1], qv[1], kvv[0], delta, decn, ol, oh, nl, nh);
        const float out = ol + oh;
        if (valid) o[(t * Hv + h) * dv + j] = ik_f2b(out);
        kvl = nl;
        kvh = nh;
        // token t + 2 replaces token t in the vector ring (its raw slot is t & 1 too: NR = 2)
        stage(ring[sl], t + 2, qv[sl], kvv[sl], vv[sl]);
    };
    auto batch = [&](int64_t t0) {
        if ((t0 & 31) == 0 && t0 > 0) {
            cbeta = nbeta;
            cdecay = ndecay;
            gate_vals(nbeta, ndecay);
            load_gates(t0 + 64);
        }
    };
    const int64_t full = ((T - 1) / 2) * 2;
    for (int64_t t0 = 0; t0 < full; t0 += 2) {
        batch(t0);
        step(0, t0, std::integral_constant<bool, true>{});
        step(1, t0 + 1, std::integral_constant<bool, true>{});
    }
    batch(full);
    if (full + 1 < T) {
        step(0, full, std::integral_constant<bool, true>{});
        step(1, full + 1, std::integral_constant<bool, false>{});
    } else {
        step(0, full, std::integral_constant<bool, false>{});
    }
    if (valid) {
        float4* dst = reinterpret_cast<float4*>(state + ((int64_t)h * dv + j) * DK);
#pragma unroll
        for (int i = 0; i < NG; ++i) dst[i] = sv[i];
    }
}


// v8: v7 with a token vector layout that also serves the norm: lane l (m = l % 16, both rows the same) holds elements
// m + 16 i in register i, so element e is register e / 16 of lane e % 16 (row_share), and v2's norm term of lane L
// (elements L, L + 32, L + 64, L + 96) is registers 0, 2, 4, 6 (L < 16) or 1, 3, 5, 7 (L >= 16) of its own lane: no
// second load, no second copy. Raw inputs one token ahead of their staging (fewer registers: no spill).
template <int E>
IK_DEVI float gdn8_fmac(float acc, const float (&vec)[8], float x) {
    return gdn_fmac_bc<E & 15>(acc, vec[E >> 4], x);
}
template <int E>
IK_DEVI float gdn8_mul(const float (&vec)[8], float x) {
    return gdn_mul_bc<E & 15>(vec[E >> 4], x);
}
template <int I, int NG, bool NEXT>
IK_DEVI void gdn_v8_loop(float4* sv, const float (&kt)[8], const float (&qt)[8], const float (&kn)[8], float delta, float decn,
                         float& ol, float& oh, float& nl, float& nh) {
    if constexpr (I < NG / 2) {
        constexpr int A = I, B = I + NG / 2, EA = 4 * A, EB = 4 * B;
        sv[A].x = gdn8_fmac<EA>(sv[A].x, kt, delta);
        sv[B].x = gdn8_fmac<EB>(sv[B].x, kt, delta);
        sv[A].y = gdn8_fmac<EA + 1>(sv[A].y, kt, delta);
        sv[B].y = gdn8_fmac<EB + 1>(sv[B].y, kt, delta);
        sv[A].z = gdn8_fmac<EA + 2>(sv[A].z, kt, delta);
        sv[B].z = gdn8_fmac<EB + 2>(sv[B].z, kt, delta);
        sv[A].w = gdn8_fmac<EA + 3>(sv[A].w, kt, delta);
        sv[B].w = gdn8_fmac<EB + 3>(sv[B].w, kt, delta);
        float qa = gdn8_mul<EA + 1>(qt, sv[A].y), qb = gdn8_mul<EB + 1>(qt, sv[B].y);
        float4 da, db;
        if constexpr (NEXT) {
            da = make_float4(sv[A].x * decn, sv[A].y * decn, sv[A].z * decn, sv[A].w * decn);
            db = make_float4(sv[B].x * decn, sv[B].y * decn, sv[B].z * decn, sv[B].w * decn);
        }
        qa = gdn8_fmac<EA>(qa, qt, sv[A].x);
        qb = gdn8_fmac<EB>(qb, qt, sv[B].x);
        float ka = 0.f, kb = 0.f;
        if constexpr (NEXT) {
            ka = gdn8_mul<EA + 1>(kn, da.y);
            kb = gdn8_mul<EB + 1>(kn, db.y);
        }
        qa = gdn8_fmac<EA + 2>(qa, qt, sv[A].z);
        qb = gdn8_fmac<EB + 2>(qb, qt, sv[B].z);
        if constexpr (NEXT) {
            ka = gdn8_fmac<EA>(ka, kn, da.x);
            kb = gdn8_fmac<EB>(kb, kn, db.x);
        }
        qa = gdn8_fmac<EA + 3>(qa, qt, sv[A].w);
        qb = gdn8_fmac<EB + 3>(qb, qt, sv[B].w);
        if constexpr (NEXT) {
            ka = gdn8_fmac<EA + 2>(ka, kn, da.z);
            kb = gdn8_fmac<EB + 2>(kb, kn, db.z);
            ka = gdn8_fmac<EA + 3>(ka, kn, da.w);
            kb = gdn8_fmac<EB + 3>(kb, kn, db.w);
            sv[A] = da;
            sv[B] = db;
        }
        ol += qa;
        oh += qb;
        if constexpr (NEXT) {
            nl += ka;
            nh += kb;
        }
        gdn_v8_loop<I + 1, NG, NEXT>(sv, kt, qt, kn, delta, decn, ol, oh, nl, nh);
    }
}
template <int G, int NG>
IK_DEVI void gdn_v8_first(float4* sv, const float (&k0)[8], float decay, float& nl, float& nh) {
    if constexpr (G < NG) {
        constexpr int E = 4 * G;
        sv[G].x *= decay; sv[G].y *= decay; sv[G].z *= decay; sv[G].w *= decay;
        float t = gdn8_mul<E + 1>(k0, sv[G].y);
        t = gdn8_fmac<E>(t, k0, sv[G].x);
        t = gdn8_fmac<E + 2>(t, k0, sv[G].z);
        t = gdn8_fmac<E + 3>(t, k0, sv[G].w);
        (G < NG / 2 ? nl : nh) += t;
        gdn_v8_first<G + 1, NG>(sv, k0, decay, nl, nh);
    }
}
struct GdnRaw8 {  // one token for one lane: q / k elements m + 16 i (m = lane % 16), its row's v
    ik_bf16 q[8], k[8], v;
};

__global__ void __launch_bounds__(32) gdn_v8_kernel(const ik_bf16* __restrict__ qkv, const ik_bf16* __restrict__ a,
                                                     const ik_bf16* __restrict__ b, const ik_bf16* __restrict__ a_log,
                                                     const ik_bf16* __restrict__ dt_bias, float eps, ik_bf16* __restrict__ o,
                                                     float* __restrict__ state, int64_t T, int Hk, int Hv, int dv) {
    constexpr int DK = 128, NG = DK / 4;
    const int lane = threadIdx.x;
    const int j = blockIdx.x * 32 + lane;
    const bool valid = j < dv;
    const int jl = valid ? j : dv - 1;
    const int h = blockIdx.y;
    const int kh = h / (Hv / Hk);
    const int C = 2 * Hk * DK + Hv * dv;
    const float scale = rsqrtf((float)DK);
    const float neg_a = -expf(ik_b2f(a_log[h]));
    const float dtb = ik_b2f(dt_bias[h]);
    const int64_t last = T - 1;
    const int m = lane & 15;
    const bool hi_row = lane >= 16;

    float s[DK];
    float4* sv = reinterpret_cast<float4*>(s);
    if (valid) {
        const float4* src = reinterpret_cast<const float4*>(state + ((int64_t)h * dv + j) * DK);
#pragma unroll
        for (int i = 0; i < NG; ++i) sv[i] = src[i];
    }
    GdnRaw8 raw;
    auto load = [&](int64_t t) {
        const ik_bf16* row = qkv + (t < last ? t : last) * C;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            raw.q[i] = row[kh * DK + m + 16 * i];
            raw.k[i] = row[Hk * DK + kh * DK + m + 16 * i];
        }
        raw.v = row[2 * Hk * DK + h * dv + jl];
    };
    ik_bf16 ga, gb;
    auto load_gates = [&](int64_t g0) {
        const int64_t t = g0 + lane < last ? g0 + lane : last;
        ga = a[t * Hv + h];
        gb = b[t * Hv + h];
    };
    float cbeta, cdecay, nbeta, ndecay;
    auto gate_vals = [&](float& beta, float& decay) {
        const float ar = ik_b2f(ga), br = ik_b2f(gb);
        beta = sigmoidf(br);
        decay = expf(neg_a * softplus(ar + dtb));
    };
    // normalise the raw token into q / k vectors and v; then load token u + 1
    auto stage = [&](int64_t u, float (&qv)[8], float (&kv8)[8], float& v) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            qv[i] = ik_b2f(raw.q[i]);
            kv8[i] = ik_b2f(raw.k[i]);
        }
        v = valid ? ik_b2f(raw.v) : 0.f;
        load(u + 1);
        float qq = 0.f, kk = 0.f;
#pragma unroll
        for (int i = 0; i < 4; ++i) {  // v2 lane L's elements L + 32 i
            const float qx = hi_row ? qv[2 * i + 1] : qv[2 * i];
            const float kx = hi_row ? kv8[2 * i + 1] : kv8[2 * i];
            qq += qx * qx;
            kk += kx * kx;
        }
        qq = gdn_warp_sum(qq);
        kk = gdn_warp_sum(kk);
        const float qn = rsqrtf(qq + eps) * scale, kn = rsqrtf(kk + eps);
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            qv[i] = qv[i] * qn;
            kv8[i] = kv8[i] * kn;
        }
    };

    load(0);
    load_gates(0);
    gate_vals(cbeta, cdecay);
    load_gates(32);
    gate_vals(nbeta, ndecay);
    load_gates(64);
    float qv[2][8], kvv[2][8], vv[2];
    stage(0, qv[0], kvv[0], vv[0]);
    stage(1, qv[1], kvv[1], vv[1]);
    float kvl = 0.f, kvh = 0.f;
    gdn_v8_first<0, NG>(sv, kvv[0], __int_as_float(__builtin_amdgcn_readlane(__float_as_int(cdecay), 0)), kvl, kvh);

    auto step = [&](int sl, int64_t t, auto next_c) {
        constexpr bool NEXT = decltype(next_c)::value;
        const int gl = (int)(t & 31);
        const float beta = __int_as_float(__builtin_amdgcn_readlane(__float_as_int(cbeta), gl));
        const int gn = (int)((t + 1) & 31);
        const float decn = gn == 0 ? __int_as_float(__builtin_amdgcn_readlane(__float_as_int(ndecay), 0))
                                   : __int_as_float(__builtin_amdgcn_readlane(__float_as_int(cdecay), gn));
        const float kv = kvl + kvh;
        const float delta = (vv[sl] - kv) * beta;
        float ol = 0.f, oh = 0.f, nl = 0.f, nh = 0.f;
        if (sl == 0) gdn_v8_loop<0, NG, NEXT>(sv, kvv[0], qv[0], kvv[1], delta, decn, ol, oh, nl, nh);
        else gdn_v8_loop<0, NG, NEXT>(sv, kvv[1], qv[1], kvv[0], delta, decn, ol, oh, nl, nh);
        const float out = ol + oh;
        if (valid) o[(t * Hv + h) * dv + j] = ik_f2b(out);
        kvl = nl;
        kvh = nh;
        stage(t + 2, qv[sl], kvv[sl], vv[sl]);
    };
    auto batch = [&](int64_t t0) {
        if ((t0 & 31) == 0 && t0 > 0) {
            cbeta = nbeta;
            cdecay = ndecay;
            gate_vals(nbeta, ndecay);
            load_gates(t0 + 64);
        }
    };
    const int64_t full = ((T - 1) / 2) * 2;
    for (int64_t t0 = 0; t0 < full; t0 += 2) {
        batch(t0);
        step(0, t0, std::integral_constant<bool, true>{});
        step(1, t0 + 1, std::integral_constant<bool, true>{});
    }
    batch(full);
    if (full + 1 < T) {
        step(0, full, std::integral_constant<bool, true>{});
        step(1, full + 1, std::integral_constant<bool, false>{});
    } else {
        step(0, full, std::integral_constant<bool, false>{});
    }
    if (valid) {
        float4* dst = reinterpret_cast<float4*>(state + ((int64_t)h * dv + j) * DK);
#pragma unroll
        for (int i = 0; i < NG; ++i) dst[i] = sv[i];
    }
}
// ---------------------------------------------------------------------------------------------------------------------
// Variant D ("v5"): v4's lane map (one lane per value row, 128 state registers, low / high chains) with the normalised q / k
// of every token written once by a pre-pass (gdn_qk_norm_kernel: v2's own norm arithmetic, one wave per token and key head)
// into scratch [T][Hk][q 128 | k 128] f32, and read in the scan with wave-uniform (scalar) loads: every product takes its
// q / k element as an SGPR operand, so there is no LDS traffic and no broadcast instruction. PF: s_prefetch_data of the
// token two ahead.
// ---------------------------------------------------------------------------------------------------------------------
__global__ void __launch_bounds__(32) gdn_qk_norm_kernel(const ik_bf16* __restrict__ qkv, float* __restrict__ nqk, float eps,
                                                          int64_t T, int Hk, int Hv, int dv) {
    constexpr int DK = 128, PL = 4;
    const int lane = threadIdx.x;
    const int64_t t = blockIdx.x;
    const int kh = blockIdx.y;
    const int C = 2 * Hk * DK + Hv * dv;
    const float scale = rsqrtf((float)DK);
    const ik_bf16* row = qkv + t * C;
    float qr[PL], kr[PL];
#pragma unroll
    for (int i = 0; i < PL; ++i) {
        qr[i] = ik_b2f(row[kh * DK + lane + 32 * i]);
        kr[i] = ik_b2f(row[Hk * DK + kh * DK + lane + 32 * i]);
    }
    float qq = 0.f, kk = 0.f;
#pragma unroll
    for (int i = 0; i < PL; ++i) {
        qq += qr[i] * qr[i];
        kk += kr[i] * kr[i];
    }
    qq = gdn_warp_sum(qq);
    kk = gdn_warp_sum(kk);
    const float qn = rsqrtf(qq + eps) * scale, kn = rsqrtf(kk + eps);
    float* dst = nqk + (t * Hk + kh) * (2 * DK);
#pragma unroll
    for (int i = 0; i < PL; ++i) {
        dst[lane + 32 * i] = qr[i] * qn;
        dst[DK + lane + 32 * i] = kr[i] * kn;
    }
}

template <bool PF>
__global__ void __launch_bounds__(32) gdn_v5_kernel(const ik_bf16* __restrict__ qkv, const float* __restrict__ nqk,
                                                     const ik_bf16* __restrict__ a, const ik_bf16* __restrict__ b,
                                                     const ik_bf16* __restrict__ a_log, const ik_bf16* __restrict__ dt_bias,
                                                     ik_bf16* __restrict__ o, float* __restrict__ state, int64_t T, int Hk,
                                                     int Hv, int dv) {
    constexpr int DK = 128, NG = DK / 4, HG = NG / 2;
    const int lane = threadIdx.x;
    const int j = blockIdx.x * 32 + lane;
    const bool valid = j < dv;
    const int jl = valid ? j : dv - 1;
    const int h = blockIdx.y;
    const int kh = h / (Hv / Hk);
    const int C = 2 * Hk * DK + Hv * dv;
    const float neg_a = -expf(ik_b2f(a_log[h]));
    const float dtb = ik_b2f(dt_bias[h]);
    const int64_t last = T - 1;
    float s[DK];
    float4* sv = reinterpret_cast<float4*>(s);
    if (valid) {
        const float4* src = reinterpret_cast<const float4*>(state + ((int64_t)h * dv + j) * DK);
#pragma unroll
        for (int i = 0; i < NG; ++i) sv[i] = src[i];
    }
    auto qp = [&](int64_t t) { return reinterpret_cast<const float4*>(nqk + (t * Hk + kh) * (2 * DK)); };
    auto kp = [&](int64_t t) { return reinterpret_cast<const float4*>(nqk + (t * Hk + kh) * (2 * DK) + DK); };
    ik_bf16 ga, gb;
    auto load_gates = [&](int64_t g0) {
        const int64_t t = g0 + lane < last ? g0 + lane : last;
        ga = a[t * Hv + h];
        gb = b[t * Hv + h];
    };
    float cbeta, cdecay, nbeta, ndecay;
    auto gate_vals = [&](float& beta, float& decay) {
        const float ar = ik_b2f(ga), br = ik_b2f(gb);
        beta = sigmoidf(br);
        decay = expf(neg_a * softplus(ar + dtb));
    };
    load_gates(0);
    gate_vals(cbeta, cdecay);
    load_gates(32);
    gate_vals(nbeta, ndecay);
    load_gates(64);
    // v of the next tokens (one per lane-row), two ahead
    ik_bf16 vraw[2];
    auto vload = [&](int64_t t) { return qkv[(t < last ? t : last) * C + 2 * Hk * DK + h * dv + jl]; };
    vraw[0] = vload(0);
    vraw[1] = vload(1);
    float kvl = 0.f, kvh = 0.f;
    {
        const float decay = __int_as_float(__builtin_amdgcn_readlane(__float_as_int(cdecay), 0));
        const float4* k4 = kp(0);
#pragma unroll
        for (int g = 0; g < NG; ++g) {
            const float4 kk4 = k4[g];
            sv[g].x *= decay; sv[g].y *= decay; sv[g].z *= decay; sv[g].w *= decay;
            (g < HG ? kvl : kvh) += gdn_dot4(sv[g], kk4);
        }
    }
    auto step = [&](int sl, int64_t t, auto next_c) {
        constexpr bool NEXT = decltype(next_c)::value;
        if constexpr (PF) {
            const int64_t tp = t + 2 < last ? t + 2 : last;
            __builtin_amdgcn_s_prefetch_data(nqk + (tp * Hk + kh) * (2 * DK), 1024);
        }
        const float v = valid ? ik_b2f(vraw[sl]) : 0.f;
        vraw[sl] = vload(t + 2);
        const int gl = (int)(t & 31);
        const float beta = __int_as_float(__builtin_amdgcn_readlane(__float_as_int(cbeta), gl));
        const int gn = (int)((t + 1) & 31);
        const float decn = gn == 0 ? __int_as_float(__builtin_amdgcn_readlane(__float_as_int(ndecay), 0))
                                   : __int_as_float(__builtin_amdgcn_readlane(__float_as_int(cdecay), gn));
        const float kv = kvl + kvh;
        const float delta = (v - kv) * beta;
        float ol = 0.f, oh = 0.f, nl = 0.f, nh = 0.f;
        const float4* k4 = kp(t);
        const float4* q4 = qp(t);
        const float4* k4n = kp(t + 1 < last ? t + 1 : last);
#pragma unroll
        for (int i = 0; i < HG; ++i) {
#pragma unroll
            for (int half = 0; half < 2; ++half) {
                const int g = half * HG + i;
                const float4 kk4 = k4[g], qq4 = q4[g];
                sv[g].x = __builtin_fmaf(kk4.x, delta, sv[g].x);
                sv[g].y = __builtin_fmaf(kk4.y, delta, sv[g].y);
                sv[g].z = __builtin_fmaf(kk4.z, delta, sv[g].z);
                sv[g].w = __builtin_fmaf(kk4.w, delta, sv[g].w);
                (half ? oh : ol) += gdn_dot4(sv[g], qq4);
                if constexpr (NEXT) {
                    const float4 kn4 = k4n[g];
                    sv[g].x *= decn; sv[g].y *= decn; sv[g].z *= decn; sv[g].w *= decn;
                    (half ? nh : nl) += gdn_dot4(sv[g], kn4);
                }
            }
        }
        const float out = ol + oh;
        if (valid) o[(t * Hv + h) * dv + j] = ik_f2b(out);
        kvl = nl;
        kvh = nh;
    };
    auto batch = [&](int64_t t0) {
        if ((t0 & 31) == 0 && t0 > 0) {
            cbeta = nbeta;
            cdecay = ndecay;
            gate_vals(nbeta, ndecay);
            load_gates(t0 + 64);
        }
    };
    const int64_t full = ((T - 1) / 2) * 2;
    for (int64_t t0 = 0; t0 < full; t0 += 2) {
        batch(t0);
        step(0, t0, std::integral_constant<bool, true>{});
        step(1, t0 + 1, std::integral_constant<bool, true>{});
    }
    batch(full);
    if (full + 1 < T) {
        step(0, full, std::integral_constant<bool, true>{});
        step(1, full + 1, std::integral_constant<bool, false>{});
    } else {
        step(0, full, std::integral_constant<bool, false>{});
    }
    if (valid) {
        float4* dst = reinterpret_cast<float4*>(state + ((int64_t)h * dv + j) * DK);
#pragma unroll
        for (int i = 0; i < NG; ++i) dst[i] = sv[i];
    }
}

}  // namespace


#define CK(x)                                                                                    \
    do {                                                                                         \
        hipError_t e_ = (x);                                                                     \
        if (e_ != hipSuccess) {                                                                  \
            fprintf(stderr, "HIP error %s at %s:%d\n", hipGetErrorString(e_), __FILE__, __LINE__); \
            exit(1);                                                                             \
        }                                                                                        \
    } while (0)

static uint16_t f2bf(float f) {
    uint32_t u;
    memcpy(&u, &f, 4);
    return (uint16_t)((u + 0x7fff + ((u >> 16) & 1)) >> 16);
}

int main(int argc, char** argv) {
    const int Hk = 16, Hv = 32, dv = 128, dk = 128, C = 2 * Hk * dk + Hv * dv;
    const int Ts[] = {1, 2, 3, 4, 5, 6, 7, 9, 31, 32, 33, 63, 64, 65, 97, 683, 4096};
    const int64_t Tmax = 4096;
    uint64_t r = 88172645463325252ull;
    auto rnd = [&]() {  // ~N(0,1)-ish: sum of uniforms
        float s = 0;
        for (int i = 0; i < 4; ++i) {
            r ^= r << 13; r ^= r >> 7; r ^= r << 17;
            s += (float)(r >> 40) / (float)(1ull << 24) - 0.5f;
        }
        return s * 1.7f;
    };
    std::vector<uint16_t> hq((size_t)Tmax * C), ha((size_t)Tmax * Hv), hb((size_t)Tmax * Hv), hal(Hv), hdt(Hv);
    for (auto& x : hq) x = f2bf(rnd());
    for (auto& x : ha) x = f2bf(rnd());
    for (auto& x : hb) x = f2bf(1.5f * rnd());
    for (auto& x : hal) x = f2bf(0.5f * rnd());
    for (auto& x : hdt) x = f2bf(0.5f * rnd());
    std::vector<float> hs((size_t)Hv * dv * dk);
    for (auto& x : hs) x = 0.3f * rnd();
    ik_bf16 *dq, *da, *db, *dal, *ddt, *o0, *o1;
    float *s0, *s1;
    CK(hipMalloc(&dq, hq.size() * 2));
    CK(hipMalloc(&da, ha.size() * 2));
    CK(hipMalloc(&db, hb.size() * 2));
    CK(hipMalloc(&dal, Hv * 2));
    CK(hipMalloc(&ddt, Hv * 2));
    CK(hipMalloc(&o0, (size_t)Tmax * Hv * dv * 2));
    CK(hipMalloc(&o1, (size_t)Tmax * Hv * dv * 2));
    CK(hipMalloc(&s0, hs.size() * 4));
    CK(hipMalloc(&s1, hs.size() * 4));
    CK(hipMemcpy(dq, hq.data(), hq.size() * 2, hipMemcpyHostToDevice));
    CK(hipMemcpy(da, ha.data(), ha.size() * 2, hipMemcpyHostToDevice));
    CK(hipMemcpy(db, hb.data(), hb.size() * 2, hipMemcpyHostToDevice));
    CK(hipMemcpy(dal, hal.data(), Hv * 2, hipMemcpyHostToDevice));
    CK(hipMemcpy(ddt, hdt.data(), Hv * 2, hipMemcpyHostToDevice));
    const float eps = 1e-6f;
    enum { V2, V3I, V3R, NOL, V4, V5, V5P, V6, V7, V8 };
    float* nqk;
    CK(hipMalloc(&nqk, (size_t)Tmax * Hk * 2 * dk * 4));
    int div = 1;  // occupancy probe: only Hv / div heads launched (timing only)
    auto launch = [&](int v, int64_t T, ik_bf16* o, float* st) {
        if (v == V2) gdn_delta_rows_v2_kernel<128, 2, 2><<<dim3(dv / 16, Hv / div), 32>>>(dq, da, db, dal, ddt, eps, o, st, T, Hk, Hv, dv);
        else if (v == V3I) gdn_v3i_kernel<128><<<dim3(dv / 16, Hv / div), 32>>>(dq, da, db, dal, ddt, eps, o, st, T, Hk, Hv, dv);
        else if (v == V3R) gdn_v3r_kernel<128><<<dim3(dv / 32, Hv / div), 32>>>(dq, da, db, dal, ddt, eps, o, st, T, Hk, Hv, dv);
        else if (v == NOL) gdn_v3i_kernel<128, true><<<dim3(dv / 16, Hv / div), 32>>>(dq, da, db, dal, ddt, eps, o, st, T, Hk, Hv, dv);
        else if (v == V8) gdn_v8_kernel<<<dim3(dv / 32, Hv / div), 32>>>(dq, da, db, dal, ddt, eps, o, st, T, Hk, Hv, dv);
        else if (v == V7) gdn_v7_kernel<<<dim3(dv / 32, Hv / div), 32>>>(dq, da, db, dal, ddt, eps, o, st, T, Hk, Hv, dv);
        else if (v == V6) gdn_v6_kernel<<<dim3(dv / 32, Hv / div), 32>>>(dq, da, db, dal, ddt, eps, o, st, T, Hk, Hv, dv);
        else if (v == V4) gdn_v4_kernel<<<dim3(dv / 32, Hv / div), 32>>>(dq, da, db, dal, ddt, eps, o, st, T, Hk, Hv, dv);
        else {
            gdn_qk_norm_kernel<<<dim3((unsigned)T, Hk), 32>>>(dq, nqk, eps, T, Hk, Hv, dv);
            if (v == V5) gdn_v5_kernel<false><<<dim3(dv / 32, Hv / div), 32>>>(dq, nqk, da, db, dal, ddt, o, st, T, Hk, Hv, dv);
            else gdn_v5_kernel<true><<<dim3(dv / 32, Hv / div), 32>>>(dq, nqk, da, db, dal, ddt, o, st, T, Hk, Hv, dv);
        }
    };
    const char* names[] = {"v2", "v3i", "v3r", "v3i-nolds", "v4", "v5", "v5pf", "v6", "v7", "v8"};
    hipEvent_t e0, e1;
    CK(hipEventCreate(&e0));
    CK(hipEventCreate(&e1));
    std::vector<uint16_t> ro((size_t)Tmax * Hv * dv), go((size_t)Tmax * Hv * dv);
    std::vector<float> rs(hs.size()), gs(hs.size());
    bool all_ok = true;
    for (int T : Ts) {
        CK(hipMemcpy(s0, hs.data(), hs.size() * 4, hipMemcpyHostToDevice));
        CK(hipMemset(o0, 0, (size_t)T * Hv * dv * 2));
        launch(V2, T, o0, s0);
        CK(hipDeviceSynchronize());
        CK(hipMemcpy(ro.data(), o0, (size_t)T * Hv * dv * 2, hipMemcpyDeviceToHost));
        CK(hipMemcpy(rs.data(), s0, hs.size() * 4, hipMemcpyDeviceToHost));
        char line[512];
        int n = snprintf(line, sizeof line, "T=%-5d", T);
        for (int v : {(int)V2, (int)V3I, (int)V7, (int)V8}) {
            CK(hipMemcpy(s1, hs.data(), hs.size() * 4, hipMemcpyHostToDevice));
            CK(hipMemset(o1, 0, (size_t)T * Hv * dv * 2));
            launch(v, T, o1, s1);
            CK(hipGetLastError());
            CK(hipDeviceSynchronize());
            CK(hipMemcpy(go.data(), o1, (size_t)T * Hv * dv * 2, hipMemcpyDeviceToHost));
            CK(hipMemcpy(gs.data(), s1, hs.size() * 4, hipMemcpyDeviceToHost));
            const bool same = memcmp(ro.data(), go.data(), (size_t)T * Hv * dv * 2) == 0 && memcmp(rs.data(), gs.data(), hs.size() * 4) == 0;
            all_ok &= same;
            float best = 1e30f;
            if (T >= 64) {
                for (int rep = 0; rep < 5; ++rep) {
                    CK(hipEventRecord(e0));
                    launch(v, T, o1, s1);
                    CK(hipEventRecord(e1));
                    CK(hipEventSynchronize(e1));
                    float ms;
                    CK(hipEventElapsedTime(&ms, e0, e1));
                    best = fminf(best, ms);
                }
                n += snprintf(line + n, sizeof line - n, "  %s %8.1f us (%.3f us/tok) %s", names[v], best * 1e3, best * 1e3 / T, same ? "bitwise" : "DIFFERS");
            } else {
                n += snprintf(line + n, sizeof line - n, "  %s %s", names[v], same ? "bitwise" : "DIFFERS");
            }
        }
        puts(line);
    }
    puts(all_ok ? "ALL BITWISE" : "SOME DIFFER");
    for (div = 1; div <= 4; div *= 2) {  // per-token time vs the number of waves (256 / div for v2 and v3i)
        for (int v : {(int)V2, (int)V3I, (int)V7, (int)V8}) {
            float best = 1e30f;
            for (int rep = 0; rep < 5; ++rep) {
                CK(hipEventRecord(e0));
                launch(v, 683, o1, s1);
                CK(hipEventRecord(e1));
                CK(hipEventSynchronize(e1));
                float ms;
                CK(hipEventElapsedTime(&ms, e0, e1));
                best = fminf(best, ms);
            }
            printf("probe heads=%2d waves=%3d %s T=683: %.3f us/tok\n", Hv / div, (dv / 16) * Hv / div, names[v], best * 1e3 / 683);
        }
    }
    return all_ok ? 0 : 1;
}
