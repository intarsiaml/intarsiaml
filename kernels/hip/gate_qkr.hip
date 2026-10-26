// Core-op fusion phase 2, P4a (docs/results/2026-10-19-core-op-fusion-p2.md): Op::GateSplitQkNormRopeKv, the gated
// attention's split folded into the fused prologue. One launch instead of gate_split_bf16 + qk_norm_rope_kv_bf16
// (elementwise.hip). The kernel below is qk_norm_rope_kv_kernel statement for statement, with two differences:
//  * a query wave reads its head's D query values from the gated row x[T,N,2D] (head h owns 2D consecutive values, the
//    first D the query) instead of from the split-off q[T,N,D]: gate_split_kernel copies those bits unchanged, so the
//    norm sees the same inputs;
//  * a query wave also copies its head's D gate values (the second half) to gate[T,N,D], which is gate_split_kernel's
//    other output (a bf16 copy).
// So every output bit is the unfused chain's (tests/fusion_p2.rs, against the two launches on the same device). The
// constraints are qk_norm_rope_kv_bf16's: D % 8 == 0, D <= 256, N / Nkv <= 8, 16-byte aligned rows.
#include "ik_hip.h"

namespace {

constexpr int kGqrMaxGroup = 8;
constexpr int kGqrMaxD = 256;

// elementwise.hip's f32_to_e4m3 (the kv_append / qk_norm_rope_kv fp8 cache conversion), copied unchanged
IK_DEVI uint8_t gqr_f32_to_e4m3(float f) {
    const uint32_t u = __float_as_uint(f);
    const uint8_t sign = (u >> 24) & 0x80u;
    if ((u & 0x7fffffffu) > 0x7f800000u) return sign | 0x7fu;  // NaN
    const float a = fabsf(f);
    if (a >= 448.f) return sign | 0x7eu;  // saturate (also covers inf and the round-up past 448)
    if (a < 0.015625f) {  // below the smallest normal 2^-6: subnormals are multiples of 2^-9
        const uint32_t q = (uint32_t)__float2int_rn(a * 512.f);  // 0..8; 8 is the smallest normal (0x08)
        return sign | (uint8_t)q;
    }
    const uint32_t ua = __float_as_uint(a);
    int e = (int)(ua >> 23) - 127;  // -6 .. 8
    uint32_t m = (ua & 0x7fffffu) >> 20;
    const uint32_t rem = ua & 0xfffffu;
    if (rem > 0x80000u || (rem == 0x80000u && (m & 1u))) ++m;
    if (m == 8u) {
        m = 0;
        ++e;
    }
    if (e > 8 || (e == 8 && m == 7u)) return sign | 0x7eu;
    return sign | (uint8_t)(((e + 7) << 3) | (int)m);
}

template <typename CacheT>
IK_DEVI void gqr_store(CacheT* c, ik_bf16 v) {
    if constexpr (sizeof(CacheT) == 2) *reinterpret_cast<ik_bf16*>(c) = v;
    else *reinterpret_cast<uint8_t*>(c) = gqr_f32_to_e4m3(ik_b2f(v));
}

// Grid (T, Nkv), block G + 1 waves (G = N / Nkv): wave w < G takes query head kvh * G + w (and copies its gate), wave G
// the k head kvh and the v copy.
template <typename CacheT>
__global__ void __launch_bounds__((kGqrMaxGroup + 1) * IK_WARP)
gate_qk_norm_rope_kv_kernel(const ik_bf16* __restrict__ qg, const ik_bf16* __restrict__ k, const ik_bf16* __restrict__ v,
                            const ik_bf16* __restrict__ qw, const ik_bf16* __restrict__ kw, ik_bf16* __restrict__ q_out,
                            ik_bf16* __restrict__ gate, CacheT* __restrict__ kcache, CacheT* __restrict__ vcache, int N,
                            int NKV, int D, int RD, int64_t cap, float eps, float woff, float theta, int64_t pos0_host,
                            const int32_t* pos_dev) {
    __shared__ __align__(16) ik_bf16 nrm[kGqrMaxGroup + 1][kGqrMaxD];
    __shared__ float fsn[kGqrMaxD / 2], fcs[kGqrMaxD / 2];
    const int64_t pos0 = pos_dev ? (int64_t)*pos_dev : pos0_host;
    const int64_t t = blockIdx.x;
    const int kvh = blockIdx.y;
    const int G = N / NKV;
    const int lane = threadIdx.x % IK_WARP, w = threadIdx.x / IK_WARP;
    const int half = RD / 2;
    for (int f = threadIdx.x; f < half; f += blockDim.x) {
        const double inv = pow((double)theta, -2.0 * (double)f / (double)RD);
        double sd, cd;
        sincos((double)(pos0 + t) * inv, &sd, &cd);
        fsn[f] = (float)sd;
        fcs[f] = (float)cd;
    }
    const bool is_k = w == G;
    const int64_t qrow = t * N + (int64_t)kvh * G + w;  // the query head's row (query waves only)
    const ik_bf16* xr = is_k ? k + (t * NKV + kvh) * D : qg + qrow * 2 * D;
    const ik_bf16* wt = is_k ? kw : qw;
    ik_bf16* nr = nrm[w];
    const int nv = D / 8;
    float ss = 0.f;
    for (int i = lane; i < nv; i += IK_WARP) {
        float f[8];
        ik_unpack8(reinterpret_cast<const uint4*>(xr)[i], f);
#pragma unroll
        for (int e = 0; e < 8; ++e) ss += f[e] * f[e];
    }
    ss = ik_warp_sum(ss);
    float tot = 0.f;  // ik_block_sum over one wave: 0 + the wave's sum
    tot += ss;
    const float r = rsqrtf(tot / (float)D + eps);
    for (int i = lane; i < nv; i += IK_WARP) {
        float f[8], g[8];
        ik_unpack8(reinterpret_cast<const uint4*>(xr)[i], f);
        ik_unpack8(reinterpret_cast<const uint4*>(wt)[i], g);
#pragma unroll
        for (int e = 0; e < 8; ++e) f[e] = f[e] * r * (g[e] + woff);
        reinterpret_cast<uint4*>(nr)[i] = ik_pack8(f);
    }
    // gate_split_kernel's second output: the gate half of the query head's gated row, copied
    if (!is_k)
        for (int m = lane; m < D; m += IK_WARP) gate[qrow * D + m] = qg[qrow * 2 * D + D + m];
    IK_LDS_BARRIER();
    ik_bf16 *dst_q = nullptr;
    CacheT* dst_k = nullptr;
    if (is_k) dst_k = kcache + ((int64_t)kvh * cap + pos0 + t) * D;
    else dst_q = q_out + qrow * D;
    for (int f = lane; f < half; f += IK_WARP) {
        const float a = ik_b2f(nr[f]), b = ik_b2f(nr[f + half]);
        const float fs = fsn[f], fc = fcs[f];
        // rope_kernel's contraction, spelled out (see qk_norm_rope_kv_kernel)
        float lof, hif;
        {
#pragma clang fp contract(off)
            lof = __builtin_fmaf(a, fc, -(b * fs));
            hif = __builtin_fmaf(b, fc, a * fs);
        }
        const ik_bf16 lo = ik_f2b(lof);
        const ik_bf16 hi = ik_f2b(hif);
        if (is_k) {
            gqr_store(dst_k + f, lo);
            gqr_store(dst_k + f + half, hi);
        } else {
            dst_q[f] = lo;
            dst_q[f + half] = hi;
        }
    }
    for (int m = RD + lane; m < D; m += IK_WARP) {
        if (is_k) gqr_store(dst_k + m, nr[m]);
        else dst_q[m] = nr[m];
    }
    if (is_k) {
        const ik_bf16* vr = v + (t * NKV + kvh) * D;
        CacheT* dv = vcache + ((int64_t)kvh * cap + pos0 + t) * D;
        for (int m = lane; m < D; m += IK_WARP) gqr_store(dv + m, vr[m]);
    }
}

}  // namespace

extern "C" {

// qk_norm_rope_kv_bf16's checks, with q replaced by the gated row qg [T,N,2D] and the gate output [T,N,D] added
int ik_gate_split_qk_norm_rope_kv_bf16(const ik_tensor* qg, const ik_tensor* k, const ik_tensor* v, const ik_tensor* qw,
                                       const ik_tensor* kw, float eps, float w_offset, int64_t pos0, float theta,
                                       int64_t rotary_dim, ik_tensor* q_out, ik_tensor* gate, ik_tensor* kcache,
                                       ik_tensor* vcache, const ik_tensor* pos, void* stream) {
    IK_TRY(ik_check(qg, IK_DTYPE_BF16, 3));
    IK_TRY(ik_check(k, IK_DTYPE_BF16, 3));
    IK_TRY(ik_check(v, IK_DTYPE_BF16, 3));
    IK_TRY(ik_check(qw, IK_DTYPE_BF16, 1));
    IK_TRY(ik_check(kw, IK_DTYPE_BF16, 1));
    IK_TRY(ik_check(q_out, IK_DTYPE_BF16, 3));
    IK_TRY(ik_check(gate, IK_DTYPE_BF16, 3));
    IK_TRY(ik_check_pos(pos));
    const int32_t* pos_dev = pos ? (const int32_t*)pos->data : nullptr;
    if (!kcache || !vcache) return IK_ERR_INVALID_ARG;
    if ((kcache->dtype != IK_DTYPE_BF16 && kcache->dtype != IK_DTYPE_FP8_E4M3) || vcache->dtype != kcache->dtype)
        return IK_ERR_DTYPE;
    IK_TRY(ik_check(kcache, kcache->dtype, 3));
    IK_TRY(ik_check(vcache, vcache->dtype, 3));
    if (qg->shape[2] % 2 != 0) return IK_ERR_SHAPE;
    const int64_t T = qg->shape[0], N = qg->shape[1], D = qg->shape[2] / 2, NKV = k->shape[1], cap = kcache->shape[1];
    const int64_t RD = rotary_dim == 0 ? D : rotary_dim;
    if (k->shape[0] != T || k->shape[2] != D || v->shape[0] != T || v->shape[1] != NKV || v->shape[2] != D ||
        q_out->shape[0] != T || q_out->shape[1] != N || q_out->shape[2] != D || gate->shape[0] != T ||
        gate->shape[1] != N || gate->shape[2] != D || qw->shape[0] != D || kw->shape[0] != D ||
        kcache->shape[0] != NKV || kcache->shape[2] != D || vcache->shape[0] != NKV || vcache->shape[1] != cap ||
        vcache->shape[2] != D || pos0 < 0 || pos0 + T > cap)
        return IK_ERR_SHAPE;
    if (NKV <= 0 || N % NKV != 0 || N / NKV > kGqrMaxGroup || D % 8 != 0 || D > kGqrMaxD || RD < 0 || RD > D || RD % 2 != 0)
        return IK_ERR_UNSUPPORTED;
    if (T == 0 || N == 0) return IK_OK;
    // qk_norm_rope_kv_bf16's vector path needs 16-byte aligned rows; a query row of qg starts at a multiple of 2D values
    // (D % 8 == 0: 32 bytes), so the base pointer decides
    if (!ik_aligned16(qg->data) || !ik_aligned16(k->data) || !ik_aligned16(qw->data) || !ik_aligned16(kw->data) ||
        !ik_aligned16(q_out->data))
        return IK_ERR_STRIDE;
    if (T > 0x7fffffffLL) return IK_ERR_SHAPE;
    const dim3 grid((unsigned)T, (unsigned)NKV);
    const unsigned threads = (unsigned)((N / NKV + 1) * IK_WARP);
    hipStream_t s = IK_STREAM(stream);
    const ik_bf16 *qp = (const ik_bf16*)qg->data, *kp = (const ik_bf16*)k->data, *vp = (const ik_bf16*)v->data;
    const ik_bf16 *qwp = (const ik_bf16*)qw->data, *kwp = (const ik_bf16*)kw->data;
    ik_bf16 *qo = (ik_bf16*)q_out->data, *go = (ik_bf16*)gate->data;
    if (kcache->dtype == IK_DTYPE_BF16)
        gate_qk_norm_rope_kv_kernel<ik_bf16><<<grid, threads, 0, s>>>(qp, kp, vp, qwp, kwp, qo, go, (ik_bf16*)kcache->data,
                                                                       (ik_bf16*)vcache->data, (int)N, (int)NKV, (int)D,
                                                                       (int)RD, cap, eps, w_offset, theta, pos0, pos_dev);
    else
        gate_qk_norm_rope_kv_kernel<uint8_t><<<grid, threads, 0, s>>>(qp, kp, vp, qwp, kwp, qo, go, (uint8_t*)kcache->data,
                                                                       (uint8_t*)vcache->data, (int)N, (int)NKV, (int)D,
                                                                       (int)RD, cap, eps, w_offset, theta, pos0, pos_dev);
    return ik_launch_status();
}

}  // extern "C"
