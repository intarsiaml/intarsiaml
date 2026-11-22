// Inkling-2-Mini routed experts (lane ink-cuda-mixmoe): Op::MoeExperts with the SiTU activation over NVFP4, and
// Op::MoeExpertsMxfp8 (e4m3 weights, E8M0 scale per 32 along the input dim; silu / gelu_tanh / situ). Contract:
// crates/intarsia-core/src/inkling2/ops.rs ("MoeExpertsMxfp8 and the situ activation"); reference quant_ref.rs / ink_ref.rs.
//
// Union-grouped decode (Metal lane ink-m-moe's design): every block lists the call's (token, slot) pairs by distinct expert in
// shared memory, and its warps take experts; a warp streams its rows of the expert once and runs every pair routed to that
// expert on them (in batches of NB accumulators; a batch past the first re-reads the rows from L1 / L2). Two launches:
//   gate_up  block = 4 rows of I; per row 8 lanes, each lane one 32-element chunk of the gate row and of the up row per step;
//            h[pair, r] = epilogue(rA(x . gate_r), rA(x . up_r)) (situ: times the routing weight)
//   down     block = 8 output rows; per row 4 lanes; y = rA(h[pair] . down_r), term = y (situ) or rA(y * w); the terms land
//            in shared memory and out[t, r] = rA(sum_k term[t, k]) in slot order.
// Per (pair, row) the chunk order, the fma chain per chunk, the lane reduction and the roundings never depend on T, on the
// routing of other tokens or on which pairs share a batch, so a T-row call is bitwise T one-row calls (rows-exact).
// More than kDecPairs pairs (prefill): the same kernels on a pair table built once in scratch by a grouping launch, the
// terms through scratch y (bf16-exact values: y and rA(y * w) are bf16) and a gather launch: the same bits.
//
// Scratch: the core `moe_experts_scratch` layout: pairs int[P] | offsets int[E+1] | active int[E+1] | h bf16[P,I] | y bf16[P,H].
#include "ik_launch.h"
#include "ik_quant.cuh"
#include "ik_cuda_abi_ink.h"
#include "ink_common.cuh"
#include <climits>

namespace {

constexpr int kThreads = 256;
constexpr int kWarpsB = kThreads / IK_WARP;
constexpr int kDecPairs = 512;  // pairs grouped in shared memory (decode / verify / short chunks)
constexpr int kEMax = 1024;
constexpr int kGuRows = 4;  // rows of I per gate_up block (8 lanes each)
constexpr int kDnRows = 8;  // output rows per down block (4 lanes each)
constexpr int kNB = 4;      // pairs per accumulator batch

enum { FMT_NV = 0, FMT_MX = 1 };
enum { ACT_SILU = 0, ACT_GELU = 1, ACT_SITU = 2 };

int g_probe = 0;

struct MoeArgs {
    const ik_bf16* x;
    const int* ids;
    const float* wts;
    const uint8_t* guw;  // gate_up / w13 payload
    const uint8_t* gus;  // its scales (e4m3 per 16 / E8M0 per 32)
    const uint8_t* dnw;
    const uint8_t* dns;
    const float* gs;  // NVFP4 global scales [E,3] (null for MXFP8)
    ik_bf16* h;       // scratch [P,I]
    ik_bf16* y;       // scratch [P,H] (table path)
    // the pair table (table path): dpair [P], doff [E+1], dexp [E] + nd at dexp[E]
    int* t_pair;
    int* t_off;
    int* t_exp;
    void* out;
    int T, K, E, H, I;
    float gc, uc;
    int table;
};

IK_DEVI float sigm(float z) { return 1.f / (1.f + expf(-z)); }

template <int ACT>
IK_DEVI float act_h(float g, float u, float w, float gc, float uc) {
    if (ACT == ACT_SITU) return ink_rbf(gc * tanhf(g / gc) * sigm(g) * uc * tanhf(u / uc) * w);
    const float a = ink_rbf(ACT == ACT_GELU ? ik_gelu_tanh(g) : g / (1.f + expf(-g)));
    return ink_rbf(a * u);
}

// ---- grouping: the block's threads list the valid pairs by distinct expert ---------------------------------------------
// first [E] (shared), nd, dexp [nd], doff [nd+1], dpair [nvalid]; pd / ps / cnt are [P] temporaries. The order of the experts
// and of the pairs within an expert comes from atomics: it decides only which warp runs what, never a bit of the result.
template <typename IntP>
IK_DEVI void group_pairs(const int* __restrict__ ids, int P, int E, int* first, int* nd, IntP dexp, IntP doff, IntP dpair, IntP pd,
                         IntP ps, IntP cnt) {
    const int tid = threadIdx.x, nt = blockDim.x;
    for (int e = tid; e < E; e += nt) first[e] = INT_MAX;
    if (tid == 0) *nd = 0;
    __syncthreads();
    for (int p = tid; p < P; p += nt) {
        const int v = ids[p];
        if (v >= 0 && v < E) atomicMin(&first[v], p);
    }
    __syncthreads();
    for (int p = tid; p < P; p += nt) {
        const int v = ids[p];
        pd[p] = -1;
        if (v >= 0 && v < E && first[v] == p) {
            const int d = atomicAdd(nd, 1);
            dexp[d] = v;
            pd[p] = d;
            cnt[d] = 0;
        }
    }
    __syncthreads();
    // the leaders (pd >= 0) replace their pair index by the expert's index d (testing first[v] == p again here would race
    // with leaders that already wrote a d equal to some other pair's index)
    for (int p = tid; p < P; p += nt) {
        if (pd[p] >= 0) first[ids[p]] = pd[p];
    }
    __syncthreads();
    for (int p = tid; p < P; p += nt) {
        const int v = ids[p];
        if (v >= 0 && v < E) {
            const int d = first[v];
            pd[p] = d;
            ps[p] = atomicAdd(&cnt[d], 1);
        }
    }
    __syncthreads();
    const int n = *nd;
    if (tid < IK_WARP) {
        int carry = 0;
        if (tid == 0) doff[0] = 0;
        for (int b = 0; b < n; b += IK_WARP) {
            int v = b + tid < n ? cnt[b + tid] : 0;
#pragma unroll
            for (int s = 1; s < IK_WARP; s <<= 1) {
                const int u = __shfl_up_sync(0xffffffffu, v, s);
                if (tid >= s) v += u;
            }
            if (b + tid < n) doff[b + tid + 1] = v + carry;
            carry += __shfl_sync(0xffffffffu, v, IK_WARP - 1);
        }
    }
    __syncthreads();
    for (int p = tid; p < P; p += nt) {
        const int v = ids[p];
        if (v >= 0 && v < E) dpair[doff[pd[p]] + ps[p]] = p;
    }
    __syncthreads();
}

struct SmemGroups {
    int first[kEMax];
    int nd;
    int dexp[kDecPairs];
    int doff[kDecPairs + 1];
    int dpair[kDecPairs];
    int pd[kDecPairs];
    int ps[kDecPairs];
    int cnt[kDecPairs];
};

// table view the kernels read
struct Groups {
    int nd;
    const int* dexp;
    const int* doff;
    const int* dpair;
};

IK_DEVI Groups get_groups(const MoeArgs& a, SmemGroups& sg) {
    if (a.table) {
        return Groups{a.t_exp[a.E], a.t_exp, a.t_off, a.t_pair};
    }
    group_pairs<int*>(a.ids, a.T * a.K, a.E, sg.first, &sg.nd, sg.dexp, sg.doff, sg.dpair, sg.pd, sg.ps, sg.cnt);
    return Groups{sg.nd, sg.dexp, sg.doff, sg.dpair};
}

// ---- weight chunks: 32 consecutive k of one row ----------------------------------------------------------------------
// NVFP4: 16 bytes of e2m1 pairs + two e4m3 block scales -> 16 bf16 pairs (exact) and two scales; dot with 16 bf16 pairs of
// x: two chains of 8 fma (one per block of 16), each folded into the accumulator times its scale.
// MXFP8: 32 e4m3 bytes -> 16 f16 pairs (exact) and the E8M0 power of two; dot: one chain of 32 fma in f32, times the scale.
struct WChunk {
    uint32_t w[16];
    float s0, s1;
};

template <int FMT>
IK_DEVI void load_chunk(const uint8_t* __restrict__ wrow, const uint8_t* __restrict__ srow, int ch, WChunk& c) {
    if (FMT == FMT_NV) {
        const uint4 v = __ldcs(reinterpret_cast<const uint4*>(wrow) + ch);
        ik_fp4x8_to_bf2(v.x, c.w);
        ik_fp4x8_to_bf2(v.y, c.w + 4);
        ik_fp4x8_to_bf2(v.z, c.w + 8);
        ik_fp4x8_to_bf2(v.w, c.w + 12);
        const unsigned short sb = *reinterpret_cast<const unsigned short*>(srow + 2 * ch);
        const float2 s = ik_e4m3x2_to_f2(sb);
        c.s0 = s.x;
        c.s1 = s.y;
    } else {
        const uint4 v0 = __ldcs(reinterpret_cast<const uint4*>(wrow) + 2 * ch);
        const uint4 v1 = __ldcs(reinterpret_cast<const uint4*>(wrow) + 2 * ch + 1);
        ik_fp8x16_to_h2(v0, c.w);
        ik_fp8x16_to_h2(v1, c.w + 8);
        const uint32_t e = srow[ch];
        c.s0 = e == 0 ? __uint_as_float(0x00400000u) : (e == 255 ? __uint_as_float(0x7fc00000u) : __uint_as_float(e << 23));
    }
}

// x: 32 bf16 (16 pairs, element 2i in the low half of word i)
template <int FMT>
IK_DEVI float chunk_dot(const WChunk& c, const uint32_t* x, float acc) {
    if (FMT == FMT_NV) {
        float p0 = 0.f, p1 = 0.f;
#pragma unroll
        for (int i = 0; i < 8; ++i) p0 = ik_fma2_bf16(c.w[i], x[i], p0);
#pragma unroll
        for (int i = 8; i < 16; ++i) p1 = ik_fma2_bf16(c.w[i], x[i], p1);
        acc = fmaf(p0, c.s0, acc);
        return fmaf(p1, c.s1, acc);
    } else {
        float p = 0.f;
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            const float2 wf = __half22float2(*reinterpret_cast<const __half2*>(&c.w[i]));
            p = fmaf(wf.x, __uint_as_float(x[i] << 16), p);
            p = fmaf(wf.y, __uint_as_float(x[i] & 0xFFFF0000u), p);
        }
        return fmaf(p, c.s0, acc);
    }
}

IK_DEVI void load_x32(const ik_bf16* __restrict__ p, uint32_t* x) {
    const uint4* q = reinterpret_cast<const uint4*>(p);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const uint4 v = __ldg(q + i);
        x[4 * i] = v.x, x[4 * i + 1] = v.y, x[4 * i + 2] = v.z, x[4 * i + 3] = v.w;
    }
}

// bytes per row of a [rows, k] weight matrix and of its scales
template <int FMT>
IK_DEVI int64_t row_bytes(int k) { return FMT == FMT_NV ? k / 2 : k; }
template <int FMT>
IK_DEVI int64_t srow_bytes(int k) { return FMT == FMT_NV ? k / 16 : k / 32; }

// ---- gate + up ---------------------------------------------------------------------------------------------------------
template <int FMT, int ACT>
__global__ void __launch_bounds__(kThreads) ink_moe_gu_kernel(MoeArgs a) {
    __shared__ SmemGroups sg;
    const Groups gr = get_groups(a, sg);
    const int warp = threadIdx.x / IK_WARP, lane = threadIdx.x % IK_WARP;
    const int q = lane >> 3, sub = lane & 7;
    const int I = a.I, H = a.H;
    const int r = blockIdx.x * kGuRows + q;
    const bool on = r < I;
    const int rr = on ? r : I - 1;
    const int nch = H / 32;
    const int64_t wb = row_bytes<FMT>(H), sb = srow_bytes<FMT>(H);
    for (int d = warp; d < gr.nd; d += kWarpsB) {
        const int e = gr.dexp[d];
        const uint8_t* wg = a.guw + ((int64_t)e * 2 * I + rr) * wb;
        const uint8_t* wu = a.guw + ((int64_t)e * 2 * I + I + rr) * wb;
        const uint8_t* sgp = a.gus + ((int64_t)e * 2 * I + rr) * sb;
        const uint8_t* sup = a.gus + ((int64_t)e * 2 * I + I + rr) * sb;
        const int o0 = gr.doff[d], o1 = gr.doff[d + 1];
        for (int pb = o0; pb < o1; pb += kNB) {
            const int nb = min(kNB, o1 - pb);
            int pp[kNB];
            float ag[kNB], au[kNB];
#pragma unroll
            for (int b = 0; b < kNB; ++b) {
                pp[b] = gr.dpair[pb + (b < nb ? b : 0)];
                ag[b] = au[b] = 0.f;
            }
            for (int ch = sub; ch < nch; ch += 8) {
                WChunk cg, cu;
                load_chunk<FMT>(wg, sgp, ch, cg);
                load_chunk<FMT>(wu, sup, ch, cu);
#pragma unroll
                for (int b = 0; b < kNB; ++b) {
                    if (b >= nb) break;
                    uint32_t xv[16];
                    load_x32(a.x + (int64_t)(pp[b] / a.K) * H + ch * 32, xv);
                    ag[b] = chunk_dot<FMT>(cg, xv, ag[b]);
                    au[b] = chunk_dot<FMT>(cu, xv, au[b]);
                }
            }
#pragma unroll
            for (int b = 0; b < kNB; ++b) {
                if (b >= nb) break;
                float sgv = ag[b], suv = au[b];
#pragma unroll
                for (int o = 4; o > 0; o >>= 1) {
                    sgv += __shfl_xor_sync(0xffffffffu, sgv, o);
                    suv += __shfl_xor_sync(0xffffffffu, suv, o);
                }
                if (sub == 0 && on) {
                    float g, u;
                    if (FMT == FMT_NV) {
                        g = ink_rbf(sgv * a.gs[e * 3 + 0]);
                        u = ink_rbf(suv * a.gs[e * 3 + 1]);
                    } else {
                        g = ink_rbf(sgv);
                        u = ink_rbf(suv);
                    }
                    a.h[(int64_t)pp[b] * I + r] = ik_f2b(act_h<ACT>(g, u, a.wts[pp[b]], a.gc, a.uc));
                }
            }
        }
    }
}

// ---- down + the token sum ----------------------------------------------------------------------------------------------
template <int FMT, int ACT, typename OutT>
__global__ void __launch_bounds__(kThreads) ink_moe_dn_kernel(MoeArgs a) {
    __shared__ SmemGroups sg;
    __shared__ float term[kDecPairs * kDnRows];
    const Groups gr = get_groups(a, sg);
    const int warp = threadIdx.x / IK_WARP, lane = threadIdx.x % IK_WARP;
    const int q = lane >> 2, sub = lane & 3;
    const int I = a.I, H = a.H;
    const int r0 = blockIdx.x * kDnRows;
    const int r = r0 + q;
    const bool on = r < H;
    const int rr = on ? r : H - 1;
    const int nch = I / 32;
    const int64_t wb = row_bytes<FMT>(I), sb = srow_bytes<FMT>(I);
    for (int d = warp; d < gr.nd; d += kWarpsB) {
        const int e = gr.dexp[d];
        const uint8_t* wd = a.dnw + ((int64_t)e * H + rr) * wb;
        const uint8_t* sd = a.dns + ((int64_t)e * H + rr) * sb;
        const int o0 = gr.doff[d], o1 = gr.doff[d + 1];
        for (int pb = o0; pb < o1; pb += kNB) {
            const int nb = min(kNB, o1 - pb);
            int pp[kNB];
            float ac[kNB];
#pragma unroll
            for (int b = 0; b < kNB; ++b) {
                pp[b] = gr.dpair[pb + (b < nb ? b : 0)];
                ac[b] = 0.f;
            }
            for (int ch = sub; ch < nch; ch += 4) {
                WChunk c;
                load_chunk<FMT>(wd, sd, ch, c);
#pragma unroll
                for (int b = 0; b < kNB; ++b) {
                    if (b >= nb) break;
                    uint32_t hv[16];
                    load_x32(a.h + (int64_t)pp[b] * I + ch * 32, hv);
                    ac[b] = chunk_dot<FMT>(c, hv, ac[b]);
                }
            }
#pragma unroll
            for (int b = 0; b < kNB; ++b) {
                if (b >= nb) break;
                float s = ac[b];
                s += __shfl_xor_sync(0xffffffffu, s, 2);
                s += __shfl_xor_sync(0xffffffffu, s, 1);
                if (sub == 0) {
                    const float yv = ink_rbf(FMT == FMT_NV ? s * a.gs[e * 3 + 2] : s);
                    const float tv = ACT == ACT_SITU ? yv : ink_rbf(yv * a.wts[pp[b]]);
                    if (a.table) {
                        if (on) a.y[(int64_t)pp[b] * H + r] = ik_f2b(tv);
                    } else {
                        term[pp[b] * kDnRows + q] = tv;
                    }
                }
            }
        }
    }
    if (a.table) return;
    __syncthreads();
    for (int i = threadIdx.x; i < a.T * kDnRows; i += blockDim.x) {
        const int t = i / kDnRows, qq = i % kDnRows;
        if (r0 + qq >= H) continue;
        float tot = 0.f;
        for (int k = 0; k < a.K; ++k) {
            const int v = a.ids[t * a.K + k];
            if (v >= 0 && v < a.E) tot += term[(t * a.K + k) * kDnRows + qq];
        }
        ik_store(reinterpret_cast<OutT*>(a.out) + (int64_t)t * H + r0 + qq, tot);
    }
}

// the table path: one block builds the pair table in scratch (temporaries in the h region, which gate_up writes later)
__global__ void __launch_bounds__(1024) ink_moe_group_kernel(MoeArgs a) {
    __shared__ int first[kEMax];
    const int P = a.T * a.K;
    int* tmp = reinterpret_cast<int*>(a.h);
    group_pairs<int*>(a.ids, P, a.E, first, a.t_exp + a.E, a.t_exp, a.t_off, a.t_pair, tmp, tmp + P, tmp + 2 * P);
}

// the table path's token sum: out[t, r] = rA(sum_k y[t*K + k, r]) over the valid slots, in slot order
template <typename OutT>
__global__ void ink_moe_gather_kernel(MoeArgs a) {
    const int t = blockIdx.y;
    for (int r = blockIdx.x * blockDim.x + threadIdx.x; r < a.H; r += gridDim.x * blockDim.x) {
        float tot = 0.f;
        for (int k = 0; k < a.K; ++k) {
            const int v = a.ids[t * a.K + k];
            if (v >= 0 && v < a.E) tot += ik_b2f(a.y[(int64_t)(t * a.K + k) * a.H + r]);
        }
        ik_store(reinterpret_cast<OutT*>(a.out) + (int64_t)t * a.H + r, tot);
    }
}

int64_t rnd(int64_t v) { return (v + 255) & ~(int64_t)255; }

template <int FMT, int ACT>
int run(MoeArgs a, int out_f32, const ik_scratch* sc, cudaStream_t st) {
    const int64_t P = (int64_t)a.T * a.K;
    const int64_t need = rnd(P * 4) + rnd((int64_t)(a.E + 1) * 4) * 2 + rnd(P * a.I * 2) + rnd(P * a.H * 2);
    if (!sc || !sc->data || sc->bytes < need) return IK_ERR_INVALID_ARG;
    char* base = (char*)sc->data;
    a.t_pair = (int*)base;
    a.t_off = (int*)(base + rnd(P * 4));
    a.t_exp = (int*)(base + rnd(P * 4) + rnd((int64_t)(a.E + 1) * 4));
    a.h = (ik_bf16*)(base + rnd(P * 4) + rnd((int64_t)(a.E + 1) * 4) * 2);
    a.y = (ik_bf16*)((char*)a.h + rnd(P * a.I * 2));
    a.table = P > kDecPairs;
    if (a.table) {
        if (P * 3 * 4 > P * a.I * 2) return IK_ERR_SHAPE;  // (the grouping's temporaries in h: I >= 6)
        ink_moe_group_kernel<<<1, 1024, 0, st>>>(a);
        IK_TRY(ik_launch_status());
    }
    if (g_probe != 2) {
        ink_moe_gu_kernel<FMT, ACT><<<(a.I + kGuRows - 1) / kGuRows, kThreads, 0, st>>>(a);
        IK_TRY(ik_launch_status());
    }
    if (g_probe != 1) {
        const unsigned g = (unsigned)((a.H + kDnRows - 1) / kDnRows);
        if (out_f32) ink_moe_dn_kernel<FMT, ACT, float><<<g, kThreads, 0, st>>>(a);
        else ink_moe_dn_kernel<FMT, ACT, ik_bf16><<<g, kThreads, 0, st>>>(a);
        IK_TRY(ik_launch_status());
        if (a.table) {
            const dim3 gg((unsigned)((a.H + 255) / 256), (unsigned)a.T);
            if (out_f32) ink_moe_gather_kernel<float><<<gg, 256, 0, st>>>(a);
            else ink_moe_gather_kernel<ik_bf16><<<gg, 256, 0, st>>>(a);
            IK_TRY(ik_launch_status());
        }
    }
    return IK_OK;
}

int common_checks(const ik_tensor* x, const ik_tensor* ids, const ik_tensor* wts, ik_tensor* out, int* out_f32) {
    IK_TRY(ik_check_a(x, IK_DTYPE_BF16, 2, 16));
    IK_TRY(ik_check(ids, IK_DTYPE_I32, 2));
    IK_TRY(ik_check(wts, IK_DTYPE_F32, 2));
    if (!out || out->ndim != 2) return IK_ERR_SHAPE;
    if (out->dtype == IK_DTYPE_F32) *out_f32 = 1;
    else if (out->dtype == IK_DTYPE_BF16) *out_f32 = 0;
    else return IK_ERR_DTYPE;
    IK_TRY(ik_check(out, out->dtype, 2));
    const int64_t T = x->shape[0], K = ids->shape[1];
    if (ids->shape[0] != T || wts->shape[0] != T || wts->shape[1] != K || out->shape[0] != T || out->shape[1] != x->shape[1])
        return IK_ERR_SHAPE;
    if (T > 65535 || K < 1 || K > 64) return IK_ERR_SHAPE;
    return IK_OK;
}

}  // namespace

extern "C" {

void ik_ink_moe_probe(int v) { g_probe = v; }

int ik_moe_experts_situ_bf16(const ik_tensor* x, const ik_tensor* ids, const ik_tensor* wts, const ik_tensor* guw,
                             const ik_tensor* gus, const ik_tensor* dnw, const ik_tensor* dns, const ik_tensor* gsc,
                             float gate_cap, float up_cap, ik_tensor* out, const ik_scratch* scratch, void* stream) {
    int of = 0;
    IK_TRY(common_checks(x, ids, wts, out, &of));
    IK_TRY(ik_check_a(guw, IK_DTYPE_U8, 3, 16));
    IK_TRY(ik_check_a(gus, IK_DTYPE_FP8_E4M3, 3, 2));
    IK_TRY(ik_check_a(dnw, IK_DTYPE_U8, 3, 16));
    IK_TRY(ik_check_a(dns, IK_DTYPE_FP8_E4M3, 3, 2));
    IK_TRY(ik_check(gsc, IK_DTYPE_F32, 2));
    const int64_t T = x->shape[0], H = x->shape[1], E = guw->shape[0], I = guw->shape[1] / 2;
    if (H % 32 || I % 32 || E < 1 || E > kEMax || guw->shape[2] != H / 2 || gus->shape[2] != H / 16 || dnw->shape[1] != H ||
        dnw->shape[2] != I / 2 || dns->shape[2] != I / 16 || gsc->shape[0] != E || gsc->shape[1] != 3)
        return IK_ERR_SHAPE;
    if (!(gate_cap > 0.f) || !(up_cap > 0.f)) return IK_ERR_INVALID_ARG;
    if (T == 0) return IK_OK;
    MoeArgs a{};
    a.x = (const ik_bf16*)x->data;
    a.ids = (const int*)ids->data;
    a.wts = (const float*)wts->data;
    a.guw = (const uint8_t*)guw->data;
    a.gus = (const uint8_t*)gus->data;
    a.dnw = (const uint8_t*)dnw->data;
    a.dns = (const uint8_t*)dns->data;
    a.gs = (const float*)gsc->data;
    a.out = out->data;
    a.T = (int)T, a.K = (int)ids->shape[1], a.E = (int)E, a.H = (int)H, a.I = (int)I;
    a.gc = gate_cap, a.uc = up_cap;
    return run<FMT_NV, ACT_SITU>(a, of, scratch, IK_STREAM(stream));
}

int ik_moe_experts_situ_bf16_rows(const ik_tensor* x, const ik_tensor* ids, const ik_tensor* wts, const ik_tensor* guw,
                                  const ik_tensor* gus, const ik_tensor* dnw, const ik_tensor* dns, const ik_tensor* gsc,
                                  float gate_cap, float up_cap, ik_tensor* out, const ik_scratch* scratch, void* stream) {
    return ik_moe_experts_situ_bf16(x, ids, wts, guw, gus, dnw, dns, gsc, gate_cap, up_cap, out, scratch, stream);
}

int ik_moe_experts_mxfp8_bf16(const ik_tensor* x, const ik_tensor* ids, const ik_tensor* wts, const ik_tensor* w13,
                              const ik_tensor* s13, const ik_tensor* w2, const ik_tensor* s2, int act, float gate_cap,
                              float up_cap, ik_tensor* out, const ik_scratch* scratch, void* stream) {
    int of = 0;
    IK_TRY(common_checks(x, ids, wts, out, &of));
    IK_TRY(ik_check_a(w13, IK_DTYPE_FP8_E4M3, 3, 16));
    IK_TRY(ik_check(s13, IK_DTYPE_U8, 3));
    IK_TRY(ik_check_a(w2, IK_DTYPE_FP8_E4M3, 3, 16));
    IK_TRY(ik_check(s2, IK_DTYPE_U8, 3));
    const int64_t T = x->shape[0], H = x->shape[1], E = w13->shape[0], I = w13->shape[1] / 2;
    if (H % 32 || I % 32 || E < 1 || E > kEMax || w13->shape[2] != H || s13->shape[2] != H / 32 || w2->shape[1] != H ||
        w2->shape[2] != I || s2->shape[2] != I / 32)
        return IK_ERR_SHAPE;
    if (act == ACT_SITU && (!(gate_cap > 0.f) || !(up_cap > 0.f))) return IK_ERR_INVALID_ARG;
    if (T == 0) return IK_OK;
    MoeArgs a{};
    a.x = (const ik_bf16*)x->data;
    a.ids = (const int*)ids->data;
    a.wts = (const float*)wts->data;
    a.guw = (const uint8_t*)w13->data;
    a.gus = (const uint8_t*)s13->data;
    a.dnw = (const uint8_t*)w2->data;
    a.dns = (const uint8_t*)s2->data;
    a.out = out->data;
    a.T = (int)T, a.K = (int)ids->shape[1], a.E = (int)E, a.H = (int)H, a.I = (int)I;
    a.gc = gate_cap, a.uc = up_cap;
    cudaStream_t st = IK_STREAM(stream);
    switch (act) {
        case ACT_SILU: return run<FMT_MX, ACT_SILU>(a, of, scratch, st);
        case ACT_GELU: return run<FMT_MX, ACT_GELU>(a, of, scratch, st);
        case ACT_SITU: return run<FMT_MX, ACT_SITU>(a, of, scratch, st);
        default: return IK_ERR_INVALID_ARG;
    }
}

int ik_moe_experts_mxfp8_bf16_rows(const ik_tensor* x, const ik_tensor* ids, const ik_tensor* wts, const ik_tensor* w13,
                                   const ik_tensor* s13, const ik_tensor* w2, const ik_tensor* s2, int act, float gate_cap,
                                   float up_cap, ik_tensor* out, const ik_scratch* scratch, void* stream) {
    return ik_moe_experts_mxfp8_bf16(x, ids, wts, w13, s13, w2, s2, act, gate_cap, up_cap, out, scratch, stream);
}

}  // extern "C"
