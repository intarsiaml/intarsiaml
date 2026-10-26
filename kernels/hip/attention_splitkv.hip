// Split-KV ("flash-decoding") causal GQA attention over a contiguous bf16 KV cache [Nkv, cap, D],
// for decode (T == 1) and verify / short prefill (T <= 16 query rows).
//
// Stage 1 (splitkv_kernel): grid (split, kv head, query row), 256 threads. The keys 0 .. pos0+t are cut into
//   `nsplit` chunks; a block owns one chunk of one kv head and serves ALL G = N/Nkv query heads of that group,
//   so K and V are read from memory once per group instead of once per head. The block is a set of
//   NG = 256/TPK key lanes of TPK = D/8 threads each; a thread owns 8 head dims (one 16 B load) of every key
//   its lane processes, so a score needs only a shuffle reduction inside the lane and the PV accumulation
//   is local. Every lane runs its own online softmax over its keys; the lanes are merged through shared
//   memory at the end and one (m, l, acc[D]) partial per (row, head, split) goes to scratch.
// Stage 2 (combine_kernel): per (row, head) merge the nsplit partials and write the bf16 output.
//
// Scratch (ik_scratch): float acc[T][N][nsplit][D] followed by float2 ml[T][N][nsplit]; see splitkv_scratch_bytes.
// The split policy depends on kv_len, the number of query rows and n_kv_heads (never on the data), so results are
// run-to-run identical for a given call shape.
// The policy is mirrored in crates/intarsia-backend-hip/src/kernels.rs (splitkv_parts); keep the two in sync.
#include "ik_hip.h"

extern "C" {
int ik_attn_prefill_bf16(const ik_tensor* q, const ik_tensor* kc, const ik_tensor* vc, int64_t pos0, float scale,
                         ik_tensor* out, void* stream);
}

namespace {

constexpr int kBlock = 256;
constexpr int kMaxGroup = 8;      // q heads per kv head handled by one block
constexpr int kMaxRows = 64;      // query rows per launch (grid.z)
constexpr int kMinKeys = 64;      // keys per split, at least
constexpr int kMaxSplit = 64;
constexpr int kTargetBlocks = 256;
constexpr float kLog2e = 1.4426950408889634f;

__host__ __device__ inline int splitkv_count(int64_t kv_len, int64_t nkv, int64_t rows) {
    int64_t by_len = (kv_len + kMinKeys - 1) / kMinKeys;
    int64_t by_fill = (kTargetBlocks + nkv * rows - 1) / (nkv * rows);
    int64_t n = by_len < by_fill ? by_len : by_fill;
    if (n > kMaxSplit) n = kMaxSplit;
    if (n < 1) n = 1;
    return (int)n;
}

__host__ __device__ inline int64_t splitkv_chunk(int64_t kv_len, int nsplit) {
    int64_t c = (kv_len + nsplit - 1) / nsplit;
    return (c + kMinKeys - 1) / kMinKeys * kMinKeys;
}

int64_t splitkv_scratch_bytes(int64_t T, int64_t N, int64_t D, int nsplit) {
    return T * N * nsplit * (D + 2) * 4;
}

// INTARSIA_HIP_SPLITKV_R8 = 0: the pre-round-8 combine (one split at a time); unset or anything else: round 8's (16 splits'
// loads in flight per batch). Both give the same bits (tests/splitkv_r8.rs). Read at every launch.
inline bool splitkv_r8() {
    const char* e = getenv("INTARSIA_HIP_SPLITKV_R8");
    return !(e && e[0] == '0' && e[1] == 0);
}

//
// RTAB (batched decode, row-table launch, `OpCall::rows`): grid.z is the SEQUENCE, each a T = 1 decode call: its position is
// `pos_dev[z]`, its K / V caches are the buffers whose addresses are entries [z][tindex] and [z][tindex + 1] of the row table `tab`,
// and its split policy is the one-query-row decode policy (`tq = 1`, not grid.z). The query row, the partials and the output row of
// sequence z are row z, as for z query rows of one sequence. All other code is shared, so sequence z is bitwise the one-row launch
// on it (the scratch holds `rows` rows of partials, one row's worth each).
template <int GP, int TPK, int KB, bool RTAB = false>
__device__ __forceinline__ void
splitkv_body(const ik_bf16* __restrict__ q, const ik_bf16* __restrict__ kc, const ik_bf16* __restrict__ vc,
               float* __restrict__ pacc, float2* __restrict__ pml, int N, int G, int64_t cap, int64_t pos0_host,
               float qscale, int nsplit, const int32_t* pos_dev, const uint8_t* tab = nullptr, int64_t tstride = 0,
               int64_t tindex = 0) {
    // `nsplit` is the grid's split count and the stride of the partials. The splits actually used follow from the key
    // count alone (device position when given), so the grid may be launched at its maximum (device-side position)
    // and the surplus blocks leave at once: the partitioning, and so every bit of the result, is the same either way.
    int64_t tq = gridDim.z;  // query rows of the sequence
    int64_t pos0;
    if constexpr (RTAB) {
        tq = 1;
        pos0 = (int64_t)pos_dev[blockIdx.z];
        const uint64_t* tp = reinterpret_cast<const uint64_t*>(tab) + (int64_t)blockIdx.z * tstride + tindex;
        kc = reinterpret_cast<const ik_bf16*>(tp[0]);
        vc = reinterpret_cast<const ik_bf16*>(tp[1]);
    } else {
        pos0 = pos_dev ? (int64_t)*pos_dev : pos0_host;
    }
    const int64_t kv_len = pos0 + tq;
    const int nsplit_eff = splitkv_count(kv_len, gridDim.y, tq);
    const int64_t chunk = splitkv_chunk(kv_len, nsplit_eff);
    if ((int)blockIdx.x >= nsplit_eff) return;
    constexpr int D = TPK * 8;
    constexpr int NG = kBlock / TPK;
    __shared__ __align__(16) float s_acc[NG * D];
    __shared__ float s_m[NG], s_l[NG];

    const int sp = blockIdx.x, kvh = blockIdx.y, t = blockIdx.z;
    const int c = threadIdx.x % TPK, gk = threadIdx.x / TPK;
    const int64_t visible = RTAB ? pos0 + 1 : pos0 + t + 1;
    const int64_t start = (int64_t)sp * chunk;
    int64_t end = start + chunk;
    if (end > visible) end = visible;

    const ik_bf16* kb = kc + (int64_t)kvh * cap * D + c * 8;
    const ik_bf16* vb = vc + (int64_t)kvh * cap * D + c * 8;

    float qr[GP][8];
#pragma unroll
    for (int g = 0; g < GP; ++g) {
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            qr[g][e] = 0.f;
            if (g < G) qr[g][e] = ik_b2f(q[((int64_t)t * N + (int64_t)kvh * G + g) * D + c * 8 + e]) * qscale;
        }
    }

    float m[GP], l[GP], acc[GP][8];
#pragma unroll
    for (int g = 0; g < GP; ++g) {
        m[g] = -INFINITY;
        l[g] = 0.f;
#pragma unroll
        for (int e = 0; e < 8; ++e) acc[g][e] = 0.f;
    }

    for (int64_t base = start; base < end; base += (int64_t)NG * KB) {
        uint4 kk[KB], vv[KB];
        bool ok[KB];
#pragma unroll
        for (int u = 0; u < KB; ++u) {
            const int64_t j = base + (int64_t)u * NG + gk;
            ok[u] = j < end;
            kk[u] = make_uint4(0, 0, 0, 0);
            vv[u] = make_uint4(0, 0, 0, 0);
            if (ok[u]) {
                kk[u] = ik_ldg(reinterpret_cast<const uint4*>(kb + j * D));
                vv[u] = ik_ldg(reinterpret_cast<const uint4*>(vb + j * D));
            }
        }
        float sc[KB][GP];
#pragma unroll
        for (int u = 0; u < KB; ++u) {
            float kf[8];
            ik_unpack8(kk[u], kf);
#pragma unroll
            for (int g = 0; g < GP; ++g) {
                float d = 0.f;
#pragma unroll
                for (int e = 0; e < 8; ++e) d += qr[g][e] * kf[e];
#pragma unroll
                for (int o = TPK / 2; o > 0; o >>= 1) d += __shfl_xor(d, o);
                sc[u][g] = ok[u] ? d : -INFINITY;
            }
        }
        float vf[KB][8];
#pragma unroll
        for (int u = 0; u < KB; ++u) ik_unpack8(vv[u], vf[u]);
#pragma unroll
        for (int g = 0; g < GP; ++g) {
            float tmax = sc[0][g];
#pragma unroll
            for (int u = 1; u < KB; ++u) tmax = fmaxf(tmax, sc[u][g]);
            const float m_new = fmaxf(m[g], tmax);
            const float m_use = (m_new == -INFINITY) ? 0.f : m_new;
            const float corr = exp2f(m[g] - m_use);  // m == -inf -> 0
            m[g] = m_new;
            float p[KB], psum = 0.f;
#pragma unroll
            for (int u = 0; u < KB; ++u) {
                p[u] = exp2f(sc[u][g] - m_use);
                psum += p[u];
            }
            l[g] = l[g] * corr + psum;
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                float a = acc[g][e] * corr;
#pragma unroll
                for (int u = 0; u < KB; ++u) a += p[u] * vf[u][e];
                acc[g][e] = a;
            }
        }
    }

    // merge the NG key lanes, one head at a time; write the block's partial for that head
#pragma unroll
    for (int g = 0; g < GP; ++g) {
        if (g < G) {
            IK_LDS_BARRIER();  // s_acc / s_m / s_l are reused for every head
#pragma unroll
            for (int e = 0; e < 8; ++e) s_acc[gk * D + c * 8 + e] = acc[g][e];
            if (c == 0) {
                s_m[gk] = m[g];
                s_l[gk] = l[g];
            }
            IK_LDS_BARRIER();
            const int64_t slot = ((int64_t)t * N + (int64_t)kvh * G + g) * nsplit + sp;
            if (threadIdx.x < D) {
                float M = -INFINITY;
                for (int i = 0; i < NG; ++i) M = fmaxf(M, s_m[i]);
                float A = 0.f, L = 0.f;
                if (M != -INFINITY) {
                    for (int i = 0; i < NG; ++i) {
                        const float w = exp2f(s_m[i] - M);  // empty lane: exp2(-inf) = 0, acc is 0 too
                        L += s_l[i] * w;
                        A += s_acc[i * D + threadIdx.x] * w;
                    }
                }
                pacc[slot * D + threadIdx.x] = A;
                if (threadIdx.x == 0) pml[slot] = make_float2(M, L);
            }
        }
    }
}

template <int GP, int TPK, int KB, bool RTAB = false>
__global__ void __launch_bounds__(kBlock)
splitkv_kernel(const ik_bf16* __restrict__ q, const ik_bf16* __restrict__ kc, const ik_bf16* __restrict__ vc,
               float* __restrict__ pacc, float2* __restrict__ pml, int N, int G, int64_t cap, int64_t pos0_host,
               float qscale, int nsplit, const int32_t* pos_dev, const uint8_t* tab = nullptr, int64_t tstride = 0,
               int64_t tindex = 0) {
    splitkv_body<GP, TPK, KB, RTAB>(q, kc, vc, pacc, pml, N, G, cap, pos0_host, qscale, nsplit, pos_dev, tab, tstride, tindex);
}

// Round 9 (docs/results/2026-10-19-hip-round9.md): the same body with the register budget of 8 waves per SIMD. GP = 8
// (the 35B: 16 q heads over 2 kv heads) takes 212 VGPRs, i.e. 7 waves per SIMD, so of the 128 blocks of 8 waves a
// long-context decode launches (64 splits x 2 kv heads) only 112 are resident and 16 run as a second round. At <= 192 VGPRs
// all 128 fit. Same instructions in the same order, only allocated differently: the same bits (tests/splitkv_occ.rs, ids).
// Opt-in, INTARSIA_HIP_SPLITKV_OCC=1: at 192 VGPRs it spills 28 bytes per lane, which costs ~30 % per launch at short
// context (12.4 -> 16.1 us, ~0.6 % of a 128-token decode step) and wins ~12 % per launch at ~17.6k tokens.
template <int GP, int TPK, int KB>
__global__ void __launch_bounds__(kBlock) __attribute__((amdgpu_waves_per_eu(8)))
splitkv_kernel_o8(const ik_bf16* __restrict__ q, const ik_bf16* __restrict__ kc, const ik_bf16* __restrict__ vc,
                  float* __restrict__ pacc, float2* __restrict__ pml, int N, int G, int64_t cap, int64_t pos0_host,
                  float qscale, int nsplit, const int32_t* pos_dev) {
    splitkv_body<GP, TPK, KB>(q, kc, vc, pacc, pml, N, G, cap, pos0_host, qscale, nsplit, pos_dev);  // (8 waves)
}

inline bool splitkv_occ() {
    const char* e = getenv("INTARSIA_HIP_SPLITKV_OCC");
    return e && e[0] == '1' && e[1] == 0;
}

// RTAB: `T` rows are sequences of one query row each (stage 1's RTAB), row `t * N + h` belongs to sequence `t = row / N`.
template <int D, int kCombBatch, bool RTAB = false>
__global__ void __launch_bounds__(D)
combine_kernel(const float* __restrict__ pacc, const float2* __restrict__ pml, ik_bf16* __restrict__ out, int nsplit,
               int T, int NKV, int64_t pos0_host, const int32_t* pos_dev, int N = 0) {
    const int64_t row = blockIdx.x;  // t * N + h
    const int64_t pos0 = RTAB ? (int64_t)pos_dev[row / N] : (pos_dev ? (int64_t)*pos_dev : pos0_host);
    const int64_t tq = RTAB ? 1 : T;
    const int nsplit_eff = splitkv_count(pos0 + tq, NKV, tq);  // same count as the stage-1 blocks used
    const float2* ml = pml + row * nsplit;
    const float* acc = pacc + row * nsplit * D;
    // Round 8: the partials are loaded kCombBatch (16) splits at a time, all loads of a batch in flight before its
    // arithmetic (the plain loop, kCombBatch = 1, waited for one load per split: ~13.6 us per launch at 64 splits, ~1 MB).
    // The max and the two sums still run over s = 0, 1, .. in order with the same operations, so the bits are the same.
    float M = -INFINITY;
    for (int s0 = 0; s0 < nsplit_eff; s0 += kCombBatch) {
        float mx[kCombBatch];
#pragma unroll
        for (int u = 0; u < kCombBatch; ++u) mx[u] = s0 + u < nsplit_eff ? ml[s0 + u].x : -INFINITY;
#pragma unroll
        for (int u = 0; u < kCombBatch; ++u)
            if (s0 + u < nsplit_eff) M = fmaxf(M, mx[u]);
    }
    float A = 0.f, L = 0.f;
    for (int s0 = 0; s0 < nsplit_eff; s0 += kCombBatch) {
        float2 v[kCombBatch];
        float a[kCombBatch];
#pragma unroll
        for (int u = 0; u < kCombBatch; ++u) {
            const int s = s0 + u < nsplit_eff ? s0 + u : s0;  // past the end: a split of this batch, unused
            v[u] = ml[s];
            a[u] = acc[(int64_t)s * D + threadIdx.x];
        }
#pragma unroll
        for (int u = 0; u < kCombBatch; ++u) {
            if (s0 + u >= nsplit_eff || v[u].x == -INFINITY) continue;  // empty split: acc is not meaningful
            const float w = exp2f(v[u].x - M);
            L += v[u].y * w;
            A += a[u] * w;
        }
    }
    out[row * D + threadIdx.x] = ik_f2b(A / L);
}

template <int GP, int TPK, int KB>
void launch_split(const ik_bf16* q, const ik_bf16* kc, const ik_bf16* vc, float* pacc, float2* pml, int64_t T,
                  int64_t N, int64_t NKV, int64_t cap, int64_t pos0, float qscale, int nsplit,
                  const int32_t* pos_dev, hipStream_t s) {
    dim3 grid(nsplit, (unsigned)NKV, (unsigned)T);
    if (splitkv_occ())
        splitkv_kernel_o8<GP, TPK, KB><<<grid, kBlock, 0, s>>>(q, kc, vc, pacc, pml, (int)N, (int)(N / NKV), cap, pos0,
                                                               qscale, nsplit, pos_dev);
    else
        splitkv_kernel<GP, TPK, KB><<<grid, kBlock, 0, s>>>(q, kc, vc, pacc, pml, (int)N, (int)(N / NKV), cap, pos0,
                                                            qscale, nsplit, pos_dev);
}

template <int D>
void launch_combine(const float* pacc, const float2* pml, ik_bf16* out, unsigned rows, int nsplit, int T, int NKV,
                    int64_t pos0, const int32_t* pos_dev, hipStream_t s) {
    if (splitkv_r8())
        combine_kernel<D, 16><<<rows, D, 0, s>>>(pacc, pml, out, nsplit, T, NKV, pos0, pos_dev);
    else
        combine_kernel<D, 1><<<rows, D, 0, s>>>(pacc, pml, out, nsplit, T, NKV, pos0, pos_dev);
}

template <int TPK>
int dispatch_group(int G, const ik_bf16* q, const ik_bf16* kc, const ik_bf16* vc, float* pacc, float2* pml,
                   int64_t T, int64_t N, int64_t NKV, int64_t cap, int64_t pos0, float qscale, int nsplit,
                   const int32_t* pos_dev, hipStream_t s) {
#define IK_SPLIT_CASE(GPV, KBV)                                                                                \
    launch_split<GPV, TPK, KBV>(q, kc, vc, pacc, pml, T, N, NKV, cap, pos0, qscale, nsplit, pos_dev, s);       \
    return IK_OK;
    if (G <= 1) { IK_SPLIT_CASE(1, 4) }
    if (G <= 2) { IK_SPLIT_CASE(2, 4) }
    if (G <= 4) { IK_SPLIT_CASE(4, 4) }
    if (G <= 8) { IK_SPLIT_CASE(8, 2) }
#undef IK_SPLIT_CASE
    return IK_ERR_SHAPE;
}

// Row-table decode stage 1 (`OpCall::rows`): `rows` sequences of one query row each; always the plain kernel (INTARSIA_HIP_SPLITKV_OCC's
// o8 build has the same instructions and so the same bits).
template <int GP, int TPK, int KB>
void launch_split_rows(const ik_bf16* q, const ik_bf16* kc, const ik_bf16* vc, float* pacc, float2* pml, int64_t T, int64_t N,
                       int64_t NKV, int64_t cap, float qscale, int nsplit, const int32_t* pos_dev, const uint8_t* tab,
                       int64_t tstride, int64_t tindex, hipStream_t s) {
    dim3 grid(nsplit, (unsigned)NKV, (unsigned)T);
    splitkv_kernel<GP, TPK, KB, true><<<grid, kBlock, 0, s>>>(q, kc, vc, pacc, pml, (int)N, (int)(N / NKV), cap, 0, qscale,
                                                              nsplit, pos_dev, tab, tstride, tindex);
}

template <int TPK>
int dispatch_group_rows(int G, const ik_bf16* q, const ik_bf16* kc, const ik_bf16* vc, float* pacc, float2* pml, int64_t T,
                        int64_t N, int64_t NKV, int64_t cap, float qscale, int nsplit, const int32_t* pos_dev,
                        const uint8_t* tab, int64_t tstride, int64_t tindex, hipStream_t s) {
#define IK_SPLIT_ROWS_CASE(GPV, KBV)                                                                                        \
    launch_split_rows<GPV, TPK, KBV>(q, kc, vc, pacc, pml, T, N, NKV, cap, qscale, nsplit, pos_dev, tab, tstride, tindex, s); \
    return IK_OK;
    if (G <= 1) { IK_SPLIT_ROWS_CASE(1, 4) }
    if (G <= 2) { IK_SPLIT_ROWS_CASE(2, 4) }
    if (G <= 4) { IK_SPLIT_ROWS_CASE(4, 4) }
    if (G <= 8) { IK_SPLIT_ROWS_CASE(8, 2) }
#undef IK_SPLIT_ROWS_CASE
    return IK_ERR_SHAPE;
}

// Row-table decode (`OpCall::rows`): `rows` sequences of one query row each (q [rows, N, D]); `kc0` / `vc0` are row 0's caches
// ([NKV, cap, D], shape carriers); sequence r's caches are the buffers at entries [r][index] and [r][index + 1] of `tab`, its
// position is pos[r]. Same grid policy as the one-row decode launch with a device position (the maximum split count), `rows` rows of
// scratch.
int splitkv_launch_rows(const ik_tensor* q, const ik_tensor* kc0, const ik_tensor* vc0, float scale, ik_tensor* out,
                        const ik_scratch* scratch, const ik_tensor* tab, int64_t stride, int64_t index, const ik_tensor* pos,
                        void* stream) {
    IK_TRY(ik_check(q, IK_DTYPE_BF16, 3));
    IK_TRY(ik_check(kc0, IK_DTYPE_BF16, 3));
    IK_TRY(ik_check(vc0, IK_DTYPE_BF16, 3));
    IK_TRY(ik_check(out, IK_DTYPE_BF16, 3));
    const int64_t R = q->shape[0], N = q->shape[1], D = q->shape[2];
    const int64_t NKV = kc0->shape[0], cap = kc0->shape[1];
    if (vc0->shape[0] != NKV || vc0->shape[1] != cap || vc0->shape[2] != D || kc0->shape[2] != D) return IK_ERR_SHAPE;
    if (out->shape[0] != R || out->shape[1] != N || out->shape[2] != D) return IK_ERR_SHAPE;
    if (NKV <= 0 || N % NKV != 0 || R > kMaxRows || N / NKV > kMaxGroup) return IK_ERR_SHAPE;
    if (D != 32 && D != 64 && D != 128 && D != 256) return IK_ERR_SHAPE;
    if (!pos || !pos->data || pos->dtype != IK_DTYPE_I32 || ik_numel(pos) != R) return IK_ERR_INVALID_ARG;
    IK_TRY(ik_check_tab(tab, R, stride, index));
    if (index + 1 >= stride) return IK_ERR_INVALID_ARG;
    if (R == 0 || N == 0) return IK_OK;
    const int32_t* pos_dev = (const int32_t*)pos->data;
    const int nsplit = splitkv_count(INT64_MAX / 2, NKV, 1);  // the one-row decode launch's grid
    const int64_t need = splitkv_scratch_bytes(R, N, D, nsplit);
    if (!scratch || !scratch->data || scratch->bytes < need) return IK_ERR_INVALID_ARG;
    float* pacc = (float*)scratch->data;
    float2* pml = (float2*)((char*)scratch->data + R * N * nsplit * D * 4);
    hipStream_t s = IK_STREAM(stream);
    const ik_bf16* qp = (const ik_bf16*)q->data;
    const ik_bf16* kp = (const ik_bf16*)kc0->data;
    const ik_bf16* vp = (const ik_bf16*)vc0->data;
    const uint8_t* tp = (const uint8_t*)tab->data;
    const float qscale = scale * kLog2e;
    int rc = IK_ERR_SHAPE;
    switch (D) {
        case 32: rc = dispatch_group_rows<4>((int)(N / NKV), qp, kp, vp, pacc, pml, R, N, NKV, cap, qscale, nsplit, pos_dev, tp, stride, index, s); break;
        case 64: rc = dispatch_group_rows<8>((int)(N / NKV), qp, kp, vp, pacc, pml, R, N, NKV, cap, qscale, nsplit, pos_dev, tp, stride, index, s); break;
        case 128: rc = dispatch_group_rows<16>((int)(N / NKV), qp, kp, vp, pacc, pml, R, N, NKV, cap, qscale, nsplit, pos_dev, tp, stride, index, s); break;
        case 256: rc = dispatch_group_rows<32>((int)(N / NKV), qp, kp, vp, pacc, pml, R, N, NKV, cap, qscale, nsplit, pos_dev, tp, stride, index, s); break;
    }
    if (rc != IK_OK) return rc;
    IK_TRY(ik_launch_status());
    ik_bf16* op = (ik_bf16*)out->data;
    const unsigned rows = (unsigned)(R * N);
    const bool r8 = splitkv_r8();
#define IK_COMBINE_ROWS(DV)                                                                                                  \
    if (r8) combine_kernel<DV, 16, true><<<rows, DV, 0, s>>>(pacc, pml, op, nsplit, (int)R, (int)NKV, 0, pos_dev, (int)N);  \
    else combine_kernel<DV, 1, true><<<rows, DV, 0, s>>>(pacc, pml, op, nsplit, (int)R, (int)NKV, 0, pos_dev, (int)N);
    switch (D) {
        case 32: IK_COMBINE_ROWS(32) break;
        case 64: IK_COMBINE_ROWS(64) break;
        case 128: IK_COMBINE_ROWS(128) break;
        case 256: IK_COMBINE_ROWS(256) break;
    }
#undef IK_COMBINE_ROWS
    return ik_launch_status();
}

int splitkv_launch(const ik_tensor* q, const ik_tensor* kc, const ik_tensor* vc, int64_t pos0, float scale,
                   ik_tensor* out, const ik_scratch* scratch, const ik_tensor* pos, void* stream, bool decode) {
    IK_TRY(ik_check_pos(pos));
    const int32_t* pos_dev = pos ? (const int32_t*)pos->data : nullptr;
    IK_TRY(ik_check(q, IK_DTYPE_BF16, 3));
    IK_TRY(ik_check(kc, IK_DTYPE_BF16, 3));
    IK_TRY(ik_check(vc, IK_DTYPE_BF16, 3));
    IK_TRY(ik_check(out, IK_DTYPE_BF16, 3));
    const int64_t T = q->shape[0], N = q->shape[1], D = q->shape[2];
    const int64_t NKV = kc->shape[0], cap = kc->shape[1];
    if (decode && T != 1) return IK_ERR_SHAPE;
    if (vc->shape[0] != NKV || vc->shape[1] != cap || vc->shape[2] != D || kc->shape[2] != D) return IK_ERR_SHAPE;
    if (out->shape[0] != T || out->shape[1] != N || out->shape[2] != D) return IK_ERR_SHAPE;
    if (NKV <= 0 || N % NKV != 0 || pos0 < 0 || pos0 + T > cap || T > kMaxRows) return IK_ERR_SHAPE;
    if (N / NKV > kMaxGroup) return IK_ERR_SHAPE;
    if (D != 32 && D != 64 && D != 128 && D != 256) return IK_ERR_SHAPE;
    if (T == 0 || N == 0) return IK_OK;
    if (!ik_aligned16(kc->data) || !ik_aligned16(vc->data)) return IK_ERR_STRIDE;

    // With a device-side position the real key count is unknown at launch: the grid is launched at its length-
    // independent maximum (the scratch bound, kernels.rs splitkv_parts(usize::MAX / 2, ..)) and the blocks past the
    // real split count exit. With a host position the grid is sized from the key count; both partition identically.
    const int nsplit = pos_dev ? splitkv_count(INT64_MAX / 2, NKV, T) : splitkv_count(pos0 + T, NKV, T);
    const int64_t need = splitkv_scratch_bytes(T, N, D, nsplit);
    if (!scratch || !scratch->data || scratch->bytes < need) return IK_ERR_INVALID_ARG;
    float* pacc = (float*)scratch->data;
    float2* pml = (float2*)((char*)scratch->data + T * N * nsplit * D * 4);

    hipStream_t s = IK_STREAM(stream);
    const ik_bf16* qp = (const ik_bf16*)q->data;
    const ik_bf16* kp = (const ik_bf16*)kc->data;
    const ik_bf16* vp = (const ik_bf16*)vc->data;
    const float qscale = scale * kLog2e;
    int rc = IK_ERR_SHAPE;
    switch (D) {
        case 32: rc = dispatch_group<4>((int)(N / NKV), qp, kp, vp, pacc, pml, T, N, NKV, cap, pos0, qscale, nsplit, pos_dev, s); break;
        case 64: rc = dispatch_group<8>((int)(N / NKV), qp, kp, vp, pacc, pml, T, N, NKV, cap, pos0, qscale, nsplit, pos_dev, s); break;
        case 128: rc = dispatch_group<16>((int)(N / NKV), qp, kp, vp, pacc, pml, T, N, NKV, cap, pos0, qscale, nsplit, pos_dev, s); break;
        case 256: rc = dispatch_group<32>((int)(N / NKV), qp, kp, vp, pacc, pml, T, N, NKV, cap, pos0, qscale, nsplit, pos_dev, s); break;
    }
    if (rc != IK_OK) return rc;
    IK_TRY(ik_launch_status());
    ik_bf16* op = (ik_bf16*)out->data;
    const unsigned rows = (unsigned)(T * N);
    switch (D) {
        case 32: launch_combine<32>(pacc, pml, op, rows, nsplit, (int)T, (int)NKV, pos0, pos_dev, s); break;
        case 64: launch_combine<64>(pacc, pml, op, rows, nsplit, (int)T, (int)NKV, pos0, pos_dev, s); break;
        case 128: launch_combine<128>(pacc, pml, op, rows, nsplit, (int)T, (int)NKV, pos0, pos_dev, s); break;
        case 256: launch_combine<256>(pacc, pml, op, rows, nsplit, (int)T, (int)NKV, pos0, pos_dev, s); break;
    }
    return ik_launch_status();
}

}  // namespace

extern "C" {

// ik_hip_abi.h (round 9 pin, tests/splitkv_occ.rs): 1 when stage 1 runs as splitkv_kernel_o8 (INTARSIA_HIP_SPLITKV_OCC=1)
int ik_hip_splitkv_occ(void) { return splitkv_occ() ? 1 : 0; }

int ik_attn_decode_bf16_splitkv(const ik_tensor* q, const ik_tensor* kc, const ik_tensor* vc, int64_t pos0,
                                float scale, ik_tensor* out, const ik_scratch* scratch, const ik_tensor* pos,
                                void* stream) {
    return splitkv_launch(q, kc, vc, pos0, scale, out, scratch, pos, stream, true);
}

int ik_attn_decode_bf16_splitkv_rows(const ik_tensor* q, const ik_tensor* kc0, const ik_tensor* vc0, float scale,
                                     ik_tensor* out, const ik_scratch* scratch, const ik_tensor* tab, int64_t stride,
                                     int64_t index, const ik_tensor* pos, void* stream) {
    return splitkv_launch_rows(q, kc0, vc0, scale, out, scratch, tab, stride, index, pos, stream);
}

int ik_attn_prefill_bf16_splitkv(const ik_tensor* q, const ik_tensor* kc, const ik_tensor* vc, int64_t pos0,
                                 float scale, ik_tensor* out, const ik_scratch* scratch, void* stream) {
    return splitkv_launch(q, kc, vc, pos0, scale, out, scratch, nullptr, stream, false);
}

}  // extern "C"
