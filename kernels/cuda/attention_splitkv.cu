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
// The split policy depends on kv_len only (never on the data), so results are run-to-run identical.
// The policy is mirrored in crates/intarsia-backend-cuda/src/kernels.rs (splitkv_parts); keep the two in sync.
//
// fp8 KV cache (attn_*_f8kv_splitkv, opt-in through an fp8_e4m3 cache): the same kernels with KT = e4m3 bytes. A thread's
// 8 dims of a key are one 8-byte load instead of 16, converted exactly to f32 (every e4m3 value is a bf16 / f32 value);
// everything after the conversion is the bf16 kernel's code, so the output is bitwise the bf16 kernel's on the cache
// upcast to bf16 (tests/attn_f8kv.rs). The bf16 instantiations compile to the same SASS as before the template parameter.
#include "ik_launch.h"
#include "ik_quant.cuh"

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

// the cache element: bf16 (16 B per thread per key) or e4m3 bytes (8 B); load_kv8 returns the raw bits of 8 dims
template <typename KT>
struct KvLoad;
template <>
struct KvLoad<ik_bf16> {
    using V = uint4;
    static __device__ __forceinline__ V zero() { return make_uint4(0, 0, 0, 0); }
    static __device__ __forceinline__ V load(const ik_bf16* p) { return __ldg(reinterpret_cast<const uint4*>(p)); }
    static __device__ __forceinline__ void unpack(const V& v, float* f) { ik_unpack8(v, f); }
};
template <>
struct KvLoad<uint8_t> {
    using V = uint2;
    static __device__ __forceinline__ V zero() { return make_uint2(0, 0); }
    static __device__ __forceinline__ V load(const uint8_t* p) { return __ldg(reinterpret_cast<const uint2*>(p)); }
    static __device__ __forceinline__ void unpack(const V& v, float* f) {
        const uint32_t w[2] = {v.x, v.y};
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const float2 a = ik_e4m3x2_to_f2(static_cast<unsigned short>(w[i] & 0xFFFFu));
            const float2 b = ik_e4m3x2_to_f2(static_cast<unsigned short>(w[i] >> 16));
            f[4 * i + 0] = a.x;
            f[4 * i + 1] = a.y;
            f[4 * i + 2] = b.x;
            f[4 * i + 3] = b.y;
        }
    }
};

//
// ROWS (batched decode, row-table launch, `OpCall::rows`): grid.z is the SEQUENCE, each a T = 1 decode call: its position is
// `pos_dev[z]`, its K / V caches are the buffers whose addresses are entries [z][tindex] and [z][tindex + 1] of the row table
// `tab`, and its split policy is the one-query-row decode policy (`tq = 1`, not grid.z). The query row, the partials and the
// output row of sequence z are row z, as for z query rows of one sequence. All other code is shared, so sequence z is
// bitwise the one-row launch on it (the scratch holds `rows` rows of partials, one row's worth each).
template <int GP, int TPK, int KB, typename KT = ik_bf16, bool ROWS = false>
__global__ void __launch_bounds__(kBlock)
splitkv_kernel(const ik_bf16* __restrict__ q, const KT* __restrict__ kc, const KT* __restrict__ vc,
               float* __restrict__ pacc, float2* __restrict__ pml, int N, int G, int64_t cap, int64_t pos0_host,
               float qscale, int nsplit, int64_t window, const int32_t* pos_dev, const uint8_t* tab = nullptr,
               int64_t tstride = 0, int64_t tindex = 0) {
    // `nsplit` is the grid's split count and the stride of the partials. The splits actually used follow from the key
    // count alone (device position when given), so the grid may be launched at its maximum (decode, CUDA graphs) and
    // the surplus blocks leave at once: the partitioning, and so every bit of the result, is the same either way.
    int64_t tq = gridDim.z;  // query rows of the sequence
    int64_t pos0;
    if constexpr (ROWS) {
        tq = 1;
        pos0 = (int64_t)pos_dev[blockIdx.z];
        const uint64_t* tp = reinterpret_cast<const uint64_t*>(tab) + (int64_t)blockIdx.z * tstride + tindex;
        kc = reinterpret_cast<const KT*>(tp[0]);
        vc = reinterpret_cast<const KT*>(tp[1]);
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
    const int64_t visible = ROWS ? pos0 + 1 : pos0 + t + 1;
    // sliding window (window > 0): the query attends keys max(0, visible - window)..visible only; a split wholly outside
    // it is empty (m = -inf), which the lane merge and the combine already handle. window 0 leaves `start` untouched.
    const int64_t lo = (window > 0 && visible > window) ? visible - window : 0;
    int64_t start = (int64_t)sp * chunk;
    int64_t end = start + chunk;
    if (start < lo) start = lo;
    if (end > visible) end = visible;

    const KT* kb = kc + (int64_t)kvh * cap * D + c * 8;
    const KT* vb = vc + (int64_t)kvh * cap * D + c * 8;
    using L = KvLoad<KT>;

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
        typename L::V kk[KB], vv[KB];
        bool ok[KB];
#pragma unroll
        for (int u = 0; u < KB; ++u) {
            const int64_t j = base + (int64_t)u * NG + gk;
            ok[u] = j < end;
            kk[u] = L::zero();
            vv[u] = L::zero();
            if (ok[u]) {
                kk[u] = L::load(kb + j * D);
                vv[u] = L::load(vb + j * D);
            }
        }
        float sc[KB][GP];
#pragma unroll
        for (int u = 0; u < KB; ++u) {
            float kf[8];
            L::unpack(kk[u], kf);
#pragma unroll
            for (int g = 0; g < GP; ++g) {
                float d = 0.f;
#pragma unroll
                for (int e = 0; e < 8; ++e) d += qr[g][e] * kf[e];
#pragma unroll
                for (int o = TPK / 2; o > 0; o >>= 1) d += __shfl_xor_sync(0xffffffffu, d, o);
                sc[u][g] = ok[u] ? d : -INFINITY;
            }
        }
        float vf[KB][8];
#pragma unroll
        for (int u = 0; u < KB; ++u) L::unpack(vv[u], vf[u]);
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
            __syncthreads();  // s_acc / s_m / s_l are reused for every head
#pragma unroll
            for (int e = 0; e < 8; ++e) s_acc[gk * D + c * 8 + e] = acc[g][e];
            if (c == 0) {
                s_m[gk] = m[g];
                s_l[gk] = l[g];
            }
            __syncthreads();
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

// ROWS: `T` rows are sequences of one query row each (stage 1's ROWS), row `t * N + h` belongs to sequence `t = row / N`.
template <int D, bool ROWS = false>
__global__ void __launch_bounds__(D)
combine_kernel(const float* __restrict__ pacc, const float2* __restrict__ pml, ik_bf16* __restrict__ out, int nsplit,
               int T, int NKV, int64_t pos0_host, const int32_t* pos_dev, int N = 0) {
    const int64_t row = blockIdx.x;  // t * N + h
    const int64_t pos0 = ROWS ? (int64_t)pos_dev[row / N] : (pos_dev ? (int64_t)*pos_dev : pos0_host);
    const int64_t tq = ROWS ? 1 : T;
    const int nsplit_eff = splitkv_count(pos0 + tq, NKV, tq);  // same count as the stage-1 blocks used
    const float2* ml = pml + row * nsplit;
    const float* acc = pacc + row * nsplit * D;
    float M = -INFINITY;
    for (int s = 0; s < nsplit_eff; ++s) M = fmaxf(M, ml[s].x);
    float A = 0.f, L = 0.f;
    for (int s = 0; s < nsplit_eff; ++s) {
        const float2 v = ml[s];
        if (v.x == -INFINITY) continue;  // empty split: acc is not meaningful
        const float w = exp2f(v.x - M);
        L += v.y * w;
        A += acc[(int64_t)s * D + threadIdx.x] * w;
    }
    out[row * D + threadIdx.x] = ik_f2b(A / L);
}

template <int GP, int TPK, int KB, typename KT, bool ROWS = false>
void launch_split(const ik_bf16* q, const KT* kc, const KT* vc, float* pacc, float2* pml, int64_t T,
                  int64_t N, int64_t NKV, int64_t cap, int64_t pos0, float qscale, int nsplit,
                  int64_t window, const int32_t* pos_dev, cudaStream_t s, const uint8_t* tab = nullptr,
                  int64_t tstride = 0, int64_t tindex = 0) {
    dim3 grid(nsplit, (unsigned)NKV, (unsigned)T);
    splitkv_kernel<GP, TPK, KB, KT, ROWS><<<grid, kBlock, 0, s>>>(q, kc, vc, pacc, pml, (int)N, (int)(N / NKV), cap, pos0,
                                                              qscale, nsplit, window, pos_dev, tab, tstride, tindex);
}

template <int TPK, typename KT, bool ROWS = false>
int dispatch_group(int G, const ik_bf16* q, const KT* kc, const KT* vc, float* pacc, float2* pml,
                   int64_t T, int64_t N, int64_t NKV, int64_t cap, int64_t pos0, float qscale, int nsplit,
                   int64_t window, const int32_t* pos_dev, cudaStream_t s, const uint8_t* tab = nullptr,
                   int64_t tstride = 0, int64_t tindex = 0) {
#define IK_SPLIT_CASE(GPV, KBV)                                                                                \
    launch_split<GPV, TPK, KBV, KT, ROWS>(q, kc, vc, pacc, pml, T, N, NKV, cap, pos0, qscale, nsplit, window, pos_dev, s, \
                                          tab, tstride, tindex);                                                   \
    return IK_OK;
    if (G <= 1) { IK_SPLIT_CASE(1, 4) }
    if (G <= 2) { IK_SPLIT_CASE(2, 4) }
    if (G <= 4) { IK_SPLIT_CASE(4, 4) }
    if (G <= 8) { IK_SPLIT_CASE(8, 2) }
#undef IK_SPLIT_CASE
    return IK_ERR_SHAPE;
}

int splitkv_launch(const ik_tensor* q, const ik_tensor* kc, const ik_tensor* vc, int64_t pos0, float scale,
                   ik_tensor* out, const ik_scratch* scratch, const ik_tensor* pos, void* stream, bool decode,
                   bool f8kv = false, int64_t window = 0) {
    IK_TRY(ik_check_pos(pos));
    const int32_t* pos_dev = pos ? (const int32_t*)pos->data : nullptr;
    IK_TRY(ik_check(q, IK_DTYPE_BF16, 3));
    const ik_dtype kvt = f8kv ? IK_DTYPE_FP8_E4M3 : IK_DTYPE_BF16;
    IK_TRY(ik_check(kc, kvt, 3));
    IK_TRY(ik_check(vc, kvt, 3));
    IK_TRY(ik_check(out, IK_DTYPE_BF16, 3));
    const int64_t T = q->shape[0], N = q->shape[1], D = q->shape[2];
    const int64_t NKV = kc->shape[0], cap = kc->shape[1];
    if (decode && T != 1) return IK_ERR_SHAPE;
    if (vc->shape[0] != NKV || vc->shape[1] != cap || vc->shape[2] != D || kc->shape[2] != D) return IK_ERR_SHAPE;
    if (out->shape[0] != T || out->shape[1] != N || out->shape[2] != D) return IK_ERR_SHAPE;
    if (NKV <= 0 || N % NKV != 0 || pos0 < 0 || window < 0 || pos0 + T > cap || T > kMaxRows) return IK_ERR_SHAPE;
    if (N / NKV > kMaxGroup) return IK_ERR_SHAPE;
    if (D != 32 && D != 64 && D != 128 && D != 256) return IK_ERR_SHAPE;
    if (T == 0 || N == 0) return IK_OK;
    if (!ik_aligned16(kc->data) || !ik_aligned16(vc->data)) return IK_ERR_STRIDE;

    // Decode launches the grid at its length-independent maximum (the scratch bound): one launch shape for every
    // position, which is what lets a captured CUDA graph be replayed at any length. Prefill / verify (host position)
    // sizes the grid from the key count; both partition identically (see splitkv_kernel).
    const int nsplit = decode ? splitkv_count(INT64_MAX / 2, NKV, T) : splitkv_count(pos0 + T, NKV, T);
    const int64_t need = splitkv_scratch_bytes(T, N, D, nsplit);
    if (!scratch || !scratch->data || scratch->bytes < need) return IK_ERR_INVALID_ARG;
    float* pacc = (float*)scratch->data;
    float2* pml = (float2*)((char*)scratch->data + T * N * nsplit * D * 4);

    cudaStream_t s = IK_STREAM(stream);
    const ik_bf16* qp = (const ik_bf16*)q->data;
    const float qscale = scale * kLog2e;
    int rc = IK_ERR_SHAPE;
    if (f8kv) {
        const uint8_t* kp = (const uint8_t*)kc->data;
        const uint8_t* vp = (const uint8_t*)vc->data;
        switch (D) {
            case 32: rc = dispatch_group<4>((int)(N / NKV), qp, kp, vp, pacc, pml, T, N, NKV, cap, pos0, qscale, nsplit, window, pos_dev, s); break;
            case 64: rc = dispatch_group<8>((int)(N / NKV), qp, kp, vp, pacc, pml, T, N, NKV, cap, pos0, qscale, nsplit, window, pos_dev, s); break;
            case 128: rc = dispatch_group<16>((int)(N / NKV), qp, kp, vp, pacc, pml, T, N, NKV, cap, pos0, qscale, nsplit, window, pos_dev, s); break;
            case 256: rc = dispatch_group<32>((int)(N / NKV), qp, kp, vp, pacc, pml, T, N, NKV, cap, pos0, qscale, nsplit, window, pos_dev, s); break;
        }
    }
    const ik_bf16* kp = (const ik_bf16*)kc->data;
    const ik_bf16* vp = (const ik_bf16*)vc->data;
    if (!f8kv) switch (D) {
        case 32: rc = dispatch_group<4>((int)(N / NKV), qp, kp, vp, pacc, pml, T, N, NKV, cap, pos0, qscale, nsplit, window, pos_dev, s); break;
        case 64: rc = dispatch_group<8>((int)(N / NKV), qp, kp, vp, pacc, pml, T, N, NKV, cap, pos0, qscale, nsplit, window, pos_dev, s); break;
        case 128: rc = dispatch_group<16>((int)(N / NKV), qp, kp, vp, pacc, pml, T, N, NKV, cap, pos0, qscale, nsplit, window, pos_dev, s); break;
        case 256: rc = dispatch_group<32>((int)(N / NKV), qp, kp, vp, pacc, pml, T, N, NKV, cap, pos0, qscale, nsplit, window, pos_dev, s); break;
    }
    if (rc != IK_OK) return rc;
    IK_TRY(ik_launch_status());
    ik_bf16* op = (ik_bf16*)out->data;
    const unsigned rows = (unsigned)(T * N);
    switch (D) {
        case 32: combine_kernel<32><<<rows, 32, 0, s>>>(pacc, pml, op, nsplit, (int)T, (int)NKV, pos0, pos_dev); break;
        case 64: combine_kernel<64><<<rows, 64, 0, s>>>(pacc, pml, op, nsplit, (int)T, (int)NKV, pos0, pos_dev); break;
        case 128: combine_kernel<128><<<rows, 128, 0, s>>>(pacc, pml, op, nsplit, (int)T, (int)NKV, pos0, pos_dev); break;
        case 256: combine_kernel<256><<<rows, 256, 0, s>>>(pacc, pml, op, nsplit, (int)T, (int)NKV, pos0, pos_dev); break;
    }
    return ik_launch_status();
}

// Row-table decode (`OpCall::rows`): `rows` sequences of one query row each (q [rows, N, D]); `kc0` / `vc0` are row 0's caches
// ([NKV, cap, D], shape carriers); sequence r's caches are the buffers at entries [r][index] and [r][index + 1] of `tab`,
// its position is pos[r]. Same grid policy as the one-row decode launch (the maximum split count), `rows` rows of scratch.
int splitkv_launch_rows(const ik_tensor* q, const ik_tensor* kc0, const ik_tensor* vc0, float scale, ik_tensor* out,
                        const ik_scratch* scratch, const ik_tensor* tab, int64_t stride, int64_t index,
                        const ik_tensor* pos, void* stream) {
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
    if (!tab || !tab->data || tab->dtype != IK_DTYPE_U8 || tab->ndim != 3 || tab->shape[0] != R || tab->shape[1] != stride ||
        tab->shape[2] != 8 || index < 0 || index + 1 >= stride || (reinterpret_cast<uintptr_t>(tab->data) & 7u))
        return IK_ERR_INVALID_ARG;
    if (R == 0 || N == 0) return IK_OK;
    const int32_t* pos_dev = (const int32_t*)pos->data;
    const int nsplit = splitkv_count(INT64_MAX / 2, NKV, 1);  // the one-row decode launch's grid
    const int64_t need = splitkv_scratch_bytes(R, N, D, nsplit);
    if (!scratch || !scratch->data || scratch->bytes < need) return IK_ERR_INVALID_ARG;
    float* pacc = (float*)scratch->data;
    float2* pml = (float2*)((char*)scratch->data + R * N * nsplit * D * 4);
    cudaStream_t s = IK_STREAM(stream);
    const ik_bf16* qp = (const ik_bf16*)q->data;
    const ik_bf16* kp = (const ik_bf16*)kc0->data;
    const ik_bf16* vp = (const ik_bf16*)vc0->data;
    const uint8_t* tp = (const uint8_t*)tab->data;
    const float qscale = scale * kLog2e;
    int rc = IK_ERR_SHAPE;
#define IK_ROWS_DISPATCH(DV, TPKV)                                                                                          \
    case DV: rc = dispatch_group<TPKV, ik_bf16, true>((int)(N / NKV), qp, kp, vp, pacc, pml, R, N, NKV, cap, 0, qscale, nsplit, \
                                                      0, pos_dev, s, tp, stride, index); break;
    switch (D) {
        IK_ROWS_DISPATCH(32, 4)
        IK_ROWS_DISPATCH(64, 8)
        IK_ROWS_DISPATCH(128, 16)
        IK_ROWS_DISPATCH(256, 32)
    }
#undef IK_ROWS_DISPATCH
    if (rc != IK_OK) return rc;
    IK_TRY(ik_launch_status());
    ik_bf16* op = (ik_bf16*)out->data;
    const unsigned rows = (unsigned)(R * N);
    switch (D) {
        case 32: combine_kernel<32, true><<<rows, 32, 0, s>>>(pacc, pml, op, nsplit, (int)R, (int)NKV, 0, pos_dev, (int)N); break;
        case 64: combine_kernel<64, true><<<rows, 64, 0, s>>>(pacc, pml, op, nsplit, (int)R, (int)NKV, 0, pos_dev, (int)N); break;
        case 128: combine_kernel<128, true><<<rows, 128, 0, s>>>(pacc, pml, op, nsplit, (int)R, (int)NKV, 0, pos_dev, (int)N); break;
        case 256: combine_kernel<256, true><<<rows, 256, 0, s>>>(pacc, pml, op, nsplit, (int)R, (int)NKV, 0, pos_dev, (int)N); break;
    }
    return ik_launch_status();
}

}  // namespace

extern "C" {

int ik_attn_decode_bf16_splitkv(const ik_tensor* q, const ik_tensor* kc, const ik_tensor* vc, int64_t pos0,
                                float scale, ik_tensor* out, const ik_scratch* scratch, const ik_tensor* pos,
                                void* stream) {
    return splitkv_launch(q, kc, vc, pos0, scale, out, scratch, pos, stream, true);
}

int ik_attn_prefill_bf16_splitkv(const ik_tensor* q, const ik_tensor* kc, const ik_tensor* vc, int64_t pos0,
                                 float scale, ik_tensor* out, const ik_scratch* scratch, void* stream) {
    return splitkv_launch(q, kc, vc, pos0, scale, out, scratch, nullptr, stream, false);
}

int ik_attn_decode_bf16_splitkv_rows(const ik_tensor* q, const ik_tensor* kc0, const ik_tensor* vc0, float scale,
                                     ik_tensor* out, const ik_scratch* scratch, const ik_tensor* tab, int64_t stride,
                                     int64_t index, const ik_tensor* pos, void* stream) {
    return splitkv_launch_rows(q, kc0, vc0, scale, out, scratch, tab, stride, index, pos, stream);
}

// g4 (Gemma 4) sliding window: the bf16 kernels above with `window` (the query at absolute position p attends keys
// max(0, p + 1 - window)..=p; 0 = all). Registered `.ext()`; the plain launchers keep window 0 and their exact bits.
int ik_attn_decode_bf16_splitkv_ext(const ik_tensor* q, const ik_tensor* kc, const ik_tensor* vc, int64_t pos0,
                                    float scale, int64_t window, ik_tensor* out, const ik_scratch* scratch,
                                    const ik_tensor* pos, void* stream) {
    return splitkv_launch(q, kc, vc, pos0, scale, out, scratch, pos, stream, true, false, window);
}

int ik_attn_prefill_bf16_splitkv_ext(const ik_tensor* q, const ik_tensor* kc, const ik_tensor* vc, int64_t pos0,
                                     float scale, int64_t window, ik_tensor* out, const ik_scratch* scratch,
                                     void* stream) {
    return splitkv_launch(q, kc, vc, pos0, scale, out, scratch, nullptr, stream, false, false, window);
}

// fp8_e4m3 KV cache (bf16 q / out), same scratch and grid policy as the two launchers above
int ik_attn_decode_f8kv_splitkv(const ik_tensor* q, const ik_tensor* kc, const ik_tensor* vc, int64_t pos0, float scale,
                                ik_tensor* out, const ik_scratch* scratch, const ik_tensor* pos, void* stream) {
    return splitkv_launch(q, kc, vc, pos0, scale, out, scratch, pos, stream, true, true);
}

int ik_attn_prefill_f8kv_splitkv(const ik_tensor* q, const ik_tensor* kc, const ik_tensor* vc, int64_t pos0, float scale,
                                 ik_tensor* out, const ik_scratch* scratch, void* stream) {
    return splitkv_launch(q, kc, vc, pos0, scale, out, scratch, nullptr, stream, false, true);
}

}  // extern "C"
