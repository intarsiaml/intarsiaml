// Split-KV sliding-window decode attention (T == 1) over a contiguous bf16 KV cache [Nkv, cap, D], for the DFlash2 draft's block rows
// (lane dw, `OpAttrs::win_split`, `INTARSIA_DFLASH_WIN_SPLIT=1`). The call is `ik_attn_decode_bf16_window`'s: the query at absolute
// position pos0 attends keys max(0, pos0 + 1 - window) ..= pos0 (window == 0: all keys), but the key range is cut into chunks of >= 64
// keys that run in parallel blocks, where `attention.hip`'s kernel scans the whole range in one block per head (a serial 2048-key loop,
// ~80 us a call, 40 calls a draft block at 4k).
//
// Stage 1 (win_splitkv_kernel): grid (split, kv head), 256 threads; the body is `attention_splitkv.hip`'s splitkv_body for one query row (a block
// serves all G = N/Nkv heads of its kv head; NG = 256/TPK key lanes of TPK = D/8 threads, a thread owns 8 head dims, online softmax per lane,
// lanes merged through shared memory, one (m, l, acc[D]) partial per (head, split) to scratch) with the key range starting at the window's first
// key. Stage 2 (win_combine_kernel): per head merge the partials and write the bf16 row.
//
// Scratch (ik_scratch): float acc[N][nsplit][D] followed by float2 ml[N][nsplit], the layout and the bound of the split-KV decode (kernels.rs
// splitkv_scratch with m = 1). The split count follows from the key count of the window (visible - first key), the kv heads and one row, never
// from the data. With a device position the grid is launched at its maximum for the window and the blocks past the real count leave at once.
//
// Not bitwise `ik_attn_decode_bf16_window` (another summation order, exp2f); admitted against the reference like any kernel, tests/win_splitkv.rs.
// A separate translation unit on purpose: attention.hip and attention_splitkv.hip, which the target model runs, are not touched.
#include "ik_hip.h"

namespace {

constexpr int kBlock = 256;
constexpr int kMaxGroup = 8;   // q heads per kv head handled by one block
constexpr int kMinKeys = 64;   // keys per split, at least
constexpr int kMaxSplit = 64;
constexpr int kTargetBlocks = 256;
constexpr int kCombBatch = 16;
constexpr float kLog2e = 1.4426950408889634f;

// The one-query-row policy of attention_splitkv.hip (splitkv_count / splitkv_chunk with rows == 1), over the window's key count.
__host__ __device__ inline int win_count(int64_t kv_len, int64_t nkv) {
    int64_t by_len = (kv_len + kMinKeys - 1) / kMinKeys;
    int64_t by_fill = (kTargetBlocks + nkv - 1) / nkv;
    int64_t n = by_len < by_fill ? by_len : by_fill;
    if (n > kMaxSplit) n = kMaxSplit;
    if (n < 1) n = 1;
    return (int)n;
}

__host__ __device__ inline int64_t win_chunk(int64_t kv_len, int nsplit) {
    int64_t c = (kv_len + nsplit - 1) / nsplit;
    return (c + kMinKeys - 1) / kMinKeys * kMinKeys;
}

// First attended key of the query at pos0 (window == 0: all keys).
__host__ __device__ inline int64_t win_first(int64_t visible, int64_t window) {
    return (window > 0 && visible > window) ? visible - window : 0;
}

template <int GP, int TPK, int KB>
__global__ void __launch_bounds__(kBlock)
win_splitkv_kernel(const ik_bf16* __restrict__ q, const ik_bf16* __restrict__ kc, const ik_bf16* __restrict__ vc,
                   float* __restrict__ pacc, float2* __restrict__ pml, int N, int G, int64_t cap, int64_t pos0_host,
                   int64_t window, float qscale, int nsplit, const int32_t* pos_dev) {
    const int64_t pos0 = pos_dev ? (int64_t)*pos_dev : pos0_host;
    const int64_t visible = pos0 + 1;
    const int64_t first = win_first(visible, window);
    const int64_t kv_len = visible - first;
    const int nsplit_eff = win_count(kv_len, gridDim.y);
    const int64_t chunk = win_chunk(kv_len, nsplit_eff);
    if ((int)blockIdx.x >= nsplit_eff) return;
    constexpr int D = TPK * 8;
    constexpr int NG = kBlock / TPK;
    __shared__ __align__(16) float s_acc[NG * D];
    __shared__ float s_m[NG], s_l[NG];

    const int sp = blockIdx.x, kvh = blockIdx.y;
    const int c = threadIdx.x % TPK, gk = threadIdx.x / TPK;
    const int64_t start = first + (int64_t)sp * chunk;
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
            if (g < G) qr[g][e] = ik_b2f(q[((int64_t)kvh * G + g) * D + c * 8 + e]) * qscale;
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
            __syncthreads();  // s_acc / s_m / s_l are reused for every head
#pragma unroll
            for (int e = 0; e < 8; ++e) s_acc[gk * D + c * 8 + e] = acc[g][e];
            if (c == 0) {
                s_m[gk] = m[g];
                s_l[gk] = l[g];
            }
            __syncthreads();
            const int64_t slot = ((int64_t)kvh * G + g) * nsplit + sp;
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

// Stage 2: block `row` = head h of the one query row. Same split count as stage 1 (it derives it from the same window).
template <int D>
__global__ void __launch_bounds__(D)
win_combine_kernel(const float* __restrict__ pacc, const float2* __restrict__ pml, ik_bf16* __restrict__ out, int nsplit, int NKV,
                   int64_t pos0_host, int64_t window, const int32_t* pos_dev) {
    const int64_t row = blockIdx.x;  // head
    const int64_t pos0 = pos_dev ? (int64_t)*pos_dev : pos0_host;
    const int64_t visible = pos0 + 1;
    const int nsplit_eff = win_count(visible - win_first(visible, window), NKV);
    const float2* ml = pml + row * nsplit;
    const float* acc = pacc + row * nsplit * D;
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
void launch_stage1(const ik_bf16* q, const ik_bf16* kc, const ik_bf16* vc, float* pacc, float2* pml, int64_t N, int64_t NKV,
                   int64_t cap, int64_t pos0, int64_t window, float qscale, int nsplit, const int32_t* pos_dev, hipStream_t s) {
    dim3 grid(nsplit, (unsigned)NKV, 1);
    win_splitkv_kernel<GP, TPK, KB><<<grid, kBlock, 0, s>>>(q, kc, vc, pacc, pml, (int)N, (int)(N / NKV), cap, pos0, window, qscale,
                                                           nsplit, pos_dev);
}

template <int TPK>
int dispatch_group(int G, const ik_bf16* q, const ik_bf16* kc, const ik_bf16* vc, float* pacc, float2* pml, int64_t N, int64_t NKV,
                   int64_t cap, int64_t pos0, int64_t window, float qscale, int nsplit, const int32_t* pos_dev, hipStream_t s) {
#define IK_WIN_CASE(GPV, KBV)                                                                                          \
    launch_stage1<GPV, TPK, KBV>(q, kc, vc, pacc, pml, N, NKV, cap, pos0, window, qscale, nsplit, pos_dev, s);         \
    return IK_OK;
    if (G <= 1) { IK_WIN_CASE(1, 4) }
    if (G <= 2) { IK_WIN_CASE(2, 4) }
    if (G <= 4) { IK_WIN_CASE(4, 4) }
    if (G <= 8) { IK_WIN_CASE(8, 2) }
#undef IK_WIN_CASE
    return IK_ERR_SHAPE;
}

int win_splitkv_launch(const ik_tensor* q, const ik_tensor* kc, const ik_tensor* vc, int64_t pos0, int64_t window, float scale,
                       ik_tensor* out, const ik_scratch* scratch, const ik_tensor* pos, void* stream) {
    if (window < 0) return IK_ERR_INVALID_ARG;
    IK_TRY(ik_check_pos(pos));
    const int32_t* pos_dev = pos ? (const int32_t*)pos->data : nullptr;
    IK_TRY(ik_check(q, IK_DTYPE_BF16, 3));
    IK_TRY(ik_check(kc, IK_DTYPE_BF16, 3));
    IK_TRY(ik_check(vc, IK_DTYPE_BF16, 3));
    IK_TRY(ik_check(out, IK_DTYPE_BF16, 3));
    const int64_t T = q->shape[0], N = q->shape[1], D = q->shape[2];
    const int64_t NKV = kc->shape[0], cap = kc->shape[1];
    if (T != 1) return IK_ERR_SHAPE;  // decode: one query row
    if (vc->shape[0] != NKV || vc->shape[1] != cap || vc->shape[2] != D || kc->shape[2] != D) return IK_ERR_SHAPE;
    if (out->shape[0] != T || out->shape[1] != N || out->shape[2] != D) return IK_ERR_SHAPE;
    if (NKV <= 0 || N % NKV != 0 || pos0 < 0 || pos0 + T > cap) return IK_ERR_SHAPE;
    if (N / NKV > kMaxGroup) return IK_ERR_SHAPE;
    if (D != 32 && D != 64 && D != 128 && D != 256) return IK_ERR_SHAPE;
    if (N == 0) return IK_OK;
    if (!ik_aligned16(kc->data) || !ik_aligned16(vc->data)) return IK_ERR_STRIDE;

    // Host position: the grid is sized from the window's real key count. Device position: the key count is unknown at launch, so the grid
    // is launched at its maximum for this window (the count never exceeds win_count(window), or the length-free maximum for window == 0),
    // and the blocks past the real count leave at once; both partition identically.
    const int64_t kv_len_max = window > 0 ? window : INT64_MAX / 2;
    int64_t kv_len_host = pos0 + 1;
    kv_len_host -= win_first(kv_len_host, window);
    const int nsplit = pos_dev ? win_count(kv_len_max, NKV) : win_count(kv_len_host, NKV);
    const int64_t need = N * (int64_t)nsplit * (D + 2) * 4;
    if (!scratch || !scratch->data || scratch->bytes < need) return IK_ERR_INVALID_ARG;
    float* pacc = (float*)scratch->data;
    float2* pml = (float2*)((char*)scratch->data + N * (int64_t)nsplit * D * 4);

    hipStream_t s = IK_STREAM(stream);
    const ik_bf16* qp = (const ik_bf16*)q->data;
    const ik_bf16* kp = (const ik_bf16*)kc->data;
    const ik_bf16* vp = (const ik_bf16*)vc->data;
    const float qscale = scale * kLog2e;
    int rc = IK_ERR_SHAPE;
    switch (D) {
        case 32: rc = dispatch_group<4>((int)(N / NKV), qp, kp, vp, pacc, pml, N, NKV, cap, pos0, window, qscale, nsplit, pos_dev, s); break;
        case 64: rc = dispatch_group<8>((int)(N / NKV), qp, kp, vp, pacc, pml, N, NKV, cap, pos0, window, qscale, nsplit, pos_dev, s); break;
        case 128: rc = dispatch_group<16>((int)(N / NKV), qp, kp, vp, pacc, pml, N, NKV, cap, pos0, window, qscale, nsplit, pos_dev, s); break;
        case 256: rc = dispatch_group<32>((int)(N / NKV), qp, kp, vp, pacc, pml, N, NKV, cap, pos0, window, qscale, nsplit, pos_dev, s); break;
    }
    if (rc != IK_OK) return rc;
    IK_TRY(ik_launch_status());
    ik_bf16* op = (ik_bf16*)out->data;
    const unsigned rows = (unsigned)N;
    switch (D) {
        case 32: win_combine_kernel<32><<<rows, 32, 0, s>>>(pacc, pml, op, nsplit, (int)NKV, pos0, window, pos_dev); break;
        case 64: win_combine_kernel<64><<<rows, 64, 0, s>>>(pacc, pml, op, nsplit, (int)NKV, pos0, window, pos_dev); break;
        case 128: win_combine_kernel<128><<<rows, 128, 0, s>>>(pacc, pml, op, nsplit, (int)NKV, pos0, window, pos_dev); break;
        case 256: win_combine_kernel<256><<<rows, 256, 0, s>>>(pacc, pml, op, nsplit, (int)NKV, pos0, window, pos_dev); break;
    }
    return ik_launch_status();
}

}  // namespace

extern "C" {

// Sliding-window decode (`OpAttrs::window`, 0 = all keys), T == 1, split over the key range; see the file header. Registered `.ext()`, accepted
// only for calls with `OpAttrs::win_split`.
int ik_attn_decode_bf16_window_splitkv(const ik_tensor* q, const ik_tensor* kc, const ik_tensor* vc, int64_t pos0, int64_t window,
                                       float scale, ik_tensor* out, const ik_scratch* scratch, const ik_tensor* pos, void* stream) {
    return win_splitkv_launch(q, kc, vc, pos0, window, scale, out, scratch, pos, stream);
}

}  // extern "C"
