// DeepSeek-V4.1 sparse MQA attention (sparse_attn, kernel.py:311-404; window rows of get_window_topk_idxs, model.py:410):
// query t attends the 128-token window rows of `win` plus the compressed rows its index list picks (-1 skipped); one latent
// vector is key and value; per-head attention sink joins the softmax denominator only; the running max starts at -1e30, so
// a query with no keys gives zeros (kernel.py:355). The output's rope part is counter-rotated by a separate rope_tail.
// Contract: docs/design/dsv41-ops.md. Written from the official inference code (MIT, DeepSeek).
//
// One block per (key split, query, head group of up to 32 heads): the block walks its tiles of the query's key list (window
// entries, then index entries), gathers a tile of keys into shared memory once for all its heads, and keeps an online softmax
// per head. Decode / verify (T <= 16) splits the list over up to kMaxSplit blocks and merges the partials in a second
// kernel (scratch); a prefill chunk runs one block per query and head group and writes the output directly.
#include "dsv41_common.cuh"

using namespace dsv41;

namespace {

constexpr int kThreads = 256;
constexpr int kWarps = kThreads / 32;
constexpr int kHeads = 32;  // heads per block
constexpr int kMaxSplit = 32;
constexpr int kMaxSplitRows = 16;
constexpr int kTileBytes = 32768;  // key tile in shared memory (native dtype)
constexpr int kMaxD = 512;
constexpr float kInitMax = -1e30f;

template <typename T> constexpr int tile_keys() { return kTileBytes / (kMaxD * (int)sizeof(T)) < 32 ? kTileBytes / (kMaxD * (int)sizeof(T)) : 32; }

template <typename T>
__global__ void __launch_bounds__(kThreads) attn_sparse_kernel(const T* __restrict__ q, const T* __restrict__ win, const T* __restrict__ cmp,
                                                               const int* __restrict__ idx, const float* __restrict__ sink, T* __restrict__ out,
                                                               float* __restrict__ part, int64_t H, int64_t D, int64_t W, int64_t Nc,
                                                               int64_t Kc, int64_t pos0, const int* __restrict__ dpos, float scale, int nsplit) {
    constexpr int KS = tile_keys<T>();
    extern __shared__ __align__(16) unsigned char smem_raw[];
    T* kt = reinterpret_cast<T*>(smem_raw);  // [KS][D]
    __shared__ float P[kHeads][32];
    __shared__ float alpha_s[kHeads];
    __shared__ int valid[32];
    __shared__ int64_t srcrow[32];  // >= 0: win row; < 0: -(cmp row) - 1

    const int split = blockIdx.x;
    const int64_t t = blockIdx.y;
    const int64_t h0 = (int64_t)blockIdx.z * kHeads;
    const int nh = (int)min((int64_t)kHeads, H - h0);
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int64_t p0 = dpos ? (int64_t)*dpos : pos0;
    const int64_t nkeys = W + Kc;
    const int ntiles = (int)cdiv(nkeys, KS);
    const int dper = (int)(D / 32);  // q elements per lane (D in 32..512)

    // q of this warp's heads (h0 + warp + 8 i), lane holds d = lane + 32 j
    float qr[kHeads / kWarps][kMaxD / 32];
#pragma unroll
    for (int i = 0; i < kHeads / kWarps; ++i) {
        const int hh = warp + kWarps * i;
#pragma unroll
        for (int j = 0; j < kMaxD / 32; ++j)
            qr[i][j] = (hh < nh && j < dper) ? ld(q, (t * H + h0 + hh) * D + lane + 32 * j) : 0.f;
    }
    // PV layout: column pair cp of CT = D/2 pairs, head slice hs of HS = 256 / CT
    const int CT = (int)(D / 2), HS = kThreads / CT;
    const int cp = threadIdx.x % CT, hs = threadIdx.x / CT;
    const int hpt = (nh + HS - 1) / HS;
    float acc[kHeads][2];
#pragma unroll
    for (int i = 0; i < kHeads; ++i) acc[i][0] = acc[i][1] = 0.f;
    // per-head running max / sum, owned by the warp that scores the head (all lanes hold the same value)
    float m_[kHeads / kWarps], l_[kHeads / kWarps];
#pragma unroll
    for (int i = 0; i < kHeads / kWarps; ++i) {
        m_[i] = kInitMax;
        l_[i] = 0.f;
    }

    for (int tile = split; tile < ntiles; tile += nsplit) {
        const int64_t e0 = (int64_t)tile * KS;
        __syncthreads();  // previous tile's keys and probabilities are consumed
        if (threadIdx.x < KS) {
            const int64_t e = e0 + threadIdx.x;
            int ok = 0;
            int64_t sr = 0;
            if (e < W) {
                const int64_t row = t + 1 + e;
                ok = (p0 + row >= W);
                sr = row;
            } else if (e < nkeys) {
                const int c = idx[t * Kc + (e - W)];
                ok = (c >= 0 && c < Nc);
                sr = -(int64_t)c - 1;
            }
            valid[threadIdx.x] = ok;
            srcrow[threadIdx.x] = sr;
        }
        __syncthreads();
        // gather: 16-byte vectors, each warp a key at a time
        {
            const int vec = 16 / (int)sizeof(T);
            const int64_t nv = D / vec;
            for (int j = warp; j < KS; j += kWarps) {
                if (!valid[j]) continue;
                const int64_t sr = srcrow[j];
                const T* src = sr >= 0 ? win + sr * D : cmp + (-sr - 1) * D;
                const uint4* s4 = reinterpret_cast<const uint4*>(src);
                uint4* d4 = reinterpret_cast<uint4*>(kt + (int64_t)j * D);
                for (int64_t v = lane; v < nv; v += 32) d4[v] = __ldg(s4 + v);
            }
        }
        __syncthreads();
        // scores: warp scores its heads against every key of the tile; lane j keeps key j's score
#pragma unroll
        for (int i = 0; i < kHeads / kWarps; ++i) {
            const int hh = warp + kWarps * i;
            if (hh >= nh) continue;  // warp-uniform
            float mys = -INFINITY;
            for (int j = 0; j < KS; ++j) {
                if (!valid[j]) continue;  // block-uniform
                const T* kr = kt + (int64_t)j * D;
                float s = 0.f;
#pragma unroll
                for (int d = 0; d < kMaxD / 32; ++d)
                    if (d < dper) s = fmaf(qr[i][d], ld(kr, lane + 32 * d), s);
                s = ik_warp_sum(s) * scale;
                if (lane == j) mys = s;
            }
            const float tmax = ik_warp_max(lane < KS ? mys : -INFINITY);
            const float mn = fmaxf(m_[i], tmax);
            const float a = expf(m_[i] - mn);
            const float p = (lane < KS) ? expf(mys - mn) : 0.f;
            l_[i] = l_[i] * a + ik_warp_sum(p);
            m_[i] = mn;
            if (lane < KS) P[hh][lane] = p;
            if (lane == 0) alpha_s[hh] = a;
        }
        __syncthreads();
        // acc = acc * alpha + sum_j p_j k_j for this thread's heads and column pair
#pragma unroll
        for (int i = 0; i < kHeads; ++i) {
            const int hh = hs + HS * i;
            if (i >= hpt || hh >= nh) continue;
            const float a = alpha_s[hh];
            float a0 = acc[i][0] * a, a1 = acc[i][1] * a;
            for (int j = 0; j < KS; ++j) {
                if (!valid[j]) continue;
                const float p = P[hh][j];
                const T* kr = kt + (int64_t)j * D + 2 * cp;
                a0 = fmaf(p, ld(kr, 0), a0);
                a1 = fmaf(p, ld(kr, 1), a1);
            }
            acc[i][0] = a0;
            acc[i][1] = a1;
        }
    }
    // publish (m, l) of each head through shared memory
    __shared__ float ml[kHeads][2];
    __syncthreads();
#pragma unroll
    for (int i = 0; i < kHeads / kWarps; ++i) {
        const int hh = warp + kWarps * i;
        if (hh < nh && lane == 0) {
            ml[hh][0] = m_[i];
            ml[hh][1] = l_[i];
        }
    }
    __syncthreads();
#pragma unroll
    for (int i = 0; i < kHeads; ++i) {
        const int hh = hs + HS * i;
        if (i >= hpt || hh >= nh) continue;
        const int64_t h = h0 + hh;
        const float m = ml[hh][0], l = ml[hh][1];
        if (nsplit == 1) {
            const float den = l + expf(sink[h] - m);
            const int64_t o = (t * H + h) * D + 2 * cp;
            st(out, o, acc[i][0] / den);
            st(out, o + 1, acc[i][1] / den);
        } else {
            const int64_t slot = (t * H + h) * nsplit + split;
            float* pa = part + slot * D;
            pa[2 * cp] = acc[i][0];
            pa[2 * cp + 1] = acc[i][1];
            if (cp == 0) {
                float* pm = part + (int64_t)gridDim.y * H * nsplit * D + slot * 2;
                pm[0] = m;
                pm[1] = l;
            }
        }
    }
}

// merge the split partials of one (query, head): o = sum_s acc_s e^(m_s - M) / (sum_s l_s e^(m_s - M) + e^(sink - M))
template <typename T>
__global__ void attn_sparse_merge(const float* __restrict__ part, const float* __restrict__ sink, T* __restrict__ out, int64_t Tn, int64_t H,
                                  int64_t D, int nsplit) {
    const int64_t t = blockIdx.x, h = blockIdx.y;
    const int64_t slot0 = (t * H + h) * nsplit;
    const float* pm = part + Tn * H * nsplit * D + slot0 * 2;
    float M = kInitMax;
    for (int s = 0; s < nsplit; ++s) M = fmaxf(M, pm[2 * s]);
    float L = 0.f;
    for (int s = 0; s < nsplit; ++s) L += pm[2 * s + 1] * expf(pm[2 * s] - M);
    const float den = L + expf(sink[h] - M);
    for (int64_t d = threadIdx.x; d < D; d += blockDim.x) {
        float v = 0.f;
        for (int s = 0; s < nsplit; ++s) v = fmaf(part[(slot0 + s) * D + d], expf(pm[2 * s] - M), v);
        st(out, (t * H + h) * D + d, v / den);
    }
}

template <typename T>
int launch(const ik_tensor* q, const ik_tensor* win, const ik_tensor* cmp, const ik_tensor* idx, const ik_tensor* sink, int64_t W,
           int64_t pos0, float scale, ik_tensor* out, const ik_scratch* scratch, const ik_tensor* pos, cudaStream_t s) {
    constexpr int KS = tile_keys<T>();
    const int64_t Tn = q->shape[0], H = q->shape[1], D = q->shape[2];
    const int64_t Nc = cmp->shape[0], Kc = idx->shape[1];
    const int64_t ntiles = cdiv(W + Kc, KS);
    int nsplit = 1;
    if (Tn <= kMaxSplitRows) nsplit = (int)(ntiles < kMaxSplit ? ntiles : kMaxSplit);
    if (nsplit < 1) nsplit = 1;
    float* part = nullptr;
    if (nsplit > 1) {
        const int64_t need = Tn * H * nsplit * (D + 2) * 4;
        if (!scratch || !scratch->data || scratch->bytes < need) return IK_ERR_INVALID_ARG;
        part = (float*)scratch->data;
    }
    const size_t smem = (size_t)KS * D * sizeof(T);
    dim3 g((unsigned)nsplit, (unsigned)Tn, (unsigned)cdiv(H, kHeads));
    attn_sparse_kernel<T><<<g, kThreads, smem, s>>>((const T*)q->data, (const T*)win->data, (const T*)cmp->data, (const int*)idx->data,
                                                    (const float*)sink->data, (T*)out->data, part, H, D, W, Nc, Kc, pos0,
                                                    pos ? (const int*)pos->data : nullptr, scale, nsplit);
    IK_TRY(ik_launch_status());
    if (nsplit > 1) {
        attn_sparse_merge<T><<<dim3((unsigned)Tn, (unsigned)H), 128, 0, s>>>(part, (const float*)sink->data, (T*)out->data, Tn, H, D, nsplit);
        IK_TRY(ik_launch_status());
    }
    return IK_OK;
}

}  // namespace

extern "C" int ik_attn_sparse(const ik_tensor* q, const ik_tensor* win, const ik_tensor* cmp, const ik_tensor* idx, const ik_tensor* sink,
                              int64_t window, int64_t pos0, float scale, ik_tensor* out, const ik_scratch* scratch, const ik_tensor* pos,
                              void* stream) {
    IK_TRY(check_float(q, 3));
    IK_TRY(ik_check(win, q->dtype, 2));
    IK_TRY(ik_check(cmp, q->dtype, 2));
    IK_TRY(ik_check(idx, IK_DTYPE_I32, 2));
    IK_TRY(ik_check(sink, IK_DTYPE_F32, 1));
    IK_TRY(ik_check(out, q->dtype, 3));
    IK_TRY(ik_check_pos(pos));
    const int64_t T = q->shape[0], H = q->shape[1], D = q->shape[2], W = window;
    if (W < 1 || win->shape[0] != W + T || win->shape[1] != D || cmp->shape[1] != D || idx->shape[0] != T || sink->shape[0] != H ||
        out->shape[0] != T || out->shape[1] != H || out->shape[2] != D)
        return IK_ERR_SHAPE;
    // D a power of two in 32..512 (the PV layout splits 256 threads into D/2 column pairs)
    if (D < 32 || D > kMaxD || (D & (D - 1)) != 0) return IK_ERR_SHAPE;
    if (!ik_aligned16(win->data) || !ik_aligned16(cmp->data) || (cmp->shape[0] > 0 && !cmp->data)) return IK_ERR_INVALID_ARG;
    if (T * H == 0) return IK_OK;
    cudaStream_t s = IK_STREAM(stream);
    switch (q->dtype) {
        case IK_DTYPE_F32: return launch<float>(q, win, cmp, idx, sink, W, pos0, scale, out, scratch, pos, s);
        case IK_DTYPE_BF16: return launch<__nv_bfloat16>(q, win, cmp, idx, sink, W, pos0, scale, out, scratch, pos, s);
        case IK_DTYPE_F16: return launch<__half>(q, win, cmp, idx, sink, W, pos0, scale, out, scratch, pos, s);
        default: return IK_ERR_DTYPE;
    }
}
