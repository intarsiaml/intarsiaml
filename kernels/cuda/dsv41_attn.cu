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
#include <stdlib.h>
#include <type_traits>

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

// Splits of a call of Tn queries whose key lists hold W + Kc entries in tiles of KS (the host policy).
__host__ __device__ inline int attn_nsplit(int64_t Tn, int64_t W, int64_t Kc, int KS) {
    int nsplit = 1;
    if (Tn <= kMaxSplitRows) {
        const int64_t ntiles = cdiv(W + Kc, KS);
        nsplit = (int)(ntiles < kMaxSplit ? ntiles : kMaxSplit);
    }
    return nsplit < 1 ? 1 : nsplit;
}

// What a call computes. Host position: the launch's values. Device position (`dpos`): P = *dpos, the visible compressed
// rows N = min((P + Tn) / ratio, Nc) of the capacity view cmp [Nc,D], the index entries Kc = min(Kw, N) (all Kw when Nc = 0)
// and the splits the host would use for an idx of Kc columns; the grid is sized for Kw and blocks past them exit.
struct AttnEff {
    int64_t p0, nc, kc;
    int nsplit;
};

template <int KS>
IK_DEVI AttnEff attn_eff(const int* dpos, int64_t pos0, int64_t Tn, int64_t W, int64_t Nc, int64_t Kw, int64_t ratio, int nsplit) {
    if (!dpos) return AttnEff{pos0, Nc, Kw, nsplit};
    AttnEff e;
    e.p0 = *dpos;
    int64_t n = (e.p0 + Tn) / ratio;
    if (n > Nc) n = Nc;
    e.nc = n;
    e.kc = Nc > 0 ? (Kw < n ? Kw : n) : Kw;
    e.nsplit = attn_nsplit(Tn, W, e.kc, KS);
    return e;
}

template <typename T>
__global__ void __launch_bounds__(kThreads) attn_sparse_kernel(const T* __restrict__ q, const T* __restrict__ win, const T* __restrict__ cmp,
                                                               const int* __restrict__ idx, const float* __restrict__ sink, T* __restrict__ out,
                                                               float* __restrict__ part, int64_t H, int64_t D, int64_t W, int64_t Nc,
                                                               int64_t Kw, int64_t pos0, const int* __restrict__ dpos, float scale, int nsplit,
                                                               int64_t ratio) {
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
    const AttnEff ef = attn_eff<KS>(dpos, pos0, gridDim.y, W, Nc, Kw, ratio, nsplit);
    if (split >= ef.nsplit) return;  // grid sized for the capacity (block-uniform)
    nsplit = ef.nsplit;
    Nc = ef.nc;
    const int64_t p0 = ef.p0, Kc = ef.kc;
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
                const int c = idx[t * Kw + (e - W)];
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


// ---- tensor-core path (bf16, heads per block a multiple of 16, D % 64 == 0) -------------------------------------------------
// Per tile of 32 keys: S = Q K^T (2 x 4 m16n8 tiles, one per warp, k = D) and O += P K (each warp owns D/8 output columns,
// both m-tiles, k = 32 keys) with mma.sync m16n8k16; the probabilities enter the second product rounded to bf16 (and the
// row sums are taken of the rounded values, so the weights stay normalised). Q and the key tile live in shared memory with
// rows padded by 8 elements (conflict-free fragment loads and ldmatrix).
constexpr int kMmaKeys = 32;

IK_DEVI void mma16816(float* c, unsigned a0, unsigned a1, unsigned a2, unsigned a3, unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

IK_DEVI void ldsm_x2_trans(unsigned& r0, unsigned& r1, const void* p) {
    const unsigned a = (unsigned)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n" : "=r"(r0), "=r"(r1) : "r"(a));
}

IK_DEVI unsigned lds32(const __nv_bfloat16* p) { return *reinterpret_cast<const unsigned*>(p); }

static size_t attn_mma_smem(int64_t D) {
    const int64_t ld = D + 8;
    return (size_t)(2 * kHeads * ld * 2 + kHeads * 33 * 4 + kHeads * 40 * 2);
}

// Online softmax of a warp's four rows 4w..4w+3 as four independent chains advanced together
// (loads, then the four max trees level by level, then exp / bf16 rounding, then the four sum trees level by level, then the stores and
// the four lane-0 updates), with one __syncwarp for the four rows instead of one per row. Per row the arithmetic is that of the former per-row loop:
// the same xor offsets 16, 8, 4, 2, 1 in the same order for max and for sum, the same bf16 rounding of p, the same alpha / l / m update.
// nh is not needed here: the rows of a head group that is not full are computed and never read, as in the loop.
IK_DEVI void softmax_rows4(const float* Ss, __nv_bfloat16* Ps, const int* valid, float* m_s, float* l_s, float* alpha_s, int warp, int lane) {
    float sv[4], mo[4], mx[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int r = warp * 4 + i;
        sv[i] = valid[lane] ? Ss[r * 33 + lane] : -INFINITY;
        mo[i] = m_s[r];
        mx[i] = sv[i];
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
#pragma unroll
        for (int i = 0; i < 4; ++i) mx[i] = fmaxf(mx[i], __shfl_xor_sync(0xffffffffu, mx[i], o));
    }
    __nv_bfloat16 pb[4];
    float mn[4], sum[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        mn[i] = fmaxf(mo[i], mx[i]);
        pb[i] = __float2bfloat16_rn(expf(sv[i] - mn[i]));
        sum[i] = __bfloat162float(pb[i]);
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
#pragma unroll
        for (int i = 0; i < 4; ++i) sum[i] += __shfl_xor_sync(0xffffffffu, sum[i], o);
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) Ps[(warp * 4 + i) * 40 + lane] = pb[i];
    __syncwarp();
    if (lane == 0) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int r = warp * 4 + i;
            const float a = expf(mo[i] - mn[i]);
            alpha_s[r] = a;
            l_s[r] = l_s[r] * a + sum[i];
            m_s[r] = mn[i];
        }
    }
}

__global__ void __launch_bounds__(kThreads) attn_sparse_mma(const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ win,
                                                            const __nv_bfloat16* __restrict__ cmp, const int* __restrict__ idx,
                                                            const float* __restrict__ sink, __nv_bfloat16* __restrict__ out,
                                                            float* __restrict__ part, int64_t H, int64_t D, int64_t W, int64_t Nc, int64_t Kw,
                                                            int64_t pos0, const int* __restrict__ dpos, float scale, int nsplit, int64_t ratio) {
    extern __shared__ __align__(16) unsigned char smem_raw[];
    const int ld = (int)D + 8;
    __nv_bfloat16* Qs = reinterpret_cast<__nv_bfloat16*>(smem_raw);
    __nv_bfloat16* Ks = Qs + kHeads * ld;
    float* Ss = reinterpret_cast<float*>(Ks + kMmaKeys * ld);  // [32][33]
    __nv_bfloat16* Ps = reinterpret_cast<__nv_bfloat16*>(Ss + kHeads * 33);  // [32][40]
    __shared__ float alpha_s[kHeads], m_s[kHeads], l_s[kHeads];
    __shared__ int valid[kMmaKeys];
    __shared__ int64_t srcrow[kMmaKeys];

    const int split = blockIdx.x;
    const int64_t t = blockIdx.y;
    const int64_t h0 = (int64_t)blockIdx.z * kHeads;
    const int nh = (int)min((int64_t)kHeads, H - h0);  // 16 or 32
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int g = lane >> 2, tq = lane & 3;
    const AttnEff ef = attn_eff<kMmaKeys>(dpos, pos0, gridDim.y, W, Nc, Kw, ratio, nsplit);
    if (split >= ef.nsplit) return;  // grid sized for the capacity (block-uniform)
    nsplit = ef.nsplit;
    Nc = ef.nc;
    const int64_t p0 = ef.p0, Kc = ef.kc;
    const int64_t nkeys = W + Kc;
    const int ntiles = (int)cdiv(nkeys, kMmaKeys);
    const int nv = (int)(D / 8);  // 16-byte vectors per row

    for (int v = threadIdx.x; v < kHeads * nv; v += kThreads) {
        const int r = v / nv, c = v % nv;
        uint4 x = make_uint4(0, 0, 0, 0);
        if (r < nh) x = __ldg(reinterpret_cast<const uint4*>(q + ((t * H + h0 + r) * D)) + c);
        *reinterpret_cast<uint4*>(Qs + r * ld + c * 8) = x;
    }
    if (threadIdx.x < kHeads) {
        m_s[threadIdx.x] = kInitMax;
        l_s[threadIdx.x] = 0.f;
    }
    // output columns of this warp: [cw, cw + D/8), D/64 n-tiles; both m-tiles
    const int ncol = (int)(D / 8), cw = warp * ncol, ntn = ncol / 8;
    float acc[2][8][4];
#pragma unroll
    for (int mt = 0; mt < 2; ++mt)
#pragma unroll
        for (int j = 0; j < 8; ++j) acc[mt][j][0] = acc[mt][j][1] = acc[mt][j][2] = acc[mt][j][3] = 0.f;

    for (int tile = split; tile < ntiles; tile += nsplit) {
        const int64_t e0 = (int64_t)tile * kMmaKeys;
        __syncthreads();
        if (threadIdx.x < kMmaKeys) {
            const int64_t e = e0 + threadIdx.x;
            int ok = 0;
            int64_t sr = 0;
            if (e < W) {
                const int64_t row = t + 1 + e;
                ok = (p0 + row >= W);
                sr = row;
            } else if (e < nkeys) {
                const int c = idx[t * Kw + (e - W)];
                ok = (c >= 0 && c < Nc);
                sr = -(int64_t)c - 1;
            }
            valid[threadIdx.x] = ok;
            srcrow[threadIdx.x] = sr;
        }
        __syncthreads();
        for (int v = threadIdx.x; v < kMmaKeys * nv; v += kThreads) {
            const int r = v / nv, c = v % nv;
            uint4 x = make_uint4(0, 0, 0, 0);
            if (valid[r]) {
                const int64_t sr = srcrow[r];
                const __nv_bfloat16* src = sr >= 0 ? win + sr * D : cmp + (-sr - 1) * D;
                x = __ldg(reinterpret_cast<const uint4*>(src) + c);
            }
            *reinterpret_cast<uint4*>(Ks + r * ld + c * 8) = x;
        }
        __syncthreads();
        // S tile (m-tile warp & 1, n-tile warp >> 1)
        {
            const int mt = warp & 1, nt = warp >> 1;
            float c4[4] = {0.f, 0.f, 0.f, 0.f};
            if (mt * 16 < nh) {
                const __nv_bfloat16* qa = Qs + (mt * 16 + g) * ld + 2 * tq;
                const __nv_bfloat16* kb = Ks + (nt * 8 + g) * ld + 2 * tq;
                for (int k0 = 0; k0 < D; k0 += 16)
                    mma16816(c4, lds32(qa + k0), lds32(qa + 8 * ld + k0), lds32(qa + k0 + 8), lds32(qa + 8 * ld + k0 + 8), lds32(kb + k0),
                             lds32(kb + k0 + 8));
            }
            float* srow = Ss + (mt * 16 + g) * 33 + nt * 8 + 2 * tq;
            srow[0] = c4[0] * scale;
            srow[1] = c4[1] * scale;
            srow[8 * 33] = c4[2] * scale;
            srow[8 * 33 + 1] = c4[3] * scale;
        }
        __syncthreads();
        // online softmax: warp w rows 4w..4w+3, lane = key
        softmax_rows4(Ss, Ps, valid, m_s, l_s, alpha_s, warp, lane);
        __syncthreads();
        // O = O * alpha + P K
        {
            float al[2][2];
#pragma unroll
            for (int mt = 0; mt < 2; ++mt) {
                al[mt][0] = alpha_s[mt * 16 + g];
                al[mt][1] = alpha_s[mt * 16 + g + 8];
            }
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                if (j >= ntn) break;
#pragma unroll
                for (int mt = 0; mt < 2; ++mt) {
                    acc[mt][j][0] *= al[mt][0];
                    acc[mt][j][1] *= al[mt][0];
                    acc[mt][j][2] *= al[mt][1];
                    acc[mt][j][3] *= al[mt][1];
                }
            }
#pragma unroll
            for (int ks = 0; ks < 2; ++ks) {
                unsigned a[2][4];
#pragma unroll
                for (int mt = 0; mt < 2; ++mt) {
                    const __nv_bfloat16* pa = Ps + (mt * 16 + g) * 40 + ks * 16 + 2 * tq;
                    a[mt][0] = lds32(pa);
                    a[mt][1] = lds32(pa + 8 * 40);
                    a[mt][2] = lds32(pa + 8);
                    a[mt][3] = lds32(pa + 8 * 40 + 8);
                }
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    if (j >= ntn) break;
                    unsigned b0, b1;
                    ldsm_x2_trans(b0, b1, Ks + (ks * 16 + (lane & 15)) * ld + cw + j * 8);
#pragma unroll
                    for (int mt = 0; mt < 2; ++mt)
                        if (mt * 16 < nh) mma16816(acc[mt][j], a[mt][0], a[mt][1], a[mt][2], a[mt][3], b0, b1);
                }
            }
        }
    }
    __syncthreads();
#pragma unroll
    for (int mt = 0; mt < 2; ++mt) {
#pragma unroll
        for (int hr = 0; hr < 2; ++hr) {
            const int r = mt * 16 + g + 8 * hr;
            if (r >= nh) continue;
            const int64_t h = h0 + r;
            const float m = m_s[r], l = l_s[r];
            const float inv = nsplit == 1 ? 1.f / (l + expf(sink[h] - m)) : 1.f;
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                if (j >= ntn) break;
                const int col = cw + j * 8 + 2 * tq;
                const float v0 = acc[mt][j][2 * hr], v1 = acc[mt][j][2 * hr + 1];
                if (nsplit == 1) {
                    *reinterpret_cast<__nv_bfloat162*>(out + (t * H + h) * D + col) = __floats2bfloat162_rn(v0 * inv, v1 * inv);
                } else {
                    const int64_t slot = (t * H + h) * nsplit + split;
                    part[slot * D + col] = v0;
                    part[slot * D + col + 1] = v1;
                    if (warp == 0 && j == 0 && tq == 0) {
                        float* pm = part + (int64_t)gridDim.y * H * nsplit * D + slot * 2;
                        pm[0] = m;
                        pm[1] = l;
                    }
                }
            }
        }
    }
}

static bool attn_mma_enabled() {
    static int on = -1;
    if (on < 0) {
        const char* e = getenv("INTARSIA_DSV41_ATTN_SIMT");
        on = !(e && e[0] == '1');
    }
    return on == 1;
}

// merge the split partials of one (query, head): o = sum_s acc_s e^(m_s - M) / (sum_s l_s e^(m_s - M) + e^(sink - M))
// Device position: the splits the attention kernel took (KS its tile); one split wrote the output directly.
template <typename T, int KS>
__global__ void attn_sparse_merge(const float* __restrict__ part, const float* __restrict__ sink, T* __restrict__ out, int64_t Tn, int64_t H,
                                  int64_t D, int nsplit, const int* __restrict__ dpos, int64_t W, int64_t Nc, int64_t Kw, int64_t ratio) {
    nsplit = attn_eff<KS>(dpos, 0, Tn, W, Nc, Kw, ratio, nsplit).nsplit;
    if (nsplit <= 1) return;
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
           int64_t pos0, int64_t ratio, float scale, ik_tensor* out, const ik_scratch* scratch, const ik_tensor* pos, cudaStream_t s) {
    constexpr int KS = tile_keys<T>();
    const int64_t Tn = q->shape[0], H = q->shape[1], D = q->shape[2];
    const int64_t Nc = cmp->shape[0], Kw = idx->shape[1];
    const int* dp = pos ? (const int*)pos->data : nullptr;
    // splits of the grid (and the scratch layout's bound) from the idx width; a device-position call takes as many of
    // them as its device index count needs (attn_eff), the merge reads that many
    float* part = nullptr;
    bool mma = false;
    if constexpr (std::is_same<T, __nv_bfloat16>::value) {
        mma = attn_mma_enabled() && H % 16 == 0 && D % 64 == 0 && D <= kMaxD && ik_aligned16(q->data) && ik_aligned16(out->data);
    }
    const int nsplit = attn_nsplit(Tn, W, Kw, mma ? kMmaKeys : KS);
    if (nsplit > 1) {
        const int64_t need = Tn * H * nsplit * (D + 2) * 4;
        if (!scratch || !scratch->data || scratch->bytes < need) return IK_ERR_INVALID_ARG;
        part = (float*)scratch->data;
    }
    dim3 g((unsigned)nsplit, (unsigned)Tn, (unsigned)cdiv(H, kHeads));
    if (mma) {
        static_assert(kMmaKeys == 32, "split policy assumes 32-key tiles");
        const size_t smem = attn_mma_smem(D);
        static bool attr = false;
        if (!attr) {
            cudaFuncSetAttribute(attn_sparse_mma, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)attn_mma_smem(kMaxD));
            attr = true;
        }
        attn_sparse_mma<<<g, kThreads, smem, s>>>((const __nv_bfloat16*)q->data, (const __nv_bfloat16*)win->data, (const __nv_bfloat16*)cmp->data,
                                                  (const int*)idx->data, (const float*)sink->data, (__nv_bfloat16*)out->data, part, H, D, W, Nc,
                                                  Kw, pos0, dp, scale, nsplit, ratio);
        IK_TRY(ik_launch_status());
        if (nsplit > 1) {
            attn_sparse_merge<T, kMmaKeys><<<dim3((unsigned)Tn, (unsigned)H), 128, 0, s>>>(part, (const float*)sink->data, (T*)out->data, Tn, H, D,
                                                                                          nsplit, dp, W, Nc, Kw, ratio);
            IK_TRY(ik_launch_status());
        }
        return IK_OK;
    }
    const size_t smem = (size_t)KS * D * sizeof(T);
    attn_sparse_kernel<T><<<g, kThreads, smem, s>>>((const T*)q->data, (const T*)win->data, (const T*)cmp->data, (const int*)idx->data,
                                                    (const float*)sink->data, (T*)out->data, part, H, D, W, Nc, Kw, pos0, dp, scale, nsplit,
                                                    ratio);
    IK_TRY(ik_launch_status());
    if (nsplit > 1) {
        attn_sparse_merge<T, KS><<<dim3((unsigned)Tn, (unsigned)H), 128, 0, s>>>(part, (const float*)sink->data, (T*)out->data, Tn, H, D, nsplit,
                                                                                dp, W, Nc, Kw, ratio);
        IK_TRY(ik_launch_status());
    }
    return IK_OK;
}

}  // namespace

extern "C" int ik_attn_sparse(const ik_tensor* q, const ik_tensor* win, const ik_tensor* cmp, const ik_tensor* idx, const ik_tensor* sink,
                              int64_t window, int64_t pos0, int64_t ratio, float scale, ik_tensor* out, const ik_scratch* scratch,
                              const ik_tensor* pos, void* stream) {
    IK_TRY(check_float(q, 3));
    IK_TRY(ik_check(win, q->dtype, 2));
    IK_TRY(ik_check(cmp, q->dtype, 2));
    IK_TRY(ik_check(idx, IK_DTYPE_I32, 2));
    IK_TRY(ik_check(sink, IK_DTYPE_F32, 1));
    IK_TRY(ik_check(out, q->dtype, 3));
    IK_TRY(ik_check_pos(pos));
    const int64_t T = q->shape[0], H = q->shape[1], D = q->shape[2], W = window;
    if (W < 1 || ratio < 1 || win->shape[0] != W + T || win->shape[1] != D || cmp->shape[1] != D || idx->shape[0] != T || sink->shape[0] != H ||
        out->shape[0] != T || out->shape[1] != H || out->shape[2] != D)
        return IK_ERR_SHAPE;
    // D a power of two in 32..512 (the PV layout splits 256 threads into D/2 column pairs)
    if (D < 32 || D > kMaxD || (D & (D - 1)) != 0) return IK_ERR_SHAPE;
    if (!ik_aligned16(win->data) || !ik_aligned16(cmp->data) || (cmp->shape[0] > 0 && !cmp->data)) return IK_ERR_INVALID_ARG;
    if (T * H == 0) return IK_OK;
    cudaStream_t s = IK_STREAM(stream);
    switch (q->dtype) {
        case IK_DTYPE_F32: return launch<float>(q, win, cmp, idx, sink, W, pos0, ratio, scale, out, scratch, pos, s);
        case IK_DTYPE_BF16: return launch<__nv_bfloat16>(q, win, cmp, idx, sink, W, pos0, ratio, scale, out, scratch, pos, s);
        case IK_DTYPE_F16: return launch<__half>(q, win, cmp, idx, sink, W, pos0, ratio, scale, out, scratch, pos, s);
        default: return IK_ERR_DTYPE;
    }
}
