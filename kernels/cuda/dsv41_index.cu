// DeepSeek-V4.1 indexer: indexer_blocks (select_candidate_blocks, model.py:584) and indexer_topk (Indexer.forward,
// model.py:527-582) on the index score score[t,c] = scale * sum_h w[t,h] relu(q[t,h] . k[c]) (c < (pos0+t+1)/ratio, -inf
// past it). Contract: docs/design/dsv41-ops.md. Written from the official inference code (MIT, DeepSeek).
//
// Tiled: a tile of query rows at a time (as many as the scratch holds, at least one), never a per-head score. Per tile,
//  1. score kernel: one lane per key row, the query's heads in registers, its q / w in shared memory; writes the head-summed
//     score row [Nk] (topk; rows outside kept candidate blocks are -inf and their keys are never read) or the block maxima
//     [ceil(Nk/B)] (blocks) to the scratch;
//  2. select kernel: one block per query row, an exact radix select of the k-th best value (value descending, ties to the
//     lower index), then one ordered pass that emits the picks in position order (topk: indices, picks past the query's
//     visible rows -1; blocks: the u8 mask with the newest visible block pinned and -inf blocks dropped).
#include "dsv41_common.cuh"
#include <stdlib.h>
#include <type_traits>

using namespace dsv41;

namespace {

constexpr int kScoreThreads = 256;  // key rows per score block (one per lane)
constexpr int kMaxHi = 32;
constexpr int kMaxQ = 8192;  // Hi * Di floats of the query in shared memory
constexpr int kSelThreads = 1024;
constexpr int kSelWarps = kSelThreads / 32;

template <typename T> IK_DEVI void load8(const T* p, float* f);
template <> IK_DEVI void load8<__nv_bfloat16>(const __nv_bfloat16* p, float* f) { ik_unpack8(__ldg(reinterpret_cast<const uint4*>(p)), f); }
template <> IK_DEVI void load8<__half>(const __half* p, float* f) {
    const uint4 u = __ldg(reinterpret_cast<const uint4*>(p));
    const __half2* h = reinterpret_cast<const __half2*>(&u);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float2 t = __half22float2(h[i]);
        f[2 * i] = t.x;
        f[2 * i + 1] = t.y;
    }
}
template <> IK_DEVI void load8<float>(const float* p, float* f) {
    const float4 a = __ldg(reinterpret_cast<const float4*>(p)), b = __ldg(reinterpret_cast<const float4*>(p) + 1);
    f[0] = a.x, f[1] = a.y, f[2] = a.z, f[3] = a.w, f[4] = b.x, f[5] = b.y, f[6] = b.z, f[7] = b.w;
}

// BLOCKS: write block maxima (row length nb) instead of the score row (row length Nk)
template <typename T, bool BLOCKS>
__global__ void __launch_bounds__(kScoreThreads) index_score(const T* __restrict__ q, const T* __restrict__ k, const T* __restrict__ w,
                                                             const unsigned char* __restrict__ cand, int64_t t0, int64_t Hi, int64_t Di,
                                                             int64_t Nk, int64_t B, int64_t ratio, int64_t pos0, float scale,
                                                             float* __restrict__ out) {
    __shared__ __align__(16) float qs[kMaxQ];
    __shared__ float ws[kMaxHi];
    const int64_t tl = blockIdx.y, t = t0 + tl;
    for (int64_t i = threadIdx.x; i < Hi * Di; i += kScoreThreads) qs[i] = ld(q, t * Hi * Di + i);
    if (threadIdx.x < Hi) ws[threadIdx.x] = ld(w, t * Hi + threadIdx.x);
    __syncthreads();
    const int64_t c = (int64_t)blockIdx.x * kScoreThreads + threadIdx.x;
    const int64_t nt = (pos0 + t + 1) / ratio;
    const int64_t nb = cdiv(Nk, B);
    bool ok = c < Nk && c < nt;
    if (ok && cand) ok = cand[t * nb + c / B] != 0;
    float score = -INFINITY;
    if (ok) {
        float acc[kMaxHi];
#pragma unroll
        for (int h = 0; h < kMaxHi; ++h) acc[h] = 0.f;
        const T* kr = k + c * Di;
        for (int64_t d = 0; d < Di; d += 8) {
            float kv[8];
            load8<T>(kr + d, kv);
#pragma unroll
            for (int h = 0; h < kMaxHi; ++h) {
                if (h < Hi) {
                    const float4 qa = *reinterpret_cast<const float4*>(qs + h * Di + d);
                    const float4 qb = *reinterpret_cast<const float4*>(qs + h * Di + d + 4);
                    float a = acc[h];
                    a = fmaf(qa.x, kv[0], a);
                    a = fmaf(qa.y, kv[1], a);
                    a = fmaf(qa.z, kv[2], a);
                    a = fmaf(qa.w, kv[3], a);
                    a = fmaf(qb.x, kv[4], a);
                    a = fmaf(qb.y, kv[5], a);
                    a = fmaf(qb.z, kv[6], a);
                    a = fmaf(qb.w, kv[7], a);
                    acc[h] = a;
                }
            }
        }
        float s = 0.f;
#pragma unroll
        for (int h = 0; h < kMaxHi; ++h)
            if (h < Hi) s = fmaf(ws[h], fmaxf(acc[h], 0.f), s);
        score = s * scale;
        if (score == 0.f) score = 0.f;  // -0 ties +0
    }
    if (BLOCKS) {
        float m = score;
        for (int o = 1; o < B; o <<= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
        if ((c % B) == 0 && c < Nk) out[tl * nb + c / B] = m;
    } else if (c < Nk) {
        out[tl * Nk + c] = score;
    }
}

// order-preserving key: larger float -> larger key; -0 and +0 were already merged
IK_DEVI unsigned fkey(float f) {
    const unsigned u = __float_as_uint(f);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

// inclusive block scan of one value per thread (kSelThreads threads); returns the inclusive prefix, `total` the block sum
IK_DEVI unsigned block_scan(unsigned v, unsigned* wsum, unsigned& total) {
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    unsigned x = v;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        const unsigned y = __shfl_up_sync(0xffffffffu, x, o);
        if (lane >= o) x += y;
    }
    __syncthreads();
    if (lane == 31) wsum[w] = x;
    __syncthreads();
    if (w == 0) {
        unsigned s = lane < kSelWarps ? wsum[lane] : 0u;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            const unsigned y = __shfl_up_sync(0xffffffffu, s, o);
            if (lane >= o) s += y;
        }
        if (lane < kSelWarps) wsum[lane] = s;
    }
    __syncthreads();
    total = wsum[kSelWarps - 1];
    return x + (w > 0 ? wsum[w - 1] : 0u);
}

// One block per row of `v` (row length N): the k best (value descending, ties to the lower index), emitted in index order.
// BLOCKS: mask row (u8, length N) of the picks whose value is > -inf, after pinning element `pin`; else idx row (length k),
// index >= nt mapped to -1.
template <bool BLOCKS>
__global__ void __launch_bounds__(kSelThreads) index_select(float* __restrict__ v, int64_t N, int64_t k, int64_t t0, int64_t B, int64_t ratio,
                                                            int64_t pos0, int* __restrict__ idx, unsigned char* __restrict__ mask) {
    __shared__ unsigned hist[256];
    __shared__ unsigned wsum[32];
    __shared__ unsigned s_digit, s_rem;
    __shared__ unsigned s_cgt, s_ceq;
    __shared__ int s_done;
    const int64_t tl = blockIdx.x, t = t0 + tl;
    const int64_t nt = (pos0 + t + 1) / ratio;
    float* row = v + tl * N;
    if (BLOCKS) {
        unsigned char* mrow = mask + t * N;
        for (int64_t i = threadIdx.x; i < N; i += kSelThreads) mrow[i] = 0;
        if (threadIdx.x == 0 && nt > 0) row[(nt - 1) / B] = INFINITY;
        __syncthreads();
    }
    if (k <= 0) return;
    unsigned prefix = 0, pmask = 0, rem = (unsigned)k;
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    for (int pass = 0; pass < 4; ++pass) {
        const int shift = 24 - 8 * pass;
        if (threadIdx.x < 256) hist[threadIdx.x] = 0;
        __syncthreads();
        for (int64_t i = threadIdx.x; i < N; i += kSelThreads) {
            const unsigned key = fkey(row[i]);
            if ((key & pmask) == prefix) {
                const unsigned dg = (key >> shift) & 255u;
                const unsigned peers = __match_any_sync(__activemask(), dg);
                if (lane == __ffs(peers) - 1) atomicAdd(&hist[dg], (unsigned)__popc(peers));
            }
        }
        __syncthreads();
        if (w == 0) {
            // lane l owns bins 255-8l .. 248-8l (descending)
            unsigned loc = 0;
#pragma unroll
            for (int i = 0; i < 8; ++i) loc += hist[255 - 8 * lane - i];
            unsigned inc = loc;
#pragma unroll
            for (int o = 1; o < 32; o <<= 1) {
                const unsigned y = __shfl_up_sync(0xffffffffu, inc, o);
                if (lane >= o) inc += y;
            }
            const unsigned hit = __ballot_sync(0xffffffffu, inc >= rem);
            const int first = __ffs(hit) - 1;  // rem <= count of matching keys, so some lane hits
            if (lane == first) {
                unsigned cum = inc - loc;
                for (int i = 0; i < 8; ++i) {
                    const unsigned b = 255 - 8 * lane - i;
                    if (cum + hist[b] >= rem) {
                        s_digit = b;
                        s_rem = rem - cum;
                        break;
                    }
                    cum += hist[b];
                }
            }
        }
        __syncthreads();
        prefix |= s_digit << shift;
        pmask |= 255u << shift;
        rem = s_rem;
        __syncthreads();
    }
    const unsigned kth = prefix;
    const unsigned need_eq = rem;
    const unsigned need_gt = (unsigned)k - rem;
    if (threadIdx.x == 0) {
        s_cgt = 0;
        s_ceq = 0;
        s_done = 0;
    }
    __syncthreads();
    for (int64_t base = 0; base < N; base += kSelThreads) {
        const int64_t i = base + threadIdx.x;
        unsigned gt = 0, eq = 0;
        if (i < N) {
            const unsigned key = fkey(row[i]);
            gt = key > kth;
            eq = key == kth;
        }
        unsigned total;
        const unsigned inc = block_scan(gt | (eq << 16), wsum, total);
        const unsigned exc = inc - (gt | (eq << 16));
        const unsigned cgt = s_cgt, ceq = s_ceq;
        const unsigned gb = cgt + (exc & 0xFFFFu), eb = ceq + (exc >> 16);
        if (gt || (eq && eb < need_eq)) {
            const unsigned o = gb + min(eb, need_eq);
            if (BLOCKS) {
                if (row[i] > -INFINITY) mask[t * N + i] = 1;
            } else {
                idx[t * k + o] = i < nt ? (int)i : -1;
            }
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            s_cgt = cgt + (total & 0xFFFFu);
            s_ceq = ceq + (total >> 16);
            s_done = (s_cgt >= need_gt && s_ceq >= need_eq);
        }
        __syncthreads();
        if (s_done) break;
    }
}


// ---- tensor-core score (bf16 q / k, Hi in {16, 32}, Di % 16 == 0, Di <= 128, block <= 8) ---------------------------------
// The index score is a GEMM [T*Hi, Di] x [Di, Nk] with a per-query epilogue (relu, weight, sum over the Hi rows). A block
// takes kMmaKeys keys (a shared-memory tile, row stride padded so the B fragments load without bank conflicts) and up to 8
// queries; each warp holds one query's A fragments in registers and runs mma.sync m16n8k16 over its n-tiles of 8 keys, then
// reduces the 16 / 32 head rows of each column across the lanes. An n-tile of 8 keys is one candidate block (block 8): tiles
// outside the query's candidates or past its visible rows are not multiplied. Same score as index_score up to f32 sum order.
constexpr int kMmaKeys = 128;
constexpr int kMmaWarps = 8;
constexpr int kMmaMaxDi = 128;

IK_DEVI void mma_bf16(float* c, const unsigned* a, unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

template <bool BLOCKS>
__global__ void __launch_bounds__(kMmaWarps * 32) index_score_mma(const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ k,
                                                                const __nv_bfloat16* __restrict__ w, const unsigned char* __restrict__ cand,
                                                                int64_t t0, int64_t tr, int Hi, int Di, int64_t Nk, int B, int64_t ratio,
                                                                int64_t pos0, float scale, int qb, float* __restrict__ out) {
    extern __shared__ __align__(16) unsigned char smem_raw[];
    __nv_bfloat16* kt = reinterpret_cast<__nv_bfloat16*>(smem_raw);  // [kMmaKeys][Di + 8]
    const int ldk = Di + 8;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int g = lane >> 2, tq = lane & 3;
    const int64_t c0 = (int64_t)blockIdx.x * kMmaKeys;
    const int64_t nb = cdiv(Nk, B);
    const int64_t q0 = (int64_t)blockIdx.y * qb;  // first query (tile-local) of this block
    const int nq = (int)min((int64_t)qb, tr - q0);
    // rows the block needs: below the last query's visible rows; with one query also inside its candidates
    const int64_t nt_last = (pos0 + t0 + q0 + nq) / ratio;
    for (int v = threadIdx.x; v < kMmaKeys * (Di / 8); v += kMmaWarps * 32) {
        const int r = v / (Di / 8), cv = v % (Di / 8);
        const int64_t c = c0 + r;
        uint4 val = make_uint4(0, 0, 0, 0);
        bool need = c < Nk && c < nt_last;
        if (need && cand && nq == 1) need = cand[(t0 + q0) * nb + c / B] != 0;
        if (need) val = __ldg(reinterpret_cast<const uint4*>(k + c * Di) + cv);
        *reinterpret_cast<uint4*>(kt + r * ldk + cv * 8) = val;
    }
    __syncthreads();
    const int wpq = kMmaWarps / qb;  // warps per query
    const int qi = warp / wpq;
    if (qi >= nq) return;
    const int64_t tl = q0 + qi, t = t0 + tl;
    const int64_t nt = (pos0 + t + 1) / ratio;
    const int mt_n = Hi / 16, ks_n = Di / 16;
    // A fragments of this query: [m-tile][k-step][4]
    unsigned af[2][kMmaMaxDi / 16][4];
    const __nv_bfloat16* qt = q + t * Hi * Di;
#pragma unroll
    for (int mt = 0; mt < 2; ++mt)
#pragma unroll
        for (int ks = 0; ks < kMmaMaxDi / 16; ++ks) {
            if (mt < mt_n && ks < ks_n) {
                const __nv_bfloat16* r0 = qt + (mt * 16 + g) * Di + ks * 16 + 2 * tq;
                const __nv_bfloat16* r1 = r0 + 8 * Di;
                af[mt][ks][0] = *reinterpret_cast<const unsigned*>(r0);
                af[mt][ks][1] = *reinterpret_cast<const unsigned*>(r1);
                af[mt][ks][2] = *reinterpret_cast<const unsigned*>(r0 + 8);
                af[mt][ks][3] = *reinterpret_cast<const unsigned*>(r1 + 8);
            } else {
                af[mt][ks][0] = af[mt][ks][1] = af[mt][ks][2] = af[mt][ks][3] = 0u;
            }
        }
    float wv[2][2];
#pragma unroll
    for (int mt = 0; mt < 2; ++mt) {
        wv[mt][0] = mt < mt_n ? __bfloat162float(w[t * Hi + mt * 16 + g]) : 0.f;
        wv[mt][1] = mt < mt_n ? __bfloat162float(w[t * Hi + mt * 16 + g + 8]) : 0.f;
    }
    for (int ntile = warp % wpq; ntile < kMmaKeys / 8; ntile += wpq) {
        const int64_t n0 = c0 + ntile * 8;
        if (n0 >= Nk) break;
        // whole tile invisible or outside the candidates (block 8 = one tile): -inf without multiplying
        bool any = n0 < nt;
        if (any && cand && B == 8) any = cand[t * nb + n0 / 8] != 0;
        float s0 = -INFINITY, s1 = -INFINITY;
        const int64_t col = n0 + 2 * tq;
        if (any) {
            float acc[2][4];
#pragma unroll
            for (int mt = 0; mt < 2; ++mt) acc[mt][0] = acc[mt][1] = acc[mt][2] = acc[mt][3] = 0.f;
            const __nv_bfloat16* kr = kt + (ntile * 8 + g) * ldk + 2 * tq;
#pragma unroll
            for (int ks = 0; ks < kMmaMaxDi / 16; ++ks) {
                if (ks < ks_n) {
                    const unsigned b0 = *reinterpret_cast<const unsigned*>(kr + ks * 16);
                    const unsigned b1 = *reinterpret_cast<const unsigned*>(kr + ks * 16 + 8);
#pragma unroll
                    for (int mt = 0; mt < 2; ++mt)
                        if (mt < mt_n) mma_bf16(acc[mt], af[mt][ks], b0, b1);
                }
            }
            float p0 = 0.f, p1 = 0.f;
#pragma unroll
            for (int mt = 0; mt < 2; ++mt) {
                p0 = fmaf(wv[mt][0], fmaxf(acc[mt][0], 0.f), p0);
                p0 = fmaf(wv[mt][1], fmaxf(acc[mt][2], 0.f), p0);
                p1 = fmaf(wv[mt][0], fmaxf(acc[mt][1], 0.f), p1);
                p1 = fmaf(wv[mt][1], fmaxf(acc[mt][3], 0.f), p1);
            }
#pragma unroll
            for (int o = 4; o < 32; o <<= 1) {
                p0 += __shfl_xor_sync(0xffffffffu, p0, o);
                p1 += __shfl_xor_sync(0xffffffffu, p1, o);
            }
            s0 = p0 * scale;
            s1 = p1 * scale;
            if (s0 == 0.f) s0 = 0.f;
            if (s1 == 0.f) s1 = 0.f;
            bool v0 = col < nt && col < Nk, v1 = col + 1 < nt && col + 1 < Nk;
            if (cand && B != 8) {
                v0 = v0 && cand[t * nb + col / B] != 0;
                v1 = v1 && cand[t * nb + (col + 1) / B] != 0;
            }
            if (!v0) s0 = -INFINITY;
            if (!v1) s1 = -INFINITY;
        }
        if (BLOCKS) {
            // block maxima: B in {1, 2, 4, 8} columns, lanes tq hold columns 2tq, 2tq+1
            if (B == 1) {
                if (g == 0) {
                    if (col < Nk) out[tl * nb + col] = s0;
                    if (col + 1 < Nk) out[tl * nb + col + 1] = s1;
                }
            } else {
                float m = fmaxf(s0, s1);
                if (B >= 4) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 1));
                if (B >= 8) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 2));
                if (g == 0 && (col % B) == 0 && col < Nk) out[tl * nb + col / B] = m;
            }
        } else if (g == 0) {
            if (col < Nk) out[tl * Nk + col] = s0;
            if (col + 1 < Nk) out[tl * Nk + col + 1] = s1;
        }
    }
}

static bool index_mma_enabled() {
    static int on = -1;
    if (on < 0) {
        const char* e = getenv("INTARSIA_DSV41_INDEX_SIMT");
        on = !(e && e[0] == '1');
    }
    return on == 1;
}

struct IdxArgs {
    const ik_tensor *q, *k, *w, *cand;
    int64_t keep, B, ratio, pos0;
    float scale;
    const ik_scratch* scratch;
};

template <typename T, bool BLOCKS>
int run(const IdxArgs& a, ik_tensor* outp, cudaStream_t s) {
    const int64_t Tn = a.q->shape[0], Hi = a.q->shape[1], Di = a.q->shape[2], Nk = a.k->shape[0];
    const int64_t nb = cdiv(Nk, a.B);
    const int64_t len = BLOCKS ? nb : Nk;
    const int64_t kk = BLOCKS ? (a.keep < nb ? a.keep : nb) : (a.keep < Nk ? a.keep : Nk);
    if (Tn == 0) return IK_OK;
    if (Nk == 0) {
        // no compressed rows: an empty idx [T, 0]; a mask [T, 0]
        return IK_OK;
    }
    if (!a.scratch || !a.scratch->data || a.scratch->bytes < len * 4) return IK_ERR_INVALID_ARG;
    int64_t rows = a.scratch->bytes / (len * 4);
    if (rows > Tn) rows = Tn;
    if (rows > 65535) rows = 65535;
    float* sc = (float*)a.scratch->data;
    const unsigned char* cand = (!BLOCKS && a.cand && ik_numel(a.cand) > 0) ? (const unsigned char*)a.cand->data : nullptr;
    for (int64_t t0 = 0; t0 < Tn; t0 += rows) {
        const int64_t tr = Tn - t0 < rows ? Tn - t0 : rows;
        bool mma = false;
        if constexpr (std::is_same<T, __nv_bfloat16>::value) {
            mma = index_mma_enabled() && (Hi == 16 || Hi == 32) && Di % 16 == 0 && Di <= kMmaMaxDi && a.B <= 8 &&
                  ik_aligned16(a.q->data) && ik_aligned16(a.w->data) && (Hi * Di) % 8 == 0;
        }
        if (mma) {
            int qb = 1;
            while (qb < kMmaWarps && qb * 2 <= tr) qb *= 2;
            dim3 g((unsigned)cdiv(Nk, kMmaKeys), (unsigned)cdiv(tr, qb));
            const size_t smem = (size_t)kMmaKeys * (Di + 8) * 2;
            index_score_mma<BLOCKS><<<g, kMmaWarps * 32, smem, s>>>((const __nv_bfloat16*)a.q->data, (const __nv_bfloat16*)a.k->data,
                                                                   (const __nv_bfloat16*)a.w->data, cand, t0, tr, (int)Hi, (int)Di, Nk,
                                                                   (int)a.B, a.ratio, a.pos0, a.scale, qb, sc);
        } else {
            dim3 g((unsigned)cdiv(Nk, kScoreThreads), (unsigned)tr);
            index_score<T, BLOCKS><<<g, kScoreThreads, 0, s>>>((const T*)a.q->data, (const T*)a.k->data, (const T*)a.w->data, cand, t0, Hi,
                                                               Di, Nk, a.B, a.ratio, a.pos0, a.scale, sc);
        }
        IK_TRY(ik_launch_status());
        index_select<BLOCKS><<<(unsigned)tr, kSelThreads, 0, s>>>(sc, len, kk, t0, a.B, a.ratio, a.pos0,
                                                                   BLOCKS ? nullptr : (int*)outp->data,
                                                                   BLOCKS ? (unsigned char*)outp->data : nullptr);
        IK_TRY(ik_launch_status());
    }
    return IK_OK;
}

int check_common(const IdxArgs& a) {
    IK_TRY(check_float(a.q, 3));
    IK_TRY(ik_check(a.k, a.q->dtype, 2));
    IK_TRY(ik_check(a.w, a.q->dtype, 2));
    const int64_t Tn = a.q->shape[0], Hi = a.q->shape[1], Di = a.q->shape[2], Nk = a.k->shape[0];
    if (a.ratio < 1 || a.pos0 < 0 || a.B < 1 || a.B > 32 || (a.B & (a.B - 1)) != 0 || a.keep < 1) return IK_ERR_SHAPE;
    if (Hi < 1 || Hi > kMaxHi || Hi * Di > kMaxQ || Di % 8 != 0) return IK_ERR_SHAPE;
    if (a.k->shape[1] != Di || a.w->shape[0] != Tn || a.w->shape[1] != Hi || Nk != (a.pos0 + Tn) / a.ratio) return IK_ERR_SHAPE;
    if (Nk > 0 && !ik_aligned16(a.k->data)) return IK_ERR_INVALID_ARG;
    return IK_OK;
}

}  // namespace

extern "C" int ik_indexer_blocks(const ik_tensor* q, const ik_tensor* k, const ik_tensor* w, int64_t keep, int64_t block, int64_t ratio,
                                 int64_t pos0, float scale, ik_tensor* mask, const ik_scratch* scratch, void* stream) {
    IdxArgs a{q, k, w, nullptr, keep, block, ratio, pos0, scale, scratch};
    IK_TRY(check_common(a));
    IK_TRY(ik_check(mask, IK_DTYPE_U8, 2));
    if (mask->shape[0] != q->shape[0] || mask->shape[1] != cdiv(k->shape[0], block)) return IK_ERR_SHAPE;
    cudaStream_t s = IK_STREAM(stream);
    switch (q->dtype) {
        case IK_DTYPE_F32: return run<float, true>(a, mask, s);
        case IK_DTYPE_BF16: return run<__nv_bfloat16, true>(a, mask, s);
        case IK_DTYPE_F16: return run<__half, true>(a, mask, s);
        default: return IK_ERR_DTYPE;
    }
}

extern "C" int ik_indexer_topk(const ik_tensor* q, const ik_tensor* k, const ik_tensor* w, const ik_tensor* cand, int64_t top_k,
                               int64_t block, int64_t ratio, int64_t pos0, float scale, ik_tensor* idx, const ik_scratch* scratch,
                               void* stream) {
    IdxArgs a{q, k, w, cand, top_k, block, ratio, pos0, scale, scratch};
    IK_TRY(check_common(a));
    IK_TRY(ik_check(idx, IK_DTYPE_I32, 2));
    const int64_t Nk = k->shape[0];
    if (idx->shape[0] != q->shape[0] || idx->shape[1] != (top_k < Nk ? top_k : Nk)) return IK_ERR_SHAPE;
    if (cand && ik_numel(cand) > 0) {
        IK_TRY(ik_check(cand, IK_DTYPE_U8, 2));
        if (cand->shape[0] != q->shape[0] || cand->shape[1] != cdiv(Nk, block)) return IK_ERR_SHAPE;
    }
    cudaStream_t s = IK_STREAM(stream);
    switch (q->dtype) {
        case IK_DTYPE_F32: return run<float, false>(a, idx, s);
        case IK_DTYPE_BF16: return run<__nv_bfloat16, false>(a, idx, s);
        case IK_DTYPE_F16: return run<__half, false>(a, idx, s);
        default: return IK_ERR_DTYPE;
    }
}
