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
#include <limits.h>
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
template <> __device__ __forceinline__ void load8<__nv_bfloat16>(const __nv_bfloat16* p, float* f) { ik_unpack8(__ldg(reinterpret_cast<const uint4*>(p)), f); }
template <> __device__ __forceinline__ void load8<__half>(const __half* p, float* f) {
    const uint4 u = __ldg(reinterpret_cast<const uint4*>(p));
    const __half2* h = reinterpret_cast<const __half2*>(&u);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float2 t = __half22float2(h[i]);
        f[2 * i] = t.x;
        f[2 * i + 1] = t.y;
    }
}
template <> __device__ __forceinline__ void load8<float>(const float* p, float* f) {
    const float4 a = __ldg(reinterpret_cast<const float4*>(p)), b = __ldg(reinterpret_cast<const float4*>(p) + 1);
    f[0] = a.x, f[1] = a.y, f[2] = a.z, f[3] = a.w, f[4] = b.x, f[5] = b.y, f[6] = b.z, f[7] = b.w;
}

// Device position (`pos`): the call's keys are a capacity view k [Ncap, Di]; the kernels read P when they run and take
// N = min((P + T) / ratio, Ncap) rows (host call: P = pos0, N = Ncap = Nk). Strides stay the capacity's.
struct DevLen {
    const int* dpos;  // null: host position
    int64_t Tn;       // rows of the whole chunk
};

IK_DEVI void dev_len(const DevLen& d, int64_t ratio, int64_t Ncap, int64_t& pos0, int64_t& N) {
    if (!d.dpos) {
        N = Ncap;
        return;
    }
    pos0 = *d.dpos;
    N = (pos0 + d.Tn) / ratio;
    if (N > Ncap) N = Ncap;
}

// rows of query t that are visible (at most N)
IK_DEVI int64_t visible(int64_t pos0, int64_t t, int64_t ratio, int64_t N) {
    const int64_t nt = (pos0 + t + 1) / ratio;
    return nt < N ? nt : N;
}

// BLOCKS: write block maxima (row length nb) instead of the score row (row length Nk). cs: the candidate row stride, os: the
// output row stride.
template <typename T, bool BLOCKS>
__global__ void __launch_bounds__(kScoreThreads) index_score(const T* __restrict__ q, const T* __restrict__ k, const T* __restrict__ w,
                                                             const unsigned char* __restrict__ cand, int64_t t0, int64_t Hi, int64_t Di,
                                                             int64_t Nk, int64_t B, int64_t ratio, int64_t pos0, float scale,
                                                             float* __restrict__ out, DevLen dl, int64_t cs, int64_t os) {
    __shared__ __align__(16) float qs[kMaxQ];
    __shared__ float ws[kMaxHi];
    const int64_t tl = blockIdx.y, t = t0 + tl;
    dev_len(dl, ratio, Nk, pos0, Nk);
    if ((int64_t)blockIdx.x * kScoreThreads >= Nk) return;  // past the device count (block-uniform)
    for (int64_t i = threadIdx.x; i < Hi * Di; i += kScoreThreads) qs[i] = ld(q, t * Hi * Di + i);
    if (threadIdx.x < Hi) ws[threadIdx.x] = ld(w, t * Hi + threadIdx.x);
    __syncthreads();
    const int64_t c = (int64_t)blockIdx.x * kScoreThreads + threadIdx.x;
    const int64_t nt = visible(pos0, t, ratio, Nk);
    bool ok = c < Nk && c < nt;
    if (ok && cand) ok = cand[t * cs + c / B] != 0;
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
        if ((c % B) == 0 && c < Nk) out[tl * os + c / B] = m;
    } else if (c < Nk) {
        out[tl * os + c] = score;
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
// The k best of N keys (key descending, ties to the lower position; positions are 0..N-1 in the order the caller wants the
// picks emitted), then one ordered pass calling emit(i, o) for every pick i with its rank o among the picks in position
// order. One block of kSelThreads; key_at(i) is an order-preserving u32 key (fkey). k <= N.
template <typename KeyAt, typename Emit>
IK_DEVI void select_core(int64_t N, int64_t k, KeyAt key_at, Emit emit) {
    __shared__ unsigned hist[256];
    __shared__ unsigned wsum[32];
    __shared__ unsigned s_digit, s_rem;
    __shared__ unsigned s_cgt, s_ceq;
    __shared__ int s_done;
    unsigned prefix = 0, pmask = 0, rem = (unsigned)k;
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    for (int pass = 0; pass < 4; ++pass) {
        const int shift = 24 - 8 * pass;
        if (threadIdx.x < 256) hist[threadIdx.x] = 0;
        __syncthreads();
        for (int64_t i = threadIdx.x; i < N; i += kSelThreads) {
            const unsigned key = key_at(i);
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
            const unsigned key = key_at(i);
            gt = key > kth;
            eq = key == kth;
        }
        unsigned total;
        const unsigned inc = block_scan(gt | (eq << 16), wsum, total);
        const unsigned exc = inc - (gt | (eq << 16));
        const unsigned cgt = s_cgt, ceq = s_ceq;
        const unsigned gb = cgt + (exc & 0xFFFFu), eb = ceq + (exc >> 16);
        if (gt || (eq && eb < need_eq)) emit(i, gb + min(eb, need_eq));
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

// One row of `v` (row length N): the k best (value descending, ties to the lower index), emitted in index order.
// BLOCKS: mask row (u8, `os` wide: the first N entries are the blocks, the rest 0) of the picks whose value is > -inf, after
// pinning the query's newest visible block; else idx row (`os` wide: k picks, index >= nt mapped to -1, then -1 fill).
template <bool BLOCKS>
IK_DEVI void select_row(float* row, int64_t N, int64_t k, int64_t t, int64_t nt, int64_t B, int* __restrict__ idx, unsigned char* __restrict__ mask,
                        int64_t os) {
    if (BLOCKS) {
        unsigned char* mrow = mask + t * os;
        for (int64_t i = threadIdx.x; i < os; i += kSelThreads) mrow[i] = 0;
        if (threadIdx.x == 0 && nt > 0) row[(nt - 1) / B] = INFINITY;
        __syncthreads();
    } else {
        for (int64_t o = k + threadIdx.x; o < os; o += kSelThreads) idx[t * os + o] = -1;
    }
    if (k <= 0) return;
    select_core(
        N, k, [&](int64_t i) { return fkey(row[i]); },
        [&](int64_t i, unsigned o) {
            if (BLOCKS) {
                if (row[i] > -INFINITY) mask[t * os + i] = 1;
            } else {
                idx[t * os + o] = i < nt ? (int)i : -1;
            }
        });
}

// v rows of `len` (the capacity's: Nk or ceil(Nk / B)); keep: top_k / blocks kept; os: the output row stride
template <bool BLOCKS>
__global__ void __launch_bounds__(kSelThreads) index_select(float* __restrict__ v, int64_t len, int64_t keep, int64_t t0, int64_t B,
                                                            int64_t ratio, int64_t pos0, int* __restrict__ idx, unsigned char* __restrict__ mask,
                                                            DevLen dl, int64_t Nk, int64_t os) {
    const int64_t tl = blockIdx.x, t = t0 + tl;
    int64_t N;
    dev_len(dl, ratio, Nk, pos0, N);
    const int64_t n = BLOCKS ? cdiv(N, B) : N;
    select_row<BLOCKS>(v + tl * len, n, keep < n ? keep : n, t, visible(pos0, t, ratio, N), B, idx, mask, os);
}

// ---- multi-block threshold select (indexer_topk, few query rows, long rows) ---------------------------------------------
// A one-block select of a 128K row runs on one SM. For T <= kFastRows rows of N >= kFastMinN: (1) a global histogram of the
// keys' top 11 bits over many blocks, (2) per row the bin holding the k-th best key, (3) compaction of every key in that bin
// or above (at most kCap; otherwise the row falls back to the one-block select), (4) per row an exact select over the
// candidates sorted by position. Same picks, same order as select_row.
constexpr int kHistBins = 2048;
constexpr int kHistShift = 21;
constexpr int64_t kSeg = 4096;
constexpr int kCap = 4096;
constexpr int64_t kFastRows = 16;
constexpr int64_t kFastMinN = 16384;

struct SelMeta {
    unsigned bin, cnt, flag, ctr;
};

// v rows of `len` (= Nk, the capacity); the segments and k are bounded by the device count N
__global__ void __launch_bounds__(256) sel_hist(const float* __restrict__ v, int64_t len, unsigned* __restrict__ hist, DevLen dl,
                                                int64_t ratio) {
    __shared__ unsigned h[kHistBins];
    const int64_t tl = blockIdx.y;
    int64_t p0 = 0, N;
    dev_len(dl, ratio, len, p0, N);
    if ((int64_t)blockIdx.x * kSeg >= N) return;
    for (int b = threadIdx.x; b < kHistBins; b += 256) h[b] = 0;
    __syncthreads();
    const float* row = v + tl * len;
    const int64_t lo = (int64_t)blockIdx.x * kSeg, hi = min(lo + kSeg, N);
    for (int64_t i = lo + threadIdx.x; i < hi; i += 256) atomicAdd(&h[fkey(row[i]) >> kHistShift], 1u);
    __syncthreads();
    for (int b = threadIdx.x; b < kHistBins; b += 256)
        if (h[b]) atomicAdd(&hist[tl * kHistBins + b], h[b]);
}

__global__ void __launch_bounds__(kSelThreads) sel_thresh(const unsigned* __restrict__ hist, int64_t k, SelMeta* __restrict__ meta, DevLen dl,
                                                          int64_t ratio, int64_t len) {
    __shared__ unsigned wsum[32];
    const int64_t tl = blockIdx.x;
    int64_t p0 = 0, N;
    dev_len(dl, ratio, len, p0, N);
    if (k > N) k = N;
    if (k <= 0) return;  // no picks (N = 0): sel_compact / sel_final skip the row
    const unsigned* h = hist + tl * kHistBins;
    // thread i owns bins 2047-2i and 2046-2i (descending); kSelThreads * 2 == kHistBins
    const int b0 = kHistBins - 1 - 2 * threadIdx.x;
    const unsigned c0 = h[b0], c1 = h[b0 - 1], c = c0 + c1;
    unsigned total;
    const unsigned inc = block_scan(c, wsum, total);
    const unsigned exc = inc - c;
    const unsigned kk = (unsigned)k;
    if (exc < kk && inc >= kk) {
        SelMeta m;
        if (exc + c0 >= kk) {
            m.bin = b0;
            m.cnt = exc + c0;
        } else {
            m.bin = b0 - 1;
            m.cnt = inc;
        }
        m.flag = m.cnt > (unsigned)kCap;
        m.ctr = 0;
        meta[tl] = m;
    }
}

__global__ void __launch_bounds__(256) sel_compact(const float* __restrict__ v, int64_t len, SelMeta* __restrict__ meta, uint2* __restrict__ cand,
                                                   DevLen dl, int64_t ratio) {
    const int64_t tl = blockIdx.y;
    int64_t p0 = 0, N;
    dev_len(dl, ratio, len, p0, N);
    if (N <= 0 || (int64_t)blockIdx.x * kSeg >= N) return;
    if (meta[tl].flag) return;
    const unsigned bin = meta[tl].bin;
    const float* row = v + tl * len;
    const int64_t lo = (int64_t)blockIdx.x * kSeg, hi = min(lo + kSeg, N);
    const int lane = threadIdx.x & 31;
    for (int64_t base = lo; base < hi; base += 256) {
        const int64_t i = base + threadIdx.x;
        unsigned key = 0;
        bool take = false;
        if (i < hi) {
            key = fkey(row[i]);
            take = (key >> kHistShift) >= bin;
        }
        // warp-aggregated append
        const unsigned m = __ballot_sync(0xffffffffu, take);
        if (m) {
            unsigned pos0 = 0;
            const int leader = __ffs(m) - 1;
            if (lane == leader) pos0 = atomicAdd(&meta[tl].ctr, (unsigned)__popc(m));
            pos0 = __shfl_sync(0xffffffffu, pos0, leader);
            if (take) cand[tl * kCap + pos0 + __popc(m & ((1u << lane) - 1))] = make_uint2(key, (unsigned)i);
        }
    }
}

__global__ void __launch_bounds__(kSelThreads) sel_final(float* __restrict__ v, int64_t len, int64_t k, int64_t t0, int64_t ratio, int64_t pos0,
                                                         const SelMeta* __restrict__ meta, const uint2* __restrict__ cand, int* __restrict__ idx,
                                                         DevLen dl, int64_t os) {
    __shared__ unsigned keys[kCap];
    __shared__ int ids[kCap];
    const int64_t tl = blockIdx.x, t = t0 + tl;
    int64_t N;
    dev_len(dl, ratio, len, pos0, N);
    if (k > N) k = N;
    const int64_t nt = visible(pos0, t, ratio, N);
    for (int64_t o = k + threadIdx.x; o < os; o += kSelThreads) idx[t * os + o] = -1;
    if (k <= 0) return;
    const SelMeta m = meta[tl];
    if (m.flag) {
        select_row<false>(v + tl * len, N, k, t, nt, 1, idx, nullptr, os);
        return;
    }
    int P = 2;
    while (P < (int)m.cnt) P <<= 1;
    for (int i = threadIdx.x; i < P; i += kSelThreads) {
        if (i < (int)m.cnt) {
            const uint2 c = cand[tl * kCap + i];
            keys[i] = c.x;
            ids[i] = (int)c.y;
        } else {
            keys[i] = 0;  // below every real key (fkey(-inf) > 0): never picked
            ids[i] = INT_MAX;
        }
    }
    __syncthreads();
    // bitonic sort by position (positions are unique)
    for (int size = 2; size <= P; size <<= 1) {
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
            for (int i = threadIdx.x; i < P; i += kSelThreads) {
                const int j = i ^ stride;
                if (j > i) {
                    const bool up = (i & size) == 0;
                    if ((ids[i] > ids[j]) == up) {
                        const int ti = ids[i];
                        ids[i] = ids[j];
                        ids[j] = ti;
                        const unsigned tk = keys[i];
                        keys[i] = keys[j];
                        keys[j] = tk;
                    }
                }
            }
            __syncthreads();
        }
    }
    select_core(
        P, k, [&](int64_t i) { return keys[i]; },
        [&](int64_t i, unsigned o) {
            const int p = ids[i];
            idx[t * os + o] = p < nt ? p : -1;
        });
}

static bool index_fast_select_enabled() {
    static int on = -1;
    if (on < 0) {
        const char* e = getenv("INTARSIA_DSV41_INDEX_ONEBLOCK_SELECT");
        on = !(e && e[0] == '1');
    }
    return on == 1;
}

/* Bytes the fast select needs past the score rows of `rows` rows (mirrored by `INDEX_FAST_EXTRA_PER_ROW` and
 * `INDEX_FAST_SLACK` in crates/intarsia-backend-cuda/src/dsv41.rs). */
static int64_t fast_extra(int64_t rows) { return rows * ((int64_t)kHistBins * 4 + (int64_t)sizeof(SelMeta) + (int64_t)kCap * 8) + 1024; }

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
                                                                int64_t pos0, float scale, int qb, float* __restrict__ out, DevLen dl, int64_t cs,
                                                                int64_t os) {
    extern __shared__ __align__(16) unsigned char smem_raw[];
    __nv_bfloat16* kt = reinterpret_cast<__nv_bfloat16*>(smem_raw);  // [kMmaKeys][Di + 8]
    const int ldk = Di + 8;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int g = lane >> 2, tq = lane & 3;
    const int64_t c0 = (int64_t)blockIdx.x * kMmaKeys;
    dev_len(dl, ratio, Nk, pos0, Nk);
    if (c0 >= Nk) return;  // past the device count (block-uniform)
    const int64_t q0 = (int64_t)blockIdx.y * qb;  // first query (tile-local) of this block
    const int nq = (int)min((int64_t)qb, tr - q0);
    // rows the block needs: below the last query's visible rows; with one query also inside its candidates
    const int64_t nt_last = (pos0 + t0 + q0 + nq) / ratio;
    for (int v = threadIdx.x; v < kMmaKeys * (Di / 8); v += kMmaWarps * 32) {
        const int r = v / (Di / 8), cv = v % (Di / 8);
        const int64_t c = c0 + r;
        uint4 val = make_uint4(0, 0, 0, 0);
        bool need = c < Nk && c < nt_last;
        if (need && cand && nq == 1) need = cand[(t0 + q0) * cs + c / B] != 0;
        if (need) val = __ldg(reinterpret_cast<const uint4*>(k + c * Di) + cv);
        *reinterpret_cast<uint4*>(kt + r * ldk + cv * 8) = val;
    }
    __syncthreads();
    const int wpq = kMmaWarps / qb;  // warps per query
    const int qi = warp / wpq;
    if (qi >= nq) return;
    const int64_t tl = q0 + qi, t = t0 + tl;
    const int64_t nt = visible(pos0, t, ratio, Nk);
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
        if (any && cand && B == 8) any = cand[t * cs + n0 / 8] != 0;
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
                v0 = v0 && cand[t * cs + col / B] != 0;
                v1 = v1 && cand[t * cs + (col + 1) / B] != 0;
            }
            if (!v0) s0 = -INFINITY;
            if (!v1) s1 = -INFINITY;
        }
        if (BLOCKS) {
            // block maxima: B in {1, 2, 4, 8} columns, lanes tq hold columns 2tq, 2tq+1
            if (B == 1) {
                if (g == 0) {
                    if (col < Nk) out[tl * os + col] = s0;
                    if (col + 1 < Nk) out[tl * os + col + 1] = s1;
                }
            } else {
                float m = fmaxf(s0, s1);
                if (B >= 4) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 1));
                if (B >= 8) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 2));
                if (g == 0 && (col % B) == 0 && col < Nk) out[tl * os + col / B] = m;
            }
        } else if (g == 0) {
            if (col < Nk) out[tl * os + col] = s0;
            if (col + 1 < Nk) out[tl * os + col + 1] = s1;
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
    const int* dpos;  // device position (null: pos0); k is then a capacity view [Ncap, Di]
};

// Every host-side decision (tiles, grids, MMA vs SIMT, fast vs one-block select) depends on the shapes only, never on the
// device position, so a captured graph replays at any P; the kernels bound their work by the device count N.
template <typename T, bool BLOCKS>
int run(const IdxArgs& a, ik_tensor* outp, cudaStream_t s) {
    const int64_t Tn = a.q->shape[0], Hi = a.q->shape[1], Di = a.q->shape[2], Nk = a.k->shape[0];
    const int64_t nb = cdiv(Nk, a.B);
    const int64_t len = BLOCKS ? nb : Nk;
    const int64_t kk = BLOCKS ? (a.keep < nb ? a.keep : nb) : (a.keep < Nk ? a.keep : Nk);
    const int64_t os = outp->shape[1];  // output row stride: nb (mask) or kk (idx)
    const DevLen dl{a.dpos, Tn};
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
    // few long rows: the multi-block threshold select, when the scratch holds its buffers for the whole chunk
    const int64_t sc_bytes = (Tn * len * 4 + 255) / 256 * 256;
    const bool fast = !BLOCKS && index_fast_select_enabled() && Tn <= kFastRows && len >= kFastMinN &&
                      a.scratch->bytes >= sc_bytes + fast_extra(Tn);
    if (fast) rows = Tn;
    unsigned char* fx = (unsigned char*)a.scratch->data + sc_bytes;
    uint2* fcand = (uint2*)fx;
    unsigned* fhist = (unsigned*)(fx + Tn * kCap * 8);
    SelMeta* fmeta = (SelMeta*)(fx + Tn * kCap * 8 + Tn * kHistBins * 4);
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
                                                                   (int)a.B, a.ratio, a.pos0, a.scale, qb, sc, dl, nb, len);
        } else {
            dim3 g((unsigned)cdiv(Nk, kScoreThreads), (unsigned)tr);
            index_score<T, BLOCKS><<<g, kScoreThreads, 0, s>>>((const T*)a.q->data, (const T*)a.k->data, (const T*)a.w->data, cand, t0, Hi,
                                                               Di, Nk, a.B, a.ratio, a.pos0, a.scale, sc, dl, nb, len);
        }
        IK_TRY(ik_launch_status());
        if (fast) {
            // hist and meta are contiguous: one memset clears both (meta's counters start at 0)
            if (cudaMemsetAsync(fhist, 0, (size_t)(tr * kHistBins * 4 + tr * sizeof(SelMeta)), s) != cudaSuccess) return IK_ERR_LAUNCH;
            const dim3 gs((unsigned)cdiv(len, kSeg), (unsigned)tr);
            sel_hist<<<gs, 256, 0, s>>>(sc, len, fhist, dl, a.ratio);
            IK_TRY(ik_launch_status());
            sel_thresh<<<(unsigned)tr, kSelThreads, 0, s>>>(fhist, kk, fmeta, dl, a.ratio, len);
            IK_TRY(ik_launch_status());
            sel_compact<<<gs, 256, 0, s>>>(sc, len, fmeta, fcand, dl, a.ratio);
            IK_TRY(ik_launch_status());
            sel_final<<<(unsigned)tr, kSelThreads, 0, s>>>(sc, len, kk, t0, a.ratio, a.pos0, fmeta, fcand, (int*)outp->data, dl, os);
        } else {
            index_select<BLOCKS><<<(unsigned)tr, kSelThreads, 0, s>>>(sc, len, a.keep, t0, a.B, a.ratio, a.pos0,
                                                                       BLOCKS ? nullptr : (int*)outp->data,
                                                                       BLOCKS ? (unsigned char*)outp->data : nullptr, dl, Nk, os);
        }
        IK_TRY(ik_launch_status());
    }
    return IK_OK;
}

int check_common(const IdxArgs& a) {
    IK_TRY(check_float(a.q, 3));
    IK_TRY(ik_check(a.k, a.q->dtype, 2));
    IK_TRY(ik_check(a.w, a.q->dtype, 2));
    const int64_t Tn = a.q->shape[0], Hi = a.q->shape[1], Di = a.q->shape[2], Nk = a.k->shape[0];
    if (a.ratio < 1 || (!a.dpos && a.pos0 < 0) || a.B < 1 || a.B > 32 || (a.B & (a.B - 1)) != 0 || a.keep < 1) return IK_ERR_SHAPE;
    if (Hi < 1 || Hi > kMaxHi || Hi * Di > kMaxQ || Di % 8 != 0) return IK_ERR_SHAPE;
    if (a.k->shape[1] != Di || a.w->shape[0] != Tn || a.w->shape[1] != Hi) return IK_ERR_SHAPE;
    // host position: exactly the visible rows; device position: any capacity (the kernels take min((P + T) / ratio, Nk))
    if (!a.dpos && Nk != (a.pos0 + Tn) / a.ratio) return IK_ERR_SHAPE;
    if (Nk > 0 && !ik_aligned16(a.k->data)) return IK_ERR_INVALID_ARG;
    return IK_OK;
}

}  // namespace

extern "C" int ik_indexer_blocks(const ik_tensor* q, const ik_tensor* k, const ik_tensor* w, int64_t keep, int64_t block, int64_t ratio,
                                 int64_t pos0, float scale, ik_tensor* mask, const ik_scratch* scratch, const ik_tensor* pos, void* stream) {
    IK_TRY(ik_check_pos(pos));
    IdxArgs a{q, k, w, nullptr, keep, block, ratio, pos0, scale, scratch, pos ? (const int*)pos->data : nullptr};
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
                               const ik_tensor* pos, void* stream) {
    IK_TRY(ik_check_pos(pos));
    IdxArgs a{q, k, w, cand, top_k, block, ratio, pos0, scale, scratch, pos ? (const int*)pos->data : nullptr};
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
