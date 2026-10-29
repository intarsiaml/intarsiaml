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

#include <stdlib.h>

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
    static __device__ __forceinline__ V load_cs(const ik_bf16* p) { return __ldcs(reinterpret_cast<const uint4*>(p)); }
    static __device__ __forceinline__ void unpack(const V& v, float* f) { ik_unpack8(v, f); }
};
template <>
struct KvLoad<uint8_t> {
    using V = uint2;
    static __device__ __forceinline__ V zero() { return make_uint2(0, 0); }
    static __device__ __forceinline__ V load(const uint8_t* p) { return __ldg(reinterpret_cast<const uint2*>(p)); }
    static __device__ __forceinline__ V load_cs(const uint8_t* p) { return __ldcs(reinterpret_cast<const uint2*>(p)); }
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

// splitkv_rows_persist_kernel's stage-1 pieces: splitkv_kernel's iteration body and lane merge, the same statements in the
// same order (splitkv_kernel keeps its own copy, so its code is unchanged). tests/row_tables.rs checks the persistent
// kernel against the solo launch bitwise.

// one iteration: the KB keys of each key lane (K / V raw bits in kk / vv, ok = in range) into the lane's (m, l, acc)
template <int GP, int TPK, int KB, typename KT>
__device__ __forceinline__ void splitkv_step(const typename KvLoad<KT>::V* kk, const typename KvLoad<KT>::V* vv, const bool* ok,
                                             const float (&qr)[GP][8], float (&m)[GP], float (&l)[GP], float (&acc)[GP][8]) {
    using L = KvLoad<KT>;
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
template <int GP, int TPK>
__device__ __forceinline__ void splitkv_merge(const float (&m)[GP], const float (&l)[GP], const float (&acc)[GP][8],
                                              float* __restrict__ pacc, float2* __restrict__ pml, int N, int G, int nsplit, int sp,
                                              int kvh, int t, float* s_acc, float* s_m, float* s_l) {
    constexpr int D = TPK * 8;
    constexpr int NG = kBlock / TPK;
    const int c = threadIdx.x % TPK, gk = threadIdx.x / TPK;
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

// the item's keys: [sp * chunk, sp * chunk + chunk) clipped to [lo, visible)
__device__ __forceinline__ void splitkv_range(int sp, int64_t chunk, int64_t lo, int64_t visible, int64_t& start, int64_t& end) {
    start = (int64_t)sp * chunk;
    end = start + chunk;
    if (start < lo) start = lo;
    if (end > visible) end = visible;
}

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

// Row-table decode on a fixed grid (INTARSIA_ATTN_ROWS_PERSIST, default on), for groups of 5..8 query heads (KB = 2; the
// 35B's 8): the blocks of splitkv_kernel<ROWS>'s grid that do work, i.e. the items (row, kv head, split < the row's split
// count), numbered row by row, then kv head, then split, dealt round-robin to a grid of (SMs x resident blocks). Each item
// runs splitkv_step / splitkv_merge on the keys, heads and partial slot its block of the full grid has, with the same q
// values and a zero for every key past the item's end, so every partial is bitwise that block's.
// What changes is the schedule. The full grid launches 64 splits per row and kv head (the length-independent maximum a CUDA
// graph needs) of which about 18 work at 1k keys, and each of its blocks waits on a position, a row-table entry, q and its
// first K / V before it streams its 4 iterations. Here a block streams its items' iterations back to back: the K / V of each
// iteration are loaded one iteration ahead and the last iteration of an item loads the next item's first, the next item's
// row-table entry is read during this one, positions live in shared memory, q is read 16 B at a time, and K / V are loaded
// as streaming (`__ldcs`, evict first) so they do not push the partials out of L2 before the combine reads them.
// KT = uint8_t (fp8_e4m3 cache): the same schedule with 8-byte K / V loads and splitkv_step's e4m3 unpack, so every partial is
// bitwise splitkv_kernel<8, TPK, 2, uint8_t, true>'s block, as the bf16 instantiation is the bf16 one's.
template <int TPK, typename KT = ik_bf16>
__global__ void __launch_bounds__(kBlock)
splitkv_rows_persist_kernel(const ik_bf16* __restrict__ q, float* __restrict__ pacc, float2* __restrict__ pml, int N, int G,
                            int NKV, int64_t cap, float qscale, int nsplit, int R, const int32_t* __restrict__ pos_dev,
                            const uint8_t* __restrict__ tab, int64_t tstride, int64_t tindex) {
    constexpr int GP = kMaxGroup, KB = 2;  // dispatch_group's (8, 2) case
    constexpr int D = TPK * 8;
    constexpr int NG = kBlock / TPK;
    using L = KvLoad<KT>;
    __shared__ __align__(16) float s_acc[NG * D];
    __shared__ float s_m[NG], s_l[NG];
    __shared__ int s_first[kMaxRows + 1];  // first item of row r; s_first[R] = the item count
    __shared__ int s_pos[kMaxRows];
    if ((int)threadIdx.x < R) {
        const int p = pos_dev[threadIdx.x];
        s_pos[threadIdx.x] = p;
        s_first[threadIdx.x + 1] = splitkv_count((int64_t)p + 1, NKV, 1) * NKV;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        s_first[0] = 0;
        for (int r = 0; r < R; ++r) s_first[r + 1] += s_first[r];
    }
    __syncthreads();
    const int items = s_first[R];
    const int c = threadIdx.x % TPK, gk = threadIdx.x / TPK;

    struct Item {
        int t, kvh, sp;
        int64_t start, end;
        const KT *kb, *vb;  // this thread's 8 dims of key 0 of the kv head
    };
    // item `it`, `t` the row of the block's previous item (a block's items increase, so its row only moves forward)
    auto describe = [&](int it, int& t, Item& d) {
        while (it >= s_first[t + 1]) ++t;
        const int64_t kv_len = (int64_t)s_pos[t] + 1;
        const int nsplit_eff = splitkv_count(kv_len, NKV, 1);
        const int local = it - s_first[t];
        d.t = t;
        d.sp = local % nsplit_eff;
        d.kvh = local / nsplit_eff;
        splitkv_range(d.sp, splitkv_chunk(kv_len, nsplit_eff), 0, kv_len, d.start, d.end);
        const uint64_t* tp = reinterpret_cast<const uint64_t*>(tab) + (int64_t)t * tstride + tindex;
        d.kb = reinterpret_cast<const KT*>(tp[0]) + (int64_t)d.kvh * cap * D + c * 8;
        d.vb = reinterpret_cast<const KT*>(tp[1]) + (int64_t)d.kvh * cap * D + c * 8;
    };
    // the K / V of the iteration at `base` of an item (zero past its end, as splitkv_item)
    auto load = [&](const Item& d, int64_t base, typename L::V* nk, typename L::V* nv) {
#pragma unroll
        for (int u = 0; u < KB; ++u) {
            const int64_t j = base + (int64_t)u * NG + gk;
            nk[u] = L::zero();
            nv[u] = L::zero();
            if (j < d.end) {
                nk[u] = L::load_cs(d.kb + j * D);
                nv[u] = L::load_cs(d.vb + j * D);
            }
        }
    };

    int it = blockIdx.x, t = 0;
    if (it >= items) return;
    Item cur;
    describe(it, t, cur);
    typename L::V nk[KB], nv[KB];  // the K / V of the next iteration
    load(cur, cur.start, nk, nv);
    while (true) {
        const bool more = it + (int)gridDim.x < items;
        Item nxt = cur;
        if (more) describe(it + (int)gridDim.x, t, nxt);

        float qr[GP][8];
#pragma unroll
        for (int g = 0; g < GP; ++g) {
#pragma unroll
            for (int e = 0; e < 8; ++e) qr[g][e] = 0.f;
            if (g < G) {
                const uint4 w = __ldg(reinterpret_cast<const uint4*>(q + ((int64_t)cur.t * N + (int64_t)cur.kvh * G + g) * D + c * 8));
                float f[8];
                ik_unpack8(w, f);  // exact, as ik_b2f per element
#pragma unroll
                for (int e = 0; e < 8; ++e) qr[g][e] = f[e] * qscale;
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

        bool fetched = false;  // whether an iteration of this item has loaded the next item's first K / V
        for (int64_t base = cur.start; base < cur.end; base += (int64_t)NG * KB) {
            typename L::V kk[KB], vv[KB];
            bool ok[KB];
#pragma unroll
            for (int u = 0; u < KB; ++u) {
                kk[u] = nk[u];
                vv[u] = nv[u];
                ok[u] = base + (int64_t)u * NG + gk < cur.end;
            }
            const int64_t nb = base + (int64_t)NG * KB;
            if (nb < cur.end) {
                load(cur, nb, nk, nv);
            } else if (more) {
                load(nxt, nxt.start, nk, nv);
                fetched = true;
            }
            splitkv_step<GP, TPK, KB, KT>(kk, vv, ok, qr, m, l, acc);
        }
        if (more && !fetched) load(nxt, nxt.start, nk, nv);  // an item without keys loaded nothing
        splitkv_merge<GP, TPK>(m, l, acc, pacc, pml, N, G, nsplit, cur.sp, cur.kvh, cur.t, s_acc, s_m, s_l);
        if (!more) break;
        cur = nxt;
        it += (int)gridDim.x;
    }
}

template <int TPK, typename KT>
int launch_rows_persist(const ik_bf16* q, float* pacc, float2* pml, int64_t R, int64_t N, int64_t NKV, int64_t cap, float qscale,
                        int nsplit, const int32_t* pos_dev, cudaStream_t s, const uint8_t* tab, int64_t tstride, int64_t tindex) {
    auto kern = splitkv_rows_persist_kernel<TPK, KT>;
    static int grid = 0;  // SMs x resident blocks, once per process
    if (!grid) {
        int occ = 0, sms = 0, d = 0;
        cudaGetDevice(&d);
        if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, kern, kBlock, 0) != cudaSuccess || occ < 1) occ = 1;
        if (cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, d) != cudaSuccess || sms < 1) sms = 48;
        grid = sms * occ;
    }
    kern<<<grid, kBlock, 0, s>>>(q, pacc, pml, (int)N, (int)(N / NKV), (int)NKV, cap, qscale, nsplit, (int)R, pos_dev, tab, tstride,
                                 tindex);
    return IK_OK;
}

// ROWS: `T` rows are sequences of one query row each (stage 1's ROWS), row `t * N + h` belongs to sequence `t = row / N`.
// DISCARD (after splitkv_rows_persist_kernel; the partials 128-byte aligned): once the block has read its row's partials it
// drops their L2 lines without writing them back (`discard.global.L2`); they are dead, and at B = 16 they are 4.7 MB per
// layer that would otherwise go to DRAM and back. The values read are the same.
template <int D, bool ROWS = false, bool DISCARD = false>
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
    if constexpr (DISCARD) {
        __syncthreads();  // every thread has read its column
        const int lines = nsplit_eff * D * 4 / 128;
        for (int i = threadIdx.x; i < lines; i += D)
            asm volatile("discard.global.L2 [%0], 128;" ::"l"(reinterpret_cast<const char*>(acc) + (int64_t)i * 128) : "memory");
    }
}

// D = 512 (Gemma 4 global layers). A key row is spanned by one warp, 16 dims per thread (two 16 B loads, dims c*8 .. +8 and
// 256 + c*8 .. +8), so a score is a warp shuffle reduction; the block has NG = 8 key lanes. The 8 query heads of a kv head are
// served in passes of kGP512 = 4 (grid.y = NKV * npass), so the accumulators stay in registers; the second pass re-reads
// K / V from L2. Everything else (split policy, window, lane merge, partial layout, combine_kernel<512>) is splitkv_kernel's.
constexpr int kGP512 = 4;
constexpr int kD512 = 512;
constexpr int kSplit512KB = 1;  // keys per key lane per iteration (registers: 2 x 16 B K and V loads each, 128 accumulators)

template <int KB>
__global__ void __launch_bounds__(kBlock)
splitkv512_kernel(const ik_bf16* __restrict__ q, const ik_bf16* __restrict__ kc, const ik_bf16* __restrict__ vc,
                  float* __restrict__ pacc, float2* __restrict__ pml, int N, int G, int npass, int64_t cap,
                  int64_t pos0_host, float qscale, int nsplit, int64_t window, const int32_t* pos_dev) {
    constexpr int D = kD512;
    constexpr int NG = kBlock / 32;
    const int nkv = gridDim.y / npass;
    const int64_t tq = gridDim.z;
    const int64_t pos0 = pos_dev ? (int64_t)*pos_dev : pos0_host;
    const int64_t kv_len = pos0 + tq;
    const int nsplit_eff = splitkv_count(kv_len, nkv, tq);
    const int64_t chunk = splitkv_chunk(kv_len, nsplit_eff);
    if ((int)blockIdx.x >= nsplit_eff) return;
    __shared__ __align__(16) float s_acc[NG * D];
    __shared__ float s_m[NG], s_l[NG];

    const int sp = blockIdx.x, kvh = blockIdx.y / npass, g0 = (blockIdx.y % npass) * kGP512, t = blockIdx.z;
    const int c = threadIdx.x % 32, gk = threadIdx.x / 32;
    const int64_t visible = pos0 + t + 1;
    const int64_t lo = (window > 0 && visible > window) ? visible - window : 0;
    int64_t start = (int64_t)sp * chunk;
    int64_t end = start + chunk;
    if (start < lo) start = lo;
    if (end > visible) end = visible;

    const ik_bf16* kb = kc + (int64_t)kvh * cap * D + c * 8;
    const ik_bf16* vb = vc + (int64_t)kvh * cap * D + c * 8;

    float qr[kGP512][16];
#pragma unroll
    for (int g = 0; g < kGP512; ++g) {
#pragma unroll
        for (int e = 0; e < 16; ++e) {
            qr[g][e] = 0.f;
            if (g0 + g < G)
                qr[g][e] = ik_b2f(q[((int64_t)t * N + (int64_t)kvh * G + g0 + g) * D + (e / 8) * 256 + c * 8 + (e % 8)]) * qscale;
        }
    }
    float m[kGP512], l[kGP512], acc[kGP512][16];
#pragma unroll
    for (int g = 0; g < kGP512; ++g) {
        m[g] = -INFINITY;
        l[g] = 0.f;
#pragma unroll
        for (int e = 0; e < 16; ++e) acc[g][e] = 0.f;
    }

    for (int64_t base = start; base < end; base += (int64_t)NG * KB) {
        uint4 kk[KB][2], vv[KB][2];
        bool ok[KB];
#pragma unroll
        for (int u = 0; u < KB; ++u) {
            const int64_t j = base + (int64_t)u * NG + gk;
            ok[u] = j < end;
            kk[u][0] = kk[u][1] = vv[u][0] = vv[u][1] = make_uint4(0, 0, 0, 0);
            if (ok[u]) {
                kk[u][0] = __ldg(reinterpret_cast<const uint4*>(kb + j * D));
                kk[u][1] = __ldg(reinterpret_cast<const uint4*>(kb + j * D + 256));
                vv[u][0] = __ldg(reinterpret_cast<const uint4*>(vb + j * D));
                vv[u][1] = __ldg(reinterpret_cast<const uint4*>(vb + j * D + 256));
            }
        }
        float sc[KB][kGP512];
#pragma unroll
        for (int u = 0; u < KB; ++u) {
            float kf[16];
            ik_unpack8(kk[u][0], kf);
            ik_unpack8(kk[u][1], kf + 8);
#pragma unroll
            for (int g = 0; g < kGP512; ++g) {
                float d = 0.f;
#pragma unroll
                for (int e = 0; e < 16; ++e) d += qr[g][e] * kf[e];
#pragma unroll
                for (int o = 16; o > 0; o >>= 1) d += __shfl_xor_sync(0xffffffffu, d, o);
                sc[u][g] = ok[u] ? d : -INFINITY;
            }
        }
        float vf[KB][16];
#pragma unroll
        for (int u = 0; u < KB; ++u) {
            ik_unpack8(vv[u][0], vf[u]);
            ik_unpack8(vv[u][1], vf[u] + 8);
        }
#pragma unroll
        for (int g = 0; g < kGP512; ++g) {
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
            for (int e = 0; e < 16; ++e) {
                float a = acc[g][e] * corr;
#pragma unroll
                for (int u = 0; u < KB; ++u) a += p[u] * vf[u][e];
                acc[g][e] = a;
            }
        }
    }

    // merge the NG key lanes, one head at a time; write the block's partial for that head
#pragma unroll
    for (int g = 0; g < kGP512; ++g) {
        if (g0 + g < G) {
            __syncthreads();  // s_acc / s_m / s_l are reused for every head
#pragma unroll
            for (int e = 0; e < 16; ++e) s_acc[gk * D + (e / 8) * 256 + c * 8 + (e % 8)] = acc[g][e];
            if (c == 0) {
                s_m[gk] = m[g];
                s_l[gk] = l[g];
            }
            __syncthreads();
            const int64_t slot = ((int64_t)t * N + (int64_t)kvh * G + g0 + g) * nsplit + sp;
            float M = -INFINITY;
            for (int i = 0; i < NG; ++i) M = fmaxf(M, s_m[i]);
            for (int d = threadIdx.x; d < D; d += kBlock) {
                float A = 0.f, L = 0.f;
                if (M != -INFINITY) {
                    for (int i = 0; i < NG; ++i) {
                        const float w = exp2f(s_m[i] - M);  // empty lane: exp2(-inf) = 0, acc is 0 too
                        L += s_l[i] * w;
                        A += s_acc[i * D + d] * w;
                    }
                }
                pacc[slot * D + d] = A;
                if (d == 0) pml[slot] = make_float2(M, L);
            }
        }
    }
}

template <int KB>
void launch_split512(const ik_bf16* q, const ik_bf16* kc, const ik_bf16* vc, float* pacc, float2* pml, int64_t T, int64_t N,
                     int64_t NKV, int64_t cap, int64_t pos0, float qscale, int nsplit, int64_t window,
                     const int32_t* pos_dev, cudaStream_t s) {
    const int G = (int)(N / NKV);
    const int npass = (G + kGP512 - 1) / kGP512;
    dim3 grid(nsplit, (unsigned)(NKV * npass), (unsigned)T);
    splitkv512_kernel<KB><<<grid, kBlock, 0, s>>>(q, kc, vc, pacc, pml, (int)N, G, npass, cap, pos0, qscale, nsplit, window,
                                                  pos_dev);
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

// INTARSIA_ATTN_ROWS_PERSIST: 1 (the default) runs the row-table decode launch for groups of 5..8 heads as
// splitkv_rows_persist_kernel, 0 as the full grid of splitkv_kernel<ROWS>. Read once per process; the bits do not depend on it.
inline int rows_persist() {
    static const int v = [] {
        const char* e = getenv("INTARSIA_ATTN_ROWS_PERSIST");
        return (e && e[0] == '0' && !e[1]) ? 0 : 1;
    }();
    return v;
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
    if (D != 32 && D != 64 && D != 128 && D != 256 && !(D == 512 && !f8kv)) return IK_ERR_SHAPE;
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
        case 512:
            launch_split512<kSplit512KB>(qp, kp, vp, pacc, pml, T, N, NKV, cap, pos0, qscale, nsplit, window, pos_dev, s);
            rc = IK_OK;
            break;
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
        case 512: combine_kernel<512><<<rows, 512, 0, s>>>(pacc, pml, op, nsplit, (int)T, (int)NKV, pos0, pos_dev); break;
    }
    return ik_launch_status();
}

// stage 1 of the row-table decode launch over a KT cache (bf16 or e4m3 bytes): the persistent kernel or the full ROWS grid
template <typename KT>
int rows_stage1(bool persist, const ik_bf16* qp, const ik_tensor* kc0, const ik_tensor* vc0, float* pacc, float2* pml, int64_t R,
                int64_t N, int64_t NKV, int64_t cap, int64_t D, float qscale, int nsplit, const int32_t* pos_dev, cudaStream_t s,
                const uint8_t* tp, int64_t stride, int64_t index) {
    const KT* kp = (const KT*)kc0->data;
    const KT* vp = (const KT*)vc0->data;
    int rc = IK_ERR_SHAPE;
    if (persist) {
        switch (D) {
            case 32: rc = launch_rows_persist<4, KT>(qp, pacc, pml, R, N, NKV, cap, qscale, nsplit, pos_dev, s, tp, stride, index); break;
            case 64: rc = launch_rows_persist<8, KT>(qp, pacc, pml, R, N, NKV, cap, qscale, nsplit, pos_dev, s, tp, stride, index); break;
            case 128: rc = launch_rows_persist<16, KT>(qp, pacc, pml, R, N, NKV, cap, qscale, nsplit, pos_dev, s, tp, stride, index); break;
            case 256: rc = launch_rows_persist<32, KT>(qp, pacc, pml, R, N, NKV, cap, qscale, nsplit, pos_dev, s, tp, stride, index); break;
        }
    } else {
#define IK_ROWS_DISPATCH(DV, TPKV)                                                                                          \
    case DV: rc = dispatch_group<TPKV, KT, true>((int)(N / NKV), qp, kp, vp, pacc, pml, R, N, NKV, cap, 0, qscale, nsplit,  \
                                                 0, pos_dev, s, tp, stride, index); break;
        switch (D) {
            IK_ROWS_DISPATCH(32, 4)
            IK_ROWS_DISPATCH(64, 8)
            IK_ROWS_DISPATCH(128, 16)
            IK_ROWS_DISPATCH(256, 32)
        }
#undef IK_ROWS_DISPATCH
    }
    return rc;
}

// Row-table decode (`OpCall::rows`): `rows` sequences of one query row each (q [rows, N, D]); `kc0` / `vc0` are row 0's caches
// ([NKV, cap, D], shape carriers); sequence r's caches are the buffers at entries [r][index] and [r][index + 1] of `tab`,
// its position is pos[r]. Same grid policy as the one-row decode launch (the maximum split count), `rows` rows of scratch.
int splitkv_launch_rows(const ik_tensor* q, const ik_tensor* kc0, const ik_tensor* vc0, float scale, ik_tensor* out,
                        const ik_scratch* scratch, const ik_tensor* tab, int64_t stride, int64_t index,
                        const ik_tensor* pos, void* stream, bool f8kv) {
    IK_TRY(ik_check(q, IK_DTYPE_BF16, 3));
    const ik_dtype kvt = f8kv ? IK_DTYPE_FP8_E4M3 : IK_DTYPE_BF16;
    IK_TRY(ik_check(kc0, kvt, 3));
    IK_TRY(ik_check(vc0, kvt, 3));
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
    const uint8_t* tp = (const uint8_t*)tab->data;
    const float qscale = scale * kLog2e;
    // groups of 5..8 heads (KB = 2), the 35B's 8: the fixed-grid kernel; the combine drops the partials from L2 after it
    const bool persist = rows_persist() && N / NKV > 4 && ik_aligned16(q->data) && !(reinterpret_cast<uintptr_t>(pacc) & 127u);
    const int rc = f8kv ? rows_stage1<uint8_t>(persist, qp, kc0, vc0, pacc, pml, R, N, NKV, cap, D, qscale, nsplit, pos_dev, s, tp,
                                               stride, index)
                        : rows_stage1<ik_bf16>(persist, qp, kc0, vc0, pacc, pml, R, N, NKV, cap, D, qscale, nsplit, pos_dev, s, tp,
                                               stride, index);
    if (rc != IK_OK) return rc;
    IK_TRY(ik_launch_status());
    ik_bf16* op = (ik_bf16*)out->data;
    const unsigned rows = (unsigned)(R * N);
    switch (D) {
#define IK_ROWS_COMBINE(DV)                                                                                                       \
    case DV:                                                                                                                      \
        if (persist)                                                                                                              \
            combine_kernel<DV, true, true><<<rows, DV, 0, s>>>(pacc, pml, op, nsplit, (int)R, (int)NKV, 0, pos_dev, (int)N);       \
        else                                                                                                                      \
            combine_kernel<DV, true><<<rows, DV, 0, s>>>(pacc, pml, op, nsplit, (int)R, (int)NKV, 0, pos_dev, (int)N);             \
        break;
        IK_ROWS_COMBINE(32)
        IK_ROWS_COMBINE(64)
        IK_ROWS_COMBINE(128)
        IK_ROWS_COMBINE(256)
#undef IK_ROWS_COMBINE
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
    return splitkv_launch_rows(q, kc0, vc0, scale, out, scratch, tab, stride, index, pos, stream, false);
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

// the row-table decode launch over fp8_e4m3 caches: row r bitwise ik_attn_decode_f8kv_splitkv on row r alone
int ik_attn_decode_f8kv_splitkv_rows(const ik_tensor* q, const ik_tensor* kc0, const ik_tensor* vc0, float scale,
                                     ik_tensor* out, const ik_scratch* scratch, const ik_tensor* tab, int64_t stride,
                                     int64_t index, const ik_tensor* pos, void* stream) {
    return splitkv_launch_rows(q, kc0, vc0, scale, out, scratch, tab, stride, index, pos, stream, true);
}

}  // extern "C"
