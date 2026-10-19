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
#include "ik_launch.h"

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

template <int GP, int TPK, int KB>
__global__ void __launch_bounds__(kBlock)
splitkv_kernel(const ik_bf16* __restrict__ q, const ik_bf16* __restrict__ kc, const ik_bf16* __restrict__ vc,
               float* __restrict__ pacc, float2* __restrict__ pml, int N, int G, int64_t cap, int64_t pos0_host,
               float qscale, int nsplit, const int32_t* pos_dev) {
    // `nsplit` is the grid's split count and the stride of the partials. The splits actually used follow from the key
    // count alone (device position when given), so the grid may be launched at its maximum (decode, CUDA graphs) and
    // the surplus blocks leave at once: the partitioning, and so every bit of the result, is the same either way.
    const int64_t pos0 = pos_dev ? (int64_t)*pos_dev : pos0_host;
    const int64_t kv_len = pos0 + gridDim.z;
    const int nsplit_eff = splitkv_count(kv_len, gridDim.y, gridDim.z);
    const int64_t chunk = splitkv_chunk(kv_len, nsplit_eff);
    if ((int)blockIdx.x >= nsplit_eff) return;
    constexpr int D = TPK * 8;
    constexpr int NG = kBlock / TPK;
    __shared__ __align__(16) float s_acc[NG * D];
    __shared__ float s_m[NG], s_l[NG];

    const int sp = blockIdx.x, kvh = blockIdx.y, t = blockIdx.z;
    const int c = threadIdx.x % TPK, gk = threadIdx.x / TPK;
    const int64_t visible = pos0 + t + 1;
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
                kk[u] = __ldg(reinterpret_cast<const uint4*>(kb + j * D));
                vv[u] = __ldg(reinterpret_cast<const uint4*>(vb + j * D));
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
                for (int o = TPK / 2; o > 0; o >>= 1) d += __shfl_xor_sync(0xffffffffu, d, o);
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

template <int D>
__global__ void __launch_bounds__(D)
combine_kernel(const float* __restrict__ pacc, const float2* __restrict__ pml, ik_bf16* __restrict__ out, int nsplit,
               int T, int NKV, int64_t pos0_host, const int32_t* pos_dev) {
    const int64_t row = blockIdx.x;  // t * N + h
    const int64_t pos0 = pos_dev ? (int64_t)*pos_dev : pos0_host;
    const int nsplit_eff = splitkv_count(pos0 + T, NKV, T);  // same count as the stage-1 blocks used
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

// ---- pipelined stage 1 (attn_*_bf16_splitkv_pipe): the arithmetic of splitkv_kernel, the loads ahead of it -----------
//
// splitkv_pipe_kernel computes exactly what splitkv_kernel computes, operation for operation and in the same order (the
// partition into splits and key lanes, each lane's online softmax over KB keys per step, the merge of the lanes, the
// combine of the splits): every output bit is the base kernel's (tests/splitkv_pipe.rs). Only the loads move. The base
// kernel loads a step's K / V rows with __ldg and then waits for them; here each thread streams its own 16 B of every row
// through a ring of ST stages in shared memory with cp.async (zero-filled past the end, as the base kernel zeroes them),
// ST - 1 steps ahead of the arithmetic, so the DRAM reads of later steps are in flight while a step is computed.
// A thread reads back only the bytes it copied itself, so the ring needs no barrier, only cp.async.wait_group. After the
// key loop the ring is reused as s_acc for the merge (same __syncthreads as the base kernel).
__device__ __forceinline__ void skv_cp16(void* dst, const void* src, bool valid) {
    const int sz = valid ? 16 : 0;  // src-size 0 zero-fills the 16 bytes
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"((uint32_t)__cvta_generic_to_shared(dst)),
                 "l"(src), "r"(sz)
                 : "memory");
}
__device__ __forceinline__ void skv_commit() { asm volatile("cp.async.commit_group;\n" ::: "memory"); }
template <int N>
__device__ __forceinline__ void skv_wait() {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N) : "memory");
}

template <int KB, int ST>
constexpr int pipe_smem_bytes() {
    // ring [ST][K, V][KB][kBlock] x 16 B; reused as s_acc[NG * D] = kBlock * 8 floats after the key loop
    return ST * 2 * KB * kBlock * 16 > kBlock * 8 * 4 ? ST * 2 * KB * kBlock * 16 : kBlock * 8 * 4;
}

template <int GP, int TPK, int KB, int ST>
__global__ void __launch_bounds__(kBlock)
splitkv_pipe_kernel(const ik_bf16* __restrict__ q, const ik_bf16* __restrict__ kc, const ik_bf16* __restrict__ vc,
                    float* __restrict__ pacc, float2* __restrict__ pml, int N, int G, int64_t cap, int64_t pos0_host,
                    float qscale, int nsplit, const int32_t* pos_dev) {
    static_assert(ST >= 2, "the ring needs at least two stages");
    const int64_t pos0 = pos_dev ? (int64_t)*pos_dev : pos0_host;
    const int64_t kv_len = pos0 + gridDim.z;
    const int nsplit_eff = splitkv_count(kv_len, gridDim.y, gridDim.z);
    const int64_t chunk = splitkv_chunk(kv_len, nsplit_eff);
    if ((int)blockIdx.x >= nsplit_eff) return;
    constexpr int D = TPK * 8;
    constexpr int NG = kBlock / TPK;
    constexpr int STEP = NG * KB;
    extern __shared__ __align__(16) uint4 s_ring[];
    float* s_acc = reinterpret_cast<float*>(s_ring);
    __shared__ float s_m[NG], s_l[NG];

    const int sp = blockIdx.x, kvh = blockIdx.y, t = blockIdx.z;
    const int c = threadIdx.x % TPK, gk = threadIdx.x / TPK;
    const int64_t visible = pos0 + t + 1;
    const int64_t start = (int64_t)sp * chunk;
    int64_t end = start + chunk;
    if (end > visible) end = visible;

    const ik_bf16* kb = kc + (int64_t)kvh * cap * D + c * 8;
    const ik_bf16* vb = vc + (int64_t)kvh * cap * D + c * 8;

    // the steps of the base kernel's key loop: base = start + it * STEP for it < niter
    const int64_t niter = end > start ? (end - start + STEP - 1) / STEP : 0;
    auto issue = [&](int64_t it, int stage) {
        uint4* st = s_ring + stage * (2 * KB * kBlock);
        const int64_t base = start + it * STEP;
#pragma unroll
        for (int u = 0; u < KB; ++u) {
            const int64_t j = base + (int64_t)u * NG + gk;
            const bool ok = j < end;
            skv_cp16(st + u * kBlock + threadIdx.x, ok ? kb + j * D : kb, ok);
            skv_cp16(st + (KB + u) * kBlock + threadIdx.x, ok ? vb + j * D : vb, ok);
        }
    };
    // prologue: steps 0 .. ST-2 in flight (one commit group per step, empty past the end)
#pragma unroll
    for (int s = 0; s < ST - 1; ++s) {
        if (s < niter) issue(s, s);
        skv_commit();
    }

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

    int rd = 0, wr = ST - 1;
    for (int64_t it = 0; it < niter; ++it) {
        const int64_t base = start + it * STEP;
        skv_wait<ST - 2>();  // this thread's copies of step `it` have landed
        uint4 kk[KB], vv[KB];
        bool ok[KB];
        const uint4* st = s_ring + rd * (2 * KB * kBlock);
#pragma unroll
        for (int u = 0; u < KB; ++u) {
            const int64_t j = base + (int64_t)u * NG + gk;
            ok[u] = j < end;
            kk[u] = st[u * kBlock + threadIdx.x];
            vv[u] = st[(KB + u) * kBlock + threadIdx.x];
        }
        // refill the stage read one step ago (by this thread only) with step it + ST - 1
        if (it + ST - 1 < niter) issue(it + ST - 1, wr);
        skv_commit();
        rd = rd + 1 == ST ? 0 : rd + 1;
        wr = wr + 1 == ST ? 0 : wr + 1;

        // ---- from here to the end of the step: splitkv_kernel's arithmetic, unchanged ----
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
                for (int o = TPK / 2; o > 0; o >>= 1) d += __shfl_xor_sync(0xffffffffu, d, o);
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
    skv_wait<0>();  // only empty groups can be pending here; the ring becomes s_acc after the first barrier below

    // merge the NG key lanes, one head at a time (splitkv_kernel's merge, unchanged)
#pragma unroll
    for (int g = 0; g < GP; ++g) {
        if (g < G) {
            __syncthreads();  // the ring / s_acc / s_m / s_l are reused for every head
#pragma unroll
            for (int e = 0; e < 8; ++e) s_acc[gk * D + c * 8 + e] = acc[g][e];
            if (c == 0) {
                s_m[gk] = m[g];
                s_l[gk] = l[g];
            }
            __syncthreads();  // every lane's partial is in s_acc / s_m / s_l
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

// combine_kernel's arithmetic with the per-split loads unrolled (issued ahead of the in-order sums): same bits
template <int D>
__global__ void __launch_bounds__(D)
combine_pipe_kernel(const float* __restrict__ pacc, const float2* __restrict__ pml, ik_bf16* __restrict__ out,
                    int nsplit, int T, int NKV, int64_t pos0_host, const int32_t* pos_dev) {
    const int64_t row = blockIdx.x;  // t * N + h
    const int64_t pos0 = pos_dev ? (int64_t)*pos_dev : pos0_host;
    const int nsplit_eff = splitkv_count(pos0 + T, NKV, T);
    const float2* ml = pml + row * nsplit;
    const float* acc = pacc + row * nsplit * D;
    float M = -INFINITY;
#pragma unroll 8
    for (int s = 0; s < nsplit_eff; ++s) M = fmaxf(M, ml[s].x);
    float A = 0.f, L = 0.f;
#pragma unroll 8
    for (int s = 0; s < nsplit_eff; ++s) {
        const float2 v = ml[s];
        if (v.x == -INFINITY) continue;  // empty split: acc is not meaningful
        const float w = exp2f(v.x - M);
        L += v.y * w;
        A += acc[(int64_t)s * D + threadIdx.x] * w;
    }
    out[row * D + threadIdx.x] = ik_f2b(A / L);
}

// ---- head-split stage 1 (INTARSIA_SPLITKV_HS=1, group sizes 5 .. 8): the 8 heads of a key lane over two warps ----------
//
// splitkv_kernel<8, TPK, 2>'s arithmetic per head, unchanged: same partition, same key lanes (same keys, same KB = 2 steps),
// same per-head online softmax and lane merge. The block has 2 x 256 threads: thread half h (0 / 1) holds heads
// 4h .. 4h+3 of the lane, so each thread carries half the accumulators (registers <= 128: 16 warps per SM instead of 8),
// and both halves load the lane's K / V rows (the second load of a row hits L1 / L2; DRAM reads them once).
template <int TPK>
__global__ void __launch_bounds__(2 * kBlock, 1)
splitkv_hs_kernel(const ik_bf16* __restrict__ q, const ik_bf16* __restrict__ kc, const ik_bf16* __restrict__ vc,
                  float* __restrict__ pacc, float2* __restrict__ pml, int N, int G, int64_t cap, int64_t pos0_host,
                  float qscale, int nsplit, const int32_t* pos_dev) {
    constexpr int GP = 4, KB = 2;  // heads per thread half, keys per lane per step (as splitkv_kernel<8, TPK, 2>)
    const int64_t pos0 = pos_dev ? (int64_t)*pos_dev : pos0_host;
    const int64_t kv_len = pos0 + gridDim.z;
    const int nsplit_eff = splitkv_count(kv_len, gridDim.y, gridDim.z);
    const int64_t chunk = splitkv_chunk(kv_len, nsplit_eff);
    if ((int)blockIdx.x >= nsplit_eff) return;
    constexpr int D = TPK * 8;
    constexpr int NG = kBlock / TPK;
    __shared__ __align__(16) float s_acc[2][NG * D];
    __shared__ float s_m[2][NG], s_l[2][NG];

    const int sp = blockIdx.x, kvh = blockIdx.y, t = blockIdx.z;
    const int half = threadIdx.x / kBlock, lt = threadIdx.x % kBlock;
    const int c = lt % TPK, gk = lt / TPK;
    const int g0 = half * GP;  // first head of this half
    const int64_t visible = pos0 + t + 1;
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
            if (g0 + g < G) qr[g][e] = ik_b2f(q[((int64_t)t * N + (int64_t)kvh * G + g0 + g) * D + c * 8 + e]) * qscale;
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
                kk[u] = __ldg(reinterpret_cast<const uint4*>(kb + j * D));
                vv[u] = __ldg(reinterpret_cast<const uint4*>(vb + j * D));
            }
        }
        // ---- splitkv_kernel's per-head arithmetic, for this half's heads ----
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
                for (int o = TPK / 2; o > 0; o >>= 1) d += __shfl_xor_sync(0xffffffffu, d, o);
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

    // merge the NG key lanes, two heads at a time (one per half), each with splitkv_kernel's merge arithmetic; the
    // barriers are outside the head guard so that both halves always reach them
#pragma unroll
    for (int g = 0; g < GP; ++g) {
        const bool live = g0 + g < G;
        __syncthreads();  // hs: s_acc / s_m / s_l are reused for every head pair
        if (live) {
#pragma unroll
            for (int e = 0; e < 8; ++e) s_acc[half][gk * D + c * 8 + e] = acc[g][e];
            if (c == 0) {
                s_m[half][gk] = m[g];
                s_l[half][gk] = l[g];
            }
        }
        __syncthreads();  // hs: every lane's partial of the pair is in s_acc / s_m / s_l
        if (live && lt < D) {
            const int64_t slot = ((int64_t)t * N + (int64_t)kvh * G + g0 + g) * nsplit + sp;
            float M = -INFINITY;
            for (int i = 0; i < NG; ++i) M = fmaxf(M, s_m[half][i]);
            float A = 0.f, L = 0.f;
            if (M != -INFINITY) {
                for (int i = 0; i < NG; ++i) {
                    const float w = exp2f(s_m[half][i] - M);  // empty lane: exp2(-inf) = 0, acc is 0 too
                    L += s_l[half][i] * w;
                    A += s_acc[half][i * D + lt] * w;
                }
            }
            pacc[slot * D + lt] = A;
            if (lt == 0) pml[slot] = make_float2(M, L);
        }
    }
}

// INTARSIA_SPLITKV_HS=1 (A/B): the head-split stage 1 for group sizes 5 .. 8 in the pipelined launchers. Read once.
bool splitkv_hs() {
    static int on = -1;
    if (on < 0) {
        const char* v = getenv("INTARSIA_SPLITKV_HS");
        on = v && v[0] == '1' ? 1 : 0;
    }
    return on == 1;
}

// stage count of the ring: INTARSIA_SPLITKV_STAGES=2|3|4 (A/B only), else the default per step size. Read once.
int pipe_stages(int KB) {
    static int env = -1;
    if (env < 0) {
        env = 0;
        if (const char* v = getenv("INTARSIA_SPLITKV_STAGES")) {
            const int x = atoi(v);
            if (x >= 2 && x <= 4) env = x;
        }
    }
    const int st = env ? env : (KB >= 4 ? 2 : 4);
    return KB >= 4 && st > 3 ? 3 : st;  // KB 4: 4 stages would need 128 KiB
}

template <int GP, int TPK, int KB, int ST>
int launch_pipe_st(const ik_bf16* q, const ik_bf16* kc, const ik_bf16* vc, float* pacc, float2* pml, int64_t T,
                   int64_t N, int64_t NKV, int64_t cap, int64_t pos0, float qscale, int nsplit, const int32_t* pos_dev,
                   cudaStream_t s) {
    constexpr int smem = pipe_smem_bytes<KB, ST>();
    {  // always: the 48 KiB default limit counts the static s_m / s_l too
        // the dynamic-smem opt-in is per function and per device context; set it once per device (outside any capture's
        // rules: relaxed mode for this one call, so a first launch inside a graph capture works too)
        static bool configured[16] = {};
        int dev = 0;
        if (cudaGetDevice(&dev) != cudaSuccess) return IK_ERR_LAUNCH;
        if (dev < 0 || dev >= 16) return IK_ERR_UNSUPPORTED;
        if (!configured[dev]) {
            cudaStreamCaptureMode mode = cudaStreamCaptureModeRelaxed;
            cudaThreadExchangeStreamCaptureMode(&mode);
            const cudaError_t e = cudaFuncSetAttribute(splitkv_pipe_kernel<GP, TPK, KB, ST>,
                                                       cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
            cudaThreadExchangeStreamCaptureMode(&mode);
            if (e != cudaSuccess) return IK_ERR_LAUNCH;
            configured[dev] = true;
        }
    }
    dim3 grid(nsplit, (unsigned)NKV, (unsigned)T);
    splitkv_pipe_kernel<GP, TPK, KB, ST><<<grid, kBlock, smem, s>>>(q, kc, vc, pacc, pml, (int)N, (int)(N / NKV), cap,
                                                                    pos0, qscale, nsplit, pos_dev);
    return IK_OK;
}

template <int GP, int TPK, int KB>
int launch_pipe(int st, const ik_bf16* q, const ik_bf16* kc, const ik_bf16* vc, float* pacc, float2* pml, int64_t T,
                int64_t N, int64_t NKV, int64_t cap, int64_t pos0, float qscale, int nsplit, const int32_t* pos_dev,
                cudaStream_t s) {
    switch (st) {
        case 2: return launch_pipe_st<GP, TPK, KB, 2>(q, kc, vc, pacc, pml, T, N, NKV, cap, pos0, qscale, nsplit, pos_dev, s);
        case 3: return launch_pipe_st<GP, TPK, KB, 3>(q, kc, vc, pacc, pml, T, N, NKV, cap, pos0, qscale, nsplit, pos_dev, s);
        default: break;
    }
    if constexpr (KB < 4) return launch_pipe_st<GP, TPK, KB, 4>(q, kc, vc, pacc, pml, T, N, NKV, cap, pos0, qscale, nsplit, pos_dev, s);
    return IK_ERR_UNSUPPORTED;
}

template <int TPK>
int dispatch_group_pipe(int G, const ik_bf16* q, const ik_bf16* kc, const ik_bf16* vc, float* pacc, float2* pml,
                        int64_t T, int64_t N, int64_t NKV, int64_t cap, int64_t pos0, float qscale, int nsplit,
                        const int32_t* pos_dev, cudaStream_t s) {
    // the same (GP, KB) per group size as dispatch_group: the step size KB is part of the arithmetic
#define IK_PIPE_CASE(GPV, KBV) \
    return launch_pipe<GPV, TPK, KBV>(pipe_stages(KBV), q, kc, vc, pacc, pml, T, N, NKV, cap, pos0, qscale, nsplit, pos_dev, s);
    if (G <= 1) { IK_PIPE_CASE(1, 4) }
    if (G <= 2) { IK_PIPE_CASE(2, 4) }
    if (G <= 4) { IK_PIPE_CASE(4, 4) }
    if (G <= 8 && splitkv_hs()) {
        dim3 grid(nsplit, (unsigned)NKV, (unsigned)T);
        splitkv_hs_kernel<TPK><<<grid, 2 * kBlock, 0, s>>>(q, kc, vc, pacc, pml, (int)N, (int)(N / NKV), cap, pos0, qscale,
                                                          nsplit, pos_dev);
        return IK_OK;
    }
    if (G <= 8) { IK_PIPE_CASE(8, 2) }
#undef IK_PIPE_CASE
    return IK_ERR_SHAPE;
}

template <int GP, int TPK, int KB>
void launch_split(const ik_bf16* q, const ik_bf16* kc, const ik_bf16* vc, float* pacc, float2* pml, int64_t T,
                  int64_t N, int64_t NKV, int64_t cap, int64_t pos0, float qscale, int nsplit,
                  const int32_t* pos_dev, cudaStream_t s) {
    dim3 grid(nsplit, (unsigned)NKV, (unsigned)T);
    splitkv_kernel<GP, TPK, KB><<<grid, kBlock, 0, s>>>(q, kc, vc, pacc, pml, (int)N, (int)(N / NKV), cap, pos0,
                                                        qscale, nsplit, pos_dev);
}

template <int TPK>
int dispatch_group(int G, const ik_bf16* q, const ik_bf16* kc, const ik_bf16* vc, float* pacc, float2* pml,
                   int64_t T, int64_t N, int64_t NKV, int64_t cap, int64_t pos0, float qscale, int nsplit,
                   const int32_t* pos_dev, cudaStream_t s) {
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

int splitkv_launch(const ik_tensor* q, const ik_tensor* kc, const ik_tensor* vc, int64_t pos0, float scale,
                   ik_tensor* out, const ik_scratch* scratch, const ik_tensor* pos, void* stream, bool decode, bool pipe = false) {
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
    const ik_bf16* kp = (const ik_bf16*)kc->data;
    const ik_bf16* vp = (const ik_bf16*)vc->data;
    const float qscale = scale * kLog2e;
    int rc = IK_ERR_SHAPE;
    if (pipe) {
        switch (D) {
            case 32: rc = dispatch_group_pipe<4>((int)(N / NKV), qp, kp, vp, pacc, pml, T, N, NKV, cap, pos0, qscale, nsplit, pos_dev, s); break;
            case 64: rc = dispatch_group_pipe<8>((int)(N / NKV), qp, kp, vp, pacc, pml, T, N, NKV, cap, pos0, qscale, nsplit, pos_dev, s); break;
            case 128: rc = dispatch_group_pipe<16>((int)(N / NKV), qp, kp, vp, pacc, pml, T, N, NKV, cap, pos0, qscale, nsplit, pos_dev, s); break;
            case 256: rc = dispatch_group_pipe<32>((int)(N / NKV), qp, kp, vp, pacc, pml, T, N, NKV, cap, pos0, qscale, nsplit, pos_dev, s); break;
        }
        if (rc != IK_OK) return rc;
        IK_TRY(ik_launch_status());
        ik_bf16* op = (ik_bf16*)out->data;
        const unsigned rows = (unsigned)(T * N);
        switch (D) {
            case 32: combine_pipe_kernel<32><<<rows, 32, 0, s>>>(pacc, pml, op, nsplit, (int)T, (int)NKV, pos0, pos_dev); break;
            case 64: combine_pipe_kernel<64><<<rows, 64, 0, s>>>(pacc, pml, op, nsplit, (int)T, (int)NKV, pos0, pos_dev); break;
            case 128: combine_pipe_kernel<128><<<rows, 128, 0, s>>>(pacc, pml, op, nsplit, (int)T, (int)NKV, pos0, pos_dev); break;
            case 256: combine_pipe_kernel<256><<<rows, 256, 0, s>>>(pacc, pml, op, nsplit, (int)T, (int)NKV, pos0, pos_dev); break;
        }
        return ik_launch_status();
    }
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
        case 32: combine_kernel<32><<<rows, 32, 0, s>>>(pacc, pml, op, nsplit, (int)T, (int)NKV, pos0, pos_dev); break;
        case 64: combine_kernel<64><<<rows, 64, 0, s>>>(pacc, pml, op, nsplit, (int)T, (int)NKV, pos0, pos_dev); break;
        case 128: combine_kernel<128><<<rows, 128, 0, s>>>(pacc, pml, op, nsplit, (int)T, (int)NKV, pos0, pos_dev); break;
        case 256: combine_kernel<256><<<rows, 256, 0, s>>>(pacc, pml, op, nsplit, (int)T, (int)NKV, pos0, pos_dev); break;
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

// the pipelined stage 1 + unrolled combine: bitwise the two launchers above (same scratch, same grid policy)
int ik_attn_decode_bf16_splitkv_pipe(const ik_tensor* q, const ik_tensor* kc, const ik_tensor* vc, int64_t pos0,
                                     float scale, ik_tensor* out, const ik_scratch* scratch, const ik_tensor* pos,
                                     void* stream) {
    return splitkv_launch(q, kc, vc, pos0, scale, out, scratch, pos, stream, true, true);
}

int ik_attn_prefill_bf16_splitkv_pipe(const ik_tensor* q, const ik_tensor* kc, const ik_tensor* vc, int64_t pos0,
                                      float scale, ik_tensor* out, const ik_scratch* scratch, void* stream) {
    return splitkv_launch(q, kc, vc, pos0, scale, out, scratch, nullptr, stream, false, true);
}

}  // extern "C"
