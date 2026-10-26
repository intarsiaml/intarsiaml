// Flash-style tiled causal GQA prefill attention on the gfx12 WMMA units (v_wmma_f32_16x16x16_bf16, wave32) over a
// contiguous bf16 KV cache [Nkv, cap, D]. Port of kernels/cuda/attention_flash.cu with the mma.sync / ldmatrix /
// cp.async parts rewritten for RDNA4.
//
// Block = 4 waves = 64 query rows of one q head; each wave owns 16 rows. K tiles of BK keys are staged row-major
// in LDS ([key][D]) and V tiles transposed ([D][key]) so every WMMA operand is one contiguous 16 B LDS read.
//
// The kernel computes the transposed problem so the accumulator layout feeds the next WMMA directly:
//   S^T[key, q]  = K[key, :] . Q[q, :]            A = K tile, B = Q fragments (registers)
//   O^T[d,   q] += V^T[d, key] . P^T[key, q]      A = V^T tile, B = P^T (== the S^T accumulator lanes, as bf16)
// WMMA 16x16x16 gfx12 wave32 layout: A/B lane l holds row/col l % 16 and k = (l / 16) * 8 .. +7; C/D lane l holds
// col l % 16 and rows (l / 16) * 8 + i. With col = query row, every lane owns exactly one query row (l % 16) and
// the two lane halves split its keys / output dims, so the softmax state is one (m, l) per lane and merging the
// halves is a single xor-16 shuffle. Online softmax in the log2 domain, f32 state.
//
// Causality / chunking: query row r of the call sees keys 0 .. pos0 + r. Tiles that lie fully below the first
// row of the block are processed without a mask. Blocks are issued longest-first.
#include "ik_hip.h"
#include <stdlib.h>

namespace {

constexpr int kBQ = 64;
constexpr int kThreads = 128;
constexpr float kLog2e = 1.4426950408889634f;

typedef __bf16 ik_v8bf __attribute__((ext_vector_type(8)));
typedef float ik_v8f __attribute__((ext_vector_type(8)));

IK_DEVI ik_v8f wmma_bf16(const uint4& a, const uint4& b, ik_v8f c) {
    return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(__builtin_bit_cast(ik_v8bf, a),
                                                             __builtin_bit_cast(ik_v8bf, b), c);
}

// D: head_dim, BK: keys per tile
template <int D, int BK>
__global__ void __launch_bounds__(kThreads)
flash_kernel(const ik_bf16* __restrict__ q, const ik_bf16* __restrict__ kc, const ik_bf16* __restrict__ vc,
             ik_bf16* __restrict__ out, int T, int N, int G, int64_t cap, int64_t pos0, float sl2) {
    constexpr int LDK = D + 8;   // padded row (elements) of the K tile: conflict-free 16 B reads
    constexpr int LDV = BK + 8;  // padded row of the transposed V tile
    constexpr int NKS = D / 16;  // k steps of S^T (head dim)
    constexpr int NKB = BK / 16; // 16-key blocks per tile
    constexpr int NDB = D / 16;  // 16-dim blocks of O^T
    constexpr int CH = D / 8;    // 16 B chunks per key row

    __shared__ __align__(16) uint16_t Ks[BK * LDK];
    __shared__ __align__(16) uint16_t Vt[D * LDV];

    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int lr = lane & 15, lk = (lane >> 4) * 8;
    const int qt = (int)gridDim.x - 1 - (int)blockIdx.x;  // longest blocks first
    const int h = blockIdx.y, kvh = h / G;
    const int q0 = qt * kBQ;
    const int qrow = q0 + warp * 16 + lr;  // this lane's query row (its l % 16 column of every accumulator)

    const ik_bf16* kb = kc + (int64_t)kvh * cap * D;
    const ik_bf16* vb = vc + (int64_t)kvh * cap * D;

    const int q_last = (q0 + kBQ < T ? q0 + kBQ : T) - 1;
    const int64_t kv_end = pos0 + q_last + 1;  // keys visible to the last row of the block
    const int ntiles = (int)((kv_end + BK - 1) / BK);
    const int64_t mask_from = pos0 + q0 + 1;   // tiles ending beyond this need the causal mask

    // Q as the B operand of S^T: lane holds query row qrow, head dims ks*16 + lk .. +7
    uint4 qb[NKS];
#pragma unroll
    for (int ks = 0; ks < NKS; ++ks) {
        qb[ks] = make_uint4(0, 0, 0, 0);
        if (qrow < T) {  // 4 B accesses: q only has to be 4-byte aligned
            const uint32_t* p = reinterpret_cast<const uint32_t*>(q + ((int64_t)qrow * N + h) * D + ks * 16 + lk);
            qb[ks] = make_uint4(p[0], p[1], p[2], p[3]);
        }
    }

    ik_v8f o[NDB];
#pragma unroll
    for (int i = 0; i < NDB; ++i)
#pragma unroll
        for (int e = 0; e < 8; ++e) o[i][e] = 0.f;
    float m = -INFINITY, l = 0.f;

    for (int kt = 0; kt < ntiles; ++kt) {
        IK_LDS_BARRIER();  // the previous tile's reads of Ks / Vt are done
        for (int idx = tid; idx < BK * CH; idx += kThreads) {
            const int r = idx / CH, ch = idx % CH;
            const int64_t key = (int64_t)kt * BK + r;
            uint4 kv = make_uint4(0, 0, 0, 0), vv = make_uint4(0, 0, 0, 0);
            if (key < kv_end) {
                kv = *reinterpret_cast<const uint4*>(kb + key * D + ch * 8);
                vv = *reinterpret_cast<const uint4*>(vb + key * D + ch * 8);
            }
            *reinterpret_cast<uint4*>(&Ks[r * LDK + ch * 8]) = kv;
            const uint32_t vw[4] = {vv.x, vv.y, vv.z, vv.w};
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                Vt[(ch * 8 + 2 * e) * LDV + r] = (uint16_t)(vw[e] & 0xffffu);
                Vt[(ch * 8 + 2 * e + 1) * LDV + r] = (uint16_t)(vw[e] >> 16);
            }
        }
        IK_LDS_BARRIER();

        // ---- S^T = K Q^T : lane holds key (kb*16 + lk + i) of its query row ----
        ik_v8f s[NKB];
#pragma unroll
        for (int b = 0; b < NKB; ++b)
#pragma unroll
            for (int e = 0; e < 8; ++e) s[b][e] = 0.f;
#pragma unroll
        for (int ks = 0; ks < NKS; ++ks) {
#pragma unroll
            for (int b = 0; b < NKB; ++b) {
                const uint4 a = *reinterpret_cast<const uint4*>(&Ks[(b * 16 + lr) * LDK + ks * 16 + lk]);
                s[b] = wmma_bf16(a, qb[ks], s[b]);
            }
        }

        // ---- online softmax (log2 domain) ----
        const bool need_mask = (int64_t)(kt + 1) * BK > mask_from;
        float mx = -INFINITY;
#pragma unroll
        for (int b = 0; b < NKB; ++b) {
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                float v = s[b][e] * sl2;
                if (need_mask) {
                    const int64_t key = (int64_t)kt * BK + b * 16 + lk + e;
                    if (key > pos0 + qrow) v = -INFINITY;
                }
                s[b][e] = v;
                mx = fmaxf(mx, v);
            }
        }
        mx = fmaxf(mx, __shfl_xor(mx, 16, 32));
        const float mn = fmaxf(m, mx);
        const float mu = mn == -INFINITY ? 0.f : mn;
        const float cr = exp2f(m - mu);
        m = mn;
        float ps = 0.f;
        uint4 pb[NKB];  // P^T as bf16: the B operand of the PV product, key block b
#pragma unroll
        for (int b = 0; b < NKB; ++b) {
            float p[8];
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                p[e] = exp2f(s[b][e] - mu);
                ps += p[e];
            }
            pb[b] = ik_pack8(p);
        }
        l = l * cr + ps;
#pragma unroll
        for (int i = 0; i < NDB; ++i)
#pragma unroll
            for (int e = 0; e < 8; ++e) o[i][e] *= cr;

        // ---- O^T += V^T P^T ----
#pragma unroll
        for (int b = 0; b < NKB; ++b) {
#pragma unroll
            for (int db = 0; db < NDB; ++db) {
                const uint4 a = *reinterpret_cast<const uint4*>(&Vt[(db * 16 + lr) * LDV + b * 16 + lk]);
                o[db] = wmma_bf16(a, pb[b], o[db]);
            }
        }
    }

    // ---- epilogue: merge the two lane halves' partial row sums, write 8 consecutive dims per lane ----
    l += __shfl_xor(l, 16, 32);
    const float inv = 1.f / l;
    if (qrow < T) {
#pragma unroll
        for (int db = 0; db < NDB; ++db) {
            float v[8];
#pragma unroll
            for (int e = 0; e < 8; ++e) v[e] = o[db][e] * inv;
            const uint4 u = ik_pack8(v);
            uint32_t* po = reinterpret_cast<uint32_t*>(out + ((int64_t)qrow * N + h) * D + db * 16 + lk);
            po[0] = u.x;
            po[1] = u.y;
            po[2] = u.z;
            po[3] = u.w;
        }
    }
}

template <int D, int BK>
int launch_flash(const ik_bf16* q, const ik_bf16* kc, const ik_bf16* vc, ik_bf16* out, int T, int N, int NKV,
                 int64_t cap, int64_t pos0, float scale, hipStream_t s) {
    dim3 grid((T + kBQ - 1) / kBQ, N);
    flash_kernel<D, BK><<<grid, kThreads, 0, s>>>(q, kc, vc, out, T, N, N / NKV, cap, pos0, scale * kLog2e);
    return ik_launch_status();
}

// ---- v2: HPB query heads of one kv head per block ---------------------------------------------------------------------
// flash_kernel with the block widened to HPB groups of 4 waves, group g taking query head blockIdx.y * HPB + g (all HPB
// heads share the kv head: the GQA group is a multiple of HPB). Every K / V tile is staged into LDS once for HPB heads
// instead of once per head, and the block keeps the same LDS (one tile) for HPB times the waves, so a CU holds HPB times
// as many waves at the same shared-memory footprint (the 35B's D = 256 tile is ~37 KB, which bounded flash_kernel to a
// few blocks per WGP). Each wave runs exactly flash_kernel's instructions for its 16 rows (same Q fragments, same tiles,
// same WMMA and softmax order), so the output is bitwise flash_kernel's.
// VPM (INTARSIA_HIP_FLASH_VP, bit mask; every value gives flash_kernel's bits):
//   bit 0 (1)  V^T staged by (key pair, chunk) with 4 B LDS stores, bank-spread (see the stage code below)
//   bit 1 (2)  the next tile's K / V global loads are issued before this tile's math (register prefetch; implies bit 0's layout)
//   bit 2 (4)  skip the o *= cr rescale when cr == 1.0f in every lane of the wave (multiplying by exactly 1.0f is the identity)
template <int D, int BK, int HPB, int VPM>
__global__ void __launch_bounds__(kThreads * HPB)
flash_hpb_kernel(const ik_bf16* __restrict__ q, const ik_bf16* __restrict__ kc, const ik_bf16* __restrict__ vc,
                 ik_bf16* __restrict__ out, int T, int N, int G, int64_t cap, int64_t pos0, float sl2) {
    constexpr int LDK = D + 8;
    constexpr int LDV = BK + 8;
    constexpr int NKS = D / 16;
    constexpr int NKB = BK / 16;
    constexpr int NDB = D / 16;
    constexpr int CH = D / 8;
    constexpr int NT = kThreads * HPB;

    __shared__ __align__(16) uint16_t Ks[BK * LDK];
    __shared__ __align__(16) uint16_t Vt[D * LDV];

    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int lr = lane & 15, lk = (lane >> 4) * 8;
    const int qt = (int)gridDim.x - 1 - (int)blockIdx.x;  // longest blocks first
    const int h = blockIdx.y * HPB + (warp >> 2), kvh = (blockIdx.y * HPB) / G;
    const int q0 = qt * kBQ;
    const int qrow = q0 + (warp & 3) * 16 + lr;

    const ik_bf16* kb = kc + (int64_t)kvh * cap * D;
    const ik_bf16* vb = vc + (int64_t)kvh * cap * D;

    const int q_last = (q0 + kBQ < T ? q0 + kBQ : T) - 1;
    const int64_t kv_end = pos0 + q_last + 1;
    const int ntiles = (int)((kv_end + BK - 1) / BK);
    const int64_t mask_from = pos0 + q0 + 1;

    uint4 qb[NKS];
#pragma unroll
    for (int ks = 0; ks < NKS; ++ks) {
        qb[ks] = make_uint4(0, 0, 0, 0);
        if (qrow < T) {
            const uint32_t* p = reinterpret_cast<const uint32_t*>(q + ((int64_t)qrow * N + h) * D + ks * 16 + lk);
            qb[ks] = make_uint4(p[0], p[1], p[2], p[3]);
        }
    }

    ik_v8f o[NDB];
#pragma unroll
    for (int i = 0; i < NDB; ++i)
#pragma unroll
        for (int e = 0; e < 8; ++e) o[i][e] = 0.f;
    float m = -INFINITY, l = 0.f;

    constexpr bool VP = (VPM & 3) != 0, PF = (VPM & 2) != 0, SK = (VPM & 4) != 0;
    constexpr int P = BK / 2;                      // key pairs per tile
    constexpr int NKI = (BK * CH + NT - 1) / NT;   // K chunks (16 B) per thread and tile
    constexpr int NVI = (P * CH + NT - 1) / NT;    // V (key pair, chunk) units per thread and tile
    uint4 rk[NKI], ra[NVI], rb[NVI];
    // K is staged as one 16 B chunk per (key, chunk), coalesced rows. V is staged by (key pair, chunk): a thread reads keys 2p
    // and 2p + 1 of one 16 B chunk and writes 8 dword pairs of V^T (dims ch * 8 .. + 7, keys 2p / 2p + 1) with 4 B stores.
    // Lanes of a wave then differ in p (the dword column), so the 32 stores of one instruction hit 16 distinct banks (2-way)
    // instead of one bank (the 2 B stores of the other path: 32 lanes = 32 chunks of the same key, whose rows are 8 * LDV * 2 B
    // = a multiple of the bank stride apart: 32-way). The same bytes land in Ks / Vt.
    auto load_regs = [&](int t) {
#pragma unroll
        for (int i = 0; i < NKI; ++i) {
            const int idx = tid + i * NT, r = idx / CH, ch = idx % CH;
            const int64_t key = (int64_t)t * BK + r;
            rk[i] = make_uint4(0, 0, 0, 0);
            if (idx < BK * CH && key < kv_end) rk[i] = *reinterpret_cast<const uint4*>(kb + key * D + ch * 8);
        }
#pragma unroll
        for (int i = 0; i < NVI; ++i) {
            const int u = tid + i * NT, p = u % P, ch = u / P;
            const int64_t k0 = (int64_t)t * BK + 2 * p;
            ra[i] = make_uint4(0, 0, 0, 0);
            rb[i] = make_uint4(0, 0, 0, 0);
            if (u < P * CH && k0 < kv_end) ra[i] = *reinterpret_cast<const uint4*>(vb + k0 * D + ch * 8);
            if (u < P * CH && k0 + 1 < kv_end) rb[i] = *reinterpret_cast<const uint4*>(vb + (k0 + 1) * D + ch * 8);
        }
    };
    auto store_regs = [&]() {
#pragma unroll
        for (int i = 0; i < NKI; ++i) {
            const int idx = tid + i * NT, r = idx / CH, ch = idx % CH;
            if (idx < BK * CH) *reinterpret_cast<uint4*>(&Ks[r * LDK + ch * 8]) = rk[i];
        }
#pragma unroll
        for (int i = 0; i < NVI; ++i) {
            const int u = tid + i * NT, p = u % P, ch = u / P;
            if (u < P * CH) {
                const uint32_t a[4] = {ra[i].x, ra[i].y, ra[i].z, ra[i].w}, b[4] = {rb[i].x, rb[i].y, rb[i].z, rb[i].w};
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    *reinterpret_cast<uint32_t*>(&Vt[(ch * 8 + 2 * e) * LDV + 2 * p]) = (a[e] & 0xffffu) | (b[e] << 16);
                    *reinterpret_cast<uint32_t*>(&Vt[(ch * 8 + 2 * e + 1) * LDV + 2 * p]) = (a[e] >> 16) | (b[e] & 0xffff0000u);
                }
            }
        }
    };
    if constexpr (PF) {
        if (ntiles > 0) load_regs(0);
    }

    for (int kt = 0; kt < ntiles; ++kt) {
        IK_LDS_BARRIER();
        if constexpr (PF) {
            store_regs();
            if (kt + 1 < ntiles) load_regs(kt + 1);  // in flight during the barrier and this tile's math
        } else if constexpr (VP) {
            load_regs(kt);
            store_regs();
        } else {
            for (int idx = tid; idx < BK * CH; idx += NT) {
                const int r = idx / CH, ch = idx % CH;
                const int64_t key = (int64_t)kt * BK + r;
                uint4 kv = make_uint4(0, 0, 0, 0), vv = make_uint4(0, 0, 0, 0);
                if (key < kv_end) {
                    kv = *reinterpret_cast<const uint4*>(kb + key * D + ch * 8);
                    vv = *reinterpret_cast<const uint4*>(vb + key * D + ch * 8);
                }
                *reinterpret_cast<uint4*>(&Ks[r * LDK + ch * 8]) = kv;
                const uint32_t vw[4] = {vv.x, vv.y, vv.z, vv.w};
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    Vt[(ch * 8 + 2 * e) * LDV + r] = (uint16_t)(vw[e] & 0xffffu);
                    Vt[(ch * 8 + 2 * e + 1) * LDV + r] = (uint16_t)(vw[e] >> 16);
                }
            }
        }
        IK_LDS_BARRIER();

        ik_v8f sacc[NKB];
#pragma unroll
        for (int b = 0; b < NKB; ++b)
#pragma unroll
            for (int e = 0; e < 8; ++e) sacc[b][e] = 0.f;
#pragma unroll
        for (int ks = 0; ks < NKS; ++ks) {
#pragma unroll
            for (int b = 0; b < NKB; ++b) {
                const uint4 a = *reinterpret_cast<const uint4*>(&Ks[(b * 16 + lr) * LDK + ks * 16 + lk]);
                sacc[b] = wmma_bf16(a, qb[ks], sacc[b]);
            }
        }

        const bool need_mask = (int64_t)(kt + 1) * BK > mask_from;
        float mx = -INFINITY;
#pragma unroll
        for (int b = 0; b < NKB; ++b) {
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                float v = sacc[b][e] * sl2;
                if (need_mask) {
                    const int64_t key = (int64_t)kt * BK + b * 16 + lk + e;
                    if (key > pos0 + qrow) v = -INFINITY;
                }
                sacc[b][e] = v;
                mx = fmaxf(mx, v);
            }
        }
        mx = fmaxf(mx, __shfl_xor(mx, 16, 32));
        const float mn = fmaxf(m, mx);
        const float mu = mn == -INFINITY ? 0.f : mn;
        const float cr = exp2f(m - mu);
        m = mn;
        float ps = 0.f;
        uint4 pb[NKB];
#pragma unroll
        for (int b = 0; b < NKB; ++b) {
            float pv[8];
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                pv[e] = exp2f(sacc[b][e] - mu);
                ps += pv[e];
            }
            pb[b] = ik_pack8(pv);
        }
        l = l * cr + ps;
        if (!SK || __any(cr != 1.0f)) {
#pragma unroll
            for (int i = 0; i < NDB; ++i)
#pragma unroll
                for (int e = 0; e < 8; ++e) o[i][e] *= cr;
        }

#pragma unroll
        for (int b = 0; b < NKB; ++b) {
#pragma unroll
            for (int db = 0; db < NDB; ++db) {
                const uint4 a = *reinterpret_cast<const uint4*>(&Vt[(db * 16 + lr) * LDV + b * 16 + lk]);
                o[db] = wmma_bf16(a, pb[b], o[db]);
            }
        }
    }

    l += __shfl_xor(l, 16, 32);
    const float inv = 1.f / l;
    if (qrow < T) {
#pragma unroll
        for (int db = 0; db < NDB; ++db) {
            float v[8];
#pragma unroll
            for (int e = 0; e < 8; ++e) v[e] = o[db][e] * inv;
            const uint4 u = ik_pack8(v);
            uint32_t* po = reinterpret_cast<uint32_t*>(out + ((int64_t)qrow * N + h) * D + db * 16 + lk);
            po[0] = u.x;
            po[1] = u.y;
            po[2] = u.z;
            po[3] = u.w;
        }
    }
}

// INTARSIA_HIP_FLASH_HPB = 1 | 2 | 4 (default 4): query heads per block of flash v2 (capped by the GQA group); every value
// gives the same bits. Default 4 since round 6 (docs/results/2026-10-19-hip-round6.md): at the 35B's D = 256 the HPB 2 kernel
// spills (236 B scratch per lane), HPB 4 does not, and the 4408-token prefill's flash time drops 54 -> 37 ms.
constexpr int kFlashHpbDefault = 4;
// Default 5 = V pair staging + skipped no-op rescale (docs/results/2026-10-22-pf-prefill-ctx.md): bitwise the 0 kernel, the 16k
// prefill's flash time 469 -> 272 ms. 0 is the pre-2026-10-22 kernel.
constexpr int kFlashVpDefault = 5;
int flash_hpb() {
    static const int v = [] {
        const char* e = getenv("INTARSIA_HIP_FLASH_HPB");
        const int x = e ? atoi(e) : kFlashHpbDefault;
        return (x == 1 || x == 2 || x == 4) ? x : kFlashHpbDefault;
    }();
    return v;
}

// INTARSIA_HIP_FLASH_VP = 0 | 1 | 3 | 5 | 7: staging / math variants of flash v2 (bit mask, see flash_hpb_kernel); all give the same bits.
int flash_vp() {
    static const int v = [] {
        const char* e = getenv("INTARSIA_HIP_FLASH_VP");
        char* end = nullptr;
        const long x = (e && *e) ? strtol(e, &end, 10) : -1;  // 0 is a valid value: "" and "x" must not parse as 0
        const bool ok = e && *e && end && *end == '\0' && (x == 0 || x == 1 || x == 3 || x == 5 || x == 7);
        return ok ? (int)x : kFlashVpDefault;
    }();
    return v;
}

template <int D, int BK, int HPB>
int launch_hpb(const dim3& grid, int vp, const ik_bf16* q, const ik_bf16* kc, const ik_bf16* vc, ik_bf16* out, int T, int N,
               int G, int64_t cap, int64_t pos0, float sl2, hipStream_t s) {
#define IK_FLASH_HPB_CASE(VPM) \
    case VPM: flash_hpb_kernel<D, BK, HPB, VPM><<<grid, kThreads * HPB, 0, s>>>(q, kc, vc, out, T, N, G, cap, pos0, sl2); break;
    switch (vp) {
        IK_FLASH_HPB_CASE(1)
        IK_FLASH_HPB_CASE(3)
        IK_FLASH_HPB_CASE(5)
        IK_FLASH_HPB_CASE(7)
        default: flash_hpb_kernel<D, BK, HPB, 0><<<grid, kThreads * HPB, 0, s>>>(q, kc, vc, out, T, N, G, cap, pos0, sl2);
    }
#undef IK_FLASH_HPB_CASE
    return ik_launch_status();
}

template <int D, int BK>
int launch_flash_v2(const ik_bf16* q, const ik_bf16* kc, const ik_bf16* vc, ik_bf16* out, int T, int N, int NKV,
                    int64_t cap, int64_t pos0, float scale, hipStream_t s) {
    const int G = N / NKV;
    int hpb = flash_hpb();
    while (hpb > 1 && G % hpb != 0) hpb /= 2;
    const dim3 grid((T + kBQ - 1) / kBQ, N / hpb);
    const float sl2 = scale * kLog2e;
    ik_rt_stamp(s, "flash", -1);  // INTARSIA_HIP_STAMP only (round 9 diagnostics); otherwise launches nothing
    const int vp = flash_vp();
    int st;
    if (hpb == 4) st = launch_hpb<D, BK, 4>(grid, vp, q, kc, vc, out, T, N, G, cap, pos0, sl2, s);
    else if (hpb == 2) st = launch_hpb<D, BK, 2>(grid, vp, q, kc, vc, out, T, N, G, cap, pos0, sl2, s);
    else st = launch_hpb<D, BK, 1>(grid, vp, q, kc, vc, out, T, N, G, cap, pos0, sl2, s);
    ik_rt_stamp(s, "flash", 0);
    return st;
}

}  // namespace

static int flash_impl(const ik_tensor* q, const ik_tensor* kc, const ik_tensor* vc, int64_t pos0, float scale, ik_tensor* out,
                      void* stream, bool v2) {
    IK_TRY(ik_check(q, IK_DTYPE_BF16, 3));
    IK_TRY(ik_check(kc, IK_DTYPE_BF16, 3));
    IK_TRY(ik_check(vc, IK_DTYPE_BF16, 3));
    IK_TRY(ik_check(out, IK_DTYPE_BF16, 3));
    const int64_t T = q->shape[0], N = q->shape[1], D = q->shape[2];
    const int64_t NKV = kc->shape[0], cap = kc->shape[1];
    if (vc->shape[0] != NKV || vc->shape[1] != cap || vc->shape[2] != D || kc->shape[2] != D) return IK_ERR_SHAPE;
    if (out->shape[0] != T || out->shape[1] != N || out->shape[2] != D) return IK_ERR_SHAPE;
    if (NKV <= 0 || N % NKV != 0 || pos0 < 0 || pos0 + T > cap) return IK_ERR_SHAPE;
    if (D != 64 && D != 128 && D != 256) return IK_ERR_SHAPE;
    if (T == 0 || N == 0) return IK_OK;
    if (T > 0x7fffffffLL / 2 || N > 65535 || (T + kBQ - 1) / kBQ > 0x7fffffffLL) return IK_ERR_SHAPE;
    // K / V are moved in 16 B vectors, q and out in 4 B words
    if (!ik_aligned16(kc->data) || !ik_aligned16(vc->data) || (reinterpret_cast<uintptr_t>(q->data) & 3) ||
        (reinterpret_cast<uintptr_t>(out->data) & 3))
        return IK_ERR_STRIDE;
    hipStream_t s = IK_STREAM(stream);
    const ik_bf16* qp = (const ik_bf16*)q->data;
    const ik_bf16* kp = (const ik_bf16*)kc->data;
    const ik_bf16* vp = (const ik_bf16*)vc->data;
    ik_bf16* op = (ik_bf16*)out->data;
    if (v2) {
        switch (D) {
            case 64: return launch_flash_v2<64, 64>(qp, kp, vp, op, (int)T, (int)N, (int)NKV, cap, pos0, scale, s);
            case 128: return launch_flash_v2<128, 64>(qp, kp, vp, op, (int)T, (int)N, (int)NKV, cap, pos0, scale, s);
            default: return launch_flash_v2<256, 32>(qp, kp, vp, op, (int)T, (int)N, (int)NKV, cap, pos0, scale, s);
        }
    }
    switch (D) {
        case 64: return launch_flash<64, 64>(qp, kp, vp, op, (int)T, (int)N, (int)NKV, cap, pos0, scale, s);
        case 128: return launch_flash<128, 64>(qp, kp, vp, op, (int)T, (int)N, (int)NKV, cap, pos0, scale, s);
        default: return launch_flash<256, 32>(qp, kp, vp, op, (int)T, (int)N, (int)NKV, cap, pos0, scale, s);
    }
}

extern "C" int ik_attn_prefill_bf16_flash(const ik_tensor* q, const ik_tensor* kc, const ik_tensor* vc, int64_t pos0,
                                          float scale, ik_tensor* out, void* stream) {
    return flash_impl(q, kc, vc, pos0, scale, out, stream, false);
}

extern "C" int ik_hip_flash_hpb(void) { return flash_hpb(); }  // ik_hip_abi.h (registry pin)
extern "C" int ik_hip_flash_vp(void) { return flash_vp(); }    // ik_hip_abi.h (registry pin)

// v2 (HIP only, ik_hip_abi.h): flash_hpb_kernel, bitwise ik_attn_prefill_bf16_flash
extern "C" int ik_attn_prefill_bf16_flash_v2(const ik_tensor* q, const ik_tensor* kc, const ik_tensor* vc, int64_t pos0,
                                             float scale, ik_tensor* out, void* stream) {
    return flash_impl(q, kc, vc, pos0, scale, out, stream, true);
}
