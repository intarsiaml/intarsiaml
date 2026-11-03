// EXL3 routed experts, prompt chunks: the grouped kernel that decodes each weight tile once for G member tiles (TensorFold's
// grouped_rows_kernel<CB, 8, 4, 1, LO, HI, 2, FOLD>; provenance, license and changes: exl3_moe.cuh).
//
// A program is (work-list entry = (expert place, group of 32 members), 128 output columns, matrix [x K split]); four warps
// split the K range; each k tile's weights are decoded once and fed to both 16-row member tiles (the mma sequence of a
// row is the decode kernel's: the same k tiles in the same order from zero, warps added in order). FOLD (gate / up): one
// program runs the K splits in order and keeps their sum from 0 in Z [mats, P, N] (0 + s0, then + s1, ..: the decode
// epilogue's order over its Z splits); without (down: one split) it writes Z [mats, SK, P, N]. The work list (exl3_moe.cu work_list_kernel) only names groups an expert has, so no program is
// launched for an empty group and an expert's programs run back to back.
#include "exl3_moe.cuh"

namespace ik_exl3x {
namespace {

constexpr int NT = 8, W = 4, PF = 2, G = 2;  // PF: k tiles of weight words in flight per warp (1: 6.38 s, 2: 5.63 s, 3: 5.88 s for a 4.1K prompt, docs/results/2026-11-02-dsv41-prefill.md)
constexpr int GR = 16 * G;
constexpr int TPT = 16 * NT * 16 / (W * 32);  // a thread's outputs of a member tile in the warp sum

// The widths the staged kernel below is compiled for (mul1 codebook, k2 4 / 6 / 8 / 10: Mia's experts are 4 and 6). Every
// other (codebook, width) an EXL3 matrix can have stays on the kernel after it: a launch of the staged kernel is followed by
// one of that kernel with SKIP set, and each matrix is run by exactly one of the two (same predicate, same device data).
__device__ __forceinline__ bool fast_width(int cb, int k2) { return cb == 2 && (k2 == 4 || k2 == 6 || k2 == 8 || k2 == 10); }
template <class F>
__device__ __forceinline__ void with_fast_width(int k2, F&& f) {
    switch (k2) {
        case 4: f(std::integral_constant<int, 2>{}, std::integral_constant<int, 4>{}); break;
        case 6: f(std::integral_constant<int, 2>{}, std::integral_constant<int, 6>{}); break;
        case 8: f(std::integral_constant<int, 2>{}, std::integral_constant<int, 8>{}); break;
        default: f(std::integral_constant<int, 2>{}, std::integral_constant<int, 10>{}); break;
    }
}

template <int CB, int K2>
__device__ __forceinline__ void warp_tiles_rows(const uint32_t* __restrict__ T, int NTILES, int kt0, int nkt, int nt0,
                                                const half* __restrict__ X, const long long (&off)[G][2], int live,
                                                int lane, float (&acc)[G][NT][2][4]) {
    constexpr int TW = Fmt<K2>::TW, LW = Fmt<K2>::LW;
    const LaneMap<K2> map(lane);
    const size_t kstride = (size_t)NTILES * TW;
    const uint32_t* tp = T + ((size_t)kt0 * NTILES + nt0) * TW + lane;

    uint32_t pf[PF][NT][LW];
#pragma unroll
    for (int d = 0; d < PF; ++d)
        if (d < nkt)
#pragma unroll
            for (int i = 0; i < NT; ++i) load_words<K2>(pf[d][i], tp + d * kstride + i * TW, lane);

    // the rows' A fragments a k tile ahead (the loads' latency behind this tile's decode and mma)
    const half* xr[G][2];
    bool ok[G][2];
#pragma unroll
    for (int g = 0; g < G; ++g)
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            ok[g][j] = off[g][j] >= 0;
            xr[g][j] = X + (ok[g][j] ? off[g][j] : 0) + kt0 * 16;
        }
    uint32_t an[G][4];
#pragma unroll
    for (int g = 0; g < G; ++g)
        if (g < live) {
            an[g][0] = load_pair(xr[g][0], ok[g][0]);
            an[g][1] = load_pair(xr[g][1], ok[g][1]);
            an[g][2] = load_pair(xr[g][0] + 8, ok[g][0]);
            an[g][3] = load_pair(xr[g][1] + 8, ok[g][1]);
        }

    for (int ib = 0; ib < nkt; ib += PF) {
#pragma unroll
        for (int d = 0; d < PF; ++d) {
            const int it = ib + d;
            if (it < nkt) {
                uint32_t w[NT][LW];
#pragma unroll
                for (int i = 0; i < NT; ++i)
#pragma unroll
                    for (int l = 0; l < LW; ++l) w[i][l] = pf[d][i][l];
                if (it + PF < nkt)
#pragma unroll
                    for (int i = 0; i < NT; ++i) load_words<K2>(pf[d][i], tp + (size_t)(it + PF) * kstride + i * TW, lane);
                uint32_t a[G][4];
#pragma unroll
                for (int g = 0; g < G; ++g)
#pragma unroll
                    for (int c = 0; c < 4; ++c) a[g][c] = an[g][c];
                if (it + 1 < nkt) {
                    const int k = (it + 1) * 16;
#pragma unroll
                    for (int g = 0; g < G; ++g)
                        if (g < live) {
                            an[g][0] = load_pair(xr[g][0] + k, ok[g][0]);
                            an[g][1] = load_pair(xr[g][1] + k, ok[g][1]);
                            an[g][2] = load_pair(xr[g][0] + k + 8, ok[g][0]);
                            an[g][3] = load_pair(xr[g][1] + k + 8, ok[g][1]);
                        }
                }
#pragma unroll
                for (int i = 0; i < NT; ++i) {
                    uint32_t b0[2], b1[2];
                    decode_tile<CB, K2>(w[i], map, lane, b0, b1);
#pragma unroll
                    for (int g = 0; g < G; ++g) {
                        if (g < live) {
                            mma16816(acc[g][i][0], a[g], b0);
                            mma16816(acc[g][i][1], a[g], b1);
                        }
                    }
                }
            }
        }
    }
}

template <bool FOLD, bool SKIP>
__global__ void __launch_bounds__(W * 32) grouped_rows_kernel(GroupedArgs a) {
    const int K = a.K, N = a.N, P = a.P, SK = a.SK, slots = a.slots, nexp = a.nexp;
    // grid.z walks the work list's (expert place, member group) pairs, then (matrix, split)
    const int zz = (int)(blockIdx.z % nexp);
    const int u = a.work[2 * zz];
    if (u >= a.ucount[0]) return;
    const int mg = a.work[2 * zz + 1];
    const int zi = (int)(blockIdx.z / nexp);
    const int splits = FOLD ? 1 : SK;
    const int split0 = FOLD ? 0 : zi % SK;
    const int mat = zi / splits;
    const half* X = mat ? a.x1 : a.x0;
    const int e = a.uids[u];
    const int cb = *a.ex.cbp;
    int k2 = 0;
    const uint32_t* T = mat_words(a.ex, e, a.j0 + mat, K, N, cb, k2);
    if (SKIP && T != nullptr && fast_width(cb, k2)) return;  // the staged kernel ran this matrix
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int g8 = lane >> 2, t = lane & 3;
    const int KT = K >> 4, NTILES = N >> 4;

    __shared__ int rows_sh[GR];
    for (int i = threadIdx.x; i < GR; i += W * 32) {
        const int m = mg * GR + i;
        const int code = m < a.mcount[u] ? a.members[a.mstart[u] + m] : -1;
        rows_sh[i] = code >= 0 ? (code >> 5) * slots + (code & 31) : -1;
    }
    __syncthreads();
    if (rows_sh[0] < 0) return;  // members come first, so this group is empty
    int live = 0;                // its occupied tiles are a prefix
#pragma unroll
    for (int g = 0; g < G; ++g) live += rows_sh[g * 16] >= 0;
    long long off[G][2];
#pragma unroll
    for (int g = 0; g < G; ++g) {
        const int r0 = rows_sh[g * 16 + g8], r1 = rows_sh[g * 16 + g8 + 8];
        off[g][0] = r0 < 0 ? -1 : (long long)r0 * K + 2 * t;
        off[g][1] = r1 < 0 ? -1 : (long long)r1 * K + 2 * t;
    }

    const int per_split = KT / SK, per_warp = per_split / W;
    const int nt0 = blockIdx.y * NT;
    __shared__ float red[W][16][NT * 16];

    for (int split = split0; split < (FOLD ? SK : split0 + 1); ++split) {
        const int kt0 = split * per_split + warp * per_warp;
        float acc[G][NT][2][4];
#pragma unroll
        for (int g = 0; g < G; ++g)
#pragma unroll
            for (int i = 0; i < NT; ++i)
#pragma unroll
                for (int h = 0; h < 2; ++h)
#pragma unroll
                    for (int c = 0; c < 4; ++c) acc[g][i][h][c] = 0.f;
        const bool ran = T != nullptr && with_width(cb, k2, [&](auto cbc, auto k2c) {
                             warp_tiles_rows<decltype(cbc)::value, decltype(k2c)::value>(T, NTILES, kt0, per_warp, nt0, X,
                                                                                          off, live, lane, acc);
                         });
        if (!ran) {
#pragma unroll
            for (int g = 0; g < G; ++g)
#pragma unroll
                for (int i = 0; i < NT; ++i)
#pragma unroll
                    for (int h = 0; h < 2; ++h)
#pragma unroll
                        for (int c = 0; c < 4; ++c) acc[g][i][h][c] = __int_as_float(0x7fc00000);
        }

        // warps' partial sums through shared memory, added in warp order, one member tile after another
#pragma unroll
        for (int g = 0; g < G; ++g) {
            if (g < live) {
#pragma unroll
                for (int i = 0; i < NT; ++i)
#pragma unroll
                    for (int h = 0; h < 2; ++h) {
                        const int col = i * 16 + h * 8 + 2 * t;
                        red[warp][g8][col] = acc[g][i][h][0];
                        red[warp][g8][col + 1] = acc[g][i][h][1];
                        red[warp][g8 + 8][col] = acc[g][i][h][2];
                        red[warp][g8 + 8][col + 1] = acc[g][i][h][3];
                    }
                __syncthreads();
#pragma unroll
                for (int j = 0; j < TPT; ++j) {
                    const int idx = threadIdx.x + j * W * 32;
                    const int row = idx / (NT * 16), col = idx % (NT * 16);
                    float s = red[0][row][col];
#pragma unroll
                    for (int w = 1; w < W; ++w) s += red[w][row][col];
                    const int r = rows_sh[g * 16 + row];
                    if (r >= 0) {
                        if constexpr (FOLD) {
                            // the splits' sum from 0 in split order, kept in Z (this thread alone owns the element)
                            float* zp = a.z + ((size_t)mat * P + r) * N + nt0 * 16 + col;
                            *zp = (split == 0 ? 0.f : *zp) + s;
                        } else {
                            a.z[(((size_t)mat * SK + split) * P + r) * N + nt0 * 16 + col] = s;
                        }
                    }
                }
                __syncthreads();
            }
        }
    }
}


// ---- gate / up: the same programs with the activation rows staged through shared memory (TensorFold's grouped_mma path) ----
// The kernel above loads each A fragment word by word from global memory (4 loads a k tile and member tile, a lane each); on a
// 2048-row chunk those loads are the largest single cost of the launch (knockout in docs/results/2026-11-02-dsv41-moegate.md).
// Here one block of 8 warps is (4 K quarters) x (2 column groups of 64): the quarter's 32 member rows x KB k tiles go to
// shared memory with cp.async (3 stages), the A fragments of a k tile come from ldmatrix, and the two column groups of a
// quarter share the staged rows. The weights, the decode, the mma sequence of every output element (same k tiles, same
// order, from zero, quarters added in order, splits folded in order) are the kernel above's, so the bits are the same.
constexpr int SCS = 2, SNTW = 4, SKB = 4, SST = 3;    // column groups, n tiles a warp, k tiles a stage, stages
constexpr int SNTB = SNTW * SCS;                      // n tiles a block (= NT: the same 128 columns, the same grid)
constexpr int SNTHR = W * SCS * 32;
constexpr int SASTR = SKB * 16 + 8;                   // halves per staged row (144 B: ldmatrix without bank conflicts)
constexpr int STPT = 16 * SNTB * 16 / SNTHR;
constexpr size_t kSmemA = (size_t)SST * W * GR * SASTR * sizeof(half);
constexpr size_t kSmemRed = (size_t)W * 16 * SNTB * 16 * sizeof(float);
constexpr size_t kSmemBytes = kSmemA > kSmemRed ? kSmemA : kSmemRed;

__device__ __forceinline__ uint32_t smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void cp_async16(uint32_t dst, const void* src, bool ok) {
    const int sz = ok ? 16 : 0;  // 0: nothing is read, 16 bytes of zeros are written (an empty member row)
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(src), "r"(sz));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void cp_wait() {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}
__device__ __forceinline__ void ldsm4(uint32_t (&r)[4], uint32_t addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(addr));
}

__global__ void __launch_bounds__(SNTHR) grouped_rows_smem_kernel(GroupedArgs a) {
    const int K = a.K, N = a.N, P = a.P, SK = a.SK, slots = a.slots, nexp = a.nexp;
    const int zz = (int)(blockIdx.z % nexp);
    const int u = a.work[2 * zz];
    if (u >= a.ucount[0]) return;
    const int mg = a.work[2 * zz + 1];
    const int mat = (int)(blockIdx.z / nexp);  // FOLD: one program per (group, matrix)
    const half* X = mat ? a.x1 : a.x0;
    const int e = a.uids[u];
    const int cb = *a.ex.cbp;
    int k2 = 0;
    const uint32_t* T = mat_words(a.ex, e, a.j0 + mat, K, N, cb, k2);
    if (T == nullptr || !fast_width(cb, k2)) return;  // the kernel above (SKIP) runs this matrix
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int wq = warp % W, cs = warp / W;
    const int g8 = lane >> 2, t = lane & 3;
    const int KT = K >> 4, NTILES = N >> 4;

    __shared__ int rows_sh[GR];
    extern __shared__ __align__(16) unsigned char smem_raw[];
    half* Asm = reinterpret_cast<half*>(smem_raw);                                   // [SST][W][GR][SASTR]
    float (*red)[16][SNTB * 16] = reinterpret_cast<float (*)[16][SNTB * 16]>(smem_raw);  // after the k loop: [W][16][SNTB * 16]
    for (int i = threadIdx.x; i < GR; i += SNTHR) {
        const int m = mg * GR + i;
        const int code = m < a.mcount[u] ? a.members[a.mstart[u] + m] : -1;
        rows_sh[i] = code >= 0 ? (code >> 5) * slots + (code & 31) : -1;
    }
    __syncthreads();
    if (rows_sh[0] < 0) return;  // members come first, so this group is empty
    int live = 0;                // its occupied tiles are a prefix
#pragma unroll
    for (int g = 0; g < G; ++g) live += rows_sh[g * 16] >= 0;

    const int per_split = KT / SK, per_warp = per_split / W;
    const int nchunk = per_warp / SKB;
    const int nt0 = blockIdx.y * SNTB + cs * SNTW;
    const int lrow = (lane & 7) + ((lane >> 3) & 1) * 8, lcol = (lane >> 4) * 8;  // this lane's ldmatrix row address

    for (int split = 0; split < SK; ++split) {
        float acc[G][SNTW][2][4];
#pragma unroll
        for (int g = 0; g < G; ++g)
#pragma unroll
            for (int i = 0; i < SNTW; ++i)
#pragma unroll
                for (int h = 0; h < 2; ++h)
#pragma unroll
                    for (int c = 0; c < 4; ++c) acc[g][i][h][c] = 0.f;
        const int kbase = split * per_split;  // quarter wq starts at k tile kbase + wq * per_warp
        auto issue = [&](int chunk) {
            constexpr int SEG = GR * (SKB * 2);  // 16-byte segments of one quarter's stage
            constexpr int PER = W * SEG / SNTHR;
#pragma unroll
            for (int j = 0; j < PER; ++j) {
                const int idx = threadIdx.x + j * SNTHR;
                const int q = idx / SEG, rem = idx % SEG;
                const int row = rem / (SKB * 2), seg = rem % (SKB * 2);
                const int r = rows_sh[row];
                const half* src = X + (r >= 0 ? (size_t)r * K : 0) + (size_t)(kbase + q * per_warp + chunk * SKB) * 16 + seg * 8;
                cp_async16(smem_u32(Asm + (((size_t)(chunk % SST) * W + q) * GR + row) * SASTR + seg * 8), src, r >= 0);
            }
        };
#pragma unroll
        for (int s0 = 0; s0 < SST - 1; ++s0) {
            if (s0 < nchunk) issue(s0);
            cp_commit();
        }
        with_fast_width(k2, [&](auto cbc, auto k2c) {
                             constexpr int CBc = decltype(cbc)::value, K2c = decltype(k2c)::value;
                             constexpr int TW = Fmt<K2c>::TW, LW = Fmt<K2c>::LW;
                             const LaneMap<K2c> map(lane);
                             const size_t kstride = (size_t)NTILES * TW;
                             const uint32_t* tp = T + ((size_t)(kbase + wq * per_warp) * NTILES + nt0) * TW + lane;
                             const int nkt = per_warp;
                             uint32_t pf[PF][SNTW][LW];
#pragma unroll
                             for (int d = 0; d < PF; ++d)
                                 if (d < nkt)
#pragma unroll
                                     for (int i = 0; i < SNTW; ++i) load_words<K2c>(pf[d][i], tp + d * kstride + i * TW, lane);
                             for (int chunk = 0; chunk < nchunk; ++chunk) {
                                 cp_wait<SST - 2>();
                                 __syncthreads();  // the chunk has landed for every warp; the stage refilled below is free
                                 if (chunk + SST - 1 < nchunk) issue(chunk + SST - 1);
                                 cp_commit();
                                 const half* As = Asm + (((size_t)(chunk % SST) * W + wq) * GR) * SASTR;
#pragma unroll
                                 for (int kk = 0; kk < SKB; ++kk) {
                                     const int it = chunk * SKB + kk;
                                     uint32_t w[SNTW][LW];
#pragma unroll
                                     for (int i = 0; i < SNTW; ++i)
#pragma unroll
                                         for (int l = 0; l < LW; ++l) w[i][l] = pf[kk % PF][i][l];
                                     if (it + PF < nkt)
#pragma unroll
                                         for (int i = 0; i < SNTW; ++i)
                                             load_words<K2c>(pf[kk % PF][i], tp + (size_t)(it + PF) * kstride + i * TW, lane);
                                     uint32_t af[G][4];
#pragma unroll
                                     for (int g = 0; g < G; ++g) ldsm4(af[g], smem_u32(As + (size_t)(g * 16 + lrow) * SASTR + kk * 16 + lcol));
#pragma unroll
                                     for (int i = 0; i < SNTW; ++i) {
                                         uint32_t b0[2], b1[2];
                                         decode_tile<CBc, K2c>(w[i], map, lane, b0, b1);
#pragma unroll
                                         for (int g = 0; g < G; ++g)
                                             if (g < live) {
                                                 mma16816(acc[g][i][0], af[g], b0);
                                                 mma16816(acc[g][i][1], af[g], b1);
                                             }
                                     }
                                 }
                             }
                         });
        cp_wait<0>();
        __syncthreads();  // the staging area becomes the reduction area
        // quarters' partial sums through shared memory, added in quarter order, one member tile after another
#pragma unroll
        for (int g = 0; g < G; ++g) {
            if (g < live) {
#pragma unroll
                for (int i = 0; i < SNTW; ++i)
#pragma unroll
                    for (int h = 0; h < 2; ++h) {
                        const int col = (cs * SNTW + i) * 16 + h * 8 + 2 * t;
                        red[wq][g8][col] = acc[g][i][h][0];
                        red[wq][g8][col + 1] = acc[g][i][h][1];
                        red[wq][g8 + 8][col] = acc[g][i][h][2];
                        red[wq][g8 + 8][col + 1] = acc[g][i][h][3];
                    }
                __syncthreads();
#pragma unroll
                for (int j = 0; j < STPT; ++j) {
                    const int idx = threadIdx.x + j * SNTHR;
                    const int row = idx / (SNTB * 16), col = idx % (SNTB * 16);
                    float sm = red[0][row][col];
#pragma unroll
                    for (int w = 1; w < W; ++w) sm += red[w][row][col];
                    const int r = rows_sh[g * 16 + row];
                    if (r >= 0) {
                        float* zp = a.z + ((size_t)mat * P + r) * N + blockIdx.y * SNTB * 16 + col;
                        *zp = (split == 0 ? 0.f : *zp) + sm;  // the splits' sum from 0 in split order
                    }
                }
                __syncthreads();
            }
        }
    }
}

// INTARSIA_DSV41_MOE_SMEMA=0: gate / up on the kernel above (per-lane global A loads) for every shape; any other value, or unset: staged A
static bool smem_a_enabled() {
    static int on = -1;
    if (on < 0) {
        const char* e = getenv("INTARSIA_DSV41_MOE_SMEMA");
        on = (e && e[0] == '0' && e[1] == 0) ? 0 : 1;
    }
    return on == 1;
}

}  // namespace

// grid: (1, N / 128, mats * (fold ? 1 : SK) * work-list length)
int launch_rows(const GroupedArgs& a, bool fold, cudaStream_t s) {
    if (a.N % (16 * NT) != 0 || (a.K / 16) % (W * a.SK) != 0 || a.nexp <= 0 || a.work == nullptr) return IK_ERR_SHAPE;
    const long long gz = (long long)a.mats * (fold ? 1 : a.SK) * a.nexp;
    if (gz > 65535) return IK_ERR_SHAPE;
    const dim3 grid(1, (unsigned)(a.N / (16 * NT)), (unsigned)gz);
    if (fold && smem_a_enabled() && (((a.K / 16) / a.SK) / W) % SKB == 0 && a.K % 8 == 0) {
        static const cudaError_t attr = cudaFuncSetAttribute(grouped_rows_smem_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)kSmemBytes);
        if (attr != cudaSuccess) return IK_ERR_LAUNCH;
        grouped_rows_smem_kernel<<<grid, SNTHR, kSmemBytes, s>>>(a);
        grouped_rows_kernel<true, true><<<grid, W * 32, 0, s>>>(a);  // the widths the staged kernel is not compiled for
    } else if (fold)
        grouped_rows_kernel<true, false><<<grid, W * 32, 0, s>>>(a);
    else
        grouped_rows_kernel<false, false><<<grid, W * 32, 0, s>>>(a);
    return ik_launch_status();
}

}  // namespace ik_exl3x
