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

constexpr int NT = 8, W = 4, PF = 1, G = 2;
constexpr int GR = 16 * G;
constexpr int TPT = 16 * NT * 16 / (W * 32);  // a thread's outputs of a member tile in the warp sum

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

template <bool FOLD>
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

}  // namespace

// grid: (1, N / 128, mats * (fold ? 1 : SK) * work-list length)
int launch_rows(const GroupedArgs& a, bool fold, cudaStream_t s) {
    if (a.N % (16 * NT) != 0 || (a.K / 16) % (W * a.SK) != 0 || a.nexp <= 0 || a.work == nullptr) return IK_ERR_SHAPE;
    const long long gz = (long long)a.mats * (fold ? 1 : a.SK) * a.nexp;
    if (gz > 65535) return IK_ERR_SHAPE;
    const dim3 grid(1, (unsigned)(a.N / (16 * NT)), (unsigned)gz);
    if (fold)
        grouped_rows_kernel<true><<<grid, W * 32, 0, s>>>(a);
    else
        grouped_rows_kernel<false><<<grid, W * 32, 0, s>>>(a);
    return ik_launch_status();
}

}  // namespace ik_exl3x
