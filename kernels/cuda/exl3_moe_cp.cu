// EXL3 routed experts, decode windows: the grouped GEMV of the gate / up (two matrices) or down projection of every
// expert a window routes to (TensorFold's grouped_cp_kernel, EPI 0; provenance, license and changes: exl3_moe.cuh).
//
// A program is (expert place, 128 output columns, matrix x K split x 16-member tile), four warps splitting the split's K
// range; each warp copies its k tiles' trellis words 16 bytes a lane (cp.async) S - 1 steps ahead into its own ring of
// shared-memory stages, decodes each 16x16 tile straight into mma.m16n8k16 B fragments and multiplies the member rows'
// fp16 rotated input (A, a k tile ahead). The warps' partial sums go through shared memory 8 rows at a time and are added
// in warp order into Z [mats, SK, P, N] (fp32): rows stay independent, K ranges are fixed by the shape, so a row's bits
// depend on (K, N, its expert) only. Programs run expert by expert, the last place (the largest id: a shared expert
// riding as the last expert) first.
#include "exl3_moe.cuh"

namespace ik_exl3x {
namespace {

constexpr int NT = 8;   // 16-column tiles a program (128 columns)
constexpr int W = 4;    // warps a program
constexpr int RR = 8;   // rows a pass of the warps' reduction
// ring stages by width: 3 up to 5 bits, 2 above, so the ring's shared memory is that of 8 bits at 2 stages (16 KB with
// the reduction's 16 KB, four programs an SM); the stage count changes nothing but how far the copies run ahead
template <int K2>
constexpr int stages() { return K2 <= 10 ? 3 : 2; }
constexpr int RING_WORDS = 2 * NT * 4 * 16;  // a warp's ring: max over widths of stages * NT * TW (8 bits, 2 stages)

template <int CB, int K2>
__device__ __forceinline__ void warp_tiles_cp(const uint32_t* __restrict__ T, int NTILES, int kt0, int nkt, int nt0,
                                              const half* x0, const half* x1, bool ok0, bool ok1, int lane,
                                              uint32_t* __restrict__ ring, float (&acc)[NT][2][4]) {
    constexpr int TW = Fmt<K2>::TW, LW = Fmt<K2>::LW, S = stages<K2>();
    constexpr int SW = NT * TW;                // words a stage
    constexpr int NV = NT * TW / 4;            // 16-byte chunks of a step's NT tiles (contiguous)
    constexpr int VPL = (NV + 31) / 32;
    static_assert(S * SW <= RING_WORDS, "the ring holds S stages");
    const LaneMap<K2> map(lane);
    const size_t kstride = (size_t)NTILES * TW;
    const uint32_t* tp = T + ((size_t)kt0 * NTILES + nt0) * TW;
    auto issue = [&](int it) {
        uint32_t* dst = ring + (it % S) * SW;
        const uint32_t* src = tp + (size_t)it * kstride;
#pragma unroll
        for (int v = 0; v < VPL; ++v) {
            const int c = v * 32 + lane;
            if ((NV % 32) == 0 || c < NV) cp_async16(dst + 4 * c, src + 4 * c, true);
        }
    };
#pragma unroll
    for (int d = 0; d < S - 1; ++d) {
        if (d < nkt) issue(d);
        cp_async_commit();
    }
    uint32_t an[4] = {load_pair(x0 + kt0 * 16, ok0), load_pair(x1 + kt0 * 16, ok1), load_pair(x0 + kt0 * 16 + 8, ok0),
                      load_pair(x1 + kt0 * 16 + 8, ok1)};
    for (int it = 0; it < nkt; ++it) {
        if (it + S - 1 < nkt) issue(it + S - 1);
        cp_async_commit();
        cp_async_wait<S - 1>();  // this step's copies (this lane's) are done
        __syncwarp();            // ... and every lane's are visible
        const uint32_t* st = ring + (it % S) * SW + lane;
        uint32_t w[NT][LW];
#pragma unroll
        for (int i = 0; i < NT; ++i)
#pragma unroll
            for (int l = 0; l < LW; ++l) w[i][l] = ((TW % 32) == 0 || l * 32 + lane < TW) ? st[i * TW + l * 32] : 0u;
        __syncwarp();  // the stage is read before a later step refills it
        const uint32_t a[4] = {an[0], an[1], an[2], an[3]};
        if (it + 1 < nkt) {
            const int k = (kt0 + it + 1) * 16;
            an[0] = load_pair(x0 + k, ok0);
            an[1] = load_pair(x1 + k, ok1);
            an[2] = load_pair(x0 + k + 8, ok0);
            an[3] = load_pair(x1 + k + 8, ok1);
        }
#pragma unroll
        for (int i = 0; i < NT; ++i) {
            uint32_t b0[2], b1[2];
            decode_tile<CB, K2>(w[i], map, lane, b0, b1);
            mma16816(acc[i][0], a, b0);
            mma16816(acc[i][1], a, b1);
        }
    }
}

__global__ void __launch_bounds__(W * 32, 4) grouped_cp_kernel(GroupedArgs a) {
    constexpr int RING = W * RING_WORDS;
    constexpr int RED = W * RR * NT * 16;
    __shared__ __align__(16) uint32_t smem[RING > RED ? RING : RED];
    __shared__ int rows_sh[16];
    const int K = a.K, N = a.N, P = a.P, SK = a.SK, maxm = a.maxm, slots = a.slots;
    // expert-major order, the last expert place first
    const int MT = (maxm + 15) / 16;
    const int NY = gridDim.y, NZ = gridDim.z;
    const int lin = blockIdx.x + gridDim.x * (blockIdx.y + NY * blockIdx.z);
    const int ur = lin / (NY * NZ), rem = lin % (NY * NZ);
    const int nu = a.ucount[0];
    if (ur >= nu) return;
    const int u = nu - 1 - ur, by = rem % NY, bz = rem / NY;
    const int mtile = bz % MT;
    const int split = (bz / MT) % SK;
    const int mat = bz / MT / SK;
    const half* X = mat ? a.x1 : a.x0;
    const int e = a.uids[u];
    const int cb = *a.ex.cbp;
    int k2 = 0;
    const uint32_t* T = mat_words(a.ex, e, a.j0 + mat, K, N, cb, k2);
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int g = lane >> 2, t = lane & 3;
    const int KT = K >> 4, NTILES = N >> 4;

    if (threadIdx.x < 16) {
        const int m = mtile * 16 + threadIdx.x;
        const int code = m < a.mcount[u] ? a.members[a.mstart[u] + m] : -1;
        rows_sh[threadIdx.x] = code >= 0 ? (code >> 5) * slots + (code & 31) : -1;
    }
    __syncthreads();
    if (rows_sh[0] < 0) return;  // members come first, so this tile is empty
    const int r0 = rows_sh[g], r1 = rows_sh[g + 8];
    const half* x0 = X + (size_t)(r0 < 0 ? 0 : r0) * K + 2 * t;
    const half* x1 = X + (size_t)(r1 < 0 ? 0 : r1) * K + 2 * t;

    const int per_split = KT / SK, per_warp = per_split / W;
    const int kt0 = split * per_split + warp * per_warp;
    const int nt0 = by * NT;
    uint32_t* ring = smem + warp * RING_WORDS;

    float acc[NT][2][4];
#pragma unroll
    for (int i = 0; i < NT; ++i)
#pragma unroll
        for (int h = 0; h < 2; ++h)
#pragma unroll
            for (int c = 0; c < 4; ++c) acc[i][h][c] = 0.f;

    const bool ran = T != nullptr && with_width(cb, k2, [&](auto cbc, auto k2c) {
                         warp_tiles_cp<decltype(cbc)::value, decltype(k2c)::value>(T, NTILES, kt0, per_warp, nt0, x0, x1,
                                                                                    r0 >= 0, r1 >= 0, lane, ring, acc);
                     });
    if (!ran) {
#pragma unroll
        for (int i = 0; i < NT; ++i)
#pragma unroll
            for (int h = 0; h < 2; ++h)
#pragma unroll
                for (int c = 0; c < 4; ++c) acc[i][h][c] = __int_as_float(0x7fc00000);
    }
    cp_async_wait<0>();
    __syncthreads();  // every warp is done with its ring: it becomes red

    // the warps' partial sums through shared memory, added in warp order: rows 0-7, then rows 8-15 when the tile has them
    float(*red)[RR][NT * 16] = reinterpret_cast<float(*)[RR][NT * 16]>(smem);
#pragma unroll
    for (int hp = 0; hp < 16 / RR; ++hp) {
        if (hp > 0) {
            if (rows_sh[8] < 0) break;
            __syncthreads();  // the first pass is read before red is overwritten
        }
#pragma unroll
        for (int i = 0; i < NT; ++i)
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int col = i * 16 + h * 8 + 2 * t;
                red[warp][g][col] = acc[i][h][2 * hp];  // row g (pass 0) or g + 8 (pass 1)
                red[warp][g][col + 1] = acc[i][h][2 * hp + 1];
            }
        __syncthreads();
        for (int idx = threadIdx.x; idx < RR * NT * 16; idx += W * 32) {
            const int row = idx / (NT * 16), col = idx % (NT * 16);
            const int r = rows_sh[row + RR * hp];
            if (r < 0) continue;
            float s = red[0][row][col];
#pragma unroll
            for (int w = 1; w < W; ++w) s += red[w][row][col];
            a.z[(((size_t)mat * SK + split) * P + r) * N + nt0 * 16 + col] = s;
        }
    }
}

// Dense decode linears on the same loader (ik_matmul_exl3_cp, exl3.cu): one matrix, rows M <= 16. A program is (128 output
// columns, K split); its four warps run warp_tiles_cp over their share of the split and add in warp order. The raw partial,
// before H and svh, goes to f32 part [S, M, N]; split_finish_kernel (exl3.cu) adds the S partials in split order and finishes.
__global__ void __launch_bounds__(W * 32, 4) dense_cp_kernel(const half* __restrict__ xh, const uint32_t* __restrict__ T,
                                                             float* __restrict__ part, const int* __restrict__ cbp, int k2,
                                                             int M, int K, int N, int S) {
    constexpr int RING = W * RING_WORDS;
    constexpr int RED = W * RR * NT * 16;
    __shared__ __align__(16) uint32_t smem[RING > RED ? RING : RED];
    const int nb = blockIdx.x, sp = blockIdx.y;
    const int cb = *cbp;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int g = lane >> 2, t = lane & 3;
    const int KT = K >> 4, NTILES = N >> 4;
    const bool ok0 = g < M, ok1 = g + 8 < M;
    const half* x0 = xh + (size_t)(ok0 ? g : 0) * K + 2 * t;
    const half* x1 = xh + (size_t)(ok1 ? g + 8 : 0) * K + 2 * t;
    const int per_split = KT / S, per_warp = per_split / W;
    const int kt0 = sp * per_split + warp * per_warp;
    const int nt0 = nb * NT;
    uint32_t* ring = smem + warp * RING_WORDS;

    float acc[NT][2][4];
#pragma unroll
    for (int i = 0; i < NT; ++i)
#pragma unroll
        for (int h = 0; h < 2; ++h)
#pragma unroll
            for (int c = 0; c < 4; ++c) acc[i][h][c] = 0.f;

    const bool ran = with_width(cb, k2, [&](auto cbc, auto k2c) {
        warp_tiles_cp<decltype(cbc)::value, decltype(k2c)::value>(T, NTILES, kt0, per_warp, nt0, x0, x1, ok0, ok1, lane, ring, acc);
    });
    if (!ran) {
#pragma unroll
        for (int i = 0; i < NT; ++i)
#pragma unroll
            for (int h = 0; h < 2; ++h)
#pragma unroll
                for (int c = 0; c < 4; ++c) acc[i][h][c] = __int_as_float(0x7fc00000);
    }
    cp_async_wait<0>();
    __syncthreads();  // every warp is done with its ring: it becomes red

    float(*red)[RR][NT * 16] = reinterpret_cast<float(*)[RR][NT * 16]>(smem);
#pragma unroll
    for (int hp = 0; hp < 16 / RR; ++hp) {
        if (hp > 0) {
            if (M <= RR) break;
            __syncthreads();  // the first pass is read before red is overwritten
        }
#pragma unroll
        for (int i = 0; i < NT; ++i)
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int col = i * 16 + h * 8 + 2 * t;
                red[warp][g][col] = acc[i][h][2 * hp];  // row g (pass 0) or g + 8 (pass 1)
                red[warp][g][col + 1] = acc[i][h][2 * hp + 1];
            }
        __syncthreads();
        for (int idx = threadIdx.x; idx < RR * NT * 16; idx += W * 32) {
            const int row = idx / (NT * 16), col = idx % (NT * 16);
            const int r = row + RR * hp;
            if (r >= M) continue;
            float v = red[0][row][col];
#pragma unroll
            for (int w = 1; w < W; ++w) v += red[w][row][col];
            part[((size_t)sp * M + r) * N + nt0 * 16 + col] = v;
        }
    }
}

}  // namespace

// grid: (N / 128, S); K / 16 a multiple of 4 * S, N of 128, T 16-aligned (cp.async)
int launch_dense_cp(const half* xh, const uint32_t* T, float* part, const int* cbp, int k2, int M, int K, int N, int S,
                    cudaStream_t s) {
    if (N % (16 * NT) != 0 || S < 1 || (K / 16) % (W * S) != 0 || (((uintptr_t)T) & 15u) != 0) return IK_ERR_SHAPE;
    dense_cp_kernel<<<dim3((unsigned)(N / (16 * NT)), (unsigned)S), W * 32, 0, s>>>(xh, T, part, cbp, k2, M, K, N, S);
    return ik_launch_status();
}

// grid: (expert places, N / 128, mats * SK * member tiles); K / 16 a multiple of 4 * SK, N of 128
int launch_cp(const GroupedArgs& a, cudaStream_t s) {
    if (a.N % (16 * NT) != 0 || (a.K / 16) % (W * a.SK) != 0 || a.nexp <= 0) return IK_ERR_SHAPE;
    const int MT = (a.maxm + 15) / 16;
    const long long gz = (long long)a.mats * a.SK * MT;
    if (gz > 65535) return IK_ERR_SHAPE;
    grouped_cp_kernel<<<dim3((unsigned)a.nexp, (unsigned)(a.N / (16 * NT)), (unsigned)gz), W * 32, 0, s>>>(a);
    return ik_launch_status();
}

}  // namespace ik_exl3x
