// Shared by the grouped MoE kernels (moe_grouped.cu, moe_nvfp4_mma_sm120a.cu): expert grouping and the weighted reduce.
// Moved out of moe_grouped.cu unchanged.
#pragma once
#include "ik_launch.h"
#include "ik_quant.cuh"

namespace {

constexpr int kMaxExperts = 1024;

IK_DEVI float round_bf16(float v) { return ik_b2f(ik_f2b(v)); }

// ---- grouping ------------------------------------------------------------------------------------------------------

// exclusive prefix sum of one int per thread over a 1024-thread block; `wsum` = 32 ints of shared scratch
IK_DEVI int block_excl_scan(int v, int* wsum, int* total) {
    const int lane = threadIdx.x % IK_WARP, warp = threadIdx.x / IK_WARP, nw = blockDim.x / IK_WARP;
    int inc = v;
#pragma unroll
    for (int o = 1; o < IK_WARP; o <<= 1) {
        const int t = __shfl_up_sync(0xffffffffu, inc, o);
        if (lane >= o) inc += t;
    }
    __syncthreads();  // wsum may still be read by a previous scan
    if (lane == IK_WARP - 1) wsum[warp] = inc;
    __syncthreads();
    if (warp == 0) {
        int x = lane < nw ? wsum[lane] : 0;
#pragma unroll
        for (int o = 1; o < IK_WARP; o <<= 1) {
            const int t = __shfl_up_sync(0xffffffffu, x, o);
            if (lane >= o) x += t;
        }
        wsum[lane] = x;
    }
    __syncthreads();
    *total = wsum[nw - 1];
    return inc - v + (warp ? wsum[warp - 1] : 0);
}

__global__ void __launch_bounds__(1024)
moe_group_tiles_kernel(const int* __restrict__ ids, int P, int E, int BM, int* __restrict__ pairs,
                       int* __restrict__ offsets, int* __restrict__ ntiles, int2* __restrict__ tiles) {
    extern __shared__ int sh[];  // counts[E], cursor[E], wsum[32]
    int* counts = sh;
    int* cursor = sh + E;
    int* wsum = sh + 2 * E;
    for (int e = threadIdx.x; e < E; e += blockDim.x) counts[e] = 0;
    __syncthreads();
    for (int p = threadIdx.x; p < P; p += blockDim.x) {
        const int e = ids[p];
        if (e >= 0 && e < E) atomicAdd(&counts[e], 1);  // out-of-range ids are dropped (contribute nothing)
    }
    __syncthreads();
    const int e = threadIdx.x;
    const int c = e < E ? counts[e] : 0;
    const int nt = (c + BM - 1) / BM;
    int total = 0, total_tiles = 0;
    const int off = block_excl_scan(c, wsum, &total);
    const int toff = block_excl_scan(nt, wsum, &total_tiles);
    if (e < E) {
        offsets[e] = off;
        cursor[e] = off;
        for (int j = 0; j < nt; ++j) tiles[toff + j] = make_int2(e, j * BM);
    }
    if (e == 0) {
        offsets[E] = total;
        *ntiles = total_tiles;
    }
    __syncthreads();
    for (int p = threadIdx.x; p < P; p += blockDim.x) {
        const int ex = ids[p];
        if (ex >= 0 && ex < E) pairs[atomicAdd(&cursor[ex], 1)] = p;
    }
}

template <typename OutT>
__global__ void __launch_bounds__(256)
moe_reduce_kernel(const ik_bf16* __restrict__ y, const int* __restrict__ ids, const float* __restrict__ w,
                  OutT* __restrict__ out, int E, int H, int K) {
    const int t = blockIdx.x;
    const int c = blockIdx.y * 256 + threadIdx.x;
    if (c >= H) return;
    float acc = 0.f;
    for (int k = 0; k < K; ++k) {
        const int id = ids[(int64_t)t * K + k];
        if (id < 0 || id >= E) continue;
        acc += round_bf16(ik_b2f(y[((int64_t)t * K + k) * H + c]) * w[(int64_t)t * K + k]);
    }
    ik_store(out + (int64_t)t * H + c, acc);
}

inline int64_t r256(int64_t v) { return (v + 255) & ~int64_t(255); }

}  // namespace
