/* Small-row NVFP4 GEMM on the matrix cores (lane wg): the NVFP4 half of the WMMA family, W4A16, M = 1 .. 64 rows. Included by
 * nvfp4.hip inside its anonymous namespace; not a standalone header.
 *
 * y[m, n] = global_scale * sum_k x[m, k] * (e2m1(w[n, k]) * e4m3(bs[n, k / 16]))     bf16 x, f32 accumulate
 *
 * The block layout is that of ik_f8_rows.h (8 waves, one 16-feature tile, weights = the A operand, activations = the B operand,
 * TM token tiles, the eight waves meet once in LDS and add in the order 0 .. 7). Here a wave owns a CONTIGUOUS range of K: K is cut
 * into super-chunks of 64 k, super-chunk s belongs to wave s / cpw (cpw = ceil(ceil(K / 64) / 8)) and a wave walks its super-chunks
 * in ascending order. Lane half h of a super-chunk holds k = 64 s + 32 h .. +31: 16 packed bytes (one uint4: two 16-element NVFP4
 * blocks, so the load is one 16-byte non-temporal read) and its two block scales; the four 8-k groups g of those 32 k are the four
 * WMMA instructions of the super-chunk, group g's weights are dword g of the uint4 and their scale is block g / 2's. A and B use the
 * same permutation of k. Every weight is dequantised ONCE per call (e2m1 value * block scale, exact in bf16: <= 6 significant bits)
 * however many rows there are.
 *
 * Row invariance: the bits of y[m, n] depend on K, the row m, the weight row n and the scales only -- not on M, the token tile or the
 * column tile. A K % 64 != 0 tail leaves the last super-chunk's missing blocks as zeros in both operands.
 */
#ifndef IK_NF4_ROWS_H
#define IK_NF4_ROWS_H

constexpr int kNf4rWaves = 8;
constexpr int kNf4rThreads = kNf4rWaves * IK_WARP;
constexpr int kNf4rSpBytes = kNf4rWaves * IK_WARP * 8 * 4;  // the cross-wave buffer of one token tile

/* 8 e2m1 codes (one dword, element j = nibble j) * scale -> 8 bf16 (one uint4). The products are exact in bf16. */
IK_DEVI uint4 nf4r_dequant8(uint32_t w, float sc, const float* lut) {
    float f[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) f[j] = lut[(w >> (4 * j)) & 15u] * sc;
    return ik_pack8(f);
}

template <int TM, int NS, typename OutT>
__device__ __forceinline__ void nf4r_body(const ik_bf16* __restrict__ x, const uint8_t* __restrict__ w, const uint8_t* __restrict__ bs,
                                          const float* __restrict__ gs, OutT* __restrict__ out, int M, int N, int K, int64_t blk) {
    extern __shared__ __align__(16) unsigned char smem[];
    __shared__ float lut[16];
    if (threadIdx.x < 16) lut[threadIdx.x] = ik_fp4_value(threadIdx.x);
    IK_LDS_BARRIER();
    const int tid = threadIdx.x, wave = tid >> 5, lane = tid & 31;
    const int h = lane >> 4, col = lane & 15;
    const int64_t n = blk * 16 + col;
    const bool live = n < N;
    const uint8_t* wr = live ? w + n * (K / 2) : nullptr;
    const uint8_t* sr = live ? bs + n * (K / 16) : nullptr;
    const int nsc = (K + 63) / 64, cpw = (nsc + kNf4rWaves - 1) / kNf4rWaves;

    ik_dq_v8f acc[TM];
#pragma unroll
    for (int t = 0; t < TM; ++t) acc[t] = ik_dq_v8f{0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

    for (int g0 = 0; g0 < cpw; g0 += NS) {
        uint4 wq[NS];
        uint32_t sb[NS];
        uint4 xb[TM][NS][4];
#pragma unroll
        for (int i = 0; i < NS; ++i) {
            const int s = wave * cpw + g0 + i;
            const int64_t k0 = (int64_t)s * 64 + 32 * h;  // this lane's first k
            const bool in = g0 + i < cpw && s < nsc;
            const bool v0 = in && k0 < K, v1 = in && k0 + 16 < K;
            wq[i] = make_uint4(0, 0, 0, 0);
            sb[i] = 0;
            if (live && v1) {
                wq[i] = ik_ldcs(reinterpret_cast<const uint4*>(wr + k0 / 2));
                sb[i] = *reinterpret_cast<const uint16_t*>(sr + k0 / 16);
            } else if (live && v0) {
                const uint2 t2 = ik_ldcs(reinterpret_cast<const uint2*>(wr + k0 / 2));
                wq[i] = make_uint4(t2.x, t2.y, 0, 0);
                sb[i] = sr[k0 / 16];
            }
#pragma unroll
            for (int t = 0; t < TM; ++t) {
                const int row = t * 16 + col;
#pragma unroll
                for (int g = 0; g < 4; ++g) {
                    xb[t][i][g] = make_uint4(0, 0, 0, 0);
                    if (row < M && in && k0 + 8 * g < K) xb[t][i][g] = ik_ldg(reinterpret_cast<const uint4*>(x + (int64_t)row * K + k0 + 8 * g));
                }
            }
        }
#pragma unroll
        for (int i = 0; i < NS; ++i) {
            const int s = wave * cpw + g0 + i;
            if (!(g0 + i < cpw && s < nsc)) continue;
            const uint32_t ww[4] = {wq[i].x, wq[i].y, wq[i].z, wq[i].w};
#pragma unroll
            for (int g = 0; g < 4; ++g) {
                const float scale = ik_e4m3_to_f((sb[i] >> (8 * (g >> 1))) & 0xffu);
                const uint4 a = nf4r_dequant8(ww[g], scale, lut);
#pragma unroll
                for (int t = 0; t < TM; ++t) acc[t] = ik_dq_wmma(a, xb[t][i][g], acc[t]);
            }
        }
    }
    float* sp = reinterpret_cast<float*>(smem);
    IK_LDS_BARRIER();
#pragma unroll
    for (int t = 0; t < TM; ++t) {
        float4* d = reinterpret_cast<float4*>(sp + (size_t)((wave * TM + t) * IK_WARP + lane) * 8);
        d[0] = make_float4(acc[t][0], acc[t][1], acc[t][2], acc[t][3]);
        d[1] = make_float4(acc[t][4], acc[t][5], acc[t][6], acc[t][7]);
    }
    IK_LDS_BARRIER();
    const float g = gs[0];
    const int rl = tid & 31, part = (tid >> 5) & 1, rt = tid >> 6;
    if (rt < TM) {
        float4 s = reinterpret_cast<const float4*>(sp + (size_t)(rt * IK_WARP + rl) * 8)[part];
#pragma unroll
        for (int wv = 1; wv < kNf4rWaves; ++wv) {
            const float4 v = reinterpret_cast<const float4*>(sp + (size_t)((wv * TM + rt) * IK_WARP + rl) * 8)[part];
            s.x += v.x;
            s.y += v.y;
            s.z += v.z;
            s.w += v.w;
        }
        const int tok = rt * 16 + (rl & 15);
        const int64_t f0 = blk * 16 + (rl >> 4) * 8 + part * 4;
        if (tok < M && f0 < N) {
            OutT* o = out + (int64_t)tok * N + f0;
            const float r[4] = {s.x * g, s.y * g, s.z * g, s.w * g};
            for (int e = 0; e < 4 && f0 + e < N; ++e) ik_store(o + e, r[e]);
        }
    }
}

template <int TM, int NS, typename OutT>
__global__ void __launch_bounds__(kNf4rThreads)
nf4r_kernel(const ik_bf16* __restrict__ x, const uint8_t* __restrict__ w, const uint8_t* __restrict__ bs, const float* __restrict__ gs,
            OutT* __restrict__ out, int M, int N, int K) {
    nf4r_body<TM, NS, OutT>(x, w, bs, gs, out, M, N, K, (int64_t)blockIdx.x);
}

template <typename OutT>
int nf4r_dispatch(const ik_bf16* x, const uint8_t* w, const uint8_t* bs, const float* gs, OutT* out, int M, int64_t N, int64_t K,
                  hipStream_t s) {
    if (M < 1 || M > 64) return IK_ERR_SHAPE;
    const int tm = (M + 15) / 16;
    const unsigned grid = (unsigned)((N + 15) / 16);
    const size_t smem = (size_t)tm * kNf4rSpBytes;
#define IK_NF4R_CASE(T, NSV)                                                                                      \
    case T:                                                                                                       \
        nf4r_kernel<T, NSV, OutT><<<grid, kNf4rThreads, smem, s>>>(x, w, bs, gs, out, M, (int)N, (int)K);        \
        break;
    switch (tm) {
        IK_NF4R_CASE(1, 4) IK_NF4R_CASE(2, 4) IK_NF4R_CASE(3, 2) IK_NF4R_CASE(4, 2)
        default: return IK_ERR_SHAPE;
    }
#undef IK_NF4R_CASE
    return ik_launch_status();
}

#endif  // IK_NF4_ROWS_H
