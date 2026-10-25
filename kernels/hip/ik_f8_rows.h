/* Small-row fp8 GEMM on the matrix cores (lane wg): one WMMA kernel family for M = 1 .. 64 rows, W8A8 (fp8 x fp8 WMMA) and W8A16
 * (bf16 WMMA, weights e4m3 -> bf16 exact). Included by fp8.hip INSIDE its anonymous namespace after the prologue helpers
 * (gated_norm_chunk, norm_add_stage, sigmoid_mul_chunk) it folds in; not a standalone header.
 *
 * y[m, n] = weight_scale * input_scale * sum_k xq[m, k] * w[n, k]   (W8A8, xq = e4m3(x / input_scale), RNE, saturating)
 * y[m, n] = weight_scale * sum_k x[m, k] * w[n, k]                    (W8A16)
 *
 * Block: 8 waves (256 threads) on one 16-feature tile of the output (A = the weights, rows = features; B = the activations,
 * columns = tokens, 16 per tile, TM tiles for M > 16). K is cut into chunks of 32; chunk c belongs to wave c % 8 and a wave
 * walks its chunks in ascending order into ONE accumulator per token tile (two WMMAs per chunk: lane half h holds bytes
 * h * 16 .. +15 of the chunk, the low 8 for the first instruction and the high 8 for the second; A and B use the same
 * permutation of k, so every k is summed exactly once). The eight waves meet once in LDS and add in the order wave 0 .. 7.
 *
 * Row invariance (the point of the family): the bits of y[m, n] depend only on K, the activation row m, the weight row n and the
 * two scales -- not on M, on the token tile the row sits in, on the column tile, on the staging window (a window is a multiple of
 * 256 k, so the chunk -> wave map and each wave's ascending chunk order do not change), not on where the activations are quantised
 * (in the block, or once per call by f8r_actq_kernel: the same f8r_quant8) and not on PRO (the folded prologues produce the staged
 * row bit for bit as the unfused kernels do). A K % 32 == 16 tail is chunk K / 32, one 8-byte-per-half WMMA.
 *
 * Two activation sources, by launch (speed only): STAGED -- the block quantises x into LDS (rows padded by one uint4 so the 16 lanes
 * that read 16 different rows hit 16 different bank groups); the only form the folded prologues (PRO != 0, M == 1) can use.
 * DIRECT -- the B fragments are loaded from global memory: the e4m3 bytes f8r_actq_kernel wrote into scratch (W8A8), or x itself
 * (W8A16); no staging, no LDS but the cross-wave sum.
 */
#ifndef IK_F8_ROWS_H
#define IK_F8_ROWS_H

constexpr int kF8rWaves = 8;          // waves on K: a constant, so the cross-wave sum order is too
constexpr int kF8rThreads = kF8rWaves * IK_WARP;
constexpr int kF8rPad = 16;           // bytes of padding after each staged row
constexpr int kF8rSpBytes = kF8rWaves * IK_WARP * 8 * 4;   // the cross-wave buffer of one token tile
constexpr int kF8rLdsBudget = 61440;  // dynamic LDS a launch may ask for (gfx1201 allows 64 KiB without opt-in)

/* 8 bf16 -> 8 e4m3 bytes (x / input_scale, RNE, saturating at +-448), bit-identical to fp8_gemv_body's Q = 2 staging. */
IK_DEVI uint2 f8r_quant8(const uint4& v, float s_in) {
    float f[8];
    ik_unpack8(v, f);
#if __has_builtin(__builtin_amdgcn_cvt_pk_fp8_f32)
    float y[8];
#pragma unroll
    for (int e = 0; e < 8; ++e) y[e] = fminf(fmaxf(__fdiv_rn(f[e], s_in), -448.f), 448.f);
    int c0 = __builtin_amdgcn_cvt_pk_fp8_f32(y[0], y[1], 0, false);
    c0 = __builtin_amdgcn_cvt_pk_fp8_f32(y[2], y[3], c0, true);
    int c1 = __builtin_amdgcn_cvt_pk_fp8_f32(y[4], y[5], 0, false);
    c1 = __builtin_amdgcn_cvt_pk_fp8_f32(y[6], y[7], c1, true);
    return make_uint2((uint32_t)c0, (uint32_t)c1);
#else
    uint32_t lo = 0, hi = 0;
#pragma unroll
    for (int e = 0; e < 4; ++e) {
        lo |= (uint32_t)ik_f_to_e4m3(__fdiv_rn(f[e], s_in)) << (8 * e);
        hi |= (uint32_t)ik_f_to_e4m3(__fdiv_rn(f[4 + e], s_in)) << (8 * e);
    }
    return make_uint2(lo, hi);
#endif
}

/* xq = e4m3(x / input_scale) for the whole call (the DIRECT form's activations). Exits at once for weight-only activations. */
__global__ void __launch_bounds__(256) f8r_actq_kernel(const ik_bf16* __restrict__ x, uint8_t* __restrict__ xq, const float* __restrict__ sc,
                                                       int64_t n8) {
    const float s_in = sc[1];
    if (!(s_in > 0.f)) return;
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n8) return;
    reinterpret_cast<uint2*>(xq)[i] = f8r_quant8(reinterpret_cast<const uint4*>(x)[i], s_in);
}

/* The 8 activations of row `row` starting at element kk (a multiple of 8) as the prologue PRO makes them. */
template <int PRO>
IK_DEVI uint4 f8r_src(const ik_bf16* __restrict__ x, const ik_bf16* __restrict__ pz, const ik_bf16* __restrict__ pnw, float peps,
                      float pwoff, const uint4* __restrict__ xn, int row, int64_t K, int64_t kk) {
    if constexpr (PRO == 2) return gated_norm_chunk(x, pz, pnw, (int)(kk >> 3), peps, pwoff);
    else if constexpr (PRO == 3) return xn[kk >> 3];
    else if constexpr (PRO == 4) return sigmoid_mul_chunk(x, pz, (int)(kk >> 3));
    else return reinterpret_cast<const uint4*>(x)[((int64_t)row * K + kk) >> 3];
}

/* One block of the family. NJ = most chunks a wave handles per window (a window is NJ * 256 k at most). TM = 16-token tiles.
 * kw: window (multiples of 256, >= 512, <= NJ * 256), M <= 16 * TM. blk: the block's 16-feature tile. PRO != 0 only at M == 1 with
 * DIRECT false and kw >= K (one window); PRO 3 needs 2 * K bytes of LDS after the staged row (the bf16 normed row). */
template <int TM, int NJ, int PRO, bool DIRECT, bool BF, typename OutT>
__device__ __forceinline__ void f8r_body(const ik_bf16* __restrict__ x, const uint8_t* __restrict__ xq, const uint8_t* __restrict__ w,
                                         const float* __restrict__ sc, OutT* __restrict__ out, int M, int N, int K, int kw, int64_t blk,
                                         const ik_bf16* __restrict__ pz = nullptr, const ik_bf16* __restrict__ pnw = nullptr,
                                         float peps = 0.f, float pwoff = 0.f, ik_bf16* __restrict__ pnormed = nullptr,
                                         ik_bf16* __restrict__ psum = nullptr, bool pwriter = false) {
    extern __shared__ __align__(16) unsigned char smem[];
    const int tid = threadIdx.x, wave = tid >> 5, lane = tid & 31;
    const int h = lane >> 4, col = lane & 15;
    // BF: the weights are bf16 (the router / gate matrices): no scales, the bf16-WMMA path of W8A16 with the weights loaded as they are
    const bool actq = BF ? false : sc[1] > 0.f;
    const float s_in = actq ? sc[1] : 1.f;
    const float s_out = BF ? 1.f : sc[0] * s_in;
    const int64_t n = blk * 16 + col;
    const uint8_t* wp = n < N ? w + n * K * (BF ? 2 : 1) : nullptr;
    // staged window: W8A8 kw bytes a row, W8A16 kw / 2 bf16 (the same bytes); a DIRECT window is kw k in registers
    const int kwa = (DIRECT || actq) ? kw : (kw / 2 / 256) * 256;
    const int stride = actq ? kw + kF8rPad : kwa * 2 + kF8rPad;
    unsigned char* xs = smem;
    float* sp = reinterpret_cast<float*>(smem);  // [wave][tile][lane][8], over the staged x once every wave is done with it
    const size_t region = DIRECT ? (size_t)TM * kF8rSpBytes
                                 : ((size_t)M * (size_t)stride > (size_t)TM * kF8rSpBytes ? (size_t)M * (size_t)stride : (size_t)TM * kF8rSpBytes);
    uint4* xn = reinterpret_cast<uint4*>(smem + region);  // PRO 3: the normed bf16 row

    ik_dq_v8f acc[TM];
#pragma unroll
    for (int t = 0; t < TM; ++t) acc[t] = ik_dq_v8f{0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

    uint4 wb[NJ][BF ? 2 : 1];
    uint4 xb[DIRECT ? TM : 1][DIRECT ? NJ : 1][DIRECT ? 2 : 1];
    for (int ws = 0; ws < K; ws += kwa) {
        const int wlen = min(kwa, K - ws);
        const int nj = (wlen + 255) >> 8;
        // the weights of the window go in flight before the activations are staged
#pragma unroll
        for (int j = 0; j < NJ; ++j) {
            wb[j][0] = make_uint4(0, 0, 0, 0);
            if constexpr (BF) wb[j][1] = make_uint4(0, 0, 0, 0);
            const int k0 = ws + (wave + kF8rWaves * j) * 32;
            if (wp && j < nj) {
                if constexpr (BF) {
                    if (k0 + 32 <= K) {
                        const uint4* p = reinterpret_cast<const uint4*>(wp + (int64_t)(k0 + 16 * h) * 2);
                        wb[j][0] = ik_ldcs(p);
                        wb[j][1] = ik_ldcs(p + 1);
                    } else if (k0 + 16 <= K) {
                        wb[j][0] = ik_ldcs(reinterpret_cast<const uint4*>(wp + (int64_t)(k0 + 8 * h) * 2));
                    }
                } else {
                    if (k0 + 32 <= K) wb[j][0] = ik_ldcs(reinterpret_cast<const uint4*>(wp + k0 + 16 * h));
                    else if (k0 + 16 <= K) {
                        const uint2 t = *reinterpret_cast<const uint2*>(wp + k0 + 8 * h);
                        wb[j][0] = make_uint4(t.x, t.y, 0, 0);
                    }
                }
            }
        }
        if constexpr (DIRECT) {
#pragma unroll
            for (int j = 0; j < NJ; ++j) {
                const int k0 = ws + (wave + kF8rWaves * j) * 32;
#pragma unroll
                for (int t = 0; t < TM; ++t) {
                    const int row = t * 16 + col;
                    xb[t][j][0] = make_uint4(0, 0, 0, 0);
                    xb[t][j][1] = make_uint4(0, 0, 0, 0);
                    if (row < M && j < nj && k0 + 16 <= K) {
                        const bool full = k0 + 32 <= K;
                        if (actq) {
                            if (full) xb[t][j][0] = *reinterpret_cast<const uint4*>(xq + (int64_t)row * K + k0 + 16 * h);
                            else {
                                const uint2 u = *reinterpret_cast<const uint2*>(xq + (int64_t)row * K + k0 + 8 * h);
                                xb[t][j][0] = make_uint4(u.x, u.y, 0, 0);
                            }
                        } else {
                            const uint4* p = reinterpret_cast<const uint4*>(x + (int64_t)row * K + k0 + (full ? 16 : 8) * h);
                            xb[t][j][0] = ik_ldg(p);
                            if (full) xb[t][j][1] = ik_ldg(p + 1);
                        }
                    }
                }
            }
        } else {
            if constexpr (PRO == 3) norm_add_stage(x, pz, pnw, pnormed, psum, xn, K, peps, pwoff, pwriter);
            const int c8 = wlen >> 3;
            for (int i = tid; i < M * c8; i += kF8rThreads) {
                const int row = i / c8, kc = (i - row * c8) << 3;
                const uint4 v = f8r_src<PRO>(x, pz, pnw, peps, pwoff, xn, row, K, ws + kc);
                if (actq) *reinterpret_cast<uint2*>(xs + (size_t)row * stride + kc) = f8r_quant8(v, s_in);
                else *reinterpret_cast<uint4*>(xs + (size_t)row * stride + kc * 2) = v;
            }
            __syncthreads();
        }
#pragma unroll
        for (int j = 0; j < NJ; ++j) {
            const int k0 = ws + (wave + kF8rWaves * j) * 32;
            const int kl = k0 - ws;
            if (j >= nj || k0 + 16 > K) continue;
            const bool full = k0 + 32 <= K;
#pragma unroll
            for (int t = 0; t < TM; ++t) {
                const int row = t * 16 + col;
                if (actq) {
                    uint4 b = make_uint4(0, 0, 0, 0);
                    if constexpr (DIRECT) {
                        b = xb[t][j][0];
                    } else if (row < M) {
                        if (full) b = *reinterpret_cast<const uint4*>(xs + (size_t)row * stride + kl + 16 * h);
                        else {
                            const uint2 u = *reinterpret_cast<const uint2*>(xs + (size_t)row * stride + kl + 8 * h);
                            b = make_uint4(u.x, u.y, 0, 0);
                        }
                    }
                    acc[t] = ik_f8_wmma(wb[j][0].x, wb[j][0].y, b.x, b.y, acc[t]);
                    if (full) acc[t] = ik_f8_wmma(wb[j][0].z, wb[j][0].w, b.z, b.w, acc[t]);
                } else {
                    uint4 b0 = make_uint4(0, 0, 0, 0), b1 = b0;
                    if constexpr (DIRECT) {
                        b0 = xb[t][j][0];
                        b1 = xb[t][j][1];
                    } else if (row < M) {
                        const unsigned char* p = xs + (size_t)row * stride + (size_t)kl * 2;
                        if (full) {
                            b0 = *reinterpret_cast<const uint4*>(p + 32 * h);
                            b1 = *reinterpret_cast<const uint4*>(p + 32 * h + 16);
                        } else {
                            b0 = *reinterpret_cast<const uint4*>(p + 16 * h);
                        }
                    }
                    if constexpr (BF) {
                        acc[t] = ik_dq_wmma(wb[j][0], b0, acc[t]);
                        if (full) acc[t] = ik_dq_wmma(wb[j][1], b1, acc[t]);
                    } else {
                        acc[t] = ik_dq_wmma(ik_e4m3x8_to_bf16(wb[j][0].x, wb[j][0].y), b0, acc[t]);
                        if (full) acc[t] = ik_dq_wmma(ik_e4m3x8_to_bf16(wb[j][0].z, wb[j][0].w), b1, acc[t]);
                    }
                }
            }
        }
        if (!DIRECT && ws + kwa < K) __syncthreads();  // the next window restages x
    }
    // cross-wave sum in a fixed order, scale, store: thread = (lane, half of the 8 features, token tile)
    __syncthreads();  // every wave has read its last staged x
#pragma unroll
    for (int t = 0; t < TM; ++t) {
        float4* d = reinterpret_cast<float4*>(sp + (size_t)((wave * TM + t) * IK_WARP + lane) * 8);
        d[0] = make_float4(acc[t][0], acc[t][1], acc[t][2], acc[t][3]);
        d[1] = make_float4(acc[t][4], acc[t][5], acc[t][6], acc[t][7]);
    }
    __syncthreads();
    const int rl = tid & 31, part = (tid >> 5) & 1, rt = tid >> 6;
    if (rt < TM) {
        float4 s = reinterpret_cast<const float4*>(sp + (size_t)(rt * IK_WARP + rl) * 8)[part];
#pragma unroll
        for (int wv = 1; wv < kF8rWaves; ++wv) {
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
            const float r[4] = {s.x * s_out, s.y * s_out, s.z * s_out, s.w * s_out};
            if (f0 + 4 <= N) {
#pragma unroll
                for (int e = 0; e < 4; ++e) ik_store(o + e, r[e]);
            } else {
                for (int e = 0; e < 4 && f0 + e < N; ++e) ik_store(o + e, r[e]);
            }
        }
    }
}

template <int TM, int NJ, bool DIRECT, bool BF, typename OutT>
__global__ void __launch_bounds__(kF8rThreads)
f8r_kernel(const ik_bf16* __restrict__ x, const uint8_t* __restrict__ xq, const uint8_t* __restrict__ w, const float* __restrict__ sc,
           OutT* __restrict__ out, int M, int N, int K, int kw) {
    f8r_body<TM, NJ, 0, DIRECT, BF, OutT>(x, xq, w, sc, out, M, N, K, kw, (int64_t)blockIdx.x);
}

/* The folded decode launchers (M == 1, STAGED): the same body with the prologue of the fp8 GEMV folds, so a solo step runs the
 * family's arithmetic on the row the unfused chain would stage. */
template <int NJ>
__global__ void __launch_bounds__(kF8rThreads)
f8r_pair_kernel(const ik_bf16* __restrict__ x, const uint8_t* __restrict__ w0, const float* __restrict__ sc0,
                ik_bf16* __restrict__ o0, int N0, const uint8_t* __restrict__ w1, const float* __restrict__ sc1,
                ik_bf16* __restrict__ o1, int N1, int K, int kw) {
    const int nb0 = (N0 + 15) / 16;
    if ((int)blockIdx.x < nb0) f8r_body<1, NJ, 0, false, false, ik_bf16>(x, nullptr, w0, sc0, o0, 1, N0, K, kw, (int64_t)blockIdx.x);
    else f8r_body<1, NJ, 0, false, false, ik_bf16>(x, nullptr, w1, sc1, o1, 1, N1, K, kw, (int64_t)blockIdx.x - nb0);
}

template <int NJ>
__global__ void __launch_bounds__(kF8rThreads)
f8r_gated_kernel(const ik_bf16* __restrict__ x, const ik_bf16* __restrict__ z, const ik_bf16* __restrict__ nw, float eps, float woff,
                 const uint8_t* __restrict__ w, const float* __restrict__ sc, ik_bf16* __restrict__ out, int N, int K, int kw) {
    f8r_body<1, NJ, 2, false, false, ik_bf16>(x, nullptr, w, sc, out, 1, N, K, kw, (int64_t)blockIdx.x, z, nw, eps, woff);
}

template <int NJ>
__global__ void __launch_bounds__(kF8rThreads)
f8r_norm_pair_kernel(const ik_bf16* __restrict__ x, const ik_bf16* __restrict__ res, const ik_bf16* __restrict__ nw, float eps,
                     float woff, ik_bf16* __restrict__ normed, ik_bf16* __restrict__ sum, const uint8_t* __restrict__ w0,
                     const float* __restrict__ sc0, ik_bf16* __restrict__ o0, int N0, const uint8_t* __restrict__ w1,
                     const float* __restrict__ sc1, ik_bf16* __restrict__ o1, int N1, int K, int kw) {
    const int nb0 = (N0 + 15) / 16;
    const bool writer = blockIdx.x == 0;
    if ((int)blockIdx.x < nb0)
        f8r_body<1, NJ, 3, false, false, ik_bf16>(x, nullptr, w0, sc0, o0, 1, N0, K, kw, (int64_t)blockIdx.x, res, nw, eps, woff, normed, sum,
                                           writer);
    else
        f8r_body<1, NJ, 3, false, false, ik_bf16>(x, nullptr, w1, sc1, o1, 1, N1, K, kw, (int64_t)blockIdx.x - nb0, res, nw, eps, woff, normed,
                                           sum, writer);
}

template <int NJ>
__global__ void __launch_bounds__(kF8rThreads)
f8r_sigmoid_kernel(const ik_bf16* __restrict__ a, const ik_bf16* __restrict__ gate, const uint8_t* __restrict__ w,
                   const float* __restrict__ sc, ik_bf16* __restrict__ out, int N, int K, int kw) {
    f8r_body<1, NJ, 4, false, false, ik_bf16>(a, nullptr, w, sc, out, 1, N, K, kw, (int64_t)blockIdx.x, gate);
}

/* bf16 weights (the MoE router / shared-expert gate, the GDN b / a gates): the two folds of the bf16 pair GEMV on the same family. */
template <int NJ>
__global__ void __launch_bounds__(kF8rThreads)
f8r_bf16_pair_kernel(const ik_bf16* __restrict__ x, const uint8_t* __restrict__ w0, ik_bf16* __restrict__ o0, int N0,
                     const uint8_t* __restrict__ w1, ik_bf16* __restrict__ o1, int N1, int K, int kw) {
    const int nb0 = (N0 + 15) / 16;
    if ((int)blockIdx.x < nb0) f8r_body<1, NJ, 0, false, true, ik_bf16>(x, nullptr, w0, nullptr, o0, 1, N0, K, kw, (int64_t)blockIdx.x);
    else f8r_body<1, NJ, 0, false, true, ik_bf16>(x, nullptr, w1, nullptr, o1, 1, N1, K, kw, (int64_t)blockIdx.x - nb0);
}

template <int NJ>
__global__ void __launch_bounds__(kF8rThreads)
f8r_bf16_norm_pair_kernel(const ik_bf16* __restrict__ x, const ik_bf16* __restrict__ res, const ik_bf16* __restrict__ nw, float eps,
                          float woff, ik_bf16* __restrict__ normed, ik_bf16* __restrict__ sum, const uint8_t* __restrict__ w0,
                          ik_bf16* __restrict__ o0, int N0, const uint8_t* __restrict__ w1, ik_bf16* __restrict__ o1, int N1, int K,
                          int kw) {
    const int nb0 = (N0 + 15) / 16;
    const bool writer = blockIdx.x == 0;
    if ((int)blockIdx.x < nb0)
        f8r_body<1, NJ, 3, false, true, ik_bf16>(x, nullptr, w0, nullptr, o0, 1, N0, K, kw, (int64_t)blockIdx.x, res, nw, eps, woff,
                                                 normed, sum, writer);
    else
        f8r_body<1, NJ, 3, false, true, ik_bf16>(x, nullptr, w1, nullptr, o1, 1, N1, K, kw, (int64_t)blockIdx.x - nb0, res, nw, eps,
                                                 woff, normed, sum, writer);
}

/* Window and shared memory of a launch from (M, K) alone -- only speed depends on them, never the bits. */
struct F8rCfg {
    int tm, nj, kw;
    size_t smem;
    bool direct;
};

/* direct: the DIRECT form (window = NJ chunks per wave in registers, no staging). */
inline bool f8r_cfg(int64_t M, int64_t K, bool single_window, size_t extra, bool direct, F8rCfg* c) {
    if (M < 1 || M > 64) return false;
    c->tm = (int)((M + 15) / 16);
    c->direct = direct;
    if (direct) {
        c->nj = c->tm >= 3 ? 4 : 8;
        c->kw = c->nj * 256;
        c->smem = (size_t)c->tm * kF8rSpBytes;
        return !single_window;
    }
    const int64_t kmax = (K + 255) / 256 * 256;
    int64_t kw = ((int64_t)kF8rLdsBudget - (int64_t)extra) / M - kF8rPad;
    kw = kw / 256 * 256;
    kw = std::min<int64_t>(kw, std::min<int64_t>(kmax, 4096));
    if (single_window && kw < kmax) return false;
    if (kw < 512) {
        if (kmax < 512) kw = 512;  // K <= 256: the window is wider than K (the staged row is K bytes)
        else return false;
    }
    c->kw = (int)kw;
    c->nj = kw > 2048 ? 16 : 8;
    c->smem = std::max<size_t>((size_t)M * (kw + kF8rPad), (size_t)c->tm * kF8rSpBytes) + extra;
    return true;
}

template <int TM, int NJ, bool DIRECT, bool BF, typename OutT>
void f8r_run(const ik_bf16* x, const uint8_t* xq, const uint8_t* w, const float* sc, OutT* out, int M, int64_t N, int64_t K,
             const F8rCfg& c, hipStream_t s) {
    const unsigned grid = (unsigned)((N + 15) / 16);
    f8r_kernel<TM, NJ, DIRECT, BF, OutT><<<grid, kF8rThreads, c.smem, s>>>(x, xq, w, sc, out, M, (int)N, (int)K, c.kw);
}

/* M rows, STAGED or (scratch given, M >= direct_min) DIRECT. */
template <bool BF, typename OutT>
int f8r_dispatch(const ik_bf16* x, const uint8_t* w, const float* sc, OutT* out, int M, int64_t N, int64_t K, uint8_t* scratch,
                 int direct_min, hipStream_t s) {
    const bool direct = (BF || scratch != nullptr) && M >= direct_min;
    F8rCfg c;
    if (!f8r_cfg(M, K, false, 0, direct, &c)) return IK_ERR_SHAPE;
    if (direct && !BF) {
        const int64_t n8 = (int64_t)M * K / 8;
        f8r_actq_kernel<<<(unsigned)((n8 + 255) / 256), 256, 0, s>>>(x, scratch, sc, n8);
    }
#define IK_F8R_CASE(T)                                                                                          \
    case T:                                                                                                     \
        if (direct) {                                                                                           \
            if (c.nj == 4) f8r_run<T, 4, true, BF, OutT>(x, scratch, w, sc, out, M, N, K, c, s);                    \
            else f8r_run<T, 8, true, BF, OutT>(x, scratch, w, sc, out, M, N, K, c, s);                              \
        } else if (c.nj == 16) f8r_run<T, 16, false, BF, OutT>(x, nullptr, w, sc, out, M, N, K, c, s);              \
        else f8r_run<T, 8, false, BF, OutT>(x, nullptr, w, sc, out, M, N, K, c, s);                                 \
        break;
    switch (c.tm) {
        IK_F8R_CASE(1) IK_F8R_CASE(2) IK_F8R_CASE(3) IK_F8R_CASE(4)
        default: return IK_ERR_SHAPE;
    }
#undef IK_F8R_CASE
    return ik_launch_status();
}

#endif  // IK_F8_ROWS_H
