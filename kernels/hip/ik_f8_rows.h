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

#ifdef IK_F8R_ABLATE
static int g_f8r_abl = 0;  // set through ik_f8r_set_abl
#endif
constexpr int kF8rWaves = 8;          // waves on K: a constant, so the cross-wave sum order is too
constexpr int kF8rThreads = kF8rWaves * IK_WARP;
constexpr int kF8rPad = 16;           // bytes of padding after each staged row
constexpr int kF8rSpBytes = kF8rWaves * IK_WARP * 8 * 4;   // the cross-wave buffer of one token tile
constexpr int kF8rWRow = 144;                      // one row of a wave's weight tile: 128 B + 16 B pad (16 rows hit 16 distinct bank groups)
constexpr int kF8rWTile = 16 * kF8rWRow;          // a wave's weight tile (fp8 weights)
constexpr int kF8rWBytes = kF8rWaves * kF8rWTile; // all 8 waves'
constexpr int kF8rLdsBudget = 49152;  // dynamic LDS a launch may ask for (48 KiB, what every other kernel here stays under)

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
template <int TM, int NJ, int PRO, bool DIRECT, bool BF, typename OutT, bool FRAG = false>
__device__ __forceinline__ void f8r_body(const ik_bf16* __restrict__ x, const uint8_t* __restrict__ xq, const uint8_t* __restrict__ w,
                                         const float* __restrict__ sc, OutT* __restrict__ out, int M, int N, int K, int kw, int64_t blk,
                                         const ik_bf16* __restrict__ pz = nullptr, const ik_bf16* __restrict__ pnw = nullptr,
                                         float peps = 0.f, float pwoff = 0.f, ik_bf16* __restrict__ pnormed = nullptr,
                                         ik_bf16* __restrict__ psum = nullptr, bool pwriter = false, int abl = 0) {
    // abl: timing ablations (IK_F8R_ABLATE builds only; wrong results): 1 no x staging, 2 no cross-wave sum, 4 no weight loads
    extern __shared__ __align__(16) unsigned char smem[];
    const int tid = threadIdx.x, wave = tid >> 5, lane = tid & 31;
    const int h = lane >> 4, col = lane & 15;
    // BF: the weights are bf16 (the router / gate matrices): no scales, the bf16-WMMA path of W8A16 with the weights loaded as they are
    const bool actq = BF ? false : sc[1] > 0.f;
    const float s_in = actq ? sc[1] : 1.f;
    const float s_out = BF ? 1.f : sc[0] * s_in;
    const int64_t n = blk * 16 + col;
    const uint8_t* wp = n < N ? w + n * K * (BF ? 2 : 1) : nullptr;  // BF only
    // staged window: W8A8 kw bytes a row, W8A16 kw / 2 bf16 (the same bytes); BF: kw bf16 (bpk = 2); a DIRECT window is kw k in registers.
    // The fp8 family's windows are multiples of 1024 k (one round of 8 waves x 128 k), so each wave's K ranges do not depend on the window.
    const int kwa = (DIRECT || actq || BF) ? kw : (kw / 2 / 1024) * 1024;
    const int stride = actq ? kw + kF8rPad : kwa * 2 + kF8rPad;
    unsigned char* xs = smem;
    float* sp = reinterpret_cast<float*>(smem);  // [wave][tile][lane][8], over the staged x once every wave is done with it
    const size_t region = DIRECT ? (size_t)TM * kF8rSpBytes
                                 : ((size_t)M * (size_t)stride > (size_t)TM * kF8rSpBytes ? (size_t)M * (size_t)stride : (size_t)TM * kF8rSpBytes);
    uint4* xn = reinterpret_cast<uint4*>(smem + region);  // PRO 3: the normed bf16 row
    // fp8 weights: each wave loads 16 rows x 128 B per range with coalesced 16 B loads (4 rows x 128 B an instruction) and re-reads them in the
    // WMMA fragment layout through a private LDS tile (no block barrier: a wave only reads what it wrote)
    unsigned char* wbuf = smem + (DIRECT ? 0 : region + (PRO == 3 ? ((size_t)K * 2 + 15) / 16 * 16 : 0)) + (size_t)wave * kF8rWTile;

    ik_dq_v8f acc[TM];
#pragma unroll
    for (int t = 0; t < TM; ++t) acc[t] = ik_dq_v8f{0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

    // slot j of this wave inside a window: the first k (relative to the window) of its chunk of 32 k
    //   BF:  chunk wave + 8 j (interleaved);   fp8: range wave + 8 (j / 4) of 128 k, chunk j % 4 of it
    auto slot_k = [&](int j) -> int { return BF ? (wave + kF8rWaves * j) * 32 : 128 * (wave + kF8rWaves * (j >> 2)) + 32 * (j & 3); };

    uint4 wb[BF ? NJ : 1][BF ? 2 : 1];            // BF: the weights of the window's chunks
    uint4 wr[BF ? 1 : NJ / 4][BF ? 1 : 4];        // fp8: the window's ranges, 4 coalesced loads each
    uint4 xb[DIRECT ? TM : 1][DIRECT ? NJ : 1][DIRECT ? 2 : 1];
    for (int ws = 0; ws < K; ws += kwa) {
        const int wlen = min(kwa, K - ws);
        // the weights of the window go in flight before the activations are staged
        if constexpr (BF) {
#pragma unroll
            for (int j = 0; j < NJ; ++j) {
                wb[j][0] = make_uint4(0, 0, 0, 0);
                wb[j][1] = make_uint4(0, 0, 0, 0);
                const int kl = slot_k(j), k0 = ws + kl;
                if (!(abl & 4) && wp && kl < kwa) {
                    if (k0 + 32 <= K) {
                        const uint4* p = reinterpret_cast<const uint4*>(wp + (int64_t)(k0 + 16 * h) * 2);
                        wb[j][0] = ik_ldcs(p);
                        wb[j][1] = ik_ldcs(p + 1);
                    } else if (k0 + 16 <= K) {
                        wb[j][0] = ik_ldcs(reinterpret_cast<const uint4*>(wp + (int64_t)(k0 + 8 * h) * 2));
                    }
                }
            }
        } else if constexpr (FRAG) {
            // fragment order ([N/16][K/32][lane = 16 h + col][16 B]; N % 16 == 0, K % 32 == 0): lane (h, col) of chunk c is ONE 16 B load of the
            // 512 contiguous bytes of (tile, c), already the A fragment (row col, k 16 h .. 16 h + 15 of the chunk): the same bytes the row-major
            // path stages through LDS, so the sums are bit for bit the same
#pragma unroll
            for (int jj = 0; jj < NJ / 4; ++jj) {
                const int kr = ws + 128 * (wave + kF8rWaves * jj);
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    wr[jj][i] = make_uint4(0, 0, 0, 0);
                    const int kc = (kr >> 5) + i;
                    if (!(abl & 4) && kr - ws < kwa && kc * 32 + 32 <= K)
                        wr[jj][i] = ik_ldcs(reinterpret_cast<const uint4*>(w + ((blk * (K >> 5) + kc) * 32 + lane) * 16));
                }
            }
        } else {
#pragma unroll
            for (int jj = 0; jj < NJ / 4; ++jj) {
                const int kr = ws + 128 * (wave + kF8rWaves * jj);  // first k of this wave's jj-th range
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    wr[jj][i] = make_uint4(0, 0, 0, 0);
                    const int64_t row = blk * 16 + 4 * i + (lane >> 3);
                    const int kp = kr + 16 * (lane & 7);
                    if (!(abl & 4) && row < N && kr - ws < kwa && kp + 16 <= K) wr[jj][i] = ik_ldcs(reinterpret_cast<const uint4*>(w + row * K + kp));
                }
            }
        }
        if constexpr (DIRECT) {
#pragma unroll
            for (int j = 0; j < NJ; ++j) {
                const int kl = slot_k(j), k0 = ws + kl;
#pragma unroll
                for (int t = 0; t < TM; ++t) {
                    const int row = t * 16 + col;
                    xb[t][j][0] = make_uint4(0, 0, 0, 0);
                    xb[t][j][1] = make_uint4(0, 0, 0, 0);
                    if (row < M && kl < kwa && k0 + 16 <= K) {
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
            for (int i = tid; i < ((abl & 1) ? 0 : M * c8); i += kF8rThreads) {
                const int row = i / c8, kc = (i - row * c8) << 3;
                const uint4 v = f8r_src<PRO>(x, pz, pnw, peps, pwoff, xn, row, K, ws + kc);
                if (actq) *reinterpret_cast<uint2*>(xs + (size_t)row * stride + kc) = f8r_quant8(v, s_in);
                else *reinterpret_cast<uint4*>(xs + (size_t)row * stride + kc * 2) = v;
            }
            if (!(abl & 1)) __syncthreads();
        }
#pragma unroll
        for (int j = 0; j < NJ; ++j) {
            const int kl = slot_k(j), k0 = ws + kl;
            if constexpr (!BF && !FRAG) {
                if ((j & 3) == 0) {  // a new range: this wave's tile <- its 16 rows x 128 B, then wave-local order
#pragma unroll
                    for (int i = 0; i < 4; ++i)
                        *reinterpret_cast<uint4*>(wbuf + (size_t)(4 * i + (lane >> 3)) * kF8rWRow + 16 * (lane & 7)) = wr[j >> 2][i];
                    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
                    __builtin_amdgcn_wave_barrier();
                    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
                }
            }
            if (kl >= kwa || k0 + 16 > K) continue;
            const bool full = k0 + 32 <= K;
            uint4 wa;  // this lane's weights of the chunk: bytes 16 h .. 16 h + 15 (a half: 8 h .. 8 h + 7)
            if constexpr (BF) {
                wa = wb[j][0];
            } else if constexpr (FRAG) {
                wa = wr[j >> 2][j & 3];  // K % 32 == 0: every chunk is full
            } else {
                const unsigned char* tp = wbuf + (size_t)col * kF8rWRow + 32 * (j & 3);
                if (full) wa = *reinterpret_cast<const uint4*>(tp + 16 * h);
                else {
                    const uint2 u = *reinterpret_cast<const uint2*>(tp + 8 * h);
                    wa = make_uint4(u.x, u.y, 0, 0);
                }
            }
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
                    acc[t] = ik_f8_wmma(wa.x, wa.y, b.x, b.y, acc[t]);
                    if (full) acc[t] = ik_f8_wmma(wa.z, wa.w, b.z, b.w, acc[t]);
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
                        acc[t] = ik_dq_wmma(wa, b0, acc[t]);
                        if (full) acc[t] = ik_dq_wmma(wb[j][1], b1, acc[t]);
                    } else {
                        acc[t] = ik_dq_wmma(ik_e4m3x8_to_bf16(wa.x, wa.y), b0, acc[t]);
                        if (full) acc[t] = ik_dq_wmma(ik_e4m3x8_to_bf16(wa.z, wa.w), b1, acc[t]);
                    }
                }
            }
        }
        if (!DIRECT && ws + kwa < K) __syncthreads();  // the next window restages x
    }
    if (abl & 2) {  // ablation: wave 0 stores its own partial, no cross-wave traffic
        if (wave == 0) {
#pragma unroll
            for (int t = 0; t < TM; ++t)
                for (int e = 0; e < 8; ++e) {
                    const int tok = t * 16 + col;
                    const int64_t f = blk * 16 + h * 8 + e;
                    if (tok < M && f < N) ik_store(out + (int64_t)tok * N + f, acc[t][e] * s_out);
                }
        }
        return;
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

template <int TM, int NJ, bool DIRECT, bool BF, typename OutT, bool FRAG = false>
__global__ void __launch_bounds__(kF8rThreads)
f8r_kernel(const ik_bf16* __restrict__ x, const uint8_t* __restrict__ xq, const uint8_t* __restrict__ w, const float* __restrict__ sc,
           OutT* __restrict__ out, int M, int N, int K, int kw, int abl) {
    f8r_body<TM, NJ, 0, DIRECT, BF, OutT, FRAG>(x, xq, w, sc, out, M, N, K, kw, (int64_t)blockIdx.x, nullptr, nullptr, 0.f, 0.f, nullptr,
                                                nullptr, false, abl);
}

/* The folded decode launchers (M == 1, STAGED): the same body with the prologue of the fp8 GEMV folds, so a solo step runs the
 * family's arithmetic on the row the unfused chain would stage. */
template <int NJ, bool FRAG = false>
__global__ void __launch_bounds__(kF8rThreads)
f8r_pair_kernel(const ik_bf16* __restrict__ x, const uint8_t* __restrict__ w0, const float* __restrict__ sc0,
                ik_bf16* __restrict__ o0, int N0, const uint8_t* __restrict__ w1, const float* __restrict__ sc1,
                ik_bf16* __restrict__ o1, int N1, int K, int kw) {
    const int nb0 = (N0 + 15) / 16;
    if ((int)blockIdx.x < nb0) f8r_body<1, NJ, 0, false, false, ik_bf16, FRAG>(x, nullptr, w0, sc0, o0, 1, N0, K, kw, (int64_t)blockIdx.x);
    else f8r_body<1, NJ, 0, false, false, ik_bf16, FRAG>(x, nullptr, w1, sc1, o1, 1, N1, K, kw, (int64_t)blockIdx.x - nb0);
}

template <int NJ, bool FRAG = false>
__global__ void __launch_bounds__(kF8rThreads)
f8r_gated_kernel(const ik_bf16* __restrict__ x, const ik_bf16* __restrict__ z, const ik_bf16* __restrict__ nw, float eps, float woff,
                 const uint8_t* __restrict__ w, const float* __restrict__ sc, ik_bf16* __restrict__ out, int N, int K, int kw) {
    f8r_body<1, NJ, 2, false, false, ik_bf16, FRAG>(x, nullptr, w, sc, out, 1, N, K, kw, (int64_t)blockIdx.x, z, nw, eps, woff);
}

template <int NJ, bool FRAG = false>
__global__ void __launch_bounds__(kF8rThreads)
f8r_norm_pair_kernel(const ik_bf16* __restrict__ x, const ik_bf16* __restrict__ res, const ik_bf16* __restrict__ nw, float eps,
                     float woff, ik_bf16* __restrict__ normed, ik_bf16* __restrict__ sum, const uint8_t* __restrict__ w0,
                     const float* __restrict__ sc0, ik_bf16* __restrict__ o0, int N0, const uint8_t* __restrict__ w1,
                     const float* __restrict__ sc1, ik_bf16* __restrict__ o1, int N1, int K, int kw) {
    const int nb0 = (N0 + 15) / 16;
    const bool writer = blockIdx.x == 0;
    if ((int)blockIdx.x < nb0)
        f8r_body<1, NJ, 3, false, false, ik_bf16, FRAG>(x, nullptr, w0, sc0, o0, 1, N0, K, kw, (int64_t)blockIdx.x, res, nw, eps, woff, normed,
                                                 sum, writer);
    else
        f8r_body<1, NJ, 3, false, false, ik_bf16, FRAG>(x, nullptr, w1, sc1, o1, 1, N1, K, kw, (int64_t)blockIdx.x - nb0, res, nw, eps, woff,
                                                 normed, sum, writer);
}

template <int NJ, bool FRAG = false>
__global__ void __launch_bounds__(kF8rThreads)
f8r_sigmoid_kernel(const ik_bf16* __restrict__ a, const ik_bf16* __restrict__ gate, const uint8_t* __restrict__ w,
                   const float* __restrict__ sc, ik_bf16* __restrict__ out, int N, int K, int kw) {
    f8r_body<1, NJ, 4, false, false, ik_bf16, FRAG>(a, nullptr, w, sc, out, 1, N, K, kw, (int64_t)blockIdx.x, gate);
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
inline bool f8r_cfg(int64_t M, int64_t K, bool single_window, size_t extra, bool direct, int bpk, F8rCfg* c, bool frag = false) {
    if (M < 1 || M > 64) return false;
    c->tm = (int)((M + 15) / 16);
    c->direct = direct;
    const bool bf = bpk == 2;  // bf16 weights (their own chunk map, no weight tile); bpk 1 = fp8 weights
    if (direct) {
        c->nj = c->tm >= 3 ? 4 : 8;
        c->kw = c->nj * 256;
        c->smem = std::max<size_t>((size_t)c->tm * kF8rSpBytes, bf || frag ? 0 : (size_t)kF8rWBytes);  // the weight tile and the cross-wave buffer share
        return !single_window;
    }
    if (bf) {
        const int64_t kmax = (K + 255) / 256 * 256;
        // bpk: bytes of LDS per staged k (2: bf16 activations)
        int64_t kw = (((int64_t)kF8rLdsBudget - (int64_t)extra) / M - kF8rPad) / bpk;
        kw = kw / 256 * 256;
        kw = std::min<int64_t>(kw, std::min<int64_t>(kmax, 4096));
        if (single_window && kw < kmax) return false;
        if (kw < 512) {
            if (kmax < 512) kw = 512;  // K <= 256: the window is wider than K (the staged row is K bytes)
            else return false;
        }
        c->kw = (int)kw;
        c->nj = kw > 2048 ? 16 : 8;
        c->smem = std::max<size_t>((size_t)M * (kw * bpk + kF8rPad), (size_t)c->tm * kF8rSpBytes) + extra;
        return true;
    }
    // fp8 weights: windows are multiples of 1024 k (8 waves x 128 k); 2048 or 4096
    const int64_t kmax = (K + 1023) / 1024 * 1024;
    const int64_t wtile = frag ? 0 : (int64_t)kF8rWBytes;  // fragment order loads straight into registers: no weight tile
    int64_t kw = ((int64_t)kF8rLdsBudget - wtile - (int64_t)extra) / M - kF8rPad;
    kw = kw >= 4096 ? 4096 : kw >= 2048 ? 2048 : 0;
    if (kw == 0) return false;
    if (kw > 2048 && kmax <= 2048) kw = 2048;
    if (single_window && kw < kmax) return false;
    c->kw = (int)kw;
    c->nj = kw / 256;
    c->smem = std::max<size_t>((size_t)M * (kw + kF8rPad), (size_t)c->tm * kF8rSpBytes) + extra + (size_t)wtile;
    return true;
}

template <int TM, int NJ, bool DIRECT, bool BF, typename OutT, bool FRAG = false>
void f8r_run(const ik_bf16* x, const uint8_t* xq, const uint8_t* w, const float* sc, OutT* out, int M, int64_t N, int64_t K,
             const F8rCfg& c, hipStream_t s) {
    const unsigned grid = (unsigned)((N + 15) / 16);
#ifdef IK_F8R_ABLATE
    f8r_kernel<TM, NJ, DIRECT, BF, OutT, FRAG><<<grid, kF8rThreads, c.smem, s>>>(x, xq, w, sc, out, M, (int)N, (int)K, c.kw, g_f8r_abl);
#else
    f8r_kernel<TM, NJ, DIRECT, BF, OutT, FRAG><<<grid, kF8rThreads, c.smem, s>>>(x, xq, w, sc, out, M, (int)N, (int)K, c.kw, 0);
#endif
}

/* M rows, STAGED or (scratch given, M >= direct_min) DIRECT. */
template <bool BF, typename OutT, bool FRAG = false>
int f8r_dispatch(const ik_bf16* x, const uint8_t* w, const float* sc, OutT* out, int M, int64_t N, int64_t K, uint8_t* scratch,
                 int direct_min, hipStream_t s) {
    const bool can_direct = BF || scratch != nullptr;
    bool direct = can_direct && M >= direct_min;
    F8rCfg c;
    // a staged window must fit the LDS (it does not for 64 bf16 rows at K >= 512): the direct form takes those calls; same bits
    if (!direct && !f8r_cfg(M, K, false, 0, false, BF ? 2 : 1, &c, FRAG)) direct = can_direct;
    if (!f8r_cfg(M, K, false, 0, direct, BF ? 2 : 1, &c, FRAG)) return IK_ERR_SHAPE;
    if (direct && !BF) {
        const int64_t n8 = (int64_t)M * K / 8;
        f8r_actq_kernel<<<(unsigned)((n8 + 255) / 256), 256, 0, s>>>(x, scratch, sc, n8);
    }
#define IK_F8R_CASE(T)                                                                                          \
    case T:                                                                                                     \
        if (direct) {                                                                                           \
            if (c.nj == 4) f8r_run<T, 4, true, BF, OutT, FRAG>(x, scratch, w, sc, out, M, N, K, c, s);                    \
            else f8r_run<T, 8, true, BF, OutT, FRAG>(x, scratch, w, sc, out, M, N, K, c, s);                              \
        } else if (c.nj == 16) f8r_run<T, 16, false, BF, OutT, FRAG>(x, nullptr, w, sc, out, M, N, K, c, s);              \
        else f8r_run<T, 8, false, BF, OutT, FRAG>(x, nullptr, w, sc, out, M, N, K, c, s);                                 \
        break;
    switch (c.tm) {
        IK_F8R_CASE(1) IK_F8R_CASE(2) IK_F8R_CASE(3) IK_F8R_CASE(4)
        default: return IK_ERR_SHAPE;
    }
#undef IK_F8R_CASE
    return ik_launch_status();
}

#endif  // IK_F8_ROWS_H
