// RMSNorm family: rmsnorm, rmsnorm_add (residual add fused), qk_norm (per-head rmsnorm).
#include "ik_hip.h"
#include "ik_mx_a8.h"  // lane fuse27: the W4A8 activation quant of the normed rows (rmsnorm_kernel<.., true>)

namespace {

// One block per row. out = bf16( x * rsqrt(mean(x^2) + eps) * (w + woff) ), woff = 1 for zero-centred
// norms (Qwen3.5), added in f32 so the weight keeps its full precision. With ADD the row is first
// summed with `res`, rounded to bf16 (that rounded sum is what the next layer carries) and written
// to `sum`; the normalisation then reads the rounded sum, exactly like the reference.
//
// Lane fuse27, QUANT (VEC only, H / 8 <= kNqKeep * blockDim.x): the block also quantises its normed row for the W4A8 prefill GEMM's
// direct kernel, exactly as mx_a8_quant_tiled_kernel (ik_mx_a8.h) would from `out`: the same bf16 values (kept in registers as stored),
// the same per-row scale (an order-independent max) and the same e4m3 bytes, written in its fragment order into xt with the scales at xs.
// The grid is the 16-padded row count: blocks row >= M write their tile row as zeros (and the K % 64 == 32 pad group of every row) and
// touch nothing else. The norm itself (the ss loop, the block sum, the normed values) is the same code as without QUANT.
constexpr int kNqKeep = 4;
template <bool ADD, bool VEC, bool QUANT = false>
__global__ void rmsnorm_kernel(const ik_bf16* __restrict__ x, const ik_bf16* __restrict__ res,
                               const ik_bf16* __restrict__ w, ik_bf16* __restrict__ out, ik_bf16* __restrict__ sum,
                               int64_t H, float eps, float woff, uint8_t* __restrict__ xt = nullptr, float* __restrict__ xs = nullptr,
                               int64_t M = 0) {
    __shared__ float red[32];
    const int64_t row = blockIdx.x;
    uint8_t* tile = nullptr;
    if (QUANT) {
        // chunk c (k = 8 c) of the row: group c >> 2, half (c >> 1) & 1, byte (c & 1) * 8 of the lane's 16 (mx_a8_quant_tiled_kernel)
        const int C = (int)(H >> 3), CP = (int)((H + 63) >> 6) * 8;
        tile = xt + (row >> 4) * (int64_t)(CP / 4) * 512 + (row & 15) * 16;
        for (int c = C + threadIdx.x; c < CP; c += blockDim.x)
            *reinterpret_cast<uint2*>(tile + (c >> 2) * 512 + ((c >> 1) & 1) * 256 + (c & 1) * 8) = make_uint2(0u, 0u);
        if (row >= M) {  // block-uniform
            for (int c = threadIdx.x; c < C; c += blockDim.x)
                *reinterpret_cast<uint2*>(tile + (c >> 2) * 512 + ((c >> 1) & 1) * 256 + (c & 1) * 8) = make_uint2(0u, 0u);
            return;
        }
    }
    const ik_bf16* xr = x + row * H;
    const ik_bf16* rr = ADD ? res + row * H : nullptr;
    ik_bf16* orow = out + row * H;
    ik_bf16* sr = ADD ? sum + row * H : nullptr;
    // the row the normalisation reads back (sum when ADD, else x)
    const ik_bf16* src = ADD ? sr : xr;

    float ss = 0.f;
    if (VEC) {
        const int64_t nv = H / 8;
        for (int64_t i = threadIdx.x; i < nv; i += blockDim.x) {
            float f[8];
            ik_unpack8(reinterpret_cast<const uint4*>(xr)[i], f);
            if (ADD) {
                float g[8];
                ik_unpack8(reinterpret_cast<const uint4*>(rr)[i], g);
#pragma unroll
                for (int k = 0; k < 8; ++k) f[k] += g[k];
                const uint4 p = ik_pack8(f);
                reinterpret_cast<uint4*>(sr)[i] = p;
                ik_unpack8(p, f);  // continue from the bf16-rounded sum
            }
#pragma unroll
            for (int k = 0; k < 8; ++k) ss += f[k] * f[k];
        }
    } else {
        for (int64_t i = threadIdx.x; i < H; i += blockDim.x) {
            float v = ik_b2f(xr[i]);
            if (ADD) {
                const ik_bf16 b = ik_f2b(v + ik_b2f(rr[i]));
                sr[i] = b;
                v = ik_b2f(b);
            }
            ss += v * v;
        }
    }
    ss = ik_block_sum(ss, red);
    const float r = rsqrtf(ss / (float)H + eps);

    if (QUANT) {
        // the normed chunks this thread stores (each element's value does not depend on the loop's shape), kept for the quant
        const int64_t nv = H / 8;
        uint4 keep[kNqKeep];
        float amax = 0.f;
#pragma unroll
        for (int j = 0; j < kNqKeep; ++j) {
            const int64_t i = threadIdx.x + (int64_t)j * blockDim.x;
            if (i < nv) {
                float f[8], g[8];
                ik_unpack8(reinterpret_cast<const uint4*>(src)[i], f);
                ik_unpack8(reinterpret_cast<const uint4*>(w)[i], g);
#pragma unroll
                for (int k = 0; k < 8; ++k) f[k] = f[k] * r * (g[k] + woff);
                keep[j] = ik_pack8(f);
                reinterpret_cast<uint4*>(orow)[i] = keep[j];
                ik_unpack8(keep[j], f);  // the stored bf16 values, as mx_a8_row_scale reads them back
#pragma unroll
                for (int k = 0; k < 8; ++k) amax = fmaxf(amax, fabsf(f[k]));
            }
        }
        amax = ik_block_max(amax, red);
        const float s = fmaxf(__fdiv_rn(amax, kMxA8Max), kMxA8MinScale);
        if (threadIdx.x == 0) xs[row] = s;
#pragma unroll
        for (int j = 0; j < kNqKeep; ++j) {
            const int64_t i = threadIdx.x + (int64_t)j * blockDim.x;
            if (i < nv) {
                const int c = (int)i;
                *reinterpret_cast<uint2*>(tile + (c >> 2) * 512 + ((c >> 1) & 1) * 256 + (c & 1) * 8) = mx_a8_quant8(keep[j], s);
            }
        }
    } else if (VEC) {
        const int64_t nv = H / 8;
        for (int64_t i = threadIdx.x; i < nv; i += blockDim.x) {
            float f[8], g[8];
            ik_unpack8(reinterpret_cast<const uint4*>(src)[i], f);
            ik_unpack8(reinterpret_cast<const uint4*>(w)[i], g);
#pragma unroll
            for (int k = 0; k < 8; ++k) f[k] = f[k] * r * (g[k] + woff);
            reinterpret_cast<uint4*>(orow)[i] = ik_pack8(f);
        }
    } else {
        for (int64_t i = threadIdx.x; i < H; i += blockDim.x) orow[i] = ik_f2b(ik_b2f(src[i]) * r * (ik_b2f(w[i]) + woff));
    }
}

// The GatedDeltaNet output norm: out = silu(z) * rmsnorm(x, w), one block per row. It is rmsnorm_kernel<false, VEC> (same block
// size, same per-thread accumulation and block reduction, same bf16 rounding of the normalised value) followed by silu_glu_kernel
// on (z, normalised): the bits of the two launches it replaces. `out` may be `x` (each element is read, then written, by the
// same thread after the reduction barrier).
template <bool VEC>
__global__ void gated_rmsnorm_kernel(const ik_bf16* __restrict__ x, const ik_bf16* __restrict__ z,
                                     const ik_bf16* __restrict__ w, ik_bf16* out, int64_t H, float eps, float woff) {
    __shared__ float red[32];
    const int64_t row = blockIdx.x;
    const ik_bf16* xr = x + row * H;
    const ik_bf16* zr = z + row * H;
    ik_bf16* orow = out + row * H;
    float ss = 0.f;
    if (VEC) {
        const int64_t nv = H / 8;
        for (int64_t i = threadIdx.x; i < nv; i += blockDim.x) {
            float f[8];
            ik_unpack8(reinterpret_cast<const uint4*>(xr)[i], f);
#pragma unroll
            for (int k = 0; k < 8; ++k) ss += f[k] * f[k];
        }
    } else {
        for (int64_t i = threadIdx.x; i < H; i += blockDim.x) {
            const float v = ik_b2f(xr[i]);
            ss += v * v;
        }
    }
    ss = ik_block_sum(ss, red);
    const float r = rsqrtf(ss / (float)H + eps);
    if (VEC) {
        const int64_t nv = H / 8;
        for (int64_t i = threadIdx.x; i < nv; i += blockDim.x) {
            float f[8], g[8], zf[8];
            ik_unpack8(reinterpret_cast<const uint4*>(xr)[i], f);
            ik_unpack8(reinterpret_cast<const uint4*>(w)[i], g);
            ik_unpack8(reinterpret_cast<const uint4*>(zr)[i], zf);
#pragma unroll
            for (int k = 0; k < 8; ++k) f[k] = f[k] * r * (g[k] + woff);
            float nrm[8];
            ik_unpack8(ik_pack8(f), nrm);  // the normalised row as the rmsnorm launch stores it (bf16)
#pragma unroll
            for (int k = 0; k < 8; ++k) f[k] = ik_silu(zf[k]) * nrm[k];
            reinterpret_cast<uint4*>(orow)[i] = ik_pack8(f);
        }
    } else {
        for (int64_t i = threadIdx.x; i < H; i += blockDim.x) {
            const float nrm = ik_b2f(ik_f2b(ik_b2f(xr[i]) * r * (ik_b2f(w[i]) + woff)));
            orow[i] = ik_f2b(ik_silu(ik_b2f(zr[i])) * nrm);
        }
    }
}

// Lane nq27: the producers of the W4A8 GEMM's other activations (silu_glu before the FFN down projection, gated_rmsnorm before the GDN out
// projection) also write the GEMM's quantised activations, the way rmsnorm_kernel's QUANT does: the layout and bytes of mx_a8_quant_tiled_kernel
// from the bf16 values as stored, an order-free row max, the grid the 16-padded row count (rows >= M and the K % 64 == 32 pad group zeroed).
// chunk c (k = 8 c) of a row's 16-byte lane slot in its tile
IK_DEVI uint2* nq_chunk(uint8_t* tile, int c) {
    return reinterpret_cast<uint2*>(tile + (c >> 2) * 512 + ((c >> 1) & 1) * 256 + (c & 1) * 8);
}

// silu_glu_kernel<true> over whole rows of K (one block per row, blockDim.x * kNqKeep >= K / 8 chunks) that also quantises its output: out =
// bf16(silu(g) * u), the same expression per element as silu_glu_kernel, so the stored bytes are the plain kernel's.
__global__ void silu_glu_quant_kernel(const ik_bf16* __restrict__ g, const ik_bf16* __restrict__ u, ik_bf16* __restrict__ out, int64_t K,
                                      uint8_t* __restrict__ xt, float* __restrict__ xs, int64_t M) {
    __shared__ float red[32];
    const int64_t row = blockIdx.x;
    const int C = (int)(K >> 3), CP = (int)((K + 63) >> 6) * 8;
    uint8_t* tile = xt + (row >> 4) * (int64_t)(CP / 4) * 512 + (row & 15) * 16;
    for (int c = C + threadIdx.x; c < CP; c += blockDim.x) *nq_chunk(tile, c) = make_uint2(0u, 0u);
    if (row >= M) {  // block-uniform
        for (int c = threadIdx.x; c < C; c += blockDim.x) *nq_chunk(tile, c) = make_uint2(0u, 0u);
        return;
    }
    const uint4* gr = reinterpret_cast<const uint4*>(g + row * K);
    const uint4* ur = reinterpret_cast<const uint4*>(u + row * K);
    uint4* orow = reinterpret_cast<uint4*>(out + row * K);
    uint4 keep[kNqKeep];
    float amax = 0.f;
#pragma unroll
    for (int j = 0; j < kNqKeep; ++j) {
        const int i = threadIdx.x + j * blockDim.x;
        if (i < C) {
            float fg[8], fu[8];
            ik_unpack8(gr[i], fg);
            ik_unpack8(ur[i], fu);
#pragma unroll
            for (int k = 0; k < 8; ++k) fg[k] = ik_silu(fg[k]) * fu[k];
            keep[j] = ik_pack8(fg);
            orow[i] = keep[j];
            ik_unpack8(keep[j], fg);  // the stored bf16 values
#pragma unroll
            for (int k = 0; k < 8; ++k) amax = fmaxf(amax, fabsf(fg[k]));
        }
    }
    amax = ik_block_max(amax, red);
    const float s = fmaxf(__fdiv_rn(amax, kMxA8Max), kMxA8MinScale);
    if (threadIdx.x == 0) xs[row] = s;
#pragma unroll
    for (int j = 0; j < kNqKeep; ++j) {
        const int i = threadIdx.x + j * blockDim.x;
        if (i < C) *nq_chunk(tile, i) = mx_a8_quant8(keep[j], s);
    }
}

// gated_rmsnorm_kernel<true> (rows of H = dv, one warp's lanes per head) whose output the W4A8 GEMM reads as [tokens, heads * H]: one block per
// 16-padded TOKEN, thread t = chunk t of the token's heads * H elements (L = H / 8 lanes per head, L a power of two <= 32 so a head lies in one
// warp). Each head's sum of squares is the plain kernel's: its 8-term per-lane sum, then the xor tree over the head's lanes (the plain kernel's
// 32-lane tree adds exact zeros above the L lanes), so rsqrt(ss / H + eps) and every stored element are the plain kernel's bits. The row scale is
// the max over the whole token (all heads).
__global__ void gated_rmsnorm_quant_kernel(const ik_bf16* __restrict__ x, const ik_bf16* __restrict__ z, const ik_bf16* __restrict__ w,
                                           ik_bf16* __restrict__ out, int64_t H, int heads, float eps, float woff, uint8_t* __restrict__ xt,
                                           float* __restrict__ xs, int64_t M) {
    __shared__ float red[32];
    const int64_t row = blockIdx.x;
    const int L = (int)(H >> 3), C = heads * L;
    const int64_t K = (int64_t)heads * H;
    const int CP = (int)((K + 63) >> 6) * 8;
    uint8_t* tile = xt + (row >> 4) * (int64_t)(CP / 4) * 512 + (row & 15) * 16;
    for (int c = C + threadIdx.x; c < CP; c += blockDim.x) *nq_chunk(tile, c) = make_uint2(0u, 0u);
    if (row >= M) {  // block-uniform
        for (int c = threadIdx.x; c < C; c += blockDim.x) *nq_chunk(tile, c) = make_uint2(0u, 0u);
        return;
    }
    const int t = threadIdx.x;
    const bool valid = t < C;
    float f[8], zf[8], gw[8];
    float ss = 0.f;
    if (valid) {
        ik_unpack8(reinterpret_cast<const uint4*>(x + row * K)[t], f);
#pragma unroll
        for (int k = 0; k < 8; ++k) ss += f[k] * f[k];
    }
    for (int o = L >> 1; o > 0; o >>= 1) ss += __shfl_xor(ss, o, IK_WARP);
    const float r = rsqrtf(ss / (float)H + eps);
    uint4 keep = make_uint4(0u, 0u, 0u, 0u);
    float amax = 0.f;
    if (valid) {
        ik_unpack8(reinterpret_cast<const uint4*>(w)[t & (L - 1)], gw);
        ik_unpack8(reinterpret_cast<const uint4*>(z + row * K)[t], zf);
#pragma unroll
        for (int k = 0; k < 8; ++k) f[k] = f[k] * r * (gw[k] + woff);
        float nrm[8];
        ik_unpack8(ik_pack8(f), nrm);  // the normalised row as the rmsnorm launch stores it (bf16)
#pragma unroll
        for (int k = 0; k < 8; ++k) f[k] = ik_silu(zf[k]) * nrm[k];
        keep = ik_pack8(f);
        reinterpret_cast<uint4*>(out + row * K)[t] = keep;
        ik_unpack8(keep, f);  // the stored bf16 values
#pragma unroll
        for (int k = 0; k < 8; ++k) amax = fmaxf(amax, fabsf(f[k]));
    }
    amax = ik_block_max(amax, red);
    const float s = fmaxf(__fdiv_rn(amax, kMxA8Max), kMxA8MinScale);
    if (threadIdx.x == 0) xs[row] = s;
    if (valid) *nq_chunk(tile, t) = mx_a8_quant8(keep, s);
}

int launch_rows(const ik_bf16* x, const ik_bf16* res, const ik_bf16* w, ik_bf16* out, ik_bf16* sum, int64_t rows,
                int64_t H, float eps, float woff, hipStream_t s) {
    if (rows == 0 || H == 0) return IK_OK;
    const bool add = res != nullptr;
    bool vec = H % 8 == 0 && ik_aligned16(x) && ik_aligned16(w) && ik_aligned16(out);
    if (add) vec = vec && ik_aligned16(res) && ik_aligned16(sum);
    const int64_t items = vec ? H / 8 : H;
    const unsigned bd = ik_round_up((unsigned)(items < 256 ? items : 256), 32);
    if (rows > 0x7fffffffLL) return IK_ERR_SHAPE;
    const unsigned g = (unsigned)rows;
    if (add) {
        if (vec) rmsnorm_kernel<true, true><<<g, bd, 0, s>>>(x, res, w, out, sum, H, eps, woff);
        else rmsnorm_kernel<true, false><<<g, bd, 0, s>>>(x, res, w, out, sum, H, eps, woff);
    } else {
        if (vec) rmsnorm_kernel<false, true><<<g, bd, 0, s>>>(x, nullptr, w, out, nullptr, H, eps, woff);
        else rmsnorm_kernel<false, false><<<g, bd, 0, s>>>(x, nullptr, w, out, nullptr, H, eps, woff);
    }
    return ik_launch_status();
}

int launch_gated(const ik_bf16* x, const ik_bf16* z, const ik_bf16* w, ik_bf16* out, int64_t rows, int64_t H, float eps,
                 float woff, hipStream_t s) {
    if (rows == 0 || H == 0) return IK_OK;
    // the same vector / block-size decisions as rmsnorm (launch_rows), so the reduction order is the same; silu_glu's own
    // vector decision does not change a single element's value
    const bool vec = H % 8 == 0 && ik_aligned16(x) && ik_aligned16(w) && ik_aligned16(out) && ik_aligned16(z);
    const int64_t items = vec ? H / 8 : H;
    const unsigned bd = ik_round_up((unsigned)(items < 256 ? items : 256), 32);
    if (rows > 0x7fffffffLL) return IK_ERR_SHAPE;
    if (vec) gated_rmsnorm_kernel<true><<<(unsigned)rows, bd, 0, s>>>(x, z, w, out, H, eps, woff);
    else gated_rmsnorm_kernel<false><<<(unsigned)rows, bd, 0, s>>>(x, z, w, out, H, eps, woff);
    return ik_launch_status();
}

// Lane fuse27: launch_rows for the vector path plus the W4A8 quant of `out` (rmsnorm_kernel<ADD, true, true>), the same block size;
// IK_ERR_UNSUPPORTED when these operands do not take the vector path or the row does not fit the kept chunks (the caller runs the plain launch).
int launch_rows_quant(const ik_bf16* x, const ik_bf16* res, const ik_bf16* w, ik_bf16* out, ik_bf16* sum, int64_t rows, int64_t H, float eps,
                      float woff, void* xq, hipStream_t s) {
    const bool add = res != nullptr;
    bool vec = H % 8 == 0 && ik_aligned16(x) && ik_aligned16(w) && ik_aligned16(out) && ik_aligned16(xq);
    if (add) vec = vec && ik_aligned16(res) && ik_aligned16(sum);
    const int64_t items = H / 8;
    const unsigned bd = ik_round_up((unsigned)(items < 256 ? items : 256), 32);
    if (!vec || rows <= 0 || H % 32 != 0 || items > (int64_t)kNqKeep * bd || rows > 0x7fffffe0LL || H > 0x7fffffffLL) return IK_ERR_UNSUPPORTED;
    const int64_t Mp = (rows + 15) & ~(int64_t)15, Kp = (H + 63) & ~(int64_t)63;
    uint8_t* xt = static_cast<uint8_t*>(xq);
    float* xs = reinterpret_cast<float*>(xt + mx_a8_xs_offset(Mp, Kp));
    if (add) rmsnorm_kernel<true, true, true><<<(unsigned)Mp, bd, 0, s>>>(x, res, w, out, sum, H, eps, woff, xt, xs, rows);
    else rmsnorm_kernel<false, true, true><<<(unsigned)Mp, bd, 0, s>>>(x, nullptr, w, out, nullptr, H, eps, woff, xt, xs, rows);
    return ik_launch_status();
}

// Lane nq27: silu_glu over [rows, K] plus the W4A8 quant of its output into xq (layout as launch_rows_quant); IK_ERR_UNSUPPORTED (nothing launched)
// when the operands do not take the plain launch's vector path or a row does not fit the kept chunks.
int launch_silu_quant(const ik_bf16* g, const ik_bf16* u, ik_bf16* out, int64_t rows, int64_t K, void* xq, hipStream_t s) {
    const int64_t C = K / 8;
    const bool vec = K % 32 == 0 && ik_aligned16(g) && ik_aligned16(u) && ik_aligned16(out) && ik_aligned16(xq);
    if (!vec || rows <= 0 || C > 1024 * (int64_t)kNqKeep || rows > 0x7fffffe0LL) return IK_ERR_UNSUPPORTED;
    const unsigned bd = ik_round_up((unsigned)((C + kNqKeep - 1) / kNqKeep), 32);
    const int64_t Mp = (rows + 15) & ~(int64_t)15, Kp = (K + 63) & ~(int64_t)63;
    uint8_t* xt = static_cast<uint8_t*>(xq);
    float* xs = reinterpret_cast<float*>(xt + mx_a8_xs_offset(Mp, Kp));
    silu_glu_quant_kernel<<<(unsigned)Mp, bd, 0, s>>>(g, u, out, K, xt, xs, rows);
    return ik_launch_status();
}

// Lane nq27: gated_rmsnorm over [rows = tokens * heads, H] plus the W4A8 quant of its output read as [tokens, heads * H] into xq; IK_ERR_UNSUPPORTED
// (nothing launched) unless the plain launch takes its single-warp vector path (H / 8 a power of two <= 32) and a token's chunks fit one block.
int launch_gated_quant(const ik_bf16* x, const ik_bf16* z, const ik_bf16* w, ik_bf16* out, int64_t rows, int64_t H, int64_t heads, float eps,
                       float woff, void* xq, hipStream_t s) {
    const int64_t L = H / 8;
    const bool vec = H % 8 == 0 && ik_aligned16(x) && ik_aligned16(z) && ik_aligned16(w) && ik_aligned16(out) && ik_aligned16(xq);
    if (!vec || heads <= 0 || rows <= 0 || rows % heads != 0 || L < 1 || L > 32 || (L & (L - 1)) != 0 || (heads * H) % 32 != 0 ||
        heads * L > 1024 || rows / heads > 0x7fffffe0LL)
        return IK_ERR_UNSUPPORTED;
    const int64_t M = rows / heads, K = heads * H;
    const int64_t Mp = (M + 15) & ~(int64_t)15, Kp = (K + 63) & ~(int64_t)63;
    uint8_t* xt = static_cast<uint8_t*>(xq);
    float* xs = reinterpret_cast<float*>(xt + mx_a8_xs_offset(Mp, Kp));
    const unsigned bd = ik_round_up((unsigned)(heads * L), 32);
    gated_rmsnorm_quant_kernel<<<(unsigned)Mp, bd, 0, s>>>(x, z, w, out, H, (int)heads, eps, woff, xt, xs, M);
    return ik_launch_status();
}

}  // namespace

extern "C" {

// Lane fuse27: ik_rmsnorm_add_bf16 (sum_out non-null, res read) or ik_rmsnorm_bf16 (res and sum_out null) that also writes the W4A8 prefill
// GEMM's quantised activations of `out` into xq (the direct pre-pass's layout: Mp = round16(rows), Kp = round64(H); fragment-ordered e4m3 at
// 0, the f32 row scales at mx_a8_xs_offset(Mp, Kp); mx_a8_scratch_bytes(Mp, Kp) bytes). out / sum_out are bitwise the plain launch's.
// Returns IK_ERR_UNSUPPORTED (nothing launched) when the operands do not take the vector path at the plain launch's block size.
int ik_rmsnorm_quant_bf16(const ik_tensor* x, const ik_tensor* res, const ik_tensor* w, float eps, float w_offset, ik_tensor* out,
                          ik_tensor* sum_out, void* xq, void* stream) {
    IK_TRY(ik_check(x, IK_DTYPE_BF16, 2));
    IK_TRY(ik_check(w, IK_DTYPE_BF16, 1));
    IK_TRY(ik_check(out, IK_DTYPE_BF16, 2));
    const int64_t R = x->shape[0], H = x->shape[1];
    if (w->shape[0] != H || out->shape[0] != R || out->shape[1] != H) return IK_ERR_SHAPE;
    if ((res == nullptr) != (sum_out == nullptr) || !xq) return IK_ERR_INVALID_ARG;
    if (res) {
        IK_TRY(ik_check(res, IK_DTYPE_BF16, 2));
        IK_TRY(ik_check(sum_out, IK_DTYPE_BF16, 2));
        if (res->shape[0] != R || res->shape[1] != H || sum_out->shape[0] != R || sum_out->shape[1] != H) return IK_ERR_SHAPE;
    }
    return launch_rows_quant((const ik_bf16*)x->data, res ? (const ik_bf16*)res->data : nullptr, (const ik_bf16*)w->data, (ik_bf16*)out->data,
                             sum_out ? (ik_bf16*)sum_out->data : nullptr, R, H, eps, w_offset, xq, IK_STREAM(stream));
}

// Lane nq27: ik_silu_glu_bf16 that also writes the W4A8 prefill GEMM's quantised activations of `out` into xq (layout and size as
// ik_rmsnorm_quant_bf16's, for Mp = round16(rows), Kp = round64(K)); out is bitwise the plain launch's. IK_ERR_UNSUPPORTED (nothing launched) when
// the operands do not take the vector path.
int ik_silu_glu_quant_bf16(const ik_tensor* gate, const ik_tensor* up, ik_tensor* out, void* xq, void* stream) {
    IK_TRY(ik_check(gate, IK_DTYPE_BF16, 2));
    IK_TRY(ik_check(up, IK_DTYPE_BF16, 2));
    IK_TRY(ik_check(out, IK_DTYPE_BF16, 2));
    for (int d = 0; d < 2; ++d)
        if (up->shape[d] != gate->shape[d] || out->shape[d] != gate->shape[d]) return IK_ERR_SHAPE;
    if (!xq) return IK_ERR_INVALID_ARG;
    return launch_silu_quant((const ik_bf16*)gate->data, (const ik_bf16*)up->data, (ik_bf16*)out->data, gate->shape[0], gate->shape[1], xq,
                             IK_STREAM(stream));
}

// Lane nq27: ik_gated_rmsnorm_bf16 on [tokens * heads, H] rows that also writes the W4A8 GEMM's quantised activations of `out` read as
// [tokens, heads * H] into xq (layout as above for Mp = round16(tokens), Kp = round64(heads * H)); out is bitwise the plain launch's.
// IK_ERR_UNSUPPORTED (nothing launched) when the operands do not take the plain launch's single-warp vector path or heads * H / 8 > 1024.
int ik_gated_rmsnorm_quant_bf16(const ik_tensor* x, const ik_tensor* z, const ik_tensor* w, float eps, float w_offset, int64_t heads,
                                ik_tensor* out, void* xq, void* stream) {
    IK_TRY(ik_check(x, IK_DTYPE_BF16, 2));
    IK_TRY(ik_check(z, IK_DTYPE_BF16, 2));
    IK_TRY(ik_check(w, IK_DTYPE_BF16, 1));
    IK_TRY(ik_check(out, IK_DTYPE_BF16, 2));
    if (z->shape[0] != x->shape[0] || z->shape[1] != x->shape[1] || w->shape[0] != x->shape[1] || out->shape[0] != x->shape[0] ||
        out->shape[1] != x->shape[1])
        return IK_ERR_SHAPE;
    if (!xq) return IK_ERR_INVALID_ARG;
    return launch_gated_quant((const ik_bf16*)x->data, (const ik_bf16*)z->data, (const ik_bf16*)w->data, (ik_bf16*)out->data, x->shape[0], x->shape[1],
                              heads, eps, w_offset, xq, IK_STREAM(stream));
}

int ik_gated_rmsnorm_bf16(const ik_tensor* x, const ik_tensor* z, const ik_tensor* w, float eps, float w_offset,
                          ik_tensor* out, void* stream) {
    IK_TRY(ik_check(x, IK_DTYPE_BF16, 2));
    IK_TRY(ik_check(z, IK_DTYPE_BF16, 2));
    IK_TRY(ik_check(w, IK_DTYPE_BF16, 1));
    IK_TRY(ik_check(out, IK_DTYPE_BF16, 2));
    if (z->shape[0] != x->shape[0] || z->shape[1] != x->shape[1] || w->shape[0] != x->shape[1] ||
        out->shape[0] != x->shape[0] || out->shape[1] != x->shape[1])
        return IK_ERR_SHAPE;
    return launch_gated((const ik_bf16*)x->data, (const ik_bf16*)z->data, (const ik_bf16*)w->data, (ik_bf16*)out->data,
                        x->shape[0], x->shape[1], eps, w_offset, IK_STREAM(stream));
}

int ik_rmsnorm_bf16(const ik_tensor* x, const ik_tensor* w, float eps, float w_offset, ik_tensor* out, void* stream) {
    IK_TRY(ik_check(x, IK_DTYPE_BF16, 2));
    IK_TRY(ik_check(w, IK_DTYPE_BF16, 1));
    IK_TRY(ik_check(out, IK_DTYPE_BF16, 2));
    if (w->shape[0] != x->shape[1] || out->shape[0] != x->shape[0] || out->shape[1] != x->shape[1])
        return IK_ERR_SHAPE;
    return launch_rows((const ik_bf16*)x->data, nullptr, (const ik_bf16*)w->data, (ik_bf16*)out->data, nullptr,
                       x->shape[0], x->shape[1], eps, w_offset, IK_STREAM(stream));
}

int ik_rmsnorm_add_bf16(const ik_tensor* x, const ik_tensor* res, const ik_tensor* w, float eps, float w_offset,
                        ik_tensor* out, ik_tensor* sum_out, void* stream) {
    IK_TRY(ik_check(x, IK_DTYPE_BF16, 2));
    IK_TRY(ik_check(res, IK_DTYPE_BF16, 2));
    IK_TRY(ik_check(w, IK_DTYPE_BF16, 1));
    IK_TRY(ik_check(out, IK_DTYPE_BF16, 2));
    IK_TRY(ik_check(sum_out, IK_DTYPE_BF16, 2));
    const int64_t R = x->shape[0], H = x->shape[1];
    if (res->shape[0] != R || res->shape[1] != H || w->shape[0] != H || out->shape[0] != R || out->shape[1] != H ||
        sum_out->shape[0] != R || sum_out->shape[1] != H)
        return IK_ERR_SHAPE;
    return launch_rows((const ik_bf16*)x->data, (const ik_bf16*)res->data, (const ik_bf16*)w->data,
                       (ik_bf16*)out->data, (ik_bf16*)sum_out->data, R, H, eps, w_offset, IK_STREAM(stream));
}

int ik_qk_norm_bf16(const ik_tensor* x, const ik_tensor* w, float eps, float w_offset, ik_tensor* out, void* stream) {
    IK_TRY(ik_check(x, IK_DTYPE_BF16, 3));
    IK_TRY(ik_check(w, IK_DTYPE_BF16, 1));
    IK_TRY(ik_check(out, IK_DTYPE_BF16, 3));
    const int64_t D = x->shape[2];
    if (w->shape[0] != D || out->shape[0] != x->shape[0] || out->shape[1] != x->shape[1] || out->shape[2] != D)
        return IK_ERR_SHAPE;
    return launch_rows((const ik_bf16*)x->data, nullptr, (const ik_bf16*)w->data, (ik_bf16*)out->data, nullptr,
                       x->shape[0] * x->shape[1], D, eps, w_offset, IK_STREAM(stream));
}

}  // extern "C"
