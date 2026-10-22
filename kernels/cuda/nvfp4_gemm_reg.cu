// Dense NVFP4 prefill GEMMs with the weights dequantised in registers (lane S1, docs/results/2026-10-22-s1-spark-prefill.md):
//
//   matmul_nvfp4_gemm_reg  out[M,N]  = gs * x[M,K] @ dequant(W)[N,K]^T                 (Op::MatmulNvfp4)
//   matmul_nvfp4_glu_gemm  out[M,N]  = silu(gate(x)) * up(x), gate/up as the line above (Op::MatmulNvfp4Glu)
//
// BITWISE the results of `matmul_nvfp4_gemm` (wq_gemm.cuh FMT_NVFP4) and, for the GLU, of two `matmul_nvfp4_gemm` calls
// followed by `silu_glu_bf16`:
//   * every output element's accumulator goes through the same chain of m16n8k16 mma, C = the accumulator, k16 steps in
//     k order, from a zero accumulator; its position inside the m16n8 fragment is the same (row % 16 and column % 8: tile
//     origins are multiples of 16 rows and of 8 columns);
//   * the B fragments hold the same bf16 numbers: e2m1 * e4m3 block scale is exact in bf16 (at most 6 significant bits),
//     whether formed in fp32 and packed (wq_gemm's shared-memory tile) or as a bf16 multiply (here, moe_w4a16.cu's
//     register decode);
//   * the A fragments are the same ldmatrix loads of the same bf16 activations;
//   * epilogue: bf16(acc * global_scale) as wq_gemm's store; the GLU keeps those two roundings and then computes
//     silu_glu_kernel's bf16(ik_silu(g) * u).
// The schedule is moe_w4a16.cu's (the grouped experts kernel, MODE 0 / 1) on dense rows: 4 warps, BM = 64 rows x R = 128
// weight rows per block (GLU: 64 gate + the matching 64 up rows), packed weights and their scales staged through a 2-unit
// cp.async ring (64 payload bytes + 8 scale bytes per row and 128-k unit), the A tile through a 3-slot ring of 64-k steps,
// each warp decoding the B fragments of its own 32 weight rows in registers. Two blocks per SM (48 KB of shared memory).
//
// Shapes served (registry): M >= 256, K % 128 == 0, N % 128 == 0 (GEMM) or N % 64 == 0 (GLU), bf16 x; the GEMM also
// writes f32. The 35B shared expert: gate / up 512 x 2048, down 2048 x 512.
#include "ik_launch.h"
#include "ik_quant.cuh"
#include <cuda_bf16.h>

namespace {

constexpr int BK = 64;          // k per pipeline step
constexpr int LDS = BK + 8;     // padded smem row stride of the A tile (bf16 elements): conflict-free ldmatrix
constexpr int kRawStride = 80;  // bytes per staged weight row: 64 payload + 8 scales + 8 pad
constexpr int R = 128;          // weight rows per block
constexpr int MT = 4;           // m16 tiles per warp: BM = 64
constexpr int JW = 4;           // n8 tiles per warp (32 weight rows)
constexpr int BM = 16 * MT;
constexpr int NT = R * 4 / JW;  // 128 threads, 4 warps
constexpr int JH = JW / 2;
constexpr int LA = 2, LR = 2;   // lookahead: A(s + LA) and the packed unit of step s + LR are requested at step s
constexpr int NA = LA + 1, NR = LR / 2 + 1;
constexpr int XS = BM * LDS;
constexpr size_t kSmem = (size_t)NA * XS * 2 + (size_t)NR * R * kRawStride;

// the 8 scale bytes of a row and unit as two 4-byte copies (block-scale tensors are only required to be 4-byte aligned)
IK_DEVI void cp_async8(void* smem_dst, const void* gmem_src) {
    const unsigned d = static_cast<unsigned>(__cvta_generic_to_shared(smem_dst));
    const char* src = static_cast<const char*>(gmem_src);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" ::"r"(d), "l"(src) : "memory");
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" ::"r"(d + 4u), "l"(src + 4) : "memory");
}

// as moe_w4a16.cu: 4 fp4 values (low 16 bits of q, nibble j = element j) -> bf16x2 {e0, e1}, {e2, e3}, unscaled
IK_DEVI void fp4x4_to_bf16x2x2(uint32_t q, uint32_t& r0, uint32_t& r1) {
    const uint32_t sel = q & 0x7777u;
    const uint32_t hi = __byte_perm(0x3F3F3F00u, 0x40404040u, sel);
    const uint32_t lo = __byte_perm(0xC0800000u, 0xC0804000u, sel);
    r0 = __byte_perm(lo, hi, 0x5140u) | (((q & 0x88u) * 0x1001000u) & 0x80008000u);
    r1 = __byte_perm(lo, hi, 0x7362u) | (((q & 0x8800u) * 0x10010u) & 0x80008000u);
}

IK_DEVI uint32_t mul_bf16x2(uint32_t a, uint32_t b) {
    __nv_bfloat162 x = *reinterpret_cast<__nv_bfloat162*>(&a), y = *reinterpret_cast<__nv_bfloat162*>(&b);
    __nv_bfloat162 r = __hmul2(x, y);
    return *reinterpret_cast<uint32_t*>(&r);
}

IK_DEVI uint32_t scale_pair(float f) {
    __nv_bfloat162 r = __float2bfloat162_rn(f);
    return *reinterpret_cast<uint32_t*>(&r);
}

IK_DEVI float round_bf16(float v) { return ik_b2f(ik_f2b(v)); }

IK_DEVI void store2(float* p, float a, float b) {
    p[0] = a;
    p[1] = b;
}
IK_DEVI void store2(ik_bf16* p, float a, float b) {
    p[0] = ik_f2b(a);
    p[1] = ik_f2b(b);
}

// GLU = 1: weight rows [0, R/2) of the block are gate rows n0.., [R/2, R) the up rows n0.. (wg / wu), output columns
// n0 .. n0 + R/2 of silu(g) * u. GLU = 0: R rows n0.. of wg, output columns n0 .. n0 + R.
template <int GLU, typename OutT>
__global__ void __launch_bounds__(NT, 2)
nvfp4_reg_kernel(const ik_bf16* __restrict__ x, const uint8_t* __restrict__ wg, const uint8_t* __restrict__ bsg,
                 const float* __restrict__ gsg, const uint8_t* __restrict__ wu, const uint8_t* __restrict__ bsu,
                 const float* __restrict__ gsu, OutT* __restrict__ out, int M, int N, int K, int vec) {
    extern __shared__ __align__(16) unsigned char smem_raw[];
    ik_bf16* const xs0 = reinterpret_cast<ik_bf16*>(smem_raw);
    auto xs = [&](int slot) { return xs0 + slot * XS; };
    unsigned char* const raw_base = smem_raw + (size_t)(NA * XS) * 2;
    auto rawu = [&](int slot) { return raw_base + slot * (R * kRawStride); };

    const int m0 = blockIdx.y * BM;
    const int rows = min(BM, M - m0);  // 1..BM
    const int n0 = GLU ? blockIdx.x * (R / 2) : blockIdx.x * R;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int wn = warp;
    const int g = lane >> 2, t4 = lane & 3;
    const int S = K / BK;   // 64-wide steps
    const int U = K / 128;  // 128-wide packed units
    const int64_t row_bytes = K / 2, srow_bytes = K / 16;
    auto wrow = [&](int r) { return GLU ? (r < R / 2 ? wg : wu) + (int64_t)(n0 + (r % (R / 2))) * row_bytes : wg + (int64_t)(n0 + r) * row_bytes; };
    auto srow = [&](int r) { return GLU ? (r < R / 2 ? bsg : bsu) + (int64_t)(n0 + (r % (R / 2))) * srow_bytes : bsg + (int64_t)(n0 + r) * srow_bytes; };

    // packed-weight copy: chunk (tid + NT q) = row r, 16-byte part of the unit's 64 bytes; threads < R also copy the 8 scale
    // bytes of row `tid`
    constexpr int RQ = R * 4 / NT;
    const int rpart = tid & 3;
    const uint8_t* rsrc[RQ];
#pragma unroll
    for (int q = 0; q < RQ; ++q) {
        const int r = (tid >> 2) + (NT / 4) * q;
        rsrc[q] = wrow(r) + rpart * 16;
    }
    const uint8_t* ssrc = srow(tid < R ? tid : 0);
    auto issue_raw = [&](int u, int slot) {
#pragma unroll
        for (int q = 0; q < RQ; ++q) {
            const int r = (tid >> 2) + (NT / 4) * q;
            ik_cp_async16(rawu(slot) + r * kRawStride + rpart * 16, rsrc[q] + u * 64, 16);
        }
        if (tid < R) cp_async8(rawu(slot) + tid * kRawStride + 64, ssrc + u * 8);
    };

    // this thread's A rows: chunk c8 (16 B = 8 bf16) of rows r = (tid >> 3) + (NT / 8) q
    constexpr int AQ = BM / (NT / 8);
    const int c8 = tid & 7;
    const ik_bf16* arow[AQ];
    bool avalid[AQ];
#pragma unroll
    for (int q = 0; q < AQ; ++q) {
        const int r = (tid >> 3) + (NT / 8) * q;
        avalid[q] = r < rows;
        arow[q] = avalid[q] ? x + (int64_t)(m0 + r) * K : x;
    }
    auto issue_a = [&](int st, int slot) {
#pragma unroll
        for (int q = 0; q < AQ; ++q) {
            const int r = (tid >> 3) + (NT / 8) * q;
            ik_cp_async16(xs(slot) + r * LDS + c8 * 8, arow[q] + st * BK + c8 * 8, avalid[q] ? 16 : 0);
        }
    };
    auto issue_group = [&](int v) {
        if (v + LA >= 0 && v + LA < S) issue_a(v + LA, (v + LA) % NA);
        if (!((v + LR) & 1) && (v + LR) / 2 < U) issue_raw((v + LR) / 2, ((v + LR) / 2) % NR);
        ik_cp_async_commit();
    };

    float acc[MT][JW][4];
#pragma unroll
    for (int i = 0; i < MT; ++i)
#pragma unroll
        for (int j = 0; j < JW; ++j)
#pragma unroll
            for (int v = 0; v < 4; ++v) acc[i][j][v] = 0.f;

    // smem row (within the R staged rows) of the j-th n8 tile of this warp; GLU: tiles j < JH gate, j >= JH the matching up
    auto nrow = [&](int j) {
        return GLU ? (j < JH ? wn * 8 * JH + j * 8 : R / 2 + wn * 8 * JH + (j - JH) * 8) : wn * 8 * JW + j * 8;
    };
    const uint32_t selt = (uint32_t)t4 | ((uint32_t)(4 + t4) << 4);

    for (int v = -LR; v < 0; ++v) issue_group(v);
    for (int s = 0; s < S; ++s) {
        ik_cp_async_wait_group<LA - 1>();
        __syncthreads();
        issue_group(s);
        const ik_bf16* xt = xs(s % NA);
        const unsigned char* rb = rawu((s >> 1) % NR) + (s & 1) * 32;
        uint32_t wd[JW][8];
        uint32_t scs[JW];
#pragma unroll
        for (int j = 0; j < JW; ++j) {
            const unsigned char* rp = rb + (nrow(j) + g) * kRawStride;
            const uint4 q0 = *reinterpret_cast<const uint4*>(rp);
            const uint4 q1 = *reinterpret_cast<const uint4*>(rp + 16);
            wd[j][0] = q0.x; wd[j][1] = q0.y; wd[j][2] = q0.z; wd[j][3] = q0.w;
            wd[j][4] = q1.x; wd[j][5] = q1.y; wd[j][6] = q1.z; wd[j][7] = q1.w;
        }
#pragma unroll
        for (int j = 0; j < JW; ++j) {
            const unsigned char* rsp = rawu((s >> 1) % NR) + (nrow(j) + g) * kRawStride + 64 + (s & 1) * 4;
            scs[j] = *reinterpret_cast<const uint32_t*>(rsp);
        }
#pragma unroll
        for (int kk = 0; kk < BK / 16; ++kk) {
            uint32_t b[JW][2];
#pragma unroll
            for (int j = 0; j < JW; ++j) {
                const uint32_t q = __byte_perm(wd[j][2 * kk], wd[j][2 * kk + 1], selt);
                uint32_t r0, r1;
                fp4x4_to_bf16x2x2(q, r0, r1);
                const uint32_t sp = scale_pair(ik_e4m3_to_f((uint8_t)((scs[j] >> (8 * kk)) & 0xFFu)));
                b[j][0] = mul_bf16x2(r0, sp);
                b[j][1] = mul_bf16x2(r1, sp);
            }
#pragma unroll
            for (int i = 0; i < MT; ++i) {
                if (i * 16 >= rows) continue;  // block-uniform: past the last row
                uint32_t a0, a1, a2, a3;
                const int r = i * 16 + (lane & 7) + ((lane >> 3) & 1) * 8;
                ik_ldmatrix_x4(a0, a1, a2, a3, xt + r * LDS + kk * 16 + (lane >> 4) * 8);
#pragma unroll
                for (int j = 0; j < JW; ++j) ik_mma_bf16_16816(acc[i][j], a0, a1, a2, a3, b[j][0], b[j][1]);
            }
        }
    }
    ik_cp_async_wait_all();

    // ---- epilogue ----
    // value of output column pair (i, jo, hh) of this thread: GLU: the two GEMMs' bf16 stores, then silu_glu_kernel's
    // arithmetic; else wq_gemm's bf16(acc * gs) (or the f32 product)
    constexpr int NCOL = GLU ? R / 2 : R;  // output columns of the block
    constexpr int JO = GLU ? JH : JW;      // output n8 tiles per warp
    const float sg = gsg[0], su = GLU ? gsu[0] : 0.f;
    // the two values of output column pair (i, jo, hh) of this thread
#define IK_NVFP4_VALUES(v0, v1, i, jo, hh)                                       \
    float v0, v1;                                                                \
    if constexpr (GLU) {                                                         \
        const float g0 = round_bf16(acc[i][jo][2 * (hh)] * sg);                  \
        const float g1 = round_bf16(acc[i][jo][2 * (hh) + 1] * sg);              \
        const float u0 = round_bf16(acc[i][(jo) + JH][2 * (hh)] * su);           \
        const float u1 = round_bf16(acc[i][(jo) + JH][2 * (hh) + 1] * su);       \
        v0 = ik_silu(g0) * u0;                                                   \
        v1 = ik_silu(g1) * u1;                                                   \
    } else {                                                                     \
        v0 = acc[i][jo][2 * (hh)] * sg;                                          \
        v1 = acc[i][jo][2 * (hh) + 1] * sg;                                      \
    }
    if (sizeof(OutT) == 2 && vec) {
        // stage the bf16 tile in shared memory, write full 16-byte row chunks
        constexpr int OS = NCOL + 8;
        __syncthreads();
        ik_bf16* const ob = xs0;  // [BM][OS], aliases the A ring
#pragma unroll
        for (int i = 0; i < MT; ++i)
#pragma unroll
            for (int jo = 0; jo < JO; ++jo)
#pragma unroll
                for (int hh = 0; hh < 2; ++hh) {
                    const int rl = i * 16 + g + hh * 8;
                    const int cl = wn * 8 * JO + jo * 8 + t4 * 2;
                    IK_NVFP4_VALUES(v0, v1, i, jo, hh)
                    *reinterpret_cast<ik_bf162*>(ob + rl * OS + cl) = __floats2bfloat162_rn(v0, v1);
                }
        __syncthreads();
        constexpr int CPR = NCOL / 8;
        for (int c = tid; c < BM * CPR; c += NT) {
            const int rl = c / CPR, part = c % CPR;
            if (rl >= rows) continue;
            *reinterpret_cast<uint4*>(reinterpret_cast<ik_bf16*>(out) + (int64_t)(m0 + rl) * N + n0 + part * 8) =
                *reinterpret_cast<const uint4*>(ob + rl * OS + part * 8);
        }
    } else {
#pragma unroll
        for (int i = 0; i < MT; ++i)
#pragma unroll
            for (int hh = 0; hh < 2; ++hh) {
                const int rl = i * 16 + g + hh * 8;
                if (rl >= rows) continue;
#pragma unroll
                for (int jo = 0; jo < JO; ++jo) {
                    const int col = n0 + wn * 8 * JO + jo * 8 + t4 * 2;
                    OutT* const p = out + (int64_t)(m0 + rl) * N + col;
                    IK_NVFP4_VALUES(v0, v1, i, jo, hh)
                    if (sizeof(OutT) == 2) {
                        ik_store(p, v0);
                        ik_store(p + 1, v1);
                    } else {
                        store2(p, v0, v1);
                    }
                }
            }
    }
#undef IK_NVFP4_VALUES
}

template <int GLU, typename OutT>
void launch(const ik_bf16* x, const uint8_t* wg, const uint8_t* bsg, const float* gsg, const uint8_t* wu,
            const uint8_t* bsu, const float* gsu, OutT* out, int M, int N, int K, cudaStream_t s) {
    static const bool attr_set = [] {
        cudaFuncSetAttribute(reinterpret_cast<const void*>(nvfp4_reg_kernel<GLU, OutT>),
                             cudaFuncAttributeMaxDynamicSharedMemorySize, (int)kSmem);
        cudaFuncSetAttribute(reinterpret_cast<const void*>(nvfp4_reg_kernel<GLU, OutT>),
                             cudaFuncAttributePreferredSharedMemoryCarveout, 100);
        return true;
    }();
    (void)attr_set;
    const unsigned nx = GLU ? (unsigned)(N / (R / 2)) : (unsigned)(N / R);
    const int vec = (sizeof(OutT) == 2 && N % 8 == 0 && ik_aligned16(out)) ? 1 : 0;  // else the scalar epilogue
    nvfp4_reg_kernel<GLU, OutT><<<dim3(nx, (unsigned)((M + BM - 1) / BM)), NT, kSmem, s>>>(x, wg, bsg, gsg, wu, bsu, gsu, out,
                                                                                         M, N, K, vec);
}

int check_w(const ik_tensor* w, const ik_tensor* bs, const ik_tensor* gs, int64_t N, int64_t K) {
    IK_TRY(ik_check_a(w, IK_DTYPE_U8, 2, 16));
    IK_TRY(ik_check_a(bs, IK_DTYPE_FP8_E4M3, 2, 4));
    IK_TRY(ik_check_a(gs, IK_DTYPE_F32, 1, 4));
    if (w->shape[0] != N || w->shape[1] != K / 2 || bs->shape[0] != N || bs->shape[1] != K / 16 || gs->shape[0] != 1)
        return IK_ERR_SHAPE;
    return IK_OK;
}

}  // namespace

extern "C" {

int ik_matmul_nvfp4_gemm_reg(const ik_tensor* x, const ik_tensor* w, const ik_tensor* block_scales,
                             const ik_tensor* global_scale, ik_tensor* out, void* stream) {
    IK_TRY(ik_check_a(x, IK_DTYPE_BF16, 2, 16));
    if (!out || (out->dtype != IK_DTYPE_BF16 && out->dtype != IK_DTYPE_F32)) return IK_ERR_DTYPE;
    IK_TRY(ik_check(out, out->dtype, 2));
    const int64_t M = x->shape[0], K = x->shape[1], N = w ? w->shape[0] : 0;
    if (K % 128 != 0 || N % R != 0) return IK_ERR_SHAPE;
    IK_TRY(check_w(w, block_scales, global_scale, N, K));
    if (out->shape[0] != M || out->shape[1] != N) return IK_ERR_SHAPE;
    if (M == 0 || N == 0) return IK_OK;
    if (M > 0x7fffffffLL || N > 0x7fffffffLL || K > 0x7fffffffLL) return IK_ERR_SHAPE;
    cudaStream_t s = IK_STREAM(stream);
    const ik_bf16* xp = (const ik_bf16*)x->data;
    const uint8_t* wp = (const uint8_t*)w->data;
    const uint8_t* bp = (const uint8_t*)block_scales->data;
    const float* gp = (const float*)global_scale->data;
    if (out->dtype == IK_DTYPE_F32)
        launch<0, float>(xp, wp, bp, gp, wp, bp, gp, (float*)out->data, (int)M, (int)N, (int)K, s);
    else
        launch<0, ik_bf16>(xp, wp, bp, gp, wp, bp, gp, (ik_bf16*)out->data, (int)M, (int)N, (int)K, s);
    return ik_launch_status();
}

int ik_matmul_nvfp4_glu_gemm(const ik_tensor* x, const ik_tensor* w_gate, const ik_tensor* block_scales_gate,
                             const ik_tensor* global_scale_gate, const ik_tensor* w_up, const ik_tensor* block_scales_up,
                             const ik_tensor* global_scale_up, ik_tensor* out, void* stream) {
    IK_TRY(ik_check_a(x, IK_DTYPE_BF16, 2, 16));
    IK_TRY(ik_check(out, IK_DTYPE_BF16, 2));
    const int64_t M = x->shape[0], K = x->shape[1], N = w_gate ? w_gate->shape[0] : 0;
    if (K % 128 != 0 || N % (R / 2) != 0) return IK_ERR_SHAPE;
    IK_TRY(check_w(w_gate, block_scales_gate, global_scale_gate, N, K));
    IK_TRY(check_w(w_up, block_scales_up, global_scale_up, N, K));
    if (out->shape[0] != M || out->shape[1] != N) return IK_ERR_SHAPE;
    if (M == 0 || N == 0) return IK_OK;
    if (M > 0x7fffffffLL || N > 0x7fffffffLL || K > 0x7fffffffLL) return IK_ERR_SHAPE;
    launch<1, ik_bf16>((const ik_bf16*)x->data, (const uint8_t*)w_gate->data, (const uint8_t*)block_scales_gate->data,
                       (const float*)global_scale_gate->data, (const uint8_t*)w_up->data,
                       (const uint8_t*)block_scales_up->data, (const float*)global_scale_up->data, (ik_bf16*)out->data,
                       (int)M, (int)N, (int)K, IK_STREAM(stream));
    return ik_launch_status();
}

}  // extern "C"
