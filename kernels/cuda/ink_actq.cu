// Inkling-2 W4A4 activation quantisers (lane ink-cuda-mixmoe): the launchers of ink_actq.cuh. NVFP4: one warp per row (the row
// amax by a warp max, then each lane whole blocks of 16); MXFP8: one thread per block of 32. A row's bits depend on that row only.
#include "ink_actq.cuh"

namespace {

__global__ void __launch_bounds__(256) ink_actq_nv_kernel(const ik_bf16* __restrict__ x, int64_t rows, int C, ik_bf16* xq, float* outer,
                                                          uint8_t* codes, uint8_t* scales) {
    const int lane = threadIdx.x & 31;
    const int64_t row = (int64_t)blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    if (row >= rows) return;
    const ik_bf16* xr = x + row * C;
    float m = 0.f;
    for (int c = lane * 8; c < C; c += 256) {
        const uint4 u = *reinterpret_cast<const uint4*>(xr + c);
        const uint32_t w[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            m = fmaxf(m, fabsf(__uint_as_float(w[i] << 16)));
            m = fmaxf(m, fabsf(__uint_as_float(w[i] & 0xFFFF0000u)));
        }
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
    float enc, dec, out;
    aq_nv_row_scales(m, &enc, &dec, &out);
    __syncwarp();  // every lane has read the whole row before any lane writes (in place)
    for (int b = lane; b < C / 16; b += 32) {
        float v[16];
        const uint4 u0 = *reinterpret_cast<const uint4*>(xr + b * 16);
        const uint4 u1 = *reinterpret_cast<const uint4*>(xr + b * 16 + 8);
        const uint32_t w[8] = {u0.x, u0.y, u0.z, u0.w, u1.x, u1.y, u1.z, u1.w};
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            v[2 * i] = __uint_as_float(w[i] << 16);
            v[2 * i + 1] = __uint_as_float(w[i] & 0xFFFF0000u);
        }
        uint32_t code[16];
        const float s = aq_nv_block(v, enc, dec, out, code);
        uint32_t o[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const ik_bf16 lo = ik_f2b(__fmul_rn(aq_e2m1_val(code[2 * i]), s));
            const ik_bf16 hi = ik_f2b(__fmul_rn(aq_e2m1_val(code[2 * i + 1]), s));
            o[i] = (uint32_t)*reinterpret_cast<const unsigned short*>(&lo) | ((uint32_t)*reinterpret_cast<const unsigned short*>(&hi) << 16);
        }
        ik_bf16* dst = xq + row * C + b * 16;
        *reinterpret_cast<uint4*>(dst) = make_uint4(o[0], o[1], o[2], o[3]);
        *reinterpret_cast<uint4*>(dst + 8) = make_uint4(o[4], o[5], o[6], o[7]);
        if (codes) {
            uint32_t p0 = 0, p1 = 0;
#pragma unroll
            for (int j = 0; j < 8; ++j) p0 |= (code[j] & 15u) << (4 * j);
#pragma unroll
            for (int j = 0; j < 8; ++j) p1 |= (code[8 + j] & 15u) << (4 * j);
            *reinterpret_cast<uint2*>(codes + row * (C / 2) + b * 8) = make_uint2(p0, p1);
        }
        if (scales) scales[row * (C / 16) + b] = ik_f_to_e4m3(s);
    }
    if (lane == 0 && outer) outer[row] = out;
}

__global__ void __launch_bounds__(256) ink_actq_mx_kernel(const ik_bf16* __restrict__ x, int64_t nblk, ik_bf16* xq) {
    const int64_t b = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= nblk) return;
    float v[32], q[32];
    const uint4* src = reinterpret_cast<const uint4*>(x + b * 32);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const uint4 u = src[i];
        const uint32_t w[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            v[8 * i + 2 * j] = __uint_as_float(w[j] << 16);
            v[8 * i + 2 * j + 1] = __uint_as_float(w[j] & 0xFFFF0000u);
        }
    }
    aq_mx_block(v, q);
    ik_bf16* dst = xq + b * 32;
#pragma unroll
    for (int j = 0; j < 32; ++j) dst[j] = ik_f2b(q[j]);
}

}  // namespace

int ink_actq_nv(const ik_bf16* x, int64_t rows, int C, ik_bf16* xq, float* outer, uint8_t* codes, uint8_t* scales, cudaStream_t s) {
    if (C % 16 || rows < 0) return IK_ERR_SHAPE;
    if (rows == 0) return IK_OK;
    const unsigned blocks = (unsigned)((rows + 7) / 8);
    ink_actq_nv_kernel<<<blocks, 256, 0, s>>>(x, rows, C, xq, outer, codes, scales);
    return ik_launch_status();
}

int ink_actq_mx(const ik_bf16* x, int64_t rows, int C, ik_bf16* xq, cudaStream_t s) {
    if (C % 32 || rows < 0) return IK_ERR_SHAPE;
    const int64_t nblk = rows * (C / 32);
    if (nblk == 0) return IK_OK;
    ink_actq_mx_kernel<<<(unsigned)((nblk + 255) / 256), 256, 0, s>>>(x, nblk, xq);
    return ik_launch_status();
}
