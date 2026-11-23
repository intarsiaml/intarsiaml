// Shared by the Inkling-2-Mini kernels (ink_proj.cu, ink_mix.cu, ink_moe.cu; lane ink-cuda-mixmoe).
#ifndef INK_COMMON_CUH
#define INK_COMMON_CUH

#include "ik_launch.h"
#include <cuda_runtime.h>

/* one segment of the tensor-core GEMV: out [T, n] = rA(x [T, D] . w [n, D]^T) */
struct InkSeg {
    const ik_bf16* x;
    const ik_bf16* w;
    ik_bf16* out;
    int n;
};

/* up to three segments; t0 = the prefix of 16-row tiles (set by ink_mma_gemv) */
struct InkSegs {
    InkSeg s[3];
    int nseg;
    int t0[4];
};

/* ink_proj.cu: one launch of the tensor-core GEMV over the segments (T rows each, D % 16 == 0, 16-byte aligned rows). */
int ink_mma_gemv(const InkSegs& sg, int T, int D, cudaStream_t st);

/* ink_moe_mma.cu: the tensor-core prefill experts (fmt 0 NVFP4 with gs, 1 MXFP8; act 0 silu, 1 gelu, 2 situ). H % 128 == 0,
 * I % 128 == 0; scratch of ink_moe_mma_scratch_bytes. */
struct InkMmaArgs {
    const ik_bf16* x;
    const int* ids;
    const float* wts;
    const uint8_t* guw;
    const uint8_t* gus;
    const uint8_t* dnw;
    const uint8_t* dns;
    const float* gs;
    void* out;
    int T, K, E, H, I;
    int fmt, act;
    float gc, uc;
    /* W4A4 (ink.act_quant = 1): x is already the quantised xq; the h rows are quantised between the GEMMs; NVFP4 row scales
     * xo [T] and ho [P] (sorted-row order is written by ink_moe_mma itself) multiply the f32 sums; null otherwise */
    int aq;
    const float* xo;
    float* ho;
    /* the NVFP4 weights are in the decode repack (ink.moe_layout = 1; ink_moe_mma only, 16-bit activations); 0 otherwise */
    int rp;
};
int64_t ink_moe_mma_scratch_bytes(int64_t P, int64_t E, int64_t H, int64_t I);

/* ink_moe_fp4_sm120a.cu: the W4A4 prefill experts on the FP4 tensor core (NVFP4, situ, aq; x already quantised to the codes
 * xq [T,H/2] / e4m3 scales xs [T,H/16] with a.xo). Available only in an sm_121a / sm_120a build (else IK_ERR_UNSUPPORTED). */
int ink_moe_fp4_available();
int64_t ink_moe_fp4_scratch_bytes(int64_t P, int64_t E, int64_t H, int64_t I);
int ink_moe_fp4(const InkMmaArgs& a, const uint8_t* xq, const uint8_t* xs, void* scratch, int out_f32, cudaStream_t s);
int ink_moe_mma(const InkMmaArgs& a, void* scratch, int out_f32, cudaStream_t s);

IK_DEVI float ink_rbf(float v) { return ik_b2f(ik_f2b(v)); }

/* an absent optional operand: a null tensor or a zero-element one */
static inline bool ink_absent(const ik_tensor* t) { return !t || ik_numel(t) == 0; }

#endif
