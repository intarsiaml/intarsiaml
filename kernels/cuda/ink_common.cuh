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

IK_DEVI float ink_rbf(float v) { return ik_b2f(ik_f2b(v)); }

/* an absent optional operand: a null tensor or a zero-element one */
static inline bool ink_absent(const ik_tensor* t) { return !t || ik_numel(t) == 0; }

#endif
