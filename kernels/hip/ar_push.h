/* Lane ARX: the all-reduce partial pushed by the GEMM that writes it (matmul_mxfp4_a8_mma's direct kernel, mxfp4_a8_gemm.hip) and the
 * short reduce launch after it (ik_ar2_pushed, allreduce.hip).
 *
 * The GEMM's epilogue stores every 16-byte C fragment word twice: into its own output (row-major, as always) and into the PEER's
 * scratch slot of the next wide all-reduce, in fragment order (each wave's 32 lanes write 512 contiguous bytes, so the posted writes
 * go out as full PCIe payloads instead of the 32-byte pieces of a row-major C fragment). The push then runs while the GEMM's later
 * tiles compute, on the one hardware queue, instead of after the GEMM in its own kernel. ik_ar2_pushed only raises the flags, waits
 * for the peer's and adds the peer's words (mapped back through the layout below) to its own partial: bitwise ar2_kernel<0>'s
 * own + peer sum.
 *
 * Fragment order: word w = ((((by * gx + bx) * nw + wave) * tn + i) * tm + j) * 32 + lane holds the C fragment of block (bx, by),
 * wave, weight tile i, token tile j and lane: token m = bx * bm + (wave / wn) * tm * 16 + j * 16 + (lane & 15), features
 * n = by * bn + (wave % wn) * tn * 16 + i * 16 + (lane >> 4) * 8 .. + 8, with bm = (nw / wn) * tm * 16 and bn = wn * tn * 16.
 * Words of tokens >= M (the last token block's padding) are never written and never read. */
#pragma once
#include <stdint.h>

typedef struct ik_ar_push {
    int32_t pushed; /* out: 1 = the GEMM pushed (direct kernel, bf16 out, 8-aligned N, the layout fits in `cap` bytes), 0 = it did not */
    int32_t M;
    int64_t N;
    int32_t wn, tn, tm, nw; /* the direct tile: waves along N, weight / token 16-tiles per wave, waves per block */
    int32_t gx;             /* blocks along M */
    int32_t pad;
    int64_t words; /* 16-byte words of the layout (gx * gy blocks * nw * tn * tm * 32) */
    int64_t cap;   /* in: bytes of the peer slot */
} ik_ar_push;
