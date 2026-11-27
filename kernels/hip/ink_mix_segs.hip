// ConvMix segment-table form (packed prefill, `OpCall::segs`, lane ink-hip-bpf; results docs/results/2026-11-26-inkling2-hip-bpf.md).
// Its own translation unit, as kernels/hip/ink_mix_rows.hip: ink_mix.hip's kernels and helpers with IK_INK_MIX_SEGS_TU defined (no
// solo launcher here), and the SEG instantiations of ink_row_kernel, which the solo and row-table modules then never hold.
#define IK_INK_MIX_SEGS_TU 1
#include "ink_mix.hip"

extern "C" {

/* ConvMix segment-table form (packed prefill, lane ink-hip-bpf): the call's T rows are consecutive segments of several sequences'
 * prefill chunks; row t is at position meta[t] (i32 [4, T]: positions, each row's segment's first and end positions, its segment
 * index) of its segment, whose ring is the buffer at entry tindex of row meta[3T + t] of `tab` (u8 [n, tstride, 8]) plus `roff` (the
 * ring carrier's byte offset); `ring` is the first segment's carrier (shape only). Segment s is bitwise its solo call
 * (crates/intarsia-models/tests/inkling2_packed.rs). */
int ik_conv_mix_bf16_segs(const ik_tensor* x, const ik_tensor* taps, const ik_tensor* res, const ik_tensor* addend,
                          const ik_tensor* src, const ik_tensor* queries, const ik_tensor* norm, const ik_tensor* msg,
                          const ik_tensor* down, const ik_tensor* gain, const ik_tensor* forget, const ik_tensor* meta, float eps,
                          ik_tensor* y, ik_tensor* ring, ik_tensor* sum, ik_tensor* mix, const ik_scratch* scratch,
                          const ik_tensor* tab, int64_t tstride, int64_t tindex, int64_t roff, void* stream) {
    if (!tab || !meta) return IK_ERR_INVALID_ARG;
    return conv_mix_run(x, taps, res, addend, src, queries, norm, msg, down, gain, forget, 0, nullptr, eps, y, ring, sum, mix, scratch,
                        tab, tstride, tindex, roff, stream, meta);
}

}  // extern "C"
