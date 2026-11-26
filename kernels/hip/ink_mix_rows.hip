// ConvMix row-table form (batched decode, `OpCall::rows`, lane ink-hip-batch; results docs/results/2026-11-26-inkling2-hip-batch.md).
// Its own translation unit: ink_mix.hip's kernels and helpers with IK_INK_MIX_ROWS_TU defined (no solo launcher here), and the RT
// instantiations of ink_row_kernel, which the solo module then never holds: ink_mix.hip's code object is main's, kernel for kernel.
#define IK_INK_MIX_ROWS_TU 1
#include "ink_mix.hip"

extern "C" {

/* tests (lane ink-hip-batch): the switches as this TU's row-table launches read them: the mask of the fast forms (mix_mask) | 256 when the
 * gate kernel's fast norm tail is on (mix_dec_tail). tests/ink_rows.rs asserts they follow ik_ink_mix_mode / ik_ink_mix_dec. */
int ik_ink_mix_rows_switches(void) { return mix_mask() | (mix_dec_tail() ? 256 : 0); }

/* ConvMix row-table form (batched decode, lane ink-hip-batch): row t of the call is the T = 1 ConvMix of sequence t at position
 * pos[t] (i32 [T]) on its own ring, the buffer at entry tindex of row t of `tab` (u8 [T, tstride, 8]) plus `roff` (the ring carrier's
 * byte offset); `ring` is row 0's carrier (shape only). Bitwise the solo call per row (tests/ink_rows.rs). */
int ik_conv_mix_bf16_rows(const ik_tensor* x, const ik_tensor* taps, const ik_tensor* res, const ik_tensor* addend,
                          const ik_tensor* src, const ik_tensor* queries, const ik_tensor* norm, const ik_tensor* msg,
                          const ik_tensor* down, const ik_tensor* gain, const ik_tensor* forget, const ik_tensor* pos, float eps,
                          ik_tensor* y, ik_tensor* ring, ik_tensor* sum, ik_tensor* mix, const ik_scratch* scratch,
                          const ik_tensor* tab, int64_t tstride, int64_t tindex, int64_t roff, void* stream) {
    if (!tab) return IK_ERR_INVALID_ARG;
    return conv_mix_run(x, taps, res, addend, src, queries, norm, msg, down, gain, forget, 0, pos, eps, y, ring, sum, mix, scratch,
                        tab, tstride, tindex, roff, stream);
}

}  // extern "C"
