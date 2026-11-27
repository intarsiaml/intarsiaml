// InkAttnDecode row-table form (batched decode, `OpCall::rows`, lane ink-hip-batch; results docs/results/2026-11-26-inkling2-hip-batch.md).
// Its own translation unit: ink_attn.hip's kernels and helpers with IK_INK_ATTN_ROWS_TU defined (no solo launcher here), and the RT
// instantiations of ink_attn_dec_k, which the solo module then never holds: ink_attn.hip's code object is main's, kernel for kernel.
#define IK_INK_ATTN_ROWS_TU 1
#include "ink_attn.hip"

/* InkAttnDecode row-table form (batched decode, lane ink-hip-batch): row r of the call (r < T <= 16) is the T = 1 call of
 * sequence r at device position pos[r] (i32 [T]); its K / V caches and K / V conv rings are the buffers at entries index ..
 * index + 3 of row r of `tab` (u8 [T, stride, 8]) plus roff[0..4] (the carriers' byte offsets: k cache, v cache, k ring, v ring);
 * kc / vc / kring / vring are row 0's carriers (shapes only). Always the fused kernel (its RT instantiation); bitwise the solo
 * call per row, fused or split (tests/ink_rows.rs). attrs->pos0 is not read. */
extern "C" int ik_ink_attn_decode_bf16_rows(const ik_tensor* pq, const ik_tensor* pk, const ik_tensor* pv, const ik_tensor* kt,
                                            const ik_tensor* vt, const ik_tensor* qs, const ik_tensor* rs, const ik_tensor* rp,
                                            const ik_tensor* kc_in, const ik_tensor* vc_in, const ik_ink_attn_attrs* a, ik_tensor* o,
                                            ik_tensor* kc, ik_tensor* vc, ik_tensor* kring, ik_tensor* vring, const ik_scratch* scratch,
                                            const ik_tensor* pos, const ik_tensor* tab, int64_t stride, int64_t index,
                                            const int64_t* roff, void* stream) {
    (void)kc_in;
    (void)vc_in;
    InkP p = {};
    IK_TRY(prologue_params(pq, pk, pv, kt, vt, qs, rs, rp, a, kring, vring, nullptr, p));
    IK_TRY(chk(o, 2));
    IK_TRY(chk(kc, 3));
    IK_TRY(chk(vc, 3));
    if (p.t == 0) return IK_OK;
    if (!pos || !pos->data || pos->dtype != IK_DTYPE_I32 || ik_numel(pos) != p.t || !roff) return IK_ERR_INVALID_ARG;
    IK_TRY(ik_check_tab(tab, p.t, stride, index));
    if (index + 3 >= stride) return IK_ERR_INVALID_ARG;
    const int G = p.h / p.hk;
    if (p.t > MAXROWS || G < 1 || G > 3 || p.d > 128 || !(p.d == 8 || p.d == 16 || p.d == 32 || p.d == 64 || p.d == 128) || p.hk > MAXHK_TICK)
        return IK_ERR_SHAPE;
    if (o->shape[0] != p.t || o->shape[1] != (int64_t)p.h * p.d || kc->shape[0] != p.hk || kc->shape[2] != p.d || vc->shape[1] != kc->shape[1])
        return IK_ERR_SHAPE;
    p.cr = (int)kc->shape[1];
    p.window = (int)a->window;
    p.scale = a->scale;
    p.sk = split_keys(a);
    const bool ring = a->kv_ring != 0;
    if (ring) p.flags |= F_RING;
    if (ring && (p.window < 1 || p.window > p.cr)) return IK_ERR_SHAPE;
    // every row is a T = 1 call: its cache row and conv-ring row are written in the kernel (the solo launcher's rules at T = 1)
    if (p.k > p.rr) return IK_ERR_SHAPE;
    p.flags |= F_RING_WRITE | F_KV_WRITE;
    // the rows' buffers are device allocations (aligned); the cache planes must stay 16-byte aligned at their offsets
    if (!ik_aligned16(kc->data) || !ik_aligned16(vc->data) || (roff[0] & 15) || (roff[1] & 15) || (roff[2] & 1) || (roff[3] & 1) ||
        roff[0] < 0 || roff[1] < 0 || roff[2] < 0 || roff[3] < 0 || !fits_u32((int64_t)p.hk * p.cr * p.d))
        return IK_ERR_INVALID_ARG;
    p.out = mb(o);
    p.kc = mb(kc);
    p.vc = mb(vc);
    p.pos = (const int*)pos->data;
    p.pos0 = 0;
    if (a->diag & 1) p.flags |= F_DIAG_NOPRO;
    if (a->diag & 2) p.flags |= F_DIAG_NOMERGE;
    if (ik_hip_dec3_mask() & 1) p.flags |= F_DPP;
    unsigned gy, nb;
    attn_geometry(p, true, &gy, &nb);  // the device-position grid (the capacity bound), as the solo captured step
    const int64_t part = (int64_t)p.t * p.h * MAXS * (p.d + 2) * 4;
    if (!scratch || !scratch->data || scratch->bytes < part) return IK_ERR_INVALID_ARG;
    p.scr = (float*)scratch->data;
    InkPR pr;
    static_cast<InkP&>(pr) = p;
    pr.rt.rtab = (const unsigned long long*)tab->data;
    pr.rt.rts = (int)stride;
    pr.rt.rti = (int)index;
    for (int j = 0; j < 4; ++j) pr.rt.roff[j] = roff[j];
    hipStream_t st = IK_STREAM(stream);
    pr.flags |= ink_tick_slot(st) << F_SLOT_SHIFT;  // the ticket slot of this stream (lane ink-hip-overlap; 0 = main's bytes)
    const size_t tbb = (size_t)G * nb * 4;
    const dim3 grid(p.hk, gy, p.t);
    const bool dpp = (p.flags & F_DPP) != 0;
    switch (G) {
        case 1: if (dpp) ink_attn_dec_k<1, true, true><<<grid, DEC_WAVES * 32, tbb, st>>>(pr); else ink_attn_dec_k<1, false, true><<<grid, DEC_WAVES * 32, tbb, st>>>(pr); break;
        case 2: if (dpp) ink_attn_dec_k<2, true, true><<<grid, DEC_WAVES * 32, tbb, st>>>(pr); else ink_attn_dec_k<2, false, true><<<grid, DEC_WAVES * 32, tbb, st>>>(pr); break;
        default: if (dpp) ink_attn_dec_k<3, true, true><<<grid, DEC_WAVES * 32, tbb, st>>>(pr); else ink_attn_dec_k<3, false, true><<<grid, DEC_WAVES * 32, tbb, st>>>(pr); break;
    }
    return ik_launch_status();
}
