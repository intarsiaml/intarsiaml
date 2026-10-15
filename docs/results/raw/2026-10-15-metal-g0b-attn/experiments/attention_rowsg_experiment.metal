// G0b experiment (NEGATIVE, not in the build): RTG rows per threadgroup. Appended to kernels/metal/attention_splitkv.metal during the experiment
// (uses ask_step, askr_nsg etc. from there); launcher selected it with INTARSIA_METAL_ROWS_RTG=<n>. Slower than one row per threadgroup at every
// k and context (docs/results/2026-10-15-metal-g0b-attn.md), bitwise equal to the 1-row kernel (tests passed for RTG 2, 3, 4, 8).

// ---------------------------------------------------------------------------------------------------------------------
// G0b variant "rowsg": RTG rows per threadgroup. The rows of a (kv head, split) share a threadgroup, so the K / V slice they all read
// is served to the RTG row groups of simdgroups from the core's cache instead of from the L2 (the arithmetic of every row is
// unchanged: a row group is nsg_r simdgroups running exactly the 1-row code, rows do not interact). Threadgroup = RTG row groups of
// NSGM simdgroups (NSGM = max over rows of nsg_r, the launcher's choice); a row group's simdgroups >= nsg_r idle. The cross-simdgroup
// merge of the row groups runs one row group after the other through one slab of threadgroup memory.
// Grid: x = Nkv * ceil(R / RTG) (x = kvh * ngroups + group), y = max_split. s2 = RTG.
template <uint GP>
kernel void attn_rowsg_splitkv_t(device const ushort* q [[buffer(0)]],
                                 device const ushort* kc [[buffer(1)]],
                                 device const ushort* vc [[buffer(2)]],
                                 device ushort* out [[buffer(3)]],
                                 device const int* posbuf [[buffer(4)]],
                                 device float* part [[buffer(5)]],
                                 constant OpP& p [[buffer(6)]],
                                 uint2 tg [[threadgroup_position_in_grid]],
                                 uint sgi [[simdgroup_index_in_threadgroup]],
                                 uint nsg_tg [[simdgroups_per_threadgroup]],
                                 uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float tm[ASK_MAX_SG * GP], tl[ASK_MAX_SG * GP];
    threadgroup float tacc[ASK_TG_FLOATS];

    const uint n = p.b, nkv = p.c, d = p.d;
    const uint cap = p.e;
    const uint rows = p.s0, rtg = p.s2;
    const uint nsgm = nsg_tg / rtg;
    const uint ngroups = (rows + rtg - 1) / rtg;
    const uint kvh = tg.x / ngroups, gidx = tg.x - kvh * ngroups;
    const uint rl = sgi / nsgm, sg = sgi - rl * nsgm;  // row group of this simdgroup, simdgroup within it
    const uint row = gidx * rtg + rl;
    const bool row_ok = row < rows;
    const int pos_raw = (p.g != 0 ? posbuf[0] : (int)p.f) + (int)row;
    const uint visible = min((uint)max(pos_raw, 0) + 1u, p.e);
    const uint nsplit = ask_nsplit(visible, nkv, p.h, p.pad, p.a);
    const uint sp = tg.y;
    const uint nsg = askr_nsg(p.g == 0, nkv, nsplit, GP, d);
    const uint grp = n / nkv;
    const uint chunk = ask_chunk(visible, nsplit);
    const uint start = sp * chunk;
    const uint end = min(visible, start + chunk);
    device float* part_r = part + row * (n * p.h * (d + 2));
    device float2* ml = (device float2*)(part_r + n * p.h * d);
    const bool live = row_ok && sp < nsplit;
    if (live && start >= end && sg == 0 && lane < grp) ml[(kvh * grp + lane) * p.h + sp] = float2(-INFINITY, 0.0f);
    const bool work = live && start < end && sg < nsg;  // this simdgroup runs the 1-row main loop

    const uint lpk = d / 8, kps = 32 / lpk;
    const uint slot = lane / lpk, c8 = lane % lpk;
    float qv[GP][8], acc[GP][8], m[GP], l[GP];
    if (work) {
        device const uint4* kb = (device const uint4*)(kc + kvh * cap * d) + c8;
        device const uint4* vb = (device const uint4*)(vc + kvh * cap * d) + c8;
        const float qs = p.f0 * ASK_LOG2E;
        const uint qrow = row * n;
        for (uint g = 0; g < GP; ++g) {
            float qf[8];
            ask_unpack8(g < grp ? ((device const uint4*)(q + (qrow + kvh * grp + g) * d + c8 * 8))[0] : uint4(0), qf);
            for (uint i = 0; i < 8; ++i) {
                qv[g][i] = g < grp ? qf[i] * qs : 0.0f;
                acc[g][i] = 0.0f;
            }
            m[g] = -INFINITY;
            l[g] = 0.0f;
        }
        const uint units = (end - start + kps - 1) / kps;
        const uint start32 = start, end32 = end;
        for (uint u0 = sg; u0 < units; u0 += nsg * ASK_U) {
            const bool full = start32 + (u0 + (ASK_U - 1) * nsg + 1) * kps <= end32;
            if (full) {
                ask_step<GP, true>(kb, vb, start32, end32, u0, nsg, kps, slot, lpk, qv, acc, m, l);
            } else {
                ask_step<GP, false>(kb, vb, start32, end32, u0, nsg, kps, slot, lpk, qv, acc, m, l);
            }
        }
        for (uint o = lpk; o < 32; o <<= 1) {
            for (uint g = 0; g < GP; ++g) {
                const float mo = simd_shuffle_xor(m[g], (ushort)o), lo = simd_shuffle_xor(l[g], (ushort)o);
                const float mn = max(m[g], mo);
                const float wa = m[g] == -INFINITY ? 0.0f : exp2(m[g] - mn);
                const float wb = mo == -INFINITY ? 0.0f : exp2(mo - mn);
                for (uint i = 0; i < 8; ++i) acc[g][i] = acc[g][i] * wa + simd_shuffle_xor(acc[g][i], (ushort)o) * wb;
                l[g] = l[g] * wa + lo * wb;
                m[g] = mn;
            }
        }
    }
    const uint lgd = ctz(d);
    for (uint r = 0; r < rtg; ++r) {  // uniform over the threadgroup
        const bool mine = work && rl == r;
        if (mine && slot == 0) {
            for (uint g = 0; g < GP; ++g) {
                if (c8 == 0) {
                    tm[sg * GP + g] = m[g];
                    tl[sg * GP + g] = l[g];
                }
                for (uint i = 0; i < 8; ++i) tacc[(sg * GP + g) * d + c8 * 8 + i] = acc[g][i];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (mine) {
            const uint tid = sg * 32 + lane;
            for (uint idx = tid; idx < grp * d; idx += nsg * 32) {
                const uint g = idx >> lgd, di = idx & (d - 1);
                float mt = -INFINITY;
                for (uint s = 0; s < nsg; ++s) mt = max(mt, tm[s * GP + g]);
                float lt = 0.0f, num = 0.0f;
                for (uint s = 0; s < nsg; ++s) {
                    const float ms = tm[s * GP + g];
                    const float w = ms == -INFINITY ? 0.0f : exp2(ms - mt);
                    lt += tl[s * GP + g] * w;
                    num += tacc[(s * GP + g) * d + di] * w;
                }
                const uint head = kvh * grp + g;
                if (nsplit == 1) {
                    out[p.s3 + (row * n + head) * d + di] = f2bf(lt > 0.0f ? num / lt : 0.0f);
                } else {
                    part_r[(head * p.h + sp) * d + di] = num;
                    if (di == 0) ml[head * p.h + sp] = float2(mt, lt);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

#define ASKG_INST(GP)                                                                                          \
    template [[host_name("attn_rowsg_splitkv_g" #GP)]] kernel void attn_rowsg_splitkv_t<GP>(                   \
        device const ushort*, device const ushort*, device const ushort*, device ushort*, device const int*, \
        device float*, constant OpP&, uint2, uint, uint, uint);
ASKG_INST(1)
ASKG_INST(2)
ASKG_INST(4)
ASKG_INST(8)
#undef ASKG_INST
