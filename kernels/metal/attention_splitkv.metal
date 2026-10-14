// AttnDecode, GQA-grouped split-KV (kernels/metal, phase B2a). Semantics as attention.metal (cpu oracle: causal, scale,
// device-side position clamped to the cache); this file only changes how the work is spread over the GPU.
//
// attn_decode_bf16_splitkv_g{1,2,4,8} (stage 1) + attn_decode_bf16_combine (stage 2). T == 1, head_dim a power of two in
// [8, 256], q and K / V caches 16-byte aligned (the launcher, crates/intarsia-backend-metal/src/attn.rs, falls back to the
// kernels of attention.metal otherwise).
//
// Stage 1: one threadgroup per (kv head, split). It reads its slice of the kv head's keys / values ONCE and applies them to
// all G = N / Nkv query heads that share the kv head (GP >= G query-head registers per lane, GP in {1, 2, 4, 8}; extra
// heads are zero and never stored), so K / V traffic is a G-th of the one-threadgroup-per-head kernels' and the grid is
// Nkv x nsplit threadgroups instead of N. Layout as attn_decode_bf16_v8: a lane loads one 16-byte chunk (8 bf16) of a K / V
// row, D / 8 lanes cover a row, a simdgroup works on KPS = 256 / D keys at once ("slots"); a slot is one online-softmax
// stream per query head. Scores are kept in the log2 domain (q is pre-multiplied by scale * log2(e)). The slots of a
// simdgroup are merged with simd_shuffle_xor, the simdgroups through threadgroup memory.
//
// Split policy (one function on host and device: the split count depends only on the clamped position, whether it comes
// from OpP or from the device tensor; the host may pick another threadgroup shape, so the two paths agree up to rounding
// order): nsplit = clamp(min(ceil(visible / min_keys), ceil(target_tgs / Nkv)), 1, max_split); the split
// chunk is ceil(visible / nsplit) rounded up to ASK_CHUNK_ALIGN keys, so trailing splits may be empty. With nsplit == 1
// stage 1 writes the bf16 output itself and stage 2 is not needed (the launcher skips it for a host position; for a
// device position it runs and returns immediately). Otherwise stage 1 writes the unnormalised partial accumulator
// (relative to its max) and (max, sum) per (query head, split) into the scratch workspace, and stage 2 combines the
// partials in split order (fixed order: run-to-run identical, no atomics).
//
// Kernel arguments (`OpP`): a = target threadgroups for the policy, b = N, c = Nkv, d = D, e = cap, f = host pos0, g = use
// device position, h = max_split (grid y; also the scratch stride), s3 = skip of out (q is 16-byte aligned, skip 0; the launcher
// falls back otherwise), f0 = scale, pad = min_keys.
// Scratch layout (floats): part[N][max_split][D] then float2 ml[N][max_split].
// stage 1 bind: 0 q, 1 k cache, 2 v cache, 3 out, 4 pos, 5 scratch, 6 OpP.  stage 2 bind: 0 out, 1 pos, 2 scratch, 3 OpP.

#include "common.h"

constant constexpr uint ASK_MAX_SG = 8;        // simdgroups per stage-1 threadgroup, at most
constant constexpr uint ASK_TG_FLOATS = 4096;  // threadgroup accumulator floats: nsg * GP * D must fit (launcher checks)
constant constexpr uint ASK_U = 2;             // units of work (KPS keys) in flight per simdgroup step
constant constexpr uint ASK_CHUNK_ALIGN = 32;
constant constexpr uint ASK_COMB_U = 16;      // splits whose partials the combine loads at once  // split chunks are a multiple of this many keys
constant constexpr float ASK_LOG2E = 1.4426950408889634f;

inline void ask_unpack8(uint4 w, thread float* f) {
    f[0] = as_type<float>(w.x << 16);
    f[1] = as_type<float>(w.x & 0xffff0000u);
    f[2] = as_type<float>(w.y << 16);
    f[3] = as_type<float>(w.y & 0xffff0000u);
    f[4] = as_type<float>(w.z << 16);
    f[5] = as_type<float>(w.z & 0xffff0000u);
    f[6] = as_type<float>(w.w << 16);
    f[7] = as_type<float>(w.w & 0xffff0000u);
}

inline uint ask_nsplit(ulong visible, uint nkv, uint max_split, uint min_keys, uint target) {
    const ulong by_len = (visible + min_keys - 1) / min_keys;
    const ulong by_fill = ((ulong)target + nkv - 1) / nkv;
    return (uint)max(min(min(by_len, by_fill), (ulong)max_split), (ulong)1);
}

inline ulong ask_chunk(ulong visible, uint nsplit) {
    const ulong c = (visible + nsplit - 1) / nsplit;
    return (c + ASK_CHUNK_ALIGN - 1) / ASK_CHUNK_ALIGN * ASK_CHUNK_ALIGN;
}

// One step of the main loop: ASK_U units (KPS keys each) of this simdgroup. FULL: every key of the step exists, so the
// bounds selects of the partial last step drop out (the arithmetic is the same, bit for bit).
template <uint GP, bool FULL>
inline void ask_step(device const uint4* kb, device const uint4* vb, uint start32, uint end32, uint u0, uint nsg, uint kps,
                     uint slot, uint lpk, thread float (&qv)[GP][8], thread float (&acc)[GP][8], thread float (&m)[GP],
                     thread float (&l)[GP]) {
    const uint row4 = lpk;  // uint4 per row
    uint4 kr[ASK_U], vr[ASK_U];
    bool ok[ASK_U];
    for (uint u = 0; u < ASK_U; ++u) {
        const uint j = start32 + (u0 + u * nsg) * kps + slot;
        ok[u] = FULL || j < end32;
        kr[u] = ok[u] ? kb[j * row4] : uint4(0);
        vr[u] = ok[u] ? vb[j * row4] : uint4(0);
    }
    float sc[ASK_U][GP];
    for (uint u = 0; u < ASK_U; ++u) {
        float kf[8];
        ask_unpack8(kr[u], kf);
        for (uint g = 0; g < GP; ++g) {
            float dot = 0.0f;
            for (uint i = 0; i < 8; ++i) dot += qv[g][i] * kf[i];
            sc[u][g] = dot;
        }
    }
    for (uint u = 0; u < ASK_U; ++u)
        for (uint g = 0; g < GP; ++g)
            for (uint o = 1; o < lpk; o <<= 1) sc[u][g] += simd_shuffle_xor(sc[u][g], (ushort)o);
    float pr[ASK_U][GP];
    for (uint g = 0; g < GP; ++g) {
        float m_new = m[g];
        for (uint u = 0; u < ASK_U; ++u) m_new = max(m_new, ok[u] ? sc[u][g] : -INFINITY);
        // a slot whose keys are all past the end keeps m = -inf; its score -inf must not meet m_new = -inf
        const float corr = m_new == -INFINITY ? 1.0f : exp2(m[g] - m_new);
        float psum = 0.0f;
        for (uint u = 0; u < ASK_U; ++u) {
            pr[u][g] = (ok[u] && m_new != -INFINITY) ? exp2(sc[u][g] - m_new) : 0.0f;
            psum += pr[u][g];
        }
        l[g] = l[g] * corr + psum;
        m[g] = m_new;
        for (uint i = 0; i < 8; ++i) acc[g][i] *= corr;
    }
    for (uint u = 0; u < ASK_U; ++u) {
        float vf[8];
        ask_unpack8(vr[u], vf);
        for (uint g = 0; g < GP; ++g)
            for (uint i = 0; i < 8; ++i) acc[g][i] += pr[u][g] * vf[i];
    }
}

template <uint GP>
kernel void attn_decode_splitkv_t(device const ushort* q [[buffer(0)]],
                                  device const ushort* kc [[buffer(1)]],
                                  device const ushort* vc [[buffer(2)]],
                                  device ushort* out [[buffer(3)]],
                                  device const int* posbuf [[buffer(4)]],
                                  device float* part [[buffer(5)]],
                                  constant OpP& p [[buffer(6)]],
                                  uint2 tg [[threadgroup_position_in_grid]],
                                  uint sg [[simdgroup_index_in_threadgroup]],
                                  uint nsg [[simdgroups_per_threadgroup]],
                                  uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float tm[ASK_MAX_SG * GP], tl[ASK_MAX_SG * GP];
    threadgroup float tacc[ASK_TG_FLOATS];

    const uint n = p.b, nkv = p.c, d = p.d;
    const ulong cap = p.e;
    const int pos_raw = p.g != 0 ? posbuf[0] : (int)p.f;
    const ulong visible = min((ulong)max(pos_raw, 0) + 1, cap);  // clamped, as the other decode kernels
    const uint nsplit = ask_nsplit(visible, nkv, p.h, p.pad, p.a);
    const uint kvh = tg.x, sp = tg.y;
    if (sp >= nsplit) return;  // uniform over the threadgroup
    const uint grp = n / nkv;  // query heads per kv head, <= GP
    const ulong chunk = ask_chunk(visible, nsplit);
    const ulong start = (ulong)sp * chunk;
    const ulong end = min(visible, start + chunk);
    device float2* ml = (device float2*)(part + (ulong)n * p.h * d);
    if (start >= end) {  // an empty split (the chunk rounding leaves trailing ones): max = -inf, sum = 0
        if (sg == 0 && lane < grp) ml[(ulong)(kvh * grp + lane) * p.h + sp] = float2(-INFINITY, 0.0f);
        return;
    }

    const uint lpk = d / 8, kps = 32 / lpk;
    const uint slot = lane / lpk, c8 = lane % lpk;
    // 16-byte aligned by the launcher (skips of k / v are 0)
    device const uint4* kb = (device const uint4*)(kc + (ulong)kvh * cap * d) + c8;
    device const uint4* vb = (device const uint4*)(vc + (ulong)kvh * cap * d) + c8;
    float qv[GP][8], acc[GP][8], m[GP], l[GP];
    const float qs = p.f0 * ASK_LOG2E;
    for (uint g = 0; g < GP; ++g) {
        // one 16-byte load of the lane's 8 q values (the launcher guarantees a 16-byte aligned q, skip 0): 8 scalar loads
        // are a chain of load issues on the ramp of every launch; the values and the scaling are unchanged
        float qf[8];
        ask_unpack8(g < grp ? ((device const uint4*)(q + (ulong)(kvh * grp + g) * d + c8 * 8))[0] : uint4(0), qf);
        for (uint i = 0; i < 8; ++i) {
            qv[g][i] = g < grp ? qf[i] * qs : 0.0f;
            acc[g][i] = 0.0f;
        }
        m[g] = -INFINITY;
        l[g] = 0.0f;
    }

    const uint units = (uint)(end - start + kps - 1) / kps;
    const uint start32 = (uint)start, end32 = (uint)end;  // < cap < 2^32 (launcher): 32-bit index math in the loop
    for (uint u0 = sg; u0 < units; u0 += nsg * ASK_U) {
        // the loads of this step, all issued before any is used: they are independent and hide each other's latency
        // (a step is "full" when every slot of every unit has a key; only the last step of a split can be partial)
        const bool full = start32 + (u0 + (ASK_U - 1) * nsg + 1) * kps <= end32;
        if (full) {
            ask_step<GP, true>(kb, vb, start32, end32, u0, nsg, kps, slot, lpk, qv, acc, m, l);
        } else {
            ask_step<GP, false>(kb, vb, start32, end32, u0, nsg, kps, slot, lpk, qv, acc, m, l);
        }
    }

    // merge the KPS slots of this simdgroup (lanes differing in the slot bits) in registers
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
    if (slot == 0) {
        for (uint g = 0; g < GP; ++g) {
            if (c8 == 0) {
                tm[sg * GP + g] = m[g];
                tl[sg * GP + g] = l[g];
            }
            for (uint i = 0; i < 8; ++i) tacc[(sg * GP + g) * d + c8 * 8 + i] = acc[g][i];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // merge the simdgroups; thread -> (query head g, dim di)
    const uint tid = sg * 32 + lane;
    for (uint idx = tid; idx < grp * d; idx += nsg * 32) {
        const uint g = idx / d, di = idx % d;
        float mt = -INFINITY;
        for (uint s = 0; s < nsg; ++s) mt = max(mt, tm[s * GP + g]);
        float lt = 0.0f, num = 0.0f;
        for (uint s = 0; s < nsg; ++s) {
            const float ms = tm[s * GP + g];
            const float w = ms == -INFINITY ? 0.0f : exp2(ms - mt);
            lt += tl[s * GP + g] * w;
            num += tacc[(s * GP + g) * d + di] * w;
        }
        const ulong head = (ulong)kvh * grp + g;
        if (nsplit == 1) {
            out[p.s3 + head * d + di] = f2bf(lt > 0.0f ? num / lt : 0.0f);
        } else {
            part[(head * p.h + sp) * d + di] = num;
            if (di == 0) ml[head * p.h + sp] = float2(mt, lt);
        }
    }
}

#define ASK_INST(GP)                                                                                           \
    template [[host_name("attn_decode_bf16_splitkv_g" #GP)]] kernel void attn_decode_splitkv_t<GP>(            \
        device const ushort*, device const ushort*, device const ushort*, device ushort*, device const int*, \
        device float*, constant OpP&, uint2, uint, uint, uint);
ASK_INST(1)
ASK_INST(2)
ASK_INST(4)
ASK_INST(8)
#undef ASK_INST

// Stage 2: one threadgroup per query head, one thread per output dim (threads == D). Combines the nsplit partials in
// split order; empty splits (max = -inf) contribute nothing.
kernel void attn_decode_bf16_combine(device ushort* out [[buffer(0)]],
                                     device const int* posbuf [[buffer(1)]],
                                     device const float* part [[buffer(2)]],
                                     constant OpP& p [[buffer(3)]],
                                     uint head [[threadgroup_position_in_grid]],
                                     uint di [[thread_position_in_threadgroup]]) {
    const uint n = p.b, nkv = p.c, d = p.d;
    const int pos_raw = p.g != 0 ? posbuf[0] : (int)p.f;
    const ulong visible = min((ulong)max(pos_raw, 0) + 1, (ulong)p.e);
    const uint nsplit = ask_nsplit(visible, nkv, p.h, p.pad, p.a);
    if (nsplit == 1) return;  // stage 1 wrote the output
    device const float2* ml = (device const float2*)(part + (ulong)n * p.h * d) + (ulong)head * p.h;
    device const float* acc = part + (ulong)head * p.h * d + di;
    // Loads come in batches of ASK_COMB_U splits, all issued before any is used: the combine is a chain of dependent
    // round trips otherwise (one per split), which made it the longest part of a short launch. The first batch's
    // accumulators are loaded together with its (max, sum) pairs. Arithmetic and order are those of a plain loop over
    // the splits, so the result is bitwise the same.
    float2 v0[ASK_COMB_U];
    float a0[ASK_COMB_U];
    for (uint u = 0; u < ASK_COMB_U; ++u) {
        const bool ok = u < nsplit;
        v0[u] = ok ? ml[u] : float2(-INFINITY, 0.0f);
        a0[u] = ok ? acc[(ulong)u * d] : 0.0f;
    }
    float mt = -INFINITY;
    for (uint u = 0; u < ASK_COMB_U; ++u) mt = max(mt, v0[u].x);
    for (uint s0 = ASK_COMB_U; s0 < nsplit; s0 += ASK_COMB_U) {
        float2 v[ASK_COMB_U];
        for (uint u = 0; u < ASK_COMB_U; ++u) v[u] = s0 + u < nsplit ? ml[s0 + u] : float2(-INFINITY, 0.0f);
        for (uint u = 0; u < ASK_COMB_U; ++u) mt = max(mt, v[u].x);
    }
    float lt = 0.0f, num = 0.0f;
    for (uint u = 0; u < ASK_COMB_U; ++u) {
        if (v0[u].x == -INFINITY) continue;
        const float w = exp2(v0[u].x - mt);
        lt += v0[u].y * w;
        num += a0[u] * w;
    }
    for (uint s0 = ASK_COMB_U; s0 < nsplit; s0 += ASK_COMB_U) {
        float2 v[ASK_COMB_U];
        float a[ASK_COMB_U];
        for (uint u = 0; u < ASK_COMB_U; ++u) {
            const bool ok = s0 + u < nsplit;
            v[u] = ok ? ml[s0 + u] : float2(-INFINITY, 0.0f);
            a[u] = ok ? acc[(ulong)(s0 + u) * d] : 0.0f;
        }
        for (uint u = 0; u < ASK_COMB_U; ++u) {
            if (v[u].x == -INFINITY) continue;
            const float w = exp2(v[u].x - mt);
            lt += v[u].y * w;
            num += a[u] * w;
        }
    }
    out[p.s3 + (ulong)head * d + di] = f2bf(lt > 0.0f ? num / lt : 0.0f);
}
