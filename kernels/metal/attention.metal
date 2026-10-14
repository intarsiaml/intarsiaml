// Causal GQA attention over a contiguous bf16 KV cache [Nkv, cap, D] (kernels/metal, phase A). Semantics follow the cpu
// oracle (crates/intarsia-backend-cpu/src/reference.rs): query t of head h sits at absolute position pos0 + t, sees keys
// 0 ..= pos0 + t of kv head h / (N / Nkv), scores are (q . k) * scale, softmax over the visible keys, output is the
// probability-weighted sum of V rounded once to bf16. bf16 in, f32 math (online softmax), bf16 out. Same structure as
// kernels/hip/attention.hip (one online-softmax pass, no split over keys); speed is phase B.
//
// Kernel arguments (`OpP`, common.h): a = T, b = N, c = Nkv, d = D, e = cap, f = host pos0, g = use device position
// (decode only), f0 = scale. s0..s3 = leading elements to skip of q, k cache, v cache, out (binding order).
// D is even and at most ATT_MAX_D. Reductions have a fixed order, so results are run-to-run identical.

#include "common.h"

constant constexpr uint ATT_MAX_D = 256;
constant constexpr uint ATT_MAX_E = ATT_MAX_D / 32;  // output dims held per lane

// ---- AttnDecode: T == 1, one threadgroup per query head ---------------------------------------------------------------
// bind: 0 q [1,N,D], 1 k cache, 2 v cache, 3 out [1,N,D], 4 pos (i32 [1], read when `g` != 0), 5 OpP.
// ATT_DEC_SG simdgroups split the visible keys round-robin; within a simdgroup the 32 lanes own output dims
// d = lane + 32 e, so every K / V row is read coalesced and a score is one simd_sum. Each simdgroup keeps its own
// online-softmax state (m, l, acc[]) and the simdgroups are merged through threadgroup memory at the end.
// Keep in sync with ATTN_DEC_SG in crates/intarsia-backend-metal/src/dense.rs (threadgroup = ATT_DEC_SG * 32 threads).
constant constexpr uint ATT_DEC_SG = 16;
constant constexpr uint ATT_DEC_U = 4;

kernel void attn_decode_bf16(device const ushort* q [[buffer(0)]],
                             device const ushort* kc [[buffer(1)]],
                             device const ushort* vc [[buffer(2)]],
                             device ushort* out [[buffer(3)]],
                             device const int* posbuf [[buffer(4)]],
                             constant OpP& p [[buffer(5)]],
                             uint head [[threadgroup_position_in_grid]],
                             uint sg [[simdgroup_index_in_threadgroup]],
                             uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float ms[ATT_DEC_SG], ls[ATT_DEC_SG];
    threadgroup float accs[ATT_DEC_SG][ATT_MAX_D];

    const uint n = p.b, d = p.d;
    const ulong cap = p.e;
    const int pos_raw = p.g != 0 ? posbuf[0] : (int)p.f;
    // a device-side position past the cache (or below 0) is clamped, not read out of bounds
    const ulong visible = min((ulong)max(pos_raw, 0) + 1, cap);
    const uint kvh = head / (n / p.c);
    device const ushort* kb = kc + p.s1 + (ulong)kvh * cap * d;
    device const ushort* vb = vc + p.s2 + (ulong)kvh * cap * d;

    float qv[ATT_MAX_E], acc[ATT_MAX_E];
    for (uint e = 0; e < ATT_MAX_E; ++e) {
        const uint di = lane + 32 * e;
        qv[e] = di < d ? bf2f(q[p.s0 + (ulong)head * d + di]) : 0.0f;
        acc[e] = 0.0f;
    }
    float m = -INFINITY, l = 0.0f;
    // ATT_DEC_U keys per step: their K / V loads are independent, which hides the memory latency a one-key step is
    // bound by; the online-softmax state is updated once per step (max over the step's scores).
    for (ulong j0 = sg; j0 < visible; j0 += (ulong)ATT_DEC_SG * ATT_DEC_U) {
        float sc[ATT_DEC_U];
        for (uint u = 0; u < ATT_DEC_U; ++u) {
            const ulong j = j0 + (ulong)u * ATT_DEC_SG;
            float dot = 0.0f;
            if (j < visible) {
                device const ushort* kr = kb + j * d;
                for (uint e = 0; e < ATT_MAX_E; ++e) {
                    const uint di = lane + 32 * e;
                    if (di < d) dot += qv[e] * bf2f(kr[di]);
                }
            }
            sc[u] = dot;
        }
        float m_new = m;
        for (uint u = 0; u < ATT_DEC_U; ++u) {
            const ulong j = j0 + (ulong)u * ATT_DEC_SG;
            sc[u] = j < visible ? simd_sum(sc[u]) * p.f0 : -INFINITY;
            m_new = max(m_new, sc[u]);
        }
        const float corr = exp(m - m_new);  // m == -inf on the first step -> 0
        float pr[ATT_DEC_U];
        float psum = 0.0f;
        for (uint u = 0; u < ATT_DEC_U; ++u) {
            pr[u] = exp(sc[u] - m_new);  // a key past the end has score -inf -> 0
            psum += pr[u];
        }
        l = l * corr + psum;
        m = m_new;
        for (uint e = 0; e < ATT_MAX_E; ++e) acc[e] *= corr;
        for (uint u = 0; u < ATT_DEC_U; ++u) {
            const ulong j = j0 + (ulong)u * ATT_DEC_SG;
            if (j < visible) {  // never multiply a probability into rows that were not read
                device const ushort* vr = vb + j * d;
                for (uint e = 0; e < ATT_MAX_E; ++e) {
                    const uint di = lane + 32 * e;
                    if (di < d) acc[e] += pr[u] * bf2f(vr[di]);
                }
            }
        }
    }

    if (lane == 0) {
        ms[sg] = m;
        ls[sg] = l;
    }
    for (uint e = 0; e < ATT_MAX_E; ++e) {
        const uint di = lane + 32 * e;
        if (di < d) accs[sg][di] = acc[e];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint tid = sg * 32 + lane;
    float mt = -INFINITY;
    for (uint s = 0; s < ATT_DEC_SG; ++s) mt = max(mt, ms[s]);
    float w[ATT_DEC_SG], lt = 0.0f;
    for (uint s = 0; s < ATT_DEC_SG; ++s) {
        w[s] = ms[s] == -INFINITY ? 0.0f : exp(ms[s] - mt);  // a simdgroup that saw no key contributes nothing
        lt += ls[s] * w[s];
    }
    for (uint di = tid; di < d; di += ATT_DEC_SG * 32) {
        float num = 0.0f;
        for (uint s = 0; s < ATT_DEC_SG; ++s) num += accs[s][di] * w[s];
        out[p.s3 + (ulong)head * d + di] = f2bf(num / lt);
    }
}

// ---- AttnDecode, 16-byte loads: head_dim a power of two in [8, 256], K / V cache 16-byte aligned --------------------------
// Same bindings and OpP as attn_decode_bf16. Each lane loads one 16-byte chunk (8 bf16) of a K / V row, so D / 8 lanes
// cover a row and a simdgroup works on KPS = 32 / (D / 8) keys at once (a "slot" per group of D / 8 lanes). Every slot
// of every simdgroup is its own online-softmax stream (m, l, acc[8] per lane, uniform over the lanes of a slot); the
// SG * KPS streams are merged through threadgroup memory at the end, as in attn_decode_bf16. The score reduction over
// the D / 8 lanes of a slot is a butterfly (simd_shuffle_xor) so it stays inside the slot.
// The 2-byte-load kernel reaches about 40 GB/s of KV traffic; Apple GPUs want wide per-lane loads (the GEMV spike
// saw the same), hence this variant.
constant constexpr uint ATT_V8_U = 2;  // units of work (KPS keys) in flight per step

inline void unpack8(uint4 w, thread float* f) {
    f[0] = as_type<float>(w.x << 16);
    f[1] = as_type<float>(w.x & 0xffff0000u);
    f[2] = as_type<float>(w.y << 16);
    f[3] = as_type<float>(w.y & 0xffff0000u);
    f[4] = as_type<float>(w.z << 16);
    f[5] = as_type<float>(w.z & 0xffff0000u);
    f[6] = as_type<float>(w.w << 16);
    f[7] = as_type<float>(w.w & 0xffff0000u);
}

kernel void attn_decode_bf16_v8(device const ushort* q [[buffer(0)]],
                                device const ushort* kc [[buffer(1)]],
                                device const ushort* vc [[buffer(2)]],
                                device ushort* out [[buffer(3)]],
                                device const int* posbuf [[buffer(4)]],
                                constant OpP& p [[buffer(5)]],
                                uint head [[threadgroup_position_in_grid]],
                                uint sg [[simdgroup_index_in_threadgroup]],
                                uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float ms[ATT_DEC_SG * 32], ls[ATT_DEC_SG * 32];
    threadgroup float accs[ATT_DEC_SG * 32 * 8];  // stream * D + d; streams * D == SG * 32 * 8

    const uint n = p.b, d = p.d;
    const ulong cap = p.e;
    const uint lpk = d / 8, kps = 32 / lpk;
    const uint slot = lane / lpk, chunk = lane % lpk;
    const int pos_raw = p.g != 0 ? posbuf[0] : (int)p.f;
    const ulong visible = min((ulong)max(pos_raw, 0) + 1, cap);
    const uint kvh = head / (n / p.c);
    // 16-byte aligned by the launcher (skips are 0)
    device const uint4* kb = (device const uint4*)(kc + (ulong)kvh * cap * d) + chunk;
    device const uint4* vb = (device const uint4*)(vc + (ulong)kvh * cap * d) + chunk;
    const ulong row4 = lpk;  // uint4 per row

    float qv[8], acc[8];
    for (uint i = 0; i < 8; ++i) {
        qv[i] = bf2f(q[p.s0 + (ulong)head * d + chunk * 8 + i]);
        acc[i] = 0.0f;
    }
    float m = -INFINITY, l = 0.0f;
    const ulong units = (visible + kps - 1) / kps;
    for (ulong u0 = sg; u0 < units; u0 += (ulong)ATT_DEC_SG * ATT_V8_U) {
        float sc[ATT_V8_U];
        for (uint u = 0; u < ATT_V8_U; ++u) {
            const ulong j = (u0 + (ulong)u * ATT_DEC_SG) * kps + slot;
            float dot = 0.0f;
            if (j < visible) {
                float kf[8];
                unpack8(kb[j * row4], kf);
                for (uint i = 0; i < 8; ++i) dot += qv[i] * kf[i];
            }
            for (uint o = 1; o < lpk; o <<= 1) dot += simd_shuffle_xor(dot, (ushort)o);
            sc[u] = j < visible ? dot * p.f0 : -INFINITY;
        }
        float m_new = m;
        for (uint u = 0; u < ATT_V8_U; ++u) m_new = max(m_new, sc[u]);
        // a slot whose keys are all past the end keeps m = -inf; its score -inf must not meet m_new = -inf
        const float corr = m_new == -INFINITY ? 1.0f : exp(m - m_new);
        float pr[ATT_V8_U], psum = 0.0f;
        for (uint u = 0; u < ATT_V8_U; ++u) {
            pr[u] = m_new == -INFINITY ? 0.0f : exp(sc[u] - m_new);
            psum += pr[u];
        }
        l = l * corr + psum;
        m = m_new;
        for (uint i = 0; i < 8; ++i) acc[i] *= corr;
        for (uint u = 0; u < ATT_V8_U; ++u) {
            const ulong j = (u0 + (ulong)u * ATT_DEC_SG) * kps + slot;
            if (j < visible) {
                float vf[8];
                unpack8(vb[j * row4], vf);
                for (uint i = 0; i < 8; ++i) acc[i] += pr[u] * vf[i];
            }
        }
    }

    const uint stream = sg * kps + slot;
    if (chunk == 0) {
        ms[stream] = m;
        ls[stream] = l;
    }
    for (uint i = 0; i < 8; ++i) accs[(ulong)stream * d + chunk * 8 + i] = acc[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint tid = sg * 32 + lane;
    const uint streams = ATT_DEC_SG * kps;
    float mt = -INFINITY;
    for (uint s = 0; s < streams; ++s) mt = max(mt, ms[s]);
    float lt = 0.0f;
    for (uint s = 0; s < streams; ++s) lt += ms[s] == -INFINITY ? 0.0f : ls[s] * exp(ms[s] - mt);
    for (uint di = tid; di < d; di += ATT_DEC_SG * 32) {
        float num = 0.0f;
        for (uint s = 0; s < streams; ++s) num += ms[s] == -INFINITY ? 0.0f : accs[(ulong)s * d + di] * exp(ms[s] - mt);
        out[p.s3 + (ulong)head * d + di] = f2bf(num / lt);
    }
}

// ---- AttnPrefill: flash-style, one threadgroup per (block of ATT_PF_BQ query rows, head) --------------------------------
// bind: 0 q [T,N,D], 1 k cache, 2 v cache, 3 out [T,N,D], 4 OpP.
// Simdgroup r of the threadgroup owns query row (block * BQ + r). The K tile of ATT_PF_BK keys is staged once in
// threadgroup memory (shared by the BQ rows; row stride D + 2 elements keeps the lane-per-key reads conflict free) and
// each lane scores one key of it, so max / sum of a tile are one simd_max / simd_sum. P.V then runs lane-per-output-dim:
// probabilities are broadcast from the lane that owns the key and V rows are read straight from the cache, coalesced.
// Tiles beyond the block's last visible key are never loaded; keys past a row's causal limit are masked (score -inf,
// and never multiplied into V, so stale tile rows cannot leak NaN into the accumulator).
constant constexpr uint ATT_PF_BQ = 8;
constant constexpr uint ATT_PF_BK = 32;

kernel void attn_prefill_bf16(device const ushort* q [[buffer(0)]],
                              device const ushort* kc [[buffer(1)]],
                              device const ushort* vc [[buffer(2)]],
                              device ushort* out [[buffer(3)]],
                              constant OpP& p [[buffer(4)]],
                              uint2 tg [[threadgroup_position_in_grid]],
                              uint sg [[simdgroup_index_in_threadgroup]],
                              uint lane [[thread_index_in_simdgroup]],
                              uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float qs[ATT_PF_BQ][ATT_MAX_D];
    threadgroup ushort2 kt[ATT_PF_BK * (ATT_MAX_D / 2 + 1)];  // [key][D + 2] bf16, as pairs

    const uint t_total = p.a, n = p.b, d = p.d;
    const ulong cap = p.e, pos0 = p.f;
    const uint head = tg.y;
    const uint qi = tg.x * ATT_PF_BQ + sg;
    const bool active = qi < t_total;
    const uint kvh = head / (n / p.c);
    const uint dp2 = d / 2 + 1;  // row stride of kt, in pairs
    device const ushort* kb = kc + p.s1 + (ulong)kvh * cap * d;
    device const ushort* vb = vc + p.s2 + (ulong)kvh * cap * d;

    if (active) {
        for (uint di = lane; di < d; di += 32) qs[sg][di] = bf2f(q[p.s0 + ((ulong)qi * n + head) * d + di]);
    }
    const ulong visible = pos0 + qi + 1;  // keys 0 .. pos0 + qi
    // keys the last row of this block can see: the whole block walks the same tiles
    const ulong block_keys = pos0 + min((ulong)t_total, (ulong)(tg.x + 1) * ATT_PF_BQ);

    float acc[ATT_MAX_E];
    for (uint e = 0; e < ATT_MAX_E; ++e) acc[e] = 0.0f;
    float m = -INFINITY, l = 0.0f;

    for (ulong tile = 0; tile < block_keys; tile += ATT_PF_BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);  // everyone is done with the previous tile (and qs is written)
        const uint rows = (uint)min((ulong)ATT_PF_BK, block_keys - tile);
        for (uint idx = tid; idx < rows * (d / 2); idx += ATT_PF_BQ * 32) {
            const uint r = idx / (d / 2), e2 = idx % (d / 2);
            device const ushort* src = kb + (tile + r) * d + 2 * e2;
            kt[r * dp2 + e2] = ushort2(src[0], src[1]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (!active) continue;
        const ulong nvalid = visible > tile ? min((ulong)ATT_PF_BK, visible - tile) : 0;
        if (nvalid == 0) continue;  // uniform per simdgroup

        float s = -INFINITY;
        if (lane < nvalid) {
            threadgroup const ushort2* kr = kt + lane * dp2;
            threadgroup const float* qr = qs[sg];
            float dot = 0.0f;
            for (uint e2 = 0; e2 < d / 2; ++e2) {
                const ushort2 kk = kr[e2];
                dot += qr[2 * e2] * bf2f(kk.x) + qr[2 * e2 + 1] * bf2f(kk.y);
            }
            s = dot * p.f0;
        }
        const float tmax = simd_max(s);
        const float m_new = max(m, tmax);
        const float corr = exp(m - m_new);  // m == -inf on the first tile -> 0
        const float pj = lane < nvalid ? exp(s - m_new) : 0.0f;
        l = l * corr + simd_sum(pj);
        m = m_new;
        for (uint e = 0; e < ATT_MAX_E; ++e) acc[e] *= corr;
        for (uint jj = 0; jj < (uint)nvalid; ++jj) {
            const float pb = simd_broadcast(pj, (ushort)jj);
            device const ushort* vr = vb + (tile + jj) * d;
            for (uint e = 0; e < ATT_MAX_E; ++e) {
                const uint di = lane + 32 * e;
                if (di < d) acc[e] += pb * bf2f(vr[di]);
            }
        }
    }

    if (active) {
        const float inv = 1.0f / l;
        for (uint e = 0; e < ATT_MAX_E; ++e) {
            const uint di = lane + 32 * e;
            if (di < d) out[p.s3 + ((ulong)qi * n + head) * d + di] = f2bf(acc[e] * inv);
        }
    }
}
