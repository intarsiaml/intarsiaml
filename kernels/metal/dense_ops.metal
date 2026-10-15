// Dense BF16 elementwise / row ops for the Qwen3 decode and prefill path. Semantics follow the cpu-ref oracle
// (crates/intarsia-backend-cpu/src/reference.rs): bf16 in, f32 math, bf16 out (round to nearest even), one rounding at
// the end of each op. Reductions have a fixed order (thread-strided partial sums, simd_sum, then a serial sum over the
// simdgroup partials), so results are run-to-run identical.
//
// Speed is phase B; these are one-element-per-thread / one-row-per-threadgroup kernels with 2-byte loads.

#include "common.h"

// ---- Embedding: out[t,:] = table[ids[t],:] ------------------------------------------------------------------------
// bind: 0 ids (i32), 1 table (bf16 [V,H]), 2 out (bf16 [T,H]). a = T, b = H, c = V. s1 / s2 skips of table / out.
// An id outside [0, V) writes zeros (the cpu oracle errors; a GPU kernel cannot).
kernel void embedding_bf16(device const int* ids [[buffer(0)]],
                           device const ushort* table [[buffer(1)]],
                           device ushort* out [[buffer(2)]],
                           constant OpP& p [[buffer(3)]],
                           uint gid [[thread_position_in_grid]]) {
    if (gid >= p.a * p.b) return;
    const uint row = gid / p.b, col = gid % p.b;
    const int id = ids[row];
    out[p.s2 + gid] = (id >= 0 && (uint)id < p.c) ? table[p.s1 + (ulong)id * p.b + col] : (ushort)0;
}

// ---- RmsNorm / QkNorm: out = x * rsqrt(mean(x^2) + eps) * (w + w_offset), per row of H --------------------------------
// One threadgroup per row. bind: 0 x, 1 w, 2 out. a = H, f0 = eps, f1 = w_offset. s0 / s1 / s2 skips. (QkNorm is the
// same kernel with rows = T*N and H = D.)
inline float tg_sum(float v, threadgroup float* part, uint lane, uint sg, uint nsg) {
    v = simd_sum(v);
    if (lane == 0) part[sg] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float t = 0.0f;
    for (uint j = 0; j < nsg; ++j) t += part[j];
    threadgroup_barrier(mem_flags::mem_threadgroup);  // part[] may be reused by the caller
    return t;
}

kernel void rmsnorm_bf16(device const ushort* x [[buffer(0)]],
                         device const ushort* w [[buffer(1)]],
                         device ushort* out [[buffer(2)]],
                         constant OpP& p [[buffer(3)]],
                         uint row [[threadgroup_position_in_grid]],
                         uint tid [[thread_position_in_threadgroup]],
                         uint tgs [[threads_per_threadgroup]],
                         uint lane [[thread_index_in_simdgroup]],
                         uint sg [[simdgroup_index_in_threadgroup]],
                         uint nsg [[simdgroups_per_threadgroup]]) {
    threadgroup float part[32];
    const uint h = p.a;
    device const ushort* xr = x + p.s0 + (ulong)row * h;
    device ushort* orow = out + p.s2 + (ulong)row * h;
    float ss = 0.0f;
    for (uint i = tid; i < h; i += tgs) {
        const float v = bf2f(xr[i]);
        ss = fma(v, v, ss);
    }
    const float tot = tg_sum(ss, part, lane, sg, nsg);
    const float r = 1.0f / precise::sqrt(tot / (float)h + p.f0);
    for (uint i = tid; i < h; i += tgs) orow[i] = f2bf(bf2f(xr[i]) * r * (bf2f(w[p.s1 + i]) + p.f1));
}

// ---- RmsNormAdd: sum = bf16(x + res); out = rmsnorm(sum) --------------------------------------------------------------
// The norm sees the bf16-rounded sum (the oracle quantises it to the sum's dtype first). bind: 0 x, 1 res, 2 w,
// 3 normed, 4 sum. a = H, f0 = eps, f1 = w_offset. s0..s4 skips in binding order.
kernel void rmsnorm_add_bf16(device const ushort* x [[buffer(0)]],
                             device const ushort* res [[buffer(1)]],
                             device const ushort* w [[buffer(2)]],
                             device ushort* normed [[buffer(3)]],
                             device ushort* sum [[buffer(4)]],
                             constant OpP& p [[buffer(5)]],
                             uint row [[threadgroup_position_in_grid]],
                             uint tid [[thread_position_in_threadgroup]],
                             uint tgs [[threads_per_threadgroup]],
                             uint lane [[thread_index_in_simdgroup]],
                             uint sg [[simdgroup_index_in_threadgroup]],
                             uint nsg [[simdgroups_per_threadgroup]]) {
    threadgroup float part[32];
    const uint h = p.a;
    const ulong base = (ulong)row * h;
    float ss = 0.0f;
    for (uint i = tid; i < h; i += tgs) {
        const float s = bf2f(f2bf(bf2f(x[p.s0 + base + i]) + bf2f(res[p.s1 + base + i])));
        ss = fma(s, s, ss);
    }
    const float tot = tg_sum(ss, part, lane, sg, nsg);
    const float r = 1.0f / precise::sqrt(tot / (float)h + p.f0);
    for (uint i = tid; i < h; i += tgs) {
        const ushort sb = f2bf(bf2f(x[p.s0 + base + i]) + bf2f(res[p.s1 + base + i]));
        sum[p.s4 + base + i] = sb;
        normed[p.s3 + base + i] = f2bf(bf2f(sb) * r * (bf2f(w[p.s2 + i]) + p.f1));
    }
}

// ---- Add: out = a + b -----------------------------------------------------------------------------------------------
// bind: 0 a, 1 b, 2 out. a = element count. s0..s2 skips.
kernel void add_bf16(device const ushort* a [[buffer(0)]],
                     device const ushort* b [[buffer(1)]],
                     device ushort* out [[buffer(2)]],
                     constant OpP& p [[buffer(3)]],
                     uint gid [[thread_position_in_grid]]) {
    if (gid >= p.a) return;
    out[p.s2 + gid] = f2bf(bf2f(a[p.s0 + gid]) + bf2f(b[p.s1 + gid]));
}

// ---- SiluGlu: out = silu(gate) * up ---------------------------------------------------------------------------------
// bind: 0 gate, 1 up, 2 out. a = element count. s0..s2 skips.
kernel void silu_glu_bf16(device const ushort* g [[buffer(0)]],
                          device const ushort* u [[buffer(1)]],
                          device ushort* out [[buffer(2)]],
                          constant OpP& p [[buffer(3)]],
                          uint gid [[thread_position_in_grid]]) {
    if (gid >= p.a) return;
    const float gv = bf2f(g[p.s0 + gid]);
    out[p.s2 + gid] = f2bf(gv / (1.0f + precise::exp(-gv)) * bf2f(u[p.s1 + gid]));
}

// ---- Rope: neox / rotate-half over the first `rd` values of each head, the rest copied ------------------------------------
// bind: 0 x [T,N,D], 1 out, 2 pos (i32 [1], read when `e` != 0). a = T, b = N, c = D, d = rd (rotary dim, resolved),
// e = use device position, f = host pos0, f0 = theta. s0 / s1 skips of x / out.
// angle = (pos0 + t) * theta^(-2k/rd); cos / sin are f32 like every real kernel's tables.
kernel void rope_bf16(device const ushort* x [[buffer(0)]],
                      device ushort* out [[buffer(1)]],
                      device const int* posbuf [[buffer(2)]],
                      constant OpP& p [[buffer(3)]],
                      uint gid [[thread_position_in_grid]]) {
    const uint d = p.c;
    if (gid >= p.a * p.b * d) return;
    const uint i = gid % d;
    const uint head = gid / d;  // t * N + n
    const uint rd = p.d;
    if (i >= rd) {
        out[p.s1 + gid] = x[p.s0 + gid];
        return;
    }
    const uint t = head / p.b;
    const uint half_ = rd / 2;
    const uint k = i < half_ ? i : i - half_;
    const uint pos0 = p.e != 0 ? (uint)posbuf[0] : p.f;
    const float ang = (float)(pos0 + t) * precise::pow(p.f0, -2.0f * (float)k / (float)rd);
    const float s = precise::sin(ang), c = precise::cos(ang);
    const ulong hb = (ulong)head * d;
    const float x1 = bf2f(x[p.s0 + hb + k]), x2 = bf2f(x[p.s0 + hb + k + half_]);
    out[p.s1 + gid] = f2bf(i < half_ ? x1 * c - x2 * s : x2 * c + x1 * s);
}

// ---- KvAppend: cache[h, pos0 + t, :] = new[t, h, :] (in place; other rows keep their bits) ------------------------------
// bind: 0 new [T,Nkv,D], 1 cache [Nkv,cap,D], 2 pos (i32 [1], read when `e` != 0). a = T, b = Nkv, c = D, d = cap,
// e = use device position, f = host pos0. s0 / s1 skips of new / cache.
kernel void kv_append_bf16(device const ushort* src [[buffer(0)]],
                           device ushort* cache [[buffer(1)]],
                           device const int* posbuf [[buffer(2)]],
                           constant OpP& p [[buffer(3)]],
                           uint gid [[thread_position_in_grid]]) {
    if (gid >= p.a * p.b * p.c) return;
    const uint e = gid % p.c;
    const uint h = (gid / p.c) % p.b;
    const uint t = gid / (p.c * p.b);
    const uint pos0 = p.e != 0 ? (uint)posbuf[0] : p.f;
    if (pos0 + t >= p.d) return;  // a device-side position past the cache is dropped, not written out of bounds
    cache[p.s1 + ((ulong)h * p.d + pos0 + t) * p.c + e] = src[p.s0 + gid];
}

// KvAppend into an fp8 e4m3 cache (`--kv-dtype fp8_e4m3`): bf16 rows, one e4m3 byte per element (round to nearest even,
// saturating, f32_to_e4m3 of common.h = the oracle's `f32_to_fp8_e4m3`). Same arguments as kv_append_bf16; s1 in bytes.
kernel void kv_append_f8(device const ushort* src [[buffer(0)]],
                         device uchar* cache [[buffer(1)]],
                         device const int* posbuf [[buffer(2)]],
                         constant OpP& p [[buffer(3)]],
                         uint gid [[thread_position_in_grid]]) {
    if (gid >= p.a * p.b * p.c) return;
    const uint e = gid % p.c;
    const uint h = (gid / p.c) % p.b;
    const uint t = gid / (p.c * p.b);
    const uint pos0 = p.e != 0 ? (uint)posbuf[0] : p.f;
    if (pos0 + t >= p.d) return;
    cache[p.s1 + ((ulong)h * p.d + pos0 + t) * p.c + e] = (uchar)f32_to_e4m3(bf2f(src[p.s0 + gid]));
}

// ---- Argmax: lowest index among the maximal values of each row (NaN never wins; an all -inf / NaN row gives 0) -----------
// One threadgroup per row. bind: 0 logits [R,V] (f32 or bf16), 1 out (i32 [R]). a = V. s0 skip of logits.
inline float ld_logit(device const float* l, ulong i) { return l[i]; }
inline float ld_logit(device const ushort* l, ulong i) { return bf2f(l[i]); }

template <typename T>
kernel void argmax_t(device const T* logits [[buffer(0)]],
                     device int* out [[buffer(1)]],
                     constant OpP& p [[buffer(2)]],
                     uint row [[threadgroup_position_in_grid]],
                     uint tid [[thread_position_in_threadgroup]],
                     uint tgs [[threads_per_threadgroup]],
                     uint lane [[thread_index_in_simdgroup]],
                     uint sg [[simdgroup_index_in_threadgroup]],
                     uint nsg [[simdgroups_per_threadgroup]]) {
    threadgroup float pv[32];
    threadgroup uint pi[32];
    const uint v = p.a;
    device const T* l = logits + p.s0 + (ulong)row * v;
    float bv = -INFINITY;
    uint bi = 0xffffffffu;
    for (uint i = tid; i < v; i += tgs) {
        const float x = ld_logit(l, i);
        if (x > bv) { bv = x; bi = i; }  // strict: the lowest index of this thread's maximum
    }
    float m = simd_max(bv);
    uint mi = simd_min(bv == m ? bi : 0xffffffffu);
    if (lane == 0) { pv[sg] = m; pi[sg] = mi; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float gm = -INFINITY;
        uint gi = 0xffffffffu;
        for (uint j = 0; j < nsg; ++j) {
            if (pv[j] > gm || (pv[j] == gm && pi[j] < gi)) { gm = pv[j]; gi = pi[j]; }
        }
        out[row] = gi == 0xffffffffu ? 0 : (int)gi;
    }
}

template [[host_name("argmax_f32")]] kernel void argmax_t<float>(
    device const float*, device int*, constant OpP&, uint, uint, uint, uint, uint, uint);
template [[host_name("argmax_bf16")]] kernel void argmax_t<ushort>(
    device const ushort*, device int*, constant OpP&, uint, uint, uint, uint, uint, uint);

// =====================================================================================================================
// Phase B2b: low-latency forms of the row norms. Bitwise the same results as `rmsnorm_bf16` / `rmsnorm_add_bf16`: the
// same element-to-thread assignment (element i belongs to thread i % 256) and the same reduction order (strided fma
// per thread, simd_sum, serial sum over the simdgroup partials). What changes is the latency: every load of the row (x,
// res, w) is issued before the reduction and the elements stay in registers for the second pass, so a row costs one
// memory round trip, one barrier and no re-read. Any alignment, any skip.
// The host picks NIT = max elements per thread and the width: H <= 256 -> NIT 1 with roundup32(H) threads (the idle
// threads of the 256-thread kernel only contributed exact zeros), H <= 4096 -> NIT 16, H <= 8192 -> NIT 32, 256 threads.
// =====================================================================================================================

// Threadgroup sum with a single barrier (part[] is used once) and none at all for one simdgroup.
inline float tg_sum1(float v, threadgroup float* part, uint lane, uint sg, uint nsg) {
    v = simd_sum(v);
    if (nsg == 1) return v;
    if (lane == 0) part[sg] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float t = 0.0f;
    for (uint j = 0; j < nsg; ++j) t += part[j];
    return t;
}

template <int NIT, bool FULL>
kernel void rmsnorm_bf16_ex_t(device const ushort* x [[buffer(0)]],
                              device const ushort* w [[buffer(1)]],
                              device ushort* out [[buffer(2)]],
                              constant OpP& p [[buffer(3)]],
                              uint row [[threadgroup_position_in_grid]],
                              uint tid [[thread_position_in_threadgroup]],
                              uint tgs [[threads_per_threadgroup]],
                              uint lane [[thread_index_in_simdgroup]],
                              uint sg [[simdgroup_index_in_threadgroup]],
                              uint nsg [[simdgroups_per_threadgroup]]) {
    threadgroup float part[32];
    const uint h = p.a;
    device const ushort* xr = x + p.s0 + (ulong)row * h;
    device const ushort* wr = w + p.s1;
    device ushort* orow = out + p.s2 + (ulong)row * h;
    float xf[NIT];
    float wf[NIT];
    for (int j = 0; j < NIT; ++j) {
        const uint i = tid + (uint)j * tgs;
        if (FULL || i < h) {
            xf[j] = bf2f(xr[i]);
            wf[j] = bf2f(wr[i]);
        }
    }
    float ss = 0.0f;
    for (int j = 0; j < NIT; ++j) {
        if (FULL || tid + (uint)j * tgs < h) ss = fma(xf[j], xf[j], ss);
    }
    const float tot = tg_sum1(ss, part, lane, sg, nsg);
    const float r = 1.0f / precise::sqrt(tot / (float)h + p.f0);
    for (int j = 0; j < NIT; ++j) {
        const uint i = tid + (uint)j * tgs;
        if (FULL || i < h) orow[i] = f2bf(xf[j] * r * (wf[j] + p.f1));
    }
}

template <int NIT, bool FULL>
kernel void rmsnorm_add_bf16_ex_t(device const ushort* x [[buffer(0)]],
                                  device const ushort* res [[buffer(1)]],
                                  device const ushort* w [[buffer(2)]],
                                  device ushort* normed [[buffer(3)]],
                                  device ushort* sum [[buffer(4)]],
                                  constant OpP& p [[buffer(5)]],
                                  uint row [[threadgroup_position_in_grid]],
                                  uint tid [[thread_position_in_threadgroup]],
                                  uint tgs [[threads_per_threadgroup]],
                                  uint lane [[thread_index_in_simdgroup]],
                                  uint sg [[simdgroup_index_in_threadgroup]],
                                  uint nsg [[simdgroups_per_threadgroup]]) {
    threadgroup float part[32];
    const uint h = p.a;
    // row pointers once: the per-access `skip + 64-bit row base + i` sums sat on every load and store
    const ulong base = (ulong)row * h;
    device const ushort* xr = x + p.s0 + base;
    device const ushort* rr = res + p.s1 + base;
    device const ushort* wr = w + p.s2;
    device ushort* nr = normed + p.s3 + base;
    device ushort* sr = sum + p.s4 + base;
    float xf[NIT];
    float rf[NIT];
    float wf[NIT];
    for (int j = 0; j < NIT; ++j) {
        const uint i = tid + (uint)j * tgs;
        if (FULL || i < h) {
            xf[j] = bf2f(xr[i]);
            rf[j] = bf2f(rr[i]);
            wf[j] = bf2f(wr[i]);
        }
    }
    float sb[NIT];  // the bf16-rounded sum, as a float: the norm sees the rounded value
    float ss = 0.0f;
    for (int j = 0; j < NIT; ++j) {
        if (FULL || tid + (uint)j * tgs < h) {
            sb[j] = bf2f(f2bf(xf[j] + rf[j]));
            ss = fma(sb[j], sb[j], ss);
        }
    }
    const float tot = tg_sum1(ss, part, lane, sg, nsg);
    const float r = 1.0f / precise::sqrt(tot / (float)h + p.f0);
    for (int j = 0; j < NIT; ++j) {
        const uint i = tid + (uint)j * tgs;
        if (FULL || i < h) {
            sr[i] = f2bf(sb[j]);
            nr[i] = f2bf(sb[j] * r * (wf[j] + p.f1));
        }
    }
}

// `_f` variants: H == NIT * threads (every thread owns exactly NIT elements, e.g. H = 4096 = 16 x 256), so the bounds
// selects are gone. The host picks them; same arithmetic, same results.
#define INST_NORM_EX(NAME, NIT, FULL)                                                                                 \
    template [[host_name(NAME)]] kernel void rmsnorm_bf16_ex_t<NIT, FULL>(                                           \
        device const ushort*, device const ushort*, device ushort*, constant OpP&, uint, uint, uint, uint, uint, uint);
#define INST_NORM_ADD_EX(NAME, NIT, FULL)                                                                             \
    template [[host_name(NAME)]] kernel void rmsnorm_add_bf16_ex_t<NIT, FULL>(                                       \
        device const ushort*, device const ushort*, device const ushort*, device ushort*, device ushort*,          \
        constant OpP&, uint, uint, uint, uint, uint, uint);
INST_NORM_EX("rmsnorm_bf16_ex1", 1, false)
INST_NORM_EX("rmsnorm_bf16_ex16", 16, false)
INST_NORM_EX("rmsnorm_bf16_ex32", 32, false)
INST_NORM_EX("rmsnorm_bf16_ex1_f", 1, true)
INST_NORM_EX("rmsnorm_bf16_ex16_f", 16, true)
INST_NORM_EX("rmsnorm_bf16_ex32_f", 32, true)
INST_NORM_ADD_EX("rmsnorm_add_bf16_ex1", 1, false)
INST_NORM_ADD_EX("rmsnorm_add_bf16_ex16", 16, false)
INST_NORM_ADD_EX("rmsnorm_add_bf16_ex32", 32, false)
INST_NORM_ADD_EX("rmsnorm_add_bf16_ex1_f", 1, true)
INST_NORM_ADD_EX("rmsnorm_add_bf16_ex16_f", 16, true)
INST_NORM_ADD_EX("rmsnorm_add_bf16_ex32_f", 32, true)

// Argmax over f32 logits, vector form: V % 4 == 0 and the row 16-byte aligned (no skip). Same result as `argmax_f32`
// (lowest index of the maximum, NaN never wins, empty / all -inf row gives 0). One threadgroup per row of up to 1024 threads;
// thread t visits the float4 chunks t, t + tgs, ... in increasing order, so its running (max, index) keeps the lowest
// index of its own maximum; the cross-thread resolution is the same as the scalar kernel's.
constant constexpr uint AMX_U = 8;
inline void amx_upd(float4 x, uint i0, thread float& bv, thread uint& bi) {
    if (x.x > bv) { bv = x.x; bi = i0; }
    if (x.y > bv) { bv = x.y; bi = i0 + 1; }
    if (x.z > bv) { bv = x.z; bi = i0 + 2; }
    if (x.w > bv) { bv = x.w; bi = i0 + 3; }
}

kernel void argmax_f32_v4(device const float4* logits [[buffer(0)]],
                          device int* out [[buffer(1)]],
                          constant OpP& p [[buffer(2)]],
                          uint row [[threadgroup_position_in_grid]],
                          uint tid [[thread_position_in_threadgroup]],
                          uint tgs [[threads_per_threadgroup]],
                          uint lane [[thread_index_in_simdgroup]],
                          uint sg [[simdgroup_index_in_threadgroup]],
                          uint nsg [[simdgroups_per_threadgroup]]) {
    threadgroup float pv[32];
    threadgroup uint pi[32];
    const uint v4 = p.a / 4;
    device const float4* l = logits + (ulong)row * v4;
    float bv = -INFINITY;
    uint bi = 0xffffffffu;
    // AMX_U chunks in flight per thread (one load round trip per AMX_U chunks instead of per chunk), visited in the
    // same increasing order as the one-at-a-time loop, so the result is the same.
    uint c = tid;
    for (; c + (AMX_U - 1) * tgs < v4; c += AMX_U * tgs) {
        float4 xs[AMX_U];
        for (uint u = 0; u < AMX_U; ++u) xs[u] = l[c + u * tgs];
        for (uint u = 0; u < AMX_U; ++u) amx_upd(xs[u], (c + u * tgs) * 4, bv, bi);
    }
    for (; c < v4; c += tgs) amx_upd(l[c], c * 4, bv, bi);
    float m = simd_max(bv);
    uint mi = simd_min(bv == m ? bi : 0xffffffffu);
    if (lane == 0) { pv[sg] = m; pi[sg] = mi; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float gm = -INFINITY;
        uint gi = 0xffffffffu;
        for (uint j = 0; j < nsg; ++j) {
            if (pv[j] > gm || (pv[j] == gm && pi[j] < gi)) { gm = pv[j]; gi = pi[j]; }
        }
        out[row] = gi == 0xffffffffu ? 0 : (int)gi;
    }
}

// KvAppend into an int8 block-scaled cache (`--kv-dtype int8`): kv_append_f8's counterpart. One thread per element; head_dim
// (p.c) must be a multiple of 32, so a simdgroup is 32 elements of one head row = one block (quantiser: common.h kvq_*).
kernel void kv_append_i8(device const ushort* src [[buffer(0)]],
                         device uchar* cache [[buffer(1)]],
                         device const int* posbuf [[buffer(2)]],
                         constant OpP& p [[buffer(3)]],
                         uint gid [[thread_position_in_grid]],
                         uint lane [[thread_index_in_simdgroup]]) {
    if (gid >= p.a * p.b * p.c) return;  // the grid is a multiple of 32 threads: a whole simdgroup leaves together
    const uint e = gid % p.c;
    const uint h = (gid / p.c) % p.b;
    const uint t = gid / (p.c * p.b);
    const uint pos0 = p.e != 0 ? (uint)posbuf[0] : p.f;
    if (pos0 + t >= p.d) return;  // uniform over the simdgroup (same t, h)
    const float x = bf2f(src[p.s0 + gid]);
    const float s = kvq_scale(x, KVQ_B, lane);
    const ulong row = (ulong)h * p.d + pos0 + t;
    cache[p.s1 + row * p.c + e] = (uchar)((int)kvq_q(x, s) & 0xff);
    if (lane == 0) ((device ushort*)(cache + p.s1 + (ulong)p.b * p.d * p.c))[row * (p.c / KVQ_B) + e / KVQ_B] = f2bf(s);
}
