// Inkling-2-Mini routed experts on HIP (lane ink-hip-moe): Op::MoeExperts with the SiTU activation over NVFP4, and
// Op::MoeExpertsMxfp8 (e4m3 weights, E8M0 scale per 32 along the input dim; silu / gelu_tanh / situ). W4A16 only, the
// checkpoint's row-major layout only (ink.moe_layout = 0). Contract: crates/intarsia-core/src/inkling2/ops.rs ("MoeExpertsMxfp8
// and the situ activation"); reference quant_ref.rs / ink_ref.rs.
//
// A port of the SIMT kernels of kernels/cuda/ink_moe.cu (union-grouped decode, Metal lane ink-m-moe's design): every block lists
// the call's (token, slot) pairs by distinct expert in shared memory, and its waves take experts; a wave streams its rows of the
// expert once and runs every pair routed to that expert on them (in batches of NB accumulators; a batch past the first re-reads
// the rows). Two launches:
//   gate_up  block = 4 rows of I; per row 8 lanes, each lane one 32-element chunk of the gate row and of the up row per step;
//            h[pair, r] = epilogue(rA(x . gate_r), rA(x . up_r)) (situ: times the routing weight)
//   down     block = 8 output rows; per row 4 lanes; y = rA(h[pair] . down_r), term = y (situ) or rA(y * w); the terms land
//            in shared memory and out[t, r] = rA(sum_k term[t, k]) in slot order.
// Per (pair, row) the chunk order, the fma chain per chunk, the lane reduction and the roundings never depend on T, on the
// routing of other tokens, on which pairs share a batch or on the load window (PRE), so a T-row call is bitwise T one-row calls
// (rows-exact), and the same per-element arithmetic as the CUDA kernels.
// More than kDecPairs pairs (prefill): the same kernels on a pair table built once in scratch by a grouping launch, the
// terms through scratch y (bf16-exact values: y and rA(y * w) are bf16) and a gather launch: the same bits. By default
// (INTARSIA_HIP_INK_MOE_WMMA unset or 1; user decision 2026-11-26; `=0` is the revert) the ordinary names run bf16 WMMA on that
// pair table instead (ink_moe_wmma_kernel below: other f32 order, within the op's tolerance).
//
// What differs from CUDA (gfx12):
//  * the e2m1 / e4m3 weights are decoded with the fp8 conversion instruction (v_cvt_pk_f32_fp8, two values per instruction; the
//    scheme of moe_wmma_r7.h kMwCvt): an e2m1 code becomes the e4m3 byte of its value * 2^-6 and the block scale carries the 2^6,
//    so every product and every fma chain is the exact 2^-6 multiple of the plain one and the scaled sum has the same bits;
//  * no prefetch and no dependent launch: RDNA4 has neither. A lane instead requests the weights of PRE of its steps before it
//    computes on the first (INTARSIA_HIP_INK_MOE_PRE, see moe_pre; the same bits for every value), so the DRAM latency of a
//    step overlaps the arithmetic of the ones before it;
//  * shared memory sized by the call (sz_dp / sz_em): 2.3 KB per down block at decode instead of 32.8 KB, because the LDS a
//    block holds caps the blocks resident per workgroup processor.
//
// Scratch: the core `moe_experts_scratch` layout: pairs int[P] | offsets int[E+1] | active int[E+1] | h bf16[P,I] | y bf16[P,H].
#include "ik_quant.h"
#include "ik_hip_abi_inkmoe.h"
#include "moe_wmma.h"
#include <limits.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

namespace {

constexpr int kThreads = 256;
constexpr int kWarpsB = kThreads / IK_WARP;
constexpr int kDecPairs = 512;  // pairs grouped in shared memory (decode / verify / short chunks)
constexpr int kEMax = 1024;
constexpr int kGuRows = 4;  // rows of I per gate_up block (8 lanes each)
constexpr int kDnRows = 8;  // output rows per down block (4 lanes each)
// pairs per accumulator batch: NB 4 on the shared-memory path (decode: mostly one pair per expert; fewer live registers),
// NB 8 on the pair-table path (prefill: an expert's rows are decoded once per 8 pairs)

enum { FMT_NV = 0, FMT_MX = 1 };
enum { ACT_SILU = 0, ACT_GELU = 1, ACT_SITU = 2 };

int g_probe = 0;  // benchmarks: 1 = only gate_up, 2 = only down, 3 / 4 = both launches with MoeArgs::probe (garbage output)
int g_pre_gu = -1, g_pre_dn = -1;  // < 0: INTARSIA_HIP_INK_MOE_PRE / the default
int g_lean = -1;                   // < 0: INTARSIA_HIP_INK_MOE_LEAN / the default (1)
int g_nb = 0;                      // tests / benchmarks: pairs per batch below the pair table (4 | 8); else the default rule
int g_wmma = -1;                   // tests / benchmarks: 0 no tensor-core prefill, 1 the rule, 32 | 64 its pair tile; < 0: INTARSIA_HIP_INK_MOE_WMMA / the default
int g_sz = 0;                      // tests / benchmarks: the smallest LDS size class a call may take (0 | 1 | 2); 4: by the call, in-wave grouping up to 32 pairs
int g_dec = -1;                    // tests / benchmarks: 0 | 1 the one-wave grouping of class 0; < 0: INTARSIA_HIP_INK_DEC_FAST bit 1 / the default (on)

// LDS size classes: the group tables (and down's terms) are sized by the call, because LDS per block decides how many blocks a
// workgroup processor holds (64 KB): class 2 is 16.4 KB of tables + 16 KB of terms (one down block, three gate_up blocks per
// WGP), class 0 is 1.3 KB + 1 KB. 0: up to 32 pairs and 128 experts (decode, T <= 4 at top-8); 1: up to 128 pairs and 128
// experts; 2: up to kDecPairs pairs and kEMax experts; 3: the pair-table path (no tables in LDS).
// 4: up to kWavePairs pairs, grouped inside each wave in registers: no tables, no barriers; LDS = down's terms. Opt-in
// (ik_ink_moe_set_sz(4)): it measured slower than class 0 (more registers per wave; window 5).
constexpr int kWavePairs = 32;
constexpr int sz_dp(int z) { return z == 0 ? 32 : (z == 1 ? 128 : (z == 2 ? 512 : 1)); }
constexpr int sz_em(int z) { return z == 0 ? 128 : (z == 1 ? 128 : (z == 2 ? 1024 : 1)); }
constexpr int sz_terms(int z) { return z == 4 ? kWavePairs : sz_dp(z); }

struct MoeArgs {
    const ik_bf16* x;
    const int* ids;
    const float* wts;
    const uint8_t* guw;  // gate_up / w13 payload
    const uint8_t* gus;  // its scales (e4m3 per 16 / E8M0 per 32)
    const uint8_t* dnw;
    const uint8_t* dns;
    const float* gs;  // NVFP4 global scales [E,3] (null for MXFP8)
    ik_bf16* h;       // scratch [P,I]
    ik_bf16* y;       // scratch [P,H] (table path)
    // the pair table (table path): dpair [P], doff [E+1], dexp [E] + nd at dexp[E]
    int* t_pair;
    int* t_off;
    int* t_exp;
    void* out;
    int T, K, E, H, I;
    float gc, uc;
    int table;
    int probe;  // benchmarks: 3 = every thread returns at once, 4 = after the grouping (the launch + start-up cost of a call)
    int wg;     // class 0 with at most 32 pairs: the one-wave grouping (group_wave) instead of group_pairs
    int y0;     // tensor-core prefill in pieces (INTARSIA_HIP_INK_PF2 bits 4 / 8): first feature tile of this launch
};

IK_DEVI float ink_rbf(float v) { return ik_b2f(ik_f2b(v)); }
IK_DEVI float sigm(float z) { return 1.f / (1.f + expf(-z)); }
// gelu_tanh as the other backends' epilogues write it
IK_DEVI float gelu_tanh(float x) { return 0.5f * x * (1.f + tanhf(0.7978845608028654f * (x + 0.044715f * x * x * x))); }

template <int ACT>
IK_DEVI float act_h(float g, float u, float w, float gc, float uc) {
    if (ACT == ACT_SITU) return ink_rbf(gc * tanhf(g / gc) * sigm(g) * uc * tanhf(u / uc) * w);
    const float a = ink_rbf(ACT == ACT_GELU ? gelu_tanh(g) : g / (1.f + expf(-g)));
    return ink_rbf(a * u);
}

// ---- grouping: the block's threads list the valid pairs by distinct expert ---------------------------------------------
// first [E] (shared), nd, dexp [nd], doff [nd+1], dpair [nvalid]; pd / ps / cnt are [P] temporaries. The order of the experts
// and of the pairs within an expert comes from atomics: it decides only which wave runs what, never a bit of the result.
IK_DEVI void group_pairs(const int* __restrict__ ids, int P, int E, int* first, int* nd, int* dexp, int* doff, int* dpair, int* pd, int* ps,
                         int* cnt) {
    const int tid = threadIdx.x, nt = blockDim.x;
    for (int e = tid; e < E; e += nt) first[e] = INT_MAX;
    if (tid == 0) *nd = 0;
    __syncthreads();
    for (int p = tid; p < P; p += nt) {
        const int v = ids[p];
        if (v >= 0 && v < E) atomicMin(&first[v], p);
    }
    __syncthreads();
    for (int p = tid; p < P; p += nt) {
        const int v = ids[p];
        pd[p] = -1;
        if (v >= 0 && v < E && first[v] == p) {
            const int d = atomicAdd(nd, 1);
            dexp[d] = v;
            pd[p] = d;
            cnt[d] = 0;
        }
    }
    __syncthreads();
    // the leaders (pd >= 0) replace their pair index by the expert's index d (testing first[v] == p again here would race
    // with leaders that already wrote a d equal to some other pair's index)
    for (int p = tid; p < P; p += nt) {
        if (pd[p] >= 0) first[ids[p]] = pd[p];
    }
    __syncthreads();
    for (int p = tid; p < P; p += nt) {
        const int v = ids[p];
        if (v >= 0 && v < E) {
            const int d = first[v];
            pd[p] = d;
            ps[p] = atomicAdd(&cnt[d], 1);
        }
    }
    __syncthreads();
    const int n = *nd;
    if (tid < IK_WARP) {
        int carry = 0;
        if (tid == 0) doff[0] = 0;
        for (int b = 0; b < n; b += IK_WARP) {
            int v = b + tid < n ? cnt[b + tid] : 0;
#pragma unroll
            for (int s = 1; s < IK_WARP; s <<= 1) {
                const int u = __shfl_up(v, s, IK_WARP);
                if (tid >= s) v += u;
            }
            if (b + tid < n) doff[b + tid + 1] = v + carry;
            carry += __shfl(v, IK_WARP - 1, IK_WARP);
        }
    }
    __syncthreads();
    for (int p = tid; p < P; p += nt) {
        const int v = ids[p];
        if (v >= 0 && v < E) dpair[doff[pd[p]] + ps[p]] = p;
    }
    __syncthreads();
}

template <int SZ>
struct SmemGroups {
    int first[sz_em(SZ)];
    int nd;
    int dexp[sz_dp(SZ)];
    int doff[sz_dp(SZ) + 1];
    int dpair[sz_dp(SZ)];
    int pd[sz_dp(SZ)];
    int ps[sz_dp(SZ)];
    int cnt[sz_dp(SZ)];
    unsigned vmask;  // group_wave: bit p = pair p is valid (the down kernel's token sum reads it instead of the ids)
};

// ---- one-wave grouping (class 0, at most kWaveGroupPairs pairs: decode; lane ink-hip-dec2) ------------------------------------------
// The class-0 tables of group_pairs, written by wave 0 alone with one barrier for the block instead of six (and no table
// initialisation): lane p holds pair p; a pair leads its expert when no earlier pair has its id; the experts are numbered by
// their first pair and an expert's pairs come in pair order. Duplicate and invalid ids are handled as group_pairs does (an
// invalid pair is in no group). Like the atomics' order, this decides only which wave runs what, never a bit of the result.
// Taken up to 16 pairs (T <= 2 at top-8): the grouping probe of the bench (window 2, #12002) went 11.8 -> 9.7 us at T = 1 and
// 11.9 -> 11.3 at T = 2, but 11.8 -> 15.4 at T = 4 (32 pairs; the call itself was flat there).
constexpr int kWaveGroupPairs = 16;
template <int SZ>
IK_DEVI void group_wave(const MoeArgs& a, SmemGroups<SZ>& sg) {
    if (threadIdx.x < IK_WARP) {
        const int lane = threadIdx.x, P = a.T * a.K;
        const int v = lane < P ? a.ids[lane] : -1;
        const bool ok = lane < P && v >= 0 && v < a.E;
        const int id = ok ? v : -1;
        int lead = lane, rk = 0;  // the first pair of this pair's expert; the earlier pairs of the same expert
#pragma nounroll
        for (int j = 0; j < P; ++j) {
            const int u = __builtin_amdgcn_readlane(id, j);
            if (j < lane && u == id) {
                if (lead == lane) lead = j;
                ++rk;
            }
        }
        const unsigned vm = (unsigned)__ballot(ok);
        const unsigned lm = (unsigned)__ballot(ok && lead == lane);
        const int ld = ok ? lead : IK_WARP;  // invalid pairs after every valid one
        int before = 0;                      // the valid pairs of the experts whose first pair precedes this pair's expert's
#pragma nounroll
        for (int j = 0; j < P; ++j) before += __builtin_amdgcn_readlane(ld, j) < ld ? 1 : 0;
        if (ok) {
            sg.dpair[before + rk] = lane;
            if (lead == lane) {
                const int d = __popc(lm & ((1u << lane) - 1u));
                sg.dexp[d] = id;
                sg.doff[d] = before;
            }
        }
        if (lane == 0) {
            const int nd = __popc(lm);
            sg.nd = nd;
            sg.doff[nd] = __popc(vm);
            sg.vmask = vm;
        }
    }
    __syncthreads();
}

// table view the kernels read
struct Groups {
    int nd;
    const int* dexp;
    const int* doff;
    const int* dpair;
};

template <int SZ>
IK_DEVI Groups get_groups(const MoeArgs& a, SmemGroups<SZ>& sg) {
    if (a.table) {
        return Groups{a.t_exp[a.E], a.t_exp, a.t_off, a.t_pair};
    }
    group_pairs(a.ids, a.T * a.K, a.E, sg.first, &sg.nd, sg.dexp, sg.doff, sg.dpair, sg.pd, sg.ps, sg.cnt);
    return Groups{sg.nd, sg.dexp, sg.doff, sg.dpair};
}

#define IK_MEMB __device__ __forceinline__
// The kernels' view of the groups: `nd` distinct experts; select(d) -> the expert and its pair count; pair(i) -> its i-th pair.
template <int SZ>
struct Grp {
    Groups g;
    int nd, o0;
    IK_MEMB void init(const MoeArgs& a, SmemGroups<SZ>& sg) {
        if constexpr (SZ == 0) {
            if (a.wg) {
                group_wave(a, sg);
                g = Groups{sg.nd, sg.dexp, sg.doff, sg.dpair};
                nd = g.nd;
                return;
            }
        }
        g = get_groups(a, sg);
        nd = g.nd;
    }
    IK_MEMB int select(int d, int& cnt) {
        o0 = g.doff[d];
        cnt = g.doff[d + 1] - o0;
        return g.dexp[d];
    }
    IK_MEMB int pair(int i) const { return g.dpair[o0 + i]; }
};

// ---- grouping inside the wave (class 4, at most 32 pairs) ---------------------------------------------------------------
// Lane p holds pair p. A pair leads its expert when no earlier pair has the same id; the leaders' ranks number the distinct
// experts. Every answer is a wave reduction, so all lanes of the wave get the same value. As with the tables, the grouping only
// decides which wave runs what: here the experts are numbered by their first pair and an expert's pairs come in pair order.
IK_DEVI int wave_max(int v) {
#pragma unroll
    for (int o = IK_WARP / 2; o > 0; o >>= 1) {
        const int u = __shfl_xor(v, o, IK_WARP);
        v = u > v ? u : v;
    }
    return v;
}
// how many lower lanes have the flag set
IK_DEVI int wave_rank(bool flag, int lane) {
    int v = flag ? 1 : 0;
    const int own = v;
#pragma unroll
    for (int sft = 1; sft < IK_WARP; sft <<= 1) {
        const int u = __shfl_up(v, sft, IK_WARP);
        if (lane >= sft) v += u;
    }
    return v - own;
}
template <>
struct Grp<4> {
    int nd, lane, id, dl, mrank;
    bool member;
    IK_MEMB void init(const MoeArgs& a, SmemGroups<4>&) {
        lane = threadIdx.x % IK_WARP;
        const int P = a.T * a.K;
        id = -1;
        bool lead = false;
#pragma nounroll
        for (int j = 0; j < P; ++j) {
            const int v = a.ids[j];
            const bool ok = v >= 0 && v < a.E;
            if (j == lane && ok) {
                id = v;
                lead = true;
            }
        }
#pragma nounroll
        for (int j = 0; j < P; ++j) {
            if (j < lane && a.ids[j] == id) lead = false;
        }
        const int r = wave_rank(lead, lane);
        dl = lead ? r : -1;
        nd = wave_max(dl + 1);
    }
    IK_MEMB int select(int d, int& cnt) {
        const int e = wave_max(dl == d ? id : -1);
        member = id >= 0 && id == e;
        mrank = wave_rank(member, lane);
        cnt = wave_max(member ? mrank + 1 : 0);
        return e;
    }
    IK_MEMB int pair(int i) const { return wave_max(member && mrank == i ? lane : -1); }
};

// ---- weight chunks: 32 consecutive k of one row ----------------------------------------------------------------------
// NVFP4: 16 bytes of e2m1 pairs + two e4m3 block scales; dot with 32 bf16 of x: two chains of 16 fma (one per block of 16),
// each folded into the accumulator times its scale.
// MXFP8: 32 e4m3 bytes and the E8M0 power of two; dot: one chain of 32 fma in f32, times the scale.
// Raw: a chunk as loaded (requested ahead of its use); WChunk: decoded to f32.
template <int FMT>
struct Raw;
template <>
struct Raw<FMT_NV> {
    uint4 v;
    uint32_t sb;
};
template <>
struct Raw<FMT_MX> {
    uint4 v0, v1;
    uint32_t e;
};

struct WChunk {
    float w[32];
    float s0, s1;
};

IK_DEVI void load_raw(const uint8_t* __restrict__ wrow, const uint8_t* __restrict__ srow, int ch, bool ok, Raw<FMT_NV>& r) {
    if (ok) {
        r.v = ik_ldcs(reinterpret_cast<const uint4*>(wrow) + ch);
        r.sb = *reinterpret_cast<const unsigned short*>(srow + 2 * ch);
    } else {
        r.v = make_uint4(0, 0, 0, 0);
        r.sb = 0;
    }
}
IK_DEVI void load_raw(const uint8_t* __restrict__ wrow, const uint8_t* __restrict__ srow, int ch, bool ok, Raw<FMT_MX>& r) {
    if (ok) {
        r.v0 = ik_ldcs(reinterpret_cast<const uint4*>(wrow) + 2 * ch);
        r.v1 = ik_ldcs(reinterpret_cast<const uint4*>(wrow) + 2 * ch + 1);
        r.e = srow[ch];
    } else {
        r.v0 = r.v1 = make_uint4(0, 0, 0, 0);
        r.e = 0;
    }
}

typedef float ink_v2f __attribute__((ext_vector_type(2)));

// 4 e4m3 bytes of one word -> 4 floats (exact)
IK_DEVI void e4m3x4_to_f32(uint32_t w, float* f) {
#if __has_builtin(__builtin_amdgcn_cvt_pk_f32_fp8)
    const ink_v2f lo = __builtin_amdgcn_cvt_pk_f32_fp8((int)w, false), hi = __builtin_amdgcn_cvt_pk_f32_fp8((int)w, true);
    f[0] = lo[0], f[1] = lo[1], f[2] = hi[0], f[3] = hi[1];
#else
#pragma unroll
    for (int j = 0; j < 4; ++j) f[j] = ik_e4m3_to_f((w >> (8 * j)) & 0xffu);
#endif
}

// 8 e2m1 codes of one word (element j = nibble j) -> 8 floats, each the code's value * 2^-6 (exact; moe_wmma_r7.h): sign and the
// 3 magnitude bits moved into an e4m3 byte (s << 7) | (mag << 2), even elements in `ev`, odd ones in `od`
IK_DEVI void fp4x8_to_f32_s6(uint32_t c, float* f) {
    const uint32_t ev = ((c << 2) & 0x1C1C1C1Cu) | ((c << 4) & 0x80808080u);
    const uint32_t od = ((c >> 2) & 0x1C1C1C1Cu) | (c & 0x80808080u);
    float e[4], o[4];
    e4m3x4_to_f32(ev, e);
    e4m3x4_to_f32(od, o);
#pragma unroll
    for (int i = 0; i < 4; ++i) f[2 * i] = e[i], f[2 * i + 1] = o[i];
}

IK_DEVI void decode(const Raw<FMT_NV>& r, WChunk& c) {
    fp4x8_to_f32_s6(r.v.x, c.w);
    fp4x8_to_f32_s6(r.v.y, c.w + 8);
    fp4x8_to_f32_s6(r.v.z, c.w + 16);
    fp4x8_to_f32_s6(r.v.w, c.w + 24);
    c.s0 = ik_e4m3_to_f(r.sb & 0xffu) * 64.f;  // exact: |scale| <= 448
    c.s1 = ik_e4m3_to_f((r.sb >> 8) & 0xffu) * 64.f;
}
IK_DEVI void decode(const Raw<FMT_MX>& r, WChunk& c) {
    e4m3x4_to_f32(r.v0.x, c.w);
    e4m3x4_to_f32(r.v0.y, c.w + 4);
    e4m3x4_to_f32(r.v0.z, c.w + 8);
    e4m3x4_to_f32(r.v0.w, c.w + 12);
    e4m3x4_to_f32(r.v1.x, c.w + 16);
    e4m3x4_to_f32(r.v1.y, c.w + 20);
    e4m3x4_to_f32(r.v1.z, c.w + 24);
    e4m3x4_to_f32(r.v1.w, c.w + 28);
    const uint32_t e = r.e;
    c.s0 = e == 0 ? __uint_as_float(0x00400000u) : (e == 255 ? __uint_as_float(0x7fc00000u) : __uint_as_float(e << 23));
    c.s1 = 0.f;
}

// x: 32 bf16 as f32
template <int FMT>
IK_DEVI float chunk_dot(const WChunk& c, const float* x, float acc) {
    if (FMT == FMT_NV) {
        float p0 = 0.f, p1 = 0.f;
#pragma unroll
        for (int i = 0; i < 16; ++i) p0 = fmaf(c.w[i], x[i], p0);
#pragma unroll
        for (int i = 16; i < 32; ++i) p1 = fmaf(c.w[i], x[i], p1);
        acc = fmaf(p0, c.s0, acc);
        return fmaf(p1, c.s1, acc);
    } else {
        float p = 0.f;
#pragma unroll
        for (int i = 0; i < 32; ++i) p = fmaf(c.w[i], x[i], p);
        return fmaf(p, c.s0, acc);
    }
}

IK_DEVI void load_x32(const ik_bf16* __restrict__ p, float* x) {
    const uint4* q = reinterpret_cast<const uint4*>(p);
#pragma unroll
    for (int i = 0; i < 4; ++i) ik_unpack8(ik_ldg(q + i), x + 8 * i);
}

// ---- the lean step: a chunk as 4 units of 8 weights, each decoded and used at once -------------------------------------------
// The same chains in the same order as decode() + chunk_dot() (elements 0..31 ascending, the fold after element 15 and 31 for
// NVFP4, after 31 for MXFP8), so the same bits; only 8 decoded weights per row are live instead of 32, and the block scales come
// from one fp8 conversion.
IK_DEVI void decode_unit(const Raw<FMT_NV>& r, int u, float* f) {
    fp4x8_to_f32_s6(u == 0 ? r.v.x : (u == 1 ? r.v.y : (u == 2 ? r.v.z : r.v.w)), f);
}
IK_DEVI void decode_unit(const Raw<FMT_MX>& r, int u, float* f) {
    const uint4& v = u < 2 ? r.v0 : r.v1;
    e4m3x4_to_f32((u & 1) ? v.z : v.x, f);
    e4m3x4_to_f32((u & 1) ? v.w : v.y, f + 4);
}
IK_DEVI void decode_scales(const Raw<FMT_NV>& r, float* s) {
#if __has_builtin(__builtin_amdgcn_cvt_pk_f32_fp8)
    const ink_v2f v = __builtin_amdgcn_cvt_pk_f32_fp8((int)r.sb, false);
    s[0] = v[0] * 64.f, s[1] = v[1] * 64.f;
#else
    s[0] = ik_e4m3_to_f(r.sb & 0xffu) * 64.f, s[1] = ik_e4m3_to_f((r.sb >> 8) & 0xffu) * 64.f;
#endif
}
IK_DEVI void decode_scales(const Raw<FMT_MX>& r, float* s) {
    const uint32_t e = r.e;
    s[0] = e == 0 ? __uint_as_float(0x00400000u) : (e == 255 ? __uint_as_float(0x7fc00000u) : __uint_as_float(e << 23));
    s[1] = s[0];
}

// one chunk of NR weight rows (gate and up: 2, down: 1) against the same 32 activations of up to NB pairs; rows[b] = the pair's
// activation row, acc[r][b] its accumulators
template <int FMT, int NR, int NB>
IK_DEVI void lean_step(const Raw<FMT>* raw, const ik_bf16* const* rows, int ch, int nb, float (*acc)[NB]) {
    float sc[NR][2];
#pragma unroll
    for (int r = 0; r < NR; ++r) decode_scales(raw[r], sc[r]);
    float p[NR][NB];
#pragma unroll
    for (int r = 0; r < NR; ++r)
#pragma unroll
        for (int b = 0; b < NB; ++b) p[r][b] = 0.f;
#pragma unroll
    for (int u = 0; u < 4; ++u) {
        float w[NR][8];
#pragma unroll
        for (int r = 0; r < NR; ++r) decode_unit(raw[r], u, w[r]);
#pragma unroll
        for (int b = 0; b < NB; ++b) {
            if (b >= nb) break;
            float x[8];
            ik_unpack8(ik_ldg(reinterpret_cast<const uint4*>(rows[b] + ch * 32) + u), x);
#pragma unroll
            for (int r = 0; r < NR; ++r)
#pragma unroll
                for (int i = 0; i < 8; ++i) p[r][b] = fmaf(w[r][i], x[i], p[r][b]);
        }
        if (FMT == FMT_NV ? (u & 1) : u == 3) {
#pragma unroll
            for (int b = 0; b < NB; ++b) {
                if (b >= nb) break;
#pragma unroll
                for (int r = 0; r < NR; ++r) {
                    acc[r][b] = fmaf(p[r][b], sc[r][FMT == FMT_NV ? (u >> 1) : 0], acc[r][b]);
                    p[r][b] = 0.f;
                }
            }
        }
    }
}

// bytes per row of a [rows, k] weight matrix and of its scales
template <int FMT>
IK_DEVI int64_t row_bytes(int k) { return FMT == FMT_NV ? k / 2 : k; }
template <int FMT>
IK_DEVI int64_t srow_bytes(int k) { return FMT == FMT_NV ? k / 16 : k / 32; }

// ---- gate + up ---------------------------------------------------------------------------------------------------------
template <int FMT, int ACT, int PRE, int NB, bool LEAN, int SZ>
__global__ void __launch_bounds__(kThreads) ink_moe_gu_kernel(MoeArgs a) {
    constexpr int kNB = NB;
    __shared__ SmemGroups<SZ> sg;
    if (a.probe == 3) return;
    Grp<SZ> gr;
    gr.init(a, sg);
    if (a.probe == 4) return;
    const int warp = threadIdx.x / IK_WARP, lane = threadIdx.x % IK_WARP;
    const int q = lane >> 3, sub = lane & 7;
    const int I = a.I, H = a.H;
    const int r = blockIdx.x * kGuRows + q;
    const bool on = r < I;
    const int rr = on ? r : I - 1;
    const int nch = H / 32;
    const int nit = (nch + 7) / 8;  // steps of a lane: its chunks sub, sub + 8, ..
    const int64_t wb = row_bytes<FMT>(H), sb = srow_bytes<FMT>(H);
    for (int d = warp; d < gr.nd; d += kWarpsB) {
        int cnt;
        const int e = gr.select(d, cnt);
        const uint8_t* wg = a.guw + ((int64_t)e * 2 * I + rr) * wb;
        const uint8_t* wu = a.guw + ((int64_t)e * 2 * I + I + rr) * wb;
        const uint8_t* sgp = a.gus + ((int64_t)e * 2 * I + rr) * sb;
        const uint8_t* sup = a.gus + ((int64_t)e * 2 * I + I + rr) * sb;
        for (int pb = 0; pb < cnt; pb += kNB) {
            const int nb = min(kNB, cnt - pb);
            int pp[kNB];
            const ik_bf16* xr[kNB];  // the pairs' token rows
            float acc[2][kNB];       // [0] gate, [1] up
#pragma unroll
            for (int b = 0; b < kNB; ++b) {
                pp[b] = gr.pair(pb + (b < nb ? b : 0));
                xr[b] = a.x + (int64_t)(pp[b] / a.K) * H;
                acc[0][b] = acc[1][b] = 0.f;
            }
            for (int it0 = 0; it0 < nit; it0 += PRE) {
                Raw<FMT> rw[PRE][2];
#pragma unroll
                for (int j = 0; j < PRE; ++j) {
                    const int ch = sub + 8 * (it0 + j);
                    load_raw(wg, sgp, ch, ch < nch, rw[j][0]);
                    load_raw(wu, sup, ch, ch < nch, rw[j][1]);
                }
#pragma unroll
                for (int j = 0; j < PRE; ++j) {
                    const int ch = sub + 8 * (it0 + j);
                    if (ch >= nch) continue;
                    if (LEAN) {
                        lean_step<FMT, 2, NB>(rw[j], xr, ch, nb, acc);
                    } else {
                        WChunk cg, cu;
                        decode(rw[j][0], cg);
                        decode(rw[j][1], cu);
#pragma unroll
                        for (int b = 0; b < kNB; ++b) {
                            if (b >= nb) break;
                            float xv[32];
                            load_x32(xr[b] + ch * 32, xv);
                            acc[0][b] = chunk_dot<FMT>(cg, xv, acc[0][b]);
                            acc[1][b] = chunk_dot<FMT>(cu, xv, acc[1][b]);
                        }
                    }
                }
            }
#pragma unroll
            for (int b = 0; b < kNB; ++b) {
                if (b >= nb) break;
                float sgv = acc[0][b], suv = acc[1][b];
#pragma unroll
                for (int o = 4; o > 0; o >>= 1) {
                    sgv += __shfl_xor(sgv, o, IK_WARP);
                    suv += __shfl_xor(suv, o, IK_WARP);
                }
                if (sub == 0 && on) {
                    float g, u;
                    if (FMT == FMT_NV) {
                        g = ink_rbf(sgv * a.gs[e * 3 + 0]);
                        u = ink_rbf(suv * a.gs[e * 3 + 1]);
                    } else {
                        g = ink_rbf(sgv);
                        u = ink_rbf(suv);
                    }
                    a.h[(int64_t)pp[b] * I + r] = ik_f2b(act_h<ACT>(g, u, a.wts[pp[b]], a.gc, a.uc));
                }
            }
        }
    }
}

// ---- down + the token sum ----------------------------------------------------------------------------------------------
template <int FMT, int ACT, typename OutT, int PRE, int NB, bool LEAN, int SZ>
__global__ void __launch_bounds__(kThreads) ink_moe_dn_kernel(MoeArgs a) {
    constexpr int kNB = NB;
    __shared__ SmemGroups<SZ> sg;
    __shared__ float term[sz_terms(SZ) * kDnRows];
    if (a.probe == 3) return;
    Grp<SZ> gr;
    gr.init(a, sg);
    if (a.probe == 4) return;
    const int warp = threadIdx.x / IK_WARP, lane = threadIdx.x % IK_WARP;
    const int q = lane >> 2, sub = lane & 3;
    const int I = a.I, H = a.H;
    const int r0 = blockIdx.x * kDnRows;
    const int r = r0 + q;
    const bool on = r < H;
    const int rr = on ? r : H - 1;
    const int nch = I / 32;
    const int nit = (nch + 3) / 4;
    const int64_t wb = row_bytes<FMT>(I), sb = srow_bytes<FMT>(I);
    for (int d = warp; d < gr.nd; d += kWarpsB) {
        int cnt;
        const int e = gr.select(d, cnt);
        const uint8_t* wd = a.dnw + ((int64_t)e * H + rr) * wb;
        const uint8_t* sd = a.dns + ((int64_t)e * H + rr) * sb;
        for (int pb = 0; pb < cnt; pb += kNB) {
            const int nb = min(kNB, cnt - pb);
            int pp[kNB];
            const ik_bf16* hr[kNB];  // the pairs' h rows
            float acc[1][kNB];
#pragma unroll
            for (int b = 0; b < kNB; ++b) {
                pp[b] = gr.pair(pb + (b < nb ? b : 0));
                hr[b] = a.h + (int64_t)pp[b] * I;
                acc[0][b] = 0.f;
            }
            for (int it0 = 0; it0 < nit; it0 += PRE) {
                Raw<FMT> rw[PRE][1];
#pragma unroll
                for (int j = 0; j < PRE; ++j) {
                    const int ch = sub + 4 * (it0 + j);
                    load_raw(wd, sd, ch, ch < nch, rw[j][0]);
                }
#pragma unroll
                for (int j = 0; j < PRE; ++j) {
                    const int ch = sub + 4 * (it0 + j);
                    if (ch >= nch) continue;
                    if (LEAN) {
                        lean_step<FMT, 1, NB>(rw[j], hr, ch, nb, acc);
                    } else {
                        WChunk c;
                        decode(rw[j][0], c);
#pragma unroll
                        for (int b = 0; b < kNB; ++b) {
                            if (b >= nb) break;
                            float hv[32];
                            load_x32(hr[b] + ch * 32, hv);
                            acc[0][b] = chunk_dot<FMT>(c, hv, acc[0][b]);
                        }
                    }
                }
            }
#pragma unroll
            for (int b = 0; b < kNB; ++b) {
                if (b >= nb) break;
                float s = acc[0][b];
                s += __shfl_xor(s, 2, IK_WARP);
                s += __shfl_xor(s, 1, IK_WARP);
                if (sub == 0) {
                    const float yv = ink_rbf(FMT == FMT_NV ? s * a.gs[e * 3 + 2] : s);
                    const float tv = ACT == ACT_SITU ? yv : ink_rbf(yv * a.wts[pp[b]]);
                    if (a.table) {
                        if (on) a.y[(int64_t)pp[b] * H + r] = ik_f2b(tv);
                    } else {
                        term[pp[b] * kDnRows + q] = tv;
                    }
                }
            }
        }
    }
    if (a.table) return;
    __syncthreads();
    // the one-wave grouping left the valid pairs in shared memory: the token sum reads them there, not the ids (the same terms
    // in the same slot order)
    const unsigned vm = (SZ == 0 && a.wg) ? sg.vmask : 0u;
    for (int i = threadIdx.x; i < a.T * kDnRows; i += blockDim.x) {
        const int t = i / kDnRows, qq = i % kDnRows;
        if (r0 + qq >= H) continue;
        float tot = 0.f;
#pragma nounroll
        for (int k = 0; k < a.K; ++k) {  // (not unrolled: its registers would set the kernel's VGPR count)
            bool ok;
            if (SZ == 0 && a.wg) {
                ok = ((vm >> (t * a.K + k)) & 1u) != 0;
            } else {
                const int v = a.ids[t * a.K + k];
                ok = v >= 0 && v < a.E;
            }
            if (ok) tot += term[(t * a.K + k) * kDnRows + qq];
        }
        ik_store(reinterpret_cast<OutT*>(a.out) + (int64_t)t * H + r0 + qq, tot);
    }
}

// the table path: one block builds the pair table in scratch (temporaries in the h region, which gate_up writes later)
__global__ void __launch_bounds__(1024) ink_moe_group_kernel(MoeArgs a) {
    __shared__ int first[kEMax];
    const int P = a.T * a.K;
    int* tmp = reinterpret_cast<int*>(a.h);
    group_pairs(a.ids, P, a.E, first, a.t_exp + a.E, a.t_exp, a.t_off, a.t_pair, tmp, tmp + P, tmp + 2 * P);
}

// the table path's token sum: out[t, r] = rA(sum_k y[t*K + k, r]) over the valid slots, in slot order
template <typename OutT>
__global__ void __launch_bounds__(256) ink_moe_gather_kernel(MoeArgs a) {
    const int t = blockIdx.y;
    for (int r = blockIdx.x * blockDim.x + threadIdx.x; r < a.H; r += gridDim.x * blockDim.x) {
        float tot = 0.f;
        for (int k = 0; k < a.K; ++k) {
            const int v = a.ids[t * a.K + k];
            if (v >= 0 && v < a.E) tot += ik_b2f(a.y[(int64_t)(t * a.K + k) * a.H + r]);
        }
        ik_store(reinterpret_cast<OutT*>(a.out) + (int64_t)t * a.H + r, tot);
    }
}

int64_t rnd(int64_t v) { return (v + 255) & ~(int64_t)255; }

// The load window (weights of this many lane steps requested before the first is used) of gate_up and of down:
// INTARSIA_HIP_INK_MOE_PRE = "v" (both) or "g:d", each one of kPres; unset or anything else: the format's default. Every value
// gives the same bits. INTARSIA_HIP_INK_MOE_LEAN=0: the first form of the step (32 decoded weights per row live, 8 pairs per
// batch), bitwise the default; kept for one round as the reference.
struct PreCfg {
    int gu, dn;
    int gu_nb, dn_nb;  // pairs per accumulator batch below the pair table (4 | 8)
    bool dn_first;     // down takes the first form (below the pair table)
};
bool pre_ok(int v) { return v == 1 || v == 3 || v == 6 || v == 9; }
PreCfg moe_pre(int fmt, int T, int64_t P, int E) {
    // Windows 3 to 5 (results doc section 3), below the pair table. Sparse calls (at most 2 pairs per expert on average: decode,
    // T <= 32 at top-8 of 128): 4 pairs per batch; down with every lane step requested up front (9); NVFP4 gate_up 6, and 9 only
    // from 8 rows (202 VGPRs: fewer waves resident, which costs at T = 1); MXFP8 gate_up 1 (it spills at 9).
    // Denser calls: 8 pairs per batch (an expert's rows are decoded once per 8 pairs), NVFP4 gate_up window 3; and from 4 pairs
    // per expert down takes the first form, which is the faster one there (T = 64: 533 against 649 us).
    const bool dense = P > 2 * (int64_t)E, dense4 = P >= 4 * (int64_t)E;
    PreCfg c;
    if (fmt == FMT_NV) c = dense ? PreCfg{3, 9, 8, 8, dense4} : PreCfg{T >= 8 ? 9 : 6, 9, 4, 4, false};
    else c = dense ? PreCfg{1, 9, 8, 8, dense4} : PreCfg{1, 9, 4, 4, false};
    int g = g_pre_gu, d = g_pre_dn;
    if (g < 0 || d < 0) {
        const char* e = getenv("INTARSIA_HIP_INK_MOE_PRE");
        if (e && e[0]) {
            const int eg = atoi(e);
            const char* colon = strchr(e, ':');
            const int ed = colon ? atoi(colon + 1) : eg;
            if (g < 0) g = eg;
            if (d < 0) d = ed;
        }
    }
    if (pre_ok(g)) c.gu = g;
    if (pre_ok(d)) c.dn = d;
    if (g_nb == 4 || g_nb == 8) c.gu_nb = c.dn_nb = g_nb;
    if (g_nb == 4 || g_nb == 8 || g_lean == 1) c.dn_first = false;  // tests / benchmarks: a forced lean form
    return c;
}
// INTARSIA_HIP_INK_DEC_FAST (lane ink-hip-dec2): bit mask of the decode latency forms, default all on; bit 1 = the one-wave grouping
int dec_fast_env() {
    static int v = -1;
    if (v < 0) {
        const char* e = getenv("INTARSIA_HIP_INK_DEC_FAST");
        v = e && e[0] >= '0' && e[0] <= '9' ? atoi(e) : 7;
    }
    return v;
}
bool moe_dec_wave() { return g_dec >= 0 ? g_dec != 0 : (dec_fast_env() & 1) != 0; }
bool moe_lean() {
    if (g_lean >= 0) return g_lean != 0;
    const char* e = getenv("INTARSIA_HIP_INK_MOE_LEAN");
    return !(e && e[0] == '0');
}

template <int FMT, int ACT, int PRE, int NB, bool LEAN, int SZ>
void launch_gu1(const MoeArgs& a, hipStream_t st) {
    ink_moe_gu_kernel<FMT, ACT, PRE, NB, LEAN, SZ><<<(a.I + kGuRows - 1) / kGuRows, kThreads, 0, st>>>(a);
}
template <int FMT, int ACT, int PRE, int NB, bool LEAN, int SZ>
void launch_dn1(const MoeArgs& a, int out_f32, hipStream_t st) {
    const unsigned g = (unsigned)((a.H + kDnRows - 1) / kDnRows);
    if (out_f32) ink_moe_dn_kernel<FMT, ACT, float, PRE, NB, LEAN, SZ><<<g, kThreads, 0, st>>>(a);
    else ink_moe_dn_kernel<FMT, ACT, ik_bf16, PRE, NB, LEAN, SZ><<<g, kThreads, 0, st>>>(a);
}

// the lean form: every window in the call's size class (4 = grouped in the wave, 0..2 LDS tables) with NB 4 below the pair table, window 3 with NB 8 on the
// pair-table path (class 3); the first form (the reference of window 1): NB 8, window 3, class 2
#define INK_PRE_SZN(CALL, N, Z)                 \
    switch (pre) {                              \
        case 1: CALL(1, N, true, Z); break;     \
        case 3: CALL(3, N, true, Z); break;     \
        case 9: CALL(9, N, true, Z); break;     \
        default: CALL(6, N, true, Z); break;    \
    }
#define INK_PRE_SZ(CALL, Z)                     \
    if (nb == 8) {                              \
        INK_PRE_SZN(CALL, 8, Z)                 \
    } else {                                    \
        INK_PRE_SZN(CALL, 4, Z)                 \
    }
#define INK_PRE_SWITCH(CALL)                    \
    if (!lean) {                                \
        CALL(3, 8, false, 2);                   \
    } else if (a.table) {                       \
        CALL(3, 8, true, 3);                    \
    } else if (sz == 4) {                       \
        INK_PRE_SZ(CALL, 4)                     \
    } else if (sz == 0) {                       \
        INK_PRE_SZ(CALL, 0)                     \
    } else if (sz == 1) {                       \
        INK_PRE_SZ(CALL, 1)                     \
    } else {                                    \
        INK_PRE_SZ(CALL, 2)                     \
    }

template <int FMT, int ACT>
int launch_pair(const MoeArgs& a0, int out_f32, hipStream_t st) {
    const int64_t P = (int64_t)a0.T * a0.K;
    const PreCfg pc = moe_pre(FMT, a0.T, P, a0.E);
    int sz = 2;
    for (int z = 1; z >= 0; --z)
        if (P <= sz_dp(z) && a0.E <= sz_em(z)) sz = z;
    if (g_sz == 1 || g_sz == 2) sz = sz < g_sz ? g_sz : sz;
    if (g_sz == 4 && P <= kWavePairs) sz = 4;  // opt-in: slower than the tables at T = 1, 2, 4 (window 5)
    MoeArgs a = a0;
    a.wg = (sz == 0 && !a.table && P <= kWaveGroupPairs && moe_dec_wave()) ? 1 : 0;
    if (g_probe != 2) {
        const int pre = pc.gu, nb = pc.gu_nb;
        const bool lean = moe_lean();
#define INK_GU(P, N, L, Z) launch_gu1<FMT, ACT, P, N, L, Z>(a, st)
        INK_PRE_SWITCH(INK_GU)
#undef INK_GU
        IK_TRY(ik_launch_status());
    }
    if (g_probe != 1) {
        const int pre = pc.dn, nb = pc.dn_nb;
        const bool lean = moe_lean() && !(pc.dn_first && !a.table);
#define INK_DN(P, N, L, Z) launch_dn1<FMT, ACT, P, N, L, Z>(a, out_f32, st)
        INK_PRE_SWITCH(INK_DN)
#undef INK_DN
        IK_TRY(ik_launch_status());
    }
    return IK_OK;
}

// ---- tensor-core prefill (S2): the pair-table path of the ordinary names -----------------------------------------------
// bf16 WMMA over the pair table, the scheme of moe_wmma.h (moe_wmma_kernel: weights dequantised straight into the A fragments
// in registers, the activation tile through shared memory, kMwRing weight blocks in flight per lane; see its header for the
// fragment layout) with this op's epilogues, both formats and this file's pair table (indexed by distinct expert). A weight
// is exact in bf16 (e2m1 x e4m3 block scale: 6 significant bits; e4m3 x 2^k: 4), so only the f32 accumulation order differs
// from the decode kernels: the result is within the op's tolerance of them, not their bits, and a row's bits do not depend on
// the other rows of the call (each output accumulates over k in the same order whatever shares its tile). The `_rows` twins
// never take this path. Grid (pair tile, feature tile, expert slot): slots past the distinct count and pair tiles past an
// expert's pairs exit; a block steps through further pair tiles by gridDim.x.
IK_DEVI uint4 pack_trunc8(const float* f, float sc) {
    uint32_t u[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) u[j] = __float_as_uint(f[j] * sc);
    uint4 r;
    r.x = __builtin_amdgcn_perm(u[1], u[0], 0x07060302u);
    r.y = __builtin_amdgcn_perm(u[3], u[2], 0x07060302u);
    r.z = __builtin_amdgcn_perm(u[5], u[4], 0x07060302u);
    r.w = __builtin_amdgcn_perm(u[7], u[6], 0x07060302u);
    return r;
}

template <int FMT, int ACT, bool GATEUP, int PAIRS>
__global__ void __launch_bounds__(kMwThreads) ink_moe_wmma_kernel(MoeArgs a) {
    constexpr int FEAT = GATEUP ? kMwFeatGateUp : kMwFeatDown;
    constexpr int J = PAIRS / 16;   // 16-pair B fragments per weight fragment
    constexpr int XR = PAIRS / 32;  // activation rows each thread stages
    __shared__ __align__(16) uint16_t sx[2][PAIRS * kMwLD];
    __shared__ __align__(16) float sred[GATEUP ? 4 * 32 * J * 8 : 1];  // up accumulators -> gate waves

    const int slot = (int)blockIdx.z;
    if (slot >= a.t_exp[a.E]) return;  // block-uniform
    const int e = a.t_exp[slot];
    const int begin = a.t_off[slot], cnt = a.t_off[slot + 1] - begin;
    const int Kd = GATEUP ? a.H : a.I, Nd = GATEUP ? a.I : a.H;
    const ik_bf16* src = GATEUP ? a.x : a.h;
    ik_bf16* dst = GATEUP ? a.h : a.y;
    const int n0 = ((int)blockIdx.y + a.y0) * FEAT;  // y0: first feature tile of a piece (INTARSIA_HIP_INK_PF2)
    const int tid = threadIdx.x, wave = tid >> 5, lane = tid & 31;
    const int lr = lane & 15, h = lane >> 4;
    const int part = GATEUP ? (wave >> 2) : 0;  // gate_up: 0 = gate rows, 1 = up rows
    const int fw = GATEUP ? (wave & 3) : wave;  // 16-feature slice of the tile
    float g0 = 1.f, g1 = 1.f;
    if (FMT == FMT_NV) {
        g0 = a.gs[e * 3 + (GATEUP ? 0 : 2)];
        g1 = GATEUP ? a.gs[e * 3 + 1] : 1.f;
    }
    const int nkb = Kd / kMwKB;

    // this lane's weight row and its half (32 elements = one chunk of this file's load_raw) of every 64-k block
    const int n = n0 + fw * 16 + lr;
    const bool wok = n < Nd;
    const int64_t wrow = (GATEUP ? ((int64_t)e * 2 + part) * Nd : (int64_t)e * Nd) + (wok ? n : 0);
    const uint8_t* wp = (GATEUP ? a.guw : a.dnw) + wrow * row_bytes<FMT>(Kd);
    const uint8_t* sp = (GATEUP ? a.gus : a.dns) + wrow * srow_bytes<FMT>(Kd);
    auto loadW = [&](int kb) {
        Raw<FMT> r;
        load_raw(wp, sp, 2 * kb + h, wok, r);
        return r;
    };

    for (int pt = (int)blockIdx.x; pt * PAIRS < cnt; pt += (int)gridDim.x) {
        // this thread's activation rows (pairs xr + 32 * r) and 8-element column chunk of the 64-wide tile
        const int xr = tid >> 3, xch = tid & 7;
        const ik_bf16* xrow[XR];
#pragma unroll
        for (int r = 0; r < XR; ++r) {
            xrow[r] = nullptr;
            const int p_idx = pt * PAIRS + xr + 32 * r;
            if (p_idx < cnt) {
                const int pair = a.t_pair[begin + p_idx];
                xrow[r] = src + (int64_t)(GATEUP ? pair / a.K : pair) * Kd;  // token row (gate_up) / own row (down)
            }
        }
        uint4 rx[XR];
        auto loadX = [&](int kb) {
#pragma unroll
            for (int r = 0; r < XR; ++r)
                rx[r] = xrow[r] ? *reinterpret_cast<const uint4*>(xrow[r] + kb * kMwKB + xch * 8) : make_uint4(0, 0, 0, 0);
        };
        auto storeX = [&](int buf) {
#pragma unroll
            for (int r = 0; r < XR; ++r) *reinterpret_cast<uint4*>(&sx[buf][(xr + 32 * r) * kMwLD + xch * 8]) = rx[r];
        };

        ik_mw_v8f acc[J];
#pragma unroll
        for (int j = 0; j < J; ++j)
#pragma unroll
            for (int i = 0; i < 8; ++i) acc[j][i] = 0.f;

        auto compute = [&](const Raw<FMT>& W, int buf) {
            float sc[2];
            decode_scales(W, sc);
#pragma unroll
            for (int st = 0; st < 4; ++st) {
                float f[8];
                decode_unit(W, st, f);
                const uint4 af = pack_trunc8(f, sc[FMT == FMT_NV ? (st >> 1) : 0]);
#pragma unroll
                for (int j = 0; j < J; ++j) {
                    const uint4 bf = *reinterpret_cast<const uint4*>(&sx[buf][(j * 16 + lr) * kMwLD + h * 32 + st * 8]);
                    acc[j] = ik_mw_wmma(af, bf, acc[j]);
                }
            }
        };

        __syncthreads();  // the previous pair tile's last reads of sx / sred are done
        Raw<FMT> W[kMwRing];
        loadX(0);
#pragma unroll
        for (int i = 0; i < kMwRing; ++i)
            if (i < nkb) W[i] = loadW(i);
        storeX(0);
        if (nkb > 1) loadX(1);
        __syncthreads();
        for (int kb0 = 0; kb0 < nkb; kb0 += kMwRing) {
#pragma unroll
            for (int i = 0; i < kMwRing; ++i) {
                const int kb = kb0 + i;
                if (kb >= nkb) break;
                compute(W[i], kb & 1);
                if (kb + kMwRing < nkb) W[i] = loadW(kb + kMwRing);
                if (kb + 1 < nkb) {
                    storeX((kb + 1) & 1);
                    if (kb + 2 < nkb) loadX(kb + 2);
                }
                __syncthreads();
            }
        }

        if (GATEUP) {
            if (part == 1) {
#pragma unroll
                for (int j = 0; j < J; ++j)
#pragma unroll
                    for (int i = 0; i < 8; ++i) sred[(fw * 32 + lane) * (J * 8) + j * 8 + i] = acc[j][i];
            }
            __syncthreads();
            if (part == 1) continue;
        }
        const int f0 = n0 + fw * 16 + h * 8;
#pragma unroll
        for (int j = 0; j < J; ++j) {
            const int pi = pt * PAIRS + j * 16 + lr;
            if (pi >= cnt || f0 >= Nd) continue;
            const int pair = a.t_pair[begin + pi];
            const float wt = a.wts[pair];
            float v[8];
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                if (GATEUP) {
                    const float g = ink_rbf(acc[j][i] * g0);
                    const float u = ink_rbf(sred[(fw * 32 + lane) * (J * 8) + j * 8 + i] * g1);
                    v[i] = act_h<ACT>(g, u, wt, a.gc, a.uc);
                } else {
                    const float yv = ink_rbf(acc[j][i] * g0);
                    v[i] = ACT == ACT_SITU ? yv : ink_rbf(yv * wt);
                }
            }
            *reinterpret_cast<uint4*>(dst + (int64_t)pair * Nd + f0) = ik_pack8(v);
        }
    }
}

// INTARSIA_HIP_INK_MOE_WMMA (not bitwise the decode kernels; default on since the user's decision 2026-11-26):
//   0: the pair-table path stays on the decode kernels (the revert);
//   1 / unset (kWmDefault): the rule: 64-pair tiles from kWmMinPerExpert pairs per expert on average. A block reads and dequantises its expert's
//      rows whatever the tile holds, so a sparse call is faster on the decode kernels (S2 window 1, NVFP4: 8 pairs per expert
//      5.08 ms against 2.92; 16 pairs per expert 5.20 against 5.49);
//   32 | 64: every pair-table call takes the tensor cores with that pair tile (tests / benchmarks).
// The default is the one constant kWmDefault (1 = the rule, 0 = the decode kernels).
constexpr int kWmDefault = 1;
constexpr int kWmPairsRule = 64, kWmMinPerExpert = 16;
int moe_wmma_pairs(int64_t P, int E) {
    int v = g_wmma;
    if (v < 0) {
        const char* e = getenv("INTARSIA_HIP_INK_MOE_WMMA");
        v = e && e[0] ? atoi(e) : kWmDefault;
    }
    if (v == 32 || v == 64) return v;
    if (v == 1) return P >= (int64_t)kWmMinPerExpert * E ? kWmPairsRule : 0;
    return 0;
}

// INTARSIA_HIP_INK_PF2 (lane ink-hip-pf2; read at every launch, every bit keeps the output bits): bit 4 = the tensor-core down in
// pieces of consecutive feature tiles (grid y) under the R9700's ~1 ms dispatch stall line (ik_hip.h), bit 8 = gate_up too (opt-in:
// on the 35B its pieces each paid the hot experts' load-imbalance tail, moe.hip round 8). A block of a piece is the block of the
// one launch with the same feature tile. Bit 16 = more pair-tile blocks per expert (wm_zp). Bits 1 / 2 / 128 are ink_attn.hip's.
// Unset = kPf2Default (1 | 2 | 4 | 16 | 128); 0 = main's forms.
constexpr int kPf2Default = 151;
int pf2_mask() {
    const char* e = getenv("INTARSIA_HIP_INK_PF2");
    return e && e[0] >= '0' && e[0] <= '9' ? atoi(e) : kPf2Default;
}
// pieces of a launch estimated at est_us (as ink_attn.hip's): budget INTARSIA_HIP_INK_PF2_US (default 900 us), trigger 950 us or
// the budget if lower
int pf2_pieces(double est_us, int64_t max_pieces) {
    const char* e = getenv("INTARSIA_HIP_INK_PF2_US");
    const double budget = e && *e ? atof(e) : 900.0;
    const double trigger = budget < 950.0 ? budget : 950.0;
    if (!(budget > 0.0) || est_us <= trigger || max_pieces <= 1) return 1;
    const double n = est_us / budget;
    const int64_t p = n >= (double)max_pieces ? max_pieces : (int64_t)n + ((double)(int64_t)n < n ? 1 : 0);
    return (int)(p < 1 ? 1 : p);
}
// the tensor-core projections' rate in window 1's profile (T = 2048, 64-pair tiles): gate_up 4.25 ms and down 2.04 ms for
// 87.0 / 43.5 GMAC, ~21 GMAC per ms (21e6 MAC per us); MXFP8 ~20
constexpr double kWmMacPerUs = 21e6;
// pair-tile blocks per expert slot (grid x); a block steps through its expert's pair tiles by gridDim.x, so any count gives the
// same bits. Main's rule: twice the average pairs per expert over the tile, 1 to 8 (4 at a 2048-row chunk), so a hot expert's
// blocks ran 2 - 3 tiles one after another while the rest of the card idled. INTARSIA_HIP_INK_PF2 bit 16 (lane ink-hip-pf2):
// eight times the average, 1 to 64 (16 at 2048 rows; window 3, mask 3: 4k 2882 -> 2970 tok/s, 16k 3007 -> 3107).
// INTARSIA_HIP_INK_PF2_ZP (tuning) sets the count.
int wm_zp(int64_t P, int E, int pairs, int mask) {
    const char* e = getenv("INTARSIA_HIP_INK_PF2_ZP");
    if (e && *e && atoi(e) > 0) return atoi(e) > 64 ? 64 : atoi(e);
    const bool wide = (mask & 16) != 0;
    const int k = wide ? 8 : 2, top = wide ? 64 : 8;
    const int zp = (int)((k * P / E + pairs - 1) / pairs);
    return zp < 1 ? 1 : (zp > top ? top : zp);
}

template <int FMT, int ACT>
int launch_wmma(const MoeArgs& a_in, int pairs, hipStream_t st) {
    MoeArgs a = a_in;
    const int64_t P = (int64_t)a.T * a.K;
    const int mask = pf2_mask();
    const int zp = wm_zp(P, a.E, pairs, mask);
    if (g_probe != 2) {
        const int Y = a.I / kMwFeatGateUp;
        const int pieces = (mask & 8) ? pf2_pieces((double)P * 2 * a.I * a.H / kWmMacPerUs, Y) : 1;
        for (int i = 0; i < pieces; ++i) {
            const int y0 = Y * i / pieces, y1 = Y * (i + 1) / pieces;
            a.y0 = y0;
            const dim3 g((unsigned)zp, (unsigned)(y1 - y0), (unsigned)a.E);
            if (pairs == 32) ink_moe_wmma_kernel<FMT, ACT, true, 32><<<g, kMwThreads, 0, st>>>(a);
            else ink_moe_wmma_kernel<FMT, ACT, true, 64><<<g, kMwThreads, 0, st>>>(a);
            IK_TRY(ik_launch_status());
        }
    }
    if (g_probe != 1) {
        const int Y = (a.H + kMwFeatDown - 1) / kMwFeatDown;
        const int pieces = (mask & 4) ? pf2_pieces((double)P * a.H * a.I / kWmMacPerUs, Y) : 1;
        for (int i = 0; i < pieces; ++i) {
            const int y0 = Y * i / pieces, y1 = Y * (i + 1) / pieces;
            a.y0 = y0;
            const dim3 g((unsigned)zp, (unsigned)(y1 - y0), (unsigned)a.E);
            if (pairs == 32) ink_moe_wmma_kernel<FMT, ACT, false, 32><<<g, kMwThreads, 0, st>>>(a);
            else ink_moe_wmma_kernel<FMT, ACT, false, 64><<<g, kMwThreads, 0, st>>>(a);
            IK_TRY(ik_launch_status());
        }
    }
    return IK_OK;
}

template <int FMT, int ACT>
int run(MoeArgs a, int out_f32, bool rows, const ik_scratch* sc, hipStream_t st) {
    const int64_t P = (int64_t)a.T * a.K;
    const int64_t need = rnd(P * 4) + rnd((int64_t)(a.E + 1) * 4) * 2 + rnd(P * a.I * 2) + rnd(P * a.H * 2);
    if (!sc || !sc->data || sc->bytes < need || !ik_aligned16(sc->data)) return IK_ERR_INVALID_ARG;
    char* base = (char*)sc->data;
    a.t_pair = (int*)base;
    a.t_off = (int*)(base + rnd(P * 4));
    a.t_exp = (int*)(base + rnd(P * 4) + rnd((int64_t)(a.E + 1) * 4));
    a.h = (ik_bf16*)(base + rnd(P * 4) + rnd((int64_t)(a.E + 1) * 4) * 2);
    a.y = (ik_bf16*)((char*)a.h + rnd(P * a.I * 2));
    a.table = P > kDecPairs;
    a.probe = g_probe >= 3 ? g_probe : 0;
    if (a.table) {
        if (P * 3 * 4 > P * a.I * 2) return IK_ERR_SHAPE;  // (the grouping's temporaries in h: I >= 6)
        ink_moe_group_kernel<<<1, 1024, 0, st>>>(a);
        IK_TRY(ik_launch_status());
    }
    // the tensor-core prefill: ordinary names only, the pair-table path, reduction dims in 64-k blocks; never the first form
    const int wm = a.table ? moe_wmma_pairs(P, a.E) : 0;
    if (a.table && !rows && wm != 0 && moe_lean() && a.H % kMwKB == 0 && a.I % kMwKB == 0) {
        IK_TRY((launch_wmma<FMT, ACT>(a, wm, st)));
    } else {
        IK_TRY((launch_pair<FMT, ACT>(a, out_f32, st)));
    }
    if (a.table && g_probe != 1) {
        const dim3 gg((unsigned)((a.H + 255) / 256), (unsigned)a.T);
        if (out_f32) ink_moe_gather_kernel<float><<<gg, 256, 0, st>>>(a);
        else ink_moe_gather_kernel<ik_bf16><<<gg, 256, 0, st>>>(a);
        IK_TRY(ik_launch_status());
    }
    return IK_OK;
}

int common_checks(const ik_tensor* x, const ik_tensor* ids, const ik_tensor* wts, ik_tensor* out, int* out_f32) {
    IK_TRY(ik_check_a(x, IK_DTYPE_BF16, 2, 16));
    IK_TRY(ik_check(ids, IK_DTYPE_I32, 2));
    IK_TRY(ik_check(wts, IK_DTYPE_F32, 2));
    if (!out || out->ndim != 2) return IK_ERR_SHAPE;
    if (out->dtype == IK_DTYPE_F32) *out_f32 = 1;
    else if (out->dtype == IK_DTYPE_BF16) *out_f32 = 0;
    else return IK_ERR_DTYPE;
    IK_TRY(ik_check(out, out->dtype, 2));
    const int64_t T = x->shape[0], K = ids->shape[1];
    if (ids->shape[0] != T || wts->shape[0] != T || wts->shape[1] != K || out->shape[0] != T || out->shape[1] != x->shape[1])
        return IK_ERR_SHAPE;
    if (T > 65535 || K < 1 || K > 64) return IK_ERR_SHAPE;
    return IK_OK;
}

int situ_impl(const ik_tensor* x, const ik_tensor* ids, const ik_tensor* wts, const ik_tensor* guw, const ik_tensor* gus,
              const ik_tensor* dnw, const ik_tensor* dns, const ik_tensor* gsc, float gate_cap, float up_cap, int aq, ik_tensor* out,
              const ik_scratch* scratch, void* stream, bool rows) {
    int of = 0;
    IK_TRY(common_checks(x, ids, wts, out, &of));
    IK_TRY(ik_check_a(guw, IK_DTYPE_U8, 3, 16));
    IK_TRY(ik_check_a(gus, IK_DTYPE_FP8_E4M3, 3, 2));
    IK_TRY(ik_check_a(dnw, IK_DTYPE_U8, 3, 16));
    IK_TRY(ik_check_a(dns, IK_DTYPE_FP8_E4M3, 3, 2));
    IK_TRY(ik_check(gsc, IK_DTYPE_F32, 2));
    const int64_t T = x->shape[0], H = x->shape[1], E = guw->shape[0], I = guw->shape[1] / 2;
    if (H % 32 || I % 32 || E < 1 || E > kEMax || guw->shape[1] != 2 * I || guw->shape[2] != H / 2 || gus->shape[0] != E ||
        gus->shape[1] != 2 * I || gus->shape[2] != H / 16 || dnw->shape[0] != E || dnw->shape[1] != H || dnw->shape[2] != I / 2 ||
        dns->shape[0] != E || dns->shape[1] != H || dns->shape[2] != I / 16 || gsc->shape[0] != E || gsc->shape[1] != 3)
        return IK_ERR_SHAPE;
    if (!(gate_cap > 0.f) || !(up_cap > 0.f)) return IK_ERR_INVALID_ARG;
    if (aq != 0) return IK_ERR_UNSUPPORTED;  // W4A4 and the decode repack (bit 1) are not built on HIP
    if (T == 0) return IK_OK;
    MoeArgs a{};
    a.x = (const ik_bf16*)x->data;
    a.ids = (const int*)ids->data;
    a.wts = (const float*)wts->data;
    a.guw = (const uint8_t*)guw->data;
    a.gus = (const uint8_t*)gus->data;
    a.dnw = (const uint8_t*)dnw->data;
    a.dns = (const uint8_t*)dns->data;
    a.gs = (const float*)gsc->data;
    a.out = out->data;
    a.T = (int)T, a.K = (int)ids->shape[1], a.E = (int)E, a.H = (int)H, a.I = (int)I;
    a.gc = gate_cap, a.uc = up_cap;
    return run<FMT_NV, ACT_SITU>(a, of, rows, scratch, IK_STREAM(stream));
}

int mxfp8_impl(const ik_tensor* x, const ik_tensor* ids, const ik_tensor* wts, const ik_tensor* w13, const ik_tensor* s13,
               const ik_tensor* w2, const ik_tensor* s2, int act, float gate_cap, float up_cap, int aq, ik_tensor* out,
               const ik_scratch* scratch, void* stream, bool rows) {
    int of = 0;
    IK_TRY(common_checks(x, ids, wts, out, &of));
    IK_TRY(ik_check_a(w13, IK_DTYPE_FP8_E4M3, 3, 16));
    IK_TRY(ik_check(s13, IK_DTYPE_U8, 3));
    IK_TRY(ik_check_a(w2, IK_DTYPE_FP8_E4M3, 3, 16));
    IK_TRY(ik_check(s2, IK_DTYPE_U8, 3));
    const int64_t T = x->shape[0], H = x->shape[1], E = w13->shape[0], I = w13->shape[1] / 2;
    if (H % 32 || I % 32 || E < 1 || E > kEMax || w13->shape[1] != 2 * I || w13->shape[2] != H || s13->shape[0] != E ||
        s13->shape[1] != 2 * I || s13->shape[2] != H / 32 || w2->shape[0] != E || w2->shape[1] != H || w2->shape[2] != I ||
        s2->shape[0] != E || s2->shape[1] != H || s2->shape[2] != I / 32)
        return IK_ERR_SHAPE;
    if (act == ACT_SITU && (!(gate_cap > 0.f) || !(up_cap > 0.f))) return IK_ERR_INVALID_ARG;
    if (aq != 0) return IK_ERR_UNSUPPORTED;
    if (T == 0) return IK_OK;
    MoeArgs a{};
    a.x = (const ik_bf16*)x->data;
    a.ids = (const int*)ids->data;
    a.wts = (const float*)wts->data;
    a.guw = (const uint8_t*)w13->data;
    a.gus = (const uint8_t*)s13->data;
    a.dnw = (const uint8_t*)w2->data;
    a.dns = (const uint8_t*)s2->data;
    a.out = out->data;
    a.T = (int)T, a.K = (int)ids->shape[1], a.E = (int)E, a.H = (int)H, a.I = (int)I;
    a.gc = gate_cap, a.uc = up_cap;
    hipStream_t st = IK_STREAM(stream);
    switch (act) {
        case ACT_SILU: return run<FMT_MX, ACT_SILU>(a, of, rows, scratch, st);
        case ACT_GELU: return run<FMT_MX, ACT_GELU>(a, of, rows, scratch, st);
        case ACT_SITU: return run<FMT_MX, ACT_SITU>(a, of, rows, scratch, st);
        default: return IK_ERR_INVALID_ARG;
    }
}

}  // namespace

extern "C" {

void ik_ink_moe_probe(int v) { g_probe = v; }
void ik_ink_moe_set_pre(int gu, int dn) { g_pre_gu = gu, g_pre_dn = dn; }
void ik_ink_moe_set_lean(int v) { g_lean = v; }
void ik_ink_moe_set_sz(int v) { g_sz = v; }
void ik_ink_moe_set_nb(int v) { g_nb = v; }
void ik_ink_moe_set_wmma(int v) { g_wmma = v; }
// tests: the pair tile (0 = the decode kernels) the ordinary names take for a call of P (token, expert) pairs over E experts
int ik_ink_moe_wmma_pairs(long long P, int E) { return moe_wmma_pairs((int64_t)P, E); }
void ik_ink_moe_set_dec(int v) { g_dec = v; }

// The `_rows` twins are the decode kernels at every size (rows-exact, bitwise the first form); so are the ordinary names up to 512
// pairs and below 16 pairs per expert; above that, by default (INTARSIA_HIP_INK_MOE_WMMA unset or 1) they take the tensor-core
// prefill on the pair-table path; INTARSIA_HIP_INK_MOE_WMMA=0 keeps them on the decode kernels.
int ik_moe_experts_situ_bf16(const ik_tensor* x, const ik_tensor* ids, const ik_tensor* wts, const ik_tensor* guw,
                             const ik_tensor* gus, const ik_tensor* dnw, const ik_tensor* dns, const ik_tensor* gsc,
                             float gate_cap, float up_cap, int aq, ik_tensor* out, const ik_scratch* scratch, void* stream) {
    return situ_impl(x, ids, wts, guw, gus, dnw, dns, gsc, gate_cap, up_cap, aq, out, scratch, stream, false);
}

int ik_moe_experts_situ_bf16_rows(const ik_tensor* x, const ik_tensor* ids, const ik_tensor* wts, const ik_tensor* guw,
                                  const ik_tensor* gus, const ik_tensor* dnw, const ik_tensor* dns, const ik_tensor* gsc,
                                  float gate_cap, float up_cap, int aq, ik_tensor* out, const ik_scratch* scratch, void* stream) {
    return situ_impl(x, ids, wts, guw, gus, dnw, dns, gsc, gate_cap, up_cap, aq, out, scratch, stream, true);
}

int ik_moe_experts_mxfp8_bf16(const ik_tensor* x, const ik_tensor* ids, const ik_tensor* wts, const ik_tensor* w13,
                              const ik_tensor* s13, const ik_tensor* w2, const ik_tensor* s2, int act, float gate_cap,
                              float up_cap, int aq, ik_tensor* out, const ik_scratch* scratch, void* stream) {
    return mxfp8_impl(x, ids, wts, w13, s13, w2, s2, act, gate_cap, up_cap, aq, out, scratch, stream, false);
}

int ik_moe_experts_mxfp8_bf16_rows(const ik_tensor* x, const ik_tensor* ids, const ik_tensor* wts, const ik_tensor* w13,
                                   const ik_tensor* s13, const ik_tensor* w2, const ik_tensor* s2, int act, float gate_cap,
                                   float up_cap, int aq, ik_tensor* out, const ik_scratch* scratch, void* stream) {
    return mxfp8_impl(x, ids, wts, w13, s13, w2, s2, act, gate_cap, up_cap, aq, out, scratch, stream, true);
}

}  // extern "C"
