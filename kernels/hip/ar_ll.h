/* Tagged-line (LL) handshake of the two-rank collectives, shared by allreduce.hip and ar_norm.hip; the protocol is described in the
 * header comment of allreduce.hip. A 16-byte data word k of an n16-word message travels as four 8-byte lines (dword j | seq << 32)
 * at ll[j * n16 + k]. */
#pragma once
#include "ik_hip.h"

#define AR_LL_SPIN_MAX 480000000u /* polls of one word (the fence path's bound: a rank that never arrives traps) */

IK_DEVI void ll_st(unsigned long long* p, unsigned long long v) { __hip_atomic_store(p, v, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM); }
IK_DEVI unsigned long long ll_ld(const unsigned long long* p) { return __hip_atomic_load(p, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM); }
IK_DEVI unsigned long long ll_line(unsigned int seq, unsigned int d) { return ((unsigned long long)seq << 32) | d; }

/* push 16-byte word k of an n16-word message as four tagged lines. `v` is taken BY VALUE on purpose: a reference to the caller's
 * x4[k] would be read lazily, after the first atomic store (the compiler cannot move a load of x4 above a store to the peer's lines,
 * they may alias), which serialises four global loads with four stores */
IK_DEVI void ll_put(unsigned long long* pl, int n16, int k, unsigned int seq, int4 v) {
    ll_st(pl + k, ll_line(seq, (unsigned int)v.x));
    ll_st(pl + n16 + k, ll_line(seq, (unsigned int)v.y));
    ll_st(pl + 2 * n16 + k, ll_line(seq, (unsigned int)v.z));
    ll_st(pl + 3 * n16 + k, ll_line(seq, (unsigned int)v.w));
}

/* the peer's words k[0..cnt), once all their lines carry `seq` (CHECK false: whatever is there now, negative control only). All loads
 * of a round are issued before any is checked, so N words cost one round trip, not N. */
template <bool CHECK, int N>
IK_DEVI void ll_get_n(const unsigned long long* ml, int n16, const int* k, int cnt, unsigned int seq, int4* out) {
    unsigned int spins = 0;
    for (;;) {
        unsigned long long a[N], b[N], c[N], d[N];
#pragma unroll
        for (int j = 0; j < N; ++j)
            if (j < cnt) {
                a[j] = ll_ld(ml + k[j]);
                b[j] = ll_ld(ml + n16 + k[j]);
                c[j] = ll_ld(ml + 2 * n16 + k[j]);
                d[j] = ll_ld(ml + 3 * n16 + k[j]);
            }
        bool ok = true;
#pragma unroll
        for (int j = 0; j < N; ++j)
            if (j < cnt) {
                if (CHECK) ok = ok && (unsigned int)(a[j] >> 32) == seq && (unsigned int)(b[j] >> 32) == seq && (unsigned int)(c[j] >> 32) == seq &&
                                (unsigned int)(d[j] >> 32) == seq;
                out[j].x = (int)(unsigned int)a[j];
                out[j].y = (int)(unsigned int)b[j];
                out[j].z = (int)(unsigned int)c[j];
                out[j].w = (int)(unsigned int)d[j];
            }
        if (ok) return;
        __builtin_amdgcn_s_sleep(1);
        if (++spins > AR_LL_SPIN_MAX) __builtin_trap();
    }
}

/* one word */
template <bool CHECK>
IK_DEVI int4 ll_get(const unsigned long long* ml, int n16, int k, unsigned int seq) {
    int4 r;
    ll_get_n<CHECK, 1>(ml, n16, &k, 1, seq, &r);
    return r;
}

IK_DEVI unsigned int ll_seq(unsigned int* ctrs, unsigned int* s_seq) {
    if (threadIdx.x == 0) *s_seq = atomicAdd(&ctrs[blockIdx.x], 1u) + 1u;
    __syncthreads();
    return *s_seq;
}

/* does a message of `bytes` use the LL path? A pure function of the size and the group's (equal on both ranks) LL region */
static inline bool ll_fits(int drain, int64_t bytes, const void* peer_ll, const void* my_ll, int64_t ll_slot_bytes) {
    return (drain == 7 || drain == 8) && peer_ll && my_ll && ll_slot_bytes > 0 && bytes * 2 <= ll_slot_bytes;
}


