// TEMPORARY hardware microbenchmark of simdgroup_matrix throughput (not part of the final change).
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
#include "common.h"
using namespace metal;

// bind 0: out (float, per-thread sink), OpP.a = iterations
template <typename AT, typename CT, uint NACC>
kernel void mma_peak(device float* sink [[buffer(0)]], constant OpP& p [[buffer(1)]], uint tid [[thread_position_in_grid]],
                     uint lane [[thread_index_in_simdgroup]]) {
    simdgroup_matrix<AT, 8, 8> a, b;
    a.thread_elements()[0] = AT(float(lane) * 0.01f);
    a.thread_elements()[1] = AT(float(lane) * 0.02f);
    b.thread_elements()[0] = AT(float(tid & 7) * 0.01f);
    b.thread_elements()[1] = AT(0.5f);
    simdgroup_matrix<CT, 8, 8> c[NACC];
    for (uint i = 0; i < NACC; ++i) c[i] = make_filled_simdgroup_matrix<CT, 8, 8>(CT(0));
    for (uint it = 0; it < p.a; ++it) {
        for (uint i = 0; i < NACC; ++i) simdgroup_multiply_accumulate(c[i], a, b, c[i]);
    }
    float s = 0;
    for (uint i = 0; i < NACC; ++i) s += float(c[i].thread_elements()[0]) + float(c[i].thread_elements()[1]);
    sink[tid] = s;
}

// threadgroup-memory load + mma: per iteration NL frag loads (non-transposed) and NL mmas
template <typename AT, typename CT, uint NACC, bool TR>
kernel void mma_ld(device float* sink [[buffer(0)]], constant OpP& p [[buffer(1)]], uint tid [[thread_position_in_grid]],
                   uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]],
                   uint tix [[thread_index_in_threadgroup]]) {
    threadgroup AT buf[64 * 72];
    for (uint i = tix; i < 64 * 72; i += 256) buf[i] = AT(float(i % 13) * 0.01f);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_matrix<AT, 8, 8> a;
    a.thread_elements()[0] = AT(float(lane) * 0.01f);
    a.thread_elements()[1] = AT(float(lane) * 0.02f);
    simdgroup_matrix<CT, 8, 8> c[NACC];
    for (uint i = 0; i < NACC; ++i) c[i] = make_filled_simdgroup_matrix<CT, 8, 8>(CT(0));
    for (uint it = 0; it < p.a; ++it) {
        simdgroup_matrix<AT, 8, 8> b[NACC];
        for (uint i = 0; i < NACC; ++i) simdgroup_load(b[i], buf + ((i + sg) % 8) * 8 * 72 + (it & 7) * 8, 72, ulong2(0, 0), TR);
        for (uint i = 0; i < NACC; ++i) simdgroup_multiply_accumulate(c[i], a, b[i], c[i]);
    }
    float s = 0;
    for (uint i = 0; i < NACC; ++i) s += float(c[i].thread_elements()[0]) + float(c[i].thread_elements()[1]);
    sink[tid] = s;
}

#define PEAK(NAME, AT, CT, N)                                                                                         \
    template [[host_name(NAME)]] kernel void mma_peak<AT, CT, N>(device float*, constant OpP&, uint, uint);
PEAK("mma_peak_hf8", half, float, 8)
PEAK("mma_peak_hh8", half, half, 8)
PEAK("mma_peak_ff8", float, float, 8)
PEAK("mma_peak_bf8", bfloat, float, 8)
PEAK("mma_peak_hf16", half, float, 16)
#define LD(NAME, AT, CT, N, TR)                                                                                       \
    template [[host_name(NAME)]] kernel void mma_ld<AT, CT, N, TR>(device float*, constant OpP&, uint, uint, uint, uint);
LD("mma_ld_hf8", half, float, 8, false)
LD("mma_ld_hf8t", half, float, 8, true)
LD("mma_ld_bf8", bfloat, float, 8, false)
LD("mma_ld_ff8", float, float, 8, false)
