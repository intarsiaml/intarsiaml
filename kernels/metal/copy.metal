// Bandwidth-ceiling kernels for the Metal spike: a plain device-to-device copy (reads n16*16 B, writes n16*16 B) and
// a read-only xor reduction (reads n16*16 B, writes 4 B per thread). 16-byte vector accesses, 4 (copy) / 16 (read)
// per thread; a simdgroup owns one contiguous 32*PER_THREAD*16 B block and each of its loads is a contiguous 512 B across the 32 lanes.
// The host dispatches a multiple of 32 threads (whole simdgroups).

#include <metal_stdlib>
using namespace metal;

constant constexpr int PER_THREAD = 4;
constant constexpr int READ_PER_THREAD = 16;

kernel void copy_u4(device const uint4* src [[buffer(0)]],
                    device uint4* dst [[buffer(1)]],
                    constant uint& n16 [[buffer(2)]],
                    uint gid [[thread_position_in_grid]]) {
    uint4 v[PER_THREAD];
    for (int u = 0; u < PER_THREAD; ++u) {
        const uint i = (gid / 32) * (32 * PER_THREAD) + u * 32 + (gid % 32);
        if (i < n16) v[u] = src[i];
    }
    for (int u = 0; u < PER_THREAD; ++u) {
        const uint i = (gid / 32) * (32 * PER_THREAD) + u * 32 + (gid % 32);
        if (i < n16) dst[i] = v[u];
    }
}

kernel void read_xor_u4(device const uint4* src [[buffer(0)]],
                        device uint* out [[buffer(1)]],
                        constant uint& n16 [[buffer(2)]],
                        uint gid [[thread_position_in_grid]]) {
    uint4 acc = uint4(0);
    for (int u = 0; u < READ_PER_THREAD; ++u) {
        const uint i = (gid / 32) * (32 * READ_PER_THREAD) + u * 32 + (gid % 32);
        if (i < n16) acc ^= src[i];
    }
    out[gid] = acc.x ^ acc.y ^ acc.z ^ acc.w;
}

// Backend::zero / Backend::copy on the GPU, appended to the stream's serial compute encoder (no host wait). The
// 16-byte kernels need 16-byte aligned bindings and a byte count that is a multiple of 16; the byte kernels take
// any alignment (they bind the whole MTLBuffer and index it with 64-bit byte offsets) and are used for the
// unaligned remainder, so they are a correctness path, not a fast one.

kernel void zero_u4(device uint4* dst [[buffer(0)]],
                    constant uint& n16 [[buffer(1)]],
                    uint gid [[thread_position_in_grid]]) {
    for (int u = 0; u < PER_THREAD; ++u) {
        const uint i = (gid / 32) * (32 * PER_THREAD) + u * 32 + (gid % 32);
        if (i < n16) dst[i] = uint4(0);
    }
}

// p = { dst byte offset, byte count }
kernel void zero_bytes(device uchar* dst [[buffer(0)]],
                       constant ulong* p [[buffer(1)]],
                       uint gid [[thread_position_in_grid]]) {
    if (gid < p[1]) dst[p[0] + gid] = 0;
}

// p = { src byte offset, dst byte offset, byte count }; the ranges must not overlap
kernel void copy_bytes(device const uchar* src [[buffer(0)]],
                       device uchar* dst [[buffer(1)]],
                       constant ulong* p [[buffer(2)]],
                       uint gid [[thread_position_in_grid]]) {
    if (gid < p[2]) dst[p[1] + gid] = src[p[0] + gid];
}
