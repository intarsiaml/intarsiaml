// Single-threadgroup streaming bandwidth and dependent-dispatch cost on this GPU.
// usage: swift tgbw.swift
import Metal
import Foundation

let src = """
#include <metal_stdlib>
using namespace metal;
// each threadgroup streams `n4` uint4 from its region (region = (tg + base) % nreg), sums, writes one float
kernel void stream(device const uint4* w [[buffer(0)]], device float* out [[buffer(1)]],
                   constant uint* prm [[buffer(2)]],
                   uint tg [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]],
                   uint tpg [[threads_per_threadgroup]]) {
    const uint n4 = prm[0], base = prm[1], nreg = prm[2], shared_ = prm[3];
    const uint reg = shared_ ? (base % nreg) : ((tg + base) % nreg);
    device const uint4* r = w + (ulong)reg * n4;
    float a0 = 0, a1 = 0, a2 = 0, a3 = 0;
    uint i = tid;
    for (; i + 3 * tpg < n4; i += 4 * tpg) {
        uint4 v0 = r[i], v1 = r[i + tpg], v2 = r[i + 2 * tpg], v3 = r[i + 3 * tpg];
        a0 += as_type<float>(v0.x ^ v0.y) ; a1 += as_type<float>(v1.z ^ v1.w);
        a2 += as_type<float>(v2.x ^ v2.w) ; a3 += as_type<float>(v3.y ^ v3.z);
    }
    for (; i < n4; i += tpg) { uint4 v = r[i]; a0 += as_type<float>(v.x ^ v.y); }
    float s = simd_sum(a0 + a1 + a2 + a3);
    if ((tid & 31) == 0) out[tg * 32 + tid / 32] = s;
}
kernel void empty(device float* out [[buffer(1)]], uint tid [[thread_position_in_grid]]) {
    if (tid == 0) out[0] += 1.0f;
}
"""

let dev = MTLCreateSystemDefaultDevice()!
let lib = try! dev.makeLibrary(source: src, options: nil)
let pStream = try! dev.makeComputePipelineState(function: lib.makeFunction(name: "stream")!)
let pEmpty = try! dev.makeComputePipelineState(function: lib.makeFunction(name: "empty")!)
let q = dev.makeCommandQueue()!
let total = 512 << 20  // 512 MB of weights to rotate over (beyond the SLC)
let wbuf = dev.makeBuffer(length: total, options: .storageModePrivate)!
let obuf = dev.makeBuffer(length: 1 << 20, options: .storageModeShared)!

func run(_ body: (MTLComputeCommandEncoder) -> Void) -> Double {
    let cb = q.makeCommandBuffer()!
    let e = cb.makeComputeCommandEncoder()!
    body(e)
    e.endEncoding()
    cb.commit()
    cb.waitUntilCompleted()
    return cb.gpuEndTime - cb.gpuStartTime
}

// warm the clocks
for _ in 0..<5 { _ = run { e in e.setComputePipelineState(pStream); e.setBuffer(wbuf, offset: 0, index: 0); e.setBuffer(obuf, offset: 0, index: 1)
    var p: [UInt32] = [UInt32(total / 16 / 64), 0, 64, 0]; e.setBytes(&p, length: 16, index: 2)
    for _ in 0..<50 { e.dispatchThreadgroups(MTLSize(width: 64, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 1024, height: 1, depth: 1)) } } }

// dependent empty dispatches
do {
    var ts: [Double] = []
    for _ in 0..<7 {
        let n = 400
        let t = run { e in e.setComputePipelineState(pEmpty); e.setBuffer(obuf, offset: 0, index: 1)
            for _ in 0..<n { e.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1)) } }
        ts.append(t / Double(n) * 1e6)
    }
    ts.sort(); print(String(format: "empty dependent dispatch: %.2f us (median of 7 x 400)", ts[3]))
}

for (tpg, bytes) in [(1024, 600_000), (1024, 1_200_000), (512, 600_000), (1024, 300_000), (768, 600_000)] {
    for ntg in [1, 3, 16, 48] {
        for shared in [0, 1] {
            if ntg == 1 && shared == 1 { continue }
            let n4 = bytes / 16
            let nreg = total / 16 / n4
            var ts: [Double] = []
            for rep in 0..<7 {
                let calls = 200
                let t = run { e in e.setComputePipelineState(pStream); e.setBuffer(wbuf, offset: 0, index: 0); e.setBuffer(obuf, offset: 0, index: 1)
                    for c in 0..<calls {
                        var p: [UInt32] = [UInt32(n4), UInt32((c + rep * calls) * ntg), UInt32(nreg), UInt32(shared)]
                        e.setBytes(&p, length: 16, index: 2)
                        e.dispatchThreadgroups(MTLSize(width: ntg, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: tpg, height: 1, depth: 1))
                    } }
                ts.append(t / Double(calls) * 1e6)
            }
            ts.sort()
            let us = ts[3]
            let gbs = Double(bytes) / (us * 1e-6) / 1e9
            print(String(format: "tpg %4d bytes/TG %8d TGs %2d shared %d: %7.2f us/call, %6.1f GB/s per TG", tpg, bytes, ntg, shared, us, gbs))
        }
    }
}
