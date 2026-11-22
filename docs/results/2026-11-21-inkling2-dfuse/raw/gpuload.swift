// Concurrent GPU load for the blend-tail contention check (lane ink-dfuse): another process that keeps every GPU core busy
// streaming a 256 MB buffer (1024 threadgroups x 1024 threads per dispatch, two command buffers in flight) for SECONDS.
// usage: gpuload SECONDS
import Metal
import Foundation
let secs = Double(CommandLine.arguments.count > 1 ? CommandLine.arguments[1] : "30")!
let src = """
#include <metal_stdlib>
using namespace metal;
kernel void stream(device const uint4* w [[buffer(0)]], device float* out [[buffer(1)]], constant uint& n4 [[buffer(2)]],
                   uint gid [[thread_position_in_grid]], uint nth [[threads_per_grid]]) {
    float a = 0.0f;
    for (uint i = gid; i < n4; i += nth) { uint4 v = w[i]; a += as_type<float>(v.x ^ v.w); }
    if (a == 1234.5f) out[0] = a;
}
"""
let dev = MTLCreateSystemDefaultDevice()!
let lib = try! dev.makeLibrary(source: src, options: nil)
let pso = try! dev.makeComputePipelineState(function: lib.makeFunction(name: "stream")!)
let bytes = 256 << 20
let w = dev.makeBuffer(length: bytes, options: .storageModePrivate)!
let out = dev.makeBuffer(length: 64, options: .storageModeShared)!
var n4 = UInt32(bytes / 16)
let q = dev.makeCommandQueue()!
let t0 = Date()
var n = 0
var last: MTLCommandBuffer? = nil
while Date().timeIntervalSince(t0) < secs {
    let cb = q.makeCommandBuffer()!
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(pso)
    enc.setBuffer(w, offset: 0, index: 0)
    enc.setBuffer(out, offset: 0, index: 1)
    enc.setBytes(&n4, length: 4, index: 2)
    for _ in 0..<8 { enc.dispatchThreadgroups(MTLSize(width: 1024, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 1024, height: 1, depth: 1)) }
    enc.endEncoding()
    cb.commit()
    last?.waitUntilCompleted()
    last = cb
    n += 1
}
last?.waitUntilCompleted()
print("gpuload: \(n) command buffers of 8 x 256 MB streams in \(String(format: "%.1f", Date().timeIntervalSince(t0))) s")
