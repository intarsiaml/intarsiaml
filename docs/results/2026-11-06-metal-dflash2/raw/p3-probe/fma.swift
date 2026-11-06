import Metal
import Foundation
let src = """
#include <metal_stdlib>
using namespace metal;
kernel void k(device float* o [[buffer(0)]], uint gid [[thread_position_in_grid]]) {
  float a0 = gid * 1e-7f, a1 = a0 + 1, a2 = a0 + 2, a3 = a0 + 3, a4 = a0+4, a5=a0+5, a6=a0+6, a7=a0+7;
  const float b = 0.999f, c = 1e-6f;
  for (int i = 0; i < 4096; ++i) {
    a0 = fma(a0, b, c); a1 = fma(a1, b, c); a2 = fma(a2, b, c); a3 = fma(a3, b, c);
    a4 = fma(a4, b, c); a5 = fma(a5, b, c); a6 = fma(a6, b, c); a7 = fma(a7, b, c);
  }
  o[gid] = a0+a1+a2+a3+a4+a5+a6+a7;
}
kernel void h(device half* o [[buffer(0)]], uint gid [[thread_position_in_grid]]) {
  half a0 = gid * 1e-4h, a1 = a0 + 1, a2 = a0 + 2, a3 = a0 + 3, a4 = a0+4, a5=a0+5, a6=a0+6, a7=a0+7;
  const half b = 0.999h, c = 1e-3h;
  for (int i = 0; i < 4096; ++i) {
    a0 = fma(a0, b, c); a1 = fma(a1, b, c); a2 = fma(a2, b, c); a3 = fma(a3, b, c);
    a4 = fma(a4, b, c); a5 = fma(a5, b, c); a6 = fma(a6, b, c); a7 = fma(a7, b, c);
  }
  o[gid] = a0+a1+a2+a3+a4+a5+a6+a7;
}
"""
let dev = MTLCreateSystemDefaultDevice()!
let lib = try! dev.makeLibrary(source: src, options: nil)
let q = dev.makeCommandQueue()!
let n = 40 * 1024 * 64
let buf = dev.makeBuffer(length: n * 4, options: .storageModeShared)!
for name in ["k", "h"] {
  let pso = try! dev.makeComputePipelineState(function: lib.makeFunction(name: name)!)
  var best = 1e9
  for _ in 0..<5 {
    let cb = q.makeCommandBuffer()!; let e = cb.makeComputeCommandEncoder()!
    e.setComputePipelineState(pso); e.setBuffer(buf, offset: 0, index: 0)
    e.dispatchThreads(MTLSize(width: n, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    e.endEncoding(); cb.commit(); cb.waitUntilCompleted()
    best = min(best, cb.gpuEndTime - cb.gpuStartTime)
  }
  print(name == "k" ? "fp32" : "fp16", String(format: "%.2f TFMA/s", Double(n) * 4096 * 8 / best / 1e12), dev.name)
}
