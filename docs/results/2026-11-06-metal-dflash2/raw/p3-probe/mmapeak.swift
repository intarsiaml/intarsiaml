// Lane dm: peak fp32 / fp16 simdgroup_multiply_accumulate throughput (independent accumulators, operands in registers).
import Metal
import Foundation
let src = """
#include <metal_stdlib>
using namespace metal;
template <typename T>
inline void body(device float* o, uint gid) {
  simdgroup_matrix<T, 8, 8> a = make_filled_simdgroup_matrix<T, 8, 8>((T)0.999f), b = make_filled_simdgroup_matrix<T, 8, 8>((T)0.001f);
  simdgroup_matrix<T, 8, 8> c0 = make_filled_simdgroup_matrix<T, 8, 8>((T)(gid * 1e-7f)), c1 = c0, c2 = c0, c3 = c0, c4 = c0, c5 = c0, c6 = c0, c7 = c0;
  for (int i = 0; i < 1024; ++i) {
    simdgroup_multiply_accumulate(c0, a, b, c0); simdgroup_multiply_accumulate(c1, a, b, c1);
    simdgroup_multiply_accumulate(c2, a, b, c2); simdgroup_multiply_accumulate(c3, a, b, c3);
    simdgroup_multiply_accumulate(c4, a, b, c4); simdgroup_multiply_accumulate(c5, a, b, c5);
    simdgroup_multiply_accumulate(c6, a, b, c6); simdgroup_multiply_accumulate(c7, a, b, c7);
  }
  o[gid] = (float)(c0.thread_elements()[0] + c1.thread_elements()[0] + c2.thread_elements()[0] + c3.thread_elements()[0]
          + c4.thread_elements()[0] + c5.thread_elements()[0] + c6.thread_elements()[0] + c7.thread_elements()[1]);
}
kernel void f(device float* o [[buffer(0)]], uint gid [[thread_position_in_grid]]) { body<float>(o, gid); }
kernel void h(device float* o [[buffer(0)]], uint gid [[thread_position_in_grid]]) { body<half>(o, gid); }
"""
let dev = MTLCreateSystemDefaultDevice()!
let lib = try! dev.makeLibrary(source: src, options: nil)
let q = dev.makeCommandQueue()!
let n = 40 * 1024 * 32
let buf = dev.makeBuffer(length: n * 4, options: .storageModeShared)!
for name in ["f", "h"] {
  let pso = try! dev.makeComputePipelineState(function: lib.makeFunction(name: name)!)
  var best = 1e9
  for _ in 0..<5 {
    let cb = q.makeCommandBuffer()!; let e = cb.makeComputeCommandEncoder()!
    e.setComputePipelineState(pso); e.setBuffer(buf, offset: 0, index: 0)
    e.dispatchThreads(MTLSize(width: n, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    e.endEncoding(); cb.commit(); cb.waitUntilCompleted()
    best = min(best, cb.gpuEndTime - cb.gpuStartTime)
  }
  let mmas = Double(n / 32) * 1024 * 8
  print(name == "f" ? "fp32 mma" : "fp16 mma", String(format: "%.2f TFMA/s", mmas * 512 / best / 1e12))
}
