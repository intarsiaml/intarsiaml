// Lane dm phase 3 probe: is fp32 simdgroup_multiply_accumulate (8x8x8) bitwise a sequential fma chain in k order?
// 4 chained MMAs from C0 (one 32-element MXFP4 group: 4 words of 8), inputs as the MXFP4 GEMV sees them (e2m1 * 2^-14 times bf16,
// every product exact in f32), plus adversarial cancellation rows. Compared to CPU references: k ascending, k descending,
// pairwise tree per 8 then added, one rounding per 8-wide dot.
import Metal
import Foundation
let src = """
#include <metal_stdlib>
using namespace metal;
kernel void mma(device const float* A [[buffer(0)]], device const float* B [[buffer(1)]], device const float* C0 [[buffer(2)]],
                device float* C [[buffer(3)]], uint tg [[threadgroup_position_in_grid]]) {
  simdgroup_float8x8 c, a, b;
  simdgroup_load(c, C0 + tg * 64, 8);
  for (int s = 0; s < 4; ++s) {
    simdgroup_load(a, A + (tg * 4 + s) * 64, 8);
    simdgroup_load(b, B + (tg * 4 + s) * 64, 8);
    simdgroup_multiply_accumulate(c, a, b, c);
  }
  simdgroup_store(c, C + tg * 64, 8);
}
"""
let dev = MTLCreateSystemDefaultDevice()!
let lib = try! dev.makeLibrary(source: src, options: nil)
let pso = try! dev.makeComputePipelineState(function: lib.makeFunction(name: "mma")!)
let q = dev.makeCommandQueue()!
let trials = 20000
var rng = SystemRandomNumberGenerator()
let e2m1: [Float] = [0, 0.5, 1, 1.5, 2, 3, 4, 6]
func bf16(_ f: Float) -> Float { Float(bitPattern: f.bitPattern & 0xFFFF0000) }
var A = [Float](repeating: 0, count: trials * 4 * 64), B = A, C0 = [Float](repeating: 0, count: trials * 64)
for t in 0..<trials {
  let adv = t % 4 == 3
  let nzc = t % 8 >= 4
  for s in 0..<4 { for i in 0..<64 {
    let w = e2m1[Int.random(in: 0..<8, using: &rng)] * (Bool.random(using: &rng) ? 1 : -1) * Float(pow(2.0, -14.0))
    var x = bf16(Float.random(in: -1...1, using: &rng) * Float(pow(2.0, Double(Int.random(in: -20...20, using: &rng)))))
    if adv && Int.random(in: 0..<4, using: &rng) == 0 { x = bf16((Bool.random(using: &rng) ? 1 : -1) * Float(pow(2.0, Double(Int.random(in: 20...40, using: &rng))))) }
    A[(t * 4 + s) * 64 + i] = w   // A[n][k] at n*8+k
    B[(t * 4 + s) * 64 + i] = x   // B[k][m] at k*8+m
  } }
  if nzc { for i in 0..<64 { C0[t * 64 + i] = Float.random(in: -1...1, using: &rng) * Float(pow(2.0, Double(Int.random(in: -30...10, using: &rng)))) } }
}
func buf(_ v: [Float]) -> MTLBuffer { dev.makeBuffer(bytes: v, length: v.count * 4, options: .storageModeShared)! }
let ba = buf(A), bb = buf(B), bc0 = buf(C0), bc = dev.makeBuffer(length: trials * 64 * 4, options: .storageModeShared)!
let cb = q.makeCommandBuffer()!; let e = cb.makeComputeCommandEncoder()!
e.setComputePipelineState(pso); for (i, b) in [ba, bb, bc0, bc].enumerated() { e.setBuffer(b, offset: 0, index: i) }
e.dispatchThreadgroups(MTLSize(width: trials, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
e.endEncoding(); cb.commit(); cb.waitUntilCompleted()
let got = bc.contents().bindMemory(to: Float.self, capacity: trials * 64)
var miss = [0, 0, 0, 0]; let names = ["k ascending fma", "k descending fma", "pairwise tree per 8", "one rounding per 8"]
var shown = 0
for t in 0..<trials { for n in 0..<8 { for m in 0..<8 {
  var r = [Float](repeating: C0[t * 64 + n * 8 + m], count: 4)
  for s in 0..<4 {
    let p = (0..<8).map { k in A[(t * 4 + s) * 64 + n * 8 + k] * B[(t * 4 + s) * 64 + k * 8 + m] }
    for k in 0..<8 { r[0] = p[k] + r[0] }                      // products exact: fma == rounded add
    for k in (0..<8).reversed() { r[1] = p[k] + r[1] }
    let t1 = [(p[0] + p[1]), (p[2] + p[3]), (p[4] + p[5]), (p[6] + p[7])]
    r[2] = r[2] + ((t1[0] + t1[1]) + (t1[2] + t1[3]))
    r[3] = Float(Double(r[3]) + p.reduce(0.0) { $0 + Double($1) })
  }
  let g = got[t * 64 + n * 8 + m]
  for i in 0..<4 where r[i].bitPattern != g.bitPattern { miss[i] += 1 }
  if r[0].bitPattern != g.bitPattern && shown < 5 { print("trial \(t) (\(n),\(m)): gpu \(g) asc \(r[0]) desc \(r[1]) tree \(r[2]) dot \(r[3])"); shown += 1 }
} } }
let total = trials * 64
for i in 0..<4 { print("\(names[i]): \(miss[i]) / \(total) differ") }
print(dev.name)
