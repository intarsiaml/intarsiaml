// Probe: does simdgroup_multiply_accumulate (float 8x8x8) equal a sequential fma chain c = fma(a_k, b_k, c), k = 0..7,
// bit for bit? And which tree does simd_sum use? Random data with wide exponents and cancellation.
import Metal
import Foundation

let src = """
#include <metal_stdlib>
using namespace metal;
kernel void mma(device const float* A [[buffer(0)]], device const float* B [[buffer(1)]], device const float* C0 [[buffer(2)]],
                device float* out [[buffer(3)]], device float* ref [[buffer(4)]], uint tg [[threadgroup_position_in_grid]],
                uint lane [[thread_index_in_simdgroup]]) {
    const ulong o = (ulong)tg * 64;
    simdgroup_float8x8 a, b, c;
    simdgroup_load(a, A + o, 8);
    simdgroup_load(b, B + o, 8);
    simdgroup_load(c, C0 + o, 8);
    simdgroup_multiply_accumulate(c, a, b, c);
    simdgroup_store(c, out + o, 8);
    // reference: per (i, j), sequential fma chain from C0 over k = 0..7, and also k = 7..0
    for (uint e = lane; e < 64; e += 32) {
        const uint i = e / 8, j = e % 8;
        float t = C0[o + e];
        for (int k = 0; k < 8; ++k) t = fma(A[o + i * 8 + k], B[o + k * 8 + j], t);
        ref[o * 3 + e] = t;
        float r = C0[o + e];
        for (int k = 7; k >= 0; --k) r = fma(A[o + i * 8 + k], B[o + k * 8 + j], r);
        ref[o * 3 + 64 + e] = r;
        // plain products summed in k order then added to C (another candidate)
        float s = 0.0f;
        for (int k = 0; k < 8; ++k) s = fma(A[o + i * 8 + k], B[o + k * 8 + j], s);
        ref[o * 3 + 128 + e] = C0[o + e] + s;
    }
}
kernel void ssum(device const float* X [[buffer(0)]], device float* out [[buffer(1)]], uint tg [[threadgroup_position_in_grid]],
                 uint lane [[thread_index_in_simdgroup]]) {
    const float v = X[tg * 32 + lane];
    const float s = simd_sum(v);
    // candidates: xor butterfly 16, 8, 4, 2, 1 and 1, 2, 4, 8, 16
    float a = v; for (ushort d = 16; d >= 1; d /= 2) a += simd_shuffle_xor(a, d);
    float b = v; for (ushort d = 1; d <= 16; d *= 2) b += simd_shuffle_xor(b, d);
    if (lane == 0) { out[tg * 4] = s; out[tg * 4 + 1] = a; out[tg * 4 + 2] = b; }
    if (lane == 0) { float q = 0.0f; for (int l = 0; l < 32; ++l) q += X[tg * 32 + l]; out[tg * 4 + 3] = q; }
}
"""
let dev = MTLCreateSystemDefaultDevice()!
let lib = try! dev.makeLibrary(source: src, options: nil)
let q = dev.makeCommandQueue()!
func rnd(_ n: Int, _ seed: inout UInt64) -> [Float] {
    (0..<n).map { _ in
        seed = seed &* 6364136223846793005 &+ 1442695040888963407
        let u = Double(seed >> 11) / Double(1 << 53)
        seed = seed &* 6364136223846793005 &+ 1442695040888963407
        let e = Int(seed >> 59) - 16  // exponent spread 2^-16 .. 2^15
        return Float((u - 0.5) * pow(2.0, Double(e)))
    }
}
var seed: UInt64 = 12345
let G = 4096
let A = rnd(G * 64, &seed), B = rnd(G * 64, &seed), C = rnd(G * 64, &seed)
func buf(_ a: [Float]) -> MTLBuffer { dev.makeBuffer(bytes: a, length: a.count * 4, options: .storageModeShared)! }
let ba = buf(A), bb = buf(B), bc = buf(C)
let bo = dev.makeBuffer(length: G * 64 * 4, options: .storageModeShared)!
let br = dev.makeBuffer(length: G * 64 * 3 * 4, options: .storageModeShared)!
let p1 = try! dev.makeComputePipelineState(function: lib.makeFunction(name: "mma")!)
var cb = q.makeCommandBuffer()!; var enc = cb.makeComputeCommandEncoder()!
enc.setComputePipelineState(p1)
for (i, b) in [ba, bb, bc, bo, br].enumerated() { enc.setBuffer(b, offset: 0, index: i) }
enc.dispatchThreadgroups(MTLSize(width: G, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
let o = bo.contents().bindMemory(to: UInt32.self, capacity: G * 64)
let r = br.contents().bindMemory(to: UInt32.self, capacity: G * 64 * 3)
var same = [0, 0, 0]
for g in 0..<G { for e in 0..<64 { for c in 0..<3 { if o[g * 64 + e] == r[g * 64 * 3 + c * 64 + e] { same[c] += 1 } } } }
print("mma vs fma chain k=0..7 from C: \(same[0]) / \(G * 64); k=7..0: \(same[1]); C + chain(0): \(same[2])")
// simd_sum
let X = rnd(G * 32, &seed)
let bx = buf(X); let bs = dev.makeBuffer(length: G * 4 * 4, options: .storageModeShared)!
let p2 = try! dev.makeComputePipelineState(function: lib.makeFunction(name: "ssum")!)
cb = q.makeCommandBuffer()!; enc = cb.makeComputeCommandEncoder()!
enc.setComputePipelineState(p2); enc.setBuffer(bx, offset: 0, index: 0); enc.setBuffer(bs, offset: 0, index: 1)
enc.dispatchThreadgroups(MTLSize(width: G, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
let s = bs.contents().bindMemory(to: UInt32.self, capacity: G * 4)
var ss = [0, 0, 0]
for g in 0..<G { for c in 1...3 { if s[g * 4] == s[g * 4 + c] { ss[c - 1] += 1 } } }
print("simd_sum vs xor 16..1: \(ss[0]) / \(G); xor 1..16: \(ss[1]); sequential: \(ss[2])")
