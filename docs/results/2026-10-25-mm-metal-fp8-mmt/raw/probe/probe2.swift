// Probe 2: MMA == fma chain on hard data (subnormals, inf, NaN, -0, exact e4m3 x bf16 products); thread_elements layout.
import Metal
import Foundation
let src = """
#include <metal_stdlib>
using namespace metal;
kernel void mma(device const float* A [[buffer(0)]], device const float* B [[buffer(1)]], device const float* C0 [[buffer(2)]],
                device float* out [[buffer(3)]], device float* ref [[buffer(4)]], device uint* lay [[buffer(5)]],
                uint tg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    const ulong o = (ulong)tg * 64;
    simdgroup_float8x8 a, b, c;
    simdgroup_load(a, A + o, 8);
    simdgroup_load(b, B + o, 8);
    simdgroup_load(c, C0 + o, 8);
    simdgroup_multiply_accumulate(c, a, b, c);
    simdgroup_store(c, out + o, 8);
    for (uint e = lane; e < 64; e += 32) {
        const uint i = e / 8, j = e % 8;
        float t = C0[o + e];
        for (int k = 0; k < 8; ++k) t = fma(A[o + i * 8 + k], B[o + k * 8 + j], t);
        ref[o + e] = t;
    }
    if (tg == 0) {
        // layout: load a matrix whose element (r, c) = r * 8 + c, read thread_elements
        threadgroup float m[64];
        for (uint e = lane; e < 64; e += 32) m[e] = float(e);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_float8x8 L;
        simdgroup_load(L, m, 8);
        auto te = L.thread_elements();
        lay[lane * 2] = uint(te[0]);
        lay[lane * 2 + 1] = uint(te[1]);
    }
}
"""
let dev = MTLCreateSystemDefaultDevice()!
let lib = try! dev.makeLibrary(source: src, options: nil)
let q = dev.makeCommandQueue()!
var seed: UInt64 = 777
func nxt() -> UInt64 { seed = seed &* 6364136223846793005 &+ 1442695040888963407; return seed >> 11 }
// e4m3 values / 256 (exact via the code table), bf16 values with wide exponents incl. subnormal floats, specials
func e4m3(_ c: Int) -> Float {
    let s: Float = (c & 0x80) != 0 ? -1 : 1; let e = (c >> 3) & 0xF; let m = c & 7
    if (c & 0x7F) == 0x7F { return Float.nan }
    if e == 0 { return s * Float(m) / 8 * powf(2, -6) / 256 }
    return s * (1 + Float(m) / 8) * powf(2, Float(e - 7)) / 256
}
func bf16(_ bits: UInt16) -> Float { Float(bitPattern: UInt32(bits) << 16) }
let G = 8192
var A = [Float](), B = [Float](), C = [Float]()
for g in 0..<G {
    let mode = g % 4
    for _ in 0..<64 {
        // A: activations (bf16): mode 0 normal wide, 1 tiny (subnormal-producing), 2 with specials, 3 cancellation
        var a: Float
        switch mode {
        case 1: a = bf16(UInt16(nxt() & 0x00FF) | UInt16((nxt() & 1) << 15) | 0x0080)  // tiny exponents
        case 2: let r = nxt() % 64; a = r == 0 ? Float.infinity : r == 1 ? -Float.infinity : r == 2 ? Float.nan : r == 3 ? -0.0 : bf16(UInt16(nxt() & 0xFFFF))
        default: a = bf16(UInt16(0x3000 + Int(nxt() % 0x1800)) | UInt16((nxt() & 1) << 15))
        }
        if !a.isFinite && mode != 2 { a = 1 }
        A.append(a)
        B.append(e4m3(Int(nxt() % 256)) .isNaN ? 0.5 : e4m3(Int(nxt() % 255)))
        C.append(mode == 3 ? -Float(Int(nxt() % 1000)) / 256 : (nxt() % 4 == 0 ? 0 : Float(Int(nxt() % 2000) - 1000) / 64))
    }
}
func buf(_ a: [Float]) -> MTLBuffer { dev.makeBuffer(bytes: a, length: a.count * 4, options: .storageModeShared)! }
let bo = dev.makeBuffer(length: G * 64 * 4, options: .storageModeShared)!
let br = dev.makeBuffer(length: G * 64 * 4, options: .storageModeShared)!
let bl = dev.makeBuffer(length: 64 * 4, options: .storageModeShared)!
let p1 = try! dev.makeComputePipelineState(function: lib.makeFunction(name: "mma")!)
let cb = q.makeCommandBuffer()!; let enc = cb.makeComputeCommandEncoder()!
enc.setComputePipelineState(p1)
for (i, b) in [buf(A), buf(B), buf(C), bo, br, bl].enumerated() { enc.setBuffer(b, offset: 0, index: i) }
enc.dispatchThreadgroups(MTLSize(width: G, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
let o = bo.contents().bindMemory(to: UInt32.self, capacity: G * 64), r = br.contents().bindMemory(to: UInt32.self, capacity: G * 64)
var same = [0, 0, 0, 0], tot = [0, 0, 0, 0], nanSame = 0, nanTot = 0
for g in 0..<G { for e in 0..<64 {
    let i = g * 64 + e; tot[g % 4] += 1
    if o[i] == r[i] { same[g % 4] += 1 }
    if Float(bitPattern: r[i]).isNaN { nanTot += 1; if Float(bitPattern: o[i]).isNaN { nanSame += 1 } }
} }
print("bitwise equal by mode (wide, tiny/subnormal, specials, cancellation): \(same) of \(tot); NaN in ref \(nanTot), NaN in mma \(nanSame)")
let l = bl.contents().bindMemory(to: UInt32.self, capacity: 64)
print("layout lane -> (r,c) of its 2 elements:", (0..<32).map { "\($0):\(l[$0*2]/8),\(l[$0*2]%8)|\(l[$0*2+1]/8),\(l[$0*2+1]%8)" }.joined(separator: " "))
