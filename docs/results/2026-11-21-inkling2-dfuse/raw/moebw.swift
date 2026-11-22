// Streaming floor of one Inkling-2-Mini NVFP4 expert visit at 1 row (lane ink-dfuse, measurement only).
// A visit reads 8 random experts of one layer: the gate+up rows (2 x 1152 x 2304 e2m1 + e4m3 scales = 2,985,984 B per
// expert) and the down rows (2304 x 1152 + scales = 1,492,992 B). A perfect streaming kernel (no math, every byte loaded
// once) reads those regions; 5 layer copies of 128 experts rotate (2.87 GB, far past the system cache), each visit its own
// 8 distinct experts. Times are GPU timestamps of one command buffer of `calls` visits (serial encoder: each dispatch waits
// for the one before, as in the model's stream), median of 7.
//   A  = the gate+up regions alone      B  = the down regions alone
//   AB = A then B (two dependent dispatches, the shipped structure)   F = all 16 regions in one dispatch (no dependency)
// Also: union 29 experts (the 4-row union) for A, and the empty dependent dispatch.
// usage: swift moebw.swift
import Metal
import Foundation

let src = """
#include <metal_stdlib>
using namespace metal;
// grid (ntgx, regions): threadgroup (x, y) streams slab x of region y; region y starts at eoff[y] (uint4) and has n4r[y]
template <uint U>
inline void body(device const uint4* w, device float* out, device const uint* eoff, device const uint* n4r, uint2 tg,
                 uint tid, uint tpg, uint ntgx) {
    const uint n4 = n4r[tg.y];
    const uint slab = (n4 + ntgx - 1u) / ntgx;
    const uint lo = tg.x * slab, hi = min(lo + slab, n4);
    device const uint4* r = w + (ulong)eoff[tg.y];
    float a = 0.f;
    uint i = lo + tid;
    for (; i + (U - 1u) * tpg < hi; i += U * tpg) {
        uint4 v[U];
        for (uint u = 0; u < U; ++u) v[u] = r[i + u * tpg];
        for (uint u = 0; u < U; ++u) a += as_type<float>(v[u].x ^ v[u].y ^ v[u].z ^ v[u].w);
    }
    for (; i < hi; i += tpg) { uint4 v = r[i]; a += as_type<float>(v.x ^ v.w); }
    const float s = simd_sum(a);
    if ((tid & 31u) == 0u && s == 1.2345f) out[0] = s;
}
#define KV(U) kernel void vis##U(device const uint4* w [[buffer(0)]], device float* out [[buffer(1)]],                  \\
    device const uint* eoff [[buffer(2)]], device const uint* n4r [[buffer(3)]], uint2 tg [[threadgroup_position_in_grid]], \\
    uint tid [[thread_index_in_threadgroup]], uint2 tpg [[threads_per_threadgroup]], uint2 ntg [[threadgroups_per_grid]]) { \\
    body<U>(w, out, eoff, n4r, tg, tid, tpg.x, ntg.x); }
KV(1)
KV(2)
KV(4)
kernel void empty(device float* out [[buffer(1)]], uint tid [[thread_position_in_grid]]) {
    if (tid == 0) out[0] += 1.0f;
}
"""

let dev = MTLCreateSystemDefaultDevice()!
let lib = try! dev.makeLibrary(source: src, options: nil)
let pso: [Int: MTLComputePipelineState] = Dictionary(uniqueKeysWithValues: [1, 2, 4].map { u in
    (u, try! dev.makeComputePipelineState(function: lib.makeFunction(name: "vis\(u)")!)) })
let pEmpty = try! dev.makeComputePipelineState(function: lib.makeFunction(name: "empty")!)
let q = dev.makeCommandQueue()!
let E = 128, LAYERS = 5
let GU = 2_985_984, DN = 1_492_992  // bytes per expert
let layerBytes = E * (GU + DN)
let wbuf = dev.makeBuffer(length: LAYERS * layerBytes, options: .storageModePrivate)!
let obuf = dev.makeBuffer(length: 1 << 16, options: .storageModeShared)!
print(String(format: "%@, %d layer copies of %.0f MB (%.2f GB)", dev.name, LAYERS, Double(layerBytes) / 1e6, Double(LAYERS * layerBytes) / 1e9))

// deterministic routing: visit c reads layer c % 5, `ne` distinct experts
var seed: UInt64 = 0x9e3779b97f4a7c15
func rnd() -> UInt64 { seed = seed &* 6364136223846793005 &+ 1442695040888963407; return seed >> 33 }
let calls = 240
func routing(_ ne: Int) -> [[Int]] {
    (0..<calls).map { _ in var s = Set<Int>(); while s.count < ne { s.insert(Int(rnd() % UInt64(E))) }; return Array(s) }
}
// per visit: region offsets (uint4) and sizes for A (gate+up), B (down), F (both)
func regions(_ route: [[Int]], _ kind: String) -> (MTLBuffer, MTLBuffer, Int) {
    var off: [UInt32] = [], n4: [UInt32] = []
    var per = 0
    for (c, ex) in route.enumerated() {
        let base = (c % LAYERS) * layerBytes
        var o: [UInt32] = [], n: [UInt32] = []
        for e in ex {
            if kind != "B" { o.append(UInt32((base + e * GU) / 16)); n.append(UInt32(GU / 16)) }
        }
        for e in ex {
            if kind != "A" { o.append(UInt32((base + E * GU + e * DN) / 16)); n.append(UInt32(DN / 16)) }
        }
        per = o.count
        off += o; n4 += n
    }
    return (dev.makeBuffer(bytes: off, length: off.count * 4, options: .storageModeShared)!,
            dev.makeBuffer(bytes: n4, length: n4.count * 4, options: .storageModeShared)!, per)
}

func run(_ body: (MTLComputeCommandEncoder) -> Void) -> Double {
    let cb = q.makeCommandBuffer()!
    let e = cb.makeComputeCommandEncoder()!
    body(e)
    e.endEncoding()
    cb.commit()
    cb.waitUntilCompleted()
    return cb.gpuEndTime - cb.gpuStartTime
}
struct Cfg { let u: Int; let tpg: Int; let ntgx: Int }
// one dispatch of cfg over visit c's regions of `r`
func disp(_ e: MTLComputeCommandEncoder, _ r: (MTLBuffer, MTLBuffer, Int), _ c: Int, _ k: Cfg) {
    e.setComputePipelineState(pso[k.u]!)
    e.setBuffer(wbuf, offset: 0, index: 0)
    e.setBuffer(obuf, offset: 0, index: 1)
    e.setBuffer(r.0, offset: c * r.2 * 4, index: 2)
    e.setBuffer(r.1, offset: c * r.2 * 4, index: 3)
    e.dispatchThreadgroups(MTLSize(width: k.ntgx, height: r.2, depth: 1), threadsPerThreadgroup: MTLSize(width: k.tpg, height: 1, depth: 1))
}
func median(_ f: () -> Double) -> Double { var v: [Double] = []; for _ in 0..<7 { v.append(f()) }; v.sort(); return v[3] }
func timeVisits(_ steps: [((MTLComputeCommandEncoder, Int) -> Void)]) -> Double {
    median { run { e in for c in 0..<calls { for s in steps { s(e, c) } } } / Double(calls) * 1e6 }
}

// warm the clocks
let r8 = routing(8)
let A8 = regions(r8, "A"), B8 = regions(r8, "B"), F8 = regions(r8, "F")
for _ in 0..<3 { _ = timeVisits([{ e, c in disp(e, F8, c, Cfg(u: 2, tpg: 256, ntgx: 72)) }]) }

let emptyUs = median { run { e in e.setComputePipelineState(pEmpty); e.setBuffer(obuf, offset: 0, index: 1)
    for _ in 0..<400 { e.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1)) } } / 400 * 1e6 }
print(String(format: "empty dependent dispatch: %.2f us", emptyUs))

// sweep: regions x configs; TGs = ntgx x regions
func sweep(_ name: String, _ r: (MTLBuffer, MTLBuffer, Int), _ bytes: Int) -> Cfg {
    var best = (1e9, Cfg(u: 1, tpg: 128, ntgx: 1))
    print("| \(name) | U | threads/TG | TGs | us/visit | GB/s |")
    print("|---|---:|---:|---:|---:|---:|")
    for u in [1, 2, 4] {
        for tpg in [128, 256, 512, 1024] {
            for ntgx in [9, 18, 36, 72, 144, 288] {
                let k = Cfg(u: u, tpg: tpg, ntgx: ntgx)
                let us = timeVisits([{ e, c in disp(e, r, c, k) }])
                let gbs = Double(bytes) / (us * 1e-6) / 1e9
                print(String(format: "| %@ | %d | %d | %d | %.2f | %.1f |", name, u, tpg, ntgx * r.2, us, gbs))
                if us < best.0 { best = (us, k) }
            }
        }
    }
    print(String(format: "BEST %@: %.2f us/visit, %.1f GB/s (U %d, %d threads/TG, %d TGs)", name, best.0,
                 Double(bytes) / (best.0 * 1e-6) / 1e9, best.1.u, best.1.tpg, best.1.ntgx * r.2))
    return best.1
}
let gu8 = 8 * GU, dn8 = 8 * DN
let bA = sweep("A (gate+up, 8 experts)", A8, gu8)
let bB = sweep("B (down, 8 experts)", B8, dn8)
let bF = sweep("F (gate+up + down, 8 experts, one dispatch)", F8, gu8 + dn8)
// the shipped geometry, mapped: gate+up 576 TGs x 128 threads (72 per expert), down 288 TGs x 128 (36 per expert)
let shipA = Cfg(u: 1, tpg: 128, ntgx: 72), shipB = Cfg(u: 1, tpg: 128, ntgx: 36)
for (label, ka, kb) in [("best A, best B", bA, bB), ("shipped TG counts (U 1, 128 threads)", shipA, shipB)] {
    let us = timeVisits([{ e, c in disp(e, A8, c, ka) }, { e, c in disp(e, B8, c, kb) }])
    print(String(format: "AB %@: %.2f us/visit, %.1f GB/s over %.2f MB", label, us, Double(gu8 + dn8) / (us * 1e-6) / 1e9, Double(gu8 + dn8) / 1e6))
}
// the 4-row union (29 experts) of gate+up and down, at the best 8-expert configs scaled to the same TGs per expert
let r29 = routing(29)
let A29 = regions(r29, "A"), B29 = regions(r29, "B")
let ab29 = timeVisits([{ e, c in disp(e, A29, c, bA) }, { e, c in disp(e, B29, c, bB) }])
print(String(format: "AB 29 experts (4-row union), best 8-expert configs: %.2f us/visit, %.1f GB/s over %.2f MB", ab29,
             Double(29 * (GU + DN)) / (ab29 * 1e-6) / 1e9, Double(29 * (GU + DN)) / 1e6))
_ = sweep("A (gate+up, 29 experts)", A29, 29 * GU)
