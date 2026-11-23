// 1-row timing of variants of the shipped NVFP4 expert kernels (lane ink-dfuse, measurement only; sources from
// .scratch/moevar_gen.py). One visit = the Auto policy at T = 1: ink_moe_gu_nv_a_x (union) then ink_moe_dn_f_nv_a_bf16,
// two dependent dispatches, as the launcher (crates/intarsia-backend-metal/src/inkmoe.rs) issues them. 5 layer copies of
// 128 experts rotate (2.87 GB), each visit its own 8 distinct experts; GPU time of one command buffer of 240 visits, median
// of 7. Per variant: both kernels, gate+up alone, down alone; and the max |out - shipped out| over 4 visits.
// usage: moevar VARIANT_DIR name ...
import Metal
import Foundation

let args = CommandLine.arguments
let vdir = args[1]
let names = Array(args.dropFirst(2))
let dev = MTLCreateSystemDefaultDevice()!
let q = dev.makeCommandQueue()!
let E = 128, K = 8, H = 2304, I = 1152, LAYERS = 5, calls = 240
// per layer: w13 u8 [E, 2I, H/2], s13 e4m3 [E, 2I, H/16], w2 u8 [E, H, I/2], s2 e4m3 [E, H, I/16]
let sz = [E * 2 * I * H / 2, E * 2 * I * H / 16, E * H * I / 2, E * H * I / 16]
let layerBytes = sz.reduce(0, +)
let wbuf = dev.makeBuffer(length: LAYERS * layerBytes, options: .storageModePrivate)!
do {  // random e2m1 codes (no NaN codes), random finite e4m3 scales 0x20..0x2f (2^-3 .. 2^-2 ish): order changes show
    let fsrc = """
    #include <metal_stdlib>
    using namespace metal;
    kernel void fillr(device uint* w [[buffer(0)]], constant uint2& pm [[buffer(1)]], uint i [[thread_position_in_grid]]) {
        if (i >= pm.x) return;
        uint v = (i + pm.y) * 0x9e3779b1u; v ^= v >> 15; v *= 0x85ebca77u; v ^= v >> 13; v *= 0xc2b2ae3du; v ^= v >> 16;
        w[i] = pm.y & 1u ? ((v & 0x0f0f0f0fu) | 0x20202020u) : v;
    }
    """
    let fl = try! dev.makeLibrary(source: fsrc, options: nil)
    let fp = try! dev.makeComputePipelineState(function: fl.makeFunction(name: "fillr")!)
    let cb = q.makeCommandBuffer()!, e = cb.makeComputeCommandEncoder()!
    e.setComputePipelineState(fp)
    for l in 0..<LAYERS { for r in 0..<4 {
        let n = sz[r] / 4
        var pm: (UInt32, UInt32) = (UInt32(n), UInt32(2 * (l * 4 + r) + (r % 2)))  // odd y = a scale region
        e.setBuffer(wbuf, offset: l * layerBytes + sz[0..<r].reduce(0, +), index: 0); e.setBytes(&pm, length: 8, index: 1)
        e.dispatchThreads(MTLSize(width: n, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    } }
    e.endEncoding(); cb.commit(); cb.waitUntilCompleted()
}
func off(_ l: Int, _ i: Int) -> Int { l * layerBytes + sz[0..<i].reduce(0, +) }
var gsc = [Float](repeating: 0.05, count: E * 3)
let gb = dev.makeBuffer(bytes: &gsc, length: gsc.count * 4, options: .storageModeShared)!
var seed: UInt64 = 0x51ed27
func rnd() -> UInt64 { seed = seed &* 6364136223846793005 &+ 1442695040888963407; return seed >> 33 }
var xs = (0..<H).map { _ in UInt16(0x3c00 + Int(rnd() % 0x300)) }  // bf16 in [0.0078, 0.5) ish
let xb = dev.makeBuffer(bytes: &xs, length: H * 2, options: .storageModeShared)!
var ids: [Int32] = [], wts: [Float] = []
for _ in 0..<calls {
    var s = Set<Int32>(); while s.count < K { s.insert(Int32(rnd() % UInt64(E))) }
    ids += Array(s); wts += (0..<K).map { _ in Float(rnd() % 1000) / 8000 }
}
let ib = dev.makeBuffer(bytes: &ids, length: ids.count * 4, options: .storageModeShared)!
let wb = dev.makeBuffer(bytes: &wts, length: wts.count * 4, options: .storageModeShared)!
let hb = dev.makeBuffer(length: K * I * 2, options: .storageModePrivate)!
let ob = dev.makeBuffer(length: H * 2 * 4, options: .storageModeShared)!
let occb = dev.makeBuffer(length: 16, options: .storageModeShared)!  // residency probe (OCC variants): [in flight, max]
struct OpP { var d: (UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32); var s: (UInt32, UInt32, UInt32, UInt32, UInt32); var f0: Float; var f1: Float; var pad: UInt32 }
var pgu = OpP(d: (1, UInt32(E), UInt32(K), UInt32(H), UInt32(I), 2 | 4, 0, 0), s: (0, 0, 0, 0, 0), f0: 4, f1: 25, pad: 0)
var pdn = pgu; pdn.d.5 = 2
precondition(MemoryLayout<OpP>.size == 64)

func run(_ body: (MTLComputeCommandEncoder) -> Void) -> Double {
    let cb = q.makeCommandBuffer()!, e = cb.makeComputeCommandEncoder()!
    body(e); e.endEncoding(); cb.commit(); cb.waitUntilCompleted()
    return cb.gpuEndTime - cb.gpuStartTime
}
func median(_ f: () -> Double) -> Double { var v: [Double] = []; for _ in 0..<7 { v.append(f()) }; v.sort(); return v[3] }
let visitBytes = Double(K * layerBytes / E)
var shipOut: [UInt16]? = nil
print("| variant | threads gu / dn | TGs gu / dn | max threads/TG gu / dn | visit us | GB/s | gate+up us | down us | max abs vs shipped (of max |out|) |")
print("|---|---|---|---|---:|---:|---:|---:|---|")
for name in names {
    let path = "\(vdir)/\(name).metal"
    let head = (try! String(contentsOfFile: path, encoding: .utf8)).split(separator: "\n", maxSplits: 1)[0]
    // "// variant NAME: SG s LG l NB n LD d NBD m [THR t]"
    let w = head.split(separator: " ").map(String.init)
    func val(_ k: String) -> Int { Int(w[w.firstIndex(of: k)! + 1])! }
    let (sg, lg, ld) = (val("SG"), val("LG"), val("LD"))
    let dthr = w.contains("THR") ? val("THR") : 128
    let lib = try! dev.makeLibrary(source: try! String(contentsOfFile: path, encoding: .utf8), options: { let o = MTLCompileOptions(); o.languageVersion = .version3_1; return o }())
    let pgk = try! dev.makeComputePipelineState(function: lib.makeFunction(name: "ink_moe_gu_nv_a_x")!)
    let pdk = try! dev.makeComputePipelineState(function: lib.makeFunction(name: "ink_moe_dn_f_nv_a_bf16")!)
    let guRows = sg * 32 / lg, rd = 32 / ld
    let ggu = MTLSize(width: (I + guRows - 1) / guRows, height: K, depth: 1)
    let gdn = MTLSize(width: 1, height: (H + rd - 1) / rd, depth: 1)
    func gu(_ e: MTLComputeCommandEncoder, _ c: Int) {
        let l = c % LAYERS
        e.setComputePipelineState(pgk)
        e.setBuffer(xb, offset: 0, index: 0); e.setBuffer(ib, offset: c * K * 4, index: 1); e.setBuffer(wb, offset: c * K * 4, index: 2)
        e.setBuffer(wbuf, offset: off(l, 0), index: 3); e.setBuffer(wbuf, offset: off(l, 1), index: 4); e.setBuffer(gb, offset: 0, index: 5)
        e.setBuffer(hb, offset: 0, index: 6); e.setBytes(&pgu, length: 64, index: 7); e.setBuffer(occb, offset: 0, index: 8)
        e.dispatchThreadgroups(ggu, threadsPerThreadgroup: MTLSize(width: sg * 32, height: 1, depth: 1))
    }
    func dn(_ e: MTLComputeCommandEncoder, _ c: Int) {
        let l = c % LAYERS
        e.setComputePipelineState(pdk)
        e.setBuffer(hb, offset: 0, index: 0); e.setBuffer(ib, offset: c * K * 4, index: 1); e.setBuffer(wb, offset: c * K * 4, index: 2)
        e.setBuffer(wbuf, offset: off(l, 2), index: 3); e.setBuffer(wbuf, offset: off(l, 3), index: 4); e.setBuffer(gb, offset: 0, index: 5)
        e.setBuffer(ob, offset: (c % 4) * H * 2, index: 6); e.setBytes(&pdn, length: 64, index: 7); e.setBuffer(occb, offset: 0, index: 8)
        e.dispatchThreadgroups(gdn, threadsPerThreadgroup: MTLSize(width: dthr, height: 1, depth: 1))
    }
    // numerics: 4 visits, then compare
    _ = run { e in for c in 0..<4 { gu(e, c); dn(e, c) } }
    let o = Array(UnsafeBufferPointer(start: ob.contents().bindMemory(to: UInt16.self, capacity: 4 * H), count: 4 * H))
    var diff = "-"
    if let s = shipOut {
        var m: Float = 0, mo: Float = 0
        for i in 0..<(4 * H) { m = max(m, abs(Float(bitPattern: UInt32(o[i]) << 16) - Float(bitPattern: UInt32(s[i]) << 16)))
                               mo = max(mo, abs(Float(bitPattern: UInt32(s[i]) << 16))) }
        diff = m == 0 ? "0 (bitwise)" : String(format: "%.3g (of %.3g)", m, mo)
    } else {
        shipOut = o
        let nz = o.filter { ($0 & 0x7fff) != 0 }.count, nf = o.filter { ($0 & 0x7f80) == 0x7f80 }.count
        diff = "reference (\(nz) of \(4 * H) nonzero, \(nf) inf / nan)"
    }
    for _ in 0..<2 { _ = run { e in for c in 0..<calls { gu(e, c); dn(e, c) } } }  // warm
    let both = median { run { e in for c in 0..<calls { gu(e, c); dn(e, c) } } / Double(calls) * 1e6 }
    let g1 = median { run { e in for c in 0..<calls { gu(e, c) } } / Double(calls) * 1e6 }
    let d1 = median { run { e in for c in 0..<calls { dn(e, c) } } / Double(calls) * 1e6 }
    if w.contains("OCC") {
        let oc = occb.contents().bindMemory(to: UInt32.self, capacity: 4)
        var mg: UInt32 = 0, md: UInt32 = 0
        for c in 0..<20 { oc[0] = 0; oc[1] = 0; _ = run { e in gu(e, c) }; mg = max(mg, oc[1]); oc[0] = 0; oc[1] = 0; _ = run { e in dn(e, c) }; md = max(md, oc[1]) }
        print("OCC \(name): max threadgroups in flight gu \(mg) of \(ggu.width * K), dn \(md) of \(gdn.height)")
    }
    print(String(format: "| %@ | %d / %d | %d / %d | %d / %d | %.2f | %.1f | %.2f | %.2f | %@ |", name, sg * 32, dthr, ggu.width * K, gdn.height,
                 pgk.maxTotalThreadsPerThreadgroup, pdk.maxTotalThreadsPerThreadgroup, both, visitBytes / (both * 1e-6) / 1e9, g1, d1, diff))
}
