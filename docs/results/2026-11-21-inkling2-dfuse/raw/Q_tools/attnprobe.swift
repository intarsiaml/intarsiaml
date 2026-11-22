// Decode attention-core probe (lane ink-dfuse, measurement only; nothing here is shipped).
// The fused InkAttnDecode visit at the real decode shapes (H 24 / Hk 8, D 128, d_rel 16, E 1024, K 4; SWA window 1024 on a
// 1040-row ring, full layers with dynamic conv + tau), launched as crates/intarsia-backend-metal/src/inkattn.rs
// `launch_ink_decode` does at T = 1: ink_attn_dec_g3 on (Hk, splits, 1) x 128 threads, then ink_attn_comb_bf16 on (H, 1)
// when there is more than one split, on one serial compute encoder.
//   cal: the shipped copy against the backend's dump (ink_attn_dump.rs): out, caches and rings bitwise.
//   warm16: 16 operand sets of one kind, 8 passes per command buffer (roofline L's bench setting): us per visit.
//   token: 40 SWA + 8 full visits per token (layer i full when i % 6 == 5), NT distinct token sets (working set past the
//          SLC, so K / V arrive cold as in the stream), 2 passes per command buffer: ms per token.
// Every number is the median over 7 rounds of one command buffer's GPU time; variants rotate order each round.
// usage: attnprobe LIB_DIR DUMP_DIR
import Metal
import Foundation

let args = CommandLine.arguments
let libDir = args[1], dumpDir = args[2]
let dev = MTLCreateSystemDefaultDevice()!
let queue = dev.makeCommandQueue()!
let H = 24, HK = 8, D = 128, DR = 16, E = 1024, K = 4, RR = K - 1 + 16, GP = 3
let CKV = HK * D
let NQ = 2 * H * D + H * DR + H
let MAXS = 32
let SCR = H * MAXS * (D + 2) * 4 + 2 * HK * D * 2

func lib(_ name: String) -> MTLLibrary { try! dev.makeLibrary(URL: URL(fileURLWithPath: "\(libDir)/\(name).metallib")) }
func pso(_ l: MTLLibrary, _ f: String) -> MTLComputePipelineState { try! dev.makeComputePipelineState(function: l.makeFunction(name: f)!) }

struct Var {
    let name: String
    let dec: MTLComputePipelineState
    let comb: MTLComputePipelineState?
    let sk: Int
    let bytes: Bool
}
let shipL = lib("ship")
let fillP = pso(lib("p_skel"), "pr_fill")
let floorP = pso(lib("p_skel"), "pr_floor")
func mk(_ name: String, _ l: String, comb: Bool = true, sk: Int = 64, fn: String = "ink_attn_dec_g3") -> Var {
    let L = lib(l)
    return Var(name: name, dec: pso(L, fn), comb: comb ? pso(L, "ink_attn_comb_bf16") : nil, sk: sk, bytes: fn == "pr_bytes")
}
let ship = mk("ship", "ship")
var vars: [Var] = [ship,
                   mk("p0", "p0"),
                   mk("nocomb", "ship", comb: false),
                   mk("nopro", "p_nopro"),
                   mk("nomath", "p_nomath"),
                   mk("skel_c", "p_skel"),
                   mk("skel", "p_skel", comb: false),
                   mk("bytes", "p_skel", comb: false, fn: "pr_bytes"),
                   mk("la1", "p_la1"), mk("la2", "p_la2"), mk("la4", "p_la4"),
                   mk("early2", "p_early2"), mk("early4", "p_early4"),
                   mk("sk32", "ship", sk: 32), mk("sk128", "ship", sk: 128)]
let only = ProcessInfo.processInfo.environment["PROBE_VARS"].map { Set($0.split(separator: ",").map(String.init)) }
if let o = only { vars = vars.filter { o.contains($0.name) } }
let sameBits: Set<String> = ["p0", "la1", "la2", "la4", "early2", "early4"]

// ---- launch parameters (inkattn.rs launch_ink_decode at T = 1, host position) ----
func up32(_ x: Int) -> Int { (x + 31) / 32 * 32 }
func nsplit(_ v: Int, _ sk: Int) -> Int { min(max((v + sk - 1) / sk, 1), MAXS) }
struct Geo { var words: [UInt32]; var gy: Int; var tb: Int }
func geo(full: Bool, ctx: Int, cr: Int, sk: Int) -> Geo {
    let window = full ? 0 : 1024, ring = !full
    let last = ctx
    let lo = window > 0 ? max(0, last + 1 - window) : 0
    let vmax = last - lo + 1
    let gy = nsplit(vmax, sk)
    let chunk = up32((vmax + gy - 1) / gy)
    let brows = min(max(chunk, up32(sk)), E)
    let tb = (max(GP * brows * 4, 16) + 15) / 16 * 16
    let kvFused = !ring || 1 + window <= cr + 1
    let ringFused = 1 + K <= RR + 1
    var w = [UInt32](repeating: 0, count: 48)
    let f = { (x: Float) -> UInt32 in x.bitPattern }
    w[0] = 1; w[1] = UInt32(H); w[2] = UInt32(HK); w[3] = UInt32(D); w[4] = UInt32(DR); w[5] = UInt32(E); w[6] = UInt32(K)
    w[7] = UInt32(RR); w[8] = UInt32(cr); w[9] = UInt32(ctx); w[10] = 0
    w[11] = (full ? 1 : 0) | (ring ? 2 : 0) | (kvFused ? 4 : 0) | (ringFused ? 16 : 0)
    w[12] = full ? 1024 : 0; w[13] = UInt32(window); w[14] = UInt32(NQ); w[15] = UInt32(full ? CKV + HK * K : CKV)
    w[16] = f(22.6); w[17] = f(2.0); w[18] = f(1e-6); w[19] = f(full ? 0.1 : 0.0)
    w[20] = f(powf(Float(D), -0.5)); w[21] = f(full ? powf(Float(2 * K + 1), -0.5) : powf(Float(K + 1), -0.5)); w[22] = 0
    w[23] = UInt32(sk)
    return Geo(words: w, gy: gy, tb: tb)
}

// ---- one visit's operands ----
final class Visit {
    let full: Bool, ctx: Int, cr: Int
    let pq, pk, pv, kt, vt, qs, rs, rp, kc, vc, out, kr, vr: MTLBuffer
    init(full: Bool, ctx: Int, pq: MTLBuffer, pk: MTLBuffer, pv: MTLBuffer, opt: MTLResourceOptions, bufs: [String: MTLBuffer]? = nil) {
        self.full = full; self.ctx = ctx; cr = full ? ctx + 17 : 1040
        func b(_ n: Int, _ name: String) -> MTLBuffer { bufs?[name] ?? dev.makeBuffer(length: n * 2 + 64, options: opt)! }
        self.pq = bufs?["pq"] ?? pq; self.pk = bufs?["pk"] ?? pk; self.pv = bufs?["pv"] ?? pv
        kt = b(K * CKV, "kt"); vt = b(K * CKV, "vt"); qs = b(H * D, "qs"); rs = b(H * DR, "rs"); rp = b(DR * E, "rp")
        kc = b(HK * cr * D, "kc"); vc = b(HK * cr * D, "vc"); out = bufs?["out0"] ?? dev.makeBuffer(length: H * D * 2 + 64, options: .storageModeShared)!
        kr = b(RR * CKV, "kr"); vr = b(RR * CKV, "vr")
    }
}
var seedN: UInt32 = 1
func fill(_ e: MTLComputeCommandEncoder, _ b: MTLBuffer) {
    let n = (b.length - 64) / 2
    var pm: (UInt32, UInt32) = (UInt32(n), seedN); seedN &+= 1
    e.setComputePipelineState(fillP); e.setBuffer(b, offset: 0, index: 0); e.setBytes(&pm, length: 8, index: 1)
    e.dispatchThreads(MTLSize(width: n, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
}
func makeVisits(_ kinds: [Bool], ctx: Int) -> [Visit] {
    let pqs = [false, true].map { full -> (MTLBuffer, MTLBuffer, MTLBuffer) in
        let pw = full ? CKV + HK * K : CKV
        return (dev.makeBuffer(length: NQ * 2 + 64, options: .storageModePrivate)!, dev.makeBuffer(length: pw * 2 + 64, options: .storageModePrivate)!,
                dev.makeBuffer(length: pw * 2 + 64, options: .storageModePrivate)!)
    }
    let vs = kinds.map { full -> Visit in
        let s = pqs[full ? 1 : 0]
        return Visit(full: full, ctx: ctx, pq: s.0, pk: s.1, pv: s.2, opt: .storageModePrivate)
    }
    let cb = queue.makeCommandBuffer()!, e = cb.makeComputeCommandEncoder()!
    for s in pqs { fill(e, s.0); fill(e, s.1); fill(e, s.2) }
    for v in vs { for b in [v.kt, v.vt, v.qs, v.rs, v.rp, v.kc, v.vc, v.kr, v.vr] { fill(e, b) } }
    e.endEncoding(); cb.commit(); cb.waitUntilCompleted()
    return vs
}

let scratch = dev.makeBuffer(length: SCR + 64, options: .storageModePrivate)!
let sink = dev.makeBuffer(length: HK * MAXS * 128 * 4 + 64, options: .storageModePrivate)!
func encode(_ e: MTLComputeCommandEncoder, _ v: Var, _ x: Visit) {
    let g = geo(full: x.full, ctx: x.ctx, cr: x.cr, sk: v.sk)
    var w = g.words
    e.setComputePipelineState(v.dec)
    if v.bytes {
        e.setBuffer(x.kc, offset: 0, index: 8); e.setBuffer(x.vc, offset: 0, index: 9); e.setBuffer(sink, offset: 0, index: 14)
        e.setBytes(&w, length: 192, index: 15)
        e.dispatchThreadgroups(MTLSize(width: HK, height: g.gy, depth: 1), threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
        return
    }
    let bs: [MTLBuffer] = [x.pq, x.pk, x.pv, x.kt, x.vt, x.qs, x.rs, x.rp, x.kc, x.vc, x.out, x.kr, x.vr, x.pq, scratch]
    for (i, b) in bs.enumerated() { e.setBuffer(b, offset: 0, index: i) }
    e.setBytes(&w, length: 192, index: 15)
    e.setThreadgroupMemoryLength(g.tb, index: 0)
    e.setThreadgroupMemoryLength(512, index: 1)
    e.dispatchThreadgroups(MTLSize(width: HK, height: g.gy, depth: 1), threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
    if g.gy > 1, let c = v.comb {
        e.setComputePipelineState(c)
        e.setBuffer(x.pq, offset: 0, index: 0); e.setBuffer(x.out, offset: 0, index: 1); e.setBuffer(x.pq, offset: 0, index: 2)
        e.setBuffer(scratch, offset: 0, index: 3); e.setBytes(&w, length: 192, index: 4)
        e.dispatchThreadgroups(MTLSize(width: H, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
    }
}
func run(_ v: Var, _ list: [Visit], passes: Int) -> Double {
    let cb = queue.makeCommandBuffer()!
    let e = cb.makeComputeCommandEncoder(dispatchType: .serial)!
    for _ in 0..<passes { for x in list { encode(e, v, x) } }
    e.endEncoding(); cb.commit(); cb.waitUntilCompleted()
    return (cb.gpuEndTime - cb.gpuStartTime) * 1e6
}
func median(_ a: [Double]) -> Double { let s = a.sorted(); return s[s.count / 2] }
func outBytes(_ x: Visit) -> [UInt8] { Array(UnsafeBufferPointer(start: x.out.contents().assumingMemoryBound(to: UInt8.self), count: H * D * 2)) }

// ---- cal: bitwise against the backend's dump ----
func loadBuf(_ path: String) -> MTLBuffer? {
    guard let d = FileManager.default.contents(atPath: path) else { return nil }
    let b = dev.makeBuffer(length: d.count + 64, options: .storageModeShared)!
    _ = d.withUnsafeBytes { memcpy(b.contents(), $0.baseAddress!, d.count) }
    return b
}
print("== cal: shipped copy (and p0) vs the backend's ink_attn_decode_bf16, one visit each")
var calOK = true
for (full, ctx) in [(false, 150), (true, 150), (false, 512), (true, 512), (false, 4096), (true, 4096), (false, 5180), (true, 5180)] {
    let tag = "\(full ? "full" : "swa")_\(ctx)"
    for v in [ship] + vars.filter({ $0.name == "p0" }) {
        var m: [String: MTLBuffer] = [:]
        for n in ["pq", "pk", "pv", "kt", "vt", "qs", "rs", "rp", "kc", "vc", "kr", "vr", "out0"] {
            guard let b = loadBuf("\(dumpDir)/\(tag)_\(n)") else { print("cal \(tag): missing \(n)"); exit(1) }
            m[n] = b
        }
        let x = Visit(full: full, ctx: ctx, pq: m["pq"]!, pk: m["pk"]!, pv: m["pv"]!, opt: .storageModeShared, bufs: m)
        _ = run(v, [x], passes: 1)
        var res: [String] = []
        for (n, b) in [("out", x.out), ("kc1", x.kc), ("vc1", x.vc), ("kr1", x.kr), ("vr1", x.vr)] {
            let want = FileManager.default.contents(atPath: "\(dumpDir)/\(tag)_\(n)")!
            let got = Data(bytes: b.contents(), count: want.count)
            let eq = got == want
            if !eq { calOK = false }
            res.append("\(n) \(eq ? "EQUAL" : "DIFF")")
        }
        print("cal \(tag) \(v.name): " + res.joined(separator: ", "))
    }
}
print("CAL \(calOK ? "PASS" : "FAIL")")

// ---- timing ----
func timeSet(_ title: String, _ list: [Visit], passes: Int, per: Double, unit: String) {
    for v in vars { _ = run(v, list, passes: 1) }  // warm-up, pipelines resident
    var t: [String: [Double]] = [:]
    for r in 0..<7 {
        let order: [Var] = r % 2 == 0 ? vars : Array(vars.reversed())
        let rot = Array(order[(r % order.count)...] + order[..<(r % order.count)])
        for v in rot { t[v.name, default: []].append(run(v, list, passes: passes)) }
    }
    let base = median(t["ship"] ?? [0]) / per
    for v in vars {
        let m = median(t[v.name]!) / per
        let lo = t[v.name]!.min()! / per, hi = t[v.name]!.max()! / per
        print(String(format: "%@ %-8@ %9.3f %@ (min %.3f max %.3f)  vs ship %+8.3f", title as NSString, v.name as NSString, m, unit as NSString, lo, hi, m - base))
    }
}
// floor: an empty dependent dispatch in this harness
do {
    var a: [Double] = []
    for _ in 0..<7 {
        let cb = queue.makeCommandBuffer()!, e = cb.makeComputeCommandEncoder(dispatchType: .serial)!
        e.setComputePipelineState(floorP); e.setBuffer(sink, offset: 0, index: 14)
        for _ in 0..<256 { e.dispatchThreads(MTLSize(width: 32, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1)) }
        e.endEncoding(); cb.commit(); cb.waitUntilCompleted()
        a.append((cb.gpuEndTime - cb.gpuStartTime) * 1e6 / 256)
    }
    print(String(format: "floor: empty dependent dispatch %.3f us", median(a)))
}
let which = ProcessInfo.processInfo.environment["PROBE_SETS"] ?? "warm,token"
if which.contains("warm") {
    print("== warm16: 16 operand sets of one kind, 8 passes per command buffer; us per visit")
    for (full, ctx) in [(false, 512), (true, 512), (false, 4096), (true, 4096)] {
        let vs = makeVisits(Array(repeating: full, count: 16), ctx: ctx)
        timeSet("warm16 \(full ? "full" : "swa ") ctx \(String(format: "%5d", ctx))", vs, passes: 8, per: 128, unit: "us/visit")
    }
}
if which.contains("token") {
    print("== token: 40 SWA + 8 full visits per token, NT distinct token sets, 2 passes per command buffer; ms per token")
    for (ctx, nt) in [(150, 6), (512, 4), (5180, 2)] {
        let kinds = (0..<(48 * nt)).map { ($0 % 48) % 6 == 5 }
        let vs = makeVisits(kinds, ctx: ctx)
        // same-bits check on the first SWA and the first full visit of the set
        for x in [vs[0], vs[5]] {
            _ = run(ship, [x], passes: 1)
            let ref = outBytes(x)
            var line: [String] = []
            for v in vars where v.name != "ship" && !v.bytes {
                _ = run(v, [x], passes: 1)
                let got = outBytes(x)
                if sameBits.contains(v.name) || v.name.hasPrefix("sk") {
                    var md: Float = 0
                    for i in stride(from: 0, to: got.count, by: 2) {
                        let a = Float(bitPattern: UInt32(ref[i]) << 16 | UInt32(ref[i + 1]) << 24)
                        let b = Float(bitPattern: UInt32(got[i]) << 16 | UInt32(got[i + 1]) << 24)
                        md = max(md, abs(a - b))
                    }
                    line.append("\(v.name) \(got == ref ? "EQUAL" : String(format: "DIFF(max %.3g)", md))")
                }
            }
            _ = run(ship, [x], passes: 1)
            print("bits ctx \(ctx) \(x.full ? "full" : "swa"): " + line.joined(separator: ", "))
        }
        timeSet("token ctx \(String(format: "%5d", ctx))", vs, passes: 2, per: Double(nt * 2) * 1000, unit: "ms/token")
    }
}
print("done")
