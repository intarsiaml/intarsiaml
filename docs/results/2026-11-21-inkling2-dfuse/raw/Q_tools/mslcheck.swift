// Runtime-compile check of the probe sources for every macro set (syntax / type check; lane ink-dfuse probe).
import Metal
import Foundation
let dir = CommandLine.arguments[1]
let dev = MTLCreateSystemDefaultDevice()!
let sets: [(String, String, [String: NSNumber])] = [
    ("ship", "ship.metal", [:]), ("p0", "probe.metal", [:]), ("p_nopro", "probe.metal", ["PR_NOPRO": 1]),
    ("p_nomath", "probe.metal", ["PR_NOMATH": 1]), ("p_noclip", "probe.metal", ["PR_NOCLIP": 1]), ("p_noconv", "probe.metal", ["PR_NOCONV": 1]), ("p_nobias", "probe.metal", ["PR_NOBIAS": 1]), ("p_skel", "probe.metal", ["PR_NOPRO": 1, "PR_NOMATH": 1]),
    ("p_la1", "probe.metal", ["PR_NB": 1]), ("p_la2", "probe.metal", ["PR_NB": 2]), ("p_la4", "probe.metal", ["PR_NB": 4]),
    ("p_early2", "probe.metal", ["PR_NB": 2, "PR_EARLY": 1]), ("p_early4", "probe.metal", ["PR_NB": 4, "PR_EARLY": 1])]
var bad = 0
for (n, f, m) in sets {
    let o = MTLCompileOptions(); o.languageVersion = .version3_1; o.preprocessorMacros = m
    if #available(macOS 15.0, *) { o.mathMode = .safe } else { o.fastMathEnabled = false }
    do {
        let l = try dev.makeLibrary(source: try String(contentsOfFile: "\(dir)/\(f)", encoding: .utf8), options: o)
        for fn in ["ink_attn_dec_g3", "ink_attn_comb_bf16"] + (f == "probe.metal" ? ["pr_bytes", "pr_floor", "pr_fill"] : []) {
            guard let fx = l.makeFunction(name: fn) else { print("\(n): no \(fn)"); bad += 1; continue }
            _ = try dev.makeComputePipelineState(function: fx)
        }
        print("\(n): ok")
    } catch { print("\(n): FAIL \(error)"); bad += 1 }
}
print(bad == 0 ? "MSL OK" : "MSL FAIL \(bad)")
