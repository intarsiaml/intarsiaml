// Dependent-dispatch cost chains (lane ink-dfuse item 2b, measurement only). The engine's setting (MetalBackend: `alloc` =
// MTLResourceStorageModeShared WITH hazard tracking, one MTLDispatchTypeSerial compute encoder, no barriers, setBytes per
// dispatch) next to dispatch_cost's (shared, untracked) and private storage / a concurrent encoder with a buffer barrier,
// so that the setting that separates 0.88 us (dispatch_cost) from 3.3 us (the attention harness) shows.
// Each chain: N dispatches of 1 threadgroup x 32 threads in one command buffer; GPU time / N, median of 9.
//   empty    no buffer bound            empty_b  a buffer bound, untouched     write  thread i writes word i
//   rmw      thread 0: a[0] += 1        add3     c = a + b, the same three buffers every time (dispatch_cost's add_bf16)
//   chain    y = x + 1 ping-pong between two buffers (each dispatch reads the previous one's output)
//   chain8   chain plus 8 more bound buffers (the real ops bind 5-16)
import Metal
import Foundation

let dev = MTLCreateSystemDefaultDevice()!
let queue = dev.makeCommandQueue()!
let src = """
#include <metal_stdlib>
using namespace metal;
kernel void k_empty(uint i [[thread_position_in_grid]]) {}
kernel void k_empty_b(device uint* a [[buffer(0)]], uint i [[thread_position_in_grid]]) {}
kernel void k_write(device uint* a [[buffer(0)]], uint i [[thread_position_in_grid]]) { a[i] = i; }
kernel void k_rmw(device uint* a [[buffer(0)]], uint i [[thread_position_in_grid]]) { if (i == 0) a[0] += 1; }
kernel void k_add3(device const ushort* a [[buffer(0)]], device const ushort* b [[buffer(1)]], device ushort* c [[buffer(2)]],
                   uint i [[thread_position_in_grid]]) { c[i] = a[i] + b[i]; }
kernel void k_chain(device const ushort* x [[buffer(0)]], device ushort* y [[buffer(1)]], uint i [[thread_position_in_grid]]) {
    y[i] = x[i] + 1;
}
"""
let opt = MTLCompileOptions(); opt.languageVersion = .version3_1
let lib = try! dev.makeLibrary(source: src, options: opt)
func pso(_ n: String) -> MTLComputePipelineState { try! dev.makeComputePipelineState(function: lib.makeFunction(name: n)!) }
let N = 1000
enum Enc { case serial, concBarrier }
var hostUs: [String: Double] = [:]  // host encode wall per dispatch (cb create -> endEncoding), same key as the GPU number
func chain(_ kernel: String, storage: MTLResourceOptions, tracked: Bool, enc: Enc) -> Double {
    var o = storage
    if !tracked { o.insert(.hazardTrackingModeUntracked) }
    let bufs = (0..<11).map { _ in dev.makeBuffer(length: 4096, options: o)! }
    let p = pso(kernel == "chain8" ? "k_chain" : "k_\(kernel)")
    var pm = [UInt32](repeating: 0, count: 16)
    var a: [Double] = [], h: [Double] = []
    for rep in 0..<11 {
        let t0 = DispatchTime.now().uptimeNanoseconds
        let cb = queue.makeCommandBuffer()!
        let e = enc == .serial ? cb.makeComputeCommandEncoder(dispatchType: .serial)! : cb.makeComputeCommandEncoder(dispatchType: .concurrent)!
        e.setComputePipelineState(p)
        for i in 0..<N {
            switch kernel {
            case "empty": break
            case "empty_b", "write", "rmw": e.setBuffer(bufs[0], offset: 0, index: 0)
            case "add3": for j in 0..<3 { e.setBuffer(bufs[j], offset: 0, index: j) }
            case "chain": e.setBuffer(bufs[i % 2], offset: 0, index: 0); e.setBuffer(bufs[1 - i % 2], offset: 0, index: 1)
            default:  // chain8
                e.setBuffer(bufs[i % 2], offset: 0, index: 0); e.setBuffer(bufs[1 - i % 2], offset: 0, index: 1)
                for j in 2..<10 { e.setBuffer(bufs[j], offset: 0, index: j) }
            }
            e.setBytes(&pm, length: 64, index: 15)
            e.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
            if enc == .concBarrier { e.memoryBarrier(scope: .buffers) }
        }
        e.endEncoding()
        let t1 = DispatchTime.now().uptimeNanoseconds
        cb.commit(); cb.waitUntilCompleted()
        if rep >= 2 { a.append((cb.gpuEndTime - cb.gpuStartTime) * 1e6 / Double(N)); h.append(Double(t1 - t0) / 1e3 / Double(N)) }
    }
    hostUs["\(kernel) \(storage == .storageModeShared ? "shared" : "private") \(tracked ? "tracked" : "untracked") \(enc)"] = h.sorted()[h.count / 2]
    return a.sorted()[a.count / 2]
}
print("device \(dev.name); \(N) dispatches of 1 tg x 32 threads per command buffer; us per dispatch, median of 9")
let cols = ["shared tracked serial*", "shared untracked serial", "private tracked serial", "private untracked serial", "shared tracked conc+bar", "shared untracked conc+bar"]
print("GPU us per dispatch")
print(("kernel  " + cols.map { $0.padding(toLength: 26, withPad: " ", startingAt: 0) }.joined()))
for k in ["empty", "empty_b", "write", "rmw", "add3", "chain", "chain8"] {
    let r = [chain(k, storage: .storageModeShared, tracked: true, enc: .serial),
             chain(k, storage: .storageModeShared, tracked: false, enc: .serial),
             chain(k, storage: .storageModePrivate, tracked: true, enc: .serial),
             chain(k, storage: .storageModePrivate, tracked: false, enc: .serial),
             chain(k, storage: .storageModeShared, tracked: true, enc: .concBarrier),
             chain(k, storage: .storageModeShared, tracked: false, enc: .concBarrier)]
    print(k.padding(toLength: 8, withPad: " ", startingAt: 0) + r.map { String(format: "%-26.3f", $0) }.joined())
}
print("host encode wall us per dispatch (command buffer create -> endEncoding)")
for key in hostUs.keys.sorted() { print(String(format: "  %@ %.3f", key as NSString, hostUs[key]!)) }
print("* the engine's setting (MetalBackend::alloc, with_encoder)")
