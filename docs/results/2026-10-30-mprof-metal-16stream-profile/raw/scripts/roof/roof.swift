import Metal
import Foundation

let dev = MTLCreateSystemDefaultDevice()!
let lib = try! dev.makeLibrary(URL: URL(fileURLWithPath: CommandLine.arguments[1]))
let q = dev.makeCommandQueue()!
func pso(_ n: String) -> MTLComputePipelineState { try! dev.makeComputePipelineState(function: lib.makeFunction(name: n)!) }
func run(_ p: MTLComputePipelineState, _ bufs: [MTLBuffer], _ u32: [UInt32], threads: Int, tg: Int) -> Double {
    let cb = q.makeCommandBuffer()!
    let e = cb.makeComputeCommandEncoder()!
    e.setComputePipelineState(p)
    var i = 0
    for b in bufs { e.setBuffer(b, offset: 0, index: i); i += 1 }
    for var v in u32 { e.setBytes(&v, length: 4, index: i); i += 1 }
    e.dispatchThreads(MTLSize(width: threads, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
    e.endEncoding(); cb.commit(); cb.waitUntilCompleted()
    return cb.gpuEndTime - cb.gpuStartTime
}
func report(_ name: String, _ v: [Double], _ unit: String) {
    let s = v.sorted()
    print(String(format: "%@: max %.1f  median %.1f  min %.1f %@", name, s.last!, s[s.count / 2], s.first!, unit))
}
print("device:", dev.name, "recommendedMaxWorkingSet GB:", Double(dev.recommendedMaxWorkingSetSize) / 1e9)
// DRAM read bandwidth: buffers of 0.5 / 1 / 2 / 3 GiB (SLC cannot hold them)
for gib in [0.5, 1.0, 2.0, 3.0] {
    let bytes = Int(gib * 1073741824)
    let src = dev.makeBuffer(length: bytes, options: .storageModePrivate)!
    let out = dev.makeBuffer(length: 4096, options: .storageModeShared)!
    // touch it so pages are resident
    let fill = q.makeCommandBuffer()!; let bl = fill.makeBlitCommandEncoder()!; bl.fill(buffer: src, range: 0..<bytes, value: 7); bl.endEncoding(); fill.commit(); fill.waitUntilCompleted()
    let p = pso("bw_read")
    var v: [Double] = []
    for rep in 0..<12 {
        let t = run(p, [src, out], [UInt32(bytes / 16)], threads: 1 << 20, tg: 256)
        if rep >= 2 { v.append(Double(bytes) / t / 1e9) }
    }
    report(String(format: "read %.1f GiB (GB/s)", gib), v, "GB/s")
}
let out = dev.makeBuffer(length: (1 << 22) * 4, options: .storageModeShared)!
do { // fp32 FMA
    let p = pso("fma_f32"); var v: [Double] = []
    let threads = 1 << 20, iters = 8192
    for rep in 0..<10 { let t = run(p, [out], [UInt32(iters)], threads: threads, tg: 256); if rep >= 2 { v.append(Double(threads) * Double(iters) * 8 * 2 / t / 1e12) } }
    report("fp32 FMA", v, "TFLOPS")
}
for (n, name) in [("mma_f32", "simdgroup float8x8 MMA (f32 in, f32 acc)"), ("mma_f16", "simdgroup half8x8 MMA (f16 in, f32 acc)")] {
    let p = pso(n); var v: [Double] = []
    let threads = 1 << 20, iters = 8192
    for rep in 0..<10 { let t = run(p, [out], [UInt32(iters)], threads: threads, tg: 256); if rep >= 2 { v.append(Double(threads / 32) * Double(iters) * 4 * 1024 / t / 1e12) } }
    report(name, v, "TFLOPS")
}
