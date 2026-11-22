// Print max threads per threadgroup (register-limited) for the named kernels in a metallib. Usage: psoinfo <metallib> names...
import Foundation
import Metal
let a = CommandLine.arguments
let dev = MTLCreateSystemDefaultDevice()!
let lib = try! dev.makeLibrary(URL: URL(fileURLWithPath: a[1]))
for n in a.dropFirst(2) {
    guard let f = lib.makeFunction(name: n) else { print("\(n): missing"); continue }
    let p = try! dev.makeComputePipelineState(function: f)
    print("\(n): maxThreads \(p.maxTotalThreadsPerThreadgroup) simdWidth \(p.threadExecutionWidth) tgmem \(p.staticThreadgroupMemoryLength)")
}
