import Metal
let dev = MTLCreateSystemDefaultDevice()!
let lib = try! dev.makeLibrary(URL: URL(fileURLWithPath: CommandLine.arguments[1]))
let fn = lib.makeFunction(name: CommandLine.arguments.count > 2 ? CommandLine.arguments[2] : "probe")!
let pso = try! dev.makeComputePipelineState(function: fn)
let n = 16 * 32 * 64
let buf = dev.makeBuffer(length: n * 4, options: .storageModeShared)!
memset(buf.contents(), 0xee, n * 4)
let q = dev.makeCommandQueue()!
let cb = q.makeCommandBuffer()!
let e = cb.makeComputeCommandEncoder()!
e.setComputePipelineState(pso); e.setBuffer(buf, offset: 0, index: 0)
e.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
e.endEncoding(); cb.commit(); cb.waitUntilCompleted()
let p = buf.contents().bindMemory(to: UInt32.self, capacity: n)
for w in 0..<11 {
  var lines: [String] = []
  for lane in 0..<32 {
    var s = "w\(w) lane\(lane):"
    for i in 0..<64 { let v = p[(w * 32 + lane) * 64 + i]; if v == 0xffffffff || v == 0xeeeeeeee { break }; s += " \(v >> 8 & 0xff),\(v & 0xff)" }
    lines.append(s)
  }
  print(lines.joined(separator: "\n"))
}
