// Dependent-dispatch chain vs in-dispatch ticket-ordered stages (spin-wait on lower tickets), with a correctness check.
import Metal
import Foundation

let src = """
#include <metal_stdlib>
using namespace metal;
// one stage of a chain: NTG threadgroups, each reads the previous stage's buffer (NTG * 1024 floats) at its slice and the
// first element of every slice, writes its slice of the next.
kernel void stage(device const float* prev [[buffer(0)]], device float* next [[buffer(1)]], constant uint& ntg [[buffer(2)]],
                  uint tg [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]]) {
    float s = 0;
    for (uint j = 0; j < ntg; ++j) s += prev[j * 1024];
    next[tg * 1024 + tid] = prev[tg * 1024 + tid] * 0.5f + s * 1e-3f + 1.0f;
}
// all stages in one dispatch: ticket t (atomic at TG start) is stage t / ntg, slice t % ntg; a stage waits until the previous
// stage's counter reaches ntg (all its TGs have lower tickets, so they are running or done)
kernel void flow(device float* bufs [[buffer(0)]], device atomic_uint* ctr [[buffer(1)]], constant uint& ntg [[buffer(2)]],
                 constant uint& nst [[buffer(3)]],
                 uint tid [[thread_position_in_threadgroup]]) {
    threadgroup uint tk;
    if (tid == 0) tk = atomic_fetch_add_explicit(&ctr[0], 1u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint t = tk, st = t / ntg, sl = t % ntg;
    if (st > 0) {
        if (tid == 0) {
            while (atomic_load_explicit(&ctr[st], memory_order_relaxed) < ntg) {}
            atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
        }
        threadgroup_barrier(mem_flags::mem_device);
    }
    device const float* prev = bufs + (ulong)st * ntg * 1024;
    device float* next = bufs + (ulong)(st + 1) * ntg * 1024;
    float s = 0;
    for (uint j = 0; j < ntg; ++j) s += prev[j * 1024];
    next[sl * 1024 + tid] = prev[sl * 1024 + tid] * 0.5f + s * 1e-3f + 1.0f;
    threadgroup_barrier(mem_flags::mem_device);
    if (tid == 0) {
        atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
        uint old = atomic_fetch_add_explicit(&ctr[st + 1], 1u, memory_order_relaxed);
        // the very last TG resets the counters for the next dispatch
        if (st + 1 == nst && old == ntg - 1) { for (uint j = 0; j <= nst; ++j) atomic_store_explicit(&ctr[j], 0u, memory_order_relaxed); }
    }
}
"""

let dev = MTLCreateSystemDefaultDevice()!
let opts = MTLCompileOptions(); opts.languageVersion = .version3_2
let lib = try! dev.makeLibrary(source: src, options: opts)
let pStage = try! dev.makeComputePipelineState(function: lib.makeFunction(name: "stage")!)
let pFlow = try! dev.makeComputePipelineState(function: lib.makeFunction(name: "flow")!)
let q = dev.makeCommandQueue()!

func run(_ body: (MTLComputeCommandEncoder) -> Void) -> Double {
    let cb = q.makeCommandBuffer()!
    let e = cb.makeComputeCommandEncoder()!
    body(e)
    e.endEncoding(); cb.commit(); cb.waitUntilCompleted()
    return cb.gpuEndTime - cb.gpuStartTime
}

for ntg in [1, 8, 24] {
    let nst = 8
    let n = ntg * 1024
    let bufs = dev.makeBuffer(length: (nst + 1) * n * 4, options: .storageModeShared)!
    let ctr = dev.makeBuffer(length: 64 * 4, options: .storageModeShared)!
    memset(ctr.contents(), 0, 64 * 4)
    func initb() { let p = bufs.contents().bindMemory(to: Float.self, capacity: (nst + 1) * n); for i in 0..<n { p[i] = Float(i % 7) }; for i in n..<((nst + 1) * n) { p[i] = -1 } }
    // CPU reference
    var ref = [Float](repeating: 0, count: (nst + 1) * n)
    for i in 0..<n { ref[i] = Float(i % 7) }
    for st in 0..<nst { var s: Float = 0; for j in 0..<ntg { s += ref[st * n + j * 1024] }
        for i in 0..<n { ref[(st + 1) * n + i] = ref[st * n + i] * 0.5 + s * 1e-3 + 1.0 } }
    var ntg32 = UInt32(ntg), nst32 = UInt32(nst)
    // chain of dispatches, repeated R times per command buffer
    let R = 50
    var tc: [Double] = [], tf: [Double] = []
    var bad = 0
    for _ in 0..<9 {
        initb()
        let t = run { e in
            e.setComputePipelineState(pStage)
            for _ in 0..<R { for st in 0..<nst {
                e.setBuffer(bufs, offset: st * n * 4, index: 0); e.setBuffer(bufs, offset: (st + 1) * n * 4, index: 1)
                e.setBytes(&ntg32, length: 4, index: 2)
                e.dispatchThreadgroups(MTLSize(width: ntg, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 1024, height: 1, depth: 1)) } } }
        tc.append(t / Double(R * nst) * 1e6)
        initb()
        let t2 = run { e in
            e.setComputePipelineState(pFlow)
            e.setBuffer(bufs, offset: 0, index: 0); e.setBuffer(ctr, offset: 0, index: 1)
            e.setBytes(&ntg32, length: 4, index: 2); e.setBytes(&nst32, length: 4, index: 3)
            for _ in 0..<R { e.dispatchThreadgroups(MTLSize(width: ntg * nst, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 1024, height: 1, depth: 1)) } }
        tf.append(t2 / Double(R * nst) * 1e6)
        let p = bufs.contents().bindMemory(to: Float.self, capacity: (nst + 1) * n)
        for i in 0..<((nst + 1) * n) where abs(p[i] - ref[i]) > 1e-3 * max(1, abs(ref[i])) { bad += 1 }
    }
    tc.sort(); tf.sort()
    print(String(format: "ntg %2d: dependent dispatch stage %.2f us, in-dispatch stage %.2f us, flow mismatches %d", ntg, tc[4], tf[4], bad))
}
