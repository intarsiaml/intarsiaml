// Programmatic dependent launch (PDL) for the Inkling-2 attention decode edges (lane ink-cuda-attn, brief 50).
//
// griddepcontrol (sm_90+): no-ops for a grid launched without the programmatic-serialization attribute. The same two
// instructions as ink_moe.cu's pdl_trigger / pdl_wait, which are file-local there and stay unchanged.
//
// The rule every PDL kernel here keeps: before griddepcontrol.wait a kernel reads only data written by kernels already
// known to be finished when it starts, and only through prefetch.global.L2 (a cache hint that returns no data, so it
// cannot change a value even if its range were wrong). Every block calls the wait before it exits or reads anything
// else, so a grid completes only after its primary did: completion stays transitive along a chain of PDL launches.
#pragma once

#include <cuda_runtime.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

namespace inkpdl {

__device__ __forceinline__ void trigger() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
#endif
}
__device__ __forceinline__ void wait() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
}
// the 128-byte lines of [p, p + bytes) into L2, lines i0, i0 + step, ... (a hint: no value reaches the thread)
__device__ __forceinline__ void l2_lines(const void* p, int64_t bytes, int i0, int step) {
    const char* c = (const char*)((uintptr_t)p & ~(uintptr_t)127);
    const int64_t n = ((const char*)p + bytes - c + 127) / 128;
    for (int64_t i = i0; i < n; i += step) asm volatile("prefetch.global.L2 [%0];" ::"l"(c + i * 128));
}

// INTARSIA_INK_ATTN_PDL: unset / anything else: on (PDL launches + the pre-wait L2 prefetches); 0 / off / false: off (plain
// launches, the old schedule); launch: PDL launches without the prefetches (diagnostic A/B). Read once.
enum Mode { OFF = 0, ON = 1, LAUNCH = 2 };
inline int mode() {
    static int m = -1;
    if (m < 0) {
        const char* e = getenv("INTARSIA_INK_ATTN_PDL");
        if (e && (strcmp(e, "0") == 0 || strcmp(e, "off") == 0 || strcmp(e, "false") == 0)) m = OFF;
        else if (e && strcmp(e, "launch") == 0) m = LAUNCH;
        else m = ON;
    }
    return m;
}

// cudaLaunchKernelEx with the programmatic-stream-serialization attribute
template <typename... KArgs, typename... Args>
cudaError_t launch(void (*k)(KArgs...), dim3 grid, dim3 block, size_t smem, cudaStream_t st, Args&&... args) {
    cudaLaunchConfig_t cfg{};
    cfg.gridDim = grid;
    cfg.blockDim = block;
    cfg.dynamicSmemBytes = smem;
    cfg.stream = st;
    cudaLaunchAttribute at[1];
    at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    at[0].val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = at;
    cfg.numAttrs = 1;
    return cudaLaunchKernelEx(&cfg, k, static_cast<KArgs>(args)...);
}

}  // namespace inkpdl
