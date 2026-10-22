#include "ik_runtime.h"
#include "ik_internal.h"
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <string.h>
#include <stdlib.h>
#include <map>
#include <mutex>

namespace {
// live ik_rt_malloc allocations (base -> bytes): matmul_fp8_gemm_cublas keys its bf16 weight copies by the allocation a
// weight lives in and drops them when that allocation is freed (lane S1)
std::mutex g_alloc_mu;
std::map<uintptr_t, size_t> g_allocs;
// only with the opt-in cuBLAS fp8 path on: the default path keeps plain cudaMalloc / cudaFree
bool track_allocs() {
    static const bool on = [] {
        const char* e = getenv("INTARSIA_CUDA_FP8_CUBLAS");
        return e && strcmp(e, "1") == 0;
    }();
    return on;
}
}  // namespace

extern "C" {

int ik_rt_device_count(int* n) { return (int)cudaGetDeviceCount(n); }

int ik_rt_set_device(int ordinal) { return (int)cudaSetDevice(ordinal); }

int ik_rt_mem_info(uint64_t* free_bytes, uint64_t* total_bytes) {
    size_t f = 0, t = 0;
    cudaError_t e = cudaMemGetInfo(&f, &t);
    *free_bytes = f;
    *total_bytes = t;
    return (int)e;
}

int ik_rt_init_device(int ordinal, ik_device_info* info) {
    cudaError_t e = cudaSetDevice(ordinal);
    if (e != cudaSuccess) return (int)e;
    memset(info, 0, sizeof(*info));
    cudaDeviceProp p;
    e = cudaGetDeviceProperties(&p, ordinal);
    if (e != cudaSuccess) return (int)e;
    memcpy(info->name, p.name, sizeof(p.name) < sizeof(info->name) ? sizeof(p.name) : sizeof(info->name) - 1);
    info->major = p.major;
    info->minor = p.minor;
    info->integrated = p.integrated;
    info->sm_count = p.multiProcessorCount;
    int v = 0;
    if (cudaDeviceGetAttribute(&v, cudaDevAttrMemoryClockRate, ordinal) == cudaSuccess) info->mem_clock_khz = v;
    if (cudaDeviceGetAttribute(&v, cudaDevAttrGlobalMemoryBusWidth, ordinal) == cudaSuccess) info->bus_width_bits = v;
    cudaGetLastError();  // clear any sticky "unsupported attribute" error
    e = cudaFree(0);     // force context creation
    if (e != cudaSuccess) return (int)e;
    e = cudaMemGetInfo((size_t*)&info->free_bytes, (size_t*)&info->total_bytes);
    if (e != cudaSuccess) return (int)e;
    if (ik_cublas_init(ordinal) != 0) return (int)cudaErrorInitializationError;
    return 0;
}

int ik_rt_malloc(void** p, size_t bytes) {
    const cudaError_t e = cudaMalloc(p, bytes);
    if (e == cudaSuccess && track_allocs()) {
        std::lock_guard<std::mutex> lock(g_alloc_mu);
        g_allocs[(uintptr_t)*p] = bytes;
    }
    return (int)e;
}
int ik_rt_free(void* p) {
    if (track_allocs()) {
        ik_fp8_bf16_cache_forget(p);
        std::lock_guard<std::mutex> lock(g_alloc_mu);
        g_allocs.erase((uintptr_t)p);
    }
    return (int)cudaFree(p);
}
int ik_rt_alloc_base(const void* p, uintptr_t* base) {
    std::lock_guard<std::mutex> lock(g_alloc_mu);
    auto it = g_allocs.upper_bound((uintptr_t)p);
    if (it == g_allocs.begin()) return 1;
    --it;
    if ((uintptr_t)p >= it->first + it->second) return 1;
    *base = it->first;
    return 0;
}

int ik_rt_memcpy_h2d(void* dst, const void* src, size_t n, void* stream) {
    return (int)cudaMemcpyAsync(dst, src, n, cudaMemcpyHostToDevice, (cudaStream_t)stream);
}
int ik_rt_memcpy_d2h(void* dst, const void* src, size_t n, void* stream) {
    cudaError_t e = cudaMemcpyAsync(dst, src, n, cudaMemcpyDeviceToHost, (cudaStream_t)stream);
    if (e != cudaSuccess) return (int)e;
    return (int)cudaStreamSynchronize((cudaStream_t)stream);
}
int ik_rt_memcpy_d2d(void* dst, const void* src, size_t n, void* stream) {
    return (int)cudaMemcpyAsync(dst, src, n, cudaMemcpyDeviceToDevice, (cudaStream_t)stream);
}
int ik_rt_memset(void* dst, int value, size_t n, void* stream) {
    return (int)cudaMemsetAsync(dst, value, n, (cudaStream_t)stream);
}

int ik_rt_stream_create(void** stream) {
    cudaStream_t s = nullptr;
    cudaError_t e = cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
    *stream = (void*)s;
    return (int)e;
}
int ik_rt_stream_sync(void* stream) { return (int)cudaStreamSynchronize((cudaStream_t)stream); }

int ik_rt_measure_bandwidth(double* gbs) {
    const size_t bytes = 128u << 20;
    void *a = nullptr, *b = nullptr;
    cudaError_t e = cudaMalloc(&a, bytes);
    if (e != cudaSuccess) return (int)e;
    e = cudaMalloc(&b, bytes);
    if (e != cudaSuccess) {
        cudaFree(a);
        return (int)e;
    }
    cudaMemset(a, 1, bytes);
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);
    cudaMemcpy(b, a, bytes, cudaMemcpyDeviceToDevice);  // warm up
    const int reps = 8;
    cudaEventRecord(t0, 0);
    for (int i = 0; i < reps; ++i) cudaMemcpyAsync(b, a, bytes, cudaMemcpyDeviceToDevice, 0);
    cudaEventRecord(t1, 0);
    e = cudaEventSynchronize(t1);
    float ms = 0.f;
    cudaEventElapsedTime(&ms, t0, t1);
    cudaEventDestroy(t0);
    cudaEventDestroy(t1);
    cudaFree(a);
    cudaFree(b);
    // a copy reads and writes `bytes`; report the traffic a read-only stream would see (the figure
    // BF16 decode is bounded by): total moved bytes / time.
    *gbs = ms > 0.f ? (2.0 * bytes * reps) / (ms * 1e-3) / 1e9 : 0.0;
    return (int)e;
}

int ik_rt_graph_begin(void* stream) {
    return (int)cudaStreamBeginCapture((cudaStream_t)stream, cudaStreamCaptureModeThreadLocal);
}
int ik_rt_graph_end(void* stream, void** exec) {
    *exec = nullptr;
    cudaGraph_t g = nullptr;
    cudaError_t e = cudaStreamEndCapture((cudaStream_t)stream, &g);  // also ends an invalidated capture
    if (e != cudaSuccess) {
        if (g) cudaGraphDestroy(g);
        cudaGetLastError();  // clear the sticky capture error: the stream is usable again
        return (int)e;
    }
    cudaGraphExec_t ge = nullptr;
    e = cudaGraphInstantiate(&ge, g, 0);
    cudaGraphDestroy(g);
    if (e != cudaSuccess) return (int)e;
    *exec = (void*)ge;
    return 0;
}
int ik_rt_graph_launch(void* exec, void* stream) { return (int)cudaGraphLaunch((cudaGraphExec_t)exec, (cudaStream_t)stream); }
int ik_rt_graph_destroy(void* exec) { return (int)cudaGraphExecDestroy((cudaGraphExec_t)exec); }

const char* ik_rt_error_string(int code) { return cudaGetErrorString((cudaError_t)code); }

}  // extern "C"

__global__ static void ik_f32_to_bf16(const float* __restrict__ s, __nv_bfloat16* __restrict__ d, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) d[i] = __float2bfloat16_rn(s[i]);
}
__global__ static void ik_f32_to_f16(const float* __restrict__ s, __half* __restrict__ d, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) d[i] = __float2half_rn(s[i]);
}

extern "C" int ik_rt_f32_round(const void* src, void* dst, size_t n, int dst_dtype, void* stream) {
    if (n == 0) return 0;
    cudaStream_t st = (cudaStream_t)stream;
    if (dst_dtype == 2) return (int)cudaMemcpyAsync(dst, src, n * 4, cudaMemcpyDeviceToDevice, st);
    unsigned blocks = (unsigned)((n + 255) / 256);
    if (dst_dtype == 0) {
        ik_f32_to_bf16<<<blocks, 256, 0, st>>>((const float*)src, (__nv_bfloat16*)dst, n);
    } else if (dst_dtype == 1) {
        ik_f32_to_f16<<<blocks, 256, 0, st>>>((const float*)src, (__half*)dst, n);
    } else {
        return (int)cudaErrorInvalidValue;
    }
    return (int)cudaGetLastError();
}
