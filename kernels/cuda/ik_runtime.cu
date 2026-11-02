#include "ik_runtime.h"
#include "ik_internal.h"
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <map>
#include <mutex>
#include <vector>

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
// The highest-priority non-blocking stream: the batched decode lane (R3d) runs beside a prefill chunk on stream 0, and its small
// kernels must get SMs as the chunk's blocks retire instead of queueing behind the chunk's remaining grid.
int ik_rt_stream_create_high_priority(void** stream) {
    int least = 0, greatest = 0;
    cudaError_t e = cudaDeviceGetStreamPriorityRange(&least, &greatest);
    if (e != cudaSuccess) return (int)e;
    cudaStream_t s = nullptr;
    e = cudaStreamCreateWithPriority(&s, cudaStreamNonBlocking, greatest);
    *stream = (void*)s;
    return (int)e;
}
int ik_rt_stream_sync(void* stream) { return (int)cudaStreamSynchronize((cudaStream_t)stream); }
int ik_rt_event_create_sync(void** ev) {
    cudaEvent_t e = nullptr;
    cudaError_t rc = cudaEventCreateWithFlags(&e, cudaEventDisableTiming);
    *ev = (void*)e;
    return (int)rc;
}
int ik_rt_event_record(void* ev, void* stream) { return (int)cudaEventRecord((cudaEvent_t)ev, (cudaStream_t)stream); }
int ik_rt_stream_wait_event(void* stream, void* ev) { return (int)cudaStreamWaitEvent((cudaStream_t)stream, (cudaEvent_t)ev, 0); }

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

// INTARSIA_CUDA_GRAPH_EDGES=1: a line per recorded graph with its node, edge and programmatic-edge counts (does a programmatic
// launch survive the stream capture? docs/results/2026-11-02-dsv41-pdl.md).
static void report_graph_edges(cudaGraph_t g) {
    static const bool on = [] {
        const char* v = getenv("INTARSIA_CUDA_GRAPH_EDGES");
        return v != nullptr && v[0] == '1';
    }();
    if (!on) return;
    size_t nodes = 0, n = 0;
    cudaGraphGetNodes(g, nullptr, &nodes);
    cudaGraphGetEdges(g, nullptr, nullptr, nullptr, &n);
    std::vector<cudaGraphNode_t> from(n), to(n);
    std::vector<cudaGraphEdgeData> data(n);
    const cudaError_t e = cudaGraphGetEdges(g, from.data(), to.data(), data.data(), &n);
    size_t prog = 0;
    for (size_t i = 0; i < n; ++i) prog += data[i].type == cudaGraphDependencyTypeProgrammatic;
    fprintf(stderr, "cuda graph: %zu nodes, %zu edges, %zu programmatic (%s)\n", nodes, n, prog, cudaGetErrorString(e));
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
    report_graph_edges(g);
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

// out = round(src[0] + src[1] + .. + src[world-1]) per element, in rank order (src: rank-major `[world, n]` f32 as an
// all-gather leaves it). At world 2 the one f32 add is the all-reduce's own (IEEE addition commutes), so this is bitwise
// `ncclAllReduce` + `ik_rt_f32_round`.
__device__ static inline void ik_store_rn(__nv_bfloat16* d, float v) { *d = __float2bfloat16_rn(v); }
__device__ static inline void ik_store_rn(__half* d, float v) { *d = __float2half_rn(v); }

template <typename T>
__global__ static void ik_f32_sum_round(const float* __restrict__ s, T* __restrict__ d, int world, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float acc = s[i];
    for (int r = 1; r < world; ++r) acc = __fadd_rn(acc, s[(size_t)r * n + i]);
    ik_store_rn(d + i, acc);
}

// L2 prefetch of weights ahead of the kernels that read them (lane dsv41/graphs, INTARSIA_DSV41_L2_PREFETCH; TensorFold's served
// settings): one block, 3 us after its start, then a wave of 128 x 8 KiB bulk prefetches every wave_ns, the ranges in order. Writes nothing.
struct IkPrefetch {
    const char* ptr[16];
    long long bytes[16];
    int n;
};

__global__ void __launch_bounds__(128) ik_l2_prefetch_kernel(IkPrefetch a, long long delay_ns, long long wave_ns) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    const long long chunk = 8 << 10;
    unsigned long long t0;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
    for (;;) {
        unsigned long long t;
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
        if (t - t0 >= (unsigned long long)delay_ns) break;
        __nanosleep(500);
    }
    long long end[16];
    long long total = 0;
    for (int t = 0; t < a.n; ++t) {
        total += (a.bytes[t] / 16 * 16 + chunk - 1) / chunk;
        end[t] = total;
    }
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
    int r = 0;
    long long wave = 0;
    for (long long c = threadIdx.x; c - threadIdx.x < total; c += blockDim.x) {
        __syncthreads();
        if (threadIdx.x == 0) {
            for (;;) {
                unsigned long long t;
                asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
                if (t - t0 >= (unsigned long long)(wave * wave_ns)) break;
                __nanosleep(256);
            }
        }
        __syncthreads();
        ++wave;
        if (c >= total) continue;
        while (c >= end[r]) ++r;
        const long long off = (c - (r ? end[r - 1] : 0)) * chunk;
        const long long left = a.bytes[r] / 16 * 16 - off;
        const unsigned size = (unsigned)(left < chunk ? left : chunk);
        asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(a.ptr[r] + off), "r"(size) : "memory");
    }
#endif
}

// ranges 16-byte aligned; at most `budget` bytes in all, the last range cut there
extern "C" int ik_rt_l2_prefetch(const void* const* ptrs, const size_t* bytes, int n, size_t budget, void* stream) {
    if (n <= 0) return 0;
    IkPrefetch a{};
    for (int i = 0; i < n && i < 16 && budget > 0; ++i) {
        const size_t take = bytes[i] < budget ? bytes[i] : budget;
        a.ptr[a.n] = (const char*)ptrs[i];
        a.bytes[a.n] = (long long)take;
        a.n += 1;
        budget -= take;
    }
    const long long wave_ns = (long long)(128.0 * 8192.0 / 230.0);
    ik_l2_prefetch_kernel<<<1, 128, 0, (cudaStream_t)stream>>>(a, 3000, wave_ns);
    return (int)cudaGetLastError();
}

extern "C" int ik_rt_f32_sum_round(const void* src, int world, size_t n, void* dst, int dst_dtype, void* stream) {
    if (n == 0) return 0;
    if (world < 1) return (int)cudaErrorInvalidValue;
    cudaStream_t st = (cudaStream_t)stream;
    unsigned blocks = (unsigned)((n + 255) / 256);
    if (dst_dtype == 0) ik_f32_sum_round<__nv_bfloat16><<<blocks, 256, 0, st>>>((const float*)src, (__nv_bfloat16*)dst, world, n);
    else if (dst_dtype == 1) ik_f32_sum_round<__half><<<blocks, 256, 0, st>>>((const float*)src, (__half*)dst, world, n);
    else return (int)cudaErrorInvalidValue;
    return (int)cudaGetLastError();
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
