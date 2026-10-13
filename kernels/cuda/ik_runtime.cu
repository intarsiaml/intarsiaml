#include "ik_runtime.h"
#include "ik_internal.h"
#include <cuda_runtime.h>
#include <string.h>

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

int ik_rt_malloc(void** p, size_t bytes) { return (int)cudaMalloc(p, bytes); }
int ik_rt_free(void* p) { return (int)cudaFree(p); }

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

const char* ik_rt_error_string(int code) { return cudaGetErrorString((cudaError_t)code); }

}  // extern "C"
