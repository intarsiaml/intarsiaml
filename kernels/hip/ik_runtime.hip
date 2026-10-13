#include "ik_runtime.h"
#include <hip/hip_runtime.h>
#include <string.h>

extern "C" {

int ik_rt_device_count(int* n) { return (int)hipGetDeviceCount(n); }

int ik_rt_set_device(int ordinal) { return (int)hipSetDevice(ordinal); }

int ik_rt_mem_info(uint64_t* free_bytes, uint64_t* total_bytes) {
    size_t f = 0, t = 0;
    hipError_t e = hipMemGetInfo(&f, &t);
    *free_bytes = f;
    *total_bytes = t;
    return (int)e;
}

int ik_rt_init_device(int ordinal, ik_device_info* info) {
    hipError_t e = hipSetDevice(ordinal);
    if (e != hipSuccess) return (int)e;
    memset(info, 0, sizeof(*info));
    hipDeviceProp_t p;
    e = hipGetDeviceProperties(&p, ordinal);
    if (e != hipSuccess) return (int)e;
    strncpy(info->name, p.name, sizeof(info->name) - 1);
    // gcnArchName looks like "gfx1201" or "gfx90a:sramecc+:xnack-"; keep the part before the first ':'
    strncpy(info->arch, p.gcnArchName, sizeof(info->arch) - 1);
    char* colon = strchr(info->arch, ':');
    if (colon) *colon = 0;
    info->major = p.major;
    info->minor = p.minor;
    info->integrated = p.integrated;
    info->sm_count = p.multiProcessorCount;
    info->warp_size = p.warpSize;
    int v = 0;
    if (hipDeviceGetAttribute(&v, hipDeviceAttributeMemoryClockRate, ordinal) == hipSuccess) info->mem_clock_khz = v;
    if (hipDeviceGetAttribute(&v, hipDeviceAttributeMemoryBusWidth, ordinal) == hipSuccess) info->bus_width_bits = v;
    (void)hipGetLastError();  // clear any sticky "unsupported attribute" error
    if (hipRuntimeGetVersion(&v) == hipSuccess) info->runtime_version = v;
    if (hipDriverGetVersion(&v) == hipSuccess) info->driver_version = v;
    e = hipFree(0);  // force context creation
    if (e != hipSuccess) return (int)e;
    size_t f = 0, t = 0;
    e = hipMemGetInfo(&f, &t);
    if (e != hipSuccess) return (int)e;
    info->free_bytes = f;
    info->total_bytes = t;
    return 0;
}

int ik_rt_malloc(void** p, size_t bytes) { return (int)hipMalloc(p, bytes); }
int ik_rt_free(void* p) { return (int)hipFree(p); }

int ik_rt_memcpy_h2d(void* dst, const void* src, size_t n, void* stream) {
    // Pageable host memory: hipMemcpyAsync may return before the staging copy of `src` is complete (this is
    // not the CUDA guarantee), and callers drop `src` right after the call. Wait for the stream.
    hipError_t e = hipMemcpyAsync(dst, src, n, hipMemcpyHostToDevice, (hipStream_t)stream);
    if (e != hipSuccess) return (int)e;
    return (int)hipStreamSynchronize((hipStream_t)stream);
}
int ik_rt_memcpy_d2h(void* dst, const void* src, size_t n, void* stream) {
    hipError_t e = hipMemcpyAsync(dst, src, n, hipMemcpyDeviceToHost, (hipStream_t)stream);
    if (e != hipSuccess) return (int)e;
    return (int)hipStreamSynchronize((hipStream_t)stream);
}
int ik_rt_memcpy_d2d(void* dst, const void* src, size_t n, void* stream) {
    return (int)hipMemcpyAsync(dst, src, n, hipMemcpyDeviceToDevice, (hipStream_t)stream);
}
int ik_rt_memset(void* dst, int value, size_t n, void* stream) {
    return (int)hipMemsetAsync(dst, value, n, (hipStream_t)stream);
}

int ik_rt_stream_create(void** stream) {
    hipStream_t s = nullptr;
    hipError_t e = hipStreamCreateWithFlags(&s, hipStreamNonBlocking);
    *stream = (void*)s;
    return (int)e;
}
int ik_rt_stream_sync(void* stream) { return (int)hipStreamSynchronize((hipStream_t)stream); }

/* events (INTARSIA_HIP_TRACE): one marker per launch, successive deltas attribute stream time to kernels */
int ik_rt_event_create(void** ev) {
    hipEvent_t e = nullptr;
    hipError_t rc = hipEventCreate(&e);
    *ev = (void*)e;
    return (int)rc;
}
int ik_rt_event_record(void* ev, void* stream) { return (int)hipEventRecord((hipEvent_t)ev, (hipStream_t)stream); }
int ik_rt_event_elapsed_ms(float* ms, void* a, void* b) { return (int)hipEventElapsedTime(ms, (hipEvent_t)a, (hipEvent_t)b); }
int ik_rt_event_destroy(void* ev) { return (int)hipEventDestroy((hipEvent_t)ev); }

int ik_rt_can_access_peer(int device, int peer, int* ok) { return (int)hipDeviceCanAccessPeer(ok, device, peer); }

int ik_rt_enable_peer_access(int device, int peer) {
    hipError_t e = hipSetDevice(device);
    if (e != hipSuccess) return (int)e;
    e = hipDeviceEnablePeerAccess(peer, 0);
    if (e == hipErrorPeerAccessAlreadyEnabled) {
        (void)hipGetLastError();  // clear the sticky error
        return 0;
    }
    return (int)e;
}

int ik_rt_malloc_finegrained(void** p, size_t bytes) { return (int)hipExtMallocWithFlags(p, bytes, hipDeviceMallocFinegrained); }

int ik_rt_measure_bandwidth(double* gbs) {
    const size_t bytes = 128u << 20;
    void *a = nullptr, *b = nullptr;
    hipError_t e = hipMalloc(&a, bytes);
    if (e != hipSuccess) return (int)e;
    e = hipMalloc(&b, bytes);
    if (e != hipSuccess) {
        (void)hipFree(a);
        return (int)e;
    }
    (void)hipMemset(a, 1, bytes);
    hipEvent_t t0, t1;
    (void)hipEventCreate(&t0);
    (void)hipEventCreate(&t1);
    (void)hipMemcpy(b, a, bytes, hipMemcpyDeviceToDevice);  // warm up
    const int reps = 8;
    (void)hipEventRecord(t0, 0);
    for (int i = 0; i < reps; ++i) (void)hipMemcpyAsync(b, a, bytes, hipMemcpyDeviceToDevice, 0);
    (void)hipEventRecord(t1, 0);
    e = hipEventSynchronize(t1);
    float ms = 0.f;
    (void)hipEventElapsedTime(&ms, t0, t1);
    (void)hipEventDestroy(t0);
    (void)hipEventDestroy(t1);
    (void)hipFree(a);
    (void)hipFree(b);
    // a copy reads and writes `bytes`; report total moved bytes / time (the engine uses it as the roofline
    // hint; the true read-only figure on the R9700 is ~627 GB/s, see docs/results/2026-10-13-r9700-prep.md)
    *gbs = ms > 0.f ? (2.0 * bytes * reps) / (ms * 1e-3) / 1e9 : 0.0;
    return (int)e;
}

const char* ik_rt_error_string(int code) { return hipGetErrorString((hipError_t)code); }

}  // extern "C"
