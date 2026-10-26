#include "ik_runtime.h"
#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* INTARSIA_HIP_STAMP (ik_runtime.h): device ring of wall-clock stamps, host-side tags */
namespace {
constexpr unsigned kStampCap = 1u << 20;
__global__ void ik_stamp_kernel(unsigned long long* buf, unsigned idx) { buf[idx] = wall_clock64(); }
__global__ void ik_spin_kernel(unsigned long long ticks) {
    const unsigned long long t0 = wall_clock64();
    while (wall_clock64() - t0 < ticks) __builtin_amdgcn_s_sleep(2);
}
struct StampState {
    int on = -1;  // -1: env not read yet
    const char* path = nullptr;
    unsigned long long* dev = nullptr;
    unsigned n = 0;
    const char** tags = nullptr;
    int* pieces = nullptr;
};
StampState g_stamp;
}  // namespace

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
/* round 9: pinned host staging (device.rs `Staging`). `src` is page-locked memory the caller keeps unchanged until an
   event recorded after this copy has completed, so nothing waits here. */
int ik_rt_memcpy_h2d_pinned(void* dst, const void* src, size_t n, void* stream) {
    return (int)hipMemcpyAsync(dst, src, n, hipMemcpyHostToDevice, (hipStream_t)stream);
}
/* `dst` page-locked: the copy, then a wait for the stream (the caller reads `dst` after it) */
int ik_rt_memcpy_d2h_pinned(void* dst, const void* src, size_t n, void* stream) {
    hipError_t e = hipMemcpyAsync(dst, src, n, hipMemcpyDeviceToHost, (hipStream_t)stream);
    if (e != hipSuccess) return (int)e;
    return (int)hipStreamSynchronize((hipStream_t)stream);
}
int ik_rt_host_alloc(void** p, size_t bytes) { return (int)hipHostMalloc(p, bytes, hipHostMallocDefault); }
int ik_rt_host_free(void* p) { return (int)hipHostFree(p); }
int ik_rt_event_create_sync(void** ev) {
    hipEvent_t e = nullptr;
    hipError_t rc = hipEventCreateWithFlags(&e, hipEventDisableTiming);
    *ev = (void*)e;
    return (int)rc;
}
int ik_rt_event_sync(void* ev) { return (int)hipEventSynchronize((hipEvent_t)ev); }

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

/* graphs: capture a stream's launches (thread-local mode, so other threads' runtime calls stay legal), instantiate, replay.
   ik_rt_graph_end always ends the capture, even for a stream whose capture was invalidated, so the stream is never left in
   capture mode. */
/* INTARSIA_GRAPH_SYNC_CAPTURE=1 (diagnostic, w27): the stream is idle before a capture begins and before the graph is instantiated (a peer rank only enqueues,
   so its previous replay can still be running when its next capture starts). */
static bool ik_sync_capture() {
    static const bool on = [] { const char* v = getenv("INTARSIA_GRAPH_SYNC_CAPTURE"); return v && v[0] == '1'; }();
    return on;
}
int ik_rt_graph_begin(void* stream) {
    if (ik_sync_capture()) (void)hipStreamSynchronize((hipStream_t)stream);
    return (int)hipStreamBeginCapture((hipStream_t)stream, hipStreamCaptureModeThreadLocal);
}
int ik_rt_graph_end(void* stream, void** exec) {
    *exec = nullptr;
    hipGraph_t g = nullptr;
    hipError_t e = hipStreamEndCapture((hipStream_t)stream, &g);
    if (e != hipSuccess) {
        if (g) (void)hipGraphDestroy(g);
        (void)hipGetLastError();  // clear the sticky capture error: the stream is usable again
        (void)hipStreamSynchronize((hipStream_t)stream);
        (void)hipGetLastError();
        return (int)e;
    }
    hipGraphExec_t ge = nullptr;
    if (ik_sync_capture()) (void)hipStreamSynchronize((hipStream_t)stream);
    e = hipGraphInstantiate(&ge, g, nullptr, nullptr, 0);
    (void)hipGraphDestroy(g);
    if (e != hipSuccess) {
        (void)hipGetLastError();
        return (int)e;
    }
    *exec = (void*)ge;
    return 0;
}
int ik_rt_graph_launch(void* exec, void* stream) { return (int)hipGraphLaunch((hipGraphExec_t)exec, (hipStream_t)stream); }
int ik_rt_graph_destroy(void* exec) { return (int)hipGraphExecDestroy((hipGraphExec_t)exec); }
int ik_rt_device_sync(void) { return (int)hipDeviceSynchronize(); }
int ik_rt_sync_capture_on(void) { return ik_sync_capture() ? 1 : 0; }

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

int ik_rt_spin(void* stream, unsigned us) {
    int khz = 0, dev = 0;
    (void)hipGetDevice(&dev);
    if (hipDeviceGetAttribute(&khz, hipDeviceAttributeWallClockRate, dev) != hipSuccess || khz <= 0) khz = 100000;
    ik_spin_kernel<<<1, 1, 0, (hipStream_t)stream>>>((unsigned long long)us * (unsigned long long)khz / 1000ull);
    return (int)hipGetLastError();
}

int ik_rt_stamp(void* stream, const char* tag, int piece) {
    StampState& g = g_stamp;
    if (g.on == 0) return 0;
    if (g.on < 0) {
        const char* e = getenv("INTARSIA_HIP_STAMP");
        g.on = (e && *e && strcmp(e, "0") != 0) ? 1 : 0;
        if (!g.on) return 0;
        g.path = e;
        if (hipMalloc(&g.dev, sizeof(unsigned long long) * kStampCap) != hipSuccess) {
            g.on = 0;
            return 0;
        }
        g.tags = (const char**)calloc(kStampCap, sizeof(const char*));
        g.pieces = (int*)calloc(kStampCap, sizeof(int));
    }
    hipStreamCaptureStatus cs = hipStreamCaptureStatusNone;
    if (hipStreamIsCapturing((hipStream_t)stream, &cs) != hipSuccess || cs != hipStreamCaptureStatusNone) return 0;
    if (g.n >= kStampCap) return 0;
    g.tags[g.n] = tag;
    g.pieces[g.n] = piece;
    ik_stamp_kernel<<<1, 1, 0, (hipStream_t)stream>>>(g.dev, g.n);
    g.n++;
    return (int)hipGetLastError();
}

int ik_rt_stamp_dump(void) {
    StampState& g = g_stamp;
    if (g.on != 1 || g.n == 0) return 0;
    (void)hipDeviceSynchronize();
    unsigned long long* h = (unsigned long long*)malloc(sizeof(unsigned long long) * g.n);
    hipError_t e = hipMemcpy(h, g.dev, sizeof(unsigned long long) * g.n, hipMemcpyDeviceToHost);
    int khz = 0;
    int dev = 0;
    (void)hipGetDevice(&dev);
    (void)hipDeviceGetAttribute(&khz, hipDeviceAttributeWallClockRate, dev);
    FILE* f = fopen(g.path, "w");
    if (f && e == hipSuccess) {
        fprintf(f, "# wall_clock_khz %d stamps %u\n", khz, g.n);
        for (unsigned i = 0; i < g.n; ++i) fprintf(f, "%llu %s %d\n", h[i], g.tags[i], g.pieces[i]);
    }
    if (f) fclose(f);
    free(h);
    g.n = 0;
    return (int)e;
}

}  // extern "C"
