/* Thin C wrappers over the HIP runtime, used by the Rust backend (not kernel launchers: these may
 * allocate and synchronise). Every function returns a hipError_t value (0 = ok) unless noted. Keeping all
 * runtime calls here means the Rust side never depends on struct layouts (hipDeviceProp_t) or on
 * runtime-version-specific symbols. Mirrors kernels/cuda/ik_runtime.h. */
#ifndef IK_RUNTIME_H
#define IK_RUNTIME_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    char    name[256];
    char    arch[64];          /* gcnArchName without feature flags, e.g. "gfx1201" */
    int32_t major, minor;      /* hipDeviceProp_t major / minor */
    int32_t integrated;        /* 1 = shares memory with the host (APU) */
    int32_t sm_count;          /* compute units */
    int32_t mem_clock_khz;     /* 0 when unknown */
    int32_t bus_width_bits;    /* 0 when unknown */
    int32_t warp_size;
    int32_t runtime_version;   /* hipRuntimeGetVersion */
    int32_t driver_version;    /* hipDriverGetVersion */
    int32_t _pad;
    uint64_t total_bytes;      /* hipMemGetInfo at init */
    uint64_t free_bytes;
} ik_device_info;

int ik_rt_device_count(int* n);
/* select the device, fill info */
int ik_rt_init_device(int ordinal, ik_device_info* info);
int ik_rt_set_device(int ordinal);
int ik_rt_mem_info(uint64_t* free_bytes, uint64_t* total_bytes);
int ik_rt_malloc(void** p, size_t bytes);
int ik_rt_free(void* p);
int ik_rt_memcpy_h2d(void* dst, const void* src, size_t n, void* stream); /* returns once `src` may be reused */
int ik_rt_memcpy_d2h(void* dst, const void* src, size_t n, void* stream); /* blocks until done */
/* round 9 pinned staging (device.rs `Staging`): `src` / `dst` are page-locked host memory. h2d returns at once (the caller
 * keeps `src` unchanged until an event recorded after the copy completed); d2h waits for the stream. */
int ik_rt_memcpy_h2d_pinned(void* dst, const void* src, size_t n, void* stream);
int ik_rt_memcpy_d2h_pinned(void* dst, const void* src, size_t n, void* stream);
int ik_rt_host_alloc(void** p, size_t bytes);
int ik_rt_host_free(void* p);
int ik_rt_event_create_sync(void** ev); /* an event without timing, for waiting on */
int ik_rt_event_sync(void* ev);
/* test seam (tests/staging.rs): a 1-thread kernel that spins on the wall clock for `us` microseconds */
int ik_rt_spin(void* stream, unsigned us);
int ik_rt_memcpy_d2d(void* dst, const void* src, size_t n, void* stream);
int ik_rt_memset(void* dst, int value, size_t n, void* stream);
int ik_rt_stream_create(void** stream);
int ik_rt_stream_sync(void* stream);
/* stream events for the per-kernel GPU-time trace (INTARSIA_HIP_TRACE) */
int ik_rt_event_create(void** ev);
int ik_rt_event_record(void* ev, void* stream);
int ik_rt_event_elapsed_ms(float* ms, void* a, void* b);
int ik_rt_event_destroy(void* ev);

/* graph capture / replay of a stream (the decode step as one hipGraph); mirrors the CUDA runtime */
int ik_rt_graph_begin(void* stream);
int ik_rt_graph_end(void* stream, void** exec);
int ik_rt_graph_launch(void* exec, void* stream);
int ik_rt_graph_destroy(void* exec);
int ik_rt_device_sync(void); /* hipDeviceSynchronize (INTARSIA_GRAPH_SYNC_DESTROY: a graph exec is destroyed only when the device is idle) */
/* tensor parallelism (kernels/hip/allreduce.hip): peer access and the fine-grained (uncached) allocations the
 * all-reduce's scratch and flags live in */
int ik_rt_can_access_peer(int device, int peer, int* ok);
/* enable access from `device` (made current) to `peer`'s memory; already enabled is success */
int ik_rt_enable_peer_access(int device, int peer);
int ik_rt_malloc_finegrained(void** p, size_t bytes);
/* two-rank one-shot all-reduce launcher (allreduce.hip); not a tensor op, so declared here */
int ik_ar2_oneshot(void* x, void* out, int dtype, int out_dtype, int64_t n_elem, void* peer_scratch, void* my_scratch,
                   unsigned int* peer_flags, unsigned int* my_flags, unsigned int* ctrs, int64_t slot_bytes,
                   int nblocks, int drain, void* stream);
/* two-rank all-gather launcher (allreduce.hip), same scratch / flags / counters as ik_ar2_oneshot */
int ik_ag2_gather(void* x, void* out, int64_t bytes, int64_t out_stride_bytes, void* peer_scratch, void* my_scratch,
                  unsigned int* peer_flags, unsigned int* my_flags, unsigned int* ctrs, int64_t slot_bytes, int nblocks,
                  int rank, int drain, void* stream);
/* D2D copy bandwidth in GB/s (read + write counted), ~256 MB working set */
int ik_rt_measure_bandwidth(double* gbs);
const char* ik_rt_error_string(int code);
/* the same all-reduce fused with the residual add + rmsnorm after it (ar_norm.hip), bitwise the pair */
int ik_ar2_add_rmsnorm(void* x, void* out, const void* res, const void* w, void* nrm, void* sum, int dtype, int64_t rows,
                       int64_t H, float eps, float w_offset, void* peer_scratch, void* my_scratch, unsigned int* peer_flags,
                       unsigned int* my_flags, unsigned int* ctrs, int64_t slot_bytes, int nblocks, int drain, void* stream);
/* Round 9, INTARSIA_HIP_STAMP=<file> (diagnostics only; unset = a no-op that launches nothing): a 1-thread kernel on
 * `stream` that stores the GPU's constant-rate wall clock (wall_clock64) when it runs, tagged `tag` / `piece` on the host.
 * Launch sites bracket the prefill's long dispatches with it, so the time between the stamps before and after a dispatch,
 * minus that dispatch's own duration, is the stream's dead time around it without a profiler attached. Skipped while the
 * stream is being captured. ik_rt_stamp_dump writes every stamp to the file (called when the backend is dropped). */
int ik_rt_stamp(void* stream, const char* tag, int piece);
int ik_rt_stamp_dump(void);

#ifdef __cplusplus
}
#endif
#endif
