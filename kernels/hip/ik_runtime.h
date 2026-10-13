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
int ik_rt_memcpy_d2d(void* dst, const void* src, size_t n, void* stream);
int ik_rt_memset(void* dst, int value, size_t n, void* stream);
int ik_rt_stream_create(void** stream);
int ik_rt_stream_sync(void* stream);
/* stream events for the per-kernel GPU-time trace (INTARSIA_HIP_TRACE) */
int ik_rt_event_create(void** ev);
int ik_rt_event_record(void* ev, void* stream);
int ik_rt_event_elapsed_ms(float* ms, void* a, void* b);
int ik_rt_event_destroy(void* ev);
/* D2D copy bandwidth in GB/s (read + write counted), ~256 MB working set */
int ik_rt_measure_bandwidth(double* gbs);
const char* ik_rt_error_string(int code);

#ifdef __cplusplus
}
#endif
#endif
