/* Thin C wrappers over the CUDA runtime, used by the Rust backend (not kernel launchers:
 * these may allocate and synchronise). Every function returns a cudaError_t value (0 = ok)
 * unless noted. Keeping all runtime calls here means the Rust side never depends on struct
 * layouts (cudaDeviceProp) or on runtime-version-specific symbols. */
#ifndef IK_RUNTIME_H
#define IK_RUNTIME_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    char    name[256];
    int32_t major, minor;      /* compute capability */
    int32_t integrated;        /* 1 = shares memory with the host (GB10) */
    int32_t sm_count;
    int32_t mem_clock_khz;     /* 0 when unknown */
    int32_t bus_width_bits;    /* 0 when unknown */
    uint64_t total_bytes;      /* cudaMemGetInfo at init */
    uint64_t free_bytes;
} ik_device_info;

int ik_rt_device_count(int* n);
/* select the device, create cuBLAS state, fill info */
int ik_rt_init_device(int ordinal, ik_device_info* info);
int ik_rt_set_device(int ordinal);
int ik_rt_mem_info(uint64_t* free_bytes, uint64_t* total_bytes);
int ik_rt_malloc(void** p, size_t bytes);
int ik_rt_free(void* p);
int ik_rt_memcpy_h2d(void* dst, const void* src, size_t n, void* stream);
int ik_rt_memcpy_d2h(void* dst, const void* src, size_t n, void* stream); /* blocks until done */
int ik_rt_memcpy_d2d(void* dst, const void* src, size_t n, void* stream);
int ik_rt_memset(void* dst, int value, size_t n, void* stream);
int ik_rt_stream_create(void** stream);
int ik_rt_stream_create_high_priority(void** stream);
int ik_rt_stream_sync(void* stream);
/* fork / join of two streams (also inside a graph capture: the wait pulls the other stream into it): an event without timing,
   recorded after the work queued so far on `stream`; later work on `stream` waits for the work `ev` was recorded after */
int ik_rt_event_create_sync(void** ev);
int ik_rt_event_record(void* ev, void* stream);
int ik_rt_stream_wait_event(void* stream, void* ev);
/* D2D copy bandwidth in GB/s (one read + one write counted once per byte copied), ~256 MB working set */
int ik_rt_measure_bandwidth(double* gbs);
/* CUDA graphs: everything launched on `stream` between begin and end is recorded, not run (thread-local capture mode,
 * so other threads may keep calling the runtime). end instantiates the graph; the returned exec is replayed with
 * ik_rt_graph_launch on any stream and freed with ik_rt_graph_destroy. A failed capture leaves the stream usable. */
int ik_rt_graph_begin(void* stream);
int ik_rt_graph_end(void* stream, void** exec);
int ik_rt_graph_launch(void* exec, void* stream);
int ik_rt_graph_destroy(void* exec);
const char* ik_rt_error_string(int code);
/* dst[i] = round-to-nearest-even bf16 (dst_dtype 0) or f16 (1) of the f32 src[i], or a copy (2: f32), n elements on
 * `stream`: the one rounding of a tensor-parallel f32 partial sum (the NCCL collective, docs/design/35b-tp2.md) */
int ik_rt_f32_round(const void* src, void* dst, size_t n, int dst_dtype, void* stream);

#ifdef __cplusplus
}
#endif
#endif
