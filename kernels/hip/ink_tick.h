// Ticket slots of the Inkling-2 kernels (lane ink-hip-overlap): the last-block tickets of the gate kernel's fused norm
// (ink_mix.hip g_gate_ctr), InkAttnDecode's split merge (ink_attn.hip ink_dec_tick) and the fused route (ink_route_fused.hip
// g_route_ctr) are device-global counters. Two streams that run one of these kernels at the same time (a prefill chunk on stream 0
// beside the decode twin's batched step on stream 1) must not count into the same entry, so each array holds kInkTickSlots copies
// and a launch uses the copy of its (device, stream), as `moe_sync_slot` does for g_moe_sync (moe.hip). The slot only picks
// which counter: no value a kernel computes depends on it.
//
// Slots are handed out per device in first-launch order. Each translation unit that includes this header has its own table
// (static), matching its own copies of the arrays (one code object per TU). When all slots of a device are taken (a test process
// that creates more streams than kInkTickSlots), slot 0 is shared, which is the single-array behaviour of main; a warning says so once.
#pragma once
#include <hip/hip_runtime.h>
#include <stdio.h>
#include <mutex>
#include <utility>
#include <vector>

constexpr int kInkTickSlots = 8;

static inline int ink_tick_slot(hipStream_t s) {
    static std::mutex mu;
    static std::vector<std::pair<int, hipStream_t>> owners;
    static bool warned = false;
    int dev = 0;
    if (hipGetDevice(&dev) != hipSuccess) dev = 0;
    std::lock_guard<std::mutex> g(mu);
    int n = 0;
    for (const auto& o : owners) {
        if (o.first != dev) continue;
        if (o.second == s) return n;
        ++n;
    }
    if (n >= kInkTickSlots) {
        if (!warned) {
            fprintf(stderr, "intarsia hip: more than %d streams on device %d launch Inkling ticket kernels; the extra ones share slot 0\n",
                    kInkTickSlots, dev);
            warned = true;
        }
        return 0;
    }
    owners.emplace_back(dev, s);
    return n;
}
