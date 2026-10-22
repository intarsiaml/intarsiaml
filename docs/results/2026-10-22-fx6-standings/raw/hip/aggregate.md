
**Decode tok/s, single stream**: median of visit medians (min-max over visits, n = visits of 3 reps each)

| engine | 128 | 1024 | 4096 | 16384 |
|---|---|---|---|---|
| intarsia HIP (origin/main cd617de2) | 159.3 (159.1-159.7, n=7) | 159.0 (158.9-159.5, n=6) | 155.5 (155.4-155.9, n=5) | 141.6 (141.4-141.6, n=5) |
| intarsia HIP (2026-10-21 standings binary, pre-J1) | 159.6 (159.5-159.8, n=2) | 158.6 (158.6-158.6, n=1) | 155.3 (155.3-155.3, n=1) | 141.5 (141.5-141.5, n=1) |
| llama.cpp v0.5.0 Vulkan MXFP4_MOE | 115.4 (114.7-115.6, n=6) | 115.3 (114.7-115.5, n=5) | 112.0 (111.6-112.5, n=5) | 106.7 (106.0-106.8, n=5) |
| llama.cpp 0.4.0-dev Vulkan MXFP4_MOE (old comparator) | 113.1 (113.1-113.1, n=2) | 112.7 (112.6-112.9, n=2) | 110.3 (110.3-110.4, n=2) | 105.3 (105.1-105.4, n=2) |
| llama.cpp v0.5.0 HIP MXFP4_MOE | 83.3 (83.0-83.4, n=4) | 83.3 (83.0-83.3, n=4) | 81.6 (81.5-81.7, n=4) | 77.7 (77.7-77.8, n=4) |
| hipfire ad10b3d MQ4R | 146.4 (145.5-146.6, n=6) | 145.2 (145.1-146.2, n=6) | 141.9 (141.2-142.6, n=4) | 127.6 (127.1-127.9, n=4) |

**Prefill tok/s (per request; 1k prompt has 990 tokens so TTFT overhead is included)**: median of visit medians (min-max over visits, n = visits of 3 reps each)

| engine | 1024 | 4096 | 16384 |
|---|---|---|---|
| intarsia HIP (origin/main cd617de2) | 5753 (5657-5824, n=6) | 9106 (9068-9122, n=5) | 7739 (7725-7783, n=5) |
| intarsia HIP (2026-10-21 standings binary, pre-J1) | 5676 (5676-5676, n=1) | 9028 (9028-9028, n=1) | 7720 (7720-7720, n=1) |
| llama.cpp v0.5.0 Vulkan MXFP4_MOE | 2796 (2784-2817, n=5) | 3198 (3183-3200, n=5) | 2967 (2963-2972, n=5) |
| llama.cpp 0.4.0-dev Vulkan MXFP4_MOE (old comparator) | 2540 (2536-2544, n=2) | 2865 (2861-2869, n=2) | 2685 (2680-2691, n=2) |
| llama.cpp v0.5.0 HIP MXFP4_MOE | 2989 (2983-3027, n=4) | 3382 (3373-3394, n=4) | 3235 (3227-3238, n=4) |
| hipfire ad10b3d MQ4R | 2762 (2754-2800, n=6) | 2343 (2341-2346, n=4) | 1283 (1283-1284, n=4) |

**Marginal prefill tok/s (delta tokens / delta TTFT)**: median of visit medians (min-max over visits, n = visits of 3 reps each)

| engine | 1024 | 4096 | 16384 |
|---|---|---|---|
| intarsia HIP (origin/main cd617de2) | 9816 (9733-10417, n=6) | 11214 (11071-11303, n=5) | 7379 (7357-7435, n=5) |
| intarsia HIP (2026-10-21 standings binary, pre-J1) | 9637 (9637-9637, n=1) | 11144 (11144-11144, n=1) | 7375 (7375-7375, n=1) |
| llama.cpp v0.5.0 Vulkan MXFP4_MOE | 4079 (3977-4156, n=5) | 3338 (3332-3358, n=5) | 2902 (2894-2905, n=5) |
| llama.cpp 0.4.0-dev Vulkan MXFP4_MOE (old comparator) | 3722 (3696-3748, n=2) | 2986 (2986-2986, n=2) | 2632 (2626-2638, n=2) |
| llama.cpp v0.5.0 HIP MXFP4_MOE | 3864 (3834-3915, n=4) | 3527 (3515-3541, n=4) | 3188 (3180-3196, n=4) |
| hipfire ad10b3d MQ4R | 3195 (3173-3269, n=6) | 2236 (2234-2237, n=4) | 1118 (1118-1119, n=4) |

**TTFT s**: median of visit medians (min-max over visits, n = visits of 3 reps each)

| engine | 1024 | 4096 | 16384 |
|---|---|---|---|
| intarsia HIP (origin/main cd617de2) | 0.17 (0.17-0.17, n=6) | 0.45 (0.45-0.45, n=5) | 2.15 (2.14-2.16, n=5) |
| intarsia HIP (2026-10-21 standings binary, pre-J1) | 0.17 (0.17-0.17, n=1) | 0.45 (0.45-0.45, n=1) | 2.16 (2.16-2.16, n=1) |
| llama.cpp v0.5.0 Vulkan MXFP4_MOE | 0.35 (0.35-0.36, n=5) | 1.28 (1.28-1.28, n=5) | 5.61 (5.60-5.63, n=5) |
| llama.cpp 0.4.0-dev Vulkan MXFP4_MOE (old comparator) | 0.39 (0.39-0.39, n=2) | 1.43 (1.43-1.43, n=2) | 6.21 (6.20-6.22, n=2) |
| llama.cpp v0.5.0 HIP MXFP4_MOE | 0.33 (0.33-0.33, n=4) | 1.21 (1.21-1.21, n=4) | 5.15 (5.14-5.16, n=4) |
| hipfire ad10b3d MQ4R | 0.36 (0.35-0.36, n=6) | 1.75 (1.74-1.75, n=4) | 12.98 (12.97-12.98, n=4) |

**Ratio to intarsia (new binary), medians; 'tie' when the min-max bands of the two arms overlap**

| engine | metric | 1024 | 4096 | 16384 |
|---|---|---|---|---|
| intarsia HIP (2026-10-21 standings binary, pre-J1) | decode (ours / theirs) | 1.00x | 1.00x | 1.00x tie |
| intarsia HIP (2026-10-21 standings binary, pre-J1) | prefill (ours / theirs) | 1.01x tie | 1.01x | 1.00x |
| llama.cpp v0.5.0 Vulkan MXFP4_MOE | decode (ours / theirs) | 1.38x | 1.39x | 1.33x |
| llama.cpp v0.5.0 Vulkan MXFP4_MOE | prefill (ours / theirs) | 2.06x | 2.85x | 2.61x |
| llama.cpp 0.4.0-dev Vulkan MXFP4_MOE (old comparator) | decode (ours / theirs) | 1.41x | 1.41x | 1.35x |
| llama.cpp 0.4.0-dev Vulkan MXFP4_MOE (old comparator) | prefill (ours / theirs) | 2.26x | 3.18x | 2.88x |
| llama.cpp v0.5.0 HIP MXFP4_MOE | decode (ours / theirs) | 1.91x | 1.91x | 1.82x |
| llama.cpp v0.5.0 HIP MXFP4_MOE | prefill (ours / theirs) | 1.92x | 2.69x | 2.39x |
| hipfire ad10b3d MQ4R | decode (ours / theirs) | 1.09x | 1.10x | 1.11x |
| hipfire ad10b3d MQ4R | prefill (ours / theirs) | 2.08x | 3.89x | 6.03x |
