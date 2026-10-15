# Metal G0b: exact q_len > 1 decode attention, k-row small ops, verify-step estimate

Date 2026-10-15. Branch `metal-g0b-attn` (off `metal-g0-multirow` 96460dd). Machine endor (Apple M5 Max). Plan: `docs/research/metal-next-levers.md`; G0 (multi-row GEMVs): `docs/results/2026-10-15-metal-g0-multirow.md`.

**Verdict.** Every kernel a k-row verify step needs now has a per-row bitwise-exact form, and none of them blocks speculation.
(1) A new multi-row split-KV attention gives, per row, exactly the bits of the 1-row serving launch at that row's position (7 shapes, k = 2..8, 3,200 position windows, all passed). It costs T(k)/T(1) = 1.4x / 2.2x / 3.1x at k = 2 / 4 / 6 at 4k context (1.5x / 2.5x / 3.4x at 16k), which is 1.2x (k = 2) to 1.6x (k = 6..8) cheaper than k sequential 1-row launches at 4k but far from free: it is the dominant extra cost of a verify step beyond 1k context.
(2) The non-GEMV small ops (RmsNorm, RmsNormAdd, Add, SiluGlu, Embedding, Argmax, QkNormRopeKv incl. KV write, unfused KvAppend) are row-parallel already: T = k rows is bitwise k 1-row launches with no new kernel, and they cost under 1 percent of a step for k <= 8.
(3) Full verify-step estimate: step(k)/step(1) = 1.011 / 1.035 / 1.064 at k = 4 for 128 / 1k / 4k context, and 1.038 / 1.066 / 1.126 at k = 6. Break-even per-draft-token acceptance (i.i.d. model) is **0.4 / 1.6 / 2.2 percent at k = 2, 1.1 / 3.4 / 6.0 percent at k = 4, 3.6 / 6.2 / 11.2 percent at k = 6** (128 / 1k / 4k).
(4) The R = 8 cliff remedy (copy the 8-mod-16 weights to 16-byte aligned memory at load) costs 2.75 GB of extra resident memory and about 0.15 s of load time (0.25 s with the source page-in), and is not needed for k <= 6.

## 1. Multi-row split-KV attention (item 1)

### What "the 1-row kernel" is (reference pinned before writing anything)
`attn_decode_bf16_splitkv` (`kernels/metal/attention_splitkv.metal`, launcher `src/attn.rs`). The Metal backend has no graph capture, so the eager decode step launches it with the **host position** `attrs.pos0 = st.len`, not a device position tensor. Its work partition is a function of the row's visible key count `v = pos0 + 1`: `nsplit = clamp(min(ceil(v/128), ceil(120/Nkv)), 1, 64)` (15 splits for Nkv = 8 from v = 1793), chunk `= roundup32(ceil(v/nsplit))`, and **simdgroups per threadgroup = 8 if Nkv x nsplit <= 64 (v <= 1024) else 4** for a host position (always 4 for a device position). The simdgroup count decides which simdgroup and slot sees which key and the merge order, so the two modes differ in the last bits (`splitkv_device_position` in `tests/attention.rs`). The reference of this work is therefore the host-position launch at `pos0 + r`; the device-position rule is implemented and tested too.

### Kernel (exact by construction)
`attn_rows_splitkv_g{1,2,4,8}` + `attn_rows_combine` (appended to `attention_splitkv.metal`, the 1-row kernels are untouched), launcher `MetalBackend::launch_attn_rows`, registered as **`attn_prefill_bf16_splitkv_rows`** (`Op::AttnPrefill`, because `OpCall::check` rejects `AttnDecode` with T != 1 and core may not change; row t of `AttnPrefill` sees keys <= pos0 + t, which is verify semantics). Priority -1 with `T in 2..=8`: `attn_prefill_bf16` (priority 0) takes any T, so serving never selects it. Scratch declaration is `T x` the 1-row bound.
Design: row r of the call runs in its own threadgroups with **exactly the partition the 1-row launch at position pos0 + r makes** (its own `visible_r`, `nsplit_r`, `chunk_r`, `nsg_r`, same key-to-(simdgroup, slot, step) assignment, same merge order, same `ask_step` code, same combine over a per-row scratch slab). The multi-row call is R 1-row calls in one dispatch pair; grid x = kvh * R + r puts the R threadgroups that read the same K/V slice next to each other. Surplus simdgroups (a row needing 4 while another needs 8) idle but reach the barrier.
Not done in the kernel: sharing the K/V loads through registers (R x GP x 16 floats per lane of state would spill) and a shared-memory variant (below, negative).

### Bitwise test (`crates/intarsia-backend-metal/tests/g0b_attn.rs`, 7 tests)
One cache with random data **everywhere** (also past `pos0 + r`, so reading a key a row must not see changes bits); every row of the k-row output is compared byte for byte with the 1-row `attn_decode_bf16_splitkv` launch at host position `pos0 + r` on the same cache.
* Qwen3-8B shape (32 q / 8 kv heads, head_dim 128), cap 4096, k = 2..8, 1,218 (k, pos0) windows placed on every partition change: first/last cache row, visible crossing every multiple of 128 up to 2048, the 1024 -> 1025 simdgroup switch, the 1792 -> 1793 split cap, chunk changes at multiples of 480.
* Same at cap 16384, k = 2, 4, 6, 8 (812 windows).
* Other shapes: 8/8/64 (group 1), 8/2/64, 16/2/256 (group 8, 2 simdgroups), 12/4/32 (group 3 padded to GP 4), 6/2/128, 8/1/16, k in {2,3,5,8}, 183 windows each; 16-byte aligned non-zero tensor offsets; misaligned q (falls back to the single-pass rows, equal to the 1-row launcher's own fallback).
* Device-position mode (through the launcher directly, `OpCall::check` refuses a position on `AttnPrefill`): bitwise the device-position 1-row launch.
* Guard bytes around the output are untouched. Result: **7 passed**, in slot #1993 (`raw/.../final/cargo-test-g0b.log`).
* Serving attention unchanged: `examples/attn_hash` output is identical to the B5 reference `raw/2026-10-14-metal-phase-b/b5/attn_hash-new3.txt` (`final/attn_hash.diff` empty).

### Timing (`examples/g0b_bench`, 30 rounds, slot #1993)
Method as G0/gemv_b3: one command buffer with a serial chain of N >= 20 ms of dispatches walking a pool of **distinct** per-layer KV caches (>= 1.5 GiB, so K/V come from DRAM as in a decode step), GPU start/end of the buffer / N, candidates interleaved in rotated order within a round, ratios taken within the round; macmon at 100 ms, rounds whose window shows a GPU clock below 1450 MHz dropped (329 of 330 rounds kept, mean clock 1605-1617 MHz). Context = visible keys of row 0 (`pos0 = ctx - 1`); row r sees ctx + r keys. The sequential column is k back-to-back 1-row launches (cache-warm after the first). "Today's AttnPrefill" is what the registry runs for T = 2..8 now (the flash prefill kernel).

| ctx | k | T(1) serving (us) | T(k) rows kernel (us) | T(k)/T(1) [p10-p90] | k sequential 1-row launches / T(1) | today's AttnPrefill (flash) / T(1) |
|---|---|---|---|---|---|---|
| 128 | 2 | 8.6 | 10.6 | **1.23** [1.23-1.24] | 2.18 | 1.6 |
| 128 | 3 | 8.6 | 11.4 | **1.33** [1.32-1.34] | - | - |
| 128 | 4 | 8.6 | 11.7 | **1.36** [1.36-1.37] | 4.32 | 1.7 |
| 128 | 5 | 8.6 | 11.8 | **1.37** [1.37-1.37] | - | - |
| 128 | 6 | 8.6 | 15.6 | **1.82** [1.81-1.83] | 6.43 | - |
| 128 | 7 | 8.6 | 15.5 | **1.80** [1.80-1.81] | - | - |
| 128 | 8 | 8.6 | 17.8 | **2.07** [2.06-2.07] | 8.56 | 1.7 |
| 1024 | 2 | 14.5 | 25.3 | **1.75** [1.73-1.76] | 2.03 | 6.0 |
| 1024 | 3 | 14.5 | 33.4 | **2.31** [2.29-2.32] | - | - |
| 1024 | 4 | 14.5 | 36.0 | **2.49** [2.47-2.50] | 4.05 | 6.0 |
| 1024 | 5 | 14.5 | 39.8 | **2.75** [2.72-2.76] | - | - |
| 1024 | 6 | 14.5 | 43.2 | **2.98** [2.95-3.00] | 6.05 | - |
| 1024 | 7 | 14.5 | 48.9 | **3.38** [3.35-3.39] | - | - |
| 1024 | 8 | 14.5 | 52.6 | **3.63** [3.59-3.66] | 8.07 | 6.0 |
| 4096 | 2 | 36.4 | 52.1 | **1.44** [1.42-1.46] | 1.78 | 9.2 |
| 4096 | 3 | 36.4 | 64.9 | **1.79** [1.78-1.80] | - | - |
| 4096 | 4 | 36.4 | 81.2 | **2.23** [2.22-2.24] | 3.34 | 9.2 |
| 4096 | 5 | 36.4 | 96.1 | **2.64** [2.62-2.66] | - | - |
| 4096 | 6 | 36.4 | 113.2 | **3.11** [3.09-3.13] | 4.89 | - |
| 4096 | 7 | 36.4 | 128.2 | **3.53** [3.50-3.55] | - | - |
| 4096 | 8 | 36.4 | 145.4 | **4.01** [3.97-4.03] | 6.44 | 9.2 |
| 16384 | 2 | 122.2 | 178.0 | **1.46** [1.44-1.46] | 1.94 | 10.9 |
| 16384 | 3 | 122.2 | 240.3 | **1.97** [1.95-1.98] | - | - |
| 16384 | 4 | 122.2 | 308.9 | **2.52** [2.50-2.54] | 3.80 | 10.9 |
| 16384 | 5 | 122.2 | 343.1 | **2.80** [2.78-2.82] | - | - |
| 16384 | 6 | 122.2 | 412.5 | **3.38** [3.35-3.48] | 5.67 | - |
| 16384 | 7 | 122.2 | 477.6 | **3.92** [3.87-4.07] | - | - |
| 16384 | 8 | 122.2 | 535.8 | **4.40** [4.34-4.64] | 7.52 | 10.9 |

Reading: the cost of an extra row is about constant, 4.6 us (1k), 15.6 us (4k), 60 us (16k) per layer, i.e. 32 / 43 / 49 percent of T(1). Because T(k) grows almost linearly in k beyond k = 3, I read the kernel as limited by per-row instruction work (online-softmax exp2, the 4-stage shuffle reduction and the accumulator updates are per row and head, and the arithmetic order is fixed by the bitwise requirement) rather than by K/V bytes; this is an inference from the scaling and the two negative experiments below, not a measured profile. Per layer-step the extra cost vs the 1-row call is (T(k) - T(1)) x 36.

### Negative experiments (code removed, data kept)
* Grid order row-major (x = r * Nkv + kvh) instead of kvh-major: equal at k = 2 and up to 32 percent slower from k = 4 (16k, k = 8: 704 vs 534 us), so L2 reuse between the rows matters and the shipped order is the better one (`raw/.../order/`, slot #1991).
* R rows per threadgroup (RTG = 2, 3, 4, `raw/.../experiments/attention_rowsg_experiment.metal`): bitwise equal (tests passed for RTG 2, 3, 4, 8) but slower at every k and context (4k, k = 4: 81 us for one row per threadgroup vs 92 / 128 / 106 us for RTG 2 / 3 / 4; `raw/.../rtg/`, slot #1992).

## 2. Small ops for k rows (item 2)

Which existing kernels already handle k rows bitwise per row: **all of them**; they are one threadgroup (or thread range) per row with no cross-row reduction, and the launcher variant choice does not depend on the row count (`norm_ex(h)`, `argmax_f32_v4` by V / alignment; `QkNormRopeKv` takes the documented `_pf` form for T > 1 at head_dim 128 and was claimed bitwise to the decode form, now proven). Proof (`tests/g0b_small.rs`, 7 tests, passed in #1993): for k = 2..8 one launch on a `[k, ...]` batch vs k launches on the 1-row slices of the same inputs, all output bytes equal:

| op | cases | result |
|---|---|---|
| RmsNorm | H 4096, 2560, 1024, 128, 5120, element offset 0 and 16 | equal |
| RmsNormAdd (normed and sum) | H 4096, 2560, 128 | equal |
| Add, SiluGlu | 4096, 12288 per row | equal |
| Embedding | V 1000, H 4096 | equal |
| Argmax f32 and bf16 | V 151936, ties, max at first/last element | equal |
| KvAppend (unfused path) | pos0 0, 5, 100, cap - k | equal |
| QkNormRopeKv (q_out, k cache, v cache; row t at pos0 + t vs 1-row decode launches at pos0 + t) | 32/8/128, 16/4/64, 8/2/128 (rotary 64), 6/2/32; pos0 0, 1, 127, 128, 1000, 1500, cap - k | equal |

**No k-row variant is needed.** Cost at the Qwen3-8B shapes (same chain method, 210 rounds, per dispatch in a dependent chain):

| op | per decode step | T(1) us | T(2) | T(4) | T(6) | T(8) |
|---|---|---|---|---|---|---|
| RmsNormAdd [k,4096] | 2 per layer | 2.09 | 2.09 (1.00x) | 2.12 (1.01x) | 2.17 (1.04x) | 2.18 (1.04x) |
| RmsNorm [k,4096] | 1 per step | 1.80 | 1.81 (1.00x) | 1.83 (1.02x) | 1.90 (1.05x) | 1.92 (1.06x) |
| QkNormRopeKv T=k (32/8/128, `_pf` form) | 1 per layer | 2.49 | 2.52 (1.00x) | 2.53 (1.01x) | 2.54 (1.02x) | 2.54 (1.01x) |
| SiluGlu [k,12288] | 1 per layer | 1.06 | 1.11 (1.05x) | 1.23 (1.16x) | 1.50 (1.41x) | 1.61 (1.52x) |
| Add [k,4096] | 0 (fused into RmsNormAdd) | 0.89 | 0.91 (1.02x) | 0.94 (1.05x) | 0.98 (1.10x) | 1.01 (1.13x) |
| Embedding [k] -> [k,4096] | 1 per step | 1.23 | 1.23 (1.00x) | 1.26 (1.03x) | 1.30 (1.05x) | 1.35 (1.10x) |
| Argmax f32 [k,151936] | 1 per step | 12.50 | 12.55 (1.00x) | 12.62 (1.01x) | 12.78 (1.02x) | 12.85 (1.02x) |

Per step (k = 8, vs k = 1): about 36 x (2 x 0.09 + 0.05 + 0.55) us + 0.6 us = 0.03 ms. Not covered and not needed for G1: the accept/compare of drafted tokens against the argmax row ids (host side), the rollback of KV rows past the accepted length (positions are just not advanced; cache rows beyond stay unread).

## 3. Verify-step estimate (item 3)

`step(k) = step(1) + dGEMV(k) + 36 x (A(k) - A(1)) + dSmall(k)`. step(1) = 27.2 / 27.3 / 28.1 ms (overnight report, SAFE build, 128 / 1k / 4k). dGEMV from G0, census-weighted layers + lm_head: +0.04 / 0.15 / 0.17 / 0.34 / 0.75 / 1.90 / 5.03 ms for k = 2..8 (the R = 7..8 rise is the 8-byte shard-1 cliff, item 4). A(k) from section 1, dSmall from section 2. Everything else in a step (dispatch count, GPU idle gap) is that of a 1-token step. Break-even: i.i.d. acceptance probability a per drafted token, a step with k rows (1 certain + k - 1 drafted) yields E = (1 - a^k)/(1 - a) tokens, break-even at E = step(k)/step(1). The engine, drafter and rollback overhead (G1) are not included.

| ctx | k | dGEMV (ms) | dAttn (ms) | dSmall (ms) | step(k) (ms) | step(k)/step(1) | break-even alpha | tokens/step needed |
|---|---|---|---|---|---|---|---|---|
| 128 | 2 | 0.04 | 0.07 | 0.00 | 27.31 | 1.004 | 0.4% | 1.004 |
| 128 | 3 | 0.15 | 0.10 | 0.01 | 27.46 | 1.010 | 0.9% | 1.010 |
| 128 | 4 | 0.17 | 0.11 | 0.01 | 27.49 | 1.011 | 1.1% | 1.011 |
| 128 | 5 | 0.34 | 0.11 | 0.02 | 27.68 | 1.018 | 1.7% | 1.018 |
| 128 | 6 | 0.75 | 0.25 | 0.02 | 28.23 | 1.038 | 3.6% | 1.038 |
| 128 | 7 | 1.90 | 0.25 | 0.03 | 29.38 | 1.080 | 7.4% | 1.080 |
| 128 | 8 | 5.03 | 0.33 | 0.03 | 32.59 | 1.198 | 16.5% | 1.198 |
| 1024 | 2 | 0.04 | 0.39 | 0.00 | 27.73 | 1.016 | 1.6% | 1.016 |
| 1024 | 3 | 0.15 | 0.68 | 0.01 | 28.14 | 1.031 | 3.0% | 1.031 |
| 1024 | 4 | 0.17 | 0.78 | 0.01 | 28.26 | 1.035 | 3.4% | 1.035 |
| 1024 | 5 | 0.34 | 0.91 | 0.02 | 28.57 | 1.047 | 4.5% | 1.047 |
| 1024 | 6 | 0.75 | 1.04 | 0.02 | 29.11 | 1.066 | 6.2% | 1.066 |
| 1024 | 7 | 1.90 | 1.24 | 0.03 | 30.47 | 1.116 | 10.4% | 1.116 |
| 1024 | 8 | 5.03 | 1.37 | 0.03 | 33.73 | 1.236 | 19.1% | 1.236 |
| 4096 | 2 | 0.04 | 0.57 | 0.00 | 28.71 | 1.022 | 2.2% | 1.022 |
| 4096 | 3 | 0.15 | 1.03 | 0.01 | 29.29 | 1.042 | 4.1% | 1.042 |
| 4096 | 4 | 0.17 | 1.61 | 0.01 | 29.89 | 1.064 | 6.0% | 1.064 |
| 4096 | 5 | 0.34 | 2.15 | 0.02 | 30.61 | 1.089 | 8.2% | 1.089 |
| 4096 | 6 | 0.75 | 2.77 | 0.02 | 31.64 | 1.126 | 11.2% | 1.126 |
| 4096 | 7 | 1.90 | 3.31 | 0.03 | 33.34 | 1.186 | 15.7% | 1.186 |
| 4096 | 8 | 5.03 | 3.93 | 0.03 | 37.09 | 1.320 | 24.2% | 1.320 |

Net speed-up E / (step(k)/step(1)) at given per-draft acceptance (same model):

| ctx | k | a=0.3 | a=0.5 | a=0.7 | a=0.9 |
|---|---|---|---|---|---|
| 128 | 2 | 1.29x | 1.49x | 1.69x | 1.89x |
| 128 | 4 | 1.40x | 1.86x | 2.51x | 3.40x |
| 128 | 6 | 1.38x | 1.90x | 2.83x | 4.52x |
| 1024 | 2 | 1.28x | 1.48x | 1.67x | 1.87x |
| 1024 | 4 | 1.37x | 1.81x | 2.45x | 3.32x |
| 1024 | 6 | 1.34x | 1.85x | 2.76x | 4.39x |
| 4096 | 2 | 1.27x | 1.47x | 1.66x | 1.86x |
| 4096 | 4 | 1.33x | 1.76x | 2.38x | 3.23x |
| 4096 | 6 | 1.27x | 1.75x | 2.61x | 4.16x |

At k = 6 and 4k context the attention is already 2.8 of the 3.5 ms added, at 16k it would dominate (k = 4: +6.7 ms, roughly a quarter of a step, before any GEMV cost). Up to about 1k context the verify step costs under 7 percent for k <= 6.

## 4. R = 8 alignment remedy (item 4)

`examples/align_copy`: Qwen3-8B shard 1 has a 9,328-byte JSON header (data starts at byte 9,336, 8 mod 16) (shards 2-5 start at 0 mod 16). 81 tensors sit at 8 mod 16, 3.996 GB, of which 2.752 GB are streamed by decode GEMVs (the rest is `embed_tokens`, gathered, 1.245 GB). Copying the 2.752 GB into freshly allocated (page-aligned, hence 16-byte aligned) device buffers: alloc 0.9 ms, upload 154 ms (17.8 GB/s, single thread, page-cache warm) plus 92 ms to page the source in (a real load reads it from disk either way). **Memory cost +2.75 GB resident** (17 percent of the 16.4 GB model) unless the mmap'd source pages are dropped after the copy (they are file-backed and evictable, so the steady-state cost is the copy; the zero-copy wrap of those tensors is given up). **Load time about +0.15 s** (+0.25 s with first touch). The alternative of re-laying out the safetensors header padding offline is cheaper at runtime but changes the model files. Gain: R = 8 census-weighted T_layer ratio from 1.208 to 1.164 and the step from 1.197 to 1.156 (G0), i.e. dGEMV(8) from 5.03 to about 3.99 ms; for k <= 6 nothing (G0: R <= 6 unchanged within 1 to 3 percent). Not worth doing for the k <= 6 the numbers above favour.

## 5. Tests

* `cargo test -p intarsia-backend-metal --release` (slot #1993, `raw/.../final/cargo-test-metal.log`): **101 passed, 0 failed** (87 at G0 + 7 `g0b_attn` + 7 `g0b_small`).
* `intarsia kernels test --backend metal -m ~/models/Qwen3-8B` (binary built with `--features metal`; `final/kernels-test-metal.log`): 26 kernels, 290 cases, all ADMITTED, 0 rejected; the new `attn_prefill_bf16_splitkv_rows` (9 cases, max_abs 1.208e-3, rel_l2 3.634e-5 against the cpu reference, Deterministic grading as the other Metal attention kernels). The harness cannot grade against the Metal 1-row launch without a core change (same limit as G0), so the bitwise property is proven in the crate tests.
* Existing serving paths unchanged: all 1-row kernels untouched, new kernels at priority -1, `attn_hash` identical to B5.

## 6. Open points and limits

* **Device position**: `OpCall::check` allows `pos` only for `Rope`, `KvAppend`, `AttnDecode`, `QkNormRopeKv`. The kernel supports a device position (tested through the launcher), but a graph-replayed multi-row step on another backend or a Metal graph would need `AttnPrefill` in that list (crates/intarsia-core). The eager Metal path uses the host position and needs nothing.
* The multi-row kernel reproduces the *serving* partition, including the nsg 8 -> 4 switch at 1024 keys. If the 1-row heuristic changes (B5-style trimming), `askr_nsg` / `rows_nsg` must follow; `g0b_attn` fails loudly if not.
* Attention for k > 3 is the weakest piece (section 1 reading); any gain needs a different arithmetic order for the 1-row kernel too (out of scope: serving unchanged).
* Estimate uses the i.i.d. acceptance model, a fixed step(1) from the overnight report, and G0's step sums; the combined step was not run end to end.

## 7. Files

`kernels/metal/attention_splitkv.metal` (appended), `crates/intarsia-backend-metal/src/{attn,kernels,lib}.rs`, tests `tests/g0b_attn.rs`, `tests/g0b_small.rs`, examples `examples/g0b_bench.rs`, `examples/align_copy.rs`, raw data and scripts `docs/results/raw/2026-10-15-metal-g0b-attn/` (`slot.sh`, `run_slot.sh`, `analyze_g0b.py`, `estimate.py`, `make_doc_tables.py`; `final/` = the data of this report, `runA/` = first full run, `order/`, `rtg/`, `experiments/`).

## 8. fleetq and slot disclosure

Slots (endor, exclusive, 20 GB): #1990 (first full run), #1991 (grid order), #1992 (rows per threadgroup), #1993 (final run: tests, kernels test, bench, align_copy); all released by trap. #1989 was queued behind #1988 (another job of the same owner) and cancelled by me; **cancelling it made the waiting `fleetq acquire --wait` return and my wrapper script ran the 7 attention tests (12 s of small GPU kernels, 09:54:33-09:54:48 EDT) outside a slot while #1988 held endor exclusively**. Nothing was timed in that run and all its results were repeated inside #1990 and #1993, but #1988's own timings in that window may carry a small disturbance. The wrapper now checks that the job is running before it executes anything. DEVELOPER_DIR was exported and asserted in every slot script.
