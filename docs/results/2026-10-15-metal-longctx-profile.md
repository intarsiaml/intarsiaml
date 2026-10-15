# Metal long-context profile: prefill and decode at 4k / 16k / 32k / 40k (read-only)

Date 2026-10-15. Machine endor (MacBook Pro, Apple M5 Max 40-core GPU, 128 GB, 614 GB/s). Branch `metal-longctx-profile`, a worktree off `spike/metal` at `c556008` (P3 chunk 2048 is the default). **No serving code changed**; the only code artefact is `raw/2026-10-15-metal-longctx-profile/nax-flash-s16-s32-experiment.patch` (two extra NAX-flash instantiations, used for one experiment, never committed to the crates). Qwen3-8B BF16, `intarsia generate` / `intarsia serve --backend metal`. Raw data and scripts: [raw/2026-10-15-metal-longctx-profile/](raw/2026-10-15-metal-longctx-profile/) (index in section 7).

## Verdict

1. **Prefill: yes, attention is the long-context limiter, and it is a FLOP-count effect, not an efficiency collapse.** Kernel time at a fixed reference clock is 1.04 s at 4k, 5.33 s at 16k, 14.2 s at 32k, 18.9 s at 40k. A two-term FLOP model (matmul 13.89 GFLOP per token at 62.9 TFLOPS, attention 294,912 N^2 FLOP at 48.5 TFLOPS) lands within 2% of every point. Attention is 9% of prefill at 4k, 30% at 16k, 46% at 32k, 51% at 40k; it equals the matmuls at about 36k tokens. The "5.6x the time for 4x the tokens" is matmul 3.98x plus attention 16.2x: 5.1x at a constant clock (the rest is the GPU clock drifting down during a long run, below).
2. **NAX flash attention runs at 46-51 TFLOPS at every position once the GPU clock is normalised** (50.5 at 4k, 49.9 at 16k, 48.2 at 32k, 48.4 at 40k, whole-prompt), against 62-63 for the matmuls in the same trace. It does not collapse with context; the raw per-chunk figure falls from 50.7 to 39 only because the GPU clock falls with it (the matmuls of the same chunk slow by the same factor). Closing the gap to the matmul rate is worth up to -12% of a 32k-40k prefill and -7% of 16k (section 6, lever P1).
3. **The GPU does not hold its clock on a long run on this laptop.** Cold, a 4k prefill runs at 1617 MHz / 75-95 W; after a few seconds to minutes of sustained load it sits at 1100-1250 MHz / 35-45 W, and a 40k prefill never held 1450 MHz. Measured plain prefill is 15-25% slower in that state (16k: 5.9 s cold, 7.1 s sustained; 40k: 21.7 s best, 25.5-27.4 s sustained). It is not a thermal-warning state (`pmset -therm` is clean, GPU die 73-79 C at the low clocks, 90+ C at the high ones), the cause is not found (section 1), and it makes every absolute long-context prefill number a range.
4. **Decode is at 89.7-91.6% of a context-aware byte floor at every context from 256 to 34k** (localeval, served), and the in-stream split-KV attention reads K/V at 550-575 GB/s (90-94% of 614) from 16k up. The fixed part of a decode step is flat at 26.4-27.2 ms (weights 24.65 ms at the floor plus 0.75 ms small ops plus 0.33 ms idle gap); everything that grows with context is the KV read, and it is already near its byte floor. Kernel tuning of decode attention is worth at most 2% at long context; **the levers that matter are byte-reducing (KV cache quantisation, +7% at 16k to +15% at 40k) and verify-step attention for speculative decoding**.
5. **Contexts above 40,960 tokens cannot be served by this engine**: `max_position_embeddings` is 40960 and `rope_scaling` is null, and `memplan.rs` clamps the context to it (KV is only 6 GB there, so memory is not the limit). 64k appears below as kernel-only microbenchmarks and a model extrapolation, not as an engine run.

## 1. Setup, method and the caveats that matter

* Binary: `cargo build --release -p intarsia-cli --features metal` at `c556008` with `DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer` (asserted in every slot script). Prompts: synthetic "A patient gardener ... on day N." text (same pattern as `prompt-16k.txt` of phase B) with an exact Qwen3 token count: 4096, 16384, 32764 (the file is named `prompt-32768.txt`, 4 tokens were trimmed), 40000 (`raw/.../mkprompts.py`, `prompts/`). Chunk 2048 (the default), 129 forced tokens (128 decode steps) unless noted.
* **Three kinds of trace** (`INTARSIA_METAL_PROFILE`, `crates/intarsia-backend-metal/src/profile.rs`, no patch needed): `kernel:` (one command buffer per launch, per-op GPU time; each record carries a few microseconds of command-buffer start/end cost), `cb:` (stream as normal; GPU busy and idle per command buffer; a decode step is three buffers of at most 128 dispatches), and plain runs (wall time from the engine's own `prefill ... | decode ...` line). The per-op split comes from `kernel:`, step and prefill totals from `cb:` and plain. Decode per-op times in this doc are **medians per call x calls per token**: the mean was pulled up by clock-state outliers (decode attention at 16k: mean 196 us, median 132 us in the same trace).
* **Speed numbers served** are from localeval (`speed`, sizes 128,4096,16384,32768, 256 forced tokens, 3 reps, `intarsia serve --max-seq 40960`, run `20261015-115639-adhoc-metal-longctx`, results in `raw/.../localeval/`). The engine-side `generate` figures are used for the breakdown only.
* **GPU clock.** `macmon pipe -i 100` ran in every slot. The prefill window of each run is isolated (process end - decode time - 0.3 s, back by the prefill time; busy samples only) and its median clock reported in `raw/.../analysis/runs.csv`. Samples/runs under 1450 MHz are **kept but labelled**, because on this machine most of the sustained part of every long run is under 1450 MHz and discarding them would leave no 32k/40k prefill number at all. Where a number needs a clock-independent form I use the **matmul thermometer**: every prefill chunk contains the same matmuls, so a chunk's matmul time says what clock/power state the GPU was in; attention time is rescaled by (best chunk matmul time / this chunk's matmul time), and this is labelled "at the reference clock". It assumes attention (compute-bound, same NAX units) scales with the clock like the matmuls do.
* **What the environment did.** The Mac is a laptop in use (Safari/WebKit at 40-60% CPU, WindowServer, OrbStack, a context-compression proxy at times burning 6 cores). Two regimes alternated: *cold* (about 1450-1617 MHz, 60-95 W, first minutes of a slot, also after idle) and *sustained* (1100-1250 MHz, 35-45 W, GPU-busy 70% by macmon, die 73-79 C). I could not find a trigger. It is not memory pressure (40% free, swap used but idle), not a thermal-warning level, not the power mode (`powermode 2`). Decode, being bandwidth-bound, is hardly affected (35.2 vs 34.2-35.5 tok/s at 4k in both); compute-bound prefill is. The earlier overnight report saw the same effect ("GPU drops from about 1600 MHz / 90 W to 900-1300 MHz / 20-50 W in the middle of a prefill") without a cause.
* fleetq: jobs #2009 (slots 1-3 of this work; the lease was kept while the scripts were replaced twice, one replacement exited 126 and released it) and #2022 (slot 4). Exclusive on endor, `--mem 60`, label "metal longctx profile". Job IDs and times in `raw/.../fleetq-jobs.txt`.
* **Not done**: no 64k engine run (limit above); no `sudo powermetrics` for the clock behaviour; 40k has no run that held 1450 MHz; one rep per kernel-mode trace.

## 2. Derived floors

Decode streams every weight once per token and reads the whole KV cache once per token.

* Weights streamed per token: per layer q 4096x4096, k and v 1024x4096 each, o 4096x4096, gate and up 12288x4096 each, down 4096x12288 = 192,937,984 parameters; x 36 layers = 6,945,767,424; plus the lm_head 151936x4096 = 622,329,856; **7,568,097,280 parameters x 2 B = 15.1368 GB** (the embedding is a gather, not streamed). At 614 GB/s: 24.65 ms, 40.6 tok/s. This is the floor of earlier docs ("weights-only").
* KV bytes read per token per visible position: 36 layers x 8 KV heads x 128 head_dim x 2 (K and V) x 2 B (bf16) = **147,456 B**. At the mean visible context L over the tokens measured: bytes/token = 15.1368 GB + 147,456 B x L; floor = bytes / 614 GB/s.
* The 87.6% at 4k in the overnight report is 35.6 / 40.6 (weights-only floor). Against the context-aware floor at 4k (39.0 tok/s) the same 35.6 tok/s is **91.3%**; the two reports do not disagree, the 4k figure just charges 1 ms of KV reads to the engine.

| mean context L | KV read per token | bytes per token | floor ms | floor tok/s |
|---:|---:|---:|---:|---:|
| 256 | 0.04 GB | 15.17 GB | 24.71 | 40.5 |
| 4,160 | 0.61 GB | 15.75 GB | 25.65 | 39.0 |
| 16,448 | 2.43 GB | 17.56 GB | 28.60 | 35.0 |
| 32,828 | 4.84 GB | 19.98 GB | 32.54 | 30.7 |
| 40,064 | 5.91 GB | 21.04 GB | 34.27 | 29.2 |
| 65,536 (kernel-only) | 9.66 GB | 24.80 GB | 40.39 | 24.8 |

Prefill floor: compute-bound. Matmul FLOPs 2 x 6,945,767,424 = 13.89 GFLOP per token (lm_head runs on the last row only); attention FLOPs 4 x 128 x 32 heads x 36 layers x N^2/2 = 294,912 N^2 (causal). Peak matmul measured in these traces is 62.9-63.3 TFLOPS (earlier docs: 64.6). At 62.9 TFLOPS the matmul part alone is 0.2208 ms/token (4,528 tok/s).

## 3. Prefill breakdown

### 3.1 Per-op GPU time (kernel mode, chunk 2048)

Per class, milliseconds for the whole prompt. The left block is as measured (the clock state of that run, see 3.3); the right block is rescaled to the reference clock (matmul best chunk 62.9 TFLOPS) with the matmul thermometer. Norms/rope is rmsnorm, rmsnorm_add and the fused qk_norm_rope_kv; "other" is silu_glu, embedding, argmax. Source: [raw/.../analysis/prefill-class-table.md](raw/2026-10-15-metal-longctx-profile/analysis/prefill-class-table.md).

| context | chunks | matmul | attention | norms/rope | other | total | attention share | matmul TFLOPS | attention TFLOPS |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4,096 | 2 | 907 | 98 | 21 | 19 | 1,045 | 9% | 62.9 | 50.4 |
| 16,384 | 8 | 3,675 | 1,646 | 78 | 75 | 5,474 | 30% | 62.0 | 48.1 |
| 32,764 | 16 | 8,427 | 7,691 | 167 | 163 | 16,449 | 47% | 54.0 | 41.2 |
| 40,000 | 20 | 9,700 | 11,075 | 197 | 193 | 21,166 | 52% | 57.3 | 42.6 |

At the reference clock:

| context | matmul | attention | norms/rope + other | total | attention share | attention TFLOPS | tok/s (kernel time) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 4,096 | 903 | 98 | 40 | 1,041 | 9% | 50.5 | 3,935 |
| 16,384 | 3,596 | 1,587 | 150 | 5,333 | 30% | 49.9 | 3,072 |
| 32,764 | 7,319 | 6,570 | 288 | 14,177 | 46% | 48.2 | 2,311 |
| 40,000 | 8,835 | 9,749 | 357 | 18,941 | 51% | 48.4 | 2,112 |

Per kernel (median microseconds per call, full 2048-row chunks, 36 layers per chunk; the matmul rows at 32k are in the sustained-clock state, 15% slower than at 16k, which is the clock and not the shape): [raw/.../analysis/perop.md](raw/2026-10-15-metal-longctx-profile/analysis/perop.md).

| kernel | n x k | calls/chunk | 4k | 16k | 32k | 40k |
|---|---|---:|---:|---:|---:|---:|
| attn_prefill_bf16_flash_nax | | 36 | 1,664 | 5,575 | 12,843 | 14,599 |
| matmul_pair_bf16_nax (gate, up) | 24576x4096 | 36 | 6,320 | 6,337 | 7,299 | 6,828 |
| matmul_bf16_nax (down) | 4096x12288 | 36 | 3,335 | 3,338 | 3,873 | 3,666 |
| matmul_bf16_nax (q, o) | 4096x4096 | 72 | 1,105 | 1,112 | 1,293 | 1,214 |
| matmul_pair_bf16_nax (k, v) | 2048x4096 | 36 | 608 | 613 | 697 | 646 |
| silu_glu_bf16 | | 36 | 256 | 257 | 286 | 261 |
| rmsnorm_add_bf16 | | 72 | 94 | 83 | 89 | 88 |
| qk_norm_rope_kv_bf16 | | 36 | 98 | 89 | 102 | 98 |

The attention row is the median over all chunks of the prompt, so it mixes positions (the last chunk of a 40k prompt sees 38k keys, the first none). Norms, rope, silu and the residual adds together are 1.5-3% at every context and flat per chunk; nothing there depends on context.

### 3.2 Why 4x the tokens costs 5x (not 5.6x) of the time

Matmul work is linear in tokens (3.98x from 4k to 16k at the reference clock), attention work is quadratic (16.2x). A two-term FLOP model reproduces the kernel-time totals:

| context | matmul s (13.89 GFLOP/token at 62.9) | attention s (294,912 N^2 at 48.5) | model total | measured, reference clock | attention share (model / measured) |
|---:|---:|---:|---:|---:|---:|
| 4,096 | 0.90 | 0.10 | 1.00 | 1.04 | 10% / 9% |
| 16,384 | 3.62 | 1.63 | 5.25 | 5.33 | 31% / 30% |
| 32,764 | 7.23 | 6.52 | 13.75 | 14.18 | 47% / 46% |
| 40,000 | 8.83 | 9.73 | 18.56 | 18.94 | 52% / 51% |
| 65,536 (extrapolated, not runnable here) | 14.47 | 26.12 | 40.6 | | 64% |

Attention equals the matmuls at N = pm/62.9 x 48.5/294,912, about 36,300 tokens. Beyond that each additional token is mostly attention. The overnight report's 5.6x (6.56 s / 1.17 s) is the same shape plus clock: its 16k run reached 87 C and the plain 16k prefill here costs 5.9 s cold and 7.1 s sustained (**5.0x** the 4k time in both states, 1.19 s cold / 1.44 s sustained at 4k).

### 3.3 Wall time, GPU busy time and the clock

| context | kernel-mode sum, reference clock | cb-mode GPU busy (run median MHz) | plain `generate` prefill, cold (median MHz) | plain, sustained (median MHz) | served TTFT (localeval, 33.9k = its 32768 row) |
|---:|---:|---:|---:|---:|---:|
| 4,096 | 1.04 s | 1.05 s (1617) | 1.19-1.21 s, 3,430 tok/s (1617) | 1.33-1.90 s (1330-1400) | 1.10 s, 3,724 tok/s (4,113 tokens) |
| 16,384 | 5.33 s | 5.68 s (1510) | 5.90 s, 2,778 tok/s (1617) | 7.1-7.7 s (1180-1240) | 6.28 s, 2,654 tok/s (16,677 tokens) |
| 32,764 | 14.18 s | 16.2-17.3 s (1300) | 15.8 s, 2,070 tok/s (1450-1473) | 18.5-18.8 s (1190-1220) | 19.06 s, 1,780 tok/s (33,926 tokens) |
| 40,000 | 18.94 s | 21.4 s (1442) | none held 1450 MHz | 25.5-27.4 s (1160-1200) | |

Cb-mode prefill GPU is busy 99.1-100% of the span at every context, so there are **no host-side gaps in prefill**; the wall time is GPU time at whatever clock the GPU had. Plain/serve prefill at 4k and 16k therefore sits 6-15% above the reference-clock kernel sum even when cold (the clock was 1510-1617 MHz and the first chunks run before the GPU ramps). The decode that follows a prefill is little affected by the same state (section 4).

### 3.4 Attention efficiency against context and chunk

* **Position (engine, in situ).** Per chunk of the 40k kernel trace (full table `analysis/norm-ctx40000.md`), attention TFLOPS raw / at the reference clock: pos0 2,048: 50.7 / 50.7; 8,192: 49.2 / 50.3; 16,384: 45.0 / 49.2; 24,576: 42.2 / 48.2; 32,768: 41.7 / 48.0; 36,864: 39.8 / 47.8. The matmuls of the same chunks fall from 62.9 to 52.3 TFLOPS over the run, i.e. the clock accounts for essentially all of the raw decline; a small real decline of about 6% from 2k to 39k (50.7 to 47.5) remains, which fits more K/V traffic per q-block as the context grows (section 3.5).
* **Kernel microbenchmark, chunk 2048, pos0 0 / 2,048 / 8,192 / 16,384 / 30,720 / 38,912** (`micro/flash-nax.txt`, slot 1, cold start, no clock gating, one ordered pass so heat drift is mixed in): 50.9 / 44.4 / 40.5 / 41.1 / 36.4 / 36.2 TFLOPS. The decline there is the same clock drift (these microbenchmarks heated the GPU up as they went); use the in-situ numbers above and the gated slot-4 points below.
* **Chunk size.** Earlier docs (`raw/2026-10-15-metal-prefill-f1b/speed/bench-final-pos0-c256.txt` and `-c2048.txt`): at chunk 256 the NAX kernel runs 21 / 37 / 41 / 42 TFLOPS at pos0 0 / 1,024 / 3,840 / 15,872, at chunk 2048 50 / 46 / 47 / 45 at pos0 0 / 2,048 / 4,096 / 14,336. <<PENDING-CHUNK>>

### 3.5 Is attention the limiter?

Yes beyond about 16k, in the sense that it is the part that grows faster than the tokens: at 32k and 40k it is the larger half of the time. But the kernel is not collapsing with context (46-51 TFLOPS everywhere at the reference clock). What attention costs at long context is its FLOP count, and the only way to make it cheaper without changing the maths is a faster kernel, 12% of a 40k prefill at the very best (lever P1), and for the very long prompts the GPU clock matters nearly as much (a 40k prefill that costs 18.9 s at the reference clock costs 25.5-27.4 s when the laptop settles into its sustained state, +35-45%).

<<PENDING-NSG>>

## 4. Decode

### 4.1 Speed against the context-aware floor

Served (localeval, `intarsia serve --max-seq 40960`, chunk 2048, 256 forced tokens, median of 3, run `20261015-115639-adhoc-metal-longctx`). The mean context is the prompt plus half the 256 generated tokens.

| prompt tok | mean context | bytes/token | floor tok/s | decode tok/s (min-max) | % of context-aware floor | % of weights-only floor (40.6) |
|---:|---:|---:|---:|---:|---:|---:|
| 136 | 264 | 15.17 GB | 40.5 | 36.3 (36.1-36.4) | 89.7% | 89.5% |
| 4,113 | 4,241 | 15.76 GB | 39.0 | 35.2 (35.0-35.3) | 90.4% | 86.8% |
| 16,677 | 16,805 | 17.61 GB | 34.9 | 31.8 (31.7-31.9) | 91.2% | 78.4% |
| 33,926 | 34,054 | 20.16 GB | 30.5 | 27.9 (27.9-28.1) | 91.6% | 68.8% |

Engine-side (`generate`, cb-mode GPU timeline, 128 decode steps after the prompt, same hardware, not served):

| context | mean context | floor ms | GPU busy per step | step period | tok/s | % of floor | idle gap between steps |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 4,096 | 4,160 | 25.65 | 27.78 ms | 28.10 ms | 35.6 | 91.3% | 0.32 ms |
| 16,384 | 16,448 | 28.60 | 31.07 ms | 31.40 ms | 31.9 | 91.1% | 0.33 ms |
| 32,764 | 32,828 | 32.54 | 34.97 ms | 35.31 ms | 28.3 | 92.1% | 0.34 ms |
| 40,000 | 40,064 | 34.27 | 37.52 ms | 37.86 ms | 26.4 | 90.5% | 0.34 ms |

GPU busy fraction is 98.9-99.1% at every context. The 16k and 32k and 40k decode speeds are new measurements (the overnight report stopped at 4k). Decode is clock-insensitive: plain runs in the cold and the sustained states give 35.2 and 34.2-35.4 tok/s at 4k, 31.1-32.0 at 16k, 26.0-28.0 at 32k-40k.

### 4.2 Where a step goes

Per generated token. Weights are the GEMVs (kernel mode, median x calls); attention is the in-stream split-KV time of `attn_bw` (a serial chain of launches over distinct per-layer caches, the same way a decode step issues them), scaled to the mean context; small ops are rmsnorm_add, qk_norm_rope_kv, silu_glu, argmax, rmsnorm, embedding at kernel-mode medians; idle is the gap between steps in cb mode.

| mean context | GEMVs (weights) | attention | small ops | idle gap | sum | measured step period | floor |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 4,160 | 25.9 ms | 1.33 ms | 0.75 ms | 0.32 ms | 28.3 ms | 28.10 ms | 25.65 ms |
| 16,448 | 25.9 ms | 4.41 ms | 0.75 ms | 0.33 ms | 31.4 ms | 31.40 ms | 28.60 ms |
| 32,828 | 25.9 ms | 8.61 ms | 0.75 ms | 0.34 ms | 35.6 ms | 35.31 ms | 32.54 ms |
| 40,064 | 25.9 ms | 10.28 ms | 0.75 ms | 0.34 ms | 37.3 ms | 37.86 ms | 34.27 ms |

The GEMVs (25.9 ms: pair 24576x4096 12.2 ms, down 6.2, q/o 4.3, lm_head 2.1, k/v pair 1.1) are the weights at 585 GB/s, 95% of 614, and **do not depend on context**. The excess over the floor decomposes (16k): 1.2 ms GEMV shortfall, 0.46 ms attention shortfall (4.41 ms against 2.43 GB / 614 GB/s = 3.95 ms), 0.75 ms small ops, 0.33 ms idle = 2.74 ms of the measured 2.80 ms. Per-op tables with the kernel-mode means (affected by clock outliers) are in `analysis/ctx*-c2048.md`; medians in [analysis/perop.md](raw/2026-10-15-metal-longctx-profile/analysis/perop.md).

### 4.3 Decode attention bandwidth on the KV read

`attn_bw` (36 layers, one distinct KV cache each so every call reads DRAM, 12 reps, cb-timed; Qwen3-8B shape 32/8/128), K+V bytes = visible keys x 4,096 B per call. The shipped policy: nsplit = clamp(min(ceil(keys/128), ceil(120/8)), 1, 64), i.e. 15 splits of 4 simdgroups (120 threadgroups) from 1,793 keys on. The older single-pass kernel is shown for scale.

| visible keys | split-KV us/call | GB/s | % of 614 | single-pass us | trace median (kernel mode) us |
|---:|---:|---:|---:|---:|---:|
| 1,024 | 14.6 | 287 | 47% | 32.4 | |
| 4,096 | 36.2 | 464 | 76% | 104.6 | 45.4 |
| 8,192 | 66.7 | 503 | 82% | 202 | |
| 16,384 | 122.0 | 550 | 90% | 392 | 131.9 |
| 32,768 | 236-245 | 549-568 | 89-93% | 777 | 246.1 |
| 40,000 | 285 | 575 | 94% | 946 | 295.0 |
| 65,536 (kernel-only) | 468 | 573 | 93% | 1,542 | |

Below 16k there is a roughly fixed 8-9 us per launch (two dependent dispatches) on top of the byte time: 4k is 36.2 us against 27.3 us of bytes (0.3 ms per step, 1.1%). From 16k up, **the kernel is within 6-10% of the DRAM bandwidth**. The kernel-mode medians are 9-11 us above the chain numbers (the extra command-buffer start/end).

<<PENDING-SPLIT>>

## 5. Speculative-decoding verify step at long context

Not re-measured (the k-row attention kernel is on `metal-g0b-attn`, not in `spike/metal`). Using its table (`docs/results/2026-10-15-metal-g0b-attn.md` on that branch): T(k)/T(1) = 2.2x at k = 4 for 4k context and 2.5x at 16k, T(1) = 36.4 us at 4k and 122.2 us at 16k, which agrees with the in-stream numbers above (36.2 and 122.0 us). Extra attention per verify step at k = 4: (T(4) - T(1)) x 36 = 1.6 ms at 4k and 6.7 ms at 16k; scaling the 16k per-row cost (60 us per extra row and layer, a constant that g0b found ALU-bound rather than byte-bound) to 40k keys gives roughly 15 ms, extrapolated and not measured. On a 31.4 ms step at 16k that is 21%, so the verify-step break-even acceptance at 16k is far above the 6.0% g0b quotes for 4k.

## 6. Ranked levers

Gains are estimated from this profile's numbers; none was implemented (read-only task) except the NSG experiment, which is a measurement. "Exact" means bitwise-identical output to today's kernels.

<<PENDING-LEVERS>>

## 7. Raw data index

Everything is under `docs/results/raw/2026-10-15-metal-longctx-profile/`:

* `mkprompts.py`, `prompts/`: the four prompts (token counts exact).
* `common.sh`, `slot.sh`, `slot2.sh`, `slotwork.sh`, `slot2work.sh`, `slot3work.sh`, `slot4work.sh`: the slot scripts as run (slot 1 `slotwork.sh`; `slot2work.sh`/`slot3work.sh` were replaced mid-run, their partial outputs are in `s2/`, `s3/`; `slot4work.sh` is the final batch in `s4/`). `fleetq-jobs.txt`, `slot*.log`.
* `kernel/`, `cb/`, `plain/`, `localeval/`, `clock/` (macmon at 100 ms, versions, idle checks): slot 1. `s2/`, `s3/`, `s4/`: the same for the later slots (`s2/kernel/ctx32768-*` is the 32k kernel and cb trace; slot 1 missed 32k because of a prompt-file name).
* `micro/`, `s*/exp/`: `flash_bench` and `attn_bw` outputs, split-policy sweeps, the NSG experiment.
* `analyze.py` (per-op tables from a trace), `chunk_norm.py` (matmul thermometer), `tables.py`, `perop.py`, `decode_table.py`, `report_plain.py` (all runs with prefill-window clock, `analysis/runs.csv`), `summarize.py`, `phase_clock.py`, `pclock_check.py`, `wclock.py`.
* `analysis/`: generated tables quoted above.
* `nax-flash-s16-s32-experiment.patch`: the experiment patch (the only code change; applied in a scratch build outside the worktree, never committed to `crates/` or `kernels/`).
