# Metal backend: overnight report (2026-10-14 22:38 → 2026-10-15 05:11 EDT)

Qwen3-8B BF16 on endor (Apple M5 Max). Branch `spike/metal` at the commit that adds this report, on top of `a2102bf`. Local only, not pushed.

## Outcome

Every phase B decode criterion and every prefill-track level, including the 85% stretch, passes in a 3-round same-slot run against mlx-lm and llama.cpp with the GPU clock logged.

- **Prefill:** 95% of mlx-lm at 1k and 101% at 4k (SAFE); about 1.4x llama.cpp. It started yesterday at 7% of mlx-lm.
- **Decode:** 90% of the byte floor at short context, 7–9% ahead of mlx-lm and 4–5% ahead of llama.cpp.
- **Raw 1k time-to-first-token:** 0.27 s against mlx-lm's 0.43 s.

## Current metrics: clock-aware same slot, fleetq #1974, 04:50–05:11 EDT

There were 3 rounds of the 4 engines, in a different order each round, with a fresh server per pass, and `macmon` logged GPU clock and power throughout. GPU idle was 0–1% before each pass, Zoom was not running, and `pmset -g therm` showed no warnings. Values are medians over the 3 passes (each a localeval median of 3 reps). Detail is in [raw/2026-10-15-metal-final-clock/summary.md](raw/2026-10-15-metal-final-clock/summary.md).

| | decode 128 | decode 1k | decode 4k | prefill 1k (marginal) | prefill 4k (marginal) |
|---|---:|---:|---:|---:|---:|
| **intarsia SAFE** (`spike/metal`, chunk 256) | 36.7 | 36.6 | 35.6 | 4056 | 3494 |
| intarsia CAND (+ P3, chunk 2048) | 36.7 | 36.5 | 35.6 | 4373 | 3719 |
| mlx-lm 0.31.3 | 34.5 | 33.8 | 32.6 | 4268 | 3459 |
| llama.cpp (build 11429) | 35.4 | 35.0 | 33.8 | 2963 | 2470 |

All figures are in tok/s.

SAFE against each criterion:

| Criterion | Value | Result |
|---|---|---|
| Decode, % of the 40.6 tok/s byte floor (128 / 1k / 4k) | 90.5 / 90.1 / 87.6 | ≥ 85% PASS |
| Decode vs mlx-lm | 106.6–109.0% | ≥ 90% PASS |
| Decode vs llama.cpp | +3.7 to +5.1% | not below it: PASS |
| Prefill vs mlx-lm (1k / 4k) | 95% / 101% | committed ≥ 50%, target ≥ 65%, stretch ≥ 85%: all PASS |

**Why prefill numbers swung earlier in the night:** the GPU sometimes drops from about 1600 MHz / 90 W to 900–1300 MHz / 20–50 W in the middle of a prefill, for reasons not yet found (no thermal warning, and the low-power setting is already off). The serve path itself adds only 1–8 ms ([2026-10-15-metal-serve-prefill.md](2026-10-15-metal-serve-prefill.md)). In this slot it hit 1 pass of 12 (SAFE round 1: 4k at 2196 tok/s, against 3494–3501 in its other two passes). The medians above absorb it; dropping that pass moves no ratio by more than 0.2 points. The earlier 02:42 slot, where the 4k stretch held only on a mean of two runs, had no clock logging; its SAFE dip is consistent with this effect. CAND's 4k also varies by about 20% at full clock (3143–3869 tok/s at 1533–1606 MHz), cause unknown, so its 75% run then may be that instead.

**Caveats:**
- The thinnest margin is decode 4k at 87.6% of the floor, 2.6 points above the line.
- The 1k marginal figure subtracts each engine's 128-token time, which hides mlx-lm's and llama.cpp's larger fixed per-request cost.
- n = 3 passes per engine; ranges are min–max, not confidence intervals. The cause of the clock drop needs `sudo powermetrics` during a slow stretch.

**Plain prefill** (`generate`, fresh process per run, ABBA/BAAB). Each build ran at its own default chunk. 1k from the 02:42 slot (6 runs each), 4k and 16k from the 03:01 re-measure (8 runs each, fleetq #1966).

| Prompt | SAFE ms (median) | CAND ms (median) | CAND speed vs SAFE |
|---|---:|---:|---:|
| 1k | 444 | 378 | 1.17x |
| 4k | 1308 (1291–1342) | 1167 (1155–1183) | 1.12x, no overlap |
| 16k | 7024 | 6558 | 1.07x (GPU reached 87 °C, noisier) |

**Accuracy** (SAFE and CAND are identical):
- Long golden prefill: top-1 1430/1434, KL 3.699e-4.
- Short golden prefill: top-1 434/434, KL 2.832e-4.
- Decode: top-1 100%, KL 3.496e-4 / 3.517e-4.
- Gate: KL ≤ 1e-3, top-1 ≥ 99.5%.
- Greedy ids are identical between SAFE and CAND. SAFE `qualify`: 10 PASS, 2 SKIP (`ab`, `tp`).

## Change since yesterday

Each row is its own same-slot localeval run; rows come from different slots and machine states.

| | decode 128 / 1k / 4k | prefill 1k / 4k |
|---|---|---|
| Same-slot baseline 2026-10-14 17:05 (`a296a78`) | 35.6 / 35.4 / 34.5 | 350 / 205 |
| After P1 + P2 (`2e1fdb1`, 22:10) | 36.2 / 36.2 / 35.1 | ~3300–3550 / ~1950–2200 |
| **Now** (`a2102bf`, 05:00 slot) | **36.7 / 36.6 / 35.6** | **4056 / 3494** |

## What merged into `spike/metal`

| Commit | Change | Gain |
|---|---|---|
| `495a447` | B5: decode split-KV attention ramp trims | 4k decode 34.2 → 34.7 |
| `9390beb` | B6: decode small-kernel ramp trims (another session, task chip) | 4k decode → 35.3 |
| `05f0b8b` | P2: flash prefill attention (simdgroup_matrix) | attention 0.6 → 12 TFLOPS |
| `2e1fdb1` | P1: NAX bf16 GEMM and pair GEMM (Metal 4 `matmul2d`) | matmul 5.5 → ~54–65 TFLOPS; 4k prefill ~380 → ~2000+ |
| `766c723` | `qualify --resume` reruns SKIP stages (another session, task chip) | n/a |
| `a710f35` | D1: decode dispatch fusion, doc only (no gain, code reverted) | none |
| `6271eec` | P4/P5: NAX tile pick for 128 ≤ M ≤ 512, prefill `qk_norm_rope_kv` | chunk-256 4k prefill +5–7% |
| `a2102bf` | F1b: NAX flash prefill attention | 38–50 TFLOPS; 4k prefill −17%, 16k −40% |

**Accuracy history** (long prefill KL / top-1):
- B4: 3.738e-4
- P2: 4.119e-4
- P1: 4.342e-4 / 1429
- F1b: **3.699e-4 / 1430**

Short prefill was 2.832e-4 with 434/434 at the end of the night.

On the F1b top-1 number, its agent notes that kernels with equal maths span 1426–1430/1434, so part of the 1430 is selection on noise. The principled change is that the softmax normaliser is now summed over the same bf16-rounded P that multiplies V.

## Held for your decision

1. **P3: Metal prefill chunk 2048** (`metal-p3-chunk`; integrated as `metal-candidate2`, which is `a2102bf` plus P3).
   - **What it buys:** plain prefill 1.17x at 1k, 1.12x at 4k and 1.07x at 16k. Serve-path prefill +8% at 1k (all 3 passes ahead, no overlap); at 4k +6% on the median but one of 3 passes was 10% below SAFE, because CAND 4k varies 3143–3869 tok/s even at full GPU clock, for a reason not found. 4k serve is unresolved. Accuracy and greedy ids are identical.
   - **Why it's held:** fresh `qualify` FAILs `acc-long` on the prefix-reuse gate. The longest long-golden prompt is 700 tokens, and any chunk above about 350 skips every prompt, which counts as a fail. It is a coverage gap, not a regression.
   - **Options:**
     - (a) Build a Qwen3-8B fp32 golden with 4k prompts (`tools/prompts/golden_4k.jsonl` exists). This needs spark1 with vLLM taken down (`--takes-down flight,guidance,capcom`, see the incident below), or an fp32 run on endor's CPU.
     - (b) Accept 2048 with the gap.
     - (c) Keep 256 for now.
   - **My recommendation:** (c) today, then (a) soon: the gain is 7–17% in plain prefill and +8% at 1k serve, which is worth closing the gate for. (a) on spark1 takes the vLLM roles down for the run; on endor's CPU it is slower but takes nothing down.
2. **Push and merge to main.** `spike/metal` is local only.
3. **CUDA/HIP no-op check.** No `intarsia-core` file changed tonight. P1, P2, P4/P5 and F1b touched only the Metal backend crate, Metal kernels and docs. The last check (`a296a78`) therefore still covers the core. A re-run is optional.

## Incident: spark1 hung 00:20–01:38 EDT

- **What happened:** my fp32 golden job (fleetq #1937, 45 GB, non-exclusive) ran beside spark1-vllm. GB10 GPU memory isn't charged to the container's memory limit, so the box thrashed. ssh hung, and spark1-vllm (flight, guidance, capcom) was unreachable for about 78 minutes.
- **Recovery:** the kernel OOM-killed the container, and the box came back without a reboot. vLLM is up, the job has ended and the temp files are removed.
- **Lesson:** an fp32 8B golden on spark1 needs `--takes-down` of vLLM, never a run beside it. unanimis `b99b5e8e`.

## Tried and dropped

- **D1, decode fusion:** three bitwise-identical fusions, down to 222 dispatches. All were flat or slower, because the GEMVs are at bandwidth and recomputing the norm in every threadgroup costs more than the dispatches it saves.
- **P4/P5:** a q+k+v triple GEMM (≤ 0.5%), a SwiGLU epilogue (slower and produced wrong bits, cause not found) and a residual-add epilogue (≤ 0.7%).
- **F1b:** fp16 and fp32 P. Neither improved accuracy, and fp16/fp32 V costs 29% speed.
- **B5:** a last-arriver fused combine. It was slower.

## Where it stops

- **Prefill matmuls:** 58–65 TFLOPS against a measured 64.6 peak.
- **Prefill attention:** 38–50 TFLOPS.
- **Decode GEMVs:** about 600 GB/s against a 614 GB/s floor.
- **Remaining decode headroom:** about 0.5–0.7 ms/token in small ops (D1 estimate).
- **Next big levers:** outside kernel tuning. They are speculative decoding, and multi-request batching for serve.

## Leftovers

- **Worktrees** from the final run: `/opt/code/intarsiaml-final-cand` and `/opt/code/intarsiaml-final-safe`, plus agent worktrees under `.claude/worktrees/`. They are safe to remove once you've decided on P3.
- **Branches:**
  - `metal-p3-chunk` and `metal-candidate2`: held.
  - `metal-f1-nax-attn` and `metal-candidate`: superseded by F1b.
  - `metal-d1-fuse`: doc only, merged.
