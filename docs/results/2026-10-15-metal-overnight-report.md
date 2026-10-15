# Metal backend: overnight report (2026-10-14 22:38 → 2026-10-15 03:00 EDT)

Qwen3-8B BF16 on endor (Apple M5 Max). Branch `spike/metal` at the commit that adds this report, on top of `a2102bf`. Local only, not pushed.

## Outcome

Every phase B decode criterion and every prefill-track level passes, including the 85% stretch, in a single same-slot run against mlx-lm and llama.cpp.

- **Prefill:** about 94–96% of mlx-lm and 140–165% of llama.cpp. It started yesterday at 7%.
- **Decode:** 90% of the byte floor at short context, and 3–5% ahead of llama.cpp.

## Current metrics: same slot, fleetq #1965, 02:42–02:53 EDT

The run order was SAFE, CAND, mlx-lm, llama.cpp, CAND, SAFE. GPU idle was 0–1% before each run and Zoom was not running. Values are localeval medians of 3 reps. Detail is in [2026-10-15-metal-final-sameslot.md](2026-10-15-metal-final-sameslot.md).

| | decode 128 | decode 1k | decode 4k | prefill 1k (marginal) | prefill 4k (marginal) |
|---|---:|---:|---:|---:|---:|
| **intarsia SAFE** (`spike/metal`, chunk 256), mean of 2 runs | 36.7 | 36.6 | 35.6 | 4054 | 3126 |
| intarsia CAND (+ P3, chunk 2048), mean of 2 runs | 36.8 | 36.5 | 35.6 | 4083 | 3169 |
| mlx-lm 0.31.3 | 34.5 | 34.0 | 32.7 | 4309 | 3244 |
| llama.cpp (build 11429) | 35.4 | 35.1 | 34.0 | 2885 | 1897 |

All figures are in tok/s.

SAFE against each criterion:

| Criterion | Value | Result |
|---|---|---|
| Decode, % of the 40.6 tok/s byte floor (128 / 1k / 4k) | 90.3 / 90.2 / 87.7 | ≥ 85% PASS |
| Decode vs mlx-lm | 106–109% | ≥ 90% PASS |
| Decode vs llama.cpp | +3.6 to +4.9% | not below it: PASS |
| Prefill vs mlx-lm (1k / 4k) | 94% / 96% | committed ≥ 50%, target ≥ 65%, stretch ≥ 85%: all PASS |

**Caveats:**
- The stretch verdict holds on the mean of two runs, not on every run. At 4k, SAFE run 1 came in at 85.1% of mlx-lm and CAND run 1 at 75%. The serve-path 4k prefill drifted +26% (SAFE) and +60% (CAND) between run 1 and run 2, while decode drifted less than 1%.
- mlx-lm decode in this slot (34.5 / 34.0 / 32.7) was lower than in the P3 slot earlier (36.0 / 36.0 / 34.7). Part of the decode margin over mlx-lm is therefore an mlx-lm dip. Against the higher figures, intarsia decode is about 102% of mlx-lm.
- mlx-lm and llama.cpp each ran once.

**Plain prefill** (`generate`, ABBA, 6 runs each). Each build ran at its own default chunk.

| Prompt | SAFE ms (median) | CAND ms (median) | CAND speed vs SAFE |
|---|---:|---:|---:|
| 1k | 444 | 378 | 1.17x |
| 4k | 1490 | 1506 | 0.99x (ranges overlap) |
| 16k | 8600 | 7794 | 1.10x |

**Accuracy** (SAFE and CAND are identical):
- Long golden prefill: top-1 1430/1434, KL 3.699e-4.
- Short golden prefill: top-1 434/434, KL 2.832e-4.
- Decode: top-1 100%, KL 3.496e-4 / 3.517e-4.
- Gate: KL ≤ 1e-3, top-1 ≥ 99.5%.
- Greedy ids are identical between SAFE and CAND. SAFE `qualify`: 10 PASS, 2 SKIP (`ab`, `tp`).

## Change since yesterday

Earlier rows are same-slot localeval runs. This morning's mlx-lm and llama.cpp runs came from a separate slot.

| | decode 128 / 1k / 4k | prefill 1k / 4k |
|---|---|---|
| Same-slot baseline 2026-10-14 17:05 (`a296a78`) | 35.6 / 35.4 / 34.5 | 350 / 205 |
| After P1 + P2 (`2e1fdb1`, 22:10) | 36.2 / 36.2 / 35.1 | ~3300–3550 / ~1950–2200 |
| **Now** (`a2102bf`) | **36.7 / 36.6 / 35.6** | **4054 / 3126** |

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
   - **What it buys:** plain prefill 1.17x at 1k and 1.10x at 16k, and noise at 4k. Serve-path prefill is about equal to SAFE. Accuracy and ids are identical.
   - **Why it's held:** fresh `qualify` FAILs `acc-long` on the prefix-reuse gate. The longest long-golden prompt is 700 tokens, and any chunk above about 350 skips every prompt, which counts as a fail. It is a coverage gap, not a regression.
   - **Options:**
     - (a) Build a Qwen3-8B fp32 golden with 4k prompts (`tools/prompts/golden_4k.jsonl` exists). This needs spark1 with vLLM taken down (`--takes-down flight,guidance,capcom`, see the incident below), or an fp32 run on endor's CPU.
     - (b) Accept 2048 with the gap.
     - (c) Keep 256. With F1b and P4/P5 merged, P3's gain is now small. **I'd do (c) now and revisit P3 with (a).**
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
