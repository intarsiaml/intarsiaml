# srv16: Spark 35B 16-stream served throughput: where the wall time goes, and packed prefill by default on CUDA

Lane srv16, 2026-10-28/29, branch `lane-srv16/serve-gap` from `main` b7c624423 (#230). spark1 (spark-4e77, GB10 sm_121),
Qwen3.6-35B-A3B NVFP4, tp1, `serve --backend cuda --max-batch 16 --max-seq 8192 --mem-budget 40`, no MTP.

**Status: PR (ready).** 16 streams +6.7 % (277.33 -> 295.80 tok/s, gate +5 %), 8 streams +5.2 %, 4 streams +1.4 %, 1 stream
+0.3 % (gate: none worse than -1 %; worst single round -0.9 % at 1 stream). Bitwise: served tokens identical to main (equal.py
21/21 cells plus 4/4 long-prompt cells), `real_35b_packed_prefill_equals_the_local_engine` passes (it failed on main),
`real_35b_batched*` pass, `rows_exact` 9 pass, `plan show` byte-identical for 8B / 35B / Gemma-4, Mac `cargo test --workspace
--release` exit 0. 16 streams is now 0.87x vLLM `--max-num-seqs 16` (340.1, vr16 window, not the same window), was 0.80x.

## 1. Attribution (W1, `raw/base-b16/`, main)

One 16-stream burst (`scripts/srv16/burst.py`: 16 x ~1150-token prompt, 256 tokens each, greedy, `ignore_eos`) under nsys with
`INTARSIA_HOST_PROFILE=1`. The host profile (`serve.err`) and the nsys pass timeline (`timeline.txt`, `scripts/srv16/timeline.py`)
agree on the wall: 15.05 s. Iteration classes from the host profile:

| part of the burst | iterations | ms each | total s | share |
|---|---:|---:|---:|---:|
| decode-only step: GPU (graph replay + sync) | 255 | 43.94 | 11.20 | 74.4 % |
| decode-only step: host (setup, launch, sampling, detok, SSE send) | 255 | 0.55 | 0.14 | 0.9 % |
| prefill chunk of one ~1.1k prompt (15 beside a decode step, 1 alone) | 16 | 199.4 | 3.19 | 21.2 % |
| decode step inside those 15 iterations (avg 8 rows) | 15 | 34.4 | 0.52 | 3.4 % |
| reap / admit / metrics / idle | | | < 0.02 | 0.1 % |
| **burst wall** | 271 | | **15.05** | 100 % |

- GPU idle between decode graph launches: 38 us per step (253 gaps); nsys: idle gaps >= 300 us total 204 ms (1.4 %, mostly around
  the prefill chunks). There is no host-gap problem: the host is 1 % of the wall.
- Decode stalls during prefill: yes. A decoding stream waits for every later prompt's 0.2 s chunk (one per iteration, decode-first).
  15 chunks x 0.2 s = 3.0 s of stall for the first stream.
- Decode step at ~1.1-1.4k context: 44.5 ms per iteration (43.9 GPU), the MOE16b micro-test's 45.7 ms at 64-264 tokens; context
  length does not make the step slower here (attention is ~2 ms of the step).
- Sampling, detokenise and streaming: 0.03 ms per step (`accept`, `on_token`, `detok`, `send`).

What vLLM does instead (docs.vllm.ai, "Optimization and Tuning", chunked prefill): decode requests are scheduled first, the
remaining `max_num_batched_tokens` budget (8192 in the vr16 run) is filled with pending prefills, which are chunked when they do not
fit, and decode and prefill tokens run in the same batch. A 16 x 1k burst is two mixed steps, so every stream gets its first token
at about the same time (vr16: TTFT max = p50). From vr16's own numbers: vLLM per-stream decode 24.6 tok/s at 16 streams (ours 19.2-19.9),
i.e. about 40.6 ms per step, against our 44.5 ms. The 3.1 s gap to vLLM at 16 streams (12.0 s vs 15.1 s wall) is therefore about
half prefill serialization (16 one-prompt passes, each reading the routed experts' weights again) and half step speed (~4 ms per step
x 256 steps). The brief's "the step is ~340 tok/s, the gap is serve level" is half right: 44.5 ms is 360 tok/s of pure decode, but
vLLM's pure decode is faster still.

## 2. Change

1. **Packed prefill is exact on CUDA.** `Model::forward_packed` (several prompts' next chunks in one layer-by-layer pass,
   `INTARSIA_PACK_PREFILL`, opt-in since the HIP lane) failed `real_35b_packed_prefill_equals_the_local_engine` on CUDA: cuBLASLt /
   cuBLAS pick tiling and split-K by row count. The macro pass (lane sk) already runs those ops (`INTARSIA_PREFILL_MACRO_SPLIT`'s
   list) once per chunk; the packed pass now does the same once per prompt (`Model::macro_ctx` set to the prompts' row ranges), where
   `Backend::pack_split_gemms` is true (CUDA only; HIP's packed pass is unchanged).
2. **A packed pass may fill the macro-pass workspace.** `Model::pack_rows_max` was the chunk (2048), so a 16 x 1.1k burst never
   packed (two bodies are 2302 rows; `raw/diag1/chunks.txt`). The workspace already holds `max(chunk, macro_rows) + tail_merge` rows
   (`[tuning.cuda] prefill_macro = 8192`), and in a packed pass only the row-wise kernels (experts, shared expert, norms, quantisers,
   top-k) see all rows, as in a macro pass; attention, the GatedDeltaNet recurrence and the split GEMMs stay per prompt. Where
   `pack_split_gemms` holds, the cap is now `max(chunk, macro_rows)`: the burst packs as 1, 7, 7, 1 prompts (`raw/diag2/chunks.txt`).
3. **On by default on CUDA.** `BatchedOptions::pack_prefill` is now `Option<bool>`: `INTARSIA_PACK_PREFILL=1` / `=0` force it,
   unset takes `Backend::pack_prefill_default()` (CUDA true, everything else false). `docs/tuning.md` row updated.

No kernel changed; `plan show` (the memory plan) is unchanged because the workspace was already that size.

Burst after the change (`raw/dev3-b16/`): wall 15.05 -> 13.52 s. Prefill 3.19 s -> 2.68 s (one solo 0.20 s chunk, packs of 7 and 7
prompts at 1.06 / 1.25 s, a last 0.24 s chunk; each prompt's last token still runs alone, 13 x ~13 ms). `moe_w4a16_kernel` calls
1280 -> 320. Decode-only steps 44.5 -> 42.1 ms (same 255 count, 15.9 vs 15.5 rows on average; not explained, the streams now start
decoding together). TTFT: median at 16 streams 1.67 -> 1.31 s, max 4.0 -> 3.1 s; at 4 and 8 streams the median goes up (0.47 -> 0.64 s,
0.86 -> 1.18 s) because the first prompts of a burst wait for the pass that carries the later ones. A decoding stream now stalls once
for a whole pack (~1.1-1.3 s) instead of once per prompt (0.2 s each).

## 3. Serve A/B (W2, `scripts/srv16/le-window.sh w2 3`, `raw/w2/`)

localeval `--sizes 1024 --gen 256 --reps 3 --concurrency 1,4,8,16 --thinking off`, 3 interleaved rounds (round 2 reversed), one
fresh server per arm. base = main b7c624423, dev3 = this branch (no env).

| streams | base r1, r2, r3 (mean) | dev3 r1, r2, r3 (mean) | dev3 vs base (per round) | median TTFT base -> dev3 |
|---|---|---|---|---|
| 1 | 79.9, 80.5, 78.3 (79.57) | 79.6, 79.8, 80.1 (79.83) | **+0.3 %** (-0.4 / -0.9 / +2.3) | 0.19-0.25 -> 0.18 s |
| 4 | 169.5, 177.8, 169.5 (172.27) | 173.1, 174.4, 176.4 (174.63) | **+1.4 %** (+2.1 / -1.9 / +4.1) | 0.47 -> 0.64 s |
| 8 | 223.7, 224.0, 221.7 (223.13) | 235.0, 233.1, 236.0 (234.70) | **+5.2 %** (+5.1 / +4.1 / +6.5) | 0.86 -> 1.18 s |
| 16 | 278.7, 277.5, 275.8 (277.33) | 295.9, 296.7, 294.8 (295.80) | **+6.7 %** (+6.2 / +6.9 / +6.9) | 1.67 -> 1.31 s |

W1's single exploratory round (`raw/w1/`): base 277.6, dev + `INTARSIA_PACK_PREFILL=1` with the 2048-row cap 285.3 (+2.8 %, 991-token
prompts pack in pairs), main + `INTARSIA_PACK_PREFILL=1` (not bitwise) 289.9.

## 4. Gates

| gate | result |
|---|---|
| +5 % at 16 streams, no stream count worse than -1 % (3 rounds) | **pass**: +6.7 % at 16; +0.3 / +1.4 / +5.2 % at 1 / 4 / 8 |
| served greedy + sampled tokens identical to main (equal.py, 21 cells) | **pass**: 21/21, 9408 ids (`raw/ident-w2/compare.log`; W1 with the env knob: `raw/ident/`) |
| long prompts that pack (300..1300 tokens, 4 and 16 at once, greedy + seeded sampled; `scripts/srv16/ident_long.py`) | **pass**: 4/4 cells, 2560 ids (`raw/ident-w2/compare-long.log`) |
| `real_35b_packed_prefill_equals_the_local_engine` (24 requests, 8 and 16 slots; `--max-seq 2048`, so 2048-row packs) | **pass** (`raw/w2-batch-real-packed.txt`; W1: `raw/batch-real-packed.txt`); failed on main per lane sk |
| `batch_cuda_real` `real_35b_batched*` (now with packing on by default) | **pass**: 2 passed (`raw/w2-batch-real.txt`) |
| `rows_exact` | **pass**: 9 tests (`raw/w2-rows-exact.txt`) |
| `plan show` byte-identical, 8B / 35B / Gemma-4, json + txt | **pass** (`raw/plan-show.txt`) |
| Mac `cargo test --workspace --release` | **pass**: exit 0 (`raw/mac-cargo-test.log`) |

Not covered: the engine test packs at most 2048 rows (its `max_seq` is 2048); the 8192-row packs are covered by the served
long-prompt cells (16 prompts of 300..1300 tokens at `--max-seq 8192`) and the burst profile, not by an engine-level test. HIP is
untouched by default (`pack_split_gemms` / `pack_prefill_default` false) and was not re-measured.

## 5. Process

GPU windows: **2 of 3**. #6709 (W1, `--quiet gpu,cpu --mem 50`, 12 min, 23:49-00:01): base nsys + host profile, packed gate, identity, one
exploratory localeval round, dev + pack nsys. #6736 (W2, 21 min, 00:20-00:41): packed and batched gates, `rows_exact`, identity base vs dev3,
3-round A/B, dev3 nsys. Diagnostics outside windows (not timing, `--mem 45`, ~2 min each): #6718 (`raw/diag1`), #6721
(`raw/diag2`), scheduler trace of which prompts share a pass. Builds / plan show `--load cpu`: #6700, #6704, #6719, #6724, #6730.
Mac tests: #6744 on endor. spark1-vllm stayed up, no `--takes-down`.

## 6. Next

1. **The prompt's last token as a decode row.** Each prompt's final token still runs as its own 1-row forward (13 x ~13 ms in the
   burst, 0.17 s, ~1.3 %); feeding it into the next batched decode step as a row would remove that, if the batched row is bitwise
   the solo last-token forward (to be checked with `batch_cuda_real`).
2. **Step speed is the other half of the vLLM gap** (~4 ms per step): routed MoE is at the bandwidth floor (MOE16), so it is in the
   GDN delta rule (8.4 ms per step at B=16) and the fp8 projections (8.2 ms vs a 5.6 ms floor, MOE16b section 4).
3. **TTFT at 4 / 8 streams** rose with packing (the first prompts wait for the pack). A pack cap below the workspace (e.g. 4096 rows)
   would trade some of the 8-/16-stream gain for it; not measured.
