# M35P: Metal 35B prefill numbers, KL against the CUDA gate, prefill profile (2026-10-27)

Lane M35P, run on the evening of 2026-10-26 (from 23:20 EDT) on endor (Apple M5 Max, 128 GB). Model Qwen3.6-35B-A3B NVFP4
(`~/models/Qwen3.6-35B-A3B-NVFP4`) and Qwen3-8B BF16 (`~/models/Qwen3-8B`). Binary: `origin/main` e0a04220
(`cargo build --release -p intarsia-cli --features metal`, unstamped). **Measurement only: no engine code changed.**
Raw data and scripts: [2026-10-27-m35p-metal-35b-prefill-kl/raw](2026-10-27-m35p-metal-35b-prefill-kl/raw/).

## Result

| question | answer |
|---|---|
| Metal 35B prefill, 1k / 4k / 16k | 1649* / **1785** / **1507** tok/s (`serve --max-batch 8`); 1631* / 1788 / 1507 (default `--max-batch 1`) |
| Metal 35B decode, 1k / 4k / 16k | 123.0 / 120.3 / 108.4 tok/s (max-batch 8); 122.4 / 120.6 / 106.3 (max-batch 1) |
| against llama.cpp Q4_K_M (endor record, 10-22) | prefill **0.75x at 4k, 0.67x at 16k** (2392 / 2261); decode 1.25x at 4k, 1.18x at 16k (96.7 / 89.9) |
| Metal 35B KL against the vLLM goldens, prefill | **PASS**: 1.04x to 1.07x CUDA on 35b-short, 0.99x on 35b-long scored rows; every difference is under 0.12 of vLLM's own run-to-run KL |
| Metal 35B KL, decode window (35b-short, 160 positions) | **FAIL the strict "no worse than CUDA by more than the noise floor"**: 1.6x to 2.0x CUDA, a gap of 1.4x to 1.9x the reference noise. Same level as HIP (0.96x to 0.99x of it). Passes the absolute floor (1.2e-2 against 3.0e-2) and Metal's own recorded baseline |
| prefill profile | top sinks at 4k: MoE 38.5 %, fp8 GEMM 34.1 %, GDN delta rule 16.0 %. The GPU is 95 % busy, so the gap is kernel throughput (10.7 TFLOPS overall, the 8B reaches ~60 on NAX) |
| code change | none. Every fix that closes the gap moves fp8 / MoE prefill to the NAX path, which changes accumulation order, so the 35B ids stop being identical. That is a user decision |

\* 1k prompts are under 1024 uncached tokens: TTFT is request overhead, not prefill throughput.

## 1. Speed (localeval, fleetq `--quiet gpu,cpu`)

`localeval speed --sizes 1024,4096,16384 --gen 256 --reps 3 --thinking off`, server
`intarsia serve --backend metal --max-seq 20000 --profile-every 0 --no-supervisor --no-ui` (scripts: `raw/scripts/speed.sh`).
Median of 3 reps after one warm-up. **Clean pass: fleetq #5854**, 23:25:58 to 23:30:30, `pgrep -x rustc` 0 before and after each arm,
a CPU/process sample every 4 s per arm in `raw/speed2/*-load.log`. Run ids `20261026-232637-adhoc-m35p-8b`, `...-232817-adhoc-m35p-35b-mb8`,
`...-232929-adhoc-m35p-35b-b1` (summaries in `raw/localeval/`).

| model, arm | prompt | TTFT s | prefill tok/s (range) | marginal prefill | decode tok/s |
|---|---:|---:|---|---:|---|
| 35B, max-batch 1 | 991 | 0.61 | 1631 * | - | 122.4 |
| | 4090 | 2.29 | 1788 (1785-1789) | 1844 | 120.6 |
| | 16656 | 11.05 | 1507 (1500-1509) | 1434 | 106.3 |
| 35B, `--max-batch 8` | 991 | 0.60 | 1649 * | - | 123.0 |
| | 4089 | 2.29 | 1785 (1600-1789) | 1835 | 120.3 |
| | 16655 | 11.05 | 1507 (1505-1516) | 1434 | 108.4 |
| 8B BF16 | 991 | 0.25 | 3997 * | - | 35.9 |
| | 4087 | 1.04 | **3914** (3893-3927) | 3889 | 35.1 |
| | 16653 | 6.33 | **2630** (2269-2740) | 2376 | 31.6 |

Prefill at max-batch 1 and 8 is the same within noise (the batching path only changes decode).

**Discarded first pass (fleetq #5849, 23:20 to 23:25).** Its 8B arm gave 4k 2204 / 16k 2032 tok/s and its max-batch-8 arm 1586 at 4k, while
another session was working on the Mac. My load check then was `pgrep -fl 'rustc|cargo'`, which also matches other sessions' command lines
(counts of 1 and 673), so it proved nothing. The 8B arm is 44 % under the clean rerun, so I treat the whole pass as contended and reran
the 8B and max-batch-8 arms and the 35B default arm (`raw/speed/`, run ids `...-232029-adhoc-m35p-35b-b1` 1769 / 1351 at 4k / 16k,
`...-232225-adhoc-m35p-8b`, `...-232415-adhoc-m35p-35b-mb8`; the 35B 16k value is 10 % under the clean rerun). The discarded runs were
logged to unanimis by localeval before I knew; they are not valid.

### Against the engines already on endor

| engine, model | prefill 1k | 4k | 16k | decode 1k | 4k | 16k |
|---|---:|---:|---:|---:|---:|---:|
| **intarsia Metal, 35B NVFP4** (this run) | 1631 * | **1788** | **1507** | 122.4 | 120.6 | 106.3 |
| llama.cpp 0.60, 35B-A3B **UD Q4_K_M** (10-22, `qwen36-35b-udq4km-lcpp060`) | - | 2392 | 2261 | 91.8 | 96.7 | 89.9 |
| intarsia / llama.cpp | | 0.75x | 0.67x | 1.33x | 1.25x | 1.18x |
| mlx-lm, 35B | no record on endor | | | | | |
| **intarsia Metal, 8B BF16** (this run) | 3997 * | **3914** | **2630** | 35.9 | 35.1 | 31.6 |
| intarsia Metal 8B, 10-14/15 figures / 10-21 gate (`sameslot`) | | 3923 / 3753-3851 | - / 2681-2717 | 35.7 / 36.0 | - / 34.9 | - / 31.2 |
| mlx-lm 8B BF16 (10-21 gate) | | 3041 | 2396 | 32.7 | 32.1 | 27.9 |
| llama.cpp 8B BF16 GGUF (10-21 gate) | | 2400-2434 | 1443-1603 | 33.8 | 33.1 | 29.6 |

The 35B comparison is **not same-weights**: llama.cpp ran a Q4_K_M GGUF (about 4.8 bits per weight, different quantisation), ours runs the NVFP4
checkpoint with fp8 attention and projections. There is no mlx-lm 35B record on endor (the mq lane's MLX 4-bit checkpoint exists at
`~/models/ref-qwen36-mlx4` but nothing was run on it: the brief said not to start other engines' servers). The 8B numbers are unchanged by the
last two weeks of Metal work: 4k prefill 3914 against 3923 (10-14/15) and 3753-3851 (10-21), decode 35.1 against 35.7 and 34.9. The 8B
BF16 stays ahead of mlx-lm (1.29x at 4k, 1.10x at 16k) and llama.cpp (1.6x, 1.6x to 1.8x) at prefill.

The marginal-prefill column shows the 35B has no big fixed cost at 1k: 1631 tok/s at 991 tokens is within 10 % of the 4k figure.

## 2. Accuracy: KL against the vLLM goldens (`intarsia accuracy`, Metal)

Goldens: the same vLLM goldens the CUDA and HIP gates use (`golden/qwen36-35b-a3b-nvfp4-vllm-{top20,long-top20,4k-top20}.safetensors`,
spark1 vLLM, fp8 KV, `guidance`). Command in `raw/scripts/accuracy.sh` (fleetq #5858, `--mode both`, chunk 2048, bf16 KV, 4-bit gate class)
and `raw/scripts/baseline-gate.sh` (fleetq #5868). Logs, verdict JSON and baseline JSON in `raw/accuracy/` and `raw/baseline-gate/`.
"KL" is the gated 21-way KL (top-20 plus the tail bucket), mean over positions. "Scored" excludes system and user-turn rows, as the gate does.
Reference noise (vLLM re-scoring its own golden): 35b-short 3.161e-3 all rows / 3.010e-3 scored, 35b-long 2.218e-2 / 8.417e-3 scored
(`golden/ref_noise/` sidecars, docs/results/2026-10-21-cuda-round7.md 3.1).

The Metal numbers are **bit-identical to the Metal baseline recorded on 2026-10-23** (`golden/baselines/*.metal.baseline.json`): the
single-stream accuracy path did not move through the later batching and fp8 / MoE row-kernel merges, and the formal "no worse than baseline
+ per-prompt noise envelope" gate passes (`raw/baseline-gate/short-vs-metal-baseline.log`, `4k-vs-metal-baseline.log`).

### 35b-short (10 prompts, 1441 prefill positions, 160 decode positions): Metal against the CUDA baseline (`...vllm-top20.cuda.baseline.json`, GB10, build c37a8d3)

| mode, rows | Metal | CUDA | ratio | gap | gap / ref noise | verdict |
|---|---:|---:|---:|---:|---:|---|
| prefill, all rows | 6.914e-3 | 6.648e-3 | 1.04 | +2.7e-4 | 0.08 | PASS |
| prefill, scored rows | 5.781e-3 | 5.408e-3 | 1.07 | +3.7e-4 | 0.12 | PASS |
| prefill top-1 | 98.33 % | 98.27 % | | | | PASS |
| decode, all rows | 1.164e-2 | 7.344e-3 | 1.58 | +4.3e-3 | 1.4 | **FAIL** |
| decode, scored rows | 1.169e-2 | 5.833e-3 | 2.00 | +5.9e-3 | 1.9 | **FAIL** |
| decode top-1 | 98.75 % | 99.38 % | | | | at the limit |
| (HIP, same golden) decode all / scored | | 1.172e-2 / 1.215e-2 | Metal 0.99 / 0.96 of HIP | | | |

The formal CUDA-baseline gate (Metal run against the CUDA baseline and CUDA envelope, `--allow-config-mismatch`,
`raw/baseline-gate/short-vs-cuda-baseline.log`): **prefill passes every aggregate and all 30 per-prompt checks; decode fails 5 aggregate
metrics (mean KL, tail-bucket KL all and scored, top-5, max KL) and 3 prompts** (`chat_system` 2.37e-2 against limit 1.89e-2, `chat_multiturn` 3.09e-2 against 1.24e-2, `long_prose_700` 1.90e-2
against 1.27e-2). The three prompts carry the gap: the other seven decode prompts are under CUDA's or within 7 % of it. Notes:

- The decode window is 16 positions per prompt. The reference noise above was measured over prefill positions, so "1.9x noise" is only a yardstick, not a measured decode noise.
- Metal's decode is at HIP's level on this golden (HIP is worse still on `chat_multiturn`: 1.49e-1 scored). 35b-short decode is the advisory gate of ADR-0008.
- Both Metal's floor gate (1.17e-2 against 3.0e-2) and its own baseline gate pass. The difference to CUDA looks like a decode-path property (Metal's single-token GEMV kernels against CUDA's), not noise from the run: it is repeatable bit for bit.
- I did not run CUDA on spark1, so there is no CUDA decode reference for the long and 4k goldens, and I did not find the cause.

### 35b-long and 35b-4k prefill, per prompt: Metal against CUDA main 6189466f, chunk 2048 (docs/results/2026-10-24-c8k-prefill-chunk-8192.md)

No CUDA baseline JSON exists for these goldens; the c8k lane's per-prompt tables are the CUDA reference. Both use KL21 scored rows.

| golden, prompt | Metal | CUDA | ratio |
|---|---:|---:|---:|
| long, long_docs_4k | 1.787e-2 | 1.754e-2 | 1.02 |
| long, long_code_4k | 7.46e-3 | 7.306e-3 | 1.02 |
| long, chat_reuse_4k | 1.953e-2 | 8.541e-3 | 2.29 |
| long, long_docs_12k | 2.140e-2 | 2.099e-2 | 1.02 |
| long, long_code_12k | 8.25e-3 | 8.577e-3 | 0.96 |
| 4k, long_docs_4k | 1.767e-2 | 1.659e-2 | 1.07 |
| 4k, long_code_4k | 7.19e-3 | 7.168e-3 | 1.00 |
| 4k, chat_reuse_4k | 1.265e-2 | 3.535e-2 | 0.36 |

`chat_reuse_4k` is the chunk-sensitive prompt (c8k: CUDA moves 8.5e-3 to 1.2e-2 on the long golden and 3.5e-2 to 2.7e-2 on the 4k one with the
chunk), so its ratio swings both ways; the other six are within 7 %.

Aggregates (35b-long, 5 prompts, 36887 prefill positions): scored rows Metal **1.430e-2** against CUDA default 1.420e-2 (cuda-round7, setkl
method, ratio 1.01, gap 1e-4 = 0.01 of the 8.417e-3 noise); all rows (tool-exact) Metal 4.259e-2 against CUDA default 4.576e-2 and `lt`
4.725e-2 (ratio 0.93 / 0.90). 35b-4k (3 prompts, 12759 positions) Metal scored 1.244e-2, all rows 9.089e-2 (the user turn of `chat_reuse_4k`). Floor
verdicts: all PASS (3.0e-2). Decode on the long and 4k goldens (teacher-forced, 80 and 192 positions): Metal 1.229e-2 (0.55 of the long
golden's 2.218e-2 noise) and 6.904e-3; no CUDA reference.

Prefix reuse on the 4k golden (`accuracy --mode reuse`) is still recorded as failed in the 10-23 Metal baseline
(`reuse.passed = false`); this lane did not rerun reuse.

MMLU 0.840 [0.820, 0.858] (n = 1425, mq lane, 10-22, `docs/results/2026-10-22-mq-metal-quality.md`) was not rerun.

## 3. Prefill profile (35B, Metal)

`intarsia generate --backend metal --max-seq 20000 --max-tokens 2 --ignore-eos --profile-every 0` on the 4096- and 16384-token prompts of
`docs/results/raw/2026-10-15-metal-longctx-profile/prompts/`, with `INTARSIA_METAL_PROFILE=kernel:<file>` (GPU timestamps per launch) and `=cb:<file>`
(per command buffer), fleetq #5859, `raw/scripts/prof.sh`, tables from `raw/scripts/perop.py` (`raw/prof/perop-{4k,16k}.txt`, traces gzipped).
Chunk 2048, so 2 chunks at 4k and 8 at 16k. **This slot overlapped other sessions' builds** (rustc 4, 1, 16, 8 seen after four of the nine runs),
so the wall times of the `generate` runs are not quoted; shares and GPU time per kernel are not affected by an idle-CPU/GPU contention of this size
(kernel-mode GPU sum 2286 ms against command-buffer GPU busy 2290 ms at 4k).

GPU time by op, summed over the whole prefill:

| op | 4k ms | 4k share | 16k ms | 16k share | us per call at 2048 rows |
|---|---:|---:|---:|---:|---:|
| `moe_experts_bf16` (gate+up, down, other, all dispatches) | 881 | **38.5 %** | 3575 | **33.5 %** | 7208 (gate+up), 3629 (down) |
| `matmul_fp8_mma` (attention q/k/v/o, GDN in/out, shared expert) | 780 | **34.1 %** | 3064 | **28.7 %** | 3002 mean |
| `gdn_delta_rule_bf16` (30 layers) | 366 | **16.0 %** | 1453 | 13.6 % | 6105 |
| `attn_prefill_bf16_flash` (10 layers) | 125 | 5.5 % | 2041 | **19.1 %** | 6253 at 4k, 25506 at 16k |
| `matmul_nvfp4_mma` | 83 | 3.6 % | 334 | 3.1 % | 347 |
| everything else (rmsnorm, rope, silu, topk, combine, conv) | 51 | 2.2 % | 195 | 1.8 % | |
| total GPU | 2286 | | 10671 | | |

Command buffers: GPU busy 94.8 % of the span at 4k, 97.1 % at 16k. The prefill is GPU-bound; scheduling gaps cost 3 to 5 %.

**Top 3 time sinks.**

1. **MoE experts (38.5 % at 4k, 33.5 % at 16k).** Gate+up 7.2 ms and down 3.6 ms per 2048-row chunk and layer. With top-8 of 256 experts, hidden
   2048, expert width 512, that is 68.7 and 34.4 GFLOP per call, so **9.5 TFLOPS** on both. Prefill uses 32x32 `simdgroup_matrix` tiles per expert
   (docs/results/2026-10-22-m35-metal-moe.md), not NAX.
2. **fp8 projections `matmul_fp8_mma` (34.1 % / 28.7 %).** 64x64x32 tile, bf16 operands staged through threadgroup memory, 8x8 `simdgroup_matrix`
   (kernels/metal/fp8.metal). It runs at **13 to 14 TFLOPS** on all four shapes (8192x2048: 5.0 ms per call, 2048x4096: 2.6 ms, 4096x2048: 2.5 ms),
   about a fifth of what the NAX bf16 GEMM reaches on the 8B at M = 2048 (44 to 65 TFLOPS, docs/results/2026-10-15-metal-gemm-midm.md).
3. **GDN delta rule (16.0 % at 4k) and, at long context, flash attention (19.1 % at 16k, GDN drops to fourth).** `gdn_delta_rule_bf16` is 6.1 ms per chunk and
   layer, a sequential recurrence over the chunk; it grows linearly with tokens (3.0 ms per 1k tokens per layer). Attention grows with the
   square of the context, 125 ms at 4k to 2041 ms at 16k, which is why the 35B loses more against llama.cpp at 16k (0.67x) than at 4k (0.75x).

Overall the prefill runs at 10.7 TFLOPS (6 GFLOP per token times 4096 tokens, over 2.29 s). The 4k gap to llama.cpp is 0.58 s (1.71 s at 2392 tok/s).

**Estimates for the next lane (not measured here).** (a) Run `matmul_fp8_mma` on the NAX path at 45 to 55 TFLOPS: 780 ms to about 200 ms at 4k, which
alone gives about 1.72 s, 2380 tok/s, parity with llama.cpp's 2392. (b) Doubling the MoE rate (an expert-grouped NAX GEMM with the NVFP4 dequant in
the operand load): another 440 ms at 4k, 1790 ms at 16k. (c) A chunked/parallel delta-rule form for GDN. Prefix: the fp8 weights are exact in bf16 and the
scale is applied at the output, so an fp8-to-bf16 operand load keeps the products exact; what changes is the accumulation order of the NAX
primitive against the present `simdgroup_multiply_accumulate` chain.

## What I did not do, and why

- **No fix.** A tile tweak of `matmul_fp8_mma` (BK 64, register prefetch) keeps the accumulation order and so the ids, but the kernel is already
  at 13 to 14 TFLOPS against a non-NAX simdgroup ceiling that the repo measured at 5.5 TFLOPS for the bf16 tile of the 8B before NAX and that I
  estimate at 16 to 20 here; the best case is about 15 % of 34 %, 5 % end to end, nowhere near the +34 % needed. The levers that matter move fp8 and
  MoE prefill to NAX, which changes accumulation order: the 35B ids would no longer be identical to today's (as when the 8B went to NAX on 10-15),
  which fails this brief's gate and wants a decision.
- No CUDA decode reference for the long and 4k goldens, and no root cause for the decode-window gap (needs CUDA on spark1 and a per-position diff).
- The 35B was not run on mlx-lm or llama.cpp again; the llama.cpp numbers are the 10-22 record.

## Reproduce

Note: `raw/accuracy/meta.txt` and `raw/baseline-gate/meta.txt` carry a process count from the false-positive `pgrep -fl` check (673); it is not contention, and `accuracy.sh` now uses `pgrep -x`. The accuracy runs are not timings.

`cargo build --release -p intarsia-cli --features metal`, then inside one fleetq slot `fleetq run --host endor --quiet gpu,cpu --mem 45 --expect 15 -- raw/scripts/speed.sh BIN OUT`
(`ARMS="8b 35b-mb8 35b-b1"` selects arms), `raw/scripts/accuracy.sh`, `raw/scripts/baseline-gate.sh`, `raw/scripts/prof.sh BIN OUT PROMPT_DIR`. fleetq jobs: #5849 (first speed pass,
discarded), #5854 (clean speed), #5858 (accuracy), #5859 (profile), #5868 (baseline gates).
