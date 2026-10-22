# Q1: intarsia vs vLLM quality on spark1 (Qwen3.6-35B-A3B-NVFP4)

**Verdict: FAIL on GSM8K (intarsia 4.1 points lower, p = 0.017); MMLU equal.** Needs an accuracy lane. This doc does not diagnose the cause.

## Setup
- Box: spark1 (GB10), fleetq #3192 (non-exclusive, 40 GB), vLLM stayed up (`guidance`, port 8000, `max_model_len` 32768, fp8 KV). No role taken down; downtime 0.
- intarsia: `g1c-base-intarsia` (built 17:17 for G1c, base = main 0294b778 as labelled by #3182; not re-verified), tp=1, `serve --backend cuda --max-seq 20480 --mem-budget 38 --port 8091`, bf16 KV. Log: `raw/intarsia-serve.log`.
- Harness: localeval 0.2.1 (deepeval 4.2.6), `--thinking off --structured none`, temperature 0, same prompts for both. Sequential runs, never concurrent. Both servers share one GPU, and other lanes' jobs ran on spark1 only after the slot was released.
- GSM8K: full test set, n = 1319. MMLU: `--limit 30` per task over 57 tasks, n = 1710 (same questions for both).

## Quality (95% CI, normal approximation, unpaired)

| suite | intarsia | vLLM | diff (intarsia - vLLM) | z | p |
|---|---|---|---|---|---|
| GSM8K n=1319 | 0.2403 [0.217, 0.263] | 0.2813 [0.257, 0.306] | **-0.0409 [-0.074, -0.008]** | -2.40 | **0.017** |
| MMLU n=1710 | 0.8386 [0.821, 0.856] | 0.8351 [0.818, 0.853] | +0.0035 [-0.021, +0.028] | 0.28 | 0.78 |

- GSM8K is significant at 5%, MMLU is not. The GSM8K gap is small next to the CIs, so it is not a large effect.
- Both GSM8K scores are low: with thinking off the model answers in a median 4-5 tokens (no chain of thought), and the 8-shot prompt scoring is strict. The absolute number is a property of the harness setting; only the comparison is meaningful.
- Output length differs: 603 of 1319 intarsia calls had a measurable decode window against 390 for vLLM (calls with enough tokens to time). Intarsia produced more long answers on the same prompts. This is a lead for the follow-up lane, not a finding.
- Limits: localeval stores no per-question results, so the test is unpaired (paired McNemar would be tighter). One run per server; greedy decoding is deterministic but NVFP4 kernels differ in rounding, so some flips either way are expected. A per-question replay of the 1319 prompts through both servers is the first thing the accuracy lane should do.

## Speed (localeval sweep, 3 reps, gen 128, both servers on one GPU, one at a time)

| prompt | intarsia prefill tok/s | vLLM prefill tok/s | intarsia decode | vLLM decode |
|---|---|---|---|---|
| ~1k | 3928 (2836-4158)* | 3881 (3770-4689)* | 75.7 | 76.4 |
| ~4k | 4870 (4455-5477) | 5278 (5235-5793) | 81.9 | 76.7 |

\* under 1024 uncached tokens: request overhead, not prefill throughput. Marginal prefill 1k to 4k: intarsia 5267, vLLM 5967 tok/s (-12%). Decode within noise at 1k, +7% at 4k (ranges overlap partly; peak-decode of intarsia shows burst values, ignore). The GPU is shared with whatever vLLM or other jobs do, so these are not clean standalone numbers; the 35B tp=2 doc has the clean ones.

## Runs (result files under /opt/code/deepeval/results/, copies in `raw/`)
- intarsia GSM8K `20261021-173333-adhoc-q1-intarsia-gsm8k`, MMLU `20261021-175808-adhoc-q1-intarsia-mmlu`, speed `20261021-183225-adhoc-q1-intarsia-speed`
- vLLM GSM8K `20261021-180552-adhoc-q1-vllm-gsm8k`, MMLU `20261021-182333-adhoc-q1-vllm-mmlu`, speed `20261021-183241-adhoc-q1-vllm-speed`
- unanimis capture failed for all runs (ssh session refused); the result files are saved.
- A 10-problem pilot (`q1-pilot`, GSM8K 0.30, `--no-unim`) preceded the runs; not counted.
