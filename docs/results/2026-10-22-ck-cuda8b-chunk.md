# ck O12: CUDA Qwen3-8B BF16 prefill chunk 256 (current default) vs 4096 (2026-10-22)

User-approved check (decision O12, "yes both", 11:46 EDT): if 256 vs 4096 is bitwise on CUDA 8B, raise the default. **It is not bitwise, so
no recipe or code change was made.** Same method as [2026-10-22-ck-prefill-chunk.md](2026-10-22-ck-prefill-chunk.md); spark2 (GB10), exclusive
fleetq slots with vLLM up (#4231, #4234), binary built from `origin/main` b325a7f8's tree plus `scripts/ck/` (#4230). `intarsia accuracy --mode
prefill|decode --write-self-golden`, top-64, compared bit by bit (`scripts/ck/chunkcmp.py --ref 256`), plus 64 greedy ids.

## Result: not bitwise, but only where a prompt leaves a short tail at chunk 256

| Prompts (tokens) | length mod 256 | prefill, 4096 vs 256 | decode mode (16 steps) |
|---|---|---|---|
| short golden (8 prompts, 18-90) | n/a | 0 of 434 rows differ | 0 differ |
| long_docs_4k 4064, long_code_4k 4064, chat_reuse_4k 4557 | 224, 224, 205 | 0 of 12685 rows | 0 |
| synth 4101 / 4200 / 8200 | 5 / 104 / 8 | 0 | 0 |
| synth 16000 / 16036 | 128 / 164 | 0 | **16036: 16 of 16 rows differ (max KL 8.6e-11, max abs d logprob 0.42)**; 16000: same, max KL 2.2e-10 |
| **synth 5686** | **54** | **54 of 5686 rows differ, all in the 54-row tail of the 256 schedule (first differing row 5632)**; max row KL 5.6e-3, max abs d logprob 0.585, 0 argmax flips | 16 of 16 rows, max KL 2.1e-3 |

* The prefill difference is the known short-tail effect (lane og's O5: CUDA tails under 128 rows are not bitwise-invariant), and it sits in the
  256 schedule, not in 4096. Prompts whose length mod 256 is below 128 reach it; the 4k/16k goldens used so far (tails 128-224) do not,
  which is why the first pass (`raw/cuda-8b-256`, 4k + 16000-token prompts) showed prefill bitwise. The decode-mode rows differ for two long
  prompts even with long tails (KL ~1e-10, no flips), so the KV/state left by the two schedules differs in the last bits.
* `generate --show-logits` on corpus prompts of 5686 and 15652 tokens (tails 54 and 36): 64 greedy ids identical, first-token top-20 logits
  (4 decimals) DIFFER at 256 vs 4096; the 4096 value equals the 2048 and 8192 values of the first lane run.
* Sanity: the comparator is the one that found the CUDA 35B difference; 0 rows differ everywhere else here.

## Speed and memory (exclusive, `bench speed --at-most`, 3 reps, 256 repeated as drift control)

| Prefill tok/s | 256 | 4096 | change |
|---|---|---|---|
| 4k | 2414 (runs 2411, 2416) | 4197 | +74 % |
| 16k | 2180 (2182, 2178) | 3631 | +67 % |

Memory: plan `scratch_bytes` 0.7 MiB at both (the 8B has no kernel scratch). The free-memory gauge is unusable on this unified-memory box; by the
Metal / HIP measurements of the same engine formula the activation workspace grows about 0.13 MB per chunk row, so about +0.5 GB at 4096 over 256.

## Recipes

`recipes/qwen3.toml` (`match.architectures = ["Qwen3ForCausalLM"]`) is the only recipe the 8B uses; every dense Qwen3 (0.6B ... 32B, the
reranker 0.6B) shares it. The 35B uses `qwen3_5_moe.toml` and is unaffected. Only the 8B was measured. Had the check passed, the change would
have been one `[tuning.cuda] prefill_chunk = 4096` line there. Not made.

## Options for the user

* Leave the CUDA 8B at 256 (bitwise to itself, 40 % slower prefill).
* Take 4096 anyway: +67-74 % prefill, logits differ from the 256 schedule only on short-tail prompts (max row KL 5.6e-3, in the tail rows)
  and in decode mode in the 1e-10 .. 2e-3 range; then a `[tuning.cuda]` line would still need the no-op and test runs.
* Use 2048 instead: lane ck's earlier run measured 2048 vs 4096 vs 8192 on the 8B bitwise (with tails >= 128 only); 256 vs 2048 was not run.

Raw: `2026-10-22-ck-prefill-chunk/raw/cuda-8b-256`, `cuda-8b-tails` (logs, plans, comparisons; dump sha256 in `cuda-8b-dump-sha256.txt`).
