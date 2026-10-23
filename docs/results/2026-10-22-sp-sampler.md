# Host sampler cost on sampled decode (lane sp, 2026-10-22)

Lane so (PR #102) measured unconstrained `temperature: 0.7` decode on endor at 26.3 tok/s against 37.0 greedy and
36.4 constrained greedy (which already reads host logits). This lane found the gap and removed it.

## Cause

`sample_row` (`crates/intarsia-core/src/sampler.rs`) sorted all 151936 vocab indices with a comparator that loads
`logits[a]` and `logits[b]` on every compare (about 2.6M random loads per token). With `top_k = 0` (the ssbench
sampled arm, and the API default) nothing truncates first.

CPU microbench (`sampler_timing`, ignored test; 151936 logits rounded to bf16 so ties are realistic, T = 0.7,
fleetq #4223, endor, non-exclusive):

| phase | ms per call |
|---|---|
| whole `sample_row`, comparator sort (main) | 9.31 |
| of which: comparator sort | 8.87 |
| of which: f64 exp + sum pass | 0.32 |
| whole `sample_row`, keyed radix sort (this PR) | 1.10 |
| of which: key build + radix sort | 0.67 |

RNG and allocation are negligible. The sort is the 11 ms.

## Fix (default-on, bit-identical)

Each logit becomes a flat `u64` key: an order-preserving transform of the f32 (descending; `-0.0` mapped to `+0.0`,
since `partial_cmp` treats them as equal), high 32 bits, and the index, low 32 bits. Keys are built in index order and
sorted by a stable 3-pass LSD radix sort (11-bit digits) on the high half. Ties therefore stay in ascending index. That
is the comparator's exact total order. The `top_k` path uses `select_nth_unstable` + `sort_unstable` on the same keys.
The softmax, top-p cut, renormalisation and cumulative walk are unchanged, so the f64 summation order and every
probability are bit-for-bit the same. Rows that contain NaN keep the old comparator path, because `partial_cmp` has no
total order there. The cpu-ref `Sample` op calls `sample_row` too, so it gets the same order.

## Gates

- **Unit equivalence** (`keyed_sort_matches_comparator_sort`): new vs old path on rows of size 1 to 151936. Covers
  bf16 ties, `±0.0`, `-inf` (min_tokens stop masking), all-equal rows, T in {0.3, 0.7, 1.0, 1.5},
  top_k in {0, 1, 40, n-1, n}, top_p in {1.0, 0.95, 0.5}, and u in {0, 1e-7, ..., 1 - 2^-24} plus random draws.
  The picks are identical everywhere.
- **End-to-end seeded identity** (`scripts/seeded_ids.py`, `return_token_ids`): 2 prompts × 3 parameter sets
  ((0.7, 1.0, 0), (1.0, 0.95, 0), (0.8, 0.9, 40)) × seeds {1, 2} × 300 `ignore_eos` tokens. That is **12 cases, 3600
  sampled tokens, 0 differ** between base and head. The seeds do change the output: seed 1 and seed 2 diverge at
  token 0 to 29 in every case.
- `cargo test --release --workspace --no-fail-fast` (fleetq #4227, separate `CARGO_TARGET_DIR`): **747 passed, 0
  failed**, 4 ignored. Log: `2026-10-22-sp-sampler/test/cargo-test-workspace.log`. Debug `cargo test --workspace
  --no-fail-fast` (fleetq #4235): also 747 passed, 0 failed, 4 ignored (`cargo-test-workspace-debug.log`). The fuzz test
  takes 19.7 s in debug and 11.7 s in release.

## Per-token time, before and after (exclusive endor, fleetq #4228, 11:47-11:57 EDT)

Qwen3-8B BF16, Metal, `--max-seq 4096`, ssbench `speed` (copied from PR #102), 177 tokens per request. Visits ran
base, head, head, base, with a fresh server each time and 2 reps per visit, so n = 4 per cell. Medians (min-max):

| arm | base (b325a7f8) tok/s | head tok/s | base ms/tok | head ms/tok |
|---|---|---|---|---|
| unconstrained greedy, `ignore_eos` | 35.95 (35.83-36.47) | 36.40 (35.84-37.06) | 27.8 | 27.5 |
| constrained greedy (fruits-40) | 34.96 (34.64-35.81) | 35.54 (35.41-35.77) | 28.6 | 28.1 |
| **sampled, T 0.7, `ignore_eos`** | **25.90 (25.75-26.02)** | **34.92 (31.50-35.24)** | **38.6** | **28.6** |

Sampled decode costs 10.0 ms less per token, and the arm now matches constrained greedy (same host-logits path). The
remaining sampled-vs-greedy gap is about 1 ms per token: the 1.1 ms sampler, against the device-argmax/token-feedback
path that greedy uses. The 31.50 low is one rep in visit 3; the other three head reps are 34.6 to 35.2.
Greedy and constrained differ by under 2% between base and head. Neither path calls the sampler, so that is noise.

Downtime: none. No role taken down, endor only. Builds: fleetq #4225 (non-exclusive). Binary md5s are in
`run/versions.txt`.

## Not done

- No device-side sampling. Sampled requests still read logits back to the host. That costs at most about 0.6 ms per token
  (constrained greedy minus device greedy, which also includes the mask).
- The 1.1 ms left could shrink further, for example by reusing the sort buffers across calls (about 3.6 MB of
  allocation per call). It was not done, because it is under 4% of a token.

Files: `2026-10-22-sp-sampler/scripts/` (`build.sh`, `run.sh`, `seeded_ids.py`, `ssbench.py`), raw `run/`
(`ab-speed.jsonl`, `ids-{base,head}.json`, server logs, `job.log`).
