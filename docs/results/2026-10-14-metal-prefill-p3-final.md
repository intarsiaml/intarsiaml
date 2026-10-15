# Metal prefill P3: chunk size (2048 on Metal)

Qwen3-8B BF16, endor (Apple M5 Max, `applegpu_g17s`, macOS 27.0). Branch `metal-p3-chunk` from `spike/metal` 2e1fdb1 (P1 NAX GEMM + P2 flash attention). Raw data: `docs/results/raw/2026-10-14-metal-prefill-p3/`.

## Outcome

* **The chunk default for Metal is now 2048** (`[tuning.metal] prefill_chunk = 2048` in `recipes/qwen3.toml`; CUDA, HIP and CPU keep 256). Plain prefill is 1.19x faster at 1k and 4k and 1.04x at 16k. Through `serve` (localeval, ABBA) marginal prefill is +12% to +22% at 1k and +23% to +49% at 4k (new vs the two pre runs), decode unchanged.
* **All numeric gates pass and are bit-for-bit unchanged from 2e1fdb1**: kernels 22 / 265 cases ADMITTED, both cargo test runs, `cargo check`, accuracy (short prefill KL 2.686e-4, long 4.342e-4, same digits as before), greedy ids identical at 48 and 256 tokens on both prompts (including the short prompt, no token-49 divergence this time), ids identical across chunks 64 to 4050, 364 dispatches per decode step.
* **`intarsia qualify --backend metal` returns FAIL (9 pass, 1 fail, 2 skip), and the cause is structural, not numerical.** `acc-long` fails only on the prefix-reuse gate: the 8B long golden's longest prompt is 700 tokens, the gate needs a reuse split of at least one chunk, so at 2048 every prompt is skipped and the tool treats "skipped" as "not a pass". The same gate would be skipped at **any chunk above 350** (512 and 1024 too), so no speedup beyond 256 can pass qualify with the current golden. At `--prefill-chunk 256` the same run passes (`prefix-reuse PASS`). I did not touch qualify or the accuracy code to hide it.
* **Merge-readiness: the numeric guardrails (long prefill KL <= 5e-4, long top-1 >= 99.6%, no greedy-id difference) hold; the qualify FAIL is a coverage gap that needs a decision.** I would call the branch NOT merge-ready as a clean qualify. Either (a) keep 256 (no P3 gain), or (b) accept 2048 with the reuse gap, or (c) build a Qwen3-8B fp32 long golden that contains 4k prompts (`tools/prompts/golden_4k.jsonl` already has `chat_reuse_4k` with `reuse_after_message`; needs `golden.py --dtype float32` on spark1, not done). Precedent: the CUDA 35B recipe shipped 2048 with a known `chat_reuse_4k` sensitivity. Supplementary evidence at the production chunk is below (warm vs cold through `serve`: identical).
* Part 2 (four-engine same-slot run) was **not run**, by the coordinator's instruction; a localeval pre/new ABBA pair was run instead (section 5).

## 1. Chunk sweep (plain prefill, `generate --max-tokens 1`, binary 2e1fdb1, 3 reps in order fwd / rev / fwd, 1 discarded warm-up)

Prompts: `prompt-1k.txt` (973 tokens), `prompt-4k.txt` (4062 tokens), `prompt-16k.txt` (about 16.2k tokens, `--max-seq 18432`); 1k and 4k at `--max-seq 8192`. fleetq #1910. Table from `sweep/table.md`.

| prompt | chunk | prefill ms median (min-max) | tok/s | vs 256 |
|---|---:|---|---:|---:|
| 1k | 256 | 459 (457-485) | 2120 | 1.00x |
| 1k | 512 | 409 (408-412) | 2379 | 1.12x |
| 1k | 1024 | 387 (372-388) | 2514 | 1.19x |
| 1k | 2048 | 386 (383-388) | 2521 | 1.19x |
| 4k | 256 | 1725 (1661-1751) | 2355 | 1.00x |
| 4k | 512 | 1560 (1501-1637) | 2604 | 1.11x |
| 4k | 1024 | 1499 (1478-1635) | 2710 | 1.15x |
| 4k | 2048 | 1448 (1447-1664) | 2805 | 1.19x |
| 16k | 256 | 12998 (12611-13812) | 1275 | 1.00x |
| 16k | 512 | 12686 (12481-12769) | 1307 | 1.02x |
| 16k | 1024 | 12627 (12471-12752) | 1313 | 1.03x |
| 16k | 2048 | 12532 (11556-12539) | 1323 | 1.04x |

Run-to-run spread is about 5% (rep 3 of 4k is slow for both 1024 and 2048). 1024 and 2048 are tied at 1k, 2048 is 3% ahead at 4k (reps 1 and 2; reversed in rep 3), 1% at 16k. The gain above 1024 is inside the noise; at 16k attention dominates and chunk barely matters.

**Memory.** `/usr/bin/time -l` peak memory footprint (the 16 GB of weights are mmapped and not in it): at `--max-seq 8192` it sits at 2.3 to 2.55 GiB for every chunk and varies by about 100 MiB between identical runs, so no chunk effect is visible; at `--max-seq 18432` it grows with chunk: 16k prompt 2877 / 2909 / 2972 / 3101 MiB for chunk 256 / 512 / 1024 / 2048 (+224 MiB for 2048), 4k prompt 3013 MiB at 256 vs 3233 MiB at 2048 (`sweep/*.err`). `plan show --json` `scratch_bytes` is 1,064,960 at every chunk (kernel scratch only, chunk independent). The 128 GB machine makes +220 MiB irrelevant; activation buffers scale with chunk (about 100 MB per 2048 rows for the gate/up intermediate).

## 2. Chosen default and why

2048: fastest or tied at all three lengths, memory cost about +220 MiB, 1024 would be an equally defensible choice (within noise at 1k and 4k). Costs to know about: a 2048-row chunk is about 0.7 s of GPU work between scheduler steps (about 0.09 s at 256), which matters only if `serve` interleaves decode of other sequences with a long prefill (Metal is single-sequence today, `max_batch = 1`); and the reuse-gate gap above. The mechanism needed no code: `[tuning.<backend>]` already accepts any backend name, and `metal` resolves to `prefill chunk 2048 (recipe qwen3, metal)` in `generate`, `serve`, `accuracy` and `plan show`. The CPU plan still shows `256 (default)` (`gates/plan/cpu-default.json`).

## 3. Files changed

* `recipes/qwen3.toml`: added `[tuning.metal] prefill_chunk = 2048` with a comment.
* `docs/adding-a-model.md`: backend list for `[tuning.<backend>]` now says `cpu | cuda | hip | metal`.
* `golden/baselines/*.json` (4 new files, see section 4), `docs/results/...` (this document and raw data).
* **No `intarsia-core` file and no `intarsia-cli` file changed** (`git diff spike/metal --stat -- crates` is empty). The recipe is embedded in the binary by `crates/intarsia-models/src/recipe.rs` (`include_str!`), so every build embeds the new `[tuning.metal]` table, but it is only read when the backend is `metal`.

## 4. Baselines and noise envelope re-recorded (Metal only)

None existed before: no Metal baseline or envelope was tracked anywhere in the repository (`golden/baselines/` did not exist; earlier Metal qualify runs wrote them only into their report directories). These are therefore the first tracked Metal files, recorded with the P3 build at chunk 2048 (`prefill_chunk: 2048, recipe qwen3, metal` is in each file's `base_run`), stamp `head=e8be186 code-tree=6c2d16b008e2 dirty=no`. The commit after it changed only a comment in `recipes/qwen3.toml`, so the code-tree hash recorded in the files differs from HEAD's by that comment. CUDA/HIP baselines were not touched.

| file | golden | flags | contents |
|---|---|---|---|
| `golden/baselines/qwen3-8b-fp32.metal.baseline.json` | short (8 prompts, 434 positions) | `--mode both --decode-positions 16 --max-seq 4096` | prefill mean KL 2.686e-4, top-1 99.77%; decode KL 3.74e-4 |
| `golden/baselines/qwen3-8b-fp32.metal.envelope.json` | short | perturbation `chunk=64` (2048 to 64) | prefill mean KL 2.686e-4 to 2.682e-4; only 1 of 8 prompts moves in prefill (max delta of 21-way KL 2.1e-6), 4 of 8 in decode (max 8.3e-5) |
| `golden/baselines/qwen3-8b-long-fp32.metal.baseline.json` | long (10 prompts, 1434 positions) | `--mode both --decode-positions 64 --max-seq 8192` | prefill mean KL 4.342e-4, top-1 99.65% |
| `golden/baselines/qwen3-8b-long-fp32.metal.envelope.json` | long | perturbation `chunk=256` (2048 to 256) | **all zero**: prefill identical to the last digit on all 10 prompts, decode max delta 1.4e-20 |

Self-checks (`--baseline` + `--noise-envelope` against its own build): both `baseline verdict ... PASS` (`baselines/*-selfcheck.out`).

Honest limits: the 8B goldens' prompts are at most 700 tokens (short golden: at most 90), so at chunk 2048 **no golden prompt crosses a chunk boundary** (accuracy prints "multi-chunk prefill is untested"). The envelopes therefore measure the effect of splitting into small chunks, which is tiny (and zero on the long golden); the long-golden envelope is degenerate and the gate falls back to the fixed margins there. `qualify`'s own envelope stage perturbs to base/4 = 512, which splits nothing, so its envelope is all zero by construction (the qualify run's `baselines/` output is in `gates/qualify/baselines/`; the files above use explicit smaller perturbations instead). Multi-chunk behaviour at the production chunk is covered separately by the 4k-prompt chunk-invariance runs (section 6) and the warm vs cold check.

## 5. Speed through `serve` (localeval, `--sizes 128,1024,4096 --gen 256 --reps 3 --thinking off`, one exclusive slot fleetq #1914, order pre new new pre)

pre = 2e1fdb1 build (chunk 256), new = P3 build (chunk 2048). Median (min-max). GPU Device Utilization before each run (median of 3): pre-1 6%, new-1 5% (3 waits), new-2 10% (3 waits), pre-2 2% (1 wait); no zoom.us process.

| run | decode 128 | decode 1k | decode 4k | marginal prefill 1k | marginal prefill 4k |
|---|---|---|---|---|---|
| pre-1 | 31.7 (30.0-33.6) | 34.6 (33.4-35.1) | 33.8 (33.8-34.2) | 3204 | 1936 |
| new-1 | 36.0 (35.6-36.0) | 35.8 (35.0-35.9) | 34.3 (33.1-34.6) | 3924 | 2858 |
| new-2 | 36.2 (36.0-36.2) | 36.0 (35.9-36.0) | 35.0 (34.9-35.0) | 3826 | 2893 |
| pre-2 | 34.9 (34.5-36.2) | 36.0 (35.8-36.0) | 35.0 (34.3-35.1) | 3428 | 2328 |

New over pre: 1k marginal prefill +19% to +22% vs pre-1, +12% to +14% vs pre-2 (mean 3875 vs 3316, +17%); 4k +48% to +49% vs pre-1, +23% to +24% vs pre-2 (mean 2876 vs 2132, +35%). TTFT at ~4090 tokens: pre 1.93 / 1.64 s, new 1.36 / 1.35 s. The pre runs are noisy (pre-1 vs pre-2 differ by 20% at 4k, pre-1 also has a slow 128 decode), so the conservative gain is the new vs pre-2 pair: +13% at 1k, +24% at 4k. Decode is unchanged within noise (new 34.3 to 36.2 vs pre 31.7 to 36.0 tok/s; the pre-1 128 point is an outlier). Against the P1+P2 doc's mlx-lm figures (4423 / 3134 tok/s at 1k / 4k, different slot, not a same-slot comparison) new is 88% / 92% at 1k / 4k.
Qualify's bench (own server run, 128 / 1024 / 4096 tokens): prefill 2407 / 3621 / 2659 tok/s vs 2315 / 3316 / 2491 in the P1+P2 run; decode 36.0 / 36.0 / 34.7 tok/s (89 / 89 / 86% of the byte floor).

## 6. Gates (fleetq #1911 for tests/accuracy/ids/chunks/qualify, #1913 for baselines and reuse; `gates/`, `baselines/`)

| gate | result |
|---|---|
| `cargo check --workspace` | pass (`gates/cargo-check.txt`) |
| `cargo test --workspace --release --exclude intarsia-backend-metal` | 267 passed, 0 failed, 1 ignored |
| `cargo test -p intarsia-backend-metal --release -- --test-threads=1` | all pass (`gates/cargo-test-metal.txt`) |
| `intarsia kernels test --backend metal -v` | 22 kernels, 265 cases, all ADMITTED |
| accuracy, short golden (bf16 gate) | prefill top-1 99.77%, KL **2.686e-4** (was 2.686e-4); decode 100%, 3.739e-4; PASS |
| accuracy, long golden | prefill top-1 99.65%, KL **4.342e-4** (was 4.342e-4); decode 100%, 3.731e-4; PASS (long KL at `--prefill-chunk` 256 and 1024: 4.342e-4 both) |
| greedy ids, 2e1fdb1 vs P3, 48 and 256 tokens, `prompt-short` and `prompt-4k` | identical in all four |
| chunk invariance, `prompt-4k`, 48 ids, default vs 64 / 128 / 256 / 512 / 1024 / 2048 / 4050 | identical for all; short-golden prefill KL at chunk 16 / 64 / 128 / 2048: 2.830e-4 / 2.682e-4 / 2.686e-4 / 2.686e-4 |
| decode dispatches per step | 364 for base and P3, short and 4k, ABBA; step 27.2 to 27.5 ms (short), 28.2 to 28.7 ms (4k) for both; tok/s 36.2 to 36.5 short, 34.4 to 35.1 4k (within noise) |
| `intarsia qualify --backend metal --bandwidth-gbs 614 --golden-dir ...` (fresh) | **FAIL**: 12 stages, 9 pass, 1 fail (`acc-long`: prefix reuse, see Outcome), 2 skip (`ab`, `tp`) |
| `accuracy --mode all` on the long golden at `--prefill-chunk 256` | floor PASS, **prefix-reuse PASS** (2 of 10 prompts span more than one chunk) |
| same at the default 2048 | floor PASS, reuse gate SKIPPED (no prompt long enough), counted as FAIL |
| warm vs cold prefix reuse through `serve`, 4k prompt, two-turn chat, greedy 48 tokens (`baselines/reuse-serve.log`) | chunk 2048: 4085 of 4109 prompt tokens served from cache, text identical to the cold server; chunk 256: same, identical |
| no-op elsewhere | `plan show --backend cpu` resolves `256 (default)`; no crate other than the embedded recipe differs |

## 7. Versions, runs, fleetq

endor, macOS 27.0 (26A428), Metal toolchain as in earlier P1/P2 docs. Binaries: base = build of 2e1fdb1, P3 = stamped build `head=e8be186 code-tree=6c2d16b008e2 dirty=no` (`INTARSIA_BUILD_STAMP`), `--features metal`. Commits: `e8be186` (recipe + sweep), `7a976b3` (baselines, gates, recipe comment), plus the doc commit and the merge of `spike/metal`. fleetq (all exclusive, released): #1909 (aborted at once, binary lacked `--features metal`; no data), #1910 sweep, #1911 gates, #1913 baselines/reuse, #1914 localeval ABBA. localeval result dirs: `speed/20261014-224228-adhoc-p3speed-pre-1`, `...224511-...-new-1`, `...224745-...-new-2`, `...224953-...-pre-2`. Scripts: `sweep.sh`, `gates.sh`, `baselines.sh`, `reuse_serve.py`, `speed_pair.sh`; `sameslot.sh` is prepared for the final four-engine run (intarsia, mlx-lm, llama.cpp, intarsia; note llama.cpp in Homebrew is now 0.6.0_1, the P3-era phase-B runs used 0.5.0 build 11146).

## 8. Not done

* The four-engine same-slot measurement and its criteria verdicts (coordinator deferred it to the final HEAD).
* A Qwen3-8B fp32 long golden with 4k prompts (needed to run the reuse gate at any chunk above 350 and to make qualify pass at 2048).
* The envelopes are not a calibrated measure of chunk noise at the production chunk (no golden prompt crosses a 2048 boundary); a long golden would fix that too.
* Memory was measured as process footprint only; GPU-side allocation per chunk was not read directly.
* 16k prompt accuracy and a 35B/other-model view of the chunk were out of scope.
