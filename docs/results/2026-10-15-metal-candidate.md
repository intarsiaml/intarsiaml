# Metal candidate: spike/metal + P3 (chunk 2048) + F1 (NAX flash attention) (2026-10-15)

Branch `metal-candidate` = `spike/metal` 6271eec (includes P4/P5) + `--no-ff` merge of `metal-p3-chunk` + `--no-ff` merge of `metal-f1-nax-attn`, then `spike/metal` 6271eec merged back in. Qwen3-8B BF16, endor (Apple M5 Max, macOS 27.0). Not merged into `spike/metal`, not pushed. Raw logs: `docs/results/raw/2026-10-15-metal-candidate/`. Build: `head=0b2e3a2 code-tree=4757814f12d4 dirty=no` (`INTARSIA_BUILD_STAMP`, `cargo build --release -p intarsia-cli --features metal`).

## Outcome

* **The merge is clean: no textual conflicts** in any of the three merges (`build.rs`, `kernels.rs`, `lib.rs`, `recipes/qwen3.toml`, docs all merged additively). Nothing needed resolving by hand.
* **Every numeric gate passes**: metal tests, 269 workspace tests, `cargo check`, 23 kernels / 272 cases ADMITTED, accuracy floors on both goldens at chunk 2048 and 256, greedy ids bit-identical to spike/metal 6271eec (48 and 256 tokens, both prompts, including the short prompt, no token-49 divergence), ids identical across chunks 64 to 4050, 364 dispatches per decode step.
* **One accuracy number is below the stricter merge guardrail the coordinator used for F1: long-golden prefill top-1 is 99.58% (1428 of 1434), guardrail 99.6%** (spike/metal 99.65%, 1429 of 1434; one position). Long prefill KL improves (4.342e-4 to 3.993e-4), short prefill KL improves (2.686e-4 to 2.602e-4), decode KL improves (3.739e-4 to 3.089e-4 short, 3.731e-4 to 3.041e-4 long). The tool's own floors (top-1 >= 99.5%, KL <= 1e-3) pass. Short prefill top-1 is 99.54% (432 of 434; spike/metal 99.77%), which is also one position and only 0.04 points above the 99.5% floor. This is F1's number carried over unchanged (it is the same value F1's doc reports); P3 does not move accuracy and P4/P5 does not either.
* **`qualify --backend metal` at the shipped chunk (2048): FAIL, 12 stages: 10 pass, 1 fail, 2 skip.** The one failure is `acc-long`, only on the prefix-reuse gate: every golden prompt is below one 2048-token chunk (longest 700 tokens), so the gate is skipped and the tool counts a skip as FAIL (the stage summary line says "prefix reuse differs from a cold run", which is misleading: `acc-long.out` says "reuse gate: SKIPPED ... not a pass"). Exactly the P3 coverage gap, nothing new. **With the recipe forced to Metal chunk 256, qualify is PASS: 10 pass, 0 fail, 2 skip** (`ab`, `tp`), including the prefix-reuse gate.
* **Baseline self-check (P3's recorded Metal baselines and envelopes, read only, not re-recorded): PASS at the default chunk 2048 on both goldens, and PASS at chunk 256 with `--allow-config-mismatch`** (floor verdict PASS and baseline verdict PASS, per-prompt limits from the envelope, k = 2). The baselines were recorded without F1; F1's differences (short prefill KL -3%, long -8%, the one-position top-1 drops) sit inside the baseline margins and per-prompt envelopes. Without `--allow-config-mismatch` the chunk 256 check is refused by the tool ("baseline recorded with prefill chunk 2048"). The self-check also needs a stamped binary ("built without INTARSIA_BUILD_STAMP" otherwise).
* **Plain prefill vs spike/metal 6271eec (ABBA-style, 4 runs each, medians): 1.15x at 1k (401 to 350 ms), 1.33x at 4k (1708 to 1287 ms), 1.75x at 16k (12245 to 7013 ms)**, decode unchanged (28.1 to 28.2 ms/step at 4k, 27.1 ms short, 364 dispatches). At the same chunk, F1 alone (cand at chunk 256 vs spike at 256) is 1.04x / 1.24x / 1.58x.

Verdict for the coordinator: numerically the candidate is correct (greedy output identical, KL better than spike/metal everywhere), the speed gain is large at 4k and 16k, and the two open items are (a) the long top-1 99.58% vs the 99.6% guardrail (one position out of 1434) and (b) the qualify prefix-reuse coverage gap at chunk 2048 (needs a 4k long golden or a decision to accept the gap). Whether (a) and (b) block a merge is the coordinator's call.

## Gates (all on the candidate build; logs in `gates/`, driver `gates.sh`, fleetq #1957)

| gate | result |
|---|---|
| `cargo check --workspace` | pass (`gates/cargo-check.txt`) |
| `cargo test --workspace --release --exclude intarsia-backend-metal` | 269 passed, 0 failed, 1 ignored (`gates/cargo-test-ws.txt`) |
| `cargo test -p intarsia-backend-metal --release -- --test-threads=1` | all pass (`gates/cargo-test-metal.txt`), includes `flash_prefill_nax`, `metal4_fallback` |
| `intarsia kernels test --backend metal -v` | 23 kernels, 272 cases, all ADMITTED (`gates/kernels-test.txt`) |
| default chunk | `plan show`: 2048 (recipe qwen3, metal) |
| accuracy floors (bf16 class), short and long, chunk 2048 and 256 | all PASS (table below) |
| greedy ids, base vs candidate, `prompt-short` and `prompt-4k`, 48 and 256 tokens | identical in all four (`gates/identity/summary.txt`) |
| chunk invariance, `prompt-4k`, 48 ids, default vs chunk 64 / 128 / 256 / 512 / 1024 / 2048 / 4050 | identical for all (`gates/chunk/summary.txt`) |
| short-golden prefill KL by chunk 16 / 64 / 128 / 256 / 512 / 2048 | 2.883e-4 / 2.613e-4 / 2.602e-4 / 2.602e-4 / 2.602e-4 / 2.602e-4 |
| decode dispatches per step | 364 for base and candidate, short and 4k (ABBA) |
| baseline self-check, chunk 2048 | PASS short and long (`gates/accuracy/selfcheck-*-cdefault.out`) |
| baseline self-check, chunk 256 (`--allow-config-mismatch`) | PASS short and long (`extras/selfcheck-*-c256.out`, fleetq #1961) |
| `qualify --backend metal --bandwidth-gbs 614`, recipe as shipped (chunk 2048), fresh dir | **FAIL**: 10 pass, 1 fail (`acc-long`: prefix-reuse gate skipped, counted as fail), 2 skip |
| `qualify`, recipe copy with `[tuning.metal] prefill_chunk = 256` (`gates/qwen3-c256.toml`, `--recipe`) | **PASS**: 10 pass, 0 fail, 2 skip |

Qualify stages, chunk 2048: doctor PASS, recipe PASS, plan PASS (36 s), kernels PASS (23 / 272), gate PASS (perplexity 1.85, 5/5 sanity), acc-short PASS (prefill 99.54% / 2.60e-4, decode 100% / 3.09e-4), bench PASS (decode 36.7 / 36.6 / 35.4 tok/s at 128 / 1024 / 4096, 90 / 90 / 87% of the byte floor; prefill 2447 / 3933 / 3478 tok/s), acc-long **FAIL** (prefill 99.58% / 3.99e-4, decode 99.80% / 3.02e-4, floor PASS, prefix reuse skipped), envelope PASS ("gate this build: PASS"), bench-long PASS, ab SKIP (no reference server), tp SKIP. Chunk 256 identical except acc-long PASS (prefix reuse runs: 2 of 10 prompts span more than one chunk), bench prefill 2418 / 3775 / 3356 tok/s, decode 36.5 / 36.6 / 35.6.

## KL and top-1 (teacher-forced vs fp32 goldens, bf16 gate class)

| run | short prefill top-1 / KL | long prefill top-1 / KL | short decode top-1 / KL | long decode top-1 / KL |
|---|---|---|---|---|
| spike/metal 6271eec (chunk 256) | 99.77% / 2.686e-4 | 99.65% (1429/1434) / 4.342e-4 | 100% / 3.739e-4 | 100% / 3.731e-4 |
| candidate, default chunk 2048 | 99.54% / 2.602e-4 | **99.58% (1428/1434)** / 3.993e-4 | 100% / 3.089e-4 | 100% / 3.041e-4 |
| candidate, `--prefill-chunk 256` | 99.54% / 2.602e-4 | 99.58% / 3.993e-4 | 100% / 3.089e-4 | 100% / 3.041e-4 |

The spike/metal row reproduces the coordinator's numbers to the digit. The candidate is identical at chunk 2048 and 256 on both goldens (the short golden never crosses a chunk; the long golden crosses 256 for two prompts and the attention kernel is chunk-invariant bitwise, per F1's tests). The numbers match F1's doc exactly, so P3 and P4/P5 add no accuracy change on top of F1. Floors: top-1 >= 99.5%, KL <= 1e-3, all PASS. Files `gates/accuracy/{base,cand,cand-c256}-{short,long}.out`. Note the golden is fp32 with an f32 KV cache (tool warning), so part of the gap is reference-side noise.

## Plain prefill time (generate --max-tokens 1, ms, fleetq #1959, `speed/`)

Order per prompt: palindromic rounds `base cand cand256 base2048 | base2048 cand256 cand base`, twice, so each arm has 4 runs; one discarded warm-up. Arms: **base** = spike/metal 6271eec as shipped (chunk 256); **cand** = candidate as shipped (chunk 2048); **cand256** = candidate at `--prefill-chunk 256` (F1 without the P3 chunk); **base2048** = spike/metal at `--prefill-chunk 2048` (the P3 chunk without F1). Median (min-max), speed-up vs base.

| prompt | base | cand | vs base | cand256 | vs base | base2048 | vs base |
|---|---|---|---:|---|---:|---|---:|
| 1k (973 tok) | 401 (399-403) | **350 (350-352)** | 1.15x | 387 (383-390) | 1.04x | 357 (355-363) | 1.12x |
| 4k (4062 tok) | 1708 (1616-1821) | **1287 (1215-1370)** | 1.33x | 1382 (1324-1437) | 1.24x | 1471 (1458-1619) | 1.16x |
| 16k (about 16.2k tok) | 12245 (11221-13132) | **7013 (6943-7547)** | 1.75x | 7747 (7550-8045) | 1.58x | 11539 (11126-11914) | 1.06x |

The 4k arms are noisy (base spans 1616 to 1821 ms, cand 1215 to 1370 ms), the ranges do not overlap between base and cand in any of the three prompts. At the start of the run the GPU read 94% utilisation on the first idle sample and 0 to 1% on the next two (`speed/idle.txt`), no zoom. F1 and the chunk compose: F1 gives the long-context gain (attention), the chunk gives the short-context gain, and P4/P5 (already in base) accounts for base being faster at chunk 256 than P3's older base numbers (1725 ms at 4k then, 1708 ms now with this noise).

Decode cross-check (`extras/`, fleetq #1961, ms/step from the cb trace, 364 dispatches everywhere): 4k pre (766c723 build) 28.16 / 28.09, base 28.35 / 28.09, cand 28.96 / 28.15; short pre 27.10 / 27.07, base 27.32 / 27.04, cand 27.16 / 27.10. The gates run's decode ABBA (`gates/driver.log`) read 29.6 to 29.9 ms at 4k for both base and cand and 27.2 to 28.3 short, while the machine was in a slower phase (other fleet activity between slots); base and cand are equal within noise in every set, so this is drift, not code. Decode is unchanged by P3 + F1.

## Conflicts resolved

None. `git merge --no-ff metal-p3-chunk` and `git merge --no-ff metal-f1-nax-attn` went through with no conflict markers (the registrations in `build.rs` / `kernels.rs` / `lib.rs` touch different hunks; P3 touches `recipes/qwen3.toml` and golden baselines, F1 the kernel and `flash_nax.rs`). Merging `spike/metal` 6271eec (P4/P5: NAX tile pick for 128 <= M <= 512, prefill `qk_norm_rope_kv_bf16_pf`; `gemm_nax_bf16.metal`, `qk_rope_kv.metal`, `gemm_nax.rs`, `dense.rs`, `kernels.rs`) was also clean. Net diff against `spike/metal`: `build.rs`, `flash_bench.rs`, `flash_nax.rs`, `kernels.rs`, `lib.rs`, the two NAX tests, `kernels/metal/attention_flash_nax.metal`, `recipes/qwen3.toml`, golden baselines, docs and raw data.

History: the first gate run was on the older tree (spike/metal a710f35 + P3 + F1) and was stopped during qualify when the coordinator advanced spike/metal to 6271eec; its partial logs are kept in `superseded-a710f35-tree/` (accuracy, ids, chunk invariance and the first 7 qualify stages all matched the final run; none of it is used in the tables above). One more thing seen there: the self-check needs a stamped build, so the final binary is stamped.

## Not done / limits

* Baselines were not re-recorded (as instructed); `qualify`'s `envelope` stage wrote its own baseline and envelope inside the qualify output dir only (`git status` shows `golden/baselines` untouched).
* The 8B long golden has no prompt over 700 tokens, so neither the prefix-reuse gate at chunk 2048 nor a multi-chunk F1 check on real text above 2048 runs in qualify. Only the 4k-prompt greedy ids and the warm/cold `serve` check from P3 cover that; the `serve` warm/cold reuse check was not repeated on the candidate.
* 16k accuracy was not measured. Only head_dim 128 has the NAX attention kernel (F1 limit).
* Speed is plain prefill through `generate`; the localeval-through-`serve` ABBA was not rerun.

## Runs and fleetq

fleetq (all exclusive on endor, released): #1956 (first gates, older tree, stopped and released), #1957 gates (final tree), #1959 prefill ABBA, #1961 extras (chunk 256 self-check, decode cross-check). Binaries: base = build of `spike/metal` 6271eec (detached worktree), cand = stamped build above. Scripts: `gates.sh`, `speed.sh`, `extras.sh`. Commits on `metal-candidate`: merge of `metal-p3-chunk`, merge of `metal-f1-nax-attn`, merge of `spike/metal` 6271eec, and this doc commit.
