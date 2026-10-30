# Lane WMR: evidence for the HIP WMMA rows family in batched serving (2026-10-30)

**DO NOT MERGE: needs user decision on the row-exact contract.** No default changed, no engine or kernel code changed; the only additions are scripts under `scripts/wmr/` and this folder.

Question: should a server started with `serve --max-batch > 1` run every row of every step (solo and batched) through the WMMA small-row kernel family of lane wg (`kernels/hip/ik_f8_rows.h`), so the +30 % at 16 streams becomes available, at the price of a one-time change of the solo output against the scalar GEMV and a few percent of single-stream decode speed? This page measures it on current main and drafts the contract wording. The decision is the user's.

Tree: `d2fe64a` (code-tree `917c430d6a38`, clean), built on tatooine as `~/intarsiaml-tp2-wmr` (release CLI with `--features hip`, plus the HIP and models GPU tests compiled), HIP, 2 x R9700 (gfx1201), Qwen3.6-35B-A3B-NVFP4, spec off. Raw logs: `2026-10-30-wmr-hip-wmma-rows-evidence/raw/`. Scripts: `scripts/wmr/` (`build.sh`, `window.sh`, `guarded.sh`, `gates.sh`, `acc.sh`, `ident.py`, `summarize.py`).

## 0. Summary

| | result |
|---|---|
| What "the flag" must be | `INTARSIA_HIP_WMMA_ROWS=1` **alone is not the +30 % configuration** and is not a no-op either. It selects the default table (W8A16 only), which on the 35B moves three fp8 decode folds (50 calls per step) to WMMA and nothing else; measured: no gain (-0.7 % at 16 streams) and -3.2 % at one stream. The speed-up needs `INTARSIA_HIP_WMMA_TABLE=fp8a8=all,bf16=all` as well (lane wg "option 1"). Everything below that says "flag on" is that pair. |
| Coverage (1) | fp8 dense projections (W8A8 and W8A16), their decode folds, bf16 router/gate matmuls and pairs. Not covered: lm_head, nvfp4 shared expert, MoE experts, attention, GDN. M = 1 goes through the family when the pair is set (plan show, section 1). |
| `rows_exact` (2) | passes (knob unset, ROWS=1 alone, option 1). Caveat: the WMMA tests drive the kernels by name, so they are independent of the env; they prove the family is M-invariant for M = 1..64, not the selection. Selection is proven by `plan show` and the served runs. |
| Served speed (3), aggregate decode, 3 interleaved rounds | TP=1: 1 / 4 / 8 / 16 streams **-4.6 / +2.0 / +14.7 / +34.6 %** (157.7 -> 150.5, 291.3 -> 297.1, 356.1 -> 408.6, 370.9 -> 499.1 tok/s). TP=2: **-2.2 / +2.4 / +10.5 / +30.0 %** (185.6 -> 181.5, 358.0 -> 366.8, 483.6 -> 534.4, 512.7 -> 666.5). |
| Single-stream cost if every process used the family (3) | `--max-batch 1`, decode: TP=1 **-4.2 / -3.6 / -3.0 %** at 1k / 4k / 16k, TP=2 **-2.8 / -2.7 / -1.6 %**. Prefill: within noise (+3.3 .. -0.9 %). With the proposed rule (family only for `--max-batch > 1`) a `--max-batch 1` server pays nothing; a lone request on a batched server pays the 1-stream row above. |
| Quality (4) | Self-golden against main's own dump, `--mode both`, TP=1 (TP=2 in 4.1): prefill unchanged (mean KL21 2.6e-12). Decode: mean KL21 **2.70e-3 short / 3.89e-3 long**, = **0.85x / 0.18x** vLLM's run-to-run noise: **PASS** under the repo's `--self-vs-ref-noise` rule; **FAIL** the default same-numerics budgets (1e-4); top-1 flips 1/160 (0.63 %) short, 9/320 (2.8 %) long. TP=2: **2.81e-3 short (0.89x), 3.51e-3 long (0.16x)**, PASS, flips 2/160 and 8/320. |
| Cross-stream identity (4) | flag on: every one of 16 prompts gives the same text at 1, 4, 8 and 16 streams (and on a repeat solo run), TP=1 and TP=2, 160 greedy tokens each. Control (flag off): same. Flag on vs flag off, solo: **16 of 16 prompts differ** (first differing character: median 237 at TP=1, 189 at TP=2): the one-time solo change is visible in text. |
| Batched == solo in one process (4) | `batch_cuda_real` 35B `batched_decode_is_bitwise_the_solo_step` and `batched_engine_equals_the_local_engine`: pass with option 1 on (and off). |

## 1. What the flag covers on main (item 1)

Mechanism. `wmma_rows_enabled()` is true only for exactly `INTARSIA_HIP_WMMA_ROWS=1` (`crates/intarsia-backend-hip/src/quantmoe.rs:108-116`); `wmma_table()` is then `WmmaTable::DEFAULT` (`:161`: `fp8a16` rows 1..=16, `fp8a8` and `bf16` off) or the `INTARSIA_HIP_WMMA_TABLE` override (`:201-213`, `:164-188`; `fp8a8=all,bf16=all` is `1..=ROWS_MAX`, `ROWS_MAX` = 16). Every registration asks `wmma_takes(kind, m)` (`:215`), and the GEMV twins carry the negation, so for a given kind and M exactly one family is admitted.

Selection by the model: a batched step marks its ops `exact_rows` (`crates/intarsia-core/src/model/batch.rs:375`; TP path `:633`), and `Registry::select` then takes only `rows_exact` kernels when the op has any (`crates/intarsia-core/src/registry.rs:358-366`). A one-row batch is the solo step (`batch.rs:362-367`). The solo step makes ordinary calls.

| op (35B decode) | flag off | option 1 (ROWS=1 + fp8a8=all,bf16=all) | where |
|---|---|---|---|
| fp8 dense matmul (W8A8, W8A16) | `matmul_fp8_gemv_v2` (M 1..4), `matmul_fp8_gemv_rows` (`exact_rows`, 1..16) | `matmul_fp8_wmma` (ordinary, M 1..4, priority 30), `matmul_fp8_wmma_rows` (`exact_rows`, 1..16); same launcher | `quantmoe.rs:427,435,445` (GEMV rejects), `:453-473` |
| fp8 decode folds (M = 1): `matmul_fp8_pair`, `gated_rmsnorm_matmul_fp8`, `rmsnorm_add_matmul_fp8_pair`, `sigmoid_mul_matmul_fp8` | `*_gemv` | `*_wmma` | `:744,759,772,781` (GEMV rejects), `:787-821` |
| bf16 router / gate matmul, pairs and the rmsnorm_add pair | `matmul_bf16_gemv`, `matmul_pair_bf16_gemv`, `rmsnorm_add_matmul_pair_bf16_gemv` | `matmul_bf16_wmma`, `matmul_pair_bf16_wmma`, `rmsnorm_add_matmul_pair_bf16_wmma` (output bf16 only) | `kernels.rs:225,242,249-263`; `quantmoe.rs:737,832,836-849` |
| lm_head (`Logits`, f32 out) | `logits_bf16_gemv(_rows)` | **unchanged** | `kernels.rs:229-235,264` |
| nvfp4 shared expert / glu / down-combine, MoE experts (`moe_block*`), attention, GDN | own row-exact twins | **unchanged** | `quantmoe.rs:748-754,822-828` |

M = 1 is in the family when the pair is set, at the three places where it matters: the ordinary solo registrations take M 1..4, the rows registrations 1..16, and the folds are M = 1 only. Evidence, `intarsia plan show` for the 35B (solo decode kernel per op, `raw/gates/plan-{off,bare,opt1}.txt`): option 1 replaces `matmul_bf16_gemv`, `matmul_fp8_gemv_v2`, `matmul_pair_bf16_gemv`, `matmul_fp8_pair_gemv`, `rmsnorm_add_matmul_pair_bf16_gemv`, `gated_rmsnorm_matmul_fp8_gemv`, `rmsnorm_add_matmul_fp8_pair_gemv`, `sigmoid_mul_matmul_fp8_gemv` by their `_wmma` kernels (calls per decode step in the plan: `matmul_fp8` 10, `matmul_fp8_pair` 30, `matmul_pair_bf16` 1, `rmsnorm_add_matmul_pair_bf16` 69, `gated_rmsnorm_matmul_fp8` 30, `rmsnorm_add_matmul_fp8_pair` 10, `sigmoid_mul_matmul_fp8` 10; `matmul_bf16` has 0 calls in the solo plan) and leaves `matmul_nvfp4_gemv_v2`, `moe_block_bf16`, `attn_decode_bf16_splitkv`, `gdn_*` as they were. `plan show` with ROWS=1 alone differs from main in exactly three lines: the three fp8 folds (`gated_rmsnorm_matmul_fp8`, `rmsnorm_add_matmul_fp8_pair`, `sigmoid_mul_matmul_fp8`), 50 calls per step. So on this model the bare flag is a mixed configuration (some fp8 ops on the family, W8A8 projections on the GEMV); each op is still M-invariant, but the solo bits move for no speed gain (section 3.3). Selection at each M is also pinned by `wmma_fp8_selection_is_pinned` (`quantmoe.rs:882`).

A partial table is the hazard the table docs name (`quantmoe.rs:146-152`): a kind that switches kernels inside 1..16 breaks M-invariance (`wmma_bf16_switch_from_the_gemv_is_not_bitwise`, passes = the switch is not bitwise). The startup rule below must therefore select the whole table, not a flag.

## 2. `rows_exact` with the flag (item 2)

`cargo test -p intarsia-backend-hip --test rows_exact --test wmma_folds --test frag_layout -- --test-threads=1` on card 1, three arms (`raw/gates/R-{off,bare,opt1}.log`, `R.out`): rows_exact 13 passed / 2 ignored (timing), wmma_folds 6 passed, frag_layout 5 passed, in all three arms. Also `quick_coverage` (knob unset) and `quick_coverage_wmma` (option 1): pass.

What this shows: `wmma_fp8_rows_are_bitwise_the_one_row_call` and `wmma_bf16_rows_are_bitwise_the_one_row_call` check that row r of an M-row call equals the family's own M = 1 launch for every M = 1..64 (wider than the 1..16 the registry admits), with nasty neighbours, 35B shapes, W8A8 and W8A16, bf16 and f32 out (`tests/rows_exact.rs:373-432`); the folds are bitwise their unfused chains in `wmma_folds`. What it does not show: that the registry picks these kernels (the tests call kernels by name, which is why all three arms are identical); that is section 1 (plan show) and section 4 (served). The GEMV-twin tests (`fp8_rows_are_bitwise_the_one_row_gemv`, nvfp4, MoE, bf16 logits) also pass, so the kernels the flag leaves alone keep their own invariance.

## 3. Served A/B (item 3)

Method. `scripts/wmr/window.sh`, derived from `scripts/r9t/window.sh`: one fleetq job holds both cards and takes radiance down (`--quiet gpu,cpu`), a guard on tatooine restores radiance when the script ends or dies, fresh `serve --backend hip --spec off --profile-every 0 --mem-budget 29 --max-seq 18432` container per (arm, round), localeval `speed` (thinking off, 1024-token prompt, gen 512, concurrency 1,4,8,16, one rep), arms interleaved with the order reversed on even rounds. Same binary for both arms; the arm is the env pair above. Round 3 of TP=1 ran in a second window (`raw/winC`). Rounds are listed in order; the bracket is the range of the per-round paired deltas. Raw: `raw/winA` (TP=1 rounds 1-2), `raw/winB` (TP=2 rounds 1-3), `raw/winC` (TP=1 round 3 and the single-stream phase), `raw/winE` (TP=2 single stream), `raw/winF` (bare flag). Tables are generated by `scripts/wmr/summarize.py`.

### 3.1 Aggregate decode, `--max-batch 16` (tok/s; includes the prefill of the call)

| config | flag off (rounds; mean) | flag on (rounds; mean) | on vs off |
|---|---|---|---|
| TP=1, 1 stream | 157.7 / 157.4 / 158.1; **157.7** | 149.4 / 152.8 / 149.4; **150.5** | -4.6 % [-5.5..-2.9] |
| TP=1, 4 streams | 295.2 / 290.3 / 288.3; **291.3** | 296.8 / 297.9 / 296.6; **297.1** | +2.0 % [+0.5..+2.9] |
| TP=1, 8 streams | 355.6 / 357.0 / 355.8; **356.1** | 406.9 / 409.4 / 409.5; **408.6** | +14.7 % [+14.4..+15.1] |
| TP=1, 16 streams | 372.4 / 370.5 / 369.8; **370.9** | 499.8 / 499.2 / 498.2; **499.1** | +34.6 % [+34.2..+34.7] |
| TP=2, 1 stream | 184.2 / 186.2 / 186.4; **185.6** | 180.5 / 182.1 / 182.0; **181.5** | -2.2 % [-2.4..-2.0] |
| TP=2, 4 streams | 356.3 / 360.4 / 357.4; **358.0** | 366.3 / 369.3 / 364.8; **366.8** | +2.4 % [+2.1..+2.8] |
| TP=2, 8 streams | 485.5 / 482.4 / 482.8; **483.6** | 537.8 / 532.0 / 533.4; **534.4** | +10.5 % [+10.3..+10.8] |
| TP=2, 16 streams | 513.5 / 511.2 / 513.3; **512.7** | 664.7 / 665.3 / 669.4; **666.5** | +30.0 % [+29.4..+30.4] |

The TP=1 gains replicate lane wg's direction (+14 % at 8 and +29 % at 16 streams on the 10-25 tree, where flag off was 335 and flag on 432 tok/s at 16; main has since moved both up, 371 and 499, and the gain at 16 is +34.6 %). The paired deltas are tight (the bracket is under 1 point at 8 and 16 streams); the 4-stream gain (+2 %) is about the size of the spread of the off arm (288..295), so treat it as flat.

### 3.2 Single stream, `--max-batch 1` (what every process would pay if the family were chosen for all of them)

Decode tok/s:

| config | flag off | flag on | on vs off |
|---|---|---|---|
| TP=1, 1k prompt | 164.5 / 164.6 / 164.4; **164.5** | 155.9 / 158.6 / 158.3; **157.6** | -4.2 % [-5.2..-3.6] |
| TP=1, 4k | 161.3 / 160.6 / 160.5; **160.8** | 155.4 / 155.0 / 154.7; **155.0** | -3.6 % [-3.7..-3.5] |
| TP=1, 16k | 145.7 / 146.2 / 145.3; **145.7** | 141.0 / 142.8 / 140.5; **141.4** | -3.0 % [-3.3..-2.3] |
| TP=2, 1k | 192.5 / 192.6 / 192.0; **192.4** | 185.2 / 187.9 / 187.8; **187.0** | -2.8 % [-3.8..-2.2] |
| TP=2, 4k | 190.4 / 190.7 / 190.6; **190.6** | 183.4 / 186.4 / 186.5; **185.4** | -2.7 % [-3.7..-2.2] |
| TP=2, 16k | 176.7 / 181.2 / 183.9; **180.6** | 177.4 / 178.3 / 177.4; **177.7** | -1.6 % [-3.5..+0.4] |

Prefill tok/s (the family only runs at 16 rows or fewer, so only prompt tails can change): TP=1 7196 -> 7432 (+3.3 %), 10178 -> 10363 (+1.8 %), 9397 -> 9335 (-0.7 %) at 1k / 4k / 16k; TP=2 8957 -> 8880 (-0.9 %), 12499 -> 12587 (+0.7 %), 11670 -> 11746 (+0.7 %). All within the round-to-round spread; TTFT likewise (`raw/winC`, `raw/winE`). A 990-token prompt is a single 990-row prefill chunk that the family (16 rows or fewer) cannot touch, and localeval itself marks TTFT under 1024 uncached tokens as per-request overhead, so the +3.3 % at 1k is overhead noise, not a gain.

The decode cost is the M = 1 W8A8 kernel being slower than the scalar GEMV (lane wg measured +0.25 ms per step at B = 1 and chased it without a fix; the fragment-order weight layout `INTARSIA_HIP_FP8_FRAG=1` did not move step time).

### 3.3 The bare flag (`INTARSIA_HIP_WMMA_ROWS=1` alone), TP=1, `--max-batch 16`, 3 rounds interleaved with off (`raw/winF`)

| streams | off | bare | bare vs off |
|---|---|---|---|
| 1 | 158.2 / 157.6 / 157.7; **157.8** | 154.1 / 151.2 / 153.2; **152.8** | -3.2 % |
| 4 | 289.3 / 292.1 / 290.0; **290.5** | 288.0 / 290.3 / 291.9; **290.1** | -0.1 % |
| 8 | 354.8 / 354.5 / 354.0; **354.4** | 354.5 / 354.3 / 355.1; **354.6** | +0.1 % |
| 16 | 369.2 / 371.8 / 374.3; **371.8** | 368.6 / 368.9 / 370.0; **369.2** | -0.7 % |

No gain and a single-stream loss: the bare flag is the wrong thing to enable. The user's brief and the wg doc both say "the flag"; the registry needs the table too.

## 4. Quality (item 4)

### 4.1 Self-golden KL / top-1, TP=1, flag on vs main

Gate: ADR-0007 section 6 / `docs/accuracy.md` "Same-weights regression gate" and "Change against base". The knob-off build writes its own dump (`--write-self-golden`, top-64 log-probs of prefill and of the teacher-forced decode window), the knob-on build is compared with `--self-golden ... --self-vs-ref-noise` (mean 21-way KL must be <= factor x the golden's vLLM run-to-run noise, every prompt <= 3 x its own noise; LOG.md 2026-10-21 11:23, 13:23, 13:58), and again with the default same-numerics budgets (mean KL 1e-4, max position KL 5e-3, flips 0.5 %, tail 5e-4). `--mode both`: prefill is the M > 16 GEMM and cannot move, the decode window is what changes. Goldens: `qwen36-35b-a3b-nvfp4-vllm-top20` (10 prompts, 16 decode positions each) and `...-vllm-long-top20` (5 prompts up to 12k, 64 decode positions each). Flags `--allow-same-code --allow-unstamped --pre-change d2fe64a8a --revlist` because both arms are the same code tree and differ only in env; the tool reports the kernels that differ per op (`differences from the self-golden's run`: all the `_gemv` -> `_wmma` swaps of section 1), so the arm really ran the family. Raw: `raw/acc/*.log`.

| golden, mode | n positions | mean KL21 vs main | vLLM run-to-run noise (all rows) | ratio | max position KL | top-1 flips | tail diff | `--self-vs-ref-noise` | default budgets |
|---|---|---|---|---|---|---|---|---|---|
| short, prefill | 1441 | 2.57e-12 | 3.161e-3 | 0.000 | 2.6e-9 | 0 | 3.8e-9 | PASS | PASS |
| short, decode | 160 | **2.697e-3** | 3.161e-3 | **0.853** | 6.53e-2 | 1 (0.63 %) | 9.67e-4 | **PASS** (tightest prompt `story`: 0.61 of its limit; worst prompt KL 8.0e-3) | FAIL (mean KL, max position KL, tail, worst prompt; flips 1 of limit 1) |
| long, prefill | 36887 | 2.68e-13 | 2.218e-2 | 0.000 | 2.7e-10 | 0 | 1.0e-8 | PASS | PASS |
| long, decode | 320 | **3.887e-3** | 2.218e-2 | **0.175** | 1.18e-1 | 9 (2.8 %) | 9.56e-4 | **PASS** (tightest prompt `long_docs_12k`: 0.23 of its limit) | FAIL (mean KL, max position KL, flips 9 of limit 1, tail, worst prompt) |
| short, decode, flag-off control against its own dump | 160 | 8.3e-12 | 3.161e-3 | 0.000 | 1.2e-9 | 0 | 6.1e-9 | PASS | PASS |

Reading. The change is a reordering of fp8/bf16 sums in decode, the same size class as the other accepted HIP kernel reorderings (the default budgets "fail any real kernel change", `docs/accuracy.md` 396-411): short-golden decode is 0.85 of vLLM's own run-to-run KL, which is close to the limit of the ref-noise rule at factor 1, long is well inside. The control shows the harness resolves 1e-11, so the 2.7e-3 is the family and not noise of the run. The flips are teacher-forced argmax flips at near-ties (1 of 160, 9 of 320). The earlier wg gate (tracked HIP baseline and envelope, floor verdicts) passed on the 10-25 tree; it was not repeated here.

Same gate at TP=2 (both cards, `--tp 2 --devices 0,1`; main's TP=2 dump against the flag-on TP=2 run; raw `raw/accT/*.log`, job #7370). Note that ADR-0006 / LOG 2026-10-21 make the TP=2-against-TP=1 self-golden report-only; this is TP=2 against TP=2, which is a legitimate change-against-base comparison:

| golden, mode | n positions | mean KL21 vs main | vLLM noise (all rows) | ratio | max position KL | top-1 flips | `--self-vs-ref-noise` | default budgets |
|---|---|---|---|---|---|---|---|---|
| short, prefill | 1441 | 2.26e-13 | 3.161e-3 | 0.000 | 1.8e-10 | 0 | PASS | PASS |
| short, decode | 160 | **2.811e-3** | 3.161e-3 | **0.889** | 1.23e-1 | 2 (1.25 %) | **PASS** (tightest prompt `chat_multiturn`: 0.79 of its limit) | FAIL |
| long, prefill | 36887 | 2.55e-13 | 2.218e-2 | 0.000 | 2.7e-10 | 0 | PASS | PASS |
| long, decode | 320 | **3.507e-3** | 2.218e-2 | **0.158** | 6.96e-2 | 8 (2.5 %) | **PASS** (tightest `long_docs_12k`: 0.20) | FAIL |
| controls (flag off against its own TP=2 dump), short / long decode | 160 / 320 | 1.6e-12 / 2.8e-12 | | 0.000 | | 0 / 0 | PASS | PASS |

The short golden's decode is the tight one in both topologies (0.85 at TP=1, 0.89 at TP=2 of the noise limit at factor 1; the tightest single prompt 0.61 and 0.79 of its own limit). It passes, but with little room; if the user wants a margin larger than 11-15 %, the short golden is where it is decided.

### 4.2 Cross-stream greedy identity, served, flag on

`scripts/wmr/ident.py` against the running server (`--max-batch 16`, temperature 0, thinking off, 160 tokens, `/v1/chat/completions`): 16 distinct prompts (half carry padding so the rows sit at different lengths and positions); each prompt first alone (1 stream, sequential), then a second solo pass, then 4, 8 and 16 at a time (concurrent requests join one batched decode step; the server log shows `continuous batching: up to 16 sequences`; the per-step batch size was not logged, so that the 16 requests really shared steps is inferred from the 16-stream throughput of 3.1, not observed here). Compared by completion text.

| server | solo repeat | 4 streams | 8 streams | 16 streams |
|---|---|---|---|---|
| TP=1 flag on (`raw/winA/ident-g512-tp1-on-r1.json`) | 16/16 | 16/16 | 16/16 | 16/16 |
| TP=2 flag on (`raw/winB/ident-g512-tp2-on-r1.json`) | 16/16 | 16/16 | 16/16 | 16/16 |
| TP=1 flag off (control) | 16/16 | 16/16 | 16/16 | 16/16 |
| TP=2 flag off (control) | 16/16 | 16/16 | 16/16 | 16/16 |

Flag on against flag off at one stream: all 16 prompts differ (first differing character median 237 at TP=1, 189 at TP=2, min 42 / 27): this is the solo change the contract wording has to name. It is a greedy chain, so one flipped near-tie changes everything after it; quality is judged in 4.1, not here. The rows are the same at every stream count, which is the property the new contract promises.

In-process bitwise check on the real 35B (`batch_cuda_real`, card 0, `raw/gatesB`): `real_35b_batched_decode_is_bitwise_the_solo_step` and `real_35b_batched_engine_equals_the_local_engine` pass with option 1 on and with it off (209 s / 215 s). The two Qwen3-8B tests skip (no `INTARSIA_QWEN3_DIR`; HIP does not batch dense models). The two long-prompt tests are `#[ignore]`d and were not run (lane wg ran them on the 10-25 tree).

## 5. Draft contract change and the code to select the family at startup (item 5)

Text only; not applied to any ADR or comment.

### 5.1 Current wording (what blocks it)

`crates/intarsia-core/src/model/batch.rs:4-5`: "Exactness is the contract: the output of row `r` is bitwise what a solo decode step of sequence `r` computes (`Model::forward` with one token), whatever the other rows are." ADR-0010 section 3 Decision: "Batching is exact only [operator, Q1]. Row r of a batched step is bitwise what the solo 1-row step computes for that row, independent of B, row order and the other rows." The registry says the same for verify calls (`registry.rs:356-358`). The family keeps this inside a process (4.2) but changes the solo step against the scalar GEMV, so a reading of "the solo step" as "the GEMV step main shipped before" forbids it. Q1 is an operator decision, which is why this is not an agent call.

### 5.2 Proposed replacement (from the Fable review, tightened)

> **Row-exact contract (batched serving), revised.** Within one server process, every row of every decode step, solo or batched, is computed by one kernel family for each op, chosen once at startup and never changed while the process runs; the family does not depend on the number of rows M (1..=16), on the other rows, or on the row's position in the batch. Row r of a batched step is bitwise what the same process computes for that sequence alone. The family is chosen at startup: the scalar GEMV family when `--max-batch` is 1 (or batching is off), the WMMA row family when `--max-batch` > 1 on a backend that has one. `tests/rows_exact.rs` proves the within-family invariance for every kernel the family registers, over M = 1..=16 (the family itself up to 64) with adversarial neighbours; a selection test proves that at every M in 1..=16 every op of the decode plan resolves to the chosen family and to no other (a table that switches kernels inside 1..=16 is a build error, not an option). Output is not promised across families: the same request on a `--max-batch 1` server and on a `--max-batch > 1` server may differ once, at the first near-tie, by the reordering of fp8/bf16 sums. That one-time difference is bounded by the repo's same-weights gate (ADR-0007 section 6) in its change-against-base form: mean 21-way KL of decode against the other family's own dump at most the vLLM run-to-run noise of the same golden, every prompt at most 3x its own noise, on the short and the long 35B goldens, at TP=1 and TP=2, in `--mode both`. Offline commands (`generate`, `accuracy`, `qualify`, `kernels test`) use the scalar family unless they are asked for the other; goldens, baselines and self-golden dumps recorded for the batched family carry its name.

Open points the user may want to change: whether to name the one-time difference's bound as "0.85x / 0.18x noise" measured here or keep the generic rule; whether `--max-batch > 1` alone is the trigger or a separate flag (a lone-request user on a batched server pays 2-5 %, section 3.1); whether the family should ever be chosen for `--max-batch 1` (no: -2..-4 % for nothing).

### 5.3 Code change it takes (not implemented: `quantmoe.rs` is HGV's file tonight)

Today the only way in is the two env vars (section 1), set by the operator; that already works for a batched server and is what every measurement above used. To select it from `--max-batch`:

1. `crates/intarsia-backend-hip/src/quantmoe.rs`: add an all-kinds table (`WmmaTable::ALL`, the value `parse("fp8a8=all,bf16=all")` returns) and a process-wide selector that `wmma_rows_enabled()` and `wmma_table()` consult first: `set_row_family(RowFamily::{Gemv, Wmma})`, backed by a `OnceLock` set at most once and rejecting a second call with a different value. Precedence: `INTARSIA_HIP_WMMA_ROWS=0` forces the GEMV family, `=1` with `INTARSIA_HIP_WMMA_TABLE` keeps today's meaning (A/B and kernel-test use), unset means "whatever `set_row_family` said, else GEMV". Today's `OnceLock`s are read at the first registry call (`kernels::all()`), so the selector has to run before the backend is built.
2. `crates/intarsia-cli/src/cmd/serve.rs` (around `:97-105`) / `common.rs` `load()` (`:420`): the effective max batch is `args.max_batch` or the recipe default (`loaded.max_batch`, `serve.rs:114`), known inside `load()` before the backend is built. There: `if backend == hip && arch is gfx12 && max_batch > 1 { set_row_family(Wmma) }`. It must run before the first `kernels::all()` (the `OnceLock`s are read once).
3. Core: expose the chosen family in the plan JSON / `plan show` header, the startup log line and the `/v1/models` info, next to `weight_layout`, so a golden or a bench can tell which family it ran (this is also how `qualify` knows which dump to compare with).
4. Tests: (a) a selection test that for M in 1..=16 and every decode op the resolved kernel is in the chosen family (generalise `wmma_fp8_selection_is_pinned`, `quantmoe.rs:882`); (b) `rows_exact` as is; (c) the served identity script of this lane as a gate for the batched family; (d) `acc-oracle` dumps `golden/baselines/qwen36-35b-a3b-nvfp4-cpu-oracle.hip.selfgolden.safetensors` (and `hip-tp2`) re-recorded for the batched family under the policy of LOG.md 2026-10-22 ("a change meant to move the numerics re-records the dumps of the backends it touches"); the scalar-family dumps stay for `--max-batch 1`.
5. Docs: `docs/tuning.md`, `docs/backends/hip.md`, ADR-0010 note, LOG.md row; `batch.rs:4-5` and `registry.rs` comments reworded as 5.2.

Not covered by the wording and worth deciding with it: speculative decode (`--spec mtp` is refused with `--max-batch`; verify passes use the same `exact_rows` path, and lane wg's MTP identity gate ran with option 1 on the 10-25 tree, not re-run here); the Metal and CUDA backends (they have no such family; the rule is per backend); TP=2 peers (rank 1 runs the same registry in the same process; the TP=2 identity result in 4.2 is the evidence).

## 6. Limits, not done, run record

* `intarsia qualify` was not run. The ADR-0007 change-against-base gate was run directly (`accuracy --self-golden --self-vs-ref-noise`, 4.1). `qualify`'s acc-oracle drift stage compares with the pinned `golden/baselines/qwen36-35b-a3b-nvfp4-cpu-oracle.hip(-tp2).selfgolden.safetensors` at the 1e-4 budget; by the 2026-10-22 policy it would fail for the batched family until those dumps are re-recorded for it.
* Not run: the two `#[ignore]`d long-prompt batched tests; MTP identity; Qwen3-8B / 27B; tracked-baseline/envelope gate for the flag-on build (wg's PASS on the 10-25 tree stands); the 4k vLLM golden (`...-vllm-4k-top20`, Metal baselines only).
* One tree, one day: the three rounds of each arm span 1-2 windows (rounds 1-2 and 3 of TP=1 were different windows); the off arms of windows A, C and F (157.7 / 157.7 / 157.8 at 1 stream, 370.9 / 371.8 at 16) agree to 0.3 %, so window-to-window drift is small.
* The TP=1 single-stream rows come from the `--max-batch 1` phase (window C), the "1 stream" rows of 3.1 from a `--max-batch 16` server; they differ by the graph/step overhead of the batched engine and are not interchangeable.
* Failed first attempts (kept in raw for the record): job #7349 exited in 3 s because `timeout` does not exist on the Mac (its folder was replaced by the rerun; script fixed; radiance came back in 15 s); `raw/acc-try1-dumpdir-denied` (dump path not writable in the container); `raw/acc-try2-prechange-needed` (the self-golden gate refuses a same-head dump without `--pre-change`/`--revlist`). in the rerun #7355 (`raw/gates`: groups R and P) the B group exited 125 (docker container-name clash with P on card 0) and was rerun alone as #7366 (`raw/gatesB`); the stale R text at the end of `raw/gatesB/gates.log` is a leftover file on tatooine, not part of that run.
* fleetq jobs (owner `claude-908e11e8`, all created and released by this lane): build #7341; gates #7349 (aborted, see above), #7355, #7366; accuracy #7350 (invalid, dump dir), #7358 (invalid, no `--pre-change`; its off-write dumps are the ones #7367 and #7370 compare against for TP=1, and #7370 wrote its own for TP=2), #7367 (TP=1), #7370 (TP=2); A/B windows #7351 (TP=1 rounds 1-2 + identity), #7360 (TP=2 rounds 1-3 + identity), #7361 (TP=1 round 3 + single stream), #7359 (TP=2 single stream), #7365 (bare flag). Every window used the guard script: radiance was down 02:15-02:21 (A), 02:33-02:36 (E), 02:36-02:43 (B), 02:44-02:49 (C), 02:58-03:05 (F) plus the gates and accuracy windows (`downtime.log` in each raw folder) and was restored each time within 15 s (`radiance answers ... unit active`).
