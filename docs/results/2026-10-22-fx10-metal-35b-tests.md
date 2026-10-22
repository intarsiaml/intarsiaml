# FX10: Metal 35B op tests and reuse-mode accuracy (Fable finding 10), 2026-10-22

Lane `fx10`, branch `lane-fx10/metal-35b-tests`, PR #92 (test-only, no kernel or engine change). Machine: endor (Apple M5 Max). Raw logs:
`2026-10-22-fx10-metal-35b-tests/raw/`.

## 1. What finding 10 named, and what was already covered

Fable C.4 items 1-3 for PR #70's kernels. Commit `8e7aa554` (merged before this lane) had already added skewed multi-tile MoE routing, odd scale offsets
with f32 output, `moe_topk`/`moe_combine` against the reference, and conv K in {2, 4, 8} / delta-rule dk in {64, 128}. Lane m35's `moe_mm_*`
replacement owns `kernels/metal/moe.metal`, `src/moe.rs` and `tests/moe.rs`; this lane did not touch them. What was still uncovered:

| op | gap before | new test (all in `crates/intarsia-backend-metal/tests/`) |
|---|---|---|
| `GateSplit` (`gate_split_bf16`) | no cargo test | `gdn.rs::gate_split_is_an_exact_per_head_split`: T in {1, 3, 300}, 16 heads, head_dim 256 (q_proj [T,16,512] = q or gate per head), bitwise vs `gdn::gate_split`, operands at a 2-byte offset |
| `SigmoidMul` (`sigmoid_mul_bf16`) | no cargo test | `gdn.rs::sigmoid_mul_matches_the_reference`: T in {1, 3, 300} x 4096, g spread +-12 (saturates both ends), within 2^-8 relative of f64 |
| `GdnConv` | only C = 300 | `gdn.rs::conv_at_the_35b_channel_count`: C = 8192, K = 4, T in {300 split 1+127+172, 1, 65 split 33+32}; chunked == one call bitwise, vs f64 reference |
| `GdnDeltaRule` | only 2 key / 4 value heads, dk 64/128 | `gdn.rs::delta_rule_at_the_35b_layer_shape`: 16 key / 32 value heads, dk = dv = 128, T = 300 (1+127+172), 1, 1024; the existing dk loops now cover 32, 64, 128, 256 |
| `MatmulFp8` | N <= 4096, K <= 4096 at M <= 200 | `quant.rs::fp8_at_the_35b_projection_shapes`: (M, N, K) = decode 1 / 8, 33, 300 over N in {8192, 4096, 2048, 512}, K in {2048, 4096, 512}; GEMV (M <= 8) and MMA kernels, bf16 and f32 out, weights at 8 mod 16 |
| `MatmulNvfp4` | same | `quant.rs::nvfp4_at_the_35b_projection_shapes`: same shapes, both kernels, both output dtypes |

Shapes are from `Qwen3.6-35B-A3B-NVFP4/config.json` (hidden 2048, 16 heads x head_dim 256, attn output gate, 16 linear key heads / 32 value heads,
dk = dv = 128, conv kernel 4, conv_dim 8192). Not done: C.4 item 4 (a `moe_topk` `top_k <= experts` predicate) was already in `8e7aa554`; item 5
(batch invariance across the `T*K <= 16` split) is a documentation item, not a test, and sits in moe, which this lane does not touch.

Checks (fleetq #3984, non-exclusive, `--mem 12`): `cargo test --release -p intarsia-backend-metal` whole crate exit 0 (`raw/cargo-test-metal-crate.log`);
the two touched test binaries alone: gdn 7 passed, quant 6 passed (`raw/cargo-test-new.log`). The first run of the fp8 shapes test failed because the
GEMV kernel serves M <= 8 only (`GEMV_MB`); the test now skips GEMV above that, as the production predicate does. No kernel bug was found.

## 2. Mutation check

Each mutation edits one kernel line in a scratch worktree (`/opt/code/wt-fx10-mut`, same commit), rebuilds, runs the named test and expects a failure,
then restores the line (`scripts/fx10/mutate.py`; logs in `raw/mutation/`). All 9 were killed (the test binary reported `test result: FAILED`).

| mutation | file | test run | result |
|---|---|---|---|
| gate half read from the q half | gdn.metal | `gate_split_is_an_exact_per_head_split` | killed |
| sigmoid_mul 1 % gain | gdn.metal | `sigmoid_mul_matches_the_reference` | killed |
| sigmoid_mul g clamped to +-6 (saturation) | gdn.metal | `sigmoid_mul_matches_the_reference` | killed |
| delta rule scale hard-coded to 1/sqrt(128) | gdn.metal | `delta_rule_chunked_prefill_equals_one_call_and_the_reference` (dk 32/64/256 now in the loop) | killed |
| delta rule key-head map `h % hk` instead of `h / (hv/hk)` | gdn.metal | `delta_rule_at_the_35b_layer_shape` | killed |
| conv input read offset by one when C > 4096 (wide-C fault; artificial) | gdn.metal | `conv_at_the_35b_channel_count` | killed |
| conv tile-boundary carry off by one | gdn.metal | `conv_at_the_35b_channel_count` | killed |
| NVFP4 GEMV global scale x1.02 | nvfp4.metal | `nvfp4_at_the_35b_projection_shapes` | killed |
| FP8 MMA drops the last K tile | fp8.metal | `fp8_at_the_35b_projection_shapes` | killed |

Several of these would also be caught by older tests; the point is that each new test has teeth on its own.

## 3. Reuse-mode accuracy on the Metal 35B

`intarsia accuracy --model ~/models/Qwen3.6-35B-A3B-NVFP4 --backend metal --mode reuse --mem-budget 30 --prefill-chunk {1024,2048}`, exclusive slot
fleetq #3999 (first attempt #3985 failed in 1 s: the CLI was built without `--features metal`; fixed in `scripts/fx10/reuse.sh`). Binary: this branch
(origin/main `06fba3a9` plus tests and scripts, no source change); built-in stream tolerance 0.25 nats / KL 1e-2; no tolerance changed. The
`--reuse-require-all` / `--reuse-measure` flags of FX5's PR are not on main, so the counts below are read from the per-prompt table and the
"skipped" line.

There is no separate 35B "4k" golden: the 35B goldens are `qwen36-35b-a3b-nvfp4-vllm-long-top20` (five prompts of 4k-12k tokens, max-seq 16384) and
`...-vllm-top20` (ten prompts of at most 700 tokens, max-seq 4096). Both were run.

| golden | chunk | judged | skipped | prefix-reuse verdict | failing case |
|---|---|---|---|---|---|
| long | 1024 | 5 | 0 | **FAIL** | `long_docs_4k` continue: streams diverge at token 12, shared rows differ by 2.108 nats (0.235 at the divergence), KL up to 8.1e-2 (tolerance 0.25 / 1e-2); warm margin 0.132, cold 0.079 |
| long | 2048 | 5 | 0 | **FAIL** | `long_docs_4k` continue: diverges at token 12, 2.704 nats (0.750 here), KL up to 8.2e-2; warm margin 0.018, cold 0.231 |
| short (max 700 tokens) | 1024 | 0 | 10 | **FAIL** (nothing ran) | no prompt crosses a chunk boundary, so no restore was tested |
| short (max 700 tokens) | 2048 | 0 | 10 | **FAIL** (nothing ran) | same |

For the long golden every other case passes: all `restore` rows are 32/32, and every `aligned` row (reused length a multiple of the chunk) has logits
bit-identical to the cold run. `long_code_4k`, `chat_reuse_4k`, `long_docs_12k` and `long_code_12k` pass at both chunks including `continue`. So the GDN
snapshot/restore path on Metal, never run before, restores state exactly. The one failure is the same prompt and the same divergence token (12) as the
CUDA run in `2026-10-22-fx5-hybrid-reuse-ci.md` (35B on spark1, `long_docs_4k` continue, chunk 1024: 0.877 nats / KL 3.1e-2; chunk 2048:
2.138 / 9.42e-2). It is the decode-versus-prefill rounding near-tie that Fable B.2 / C.4 item 5 predicted (warm margin 0.018-0.132 nats), not a Metal kernel
fault: the cold-vs-warm gap appears only in the `continue` case, where reply tokens produced by decode kernels enter the cache and the cold run
prefills them. Whether the tolerance should absorb it is the open decision in PR #89; it is untouched here.

Short-golden note: with the Metal default chunk (256) the 300- and 700-token prompts of the short golden do cross a boundary; that run was not made
(the task named 1024 and 2048). Verdict-gate consequence: a Metal reuse gate on the short golden at the defaults needs `--prefill-chunk 256` to judge anything.

After merging origin/main (which brought lane m35's `moe_mm_*` kernels) the gdn and quant binaries were re-run: 7 and 6 passed (#4010, `raw/cargo-test-after-merge.log`).

Fleet jobs: #3978 (first build and test), #3984 (crate tests and mutation), #3985 (failed reuse launch), #3999 (reuse runs), #4010 (re-run after merge).
