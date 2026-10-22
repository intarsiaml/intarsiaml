# llama3 / yarn rope scaling and q/k/v bias: Llama-3.1-8B-Instruct and Qwen2.5-7B-Instruct (2026-10-22, lane fx4)

**Outcome.** Both models load and generate correctly on Metal from built-in recipes (`llama`, `qwen2`), with no core edit beyond the
two features. Qwen3-8B on Metal is unchanged (plan, 2 x 256 greedy ids, both accuracy goldens, byte for byte). **Accuracy is not a clean
PASS**: the KL gate passes for both models, the top-1 floor (99.5%) fails for both. The fp32 golden's own bf16 forward fails
it by the same margin (below), so this is a gate-policy question for a human, not a bug I can see. CUDA and HIP kernels are written but
were never compiled or run.

## What was built

| Piece | Where |
|---|---|
| `rope_type` llama3 and yarn: frequency table (transformers formulas, f64 then f32) | `crates/intarsia-core/src/rope.rs` |
| ops `RopeScaled` (x, `inv_freq` table; cos/sin times `rope_mscale`) and `AddBias`, appended to `Op::ALL` | `crates/intarsia-core/src/ops.rs` |
| model: table uploaded once at load; scaled models call `RopeScaled`, unscaled models emit the same calls as before; q/k/v bias after each projection; fused qk-norm+rope prologue bails when scaled | `crates/intarsia-core/src/model.rs` |
| recipe: `[layer.attn] q_bias / k_bias / v_bias`; config `rope_parameters` / `rope_scaling` read into `ModelDims.rope_scaling`; unsupported `rope_type`s and an `o_proj` bias still rejected by name; a checkpoint bias the recipe does not map is a load error | `recipe.rs`, `model.rs` |
| CPU reference + fast kernels | `crates/intarsia-backend-cpu/src/{reference,fast}.rs` |
| Metal kernels `rope_scaled_bf16`, `add_bias_bf16` | `kernels/metal/dense_ops.metal`, `crates/intarsia-backend-metal/src/{kernels,dense}.rs` |
| CUDA / HIP kernels (**not compiled, not run**) | `kernels/{cuda,hip}/elementwise.*`, `kernels/abi/ik_abi.h`, `crates/intarsia-backend-{cuda,hip}` |
| recipes `llama`, `qwen2`; `recipe new` scaffolds biases and llama3/yarn; `llama3` turn format for the KL floor | `recipes/`, `crates/intarsia-cli/src/cmd/{recipe.rs,accuracy/turns.rs}` |
| `generate`: plain prompts tokenise with the tokenizer's special tokens (Llama BOS); `golden.py`: no second BOS on a rendered chat template | `generate.rs`, `tools/golden.py` |

Design choice: a new op that takes a precomputed table, not new attrs on `Rope`, so existing models' OpCalls (and plans) are
unchanged by construction. `AddBias` is a separate op after the projection (one extra bf16 rounding versus HF's fused bias; see
"Open").

## Tests

* CPU reference, hand-computed against transformers' values: `rope::tests::{llama3_head16, llama3_head128_llama31, yarn_head16, yarn_head128_qwen_style}` (values from `ROPE_INIT_FUNCTIONS` of transformers 5.19; the head-16 cases carry the hand derivation in comments), `ops.rs` `rope_scaled_llama3_known_angles` (hand-computed angles, mscale), `rope_scaled_with_the_default_table_is_rope`, `add_bias_broadcasts_over_rows`.
* End to end on the CPU backend against a naive f64 decoder: `crates/intarsia-models/tests/rope_bias.rs` (llama3 without qk-norm, Qwen2-shaped biases, bias+llama3, bias+yarn, qk-norm model with scaling, decode equals prefill, tests are not vacuous, unmapped bias is a load error).
* Metal against an f64 reference: `rope_scaled_matches_f64_reference` (llama3 and yarn tables, partial rotary, odd byte offsets, host and device position, positions up to 70000), `add_bias_matches_reference_in_place_and_at_odd_offsets`.
* Admission sweeps for both ops (`admission.rs`): `intarsia kernels test --backend metal` ADMITTED, rope_scaled 36 cases (max abs 1.7e-3), add_bias 12 cases.
* Chat templates against transformers (`tests/chat_templates_real.rs`, needs `INTARSIA_LLAMA31_DIR` / `INTARSIA_QWEN25_DIR`): both byte-equal.
* Full `cargo test --workspace` (release, `--features intarsia-cli/metal`, excl. CUDA/HIP crates, on endor): 662 tests, the only failure was the Metal pinned kernel list (fixed: it gains exactly `add_bias_bf16` and `rope_scaled_bf16`, `cargo test -p intarsia-backend-metal --lib` 12/12 after).

## No-op on Qwen3-8B, Metal (main f9087322 vs this branch, one exclusive endor slot, `raw/2026-10-22-fx4/noop.sh`)

| Check | Result |
|---|---|
| `plan show --json` and text | byte-identical (`raw/.../plan-base.txt` = `plan-new.txt`) |
| greedy ids, 256 tokens, prompt-short (122 tok) and prompt-4k (4062 tok) | identical id lists |
| `accuracy` vs `qwen3-8b-fp32` and `qwen3-8b-long-fp32` (bf16 class, timing lines removed) | identical output |
| `kernels test` | same results for the 28 old kernels; two new rows (rope_scaled, add_bias) |

Decode speed was not re-measured: the unscaled path emits the same OpCalls, so a change would show in the plan.

## Onboard on Metal (M5 Max, `intarsia onboard --quick-only --golden-from <HF fp32 golden>`)

Weights: Qwen2.5-7B-Instruct from `Qwen/Qwen2.5-7B-Instruct`; Llama from `unsloth/Llama-3.1-8B-Instruct` because `meta-llama/` is gated and the
token on endor has no access (decision in `docs/decisions/LOG.md`). Both in the HF cache, linked under `/Users/josef/models/`. Disk was 1.0 TiB free.
Goldens: `tools/golden.py --dtype float32` on endor CPU (HF transformers 5.19), short plus long set. Llama uses `--template hf` (HF chat template).

| | Llama-3.1-8B-Instruct | Qwen2.5-7B-Instruct |
|---|---|---|
| recipe / smoke / plan / kernels | PASS (builtin:llama, 21/21 templates) | PASS (builtin:qwen2, 21/21) |
| quick gate (perplexity, sanity) | PASS (ppl 1.28, 5/5 sanity) | **FAIL**: sanity "max 4-gram repeat 4" on "The three primary colors are red, yellow, and"; the text is correct ("If you mix red and yellow, you get orange...") and coherent, perplexity 1.38. A false positive of the repeat heuristic |
| acc-short prefill top-1 / full-vocab KL (all rows) | **99.09%** (floor 99.5) / 5.3e-4 | **99.30%** / 3.2e-3 |
| acc-short prefill KL, scored rows, vs threshold | 9.8e-5 vs 1.0e-3 PASS | 1.41e-3 vs 1.36e-3 (ref-noise-bound) PASS |
| acc-short decode top-1 (teacher-forced) | **99.38%** | **98.12%** |
| decode tok/s at 128 / 1024 / 4096 (% of byte floor) | 23.2 / 23.9 / 22.4 (57 / 58 / 55%) | 31.7 / 32.2 / 27.6 (73 / 74 / 64%) |
| prefill tok/s at 128 / 1024 / 4096 | 1081 / 1306 / 1533 | 1494 / 1014 / 553 |
| onboard wall time | 5m17s (plan load 59s, kernels 80s) | 5m17s (plan load 79s, kernels 69s, bench 2m08s) |

The PASS in the KL rows needs the `ref_noise` sidecars committed here (`golden/ref_noise/*-fp32.safetensors.json`, from
`intarsia ref-noise --method precision_spread` between the fp32 and bf16 HF goldens). Without the Qwen2.5 sidecar its prefill KL
(1.405e-3) is over the 1e-3 static floor; the second accuracy run (raw `acc-*.out`) used the sidecars.

**Why top-1 fails, and why I did not touch the gate.** The reference's own bf16 forward has the same problem. HF transformers bf16 vs HF fp32 on the same ids:

| | HF bf16 vs HF fp32 top-1 | engine (Metal bf16) vs HF fp32 top-1 | HF bf16 vs fp32 full-vocab KL | engine full-vocab KL |
|---|---|---|---|---|
| Llama-3.1-8B-Instruct | 99.22% (1521/1533) | 99.09% | 4.6e-4 | 5.3e-4 |
| Qwen2.5-7B-Instruct | 98.74% (1416/1434) | 99.30% | 3.8e-3 | 3.2e-3 |

So the engine is as close to fp32 as HF's own bf16, and no bf16 engine reaches 99.5% on these two. Qwen3-8B (100.00% on the same gate) is the
exception, not the rule. Whether the top-1 floor should widen for models whose bf16 noise is larger is the open decision; I left
`docs/accuracy.md` thresholds alone.

## Open

1. **Top-1 gate / sanity heuristic** (above): needs a decision. Options: a `ref_noise`-aware top-1 floor, or a per-model `--min-top1`.
2. **CUDA / HIP**: kernels written (commit "CUDA/HIP: rope_scaled and add_bias kernels"), `cargo check`/`test --lib` pass for both crates, nothing built with nvcc/hipcc, `kernels test` not run (spark1 never granted a slot in 80 minutes; tatooine blocked). Until then a Llama/Qwen2 model on those backends fails at admission, not silently.
3. **Qwen2.5 prefill 553 tok/s at 4k** against Llama's 1533 at the same size: not investigated (28 heads / 4 KV heads, group of 7, may miss a fast attention path; unconfirmed). Llama decode at 57% of the byte floor with scaled rope bailing out of the fused prologue: not compared to an unfused Qwen3 baseline.
4. **Bias rounding**: bias is added after a bf16-rounded projection (two roundings); HF rounds once. Qwen2.5's large k-bias values may make this the dominant error term. Not measured; the engine is already at HF-bf16 level, so I did not fuse the bias into the matmul epilogue.
5. Not done: `o_proj` bias (HF Llama `attention_bias=true`), sliding window, other activations, `linear`/`dynamic`/`longrope` rope.
6. Not proven non-vacuous on the real model: no run of Llama with the scaling switched off. The unit tests show scaling moves logits by >1e-2 (`the_scaling_and_the_biases_are_not_vacuous`).
