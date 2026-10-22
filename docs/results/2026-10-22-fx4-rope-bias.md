# llama3 / yarn rope scaling and q/k/v bias: Llama-3.1-8B-Instruct and Qwen2.5-7B-Instruct (2026-10-22, lane fx4)

**Outcome.** Both models load and generate correctly on Metal from built-in recipes (`llama`, `qwen2`), with no core edit beyond the
two features. Qwen3-8B on Metal is unchanged (plan, 2 x 256 greedy ids, both accuracy goldens, byte for byte). **Accuracy is not a clean
PASS**: the top-1 floor (99.5%) fails for both models, so acceptance criterion 1 ("accuracy PASS") is **not met**. The KL gate passes
for both, but for Qwen2.5 only through a reference-noise sidecar that this lane measured and committed (see below). The fp32 golden's
own bf16 forward misses the top-1 floor too, so this is a gate-policy question for a human, not a bug I can see. CUDA and HIP kernels
are written but were never compiled or run, so they are **not in this branch** (they sit on `lane-fx4/cuda-hip-kernels`).

## What was built

| Piece | Where |
|---|---|
| `rope_type` llama3 and yarn: frequency table (transformers formulas, f64 then f32) | `crates/intarsia-core/src/rope.rs` |
| ops `RopeScaled` (x, `inv_freq` table; cos/sin times `rope_mscale`) and `AddBias`, appended to `Op::ALL` | `crates/intarsia-core/src/ops.rs` |
| model: table uploaded once at load; scaled models call `RopeScaled`, unscaled models emit the same calls as before; q/k/v bias after each projection; fused qk-norm+rope prologue bails when scaled | `crates/intarsia-core/src/model.rs` |
| recipe: `[layer.attn] q_bias / k_bias / v_bias`; config `rope_parameters` / `rope_scaling` read into `ModelDims.rope_scaling`; unsupported `rope_type`s and an `o_proj` bias still rejected by name; a checkpoint bias the recipe does not map is a load error | `recipe.rs`, `model.rs` |
| CPU reference + fast kernels | `crates/intarsia-backend-cpu/src/{reference,fast}.rs` |
| Metal kernels `rope_scaled_bf16`, `add_bias_bf16` | `kernels/metal/dense_ops.metal`, `crates/intarsia-backend-metal/src/{kernels,dense}.rs` |
| CUDA / HIP kernels (**not compiled, not run; separate branch `lane-fx4/cuda-hip-kernels`, commit 035f2aea, not part of this PR**) | `kernels/{cuda,hip}/elementwise.*`, `kernels/abi/ik_abi.h`, `crates/intarsia-backend-{cuda,hip}` |
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
| acc-short prefill full-vocab KL, scored rows (static floor 1.0e-3) | 9.8e-5, under the floor: PASS | 1.405e-3, **over** the floor: passes only through the noise term (next row) |
| Qwen2.5 noise term (needs the sidecar) | n/a | 21-way KL on scored rows 1.19e-3 <= ref noise 1.355e-3 (ratio 0.880) => PASS; the full-vocab 1.405e-3 above is not what the noise term compares |
| acc-short decode top-1 (teacher-forced) | **99.38%** | **98.12%** |
| decode tok/s at 128 / 1024 / 4096 (% of byte floor) | 23.2 / 23.9 / 22.4 (57 / 58 / 55%) | 31.7 / 32.2 / 27.6 (73 / 74 / 64%) |
| prefill tok/s at 128 / 1024 / 4096 | 1081 / 1306 / 1533 | 1494 / 1014 / 553 |
| onboard wall time | 5m17s (plan load 59s, kernels 80s) | 5m17s (plan load 79s, kernels 69s, bench 2m08s) |

**The Qwen2.5 KL PASS depends on a gate-affecting artifact this lane created.** `golden/ref_noise/*-fp32.safetensors.json` are measured by
`intarsia ref-noise --method precision_spread` between the HF fp32 and HF bf16 goldens (same ids). The first onboard ran without them and
Qwen2.5's prefill KL FAILED (1.405e-3 over the 1e-3 floor, `raw/.../onboard-Qwen2.5-7B-Instruct-quick.md`); the second accuracy run
(`raw/.../acc-*.out`) used them. Llama passes its KL gate with or without its sidecar. Reviewers should decide whether a self-measured
precision-spread record is an acceptable gate input for a new model. The goldens (four safetensors, 1.6 GB, hash-bound to the sidecars)
are in `/Users/josef/models/golden/` (`INTARSIA_GOLDEN_DIR=/Users/josef/models/golden`); they are not in git.

**Why top-1 fails, and why I did not touch the gate.** The reference's own bf16 forward has the same problem. HF transformers bf16 vs HF fp32 on the same ids:

| | HF bf16 vs HF fp32 top-1 | engine (Metal bf16) vs HF fp32 top-1 | HF bf16 vs fp32 full-vocab KL | engine full-vocab KL |
|---|---|---|---|---|
| Llama-3.1-8B-Instruct | 99.22% (1521/1533) | 99.09% | 4.6e-4 | 5.3e-4 |
| Qwen2.5-7B-Instruct | 98.74% (1416/1434) | 99.30% | 3.8e-3 | 3.2e-3 |

So the engine is in the same range as HF's own bf16 (slightly worse on Llama top-1, 99.09% vs 99.22%; slightly better on Qwen2.5), and a
bf16 forward reaching 99.5% on these two looks out of reach. Qwen3-8B (100.00% on the same gate) is the
exception, not the rule. Whether the top-1 floor should widen for models whose bf16 noise is larger is the open decision; I left
`docs/accuracy.md` thresholds alone.

## Open

1. **Top-1 gate / sanity heuristic** (above): needs a decision. Options: a `ref_noise`-aware top-1 floor, or a per-model `--min-top1`.
2. **CUDA / HIP**: kernels written on branch `lane-fx4/cuda-hip-kernels` (one commit on top of this lane's second commit). Update: CUDA later built and passed `kernels test --quick` on spark1 (see Round 2); HIP was never compiled with hipcc (tatooine blocked). They are kept out of this PR so an uncompiled `.cu`/`.hip` file cannot break the CUDA/HIP build. On this branch `RopeScaled` / `AddBias` stay in `NOT_ON_CUDA` / `UNIMPLEMENTED`, so a Llama/Qwen2 model on those backends fails at load with `NoKernel` and never runs the unscaled rope.
3. **Qwen2.5 prefill 553 tok/s at 4k** against Llama's 1533 at the same size: not investigated (28 heads / 4 KV heads, group of 7, may miss a fast attention path; unconfirmed). Llama decode at 57% of the byte floor, scaled rope bailing out of the fused prologue: not compared to a Qwen3-8B number from the same slot.
4. **Bias rounding**: bias is added after a bf16-rounded projection (two roundings); HF rounds once. Qwen2.5's large k-bias values may make this the dominant error term. Not measured; the engine is already at HF-bf16 level, so I did not fuse the bias into the matmul epilogue.
5. Not done: `o_proj` bias (HF Llama `attention_bias=true`), sliding window, other activations, `linear`/`dynamic`/`longrope` rope.
6. Not proven non-vacuous on the real model: no run of Llama with the scaling switched off. The unit tests show scaling moves logits by >1e-2 (`the_scaling_and_the_biases_are_not_vacuous`).

## Round 2 (coordinator follow-ups, same day)

### Bias: one rounding (commit 761ad319)
A bf16 weight with a bias now projects through the f32-output matmul (`Op::Logits`) and `Op::AddBias` adds the bias in f32 and rounds once into the bf16 activation (Metal kernel `add_bias_f32_bf16`, CPU generic), which is how HF's fused bf16 `Linear` rounds. A biased k/v skips the pair kernel. Unit tests: `add_bias_f32_input_rounds_once` (CPU and Metal, bit-exact, the data separates one rounding from two), `bf16_biased_projections_round_once_through_the_f32_output`. Found on the way and fixed: `RopeScaled` was missing from `plan.rs` `PER_ROW_OPS`, so a batched step of a scaled-rope model would have failed `check_batch_ops` at load; `batched_decode_with_scaled_rope_and_bias_is_bitwise_the_solo_steps` covers it.

Qwen2.5-7B-Instruct against the HF fp32 golden (`raw/2026-10-22-fx4/step23/acc-qwen2.5-*.out`, with the ref-noise sidecar), same prompts and binary family:

| Variant | prefill top-1 | prefill KL (full vocab, all rows) | prefill KL scored rows (noise bound 1.355e-3) | decode top-1 |
|---|---|---|---|---|
| double rounding (first branch build) | 99.30% | 3.16e-3 | 1.405e-3 | 98.12% |
| single rounding | 98.81% | 3.50e-3 | 1.471e-3 | 97.50% |
| single rounding + flash group 7 (final) | 98.88% | 3.59e-3 | 1.566e-3 | 98.12% |
| HF bf16 vs HF fp32 (reference spread) | 98.74% | 3.8e-3 | n/a | n/a |

**Single rounding did not improve accuracy against fp32; it moved the engine from 99.30% to 98.9%, i.e. onto HF bf16's own 98.74%.** The coordinator's target (match or beat HF bf16's top-1) is met by every variant. With 1434 rows, 0.4 points is 7 rows, so I read the move as noise from a different rounding realisation, not as a regression, but it is not the improvement the hypothesis predicted. All variants still fail the 99.5% floor and pass the KL gate through the noise sidecar.

Cost: the f32-output prefill GEMM (`gemm_bf16_o_f32`) is not the NAX kernel. In the 4k op profile the q/k/v `logits` calls take 1.39 s of 3.76 s (`raw/.../prof-Qwen2.5-7B-Instruct-final.err`), against about 0.1 s for the q projection before. Decode 32.5 -> 31.4 tok/s (-3.4%, 2 reps each, `dec-*.err`). A NAX GEMM with an f32 output or a bias epilogue would remove the prefill cost; that is a kernel project, not done. Reverting is the `_` arm of `Model::proj_bias` (project, round, add).

### Qwen2.5 prefill at 4k: the flash kernel did not take group 7 (commit "Metal flash prefill ... groups 3, 5, 6, 7")
Cause (confirmed by the op profile): Qwen2.5-7B has 28 query heads over 4 kv heads, a group of 7. `flash_ok` / `nax_flash_ok` accepted only groups 1, 2, 4, 8, so every prefill ran the scalar `attn_prefill_bf16`: 4413 ms of 6637 ms (66%) in a 4k profile. Fix: both flash shaders take any group of 1 to 8. Groups that do not divide the simdgroups (3, 5, 6, 7) run the next larger variant; the surplus simdgroups still load K/V and hit the barriers but own no head (`live = hg < G`: no Q load, no store). Each real head runs the arithmetic of the dividing group, so groups 1, 2, 4, 8 keep their variants and code path.

| 4k prompt, `generate --max-tokens 1`, 2 reps | before | after |
|---|---|---|
| Qwen2.5-7B prefill tok/s | 727 / 722 | 1797 / 1806 (2.5x) |
| attn_prefill share, profiled run | 4413 ms (66%) | 186 ms (5%) |

Tests: `flash_matches_f64_reference` (plain and NAX) gained groups 3, 5, 6, 7 at head_dim 64, 128, 256 including the 4k continuation chunk; `flash_predicate` accepts 3, 5, 6, 7 and rejects 9. All pass on Metal.

### Qwen3-8B no-op for this round (Metal, main 659f770c vs branch f96942c6+, exclusive slot)
Kernel registry grows (39 -> 42 kernels, `rope_scaled`, `add_bias` x2); everything else byte-identical: `plan show` JSON and text, the 256 greedy ids for the short and 4k prompts, both accuracy goldens (`raw/.../step23/`). The flash change touches shaders Qwen3-8B (group 4) runs, so this is the check that matters for it.

### CUDA check of the side branch (fleetq #3835 and #3987, spark1 GB10, nvcc 13.0)
`lane-fx4/cuda-hip-kernels` (035f2aea) built with `scripts/deploy.sh spark1 --backend cuda` and `intarsia kernels test --quick --backend cuda` ran: `rope_scaled_bf16` 8 cases ADMITTED (max abs 0), `add_bias_bf16` 2 cases ADMITTED, 56 kernels, 0 failed, smoke pass. So the CUDA kernels compile and pass admission on the GB10. Not covered: that branch predates the f32-input `AddBias` of round 2, so a biased model on CUDA would still stop at load with `NoKernel` for `add_bias` (f32 x -> bf16) until that variant exists (the f32-output `Logits` it follows exists on CUDA). No model was run on CUDA. HIP: nothing compiled with hipcc.

### Still open
Top-1 floor policy; an f32-input `add_bias` kernel for CUDA/HIP and the HIP compile check; a NAX/f32-out GEMM for the biased prefill projections.
