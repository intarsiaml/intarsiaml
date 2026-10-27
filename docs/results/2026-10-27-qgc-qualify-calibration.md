# QGC: calibrated top-1 floor, auto short golden, dtype error naming (2026-10-27)

Lane QGC, branch `lane-qgc/qualify-calibration`, build origin/main `d5810616c` plus this branch, endor (Apple M5 Max), Metal. Fixes frictions 2, 3 and 1 of `2026-10-27-nmd-new-model-drill-metal.md`. Fleet jobs on endor through fleetq (#6164-#6173), `endor-kev` and `endor-tensorfold` stayed up.

## 1. Top-1 floor from the golden's reference spread

`tools/golden.py --dtype float32` now runs a second pass with the same model in bf16 (same ids) and writes, into the golden's `ref_spread` metadata, the top-1 agreement bf16-vs-fp32 over all rows and the mean KL (21-way for a top-k golden, full-vocabulary otherwise). `intarsia accuracy` takes the floor for a mode that judges `n` rows as

    floor(n) = min(floor in force, agreement - 2 x sqrt(p (1 - p) / n)),   p = agreement / 100

and keeps the KL floor as it was (`max(1e-3, factor x ref noise)`). `ref_spread` is measured over all rows of the golden, the gate's top-1 judges all rows too; decode mode reuses the same agreement with the standard error at its own row count, an approximation (not the agreement over the decode window alone). Making a golden with the spread pass about doubles its time (a 4k golden of the 4B: 34 min -> about an hour; `--no-bf16-spread` skips it). Margin: two binomial standard errors at the rows judged. An engine exactly as noisy as HF's bf16 run is a draw from the reference's own distribution and falls under this floor in about 2 % of runs; the drill's engine, closer to fp32 than HF bf16 is (99.72 % against 98.96 %), almost never. The margin depends on `n` because a fixed one cannot serve both mode sizes: one decode flip is 0.6 points on 160 rows, so a fixed margin of a few tenths fails on one extra flip. Never above the floor in force (a 99.9 % reference does not raise the bar over 99.5 %), not used when the reference agrees with itself on under 95 % of rows (a broken golden, the class floor then fails loudly), `--min-top1` wins. Code: `crates/intarsia-cli/src/cmd/accuracy/top1ref.rs`, `Gates::top1_floor`.

Qwen3.5-4B, auto-made short golden (8 prompts, 441 rows, `golden.py` output: `ref_spread: bf16 vs fp32 top-1 agreement 98.87% (5 flips in 441 rows), full-vocab KL 4.957e-04`):

| mode | engine top-1 | old floor | new floor | KL (floor 1e-3) | verdict old -> new |
|---|---|---|---|---|---|
| prefill (441 rows) | 99.77 % | 99.5 | 97.86 | 8.5e-5 | PASS -> PASS (this set happens to clear 99.5; the drill's 1441-row set did not) |
| decode (128 rows) | 99.22 % | 99.5 | 96.99 | 2.0e-4 | FAIL (one flip in 128 rows is 99.22 %) -> PASS |

The drill's own numbers (98.96 % over 1441 rows) give floors 98.43 (prefill) and 97.36 (160 decode rows): its engine's 99.72 % and 98.75 % pass, 5 flips in 160 rows (96.9 %) still fail (unit test `the_qwen35_4b_drill_numbers_pass_and_a_real_regression_fails`).

### Existing goldens: same verdict as before

A golden without `ref_spread` takes the old code path exactly (`top1_spread` is `None`; unit test `ref_spread_calibrates_the_top1_floor_and_its_absence_changes_nothing` pins floor and printed text). Checked by reading the metadata of every golden / oracle file found: endor `/opt/code` (all worktrees) and `/Users/josef` 49 distinct files (`format` golden / oracle, copies in worktrees and `~/intarsia-deploy/*` collapsed), 0 with `ref_spread`; spark1 1161 files (copies included), spark2 522, tatooine 255 (read-only header scan of `~`, `/models`, `/opt`): 0 with `ref_spread` everywhere.

Real before/after runs (`intarsia accuracy --golden`, origin/main binary against this branch, Metal, same weights), floor, top-1 and verdict lines identical:

| golden | model | prefill floor / top-1 | decode floor / top-1 | golden gate before -> after |
|---|---|---|---|---|
| `golden/qwen3-8b-fp32.safetensors` | Qwen3-8B | 99.5 / 100.00 | 99.5 / 100.00 | PASS -> PASS |
| `~/Models/golden/qwen2.5-7b-instruct-fp32.safetensors` | Qwen2.5-7B-Instruct | 98.24 (recipe) / 99.09 | 98.24 (recipe) / 97.50 | FAIL -> FAIL (decode top-1, as before) |
| `~/Models/golden/llama-3.1-8b-instruct-fp32.safetensors` | Llama-3.1-8B-Instruct | 98.72 (recipe) / 99.09 | 98.72 (recipe) / 99.38 | PASS -> PASS |

The other goldens (not run: the models are 35B / 26B / 27B, the golden is the evidence): `qwen3-8b-long-fp32`, `qwen3-8b-4k-fp32-top20`, `qwen3-reranker-0.6b[-long][-fp32]`, `qwen2.5-7b-instruct-bf16`, `llama-3.1-8b-instruct-bf16`, `gemma4-26b-a4b-nvfp4-{fp32,hfbf16,cpu-oracle}`, `gemma4-26b-vllm.golden`, `qwen36-35b-a3b-nvfp4-{vllm-top20,vllm-long-top20,vllm-4k-top20,vllm-long-reuse-s9500,cpu-oracle,oracle-set-vllm-top20}`, the `*.selfgolden` and `continue-oracle` files: none has the key, so the floor is the class floor or the recipe's `min_top1` as before.

## 2. `qualify` makes the short golden of a hybrid model

`auto_golden_decision` (qualify.rs): a model with no golden found gets an HF fp32 short golden made when it is not MoE and not quantized (hybrid allowed), its weights are at most `--make-golden-max-gb` (default 5 -> 16) and its fp32 copy (2 x weights + 3 GB) fits in the available RAM and in `--mem-budget`. The command runs inside a fleetq slot on this machine (`fleetq run --for-url http://127.0.0.1 --load cpu --expect MIN --mem GB`) unless `FLEETQ_JOB` is set; without `fleetq` on the box it just runs. MoE / quantized, too large and no-RAM cases are skipped with the reason. A hybrid model's long golden and every 4k golden stay manual and the SKIPPED reason carries the exact command.

Qwen3.5-4B, `intarsia onboard --model Qwen/Qwen3.5-4B --backend metal --quick-only` (fleetq #6166, `--quiet gpu,cpu`), nothing in any golden directory:

| | drill (2026-10-27, by hand) | this branch |
|---|---|---|
| weights to first token (recipe, smoke) | 45 min 36 s (golden 34 min + two engine fixes) | 1 s (engine fix already in main) |
| short golden | 1 min 5 s by hand (default prompts, no spread pass) | 3 min 20 s stage, made by qualify (golden.py 185 s: fp32 pass 96 s, bf16 spread pass 89 s) |
| weights to `quick.md` | 48 min | **5 min 37 s**, ONBOARD PASS |
| acc-short | FAIL (floor 99.5) | PASS (prefill 99.77 %, decode 99.22 %) |
| 4k golden | 34 min by hand, to get the in-depth run | not made, command printed |

The 3 min 20 s is slower than the drill's one minute: the two passes are 8 prompts at about 12 s each on this box today (the HF PyTorch GatedDeltaNet fallback dominates, not the length), and the bf16 pass costs as much as the fp32 one on the CPU. `--no-bf16-spread` drops the second pass; the fp32 pass alone is 1 min 36 s.

## 3. Load-time error names the tensor

The plan-time `no kernel` error for an op whose weight dtype no kernel takes lists the checkpoint tensors, their dtype and the op. Both errors of the drill are now plan-time. Reproduced on Qwen3.5-4B / Metal with the `gpu_f32_to_bf16` cast switched off for the test (temporary edit, reverted), `plan show`:

A_log cast off:

    - gdn_delta_rule: 27 shape(s) without a kernel, e.g. no kernel for gdn_delta_rule on metal/applegpu_g17s (m=1 n=128 k=128 act=Some(Bf16) w=Some(F32)) ...; needed by Decode t=1, ...
        weight tensor(s) of a dtype no gdn_delta_rule kernel takes on metal/applegpu_g17s: 'model.language_model.layers.0.linear_attn.A_log' (f32), 'model.language_model.layers.1.linear_attn.A_log' (f32), ... (24 in all)

Gated-norm weight cast off (A_log cast kept); this was the drill's second error, `metal w: expected bf16, got f32` at the first launch:

    - rmsnorm: 27 shape(s) without a kernel, e.g. no kernel for rmsnorm on metal/applegpu_g17s (m=32 n=0 k=128 act=Some(Bf16) w=Some(F32)) ...
        weight tensor(s) of a dtype no rmsnorm kernel takes on metal/applegpu_g17s: 'model.language_model.layers.0.linear_attn.norm.weight' (f32), ... (24 in all)

Two parts. (a) `Model` keeps a name index of the dense tensors `load_shard` read (`weight_names`, matched by buffer and offset; read on the error path only) and the plan error lists the tensors of the dtype the failed call asked for. (b) The Metal norm family (`rmsnorm_bf16`, `rmsnorm_add_bf16`, `qk_norm_bf16`) now declares `.weights(bf16)`, and `ShapeInfo` reports the norm weight dtype for `RmsNorm`, `RmsNormAdd` and `QkNorm` (it was empty there, so a `.weights(..)` list matched everything). Other backends register no weight dtype for these ops, so they are unchanged. Checked unchanged on Metal against the origin/main binary: `kernels test --backend metal --quick` 46 kernels / 219 cases, all admitted, both builds; `plan show` of Qwen3-8B and Qwen3.5-4B byte-identical (kernel per op).

Still launch-time only: any other Metal kernel that binds a fixed dtype without declaring it. A grep of `bind(.., "w")` found the dense matmul / logits (already `.weights(bf)`), the norm family (now declared), and the fp8 / mxfp4 / nvfp4 / moe launchers, whose weight dtypes are fixed by the quantisation format and selected by op, not by dtype. No other dense bf16 op binds a weight.

## Raw

`raw/2026-10-27-qgc/onboard/` (quick.md, in-depth.md, golden.py output, acc-short log, onboard logs), `raw/2026-10-27-qgc/before-after/` (accuracy runs of the origin/main and branch binaries on the three existing goldens, `kernels test --backend metal --quick` of both).

Full onboard: `ONBOARD PASS` after the resumed in-depth tier (71 s: `acc-long`, `acc-4k`, `ab`, `tp`, `acc-oracle`, `acc-format` SKIPPED with reasons and the exact golden commands; envelope, bench-long and the rest PASS). Final verdict PASS, not INCOMPLETE: only a missing *short* golden makes the run INCOMPLETE (#197).

## Tests

`cargo check --workspace --locked`, `cargo test --workspace --locked --release --no-fail-fast`, `cargo check -p intarsia-backend-metal` and `-p intarsia-cli --features metal` pass (fleetq #6195, the commands CI runs; CI runs no fmt / clippy). New: `top1ref::tests` (5), `accuracy::tests::ref_spread_calibrates_the_top1_floor_and_its_absence_changes_nothing`, `qualify::tests` (auto-golden decision, manual commands and slot wrapper), `plan::tests::the_no_kernel_error_names_the_weight_tensors_their_dtype_and_the_op`, `model::weights_named_tests`, `ops::tests::norm_calls_report_the_norm_weight_dtype_so_a_kernel_can_declare_it`.
