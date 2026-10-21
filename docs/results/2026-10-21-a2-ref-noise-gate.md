# A2: a reference-noise-aware KL gate (2026-10-21)

## 1. Headline

The accuracy gate now takes the golden's own measured spread into account, as the two user decisions of 2026-10-21 ask
(11:23 item 3, "a change must be no larger than vLLM's own run-to-run KL on the same golden"; 12:39 item 2, M1's reference-noise
floor). One gate, two uses:

* **Engine against a golden:** PASS when `KL <= max(static floor, factor x ref_noise)`. The run prints and writes (`--write-verdict`)
  the static floor, the noise, the factor and where it came from, the effective threshold, and which of the two bound.
* **Change against base** (`--self-golden --self-vs-ref-noise`): the change's mean 21-way KL against the base build must be at most
  `factor x ref_noise`, and the worst prompt's mean KL at most `prompt_factor x ref_noise` (3.0 by default, configurable); non-finite rows
  fail; largest position KL, flips and tail mass are reported only (user decision 2026-10-21 13:23, "1 rec, 2 yes, 3 later").
* **Measuring** (`intarsia ref-noise`): offline, no model and no GPU, from a second reference run or a different-precision reference.

Re-scored from committed logs and documented numbers (nothing measured on a GPU), at factor 1.0:

* **One verdict flips on the engine-vs-golden gate:** Qwen3.8-27B W4A4 prefill, 0.180 against a noise of 0.197, FAIL to PASS. Its decode mode
  stays FAIL at 0.274 (1.39 times the noise).
* **No change-vs-base verdict flips.** CUDA round 7's `lt` against the default is inside vLLM's run-to-run KL on the mean for the long golden
  (0.85x, 0.90x and 0.93x), but the worst-prompt check (3.0 x ref noise) fails it on `chat_reuse_4k`, the prompt vLLM itself does not reproduce
  (0.097 / 0.107 / 0.106 against a limit of 0.0665, 1.46x to 1.60x). On the short golden `lt` is 1.06x (chunk 2048) and 1.11x (chunk 256) on the mean and
  1.03x on the worst prompt (`chat_system`): FAIL on both checks. Before the worst-prompt decision the three long rows were PASS on the mean alone.
* **Nothing else changes.** On the 35B the noise (3.0e-3 short, 8.4e-3 long, scored rows) is far below the 4-bit class floor of 3e-2, so the floor
  binds and the new term never decides. If the noise were the only bound, **no 35B build would pass at factor 1.0**: every build is 1.5 to
  1.8 times vLLM's own spread from the golden. HIP main on 35b-long (scored rows 1.420e-2) is 1.69 times the noise.

## 2. Numbers

All KL are mean 21-way KL, nats. Sources: `raw/rescore-table.txt` (printed by the tests `rescore_cuda_round7_from_the_committed_table` and
`rescore_m1_w4a4_from_the_documented_numbers`, which read `docs/results/2026-10-21-cuda-round7/raw/05-kl-table.txt` and the back-filled sidecars
in `golden/ref_noise/`), `raw/rescore-hip-main-long.txt` (the M3 per-row dump of HIP main, re-scored by `rescore_m3_dump_with_the_row_mask --ignored`,
which now also loads the sidecar by file hash).

### 2.1 Engine against the golden (floor = max(class floor, factor x noise))

Class floor 3e-2 for the 35B (4-bit class) and for the Qwen3.8-27B (floor from its drill doc). "Logged" is the verdict in the run's own log; "main"
is the gate of origin/main today (scored rows, no noise); "new" is this PR at factor 1.0.

| case | rows | KL | ref noise | KL / noise | logged | main today | new | bound by |
|---|---|---|---|---|---|---|---|---|
| Qwen3.8-27B W4A4, prefill | all (n=1441) | 0.180 (0.176 by `golden_kl.py`) | 0.197 | 0.91 | FAIL | FAIL | **PASS** | ref noise |
| Qwen3.8-27B W4A4, decode | all | 0.274 | 0.197 | 1.39 | FAIL | FAIL | FAIL | ref noise |
| 35b-short, default c2048 | scored (n=1279) | 4.539e-3 | 3.010e-3 | 1.51 | PASS (5.430e-3, all rows) | PASS | PASS | static floor |
| 35b-short, lt c2048 | scored | 5.303e-3 | 3.010e-3 | 1.76 | PASS (6.648e-3) | PASS | PASS | static floor |
| 35b-short, lt c256 | scored | 5.063e-3 | 3.010e-3 | 1.68 | PASS (6.434e-3) | PASS | PASS | static floor |
| 35b-long, default c2048 | scored (n=32349) | 1.420e-2 | 8.417e-3 | 1.69 | FAIL (4.576e-2, all rows) | PASS | PASS | static floor |
| 35b-long, lt c2048 | scored | 1.355e-2 | 8.417e-3 | 1.61 | FAIL (4.725e-2) | PASS | PASS | static floor |
| 35b-long, lt c256 | scored | 1.409e-2 | 8.417e-3 | 1.67 | FAIL (4.492e-2) | PASS | PASS | static floor |
| 35b-long, lt + c4096 | scored | 1.360e-2 | 8.417e-3 | 1.62 | FAIL (4.588e-2) | PASS | PASS | static floor |
| 35b-long, HIP main c4096 (tool-exact, M3 dump) | scored (n=32349) | 1.4198e-2 | 8.417e-3 | 1.69 | n/a | PASS | PASS | static floor |

Reading:

* The "logged" 35B long FAILs are the pre-mask all-rows numbers (they were run before the scored-rows floor). The scored-rows floor (already on main) flips them,
  not the noise. All rows would still FAIL under the new rule: 4.576e-2 is 2.06 times the all-rows noise of 2.218e-2 and above 3e-2.
* The 35B scored-rows values other than HIP main's come from `setkl.py` (both sides truncated to top-20), not from the tool. On our own rows the two agree to
  about 1e-4 relative (HIP main: 1.4198e-2 by the tool, 1.420e-2 by `setkl.py`, `raw/rescore-hip-main-long.txt`).
* The M1 noise (0.197) has no scored-rows value and its row set is not recorded (taken as all rows). The 0.180 and 0.274 are the drill's gate numbers, which predate
  the scored-rows floor, so they are all-rows too (checked: `turns.rs` first landed on main at 11:09 EDT in 6952b26b, M1 branched from `4be7b7d4` and ran its gate at 10:19; the round-7 logs in
  `raw/r7A/acc-*.log` carry no "scored rows" text); compared like with like. The decode mode scores the last 16 positions of each prompt, the noise was measured over
  every position: the gate applies it as measured and says so.
* "Noise alone" means the rule without the static floor, `KL <= factor x noise`. At factor 1.0 it fails every 35B row above; the factor each would need is the KL / noise column
  (1.5 to 1.8).

### 2.2 Change against base (`--self-vs-ref-noise`)

CUDA round 7: `lt` (opt-in e4m3 cuBLASLt GEMMs) against the default build, KL21(default || lt), against vLLM's own run-to-run KL on the same golden. A self-golden compares all rows, so
the verdict is on all rows; the scored-rows ratio is given too (the same verdicts). "Old" is the self-golden gate at its default budgets (mean 21-way KL <= 1e-4); it was not run for this pair, so
"over" is the ratio KL / 1e-4 computed from the round-7 numbers. The worst prompt is the largest per-prompt mean KL21 of the same log (all rows, as a self-golden reports it), against
3.0 x the all-rows noise (the limit is 9.48e-3 on the short golden, 6.65e-2 on the long one). The noise of a single prompt is not used.

| golden | build | KL(default \|\| lt), all rows | vLLM run-to-run | mean ratio | scored rows ratio | worst prompt | / limit (3.0 x noise) | old | new |
|---|---|---|---|---|---|---|---|---|
| 35b-short | lt c2048 | 3.358e-3 | 3.161e-3 | 1.062 | 1.048 | chat_system 9.774e-3 | 1.03 | FAIL (34x over) | **FAIL** (mean needs factor 1.06, worst prompt 3.09) |
| 35b-short | lt c256 | 3.503e-3 | 3.161e-3 | 1.108 | 1.102 | chat_system 9.774e-3 | 1.03 | FAIL (35x over) | **FAIL** (mean needs 1.11, worst prompt 3.09) |
| 35b-long | lt c2048 | 1.884e-2 | 2.218e-2 | 0.849 | 0.904 | chat_reuse_4k 9.714e-2 | 1.46 | FAIL (188x over) | **FAIL** on the worst prompt (mean passes; needs prompt factor 4.38) |
| 35b-long | lt c256 | 2.000e-2 | 2.218e-2 | 0.902 | 0.897 | chat_reuse_4k 1.067e-1 | 1.60 | FAIL (200x over) | **FAIL** on the worst prompt (needs 4.81) |
| 35b-long | lt + c4096 | 2.057e-2 | 2.218e-2 | 0.927 | 0.984 | chat_reuse_4k 1.062e-1 | 1.60 | FAIL (206x over) | **FAIL** on the worst prompt (needs 4.79) |

On the mean this restates round 7's own reading (docs/results/2026-10-21-cuda-round7.md 3.2): long inside the noise, short at the noise level and not under it. The worst-prompt check is what fails the long
rows. `chat_reuse_4k` is the prompt where vLLM's own re-run is 0.118 from its golden and where the default build is already at 0.178 (round 7, 3.2): a global 3.0 x noise limit is below what that single prompt does
between two correct builds, so the check keeps flagging it. A per-prompt noise limit (the log has each prompt's vLLM re-run value) would pass the long rows; it is not what was decided and was not added. Honest note on the short golden:
M3 measured two more vLLM re-runs of it at 2.5e-3 and 2.9e-3 (against 3.161e-3), so a 6 % excess over one re-run is inside the spread of the noise itself. The gate takes the one measured value in the record
and does not average; `runs` and `how` say so.

## 3. Gates run

| check | result |
|---|---|
| unit tests: threshold logic (`judge_kl`, `resolve_factor`, `resolve_prompt_factor`, `judge_worst_prompt`, `apply_to_self_checks`), record parsing and binding, sidecar lookup, top-k pair KL, dates | 20 pass (`cargo test -p intarsia-cli refnoise`, which includes the two re-score tests) |
| re-score tests on the committed round-7 log and the M1 numbers | pass (`rescore_`) |
| `rescore_m3_dump_with_the_row_mask --ignored` with the real 35B long golden, tokenizer and M3 dump | pass: 36887 / 32349 rows, 4.3560e-2 / 1.4198e-2, sidecar bound by file hash, verdict PASS bound by static floor, 1.687 times the noise |
| `intarsia ref-noise` against `scripts/m3-accuracy/setkl.py` on a perturbed copy of the 35B short golden | identical: all rows 3.211e-2, scored rows 3.482e-2 (1441 and 1279 rows; `raw/crosscheck-setkl.txt`) |
| `intarsia ref-noise` HF fp32 against HF bf16, Qwen3-Reranker-0.6B short set | full-vocabulary KL 8.86e-4, documented 8.7e-4 (`docs/accuracy.md`); KL21 6.69e-4 (`raw/crosscheck-reranker.txt`) |
| `cargo check --workspace` | pass |
| `cargo test --workspace --exclude intarsia-backend-metal` | pass |
| `cargo test --workspace` in `fleetq run --host endor --mem 8` | see section 5 |

No GPU was used. The two goldens that would measure the M1 decode-window noise and the vLLM re-run noise from scratch (the bf16 and f32 W4A4 goldens, `vllm2-top20`) were not on this host.

## 4. Files

* `crates/intarsia-cli/src/cmd/accuracy/refnoise.rs` (new): the record, binding, sidecar lookup, factor, `judge_kl`, the change-vs-base transform, the `ref-noise` measurement, tests.
* `crates/intarsia-cli/src/cmd/accuracy.rs`: flags `--ref-noise`, `--no-ref-noise`, `--ref-noise-factor`, `--ref-noise-prompt-factor`, `--write-verdict`, `--self-vs-ref-noise`; the floor uses `judge_kl`; the self-golden path; verdict file; the re-score test.
* `crates/intarsia-cli/src/cmd/accuracy/selfgolden.rs`: `SelfCheck.gated`; `SelfMeta.ref_noise` (optional, copied from the source golden when a self-golden is written).
* `crates/intarsia-cli/src/main.rs`: the `ref-noise` subcommand.
* `crates/intarsia-core/src/recipe.rs`, `crates/intarsia-models/src/recipe.rs`: `[accuracy] ref_noise_factor` and `ref_noise_prompt_factor` (optional; no built-in recipe sets them) and their test.
* `golden/ref_noise/*.json`: three back-filled sidecars (the two 35B vLLM goldens, bound by hash; the Qwen3.8-27B W4A4 golden, matched by position count only because its file was not available).
* `scripts/a2-ref-noise/`: `perturb_topk_golden.py` (the cross-check below) and `make_backfill_sidecars.py` (regenerates the three sidecars).
* `docs/accuracy.md` (new section "Reference noise"), `docs/decisions/LOG.md` (rows for the 11:23, 12:39 and 13:23 decisions), this doc and `raw/`.

Effect on each backend: none. Nothing here touches a kernel, an engine or a model path; the one core change is an optional recipe field. The visible behaviour change is in `intarsia accuracy`: it prints one extra
line per mode, and a golden with a sidecar gets the noise-aware floor automatically (`--no-ref-noise` turns it off). With the sidecars in this PR that changes a verdict only for the Qwen3.8-27B W4A4 golden (its golden was not on this host; the sidecar is found by the golden's file name, `qwen3.8-27b-mxfp4-hfquark-top20.safetensors`, the name in `tools/golden_quark_mxfp4.py`'s usage line).
Baseline JSON, `ModeStats` and the SELF-STATS line are unchanged.

## 5. Downtime

None. `cargo test --workspace` on endor took a `fleetq` slot (job and result in the PR description).

## 6. Decisions for the user

Decided by the user at 13:23 EDT, "1 rec, 2 yes, 3 later" (relayed by the coordinator), and implemented here:

1. **Change vs base:** largest position KL, top-1 flips and tail mass are reported only; the worst prompt's mean KL is a hard check at `prompt_factor x ref_noise` with `prompt_factor` 3.0
   (`--ref-noise-prompt-factor`, else the record's `prompt_factor`, else the recipe's `[accuracy] ref_noise_prompt_factor`, else 3.0); non-finite rows still fail. Result on the round-7 data: the worst-prompt check
   fails all five rows (section 2.2), so the long `lt` rows no longer pass. **Open for the user:** `chat_reuse_4k` alone needs a prompt factor of 4.4 to 4.8; a per-prompt noise limit would pass it.
2. **Keep `max(static floor, factor x ref_noise)`.** Unchanged. On the 35B it does nothing, because 3e-2 is 3.6 to 10 times the noise; nothing passes on the noise alone at factor 1.0 (factors 1.5 to 1.8, 2.06 on all rows).
3. **Qwen3.8-27B W4A4 decode-window noise: deferred, an open follow-up.** The decode mode stays FAIL at 1.39 times the noise. The noise was measured over every position and the decode mode scores a 16-position window per prompt;
   measuring the noise on that window needs the two W4A4 goldens (about 11 min of CPU on endor). No factor was chosen to make it pass.
4. **Existing goldens use sidecars, not embedded metadata**: rewriting a golden's metadata changes its hash and orphans every baseline and self-golden. Goldens built from now on can carry the `ref_noise` key; `tools/golden*.py` do not write it yet.
5. **M1's record is matched by position count only** (file and hash unavailable) and its row set is assumed to be all rows. The sidecar is found by the name `qwen3.8-27b-mxfp4-hfquark-top20.safetensors`; if the golden is named
   differently, pass `--ref-noise golden/ref_noise/qwen3.8-27b-mxfp4-hfquark-top20.safetensors.json`. Once the golden is on a box, `intarsia ref-noise` re-measures it with a hash and the scored-rows value.
