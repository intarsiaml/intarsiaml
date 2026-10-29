# Lane ARX: BI8's prefix-reuse bug on HIP, and the exact rule there (2026-10-29)

Branch `lane-arx/hip-prefix-exact`, on `origin/main` `200be3f05` (#233). tatooine, one R9700 (gfx1201) per run, Qwen3-8B BF16
(`/data/models/intarsia/Qwen3-8B`), lane PRB's probe `crates/intarsia-models/tests/prefix_reuse_real.rs` with
`INTARSIA_PROBE_BACKEND=hip`. Raw: `docs/results/2026-10-29-arx-hip-prefix-reuse/raw/`. Asked by the coordinator after PRB (#233) found
Metal affected and could not measure HIP (Radiance held the cards).

## Result

**HIP has the bug.** On main (`prefix_reuse_any_cut` true on HIP, so the default is the old `any` rule):

| probe (main, default = `any`) | result |
|---|---|
| `prefix_reuse_probe` OLDRULE: a repeated prompt resumed with its last token alone (a 1-row pass) | **15 / 15 differ** in the first logits row (max abs 0.098-0.304); greedy tokens flip within 64 steps in 2 of them (prompts 5 and 8, at tokens 3 and 55). The run was cut at the job's 175 s cap before prompt 15 and the split / cross-prompt cases |
| `engine_repeats_equal_the_cold_reference` | **FAILED**: single-sequence engine request 6 (prompt 8, 12 tokens cached) and the 4-slot batched engine rounds 1 and 2 of prompt 5 (7 cached) give other tokens than the cold reference |

`raw/main-default.txt` and `raw/main-any.txt` (`INTARSIA_PREFIX_REUSE=any`) are the same run twice, as they must be.

**Fix:** `Backend::prefix_reuse_any_cut` returns false on HIP (`crates/intarsia-backend-hip/src/device.rs`), as on CUDA (BI8) and Metal (PRB):
a dense request resumes only where its own cold chunk schedule cuts (`Model::reuse_len`). With it (`raw/fix-default.txt`):
`engine_repeats_equal_the_cold_reference` and `reuse_rule_resumes_with_cold_bits` **pass** (every repeated prompt resumes at 0, under one
2048-row chunk); `INTARSIA_PREFIX_REUSE=any` on the same binary still fails both (`raw/fix-any.txt`, the negative control: the override
restores the old rule).

Cost: as on CUDA / Metal, a dense prompt shorter than one prefill chunk never resumes from a held prefix (multi-turn chat re-prefills its
history); not measured on HIP. Hybrid models (Qwen3.8-27B, Qwen3.6-35B: GDN recurrent state) keep their snapshot rule, which this change does
not touch (`SequenceState::restorable_len` comes first in `reuse_len`).

## PF27b's one-off max-batch-16 text difference

Not explained by this: it was the 27B, whose recurrent state never takes the `any` cut. Lane ARX re-ran that case (main `f9fabeb13`,
`--max-batch 16`, stream A 200 greedy tokens with a 4k-0 prefill admitted beside it) on 5 fresh servers with the default and 5 with
`INTARSIA_PREFIX_REUSE=exact`, each a cold pair and a warm pair: **20 / 20 pairs gave PF27b's majority text** (`915f1c0953`, B `dd42cb5b78`;
`docs/results/2026-10-29-arx-hip-ar-overlap/raw/w1/mixids-*`). Not reproduced.

## Gates

| gate | result |
|---|---|
| HIP probe on main shows the bug | met (15 / 15 OLDRULE, engine repeats differ) |
| fix: engine repeats equal the cold reference, reuse resumes with cold bits | met (2 / 2 pass); `any` override still fails (negative control) |
| `cargo test --workspace --release` (Mac) | met: 204 suites, 1090 passed, 0 failed |

## GPU jobs

fleetq #6742 (build, cpu), #6738 m6 (probe on main, both cards, radiance down 00:24-00:26:41), #6743 m7 (probe with the fix,
00:29-00:31:47). Radiance answered on :8080 after each.
