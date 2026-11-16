# HIP: re-record the four Qwen3.8-27B accuracy baselines after the chunked GDN default (lane b27-rerecord)

Status: done (runs 2026-11-15 21:13-21:35 EST on tatooine; PR not merged). The decision section was written before the re-record, so it survives context compaction.

## Lane brief and coordinator decision (kept here so it survives compaction)

PR #378 (merged 2026-11-15) made the HIP chunked GatedDeltaNet delta rule the default for prefill calls of >= 64 tokens at dk = dv = 128
(`INTARSIA_GDN_CHUNK=0` restores the old kernels, bit-identical to main before #378). The 35B HIP baselines were re-recorded then
(`2026-11-15-hip-gdn-default.md`); the four `golden/baselines/qwen3.8-27b-mxfp4-hfquark{,-w4a16}-top20.hip.{baseline,envelope}.json` were not.

**Coordinator decision (2026-11-16, after step 1 below): re-record all four at the recipe default prefill chunk (2048).** Reasoning: the
pre-#378 W4A16 staleness (the HIP build now selects `matmul_mxfp4_mma_fold`; the W4A16 baseline predates it) and GDN default-on
(#378) are both accepted main defaults, so the baselines record what main produces now and what `qualify` plans (`qualify` ignores a
baseline whose `run.prefill_chunk` differs from the plan's chunk, so baselines at 256 would never be used).

Steps ordered by the coordinator:
1. W4A4: `qualify` defines the run (baselines moved away, acc-short + envelope stages, others skipped, as `scripts/gdn-default/stages.sh` `rec`), then install and rerun `qualify` (expect PASS). If `qualify` does not cover the W4A4 golden by itself: the explicit accuracy command at chunk 2048 with the act mode `qualify` would plan (say which).
2. W4A16: the explicit accuracy command of `2026-10-25-ac-27b-accuracy-gate.md` section 7 with `--mxfp4-act none` at chunk 2048, write baseline and envelope, gate again (expect PASS).
3. The 27B's own change-vs-base number for the GDN flip, the method of `2026-11-14-hip-gdn-chunk.md` section 4 (`scripts/gdn-chunk/accuracy-window.sh`): same binary, `INTARSIA_GDN_CHUNK=0` as base, `--self-vs-ref-noise`, mean KL as a ratio of the reference-noise limit (the 35B got 1.09x short, 1.26x long). W4A16 at chunk 2048 at least; W4A4 too if it fits.
4. This doc keeps the step-1 table below, then the re-record results and the step-3 ratios. LOG row, a pointer line in `2026-11-15-hip-gdn-default.md`, PR (not merged).

Fleet: tatooine, one R9700 (tp=1), radiance down only inside fleetq windows (under 45 min, EXIT-trap restore). Checkpoint `/mnt/models/Qwen3.8-27B-Quark-AWQ-MXFP4` (the original runs used the HF-cache snapshot 5233554c, same weights), goldens from `/data/models/intarsia/golden` (sha256 equal to the provenance files). Build: head f219eec (scripts only on top of main 5c9cb111e), stamp `head=f219eec code-tree=aa4946c84c9a dirty=no`. Scripts: `scripts/b27-rerecord/`.

## Step 1: the CURRENT (stale) baselines, chunk 256 (raw/step1-chunk256)

Gates with `--baseline` + `--noise-envelope` + `--pre-change <baseline head>` `--allow-same-code`, `--mode both --decode-positions 16 --max-seq 4096 --mem-budget 29`, **`--prefill-chunk 256`** (the baselines' chunk; the recipe default is now 2048 since 98f260f96, 2026-10-27, so without the flag the gate refuses: "recorded with a different configuration"), `--mxfp4-act dynamic` (W4A4; the HIP 27B recipe default is now `fp8`) or `none` (W4A16).

| run | baseline gate | prefill mean KL21 (baseline -> this run, limit) | prefill top-1 | decode mean KL (baseline -> this run, limit) | floor (golden) |
|---|---|---|---:|---|---|
| W4A4, `INTARSIA_GDN_CHUNK=0` (1a) | **PASS** | 0.17153 -> 0.17160 (0.19726) | 91.95 % (= baseline) | 0.2106 -> 0.2102 (0.2422) | PASS (decode report-only) |
| W4A16, `INTARSIA_GDN_CHUNK=0` (1a) | **FAIL** (7 of 10 prompts) | 1.385e-3 -> 1.635e-3 (1.592e-3) | 99.44 % (baseline 99.38) | 8.30e-4 -> 1.218e-3 (9.54e-4) | PASS |
| W4A4, chunk on (1b) | **FAIL** | 0.17153 -> 0.19208 (0.19726, ok) | 90.56 % (baseline 91.95, limit 91.65, FAIL) | 0.2106 -> 0.2544 (0.2422, FAIL); top-5 81.1 % (limit 81.75, FAIL) | PASS |
| W4A16, chunk on (1b) | **FAIL** (8 of 10 prompts) | 1.385e-3 -> 1.713e-3 (1.592e-3) | 99.58 % | 8.30e-4 -> 1.343e-3 (9.54e-4) | PASS |

* The setup is sound: W4A4 with the knob off reproduces its 2026-10-25 baseline to four digits (0.17160 vs 0.17153; top-1 identical).
* **The W4A16 baseline was stale before #378.** It was recorded 2026-10-25 at `bbc0cde` with `matmul_mxfp4_gemv+matmul_mxfp4_mma`; the build now plans `matmul_mxfp4_gemv+matmul_mxfp4_gemv_rows+matmul_mxfp4_mma_fold` (the W4A4 baseline, recorded later at `092c961`, already had `mma_fold`, and reproduces). With the GDN knob off the W4A16 run still reads 1.635e-3 against 1.385e-3, so this staleness is independent of the chunked delta rule. The run's own log lists the kernel differences (`s1-w4a16-chunk0.err`, "differences from the baseline's run").
* Change from the GDN knob alone (same binary, chunk 256): W4A16 prefill 1.635e-3 -> 1.713e-3 (+4.8 %), decode 1.218e-3 -> 1.343e-3; W4A4 prefill 0.1716 -> 0.1921 (+12 %), decode 0.2102 -> 0.2544 (+21 %), prefill top-1 91.95 -> 90.56 %. The absolute floors against the goldens pass in all four runs.

## Re-record (coordinator steps 1-2) at the recipe default chunk 2048 (raw/step2-3-chunk2048, raw/qualify-attempt)

**`qualify` cannot plan the 27B on HIP, so both variants were recorded with the explicit accuracy command.** The first attempt (stale W4A4
files moved away, `qualify --backend hip --golden <W4A4 golden>` with acc-short + envelope, the other stages skipped) stopped in its `plan`
stage: `plan show --admit` reports one op, `attn_decode:attn_decode_bf16_splitkv_rows` (the MTP verify-rows attention kernel; recipe default
`--spec mtp --spec-k 4`, 0 calls per decode step), as not admitted, so gate, acc-short and envelope are "blocked by plan"
(raw/qualify-attempt/q-rec-w4a4.out, plan.json). That is a separate finding about `qualify` on the 27B, not changed here. The explicit commands
use the activation mode `qualify` would plan **for the W4A4 golden**: its accuracy stage adds `--mxfp4-act` from the golden's own mode
(`facts::golden_act`: metadata "activations dynamic") instead of the HIP recipe default `fp8`, so `dynamic`; the W4A16 golden is run with
`none`, as in `2026-10-25-ac-27b-accuracy-gate.md` section 7. (`qualify` would grade the HIP default W4A8 against the W4A16 golden; neither
file here is a W4A8 baseline.)

Flags, in the order the baselines' `exec.cmdline` records them: `accuracy --backend hip --model /models/Qwen3.8-27B-Quark-AWQ-MXFP4 --mem-budget 29
--mode both --decode-positions 16 --max-seq 4096 --mxfp4-act {dynamic|none} --golden <golden>` (+ `--ref-noise <sidecar>` for W4A4, as the original)
`--write-baseline B --write-self-golden D`, then `... --noise-ref B --noise-perturb chunk --write-noise-envelope E --allow-same-code`.
No `--prefill-chunk`: the plan reads 2048 from the recipe (`prefill chunk 2048 (recipe qwen3_5_vl, hip)`), recorded in `run.prefill_chunk`.
Build `head=f219eec code-tree=aa4946c84c9a dirty=no` (the baselines' `code`); the script commits after f219eec do not touch the engine.
`--write-self-golden` dumps (0.8 MB each) are not committed.

| file (golden/baselines/) | recorded | prefill (top-1, mean KL21 all / scored) | decode (top-1, mean KL21 all / scored) | envelope |
|---|---|---|---|---|
| `qwen3.8-27b-mxfp4-hfquark-top20.hip.*` (W4A4, chunk 2048) | 21:32 | 90.56 %, 0.1913 / 0.1884 | 88.12 %, 0.2516 / 0.3011 | `prefill chunk 2048 -> 512` |
| `qwen3.8-27b-mxfp4-hfquark-w4a16-top20.hip.*` (W4A16, chunk 2048) | 21:28 | 99.58 %, 1.697e-3 / 1.789e-3 | 99.38 %, 1.345e-3 / 1.488e-3 | `prefill chunk 2048 -> 512` |

Caveats: **no golden prompt crosses a 2048-row chunk boundary (longest 700 tokens), and only the 700-token prompt exceeds 512, so the
`chunk` envelope perturbs one prompt and the gate is nearly a reproduction check**, as in the 2026-10-25 record (256 -> 64). The recorded
chunk differs from the old files (2048 vs 256); the step 1 numbers (256) and these are not a like-for-like pair, only the same-binary,
knob on/off pairs within a chunk are. The W4A4 prefill KL21 on scored rows (0.1884) sits at 0.946x of the sidecar's reference noise
(0.1991) and under the 0.2 floor; its decode floor (0.3011 vs 0.2) FAILs and is report-only by the golden's sidecar (`floor_report_only`), as
since 2026-10-25.

### Gate with the new files (the coordinator's steps 1 and 2, second half) (same build, `--pre-change f219eec --allow-same-code`: a reproducibility gate)

| gate | floor vs golden (absolute) | baseline gate | prefill KL / top-1 | decode KL / top-1 |
|---|---|---|---|---|
| W4A4 `chk-w4a4` | PASS (decode report-only) | **PASS** | 0.19154 / 90.56 % (limit 0.2203) | 0.2430 / 88.12 % (limit 0.2794) |
| W4A16 `chk-w4a16` | PASS | **PASS** | 1.717e-3 / 99.58 % (limit 1.975e-3) | 1.364e-3 / 99.38 % (limit 1.568e-3) |

(`qualify` rerun with the new files: not possible, see above. The repo test `committed_27b_baselines_pair_with_their_envelopes_and_goldens`
reads the four committed files and passes: `cargo test --release -p intarsia-cli committed_27b`, 1 passed.)

## Change vs base (coordinator step 3): the 27B's change vs base for the GDN flip (raw/step2-3-chunk2048/self-*)

Method of `2026-11-14-hip-gdn-chunk.md` section 4: same binary, **base = `INTARSIA_GDN_CHUNK=0`** writing a self-golden (`--write-self-golden`),
**change = default (chunk on)** run `--self-golden base --self-vs-ref-noise` (mean KL21 of the change against the base's logits, all rows, must be at most
1 x the golden's reference noise; every prompt's KL stays a hard check at 3 x its noise), recipe chunk 2048, the same `--mode both
--decode-positions 16 --max-seq 4096` as the baselines; the control is the knob-off binary against its own base (expect ~0).

| variant | prefill change-vs-base mean KL21 | / limit (ref noise) | decode change-vs-base | / limit | flips (prefill, decode) | verdict |
|---|---|---|---|---|---|---|
| **W4A4** (sidecar noise 0.1969, all rows) | **0.1653** | **0.839x** (limit 0.1969) | 0.1091 | 0.554x | 111 of 1441 (7.7 %), 7 of 160 | mean **PASS** both modes; per-prompt check **FAIL**: decode `long_prose_700` KL 0.928 > 3 x noise 0.591 (1.57x; needs prompt factor 4.71); prefill worst prompt 0.43 of its limit |
| W4A16 | 3.204e-4 | no reference-noise record exists | 4.661e-4 | n/a | 4 of 1441 (0.28 %), 0 of 160 | tight self-golden budgets (mean KL <= 1e-4) FAIL, as for any non-bitwise kernel; no ratio |
| controls (knob off vs own base) | 2.6e-12 (W4A4), 4.3e-12 (W4A16) | 0 | 2.9e-11 / 1.1e-11 | 0 | 0, 0 | PASS |

* The 35B for comparison: 1.09x (short golden), 1.26x (long golden) of its reference noise in prefill, both FAIL; the 27B W4A4 is **0.84x in prefill and 0.55x in decode, inside the noise** (the per-prompt hard check fails on one prompt's 16 decode rows).
* **W4A16 has no ratio.** The only 27B reference-noise record is the W4A4 sidecar, hash-bound to the W4A4 golden (a run with it against the W4A16 golden is refused: "ref_noise was measured on golden hash 725eb3ea093c0264, this golden is b2665d1de37eb72c"), and it measures HF W4A4 bf16 against f32, dominated by fp4 activation rounding. There is no f32 W4A16 HF golden (one CPU run of `tools/golden_quark_mxfp4.py` on endor would give it). Reading the absolute number: W4A16 prefill KL 3.2e-4 is 0.19x of the engine's own KL to the W4A16 golden (1.70e-3), decode 4.7e-4 is 0.35x of 1.35e-3; and both are 0.2-0.3 % of the W4A4 noise limit, which is not a valid denominator.
* Which prompts move: every prompt under 64 tokens is bitwise unchanged (KL ~1e-14: count, story, chat_short, chat_system, chat_multiturn), because the chunked kernel needs T >= 64. The change is in `code` (87 tokens, 1.5e-2 prefill KL under W4A4), `science` (79), `chat_reasoning` (90), `long_prose_300` and `long_prose_700`.
* W4A4 amplifies the same kernel difference about 500x against W4A16 (0.165 vs 3.2e-4) because one-ulp differences flip fp4 activation roundings (`2026-10-25-ac-27b-accuracy-gate.md` section 5); the W4A4 base-to-change KL (0.165) is as large as the engine-to-golden KL (0.19), which is why its noise limit (0.197, itself a W4A4 effect) is not exceeded.

## Reading

The W4A4 27B change from the GDN flip is within the reference noise on both modes (0.84x / 0.55x), better than the 35B (1.09x / 1.26x), with one prompt over its per-prompt limit. W4A16 moves by 3e-4 (prefill) and 5e-4 (decode) of KL, 4 top-1 flips in 1441 positions, against a golden gap of 1.7e-3; no noise ratio can be formed for it today. The step 1 table (chunk 256, same binary) shows the same direction at a larger size for W4A4 (prefill KL 0.1716 -> 0.1921 against the recorded baseline), which is the baselines' own run-to-run gate, not the noise ratio.

Not done / not verified: `qualify` on the 27B HIP (plan admission of `attn_decode_bf16_splitkv_rows`); a W4A16 reference-noise record; the W4A8 (fp8) default run is not baselined; multi-chunk prefill on these goldens (no prompt crosses 2048); tp=2.
