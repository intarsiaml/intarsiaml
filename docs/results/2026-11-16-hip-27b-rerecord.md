# HIP: re-record the four Qwen3.8-27B accuracy baselines after the chunked GDN default (lane b27-rerecord)

Status: 2026-11-16, in progress (this file is written before the re-record, so the decision survives context compaction).

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
