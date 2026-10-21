# Metal: 8B fp32 4k golden, `acc-4k` stage, qualify at prefill chunk 2048

> **Update (2026-10-21, docs-accuracy pass, see `docs/decisions/LOG.md`):** Merged to main in `d4c1a146` (Merge branch 'metal-golden-4k' into metal-p3-final); "not pushed" / "nothing merged" wording in this page describes the state when written.

Qwen3-8B BF16 on endor (Apple M5 Max, macOS 27.0). Branch `metal-golden-4k` = `metal-tail-fix` (b1c38d9, `NAX_MIN_M` 16 -> 2)
+ the qualify wiring below (f67c379, then 7447269 / b71ee56 for the baseline-gated `acc-4k`). The P3 build is a scratch merge of that branch with `metal-p3-chunk`
(branch `metal-golden-4k-p3-scratch`, 03ee339, worktree `/opt/code/intarsiaml-golden-4k-p3`; scratch, kept for reproduction, delete when done). Raw data: `docs/results/raw/2026-10-15-metal-golden-4k/`. Nothing merged or pushed.

## Outcome

* **Coverage gap closed.** The 4k golden exists and `qualify` uses it (stage `acc-4k`); the **prefix-reuse gate PASSES at chunk 2048**
  on `chat_reuse_4k` (continue, restore, aligned) for the P3 build. Without the 4k golden the P3 build FAILs `acc-long` on the
  reuse gate exactly as before (`round2/p3-no4k.log`).
* **`acc-4k` is baseline-gated, with no absolute top-1 floor** (the user's decision, after the 98.0 floor of an earlier revision was
  reverted). The class floor top-1 >= 99.5% fails on every build, with identical numbers: 99.28% prefill, 98.96% decode on the
  tail-fix base at chunk 256 and on P3 at 2048, and 99.45% on the CPU engine, so it measures bf16 noise at near-ties of the fp32
  reference, not a regression (section 3). `acc-4k` gates KL <= 1e-3 (measured 2.3e-4 / 2.5e-4), tail mass, non-finite rows,
  prefix reuse, and, when the backend has a recorded baseline for this golden, top-1 / KL / tail per mode and per prompt against that
  baseline plus fixed margins and the noise envelope (section 2). With no matching baseline it is KL-only and says so.
* **`qualify --backend metal` passes at 2048 and at 256.** Round 3 (section 5) ran the final stage: `acc-4k` PASS with the baseline
  (chunk 256) and PASS KL-only with the note "not used (recorded at prefill chunk 256, this run uses 2048)".
* **Accuracy is identical for the two builds, and for chunk 256 and 2048 on either build**, to the printed precision, over 12685
  prefill and 192 decode positions in all four runs (SAFE at 256 and 2048, P3 at 2048 and 256). The noise-envelope deltas of the chunk
  perturbation (256 <-> 2048, every 4k prompt crosses a boundary) are at most 1e-19. P3 changes no 4k number, so the baseline is the same
  at 256 and 2048 (a recorded baseline is still tied to its chunk; section 2).

## 1. The golden

| | |
|---|---|
| file | `/opt/code/intarsiaml-golden-4k/golden/qwen3-8b-4k-fp32-top20.safetensors` (git-ignored; 2,086,924 bytes) |
| sha256 | `7371a77f417c86b9ca50518fe9872e05dc5935a744254e6040a1307cb1a394f8` |
| format | `intarsia-golden-topk-v1`, top-20, 3 prompts, 12685 positions (`long_docs_4k` 4000+64, `long_code_4k` 4000+64, `chat_reuse_4k` 4493+64, `reuse_split` 4447) |
| reference | HF transformers 5.19.0, torch 2.14.1, **fp32 weights and activations, CPU**, model `/Users/josef/Models/Qwen3-8B`, ChatML, 64 greedy continuation tokens per prompt |
| command | `fleetq run --host endor --mem 50 -- uv run python -u golden.py --model /Users/josef/Models/Qwen3-8B --device cpu --dtype float32 --topk 20 --prompts prompts/golden_4k.jsonl --out ../golden/qwen3-8b-4k-fp32-top20.safetensors` (fleetq #1977, in `tools/`) |
| runtime / memory | generation 303 s (99 / 104 / 100 s per prompt); the whole job 4220 s of wall clock, almost all of it queued behind #1975; max RSS 41.5 GB (budget 50 GB) |

Top-20 follows `docs/accuracy.md` ("HF (8B), top-20 only"). Consequence: the gated KL on this golden is the 21-way KL (a lower bound
of the full-vocabulary KL, doc "How a top-k golden is compared"), with the tail-mass check; it is not the exact full-vocabulary KL
that the guardrails on the 8 short / 10 long full-vocabulary goldens use. The file is outside git like every other golden. It lives in this worktree's
ignored `golden/`; `qualify` finds it by name only from there, from `--golden-dir`, or from the main checkout's `golden/` (which
I was told not to touch). It was copied (file only) into `/opt/code/intarsiaml/golden/` at the user's request, which makes it the default for every worktree.
Also reproducible: the command above is deterministic CPU fp32 (not re-run to confirm bit-identity).

## 2. Qualify wiring (files)

* `crates/intarsia-cli/src/cmd/qualify/facts.rs`: `find_golden_4k()` (a `<model>-4k...` file, fp32 first); `find_golden` shares one
  `find_scored` with it, scoring and first-directory-wins ties unchanged. Unit test extended.
* `crates/intarsia-cli/src/cmd/qualify.rs`: `--golden-4k FILE` (else found by name in the golden dirs, never made: the HF run is
  ~5 min of CPU at 41 GB); new stage `acc-4k` (`accuracy --mode all --decode-positions 64 --max-seq 8192 --min-top1 0`, i.e. no absolute top-1 floor; the
  class KL / tail / non-finite floors and the reuse gate stay), and `Ctx::acc_4k_baseline`: it looks for
  `<golden stem>.<backend>.baseline.json` (+ `.envelope.json`) next to the golden in `baselines/`, in the repo's `golden/baselines/`
  and in the main checkout's, and uses it (`--baseline`, `--noise-envelope`, `--pre-change <its commit>`, `--allow-same-code`) only
  when it was recorded for the same backend, prefill chunk, decode window and golden file size. Otherwise the stage is KL-only
  and the note says why; it never fails on top-1 for want of a baseline. `acc-long` runs `--mode both` instead of `all` **only when**
  a 4k golden is found and differs from the long golden (the reuse gate then lives in `acc-4k`).
* `crates/intarsia-cli/src/cmd/qualify/report.rs`: `acc-4k` listed with `acc-short` / `acc-long`.
* `docs/adding-a-model.md`, `docs/accuracy.md`: stage table, golden discovery, where the golden lives.

**Backward compatibility (CUDA / HIP / cpu / other models):** `intarsia-core` and the engine are untouched; only `qualify` changed.
With no `<model>-4k...` golden in the golden dirs (true for every other model and box today; the planned spark1 8B top-20 was never
built) `acc-long` is unchanged (`--mode all`, same golden) and `acc-4k` is SKIP, which does not change the overall verdict. Verified:
`round2/safe-no4k` (acc stages only, no 4k golden) PASS with `acc-4k` SKIP, `acc-long` numbers equal to the 4k-golden run. The one
visible difference on those boxes is an extra SKIP row in the report. Someone copying a 4k golden to a CUDA box changes `acc-long`
there to `--mode both`, with the reuse gate moving to `acc-4k` (where it also runs at chunk 256).

## 3. Accuracy, `intarsia accuracy --mode all --decode-positions 64 --max-seq 8192` on the 4k golden

| build, chunk | prefill top-1 | prefill KL(21) | decode top-1 | decode KL(21) | reuse gate | class floor (99.5 / 1e-3) |
|---|---|---|---|---|---|---|
| SAFE (tail-fix + wiring), 256 (default) | 99.28% | 2.274e-4 | 98.96% | 2.548e-4 | PASS (9 cases: all three prompts at 256; at 2048 only `chat_reuse_4k` runs, the raw prompts print "skipped", split 2000 < 2048) | top-1 FAIL, KL ok |
| SAFE, `--prefill-chunk 2048` | 99.28% | 2.274e-4 | 98.96% | 2.548e-4 | PASS | same |
| P3, 2048 (default) | 99.28% | 2.274e-4 | 98.96% | 2.548e-4 | PASS | same |
| P3, `--prefill-chunk 256` | 99.28% | 2.274e-4 | 98.96% | 2.548e-4 | PASS | same |

Per prompt (prefill top-1 / KL21): `long_docs_4k` 99.34% / 1.83e-4, `long_code_4k` 99.53% / 1.83e-4, `chat_reuse_4k` 99.01% / 3.07e-4.
The one reuse row beyond the rounding tolerance, `chat_reuse_4k` continue, is accepted by the existing logit criterion (warm and cold
diverge at token 24, rounding-scale KL; identical in all four runs). The existing goldens (qualify stages `acc-short` / `acc-long`),
both builds: short prefill 100.00% / 2.83e-4, decode 100.00% / 3.31e-4; long prefill 99.72% / 3.70e-4, decode 99.80% / 3.54e-4,
all above the merge bar (KL 5e-4, top-1 99.6%). **The merge bar (top-1 99.6%) cannot be applied to the 4k golden**: no build reaches it.

### The top-1 floor

* Positions are near-ties for the fp32 reference. Of the 91 prefill misses, the reference's top-1 minus top-2 gap is under 0.1
  nats at 82 (90%) and under 0.02 at 43 (47%); median 0.022 nats, max 0.45. Such gaps hold 4.2% of all 12685 positions
  (`posdump.tsv`; 91 = 12685 x (1 - 0.9928)). A bf16 logit noise of a few 0.01 nats flips them.
* The intarsia **cpu** engine (bf16 weights, independent kernels, `--backend cpu --mode prefill`, fleetq #1981): top-1 99.45%,
  KL21 2.05e-4, also below 99.5. Metal is 0.17 points lower (21 more of 12685 flips) at 1.1x the CPU KL.
* The decode window is 192 positions: one flip is 0.52 points, so a floor of 99.5 allows 0 flips; the measured two flips is 98.96%.

So the class floor top-1 is not informative on 4k prompts, while KL (2.3e-4, below the long golden's 3.7e-4), the tail mass
(excess -1e-4) and the zero non-finite rows are. An intermediate revision of this branch fixed top-1 at 98.0%; it was reverted for the
baseline gate below (the decode window alone gives that floor a few percent chance of failing on noise).

### The baseline gate and why it does not fail on noise

Per mode and per prompt, `accuracy --baseline --noise-envelope` fails a run only if it is worse than the recorded run by more than
`max(fixed margin, 2 x the prompt's envelope delta)`; the fixed margins are KL x1.75 + 1e-5, top-1 -1 point (at least one position),
|tail diff| +0.002. Against the numbers measured here a correct build does not trip it: SAFE at 256 and 2048 and P3 at 2048 and 256 are
identical to the printed precision, the chunk envelope is zero, and the self-checks pass; the tail-fix itself moved the existing
goldens' decode KL by 5% (short 3.496 -> 3.313e-4, long 3.517 -> 3.371e-4, per the coordinator), well inside the x1.75 margin. A
control in the final run (`metal-p3-final`, `accuracy/4k-control`) hands the gate a baseline that is better than the build by 1.5
top-1 points and half the KL, and the gate must FAIL it.

## 4. Baselines and envelopes (Metal, on tail-fix)

Recorded like the P3 baselines (`--mode both --decode-positions 64 --max-seq 8192`, perturbation = the other chunk, `--allow-same-code`,
self-check with `--pre-change <head>`; `run.sh`):

| file | build / chunk | contents |
|---|---|---|
| `golden/baselines/qwen3-8b-4k-fp32-top20.metal.baseline.json` | SAFE, 256 (tracked, head=f67c379 code-tree=b7b7b80a8f67) | prefill top-1 99.283%, KL21 2.2737e-4; decode 98.958%, 2.5475e-4 |
| `golden/baselines/qwen3-8b-4k-fp32-top20.metal.envelope.json` | perturbation chunk 256 -> 2048 | nothing moved: every per-prompt delta is 0 to 1e-19, and the tool warns that the perturbation did not touch the prompts, so the fixed margins apply |
| `docs/results/raw/2026-10-15-metal-golden-4k/p3-baselines/*.json` | P3, 2048 (head=0af824d), perturbation 2048 -> 256 | same numbers |

A recorded baseline belongs to one prefill chunk; `acc-4k` uses it only at that chunk (this branch's default is 256). On the P3 tip the
pair is re-recorded at 2048 (`metal-p3-final`, `docs/results/2026-10-15-metal-p3-final.md`).

## 5. Qualify verdicts (fresh, `--backend metal --bandwidth-gbs 614`, goldens: short, long, 4k)

| round | build | chunk | verdict | notes |
|---|---|---|---|---|
| 2 | SAFE (tail-fix + wiring + 98.0 floor, since reverted) | 256 | PASS | full run |
| 2 | P3 scratch merge | 2048 | PASS | full run |
| 2 | SAFE, no 4k golden | 256 | PASS | `acc-long` `--mode all` PASS, `acc-4k` SKIP |
| 2 | P3, no 4k golden | 2048 | **FAIL** | `acc-long` prefix reuse (SKIPPED counts as fail), the original failure |
| 3 | SAFE (b71ee56, baseline-gated), acc stages | 256 | **PASS** | `acc-4k` gated against the recorded baseline + envelope: floor PASS, baseline verdict PASS, reuse PASS |
| 3 | P3 scratch merge (59af184), acc stages | 2048 | **PASS** | baseline recorded at 256 is "not used (recorded at prefill chunk 256, this run uses 2048)": KL-only PASS, reuse PASS |

Round 1 (`safe/`, `p3/`, no top-1 override): both builds FAIL only on `acc-4k` top-1 (99.28% / 98.96% < 99.5), which found the issue in
section 3. Round 3 skipped the stages that do not touch this change (kernels, gate, bench, bench-long, envelope, acc-short); the full
run on the final P3 tip is in `docs/results/2026-10-15-metal-p3-final.md`. Speeds in the qualify reports are not benchmark data.

## 6. Notes for merging

1. `metal-golden-4k` is a prerequisite of merging P3: without the 4k golden and this wiring, P3's qualify FAILs at chunk 2048.
2. The baseline pair for 4k is per chunk. P3 must carry the 2048 pair (in `metal-p3-final`), not the 256 pair of this branch.
3. P3's committed short/long Metal baselines predate the tail-fix and are re-recorded in `metal-p3-final`.

## 7. Method and ids

* fleetq jobs (all released): #1977 golden generation (CPU, 50 GB); #1979 accuracy + baselines + round-1 qualify (exclusive); #1981 CPU
  engine on the 4k golden (30 GB); #1982 Metal position dump (12 GB); #1984 round-2 qualify (exclusive); #1986 round-3 `acc-4k` check
  (exclusive). (#1976 was my first launch with full-vocabulary output, cancelled before it started: the doc says top-20.)
* Builds, stamped (`scripts/tree-stamp.sh`, `dirty=no`, `cargo build --release -p intarsia-cli --features metal`): round 1 SAFE head=f67c379
  code-tree=b7b7b80a8f67, P3 head=0af824d code-tree=fdde68198b2c; round 2 SAFE head=cd4c7d3, P3 head=03ee339; round 3 SAFE head=b71ee56
  code-tree=20fc48756134, P3 head=59af184 code-tree=1e0dfabdf84a.
* `cargo test -p intarsia-cli`: 75 + 1 passed, 0 failed (includes `stage_table_is_consistent`, the golden discovery and baseline discovery
  tests, report tests). `cargo fmt --check` shows pre-existing differences in `accuracy/baseline.rs` only; none in the files I changed.
