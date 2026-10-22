# fx1: an independent accuracy reference for the 35B, and a gate that kills a 1 % scale bug (2026-10-22)

Lane fx1 answers finding 1 and section D of the adversarial review (`fable-review-2026-10-22.md`, scratchpad). Base
`origin/main` cd617de2 (PR #70 merged: the 35B runs on Metal). Model `nvidia/Qwen3.6-35B-A3B-NVFP4` (MoE, GDN hybrid,
NVFP4 experts, fp8 attention). Only `intarsia-cli` qualify code changes: `git diff --stat origin/main -- crates kernels
recipes` lists `qualify.rs`, `qualify/{facts,report,state}.rs` and `onboard.rs` only. Engine numerics and defaults are
untouched (section 6 checks the logits).

## 1. The problem, measured

The 35B's only reference was a vLLM golden (`guidance` on spark1, fp8 KV cache). On the short set, a 1 % row-parallel
scale bug passes every shipped gate, on every backend:

| run (acc-short args, vLLM short golden, 4-bit floor KL21 <= 3e-2) | prefill KL21 scored | decode KL21 scored | old verdict |
|---|---|---|---|
| Metal clean | 5.78e-3 | 1.12e-2 | PASS |
| Metal x1.01 mutant | 4.83e-3 | 1.50e-2 | PASS |
| Metal x1.005 mutant | 4.83e-3 | 8.25e-3 | PASS |
| HIP tp=1 clean | 5.98e-3 | 1.21e-2 | PASS |
| HIP tp=1 x1.01 mutant | 4.80e-3 | 1.28e-2 | PASS |
| HIP tp=1 x1.005 mutant | 4.94e-3 | 1.23e-2 | PASS |
| HIP tp=2 clean | 5.97e-3 | 7.10e-3 | PASS |
| HIP tp=2 `INTARSIA_TP_TEST_BUG=scale:1.01` | 4.91e-3 | 1.20e-2 | PASS |

The mutants read *closer* to vLLM than clean does. The vLLM golden is one non-deterministic sample. tatooine's
`/models/golden` copy of the "same" short golden is another sample (sha256 89d650a8, against the canonical b6b51e1b);
it disagrees with its own `ref_noise` sidecar, so the accuracy tool refuses it (window 1, first attempt). Without a
committed baseline, qualify fell back to this floor without saying so (`acc_4k_baseline`, `qualify.rs:1847` on main).

**The mutant.** `scripts/fx1-oracle/make_scale_mutant.py` writes an overlay model directory that symlinks every file and
adds one shard. That shard multiplies by F the per-tensor scale of every row-parallel output projection on odd layers:
fp8 `o_proj` / `out_proj` `weight_scale`, NVFP4 expert and shared-expert `down_proj` `weight_scale_2`, 5160 scalars.
The loader takes the last file's copy of a tensor, so no weight file and no engine code changes. It works at tp=1 on
every backend, Metal included, where `INTARSIA_TP_TEST_BUG` cannot run. It is not the TP hook, which scales rank 1's
partial sums of every layer. Both are the same class of bug, a 1 % scale error on part of the row-parallel output, and
both move the logits by the same amount: drift KL21 9.2e-3 (mutant) against 1.15e-2 (TP hook) on HIP.

## 2. The oracle

`intarsia accuracy --backend cpu` is the f64 CPU oracle of M3: f32 accumulation, NVFP4 / MoE / GDN in f64, the same
weights. It runs teacher-forced prefill only and writes a top-64 self-golden.

* **Prompt set** (`golden/oracle/qwen36-35b-a3b-nvfp4-oracle-set-vllm-top20.safetensors`, made by
  `scripts/fx1-oracle/merge_goldens.py`): the 10 short prompts plus `long_docs_4k` from the long golden, 11 prompts and
  5505 rows. The short set alone (1441 rows, longest prompt 700 tokens) crosses no prefill-chunk boundary at the
  default chunk ("multi-chunk prefill is untested", the tool warns). `long_docs_4k` adds the long positions and the
  chunk boundaries for the cost of one prompt.
* **Oracle** (`golden/oracle/qwen36-35b-a3b-nvfp4-cpu-oracle.safetensors`, 2.8 MB, committed): made on endor (M5 Max,
  40 GB budget, chunk 256). The short set took 10 min 17 s; the oracle set took 20 min 32 s. qualify finds it by model
  name. Without one it makes it with the cpu backend from the short golden's prompts. That fallback happens only for
  MoE / hybrid / quantized models whose golden is not fp32.
* **Determinism**: the oracle is deterministic. Its distance to vLLM is the vLLM sample's noise: KL21 on the top-20
  support is 1.42e-2 against the oracle, while clean Metal reads 8.90e-3 on the same rows.

## 3. The floor, measured before it was chosen

The quantity is KL(oracle || engine) over the oracle's top-64 plus a tail bucket, on scored prefill rows (ChatML system
and user turns hidden, n = 5343 of 5505), the same number `intarsia accuracy` gates.

| engine vs CPU oracle (oracle set) | KL scored | top-1 |
|---|---|---|
| Metal clean (also at chunk 512: identical) | 9.88e-3 | 96.26 % |
| HIP tp=1 clean (GPU 0, GPU 0 again, GPU 1: identical) | 9.86e-3 | 95.86 % |
| HIP tp=2 clean | 1.12e-2 | 95.82 % |
| CUDA | not measured (spark1 queue, section 7) | |
| Metal x1.01 / x1.005 mutant | 9.05e-3 / 1.00e-2 | 96.44 / 95.93 % |
| HIP tp=1 x1.01 mutant | 9.51e-3 | 96.29 % |
| HIP tp=2 `scale:1.01` | 1.05e-2 | 96.04 % |

The clean spread (9.9e-3 to 1.1e-2) is bf16 activation rounding against f32 / f64 math on the same NVFP4 weights. The
spread is not noise, because each build reads the same value every run. The floor is **2e-2**, about 1.8x the largest clean
reading (HIP tp=2): margin for CUDA and for a future backend's different rounding, and still far below a wrong kernel
(10x and more; the old 4-bit floor sits at 3e-2 against a reference that is itself off by 1.4e-2). Top-1 is not gated:
the 4 % of rows where clean builds disagree with the oracle's argmax move by a few rows between correct builds
(HIP tp=1 / tp=2 / Metal: 95.86 / 95.82 / 96.26 %), and a mutant can read higher than clean.

**The floor does not see the 1 % bug, and nothing that compares two implementations can.** Every mutant reads inside
the clean spread, some of them closer to the oracle than clean. The bug's effect on the logits (KL ~1e-2) is the size of
the bf16-vs-f64 difference itself. Section 4 describes what does see it. A logit-sharpness statistic was tried as well:
the least-squares slope of the engine's log-probs on the oracle's over the oracle top-8. It reads +7e-3 for the Metal
mutant against Metal clean, but clean HIP already sits at +9e-3 against the oracle, so it does not separate either. It
was dropped.

## 4. The kill: a drift check against a recorded same-backend dump

The engine is deterministic per backend and tp degree: HIP clean reads KL 0 run to run and GPU 0 to GPU 1, and Metal
reads 2.3e-13. `acc-oracle` therefore runs in two parts:

1. **Floor**: the engine against the oracle at <= 2e-2. This guards absolute accuracy. The run also writes the engine's
   own top-64 dump on the oracle prompts.
2. **Drift check**: the engine against the recorded dump of clean main on the same backend
   (`golden/baselines/<oracle stem>.<backend>[-tpN].selfgolden.safetensors`), with the self-golden budgets (mean KL21
   <= 1e-4). Clean main reads 0. The x1.01 mutant reads 9.2e-3 on HIP and 9.7e-3 on Metal, about 100x the budget.

The `tp` stage runs both parts at tp=N against the `-tpN` dump. `INTARSIA_TP_TEST_BUG=scale:1.01` reads 1.15e-2 against
the tp=2 dump.

**Policy.** The dumps pin the logits of main. A change that is meant to move the numerics (a new kernel, a different
accumulation order) fails the drift check by design. Its PR re-records the dumps of every backend it touches: run
qualify on the new main, copy `baselines/*.selfgolden.safetensors` from the report into `golden/baselines/`, and justify
the floor reading in the PR. The floor stays as the guard of absolute accuracy.

## 5. MISSING-BASELINE

A model whose golden is not a full-precision reference reads **MISSING-BASELINE** (exit 1, counted as failing) instead
of a silent floor-only PASS. Such a model is MoE, hybrid or quantized, and its golden has no `dtype = float32`
(`Ctx::needs_baseline`). The rule applies in three places:

* `acc-short` / `acc-4k`: no backend baseline + envelope applies (none found, or one recorded for another chunk, decode
  window or golden file).
* `acc-oracle`: no recorded dump for this backend.
* `tp`: no recorded `-tpN` dump.

A dense full-precision model keeps the floor-only fallback, unchanged. `acc-long` is unchanged: it never had a baseline
mechanism, and the review finding is about `acc-short` / `acc-4k`. A baseline keyed to the canonical short golden
(b6b51e1b) does not apply on a box that holds another vLLM sample under the same name, as tatooine's `/models/golden`
does. Such a run reads MISSING-BASELINE until its golden is replaced with the canonical file.

## 6. Acceptance

qualify, quick section (kernels, bench, acc-long, acc-4k, bench-long and ab skipped for time), `--golden` the canonical
short golden, `--oracle-golden` the committed oracle:

| backend | no baseline committed | clean main | x1.01 mutant | tp=2 `scale:1.01` |
|---|---|---|---|---|
| HIP (tatooine, tp=2 stage included) | MISSING-BASELINE, exit 1 (acc-short, acc-oracle, tp) | PASS, exit 0 | FAIL, exit 1: acc-oracle drift KL 9.19e-3; acc-short baseline gate (top-1, 2 of 10 prompts beyond the envelope) | FAIL, exit 1: tp drift KL 1.15e-2 |
| Metal (endor) | MISSING-BASELINE, exit 1 (acc-short, acc-oracle) | PASS, exit 0, twice (drift 2.3e-13) | FAIL, exit 1: acc-oracle drift KL 9.70e-3; acc-short baseline gate (2 of 10 prompts) | n/a (tp=1 only) |
| CUDA (spark1) | not run | not run | not run | not run |

Logs: HIP window 4 (fleetq #3769), Metal job #3744, under the lane's scratchpad (`fx1/tatooine/w4`, `fx1/acc`).

**Logits unchanged.** The Metal dump of main + fx1 equals the m35 binary's dump (PR #70 head before merge) at KL 0 over
5505 rows. The HIP dump after the `accuracy.rs` experiment equals the one before it at KL 0. The diff touches no engine
crate.

## 7. Not done

* **CUDA.** The job (`scripts/fx1-oracle/cuda-job.sh`) builds fx1, runs no-baseline, installs the recorded files, then
  runs clean twice and the mutant. It is queued on spark1 behind #3555, which waits for a spark2 reboot. Requirement 3
  asks for committed CUDA baselines; they are not committed, and a CUDA qualify of the 35B reads MISSING-BASELINE until
  they are. CUDA determinism of the MoE path, which the drift check relies on, is unmeasured.
* The oracle covers prefill only. Decode is gated by the acc-short baseline + envelope, which caught the mutant on both
  backends. Decode has no oracle.
