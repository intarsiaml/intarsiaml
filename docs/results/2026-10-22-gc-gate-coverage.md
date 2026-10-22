# gc: gate coverage for the 35B (4k golden, CUDA tp=2 dump)

Lane `gc`, 2026-10-22. Closes the two gaps lanes FX1 and fx10 reported ([fx1](2026-10-22-fx1-oracle-gate.md) section 7,
[fx10](2026-10-22-fx10-metal-35b-tests.md)). Only data files and scripts change; no engine or qualify code.

## 1. 35B 4k golden

`golden/qwen36-35b-a3b-nvfp4-vllm-4k-top20.safetensors` (2,100,484 bytes, 3 prompts from `tools/prompts/golden_4k.jsonl`:
`long_docs_4k`, `long_code_4k`, `chat_reuse_4k`; 64 greedy continuation tokens each, top-20) with the sidecar
`golden/qwen36-35b-a3b-nvfp4-vllm-4k-top20.safetensors.provenance.json` (sha256 `304b3dc7...5034`, fnv1a64, source, date).
Made by `scripts/onboard/make-goldens.sh` on the Mac (uv, no numpy problem) against the spark1 vLLM `guidance`
(vLLM 0.26.1rc1.dev602, fp8 KV cache, the server stayed up; fleetq #4144, non-exclusive, queries only, 24 s for the 4k
set). The file is git-ignored by `golden/*.safetensors` like the 35B short and long goldens, so it is committed with
`git add -f` (2 MB), the same way the oracle files in `golden/oracle/` are. qualify finds it by name (`find_golden_4k`:
`<model name>-4k...`); `--model-name Qwen3.6-35B-A3B-NVFP4` resolved it on both backends below.
vLLM is not bit-deterministic run to run; no `ref_noise` sidecar was measured for this golden, so `acc-4k` gates on the
recorded per-backend baseline (below) and on fixed margins.

## 2. acc-4k on main

Main `acba5987` code (only data files differ). `qualify` with every stage but `acc-4k` skipped (prerequisites doctor,
recipe, plan still run). Per backend: q-0 had no baseline for the 4k golden, recorded one, installed it as
`golden/baselines/qwen36-35b-a3b-nvfp4-vllm-4k-top20.<backend>.baseline.json` (same naming as the 8B 4k one), then q-1
(and q-1b) ran against it.

| backend | q-0 (no 4k baseline) | q-1 / q-1b against the recorded baseline | floor | baseline gate | prefix reuse |
|---|---|---|---|---|---|
| Metal (endor, fleetq #4196, non-exclusive 45 GB, chunk 256) | FAIL (prefix reuse) | FAIL, FAIL (identical) | PASS | PASS | FAIL |
| CUDA (spark1, vLLM up, fleetq #4207) | SKIP | SKIP | n/a | n/a | n/a |

CUDA: **no verdict.** Both runs read `SKIP (the 4k golden is the long golden: covered by acc-long)`: with no 35B long golden
in the tree, qualify's long-golden lookup falls back to the `-4k` file (`find_golden`, score 60), `acc-4k` sees the same
file twice and skips. Metal ran because its lookup found no long golden at all (or a different one); the CUDA copy of the tree
had only the short and 4k goldens. The fix is to put the long golden next to them (`acc4k-cuda.sh` now copies it,
`LONG=`); the rerun (#4216) aborted at its own memory guard (22 GB available on spark1, needs 34, other jobs were running) at
11:21, too close to the 11:30 stop to wait again. This is a qualify quirk worth a fix on its own: a model with a 4k golden
and no long golden never grades `acc-4k` (it is graded as `acc-long` instead, which is the intent, but only when the
long-golden stage runs). Raw run 1: [raw/cuda-acc4k-run1-skipped](2026-10-22-gc-gate-coverage/raw/cuda-acc4k-run1-skipped).

Metal: the only failing case is `long_docs_4k` `continue` (split 2000, 2015 tokens cached): the streams diverge at
token 8, "warm margin 0.346, cold margin 0.132, shared rows differ by 0.252 nats, KL up to 2.9e-2; fixed rounding scale
0.250 nats, KL 1e-2". `restore`, `aligned` and every `long_code_4k` / `chat_reuse_4k` case pass (aligned: bit-identical
logits). Prefill KL 1.24e-2 (scored rows, floor 3e-2), top-1 93.14 %, decode KL 6.9e-3, all PASS. This is a **finding on
main, not fixed here**: the unaligned-continue case is the same prefix-reuse differences the HIP onboard run saw
([on-onboard](2026-10-22-on-onboard.md), HIP `acc-4k`). A recorded baseline from a run whose reuse gate failed is still a
valid KL / top-1 baseline (the baseline mechanism ignores the reuse mode), but re-record it if the reuse behaviour
changes numerics. Raw: [raw/metal-acc4k](2026-10-22-gc-gate-coverage/raw/metal-acc4k).

## 3. CUDA tp=2 drift dump

**Not recorded: the job never ran.** The CUDA tp=2 job (`scripts/tp2-cuda.sh`: ONE fleetq job over spark1 + spark2, vLLM up,
`--mem spark1=20,spark2=20`; qualify's `--bin` is `scripts/tpshim.py`, which starts the matching `tp-worker` on spark2 and adds
`--tp-remote`, because qualify itself cannot pass `--tp-remote`) waited in the queue behind exclusive spark1 jobs
(#4171 `ck`, 46+ min overdue) from 09:50 to the 11:30 hard stop, and was cancelled at 11:12 because a 3-run job
(record, clean, `INTARSIA_TP_TEST_BUG=scale:1.01` on the worker) needs about 30 min. No `*.cuda-tp2.selfgolden.safetensors`
is committed and the CUDA tp stage still reads MISSING-BASELINE. The scripts are untested on the boxes beyond a binary
build (the main-code CUDA binary `~/intarsia-tp2/bin/gc-intarsia` is on both Sparks, fleetq #4145). Untested parts:
the shim's worker flag filter (derived from `tp-worker --help`), `plan show --tp 2` in the tp stage (rank 0 alone has
one GPU), and the worker env for the mutant.

## 4. Not done / uncertain

* CUDA tp=2 dump and the tp-stage clean / mutant verdicts (section 3).
* CUDA acc-4k verdict and baseline (no `...4k-top20.cuda.baseline.json` committed): rerun `acc4k-cuda.sh` with the long golden present and 34+ GB free on spark1.

Scripts: `2026-10-22-gc-gate-coverage/scripts/` (`acc4k-cuda.sh`, `acc4k-metal.sh`, `tp2-cuda.sh`, `tpshim.py`).
