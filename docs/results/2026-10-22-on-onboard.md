# On: `intarsia onboard` on CUDA (spark1) and HIP (tatooine) against the 2 h / 4 h targets

**Verdict: both targets are met by a wide margin for a model whose recipe exists.** Qwen3.6-35B-A3B-NVFP4, weights already on the box: the quick numbers were published 7 min 24 s (CUDA) and 1 min 24 s (HIP) after `onboard` started, and the whole pass (quick plus in-depth) took 15 min 24 s on CUDA and 2 min 47 s on HIP. On CUDA all gates PASS. On HIP the in-depth tier reports a real gate FAIL (`acc-4k` prefix reuse), which is an engine finding, not an onboarding fault. Three onboard/qualify faults were found and fixed (below). Real wall-clock risk is outside the tool: the fleetq queue (4.5 h on the Sparks tonight), a recipe that needs a human, a missing kernel.

Branch `lane-on/onboard-cuda-hip`. Docs: `docs/adding-a-model.md` section 0 (measured timings and gotchas). Raw logs: `docs/results/2026-10-22-on-onboard/raw/`.

## Model choice (step 1)

Weights already on the boxes (listed under fleetq slots, nothing downloaded):

* spark1 (`~/.cache/huggingface/hub`): Qwen3.6-35B-A3B-NVFP4, Qwen3-8B, Nemotron/Gemma/GLM/DeepSeek checkpoints (other lanes' downloads; no recipe, the DeepSeek and GLM ones are large and the spark2 disk is reserved), Qwen3.8-27B variants.
* tatooine (`/data/models/intarsia` as `/models`): Qwen3-8B, Qwen3.6-35B-A3B-NVFP4.

Chosen: **Qwen3.6-35B-A3B-NVFP4** on both. It is not Qwen3-8B dense: a hybrid GatedDeltaNet + MoE model, NVFP4 with FP8 parts (recipe `qwen3_5_moe`, 61/61 tensor templates resolve), and it runs on both backends. It has cached goldens, so for the CUDA run and the fresh HIP run the goldens were made again (below) to exercise the generation step. A model with no recipe (which would exercise `recipe new` and the exit-3 path) is not on either box, so that path was **not run here**.

## Per-stage timing

All times from `onboard`'s own stage timeline (`raw/*/report/in-depth.md`). Targets: quick numbers within 2 h, in-depth within 4 h, of the weights being present.

| Stage | CUDA spark1 (`35b-cuda`) | HIP tatooine, fresh goldens (`35b-hip-fresh`) | HIP tatooine, cached goldens (`35b-hip-r3`) |
|---|---|---|---|
| goldens made (short + long + 4k, on the Mac from the spark1 vLLM) | 29 s (10 + 5 + 14) | reused from the CUDA run | cached, none made |
| build (dependency cache warm) | 44 s | 6 s (CLI crate only) | 6 s |
| `recipe` | 0 s | 0 s | 0 s |
| `smoke` (load + 16 tokens) | 1 m 33 s (weights cold) | 7 s | 3 s |
| `quick` (qualify --quick-only) | 5 m 50 s | 1 m 17 s | 1 m 20 s |
| of which `plan` / `kernels` / `gate` | 70 s / 44 s / 74 s | 9 s / 39 s / 6 s | 9 s / 39 s / 6 s |
| of which `acc-short` / `bench` | 69 s / 1 m 31 s | 5 s / 17 s | 8 s / 17 s |
| **quick numbers published** (`quick.md` after onboard start) | **7 m 24 s** | **1 m 24 s** | **1 m 23 s** |
| `in-depth` | 8 m 00 s | 1 m 24 s | 2 m 36 s |
| of which `acc-long` / `acc-4k` | 79 s / 1 m 59 s | 11 s / 40 s (FAIL) | 1 m 54 s / skipped (no 4k golden) |
| of which `envelope` / `bench-long` | 3 m 19 s / 1 m 22 s | 16 s / 18 s | 24 s / 18 s |
| **whole pass** | **15 m 24 s** (PASS) | **2 m 47 s** (FAIL, see below) | **3 m 59 s** (PASS) |
| 2 h target (quick) | met, 6 % of it | met, 1 % | met, 1 % |
| 4 h target (in-depth) | met, 6 % of it | met, 1 % | met, 2 % |
| skipped | `acc-format`, `ab` (no localeval on the box), `tp` | same | + `acc-4k` |

Numbers in the reports (not a speed gate; vLLM was up on spark1, radiance was down and the card exclusive on tatooine): CUDA decode 84.1 / 83.6 / 81.1 tok/s at 128 / 1024 / 4096 (86 / 85 / 83 % of the byte floor), prefill 4640 tok/s at 1k, 5383 at 4k; HIP decode 160.3 / 160.0 / 155.4 tok/s (58 / 58 / 56 %), prefill 6535 and 8684. Accuracy against the freshly made vLLM goldens, CUDA: short top-1 96.83 % KL 7.65e-3, long 97.85 % KL 5.56e-3, 4k prefill 92.40 % (all rows) with the 4k gate PASS.

Queue wait is not part of the target and is reported separately: the CUDA job was requested at 22:51 EDT and started at 03:22 after four attempts (a two-Spark TP2 job, #3555, ahead of it held both Sparks and took vLLM down while it ran; three of my own attempts were lost to client drops and a wrong reference address, see Fleetq notes).

## What the generation path covered, and what it did not

* The goldens were made from the vLLM server (`tools/golden_vllm.py`, top-20 `prompt_logprobs`) by `scripts/onboard/make-goldens.sh` **on the Mac**, copied to the box and found by model name from `-- --golden-dir`. `qualify state.json` lists all three as "found by model name" with their fnv fingerprints.
* `qualify`'s own make step could **not** run on either box: spark1 has no `uv` and no numpy (`ModuleNotFoundError: numpy`), and the tatooine dev image has none either (fleetq #3614). That is documented as a gotcha instead of fixed: porting `golden_vllm.py` to Rust was out of scope. So the "golden generated on the box" path is untested on CUDA and HIP; the generated-from-server goldens themselves are exercised.
* The 4k golden (`tools/prompts/golden_4k.jsonl`) is never made by `qualify`; the script makes it, and with it `acc-4k` ran on both backends.

## Faults found and fixed (no numerics change)

1. **A failed subprocess showed a warning, not the error.** `acc-short` ERRORed on tatooine and `onboard` printed "prefill chunk 4096; longest prompt 700 ..." (a warning line) while the real cause was `Error: ref_noise was measured on golden hash ...`. `proc::Run::tail` now starts at the last `Error:` line when it is within the last 8 lines (`crates/intarsia-cli/src/cmd/qualify/proc.rs`, unit test `tail_starts_at_the_error_report`).
2. **`acc-long` failed on a long golden with a 12k prompt.** The fixed `--max-seq 8192` could not hold `long_docs_12k` (12064 tokens): `sequence of 12064 tokens exceeds the planned context of 8192`. `qualify` now uses `max(8192, longest prompt + decode positions + 64)` rounded up to 256, read from the golden's header (`golden_max_seq`, unit test `golden_context_comes_from_the_longest_prompt`). The value only grows where the old one was a hard error, so no previously produced number changes. The `--max-seq` is KV capacity, not arithmetic.
3. **Environment, documented (not code):** a stale `/models/golden` 35B file on tatooine whose hash does not match the tracked `golden/ref_noise` sidecar (the repository's golden goes to the tree instead); a dirty tree makes `envelope` ERROR (build from a committed tree); the vLLM on spark1 listens only on its Tailscale address.

Not changed: no default, no kernel, no gate threshold.

## The HIP `acc-4k` failure (engine finding)

With the 4k golden, tatooine fails `acc-4k` on prefix reuse: `chat_reuse_4k continue` (split at 4521, prefill chunk 4096) diverges at token 5, shared rows differ by 0.97 / 0.34 nats, KL up to 1.8e-2 against the fixed 0.25 nats / 1e-2 tolerance. The aligned and restore cases pass bit-identically, and the same gate PASSes on CUDA (chunk 2048). This is the area lane fx5 is checking ("35B reuse logit deviation"); the onboard run only surfaced it, because the cached HIP goldens have no 4k member. Log: `raw/hip/35b-hip-fresh/report/qualify/logs/acc-4k.out`.

## Runs

| Run | What | Result |
|---|---|---|
| `raw/hip/35b-hip` | first HIP try, cached tatooine goldens, older binary | `acc-short` ERROR (stale golden vs sidecar, fault 1/3), 78 s |
| `raw/hip/35b-hip-r2` | tatooine golden dir with the repository's goldens | `acc-long` ERROR (12k prompt, fault 2) and `envelope` ERROR (dirty tree), 2 m 33 s |
| `raw/hip/35b-hip-r3` | fixes, clean tree | PASS, 3 m 59 s |
| `raw/hip/35b-hip-fresh` | freshly made goldens incl. 4k | FAIL (`acc-4k` reuse), 2 m 47 s |
| `raw/cuda/35b-cuda` | spark1, goldens made from vLLM | PASS, 15 m 24 s (`aborted-attempts-steps.log`: two lease slots lost to preflight aborts) |

Commits are on a clean tree for the CUDA run and the last two HIP runs (build stamp in each report's provenance table).

## Downtime used

* spark1 vLLM: none (non-exclusive lease #3892, 40 GB, vLLM stayed up and was the reference).
* tatooine radiance (EXIT trap restores it, `raw/hip/downtime.log`): 23:28:55-23:30:24, 23:42:10-23:44:53, 23:47:21-23:51:31 (8.4 min before 00:00, under the 60 min cap), 03:40:10-03:43:13. Every window is under 20 min.

## Fleetq notes

* Jobs: #3892 (spark1 lease for the CUDA run), #3911, #3683, #3665, #3637 (tatooine windows), #3910, #3679, #3662, #3634, #3633, #3610 (tatooine CPU build), #3614 (read-only check), #3729. Cancelled or lost: #3587, #3593, #3597, #3652, #3616, #3699, #3739 (all mine; none held a slot), #3762 and #3851 (ran 1 s and aborted in preflight before any work: the box has no numpy, then vLLM does not listen on localhost).
* A `fleetq run`/`acquire` client launched with a bare `&` died when its tool call returned, and a queued client without a heartbeat is dropped after three minutes; start it detached. The queue is strict, so a job that fails fast at the head of the line costs the whole wait again: take one lease (`fleetq acquire --wait`) and drive the steps by hand with `FLEETQ_JOB=<id>`.
* spark1 sat idle for hours behind a two-Spark job (#3555) that itself waited for spark2; `--expect` backfill did not help a job that is behind a job without `--expect`.

## Scripts

`scripts/onboard/`: `make-goldens.sh` (short, long, 4k from a reference server, on a host with uv), `build-tatooine.sh` (HIP CLI build in the dev image + golden sync), `hip-window.sh` (one radiance-down window, resumable onboard run under a time cap), `run-spark1-cuda.sh` (the unattended version of the CUDA flow; the numbers above came from the same steps driven by hand inside lease #3892, because the first two unattended slots were lost).

## Not done

* A model with no recipe (`recipe new` scaffold, TODO stop, exit 3) was not run: no such weights are on either box.
* `ab` (needs localeval and `uv` on the box) and `tp` (`--tp 2`) were skipped.
* HIP stayed in 20 min windows; the whole pass is 2 m 47 s so the cap never mattered.
* The in-box golden generation (numpy + uv on the box, or a Rust port of `golden_vllm.py`) is not implemented.
