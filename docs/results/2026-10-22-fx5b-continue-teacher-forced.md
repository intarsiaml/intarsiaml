# FX5b: is the 35B `continue` failure a free-running-stream artefact? No. (2026-10-22)

Lane `fx5b`, stacked on PR #87 (`lane-fx5/hybrid-reuse-ci`, head `f1fe8650`). User decision (option 4): fix the `continue` check, do not loosen
the tolerance. **No comparison code was changed**, because the premise (two free-running streams, near-tie flip at token 12) is falsified by the
code and by a measurement. Decision for the user at the end.

## 1. What the check compares (quoted, `crates/intarsia-cli/src/cmd/accuracy.rs` at `f1fe8650`)

The two sides do free-run (`reuse_gate`, `continue`): `gen_tokens_traced(engine, turn2.clone(), m, t)` is greedy decode on each side, and
`first_divergence(warm, cold)` locates the split. But the verdict never reads rows after the split (`judge_divergence`):

```
let nu_pre = (0..d).map(|j| row_nu(&warm_rows[j], &cold_rows[j])).fold(0.0, f64::max);
let nu_div = row_nu(&warm_rows[d], &cold_rows[d]);
let kl_max = (0..=d).map(|j| kl_rows(&cold_rows[j], &warm_rows[j])).fold(0.0, f64::max);
```

with the doc comment "rows `0..=d` share one context (the streams agree before `d`), so any difference between them is numerical". So the
failing numbers (fx10 Metal: "shared rows differ by 2.108 (before) / 0.235 (here) nats, KL up to 8.1e-2") are the `nu_pre` / `kl_max` over rows 0..12,
where both runs fed the same tokens. Teacher forcing on the cold stream recomputes exactly those rows with identical inputs and only adds rows after
the split, which can raise the maximum, never lower it. It cannot turn the verdict to PASS at 0.25 / 1e-2, and no test of "a flip that fails
free-running passes teacher-forced" exists to write: the current judge already ignores the post-flip text.

## 2. Measurement (CUDA GB10 spark1, 35B NVFP4, long golden, `long_docs_4k`, chunk 1024, fleetq #4037, non-exclusive, numerics only)

`accuracy --mode reuse --reuse-diag --reuse-measure --reuse-prompt long_docs_4k` with FX5's binary (same code as #87).
Raw: `2026-10-22-fx5b-continue-teacher-forced/raw/cuda-diag-c1024.log`. Rows are 32 logit rows, teacher-forced through `Engine::forward_script`
on the cold stream unless noted. (The command exited 1 afterwards in the `restore` diagnostic, "cannot roll recurrent state back to 1024
tokens"; that is the diagnostic's own replay, not the gate, and the `restore` row printed 32/32 before it.)

| comparison | argmax flips | KL mean | KL max | max abs dlogit |
|---|---|---|---|---|
| warm vs warm again; cold vs cold again | 0 | 0 | 0 | 0 (bit-identical) |
| cold chunk 1024 vs cold chunk 512 / 1023 / 2048 | 0 | 0 | 0 | 0 (bit-identical) |
| **warm vs cold, same (cold) context = teacher-forced `continue`** | 3 | 1.37e-2 | **8.54e-2** | 2.23 |
| cold vs warm with the reply prefilled instead of decoded | 2 | 1.13e-2 | 6.41e-2 | 3.27 |
| warm vs warm with prefilled reply (decode-written KV only) | 3 | 1.37e-2 | 6.00e-2 | 3.85 |
| engine cold vs `forward_script` replay of the same cold run | 2 | 7.63e-3 | 3.96e-2 | 3.63 |
| engine warm vs `forward_script` replay of the same warm run | 4 | 8.67e-3 | 3.25e-2 | 2.23 |

First engine-stream divergence: token 12, warm took 4670 / cold took 2186; warm margin +0.182, cold margin +0.405 nats; KL at that row 3.10e-2.

Reading:

* The teacher-forced `continue` comparison (row 3) fails the 1e-2 KL tolerance at 8.5e-2 with 3 argmax flips across 32 positions. The failure is
  not about which text each side wrote.
* A cold prefill is bit-identical across chunk sizes 512 / 1023 / 1024 / 2048, so "a different chunk split" is not the mechanism for a cold run.
  Resuming at token 2015 (not a multiple of the chunk, or presumably of any internal tile) is not bit-identical to the cold run even when the
  reply is prefilled (row 4): a resumed prefill is numerically a different computation. Decode-written KV adds a second, similar-sized part (row 5).
* Two paths that are supposed to be equivalent, the engine's decode and `forward_script`'s decode of the same run, differ by KL 4e-2 (rows 6, 7),
  the same size as the `continue` deviation. The 35B MoE's logits move by KL ~3e-2 to 9e-2 and ~2 nats on top-5 tokens under any non-bit-identical
  but equivalent path (expert-routing flips are a candidate; not tested). The tolerance 0.25 / 1e-2 was calibrated on dense Qwen3-8B / HIP.
* All restore and aligned rows stay bit-exact (nu 0, kl 0): the restore path is not implicated.

## 3. Not done, and why

* No per-layer dump: the repo has no layer or activation dump hook, and the evidence above says the deviation is a numerical-noise floor of the
  non-identical paths, not a single wrong op. Finding the first op to differ would need a new dump facility on CUDA and Metal; not built.
* Steps 5 (CUDA 1024 / 2048, Metal, 8B no-regression) validate a fix that does not exist; not run. The CUDA 1024 / 2048 and Metal 1024 / 2048 numbers
  for the unchanged gate are in `2026-10-22-fx5-hybrid-reuse-ci.md` section 3 and `2026-10-22-fx10-metal-35b-tests.md`.
* Offset sensitivity (resume at 2048 vs 2015 vs 2016) would show whether the resume-alignment part is exact at an aligned offset; not run.

## 4. Options for the user (a decision, not made here)

1. Per-model tolerance for the 35B from a measured noise floor of equivalent paths (`reuse_logit_tol` / `reuse_kl_tol` recipe keys already exist,
   draft PR `lane-fx5/reuse-tol`). This is the loosening that option 4 declined; the table above is the evidence for it (a floor of ~4e-2 KL between
   engine decode and `forward_script` decode, bit-identical repeatability).
2. Make the engine bit-exact on reuse: on `continue`, resume from the last chunk-aligned snapshot instead of the decode-extended state, so the warm
   run executes the cold run's operations (as `aligned` already does). A serving behaviour change that costs up to a chunk of re-prefill per turn;
   numerics-affecting, so opt-in at most. Not done.
3. Keep the gate as is: `continue` on the 35B stays FAIL and the report says so.

Fleet jobs: #4037 (spark1, diag, done). Cancelled before running: #4036 (Metal build, never started). No slot held.
