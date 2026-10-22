# FX5: a measured 35B prefix-reuse tolerance (PROPOSED: needs a user decision) (2026-10-22)

Draft PR `lane-fx5/reuse-tol`, split out of PR #87 at the coordinator's request. **Nothing here is in force until the user decides.** It loosens a gate that failed (reuse logit tolerance 0.25 -> 5.1 nats, KL 1e-2 -> 0.12, i.e. 20x and 12x) on the strength of one failing case, which is the pattern Fable findings B/H warn about; the FX2 cooling rule is still only proposed. The measurement tooling (`accuracy --reuse-measure`, `--reuse-require-all`) is in PR #87; this branch is based on main and needs #87 merged to reproduce the runs.

## What changes

`recipes/qwen3_5_moe.toml` `[accuracy]`: `reuse_logit_tol = 5.1`, `reuse_kl_tol = 0.12` (built-in: 0.25 / 1e-2, calibrated on dense bf16 Qwen3-8B / HIP). One test line in `accuracy.rs` (`the_reuse_tolerance_is_a_fixed_constant_...`) pins the recipe value. The recipe applies to every backend that loads `qwen3_5_moe`; it was measured on CUDA GB10 only.

## Derivation

`accuracy --mode reuse --reuse-measure --reuse-require-all`, 35B NVFP4, spark1 CUDA (vLLM up), the 35B long golden (5 prompts: two 4k, `chat_reuse_4k`, two 12k; the 12k ones are cut to 8152 tokens by `--max-seq 8192`), bf16 KV, fleetq #3763 (`scripts/fx5/run.sh`). `nu` = largest |dlog-prob| over either row's top-5 tokens, `kl` = KL(cold || warm), over the rows that share a context (to the first divergence), as the judge uses them. Every `restore` case (aligned or not) was bit-identical to cold (nu 0, kl 0); only `continue` (reply tokens written by the decode kernels) moves:

| prompt (continue) | chunk 1024 nu / kl (run 1 = run 2 to every digit) | chunk 2048 nu / kl |
|---|---|---|
| long_docs_4k | 0.877 / 3.10e-2 (diverged at token 12) | 2.138 / 9.42e-2 |
| long_code_4k | 2.420 / 3.54e-2 | **4.056** / 4.56e-2 |
| chat_reuse_4k | 1.183 / 4.40e-2 | 1.075 / 5.66e-2 |
| long_docs_12k | 1.949 / **8.49e-2** | 2.399 / 5.61e-2 |
| long_code_12k | 1.878 / 1.50e-2 (diverged at 26) | 1.977 / 4.00e-2 |

3 runs, 15 measurements, 10 distinct. Max nu 4.056, max KL 9.42e-2. Margin 25 %: 4.056 x 1.25 = 5.07 -> 5.1; 9.42e-2 x 1.25 = 0.118 -> 0.12.

## Old vs new verdicts, side by side

Same binary and golden, only the tolerance differs (`raw/reuse-c1024-run1.log` / `reuse-recipe-c1024.log`, `reuse-c2048-run1.log` / `reuse-recipe-c2048.log`; 15 cases per run: continue, restore, aligned for 5 prompts).

| run | built-in 0.25 / 1e-2 | recipe 5.1 / 0.12 | cases that change |
|---|---|---|---|
| chunk 1024 (qualify's) | gate **FAIL** | gate PASS | `long_docs_4k continue`: FAIL -> ok (the only one) |
| chunk 2048 (CUDA default) | gate PASS | gate PASS | none |

What the table does not say, so the decision is informed:
* Only the diverged streams are ever judged by the tolerance. 14 of the 15 chunk-1024 cases pass without consulting it: 13 have identical warm and cold streams (including `long_code_4k continue`, whose rows differ by nu 2.42 / KL 3.5e-2, far above the built-in values), and `long_code_12k continue` (diverged at 26) passes on the older rule "at least 16 leading tokens agree". The single case the new values rescue is a divergence at token 12 with the warm and cold top-1 margins 0.18 and 0.41 nats and a KL of 3.1e-2.
* 5.1 nats is wide enough that nu stops discriminating; the KL bound (0.12) is the working part of the judge. The detector for a wrong restore is the bit-exact check on every chunk-aligned restore (default on), which is unaffected, and the CI test in PR #87.
* The deviation is the decode-kernel vs prefill-kernel difference in KV/GDN state (M3 2.8: "plausible cause, not proven"), unexplained and unbisected. A fix upstream of the tolerance (same kernels for the reply tokens, or a bisect to the MoE/GEMM that flips) would make this unnecessary.
* Not measured on HIP or Metal, nor with fp8 (lane F2, PR #66, diagnoses `chat_reuse_4k` fp8-lt against the old values).

## Decision needed

(a) Commit these values (this PR), (b) leave the 35B at the built-in values, so `qualify`'s reuse gate at chunk 1024 reads FAIL on `long_docs_4k continue` until the deviation is understood, or (c) a per-case rule instead of a wider constant. A flag-level alternative that needs no recipe change: `--reuse-logit-tol 5.1 --reuse-kl-tol 0.12` on a single run.

Verification of the committed values: `scripts/fx5/verify.sh` (fleetq #3869) -> `raw/reuse-recipe-c1024.log`, `raw/reuse-recipe-c2048.log`: both `5 prompt(s) judged, 0 skipped`, `prefix-reuse gate: PASS`.
