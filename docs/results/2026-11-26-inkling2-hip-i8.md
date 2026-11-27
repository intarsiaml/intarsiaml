# Inkling-2-Mini on the R9700: opt-in int8 per-row weights (lane ink-hip-i8, 2026-11-26)

Branch `ink/hip-i8`, stacked on `ink/q8-study` (PR #540: the cpu accuracy study that picked this format). Opt-in and lossy, never
default-on. Status: @@STATUS@@

## Briefs received

2. 2026-11-26 ~19:45 EST (coordinator), after the PR: **Part 1, get #540 ready to merge.** The branch merged main at `80719483a`; main is
   `85e14bbcb`, seven merges later, one is P12 (#533), which edits the same `model.rs` / `inkling2.rs` (`decide_load`, `pack12`,
   `drop_p12_for_batch`, `LoadOptions`) and loads P12 by default on HIP. (1) Merge current origin/main (merge, not rebase). (2) Rerun
   on endor through fleetq: the 8 head tests, the `inkling2` tiny suite, `intarsia-cli --test docs_env_vars` (an intarsia-cli target),
   `intarsia-models --test recipes_doc`, and the C-short tensor-identity check against the earlier C run. (3) Confirm by test or code
   path (file:line) that `fp8_row` / `int8_row` are still refused on any non-cpu backend now that P12 loads by default on HIP, and say
   what happens when the env hook and `INTARSIA_INK_PACK12` both apply to the same tensor. (4) Push through fleetq, tell the coordinator the head.
   **Part 2, brief 2: HIP int8 per-row (arm C) decode kernel, opt-in**, stacked on `ink/q8-study` as branch `ink/hip-i8`, worktree
   `/opt/code/intarsiaml/.worktrees/ink-hip-i8`, tatooine folder `~/intarsiaml-ink-hip-i8`, docker volume
   `intarsia-cargo-target-ink-hip-i8`; follow `.worktrees/hip-swarm/COMMON-1126.md` (windows, `HIP_VISIBLE_DEVICES=0` on `--host
   tatooine-r9700`, radiance handling, trap + kill the process tree in window scripts, `env VAR=x cmd` in sh). Goal: opt-in int8-per-row
   weights on HIP for the in-projection, `wo` and the unembed; replaces P12 on those tensors when enabled; lossy, never default-on, no
   default changes. Placement: new files only following P12 (`kernels/hip/ink_i8.hip`, `crates/intarsia-backend-hip/src/inki8.rs`),
   shared files add-only; keep off `kernels/hip/ink_mix.hip`, `ink_attn.hip`, `ink_route_fused.hip` and the serve scheduler (lane
   ink-hip-bpf) and spec routing / `dflash*.rs` (lane ink-dflash2). Order: (1) cpu-side, no GPU: the kernel, the `MatmulI8` /
   `InkInProj` per-row-scale contract, admission cases at the real shapes rows 1, 4, 16, 2048, the re-derived exact-unembed-screen
   per-row bound for int8 (if hard, document: run without the screen under int8, or a looser bound). (2) First GPU window: kernel-only
   microbench, <= 3 min radiance down, int8 GEMV vs P12 GEMV at the real shapes rows 1, 2, 4, 8, 16; **stop rule: if int8 is not at
   least 10 % faster than P12 per call at rows 1 and 4, stop and report** (decode step about 13 ms against an 8.7 ms byte floor). (3) If
   it continues: end-to-end greedy decode `--spec off` and spec default, short / 1k / 4k / 16k, int8 vs main (P12), ABBA 4 + 4 with a 95 %
   CI of the ratio; prefill 4k / 16k must not get slower; an accuracy gate. **Accuracy gate (pre-registered):** the cpu G2 verdict does
   not carry over (HIP's accepted G2 short is already 4.49e-3, over 4e-3, since the flash-prefill flip); gate on KL of HIP int8 against
   HIP main on the G2 / G3 sets, short / long / 4k and confident-row misses, and the G3 absolute bounds (1.05e-2 / 1.65e-2 / 1.4e-2).
   Pass = KL against main at or under the cpu C-vs-bf16 numbers (4.1e-3 / 4.6e-3 / 5.3e-3) plus 20 %, and G3 passes. Budget: radiance 20
   min total, windows <= 5 min; two other lanes queue for tatooine windows, so do all cpu-side work before queueing a window; Metal out
   of scope. Report: lead with the speed table (prefill 4k / 16k, decode short / 4k / 16k, plain and spec, main (P12) vs int8, %
   change, CI), then microbench, accuracy, checks (command + result), radiance down per window, fleetq ids, PR link + head, a LOG.md
   row (add-only), the unanimis id, leftovers.

## Format and contract

Int8 weight `q [N, K]` (levels -127..=127) with one f32 scale per output row (`scales [N, 1]`), W8A16: `out[m, n] = rA(scale[n] *
sum_k x[m, k] * q[n, k])` (`quant::i8row`, `Op::MatmulI8` in slot 91, the spare placeholder of the finished ink-fp8 lane, so the
80-83 block that ink-pack12 holds is untouched; the i8 form of `InkInProj` takes the third input `scales [Nq+Nk+Nv, 1]`). Quantiser:
`head::int8_best_clip` (MSE-optimal symmetric clip, round to nearest even), the arm C of `2026-11-26-inkling2-q8-study.md`.
The cpu reference is f64 in the contract's order (`quant::reference::matmul_i8`, `ink_ref` for the in-projection), rows-exact.

## Kernels (kernels/hip/ink_i8.hip, crates/intarsia-backend-hip/src/inki8.rs)

One wave per R weight rows; lane L reads 16-byte chunks (16 int8) L, L + 32, ... with U loads in flight, sign-extends and converts,
runs the f32 fma chain in element order, `ik_warp_sum`, then the row scale and the output rounding. The chain of an output does not
depend on M, R or U, so the kernels are rows-exact (up to a signed zero on a masked tail). Registrations: `matmul_i8[_logits]_gemv`
(1..=4 rows) and the `_rows` forms (1..=16, rows-exact) for `MatmulI8`, `ink_in_proj_i8_gemv[_rows]` for the i8 `InkInProj`.
Calls with more than 16 rows have no i8 kernel (the model must keep `bf16(q * scale)` for prefill chunks: step 3).

## Pre-registered reading of the stop rule (written before window 1)

- Metric: mean microseconds per call over 200 calls (unembed 40) with the weight rotated through copies totalling at least 400 MB
  (the Infinity Cache does not hold it), `ink_i8.rs::i8_microbench`; i8 at its default launch shape against the P12 kernel exactly as
  main runs it (`matmul_p12_gemv` / `_logits_gemv` at <= 4 rows, the `_rows` names above, `ink_in_proj_p12_gemv[_rows]`; the unembed
  with main's default dec3 split). Rows 1 and 4.
- Shapes: wo `[2304, 3072]` (48 visits per token), the in-projection stack `[8664, 2304]` (48 visits), the unembed `[200058, 2304]`
  (one call per token). **HIP main runs no unembed screen** (`HeadScreen` is unimplemented on HIP, `INTARSIA_INK_HEAD_SCREEN` is
  off outside Metal and CUDA), so the unembed cost main pays is the full P12 GEMV and the comparison is like for like.
- **Stop rule (the brief's): "int8 not at least 10 % faster than P12 per call at rows 1 and 4" is read per token, not per call:
  the sum 48 * in_proj + 48 * wo + 1 * unembed of the measured microseconds, int8 against P12, at rows 1 and at rows 4. If either
  sum improves by less than 10 % (P12 sum / int8 sum < 1.10), stop and report. If the sums clear 10 % but one shape does not (wo is
  7 MB int8 and may be launch-bound), continue with that tensor left on P12 and say so.** Per-shape numbers are reported either way.
- Not part of the decision: prefill speed (no i8 kernel; step 3 checks it does not get slower), accuracy (step 3's gate).

## Results

@@RESULTS@@
