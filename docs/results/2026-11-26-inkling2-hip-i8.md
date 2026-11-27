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

3. 2026-11-26 ~20:10 EST (coordinator): #540 is merged as `a4c8f3b19`, which is main's tip now; it also has #538 (packed prefill
   default-on with `--max-batch`). (1) Merge `origin/main` into `ink/hip-i8` so this PR's diff becomes int8-only. (2) Tell the
   coordinator in the next report whether the Mac worktree `.worktrees/ink-q8`, the remote branch `ink/q8-study` and its `target/` folder
   can be removed now, or whether scripts or data from them are still used; the coordinator touches none of them until answered.

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

## Parked 2026-11-26 20:40

Head: see `git log -1` on `ink/hip-i8` (local commit, not pushed after `1dec009bc`; merged `origin/main` a4c8f3b19). Radiance: up (checked
`fleetq status` 20:38; every window of this lane restored it, 15 s). No job of this lane is running; #12334 (a build) was released unstarted.

**Go/stop rule (>= 10 % faster per call than P12 at rows 1 and 4): PASSED.** Window 1 (#12302) and window 2 (#12323), kernel-only, rotated
weights, mean us per call, i8 at its best launch shape against main's P12 kernel:

| shape (calls / token) | rows 1: P12 -> i8 | rows 4: P12 -> i8 | rows 8 | rows 16 |
|---|---|---|---|---|
| in_proj [8664, 2304] (48) | 55.1 -> 35.1 (1.57x) | 55.8 -> 37.3 (1.50x) | 63.3 -> 57.0 (1.11x) | 134.4 -> 146.8 (0.92x, i8 slower) |
| wo [2304, 3072] (48) | 22.8 -> 14.8 (1.54x) | 23.6 -> 16.2 (1.46x) | 28.0 -> 22.4 (1.25x, R2U1) | 44.9 -> 34.8 (1.29x, R2U1) |
| unembed [200058, 2304] (1) | 1101 -> 733 (1.50x) | 1281 -> 754 (1.70x, R2U2) | 1444 -> 1002 (1.44x, R2U1) | 2117 -> 2121 (1.00x, R2U1) |

Per token (48 in_proj + 48 wo + 1 unembed), window-1 numbers: rows 1: 4848 -> 3116 us (1.56x, -1732 us); rows 4: 5104 -> 3275 us (1.56x). The
window-1 defaults spilled at 4 rows (U >= 4: wo 61 us, unembed 4485 us), and at 8 / 16 rows in window 2 (R2U2: wo 58 / 85 us); both were
retuned (committed: M <= 4 R1U2, M 5..16 R2U1; in_proj U3 at one row, U1 above); the R2U1 default at 5..16 rows is **not yet run on the GPU**
(the table's rows 8 / 16 are the best-variant numbers of the window-2 sweep, which includes R2U1).

**Window 2 (#12323, `docs/results/2026-11-26-inkling2-hip-i8/raw/window2-12323.log`, radiance down 20:30:34-20:33:04):**
- correctness (`ink_i8`): 3 / 3 pass; real shapes (wo, unembed, in_proj S = 1 and 3, rows 1..16) 168 060 outputs and small / tail shapes with
  every launch shape 143 616 outputs match the f64 reference; rows-exact (row m of an M-row call = the one-row call) holds.
- greedy md5 at 64 tokens, short, `--spec off`: **main = i8 = 04cdb28a6849728e8aa5be39bf65c07e**; decode 82.79 -> 96.22 tok/s (+16.2 %, one run each,
  not a paired CI).
- accuracy against the vLLM goldens, HIP, prefill / decode mode (decode: 64 positions per prompt), KL21 (scored rows), confident misses 0 everywhere:

| set | main prefill | i8 prefill | main decode | i8 decode | i8 vs main prefill | i8 vs main decode |
|---|---|---|---|---|---|---|
| short | 8.565e-3 | 8.490e-3 | 7.489e-3 | 9.957e-3 | 3.81e-3 | 2.77e-3 |
| long | 1.197e-2 | 1.252e-2 | 9.292e-3 | 1.101e-2 | 3.46e-3 | 2.89e-3 |

  Pre-registered gate (KL against HIP main at or under the cpu C-vs-bf16 numbers 4.1e-3 / 4.6e-3 / 5.3e-3 plus 20 %, and the G3 absolute
  bounds 1.05e-2 / 1.65e-2 / 1.4e-2): short and long PASS on every number (decode short 9.96e-3 against 1.05e-2 is the tightest). 4k not run.
  Self-goldens: `.scratch/acc-hip/{main,i8}-{short,long}.safetensors`, logs `.scratch/acc-hip/w2/`; script `raw/klpair.py`.

**Not verified:** 4k accuracy (both arms); the R2U1 / retuned defaults on the GPU; paired decode speed with a CI (short / 1k / 4k / 16k, plain and
spec default; ABBA 4 + 4); prefill 4k / 16k not slower (the prefill path takes the dequantised bf16 weight, unchanged kernels; unmeasured);
spec-default md5 equal to spec-off for the i8 arm; `inkling2` tiny and `inkling2_batch` on hip with the i8 env; the i8 `InkInProj` at 9..16 rows
is slower than P12 in the microbench (0.92x at 16): the model sends it there; unembed at 16 rows is equal to P12. No PR opened.

**fleetq ids:** #12287 / #12295 / #12316 (tatooine cpu builds), #12302 (W1, `raw/window1-12302.log`), #12323 (W2), #12333 (pull), #12313 / #12315 /
#12296 / #12308 (endor cpu tests: core lib 275 / 275, tiny suite 18 / 18 x3, docs tests, hip check-only). Tatooine tree: `~/intarsiaml-ink-hip-i8`
(binary `.scratch/bin/intarsia-a` built before the R2U1 default; goldens in `golden/inkling2-mini-vllm-*` excluded from sync; logs in `logs/{w2,acc}`).

**Exact next step:** (1) `fleetq run --host tatooine-r9700 --load cpu --expect 20 --label ... -- bash .scratch/tat-build.sh` (sync + build, ~2 min);
(2) W3: `CAP=285 PROMPTS=short bash ~/intarsiaml-ink-hip-i8/.scratch/window.sh ~/intarsiaml-ink-hip-i8/docs/results/2026-11-26-inkling2-hip-i8/raw/stage_w3.sh`
under `fleetq run --host tatooine-r9700 --takes-down radiance --quiet gpu,cpu --expect 6` (4k accuracy both arms + plain decode short ABBA 4 + 4);
(3) the same with `stage_speed.sh` for 4k, 16k (`PROMPTS=16k REPS="1 2"` if the radiance budget, 20 min total, ~3 used, is short) and `SPEC=default`;
(4) fill `@@RESULTS@@` / `@@STATUS@@` here, speed table first; LOG.md row; unanimis capture; push `ink/hip-i8` and open the PR.

## Results

@@RESULTS@@
