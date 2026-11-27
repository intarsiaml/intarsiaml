# Inkling-2-Mini on the R9700: opt-in int8 per-row weights (lane ink-hip-i8, 2026-11-26)

Branch `ink/hip-i8`, stacked on `ink/q8-study` (PR #540: the cpu accuracy study that picked this format). Opt-in and lossy, never
default-on. Status: **opt-in, lossy, PR open, not merged.** Decode +15 to +16 % plain and +8 to +31 % with the speculative default against main (P12), prefill unchanged; accuracy gate passes at short, long and 4k; spec-default ids equal spec-off for the i8 arm. Known limits at the end of the Results section.

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

4. 2026-11-26 ~21:00 EST (coordinator), new agent after the 20:40 restart: finish the evidence and open the PR (do not merge). (1) Merge `origin/main`
   (aee06201c; done, merge commit `dafa72379`, no conflicts). (2) Rebuild on tatooine, `--load cpu` (the old `intarsia-a` predates the R2U1
   retune). (3) W3: 4k accuracy both arms + plain-decode short ABBA 4 + 4, also the first GPU run of the retuned R2U1 / M 5..16 defaults.
   (4) `stage_speed.sh` windows: 4k, 16k, `SPEC=default`; prefill 4k / 16k not slower; spec-default md5 = spec-off for the i8 arm; hip `inkling2`
   tiny and `inkling2_batch` pass with the i8 env. (5) A row count slower than P12 (in_proj 9..16 rows 0.92x) goes back to P12 only if that is a
   small dispatch rule with a test, else a known limit. (6) Fill results / status, LOG.md row, unanimis capture. (7) cpu tests on the final head
   (endor `--load cpu`), push via `fleetq run --host endor --load cpu` (pre-push hook), PR to `main`. Budget: radiance about 17 of this lane's 20 min
   left; 16k as `PROMPTS=16k REPS="1 2"` (give n) if it does not fit; windows <= 5 min each, none after 06:00 EST; never `fleetq release` a
   running window job (kill the driver's process tree, confirm `curl -s http://tatooine:8080/v1/models`); radiance answering after every window.

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

## Earlier results and the 20:40 park (window 1 and 2, kept for the numbers)

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

(The parked-state list of what was unverified and the next steps are done: see Results. fleetq ids of that stretch: #12287 / #12295 / #12316 (tatooine cpu builds), #12302 (W1), #12323 (W2), #12313 / #12315 / #12296 / #12308 (endor cpu tests).)

## Results

## 1. Speed: main (P12 default) against `INTARSIA_INK_ATTN_QUANT=int8_row INTARSIA_LMHEAD_QUANT=int8_row`

Card 0 (R9700), `generate`, greedy, 256 tokens, graphs, one binary (the branch with the env unset is main's code path), runs interleaved ABBA
(`raw/stage_speed.sh`, `raw/stage_w45.sh`, ratio per rep, 95 % CI = t interval over the reps, `raw/speed.py`). The runs are deterministic to
about 0.1 %, so the intervals are narrow; n is the number of pairs and the interval at n = 2 uses t = 4.3, at n = 1 there is none.

| decode tok/s | main (P12) | int8 | change | 95 % CI of the ratio | n |
|---|---|---|---|---|---|
| plain (`--spec off`), short | 83.25 | 96.81 | **+16.3 %** | [1.162, 1.164] | 4 |
| plain, 4k | 80.03 | 92.48 | **+15.6 %** | [1.155, 1.156] | 4 |
| plain, 16k | 75.11 | 86.14 | **+14.7 %** | [1.144, 1.149] | 2 |
| spec default (DSpark k5), short | 94.49 | 101.86 | **+7.8 %** | [1.068, 1.088] | 2 |
| spec default, 4k | 108.00 | 141.72 | +31.2 % (see below) | [1.308, 1.317] | 2 |
| spec default, 16k | 119.99 | 131.46 | **+9.6 %** | none | 1 |

| prefill tok/s (the `generate` prefill line) | main | int8 | change | 95 % CI | n |
|---|---|---|---|---|---|
| 4k, plain | 3061.4 | 3065.3 | +0.1 % | [0.998, 1.004] | 4 |
| 4k, spec default | 3054.9 | 3057.5 | +0.1 % | [0.992, 1.009] | 2 |
| 16k, plain | 3182.1 | 3188.4 | +0.2 % | [0.990, 1.014] | 2 |
| 16k, spec default | 3183.0 | 3181.9 | 0.0 % | none | 1 |

Prefill is not slower: a chunk of more than 16 rows reads `bf16(q * scale)` and the unchanged bf16 kernels.

- **The spec-default 4k and short rows are not a pure speed ratio.** The int8 model is a different model: its greedy text differs from main's
  at short and 4k (md5 below), so the two arms speculate on different text. At 4k the int8 text is accepted much better (256 tokens in 77
  verify passes against 106 for main), which is most of the +31 %. At 16k both arms emit the same ids, so +9.6 % there is the kernel gain on its own.
  The plain rows (same number of steps) are the clean speed comparison.
- **Load costs +11 s** (host quantisation with the clip search: 16.3 s against 5.2 s) and the resident weights are 14.76 GB against 14.97 GB
  (`0.647 GB` of i8 in_proj + wo, `0.46 GB` of i8 unembed, in place of main's P12 copies; the bf16 `q * scale` weights stay for prefill).
- Microbench per call (windows 1 and 2, table in the earlier-results section): in_proj / wo / unembed 1.5-1.7x P12 at 1 and 4 rows,
  per token (48 in_proj + 48 wo + unembed) 4848 -> 3116 us at one row.

## 2. Accuracy (pre-registered gate)

KL21 = the 21-way KL (ref top-20 plus the rest) of the HIP self-golden of the int8 model against HIP main's, prefill mode (all prompt rows) and
decode mode (teacher-forced decode positions), `raw/klpair.py`; confident misses = rows where the golden's top-1 beats its top-2 by 0.5 and the
argmax differs. Gate: KL against main at or under the cpu C-vs-bf16 number plus 20 % (4.1e-3 / 4.6e-3 / 5.3e-3 for short / long / 4k, so 4.9e-3 /
5.5e-3 / 6.4e-3) and the G3 absolute bounds on the KL against the vLLM goldens (1.05e-2 / 1.65e-2 / 1.4e-2).

| set | int8 vs main, prefill | int8 vs main, decode | confident misses vs main (prefill / decode) | int8 vs vLLM prefill / decode (main) | G3 bound | verdict |
|---|---|---|---|---|---|---|
| short | 3.81e-3 | 2.77e-3 | 0 / 368, 0 / 287 | 8.49e-3 / 9.96e-3 (8.57e-3 / 7.49e-3) | 1.05e-2 | PASS |
| long | 3.46e-3 | 2.89e-3 | 0 everywhere | 1.252e-2 / 1.101e-2 (1.197e-2 / 9.29e-3) | 1.65e-2 | PASS |
| 4k | 4.67e-3 | 3.79e-3 | 0 / 8268, 0 / 63 | 1.010e-2 / 1.032e-2 (9.96e-3 / 1.069e-2) | 1.4e-2 | PASS |

Against the vLLM goldens the confident-row top-1 misses are the same as main's: short 0 and 0; 4k 3 of 8229 in prefill for both arms, 0 of 64
in decode. The short and 4k numbers were taken again after the dispatch fix of section 4 (window 7) and are unchanged to the printed digits;
long was measured before it (window 2) and not repeated (budget). Self-goldens and logs: `raw/acc-*`, `raw/w*-` logs.

## 3. Greedy ids (md5 of the 256-token output, `generate`)

| prompt | main spec off | main spec default | int8 spec off | int8 spec default |
|---|---|---|---|---|
| short | 9a27c21b | 9a27c21b | 51b358e2 | **51b358e2** (equal) |
| 4k | 9675bbb9 | 9675bbb9 | c96e82b7 | **c96e82b7** (equal) |
| 16k | 93e0a151 | 93e0a151 | 93e0a151 | **93e0a151** (equal) |

The int8 arm differs from main at short and 4k (lossy by design; the short text first differs at one near-tie, "grateful" against "appreciate")
and equals main's ids at 16k. At 64 tokens, short, plain: main = int8 = 04cdb28a (window 2). **The spec-default check found a bug in the first
run** (window 5, binary `0ffcc066`): the short md5 was 754611c1, not the spec-off 51b358e2. Cause in section 4; fixed; window 5b (binary
`185d01e0`) gives the table above.

## 4. The verify-rows bug and its fix (the one dispatch change)

In the model the i8 copies sit in the `in_p12` / `wo_p12` slots, and those are admitted on the P12 row limit (`DENSE_ROWS_MAX` = 4 on HIP) rather than
`quant::i8row::DENSE_ROWS_MAX` = 16, which was defined and never applied. The default speculative verify pass has up to 6 rows (k5), so its
in-projection and `wo` read the dequantised bf16 weight while a one-row decode step read the int8 codes: not rows-exact, and a near-tie flipped.
`crates/intarsia-core/src/model/inkling2.rs` now takes the i8 copy of in_proj and `wo` on 1..=16 rows (P12 keeps its limit), and
`inkling2_tiny_int8_row_real_weights_match_the_emulation` (`crates/intarsia-models/tests/inkling2.rs`) asserts a 6-row pass launches the i8
in-projection and `MatmulI8` (it fails without the change: "a 6-row pass took no i8 in-projection", fleetq #12373, and passes with it).
The kernels are unchanged, so windows 1 and 2's microbench and `ink_i8` correctness hold.

Step 5 (rows slower than P12): in_proj at 9..16 rows is 0.92x of P12 in the microbench (146.8 against 134.4 us at 16 rows). Main takes P12 only at 1..4 rows
(the verify pass of k5 has 6), and the alternative at 5..16 rows is the bf16 GEMV rows kernel, which was not measured against it. Sending
those rows to bf16 would break rows-exactness (spec on = spec off), the thing the md5 table needs, so **no row count is routed back**; the i8
in_proj at 9..16 rows (a verify pass of k >= 8, none by default) is recorded as a known limit.

## 5. Checks

| check | where | result |
|---|---|---|
| `intarsia-core --lib` | endor cpu, #12372 | 276 / 276 |
| `inkling2` tiny suite (cpu), incl. the real-i8 test with the 6-row assertion | endor cpu, #12372 | 18 / 18 |
| `intarsia-cli --test docs_env_vars`, `intarsia-models --test recipes_doc` | endor cpu, #12372 | 2 / 2, 3 / 3 |
| hip `cargo check --tests` | endor cpu, #12372 | ok |
| `ink_i8` device test (3 tests: every shape, every launch shape, rows-exact) | tatooine, window 2 (#12323) | 3 / 3 (kernels unchanged since) |
| hip tiny `inkling2`, main env (window 6, `raw/w6-inkling2-main.log`) | tatooine #12357 | 16 / 18: the screened-head pair (`head_screen` has no HIP kernel), same as main before this branch |
| hip tiny `inkling2_batch`, main env | #12357 | 9 / 9 |
| hip tiny `inkling2`, int8 env, after the fix (window 7, #12379) | tatooine | 15 / 18: the screened-head pair and `g1a_f32_vs_f64` (the cpu f32 path against the f64 oracle: the env quantises the weights, 7 of 40 top-1 agree) |
| hip tiny `inkling2_batch`, int8 env, after the fix | #12379 | 7 / 9: `batched_engine_equals_solo_engine`, `loaded_to_batch_packs_no_p12` (known limit below) |

Before the dispatch fix the int8 env also failed `inkling2_tiny_chunked_and_decode_match_prefill_bitwise` (window 6, 14 / 18); it passes now.

## 6. Known limits (all opt-in)

- **Batching (`serve --max-batch` > 1):** `Model::drop_p12_for_batch` drops the P12 slots, which now hold the i8 copies, so a batched engine reads
  `bf16(q * scale)`: the same quantised weights, no int8 speed-up, and its tokens are not bitwise a solo (max-batch 1) engine's. The two failing
  batch tests above are exactly this. `loaded_to_batch_packs_no_p12` fails because the copies are dropped at `enable_batch`, not at the load.
- Calls of more than 16 rows (prefill chunks) have no int8 kernel and read the bf16 dequantised weight.
- The in_proj at 9..16 rows is slower than P12 (microbench) and has no bf16 comparison; the unembed at 16 rows equals P12.
- Load +11 s (host quantisation). The vocabulary-sharded head (`INTARSIA_TP_VOCAB`) refuses `int8_row`.
- Not verified: long accuracy after the dispatch fix (identical code path expected); decode with k other than the default; Metal and CUDA (out of
  scope, not exercised); the spec-default speedup at n >= 4 and at 16k with a CI.

## 7. Radiance and fleetq

Radiance down this lane (each window restarted it, `/v1/models` answered after 10-15 s): W1 20:03:04-20:03:24 (0:20), W2 20:30:34-20:33:04 (2:30),
W3 21:08:04-21:12:02 (3:58), W4 21:25:07-21:28:44 (3:37), W6 21:38:08-21:38:41 (0:33), W5 21:48:07-21:51:18 (3:11, the run that found the bug),
W5b 21:57:36-22:00:48 (3:12), W7 22:01:37-22:03:30 (1:53): about 19.2 min of the 20 (W3-W7: 14.4 min). Logs: `raw/window*.log`.
fleetq: build #12339 (merged main) and #12375 (after the fix, binary `intarsia-a` md5 185d01e0, plus the models test binaries), windows #12347
(W3), #12354 (W4), #12363 (W5), #12357 (W6), #12378 (W5b), #12379 (W7); endor cpu #12364, #12372, #12373 (the test fails without the fix).
Scripts: `raw/stage_w3.sh`, `stage_w45.sh`, `stage_w6.sh`, `stage_w7.sh`, `stage_speed.sh`, `window.sh`, `run.sh`, `speed.py`, `klpair.py`.

