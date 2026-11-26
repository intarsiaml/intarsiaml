# Inkling-2-Mini on HIP: the cost of a verify row, and the draft length (lane ink-hip-verify, 2026-11-26)

Branch `ink/hip-verify` (worktree `.worktrees/ink-hip-verify`), base main `3226af2b9` (#528). Goal: make DSpark speculative decode
(default-on for Inkling-2-Mini on HIP since #527: cut 0.5, k3) faster on one R9700 (tatooine, gfx1201) by making the verify pass cheaper
per row and by checking the draft length; greedy output stays identical to `--spec off`. Shared rules:
`.worktrees/hip-swarm/COMMON-1126.md`.

## Briefs received

1. 2026-11-26 ~14:00 EST (coordinator, lane brief), summarised:
   - Measured (`2026-11-26-inkling2-hip-spec.md`, "Verify cost"): decode 13.30 ms; verify 2 / 3 / 4 rows 16.48 / 19.00 / 20.80 ms (1.24x /
     1.43x / 1.56x); draft ~2.1 ms. Per-op lines lost. Modelled (Fable, `docs/plans/2026-11-26-inkling2-drafter-quant/fable.md`): 4-row
     byte floor ~12.8 ms (8.76 one-row floor + ~4 ms of extra expert bytes, union 8 -> 18.9 per visit) against 20.8 measured, so ~3.5 ms of
     row-dependent excess; on Metal the experts were 70 % of the growth.
   - Step 0: search `ink/*` results docs for prior HIP verify / rows work. Step 1: one GPU window, per-op kernel trace (rocprofv3
     `--kernel-trace`) of verify passes at 1, 2, 3, 4, 6 and 8 rows, graph on; table of ms per op class per row count, byte floor per
     class, row-dependent excess. Same window if it fits: draft-length sweep at cut 0.5, k 3 / 4 / 5 / 7 at short / 1k / 4k / 16k, 2 + 2
     processes per arm (rows >= 5 move `ink_moe.hip` from LDS class 0 to 1); a k that beats k3 by >= 2 % at >= 2 lengths gets a 4 + 4
     ABBA confirmation.
   - Step 2: fix the largest row-dependent excess in this lane's kernels (`kernels/hip/ink_moe.hip` decode / rows expert forms, not
     #528's PF2 prefill forms; `kernels/hip/ink_route*.hip`; `kernels/hip/ink_spec.hip`). If the largest excess is in the in-projection,
     wo or unembed (lane ink-hip-p12) or attention / mix (lane ink-hip-batch): stop and report the numbers first. Step 3: 1-row decode
     and rows 2..16 stay bitwise as on main.
   - Gates (card 0): greedy md5 = main at short and 4k, 64 and 256 tokens, `--spec off` and the spec default; verify probe rows 2..k+1
     bitwise the 1-row decode; hip tests `inkling2`, `inkling2_dspark`, `inkling2_dspark_cut`, `inkling2_verify_softcap` as on main,
     admission 0 FAIL for touched ops; G2 short / long = main; speed 4 + 4 ABBA, spec default at short / 1k / 4k / 16k and plain short /
     4k, 95 % CI. Default-on only with gates 1-4, spec >= +2 % with CI lower bound > 1.0 at >= 2 lengths and no length slower; else an
     env knob with a `docs/tuning.md` row. A k change ships via `[tuning.hip] mtp_default_k` under the same bar.
   - Overlaps: ink-hip-p12 owns `kernels/hip/ink_proj.hip`, the wo / unembed GEMVs in `matmul.hip`, `quant/p12.rs`; ink-hip-batch
     (#529) owns `ink_attn*.hip`, `ink_mix*.hip`, `inkattn.rs`, `inkmix.rs`; ink-sampled owns `spec.rs` and `model/dspark.rs`;
     `kernels.rs` append-only.
   - Radiance: at most 5 windows, each <= 5:00 down, <= 15:00 in total.

## 0. Prior work (step 0)

`git log --all -i --grep` (verify, union, rows, spec-k) and the results docs on `ink/*` branches:

| Where | What it measured | What it means here |
|---|---|---|
| `2026-11-26-inkling2-hip-spec.md` (#527) | verify totals 2-4 rows (above); the verify forward runs only `rows_exact` kernels (`moe_experts_situ_bf16_rows`, `ink_attn_decode_bf16`, `_rows` GEMV twins); verify graph saves 0.3-0.6 ms a pass | the per-op split was never recorded on HIP |
| `2026-11-25-inkling2-hip-moe.md` (lane ink-hip-moe) | MoE bench by T, **random** routing (union 8 / 15.5 / 29 / 52 at T 1 / 2 / 4 / 8): NVFP4 82.3 / 140.8 / 246.5 / 401.7 us a call (435 / 498 / 529 / 571 GB/s of the union's bytes), MXFP8 121.9 / 216.3 / 353.3 / 558.0; ~17 us a call that does not scale with bytes (two launches 5.9, in-block grouping 5.9); the `moe_pre` rule (load windows, pairs per batch) was swept by T there; the in-wave grouping (class 4) was slower | at bench unions the multi-row expert kernels already run at 85-91 % of 627 GB/s; real routing has smaller unions (below) |
| `2026-11-26-inkling2-hip-dec2.md` | one-wave grouping for <= 16 pairs (T <= 2): 11.8 -> 9.7 us at T = 1, but 15.4 at T = 4 | why `kWaveGroupPairs` = 16 |
| `2026-11-21-inkling2-mtp.md` (Metal) | expert union per visit over R consecutive real rows: 8.0 / 12.5 / 16.0 / 18.9 / 21.4 / 23.7 / 25.7 / 27.6 (R 1..8, NVFP4; MXFP8 within 0.3); verify per-op attribution 1 -> 4 rows: experts +12.4 of +18 ms profiled (70 %), conv_mix +2.1, matmul +5.6 (target + draft), attention / route / in_proj flat; a union-fused down kernel sized at 1.0-1.5 ms per 4-row pass and parked | the byte floor below uses this union table |
| `2026-11-25-inkling2-verify.md` (Metal, #517) | a screened greedy head for 2..4 verify rows and a row-table DSpark draft attention: both at noise | head and draft attention are not levers on Metal |
| `2026-11-21-inkling2-cuda-mixmoe.md` | "verify-pass MoE: gu 259 us vs 116 at decode (2.2x at ~4.5 rows): expert union, bandwidth-bound; no target" | on CUDA the experts were at their floor |
| `2026-11-22-inkling2-spec-k.md` (Metal, branch `ink/spec-k`) | k1-k7 at gate 0.6 / 0.8: k5 / k7 lose 2-17 % on greedy and code under every rule (an extra row costs ~21 % of a plain step on Metal) | |
| `2026-11-26-inkling2-spec-cut.md` (Metal) | cut 0.5 k5 at most +1.5 % over cut 0.5 k3 (1k, 16k), below it at short and 4k | no k > 3 won on Metal |

No HIP draft-length sweep and no HIP per-op verify profile exist.

## 1. Window 1: where a verify row goes, and the draft length

fleetq #12118, radiance down 14:22:36-14:26:22 EST (**3:46**), both cards (the slot held tatooine-r9700 and -2). Binary `main` = main
`3226af2b9` (md5 `a1408dd5`); the probe test (`tests/verify_rows_real.rs`) built from this branch, which only adds its timing-only knobs
(`INTARSIA_VERIFY_TIME_ONLY`, `INTARSIA_VERIFY_PROMPT`, `INTARSIA_VERIFY_TIME_REPS`, `docs/tuning.md` rows). Verify graph on (the
default), nothing else set. Logs: `raw/window1-12118.log`, `raw/w1/`.

**How the passes were forced.** The probe's cost loop (`INTARSIA_VERIFY_TIME_ONLY=1`) runs a decode step and verify passes of exactly 2..8
rows, each pass rolled back before the next, at the real 1021-id prompt of the speed set (`INTARSIA_VERIFY_PROMPT`); its verify rows are
the model's own greedy continuation of that prompt, so every row is a token an all-accepted draft would carry and the experts route as on
real text (random ids route nearly independent rows: union 29 at 4 rows against 18.9). Each pass ends in one argmax dispatch, so the kernel
trace splits into passes; reps 3-8 of 8 kept, every kept pass of a row count has the same dispatch count (`raw/rowtrace.py`; the trace,
`raw/w1/rp/run_kernel_trace.csv.gz`, 47 741 dispatches).

**Verify cost, unprofiled** (`PROBE_TIME`, median of 12, ms; `raw/w1/probe1k.log`, `probe4k.log`):

| context | decode (1 row) | 2 rows | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1021 ids | 13.36 | 16.62 (1.24x) | 20.06 (1.50x) | 22.36 (1.67x) | 24.74 (1.85x) | 27.03 (2.02x) | 29.35 (2.20x) | 31.56 (2.36x) |
| 4108 ids | 13.54 | 17.00 (1.26x) | 19.37 (1.43x) | 21.57 (1.59x) | 24.74 (1.83x) | 27.20 (2.01x) | 29.23 (2.16x) | 31.09 (2.30x) |

Under the profiler a pass is ~0.9-1.1 ms longer; the class table is in GPU kernel time per pass (median of 6 reps), with the byte floor of
each class at 627 GB/s and the row-dependent excess = (t(R) - t(1)) - (floor(R) - floor(1)):

| class (owner) | 1 row | 2 rows | 3 rows | 4 rows | 6 rows | 8 rows | floor 1 / 4 / 8 | excess at 2 / 3 / 4 / 6 / 8 rows |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| experts NVFP4, 44 visits (this lane) | 3.59 | 5.61 | 7.59 | 8.97 | 11.78 | 14.61 | 2.51 / 5.94 / 8.68 | +0.61 / +1.49 / **+1.96** / +3.26 / +4.87 |
| experts MXFP8, 4 visits (this lane) | 0.47 | 0.78 | 1.07 | 1.30 | 1.74 | 2.18 | 0.42 / 1.01 / 1.45 | +0.07 / +0.16 / +0.24 / +0.44 / +0.67 |
| route: router GEMV + top-k (this lane) | 0.47 | 0.50 | 0.52 | 0.55 | 0.54 | 0.54 | 0.04 / 0.04 / 0.04 | +0.03 / +0.05 / +0.08 / +0.07 / +0.07 |
| attention: `ink_attn_dec_k`, `_s1_k`, comb, prologue (ink-hip-batch) | 0.96 | 1.40 | 1.81 | 2.17 | 2.96 | 3.60 | 0.32 / 0.32 / 0.32 | +0.43 / +0.84 / **+1.20** / +2.00 / +2.63 |
| blend: `ink_blend_gate`, from 3 rows + `ink_ctl_gemv` + `ink_blend_norm` (`ink_mix.hip`, ink-hip-batch) | 0.36 | 0.46 | 1.04 | 1.21 | 1.56 | 1.90 | 0.18 / 0.18 / 0.18 | +0.09 / +0.68 / **+0.85** / +1.19 / +1.53 |
| in-projection (ink-hip-p12) | 3.09 | 3.08 | 3.25 | 3.31 | 3.54 | 3.75 | 3.05 | -0.01 / +0.16 / +0.22 / +0.45 / +0.67 |
| wo (ink-hip-p12) | 1.17 | 1.22 | 1.26 | 1.33 | 1.40 | 1.48 | 1.08 | +0.05 / +0.08 / +0.16 / +0.22 / +0.31 |
| unembed (ink-hip-p12) | 1.45 | 1.45 | 1.45 | 1.46 | 1.48 | 1.49 | 1.48 | +0.00 / +0.00 / +0.01 / +0.03 / +0.05 |
| ConvMix / AttnResMix row kernels (ink-hip-batch) | 0.96 | 1.03 | 0.87 | 0.86 | 0.85 | 0.85 | ~0 | +0.07 / -0.09 / -0.10 / -0.11 / -0.11 |
| kernel time, all classes | 12.56 | 15.56 | 18.89 | 21.18 | 25.87 | 30.45 | 8.79 / 12.80 / 16.00 | +1.34 / +3.38 / +4.61 / +7.55 / +10.70 |
| GPU span of the pass | 14.23 | 17.47 | 21.07 | 23.38 | 28.06 | 32.65 | | |
| dispatches | 538 | 536 | 632 | 632 | 632 | 632 | | |

Floors: experts = visits x union(R) x bytes of an expert (NVFP4 4.48 MB, MXFP8 8.21 MB) with the union of R consecutive real rows
(`2026-11-21-inkling2-mtp.md` section 1: 8 / 12.5 / 16 / 18.9 / 21.4 / 23.7 / 25.7 / 27.6; this probe's own union was not read back, so
the experts' floor is that table's, not this pass's); dense classes are read once a pass (`2026-11-25-inkling2-hip.md` 4.3); attention =
one read of the 1k context's K / V per visit (0.32 ms), whatever the rows. Per-kernel lines: `raw/w1/rowtrace.txt`.

Reading (revised after window 2's bench, section 2):
- Against 627 GB/s the experts carry the largest excess (+1.96 ms NVFP4 + 0.24 MXFP8 at 4 rows), but that floor charges the extra union
  bytes at 627 GB/s while the expert kernels run at ~470 GB/s **at every row count, 1 row included** (bench, section 2). Charged at
  the kernels' own 1-row rate, the experts' row-dependent excess at 4 rows is ~0.5 ms with Metal's union table, and ~0 if this probe's
  real union is ~21 (the probe's gate_up takes 131.5 us a call at 4 rows against 111.2 us in the bench at union 19, while down is
  within 3 % of the bench; the probe's own union was not read back, so the experts' floor here is approximate). So the experts'
  growth is the union's bytes, not a row cost of the kernels.
- **The largest row-dependent excess with a flat floor is attention: +1.20 ms at 4 rows** (`ink_attn_dec_k` 0.78 -> 1.88 ms; the
  verify attention is one decode call per row, +0.37 ms a row, each row reading the context's K / V again), then **blend: +0.85 ms
  at 4 rows**: `kernels/hip/ink_mix.hip` `kFuseRowsMax = 2`, so from 3 rows the loop blend leaves the fused launch for
  `ink_ctl_gemv_kernel` + `ink_blend_gate` + `ink_blend_norm` (0.46 -> 1.04 ms between 2 and 3 rows, +96 dispatches a pass). Both
  are in lane ink-hip-batch's files (`ink_attn*.hip`, `ink_mix*.hip`): the brief's stop-and-report case, reported to the
  coordinator (section 3) and not touched here.
- The in-projection adds 0.22 ms at 4 rows (ink-hip-p12); wo, unembed, route and the ConvMix / AttnResMix row kernels are flat.
  Gaps between dispatches add +0.5 ms from 1 to 4 rows (span minus kernel time 1.67 -> 2.20 ms).

**Draft-length screen** (cut 0.5, k 3 / 4 / 5 / 7, `INTARSIA_DSPARK_TAU=0 INTARSIA_DSPARK_CUT=0.5 --spec dflash --spec-k K`; sweep.json =
a warm-up + 6 of the 12 short golden prompts (count, story, code, science, chat_system, chat_reasoning; 35-101 ids) + 1021 / 4108 /
16455 ids, 128 tokens, greedy, EOS honoured). Three processes per arm: card 1 in the order k3 k4 k5 k7 (c1a) then k7 k5 k4 k3 (c1b),
card 0 k3 k4 k5 k7 (c0a) after the probes, so the screen shared the box with the profiler. Ratio to k3 per (prompt, process), geometric
mean per class, t-interval over the 3 processes [bootstrap over processes x prompts] (`raw/w1/sweep-analysis.txt`):

| decode, k / k3 (cut 0.5) | short (6) | 1k | 4k | 16k | tokens / pass | rows / pass |
|---|---:|---:|---:|---:|---:|---:|
| k3 (114.1 / 90.0 / 115.2 / 91.9 tok/s) | 1 | 1 | 1 | 1 | 2.68 | 3.44 |
| k4 | **1.035** [1.029-1.040; 1.020-1.050] | 1.003 [0.996-1.010] | **1.060** [1.058-1.062] | **1.034** [1.029-1.039] | 2.89 | 3.91 |
| k5 | **1.021** [1.016-1.025; 1.005-1.037] | 1.012 [1.004-1.020] | **1.081** [1.076-1.086] | **1.048** [1.042-1.055] | 2.97 | 4.26 |
| k7 | 1.000 [0.996-1.004; 0.979-1.023] | 1.005 [1.000-1.010] | **1.093** [1.090-1.095] | **1.041** [1.035-1.047] | 3.07 | 4.73 |

Ids identical in all 12 processes. k4 and k5 each beat k3 by >= 2 % at three lengths (short, 4k, 16k), so the brief's confirmation
applies (section 4). Unlike Metal (where k5 / k7 lost, `2026-11-22-inkling2-spec-k.md`), a verify row on HIP costs ~2.3 ms against a
13.4 ms step (r / P ~0.17), so drafts 4 and 5 pay for their rows.

## 2. Window 2: expert slices for 2..16-row calls (negative, reverted)

fleetq #12133, radiance down 14:39:20-14:41:24 EST (**2:04**), both cards. Logs: `raw/window2-12133.log`, `raw/w2/`.

Idea: a 2..16-row call routes more distinct experts (real text: 12.5 / 18.9 / 27.6 at 2 / 4 / 8 rows) than a block has waves (8), so
a wave ran 2-4 experts in a row while the grid stayed as wide as at decode; with S "expert slices" (grid y) block y's waves take the
experts y * 8 + w, + 8 S, .. and down's terms go through scratch and the gather launch (bf16-exact, slot order; an env knob
`INTARSIA_HIP_INK_MOE_SPLIT`, default the rule ceil((8 + 3.6 (T - 1)) / 8)). The bench gained the real-text union mode
(`INK_MOE_BENCH_REAL=1` in `tests/ink_moe.rs`: a T-row call reads the union of T consecutive real rows, not T independent top-8 draws).

MoE bench, NVFP4, real-text unions, us per call (`raw/w2/bench.log`; "1:1" = main's grid):

| T (union) | main's grid: both / gate_up / down | best slices (gate_up:down) | gain | GB/s, main's grid |
|---|---|---|---:|---:|
| 1 (8) | 80.6 / 49.6 / 29.8 | n/a | | 445 |
| 2 (13) | 125.6 / 80.4 / 46.8 | 122.1 (4:1) | -2.8 % | 464 |
| 3 (16) | 150.0 / 96.2 / 56.1 | 149.8 (2:1) | 0 | 478 |
| 4 (19) | 181.4 / 115.5 / 68.6 | 177.2 (0:1) | -2.3 % | 469 |
| 6 (24) | 224.0 / 147.5 / 81.6 | 222.6 (0:1) | -0.6 % | 480 |
| 8 (28) | 278.1 / 183.0 / 100.3 | none (every split slower) | | 451 |

Slicing down is slower at every T (its extra gather launch), slicing gate_up gains at most 4 us a call. In the model (verify probe at
the 1k prompt, `PROBE_TIME_MIN`, ms): main's grid 16.52 / 19.93 / 22.26 at 2 / 3 / 4 rows, the rule 16.59 / 19.96 / 22.18, gate_up
sliced only 16.41 / 19.83 / 22.08: <= 0.18 ms a pass (< 1 %). At 5..8 rows the rule broke outputs: `ink_moe_rows_exact` failed at T = 5
and the bitwise probe failed (rc 101). Cause: every grid-y block runs `group_pairs` itself, whose expert numbering comes from atomics,
deterministic only while all pairs sit in one wave (P <= 32); above that, blocks number the experts differently, so the slices
overlap and miss experts (the 5-8-row passes ran in 13.7-16.1 ms, the k4 / k5 screen arms on that binary accepted 40 of 120 drafts).
A fix needs a deterministic numbering pass in `group_pairs` for a gain below 1 % of a pass, so the code is **reverted**:
`kernels/` is identical to main on this branch. Kept: the bench's real-text union mode and the probe's timing knobs.

What this means for the experts: the decode / rows kernels already run a multi-row call at the 1-row bandwidth (445 GB/s at T = 1,
464-480 at T = 2..6 in the bench). A verify row costs its union's bytes; making that cheaper means raising the kernels' base rate
(470 of ~600 GB/s attainable), which is decode work as much as verify work, or fewer distinct experts per pass (drafter side).
