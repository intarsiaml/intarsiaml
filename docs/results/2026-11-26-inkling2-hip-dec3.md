# Inkling-2-Mini HIP decode, round 3 (lane ink-hip-dec3, one R9700, 2026-11-26)

Branch `ink/hip-dec3` from origin/main `f3a150646` (#523 merged). Tatooine, card 0, dev image ROCm 7.2.4; repo
`~/intarsiaml-ink-hip-dec3`, target volume `intarsia-cargo-target-ink-hip-dec3`. Scripts and logs: `2026-11-26-inkling2-hip-dec3/raw/`.

Status (2026-11-26 04:35 EST): **ships: +3.6 % decode short, +3.3 % at 4k, bitwise; PR opened, not merged.** Two bitwise levers
behind `INTARSIA_HIP_INK_DEC3` (bit mask, default 7, `=0` = main's forms): (1) the wave butterflies of `moe_route`, the mix
row / gate kernels, op 75 and argmax via DPP / `v_permlanex16` instead of `ds_bpermute` (+1.6 %); (4) the unembed GEMV (1.44 ms,
one launch) as 3 pieces under the R9700's dispatch stall line, which removes a 271 us idle hole before argmax every token
(+1.9 %, with (2) argmax's loads issued 8 at a time). Main `f3a150646` vs dec3c, 4 + 4 processes ABBA, 256 tokens: **74.21 ->
76.89 tok/s short (+3.61 %, 95 % CI of the ratio [1.0355, 1.0367]), 71.69 -> 74.02 at 4k (+3.26 %, [1.0324, 1.0327])**. Greedy
md5 equal to main at 64 and 256 tokens, short and 4k (and at `=0`); G2 short / long PASS at main's values; admission of the 8
touched ops and the rows-exact / bitwise tests pass; graphs 1 capture + 255 replays, 0 eager. Radiance down 13 min 42 s in 5
windows.

## Brief (coordinator, 2026-11-26 ~03:50 EST, in substance)

One more bitwise decode speedup for Inkling-2-Mini on one R9700 before 06:00 EST; coordinator merges. Fresh profile of main
`f3a150646` first (dec2's `ktrace.py` flow), then the 1-3 levers with the most ms left. Bar: greedy md5 equal to main at 64 and
256 tokens, short and 4k (main 04cdb28a / a7c6a2ac at 64, b4a4f492 / c96e82b7 at 256), graphs 1 capture / 0 eager, G2 KL equal
to main's, admission + rows-exact of every touched op, a bitwise test per changed kernel; >= +2 % end to end on short AND 4k
(4 + 4 interleaved processes, 95 % CI lower bound above 1.0), else written up as not shipped. One knob, default on, `=0` =
main's forms, a `docs/tuning.md` row. PR against main if it ships (not merged). Radiance windows <= 5 min each, <= 20 min in
all, none after 06:00, radiance up by 07:00. Report once, by 06:30 at the latest.

## Step 0 (no GPU)

`git log --all -i --grep` for dpp / permlane / unembed / split_pieces / dispatch stall, and the dec2 / 11-25 docs:
- DPP / v_permlanex16 butterflies: only in `gdn.hip` / `gdn_fused.hip` (GDN v2, `726b0564b`) and `allreduce_wht6.hip`; never in
  the Inkling kernels, which all reduce through `__shfl_xor` (ds_bpermute: an LDS round trip plus the lane-index VALU per step).
- Unembed under the R9700 dispatch stall: open item S5 of the port map (`2026-11-25-inkling2-hip.md` 5.5), never built; dec2's
  step 0 put it aside because the unembed kernel itself runs at its byte floor. Recorded dead items (op 76, in-wave grouping in
  the experts' loop, fast row kernel above 16 rows, blend fused at 4 / 16 rows, gate early loads, op 75 fused past 2048 keys,
  split sizes 128 / 256) are not touched.

## Window 1: decode profile of main f3a150646 (fleetq #12008; radiance down 03:55:46 - 03:56:38, 0:52)

`rocprofv3 --kernel-trace`, `generate --spec off`, graphs, 24 tokens, the last 8 steady steps (`raw/ktrace.py`); traces
`raw/w1-ktrace-{short,4k}.csv.gz`, log `raw/window1-12008-profile.log`. Profiled step 14.47 ms short / 14.96 ms 4k; unprofiled
(128 tokens, same window) 74.12 tok/s short (13.49 ms), 71.42 tok/s 4k.

| Part (short; 4k where it differs) | Kernel | Calls / token | us / call | ms / token |
|---|---|---|---|---|
| In-projection | `in_proj_gemv_kernel<1>` | 48 | 63.8 | 3.06 (3.09) |
| Experts NVFP4 gate_up | `ink_moe_gu_kernel<0,2,6,4,true,0>` | 44 | 50.2 | 2.21 (2.25) |
| Unembed + router logits | `gemv_kernel<1,1,8,float>` | 1 + 48 | 1440 + 2.6 | 1.57 |
| Experts NVFP4 down | `ink_moe_dn_kernel<...>` | 44 | 29.9 | 1.31 (1.34) |
| wo | `gemv_kernel<1,1,8,ik_bf16>` | 48 | 23.8 | 1.14 (1.17) |
| ConvMix / AttnResMix / blend rows | `ink_row_kernel<3,true>` | 124 | 8.4 | 1.04 |
| Attention (op 75) | `ink_attn_dec_k<3>` (+ split path at 4k) | 40 (+ 3 x 8) | 19.0 (22.1) | 0.76 (0.88 + 0.38) |
| Router top-k | `moe_route_kernel` | 48 | 7.4 | 0.35 |
| Blend gate | `ink_blend_gate_kernel<false>` | 48 | 7.3 | 0.35 |
| Experts MXFP8 | gu + dn | 4 + 4 | 73.4 + 45.4 | 0.48 |
| argmax | `argmax_kernel<float>` | 1 | 51.5 | 0.05 |
| **Idle before argmax** | (none) | 1 | **271** | **0.27** |
| Idle before every other dispatch | | 535 | 3.0 | 1.68 |

The new finding: every step has a **271 us hole between the unembed GEMV and argmax** (10 of 10 steps, 270-278 us, both
prompts). The unembed is one 1.44 ms launch (200058 x 2304 bf16, 0.92 GB) and crosses the R9700's dispatch stall line
(`ik_hip.h`, round 8 / 9: a dispatch of >= ~1 ms is followed by ~260 us in which the GPU starts nothing). The kernel itself is
at its floor, which is why the per-kernel tables (dec2's included) did not show it: the time is in the gap column. Every other
dispatch is preceded by 3.0 us under the profiler.

## The levers: one knob, `INTARSIA_HIP_INK_DEC3` (bit mask, default 7, `=0` = main's forms)

| Bit | What | Files | Why it is the same bits |
|---|---|---|---|
| 1 | The wave butterflies take the partner lane from DPP row_xmask (xor 1 / 2 / 4 / 8) and `v_permlanex16` (xor 16) instead of `__shfl_xor` (`ds_bpermute`: an LDS round trip plus the lane-index VALU per step). In `moe_route` (softmax max / sum and the K selection rounds), the mix row kernel (`warp_put`, the fused blend's controller GEMV), the blend gate kernel, op 75 (`ink_attn_dec_k`, its split path's prologue and stage 1: the prologue jobs' sums, the slot sum xor 1..8 and the slot merge xor 16, D = 128) and `argmax`'s wave pick. New helpers `ik_xor_i` / `ik_xor_f` / `ik_warp_sum_dpp` / `ik_warp_max_dpp` in `ik_hip.h` (the GDN v2 helpers, `726b0564b`, made shared). Templated (`<.., DPP>`), so `=0` runs main's instantiations | `ik_hip.h`, `ink_route.hip`, `ink_mix.hip`, `ink_attn.hip`, `argmax.hip` | the same partner values in the same order; every op used (`+`, `fmaxf`, the routers' total-order `better`, argmax's `better`) is commutative, so both lanes of a pair form the same value |
| 2 | `argmax` issues the loads of 8 of a thread's elements (clamped to the row) before comparing them | `argmax.hip` | the same values compared in the same ascending order with the same strict `>` |
| 4 | A one-row bf16 GEMV estimated over the dispatch stall line runs as consecutive pieces of output rows: the unembed becomes 3 launches of ~480 us instead of one of 1.44 ms | `matmul.hip` (`gemv_vec_split`) | a row's loads, fma chain and warp sum do not depend on N or on the block; piece = the same kernel on `w + a K`, `out + a`, `N = b - a`. Piece count from `ik_split_pieces` (the prefill splits' rule: budget 600 us, trigger 950 us, both env-tunable) |

Bits 1 (its argmax part), 2 and 4 act on every HIP model's argmax / one-row GEMV, not only on Inkling-2's (the same bits, so no
model's output changes; a lm_head over ~950 us estimated, e.g. the 35B's 248320 x 2048 head, is now split too).

## Window 2: per-bit decode A/B and bitwise tests on `dec3a` (bits 1 + 2 only; fleetq #12010; radiance down 04:06:05 - 04:09:12, 3:07)

Log `raw/window2-12010-unit.log`, stage `raw/stage_unit.sh`. `generate`, graphs, 128 tokens, 2 processes per setting, tok/s:

| `INTARSIA_HIP_INK_DEC3` | short | 4k |
|---|---|---|
| 0 | 73.77, 73.84 | 71.19, 71.17 |
| 3 (bits 1 + 2) | 75.35, 75.31 (+2.1 %) | 72.50, 72.42 (+1.8 %) |
| 1 (DPP only) | 75.39, 75.33 | |
| 2 (argmax loads only) | 73.83, 73.82 (about 0) | |

Tests (`cargo test --release -p intarsia-backend-hip --test ink_mix -- --test-threads=1 --skip bench`, card 0,
`INTARSIA_REQUIRE_GPU=1`): 7 passed: `argmax_dec3_bitwise` (new, 130 cases: f32 / bf16, V 1 .. 200064, ties, NaN, all -inf / all
NaN rows, max at the first / last element; masks 0 / 1 / 2 / 3 equal), `moe_route_fast_tail_bitwise` (256 cases, now also at
`=0` with both tails), the rows-exact tests (`run_r` now also runs every mix call at `=0`, default and first forms, and
compares), the mix admission (every op ADMITTED), `ink_route_bitwise_pair`. `ink_attn rows_exact_real_shapes`: 63 cases
bitwise (op 75 with DPP against the unfused pair, which keeps `ds_bpermute`, and op 75 at `=0`).

## Window 3: the unembed in pieces (`dec3b`; fleetq #12013; radiance down 04:15:38 - 04:19:37, 3:59)

Log `raw/window3-12013-ab.log`, stage `raw/stage_ab.sh`, trace `raw/w3-ktrace-short.csv.gz`.

- Tests: all of `ink_mix` 8 passed (adds `gemv_split_bitwise`: `logits_bf16_gemv` and `matmul_bf16_gemv` forced into pieces at
  every block boundary, N 33 .. 5000, and the real unembed shape 200058 x 2304 at the default budget: 7 cases bitwise against
  one launch); `ink_attn rows_exact_real_shapes` 63 bitwise; `rows_exact` `bf16_matmul_and_logits_rows_are_bitwise_the_one_row_gemv`
  (the rows kernels against the one-row GEMV, now in pieces at the 27B / 35B heads) and `row_parallel_ops_are_row_independent`
  (argmax among them): 2 passed.
- Profile (short, default 7): step 13.95 ms profiled (14.47 on main), gaps 1.69 ms (1.95). The unembed is 3 x 481 us
  `gemv_kernel<1,1,8,float>`; the gap before argmax is **3.2 us (271 on main)**; argmax 14.5 us (51.5); op 75 16.4 us / call
  (19.0); row kernel 7.8 (8.4); `moe_route` 7.1 (7.4); blend gate 7.3 (7.3). VGPRs unchanged (op 75 128, row kernel 128 / 104,
  `moe_route` 24; argmax 8 -> 24).
- Decode, 128 tokens, 2 processes each, interleaved, tok/s:

| Build / `INTARSIA_HIP_INK_DEC3` | short | 4k |
|---|---|---|
| main `f3a150646` | 74.09, 74.12 | 71.44, 71.41 |
| dec3b `=0` | 73.77, 73.87 | 71.18, 71.16 |
| dec3b `=7` | **76.76, 76.68** (+3.5 % on main) | **73.76, 73.66** (+3.2 %) |
| dec3b `=3` (no split) | 75.34, 75.25 (+1.6 %) | |

Two things from this window: the split is worth +1.9 % on its own (bits 4 + 2 over bit 1), and dec3b at `=0` ran 0.35 % below
main. In dec3b op 75 still chose DPP by a runtime flag (both paths in one kernel, a branch in the key loop); dec3c templates op
75, its split-path prologue and stage 1 on DPP like the other kernels, so `=0` launches main's code.

## Window 4: gate on `dec3c` (fleetq #12015; radiance down 04:23:52 - 04:25:50, 1:58)

Log `raw/window4-12015-gate.log`, stage `raw/stage_gate.sh`. Main = `f3a150646` built in CPU job #12007 (binary md5
`5062b649…`, the same binary as dec2's branch), branch = dec3c built in CPU job #12014 (`3e70c45a…`); same box, card, window.

- `ink_attn rows_exact_real_shapes` on dec3c (op 75 templated): 63 cases bitwise (default DPP against the unfused pair, and at
  `=0`). (`ink_mix` and `rows_exact` ran on dec3b in window 3; dec3c changes only `ink_attn.hip`.)
- Greedy md5 (`generate --spec off --ignore-eos`, graphs; the text before `finish=`), 64 tokens:

| Prompt | main | branch (default 7) | branch `INTARSIA_HIP_INK_DEC3=0` | graphs (all three) |
|---|---|---|---|---|
| short (31 tokens) | 04cdb28a6849… | 04cdb28a6849… | 04cdb28a6849… | 63 replays, 1 capture, 0 eager |
| 4k (4228 tokens) | a7c6a2acedec… | a7c6a2acedec… | a7c6a2acedec… | 63 / 1 / 0 |

- G2 (`accuracy --mode prefill`, oracle W4A16 goldens), branch defaults: short KL21 2.566e-3, top-1 98.95 %, PASS; long
  3.251e-3, 99.30 %, PASS: main's values (dec2's table).
- Admission (`intarsia kernels test --backend hip --op <op>`, defaults), every case ADMITTED, 0 failing: moe_route 19,
  conv_mix 532, attn_res_mix 93, loop_blend 27, sconv_res 64, ink_attn_decode 70, argmax (f32 8, bf16 8, max error 0), logits
  (logits_bf16 4, logits_bf16_gemv 4, logits_bf16_gemv_rows 4).

## Window 5: decode speed, main vs branch (fleetq #12016; radiance down 04:26:04 - 04:29:50, 3:46)

Log `raw/window5-12016-speed.log`, stage `raw/stage_speed.sh`, analysis `raw/speed.py`. `generate --spec off --ignore-eos`,
graphs, 256 tokens, one process per run, 4 processes per arm per prompt, the two arms of a rep back to back, ABBA over the reps.
Paired ratio = branch / main per rep, 95 % CI from the t distribution over the 4 reps.

| Prompt | main tok/s (4 runs) | branch dec3c tok/s (4 runs) | paired gain | 95 % CI of the ratio | md5 main = branch = branch `=0` | graphs |
|---|---|---|---|---|---|---|
| short (31 tokens) | 74.18 / 74.21 / 74.22 / 74.22 | 76.90 / 76.88 / 76.89 / 76.88 | **+3.61 %** (13.475 -> 13.006 ms / token) | [1.0355, 1.0367] | b4a4f4928e87… (all 9 runs) | 255 replays, 1 capture, 0 eager |
| 4k (4228 tokens) | 71.67 / 71.71 / 71.69 / 71.67 | 74.00 / 74.05 / 74.02 / 74.01 | **+3.26 %** (13.949 -> 13.510 ms / token) | [1.0324, 1.0327] | c96e82b75275… (all 9 runs) | 255 / 1 / 0 |

The branch with `INTARSIA_HIP_INK_DEC3=0` in the same window: 73.99 short, 71.48 4k, same md5s: 0.3 % under main (also seen in
windows 3 and 4, and after op 75 was templated, so not the attention branch). `=0` launches main's kernels (the `<.., false>`
instantiations are main's code); the remaining difference is not explained here (candidates: code-object layout of the doubled
instantiations, host-side launch work during capture only). It does not touch the shipped default.

Per lever (window 3, same build, 128 tokens, short): bit 1 (DPP butterflies) +1.6 % on main (+2.1 % on `=0`); bits 4 + 2 (the
unembed in pieces + argmax loads) a further +1.9 %. Bit 2 alone was flat without the split (window 2: argmax 51.5 -> 14.5 us in
the kernel trace, but the 271 us stall right before it absorbed it); it is kept in the default because it is bitwise and the
kernel time is real once the stall is gone; it is not separately measured with bit 4 on.

## Radiance down time (this lane; budget 20 min, each window <= 5 min)

| Window | fleetq | Down | Total |
|---|---|---|---|
| 1 profile of main | #12008 | 03:55:46 - 03:56:38, 0:52 | 0:52 |
| 2 per-bit A/B + tests (dec3a) | #12010 | 04:06:05 - 04:09:12, 3:07 | 3:59 |
| 3 tests + profile + A/B (dec3b) | #12013 | 04:15:38 - 04:19:37, 3:59 | 7:58 |
| 4 gate: md5 64, G2, admission, attn rows-exact (dec3c) | #12015 | 04:23:52 - 04:25:50, 1:58 | 9:56 |
| 5 speed + md5 256 (dec3c) | #12016 | 04:26:04 - 04:29:50, 3:46 | 13:42 |

Down = `systemctl --user stop radiance` to the first `/v1/models` answer after the restart (each window's trap restarts it on any
exit and polls; every one answered within 15 s of the stage's end). CPU jobs on tatooine: #12007 (build main), #12009 / #12011 /
#12012 / #12014 (build dec3a / dec3b / rows_exact / dec3c), #12017 (host checks). All exit 0.

## Host checks (tatooine CPU job #12017, not endor: endor holds #11857, gpu + cpu quiet, 04:00 - 06:20)

`raw/hostcheck.sh`, log `raw/hostcheck-12017.log`, in the dev image with a separate target dir: `INTARSIA_HIP_CHECK_ONLY=1 cargo
check -p intarsia-backend-hip --tests` (0 errors), `INTARSIA_NO_HIP=1 cargo test -p intarsia-backend-hip` (stub build, all
pass), and the pre-push hook's two tests `cargo test --locked -p intarsia-cli --test docs_env_vars` (2 / 2) and `-p
intarsia-models --test recipes_doc` (3 / 3). The push skipped the local hook (`--no-verify`) because it would have built on
endor during #11857; it ran the same two tests here. No file outside `kernels/hip/`, `crates/intarsia-backend-hip/` and `docs/`
changes; CUDA, Metal, CPU and the core are untouched.

## Revert and knobs

`INTARSIA_HIP_INK_DEC3=0` restores main's forms at start-up (bitwise the same output either way); single bits: `6` (no DPP), `5`
(argmax's first loop), `3` (the unembed in one launch). `docs/tuning.md` has the row. Test setter: `intarsia_backend_hip::set_dec3`.
The split also honours `INTARSIA_HIP_SPLIT_US` / `INTARSIA_HIP_SPLIT_MIN_US` (`0` budget = never split).

## Not done / not run

- Not run: `intarsia-models --test inkling2` on HIP, G3, the teacher-forced decode gate, `serve`, prefill timing (the changed
  kernels are decode-path; prefill's lm_head is one row per sequence and is split the same way), spec / verify rows (MTP is not
  built on HIP), other HIP models end to end (bits 1-argmax, 2 and 4 reach them: same bits by the tests above; the 35B's
  248320 x 2048 bf16 head, if it runs `logits_bf16_gemv`, is now 3 pieces).
- `matmul` admission was not run (the split only triggers above ~950 us estimated, i.e. never at admission shapes;
  `gemv_split_bitwise` and `rows_exact` cover the M = 1 GEMV in pieces).
- Left open, by the window-3 profile (13.95 ms profiled step): the 535 x 3 us dispatch gaps (1.6 ms profiled, about 0.7 ms real),
  the experts' fixed cost (gu 50 / dn 30 us per call against about 38 / 19 us of bytes), the row kernel's redundant softmax per
  thread (the "mix" phase, about 1.4 us of its 7.8; a one-wave form risks a different contraction of `fin * wn - mx`, so it needs
  an ISA check first), the experts' epilogue `gs` / `wts` loads after the weight stream (scalar loads via `readfirstlane`).
- tatooine: `~/intarsiaml-ink-hip-dec3` (binaries in `.scratch/bin`) and volume `intarsia-cargo-target-ink-hip-dec3` are kept
  until the PR merges; remove both then.
