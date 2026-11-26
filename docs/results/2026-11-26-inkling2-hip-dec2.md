# Inkling-2-Mini HIP decode, round 2 (lane ink-hip-dec2, one R9700, 2026-11-26)

Branch `ink/hip-dec2` from origin/main `f569b46a1`. Tatooine, card 0, dev image ROCm 7.2.4; repo `~/intarsiaml-ink-hip-dec2`, target
volume `intarsia-cargo-target-ink-hip-dec2`. Scripts and logs: `2026-11-26-inkling2-hip-dec2/raw/`.

Status (2026-11-26 03:25 EST): **one bitwise lever, +2.0 % decode at short and 4k, gated; PR #523 (https://github.com/intarsiaml/intarsiaml/pull/523), not merged.** Three
order-preserving latency cuts in the small decode kernels behind `INTARSIA_HIP_INK_DEC_FAST` (bit mask, default 7 = on, `=0` =
main's forms): the experts' one-wave grouping at <= 16 pairs (+0.8 %), the blend gate kernel's fused-norm tail (+1.1 %), the
`moe_route` tail (+0.2 %). Main `f569b46a1` vs the branch `751f15646`, 4 + 4 processes ABBA, 256 tokens: **72.75 -> 74.22 tok/s
short (+2.01 %, 95 % CI of the ratio [1.0195, 1.0208]), 70.27 -> 71.70 at 4k (+2.05 %, [1.0200, 1.0210])**. Greedy md5 equal to
main at 64 and 256 tokens, short and 4k (and with the knob at 0); G2 short / long PASS at main's values; admission of the five
touched ops and the rows-exact tests pass; graphs 1 capture + 255 replays, 0 eager. Radiance down 12 min 55 s in 4 windows.

## Brief (coordinator, 2026-11-26 ~02:50 EST, in substance)

The next bitwise decode speedup on one R9700, gated and opened as a PR (not merged), before 06:00 EST. Step 0 without the GPU: the
bitwise levers the 11-25 docs leave open; one short window for a fresh decode profile of main (short and 4k: per-op time against
its byte floor, dispatch count, idle gaps); pick the top lever by ms / token. Build it behind an env knob (default on only if
bitwise, `=0` reverts). Bitwise = greedy `generate` md5 equal to main at 64 and 256 tokens, short and 4k prompts; the HIP
rows-exact tests and the admission pass. Gate: admission of the touched ops, rows-exact, md5 vs main, G2 short / long, graph
counts, decode short / 4k for both builds (>= 4 processes per arm, interleaved), the paired gain with a CI. PR if >= 1 % at short
or 4k with the CI's lower bound above 1.0; else the doc on the branch, not pushed. Radiance windows <= 5 min each, <= 20 min in
all, none after 06:00, radiance answers `/v1/models` before release. Stop by 05:50.

## Step 0: levers left by the 11-25 docs (bitwise, not dead)

From `2026-11-25-inkling2-hip.md` (port map, profile), `-hip-attn.md`, `-hip-mix.md`, `-hip-moe.md`:

| Lever | Source | State before this lane |
|---|---|---|
| Fused router `InkRoute` (op 76) on HIP | mix doc, speed round (a) | built, admitted, bitwise its pair; not faster in a microbench (15.9 vs 15.7 us); end to end untested |
| Experts: one launch for gate_up + down | moe doc section 4.3 / 5.9 | not built (about 3 us of 82 per call; needs an in-kernel grid dependency) |
| Experts: the call's fixed cost (5.9 us launches, 5.9 us grouping, 12.7 us tail of the 81.6 us at T = 1) | moe doc windows 3-4 | grouping inside the wave (class 4) measured slower (registers in the main loop): dead in that form |
| Blend: 2 launches (row kernel + gate with the fused norm), ~20 us against a 5.6 us weight floor | mix doc | open |
| Row kernel (ConvMix / AttnResMix) reduction chain, ~8-10 us per call | mix doc | load phase fixed (#518); the rest is the reduction chain |
| `moe_route` (top-k on one wave), 7.9 us per call | port map profile | open |
| Unembed split under the 1 ms dispatch stall | port map 5.5 | the graph profile shows the unembed at its floor (below) |
| Dead | | fast row kernel above 16 rows; blend fused at 4 / 16 rows; gate kernel early loads; op 75 fused on full layers past 2048 keys; split-size 128 / 256 |

Prefill items (flash, WMMA experts, the moe doc's section 5) are not this lane's target.

## Window 1: decode profile of main f569b46a1 (fleetq #12000; radiance down 02:53:35 - 02:55:28, 1 min 53 s)

`rocprofv3 --kernel-trace`, `generate --spec off`, graphs, 24 tokens, the last 8 steady steps (`raw/ktrace.py`); traces
`raw/w1-ktrace-{short,4k}.csv.gz`, log `raw/window1-12000-profile.log`. Under the profiler a step is 14.73 ms (short) / 15.24 ms
(4k) against 13.75 / 14.28 ms unprofiled (72.7 / 70.0 tok/s), so about 1 ms of the gaps is the profiler's.

| Part (short prompt; 4k in brackets where it differs) | Kernels | Calls / token | ms / token | Byte floor ms |
|---|---|---|---|---|
| Experts NVFP4 | `ink_moe_gu_kernel` 2.25 (2.29) + `ink_moe_dn_kernel` 1.35 (1.38) | 44 + 44 | 3.60 (3.66) | 2.51 |
| In-projection | `in_proj_gemv_kernel<1>` | 48 | 3.06 (3.10) | 3.05 |
| Idle between dispatches (profiled) | | 536 | 1.96 (1.96) | 0 |
| Unembed + router logits | `gemv_kernel<..float>` | 49 | 1.57 | 1.51 |
| wo | `gemv_kernel<..ik_bf16>` | 48 | 1.14 (1.17) | 1.08 |
| ConvMix / AttnResMix / blend rows | `ink_row_kernel<3, true>` | 124 | 1.03 (1.05) | about 0 |
| Attention | `ink_attn_dec_k` 0.76 (0.88) + `ink_attn_s1_k` 0.12 (0.30) + prologue 0.05 + comb 0.01 (0.02) | 40 + 3 x 8 | 0.94 (1.27) | about 0 (short) |
| Blend gate (+ fused norm) | `ink_blend_gate_kernel<false>` | 48 | 0.50 (0.53) | 0.14 (gain + forget, 1.77 MB per call) |
| Experts MXFP8 | gu 0.30 + dn 0.19 | 4 + 4 | 0.49 | 0.42 |
| Router top-k | `moe_route_kernel` | 48 | 0.38 (0.39), 7.9 us per call | about 0 |
| argmax, copies, norm, embedding | | 11 | 0.07 | |

Since the pre-#518 profile (712 dispatches, 2.56 ms idle, 17.4 ms): 176 fewer dispatches (op 75 on the window layers, blend in 2
launches); the dense GEMVs (in-projection, wo, unembed: 5.8 ms) stay at 95-100 % of their floor. The excess left: experts 1.1 ms,
idle 1.0-2.0 ms, latency-bound small kernels 2.9 ms (row kernel 1.03, attention 0.94, blend gate 0.50, route 0.38).

Same window, the existing fused router end to end (main's binary, 128 tokens, 2 processes per arm): `INTARSIA_INK_ROUTE_FUSED=1`
72.53 / 72.52 tok/s short, 69.93 / - 4k against unset 72.65 / 72.68, 70.03 / 70.00: about -0.15 %. **Op 76 on HIP is dead**
(its last-block tail costs more than the dispatch it saves, in the graph too). Run-to-run spread of one build: 0.05 %.

## The lever: three latency tails, one knob (`INTARSIA_HIP_INK_DEC_FAST`, bit mask, default 7, `=0` = main's forms)

No single open item was worth much more than 1 % on its own, so the lane took the three cheapest order-preserving cuts of the
small kernels' fixed latency and gated them together, one bit each (per-bit gains measured below). None changes an operation,
an operand or an order: only where values are loaded from and which lane writes them.

| Bit | Kernel (file) | Before (main) | After |
|---|---|---|---|
| 1 | `ink_moe_gu_kernel` / `ink_moe_dn_kernel`, LDS size class 0 at <= 16 pairs (decode: T <= 2 at top-8), NVFP4 and MXFP8 (`ink_moe.hip`) | `group_pairs`: all 256 threads build the expert tables with LDS atomics, six block barriers; the down kernel's token sum reloads the K ids from global memory one at a time | `group_wave`: wave 0 builds the same class-0 tables (lane p = pair p; leaders by `readlane`, ranks by ballot / popcount), one barrier; the valid-pair mask stays in LDS for the token sum. Duplicate / invalid ids handled as before. The table order decides only which wave runs what (the file's own invariant) |
| 2 | `ink_blend_gate_kernel`, the fused norm in the last block of a (stream, row group) (`ink_mix.hip`) | the row's 18 slice partials summed by one thread straight from global memory, then the 2304-wide norm loop one load pair at a time | every thread stages the partials into LDS first (the sum still runs on one thread in slice order from 0.f); the norm loop issues 4 iterations' loads before their products |
| 4 | `moe_route_kernel` (`ink_route.hip`) | lane 0 keeps the K picks in LDS, sums their scores and writes the K weights / ids serially | lane k keeps pick k in a register and writes output k; every lane forms the sum over `readlane` of the same values in pick order |

Not taken: the one-launch gate_up + down (an in-kernel grid dependency: deadlock / graph-replay risk), moving the experts'
epilogue loads (`gs`, `wts`) ahead of the weight stream (more live registers in a kernel at the occupancy edge), any change to
`moe_route`'s selection rounds (its tie / NaN / exhaustion behaviour), and op 76 (dead above).

## Window 2: bitwise tests, expert bench, per-bit decode (fleetq #12002 on `dc123d783`; radiance down 03:07:34 - 03:10:54, 3 min 20 s)

Log `raw/window2-12002-unit.log`, stage `raw/stage_unit.sh`. Tests (`cargo test --release -p intarsia-backend-hip`, card 0,
`INTARSIA_REQUIRE_GPU=1`): `ink_moe_rows_exact` (now also class 0 with the one-wave grouping on and off against the first form,
T 1 .. 70) and the new `ink_moe_wave_grouping_odd_ids` (60 cases: duplicates, ids -1 / >= E, one expert, all invalid, T 1 .. 4,
K 1 .. 32, both formats) pass; all of `ink_mix` passes (`run_r` now also runs every mix call with the first fused-norm tail and
compares with the first forms; the new `moe_route_fast_tail_bitwise`, 256 cases incl. ties, NaN / inf logits and fewer finite
scores than K; `ink_route_bitwise_pair`, which compares op 76's unchanged tail with the new `moe_route`; the admission of the
mix ops).

Expert bench (`INK_MOE_BENCH=1`, real shape, 12 routings, eager, us per call; `INTARSIA_HIP_INK_DEC_FAST=0` -> `7`):

| Format, T | both launches | gate_up | down | fixed-cost probe "after the grouping" |
|---|---|---|---|---|
| NVFP4, 1 | 83.1 -> **80.2** | 50.1 -> 48.9 | 31.4 -> 29.5 | 11.8 -> 9.7 |
| NVFP4, 2 | 141.0 -> **138.3** | 91.7 -> 91.1 | 50.4 -> 47.9 | 11.9 -> 11.3 |
| NVFP4, 4 (32 pairs) | 246.4 -> 247.0 | 161.8 -> 162.0 | 85.3 -> 85.1 | 11.8 -> 15.4 |
| MXFP8 (64 experts), 1 | 123.0 -> **119.7** | 74.8 -> 73.4 | 48.2 -> 46.4 | 12.5 -> 9.8 |
| MXFP8, 2 | 217.4 -> 216.1 | 134.4 -> 134.0 | 83.2 -> 82.2 | 12.6 -> 11.9 |

The probe at 32 pairs got slower, so after this window the one-wave grouping is limited to 16 pairs (`kWaveGroupPairs`,
`751f15646`): T <= 2 at top-8. Plain decode is T = 1.

Decode in the window-2 binary by the knob (`generate`, graphs, 128 tokens, 2 processes per setting, interleaved), tok/s:

| `INTARSIA_HIP_INK_DEC_FAST` | short | 4k |
|---|---|---|
| 0 (main's forms) | 72.61, 72.59 | 69.96, 70.00 |
| 7 (all) | 74.06, 74.15 (+2.07 %) | 71.44, 71.42 (+2.07 %) |
| 6 (bit 1 off) | 73.47, 73.56: bit 1 = +0.80 % | |
| 5 (bit 2 off) | 73.31, 73.24: bit 2 = +1.13 % | |
| 3 (bit 4 off) | 73.91, 73.94: bit 4 = +0.24 % | |

The three bits add up (0.80 + 1.13 + 0.24 = 2.17 % against 2.07 % measured together). Main's binary in window 1, same lengths:
72.65 / 72.68 short, 70.03 / 70.00 4k, so `=0` is main's speed.

## Window 3: gate on `751f15646` (fleetq #12004; radiance down 03:13:47 - 03:17:41, 3 min 54 s)

Log `raw/window3-12004-gate.log`, stage `raw/stage_gate.sh`. Main = `f569b46a1` built in CPU job #11999 (binary md5
`3035a586…`), branch = `751f15646` built in CPU job #12003 (`5062b649…`); same box, card, window.

- Tests: `ink_moe_rows_exact`, `ink_moe_wave_grouping_odd_ids` (60 bitwise), all of `ink_mix` (6 passed: the rows-exact tests,
  `moe_route_fast_tail_bitwise` 256 bitwise, `ink_route_bitwise_pair`, the mix admission): pass.
- Admission (`intarsia kernels test --backend hip --op <op>`, defaults): moe_route 19, moe_experts_mxfp8 19 / 14 (`_rows`),
  loop_blend 27, conv_mix 532, attn_res_mix 93, moe_experts (situ 10, situ `_rows` 7, and the bf16 forms 12 / 12 / 12 / 4):
  every case ADMITTED, 0 failing.
- G2 (`accuracy --mode prefill`, oracle W4A16 goldens), branch defaults: short KL21 2.566e-3, top-1 98.95 %, PASS; long
  3.251e-3, 99.30 %, PASS: the values of main / #518.
- Greedy md5 (`generate --spec off --ignore-eos`, graphs; the text before `finish=`), 64 tokens:

| Prompt | main | branch (default) | branch `INTARSIA_HIP_INK_DEC_FAST=0` | graphs (all three) |
|---|---|---|---|---|
| short (31 tokens) | 04cdb28a6849… | 04cdb28a6849… | 04cdb28a6849… | 63 replays, 1 capture, 0 eager |
| 4k (4228 tokens) | a7c6a2acedec… | a7c6a2acedec… | a7c6a2acedec… | 63 / 1 / 0 |

(The same md5s as the #518 PR table: main's output has not changed since #516.)

## Window 4: decode speed, main vs branch (fleetq #12005; radiance down 03:18:20 - 03:22:08, 3 min 48 s)

Log `raw/window4-12005-speed.log`, stage `raw/stage_speed.sh`, analysis `raw/speed.py`. `generate --spec off --ignore-eos`,
graphs, 256 tokens, one process per run, 4 processes per arm per prompt, the two arms of a rep back to back, ABBA over the reps.
Paired ratio = branch / main per rep, 95 % CI from the t distribution over the 4 reps.

| Prompt | main tok/s (4 runs) | branch tok/s (4 runs) | paired gain | 95 % CI of the ratio | md5 main = branch = branch `=0` | graphs |
|---|---|---|---|---|---|---|
| short (31 tokens) | 72.72 / 72.73 / 72.78 / 72.78 | 74.17 / 74.22 / 74.21 / 74.27 | **+2.01 %** (13.746 -> 13.474 ms / token) | [1.0195, 1.0208] | b4a4f4928e87… (all 9 runs) | 255 replays, 1 capture, 0 eager |
| 4k (4228 tokens) | 70.23 / 70.29 / 70.29 / 70.25 | 71.66 / 71.71 / 71.76 / 71.69 | **+2.05 %** (14.231 -> 13.947 ms / token) | [1.0200, 1.0210] | c96e82b75275… (all 9 runs) | 255 / 1 / 0 |

The branch with `INTARSIA_HIP_INK_DEC_FAST=0` in the same window: 72.69 short, 70.22 4k (main's speed), same md5s. 0.27 ms /
token saved at short context, of which (window 2) about 0.11 ms is the experts' grouping, 0.15 ms the blend gate tail and
0.03 ms the router tail.

## Radiance down time (this lane; budget 20 min, each window <= 5 min)

| Window | fleetq | Down | Total |
|---|---|---|---|
| 1 profile + op 76 A/B | #12000 | 02:53:35 - 02:55:28, 1:53 | 1:53 |
| 2 tests + bench + per-bit A/B | #12002 | 03:07:34 - 03:10:54, 3:20 | 5:13 |
| 3 gate: tests, md5 64, G2, admission | #12004 | 03:13:47 - 03:17:41, 3:54 | 9:07 |
| 4 speed + md5 256 | #12005 | 03:18:20 - 03:22:08, 3:48 | 12:55 |

Down = `systemctl --user stop radiance` to the first `/v1/models` answer after the restart (each window's trap restarts it on any
exit and polls; every one answered within 15 s of the start). CPU jobs: #11999 (clean the merged lanes' leftovers on tatooine:
`~/intarsiaml-ink-hip{,-mix,-attn,-speed}`, volumes `intarsia-cargo-target-ink-hip-{mix,attn}`; `/` 94 % -> 91 %; build main),
#12001 and #12003 (build the branch). All exit 0.

## Revert and knobs

`INTARSIA_HIP_INK_DEC_FAST=0` restores main's three forms at start-up (bitwise the same output either way); single bits: `6`
(no one-wave grouping), `5` (first blend gate tail), `3` (first `moe_route` tail). `docs/tuning.md` has the row (`docs_env_vars`
passes). Test setters: `inkmoe::set_dec`, `inkmix::set_mix_dec`, `inkroute::set_route_dec`.

## Host checks (endor)

`cargo test -p intarsia-backend-hip` (lib 32 + every test target, stub build), `INTARSIA_HIP_CHECK_ONLY=1 cargo check -p
intarsia-backend-hip --tests` (0 errors), `cargo test -p intarsia-cli --test docs_env_vars` (2 / 2). No file outside
`kernels/hip/`, `crates/intarsia-backend-hip/` and `docs/` changes; CUDA, Metal, CPU and the core are untouched.

## Not done / not run

- Not run: `intarsia-models --test inkling2` on HIP (G1 tiny fixtures; the touched ops are covered by the admission, the
  rows-exact tests and G2), G3, the teacher-forced decode gate, `serve`, prefill timing (decode-only change: prefill chunks have
  more than 16 pairs and take the unchanged paths), a profile of the branch (the per-bit A/B stands in for it), spec / verify
  rows (MTP is not built on HIP).
- The one-wave grouping at 17-32 pairs (T 3-4) stays off: its grouping probe was slower at 32 pairs (window 2).
- Left open for a next round, by the window-1 table: the experts' remaining fixed cost (about 20 us per call above the
  bandwidth line: the last round of blocks, the term barrier, the epilogue's `gs` / `wts` loads after the weight stream), the
  row kernel's reduction chain (124 calls, 8.3 us each), op 75's 19 us at short context, `moe_route`'s selection rounds (7.5 us
  of one wave), and the blend gate kernel's `dbuf` / `msg` loads after its fma chains.
- tatooine: `~/intarsiaml-ink-hip-dec2` (binaries in `.scratch/bin`) and volume `intarsia-cargo-target-ink-hip-dec2` are kept
  until the PR merges; remove both then.
