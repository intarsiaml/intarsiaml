# Inkling-2-Mini MoE experts prefill, round 2 (lane ink-moepf2, endor M5 Max, 2026-11-25)

Branch `ink/moepf2` from main 072dd0863. Not pushed. **Status (final): the two levers of brief 1 were already measured slower on `ink/prefill` (below); brief 2's mixed-tile form is built, bitwise, opt-in, about 1.8 % of prefill GPU time.** Brief 1 status was: stopped before any code or GPU work, both levers of the
brief were already built and measured on the unmerged branch `ink/prefill`, and both were slower.** No fleetq slot taken.

## Briefs received

2. 2026-11-25 ~21:30 EST (coordinator): **go ahead with the small-expert tile idea.**
   1. Bitwise form first: per-expert M tile size (experts with few rows get a 16- or 32-row M tile instead of 64) in the
      same kernel family, same K loop, same per-element accumulation order. Before building, read the kernel and confirm
      the MMA fragment order per output element does not depend on the M tile size; if it does, say so and stop this form.
   2. Probe first, cheap: kernel bench with real routing captured from a 4k prompt; time per call of the current form vs
      the small-tile form vs the balanced-routing floor (the ~1.05 ms / call gap). Under ~2 % of a chunk: write up, stop.
   3. If it wins: admission cases, `ink_prefill_moe` / `inkmoe` tests, same greedy ids at 1k / 4k / 16k, prefill ABBA A/B
      at 1k / 4k / 16k with >= 6 runs per arm. Default-on only if bitwise and faster than noise; else env knob + tuning row.
   4. If the bitwise form is impossible: a GEMV-style path for experts under 16 rows, opt-in only, with the KL gate.
   - endor: 10-20 min `--exclusive` kernel-bench slots; #11871, #11856, #11857 untouchable. Report by 08:00 EST with a
     table, the bitwise result, tests, branch head, fleetq ids. The coordinator moves the `ink/prefill` doc to main;
     this lane does not touch it.

3. 2026-11-25 ~21:47 EST (coordinator): lane ink-mode (`.worktrees/ink-mode`, `ink/mode`) hunts a per-process fast / slow
   mode on endor (~14 % at 16k) and was told of this lane's 7.5 -> 15 ms in-process drift as a possible shared cause. The
   interleaved `INKMOE_ROUNDS` plan is right; the doc must also give each form's best round and the per-round series, not
   only medians. Bitwise on 36 cases is good; no merge before a clean speed win.

## Blocker: the two levers are already measured (overlap check, step 0 of the brief)

The brief took its levers from "Next ideas" item 1 of `docs/results/2026-11-21-inkling2-prefill.md` as it stands on main.
The `ink/prefill` branch (worktree `.worktrees/ink-prefill`, head 50d0ad83d, 96 commits ahead of main) carries a later
version of that doc, and its MoE work never reached main (commits 33bb055d4, 9acbb5163, f753bb1c8, e09f536f2 are on
`ink/prefill`, `ink/blend-piece`, `ink/prefill-rel2`, `ink/prologue-split` only). All numbers below are quoted from that
doc; this lane re-measured none of them.

| Brief lever | Already built as | Result there | Job |
|---|---|---|---|
| Bitwise: one decoded weight stage shared by two 64-row M tiles of the same expert (256 threads) | `INTARSIA_INK_MOE_NAX2=m128_k64_v65` (33bb055d4) | bitwise, admitted; bench 11.0 vs 7.3 ms per NVFP4 call; e2e 8k ~2406 vs ~2670 tok/s (-10 %) | #10981 / #10982 |
| Bitwise: cheaper fp4 -> bf16 decode | upper bound only: probe B, a bit copy in place of the decode (f753bb1c8) | saves 392 of 6188 µs per call (6.3 % of the experts, about 3.4 % of a chunk) with the decode removed entirely | #11235 |
| Lossy: dequantize each expert once to bf16, then the NAX bf16 GEMM | `INTARSIA_INK_MOE_NAX2=dq1`, kernel `ink_moe_dq_nv` (9acbb5163) | 14.8 vs 7.3 ms per call, the decode pass alone 9.3 ms (2 GB of bf16 written and re-read per call); output was bitwise gen1, so the "not bitwise" part of the review claim is also wrong | #10981 |

Related closed probes on the same branch: 128-row tiles (m128) 13-17.5 ms; K tile 128 instead of 64 +98 % (#11282, the
32768 B threadgroup limit); no weight loads at all saves 746 µs (#11235). That doc's reading: the cost is ~15.2 µs per
64-row tile (325 tiles per call with real routing), of which decode + DRAM loads explain ~4.5 µs and the rest is the K
loop's structure (36 staged steps, two barriers each). Its conclusion: "MoE decode work stops here", and the later ranking
excludes the expert kernel ("only a format change would move it").

The Neon estimate of +15-18 % prefill for the dequant-once lever does not hold: the measured form is 2x slower per call.
A variant that decodes once per prefill instead of once per call would need about 2.3 GB of bf16 per NVFP4 layer x 44
layers (~100 GB), which does not fit endor.

## Step 1 (re-profile on main): not run

endor's first exclusive slot tonight was 01:25 (`fleetq when`, behind #11915 and #11916, then #11871). The latest profile
on record is the Metal lane's on main ca3b4e55b (#11757, 4k prompt, chunk 2048): experts 759 of 1337 ms GPU (57 %), floor
404. No commit since changes the expert kernels, so the experts are still the top item; this lane did not re-measure it.

## Not tried anywhere (candidates, none measured)

- Fewer tiles under real routing: 21 % of tile rows are padding and a median of 27 experts per call have under 16 rows,
  each paying a full 64-row tile (~15 µs). Real routing costs ~1.05 ms per call over balanced (~46 ms per chunk, ~7 %).
  A form that sends small experts to a cheaper path, or packs them, would be bitwise only if the per-row chunk order
  and reductions stay the same.
- The per-tile ~10 µs that the probes did not explain (K-loop staging and barriers). No probe isolates it yet.

## Mixed-tile form (brief 2): `INTARSIA_INK_MOE_MIX`, opt-in

In a prefill call on 64-row tiles, an expert's last (rows mod 64) rows run on a 16- or 32-row tile instead of a zero-padded
64-row tile. `ink_moe_group_mix` (inkling2_moe.metal) leaves those remainders out of the 64-row tile table; the remainder
kernels `ink_moe_nax_{gu,dn}_{nv,mx}_r{16,32}[k64]` (inkling2_moe_nax.metal, the same `inx_mm_impl` with a remainder
lookup by expert) run one threadgroup row per expert and return unless that expert's remainder is in their class.
Forms: `16`, `32`, `16+32`, each with K step 128 (default) or 64 (`k64` suffix). No extra scratch.

**MMA order (brief 2 step 1).** The kernel source cannot settle it: the K reduction runs inside Apple's `matmul2d`
(multiply_accumulate into an f32 cooperative tensor). The launcher-visible differences of a small tile are the M / N
sub-tile of the descriptor (16 x 16 or 16 x 32 instead of 32 x 32) and, for the existing small kernels, the K step (128
instead of 64). So the check is empirical, and it holds for every case tried.

**Bitwise (fleetq #11920, `tests/ink_moe_mix.rs`, 31b081298):** real shape (H 2304, I 1152, 128 experts, top-8, SiTU),
NVFP4 and MXFP8, 3 routings (random at 2048 rows, random at 600 rows, a routing with experts of 0 / 1 / 2 / 7 / 8 / 9 /
15 / 16 / 17 / 31 / 32 / 33 / 63 / 64 / 65 ... rows at 2048 rows) x 6 forms: **0 differing bf16 elements in all 36
cases**, output poisoned before each run, counter confirms the mixed path ran. The bench's output hash is also the same
for every form on the replayed real routing.

Real routing used: `ink/prefill`'s dump of an 8k prompt (`.scratch/routes8k.txt`, copied from that lane's
`.scratch/md/routes.txt`; 132 NVFP4 calls at 2048 rows), not a new 4k capture. Per call (median): 325 64-row tiles, of
which 24 remainders of 1-8 rows, 18 of 9-16, 31 of 17-32, 47 above 32.

**Timing, probe 1 (#11920): not usable.** Forms ran one after another in one process, and every process slowed from
~7.5 to ~15 ms per call within about 4 s whatever the form (the uniform form measured 7.9 ms first and 16.8 ms last in the
same process). The only undisturbed stretch (first 2 s of the first process): uniform 7889, `16` 7418, `32` 7458 µs per
call. Probe 2 interleaves the forms round by round.

### Probe 2 (fleetq #11934, exclusive, bench at 35af9af69, md5 84b29f4f...): interleaved rounds

`inkmoe_bench` with `INKMOE_ROUNDS`: every form gets 12 calls per round, the order rotates each round, 24 rounds (control:
16). Raw log with every series: `2026-11-25-inkling2-moepf2/raw/probe2-11934.log` (probe 1: `raw/probe1-11920.log`).
`pmset -g therm` reported no thermal or performance warning before or after; powermode 2.

**The drift is in the uniform form too (control: the uniform form four times, real routing).** µs per call per round:

| slot | series (16 rounds) |
|---|---|
| 1 | 7235 9677 9330 9162 9104 9056 9027 9069 9085 8987 9000 8789 8878 9104 9138 9124 |
| 2 | 7251 7580 9229 9134 9090 9077 9028 9091 9063 9034 8932 8848 8910 9000 9107 9154 |
| 3 | 7275 9324 9555 9139 9081 9056 9059 9089 9072 8979 8976 8854 8952 9065 9158 9144 |
| 4 | 7244 9716 9413 9199 9093 9075 9016 9030 9009 8993 8980 9045 8964 9128 9138 9154 |

Every process starts at about 7.2 ms per call and settles 25 % slower (about 9.1 ms) after 0.4 to 2 s of GPU work; a new
process starts fast again. Slot-to-slot ratio per round: median 1.000 to 1.002. So the step belongs to the process or
the box, not to a form, and per-round ratios are the usable number. (Passed to lane ink-mode by the coordinator.)

**NVFP4, 2048 rows, real routing (8k dump), two processes:**

| form | best round µs | median µs (rep 1 / rep 2) | per-round ratio to uniform, median (rep 1 / rep 2) | ratio min .. max (rep 1) |
|---|---:|---:|---:|---|
| uniform (`0`) | 7238 / 7263 | 8859 / 8749 | 1 | |
| `16` | 7131 / 7115 | 8659 / 8653 | 0.981 / 0.981 | 0.955 .. 0.998 |
| `32` | 7157 / 7164 | 8693 / 8680 | 0.984 / 0.985 | 0.963 .. 1.026 |
| `16+32` | 7138 / 7130 | 8586 / 8634 | 0.974 / 0.975 | 0.960 .. 1.009 |
| **`32k64`** | **7010 / 7010** | 8501 / 8540 | **0.960 / 0.961** | 0.947 .. 1.001 |
| `16+32k64` | 7023 / 7053 | 8503 / 8526 | 0.959 / 0.963 | 0.953 .. 0.980 |

Series, rep 1 (µs per call, rounds 1-24):

| form | series |
|---|---|
| uniform | 7238 7264 7264 7283 7801 8088 8518 8884 8938 8896 8844 8773 8745 8819 8849 8978 9021 8993 8972 8892 8895 8920 8929 8859 |
| `16` | 7143 7131 7138 7267 7695 7988 8389 8488 8768 8711 8656 8578 8574 8536 8659 8826 8847 8859 8780 8692 8722 8730 8722 8734 |
| `32` | 7165 7157 7169 7473 7810 8148 8472 8550 8784 8733 8676 8620 8604 8623 8693 8842 8863 8827 8827 8723 8756 8755 8756 8754 |
| `16+32` | 7171 7137 7150 7157 7781 8159 8422 8526 8697 8731 8568 8499 8518 8509 8586 8715 8917 8729 8728 8650 8637 8694 8621 8688 |
| `32k64` | 7015 7010 7012 7023 7387 8097 8337 8446 8558 8571 8501 8408 8398 8384 8437 8560 8678 8593 8573 8525 8539 8556 8588 8556 |
| `16+32k64` | 7046 7037 7023 7027 7463 7717 8344 8530 8558 8546 8497 8418 8380 8415 8432 8606 8620 8669 8567 8512 8503 8543 8543 8503 |

**Other routings and MXFP8 (ratio to uniform per round, median; best round µs uniform -> form):**

| case | `16` | `32` | `16+32` |
|---|---|---|---|
| NVFP4 balanced 128 x 128 rows (no remainders: only the empty small dispatches) | 1.003 (6273 -> 6299) | 1.003 (6273 -> 6343) | 1.001 (6273 -> 6431) |
| NVFP4 random routing (noisy process: uniform 7.5 -> 12.5 -> 9.2 ms) | 0.958 | 0.980 | 0.964 |
| MXFP8 real routing (noisy: 7.7 -> 11-12 ms) | 0.977 | 0.980 | 0.978 |

Reading:
- The K step 64 small tiles beat the K step 128 ones (0.960 vs 0.984 for `32`), and one 32-row class does as well as
  two classes. **Pick: `32k64`** (one extra dispatch per stage).
- **Gain: 4.0 % of an NVFP4 experts call in the steady state, 3.1 % by best rounds (7238 -> 7010 µs).** With the experts
  at 54-57 % of a chunk that is **about 2.2 % of a prefill chunk** (1.7 % by best rounds): at the brief's stop line of
  ~2 %, not clearly over it.
- The empty threadgroups of the small dispatches cost 0.1-0.3 % on a call with no remainders: inside the noise.
- It recovers 0.23 of the 0.97 ms between real routing (7238) and balanced (6273). A 32-row tile is only about 3 µs
  cheaper than a 64-row tile (0.23 ms over ~74 tiles): the per-tile cost is the K loop and the weight stream, which do
  not shrink with M. The 47 remainders above 32 rows keep their 64-row tile. Smaller M tiles cannot give more than this.
- `k64` forms were not timed on MXFP8, balanced or random routing.

### End to end (fleetq #11955, exclusive, 23:00-23:11; one binary at f615aa551, md5 56081f24...)

Arm A = knob unset (the uniform 64-row tiles, main's code path), arm B = `INTARSIA_INK_MOE_MIX=32k64`. One binary for both
arms, not a separate main build. `intarsia generate --backend metal --max-seq 20480 --temperature 0`, prompts of lane
ink-relbias-mma (`p4k.txt`, `p16k.txt`; 1k = the first 3100 characters of the 4k prompt). Raw: `raw/e2e-11955.log`.

**Greedy output, 32 tokens:** md5 the same in both arms at 1k (a5aa3433...), 4k (50799fe1...) and 16k (cdbf7d72...). The
trace in arm B shows the remainder kernels ran at every length.

**Kernel trace, 4k prompt, warm second run, order A B B A (GPU ms over the prefill):**

| op | A (1) | B (2) | B (3) | A (4) | B / A (means) |
|---|---:|---:|---:|---:|---:|
| moe_experts_situ_bf16 (NVFP4) | 1349.8 | 1294.6 | 1284.3 | 1328.6 | **0.963** |
| moe_experts_mxfp8_bf16 | 128.7 | 128.5 | 127.5 | 126.5 | 1.003 |
| ink_in_proj_bf16_nax (untouched, reference) | 261.3 | 261.6 | 257.7 | 256.9 | 1.002 |
| all ops | 2603 | 2546 | 2526 | 2561 | **0.982** |

In the stream the NVFP4 experts are 3.7 % faster (the bench said 4.0 %), MXFP8 is unchanged, and the whole prefill's GPU
time drops 1.8 %. The experts are 52 % of the GPU time in arm A, still the top item. (This run was in a slow state: 2.6 s
of GPU time for a 4k prefill against 1.34 s in the Metal lane's #11757 profile.)

**Prefill wall time, ms, ABBA x 3 (warm third run of `--repeat 3`), in run order:**

| length | A | B | median A / B | best A / B |
|---|---|---|---:|---:|
| 4k | 1466 1595 1447 1774 1957 1841 | 1440 1571 1857 1663 1911 1761 | 1684 / 1712 (+1.6 %) | 1447 / 1440 |
| 16k | 7924 10075 8726 7442 6761 6977 | 9680 9492 8110 6888 6825 6843 | 7683 / 7499 (-2.4 %) | 6761 / 6825 |
| 1k | 348 344 343 471 345 383 | 328 535 520 327 397 329 | 346 / 363 (+4.8 %) | 343 / 327 |

**Not faster than noise.** The spread inside one arm is 30-50 % (4k: 1447 to 1957 ms; the usual 4k figure on this box is
about 1250 ms), against an expected gain of 1.8 %. Six runs per arm cannot show it either way, and no rest was taken
between runs.

### Verdict

- **Bitwise: yes.** 36 kernel cases, 0 differing elements; same greedy output at 1k / 4k / 16k.
- **Speed: -3.7 % on the NVFP4 experts in the stream, -1.8 % of prefill GPU time at 4k.** That is at or just under the
  brief's stop line of ~2 % of a chunk, and the wall-time A/B does not resolve it.
- **Stays opt-in** (`INTARSIA_INK_MOE_MIX=32k64`, row in `docs/tuning.md`). Not default-on: the rule was bitwise and
  faster than noise, and only the first holds. No merge asked.
- Brief 2 step 4 (a GEMV-style path for experts under 16 rows) was for the case that the bitwise form is impossible; it
  is possible, so that path was not built.
- What would settle a default: a within-process A/B (lane ink-verify's method) or a rested ABBA with far more runs, in a
  quiet window. The form's ceiling is known either way: about 2 % of a chunk.

## Tests

- `tests/ink_moe_mix.rs` (new): pass in #11920 and again in #11955.
- #11955: `cargo test --release -p intarsia-backend-metal --lib --test inkmoe --test ink_prefill_moe --test ink_moe_mix`:
  lib 18 passed, inkmoe 2, ink_prefill_moe 1, ink_moe_mix 1; `-p intarsia-cli --test docs_env_vars`: 2 passed.
- #11955: `intarsia kernels test --backend metal --quick --op moe_experts` and `--op moe_experts_mxfp8` with the knob on:
  all ADMITTED (12 and 21 cases). These cases are at most 40 rows and never reach 64-row tiles, so they show only that the
  knob leaves the other paths alone; the mixed path itself is covered by `ink_moe_mix`.
- Not run: the full workspace `cargo test`, G2 / G3 (not needed for a bitwise form), localeval `speed-plain`.

## fleetq jobs

#11919 build (cpu), #11920 probe 1 (exclusive), #11933 build (cpu), #11934 probe 2 (exclusive), #11955 end to end + tests
(exclusive). All exit 0, none overlapped a reservation (#11955 started at 23:00 in the slot #11871 vacated when it died).
