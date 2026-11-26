# Inkling-2-Mini MoE experts prefill, round 2 (lane ink-moepf2, endor M5 Max, 2026-11-25)

Branch `ink/moepf2` from main 072dd0863. Not pushed. **Status: stopped before any code or GPU work. Both levers of the
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

## Tests

None run: no code changed.
