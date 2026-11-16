# HIP tp2 all-reduce wire probe: does the change-vs-base KL gate see wire precision? (lane wire-probe), 2026-11-16

Question (from lane wht6-8b, `docs/results/2026-11-16-hip-wht8-ar.md`, PR #383): the wht8 wire has a 4x smaller error than wht6
(rms 0.41 % against 1.66 % of the reduced partial) and the change-vs-base KL was no better (1.76 x / 1.46 x the reference noise
against 1.51 x / 1.42 x). Hypothesis: the rule measures a floor that **any** change beyond bf16 rounding hits (a different
reduction order is enough), so it cannot see the wire's precision. Test: the same kernel at 12 and 16 bits, and a control with no
quantisation at all. Opt-in only; `exact` stays the default; nothing was flipped. This branch includes #383's kernel and wiring
(merged `origin/wht6-8b`) and supersedes it as the kernel base.

## Summary

- Built (opt-in values of `INTARSIA_HIP_AR_WIRE`): `wht12` (12.25 bits per element), `wht16` (16.25), and the control **`whtf`**:
  the same rotation, group of 64 and block layout, but the rotated **f32 values** go on the wire unquantised (32 bits per element,
  the scale slot unused), the two ranks add the rotated values in f32 and rotate back. It is the exact all-reduce up to f32
  rounding of the rotation (rms error below 5e-6 % of the sum, i.e. at most f32 round-off), with the rotation and reduction order
  of the lossy wires.
- **The gate does not see wire precision.** The control, whose wire error is about 1e-7, already sits at **1.10 x** (short golden)
  and **1.35 x** (long golden) the reference noise and flips 19 / 1596 top-1 positions: it **fails the rule too**. Across the five
  wires the KL moves between 1.10 and 1.76 x (short) and 1.35 and 1.68 x (long) with no order in the wire error, which spans
  five orders of magnitude (1.7 % down to f32 round-off). The widest lossy wire (wht16, error 0.0016 %) is above the control, and
  wht12 on the long golden is the worst of all. Every number is deterministic (wht6 and wht8 reproduce #383's figures to all digits:
  5.559e-3 / 3.240e-2 and 4.786e-3 / 3.142e-2), so the spread is not run-to-run noise but the model's response to which bits flip
  (MoE routing near-ties, bf16 rounding of the hidden state): chaotic, not precision-driven.
- **No width passes the change-vs-base rule**, and none can on this model: the floor of a perturbation this small is already over
  the limit. The control's 3.47e-3 matches the chunked GDN delta rule (a reordering of exact arithmetic, 3.455e-3, LOG
  2026-11-15 12:05). The vLLM-golden gates (absolute 3e-2 floor) pass for every width and for the control.
- Speed (35B NVFP4 tp2, 4k prompt, 2 interleaved rounds x 3 reps, one binary): prefill **+22.9 % (wht6), +15.0 % (wht12), +10.1 %
  (wht16)** against exact; decode unchanged. wht12 keeps 65 % of wht6's gain at a 65x smaller wire error; wht16 keeps 44 %.
- Recommendation: the rule as written cannot be a decision tool for any lossy wire here (nor for any non-bit-exact change). If the
  prefill gain is wanted, the candidate is **wht12** (+15 % at 4k, wire error 0.026 %, vLLM-golden gates pass), accepted by a
  user decision the way the GDN chunk kernel was; judge it on a floor-aware criterion (compare with the `whtf` control, see
  section 4), the greedy ids and the vLLM-golden gates, not on KL / reference noise. Not offered as a serving profile here.

## 1. What was built

| file | what |
|---|---|
| `kernels/hip/allreduce_wht6.hip` | symbol width `B` in {6, 8, 12, 16, 32}; new generic exports `ik_ar2_whtw`, `ik_ar2_whtw_capacity`, `ik_ar2_whtw_loc_bytes` (the `ik_ar2_wht6` / `wht8` exports are unchanged); scale rounding for `B > 8`; `B = 32` sends the rotated f32 bit patterns |
| `crates/intarsia-backend-hip/src/{comm.rs,ffi.rs}` | `wire_env` parses `wht12`, `wht16`, `whtf`; `LossyWire` dispatches every width through `ik_ar2_whtw`; the start-up line and `debug_state` print the wire's name |
| `crates/intarsia-backend-hip/tests/ar_wht6.rs` | the whole suite also at 12, 16 and 32 bits (bound, floor, delayed-rank stress, negative control, rate) |
| `scripts/wire-probe/` | the wht6-8b scripts: tree `~/intarsiaml-wire-probe`, volume `intarsia-cargo-target-wire-probe`, five arms in the accuracy window, four in the speed window, `table.py` for the KL table |
| `docs/tuning.md`, `docs/backends/hip.md`, `docs/accuracy.md`, `docs/decisions/LOG.md` | knob row, design note, accuracy note, LOG row |

Packing (bits per element include the bf16 scale of 0.25 bit per element; a lane holds 8 elements of a group and emits 2 x B dwords
per 2048-element chunk, the layout and the coalesced 16-byte stores are as wht6's):

| wire | symbol | dwords per lane and chunk | chunk bytes | bits per element | elements per 20 MiB slot | note |
|---|---|---:|---:|---:|---:|---|
| wht6 | 6 bit, straddles dwords | 12 | 1536 | 6.25 | 26.2 M | #375 |
| wht8 | 8 bit, byte aligned | 16 | 2048 | 8.25 | 19.7 M | #383 |
| wht12 | 12 bit, straddles dwords (generic shift / or) | 24 | 3072 | 12.25 | 13.1 M | new |
| wht16 | 16 bit, half-word aligned | 32 | 4096 | 16.25 | 9.8 M | new |
| whtf | the rotated f32 bit pattern | 64 | 8192 | 32 (+ the unused scale slot) | 4.9 M | control; same bytes as an f32 partial |

Messages above a slot's capacity go out as several launches (the existing loop), e.g. a 4096 x 2048 f32 chunk (8.4 M elements) is
one launch for wht6 to wht16 and two for whtf.

One change that matters above 8 bits: the per-group scale is shipped as a bf16, and nearest-even rounding of `amax / qmax` can round
it **down** by up to 2^-8, which clamps the extreme symbol and costs up to `amax / 256` on that element. That is fine for 6 and 8
bits (the half step is `amax / 62` and `amax / 254`) but at 12 and 16 bits it would be the dominant error, 16x and 256x the half
step. For `B > 8` the scale is therefore rounded **up** (`w6_bf16_ceil`): no clamp, the step is at most 2^-7 larger. `B <= 8`
keeps the nearest-even scale, bit-identical to #375 / #383 (their kernel results reproduced, section 2).

## 2. Kernel gate (both R9700s) - PASS

`scripts/wire-probe/kernels-window.sh` (`raw/kern2/`): `cargo test --release -p intarsia-backend-hip --test ar_wht6`: **18 passed**,
1 ignored (rate); the existing suites with the wire unset (`--test comm --test ar_push --test ar_norm`): 13 + 4 + 4 passed. Per
width: both ranks bit-identical at every size and dtype pair, bitwise the exact sum below the floor, at 128 KiB and for sizes that
are not multiples of 64, every rotated error element inside the codec's bound, the f32 to bf16 output the f32 result rounded once,
the delayed-rank stress test clean (0 wrong of 30 rounds), the negative control (no wait) sees 94 to 124 wrong results.
`whtf` is held to the f32 round-off bound only. (The first window, not kept, failed three tests that were test artefacts: the stress
test's rms limit did not allow for the bf16 output rounding of two shapes at the tight widths, and the wht6 negative control
asserted instead of counting; both fixed in `ar_wht6.rs`, then the whole window rerun.)

| wire | worst rotated error / bound | rms error / rms of the sum (4096 x 2048 f32 size, 26 M elements) | allreduce us, 4096 x 2048 f32 (exact 1333) | 2048 x 5120 bf16 (exact 839) |
|---|---:|---:|---:|---:|
| wht6 | 0.996 | 1.66 % | 329 | 389 |
| wht8 | 0.993 | 0.41 % | 396 | 476 |
| wht12 | 0.985 | 0.025 % | 550 | 673 |
| wht16 | 0.836 | about 0.002 % (printed 0.001-0.002 %) | 705 | 905 |
| whtf (control) | 0.123 of the f32 round-off bound | below 0.0005 % (printed 0.000 %) | not timed | not timed |

All-reduce rate is wire bytes: at 12 bits an f32 partial still goes 2.4x faster than exact, at 16 bits 1.9x; a bf16 partial
(the 27B's) gains 1.25x at 12 bits and is **slower than exact at 16 bits** (905 against 839 us).

## 3. Accuracy: change-vs-base, 35B NVFP4 tp2, same binary

`scripts/wire-probe/accuracy-window.sh` (`raw/accA/`, `raw/accB/`), the same method as the wht8 lane: base = the exact-wire
self-golden dump of this binary (`--write-self-golden`), change = the wire under test against it with `--self-golden ...
--self-vs-ref-noise --ref-noise <golden's record>` (mean 21-way KL against the reference noise at factor 1, per-prompt check at 3 x
the prompt's own noise), `accuracy --tp 2 --devices 0,1 --mode both`, build stamped clean (`head=e3a5102 code-tree=717911054e84
dirty=no`, declared as its own pre-change commit as in the wht8 lane). Control: exact against its own dump gives 2.8e-13 and 0 flips.
`table.py` makes the tables.

**Prefill, short golden** (1441 rows, reference noise 3.161e-3; limit 1.0):

| wire | bits/elem | wire rms error | prefill mean KL21 | KL / limit | top-1 flips | per-prompt rule (3 x own noise) | decode KL / ref noise |
|---|---:|---:|---:|---:|---:|---|---:|
| whtf (control) | 32 | about 1e-7 | 3.468e-3 | **1.10** | 19 (1.32 %) | all inside | 0.86 |
| wht16 | 16.25 | 0.002 % | 4.513e-3 | 1.43 | 17 (1.18 %) | `code` over (needs 6.72) | 0.84 |
| wht12 | 12.25 | 0.025 % | 3.922e-3 | 1.24 | 15 (1.04 %) | all inside | 1.14 |
| wht8 | 8.25 | 0.41 % | 5.559e-3 | 1.76 | 18 (1.25 %) | `code`, `chat_system` over | 1.62 |
| wht6 | 6.25 | 1.66 % | 4.786e-3 | 1.51 | 16 (1.11 %) | `code`, `chat_multiturn` over | 0.94 |

**Prefill, long golden** (36887 rows, reference noise 2.218e-2; prompts up to 16k tokens):

| wire | prefill mean KL21 | KL / limit | top-1 flips | per-prompt | decode KL / ref noise |
|---|---:|---:|---:|---|---:|
| whtf (control) | 2.991e-2 | **1.35** | 1596 (4.33 %) | all inside | 0.30 |
| wht16 | 3.194e-2 | 1.44 | 1662 (4.51 %) | all inside | 0.43 |
| wht12 | 3.732e-2 | 1.68 | 1843 (5.00 %) | all inside | 0.40 |
| wht8 | 3.240e-2 | 1.46 | 1683 (4.56 %) | all inside | 0.29 |
| wht6 | 3.142e-2 | 1.42 | 1765 (4.78 %) | all inside | 0.58 |

Against the vLLM short golden (`--golden` with the reference-noise record, static floor 3.0e-2 binding; prefill / decode 21-way KL,
all PASS): exact 5.448e-3 / 7.977e-3, whtf 5.260e-3 / 1.006e-2, wht16 5.036e-3 / 9.954e-3, wht12 5.035e-3 / 7.221e-3, wht8
5.816e-3 / 8.779e-3, wht6 4.931e-3 / 1.050e-2. The control and every width sit within a factor 0.9 to 1.3 of the exact wire, and none
is ordered by width either.

Greedy ids, 256 tokens, first id that differs from the exact wire's (`diverge.py`; short prompt of 24 tokens / 4k prompt):

| wire | short | 4k |
|---|---:|---:|
| whtf | 15 | 68 |
| wht16 | 15 | **4** |
| wht12 | 15 | 82 |
| wht8 | 28 | 82 |
| wht6 | 15 | 69 |

The short prompt leaves the exact wire's ids at id 15 for four of the five wires, **including the control**: a near-tie token that
any bit-level change flips; and the 4k prompt diverges for the control at id 68 and for wht16 at id 4. Divergence point is not an
accuracy measure either.

## 4. Answer

**Does the gate see wire precision? No.** The wire error from wht6 to the control varies by a factor of about 1.7e5 (1.66 % to f32
round-off) while KL / limit moves 1.10 to 1.76 (short) and 1.35 to 1.68 (long) with no monotone relation. By wire error the order
(best first) is whtf, wht16, wht12, wht8, wht6; by KL on the short golden it is whtf, wht12, wht16, wht6, wht8; on the long golden
whtf, wht6, wht16, wht8, wht12. The control, exact up to f32 rounding, already **fails** both goldens (1.10 x, 1.35 x). So the
hypothesis holds: the rule's floor is set by changing the reduction at all (the rotation re-associates the f32 sum, a bf16
rounding of the hidden state then flips, then MoE near-ties), about 1.1 x the reference noise on the short golden and 1.35 x on
the long one, on this binary. Any width, up to 32 bits, fails at factor 1.

**Is there a width that passes with most of the speed?** None passes the rule as written, and none can. Measured against the
control as a floor instead, every width is within 1.13-1.60 x of it (short) and 1.05-1.25 x (long), but the spread is of the order
of the differences between widths, so these two goldens cannot rank widths; that needs a larger prompt set or the absolute
vLLM-golden gates (which all widths pass, and the exact wire passes by a similar margin). Among the widths, **wht12 is the sensible
trade**: 65 % of wht6's 4k prefill gain (+15.0 % against +22.9 %), 65x lower wire error, vLLM-golden gates passing, short-golden
KL / limit of 1.24 (the second best after the control; the long golden's 1.68 is the worst but all prompts are inside their own
noise). wht16 buys +10.1 % and is slower than exact for bf16 partials (the 27B); there is no reason to take it over wht12.

## 5. Speed (35B NVFP4 tp2, one interleaved window: cheap, not a full table)

`scripts/wire-probe/speed-window.sh` from the Mac (`localeval speed`, one prompt size of 4096, 128 decoded tokens, 3 reps, thinking
off), `serve --tp 2 --devices 0,1`, one binary, a fresh server per arm, 2 rounds with the arm order rotated (`raw/speed/`,
`speed-table.py`). Median (range) over 6 measured reps per arm, tok/s, prefill = prompt tokens / TTFT. Absolute numbers include
the chunked GDN of #378; compare within the table only:

| prompt tok | prefill exact | wht6 | wht12 | wht16 | wht6 / exact | wht12 / exact | wht16 / exact |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 4092 | 14522 (10872-14801) | 17851 (17575-17984) | 16698 (16577-16790) | 15982 (15939-16044) | 1.229 | **1.150** | 1.101 |

Decode (one row per step, exact wire by construction): 189.7 exact, 188.5 / 188.9 / 188.3, ratios 0.994 / 0.996 / 0.993. The exact
arm's lowest rep (10872) is an outlier; the median of the other arms is not affected. Two rounds of three reps: indicative, the
ratios of the 4k figure agree with the wht8 lane's (wht6 1.220, wht8 1.188 there).

## 6. Not done, not verified

- Prefill speed at 1k and 16k and the 27B: not measured (27B checkpoint absent, as before). Only one 4k window; the order wht6 >
  wht12 > wht16 follows the rate table.
- The floor-aware criterion (compare a change with the `whtf` control, or a repeat-perturbation spread) is proposed, not implemented
  in the accuracy tool.
- Batched decode above 16 rows per step with a wire on: not exercised.
- Fleet: kernel window (1.5 min), two accuracy windows (short about 3 min, long about 7 min) and the speed window (about 2.5 min),
  each with radiance down and answering after (15-20 s); an earlier kernels job (#9604) was stopped by hand after a quoting
  mistake in my own launcher never started its window (radiance had not been stopped). The tatooine tree `~/intarsiaml-wire-probe`
  and the volume `intarsia-cargo-target-wire-probe` are removed (`docker volume ls` shows none). `raw/` holds the `.out` / `.err` /
  step logs, not the `.safetensors` dumps.
