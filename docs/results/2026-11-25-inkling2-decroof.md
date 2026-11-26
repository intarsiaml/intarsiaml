# Inkling-2-Mini Metal decode toward the bandwidth floor (lane ink-decroof, 2026-11-25)

Status (2026-11-26 00:25): three bitwise levers on branch `ink/decroof` (tip eceba6809 + docs), all default on, every gate
green at the tip (#11992). Plain decode (`--spec off`) within one process, vs main's code paths: **+2.1 % at 143, +2.4 % at
1k, +2.0 % at 4k** (CONV_PRE + BIAS_PRE, #11981), plus **+0.8 % at 4k, +0.9 % at 8k, +0.6 % at 16k** from the fused-attention
threshold (#11983). **Tip binary vs main's binary across processes (#11995, 8 + 8 processes ABBA, every process in the fast
mode, no overlap between the arms at any length): 77.65 -> 79.03 tok/s at 143 (+1.8 %), 75.28 -> 76.45 at 1k (+1.6 %),
72.86 -> 74.64 at 4k (+2.4 %)**; fused attention per dispatch 30.4 -> 26.2 us at short context. Greedy ids equal to main at 143 / 1k / 4k (spec off and spec default) and 8k / 16k. Branch from
origin/main d8c3c7148, worktree `.worktrees/ink-decroof`, not pushed.

| lever | commit | knob (`=0` reverts) | gain (within-process, paired, 95 % CI above 1) |
|---|---|---|---|
| K / V conv loads before the q prologue (ink/solo-38 port) | 1760c5228 | `INTARSIA_INK_CONV_PRE` | +0.7 % at 143 / 1k / 4k on top of BIAS_PRE |
| rel bias one distance per thread, relproj loaded together | bb8457ccf, on in 7bab2fa92 | `INTARSIA_INK_BIAS_PRE` | +1.3-1.6 % at 143 / 1k / 4k |
| solo one-row fused attention up to 16896 keys (was 4096) | eceba6809 | `INTARSIA_INK_ATTN_FUSED=0` forces the pair everywhere | +0.6-0.9 % at 4k-16k |
| (bias loads at kernel entry) | bb8457ccf | `INTARSIA_INK_BIAS_EARLY=1` (stays off) | not significant |

## Brief (2026-11-25, coordinator, in substance)

Make plain Inkling-2-Mini decode on Metal (endor, M5 Max) faster toward the bandwidth floor, bitwise only. Step 0: search all
branches first and write what is done / dead / open. Then a fresh per-dispatch profile on main (`--spec off`, short and 4k),
pick the biggest excess no branch has tried or killed, build it bitwise (same greedy ids at 143 / 1k / 4k with `--spec off` and
with the spec default; Metal Inkling tests pass), A/B plain decode tok/s interleaved ABBA >= 6 runs per arm at 143 / 1k / 4k
(processes land in a fast or slow mode 10-14 % apart: interleave, report spreads, prefer within-process comparisons through a
run-time knob). Default on only if bitwise and faster than noise, else an env knob with a docs/tuning.md row. endor: exclusive
fleetq slots <= 30 min, never touch reservations #11871 / #11856 / #11857, separate CARGO_TARGET_DIR per arm, md5 the binaries.

## Step 0: what other branches already did (searched 2026-11-25 21:40)

Sources: `git for-each-ref refs/heads/ink`, the results docs on main (dfuse, batch, metal, hostgap) and on the unmerged
branches ink/metal (1dbdd6679, sections "Solo levers" .. "Brief 47"), ink/batch (sections 35, 36, 45), ink/solo-38,
ink/mixsplit, ink/dec-probe, ink/attn-prof, ink/attn-sweep, ink/dfuse-route, ink/batch-34; Fable's
`.worktrees/ink-fable/decode-structural-design.md` (read through ink/metal's brief 47 summary).

The 11-21 table in `2026-11-21-inkling2-metal.md` is stale: since then the controllers, router and experts changed as below.
The last ranked solo-decode profile (ink/metal "Solo levers", 13.1 ms / token at short context, main 22303d704) has, in ms
over the floor: mix / blend / norm 2.09 (248 dispatches), attention 1.52 (48), experts 0.95 (96), router 0.51 (48),
in_proj / wo / head 0.10 (100).

### Done (on main, default on)

| Change | Where | Measured |
|---|---|---|
| InkRoute (op 76): router GEMV + MoeRoute in one launch | dfuse, `INTARSIA_INK_ROUTE_FUSED` | -48 dispatches, -0.13 ms |
| Loop messages per pass (one AttnResMix with Q 48 at the restart) | dfuse, `INTARSIA_INK_MSG_PRE` | -22 dispatches, -0.14 ms |
| Row kernels issue all operand loads first (hoist) | dfuse, `INTARSIA_INK_MIX_HOIST` | -0.25 ms |
| Blend norm one channel per thread | dfuse, `INTARSIA_INK_BLEND_NORM_C` | -0.23 ms |
| Route rank on float4 reads | batch brief 45, PR #504, `INTARSIA_INK_ROUTE_RANK4` | +1.1 % short |
| One-row NVFP4 expert instances | dfuse job G, flipped by batch brief 35, `INTARSIA_INK_MOE_T1` | +2.2 % short, +3.0 % 5k |
| P12 in_proj / wo / unembed packing, screened greedy head, prologue split, fp8 paths | pack12, cuda/metal screen lanes | projections and unembed at the floor |
| Host gap (profiler default, device sampler opt-in) | hostgap | GPU idle 0.001-0.011 ms / token |

### Dead (built or probed, measured, closed)

| Idea | Where | Why closed |
|---|---|---|
| One-dispatch blend tail (h GEMV + gate + norm, ticketed) | dfuse change 3 | slower net (per-call 32.5 vs 25.4 us at S 3) |
| Blend tail 3 -> 2 (gate + norm) | ink/metal solo 38 | needs a cross-threadgroup wait; spin form ruled out (no forward progress on Metal), non-waiting form is the slow one-threadgroup norm |
| Split ConvMix row kernel (one row over W threadgroups, 3 dispatches) | ink/mixsplit | bitwise, 0.5-0.7 ms / token slower at every W |
| ink_route preload | ink/dfuse-route | bitwise, slower |
| Row-kernel cold operands | dfuse job P | ceiling 0.085 ms / token |
| Attention key-loop lookahead, early K / V loads | dfuse job Q | <= 0.02 ms, slower at 512 / 5k |
| Split keys / split cap sweeps | dfuse Q, ink/attn-sweep | non-bitwise; best sk32 +1.2 % short, cap > 32 slower |
| Concurrent encoder with barriers | dfuse job K, ink/metal | +0.4-0.6 us per dependent transition, net ~0.2 ms slower; only ~6 independent transitions per step |
| Untracked hazards | dfuse job K | no difference |
| Expert-prologue router fold | ink/batch brief 36 | ~0 (the fused InkRoute already costs the GEMV alone) |
| Attention touch (idle split prefetch) | ink/dec-probe | 0.07-0.10 ms, below go |
| Barrier diet in ink_mix_row | ink/dec-probe | 0.03-0.09 us per barrier: nothing to save |
| Fable option A (dissolve ConvMix) | ink/metal brief 47 | ~0.3 ms on paper, 9 days, contract changes on 4 ops |

### Open (built or sized, not on main)

| Item | State | Estimate |
|---|---|---|
| **K / V conv prefetch in `ink_attn_dec` (`INTARSIA_INK_CONV_PRE`)** | built on ink/solo-38 46e30ba3c, bitwise (md5 64 / 256 equal, inkling2_attn, rows_exact), closed only for the old 2 % bar: +1.3 % short, +0.8 % 5k (#11662) | eligible under this brief's bar; **this lane's first lever** |
| Fable option (b): transposed relproj load in the attention prologue | not built, unmeasured | 0.05-0.14 ms / token, bitwise |
| Router GEMV folded into the MoE-input norm / blend kernel | owned by the batch lane (coordinator FYI on ink/metal: "the batch lane (a716d7e) owns a decode fold of the router GEMV into the MoE-input norm / blend kernel (brief 45, ceiling +4.7 %) ... Do not change that decode norm / blend kernel without telling them") | not this lane's |
| Op 77 (ConvMix + router) | dfuse, skipped (~0.1 ms) | touches the same kernels as the fold above |
| One split per kv head at short context (no combine) | ink/metal brief 44 #3 | ~0.15 ms, **not bitwise**: out of scope |

Conclusion of step 0: the brief's "likely candidate", controller fusion, is spent: what was bitwise and positive shipped, and
the rest measured slower or below 0.1 ms. Every lane that probed the remainder found the decode gap to be warm serial chains
inside each kernel plus ~1 us per dependent dispatch, with no single bitwise lever over ~0.3 ms. Under this brief's bar
(bitwise and faster than noise) the sub-2 % items are back in play; CONV_PRE is the only one already built.

## Pre-registered rule for CONV_PRE (written 2026-11-25 22:00, before any timing)

- Bits: greedy ids (`--show-ids`, 256 tokens, `--ignore-eos`) of the 143 / 1021 / 4108-token prompts equal between the base
  binary (d8c3c7148) and the candidate (1760c5228), with `--spec off` and with the spec default, all in the same job. Any
  difference stops the lever.
- Tests on the candidate: backend-metal inkling2_attn, rows_exact, inkmix, inkctrl; models inkling2 (G1 tiny), inkling2_batch;
  docs_env_vars. G2 short decode-mode self-golden byte-equal to the base binary's.
- Speed: (1) within one process (diag binary, `CONV_PRE.set` per prompt, arms alternating ABBA by cycle, cycle 0 dropped):
  paired on / off ratio per length, median with a bootstrap 95 % CI; (2) across processes, base vs candidate binary ABBA, >= 6
  processes per arm, all three lengths per process. Default on if (1)'s CI lies above 1.0 at 143 and the median is >= 1.0 at
  1k and 4k, and (2) does not contradict it (candidate median not below base at any length beyond the process spread).
  Otherwise the knob stays and the default goes off.

## Lever 2: the decode attention's rel bias, one distance per thread (`INTARSIA_INK_BIAS_PRE`, `_BIAS_EARLY`; bb8457ccf)

Built before any timing, default off. Fable's option (b) ("relproj stored transposed at load: a bias element is one 32-byte
load, not 16 strided loads") without a weight copy: in `ink_attn_dec` the split's bias loop ran `x` over GP x nb elements
(192 at 3 heads x 64 distances, 2 rounds over 128 threads), each a runtime-`dr` loop of a load and a dependent fma, 16 strided
loads per element. The new form (InkP flag 64, d_rel 16 and at most 128 distances per split) gives each thread one distance:
its 16 relproj values are loaded in one unrolled round and reused by the GP heads' fma chains, each chain in the same order
(j ascending, `fma(rsh[h][j], w_j, a)`) and rounded as before, so the bias table is bit for bit the old one. Flag 128 issues
those loads at kernel entry, next to CONV_PRE's, since they depend only on the split geometry. Wider splits (more than 128
distances: the 8 full visits from ~4k on, every visit at 16k) keep the old loop. The prefill prologue already uses a
preloaded vector form (flag 8), so this only brings decode in line with it.

Rule (same as CONV_PRE): bitwise first (greedy ids base == diag arms C and D, `--spec off` and spec default; inkling2_attn
and rows_exact with the knobs on; inkling2_batch with both on: the row-table form; G2 short self-golden byte-equal), then the
within-process ratio C / B and D / C at 143 / 1k / 4k. Default on only if the CI of C / B (or D / B for EARLY) lies above 1.0
at 143 and the median is >= 1.0 at 1k and 4k.

## Jobs

| fleetq | what | state |
|---|---|---|
| #11930 | build base d8c3c7148 (md5 c887ead6), cand 84bd490dd (= bb8457ccf code; 277c7697), diag (cand + the arm patch; 17cfcf8c), each its own target dir; test binaries; docs_env_vars 2 / 2 (`--load cpu`, 22:07-22:09) | done |
| #11936 | gates (ids, Metal tests, G2) + fresh profile of main (exclusive, 22:10-22:17) | done, all pass |
| #11941 | A/B: within-process 4 arms x 3 processes + cross-process ABBA 6 x 2 (exclusive, 22:17-22:38) | done |
| #11981 | A2: within-process A / E (BIAS_PRE alone) / C (both), 3 processes; tip build, ids vs base, docs_env_vars, attn tests (23:11-23:25) | done |
| #11983 | A3: fused attention above 4096 keys, within-process C / F at 4k / 8k / 16k (23:25-23:32) | done |
| #11992 | T: tip gates after the threshold change (00:18-00:21) | done, all pass |
| #11995 | X: tip binary vs main binary, cross-process ABBA 8 x 2 + interleaved per-dispatch traces (01:14-01:23) | done |

Diag patch (`.scratch/diag`, never committed): `generate --prompt-ids` reads `INTARSIA_DIAG_ARMS` (one letter per prompt) and
sets the Switch overrides before each prompt: A all off, B CONV_PRE, C + BIAS_PRE, D + BIAS_EARLY; it prints per prompt how
many attention launches carried flags 32 / 64 / 128, so an arm that does not engage shows up as zero.

## Fresh decode profile of main d8c3c7148 (#11936; `--spec off`, greedy, 143- and 4108-token prompts)

Method: the per-dispatch timeline (`INTARSIA_METAL_COMMIT_EVERY=1 INTARSIA_TOKEN_FEEDBACK=0
INTARSIA_METAL_PROFILE=cb`, 33 tokens, `raw/disp2.py`: steps split at the embedding launch, prefill and odd steps dropped),
grouped by class (`raw/classes.py`) and scaled to the normal step's span (token feedback, 257 tokens, `cbstat.py fb`), since
one buffer per dispatch overstates each by ~1-2 us. Floors at 560 GB/s; bytes per class from dfuse's roofline L, the head
re-counted for the fp8 screen (461 M weights x 1 B), attention KV at 4 KiB per key per visit.

Normal step: **12.82 ms at short context (78.0 tok/s), 13.65 ms at 4k (73.3 tok/s)**; 545 / 553 dispatches, 2 command
buffers per step. Floor 7.92 / 8.41 ms, so decode runs at **62 % of the floor** at both lengths (the same share as 11-21: the
P12 / screen / dfuse gains moved floor and step together).

Short context (ctx ~150):

| class | per-dispatch us / token | scaled to the real step | dispatches | MB / token | floor (ms) | excess (ms) |
|---|---:|---:|---:|---:|---:|---:|
| controllers: ConvMix row + blend, LoopBlend, AttnResMix | 2411 | 2316 | 246 | 113 | 0.202 | **2.115** |
| attention core (`ink_attn_decode`, 48 x 30.3 us) | 1466 | 1409 | 48 | 29 | 0.053 | **1.356** |
| experts NVFP4 (gate+up 50.8 us, down 31.4 us) | 3621 | 3479 | 88 | 1577 | 2.816 | **0.663** |
| router (`ink_route`, 9.7 us) | 501 | 482 | 49 | 28 | 0.050 | **0.432** |
| attention projections (in_proj 57.8 us, wo 22.5 us; P12) | 3865 | 3713 | 96 | 1950 | 3.482 | 0.231 |
| LM head (fp8 screen GEMV 781 us + screen + argmax) | 924 | 888 | 3 | 469 | 0.838 | 0.050 |
| experts MXFP8 (4 visits) | 531 | 510 | 8 | 263 | 0.470 | 0.041 |
| norms, embedding, other | 21 | 20 | 7 | 8 | 0.014 | 0.006 |
| **sum** | 13339 | 12817 | 545 | 4437 | 7.924 | **4.893** |

4k (ctx ~4120): controllers 2.171 ms over, **attention 1.601** (40 fused sliding-window visits at 39.1 us; the 8 full visits
leave the fused kernel at this context and run `ink_kv_prologue` + `rel_sink_attn` at 66.7 us), experts NVFP4 0.676, router
0.431, projections 0.256, head 0.048; sum 5.234 ms over a 8.412 ms floor. Per kernel: `raw/disp_base_{143,4k}.txt`.

Against the 11-21 table: controllers 2.94 -> 2.12 ms over (dfuse), attention 1.44 -> 1.36, experts 1.22 -> 0.66 (T1),
projections and head at the floor. The ranking is unchanged: controllers, attention, experts, router.

## Gates of CONV_PRE and BIAS_PRE / BIAS_EARLY (#11936, all pass; `raw/ids.txt`, `tests.txt`, `g2.txt`)

- **Greedy ids** (256 tokens, `--ignore-eos`, 143 / 1021 / 4108-token prompts): base == cand == diag arms A / B / C / D for
  every prompt, with `--spec off` and with the spec default (DSpark k3: 219 verify passes, 80.2 % acceptance in every spec
  run). md5 of the id lists: 66d75868 / 442bd780 / f9adc9df.
- **Engagement** (diag counters): every arm-B prompt carried flag 32 on all its attention launches (12240 / 12240 / 10200
  at `--spec off`; 10200 at 4k because the 8 full visits leave the fused kernel), arm C flags 32 + 64, arm D 32 + 64 + 128,
  arm A none; the same under spec (4272 / 4560 / 1400 launches).
- **Metal tests on cand**: backend-metal inkling2_attn, rows_exact, inkmix, inkctrl, inkload 24 / 24; inkling2_attn +
  rows_exact with `INTARSIA_INK_BIAS_PRE=1` 12 / 12, with BIAS_PRE + BIAS_EARLY 12 / 12, with `INTARSIA_INK_CONV_PRE=0` 12 / 12;
  models inkling2 (G1 tiny) + inkling2_batch 25 / 25; inkling2_batch (row-table form) with BIAS_PRE + BIAS_EARLY 8 / 8;
  docs_env_vars 2 / 2 (#11930).
- **G2 short (decode mode)**: mean KL 2.928e-3 (limit 4e-3) for base, cand and cand + BIAS_PRE + BIAS_EARLY; the
  self-goldens are byte-equal to the base's (stcmp EQUAL, both).
- First timing signal (one run each, not the A/B): `--spec off` base 73.1 / 70.5 / 71.3 tok/s, cand 77.3 / 74.8 / 72.2; in
  the diag process arm C/D prompts ran 63-69 tok/s against 71-77 for A/B, a hint that BIAS_PRE is slower. The A/B decides.

## A/B (#11941, 22:17-22:38; `raw/jobA.log`)

Within one process (diag binary, arms set per prompt through the Switch overrides; 7 cycles ABCD / DCBA x 143 / 1k / 4k, cycle 0
dropped, 3 processes = 18 pairs per comparison and length; `--spec off`, 256 greedy tokens, `--ignore-eos`). Paired ratio of
decode tok/s, median, bootstrap 95 % CI, pairs above 1, per-process medians:

| comparison | 143 | 1k | 4k |
|---|---|---|---|
| B / A (CONV_PRE) | 1.0050 [0.9950-1.0079], 12 / 18, (0.998 1.005 1.007) | 1.0064 [1.0018-1.0108], 14 / 18 | 1.0065 [1.0041-1.0142], 14 / 18 |
| C / B (BIAS_PRE on top) | **1.0136 [1.0069-1.0173]**, 14 / 18, (1.009 1.014 1.016) | **1.0131 [1.0082-1.0218]**, 16 / 18 | **1.0133 [1.0093-1.0165]**, 17 / 18 |
| D / C (BIAS_EARLY on top) | 1.0047 [1.0022-1.0134], 15 / 18 | 0.9969 [0.9876-1.0023], 6 / 18 | 1.0017 [0.9968-1.0053], 10 / 18 |
| C / A (both, vs main's code paths) | **1.0182 [1.0081-1.0221]** | **1.0237 [1.0167-1.0257]** | **1.0201 [1.0140-1.0267]** |

Arm medians (tok/s): 143 A 77.01, B 76.85, C 76.96, D 78.53; 1k A 73.98, B 74.11, C 75.22, D 74.69; 4k A 71.95, B 72.35,
C 73.06, D 73.36. Within a process the arms spread 70-79 tok/s (the fast / slow mode moves every arm of a cycle together),
so the arm medians are noisier than the paired ratios; the per-process ratio medians agree within ~0.5 %.

Cross-process ABBA (base binary d8c3c7148 vs cand = CONV_PRE on, BIAS_PRE then off; 6 processes each, the three prompts per
process): the first five processes ran in the slow mode (63-69 tok/s), the last seven in the fast one (72-77), so the
spread (143: base 65.1-77.5, cand 68.5-77.1) is ~10x the effect. cand / base by median: 0.987 at 143, 1.030 at 1k, 1.051 at
4k; restricted to the seven fast-mode processes, 143 is base 76.7-77.5 vs cand 75.5-77.1. Nothing here resolves a 0.5-1 %
effect; it does not contradict the within-process result beyond the process spread.

Decisions by the pre-registered rules:
- **BIAS_PRE: default on** (7bab2fa92). C / B's CI lies above 1 at all three lengths. `INTARSIA_INK_BIAS_PRE=0` reverts.
- **BIAS_EARLY: stays off.** Its 1k median is below 1.0.
- **CONV_PRE: the rule is not met at 143** (CI 0.995-1.008 includes 1; +0.6 % with CIs above 1 at 1k and 4k). By the rule
  its default would go off, but that leaves BIAS_PRE without CONV_PRE, a combination no arm timed. #11981 times exactly
  that (A all off / E BIAS_PRE alone / C both, same within-process method): CONV_PRE stays on only if C / E's CI lies above
  1 at 143 and its median is >= 1.0 at 1k and 4k; otherwise its default goes off and E is the shipped state.

## Follow-up A2 (#11981, 23:11-23:25; `raw/jobA2.log`): CONV_PRE on top of BIAS_PRE, and the tip

Same method, arms A (all off) / E (BIAS_PRE alone) / C (both), 7 cycles AEC / CEA, 3 processes, 18 pairs per length. Every
process: 3 distinct id lists (one per prompt, across all arms), launch counts as intended (E 0 / 12240, C 12240 / 12240).

| comparison | 143 | 1k | 4k |
|---|---|---|---|
| E / A (BIAS_PRE alone) | 1.0141 [1.0132-1.0148], 16 / 18 | 1.0160 [1.0143-1.0167], 17 / 18 | 1.0134 [1.0126-1.0165], 17 / 18 |
| **C / E (CONV_PRE on top)** | **1.0070 [1.0052-1.0088]**, 16 / 18, (1.007 1.007 1.007) | **1.0070 [1.0047-1.0093]**, 17 / 18 | **1.0069 [1.0041-1.0080]**, 17 / 18 |
| **C / A (shipped vs main)** | **1.0212 [1.0185-1.0249]** | **1.0235 [1.0203-1.0263]** | **1.0204 [1.0172-1.0241]** |

Arm medians (tok/s): 143 A 77.16 / C 78.82; 1k A 74.73 / C 76.39; 4k A 72.19 / C 73.70.

**CONV_PRE stays default on**: on top of BIAS_PRE its CI lies above 1 at all three lengths (#11941's +0.5 % at 143 without
BIAS_PRE was the same size, just not resolved). The shipped state (both on) is **+2.1 % / +2.4 % / +2.0 %** over main at
143 / 1k / 4k, every CI above 1, all bitwise.

Tip check in the same job (tip 699d0bfc7, md5 2d61b4a6, built in the worktree's own target dir): greedy ids at the three
lengths equal to the base binary's from #11936, `--spec off` and spec default; docs_env_vars 2 / 2; inkling2_attn 1 / 1,
rows_exact 11 / 11 with the new defaults. Single-run tip speed (not an A/B): `--spec off` 79.2 / 76.9 / 74.1 tok/s, spec
default 123.7 / 80.8 / 99.8.

## Lever 3 (measure first): the fused decode attention above its 4096-key rule

The 4k profile shows the 8 full visits leaving `ink_attn_decode` at ~4.1k context: `ink_attn_fused_rule` takes the fused
kernel for one row up to 4096 keys only (lane ink-m-attn measured fused faster at 512 / 1024 / 4096 keys and the pair faster
at 32768; 8k / 16k were never measured). Above it each full visit runs `ink_kv_prologue` (21.6 us) + `rel_sink_attn` (66.7 us),
88 us per-dispatch, against ~62 us for the fused kernel at 4096 keys (dfuse L / K), so up to ~0.2 ms per token at 4-8k. The
fused kernel is bitwise the pair (tests/inkling2_attn.rs), so a new threshold changes no bits.

#11983 (`raw/jobA3.sh`, `raw/diag3.patch`): within one process, arm C (today's defaults) vs F (C with the fused kernel forced
on every visit through a diag override of `INTARSIA_INK_ATTN_FUSED`), prompts 4108 / 8216 / 16432 tokens (the 4k prompt
repeated), 5 cycles CF / FC, 2 processes; ids per length must be one list across arms. Rule: raise the 1-row threshold to the
largest measured length whose F / C CI lies above 1 (and every shorter one's median >= 1); none above 1 -> no change.

### #11983 results (23:25-23:32; `raw/jobA3.log`)

Both processes: 3 distinct id lists (one per length across C and F), F's launches all fused (12240 at 4k / 8k, 12288 at 16k)
against C's 10200 (the 8 full visits on the pair). 8 pairs per length:

| length | C (tok/s) | F (tok/s) | F / C, 95 % CI | pairs above 1 | per process |
|---|---:|---:|---|---|---|
| 4108 | 73.90 | 74.52 | **1.0083 [1.0061-1.0099]** | 8 / 8 | 1.0085, 1.0079 |
| 8216 | 72.03 | 72.66 | 1.0086 [1.0000-1.0111] | 6 / 8 | 1.0045, 1.0087 |
| 16432 | 69.06 | 69.54 | **1.0059 [1.0007-1.0085]** | 7 / 8 | 1.0069, 1.0048 |

By the rule the largest length whose CI lies above 1 is 16.4k (8k's lower bound touches 1.0000, its median is above 1), so
the one-row threshold goes to **16896 keys** for solo steps (eceba6809: the 16.4k prompt plus 256 tokens, rounded up; the
pair stays above it, where ink-m-attn measured it faster at 32k). Batched steps' one-row calls keep 4096: no batched run
was timed above it. Below ~4.1k context nothing changes (every visit was already fused). Tip gates: #11992 (`raw/jobT.sh`).

## Tip gates (#11992, tip 1d4a1424a = eceba6809 + docs, md5 e6c4494a; `raw/jobT.log`)

- Greedy ids, tip vs main's base binary: 143 / 1021 / 4108 tokens EQUAL with `--spec off` and with the spec default (the
  same md5s as #11936); 8216 and 16432 tokens (`--spec off`, both binaries in this job) EQUAL (463e6cff / 95fa80ba).
- Tests: backend-metal inkling2_attn, rows_exact, inkmix, inkctrl, inkload 24 / 24; models inkling2 (G1 tiny) +
  inkling2_batch 25 / 25; intarsia-core lib inkling2 4 / 4; docs_env_vars 2 / 2.
- Single runs (one process each, not an A/B): tip `--spec off` 79.2 / 77.0 / 75.1 tok/s at 143 / 1k / 4k; 8k base 70.9 vs
  tip 72.6, 16k base 67.7 vs tip 69.4; spec default 126.7 / 80.6 / 100.6.

## Tip vs main's binary (#11995, 01:14-01:23; `raw/jobX.log`, `raw/ab_proc2.py`)

The in-process ratios above compare the new binary's arms, so its "off" arm still runs the new kernel source. This job compares
the shipped tip binary (e6c4494a) against main's (c887ead6) directly: 16 processes in the order base tip tip base ..., each
running the three prompts twice (`--spec off`, 256 greedy tokens), per-process mean per length:

| length | base, median [range], n = 8 | tip, median [range], n = 8 | tip / base |
|---|---|---|---:|
| 143 | 77.65 [77.41-78.00] | 79.03 [78.94-79.19] | **1.0178** |
| 1021 | 75.28 [74.88-75.63] | 76.45 [75.89-76.59] | **1.0156** |
| 4108 | 72.86 [72.66-73.06] | 74.64 [74.06-74.75] | **1.0245** |

Every process ran in the fast mode (all within 1 % of their arm's best), and the arms do not overlap at any length: the
slowest tip process is faster than the fastest base process. So the shipped state is **+1.6-2.5 % over main**, matching the
in-process +2.0-2.4 % (+ the threshold's +0.8 % at 4k).

Per-dispatch traces, interleaved, 3 per arm (`ink_attn_decode` us per dispatch): 143 base 30.50 / 30.38 / 30.46, tip
26.12 / 26.33 / 26.04 (-4.3 us x 48 visits = -0.21 ms per token); 4k base 39.25 / 38.75 / 39.00 on 40 visits + 8 x
(66.6 + 22.0) us on the pair, tip 34.21 / 34.25 / 34.00 on all 48 visits (attention 2.33 -> 1.96 ms per token per-dispatch).

Serve: `serve --max-batch 1` (the default) runs the single-sequence engine (`Seqs::One`, `model.rs` ~4443), the same solo path
as `generate`, so all three levers reach it. With `--max-batch` > 1 decode runs the batched step: CONV_PRE and BIAS_PRE apply
(row-table form, inkling2_batch passes with them on), the fused threshold stays at 4096 there. No localeval run was made.

## What is left (ranked by the fresh profile, short context)

1. Controllers, 2.1 ms over the floor: no bitwise lever left after dfuse / mixsplit / dec-probe (see step 0); the router
   GEMV fold into the norm / blend kernel belongs to the batch lane.
2. Attention core, ~1.3 ms over: this lane took ~0.3 ms of it; the rest is the body's serial key steps and the combine
   dispatch, whose levers (one split per head, fewer simdgroup merges) change the merge order (not bitwise).
3. Experts NVFP4, 0.66 ms over; router 0.43 ms (mostly the routing tail after rank4).
Not built: BIAS_EARLY's register cost cancels its overlap; a batched-step fused threshold above 4096 is unmeasured.
