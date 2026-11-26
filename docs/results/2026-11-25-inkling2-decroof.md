# Inkling-2-Mini Metal decode toward the bandwidth floor (lane ink-decroof, 2026-11-25)

Status: step 0 done (prior work below); CONV_PRE ported to main (1760c5228), gates and A/B queued. Branch `ink/decroof` from
origin/main d8c3c7148 (includes #496: DSpark k=3 default on Metal, so `--spec off` is plain decode), worktree
`.worktrees/ink-decroof`. Not pushed.

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
