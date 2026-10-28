# det: run-to-run nondeterminism of greedy decode, Qwen3.8-27B MXFP4 tp=2 on the R9700s (HIP), 2026-10-28

Lane `lane-det/tp2-nondeterminism`, base `origin/main` `bb48df640`. tatooine, 2x R9700 (gfx1201), `amd/Qwen3.8-27B-Quark-AWQ-MXFP4`, `generate --tp 2 --devices 0,1`, W4A8 (recipe default),
greedy, `--ignore-eos`. Raw logs: `2026-10-28-det-tp2-nondeterminism/raw/`. Scripts: `scripts/det/`. Comparison: `tools/det/detcmp.py`.
Status: HANDED OFF 2026-10-28 ~12:00 (see Handoff); lane ar measured the speed cost 2026-10-28 14:00-14:45 (section 5): coordinator decision 2026-10-28 (lane ars): correctness wins over ~2% decode, so **`INTARSIA_HIP_AR_DRAIN=1` is the default for HIP tp>1; `=3` restores the fast, racy mode.** PR intarsiaml/intarsiaml#221.

Trigger: lane vat saw one plain `--spec off` run of ids prompt short-5 part from four other off runs (`2026-10-28-vat-verify-attn-rows/raw/w2/ids/gen-tp2-short-5-off.ids`,
first different token 27), and ho-chat `chat-sleep` differ from the off text in 1 of 5 fresh-server rounds of each dflash arm; lane vrc saw one old-path dflash text (ho-recall round 1) part from off.

## Headline

In about 1.5% of fresh tp=2 processes (`--spec mtp`; rarer with `--spec off`), one rank ends a layer of the prompt prefill holding a few rows that
the other rank does not: a one-sided wrong read at or right after a TP all-reduce. The shift is small at the logits (it flips only the prompt's
near ties, margins 0.01 to 0.3) and never repeats inside one process. tp=1 is clean (0/200). Not the cause: never-written memory, the peer rank's
graph capture, an order-dependent float reduction. The P2P all-reduce's flag can reach the peer before the pushed data: with
the default `INTARSIA_HIP_AR_DRAIN=3` (wait for the store counter) about 1.5% of tapped fresh processes show it (3/200); waiting ~25 us after the
flag (mode 5, 0/300), a host-staged all-reduce (`--comm host`, 0/200) and a system fence in every thread before the flag (mode 1, 0/250) each
remove it, modes 1 and 5 with output bitwise equal to the normal run. **Fix: `INTARSIA_HIP_AR_DRAIN=1`, now the default (`=3` restores the old, faster mode with the race).** Lane ar
measured its cost (section 5): decode -1.9 to -2.7% on the 27B (MTP off), -1.5% with MTP k2, -3.4 to -4.6% on the 35B; mode 5 costs -12% / -22%. Both
are over the 1% bar the lane set for a speed-only change, but this is a correctness fix: the coordinator decided (2026-10-28) that correctness wins over about 2% decode, so the default is now 1. `INTARSIA_HIP_AR_DRAIN=3` restores the fast mode (and the ~1.5% race); the knob is in `docs/tuning.md`.

## Handoff (2026-10-28, lane det)

**Found.** See Headline; evidence in section 3, w5 and w7 (`raw/w7/tap-{33,88,118}.tap`, `tools/det/tapdiff.py`).

**Ruled out** (by rate, ~100-450 fresh processes each): poisoned allocations (`INTARSIA_HIP_POISON=ff`), `INTARSIA_GRAPH_SYNC_CAPTURE=1` +
`INTARSIA_TP_PEER_SYNC=1`, eager decode, tp=1 on either card; float atomics or split-K on the prefill path (audit, section 3).

**Open.**
- Decision (coordinator, 2026-10-28, lane ars): the A/B of mode 1 against 3 costs 2% decode; correctness wins, so the default is mode 1 and `=3` is the fast, racy opt-in. The all-gather (`ag2_kernel`, vocab-parallel
  logits) has the same handshake and takes the same mode.
- Mode 6's both-ranks events (w9): unexplained; a second mechanism is not excluded. Rerun a tapped mode-1 arm of ~500 to bound it.
- Why `--spec off` is rarer (0 in ~450 here, 1 in vat's 5) and why all three tapped events are at layer 2.
- The 08:02:40 hard reset of tatooine during w4 (section 3): one event, cause unknown.
- Knobs from the brief never bisected (the event was found without them): `INTARSIA_TP_PARTIAL`, `INTARSIA_TP_ARNORM`,
  `INTARSIA_HIP_MXFP4_A8_FUSE`, `INTARSIA_HIP_GDN_VERIFY_FUSE`, `INTARSIA_HIP_VERIFY_ATTN_ROWS`, `GPU_MAX_HW_QUEUES`.
- vat's server-mode dflash misses (chat-sleep, recall-tls) were not reproduced in 51 localeval-style runs (w3); they are probably the same event
  in the first request of a fresh server, not shown.

**Next step.** (A/B done, section 5.) Flipping the default to mode 1 in `crates/intarsia-backend-hip/src/comm.rs:163` (and its `docs/tuning.md` row)
costs 2% decode; a cheaper fix is to design a handshake that does not trust the store counter (for example a sequence number in every pushed 16-byte word, read until it matches, as NCCL's
LL protocol does) and test it with a 300-run tapped arm (`scripts/det/stages-w9.sh` as the model).

**How to rerun.** `scripts/det/window.sh` under fleetq (header of the script), stages in `scripts/det/stages-w*.sh`, one fresh process per run via
`loop` in `scripts/det/lib.sh`; `TAP=1` writes per-process taps; `python3 tools/det/tapdiff.py <raw dir> <arm>` lists the taps that differ.
A tapped arm of 200 fresh `--spec mtp --spec-k 2` processes, 16 tokens, takes ~10 min with radiance down.

## 1. Tools added

| what | where | does |
|---|---|---|
| `generate --repeat N` | `crates/intarsia-cli/src/cmd/generate.rs` | the request N times in one process, the sequence reset before each run (no prefix reuse), `ids[r]:` per run |
| `generate --trace-margins` | same | host-logits path; per token `id:margin:hash` (top-1 minus top-2, FNV-1a of the whole f32 logits row). It turns speculation off. |
| `INTARSIA_DET_TAP=<file>` | `crates/intarsia-core/src/model/det_tap.rs` | per single-sequence pass (forward and verify), per rank, an FNV-1a hash of every layer's output (last row of residual + pending block output). Both ranks of a tp group must be equal after every all-reduce; across runs the first different (pass, layer) is where runs part. Off by default (one `OnceLock` read per layer when off). |
| `tools/det/detcmp.py` | | distinct id lists per arm, first parting token, logits-row hash differences, rank 0 vs rank 1 per pass, first different pass and layer run vs run |

## 2. What is deterministic (w1 to w3)

**Plain decode, tp=2.** short-5, 256 tokens. Every run bitwise equal: ids, every logits row (the `--trace-margins` hash), every layer output of every pass (the tap).

| window | arm | processes | runs | distinct id lists | logits rows | tap rank 0 vs 1 |
|---|---|---|---|---|---|---|
| w1 | default, on-device argmax | 1 | 10 | 1 | | |
| w1 | tap + margins | 3 | 18 | 1 | 1 set | 0 of 4626 passes differ |
| w1 | `INTARSIA_HIP_AR_DRAIN=1` | 1 | 6 | 1 (same) | same bits | 0 differ |
| w1 | `INTARSIA_HIP_POISON=ff` / `=00` | 2 | 8 | 1 (same) | same bits | 0 differ |
| w1 | `INTARSIA_TP_PEER_SYNC=1` | 1 | 10 | 1 (same) | | |
| w1 | eager decode (`INTARSIA_SOLO_GRAPHS=0`) | 1 | 4 | 1 (same) | same bits | 0 differ |
| w2 | fresh process each, after an mtp process (vat's order) | 30 | 30 | 1 (same) | | |
| w2 | fresh process each, tap | 40 | 40 | 1 (same) | | 0 of 10280 |
| w2 | one process, tap | 1 | 60 | 1 (same) | | 0 of 15420 |

230 plain runs (72 fresh processes), one id list, equal to the majority of vat's runs (`74621ab9`). The knob arms change nothing, not even a bit: the drain mode, poison (no never-written allocation is
read), the peer sync and eager launch give the graph path's bits. They could not discriminate a cause, because no run diverged in them.

**Speculative decode equals plain decode.** w3, 512 forced tokens (localeval's setting), the two prompts the server-mode runs missed on:

| prompt | off | dflash k7 | mtp k2 | dflash k7 + tap |
|---|---|---|---|---|
| ho-chat chat-sleep | 6 runs (2 proc) | 18 runs (6 proc) | 6 runs (2 proc) | 6 runs (2 proc), 0 of 876 passes rank-unequal |
| ho-recall recall-tls | 3 runs (1 proc) | 12 runs (4 proc) | | |

One id list per prompt across all arms: dflash and mtp are bitwise plain decode here.

**What this excludes.** An order- or timing-dependent reduction would show as many slightly different runs that part at the smallest margins first. short-5's smallest
off margins are at token 237 (0.0073) and token 17 (0.0123); no run ever parted there. The only float reductions across blocks or ranks are fixed-order: the explorer audit found no
float atomics in `kernels/hip` (integer counters and flags only: all-reduce, GDN conv ticket, MoE cursors), the all-reduce adds two operands (`a + b` commutes, so both ranks get the same bits), the split-KV
split count is a function of the KV length only (`kernels.rs` `splitkv_parts`), and the tap found rank 0 and rank 1 equal in every pass.

## 3. The event: a rare one-step corruption early in a fresh process

| where | run | first different token | off margin there | 
|---|---|---|---|
| vat w2 | `--spec off`, fresh process | 27 | 0.0777 |
| det w2 `soakA-mtp10` | `--spec mtp --spec-k 2`, fresh process (1 of 30) | 20 | 0.1240 |
| vat w3 | serve dflash k7 round 3, chat-sleep (the first request of a fresh server) | char 74 | |
| vat w3 | serve dflash k7 old path round 5, chat-sleep (first request) | char 107 | |
| det w4 `mtptap20` | `--spec mtp --spec-k 2`, fresh process, tap on (1 of 43; off 0 of 43) | 38 | 0.3093 |
| det w5 `tp2` #23, `tp2sync` #4 | mtp, fresh process | 7 (3775 -> 563, both) | 0.1898 |
| det w5 `tp2` #51 | mtp, fresh process | 17 | 0.0123 |
| det w5 `tp2poison` #39 | mtp, fresh process | 27 | 0.0777 |

short-5's lowest off margins in its first 40 tokens are tokens 17 (0.012), 27 (0.078), 20 (0.124), 7 (0.190), 38 (0.309), 18 (0.357), 8 (0.378) (`raw/w1/base-tap1.err`). Every event parts at one of
the first five. So an event is a small shift of the logits, of order 0.01 to 0.3, that flips whichever near tie it reaches first; sometimes 7 holds and 17 flips. It is not a large
discrete wrong value (an earlier draft of this doc said so, from two events). Repeats inside one process (130 runs, all `--spec off`) never parted; at the mtp rate that is weak evidence.

**w4: the wrong value is already in the prefill.** `mtptap20`'s tap (`raw/w4/mtptap20.tap`, reference `mtptap18.tap`; the other 42 mtp and 43 off taps are equal to each other) differs
from pass 0 on, the 38-row prefill (`pos=0 t=38`): layer 0's last row is equal, layer 1 (GatedDeltaNet) differs, and every later pass differs. Rank 0 equals rank 1 on that last row, which
does NOT clear the all-reduce: if one rank read a stale peer value in an earlier row, its residual row differs from the other rank's, its GDN heads carry that into row 37, and the
next all-reduce sums the shift into both ranks' row 37 equally. Only a per-row tap separates the cases. The w4 tap hashed only the last row of each layer
(the build predates `398c2c943`), and GDN's last row depends on every earlier row, so the bad value can be in any row of layer 0's output or inside layer 1. At t=38 the GDN layers run
`gdn_conv_bf16` + `gdn_delta_rule_bf16_v2` (the fused conv+delta launch is for t <= 32, v3 for t >= 64). The token only flips 38 tokens later: a small prefill error that the
greedy argmax absorbs until a near tie.

**tatooine hard-reset at 08:02:40**, the second the next w4 process (`mtptap44`) started its container (the previous boot's journal stops mid-container-start with no shutdown, the next
boot is at 08:03:03; `raw/w4/steps.log` ends at `offtap43`). Every det process before it today (about 300) started cleanly. One event, cause unknown (power transient at load, PCIe, driver);
noted because a marginal host is also a way to get rare, non-reproducible wrong values.

**Audit of the t=38 prefill path (read-only, no kernel suspect).** Every kernel there zero-fills or clamps its tile tails and guards its stores; no float atomics, no split-K; the
activation scale is per row (`ik_mx_a8.h:63-90`), so padded rows never enter it; kernel choice is env and shape only. `mxfp4_a8_gemm` at M=38 uses 128x32 tiles and the conv
32-token tiles, so a tail bug would first show at rows 32..37, which w6 can see. A bug that repeats would also repeat inside one process, so a fresh-process-only event points at a
race at start-up (load or MTP set-up work on another stream, not yet audited) or the host. Off has 0 events in ~380 fresh processes against mtp's 2 in 73: the mtp head's load and
set-up is the obvious difference. Both arms run the same 38 + 1 prefill (no prefix store, recurrent snapshot) with equal hashes. A second audit (MTP load, state and first
step) found nothing of the head queued before or during the 38-row pass on either rank: its first kernels follow the pass on stream 0, and its prompt catch-up is K/V only with no
collective, so the all-reduce counters cannot drift. What mtp does change before the pass is the number, size and order of the (never zeroed, `device.rs:500-521`) allocations:
the head's KV, per-row recurrent copies, verify buffers, possibly a larger shared scratch. A read-before-write somewhere on the path would see different leftovers per process and
the same in a repeat, which fits; w5's poison arm tests it.

**w6 (per-row tap, 09:40-09:47): no event.** 70 fresh `--spec mtp` and 70 fresh `--spec off` processes, 8 tokens each: every id list and every tap file byte-identical within
its arm (`raw/w6/`, one reference tap kept per arm). Fresh mtp processes so far: 2 events in 143 (w2 1/30, w4 1/43, w6 0/70); fresh off: 0 in ~450 here, 1 in vat's 5.

**w5 (09:48-10:08, rate, all `--spec mtp --spec-k 2`, 48 tokens, fresh processes, `raw/w5/<arm>.ids`):**

| arm | processes | events |
|---|---|---|
| tp=2 | 100 | 2 (tokens 7, 17) |
| tp=2, `INTARSIA_HIP_POISON=ff` (every allocation filled with 0xff) | 100 | 1 (token 27) |
| tp=2, `INTARSIA_GRAPH_SYNC_CAPTURE=1 INTARSIA_TP_PEER_SYNC=1` | 100 | 1 (token 7) |
| tp=1 on card 0 / card 1 (at the same time) | 100 / 100 | 0 / 0 (both cards the same ids) |

Poison does not change it: a read of never-written memory would give a large or NaN value with 0xff fill, and turn the event always or never; it stays ~1%. The capture and peer syncs
do not change it either, so it is not the peer rank's first capture racing its prefill. tp=1 is clean on both cards (0/200; tp=1's ids differ from tp=2's, as expected from a
different reduction order). Fresh mtp processes in total: 6 events in 443 (1.4%); fresh off: 0 in ~450. With rates this low the arms show only what does not matter much
(poison, syncs); tp=2 against tp=1 (6/443 against 0/200, Fisher p ~ 0.15 on its own: suggestive, not proof) points at the two-GPU part, which brings the P2P all-reduce back.

**w7 (10:10-10:37, all `--spec mtp`, fresh processes): the per-row tap catches it, one rank, a few rows, in an all-reduce.**

| arm | processes | id events (16 / 48 tokens) | taps different |
|---|---|---|---|
| per-row tap, 16 tokens | 200 | 0 | 3 (`tap-33`, `tap-88`, `tap-118`) |
| `INTARSIA_HIP_AR_DRAIN=5` (~25 us wait after the flag before the peer's data is read), 48 tokens | 200 | 0 | (no tap) |
| `--comm host` (host-staged all-reduce, no P2P kernel), 48 tokens | 150 | 0 | (no tap) |

`tools/det/tapdiff.py raw/w7 tap` on the three (the other 197 taps byte-identical):

| tap | pass | rank 0, layer 2 | rank 1, layer 2 | layer 3 |
|---|---|---|---|---|
| `tap-33` | 38-row prefill | equal | row 21 | rows 21..37 both ranks |
| `tap-88` | 38-row prefill | rows 31, 33, 34, 35 | equal | rows 31..37 both ranks |
| `tap-118` | 38-row prefill | equal | rows 24, 35, 36 | rows 24..37 both ranks |

The residual stream is replicated: after a layer's all-reduces both ranks must hold the same bits. At the end of layer 2 one rank, either one, holds a
few rows that the other rank (and 197 other processes) does not. Layer 3 (full attention) then spreads it to every later row on both ranks, and the
next all-reduce makes the ranks equal again, which is why w4's last-row tap showed rank 0 == rank 1. Two explanations fit w7 alone: (a) that rank's
all-reduce summed something other than the peer's partial for those rows, or (b) a race in an op each rank runs on its own after the all-reduce
(residual add, the next norm). w8's tapped `--comm host` arm separates them (no P2P kernel: (a) gives 0, (b) does not).

Hypothesis (a), the P2P handshake: `ar2_kernel` (`kernels/hip/allreduce.hip`) gives each block a contiguous range of 16-byte words, pushes it into
the peer's scratch with posted PCIe writes, waits `s_wait_storecnt 0` (mode 3, the default), then raises the peer's flag with a system-scope release
store; the peer reads its scratch once it sees the flag. The store counter only says the writes left this GPU; if a write lands at the peer after the
flag, the peer adds the slot's old contents (a partial from two all-reduces earlier: the scratch is double buffered). Predictions: per-block
chunks (scattered rows such as 31, 33, 34, 35 but not 32 fit); the affected rows are badly wrong (a different tensor), and only a few prompt rows,
so the logits move only slightly; and `INTARSIA_HIP_AR_DRAIN=5` (wait ~25 us after the flag, far longer than a PCIe write) removes it. w8 tests
the last one with the tap. It does not explain why `--spec off` is rarer, or why all three w7 events are at layer 2 (w4's last-row tap first
differed at layer 1; both tap builds index layers from 0, but the last-row view cannot see where a non-last row first went wrong).

**w8 (10:39-11:04, tapped, `--spec mtp`, 16 tokens, fresh processes): the P2P handshake hypothesis holds.**

| arm | processes | taps different | if the arm changed nothing (baseline 3/200 tapped, w7) |
|---|---|---|---|
| `INTARSIA_HIP_AR_DRAIN=5` (~25 us wait after the peer's flag before reading its data) | 300 | 0; all 300 byte-equal to w7's baseline majority `tap-1` | P(0) ~ 1%; Fisher 0/300 vs 3/200 p ~ 0.06 |
| `--comm host` (host-staged all-reduce, no P2P kernel) | 200 | 0 (all equal; own bits, same ids as P2P) | P(0) ~ 5% |

With w7's untapped arms (drain5 0/200, host 0/150 at the id level, against mode 3's 5/400 in w5 and w7) this answers w7's question for (a): delaying
the read until long after the flag removes the event without changing a single bit of the normal output, so the late part is the data, not a
computation after the all-reduce (that would not care about a wait before the read), and a host all-reduce has no event either.

**w9 (11:06-11:31, tapped, `--spec mtp`, 16 tokens, fresh processes): mode 1 fixes it; the new mode 6 does not.**

| arm | processes | taps different | first difference |
|---|---|---|---|
| `INTARSIA_HIP_AR_DRAIN=1` (existing: `__threadfence_system()` in every thread before the flag) | 250 | 0; all 250 byte-equal to w7's baseline majority | P(0) ~ 2.4% if it changed nothing |
| `INTARSIA_HIP_AR_DRAIN=6` (new: mode 3, then each thread reads back its last pushed word) | 250 | 4 | `drain6-8`: rank 1 only, layer 2, row 31 (the w7 pattern); `drain6-13`: both ranks the same, layer 0, rows 0, 2, 4, 19; `drain6-131`: both ranks, layer 0, row 33; `drain6-127`: both ranks, layer 2, rows 14, 16 |

Mode 6 is no fix (4/250 against mode 3's 3/200), and three of its events are a pattern not seen with mode 3: both ranks hold the same wrong rows,
from layer 0. Same wrong bits on both ranks means the sum itself was wrong, i.e. one rank's partial (or the data both read) was wrong before the
reduction; a one-sided late read cannot do that. Not explained: the extra PCIe reads may change the timing so that the late data hits the
other direction or the all-gather, or there is a second mechanism that mode 3 hides. Mode 6 stays an opt-in diagnostic; do not use it.

## 4. Raw

- `raw/w4/` .. `raw/w9/`: per arm `<arm>.ids` (one line per fresh process), `steps.log`, `window.log`; taps only for the events and one reference
  per arm (`raw/w4/mtptap20.tap` vs `mtptap18.tap`, `raw/w7/tap-{33,88,118}.tap` vs `tap-1.tap`, `raw/w9/drain6-{8,13,127,131}.tap`); the other
  taps were byte-identical to their arm's reference and are deleted. w1/w2 taps were dropped from the branch (all equal).
- `raw/w1/` .. `raw/w3/`: per arm `<tag>.err` (ids, margins, stats), `<tag>.tap` (layer hashes), `steps.log`, `window.log`; `raw/downtime.log` (radiance windows).

## 5. Speed A/B of `INTARSIA_HIP_AR_DRAIN` (lane ar, 2026-10-28 13:57-14:43)

Question: is mode 1 (system fence in every thread) or mode 5 (~25 us wait) within 1% of the default (mode 3) on every metric, so that it can become
the default? Setup: tatooine, both R9700s, `serve --tp 2 --devices 0,1 --max-batch 1`, one binary (`origin/main` `ff31c2081` + this branch, built
once), a fresh server per arm and round, the only difference `INTARSIA_HIP_AR_DRAIN=<d>`. Rounds interleaved with a rotating arm order
(`d3 d1 d5; d5 d1 d3; d1 d5 d3; d3 d5 d1`), fleetq #6461, driver `scripts/ar/ab-window.sh`, tables `scripts/ar/ab_table.py`,
`scripts/ar/ab_cats.py`, raw `raw/ar-a1/` (speed) and `raw/ar-a2/` (categories), localeval results logged under the labels `ar-a1-*` / `ar-a2-*`.
Each cell: mean of the per-round medians (3 reps and a warmup per call), (min-max over rounds); delta = mean of the paired same-round ratios
vs d3, worst = the worst single round. d3's own round-to-round spread is 0.1 to 0.5%, so the deltas below are not noise.

**27B MXFP4, `--spec off`, `localeval speed` (gen 128), 4 rounds.**


ar-a1-speed27 decode tok/s: mean of per-round medians (min-max over rounds, n rounds); paired delta vs d3 (mean, worst round)
| size | d3 | d1 | d5 |
|---|---|---|---|
| 1024 | 56.74 (56.67-56.79, n=4) | 55.19 (54.62-55.70, n=4) **-2.72%** (worst -3.77%) | 49.76 (49.62-49.93, n=4) **-12.30%** (worst -12.58%) |
| 4096 | 56.03 (55.98-56.07, n=4) | 54.87 (54.86-54.89, n=4) **-2.07%** (worst -2.13%) | 49.22 (49.19-49.24, n=4) **-12.17%** (worst -12.24%) |
| 16384 | 52.98 (52.93-53.03, n=4) | 51.96 (51.93-51.98, n=4) **-1.93%** (worst -2.08%) | 46.82 (46.81-46.83, n=4) **-11.63%** (worst -11.71%) |

ar-a1-speed27 prefill tok/s: mean of per-round medians (min-max over rounds, n rounds); paired delta vs d3 (mean, worst round)
| size | d3 | d1 | d5 |
|---|---|---|---|
| 1024 | 3579.79 (3567.60-3596.43, n=4) | 3543.48 (3522.50-3559.61, n=4) **-1.01%** (worst -1.43%) | 3456.39 (3448.39-3471.75, n=4) **-3.45%** (worst -4.06%) |
| 4096 | 3888.26 (3881.73-3892.85, n=4) | 3866.12 (3861.06-3874.01, n=4) **-0.57%** (worst -0.82%) | 3792.15 (3786.80-3796.84, n=4) **-2.47%** (worst -2.55%) |
| 16384 | 3557.48 (3552.74-3566.50, n=4) | 3539.47 (3533.73-3547.08, n=4) **-0.51%** (worst -0.92%) | 3483.02 (3473.05-3499.19, n=4) **-2.09%** (worst -2.24%) |

**Qwen3.6-35B-A3B NVFP4, tp=2 (`INTARSIA_TP_FUSION` default on at HIP tp>1), `localeval speed` (gen 256), 2 rounds.**


ar-a1-speed35 decode tok/s: mean of per-round medians (min-max over rounds, n rounds); paired delta vs d3 (mean, worst round)
| size | d3 | d1 | d5 |
|---|---|---|---|
| 1024 | 191.58 (191.21-191.96, n=2) | 182.76 (180.42-185.11, n=2) **-4.60%** (worst -5.64%) | 148.90 (148.69-149.11, n=2) **-22.28%** (worst -22.32%) |
| 4096 | 188.91 (188.91-188.92, n=2) | 182.57 (182.30-182.84, n=2) **-3.36%** (worst -3.50%) | 148.24 (146.35-150.12, n=2) **-21.53%** (worst -22.53%) |

ar-a1-speed35 prefill tok/s: mean of per-round medians (min-max over rounds, n rounds); paired delta vs d3 (mean, worst round)
| size | d3 | d1 | d5 |
|---|---|---|---|
| 1024 | 8833.38 (8731.25-8935.51, n=2) | 8809.47 (8798.23-8820.72, n=2) **-0.26%** (worst -1.54%) | 8588.01 (8586.04-8589.98, n=2) **-2.77%** (worst -3.87%) |
| 4096 | 11996.07 (11930.18-12061.96, n=2) | 12091.00 (12060.09-12121.91, n=2) **+0.79%** (worst +0.50%) | 11135.70 (10513.69-11757.71, n=2) **-7.14%** (worst -12.84%) |

**27B MXFP4, `--spec mtp --spec-k 2`, category harness (`scripts/dfh/cats/rcp-*.jsonl`, 8 rows each, 512 forced tokens, temperature 0), 4 rounds,
arms d3 and d1 only (d5 was already -12% in the plain decode and was dropped from this phase).**

ar-a2 MTP k2 decode tok/s per category (mean of rows; mean over rounds (min-max, n)); paired delta vs d3 (mean, worst round)
| category | d3 | d1 |
|---|---|---|
| rcp-code | 108.68 (108.62-108.75, n=4) | 107.15 (106.90-107.29, n=4) **-1.40%** (worst -1.63%) |
| rcp-prose | 83.60 (83.37-83.96, n=4) | 82.11 (82.06-82.17, n=4) **-1.78%** (worst -2.18%) |
| rcp-recall | 99.05 (98.94-99.20, n=4) | 97.59 (97.50-97.62, n=4) **-1.47%** (worst -1.72%) |
| all | 97.11 (96.99-97.23, n=4) | 95.62 (95.53-95.69, n=4) **-1.54%** (worst -1.62%) |

**Decision.** Rule: pick the faster of 1 and 5 if no arm or metric is slower than the default by more than 1%. Mode 1 is slower by 1.9 to 2.7% (27B
decode), 1.4 to 1.8% (MTP k2 categories) and 3.4 to 4.6% (35B decode); mode 5 by 11.6 to 12.3% (27B) and 21.5 to 22.3% (35B). Prefill moves by
0.5 to 1.0% (d1) and 2.1 to 3.5% (d5) on the 27B. Both arms miss that speed bar, but the coordinator overruled it on correctness grounds (2026-10-28): the default is now `INTARSIA_HIP_AR_DRAIN=1` (cost: ~2% decode on the 27B at tp=2, ~4% on the 35B) and `=3` restores the old fast mode. The merge gates for the flip are in section 6; `kernels test --backend hip --quick` passes on this branch (raw/ar-g1/kernels-test.log, exit 0, 10 kernels without cases in the quick sweep).
The cost split is not measured: mode 1 adds a fence to every thread of the all-reduce block, so a cheaper sufficient variant
(fence only in the threads that pushed the last word, or a sequence number in every pushed word as in the next-step note above) could close the gap.

## 6. Default flip to `INTARSIA_HIP_AR_DRAIN=1`: merge gates (lane ars, 2026-10-28)

Coordinator decision: correctness wins over about 2% decode. Mode 3 (the old default) can let a rank add stale peer rows, a data race, so the sums are wrong,
in about 1.5% of fresh tp=2 processes. `INTARSIA_HIP_AR_DRAIN=1` is now the default for HIP tp>1; **`INTARSIA_HIP_AR_DRAIN=3` restores the fast, racy mode**
(`docs/tuning.md`). One window on tatooine (fleetq #6537, radiance down 15:29:54, answering again 15:49:09), tree `~/intarsiaml-ars` at `1d96bc0`,
27B `amd/Qwen3.8-27B-Quark-AWQ-MXFP4`, `generate --tp 2 --devices 0,1`. Scripts `scripts/ar/stages-soak.sh` (stages `kt ids18 soak`), `scripts/ar/ids18_verdict.py`;
raw files in `raw/ars-w1/`.

| gate | result |
|---|---|
| greedy ids vs main (`ids18`): the 18 prompts of `scripts/dfh/ids-prompts` (short, 4k, 16k x 0-5), plain decode (`--spec off`), 256 tokens; 3 fresh runs with `INTARSIA_HIP_AR_DRAIN=3` (main's behaviour) give the reference (majority), 2 fresh runs with the default must equal it | 18/18 equal (`ids18_verdict.py raw/ars-w1`); all 18 prompts: 3/3 main runs agreed with each other |
| confirm soak: 120 fresh tapped processes, short-5, `--spec mtp --spec-k 2`, 16 tokens, default mode, per-row layer taps (`INTARSIA_DET_TAP`) | 0 divergent: 120 of 120 taps byte-identical (`tapdiff.py`), 120 of 120 id lines equal, all exit 0 (mode 1 already had 0/250 tapped in w9) |
| `kernels test --backend hip --quick` | exit 0 |
| Mac `cargo test --workspace --release` | green (202 result lines, 0 failed) |

Cost of the flip, measured in section 5 (same binary, 4 interleaved rounds): decode -1.9 to -2.7% on the 27B (MTP off), -1.4 to -1.8% with MTP k2, -3.4 to -4.6%
on the 35B NVFP4 (2 rounds). `docs/results/2026-10-28-scoreboard.md` carries the new numbers ("after #221 fence default").
