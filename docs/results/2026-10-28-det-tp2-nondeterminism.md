# det: run-to-run nondeterminism of greedy decode, Qwen3.8-27B MXFP4 tp=2 on the R9700s (HIP), 2026-10-28

Lane `lane-det/tp2-nondeterminism`, base `origin/main` `bb48df640`. tatooine, 2x R9700 (gfx1201), `amd/Qwen3.8-27B-Quark-AWQ-MXFP4`, `generate --tp 2 --devices 0,1`, W4A8 (recipe default),
greedy, `--ignore-eos`. Raw logs: `2026-10-28-det-tp2-nondeterminism/raw/`. Scripts: `scripts/det/`. Comparison: `tools/det/detcmp.py`.
Status: IN PROGRESS (sections marked TODO are filled as windows land).

Trigger: lane vat saw one plain `--spec off` run of ids prompt short-5 part from four other off runs (`2026-10-28-vat-verify-attn-rows/raw/w2/ids/gen-tp2-short-5-off.ids`,
first different token 27), and ho-chat `chat-sleep` differ from the off text in 1 of 5 fresh-server rounds of each dflash arm; lane vrc saw one old-path dflash text (ho-recall round 1) part from off.

## Headline

TODO

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
(poison, syncs); tp=2 against tp=1 (6/443 against 0/200, p ~ 0.1 on its own) points at the two-GPU part, which brings the P2P all-reduce back.

## 4. Raw

- `raw/w1/` .. `raw/w3/`: per arm `<tag>.err` (ids, margins, stats), `<tag>.tap` (layer hashes), `steps.log`, `window.log`; `raw/downtime.log` (radiance windows).
