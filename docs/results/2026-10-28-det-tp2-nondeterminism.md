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

Common to all: a fresh process, the first ~30 generated tokens, a flip at a margin (0.08, 0.12) well above the smallest margins before it (token 17 at 0.0123 did not flip). That is a discrete wrong value in
one step, not rounding noise. Repeats inside one process (130 runs) never parted.

TODO: w4 (tapped fresh processes: which pass and layer), w5 (rate, tp=1 per card), the explorer audit of work in flight at process start.

## 4. Raw

- `raw/w1/` .. `raw/w3/`: per arm `<tag>.err` (ids, margins, stats), `<tag>.tap` (layer hashes), `steps.log`, `window.log`; `raw/downtime.log` (radiance windows).
