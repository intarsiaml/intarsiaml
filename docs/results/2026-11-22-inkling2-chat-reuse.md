# Inkling-2 serve: KV prefix reuse across chat turns (lane ink-spec-land)

Branch `ink/chat-reuse` from main 22303d704 (worktree `.worktrees/ink-chat-reuse`). Not merged during the #491 freeze; report
before merging (it changes output on multi-turn chat).

## Brief (coordinator, 2026-11-22 ~16:30; the coordinator's message names it "L2c: chat prefix reuse")

- Problem: chat turn 2 TTFT is 1.55 s at about 4.6k against 0.085 s when the prefix is reused, about 18x (#11634, `--spec off`
  server: turn 2 4593 prompt rows, 0 cached; the `/v1/completions` continuation 4560 cached).
- mlx-serve keeps a ring of window + 2048 rows per sliding visit (`.worktrees/mlx-serve-src/src/plugins/inkling2/AGENTS.md`, "A
  sliding visit keeps a ring of window + 2048 rows"). Read it for the design; don't copy it.
- Before building: the memory cost per stream and at B16 for margins 256 / 1024 / 2048, and which margin covers the template's
  ~30-row divergence with room to spare. Compare a larger margin with a ring snapshot at each prompt boundary; pick one.
- Build here, with an env that reverts to margin 16.
- Gates: single-turn md5 64/256 and G2/G3 unchanged; batched == solo digests; prefix-store tests pass; chat flow: turn 2 reuses
  >= 4000 rows and TTFT <= 0.2 s at about 4.6k; the turn-2 text equals a full re-prefill's, or the difference is explained as
  prefill-vs-decode numerics with a greedy G2-style KL on turn 2 against a full re-prefill; server memory at max-batch 16 within
  the measured budget.
- Report before merging; nothing goes to main during the #491 freeze.

## Design: larger margin vs ring snapshot (before building)

**Why turn 2 misses today.** `Model::reuse_len` on Metal resumes only inside rows that prefill passes of >= 2 rows wrote
(`Backend::prefix_reuse_min_pass_rows`, lane prc); a decode step's rows never count. So turn 2 can resume at most at turn 1's
prompt end L1, whatever the template does, and the cut is `reply length` rows below the cached end (the reply's reasoning and
answer were decode rows). The ring slots allow a cut of at most `INK_RING_MARGIN` = 16 rows (`SequenceState::ring_cut_ok`), so
any reply longer than 16 tokens sends turn 2 back to 0. The ~30-row divergence of #11620 was a 32-token test reply; real
Inkling replies with reasoning run hundreds to thousands of tokens.

**Ring bytes per stream** (from the slot specs, `model/inkling2.rs` `push_slots`; bf16; 48 visits, 40 of them sliding; window
1024, Hk x Dh = 1024, hidden 2304, conv taps 4): KV ring 40 x (1024 + m) x 4 KiB, conv rings 48 x (3 + m) x 13 KiB. Each margin
row costs 0.766 MiB per stream, 82% of it the conv rings (every visit has two).

| margin | rings per stream | over today | at B16 | over today at B16 | covers a turn-1 reply of |
|---:|---:|---:|---:|---:|---|
| 16 (today) | 174.1 MiB | - | 2.72 GiB | - | < 16 tokens |
| 256 | 357.8 MiB | +183.8 MiB | 5.59 GiB | +2.87 GiB | < 256 tokens |
| 1024 | 945.8 MiB | +771.8 MiB | 14.78 GiB | +12.06 GiB | < 1024 tokens |
| 2048 (mlx-serve's) | 1729.8 MiB | +1555.8 MiB | 27.03 GiB | +24.31 GiB | < 2048 tokens |
| 16 + one ring snapshot at each prompt's end | 348.2 MiB | +174.1 MiB | 5.44 GiB | +2.72 GiB | any length |

A prefix-store entry holds a sequence's rings too, so a larger margin also grows every stored entry; a snapshot does not.

**Pick: the ring snapshot.** At the end of each prompt's prefill (`Model::forward_prefill` with a last-row head) the sequence
copies its ring slots (one 174 MiB device copy, about 0.5 ms) and keeps the latest. A later request in that slot whose reuse cut r
lies in [s - 16, s] (s = the snapshot's length) restores the rings from it and cuts by length as today. It costs what a margin
of about 243 rows costs and covers any reply length, where margin 2048 costs 9x as much and still misses longer replies. The
snapshot buffer is reserved with the slot and counted in the memory plan (no allocation while serving).
`INTARSIA_INK_RING_SNAPSHOT=0` reverts to today's behaviour (margin 16, no snapshot, no reserved bytes).

Exactness: the restore puts back the exact ring bytes of turn 1's prefill, and Metal passes of >= 2 rows agree bitwise whatever
row they start at (the rule `reuse_len` already relies on), so turn 2 is expected to equal a full re-prefill bit for bit; the
gate checks it.

## Coordinator ruling (2026-11-22 ~17:00): snapshot approved, no margin change

Keep the bitwise gate (turn 2 == a full re-prefill). Added gates:
1. A 3-turn chat with replies over 1k tokens (reasoning on): turns 2 and 3 each reuse the previous prompt's end, and their text
   equals a full re-prefill's.
2. B16 serve with 16 concurrent chats: each slot restores only its own snapshot (slots crossed or reordered give the same
   per-slot text), and the memory plan counts the +2.72 GiB.
3. A restore whose cut is more than 16 rows below the snapshot falls back to a full prefill, with correct output.
4. `INTARSIA_INK_RING_SNAPSHOT=0` behaves the same as main.
Report the chat turn-2 TTFT at about 4.6k (target <= 0.2 s) and at 16k. Nothing to main during the #491 freeze.

## Gates on b7351ee4b (endor M5 Max, Metal)

**#11644** (`--quiet gpu,cpu`; main binary 22303d704). The job died after the solo chat flows: `gate.sh` was edited while bash
was still reading it (a fix to the G comparison below), so its last two sections never ran. The flows' JSON results were complete
and were scored afterwards with the job's own scoring code; gate B ran as its own job.
- T PASS: tiny Metal tests `inkling2_chat_reuse` 5/5, `inkling2` 11/11, `inkling2_batch` 8/8 (cpu on bespin #11641: the same,
  plus `docs_env_vars` 2/2 and intarsia-core lib 260/260).
- M PASS: greedy md5 64 `65cde8f3...` and 256 `61fd59fe...`, identical for main, on (default) and off (`INTARSIA_INK_RING_SNAPSHOT=0`).
- G PASS: G2 short / long and G3 short / long / 4k self-goldens, on vs main: every tensor bitwise identical (48 / 60 / 48 / 60 /
  18 tensors). The job's whole-file `cmp` reported FAIL. That was a measurement bug: the files differ only in `__metadata__`,
  which records the binary's path and build time. The script now compares tensors (`.scratch/st_same.py`). OVERALL lines are
  identical too (g3-short exits 1 on main as well: its mean KL 1.204e-2 is above the 1.05e-2 bar on main, unchanged).
- C PASS, solo server (`--max-batch 1`), on vs off (off prefills every turn again, so it is the full re-prefill reference):

| chat | turn-1 prompt | turn-1 reply | turn-2 prompt | on: cached / TTFT | off: cached / TTFT | turn-2 text on == off |
|---|---:|---:|---:|---:|---:|---|
| ~4.6k | 4575 | 189 | 4633 | 4575 / **0.132 s** | 0 / 1.550 s | yes (and turn 1) |
| ~16k | 16125 | 256 | 16152 | 16125 / **0.136 s** | 0 / 7.092 s | yes (and turn 1) |

  Three turns, reasoning on (effort 0.9), replies 1169 and 1200 tokens: turn 2 (2358 rows) cached 1957 = turn 1's prompt, TTFT
  0.305 s vs 1.238 s off; turn 3 (3043 rows) cached 2358 = turn 2's prompt, 0.405 s vs 1.405 s; all three turns' text equal to
  off's. Device memory in use, idle solo server: on 33.08 GiB, off 32.91 GiB (+0.17 GiB = one 174 MiB snapshot).

## Coordinator ruling (2026-11-22 ~17:30): merge on the coordinator's judgement, default on

1. When gate B (#11649) passes, open the PR from `ink/chat-reuse`, with the TTFT table, the memory numbers (solo +0.17 GiB, B16
   as measured) and the revert env (`INTARSIA_INK_RING_SNAPSHOT=0`) in the body.
2. Hold the merge for the #491 freeze, as with #496. After #491 lands: merge main in, rerun md5 and the chat-reuse tests, then
   tell the coordinator, who merges #496 and this PR one after the other.

**#11649, gate B: FAIL** on b7351ee4b (endor; B16 serve, 16 concurrent chats sharing a 20-paragraph system preamble).
- Turn 1 was identical across all arms: on (forward), on (shuffled), off, and the solo server.
- On turn 2, on forward == on shuffled, and every conversation resumed at its turn-1 prompt end. But 7 of 16 replies (0, 1, 2, 3,
  7, 10, 15) differed from off and from the solo server. Off == solo on all 16.
- Memory in use at start: on 46.02 GiB, off 43.30 GiB (+2.72 GiB, the 16 snapshots as planned).

**Cause.** Every turn 2 had a 14-row remainder (808 prompt rows, 794 resumed).
- Inkling-2 on Metal runs passes of 2..=16 rows on the row-exact kernels (`inkling2::ink_rows_exact`). Those are bitwise the decode
  step per row, not the fast prefill kernels that a full re-prefill runs for those rows.
- Metal's reuse rule (`prefix_reuse_min_pass_rows` = 2) assumes any split of a prompt into passes of >= 2 rows is bitwise. That was
  measured on Qwen3-8B, and it does not hold for Inkling-2.
- The solo flows above had remainders of 27-685 rows. The tiny Metal tests use chunk 16, where every pass is row-exact. So neither
  caught it.
- main has the same latent issue for any Inkling-2 resume with a 2..=16-row remainder, but the 16-row margin made such resumes rare.

**Fix (bdcdee4b1).** `Model::model_min_pass_rows`: for Inkling-2 on Metal with a prefill chunk above 16, the minimum pass is 17
rows. So the reused rows must come from passes of >= 17 rows, and the resume point backs off until the remainder has >= 17 rows
(791 instead of 794 here), which is still inside the ring margin below the snapshot.
- New tiny test `inkling2_chat_short_remainder_matches_cold` (chunk 64). On Metal, a 10-row remainder resumes at 13, bitwise the
  cold prefill. On cpu (any cut) nothing changes.
- Gate C gains a "short remainder" flow (`/v1/completions`, turn 2 = turn 1's prompt + a few tokens; the expected resume is 17
  rows before the prompt's end).
- The full gate set reruns on the new code.

## Coordinator rulings (2026-11-22 ~17:00-17:15)

- The fix is approved: a minimum pass of 17 rows for Inkling-2 on Metal when the chunk is above 16, plus `wide_reuse_len` for the
  cached rows. Two additions before the PR:
  1. Gate R (B16): set the turn-2 remainders on purpose to cover 1, 2, 8, 13, 16, 17, 18 and 40 rows across the 16 slots. All 16
     turn-2 replies must be identical to the full re-prefill, in both forward and shuffled order.
  2. Add a snapshot=0 arm to that sweep, since main has the same latent issue for 2-16-row remainders. The PR body says the change
     also fixes main's `/v1/completions` resumes.
- The back-off must stay within the 16-row ring margin. Any remainder that would need more falls back to a cold prefill (gate 3).
  Open the PR only when all gates pass.
- #491 merged as main 8c7bb7cb1; the freeze is lifted. Merge order on main:
  1. #497
  2. the CUDA flips PR
  3. #496, after the 05:45 night speed run passes
  4. this PR, after all its gates pass
- Before merging either of this lane's PRs: merge main in, rerun md5 and the affected tests, and ask the coordinator for the go.

## Gates on bdcdee4b1, the 17-row fix (#11659, endor `--quiet gpu,cpu`, main binary 22303d704): all pass

- T PASS: tiny Metal `inkling2_chat_reuse` 6/6 (with `inkling2_chat_short_remainder_matches_cold`: a 10-row remainder resumes 3
  rows earlier, bitwise the cold prefill), `inkling2` 11/11, `inkling2_batch` 8/8. CPU on bespin (#11658): the same plus
  `inkling2_dspark` 3/3, `docs_env_vars` 2/2 and intarsia-core lib 260/260.
- M PASS: md5 64 `65cde8f3...` and 256 `61fd59fe...`, the same for main, on and off.
- G PASS: G2 short / long and G3 short / long / 4k self-goldens, on vs main, every tensor bitwise identical.
- C PASS, solo server, on vs off (off = full re-prefill), text equal everywhere:

| flow | turn-2 prompt | on: reused / TTFT | off: reused / TTFT |
|---|---:|---:|---:|
| ~4.6k chat, 189-token reply | 4633 | 4575 / 0.137 s | 0 / 2.108 s |
| ~16k chat, 256-token reply | 16152 | 16125 / 0.135 s | 0 / 6.918 s |
| ~4.6k completions, 40 decoded, turn 2 = prompt + 3 tokens | 4560 | 4543 (17 before the end) / 0.075 s | 0 / 1.780 s |
| 3 turns, reasoning on: turn 2 (reply 1169 before) | 2358 | 1957 / 0.274 s | 0 / 1.187 s |
| 3 turns: turn 3 (reply 1200 before) | 3043 | 2358 / 0.384 s | 0 / 1.547 s |

- B PASS: B16 serve, 16 concurrent chats. Every turn 2 resumed 3 rows before its turn-1 prompt end (14-row remainders, now 17),
  and all 16 turn-2 replies equal off and the solo server, forward and shuffled. Device memory in use at start: on 46.02 GiB, off
  43.30 GiB (+2.72 GiB, the 16 snapshots as planned).
- R PASS (B16, token-id `/v1/completions`, 16 prompts of 600-660 rows with distinct first tokens, so no request reuses another's
  rows; turn 2 = turn-1 prompt + 1 / 2 / 8 / 13 / 16 / 17 / 18 / 40 tokens, twice each; reference = the 16 turn-2 prompts cold,
  0 rows reused):
  - on, forward and shuffled (turn 1 = 40 tokens decoded, so only the snapshot can serve): resume points 16 / 15 / 9 / 4 / 1 / 0 /
    0 / 0 rows before the turn-1 prompt end, as the 17-row rule says; all 16 turn-2 replies equal the cold ones.
  - snapshot off (`INTARSIA_INK_RING_SNAPSHOT=0`, turn 1 = 4 tokens, so the live rings can serve): remainders 1 and 2 go cold
    (the back-off leaves the live margin), the others resume as on, except two that went cold (8 and 17 in the second half; most likely the
    four cold requests took the least recently used free slots, two of which held those conversations); all 16 turn-2 replies
    equal the cold ones.
  - main 22303d704 (informational, the same snapshot-off flow): resumes at the prompt end with 2..16-row remainders (1 row before
    it for the 1-token remainder), and 8 of 16 turn-2 replies differ from the cold ones. That is main's latent bug, which the fix
    removes.

Marker `.scratch/gate_pass_bdcdee4b1`.

## Coordinator rulings (2026-11-22 ~17:30-17:45)

- New merge order: #497, then the CUDA flips PR, then this PR, then #496 (after #11621 passes).
- Open the PR when push #11668 lands. Body: the C table, B16 memory (+2.72 GiB), the R result including main's 8/16 mismatch as
  the bug it fixes, and `INTARSIA_INK_RING_SNAPSHOT=0` as the revert.
- On the go: merge main in, rerun md5 and the tiny / chat-reuse tests, then merge.
- Push from endor (bespin and the Sparks cannot push to GitHub). #11668 stays queued with `--load cpu`.
