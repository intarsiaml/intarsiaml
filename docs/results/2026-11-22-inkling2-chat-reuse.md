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
