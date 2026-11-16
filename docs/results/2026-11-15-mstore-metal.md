# mstore: the prefix store with MTP on Metal (roadmap idea #8)

Lane `mstore`, branch `mstore` from origin/main `5c9cb111e`, worktree `.worktrees/mstore`, box endor (M5 Max, 128 GB,
macOS 27.0). Raw files: [`2026-11-15-mstore-metal/raw/`](2026-11-15-mstore-metal/raw/). Scripts: [`scripts/mstore/`](../../scripts/mstore/).

## Brief + progress (kept at the top so it survives compaction)

Brief (coordinator, 2026-11-15 evening, condensed):

- Goal: make the prefix store work with MTP on Metal, then build the case for default-on for `serve` on Mac. Idea #8 of
  `docs/design/2026-11-15-mac-roadmap.md` (section 2.1). CUDA side: lane mtp-store, PR #371 (`MtpState::base`: the head's
  context restarts at the restore point; no drafting inside a parked prefill chunk). Metal guard: `load.rs` near line 710
  (a configured store turns drafting off; `INTARSIA_MTP_STORE=1` forces it on).
- Step 1, gates on Metal with `INTARSIA_MTP_STORE=1`, 35B and 27B: ids warm == cold with MTP (greedy and seeded sampled,
  single and `--max-batch`); the store's reuse gates; a restart test (the store survives a server restart and restores);
  acceptance after a restore. Fix failures bitwise-neutrally.
- Step 2, measure: TTFT for a 4k / 32k shared prefix cold vs restored (MTP on); decode tok/s after a restore vs cold;
  the cost of the head catch-up at restore; memory and disk footprint at a sensible default budget.
- Step 3, only if measurement says so: persist the MTP head's KV in the entry; an ahead-of-time head catch-up.
- Step 4: PR 1 = fixes + this doc, default unchanged (coordinator merges on green gates). PR 2, separate, unmerged:
  Metal `serve` defaults to MTP-with-store (`INTARSIA_MTP_STORE` unset = on for Metal) and, if the data supports it, the
  store on by default for `serve` on Metal with a disk budget; title "... (USER DECISION, do not merge)", numbers in the body.
- **Stop rule:** if the head catch-up at restore costs more than 20 % of the saved prefill, write it up and stop.
- Rules: a restore never changes tokens; knobs unset leave `plan show --json` byte-identical; worktree and store dirs only
  under `.worktrees/mstore`; fleetq for every GPU run (`--quiet gpu`, jobs <= 32 GB, never `--force` / `--no-queue` /
  `--exclusive`); ABBA with fresh servers; plain per-chunk gap p99, not localeval's weighted ITL p99; lanes mtt / mnr / mka
  edit `model/mtp.rs`, `model/spec_pipe.rs`, `engine.rs`, `engine/batch.rs` and Metal kernels: keep edits there small.

Progress:

- 21:04 build #9478 (CLI), #9494 (gate test). New gate `crates/intarsia-models/tests/mtp_store_metal.rs`; scripts
  `scripts/mstore/{gate.sh,window.sh,req.py,prompts.py}`.
- Queued: gates #9498 (35B), #9499 (27B); windows #9502 (35B 4k), #9503 (35B 32k), #9504 (27B 4k), #9505 (27B 32k).
- Definition of the head catch-up cost (the restart design has no catch-up compute; the head starts empty at the restore
  point, so the cost is lower acceptance): decode time over the same tokens restored minus cold, divided by TTFT cold minus
  TTFT restored, per cell (4k / 32k x new turn after the restore point / repeated prompt).
- 21:50 gate 35B #9498: **PASS** (1 passed, 199 s; `raw/gate-35b.log`). Single greedy, cold-engine greedy, single
  sampled restored == cold, batched greedy, batched sampled restored == cold, restart from disk: all true. Store off gives
  other ids than store on (5 of 6), as documented for Metal (the boundary-cut schedule changes prefill bits).
  Acceptance: cold turns 77.8 %, restored turns with a new turn 77.5 %, store-off whole head 81.0 % (later turns);
  all but 2 tokens restored from the store 58.7 to 68.7 % (63.0 % greedy batched, 63.3 % sampled).
- 22:0x gate 27B #9499 (`INTARSIA_BATCHED_MTP=1` for the batched arms, opt-in on the 27B): **PASS** (1 passed, 385 s;
  `raw/gate-27b.log`), all six checks true. Acceptance: cold turns 77.3 %, restored turns with a new turn 61.0 %
  against 66.3 % with the whole head context (store off; not the same tokens), all but 2 restored 45.2 % (sampled) /
  54.4 % (greedy, batched).
- 22:45 reuse gate 35B #9514 (`raw/reuse/35b-*.out`, chunk 1024, 4k golden): store off (main's default, MTP on, the
  engine's own hybrid reuse) FAILS `long_code_4k continue` (3 of 32 agree, KL 1.6e-1), the known failure on main (lane hrr,
  `2026-10-29-hrr-hybrid-reuse.md`). Store on: all 9 rows agree **32 of 32** with the "cold" run, but the gate reports FAIL
  on its own bookkeeping, not on tokens: "the cold run was not cold" (its cache flush replaces the held sequence, not the
  store, so the cold run restores too), "reused N > the shared prefix" (an earlier case of the same prompt left a longer
  entry; the store compares every token id, so it is the same prefix), "reuse covered 1024 < 2000" (the store resumes only
  at its own cut points). So `accuracy --mode reuse` cannot judge a configured store as written; the store's warm == cold
  gate is the fresh-engine comparison of `tests/mtp_store_metal.rs`. Leftover: a store-aware reuse mode.
- 22:40 pinned measurement binary `target/mstore-run/intarsia-measure` from `345aec626` (no local crate changes).
- PR 2 draft on branch `mstore-default` (worktree `target/pr2wt`): `mtp_store_default_for` = cuda | metal, tests and docs.
