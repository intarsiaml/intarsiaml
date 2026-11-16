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
