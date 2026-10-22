# Lane fx12: acknowledged `Truncate` / `Snapshot` in the cross-process tp protocol (2026-10-22)

Finding 12 of the 2026-10-22 review (requirement 12, tp correctness). Branch `lane-fx12/tp-remote-acks` off main `659f770c`.
Raw logs: [raw](2026-10-22-fx12-tp-remote-acks/raw/).

## Headline

- **A remote-rank failure on `Truncate` or `Snapshot` now fails the request that caused it, at once, and changes nothing.**
  Before, rank 1 swallowed the error and reported it on the next `Forward`, with the wrong message context.
- **`SequenceState::truncate` no longer moves a rank before it knows the primary can follow.** Before, in-process peers
  were truncated first and the primary's snapshot lookup could still fail, leaving ranks at different lengths.
- **The wire protocol is versioned (v2).** A rank-0 and a rank-1 built with different versions refuse to pair, and both name the versions.
- **No numerics change.** 256 greedy ids identical to main on a 114-token and a 4449-token prompt; a 100-request soak with prefix
  reuse gives 100 of 100 byte-identical answers to main's. The 35B at tp=2 across spark1 and spark2.

## What changed

Files (all in `intarsia-core` plus a test and scripts):
`crates/intarsia-core/src/tp_remote.rs`, `crates/intarsia-core/src/state.rs`, `crates/intarsia-core/src/model.rs` (one line),
`crates/intarsia-models/tests/tp_remote_acks.rs`, `docs/design/35b-tp2.md`, `scripts/tp2-acks/`.

**Wire protocol v2** (`tp_remote.rs`):

- `Truncate` and `Snapshot` are answered with `T_ACK` = `[status u8][remote len u64][message]`. The primary holds the reply queue
  across send and wait, so the next reply is this command's.
- Statuses: `0` ok; `1` refused, remote copy unchanged, unsupported (no snapshot at that length); `2` refused, unchanged, invalid
  (unknown sequence, beyond the length); `3` failed part-way (device error), copy undefined.
- The primary maps them to `Error::Unsupported`, `Error::InvalidCall`, `Error::Backend`. `Backend` is fatal to the engine
  (`is_fatal`), the refusals are not: the request fails, the slot is reset, the next request is served. A status `3` aborts the
  communicator, like a failed forward.
- A failed `NewState` is still fire-and-forget; it is reported by the next answered command (`Truncate`, `Snapshot` or `Forward`).
  `Drop` and `Shutdown` are unchanged.
- The handshake `Hello` carries `tp-protocol=2;` in front of the fingerprint text. A v2 rank reading a v1 hello (bare fingerprint)
  or any other version fails with `tensor-parallel ranks speak different protocols: this rank speaks version 2, the other rank
  version 1. Run the same intarsia build on both ranks`. A v1 rank reading a v2 hello fails on its fingerprint comparison, which
  prints both texts (the old binary cannot be changed). `PROTOCOL_VERSION` is a public constant; bump it on any wire change.
- `handshake` is generic over `Read + Write` (callers pass a `TcpStream` as before).

**Rollback order** (`state.rs`):

- `truncate`: `check_truncate` (length, in-process peers' lengths, snapshots on the primary and the peers; changes nothing), then
  the remote rank (acknowledged), then the in-process peers, then the primary. A refusal anywhere in the first two steps leaves
  every rank as it was. Only a device error in the last two steps can leave ranks apart; it is fatal for the group, and the
  next forward's existing "ranks diverged" check catches it.
- `snapshot`: remote rank first (acknowledged), then peers, then the primary. A failure can leave another rank holding an extra
  snapshot, never the primary; `truncate` checks the primary's snapshots before it asks any other rank, so an extra snapshot is harmless.

**`mark_unusable`** (`model.rs`, one line): when the primary's own pass fails in `forward_remote`, the peer is marked unusable.
Without it, the `st.reset()` that `Model::forward` runs after a failed recurrent forward would wait for an acknowledgement from a
rank stuck in a collective, up to `INTARSIA_TP_REMOTE_TIMEOUT_S` (600 s). A reply timeout marks the peer unusable too.

**Cost:** one round trip per `Truncate` (each admission that reuses a prefix, each forward after a failed one) and one per
`Snapshot` (each prefill chunk of a hybrid model, plus the last-token snapshot). The remote rank has finished the chunk's forward
by then, so the wait is a network round trip plus the small recurrent-state copy. Not separately measured; see the GPU gate for
end-to-end numbers (prefill 4449 tokens: 2284 ms main, 2443 ms this change, single runs, inside run-to-run noise).

## Tests (CPU, `cargo test -p intarsia-models --test tp_remote_acks`)

A scripted fake rank on a loopback socket speaks the wire protocol; the primary side is the real `RemotePeer` and
`SequenceState` on the cpu backend, with a hybrid state (attention KV plus a recurrent slot, snapshots at 3 and 6 tokens).

| Test | Injects | Asserts |
|---|---|---|
| `remote_truncate_refusal_fails_that_request_and_changes_nothing` | rank 1 refuses `Truncate` (no snapshot there) once | `truncate` returns `Unsupported` naming the remote rank and `truncate to 3`; primary length, tokens, snapshots and the remote length unchanged; the same rollback, a snapshot and a reset then work |
| `remote_truncate_failure_is_a_backend_error_and_leaves_the_primary_alone` | rank 1 fails `Truncate` (status 3) | `Backend` error with the message; primary unchanged |
| `remote_snapshot_refusal_fails_that_request_and_changes_nothing` | rank 1 refuses the first `Snapshot` (unknown sequence) | `InvalidCall`; no snapshot on the primary; the next snapshot works; the fake saw 2 commands |
| `truncate_checks_the_primary_before_touching_any_other_rank` | no snapshot at 4 on the primary, one on the in-process peer | `Unsupported`; the remote rank got no `Truncate`; the peer is still at 6 |
| `ranks_of_different_protocol_versions_refuse_to_pair` | a v1 hello, a v3 hello, a v2 hello with another fingerprint, a matching v2 hello | the first three are errors naming the versions or both fingerprints; the last pairs at the smaller capacity |

Existing tests that run the real `serve` loop over a loopback socket still pass: `tp_hybrid` (7, including the remote rank
bitwise equal to the in-process group, snapshot rollback through the remote rank), `hybrid` (10), `intarsia-core` (97 unit, 21 +
2 integration). `cargo check --workspace` passes. The same set ran again on the rebased head (main `34ca7267`, fleetq #3898, [raw/final-*.log](2026-10-22-fx12-tp-remote-acks/raw/)): all pass.

### Each test fails on the old code

The five new tests, copied unchanged into a checkout of main (`659f770c`, 0 occurrences of `T_ACK` in `tp_remote.rs`), all fail
at their own assertion (`raw/models-oldcode.log`):

- `ranks_of_different_protocol_versions_refuse_to_pair`: `tp_remote_acks.rs:234`, the v1 hello pairs.
- the three remote tests: `:131`, `:158`, `:179`, `expect_err` on a `truncate` / `snapshot` that old code reports as `Ok`.
- `truncate_checks_the_primary_before_touching_any_other_rank`: `:213`, `peer unchanged` left `(6, 4, [4])`, right `(6, 6, [4])`: the old order truncated the peer to 4, then the primary failed.

Method note: the first attempt to run them on main passed, wrongly. Cargo's unit hashes for workspace path crates ignore the
worktree path, so two worktrees sharing one `CARGO_TARGET_DIR` reuse each other's artifacts when the sources look older. The proof
ran in its own target directory with all sources touched.

## GPU gate: tp=2 across spark1 and spark2, Qwen3.6-35B-A3B-NVFP4 (CUDA)

ONE fleetq job, #3828 (`--host spark1,spark2 --exclusive --takes-down flight,guidance,capcom,retro`), 02:26 to 02:34 EDT, inside
the 00:00 to 07:00 window. Binaries built in a non-exclusive slot (#3827): dev = this change at `dbf7fa16` (code tree `203f21110`,
clean), base = main at `73e1ebd` (code tree `564431e01`), copied to both Sparks. Script: `scripts/tp2-acks/run.sh`; raw:
[raw/window-0226](2026-10-22-fx12-tp-remote-acks/raw/window-0226/).

| Check | Result |
|---|---|
| 256 greedy ids, 114-token prompt (`generate --chat --no-think --ignore-eos --show-ids`), base vs dev | identical, 256 of 256 |
| 256 greedy ids, 4449-token prompt, base vs dev | identical, 256 of 256 |
| Seeded 100-request soak against `serve --tp 2 --tp-remote` (scripts/tp2-acks/soak100.py: 20 short, 40 reuse, 20 repeat, 10 turn, 10 long; greedy, serial) | dev: 100 requests, 0 errors, 30 prefix-reuse hits (`cached_tokens > 0`), health ok after, 0 error lines in the server log, 57.7 s. main: same, 59.5 s |
| Soak answers, base vs dev | 100 of 100 byte-identical |
| Base rank 0 paired with a dev worker | refused on both sides: the dev worker says `this rank speaks version 2, the other rank version 1`; the base rank prints its fingerprint mismatch with `tp-protocol=2;` in the other rank's text. Exit 1, no output |

Decode tok/s in the id runs (single runs, not a benchmark): short 91.3 main, 95.9 dev; 4k 91.5 main, 91.2 dev.

Downtime (`raw/downtime.log`): spark1-vllm 464 s and spark2-vllm 464 s, both restored and answering `/v1/models`.
The 100-request soak does not inject a failure on the GPU; the failure paths are covered by the CPU tests above.

## Decisions

- Refusals are non-fatal errors and device errors stay fatal; the status byte carries which (LOG.md row of 2026-10-22 01:50).
- The version rides inside the fingerprint text so that a v1 binary refuses a v2 hello with a visible mismatch instead of
  misreading it. The alternative (a new field after the capacity) would have been read by v1 as part of the fingerprint too, with
  an unreadable message.
- `NewState` stays unacknowledged. The finding names `Truncate` and `Snapshot`; `NewState` runs at start-up and per new
  sequence, and its failure already fails the next answered command.
- `Truncate` is not skipped when the remote copy is already at `len`: truncating to 0 must zero the recurrent state even when empty.

## Not done / uncertain

- The failure paths are tested with a fake rank, not injected on the GPUs.
- Per-request cost of the added round trips is bounded by the 4k prefill number above, not measured on its own.
- The PR changes `intarsia-core` (shared by every backend): the coordinator runs the Metal no-op check before merge.
