# Lane copyflipm: copy drafts default-on for Metal single-sequence MTP (decision PR)

## Brief

Lane `copyflipm` in /opt/code/intarsiaml (Rust inference engine), on endor (Apple-silicon Mac, Metal). Prepare a **decision PR**: copy drafts default-on for Metal single-sequence MTP serving. Open it; DO NOT merge — the user decides.

First, write this brief (verbatim, as a "Brief" section) at the top of `docs/results/2026-11-18-copyflipm.md` so it survives context compaction.

### The change
Today `INTARSIA_SPEC_COPY` unset = off everywhere (`crates/intarsia-core/src/spec.rs` `copy_from_env`, `set_copy_backend`; load-time checks in `crates/intarsia-models/src/load.rs` ~line 1030 refuse it with `--max-batch` > 1, tp > 1, or a backend other than CUDA/Metal/cpu; engine use in `crates/intarsia-core/src/engine.rs`). New behaviour:
- **Unset on Metal** with `--spec mtp`, single-sequence engine (`--max-batch 1` / `generate`), tp 1 → copy drafts on as if `INTARSIA_SPEC_COPY=20` (len resolves to 15 on Metal per PR #456).
- **Unset anywhere else** (CUDA, HIP, cpu, `--max-batch` > 1, tp > 1, no `--spec mtp`) → off, silently, exactly as today — no refusal, no warning, plan unchanged. This is critical: a default must never make a batched server refuse to start.
- **Explicit** `INTARSIA_SPEC_COPY=0` (or empty) → off everywhere. An explicit value in an unsupported config keeps today's refusal.
- The default and the explicit value must go through ONE resolution point so the load-time verify-row width / memory plan and the engine's chain agree (look at how PR #456 did `set_copy_backend`; `gh pr diff 456`).
Pattern: PRs #445 / #456 (`gh pr view 456`): knob change, tests (unset-on-metal-mb1 = on; unset at mb>1 = off with no error; `=0` = off; CUDA unset = off), `docs/tuning.md` row (default, revert `INTARSIA_SPEC_COPY=0`), results doc, LOG row at the TOP of the dated table (newest-first) "proposed, user decision", plus the LOG defaults-table row if there is one (hand-merge that table on conflicts), gate scripts in `scripts/copyflipm/`.

### Evidence (cite, don't redo)
`docs/results/2026-11-18-copymetal.md` (PR #459): Metal 35B, 4 ABBA rounds, `=20` (len 15) vs base: copy3k 2.10x, json-xform 1.64x, agent-edit 1.22x, agent-batch 1.075x, chat-short 1.037x, reason 1.032x, code-gen 1.024x; passes the bar (no cell <0.995, agent ≥1.03) on all 7; ids equal `--spec off`. Caveats from that doc: code-gen round 3 alone 0.983x; a ~2% offset on control cells in both copy arms not explained; not measured: sampled, thinking, real agent transcripts. Copy chains only fire for greedy requests — confirm in code that sampled requests are untouched, and say so.

### Gates (endor, inside fleetq slots)
- Metal crate tests, intarsia-core spec tests, intarsia-models load tests if any, `cargo test -p intarsia-cli --test docs_env_vars`.
- `plan show --json` (model `$HOME/models/Qwen3.6-35B-A3B-NVFP4`, metal): (a) `--spec mtp --spec-k 2` mb1 unset == origin/main with `INTARSIA_SPEC_COPY=20`; (b) same with `INTARSIA_SPEC_COPY=0` == origin/main unset (byte-identical); (c) `--max-batch 8` unset == origin/main mb8 unset; (d) `--spec off` unchanged; (e) `--backend cpu` (or whatever plan show allows without a GPU) unset unchanged, as a stand-in for non-Metal.
- `serve --max-batch 8 --spec mtp` with the variable unset starts and answers one request (then stop it).
- Greedy ids: default vs `=0` vs `--spec off`, on copy3k, json-xform, code-gen (generate). And one sampled request (temperature 0.7, fixed seed) default vs `=0`: identical output.
- One sanity tok/s line per arm; no long A/B.

### Fleet rules
Read /opt/code/fleetq/README.md and /opt/code/fleetq/deploy/LOCAL.md; `fleetq status` first; `--host endor`, always `--expect MIN` (guess high), `--quiet gpu,cpu` for timing, `--load cpu` for builds/tests; label "copyflipm ...". FLEETQ_OWNER=claude-0b386a17-copyflipm. endor runs standing services (kev, tensorfold) — leave them alone. Never leave a slot held; never release another owner's job; kill only by PID.

### Work folder & hygiene
Only `/opt/code/intarsiaml/.worktrees/copyflipm/` (`git worktree add /opt/code/intarsiaml/.worktrees/copyflipm -b copyflipm origin/main`; if PR #459 hasn't merged yet when you start, that's fine — rebase on main before opening the PR so the evidence doc is on main). Per-worktree CARGO_TARGET_DIR, and a separate one for tests vs the `--features metal` binary. PR body in `.worktrees/copyflipm/pr-body-copyflipm.md`. Commit messages end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>` (use exactly this line); PR body ends with `🤖 Generated with [Claude Code](https://claude.com/claude-code)`. PR title `[decision] Metal: copy drafts default-on (single-sequence MTP)`. No single tool call or wait over 4 min (Bash timeout ≤ 240000): long work with nohup, poll at most every 4 min. Wait for CI green (≥6 SUCCESS on current head).

Report: PR URL, CI state, each gate pass/fail with one-line evidence, where the default is resolved (file:line), anything skipped.

Date 2026-11-18, endor (Apple silicon, Metal), Qwen3.6-35B-A3B NVFP4, `--spec mtp --spec-k 2`. **Decision PR, not merged.**

## The decision

With `INTARSIA_SPEC_COPY` unset, copy drafts are **on as if `INTARSIA_SPEC_COPY=20`** (chain length 15 on Metal, #456) when all of these hold: backend Metal, `--spec mtp`, the single-sequence engine (`--max-batch 1`, `generate`, the default), tp 1. Everywhere else unset stays off, silently: CUDA, HIP, cpu, `--max-batch` above 1, tp above 1, no `--spec mtp`. No refusal, no warning, the plan unchanged. `INTARSIA_SPEC_COPY=0` (or empty, or an unusable value) is off everywhere; an explicit value in an unsupported config is refused at load exactly as before. **Revert: `INTARSIA_SPEC_COPY=0`, no rebuild.**

### Which runs this reaches (read before deciding)

The default is defined by backend, batch size, tp and `--spec mtp`, as the brief asks, **not by model and not by an explicit `--spec` flag**. On Metal at tp 1 and one sequence the recipe turns `--spec mtp` on by itself (`[tuning.metal] mtp_default_k`: the Qwen3.6-35B-A3B, lane mpf / 2026-11-05; the Qwen3.8-27B, lane m27), and the default is read after that, so **a bare `serve` or `generate` (no `--spec` flag) on either model now has copy drafts on**, and so does any other model with an MTP head on Metal that a user runs with `--spec mtp`. `INTARSIA_SPEC_DEFAULT=0` (no recipe MTP) or `--spec off` still means no copy drafts, since the chain needs `--spec mtp`. Only the 35B has the A/B speed evidence (#459). The Qwen3.8-27B has the ids gate below (3 cells, greedy ids equal `--spec off`) and one sanity tok/s line per arm, no A/B; any other MTP model has neither. Restricting the default per recipe (for example a `[tuning.metal]` key) is a choice for the user; this PR does not narrow it.

## Evidence (not re-measured here)

`docs/results/2026-11-18-copymetal.md` (#459): Metal 35B, served, 4 ABBA rounds, `INTARSIA_SPEC_COPY=20` (len 15) against off: copy3k 2.10x, json-xform 1.64x, agent-edit 1.22x, agent-batch 1.075x, chat-short 1.037x, reason 1.032x, code-gen 1.024x. No cell below 0.995 and the agent cells at least 1.03, on all 7; ids equal `--spec off`.

Caveats carried over from that doc: code-gen round 3 alone was 0.983x; a ~2 % offset on the control cells appears in both copy arms and is not explained; sampled requests, thinking mode and real agent transcripts were not measured. The sampled case is covered below by construction (chains never fire for it) and by one identical-output check; the other two remain unmeasured.

## Where the default is resolved

One place: `copy_from_env` (`crates/intarsia-core/src/spec.rs:561`), via the private `resolve_copy` (explicit value wins; `0` / empty / unusable is off; unset is `20` when the loader said the default applies; a value without `,<len>` gets the backend's length from `set_copy_backend`, #456). Both consumers call it, so they cannot disagree: the load-time verify-row width and the memory plan (`SpecConfig::save_rows` / `verify_rows`, spec.rs) and the engine's chain (`crates/intarsia-core/src/engine.rs:801`, which also names the default in its one startup line). The decision whether the load qualifies is made once, in the loader, where `max_batch` is known and before any `verify_rows`: `crates/intarsia-models/src/load.rs:828` calls `set_copy_default(copy_default_applies(backend, max_batch, tp))` (spec.rs:525 / 532); the tensor-parallel rank entry `serve_remote_rank` sets it false (load.rs:305), and the flag is false until a loader sets it, so any other entry point (tests, other engines) keeps the old behaviour. The load-time refusal (load.rs:1040) now looks at `copy_explicit_from_env` (spec.rs:548), the user's own value only, so the default can never make a batched server refuse to start.

**Sampled requests are untouched.** The chain is attached per request as `seq.copy = sh.copy.filter(|_| seq.spec_on && seq.greedy && st.peers.is_empty() && st.remote.is_none())` (`engine.rs`, in `Seq::build`), so a request with temperature above 0 never proposes a chain and runs the MTP step it ran before; the batched scheduler never sets `copy` at all. The one cost a sampled request on the default can see is the larger verify-pass width and per-row state reservation made at load (below).

## Memory

With the default the load reserves 15 per-row recurrent-state buffers instead of 2 and runs 16-row verify passes. Three measurements (`scripts/copyflipm/mem.sh`, `raw/mem.log`):

- **Reservation**: `serve --max-batch 1 --spec mtp --spec-k 2`, default against `INTARSIA_SPEC_COPY=0`: `intarsia_kv_bytes_total` 233,635,840 and `intarsia_kv_tokens_capacity` 8192 in both (the KV cache is unchanged), and `intarsia_device_memory_free_bytes` 72.31 GB against 71.46 GB, i.e. **about 0.86 GB less free with the default** (one measurement; it matches 13 more buffers at about 63 MB each, the per-row size the engine comments give for this model). That is the cost of the default on a 22 GB model; it comes out of the memory budget, so a server run near its `--mem-budget` has 0.86 GB less for KV.
- **Footprint of a run in which chains fire** (600-token copy3k `generate`, 37 chain passes in each default run, `/usr/bin/time -l`, 2 rounds): peak memory footprint default 21.93 / 21.91 GB, `=0` 21.84 / 22.00 GB; maximum resident set size 41.6-41.7 GB in all four. Within noise: the reserved buffers are mostly untouched pages.
- A first 64-token run in which **no chain fired** (earlier draft of `mem.sh`) showed the same footprint and says nothing about the reservation; it is superseded by the two above.

`plan show --json` differs from `=0` only in two op scratch sizes (`ops/12/scratch_bytes` 65552 to 131088, `ops/21/scratch_bytes` 125696 to 658432 bytes).

## Gates

Binaries: `base` = origin/main `70e39e19d` (the branch point, #456 merged), `new` = this branch's code; built by `scripts/copyflipm/build.sh`, **before the rebase onto #459** (`1be12e565`), which touched no `crates/`, `kernels/` or Cargo file (`git diff 70e39e19d origin/main -- crates kernels Cargo.toml Cargo.lock` is empty), so the gated code is the code in this PR. Every gate ran in a fleetq slot on endor (`--load cpu` for builds and tests, `--quiet gpu,cpu` for the model runs). Scripts: `scripts/copyflipm/{build,tests,plan,gates,gates27,serve,mem}.sh`; raw logs in `docs/results/2026-11-18-copyflipm/raw/`.

| Gate | Result | Evidence |
|---|---|---|
| Metal crate tests (`cargo test --release -p intarsia-backend-metal`) | PASS | `raw/tests.log`, `test fails=0` |
| intarsia-core spec tests (`--lib spec`: 52 passed, incl. new `copy_default_applies_only_to_metal_one_sequence_one_rank`, `copy_resolves_unset_to_the_default_only_when_it_applies`) | PASS | `raw/tests.log` |
| New own-process tests `crates/intarsia-core/tests/copy_default_{unset,zero,empty,explicit}.rs`: unset on Metal mb1 tp1 = on (n 20, len 15, verify rows 16); unset at mb 8 / tp 2 / cuda / hip / cpu = off and verify rows 3, no error; `=0` and empty = off on Metal mb1; explicit value wins and is what the refusal sees | PASS (4 binaries, 1 test each) | `raw/tests.log` |
| intarsia-models lib (73) and `--test mtp --test batch_recurrent_load` (66 + 1) | PASS | `raw/tests.log` |
| `cargo test -p intarsia-cli --test docs_env_vars` | PASS | `raw/tests.log` |
| `plan show --json` (a): `--spec mtp --spec-k 2`, unset == base with `INTARSIA_SPEC_COPY=20` | PASS: json identical (sha `ca5d52ba7fc2f435`); differs from base unset (`a3bb42031070215f`) as intended | `raw/plan/`, `raw/plan.log` |
| (b): new `INTARSIA_SPEC_COPY=0` (and empty) == base unset | PASS: byte-identical json (13188 bytes, sha `a3bb42031070215f`), stderr and exit equal | `raw/plan.log` |
| explicit `=20,7` unchanged | PASS (sha `cc3a803ba2f21696` both) | `raw/plan.log` |
| (d) `--spec off` unset, and with explicit `=20`, unchanged | PASS (sha `c61acc4d0bbc1c31` all four) | `raw/plan.log` |
| (e) `--backend cpu` unset, and explicit `=20`, unchanged (stand-in for non-Metal) | PASS (sha `36f5a5d6bbcb6ee2` all four) | `raw/plan.log` |
| (f) `plan show --json` with **no `--spec`** (recipe MTP default), 35B: default == base with `INTARSIA_SPEC_COPY=20`; `=0` == base unset; `INTARSIA_SPEC_DEFAULT=0` unchanged | PASS | `raw/plan.log` |
| (h) same, **Qwen3.8-27B** (`~/models/Qwen3.8-27B-Quark-AWQ-MXFP4`, Metal): default == base `=20`, `=0` == base unset, `--spec off` unchanged | PASS | `raw/plan.log` |
| (c) `--max-batch 8`: `plan show` has no `--max-batch`, so it is served: `serve --max-batch 8 --spec mtp --spec-k 2`, unset, starts like base, startup lines equal base (14 lines, timings and addresses blanked), answers one request (24 tokens), no copy line in the log; also with `INTARSIA_BATCHED_MTP=4` | PASS | `raw/serve/`, `raw/serve.log` |
| served `--max-batch 1`, unset: starts, answers one request, log says "copy drafts on: a chain of up to 15 tokens ... (default on Metal at one sequence; INTARSIA_SPEC_COPY=0 turns it off)" | PASS | `raw/serve/new-mb1.log` |
| Greedy ids, `generate`, copy3k / json-xform / code-gen: `--spec off` = default = `INTARSIA_SPEC_COPY=0` = base (no copy) | PASS: 2902 / 2600 / 500 ids identical across the four arms | `raw/gates.log`, `raw/ids/` |
| The default is really on (len 15) and `=0` / base really off | PASS: "chain of up to 15 tokens ... default on Metal" in every default run; no "copy drafts on" in `=0`, base or `--spec off` runs | `raw/gates.log` |
| **Qwen3.8-27B**, no `--spec` flag (recipe MTP default), greedy ids, copy3k / json-xform (1000 tokens) and code-gen (300): `--spec off` = default = `=0` = base | PASS: 1000 / 1000 / 300 ids identical across the four arms; the default runs say "chain of up to 15 tokens ... default on Metal", the others no chain | `raw/gates27.log`, `raw/ids27/` |
| One sampled request (temperature 0.7, seed 7, json-xform prompt, 600 tokens): default vs `=0` | PASS: ids and text identical; no copy chain pass ran in either | `raw/gates.log`, `raw/ids/gen-sampled-*` |

Sanity tok/s (decode, one run per arm, `generate`; not an A/B). Single-run noise is wider than a few percent: code-gen `=0` ran 199 against base unset 214 although their plans and ids are identical, so read the code-gen row as a tie, and only the copy3k / json-xform gains as real:

| Cell | `--spec off` | `=0` (MTP, no copy) | base unset | default (copy on) |
|---|---|---|---|---|
| copy3k | 120.6 | 211.7 | 209.5 | **445.0** (180 chains, 2680 of 2700 drafted tokens accepted) |
| json-xform | 119.0 | 202.6 | 201.2 | **295.4** (183 chains, 2233 of 2744 accepted) |
| code-gen | 124.6 | 199.3 | 213.8 | 213.7 (no cheaper or dearer than base within noise) |
| sampled json-xform (T 0.7) | - | 162.5 | - | 162.9 |
| 27B copy3k (1000 tokens) | 31.9 | 61.5 | 61.5 | **91.0** (62 chains, 919 of 919 accepted) |
| 27B json-xform (1000 tokens) | 29.8 | 63.0 | 63.3 | **73.0** (67 chains, 833 of 996 accepted) |
| 27B code-gen (300 tokens) | 31.3 | 64.8 | 64.9 | 66.0 |

An earlier draft of `gates.sh` also ran the `plan show` comparison; that part gave 8 FAIL lines because it compared timing-bearing stderr and passed a `--max-batch` flag `plan show` does not take. It was moved to `plan.sh` with those two fixed (rerun above, all PASS) and cut from `gates.sh`; the ids and sampled half of `raw/gates.log` is from the run of that earlier draft, whose ids section is unchanged.

## What is not covered

- Sampled, thinking and real-agent-transcript speed were not measured (see Caveats); sampled output equality is shown for one request, and by the code path above.
- Speed of the Qwen3.8-27B and of any other MTP model on Metal: one tok/s line per arm only (27B), nothing for others.
- CUDA and HIP behaviour is unchanged by construction (plan equal on cpu as a stand-in, the loader sets the default only for Metal); not run on a CUDA box.
- This lane did not run an A/B: the speed claim is #459's.

## Revert

`INTARSIA_SPEC_COPY=0` per process, no rebuild; or revert this PR (the code change is `crates/intarsia-core/src/spec.rs`, `engine.rs`, `crates/intarsia-models/src/load.rs`, `docs/tuning.md`, `docs/decisions/LOG.md`).
