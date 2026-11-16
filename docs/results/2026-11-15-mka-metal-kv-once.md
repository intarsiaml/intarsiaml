# mka: read the KV cache once per decode / verify pass on Metal (2026-11-15)

Lane `mka`, branch `mka`, worktree `/opt/code/intarsiaml/.worktrees/mka`, base `origin/main` `c5ce87bc8`.

## 0. Brief + progress (kept here so context compaction cannot lose it)

### Progress

- [x] 1. attribution (`examples/attn_bw.rs`, both shapes, visible 4k / 16k / 34k, M 1 / 3 / 8) -> section 1
      (fleetq #9414 build, #9416 attrib1, #9426 build, #9445-ish attrib2 / attrib3, ce1-base e2e per-dispatch)
- [x] stop-rule check: M = 3 is 2.84x / 2.45x M = 1 -> rule 1 does not fire. BUT the extra row is in-core work, not a DRAM
      re-read (slope per key-row flat from 256 hot keys to 34k cold; rows adjacent in the grid only 4 to 8 % faster).
- PLAN OF RECORD (decided 2026-11-15 ~21:10 EST, after the attribution):
  - Part 1: under the knob the MTP head's short catch-up (t = 2..3 rows, `Model::head_rows_attn`) takes the rows-exact
    kernel (`attn_prefill_bf16_splitkv_rows`) instead of `attn_prefill_bf16_flash_nax` (3.4 to 3.8 ms per launch at 32k
    whatever the row count, 0.42 ms at 4k; the rows kernel 0.4 to 0.65 / 0.05 to 0.1 ms). New `Backend` capability
    (default false; Metal: the knob), a branch in `model.rs` `attn_block`; `mtp.rs` untouched (lane C).
  - Part 2: "packed" stage-1 kernel (`attn_pack_g*`): one threadgroup per (kv head, split) hosts several (row, head group)
    items, each item keeping its own simdgroup streams exactly (same keys, steps, merges) -> bitwise. Launchers: solo
    (`launch_attn_splitkv`), G0b rows (`launch_attn_rows`, the trunk verify on Metal), row table (`rowtab.rs`).
    Evidence it helps: 27B plain HS = 3 (4-simdgroup threadgroups, not bitwise) solo 285 vs 391 us, M = 3 761 vs 959;
    35B row-table idle simdgroups 250 vs 147 us at M = 1.
  - Evaluate with an in-situ-like bench (isolated / interleaved launches): chained launches flatter the kernel by 1.6x.
- [x] part 1 built (head catch-up reroute; `Backend::head_rows_exact`)
- [x] part 2 built; first bitwise run PASS on 5 shapes (fleetq #9477, `raw/test-kvonce1.log`); kbench1 (same job) -> section 3
- POLICY DECIDED 21:25 EST from kbench1: pack only launches whose threadgroup has < 4 simdgroups (the 27B shape,
  `PK_PACK_BELOW = 4`); the 35B stays unpacked (packed it is 0.93x to 1.30x) and gets part 1 only. `scripts/mka/policy.env`
  empty = shipped default. The bitwise test forces every shape packed (`INTARSIA_METAL_ATTN_KVO_BELOW=16`).
- QUEUED 21:30 EST (endor line was ~11 h long; outputs in `raw/q/<name>.out`, fleetq ids in `raw/q/<name>.fleetq`):
  #9519 ksweep, #9536 tests, #9537 smoke (writes `raw/gates/SMOKE_OK`; later jobs skip without it), #9538 kernels,
  #9539 plan, #9540 ids35, #9541 ids27, #9542 served ident, #9543 sp35d, #9544 sp35o, #9545 sp27d, #9546 sp27o, #9547 ce1on.
- 21:45 EST: origin/main (fd8a3af08, docs / scripts / a hostprof test only) merged, binaries rebuilt (#9555), branch pushed,
  DRAFT PR opened. The current tree (accumulator instances + policy, after #9477's run) has compiled but has NOT had its
  GPU bitwise run yet: that is #9536.
- RESUME HERE when the queue has drained (`ls raw/q/*.out`; each ends with `## done`):
  1. `raw/q/tests.out`: 7 kv_once lines must say exit 0 / 5 shapes; existing tests exit 0 at kvo=0 and kvo=1.
  2. `raw/q/smoke.out` (3 x SAME + SMOKE OK), `kernels.out`, `plan.out` (base==off IDENTICAL x 9), `ids35.out`, `ids27.out`
     (every line SAME), `ident.out` (16 of 16, 24 of 24 twice).
  3. `raw/q/sp35d|sp35o|sp27d|sp27o.out` -> speed table (off/on/on/off); `raw/ksweep1/ksweep.txt`; `raw/ce1-on/`
     with `scripts/mka/ce1.py`.
  4. Write sections 4 (gates), 5 (speed), 6 (verdict + default-flip recommendation); LOG row in `docs/decisions/LOG.md`;
     mark the PR ready (do not merge); `unim store`; `rm -rf target`.
- [x] steps 1 to 3 read 2026-11-16 (all queued jobs exit 0) -> sections 4, 5, 6
- 2026-11-16: coordinator: knob stays opt-in, no flip; merge origin/main (f0ecebe2b: #389, #393, #380), rerun build + suites,
  push, undraft #387 if green, do not merge, clean build folders.
- [x] post-merge rerun green (#9635, section 4.1); pushed; #387 marked ready; `target/` removed. LANE DONE.
- [ ] gates 1..7
- [ ] LOG row, merge origin/main, PR (do not merge)

### Coordinator overrides (these win over the brief)

- Worktree `/opt/code/intarsiaml/.worktrees/mka`, branch `mka` (not `lane-mka/metal-kv-once`). Only that folder; delete
  its `target/` when done.
- Bitwise is required. Ships opt-in (`INTARSIA_METAL_ATTN_KV_ONCE=1`). If all gates pass (ids identity on both models at
  4k and 34k, MTP on and off, B = 1 and multi-stream), say so; the coordinator may flip the default. Do not flip it here.
  With the knob unset `plan show --json` must be byte-identical.
- Attribute first; follow the brief's stop rule. If KV re-reads are not the cost, write that up and stop.
- Parallel lanes `mnr` (Brief A) and `mtt` (Brief C) run on endor. `mnr` also appends to
  `crates/intarsia-backend-metal/src/kernels.rs` and its pinned kernel list: keep that edit minimal, on conflict keep
  both. `git fetch` and merge origin/main before opening the PR.
- PR against main, do not merge.
- Fleet: every GPU run on endor through `fleetq run --host endor --expect <MIN> --mem <GB> --quiet gpu --label "mka: ..."`;
  builds / CPU tests `--load cpu`; each job <= 32 GB; never stop `endor-kev` / `endor-tensorfold`; never `--force`,
  `--no-queue`, `--exclusive`; never release others' jobs; acquire / release for several steps, never leave a slot held.
- No tool call or wait over 4 minutes. macOS has no `timeout`: `perl -e 'alarm N; exec @ARGV'`.
- Speed numbers from ABBA runs with fresh processes. Never print secrets.
- unanimis MCP down -> `unim store`, source "intarsiaml lane mka", never project / scope.
- Commits end with `Co-Authored-By: Claude Fable 5.2 <noreply@anthropic.com>`; PR body ends with the Claude Code line.
- Report: attribution table, what was built, gate table, speed ABBA (35B and 27B, 4k and 34k, MTP default and plain),
  default-flip recommendation, PR number, fleetq job ids, leftovers.

### The brief (verbatim from `docs/design/2026-11-15-mac-roadmap.md` section 4, lane `mac-plan`)

### Brief B: lane `mka`, read the KV cache once per decode / verify pass (Metal, both hybrid models)

**Goal.** Make decode attention on Metal cost about one read of the visible K / V rows per layer and pass, for one row
and for 2..16 verify / batch rows, without changing any output bit. Ship opt-in behind `INTARSIA_METAL_ATTN_KV_ONCE=1`
first; default-on is the coordinator's call after the gates.

**Why.** Measured on endor today (`docs/design/2026-11-15-mac-roadmap.md` sections 1.2 to 1.4; served, localeval, filler
prompts):

| | 4k | 34k | extra per pass at 34k | KV bytes at 34k / 550 GB/s |
|---|---:|---:|---:|---:|
| 35B plain | 117.7 tok/s | 91.0 | +2.5 ms | 0.68 GB = 1.2 ms |
| 35B MTP k2 (default) | 129.3 | 87.3 | about +9 ms (3 rows) | 1.2 ms |
| 27B plain | 30.6 | 24.1 | +8.8 ms | 2.2 GB = 4.0 ms |
| 27B MTP k2 (default) | 47.0 | 35.1 | about +17.5 ms (3 rows) | 4.0 ms |

The 3-row verify pass pays about three times the one-row attention cost: each row re-reads the cache. On the 35B
(G = 8 query heads per kv head, head_dim 256) the one-row kernel also splits the query heads over `head_split` = 4
threadgroups per kv head, each reading the same K / V slice (`crates/intarsia-backend-metal/src/attn.rs` `head_split`;
`kernels/metal/attention_splitkv.metal` header, "Head split"). Lane mprof measured the batched kernel at 36 % of the KV
read roofline. Expected: 35B MTP at 34k 87 -> about 120 tok/s, 27B MTP at 34k 35 -> about 43, plain +12 to +18 % at 34k,
about +5 % at 4k, and the same saving on every batched step.

**What exists.** `kernels/metal/attention_splitkv.metal`: stage 1 `attn_decode_bf16_splitkv_g{1,2,4,8}` (one
threadgroup per (kv head, split); a lane loads one 16-byte chunk of a K / V row; KPS = 256 / D keys per simdgroup step,
i.e. one key at D = 256; slots merged with `simd_shuffle_xor`, simdgroups through threadgroup memory), stage 2
`attn_decode_bf16_combine` (partials combined in split order). `attn_rows_splitkv_g*` + `attn_rows_combine` serve the
`rows_exact` multi-row path, `attn_decode_rowtab_g*` the batched row-table path (`src/rowtab.rs`). Split policy:
`decode_splits` / `ask_nsplit` (`MIN_KEYS` 128, `TARGET_TGS` 120, `MAX_SPLIT` 64, chunk rounded up to 32 keys);
`INTARSIA_METAL_SPLITKV`, `INTARSIA_METAL_ATTN_HS`, `INTARSIA_METAL_ATTN_HSB` are diagnostic overrides.

**Design.** First measure, then build; the mechanism is yours to choose inside the bitwise rule.

1. Attribution (first slot): `examples/attn_bw.rs` at the two real shapes (35B: N 16, Nkv 2, D 256; 27B: N 24, Nkv 4,
   D 256), visible 4k / 16k / 34k, M = 1 / 3 / 8: time and effective GB/s against unique K + V bytes; split it into
   re-reads (rows x head split) and per-read rate (load-issue-bound at 16 bytes per lane and one key per step?). Write
   the table into the results doc before changing a kernel.
2. Share the read across rows. Verify rows of one sequence see keys 0 ..= pos0 + i, so they share every K / V tile; one
   threadgroup per (kv head, split) should load each tile once and apply it to all rows' query heads. Register pressure
   is the known limit (GP = 8 at D = 256 already spills, hence the head split), so stage the K / V tile in threadgroup
   memory (16 keys x 256 x 2 B = 8 KB per operand) and let R x G / GP simdgroup sets read it from there, or loop the
   rows over a tile held in registers. Rows whose split count or chunk differs from the first row's (it depends on the
   row's visible length) fall back to today's launches for that pass.
3. Remove the 35B's four-fold head-split read the same way (the four head groups of a kv head share one staged tile).
4. Only if 1 shows the single read itself is far below 550 GB/s: wider loads per step for D = 256.

**Bitwise rule.** Each (row, query head) keeps today's arithmetic exactly: the same keys per slot and simdgroup, the same
online-softmax stream per slot, the same `simd_shuffle_xor` and threadgroup merge order, the same split boundaries and
combine order. Only who loads which bytes, and how many rows a threadgroup serves, may change. The header already states
"each query head's arithmetic is independent of the others', so HS changes no bit at the same simdgroup count": keep the
simdgroup count per head group. If a faster form needs another reduction order, stop and report it as a
numerics-changing option with its measured gain; do not ship it in this lane.

**Files.** `kernels/metal/attention_splitkv.metal`, `crates/intarsia-backend-metal/src/attn.rs`, `src/rowtab.rs`,
`src/kernels.rs` (appended entries; lane A appends there too: two-line conflict, keep both), `examples/attn_bw.rs`,
attention tests under `crates/intarsia-backend-metal/tests/`. Do not edit `mxfp4.*` (lane A) or
`crates/intarsia-core/src/model/mtp.rs`, `spec_pipe.rs`, `engine.rs` (lane C). `kernels/metal/dflash.metal`'s windowed
decode (`attn_decode_bf16_window`) reuses the decode shaders over a window: keep it bitwise too (its test is
`tests/dflash2.rs`).

**Pre-registered rules.**

* Kernel: at visible 34k, M = 3 costs <= 1.3x M = 1 (today about 3x), and M = 1 reaches >= 70 % of unique K + V bytes /
  550 GB/s on both shapes (today about 40 %).
* End to end, same binary, knob off against on, served, ABBA, localeval `speed --sizes 4096,16384,32768 --gen 256`:
  35B default (MTP k2) decode at 34k >= 1.25x and at 4k >= 0.98x; 27B default at 34k >= 1.12x; `--spec off` no cell
  below 0.98.

**Gates.**

1. Bitwise kernel tests: every row of the new multi-row launch equals the one-row launch (M 2..16, visible lengths
   around every split boundary of `decode_splits` between 1 and 40000, both shapes plus the 8B's G = 4, D = 128, cache
   offsets); the new one-row launch equals today's one-row launch (same list); a planted fault (one tile skipped) is
   caught. Existing `rows_exact.rs`, `tests/dflash2.rs` and the row-table tests pass unchanged.
2. `intarsia kernels test --backend metal`: existing rows identical, 0 FAIL.
3. `intarsia plan show --backend metal` byte-identical to main with the knob unset: 8B, 27B, 35B x default /
   `--spec off` / `--spec mtp`.
4. Greedy ids, 256 tokens, knob on == knob off == main: 35B and 27B, default and `--spec off`, short / 4k (two prompts)
   / 16k / 32k (`docs/results/raw/2026-10-15-metal-longctx-profile/prompts/`); served `--max-batch 8` on the 35B with 4
   and 8 streams == solo; Qwen3-8B short and 4k.
5. `crates/intarsia-models/tests/batch_metal_real.rs` (35B) passes.
6. `cargo test --workspace --release --locked`.
7. The speed rules.

**Stop rules.** (1) If the attribution shows re-reads are not the cost (M = 3 already <= 1.5x M = 1 in the microbench),
stop and report where the +9 ms per verify pass at 34k goes (then it is the engine's verify path, not the kernel).
(2) If the shared-tile kernel is bitwise but slower than today at 4k on either model and cannot be gated by visible
length cleanly, ship it for visible >= the crossover only and say so. (3) Two kernel forms miss the kernel rule: stop,
write up.

**Deliverables.** Branch `lane-mka/metal-kv-once` from `origin/main`, worktree `/opt/code/intarsiaml/.worktrees/mka`
(yours alone; delete `target/` when done). Results doc `docs/results/2026-11-15-mka-metal-kv-once.md`, raw logs under
`docs/results/2026-11-15-mka-metal-kv-once/raw/`, scripts under `scripts/mka/`. A row in `docs/decisions/LOG.md`. PR
against main; report the PR number, the rule table and the fleetq job ids.

**Rules.** GPU and timing work on endor goes through fleetq: `fleetq run --host endor --expect <MIN> --mem <GB> --quiet
gpu --label "mka: ..." -- <cmd>`; builds and CPU tests use `--load cpu`. Never `--force`, `--no-queue` or `--exclusive`,
never release another owner's job, keep memory within 32 GB, do not stop `endor-kev` or `endor-tensorfold`. No tool call
or wait over 4 minutes: start long runs with `nohup ... &` inside the fleetq command and poll. macOS has no `timeout`: use
`perl -e 'alarm N; exec @ARGV'`. Commits end with `Co-Authored-By: Claude Fable 5.2 <noreply@anthropic.com>`; the PR body
ends with `🤖 Generated with [Claude Code](https://claude.com/claude-code)`. Never print secrets.

When you change code that can be run, built, or type-checked, run a real check that exercises the change before reporting it done: the project's tests, type-checker, or build, or the changed command itself. A syntax-only check, or a check command that failed to start, does not count; if all that is missing is the project's declared dependencies, install them with its own package manager and lockfile (e.g. npm install, pip install -r requirements.txt), never via sudo or the system package manager, unless told not to. Only if no real check can run here, say which one you did not run and why instead of reporting the change as done.

---

## 1. Attribution (before any kernel change)

Microbench `crates/intarsia-backend-metal/examples/attn_bw.rs` (extended in this lane: `rowtab` = the row-table kernel with
M rows on the SAME cache at positions `ctx - M .. ctx - 1`, which is the launch an MTP verify pass makes
(`crates/intarsia-core/src/model/verify_attn.rs`); `rowtabd` = M rows on M caches, a batched step; `rows` = the G0b kernel
with the rows adjacent in the grid; `--hot` = one cache for every launch). One K / V cache per layer, GPU time from the
command-buffer trace, median of 20 reps. Raw: `raw/attrib1/attrib.txt` (fleetq #9416), script `scripts/mka/attrib.sh`.
"floor" = unique K + V bytes of the launch / 550 GB/s.

### 1.1 Today's launches, cold caches (us per launch)

| shape | visible | floor | solo 1 row (`splitkv`) | row-table M = 1 | M = 3 (verify) | M = 8 | M = 3 / solo | M = 8 / solo |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 35B 16 / 2 / 256 | 4096 | 15.3 | 41.1 | 32.1 | 55.7 | 111.6 | 1.36 | 2.72 |
| 35B | 16384 | 61.0 | 75.3 (81 %) | 104.7 | 203.1 | 440.5 | 2.70 | 5.85 |
| 35B | 34000 | 126.6 | 147.4 (86 %) | 250.4 | 418.1 | 939.5 | 2.84 | 6.37 |
| 27B 24 / 4 / 256 | 4096 | 30.5 | 55.8 | 56.6 | 119.1 | 256.4 | 2.13 | 4.59 |
| 27B | 16384 | 122.0 | 194.9 (63 %) | 196.7 | 447.3 | 1007.5 | 2.29 | 5.17 |
| 27B | 34000 | 253.2 | 390.8 (65 %) | 391.8 | 958.6 | 2287.4 | 2.45 | 5.85 |

Stop rule 1 (M = 3 <= 1.5x M = 1) does not fire: 2.84x and 2.45x at 34k. The lane continues.

What the table already says against the brief's premise:

* **The one-row launch is not far from the roofline on the 35B**: 86 % at 34k, 81 % at 16k, with the four-fold head split
  in place. The head-group "re-read" does not go to DRAM (HS = 2 gives 93 %, HS = 8 57 %, HS = 1, no re-read but spilled
  registers, 26 %). The 27B is at 65 %.
* **The row-table kernel at one row is 1.7x slower than the solo kernel on the 35B at 34k** (250 against 147 us) although it
  does the same arithmetic: it launches threadgroups of 8 simdgroups of which a long row uses 4 (the surplus idle to the
  barrier). The verify pass pays that on every row.
* **An extra verify row costs the same per key at every length, hot or cold**: 35B (M = 8 minus M = 1) / 7 rows / keys =
  2.8 ns (4k cold), 2.3 ns (4k hot), 2.9 ns (16k), 2.9 ns (34k); 27B 6.9 / 7.6 / 7.1 / 8.0 ns. On the 35B at 34k that is
  99 us per extra row, BELOW the DRAM floor of one read (127 us): the extra row's bytes do come from the caches. So the
  extra row is in-core work (load issue and ALU), not a DRAM re-read. Part 2 below tests that directly.

### 1.2 Other arms of the first run (34k unless said; us per launch)

| arm | 35B | 27B |
|---|---:|---:|
| solo, default head plan (35B HS 4, 27B bitwise HSB 3) | 147.4 | 390.8 |
| solo, no head split (GP 8, spilled) | 490.3 | 1209.0 |
| solo, 2 head groups (GP 4) | 135.8 | 536.1 |
| solo, GP 1 (35B HS 8, 27B HSB 6) | 220.9 | 549.0 |
| solo, device position | 148.2 | 375.6 |
| one hot cache: solo / M = 3 / M = 8 | 157.0 / 441.5 / 1055.5 | 407.1 / 1016.1 / 2742.5 |
| M rows on M caches (batched): M = 1 / 3 / 8, 4 layers | 416.9 / 1092.6 / 3824.4 | 491.7 / 1362.8 / 3603.5 |
| the same as % of M x floor | 30 / 35 / 26 % | 51 / 56 / 56 % |

### 1.3 Is the extra row memory or in-core work? (`raw/attrib2/attrib.txt`, `scripts/mka/attrib2.sh`)

| test | 35B | 27B | reading |
|---|---|---|---|
| slope per key and extra row, row table on ONE hot cache, 256 / 1024 / 4096 keys | 3.4 / 3.4 / 2.7 ns | - / 6.6 / 7.1 ns | the same as at 34k cold (2.9 / 8.0 ns): no DRAM in it |
| rows adjacent in the grid (G0b kernel) against the row table, 34k, M = 3 / 8 | 402 / 880 vs 419 / 940 us | 876 / 2010 vs 908 / 2180 us | sharing "through the caches" already works; adjacency buys 4 to 8 % |
| plain head split, 4-simdgroup threadgroups (27B `HS=3`: 120 threadgroups of 4 simdgroups instead of 360 of 2; NOT bitwise, another key partition), 34k solo / M = 3 / M = 8 | | 285 / 761 / 1816 vs 391 / 959 / 2287 us | the threadgroup shape is worth 21 to 27 % on the 27B |
| 35B `HS=2` (GP 4, 30 splits; not bitwise), 34k solo / M = 3 / M = 8 | 133 / 332 / 747 vs 147 / 418 / 940 us | | 21 % per extra row |
| row table with no idle simdgroups (35B, split target 60 -> 8 simdgroups per row), 34k M = 1 / 3 | 147 / 395 vs 250 / 418 us | | the idle half costs 100 us at one row, 6 % at three |

**The extra verify row is ALU work, not a re-read.** One key costs a lane about 100 scalar operations for two query heads
(8-wide dot, a 5-level `simd_shuffle_xor` sum, the online-softmax bookkeeping with two `exp2`, the 8-wide accumulate, the
bf16 unpack); 34k keys x 16 heads x 32 lanes is 0.77 G lane-operations per row on the 35B, and the measured 99 us per
extra row is 7.7 T operations/s: the GPU's ALU rate (8.07 TFMA/s, lane dm). Reading the K / V bytes once instead of M
times cannot make a row cheaper than that. What a bitwise kernel can still change is how well the threadgroups fill the
cores.

### 1.4 What the engine launches, and what it costs there (`raw/ce1-base/`, fleetq job in the progress list, `scripts/mka/ce1.sh`, `ce1.py`)

`generate`, 97 tokens, `INTARSIA_METAL_PROFILE=cb` + `INTARSIA_METAL_COMMIT_EVERY=1` (one command buffer per dispatch).
Per verify pass (default, MTP k2) or per token (`--spec off`), GPU busy time:

| model, prompt | mode | busy ms | decode attention: kernel, launches, us each | of the pass | MTP head catch-up attention |
|---|---|---:|---|---:|---|
| 35B 4k | off | 8.11 | `attn_decode_bf16_splitkv` 10 x 44 | 5.5 % | |
| 35B 32k | off | 10.38 | 10 x 249 | 24.0 % | |
| 35B 4k | MTP | 17.44 | `attn_prefill_bf16_splitkv_rows` (3 rows) 10 x 104 | 6.0 % | `attn_prefill_bf16_flash_nax` 0.8 x 582 us = 0.47 ms |
| 35B 32k | MTP | 23.97 | 10 x 653 | 27.2 % | 0.8 x 3824 us = **3.18 ms (13.3 %)** |
| 27B 4k | off | 33.46 | 16 x 107 | 5.1 % | |
| 27B 32k | off | 40.97 | 16 x 600 | 23.4 % | |
| 27B 4k | MTP | 45.14 | 16 x 184 | 6.5 % | 0.8 x 478 us = 0.40 ms |
| 27B 32k | MTP | 66.82 | 16 x 1274 | 30.5 % | 0.9 x 3778 us = **3.28 ms (4.9 %)** |

Three things the brief did not know:

1. **The verify pass on Metal does not launch the row table.** It launches the G0b kernel (`launch_attn_rows`, rows
   adjacent in grid x), because `Backend::verify_attn_per_row` is false on Metal. The kernel to change is that one.
2. **The MTP head's catch-up runs the PREFILL attention kernel on 2 or 3 rows.** After a pass that accepted a draft the head
   catches up on 2 or 3 rows (`Model::mtp_run_pending`); that call is an ordinary `AttnPrefill`, the registry gives it
   `attn_prefill_bf16_flash_nax`, and that kernel costs the same for 2 rows as for a chunk (microbench, `raw/attrib3/`: 35B
   3.77 / 3.78 / 3.80 ms for 2 / 3 / 8 rows at 32k, 0.43 to 0.45 ms at 4k; 27B 3.37 ms and 0.42 ms). The rows kernel does 3
   rows at 32k in 0.39 to 0.65 ms. This is 2.7 ms of the 35B's "+9 ms per verify pass at 34k" and it is not attention over
   the trunk's cache at all. (`NAX_FLASH_MIN_M = 2` is deliberate for the TRUNK, so that the chunking of a prompt never
   changes a row's kernel; the head's rows only feed drafts.)
3. **A launch costs up to 1.6x more in the engine than chained in the microbench** (35B 3 rows at 32k: 653 us in the
   engine, 386 us chained, 625 us as one launch per command buffer; 27B 1274 / 850 / 1137). Chained attention launches
   overlap each other; in a decode step the neighbours are projections that are themselves bandwidth-bound. The rule
   table below therefore quotes the microbench with a 32 MB copy between the launches (`--gap 32`) next to the chained
   number.

Where the 35B's +9 ms per pass at 34k goes (trace, 4k -> 32k): verify rows attention +5.5 ms, head catch-up through the
prefill kernel +2.7 ms, the head's one-row attention +0.2 ms, host gaps +1.7 ms; the other kernels get slightly faster.
27B: +17.5 ms, +2.9 ms, +0.5 ms.

### 1.5 What follows for the brief's expectations

* The kernel rule "M = 3 <= 1.3x M = 1" asks for extra rows at 15 % of a row. The rows are ALU-bound, so no bitwise
  form reaches it: the arithmetic per (row, head, key) is fixed by the bitwise rule. Stop rule 3 territory; section 3
  reports the numbers of the form built here.
* The end-to-end rule (35B default at 34k >= 1.25x) needs 5.6 ms out of a 28 ms pass. Attention is 6.5 ms of it in the
  engine; a bitwise kernel cannot remove 85 % of that. What is available bitwise: the head catch-up (2.5 ms per pass on
  both models) and the threadgroup shape (20 to 27 % of the attention time).

---

## 2. What was built (all behind `INTARSIA_METAL_ATTN_KV_ONCE=1`, default off)

### 2.1 The MTP head's short catch-up takes the verify rows kernel

`Backend::head_rows_exact() -> usize` (new, default 0; Metal: 8 with the knob). In `Model::attn_block`
(`crates/intarsia-core/src/model.rs`), a pass that `mtp.rs` already marks as a short head catch-up
(`head_rows_attn`: 2 to `VERIFY_MAX_ROWS + 1` rows, host positions) and that fits the backend's bound runs its attention
as ONE `AttnPrefill` call with `exact_rows` set, so the registry gives it the rows-exact kernel
(`attn_prefill_bf16_splitkv_rows`, bitwise the decode step per row) instead of `attn_prefill_bf16_flash_nax`.
`mtp.rs`, `spec_pipe.rs` and `engine.rs` are untouched. The trunk's kernels do not change: a prompt's tail chunk of 2..8
rows still takes the flash kernel (chunking invariance), and the captured-step path (device positions) is as before.
What changes in the numbers: the head's catch-up rows (its hidden state for the next draft) are computed by a
decode-class kernel instead of the NAX prefill kernel, so a draft can differ; every emitted token is still the trunk's
verified token. Greedy ids are identical by construction and by gate 4.

### 2.2 Packed threadgroups for the decode attention's stage 1

New shaders `attn_pack_g{1,2,4,8}` (rows of one sequence: the 1-row decode launch and the G0b verify rows) and
`attn_pack_rowtab_g{1,2,4,8}` (row table), in `kernels/metal/attention_splitkv.metal`; launchers
`launch_attn_splitkv`, `launch_attn_rows` (`attn.rs`) and `launch_attn_rowtab` (`rowtab.rs`) pick them through
`attn::pack_plan`. No registry entry is added (the pinned kernel list is unchanged; `kernels.rs` gains two lines in
`shader_names`). The combine stages are the existing ones.

An ITEM is one row's head group on one K / V split: `nsg` simdgroup streams. Today every item is a threadgroup. Packed, a
threadgroup of `W * IPT` simdgroups holds the items of the same kv head and split: `W` (the window) is the largest `nsg` a
row of the launch can have, a row with `nsg = W / sub` puts `sub` items in a window, `IPT` windows fill a threadgroup up
to 8 simdgroups (`INTARSIA_METAL_ATTN_KVO_SG` to sweep), bounded by 16 simdgroups and 30 KB of threadgroup memory.

| launch | today | packed |
|---|---|---|
| 35B solo, long (nsg 4, HS 4) | 120 threadgroups x 4 simdgroups | 60 x 8 (2 head groups each) |
| 35B verify, 3 rows | 360 x 4 | 180 x 8 |
| 35B row table (batched; window 8, long rows use 4) | 120 R x 8, half idle | 60 R x 8 full (+ 60 R that return at once) |
| 27B solo (nsg 2, bitwise HS 3) | 360 x 2 | 120 x 6 (the kv head's 3 head groups) |
| 27B verify, 3 rows | 1080 x 2 | 360 x 6 |
| 8B (HS 1, nsg 4 or 8) | as today's shape | unchanged shape (1 item per threadgroup) |

Bitwise argument: an item keeps its split count, chunk, `nsg`, key-to-(stream, step, unit) assignment, loop body
(`ask_step`), slot merge, stream merge order and combine. Only the threadgroup its simdgroups live in and the index of
its streams in threadgroup memory change.

One ALU trim rides along (`trim`, bit 27 of the packed plan, `INTARSIA_METAL_ATTN_KVO_TRIM=0` to switch it off): a step
that does not raise a stream's running maximum has `corr = exp2(0) = 1`, so `l * 1 + psum` and `acc * 1` are `l + psum`
and `acc` bit for bit, and the 9 multiplies and the `exp2` are skipped. At head_dim 256 a simdgroup holds one slot, so the
branch is uniform. It is only in the packed kernels (`ask_step`'s new argument defaults to false).


---

## 3. Kernel numbers (microbench, fresh process per arm, off / on / on / off; `raw/kbench1/kbench.txt`, fleetq #9477)

Every shape packed (target 8 simdgroups, trim on), before the policy was set. us per launch, mean of the two runs per arm.
"gap32" = a 32 MB copy between the launches (the in-engine-like number), "chained" = attention launches back to back.

| shape, visible | launch | chained off -> on | gap32 off -> on | on / off (gap32) |
|---|---|---:|---:|---:|
| 27B 34k | solo 1 row | 370 -> 226 | 403 -> 244 | **0.60** |
| 27B 34k | verify rows M = 3 (G0b) | 880 -> 689 | 994 -> 766 | **0.77** |
| 27B 34k | verify rows M = 8 | 2021 -> 1654 | 2315 -> 1869 | 0.81 |
| 27B 34k | row table M = 3 / 8 | 913 -> 742 / 2293 -> 1764 | 1015 -> 788 / 2496 -> 1903 | 0.78 / 0.76 |
| 27B 16k | solo / M = 3 | 186 -> 117 / 426 -> 328 | 193 -> 116 / 469 -> 359 | 0.60 / 0.77 |
| 27B 4k | solo / M = 3 | 56 -> 57 / 117 -> 99 | 45 -> 27 / 113 -> 97 | 0.60 / 0.86 |
| 35B 34k | solo 1 row | 150 -> 159 | 169 -> 181 | 1.07 |
| 35B 34k | verify rows M = 3 (G0b) | 407 -> 389 | 518 -> 483 | 0.93 |
| 35B 34k | row table M = 3 / 8 | 418 -> 530 / 937 -> 1016 | 516 -> 669 / 1249 -> 1303 | 1.30 / 1.04 |
| 35B 4k | solo / M = 3 (G0b) | 36 -> 73 / 50 -> 85 | (below the copy's noise) / 41 -> 35 | |

Reading:

* **27B: the packed launch is 1.65x faster at one row and 1.3x at three**, at every length. The solo launch at 34k
  streams its 139 MB of K + V in 226 us chained = 616 GB/s, the machine's peak (the bench's old constant was 614): the
  2-simdgroup threadgroups of the bitwise head split were what held it at 65 %.
* **35B: packing does not pay** (its threadgroups already have 4 simdgroups): solo 7 % slower, the verify rows 7 % faster,
  the row table up to 30 % slower. The shipped policy therefore packs only launches whose own threadgroup has fewer than
  4 simdgroups; the 35B's launches are byte-for-byte today's code path under the knob.
* The brief's kernel rule (M = 3 <= 1.3x M = 1) is not met and cannot be (section 1.3): 27B packed 766 / 244 = 3.1x.
  Its second half (M = 1 >= 70 % of the roofline at 34k) holds: 27B 112 % of 550 GB/s chained, 35B 86 % (unchanged).

The threadgroup-size sweep (`raw/ksweep1/ksweep.txt`, fleetq #9519, 34k, 32 MB gap) agrees: 27B solo 407 us unpacked,
403 at 2 windows per threadgroup, 243 at 3 (the shipped layout), 243 to 246 at 4 and 6; its verify rows 995 -> 803 to 833
at 1 to 3 windows, worse again at 4 and 6. 35B solo 181 unpacked, 175 at 1 window, 282 to 285 at 3 and more. The trim
changes nothing measurable (243.1 vs 242.7 us).

---

## 4. Gates (pre-merge tree 8750c8955 against origin/main c5ce87bc8; all jobs exit 0)

| gate | job | result |
|---|---|---|
| 1. packed vs unpacked bit-identical: 5 shapes x (1-row, G0b verify rows, row table, one-cache table), host and device position, guards; planted fault seen 5 / 5 per launch kind | #9536 | PASS in 7 layouts: every shape packed, 1 / 2 / 3 windows per threadgroup, trim off, largest accumulator instance, shipped policy |
| existing attention tests (`hs_bitwise`, `g0b_attn`, `row_tables`, `rows_exact`, `dflash2`, `attention`, `multirow`), knob 0 and 1 | #9536 | PASS, all |
| 2. `kernels test --backend metal`, knob off and on (+ 35B / 27B `attn_decode`, 35B `attn_prefill` shapes) | #9538 | 55 kernels, 1276 cases, 0 FAIL in both; reports differ only in the run time line |
| 3. `plan show --json` knob unset vs origin/main: 8B, 27B, 35B x default / `--spec off` / `--spec mtp` | #9539 | byte-identical, 9 of 9 (8B `--spec mtp` is refused identically: no MTP head) |
| `plan show --json` knob set vs unset | #9539 | identical where MTP is off; with MTP one field differs: the `attn_prefill` scratch (`scratch_bytes` 3170304 -> 8454144 on the 35B), because the head catch-up now needs the rows kernel's workspace |
| 4. greedy ids, 256 tokens, 35B: short / 4k / 4k (second text) / 16k / 32k x MTP default / `--spec off`; main, knob unset, knob set | #9537, #9540 | SAME, 10 of 10 |
| greedy ids, 27B, same grid (main arm on short and 4k) | #9537, #9541 | SAME, 10 of 10 |
| served, 35B `--max-batch 8`: 8 prompts (400 and 2500 tokens) at 1 / 4 / 8 streams | #9542 | every stream equals its solo run in all three arms (16 of 16); main == unset and unset == set, 24 of 24 cells |

With the knob set the ids are identical to unset everywhere tested. The one visible difference is in the MTP draft
statistics of one cell (35B 4k: 99 passes / 155 accepted unset, 98 / 156 set): the head's catch-up rows come from another
kernel, so a draft can differ, as section 2.1 says; the emitted tokens are the same.

Not covered: 27B multi-stream (the 27B has no batched path gated here), sampled requests.

### 4.1 Rerun after merging origin/main f0ecebe2b (#389, #393, #380); fleetq #9635, `raw/q2/`

| check | result |
|---|---|
| `kv_once` in the 7 layouts; `hs_bitwise`, `g0b_attn`, `row_tables`, `rows_exact`, `dflash2`, `attention`, `multirow` at knob 0 and 1 | PASS, all |
| `plan show --json` knob unset vs the new main, 8B / 27B / 35B x default / off / mtp | byte-identical, 9 of 9 |
| greedy ids main / unset / set: 35B 4k and 32k MTP, 27B 4k MTP, 27B short plain | SAME, 4 of 4 |
| `kernels test`, knob off and on | 55 kernels, 1276 cases, 0 FAIL in both |

## 5. Speed

### 5.1 End to end (`localeval speed`, fresh server per arm, order off / on / on / off, 2 reps, 256 tokens; #9543 to #9546)

Decode tok/s, the two arms of each setting and their mean:

| model, mode | prompt | off | on | on / off |
|---|---:|---|---|---:|
| 35B MTP default | 4k | 129.4, 141.9 (135.7) | 154.5, 134.5 (144.5) | 1.07 |
| 35B MTP default | 32k | 87.2, 90.3 (88.8) | 98.6, 95.4 (97.0) | 1.09 |
| 35B `--spec off` | 4k | 112.3, 115.6 (114.0) | 111.3, 98.0 (104.7) | 0.92 |
| 35B `--spec off` | 32k | 93.5, 93.8 (93.7) | 84.0, 69.8 (76.9) | 0.82 |
| 27B MTP default | 4k | 51.5, 43.7 (47.6) | 47.0, 46.7 (46.9) | 0.98 |
| 27B MTP default | 32k | 38.6, 39.2 (38.9) | 39.0, 38.9 (39.0) | 1.00 |
| 27B `--spec off` | 4k | 29.8, 28.0 (28.9) | 30.1, 30.1 (30.1) | 1.04 |
| 27B `--spec off` | 32k | 24.5, 24.2 (24.4) | 26.4, 24.5 (25.5) | 1.05 |

**These runs do not establish a speedup.** The 35B `--spec off` rows are a null test: under the knob the 35B's plain decode
runs exactly today's launches (its shape is not packed and there is no head catch-up), yet the arms differ by 8 and 18 %.
The box was shared with other lanes' CPU jobs during the window, and the arm-to-arm spread inside one setting reaches
15 %. So the noise of this run is at least as large as every effect in the table. Prefill and TTFT are unchanged within
the same noise (35B 32k 11.0 to 12.0 s, 27B 51.5 to 54.6 s in all arms).

### 5.2 In the engine, per dispatch (32k prompt, MTP default, one command buffer per dispatch; `raw/ce1-base`, `raw/ce1-on` #9547)

| | 35B off | 35B on | 27B off | 27B on |
|---|---:|---:|---:|---:|
| verify rows attention, us per launch | 653 | 662 | 1274 | 1054 |
| solo attention (MTP head's one-row step), us per launch | 246 | 249 | 535 | 332 |
| head catch-up through `attn_prefill_bf16_flash_nax`, ms per pass | 3.18 | 0 (rows kernel instead, about 0.5) | 3.28 | 0 (about 0.9) |
| GPU busy per verify pass, ms (on: scaled to true passes) | 24.0 | 21.5 | 66.8 | 60.8 |

One run per arm. The kernels do what the microbench says (27B verify rows 0.83x, solo 0.62x; the flash launch is gone), and
GPU busy time per pass drops 9 to 10 % on both models at 32k. That this does not show as tok/s on the 27B in 5.1 is not
explained here; a quiet-box rerun of 5.1 would settle it.

## 6. Verdict

* Brief's stop rules: rule 1 did not fire; the premise behind the build (KV re-reads) is wrong (section 1), and the
  kernel rule "M = 3 <= 1.3x M = 1" is out of reach for any bitwise kernel. The end-to-end rule (35B default at 34k
  >= 1.25x) is **not met**: 1.09x measured, inside the run's noise.
* What ships, opt-in: bitwise packed threadgroups (27B shape only) and the MTP head catch-up on the rows kernel (both
  models). All correctness gates pass; knob unset is byte-identical to main.
* **Default flip: not recommended yet.** Correctness would allow it (ids identical in every cell, plan identical with the
  knob unset), but the speed evidence is a microbench and one trace per arm; the end-to-end ABBA is inconclusive. Needed
  first: sections 5.1 rerun on a box with no other jobs (`--quiet gpu,cpu`), 3 reps.
* Follow-ups the attribution points at (not built): the verify rows are ALU-bound, so the next step is fewer ALU
  operations per key (a non-bitwise change), not memory; the row table's idle simdgroups on the 35B batched path (250 vs
  147 us at one row) are still there.
