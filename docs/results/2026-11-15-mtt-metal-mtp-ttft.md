# Lane mtt: the MTP default's time-to-first-token cost under `serve` on Metal (2026-11-15)

## Brief + progress (kept at the top: context compaction loses briefs)

### Progress

- [ ] build, reproduce
- [ ] bisect cause
- [ ] fix + off knob `INTARSIA_METAL_MTP_TTFT_FIX=0`
- [ ] gates (ids, plan show, tests, regression test, speed rule)
- [ ] 27B, first-token emission, prefix-store numbers (if time)
- [ ] LOG row, merge origin/main, PR (do not merge), delete `target/`

### Working notes (newest last; rewrite into the report at the end)

- 20:33 #9411 build ok. #9413 bisect1 (cool box): MTP default TTFT 745..813 ms == `--spec off` 742..804. **Does not reproduce.**
- 20:37 #9418 rep2 ABBA + cb (cool box): off 744..776, mtp 752..813; prefill buffers 96..100 ms in both arms.
- Planner's #9398 "off" arm had NO `--spec off` flag (`ttft.sh mtp off mtptrace` -> `## off [] []`, MTP gap pattern): it was an MTP-default
  server and it was FAST (738..771). So in one job identical MTP servers went slow, fast, slow: a per-instance state, not MTP vs off.
- In the planner's slow instances EVERY GPU buffer is 1.2..1.45x slower (verify rows 1.84 vs 1.53 ms, 3-row tail 4.70 vs 3.22), not just prefill.
- No host thread spins under MTP (`sample`, bisect1/mtpsmp.sample.txt). No other fleetq job overlapped the planner's window (fleetq log).
- 20:41 #9430 hot1: 160 s of 32k prefills first (TTFT of those climbs 9.55 -> 12.6 s; `thermalpressurelevel` 2), then alternating arms:
  mtp1 888/1044/1020 SLOW, off1 777/751/754, mtp2 782/751/749 FAST, **off2 813/918/946 SLOW (gaps 10.7 -> 13.7 ms)**, mtp3 833/1025/893 SLOW, off3 fast.
  => thermal throttling of the GPU on a hot box; hits `--spec off` too. Open: is MTP hit more often (longer, CPU-heavy load: 10.4 s vs 4.3 s)?
- next: sample the MTP start-up (#load), controlled hot-box test of "CPU burn right before the request".

### Coordinator overrides (these win over the brief)

- **Worktree and branch:** `/opt/code/intarsiaml/.worktrees/mtt`, branch `mtt` from origin/main (not the brief's branch name).
  Use only that folder locally; delete its `target/` when done.
- **Bitwise is required:** no token may change.
  - If the cause is a defect (wasted work, a serialisation bug), fix it on by default with `INTARSIA_METAL_MTP_TTFT_FIX=0` as the off switch.
  - Gates: ids identical on and off, `plan show` identical where applicable, decode no cell below 0.98.
  - If the fix trades decode speed for TTFT, make it opt-in.
- **Backend scope:** the fix may live in backend-neutral files (`model/mtp.rs`, `spec_pipe.rs`, `engine.rs`, `engine/batch.rs`).
  If shared code paths change, CUDA behaviour must stay identical: guard the change to Metal, or prove CUDA unaffected
  (CPU tests plus a CUDA ids check on a Spark via fleetq: `--host spark1` or `spark2`, vLLM stays up, `--mem-budget 40`, no `--takes-down`).
- **Parallel lanes:** `mnr` and `mka` run Metal kernel work on endor at the same time; files differ. `git fetch` and merge origin/main before the PR.
- **PR:** against main, do not merge; the coordinator merges on green gates.
- **Fleet rules (strict):** every GPU run on endor through fleetq: `fleetq run --host endor --expect <honest MIN> --mem <GB> --quiet gpu --label "mtt: ..." -- <cmd>`
  (guide `/opt/code/fleetq/README.md`). Builds and CPU tests `--load cpu`. Each job <= 32 GB (`endor-kev`, `endor-tensorfold` hold about 50 GB; do not stop them).
  Never `--force`, `--no-queue`, `--exclusive`; never release others' jobs. Several steps: `acquire` / `release`, never leave a slot held.
  No tool call or wait over 4 minutes (Bash timeout <= 240000), poll at most every 4 min. macOS has no `timeout`: `perl -e 'alarm N; exec @ARGV'`.
  Speed and TTFT numbers come from ABBA runs with fresh servers. Never print secrets.
  unanimis MCP down -> `unim store` CLI, source "intarsiaml lane mtt", never pass project or scope.
- **Finish:** commits end with `Co-Authored-By: Claude Fable 5.2 <noreply@anthropic.com>`; PR body ends with the Claude Code line.
  Report: root cause with evidence; the fix; gate table; TTFT ABBA table (35B and 27B at 4k and 34k, `serve`, MTP on and off, before and after);
  decode check; PR number, fleetq job ids, leftovers.

### Brief C (verbatim from `docs/design/2026-11-15-mac-roadmap.md` section 4, branch `mac-plan`)

### Brief C: lane `mtt`, the MTP default's time-to-first-token cost under `serve` (Metal)

**Goal.** Find and remove the extra 0.3 to 0.45 s of TTFT that the MTP default adds to a 4k prompt on the 35B under
`intarsia serve` on Metal, without changing any output token. Then check the 27B and the first request after load.

**Evidence (all on main `c5ce87bc8`, endor, 2026-11-15; scripts in `scripts/mac-plan/`, logs in
`scripts/mac-plan/raw/`).**

* `scripts/mac-plan/ttft.sh` (one `serve` per arm, `/v1/completions`, `prompt-4096.txt` with a fresh nonce, 3 calls):
  default (MTP k2) TTFT 1033 / 1221 / 1188 ms, `--spec off` 771 / 738 / 758 ms (fleetq #9398; repeated in #9401:
  962 / 1074 / 1048 against 768 / 728 / 748).
* localeval `speed` (#9391): 35B TTFT 1.15 s against 0.71 s at 4k (prefill 3551 against 5781 tok/s), 12.29 against
  11.51 s at 34k; 27B 5.97 against 5.47 s at 4k (one window).
* Command-buffer trace (`INTARSIA_METAL_PROFILE=cb:<file>`, `raw/ttft/mtpcb.jsonl`, `offcb.jsonl`; print with
  `scripts/mac-plan/cbgap.py <file> 8`): a 4096-row prefill chunk is seven 128-dispatch buffers. `--spec off`: 96 to
  107 ms each on every request. MTP default: first request 100 / 97 / 96 / 111 / 130 / 151 / 166 ms; second and third
  requests 124 to 154 ms each. These are GPU start-to-end times.
* `intarsia generate` with the MTP default: 96 to 101 ms per buffer, prefill 907 ms against 881 ms off
  (`raw/p35mtp/`, `raw/p35/`). So the plain prefill path is fine and the cost is specific to the serve path with MTP on.
* The extra is about 0.3 s at 4k and 0.8 s at 34k: not proportional to the prompt length.

**Hypotheses, in the order to test them** (none verified):

1. GPU work submitted beside the prefill: the MTP head's catch-up over queued prompt rows
   (`crates/intarsia-core/src/model/mtp.rs`: `mtp_queue`, `mtp_catch_up`, `mtp_catch_up_rank`), run in small pieces whose
   command buffers overlap the prefill's, or the pipelined draft of `model/spec_pipe.rs` (`pipe_draft`, "drafted ahead")
   still running from the previous request.
2. A host thread that takes memory bandwidth during prefill under MTP (a readback, a zeroing or copy of the head's KV or
   workspace sized by `--max-seq`, the hidden-row queue). Check with `sample <pid> 1 -file ...` during a prefill and by
   repeating the probe at `--max-seq 8192`.
3. A different prefill schedule under MTP in the serve engine (`crates/intarsia-core/src/engine.rs`
   `Seq::prefill_step`; peel and tail passes): compare the dispatch count and ops per buffer of the two traces (`nd`, `t`
   fields).

**Method.** One `serve` per arm, the `ttft.sh` probe, the cb trace, and `INTARSIA_METAL_PROFILE=kernel:<file>` or
`INTARSIA_METAL_COMMIT_EVERY=1` with `cb` to see which ops share the GPU with the prefill buffers. Bisect with existing
knobs before reading code: `INTARSIA_METAL_MTP_GPU_ACCEPT=0`, `INTARSIA_BATCHED_MTP=0`, `--spec mtp --spec-k 1`,
`INTARSIA_SPEC_DEFAULT=0` (must reproduce the `--spec off` arm).

**Fix.** Whatever the cause, the fix must keep drafts and output bits: the head sees the same hidden rows in the same
order. Likely shapes: run the head's prompt catch-up as one grouped pass after the prefill chunk (prefill kernels at
M = chunk rows) instead of beside it; or defer it until after the first token is emitted, so TTFT is the prefill plus one
plain step; or remove a host-side copy. If the fix is a clear defect repair, make it the default with an off knob
`INTARSIA_METAL_MTP_TTFT_FIX=0` that restores main's behaviour for A/B; if it changes scheduling in a way that could move
decode speed, ship it opt-in (`=1`) and say so.

**Then, in the same lane, if time remains:** (a) the 27B (`~/models/Qwen3.8-27B-Quark-AWQ-MXFP4`): same probe, same fix
or its own cause; (b) confirm that under MTP the first token is streamed as soon as the prefill's logits exist and does
not wait for the first verify pass (if it waits, emitting it first is worth about 20 ms on the 35B and 50 ms on the 27B);
(c) measure a `--prefix-store` server with `INTARSIA_MTP_STORE=1` on Metal against the store alone (TTFT and decode on a
repeated 16k prefix): numbers only, no default change (it is idea 8 of the roadmap).

**Files.** `crates/intarsia-core/src/model/mtp.rs`, `model/spec_pipe.rs`, `engine.rs`, `engine/batch.rs`, possibly
`crates/intarsia-server` request handling. Do not edit `crates/intarsia-backend-metal/src/mxfp4.rs`, `kernels.rs`,
`attn.rs`, `rowtab.rs` or anything under `kernels/metal/` (lanes A and B).

**Pre-registered rule.** Same binary, fix off against on, `ttft.sh`-style probe with 6 fresh calls per arm, arms
interleaved A B B A: 35B median TTFT at 4k within 40 ms of the `--spec off` arm's (today +300 to +450 ms); at 34k within
150 ms. Decode tok/s (localeval `speed --sizes 4096,16384 --gen 256`, natural-v1 where the suite offers it) no cell below
0.98.

**Gates.**

1. Greedy ids, 256 tokens, fix on == fix off == `--spec off`: 35B and 27B, short / 4k (two prompts) / 16k, served; the
   acceptance line (`spec: verify passes=... accepted=...` on the server's stderr) identical between fix on and off for
   the same prompt (drafts unchanged).
2. `intarsia plan show --backend metal` byte-identical to main: 8B, 27B, 35B x default / `--spec off` / `--spec mtp`.
3. `cargo test --workspace --release --locked`; `crates/intarsia-models/tests/batch_metal_real.rs` (35B).
4. A regression test where CPU can show it (e.g. the number or order of head catch-up calls for a prompt of N rows);
   if only Metal timing shows it, say so and add the probe script to the results doc instead.
5. The speed rule.

**Stop rules.** (1) If the first two hours of bisecting do not isolate a cause, write up the traces and the knobs that
do and do not move it, and stop. (2) If the cause is on CUDA / HIP too and the fix touches shared scheduling that this
lane cannot test there, ship it Metal-only behind `Backend`-level gating or opt-in and flag it. (3) If the fix costs
decode speed (a cell below 0.98 twice), report both numbers and leave it opt-in.

**Deliverables.** Branch `lane-mtt/metal-mtp-ttft` from `origin/main`, worktree `/opt/code/intarsiaml/.worktrees/mtt`
(yours alone; delete `target/` when done). Results doc `docs/results/2026-11-15-mtt-metal-mtp-ttft.md`, raw logs under
`docs/results/2026-11-15-mtt-metal-mtp-ttft/raw/`, scripts under `scripts/mtt/` (copy `scripts/mac-plan/ttft.py`,
`ttft.sh`, `cbgap.py` from branch `mac-plan`, PR linked in the roadmap, if they are not on main yet). A row in
`docs/decisions/LOG.md`. PR against main; report the cause in one paragraph, the before / after TTFT table, the PR number
and the fleetq job ids.

**Rules.** GPU and timing work on endor goes through fleetq: `fleetq run --host endor --expect <MIN> --mem <GB> --quiet
gpu --label "mtt: ..." -- <cmd>`; builds and CPU tests use `--load cpu`. Never `--force`, `--no-queue` or `--exclusive`,
never release another owner's job, keep memory within 32 GB, do not stop `endor-kev` or `endor-tensorfold`. No tool call
or wait over 4 minutes: start long runs with `nohup ... &` inside the fleetq command and poll. macOS has no `timeout`: use
`perl -e 'alarm N; exec @ARGV'`. Commits end with `Co-Authored-By: Claude Fable 5.2 <noreply@anthropic.com>`; the PR body
ends with `🤖 Generated with [Claude Code](https://claude.com/claude-code)`. Never print secrets.

When you change code that can be run, built, or type-checked, run a real check that exercises the change before reporting it done: the project's tests, type-checker, or build, or the changed command itself. A syntax-only check, or a check command that failed to start, does not count; if all that is missing is the project's declared dependencies, install them with its own package manager and lockfile (e.g. npm install, pip install -r requirements.txt), never via sudo or the system package manager, unless told not to. Only if no real check can run here, say which one you did not run and why instead of reporting the change as done.

