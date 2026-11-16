# Lane mtt: the MTP default's time-to-first-token cost under `serve` on Metal (2026-11-15)

Branch `mtt`, base `origin/main` `c5ce87bc8`, merged up to `930411f58`. Box endor (Apple M5 Max, macOS 27.0); every GPU run in
a fleetq slot (section 7). Scripts: [`scripts/mtt/`](../../scripts/mtt/); raw logs:
[`2026-11-15-mtt-metal-mtp-ttft/raw/`](2026-11-15-mtt-metal-mtp-ttft/raw/). The brief and the coordinator's overrides are kept
at the end of this file.

## Result in short

1. **The MTP default does not cost time to first token.** On an unthrottled box the 35B answers a 4k prompt in 754 ms with
   the MTP default and 746 ms with `--spec off` (median of 6 fresh calls each, #9507; section 3). The 27B and the 34k prompts
   show no cost either. The roadmap's +0.3 to +0.45 s was the box: its slow servers ran right after a 14-minute sweep, and
   its "`--spec off`" arm in #9398 was started without the flag (an MTP server, and it was fast). Nothing was changed in the
   prefill or MTP path.
2. **A hot box slows every GPU buffer by 1.2 to 1.45x for whichever server happens to run**, `--spec off` included
   (section 2). It comes and goes per server instance, and macOS's thermal pressure level does not announce it. The decode
   step time of the same server does: 14.4 to 14.6 ms per MTP pass and 8.2 to 8.5 ms per plain token on the 35B when the box
   is not throttling.
3. **Found on the way and fixed: `serve` start-up.** Admission built the random data of every test case only to read the
   case's label, and then took the result from its cache. With the MTP default that was 7 of the 10 s of a 35B start and 62 of
   the 65 s of a 27B start, on one CPU core, on every start. Cases are now built only when their result is not cached: the
   35B is up in 2.9 s and the 27B in 2.0 to 2.5 s. On by default on Metal; `INTARSIA_METAL_MTP_TTFT_FIX=0` (or
   `INTARSIA_ADMIT_LAZY=0`) restores main's behaviour. No token, admission verdict or cache byte changes.

## 1. Why the roadmap's numbers looked like an MTP cost

* **One arm was mislabelled.** #9398 ran `ttft.sh mtp off mtptrace`. The arm named `off` has no flags (`## off [] []` in
  `scripts/mac-plan/raw/diag.log`) and its token gaps are MTP passes (`15.0, 0.0, 15.1`). So that job ran three MTP-default
  servers in a row: slow (1033 / 1221 / 1188 ms), fast (771 / 738 / 758), slow (1020 / 1205 / 1124). Same binary, same
  flags.
* **In a slow server everything on the GPU is slow**, not only the prefill (the planner's `mtpcb.jsonl` against a trace of
  the same commit on a cool box, `raw/rep2/mtp1cb.jsonl`): prefill buffers 142 against 98 ms, 3-row verify buffers 1.84
  against 1.53 ms, the 3-row prompt tail 4.70 against 3.22 ms. It starts about 300 ms into the first request (100 / 97 / 96 /
  111 / 130 / 151 / 166 ms) and stays.
* **Nothing runs beside the prefill.** The traces hold no other command buffer during a prefill chunk; the head's catch-up
  over the 4095 prompt rows is one 3.2 ms buffer after it. `sample` of both servers during a request shows every host thread
  waiting (`raw/bisect1/*.sample.txt`). `--max-seq 8192` changes nothing. The brief's three hypotheses are ruled out.
* **Cool box, same commit, same script:** #9413 and #9418, 13 servers. MTP default 745 to 813 ms, `--spec off` 742 to
  804 ms (and one call at 974). Every knob of the brief lands in that range (`raw/bisect1.log`: `INTARSIA_METAL_MTP_GPU_ACCEPT=0` 763 to 797,
  `INTARSIA_BATCHED_MTP=0` 782 to 830 beside another lane's build, `--spec-k 1` 753 to 808, `INTARSIA_SPEC_DEFAULT=0` 742
  to 777), so there was nothing to bisect.
* The first token does not wait for the first verify pass: `run_seq` sends it (`Seq::on_token`) before the first
  `advance` (`crates/intarsia-core/src/engine.rs`), and the TTFT difference of 8 ms is less than a pass.

## 2. What does slow a server: a hot box

`scripts/mtt/hot.sh` runs 150 s of 32k prefills first (the TTFT of those requests climbs from 9.6 to 12.6 s on its own, and
`com.apple.system.thermalpressurelevel` goes to 2), then the probe on alternating fresh servers. 35B, 4k prompt, 3 calls per
server; "step" is the median gap between token events of that server (one MTP pass or one plain token).

| run | arm | TTFT ms | step ms | throttled |
|---|---|---|---:|---|
| #9430 | MTP | 888 / 1044 / 1020 | 17.5 | yes |
| | `--spec off` | 777 / 751 / 754 | 8.6 | no |
| | MTP | 782 / 751 / 749 | 14.8 | no |
| | `--spec off` | 813 / 918 / 946 | 10.8 | yes |
| | MTP | 833 / 1025 / 893 | 17.9 | yes |
| | `--spec off` | 772 / 739 / 740 | 8.3 | no |
| #9442 | `--spec off` | 796 / 759 / 765 | 9.1 | a little |
| | MTP | 796 / 975 / 822 | 16.8 | yes |
| | `--spec off` | 801 / 905 / 826 | 9.6 | yes |
| | MTP | 803 / 975 / 963 | 16.7 | yes |
| | `--spec off` | 781 / 756 / 779 | 8.5 | no |
| | MTP | 815 / 768 / 786 | 16.3 | yes |
| | `--spec off` | 809 / 832 / 920 | 9.6 | yes |
| | MTP | 811 / 845 / 874 | 17.5 | yes |
| #9506 (no heating, after other lanes' jobs) | MTP | 785 / 747 / 746 | 14.6 | no |
| | MTP | 787 / 755 / 771 | 15.4 | a little |
| | `--spec off` | 808 / 992 / 1149 | 11.4 | yes |
| | `--spec off` | 808 / 794 / 833 | 9.3 | yes |
| | `--spec off` | 788 / 892 / 951 | 10.2 | yes |
| | `--spec off` | 808 / 759 / 766 | 8.9 | a little |
| | MTP | 792 / 759 / 789 | 14.5 | no |
| | MTP | 782 / 746 / 746 | 14.6 | no |

Both arms are hit, in no order: in #9506 it is the four `--spec off` servers that are slow and the MTP ones that are not. The
thermal level read at the start of each arm was 2 or 1 throughout #9430 and #9442 and 0 for seven of the eight arms of #9506, so
it is no guide. The first call of a throttled server is usually still fast, as in the planner's trace.

What it is not:

* **Not CPU load from other jobs.** #9507, unthrottled, with 12 busy CPU threads beside the calls: `--spec off` 758 against
  746 ms, MTP 762 against 754 ms, step time +3 %. A build beside a probe costs about 10 ms, not 300.
* **Not the start-up work before the first request.** #9453 on a hot box: MTP servers with the start-up fix (3 to 5.6 s to
  load) 832 / 823 / 929, 830 / 907 / 863, 782 / 744 / 766; without it (10 to 25 s) 810 / 766 / 812, 792 / 752 / 751, 787 /
  920 / 878; `--spec off` 833 / 801 / 870, 825 / 836 / 907, 779 / 745 / 739.

This lane cannot say which controller does it (`powermetrics` needs root). It can say that the engine has no part in it:
identical command buffers take 1.2 to 1.45x longer.

For timing work on endor this means: fresh-server A/B probes are only comparable when the arms are interleaved several
times, and a throttled instance has to be recognised and dropped or repeated. `scripts/mtt/ttft.sh` prints the thermal level
and the load time per arm and can wait for level 0 (`COOL=N`); `scripts/mtt/table.py` prints the step time per instance. The
runs up to #9470 also had this lane's and other lanes' builds beside them (`--quiet gpu` lets `--load cpu` jobs in);
the later runs used `--quiet gpu,cpu`.

## 3. Time to first token, MTP default against `--spec off`, before and after

"Before" is `INTARSIA_METAL_MTP_TTFT_FIX=0` (main's start-up), "after" the default. The change only touches start-up, so
before and after differ only by noise; the column that moves is the load time (section 4). Fresh server per arm, arms in the
order after, before, off, off, before, after (35B 4k: with a fourth arm, `--spec off` before), `COOL=90`, `--quiet gpu,cpu`.
Median, with the range in brackets.

| model, prompt | MTP default, before | MTP default, after | `--spec off` | job |
|---|---|---|---|---|
| 35B, 4.1k tokens | 779 (755..792), 6 calls | 746 (746..785), 6 calls | throttled in this run: 808 (759..1149), 6 calls; before: 820 (788..951) | #9506 |
| 35B, 4.1k tokens, unthrottled run | | 754 (745..786), 6 calls | 746 (742..780), 6 calls | #9507 |
| 35B, 32.8k tokens | 8836 (8444..10349), 4 calls | 9310 (8620..10205), 4 calls | 10376 (9688..10913), 4 calls | #9509 |
| 27B, 4.1k tokens | 4524 (4518..4580), 4 calls | 4646 (4527..4923), 4 calls | 4676 (4514..4897), 4 calls | #9508 |
| 27B, 32.8k tokens | 44240 / 46705, 2 calls | 43614 / 46141, 2 calls | 47012 / 61444, 2 calls (the second one throttled: started 53 s after the first) | #9510 |

* **Pre-registered rule** (35B median TTFT at 4k within 40 ms of `--spec off`, at 34k within 150 ms): met, and it was already
  met on main. At 4k the MTP default is 8 ms behind `--spec off` in the unthrottled run (754 against 746) and ahead of it in
  the throttled one. At 34k the MTP arms are ahead of `--spec off` (by 1.1 to 1.5 s), which is the order of the arms: the
  `--spec off` servers ran in the middle, when the box was warmest, and the second call of every server is 0.4 to 1.5 s
  slower than its first. A 34k prefill is 9 to 10 s of full GPU load, enough to heat the box by itself.
* The 27B shows the same thing: at 4k first calls 4514 to 4621 ms in every arm and second calls up to 4923; at 34k one
  call per server, 43.6 to 47.0 s in all arms but for the one `--spec off` server that started without a cool-down (61.4 s).
  27B start-up before the fix at 34k was again 66 to 70 s against 2.0 to 2.1 s.
* The prompt files are the roadmap's (`prompt-4096.txt`, `prompt-32768.txt`); "34k" there is this 32.8k prompt.

## 4. The start-up fix

`plan::admit` checks every kernel the plan selects against the CPU reference on the plan's shapes and caches the results
per binary. Its loop called `case_from_call` (random inputs, quantised random weights) for every planned call before it
looked the result up, because the cache key contains the case's label. `case_label` gives the same label from the shapes
alone. `sample` of a 35B start with the MTP default (#9437, `raw/load/mtp.sample.txt`): 5.7 s of 11.1 s in
`admission::rand_t` (Box-Muller, one thread) and the quantisers, followed by
`admission on plan shapes: 32 kernel(s) checked, 0 case(s) run, 558 cached`. The MTP default adds the verify kernels'
cases, which are the expensive ones; that is why `--spec off` starts suffered less.

The change (`crates/intarsia-core/src/plan.rs`):

* With `PlanOptions::lazy_cases`, a call whose result is cached under its label is answered without building the case, and
  the call is remembered. The first call that has to be built first replays the remembered ones through the generator (data
  dropped), then the loop goes on as before. So every case that runs gets the data it always got, and a run writes the
  cache file main writes.
* Default: on for Metal, off elsewhere (`plan::lazy_cases_for`). `INTARSIA_ADMIT_LAZY=1` / `0` forces it on any backend;
  `INTARSIA_METAL_MTP_TTFT_FIX=0` turns the Metal default off (the switch the coordinator asked for). Both have rows in
  `docs/tuning.md`.
* `AdmitSummary::cases_built` counts the cases built (0 on a start served from the cache).

Time from process start to `loaded ...` (`serve`, `--max-seq 40000`, admission cache warm, quiet box):

| model | flags | before | after | jobs |
|---|---|---:|---:|---|
| 35B | MTP default | 9.7 to 11.4 s | 2.8 to 3.0 s (one start 4.5 s) | #9506, #9507, #9509 |
| 35B | `--spec off` | 5.3 to 5.9 s | 2.3 to 3.6 s (one start 4.3 s) | #9506, #9507 |
| 27B | MTP default | 64.0 to 65.5 s | 2.0 to 2.5 s | #9508 |
| 27B | `--spec off` | 6.4 s (roadmap's `s27off`) | 1.9 to 2.1 s | #9391, #9508 |

Beside other jobs or on a hot box the "before" numbers are worse (35B 25.5 s in #9453, 27B 79.6 s in #9463); "after" stays
at 3 to 5.6 s.

Not changed: the first start of a newly built binary still runs every case, because the cache is keyed by the executable's
size and mtime (35B 15 to 16 s, 27B 88 s). CUDA and HIP run the same eager loop; they are left off here (section 6).

## 5. Gates

| gate | result | where |
|---|---|---|
| 1. Served greedy ids, 256 tokens, fix on == fix off == `--spec off`; short, 4k (two prompts), 16k | **PASS** 35B and 27B, 4 / 4 cases each | #9461, #9463; `raw/gate-ids-*.log`, `raw/ids/` |
| 1. MTP counters (`intarsia_spec_steps / proposed / accepted`) identical, fix on against off | **PASS** 35B and 27B (e.g. 35B 4k 95 passes, 190 proposed, 159 accepted in both) | same |
| 2. `plan show --backend metal` byte-identical to main: 8B, 27B, 35B x default / `--spec off` / `--spec mtp`, each also with `--admit` | **PASS** 18 / 18 (8B `--spec mtp`: the same refusal from both) | #9484; `raw/gate-plan.log`, `raw/plan/` |
| 3. `cargo test --workspace --release --locked` | **PASS** 1495 passed, 0 failed, 33 ignored (merged with main `fd8a3af08`); again after the merge with main `930411f58`: 1509 passed, 0 failed, 33 ignored, and the Metal build | #9486, #9598; `raw/test-workspace.log`, `raw/test-workspace-merged.log` |
| 3. `crates/intarsia-models/tests/batch_metal_real.rs` (35B) | **PASS** both 35B tests (`real_35b_batched_decode_is_bitwise_the_solo_step`, `real_35b_batched_engine_equals_the_local_engine`). The file's LFM2 test reports FAILED only because the run set `INTARSIA_REQUIRE_GPU=1` and there is no LFM2 checkpoint on endor (`SKIPPED: INTARSIA_LFM2_DIR not set`) | #9534; `raw/test-batch-metal-real.log` |
| 4. Regression test on CPU | **PASS** `lazy_admission_cases_build_nothing_when_cached_and_keep_the_data_of_those_that_run` (`crates/intarsia-models/tests/plan.rs`) and `plan::tests::lazy_admission_cases_default_to_metal_and_follow_the_switches` | #9446, #9448 |
| 5. Speed rule, TTFT | **PASS** (section 3; already true on main) | #9506, #9507, #9509 |
| 5. Speed rule, decode, no cell below 0.98 | **PASS** lowest cell 1.03 (table below) | #9511, #9512; `raw/decode-*.log`, `raw/decode/` |

Decode check (localeval `speed --sizes 4096,16384 --gen 256 --reps 2`, MTP default, fresh server per arm, order after,
before, before, after, `--quiet gpu,cpu`; tok/s, median of 2 reps per server):

| model | prompt | before (two servers) | after (two servers) | after / before |
|---|---|---|---|---:|
| 35B | 4k | 147.7, 141.8 | 151.3, 148.7 | 1.04 |
| 35B | 16k | 108.6, 113.5 | 121.2, 124.7 | 1.11 |
| 27B | 4k | 53.2, 52.9 | 54.5, 55.1 | 1.03 |
| 27B | 16k | 46.8, 48.3 | 49.0, 49.6 | 1.04 |

The decode path is untouched, so the 3 to 11 % in favour of "after" is not a decode gain. The "before" servers come out of
10 to 68 s of CPU work and ran back to back; this lane reads the difference as the box's state (section 2). TTFT of the same
runs: 35B 4k 0.70 to 0.75 s in all four servers, 27B 4k 4.50 to 4.90 s.

The regression test runs admission on the tiny model three ways, eager and lazy side by side: cold (same cache file, byte
for byte), warm (eager builds every case, lazy builds none, same counts and kernels), and after a quarter of the cache
entries are removed (both re-measure exactly those, and the files match each other and the cold run's values, which they
only can if the lazy loop replays the skipped draws). Checked by mutation: with the replay removed the test fails (#9449).

The Metal binary of gates 1 and 2 and of the timing runs was built on the lane's base. `origin/main` was merged in twice
afterwards: first a refactor inside `hostprof.rs` for a test (#379), then lanes `dsubset` (#389, opt-in) and `hip-reuse`
(#392, HIP). The workspace tests ran after each merge. MERGEDIDS

The first run of gate 2 (#9462) is void: `cargo test --workspace` rebuilt `target/release/intarsia` without the Metal feature
while it ran. The lane's Metal binary lives in `target/metal/` since.

## 6. Not done, and follow-ups

* **CUDA and HIP still build every case.** The loop is shared and the gain should be the same or larger (slower CPU cores on
  the Sparks). It is off there because this lane did not gate those backends; the override asked for a guard or a proof.
  `INTARSIA_ADMIT_LAZY=1` turns it on for a trial; the check is the served ids and the `admission on plan shapes` line of
  one start each way.
* **A new binary starts cold** (27B 88 s): the cache key is the executable's size and mtime. A key from the kernel sources
  would keep the cache across rebuilds that do not touch kernels. Not done: it changes when cases are re-run, which is a
  decision about admission, not a defect.
* **Brief item (c)**, the `--prefix-store` server with `INTARSIA_MTP_STORE=1`: not measured here; lane `mstore` ran exactly
  those runs tonight.
* The roadmap (`docs/design/2026-11-15-mac-roadmap.md`, item 3 of section 0 and section 1.5) carries a correction note
  pointing here.

## 7. fleetq jobs (endor)

Builds and CPU tests (`--load cpu`): #9411, #9446, #9448, #9449, #9451, #9454, #9482, #9474 (stopped at the missing
`docs/tuning.md` rows, fixed). GPU (`--quiet gpu`): #9413 (knobs), #9418 (ABBA + traces), #9430, #9442, #9453 (hot box),
#9437 (start-up `sample`), #9461, #9463 (ids), #9462 (void), #9484 (plan show), #9486 (workspace tests, in line),
#9470 (ABBA beside other jobs' builds, kept as `raw/busy-abba-35b-4k.log`), #9471 to #9473 (stopped or released by this lane
before they produced data). GPU (`--quiet gpu,cpu`): #9506, #9507, #9508, #9509, #9510 (TTFT), #9511, #9512 (decode check, localeval). GPU (`--quiet gpu`): #9534 (35B batch test).

---

## Brief + progress (kept for the record)

### Progress

- [x] build, reproduce (does not reproduce on a cool box; reproduces on a hot one, in both arms)
- [x] bisect cause (hot box; mislabelled arm in #9398)
- [x] fix: none needed for TTFT; start-up waste in admission fixed, off switch `INTARSIA_METAL_MTP_TTFT_FIX=0`
- [x] gates (ids, plan show, tests, regression test, speed rule)
- [x] 27B; first-token emission checked in code; prefix-store numbers left to lane `mstore`
- [x] LOG row, origin/main merged, PR #393 (not merged), `target/` deleted

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

