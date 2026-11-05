# m35 Metal tail peel with the busy chunk as its period (2026-11-04)

Question: the first attempt (`2026-11-04-m35-metal-peel.md`, branch `m35/metal-peel`) fixed the Qwen3.6-35B's load-dependent output flip on Metal with `prefill_peel = 8`, but `peel_tail` cuts at multiples of 256 and the 35B's busy chunk is 2048, so it also changed the bits of 120 of every 4096 prompt lengths where solo and busy already agreed (8 were exposed). Its section 8 option (a): make the period a recipe value. Does a peel with period 2048 fix the 8 exposed lengths and move nothing else?

**Answer.** Branch `m35/peel-period` adds the recipe key `prefill_peel_period` (default 256) and sets `prefill_peel = 8`, `prefill_peel_period = 2048` under `[tuning.metal]` of `recipes/qwen3_5_moe.toml`. **Gates 1 to 5 pass**: solo equals behind-a-decoding-row on 7 of 7 exposed prompts (main: 0 of 7), the busy schedule's bits are unchanged (peel behind equals main behind, 7 of 7), and all 19 other lengths tested are bit-equal to main, solo and behind, including the 8 lengths that failed in the first attempt (258, 260, 265, 516, 772, 1028, 1540, 4356). Plan text and JSON are byte-identical to main for the 27B, Qwen3-8B, Llama-3.1-8B and Qwen2.5-7B; the 35B differs only in the peel lines. Speed: controls unchanged to the 0.01 s print resolution in the second round (the first round's 4096 control moved +0.06 s with the arm order, see section 5), the exposed lengths pay one extra short pass, +0.02 to +0.04 s TTFT. Gate 6 passes; gate 7 is a measurement (section 5).

* What changes for users: on the 35B on Metal, a prompt whose length mod 4096 is in 2050..2057 (8 of 4096 lengths, 0.2 %) is now computed on an idle server with the bits it already had behind a decoding neighbour (peel solo equals main behind on 7 of 7). Everything else is bitwise what main computes.
* The period is an explicit recipe key, not derived from `busy_chunk`. The 27B (HIP peel 16, Metal peel 8) and every other recipe keep period 256 and print nothing new.
* Exposure set: the unit sweep (section 4) shows that at the forward-call level the new peel changes the pass schedule exactly when the rows mod 4096 are in 2049..2056 (serving prompt lengths 2050..2057 because the body is p - 1 rows), for every length from 1 to 20000, and never changes the busy (2048) schedule. The measured lengths (section 1) agree with it; they are 26 lengths, not a census.

Raw data: `2026-11-04-m35-peel-period/raw/`. Scripts: `raw/scripts/` (they are the first attempt's, with this tree's paths, plus `gates-ids.py`, which applies gates 1 and 2 and checks the trace). Binaries (built from this tree with `--features metal`): `intarsia.main` (origin/main `ba8658736`, sha256 c8d488d7...) and `intarsia.peel` (this branch's code, sha256 9bbfcb68..., built before the `docs/tuning.md` fix and the `mtp.rs` test, neither of which is compiled into it; the workspace test run #8473 covered the head's code).

## 1. The change

* `crates/intarsia-core/src/scheduler.rs`: `peel_tail(n, peel, period)` (was `(n, peel)`): a pass of more than `period` rows whose length mod `period` is in 1..=`peel` runs its first `n - n % period` rows first. `parse_peel_period` and the env knob `INTARSIA_PREFILL_PEEL_PERIOD` (at least 1; unset: the recipe).
* `crates/intarsia-core/src/recipe.rs`: `prefill_peel_period` in `[tuning]` and `[tuning.<backend>]`, `Recipe::resolve_prefill_peel_period` (backend, base, default 256 = `DEFAULT_PEEL_PERIOD`). Validation: period at least 1, and the peel below its period, checked on the effective pair (a backend table that sets only one of the two is checked against the base value of the other). The old rule (peel below 256) is the same rule at the default period.
* `crates/intarsia-core/src/model.rs`: `Model::prefill_peel_period` resolved at load (`INTARSIA_PREFILL_PEEL_PERIOD`, then the recipe) and checked against the peel in use (a knob can break the recipe's pair, so the load fails if the peel is not below the period). The only caller of `peel_tail` is `Model::forward_chunks` (`model.rs:2565`), which every forward pass goes through: the serving scheduler hands the model its chunks and the model cuts them, as does the single-pass path of the accuracy tool. `model/batch.rs` copies the period into the decode twin with the peel. `scheduler.rs` itself never called `peel_tail`.
* `min_rows` (`model.rs:750`): the condition `prefill_peel == 0` is unchanged and still right. A peel of any period can end a pass in a 1-row tail the prefix-reuse schedule does not show, so any peel above 0 keeps the "no wide reuse" branch; the period does not enter. (The 35B is recurrent and returns before that branch anyway.)
* `plan.rs` and `cmd/plan.rs`: `prefill_peel_period` in the JSON, and in the text a peel line that names the period, **only when the period is not 256**. `prefix_store.rs`: the key suffix is `|peel8` as before for period 256 and `|peel8|period2048` otherwise, so the 35B on Metal has a suffix no period-256 peel has and the suffix of every other model is unchanged. **The full key string of every model does change**: `describe_build` embeds the recipe's `Debug` dump, which now spells the new field (`prefill_peel_period: None`). That is harmless in practice, because the key also carries the running executable's size and modification time (`describe_build`, `exe`), so any rebuild already starts a new key space. A byte-identical full string would need a hand-written `Debug` for `Tuning` and `BackendTuning` that skips the field when unset; I did not do that, the coordinator may.
* `recipes/qwen3_5_moe.toml`: `[tuning.metal] prefill_peel = 8`, `prefill_peel_period = 2048`, with the comment. `docs/tuning.md` (the `INTARSIA_PREFILL_PEEL` row, a new `INTARSIA_PREFILL_PEEL_PERIOD` row, which the env-var docs test requires) and `docs/recipes.md`.
* Tests: `tests/scheduler.rs` (`peel_tail` at both periods: n = 2049..2056 cut to 2048, 2057 and 2058 not, 6145 cut to 6144, 1..=2048 never cut, 258/260/265 cut at 256 and not at 2048; the pass sequences of n = 6145 at chunks 4096 and 2048; a sweep over len 1..=20000: the busy schedule is never changed, the solo schedule changes exactly when len mod 4096 is in 2049..2056, and the solo small passes equal the busy schedule's), `crates/intarsia-models/src/recipe.rs` (the test that pins which recipes set a peel now expects the 35B on Metal with period 2048, every other recipe with period 256, and the validation), `crates/intarsia-models/tests/mtp.rs` (`prefill_peel_period_reaches_the_model_and_the_prefix_store_key`: a CPU model loaded with peel and period, its prefix-store key suffix, and the refusal of a peel not below its period).

## 2. Gates, fixed before any data (`raw/`)

| gate | result |
|---|---|
| 1. exposed lengths 2050, 2052, 2055, 2056, 2057, 6146, 6153: 256 greedy ids, solo == behind a decoding row | **PASS, 7 of 7** (main: 0 of 7, first id differs at token 23, 25, 70, 147, 24, 46, 10). Overlap confirmed per row in the server trace: the behind request's first chunk was planned with `decoding: 1` and its schedule is the busy one (`[2048, 2..9]`, `[2048, 2048, 2048, 2]`); the background row was a 1200-token generation started 1 s before (the first attempt's 300-token row ended too early) (`raw/ids35/gates-1-2.txt`) |
| 2. no other length moves: 258, 260, 265, 516, 772, 1028, 1540, 4356, 2058, 2064, 2300, 6154, 48, 233, 266, 4099, 4101, 4105, 5770, 256 ids equal to main, solo and behind | **PASS, 19 of 19** (38 of 38 comparisons; overlap confirmed on every behind row) |
| 3. `plan show` text and `--json` against main | **PASS**: 27B, Qwen3-8B, Llama-3.1-8B, Qwen2.5-7B byte-identical, 8 of 8; the 35B differs by one text line (`prefill tail peel 8 rows, period 2048 (...)`) and two JSON lines (`"prefill_peel": 8`, `"prefill_peel_period": 2048`) (`raw/plan/`, `raw/gates-run.log`) |
| 4. `kernels test --backend metal` | **PASS**: 49 kernels, 980 cases, 0 FAIL; the diff of main's and the branch's logs is the wall-time line (`raw/kernels/`) |
| 5. MTP on the branch, 35B, p = 2050, 2057, 6146: `--spec mtp --spec-k 2` == `--spec off`, default == off | **PASS, 3 of 3** (the head ran: acceptance 71.8 %, 80.9 %, 71.8 % at 2050, 2057, 6146) (`raw/mtp/`) |
| 6. `cargo test --workspace --no-fail-fast` | **PASS**: 1300 passed, 0 failed, 29 ignored (debug profile, job #8473, on the code and test state of the head; `raw/cargo-test-workspace.txt`). The first run (#8462, `raw/cargo-test-workspace-first-run.txt`) had 1 failed, the env-var docs test, fixed before this run |
| 7. speed: controls unchanged, TTFT cost at exposed lengths | controls: round 2 within 0.005 s on all four (PASS); round 1 failed the strict criterion at the 4096 control (+0.060 s), attributed to arm order, see section 5. Exposed cost: +0.02 to +0.04 s |

How the other columns of the first attempt's table look now (`raw/ids35/exposed-vs-main.txt`): on the 7 exposed prompts, peel behind equals main behind (7 of 7, the busy bits are unchanged) and peel solo equals main behind (7 of 7), peel solo differs from main solo (7 of 7, they move onto the busy bits). On the other 19 lengths peel solo equals main solo and peel behind equals main behind (19 of 19).

## 3. Environment overrides

`plan show` of the 35B on Metal with the branch binary (`raw/envcheck/`, `raw/env-check.log`, job #8474): `INTARSIA_PREFILL_PEEL_PERIOD=256` prints the old line `prefill tail peel 8 rows (... multiple of 256 ...)` and no period line (the first attempt's cut); `INTARSIA_PREFILL_PEEL_PERIOD=8` fails the load (`prefill tail peel 8 rows must be below its period 8`, exit 1); `=0` fails (`a row count of at least 1`); `INTARSIA_PREFILL_PEEL=0` prints no peel line; `INTARSIA_PREFILL_PEEL=16` prints `prefill tail peel 16 rows, period 2048 (...)`. Greedy ids with a period override were not run.

## 4. Unit tests of the rule (`crates/intarsia-core/tests/scheduler.rs`)

`peel_tail(n, 8, 2048)`: n = 1..=2048 unchanged; n = 2049..=2056 cut to 2048; n = 2057, 2058, 2064, 2300, 4095 unchanged; 4096 unchanged; 4097 and 4104 cut to 4096; 4105 unchanged; 6145 cut to 6144. At period 256 the same n = 258, 260, 264, 516, 1028, 1540, 4360 are cut to a multiple of 256 and at 2048 they are not. Pass sequences at chunk 4096: 6145 rows run `(0, 4096), (4096, 2048), (6144, 1)` (without the peel `(0, 4096), (4096, 2049)`); at chunk 2048: `(0, 2048), (2048, 2048), (4096, 2048), (6144, 1)`. Over every length 1..=20000: the chunk-2048 schedule is the same with and without the peel, the chunk-4096 schedule changes exactly when len mod 4096 is in 2049..=2056, and the passes of 8 rows or fewer of the peeled chunk-4096 schedule are the chunk-2048 schedule's.

## 5. Speed (`raw/speed-round1/`, `raw/speed-round2/`)

`localeval speed`, single stream, `serve --max-batch 1 --spec off`, `--gen 128 --reps 5`, one quiet slot (`--quiet gpu,cpu`), each rep classed by its own prompt length (the nonce moves it by a few tokens). TTFT is printed to 0.01 s.

**Round 2** (order peel, main, main, peel; sizes of the first attempt; `raw/speed35b-table.txt`):

| size | class | prompt tokens | reps main / branch | TTFT median main -> branch | min | per-arm medians |
|---|---|---|---|---|---|---|
| 2099 | exposed | 2052..2057 | 10 / 10 | 0.380 -> 0.415 s (+0.035 s, +9.2 %) | 0.38 -> 0.39 | peel1 0.430, main2 0.380, main3 0.380, peel4 0.400 |
| 2105 | exposed | 2051..2057 | 9 / 9 | 0.380 -> 0.400 s (+0.020 s, +5.3 %) | 0.38 -> 0.39 | 0.400, 0.380, 0.380, 0.405 |
| 6148 | exposed | 6148..6153 | 10 / 10 | 1.180 -> 1.220 s (+0.040 s, +3.4 %) | 1.12 -> 1.18 | 1.310, 1.120, 1.200, 1.210 |
| 1063 | control | 1029..1037 | 9 / 10 | 0.220 -> 0.220 s (+0.000 s) | 0.22 -> 0.22 | 0.220 in all four |
| 1067 | control | 1027..1034 | 9 / 10 | 0.220 -> 0.220 s (+0.000 s) | 0.22 -> 0.22 | 0.220 in all four |
| 2100 | control | 2059..2065 | 10 / 10 | 0.380 -> 0.385 s (+0.005 s) | 0.38 -> 0.38 | 0.390, 0.380, 0.380, 0.380 |
| 4096 | control | 4084..4091 | 10 / 10 | 0.715 -> 0.710 s (-0.005 s) | 0.71 -> 0.70 | 0.710, 0.710, 0.720, 0.730 |

The 1063 and 1067 sizes are the first attempt's "peeled, not exposed" cells, now controls: they ran +0.02 to +0.03 s slower with the period-256 peel and are unchanged now, as they must be (the same passes).

**Round 1** (order main, peel, main, peel; `raw/speed-round1-cmp.txt`): the scout found no size that lands in 2050..2057, so only 6148 (exposed: 1.320 -> 1.360 s, +0.040 s, +3.0 %) and the controls 2100 (0.380 -> 0.380 s) and 4096 ran. **The 4096 control read 0.725 -> 0.785 s (+0.060 s)** although the passes are identical (the prompts are 4085..4092 rows, below every cut): TTFT rose with arm order inside this round (main-3 was slower than main-1 as well; every arm's best rep was 0.70 to 0.71 s), which is why round 2 reversed the order and ran the exposed sizes of the first attempt. I report the round 1 control as what it was, a failure of the "controls within 0.01 s" criterion, and the evidence that it is drift, not the peel: the same cell in round 2 reads -0.005 s, and no code path differs for those prompts (section 4).

Round 2 ran the peel arm first and the first arm reads high even on a control (2100: 0.390 against 0.380), so the pooled exposed medians carry some first-arm bias; the fourth arm (peel, last) alone gives +0.020 s at 2099 (0.400 against 0.380) and +0.025 s at 2105 (0.405 against 0.380), the lower end of the cost. The cost is one more short pass, as in the first attempt: +20 to +40 ms TTFT at 1k-2k and 6k tokens on the 8 of 4096 lengths that are exposed, nothing elsewhere. Decode is unchanged within noise (122.6 vs 121.2 tok/s at 2100, round 1; drift again). The 6148 cell is noisy (arm medians 1.12 to 1.31 s), so +0.04 s is the median difference, not a resolved size.

## 6. Not established

* Accuracy against a reference (not a gate; section 7 of the first attempt already says neither side is shown to be closer, and this branch moves the same eight lengths onto the same busy bits as that one did).
* The exposure set beyond the 26 measured lengths and the unit sweep: no census of all 4096 residues on the real model. Lengths above 6154 and `--max-batch` above 2 were not run.
* The 35B on CUDA and HIP: not run, recipes not changed (their `prefill_peel` stays unset). The 27B was checked by `plan show` only (byte-identical) and by its unchanged recipe, not by a new greedy run.
* A 35B behind-a-neighbour run at a busy chunk other than 2048 (`--busy-chunk`): the period is a recipe constant, so a user who changes `busy_chunk` should change `prefill_peel_period` with it; nothing enforces that.
* Qwen3-8B and the other models of the first attempt's section 3: not re-run; their recipes and plans are unchanged.
* MiniCPM-V: does not load here, as in the first attempt.
* `INTARSIA_PREFILL_PEEL` still rejects 256 or more, while a recipe may now set a peel up to its period minus 1 (an inconsistency I left alone).

## 7. Jobs

Owner `claude-299136c4`, `--host endor`, no `--force`, no `--no-queue`. Builds #8454 (main and branch), #8458 and #8461 (branch again; the first copy of the target directory carried newer mtimes than my edits and gave a binary without the recipe change, whose gates #8456 and #8459 I released and discarded); gates 1 to 5 #8463; speed round 1 #8467, round 2 #8468; tests: #8462 (first workspace run: 1298 passed, 1 failed, `every_env_var_in_the_code_has_a_row_in_docs_tuning_md` because the new env row cited a name instead of file:line; fixed, #8464 passes), #8471 and #8472 (new load/key test), env override check #8474, final workspace run #8473. #8457, #8460 and #8470 (a workspace run cut short because I then added the `mtp.rs` test) were released before finishing.
