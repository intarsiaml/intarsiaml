# Lane copyflipc: CUDA copy drafts default-on (single-sequence MTP, 35B recipe only; user decision #9)

## Brief

(Coordinator brief to lane copyflipc, 2026-11-20, written here verbatim so it survives compaction.)

Lane `copyflipc`: make CUDA copy drafts (`INTARSIA_SPEC_COPY`) default-on for the 35B recipe on the single-sequence MTP engine, mirroring the Metal flip. This is user decision #9, approved. Merge after the gates pass and CI is green.

Repo intarsiaml. Your worktree is `/opt/code/intarsiaml/.worktrees/copyflipc` (branch `copyflipc`, from origin/main 87c8bb19a). Put all scratch and build dirs inside it. Do not create other work folders and do not delete this one. Your first act: write this brief to `docs/results/2026-11-20-copyflipc.md` as a "Brief" section, and add progress notes as you go.

#### Context
- Metal flip: PR #460, lane copyflipm. Read `docs/results/2026-11-18-copyflipm.md` and the `INTARSIA_SPEC_COPY` row in `docs/tuning.md`. Code: `crates/intarsia-core/src/spec.rs` (`copy_default_applies`, `set_copy_default`, `copy_from_env`, `COPY_LEN_DEFAULT` / `COPY_LEN_METAL`), `crates/intarsia-core/src/recipe.rs` / `crates/intarsia-models/src/recipe.rs` (`[tuning.metal] copy_default = true` in `recipes/qwen3_5_moe.toml`), and `crates/intarsia-models/src/load.rs`. Tests: `crates/intarsia-core/tests/copy_default_*.rs`.
- CUDA evidence: `docs/results/2026-11-18-copycuda.md` (35B NVFP4, spark, `20` vs off). code-gen 0.992x (the trade the user accepted), agent-batch 1.041x, agent-edit 1.09x, copy3k 1.69x, json 1.51x. Use the same `n` and `len` that copycuda measured (check the doc; the CUDA default len is `COPY_LEN_DEFAULT` = 7).
- Production serves `--max-batch 2`, where copy is off anyway. The default only acts at `--max-batch 1` (and in `generate`) with `--spec mtp`, tp 1.

#### Do
1. Extend the default to CUDA for recipes that opt in:
   - Add a `[tuning.cuda] copy_default = true` key, or generalise the existing key to whatever is cleanest. Opt in the 35B recipe only (`recipes/qwen3_5_moe.toml`). The 27B and other recipes stay off.
   - Unset = on for those configs. `INTARSIA_SPEC_COPY=0` or empty = off everywhere.
   - Explicit values behave as before.
   - HIP, cpu, `--max-batch` > 1 and tp > 1 stay off silently.
   - The memory plan must stay correct (it reserves `len` per-row recurrent buffers when copy is on).
2. Tests: extend the `copy_default_*` tests for CUDA (unset / zero / empty / explicit; the 27B recipe stays off; mb2 and tp2 stay off). Run `cargo test -p intarsia-core -p intarsia-models -p intarsia-cli --lib --tests` on the Mac, including docs_env_vars.
3. Update `docs/tuning.md` (the `INTARSIA_SPEC_COPY` row, which is long: edit it carefully), the help text, the results doc, and a LOG row in `docs/decisions/LOG.md` (keep both rows on conflict).
4. Hardware gates on spark1 (35B NVFP4, `serve --max-batch 1 --spec mtp`, beside production vLLM):
   - (a) The startup log shows copy on with nothing set, and off with `=0`.
   - (b) Greedy ids with the default equal `INTARSIA_SPEC_COPY=0` and `--spec off`, on copycuda's cells (reuse its scripts/prompts).
   - (c) A short ABBA, >= 3 rounds, on code-gen and copy3k: default vs `=0`, run with `INTARSIA_CUDA_ATTN_KV_ONCE=1` set in BOTH arms. That knob is about to become default-on (PR #472 + lane kvflip), and the chain verify pass has 1+len rows, so check the trade still holds. Report the ratios.
   - Gate: (a) and (b) must pass. (c) is informational unless code-gen falls below 0.98x. If it does, stop and report without merging.
5. Open the PR, titled `[decision] CUDA: copy drafts default-on (single-sequence MTP, 35B recipe only; user decision #9)`. Then run `zsh /opt/code/intarsiaml/.worktrees/scan/bin/merge-when-green.sh <PR>`. If LOG.md conflicts, merge origin/main, keep both rows, push and rerun.
   Lane `kvflip` is editing `docs/tuning.md` (its KV-once row and the `INTARSIA_SPEC_SAMPLED_CTX_K` row) and `crates/intarsia-backend-cuda` at the same time. Don't touch those rows or that crate.

#### Fleet rules (hard)
- Every spark command that builds, starts a server or benchmarks goes through fleetq (read `/opt/code/fleetq/README.md`): `fleetq run --host spark1 --expect <MIN, guess high> --mem <GB peak> --label "copyflipc ..." [--load cpu | --quiet gpu,cpu] -- <cmd>`. Builds use `--load cpu`; timing runs use `--quiet gpu,cpu`. Never `--force` / `--no-queue`, and never release another owner's job. Lane kvflip holds both Sparks for tp2 runs at times, so expect to queue.
- Don't stop vLLM (spark1 runs flight/guidance/capcom).
- GB10 memory: 121 GB unified. Budget = static pool + vLLM (~34 GB on spark1) + 8 GB per parallel nvcc job + 10 GB OS. Build with `MAX_JOBS=4`. Poll `free -g` during first launches and kill your server below 15 GB available.
- Check SM clocks per window (the clk.sh pattern in `scripts/samp64/`). Discard windows below 2300 MHz.
- No single tool call or wait over 4 min (Bash timeout <= 240000). Poll at most every 4 min.
- At the end: no fleetq slot held, no server running, remote box dir removed. Keep the local worktree.

Report (short): PR number and merge SHA, the gate results (a/b/c with ratios), the test counts, the revert line, and the cleanup state.

## Progress notes

- Code (commit "CUDA: copy drafts default-on..."): `copy_default_applies` (`crates/intarsia-core/src/spec.rs:564`) now accepts `metal` and `cuda` (still needs `--max-batch 1`, tp 1 and the recipe's `[tuning.<backend>] copy_default`). The existing per-backend recipe key is reused; `[tuning.cuda] copy_default = true` is set in `recipes/qwen3_5_moe.toml` only. Chain length on CUDA is `COPY_LEN_DEFAULT` = 7 (what copycuda measured: `INTARSIA_SPEC_COPY=20`, len 7), from `set_copy_backend`. The ONE resolution point (`copy_from_env`) is unchanged, so the load-time verify rows (8), the saved per-row recurrent buffers (7 instead of 2) and the engine's chain agree. HIP, cpu, `--max-batch` > 1, tp > 1: off silently (the loader never sees an explicit value there, so no refusal). Startup line now says "default for this recipe on Metal / CUDA at one sequence; INTARSIA_SPEC_COPY=0 turns it off".
- Reach: the CUDA 35B recipe turns `--spec mtp` on by itself at tp 1 (`mtp_default_k = 2`), so a bare `generate` or `serve` (no `--max-batch`, no `--spec`) of the 35B on CUDA now has copy drafts on. The production `serve --max-batch 2` (batched MTP) is unaffected (off by `max_batch`); the 27B and HIP are unaffected.
- Sampled requests are untouched (chains attach only to greedy requests, `engine.rs`, `Seq::build`); only the larger verify-pass width and the per-row buffer reservation at load apply to them.
- Mac tests (fleetq #10430, endor, `scripts/copyflipc/tests.sh`): `cargo test -p intarsia-core -p intarsia-models -p intarsia-cli --lib --tests`: 103 test binaries, 1045 passed, 0 failed, 27 ignored (`raw/tests-full.log`); includes docs_env_vars, the 4 `copy_default_*` binaries (CUDA cases added: unset on = 20 / len 7 / 8 verify rows / 7 save rows; mb2 / mb8 / tp2 / no key / hip / cpu off; `=0` and empty off; explicit n 12 len 7 wins and is what the refusal sees), `copy_default_applies...` and the recipe test (`only_the_35b_recipe_opts_into_the_copy_default_on_metal_and_cuda`: every built-in recipe on every backend, only `qwen3_5_moe` on Metal and CUDA is true; the 27B stays off).
- Build on spark1 (fleetq #10431, 6 min 20 s, `MAX_JOBS=3`), then one gates slot (`scripts/copyflipc/gates.sh`): ids gate (generate; copy3k, code-gen, json-xform x arms `off`, `zero` (`=0`), `def` (unset)), then the ABBA (code-gen, copy3k; 4 rounds, 3 reps; arms def / zero, `INTARSIA_CUDA_ATTN_KV_ONCE=1` in both), memory watchdog on throughout, SM clock sampled every 2 s per server window.

## Gates (spark1, GB10, 35B NVFP4, CUDA tp 1; beside production vLLM, which stayed up and was not touched)

Binary: this branch (`copyflipc-<sha>-dirty`, built in fleetq #10431, 6 min 20 s), one binary for all arms; arms differ by environment. One slot (#10434, `--quiet gpu,cpu`) ran the ids gate and the ABBA; the free-memory watchdog (`wd.sh`, kill below 15 GB available) ran throughout and never fired, available memory was 56-57 GB at every server start. After the slot: remote box dir removed (#10446), no process left, no slot held. Raw: `docs/results/2026-11-20-copyflipc/raw/` (`gates-run.out`, `ids/`, `ab1/`, `ab-an.txt`, `clk-an.txt`). Scripts: `scripts/copyflipc/`.

| Gate | Result | Evidence |
|---|---|---|
| (a) startup log, nothing set | PASS: "spec: copy drafts on: a chain of up to 7 tokens when the last 20 tokens occurred earlier (default for this recipe on Metal / CUDA at one sequence; INTARSIA_SPEC_COPY=0 turns it off)" in all 4 served `--max-batch 1 --spec mtp` default servers and in the 3 default `generate` runs | `raw/ab1/serve-def-r*.log`, `raw/gates-run.out` |
| (a) `INTARSIA_SPEC_COPY=0` | PASS: no "copy drafts on" line in the 4 `=0` servers and the 3 `=0` `generate` runs | same |
| (b) greedy ids, `generate`, default = `=0` = `--spec off` | PASS (6 of 6 compared arms, `ids fails=0`): copy3k 2902 ids, code-gen 500, json-xform 2600, identical in all three arms. Chains fired in the default arm on copy3k (359 chain passes, 357 accepted 7 of 7); code-gen's 500-token `generate` fires no chain (as copycuda found), so for code-gen the chain path is covered by the served replies: **reply shas equal between default and `=0` in 8 of 8 round-cells** (code-gen and copy3k, 4 rounds, 3 reps each; `raw/ab-an.txt`) | `raw/gates-run.out`, `raw/ab-an.txt` |
| (c) ABBA, KV-once on in both arms (`INTARSIA_CUDA_ATTN_KV_ONCE=1`), 4 rounds, 3 reps, served `--max-batch 1 --spec mtp --spec-k 2` | code-gen **0.9924x** (rounds 0.9904 0.9863 0.9943 1.0047), copy3k **1.7269x** (1.7264 1.7115 1.7273 1.7452). Informational; code-gen is above the 0.98 stop line | below |

(c) decode tok/s, median of the 3 reps per server, rounds 1-4: code-gen default 161.6 / 160.0 / 161.3 / 163.1, `=0` 163.2 / 162.2 / 162.2 / 162.4; copy3k default 279.8 / 277.1 / 279.8 / 281.4, `=0` 162.1 / 161.9 / 162.0 / 161.3. Ratio = default / `=0` of the round's median. The trade is the one copycuda measured without KV-once (code-gen 0.992x, copy3k 1.693x): the 1+len-row chain verify pass did not make it worse with KV-once on, and copy3k reads higher (1.727x). SM clock per server window (busy samples, util >= 50 %): median 2437 MHz (default) / 2444 MHz (`=0`), minimum 2411; no window under the 2300 MHz line (`raw/clk-an.txt`). The code-gen cost is the one missed chain per request copycuda explained (about 0.25 % by pass times, 0.8 % measured); not investigated further here. The user accepted this trade in decision #9.

Not measured: sampled requests (never take a chain), thinking mode, real agent transcripts, `--max-batch` > 1 (off by construction), the memory plan on a box near its budget (the default reserves 7 per-row recurrent buffers instead of 2 for the 35B; about 63 MB each as on Metal, so about 0.3 GB, not measured on CUDA).

## Revert

`INTARSIA_SPEC_COPY=0` per process, no rebuild; or delete `copy_default = true` from `[tuning.cuda]` in `recipes/qwen3_5_moe.toml`; or revert this PR.

Post-merge recheck (after merging origin/main `0b417ec2`, which carries PR #473): `docs_env_vars` 2 passed, `intarsia-core --lib` 232 passed (2 ignored), the 4 `copy_default_*` binaries 1 passed each, `intarsia-models --lib` 74 passed, 0 failed.
