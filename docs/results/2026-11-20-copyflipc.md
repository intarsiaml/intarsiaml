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

