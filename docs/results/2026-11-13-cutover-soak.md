# Cutover soak: load, soak and quality (blockers 6 and 7)

Lane `soak`, branch `cutover-soak` from origin/main `45e6ce56b`, measurement only (no engine changes).

## Brief (verbatim from the launching agent, kept so it survives compaction)

You are lane `soak` in intarsiaml, a Rust LLM inference engine. Working folder `/opt/code/intarsiaml/.worktrees/soak` (git worktree, branch `cutover-soak` from origin/main `45e6ce56b`). Work only there; no other work folders. Results go to `docs/results/2026-11-13-cutover-soak.md`.

### Goal
Collect measurement-only evidence for cutover blockers 6 and 7 in `docs/results/2026-11-12-ab35b-cutover-followup.md` (section "Cutover blockers, ranked"). The question is whether intarsiaml can replace production vLLM for Qwen3.6-35B-A3B-NVFP4 on spark2. Do not change engine code. If a bug is found, write it down with evidence and stop that arm.

### Already established (do not redo)
- Blockers 1 to 3 closed by PRs #351 to #354 and #357.
- 2-stream replay with `--max-batch 2 --prefix-store on` matches vLLM: TTFT sum 73 s vs 77.8 s, decode 39.9 vs 39.8 tok/s per stream.
- localeval's ITL p99 is computed per character and inflates single-token stalls; use the engine's own `/metrics` gap histogram for stall claims (`docs/results/2026-11-13-interleave.md`, `2026-11-13-stream-burst.md`).
- GSM8K n=200 baseline for vLLM: 0.610 and 0.585. intarsiaml batched bf16 store-off: 0.710 (Run 2 of the ab35b doc).

### What to measure (intarsiaml on main HEAD; production-shaped serve: `--max-batch 2 --prefix-store on`, bf16 KV, defaults otherwise)

Window A, at most 55 min with vLLM down: load and soak
1. More than 2 streams: replay2 interleaved with K=3 and K=4, one seed each. Compare against vLLM at K=3 and K=4 (vLLM runs `--max-num-seqs 2` in production, so both engines queue). Run the vLLM arms before taking vLLM down, while it is up and quiet. Production vLLM is a live service: check `fleetq status`, use `--quiet gpu,cpu` on spark2. If vLLM can't be measured cleanly, say so.
2. Soak: about 35 min of continuous concurrent multi-turn load at K=2. Loop replay2 over fresh seeds. Shrink the store's host budget so eviction runs: pick `host_mb` from the store's bytes-per-token (about 66 KB per token) so evictions start within about 5 min. Sample every minute: `free -g` and MemAvailable, server RSS, prefix-store tier bytes, hits/evictions/errors from `/metrics`, p50 TTFT and decode per replay. Pass criteria: no errors; no RSS or MemAvailable drift beyond noise; store bytes within budget; TTFT and decode flat over time. Confirm earlyoom is active on spark2.

Window B, at most 55 min with vLLM down: quality (blocker 7). `local-model-bench` skill (localeval at `/opt/code/deepeval`; `uv run localeval ...`), same pinned GSM8K setup as ab35b Run 2: `--limit 200 --max-tokens 1024 --pin-gsm8k`, thinking off, temperature 0. Arms:
- (a) store-on bf16 at `--max-batch 2` (new; the prefix store changes greedy bits on CUDA, check accuracy against store-off).
- (b) `--kv-dtype fp8` with the store on.
- (c) one thinking-on quality run: GSM8K thinking on, n=100, `--max-tokens` sized for thinking, against vLLM thinking-on. Run the vLLM arm while vLLM is up and quiet.
Use localeval `compare` for paired verdicts against the existing vLLM and B runs.

### How to run
- Build: `REPO=/opt/code/intarsiaml/.worktrees/soak ABL='~/intarsia-soak' BHOST=spark2 FLEETQ_OWNER=claude-soak fleetq run --host spark2 --load cpu --mem 24 --expect 20 --label "soak: build" -- bash /opt/code/intarsiaml/.worktrees/ab35c/build.sh`
- Windows: template `scripts/interleave/window.sh` (stops `missioncntrl-guidance`, trap restores, waits for `/v1/models` 200); copy to `scripts/soak/` and adapt. Each window: `FLEETQ_OWNER=claude-soak fleetq run --host spark2 --takes-down retro --quiet gpu,cpu --mem 58 --expect 60 ...`, hard cap 55 min of vLLM downtime per window. Standing OK covers spark2 vLLM-down windows up to 60 min. Leave at least 15 min with vLLM up between the windows.
- Replay tooling: `scripts/interleave/replay2.py` on main; GSM8K arms via `scripts/ab35b/` and `scripts/ab35c/lib.sh`.

### Rules
- Run a real check for anything runnable you change; say which check could not run.
- No tool call or wait over 4 min (Bash timeout <= 240000); poll at most every 4 min; run long windows in the background.
- `fleetq status` first; always `--expect`; never `--force` or `--no-queue`.
- GB10: 121 GB unified, swaps and hangs instead of OOM-kill; give `--mem` the peak; kill a run if MemAvailable < 10 GB (15 GB if earlyoom is not active).
- Commits end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`; no secrets.
- Finish: DRAFT docs-only PR against main with the results doc and raw files; PR body ends with the Claude Code line. Report a per-blocker verdict (closed, open, new issue) with key numbers.

## Log

(filled in as the work proceeds)
