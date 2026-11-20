User decision #9 (approved): CUDA copy drafts default-on for the 35B recipe on the single-sequence MTP engine, mirroring the Metal flip (#460).

## Change
- `INTARSIA_SPEC_COPY` unset = `20` (chain len 7, `COPY_LEN_DEFAULT`, the value copycuda measured) on CUDA with `--spec mtp` at `--max-batch 1` / `generate`, tp 1, for a recipe that opts in: `[tuning.cuda] copy_default = true` in `recipes/qwen3_5_moe.toml` only (existing per-backend key reused; the 27B and every other recipe stay off).
- `copy_default_applies` (`crates/intarsia-core/src/spec.rs:564`) accepts Metal and CUDA. HIP, cpu, `--max-batch` > 1 (production's `--max-batch 2`), tp > 1: off silently, no refusal. `INTARSIA_SPEC_COPY=0` or empty: off everywhere. Explicit values behave as before. One resolution point (`copy_from_env`), so the verify rows, the 7 reserved per-row buffers and the engine's chain agree.
- Reach: a bare `serve` / `generate` of the 35B on CUDA (the recipe turns MTP on at tp 1). Sampled requests never take a chain.

## Evidence
copycuda (35B NVFP4, `20` vs off, 4 ABBA): copy3k 1.693x, json 1.507x, agent-edit 1.092x, agent-batch 1.041x, code-gen 0.992x (the accepted trade).
Gates on spark1 (35B NVFP4, `serve --max-batch 1 --spec mtp`): (a) startup line on with nothing set, absent with `=0`: PASS. (b) greedy ids default = `=0` = `--spec off` on copy3k / code-gen / json-xform: PASS; served replies equal 8/8. (c) ABBA with `INTARSIA_CUDA_ATTN_KV_ONCE=1` in both arms, 4 rounds: code-gen 0.9924x, copy3k 1.7269x; SM clocks 2411+ MHz. Mac tests: 1045 passed, 0 failed (`cargo test -p intarsia-core -p intarsia-models -p intarsia-cli --lib --tests`).

Details: `docs/results/2026-11-20-copyflipc.md`.

Revert: `INTARSIA_SPEC_COPY=0` (no rebuild), or delete the recipe key.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
