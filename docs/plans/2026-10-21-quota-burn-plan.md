# Quota-burn plan: 2026-10-21 ~09:00 → 2026-10-22 12:00 EDT (~27 h)

**Goal:** spend about 85 % of a fresh weekly quota on work that moves intarsiaml's requirements forward, in priority order. Two things limit this:

- **Fleet boxes.** Each box runs one GPU lane at a time, so the plan runs 3 GPU lanes and 4–5 Mac-only lanes in parallel.
- **The 5 h rolling limit.** The window spans about 5.4 such periods. Each 5 h block should therefore run near full, not idle and then burst.

Per-lane cost estimates are rough. They are scaled from last week, where 39 % paid for about 10 opus-sized rounds, so about 4 % per heavy opus round and 1–2 % per sonnet lane.

## Sharing with agent-harness (added 23:20)

The `agent-harness` project spends from the same quota in the same window. It also uses the same fleet: harness2's JobBench ran against spark1-vllm and spark2-vllm (#2782).

1. **Quota split.** The table below sums to about 83 % for intarsiaml alone. With an even split, intarsiaml's share is about 45 %, so run the lanes marked ★ first and treat the rest as fill.
   - ★ lanes: G1, G2, G2b, M1, M3, M4.
   - Checkpoints every 5 h compare total usage against the combined target. Whichever project is behind gets the next block.
2. **Fleet contention.**
   - **Sparks:** agent-harness probably needs spark1-vllm and spark2-vllm *up*, since it benchmarks against them. G1b (35B TP=2 on both Sparks) takes both down, so it moves to the overnight window, 00:00–07:00, or after agent-harness's runs.
   - **tatooine and endor:** agent-harness presumably doesn't use these, so G2/G2b/G3 and M1 go first.
   - **Coordination:** everything goes through fleetq. The two projects never take down a role the other one is benchmarking.
3. **Fewer parallel agents.** In block 1, run 4–5 agents, not 7, so both projects stay under the 5 h limit.

## Lanes

| # | Lane | Box | Model | Est. h | Est. quota | Requirement |
|---|---|---|---|---|---|---|
| G1 | CUDA round 7: e4m3 activations, opt-in. Gate: KL21 within vLLM's own KL | spark1 | opus | 5 | 5 % | 4 fast, 3 accurate |
| G1b | **35B TP=2 across both Sparks**. Not built yet; the 8B TP=2 exists | spark1+2 | opus | 8 | 10 % | 12 tp=X, 10 Spark |
| G2 | HIP round 10 resume: 8B GEMM split test, A/B, gates, merge | tatooine | sonnet | 1 | 1.5 % | 4, 11 |
| G2b | **35B TP=2 on 2× R9700** (RCCL path exists for the 8B) | tatooine | opus | 6 | 8 % | 12, 11 |
| G3 | Metal round: 4k decode floor at 84.5 % vs 85 %, then a prefill lever | endor | opus | 5 | 5 % | 4 |
| M1 | **New-model drill, timed:** onboard Qwen3.8-27B (what radiance serves) via `docs/adding-a-model.md` and `intarsia qualify`. Needs a new MXFP4 quant format (CPU reference, then CUDA/HIP/Metal kernels). After that, a head-to-head against radiance-native on the same box | Mac → tatooine | opus | 10 | 10 % | 1 modular, 2 easy |
| M2 | **EXL3 first-class format** (ABI id 5): CPU reference decoder plus a CUDA decode-GEMV, starting from TensorFold's MIT decoder | Mac → spark1 | opus | 8 | 10 % | 1 |
| M3 | Accuracy debt: root-cause the 35b-long prefill-floor KL FAIL (HIP and CUDA), and HIP prefix-reuse at chunk 1024 | Mac → spark/tatooine | opus | 5 | 6 % | 3 |
| M4 | Metrics: Grafana dashboard JSON and Prometheus scrape config shipped in the repo, plus the obs scrape job | Mac / obs | sonnet | 2 | 3 % | 8 |
| M5 | Reproducibility: one-command host-to-host deploy, bench bundle and lockfiles; CI for `cargo test`/`check` | Mac | sonnet | 4 | 4 % | 5 |
| M6 | TUI with a live architecture map and metrics overlay (the web UI already has one) | Mac | sonnet | 4 | 4 % | 7, 9 |
| R1 | Fable adversarial review of PRs #27–#32 (**ask first**, per CLAUDE.md) | none | fable | 2 | 5 % | all |
| D | Decision-log and docs agent, running continuously | none | haiku/sonnet | — | 2 % | — |
| X | Overnight follow-ups: CUDA round 8 / HIP round 11 on whatever G1/G2 leave open | fleet | opus | 8 | 10 % | 4 |
| | **Total** | | | | **~83 %** | |

## Schedule

1. **09:00, block 1:** check `fleetq status` and quota. Start G1, G2, G3, M1, M2, M3 and D in parallel; that is 7 agents, 3 of them on GPU boxes.
2. **~10:00:** G2 finishes and G2b starts on tatooine. M4 starts.
3. **~14:00, block 2:** G1 finishes and G1b starts on both Sparks. G1b needs both vLLMs down, so it uses longer windows; see decision 2. M5 and M6 start.
4. **~19:00, block 3:** merge whatever passed. R1 runs on the merged diff, if approved. M1 runs its head-to-head on tatooine after G2b.
5. **Overnight, blocks 4–5:** a chained Workflow runs X plus any lane left unfinished. Each stage commits and merges on green, under the PR/merge authorization of 2026-10-19 22:04.
6. **Every 5 h:** check quota with `get_usage`. If a block is behind, add Mac-only work. If it's ahead, slow the spend down.
7. **10:00 on 10-22:** wrap up. Bring every role back up, release all slots, write the final report, and file the decision log.

## Safety rails (unchanged)

- Every GPU window is a single `fleetq run`. It uses a `trap` that restores the role on exit, so an agent that dies mid-window still brings the service back.
- Bitwise defaults, or opt-in plus KL; a changed default needs your OK.
- No `--force` or `--no-queue`, and other owners' jobs are never touched.
- Shared-file changes get the Metal and CUDA no-op checks before merge.

## User decisions (2026-10-20 23:24 EDT)

The user's reply, verbatim: "1 yes, 2 yes, 3 yes, 4 yes, 5 yes, 6 60/40 harness but coordinate when harness can be without the sparks so you can use"

1. **Opus:** yes for implementation lanes in this window.
2. **Downtime caps:** 3 h per role per day, windows up to 30 min, unlimited 00:00–07:00.
3. **Workflow:** yes for the overnight chain. This counts as the opt-in.
4. **Fable review R1:** yes.
5. **obs Prometheus scrape job:** yes.
6. **Quota split:** 60 % agent-harness, 40 % intarsiaml, so intarsiaml targets about 40 % of the weekly quota.
   - At kickoff, message the agent-harness session (find it with ListAgents) and agree the windows when it can run without spark1/spark2 vLLM.
   - G1, G1b and M2's CUDA step get only those windows. Until then, tatooine, endor and Mac lanes go first.

## Decisions needed before 09:00 (answered above)

1. **Opus for implementation lanes in this window?** CLAUDE.md limits opus to tricky or judge work, which these lanes mostly are, but burning quota argues for yes.
2. **Downtime caps for TP=2 work:**
   - Sparks: vLLM on spark1 (flight/guidance/capcom) and spark2 (retro).
   - tatooine: radiance.
   - Proposal: 3 h per role per day, windows up to 30 min, and unlimited between 00:00 and 07:00.
3. **Workflow opt-in for the overnight chain** (stage X): yes or no.
4. **Fable review R1:** yes or no.
5. **Prometheus scrape job on obs (M4):** yes or no. This changes obs's config.
6. **Quota split with agent-harness:** 50/50, or some other ratio? Which project gets the Sparks during the day?
