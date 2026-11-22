## Brief L1 (coordinator, 2026-11-22), lane ink-spec-land

Goal: Inkling-2-Mini DSpark speculative decoding on Metal onto main as OPT-IN (default off), then measure it on today's main and
hand the coordinator a default-on proposal. User approved the lane on 2026-11-22.

- Base: origin/main 1f226a2e3 (has #488 Inkling Metal+CPU, #489 screened greedy head default-on on Metal
  (`INTARSIA_INK_HEAD_SCREEN=0` reverts), #490 dfuse tools; Metal decode without spec ~76 tok/s at 256, ~72 at 5k).
- Branch `ink/spec-land` in `.worktrees/ink-spec-land` from origin/main; merge `ink/mtp` (c6dffe0d6: DSpark k3 gate 0.8,
  device ingest 85265f9d5, serve under admitted kernels a3b72495b), then `ink/mtp-softcap` (36cbb9536, brings `ink/mtp-cut`
  5d73d0fa8: INTARSIA_DSPARK_CUT default off; softcapped verify rows for samplers, `tests/inkling2_verify_softcap.rs`).
  NOT `ink/mtp-flip` (f130b8855): this PR changes no defaults. Land only built and gated code; design notes stay doc text.
  Scratch only under `.worktrees/ink-spec-land/.scratch/`; read-only for `.worktrees/ink-mtp`, `ink-mtp-softcap`, `ink-mtp-cut`.
  Do not touch `ink/cuda-mtp`, `ink/cuda-mixmoe`, `ink/fp8-default`, `ink/batch`, `ink/prefill`; second to land merges main.
- Last numbers (#10991, base a3b72495b): decode off -> DSpark 63.6->80.2 (142), 61.6->77.5 (1k), 60.9->87.5 (4k),
  57.1->72.1 (16k); acceptance 77%, 2.55 tokens/pass; prefill -6/-7% at 4k/16k (TTFT +0.2 s / +0.75 s) from the draft's
  context ingest after each prefill chunk; that held the flip.
- Gates (rules written in this doc before running):
  a. Spec off (default) is a no-op vs main on Metal: Inkling greedy md5 at 64 / 256 equal to main's; `plan show` text + JSON
     byte-identical to main for Inkling, Qwen3-8B, Llama-3.1-8B; Metal Inkling tests + workspace tests pass; CUDA `plan show`
     unchanged and HIP builds (spark1, CPU slot).
  b. `--spec` DSpark k3 gate 0.8: greedy ids identical to spec off on the 12 greedy prompts x 128 plus the prose texts;
     verify-softcap test passes and sampled requests read softcapped rows; serve smoke under admitted kernels: one ~1.4k
     greedy request returns 200 with verify passes > 0.
  c. The real ubuntu check is the PR's own CI (bespin is macOS).
- Speed on the branch tip, endor, quiet slot: spec off vs DSpark at 142 / 1k / 4k / 16k, decode tok/s, prefill tok/s, TTFT, via
  localeval (`local-model-bench`, serve --max-batch 1, labelled as a spec run). Endor's GPU slows under sustained load: ABBA
  within one slot after a warm-up; absolute numbers next to ratios.
- Push `ink/spec-land`, open the PR with every gate by job number and the speed table; merge only when all CI jobs are green.
  Never force-push.
- Then report a default-on proposal: decode x and TTFT cost per length today; whether f130b8855's conditions still fit (one
  sequence, greedy only, max-batch <= 1); what deploy needs (drafter weights source); one concrete idea with an estimate to
  remove the draft-ingest prefill/TTFT cost (overlap with next chunk, or defer). Build it only if bitwise and < half a day.
- Rules: every box-loading command via fleetq (`fleetq status` first; builds/tests `--load cpu`, timing `--quiet gpu,cpu`;
  no bare cargo/rustc on endor outside a slot); job scripts trap and kill their process tree; no tool call or wait > 4 min;
  never release others' jobs, no --force / --no-queue. If a gate fails, stop and report instead of loosening the rule.
  Report to the coordinator with job numbers at: merge result, gate results, speed table, PR link, default-on proposal.
