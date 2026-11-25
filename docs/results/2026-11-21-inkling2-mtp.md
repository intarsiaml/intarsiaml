# Inkling-2-Mini speculative decoding with its DSpark draft (lane ink-mtp, 2026-11-21)

Status: landing as opt-in (default off) on branch `ink/spec-land` (lane ink-spec-land, Brief L1); the old lane's branch was `ink/mtp`.

## Briefs received

- 2026-11-21 (coordinator, SendMessage): Fable's decode review ranks this lane #2 (`.worktrees/ink-fable/decode-profile-review.md`,
  item 2, sections 3.5 and 4; read section 4 before designing the verify path).
  1. Shape: short confidence-gated drafts, 2-3 verify rows; the drafter rows share the verify's unembed launch.
  2. Estimated saving 3.0-4.4 ms of the real 15.76 ms step at acceptance 0.6-0.7, about 6 ms if consecutive tokens share experts.
  3. #10743 decides it: report acceptance by draft position and the expert-union size for 2 and 3 consecutive tokens
     (union over the 8-of-128 per layer) as soon as available.
  4. Device-side accept / verify readback is ink-hostgap's job: send interface needs to the coordinator, who relays.
  5. Measure Metal decode with `--profile-every 0` (real greedy about 63.4 tok/s; the default profiler adds about 4 ms/token).

- 2026-11-21 (coordinator, SendMessage): endor's queue is about 5 h deep. Move the real-model CPU token-identity job (#10807) to
  spark2's CPU (released on endor, resubmitted there); keep the Metal decode tok/s job (#10806) on endor. Every other CPU-only
  job of this lane goes to a Spark.
- 2026-11-21 (coordinator, SendMessage): priorities in order. (1) Fold the drafter's base-logit read into the verify's unembed
  (Fable §4.3; a separate 922 MB read is about 1.6 ms a cycle). (2) Move the tap mixes and the top-64 Markov head to the device;
  send the interface needs (what the device hands back per cycle: accepted count, next ids, draft confidences) after the
  per-cycle profile. (3) Tune the policy on a mixed set (greedy golden + prose + chat turns), report tokens per pass and real
  tok/s per policy; the default must never be slower than spec off on prose; the p1 >= 0.6 gate is the right shape.
  (4) Sampling: confirm whether temperature > 0 uses exact speculative rejection sampling; if not, keep spec greedy-only and
  say so. Speculation stays opt-in until #10806's timed numbers; default-on is the coordinator's call. If drafter changes land
  before #10806 starts, edit its script rather than queue a new job (within its --expect).
- 2026-11-21 (coordinator, SendMessage): §4.3 agreed circular for the anchor; the separate MXFP4 draft head is the fix, its
  precision a measured choice in #10806. Ownership: this lane's drafter device ops ((c) top-k ids / values + p1, the tap mixes)
  take discriminants 92-95; ink-hostgap owns (a) device argmax over the R verify rows with the accepted length and (d) the single
  per-cycle readback. Send the interface after the #10806 profile (relayed to hostgap). Remove the 7 x V head download + host
  top-64 sync first ((c) on device). Greedy-only DSpark is fine: say so in the doc and the serve / CLI `--spec` help.
- 2026-11-21 (coordinator, SendMessage): endor's quiet-GPU line is about 9 h. Move the token-identity checks out of #10806 into
  a default-GPU-load job with a short --expect (backfills); keep #10806 for tok/s plus the DSPARK_PROFILE run. ink-metal fixed
  the rows_exact pin-list test at a0751f302: merge ink/metal.
- 2026-11-21 (coordinator, relay from ink-hostgap; their design note docs/results/2026-11-21-inkling2-hostgap.md on ink/hostgap,
  6855e486a, nothing built): a DSpark cycle has 3 host syncs, S1 ingest (taps down, host mix, upload per layer), S2 draft head
  (base-logit download, host top-64, host Markov + gate), S3 verify (argmax readback, accept). hostgap can provide H1 (one 40 B
  per-cycle record [n_acc, bonus, k_eff, drafts[7]]) and H2 (encode cycle c+1 ahead, discard on reject). Removing the syncs needs
  this lane's M1 (device ingest mix), M2 (device Markov head + gate), M3 (verify always at kmax+1 rows + an accept op), M4
  (device positions for Inkling's ops). Ask: report per-cycle host ms split S1 / S2 / S3 from INTARSIA_DSPARK_PROFILE=1 (a short
  default-load job if #10806 lacks it); under about 1 ms per cycle in total drops M1-M4. And: may hostgap add
  INTARSIA_HOST_PROFILE hooks (profiling lines only) in the spec loop; name the file to keep out of.
- 2026-11-21 (coordinator, SendMessage): the arithmetic points at the verify pass, not the drafter: k3 is 35.0 ms per cycle,
  so verify at 4 rows is about 29 ms against 16.6 ms at 1 row (about +4.3 ms per row; k2 about +4.7), while ink-batch measures
  about 2.3 ms per extra row for batch_step 2-8 rows. (1) Before the AttnDecode work, profile one verify pass at T = 3 and 4
  per op (kernel trace, attribution only) next to ink-batch's B = 3 / 4 rows path: does verify route wo / unembed / InkInProj /
  experts to generic GEMM / prefill-style kernels or to the rows kernels; attention and the rel prologue at T > 1 too. (2) If it
  is the routing, send verify T <= 16 through ink-batch's rows kernels (bitwise per row with the 1-row kernels). Merge ink/metal
  first (m-mix 47570539e, serve fix 761a1561b). (3) Default-on candidate k2 tau 0.6 greedy: flip as its own commit with an env
  off switch, only after #10874 shows the gain; sampled requests stay on the normal path.
- 2026-11-21 (coordinator, SendMessage): flip choice k3 tau 0.8. Flip only after #10874's serve + localeval speed-plain shows
  on >= off at every prompt length it runs. Greedy requests only (sampled stay plain); single sequence only (spec off when
  ink/batch's batched decode has B > 1 live sequences); an env off switch named in docs/tuning.md with the numbers; its own
  revertable commit; plan show / greedy md5 unchanged with the switch off. If localeval shows on < off at any length, no flip:
  send the table. Also report: verify at T = 3 / 4 per-op profile, whether verify routing changed between #10806 and #10874, and
  verify ms per cycle now; if verify still costs about +4 ms per extra row, that is the next lever before AttnDecode.
- 2026-11-22 (coordinator, SendMessage): decision (b): make the prompt-time drafter ingest cheaper or deferred first, then flip
  (16k prompt + 100 tokens is slower end to end with DSpark: 11.97 + 1.39 s vs 11.22 + 1.75 s; the flip is not urgent).
  (1) Check and say here whether the draft ingests the prompt on requests that can never speculate (sampled, the serve
  default t1.0, or batched); if so, skip the ingest entirely for them, before any flip (no lazy catch-up: sampling is per
  request). (2) Greedy: cut the ingest cost, preferring (i) ingest only the rows the draft can attend to if its context is
  windowed, (ii) fold the ingest into the prefill chunk's command buffer so it overlaps, (iii) defer it to the first decode
  step; ids identical spec on vs off. (3) Flip gate: localeval speed-plain off vs DSpark k3 tau 0.8 at 142 / 1k / 4k / 16k,
  quiet slot: prefill within 2% of off at every length with the decode gain kept, plus a sampled (t1.0) run with prefill ==
  off; fix the run label (not spec:unknown). (4) If (1) is done but (2) cannot get under 2% in about 2 hours of work, report
  the best number. (5) After the flip: the union-fused experts pass at 2-4 verify rows is this lane's; write a plan and send it
  before building. Rules: compile on bespin, CPU tests on spark2, GPU only inside endor fleetq slots.
- 2026-11-22 (coordinator, SendMessage): experts plan approved as proposed. (1) Run the probe: one default-load endor job,
  per-visit µs at T = 1..4 with real routing plus the gate+up / down split (fold it into #11147 if simpler; do not delay the
  ingest gates). (2) Build the union-grouped fused down only if gather + dn_u exceed about 10% of the experts time at 4 rows:
  bitwise, flag default off. (3) skipped for now. Design only while the gates wait: adaptive draft length, k in {0,1,2,3} per
  cycle from a running acceptance estimate and/or the draft's top-1 confidence, simulated offline on CPU from the oracle
  acceptance logs (no GPU jobs). Report the expected tokens per pass and tok/s against fixed k3 tau 0.8 on greedy golden and
  prose, with the measured +2.9 ms per extra verify row; send the table before building anything.
- 2026-11-22 (coordinator, SendMessage): go on the cumulative cut; skip EMA. (1) Build env knobs for the cut c and kmax, default
  off (k3 tau 0.8 unchanged when unset); a few lines in the gate only; bespin compile + spark2 CPU DSpark tests. (2) One quiet
  endor job behind #11147 / #11148 (do not delay them): c in {0.35, 0.5} x kmax in {3, 4} against k3 tau 0.8, interleaved, on
  greedy golden and the real-decode prose prompts, with server counters (tokens / pass, rows / pass). (3) Decision rule
  (coordinator's, pre-set): if the best (c, kmax) beats k3 tau 0.8 beyond spread on greedy and is not worse beyond spread on
  prose, it becomes the flip commit's policy (amend f130b8855's message with the numbers). kmax 4 needs verify at 5 rows:
  confirm it routes to the rows kernels with no slow fallback and that the 4-row crossover is not a cliff at 5; report
  per-pass ms at 4 and 5 rows. If nothing clears, keep k3 tau 0.8 and report. (4) The flip still needs the ingest gates first
  (prefill within 2% at every length, sampled prefill equal to off).
- 2026-11-22 (coordinator, SendMessage): cut sweep #11188 down to one candidate, cumulative 0.5 kmax 3 vs fixed k3 tau 0.8,
  ABAB (at least 2 rounds each); the refit gain (+2-4%) is inside model error, and kmax 4 and c 0.35 lose on prose. Keep it only
  if greedy tok/s beats k3 tau 0.8 by more than the spread at all 4 lengths, prose is not worse, and ids equal spec off;
  otherwise the cut stays opt-in and the flip keeps k3 tau 0.8. Commit the cut in .worktrees/ink-mtp-cut only after #11214
  passes; if #11188 reaches the front first, release and requeue it behind the others (never build uncommitted or failing
  code). Send the refit correction with the #11214 result.
- 2026-11-22 (coordinator, SendMessage): the fused-down kernel goes to a fresh lane; this lane keeps the flip. (1) Write a
  self-contained build brief as a section of this doc (design, rows-exact contract, files, bench and admission commands, gates,
  probe numbers); worktree .worktrees/ink-mtp-dn from ink/mtp-cut 5d73d0fa8; the lane starts only if quiet #11231 confirms down
  >= 10% of experts time at T = 4, else parked. (2) Flip rule unchanged (prefill within 2% of off at all 4 lengths, sampled
  prefill equal to off, decode gain kept). Localeval 4k at -6% fails today; if #11226's engine runs show 4k / 16k equal, the 4k
  number counts as noise only if one more ABAB at 4k lands within 2% (queue it only after #11226); if engine 4k shows a real
  gap, find the ingest cost before any flip. (3) The -1.7 pp acceptance gap at k1 tau 0 is not blocking: note it as a known
  difference. (4) Cut: wait for #11227 (counts only with ids equal off, prose not worse, greedy > k3 tau 0.8 beyond spread at
  every length); keep the clean-tree + passed-tests guard on every queued build job.
- 2026-11-22 (coordinator, SendMessage): low priority, after the flip gates. Latent bug found by ink-cuda-attn (gate5, fleetq
  #11254): `verify_head` (model.rs:2890) fills `ws.logits` through a plain proj and never applies the final logit softcap (30);
  decode applies it on host readback in `Model::softcap_logits` (decode == 30*tanh(verify/30) to every printed digit, cpu and
  CUDA). Greedy is safe (monotone); a sampled or probability-based verify reading `forward_verify_logits` would see uncapped
  logits, unreachable today. Fix as one small commit: apply the cap wherever verify rows are read as probabilities, or a hard
  assert that sampled verify is unreachable plus a comment that the rows are pre-softcap. Bitwise-neutral for greedy (md5
  equal). Do not change it while #11226 / #11227 are pending on the current binary.
- 2026-11-22 (coordinator, SendMessage, from the CUDA attention lane ink/cuda-mtp b1d84ba5b): the same verify_head pre-softcap
  bug. Any sampled spec path (rejection sampling, Gumbel acceptance) must sample from the capped logits, so fix it before
  sampled spec is enabled on any backend, with a test: a sampled verify-pass distribution equal to plain sampled decode
  (chi-square), or at least the verify logits equal to the decode logits after softcap. Report the commit.
- 2026-11-22 (coordinator, SendMessage): 36cbb9536 ships with the Inkling merge, no hotfix on main (main is not affected:
  only Gemma 4 caps there, and Gemma 4 cannot load the MTP head that every sampled spec path needs). **Condition before
  sampled speculation is turned on for Inkling-2:** a test that runs sampled verify with softcap 30 and checks the verify rows
  are capped before sampling.
  Status (ink/mtp-softcap 36cbb9536), `crates/intarsia-models/tests/inkling2_verify_softcap.rs`
  (`verify_logits_rows_are_the_softcapped_decode_logits`):
  - covers `Model::forward_verify_logits`, the readback the sampled verify calls (engine.rs `forward_verify_logits` call in
    the sampled pass), on the tiny Inkling-2 with `final_logit_softcapping` 30: verify rows bitwise the one-token decode steps'
    capped logits; unfixed it fails (299 / 300 differ, max |d| 0.45);
  - does not cover `forward_verify_topk` (device top-k; CPU has no device top-k, so it falls back to whole rows), so run the
    same test on Metal / CUDA with device sampling on before enabling sampled spec there;
  - does not run an engine-level sampled verify: Inkling-2 has no sampled spec path yet (`engine.rs` admits sampled requests
    only under `--spec mtp`). Whoever wires sampled DSpark adds the engine-level test (accepted-token distribution vs plain
    sampled decode, chi-square) in the same change.

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
- 2026-11-22 (coordinator, rule found by ink-metal): `git push` runs the repo's pre-push hook (`.githooks/pre-push` ->
  `scripts/prepush.sh`, two `cargo test`s), so a push from endor is a cargo run: push only inside a slot,
  `fleetq run --host endor --load cpu --expect 10 -- git push ...`. This lane's first two pushes (13:4x, before the rule
  reached it) ran outside a slot; every later push goes through fleetq.
  Report to the coordinator with job numbers at: merge result, gate results, speed table, PR link, default-on proposal.

## Brief L2 (coordinator, 2026-11-22 ~14:40), lane ink-spec-land

Merge #493 on all-green CI as planned. User decision 5 (2026-11-22, "do what you would recommend", relayed by the coordinator)
covers spec default-on on Metal as well as CUDA. After #493 merges, make Metal DSpark default-on for Inkling-2 as its own PR from
main, in a new worktree `.worktrees/ink-spec-flip`, removed after merge.
- Mechanism: the recipe default (`mtp_default_k` or its DSpark equivalent) in `recipes/inkling2.toml` `[tuning.metal]`, k3 /
  gate 0.8. Revert: `INTARSIA_SPEC_DEFAULT=0` or `--spec off`. Drafter weights missing at load: spec stays off, one stderr line,
  never a load error.
- Sampled requests: #494 (device Gumbel sampler default-on) is in flight; spec requests stay on the host sampler. Measure t=1.0
  decode, short and 5k: spec default vs spec off with the device sampler. Sampled requests take whichever is faster; if plain
  wins, sampled requests skip spec by default. Same seed must give the same tokens as the chosen path's opt-in run.
- A batched server (`--max-batch` > 1) must not change output or get slower; turn spec off there if needed.
- Gates: greedy md5 on == off at 64 / 256; plan show for Qwen3-8B and Llama identical to main; serve smoke; a night quiet slot
  after #11567 (`fleetq --at 05:45` or the earliest slot after it), box check first: serve-path TTFT and prefill, spec default
  vs off, at 1k / 4k / 16k, ABBA x3 (more reps: today's 4k / 16k localeval cells favoured off by 3-4%, inside noise).
- Bar: decode gain kept at every length; serve TTFT and prefill within 2% at every length. Otherwise stop and report.
- Endor rules unchanged: cargo and git push only inside `--load cpu` slots.

## Brief L2a (coordinator, 2026-11-22 ~15:20), lane ink-spec-land

The flip plan is approved as written, prefix-store skip included (consistent with the MTP default). One change: the re-prefill
cost is not acceptable as a known cost. A greedy turn after a sampled turn would re-prefill the whole context: at 4k a re-prefill
costs ~2.1 s while spec saves ~0.85 s over a 256-token reply (65 -> 83 tok/s); at 1k 0.68 s lost against 0.42 s saved, so prefix
reuse wins at every length.
- Rule: when the draft is unsynced and a reusable KV prefix exists, run that request without drafting and keep the prefix. Draft
  again once a request starts with the draft synced, e.g. a fresh conversation.
- Output is identical either way (spec on == off): this only changes the schedule. Gate: a multi-turn serve test with a sampled
  turn, then a greedy turn, at ~4k; the greedy turn's TTFT stays at prefix-reuse level and its ids equal `--spec off`.
- If the rule takes more than half a day in engine.rs, stop and report instead of shipping the cliff.
- Then run the gate job, and queue the night job at 05:45 as planned.

## Brief L2c (coordinator, 2026-11-22 ~16:00), lane ink-spec-land

- Rerun approved with both fixes, as written; get the slot so it ends well before #11621 at 05:45. F4: kernels checked, total
  cases and the full greedy reply (reasoning_content + content); the run / cached split depends on run order and was never a fair
  bar. F8 v2: bars as proposed; keep the void rule (`--spec off` reusing fewer than 4000 rows fails, never passes). F6: the 2%
  sampled gap counts only if the A B B A night run shows it; report it either way.
- Merge freeze: merge nothing to main until the coordinator says PR #491 (CUDA mixmoe) has landed. Draft #496 may stay open with
  CI running. After #491 lands: merge main into the branch, rerun F1, F7 and the md5 checks before landing.
- Write the chat-reuse finding (Inkling-2 serve re-prefills every normal chat turn: 16-row ring margin + template re-render) as
  its own section with numbers (prompt rows, re-prefill time, TTFT at turn 2). A TTFT lever, separate from the flip; build
  nothing for it.

## Brief L3 (coordinator, 2026-11-22 ~16:00), lane ink-spec-land; queued after the F4/F8 rerun and #11621

Measurement only, no default change. Work folder `.worktrees/ink-spec-k/`, new branch; no merge to main while the #491 freeze is on.
Report before building any default.

- Reference: mlx-serve v26.11.4 runs the same DSpark head (mtp.safetensors, block 8) with another verify rule. Source to read (not
  copy): `.worktrees/mlx-serve-src/src/plugins/inkling2/plugin.zig:290-435`. They draft 7 per round; verify a variable-length
  prefix: draft while the running product of drafter confidences >= `cut`, never past a stop id; `cut = clamp(1.34*sqrt(r), 0.3,
  0.8)`, r = marginal cost of one more verified row over a plain step (running linear fit of verify ms vs rows), cut 0.5 until
  measured; verify appends nothing, then commits accepted rows (no rollback). Headline 116 tok/s single stream, M4 Max, greedy,
  best scenario. Ours on endor (M5 Max): 1.12-1.28x over spec off at k3, gate 0.8.
1. If k or the gate were swept before, show those numbers first.
2. Metal, greedy, quiet slot, A B B A: k 3 / 5 / 7 x three rules: today's gate 0.8; cumulative-product cut fixed 0.5; cumulative
   with cut from our measured r.
3. Contexts short, 1k, 4k, 16k. Prompts: the usual set plus one (predictable) code prompt.
4. Measure verify ms for 1..7 rows (gives r), acceptance, tokens per round, decode tok/s, output identical to `--spec off`.
- A rule that needs a code change goes behind a measurement-only env on the new branch.

## Coordinator ruling (2026-11-22 ~17:15), lane ink-spec-land: merge order after the #491 freeze

#491 merged as main 8c7bb7cb1. Merge order on main:
1. #497
2. the CUDA flips PR
3. #496, after the 05:45 night speed run (#11621) passes
4. chat-reuse, after all its gates pass

Before merging: merge main in, rerun md5 and the affected tests, and ask the coordinator for the go.

## #496 on a batching server (coordinator, 2026-11-22 ~19:05), lane ink-spec-land

Question: what does `serve --max-batch 16` do with #496 applied? (a) refuse to start, (b) fall back to one sequence, or (c) leave
spec off and batch.

**Code path, 34525d854: (c).**
- `load_engine_group` (`crates/intarsia-models/src/load.rs`) first resolves `max_batch` (16 with `--max-batch 16`; 1 without the
  flag). `dspark_default` then returns `Off` when `opts.max_batch > 1`, so no draft is opened and no line is printed.
- The batching engine starts as on main.
- An explicit `--spec dflash` with `--max-batch > 1` still fails before any weights load: "--spec dflash with the batching engine
  (use --max-batch 1)".
- `serve.rs` `refuse_spec_with_store` only acts on an explicit `--spec` with a prefix store.
- Unit test `dspark_default_only_where_measured` covers `max_batch: Some(4)` -> `Off` (and `Some(1)` -> `On(3)`).
- The real-model gate F4 (#11634) ran `--max-batch 4`: no draft, the same kernels and cases as main, and the same greedy reply.
- No guard needed. Added for the record: a `Some(16)` case in that test, committed with the post-night merge of main so that the
  night job's code hash (34525d854) stays put.

**Conc check rules (written before the run; `.scratch/flip_conc.sh`, endor `--quiet gpu,cpu`).** Flip binary 34525d854 vs main
22303d704, `serve --max-batch 16`, A B B A (main, flip, flip, main). PASS when all of these hold:
1. Every server starts and prints `continuous batching: up to 16`, and no server prints the DSpark default line.
2. The spec steps counter is 0 after every arm.
3. 4 concurrent greedy completions (1021 / 1021 / 1021 / 1500 prompt tokens, 64 out) give main's texts.
4. localeval `suite --cells conc-plain-1k --max-num-seqs 4` (4 closed-loop streams): the flip's aggregate tok/s is within noise of
   main's (`localeval compare`: under 10%, both reps of each arm shown).

**Conc check result: PASS (#11695, endor quiet, 19:15-20:34, A B B A).**
1. All four servers printed `continuous batching: up to 16`, and none printed the DSpark default line.
2. The spec steps counter was 0 after every arm.
3. The 4 concurrent greedy completions on the flip equal main's: [True, True, True, True].
4. localeval `conc-plain-1k`, aggregate tok/s, median per run, arms in run order (main, flip, flip, main):

| streams | main (A1, A2) | flip (B1, B2) | flip vs main (medians of both reps) |
|---:|---|---|---:|
| 1 | 55.0, 50.8 | 55.0, 55.0 | +4% (one 50.8 rep on main; ITL steps, noise) |
| 4 | 84.6, 84.6 | 84.6, 84.6 | 0% |
| 16 | 121.5, 118.4 | 121.9, 122.7 | +2% |

TTFT p99 at 16 streams: main 11.12 / 11.54 s, flip 11.11 / 11.35 s. Runs: `20261122-191557-…-conc4-main`, `20261122-193455-…-conc4-flip`,
`20261122-195626-…-conc4-flip`, `20261122-201518-…-conc4-main` (localeval results dir). Under `--max-batch 16` the flip
behaves as main: spec off, the batching engine unchanged.

## #496 merge of main 08ae803e0 (coordinator brief, 2026-11-22 ~21:15), lane ink-spec-land

Brief (condensed): prepare #496's merge without touching `ink/spec-flip`, so #11621's code-hash marker stays valid. (1) New
worktree `.worktrees/ink-spec-flip-merge`, branch `ink/spec-flip-merge` from `ink/spec-flip` 9c78ba5e0, merge main 08ae803e0.
(2) Resolve `tests/mtp.rs`; check `load.rs`, `recipe.rs` and `recipes/inkling2.toml` against #498's CUDA DSpark default: Metal's
default k=3 and CUDA's must both survive, and `--max-batch` > 1 must still turn the Metal default off. (3) Non-timing gates on endor
via fleetq: md5 64 / 256 spec default == `--spec off`, the Metal spec / mtp tests, the unit test with the `Some(16)` case, the conc
behaviour check (4 concurrent greedy identical, no DSpark line at `--max-batch 16`); skip timing. (4) After #11621 passes (Mon 05:45):
fast-forward `ink/spec-flip` to the prepared merge, push, wait for green CI, report. Merging #496 still needs the user's go (the
coordinator asks in the morning). If #11621 fails the bar: stop and report.

**Resolution.** #498 brought a second implementation of the same default for CUDA: `[tuning.cuda] mtp_default_k = 3`, which on a
recipe without `[mtp]` is the DSpark draft's default (`Recipe::resolve_draft_default`, `load.rs` `dspark_default` /
`dspark_draft_present`). This branch had its own key, `[tuning.metal] dspark_default_k = 3` (`resolve_dspark_default_k`, a
`DsparkDefault` enum). One key for one behaviour: the merge keeps main's code (`recipe.rs`, `load.rs`, `tests/mtp.rs` as on main)
and moves Metal onto it, `[tuning.metal] mtp_default_k = 3`; the branch's `dspark_default_k` field and function are gone.
- Same as before on Metal: no `--spec`, one rank in process, `--max-batch` unset or 1, no `--spec-draft` / `--dcoco`, `--spec-k 0`
  off, a lone `--spec-k N` sets k, `INTARSIA_SPEC_DEFAULT=0` / `--spec off` opt out, greedy requests draft and sampled ones take plain
  steps (engine unchanged), the startup line ("the DSpark default of recipe inkling2 on metal ([tuning.metal] mtp_default_k ...").
- Changed on Metal, from main's code: a configured prefix store turns the default off silently (the branch printed one "not applied"
  line); a checkpoint without the draft prints main's line ("not applied: no DSpark draft in this checkpoint").
- The branch's other code is unchanged by the merge: `engine.rs` (an out-of-sync draft keeps the KV prefix, Brief L2a) and its test
  in `inkling2_dspark.rs`.
- The unit test is main's `dspark_default_only_where_the_recipe_sets_it`, extended: Metal `Some(3)`; CUDA and Metal off at tp 2, on
  a remote rank, at `--max-batch` 2 / 4 / 16 and with a prefix store; on at `--max-batch 1`; an explicit `--spec`, `--spec-k 2 / 0`
  and `--spec-draft` on both; no other recipe has a draft default on any backend.
- Docs: `docs/recipes.md` and `docs/tuning.md` take main's rows plus "and Metal"; the decision-log rows name the new key.
- Interaction with #499 (ring snapshot, now on main): a chat's turn 2 resumes at turn 1's prompt end. The DSpark draft's cache is a
  plain attention cache (no ring) and `DflashState::cut` truncates it to the resume point, so the draft stays in sync; the `f_rnn`
  ring row a draft reads is the last committed position's, which the remainder prefill writes. Greedy ids cannot change either way
  (verify is row-exact). Gate MG4 checks it.

**Gate rules (written before the run; `.scratch/flipm_gate.sh` in this worktree, endor, default load, not timing).** Merged tip vs
main 08ae803e0 (both built fresh: own target dirs, main from `git archive` with touched files; the merged binary must carry the new
recipe text and main's must not). PASS when all hold:
- MG1: Inkling greedy md5 at 64 / 256 (the A1 prompt): default (no `--spec`) == `--spec off` == main; the default run prints the
  DSpark default line naming `mtp_default_k` and has verify passes > 0; `--spec off` prints no default line.
- MG2: `serve --max-batch 16`, main then merged: each prints `continuous batching: up to 16` and no DSpark default line, the spec
  steps counter is 0 after the arm, and the 4 concurrent greedy completions (1021 / 1021 / 1021 / 1500 prompt tokens, 64 out) on the
  merged server equal main's.
- MG3: default server (no `--spec`, no `--max-batch`): the ~1.4k greedy chat request HTTP 200 with verify passes rising; a t1.0
  request HTTP 200 with the steps counter unchanged.
- MG4: two-turn chat at ~4.6k (`chat_gate.py flow 130`, greedy), default server vs `--spec off` server: turn 1 and turn 2 replies
  equal, turn 2's cached tokens equal on both and > 0. Info: the default server's spec steps over the flow.
- MG5: a model dir without `mtp.safetensors`: rc 0, exactly one "not applied: no DSpark draft" line, no speculation, md5 = main's.
- MG6 tests pass (none skipped where the fixture is present): `intarsia-models --lib dspark_default` (the extended unit test),
  `--lib recipe`, `--test recipes_doc`, `--test recipe_gumbel`, `--test mtp`, `intarsia-cli --test docs_env_vars`; on Metal
  (`INTARSIA_INKLING2_BACKEND=metal`) `inkling2_dspark`, `inkling2_dspark_cut`, `inkling2_verify_softcap`, `inkling2_chat_reuse`,
  `inkling2`, `inkling2_batch`.

**Gate results, #11738 (endor, default load, 21:22-21:41; code a6e3cc43a vs main 08ae803e0, both built fresh; binaries differ:
merged 26d1ad1a…, main 2324bd77…): all pass; MG4 after a script fix, evaluated on the job's saved replies.**
- MG1 PASS: md5 default == `--spec off` == main at 64 (65cde8f3…) and 256 (61fd59fe…); the default run printed the line naming
  `[tuning.metal] mtp_default_k` and speculated (57 verify passes, acceptance 80.7 %, 2.61 tokens per pass); `--spec off` and main
  printed no default line.
- MG2 PASS: `--max-batch 16`, main and merged: no default line, batching up to 16, spec steps 0; the 4 concurrent greedy
  completions equal main's [True, True, True, True].
- MG3 PASS: default server, ~1.4k greedy HTTP 200 with verify passes 0 -> 13; t1.0 HTTP 200, steps unchanged (13).
- MG4 PASS (re-evaluated): the job printed "FAIL ... t2 reply differs", but the replies are byte-identical. The script's
  non-empty check read only `content`; turn 2 hit `max_tokens` 128 inside the reasoning (content "" on both servers, reasoning 547
  chars). Fixed check: reasoning and content both equal, and the reply (reasoning or content) non-empty. On the saved replies: turn 1
  (189 tokens) and turn 2 (128 tokens) equal, turn 2 cached 4575 of 4633 on both, TTFT 0.136 s on both. The default server
  speculated over the flow (spec steps 0 -> 75), so the draft stays in use when a chat turn resumes at the prompt end (#499).
- MG5 PASS: no `mtp.safetensors`: rc 0, one "not applied: no DSpark draft" line, no speculation, md5 = main's.
- MG6 PASS: dspark_default (1, the extended unit test), recipe (17), recipes_doc (3), recipe_gumbel (1), mtp (69), docs_env_vars
  (2); on Metal inkling2_dspark (4), inkling2_dspark_cut (1), inkling2_verify_softcap (1), inkling2_chat_reuse (6), inkling2 (17),
  inkling2_batch (8).

**Coordinator on #496 (2026-11-22 ~21:50): gates accepted; HOLD on publishing.**
1. Rerun MG4 clean with the fixed check (fleetq, endor, `--quiet gpu,cpu`, `--expect 30`, behind #11739); record it here.
2. Main moved to dabd0d448 (#502: tests and docs only). After #11621 passes: merge origin/main in, rerun md5 64 / 256 and the Metal
   test set of MG6 on the merged tree before any push.
3. **HOLD: no push, no PR, no merge until the coordinator relays the user's own go.** The repo is public; pushing to it is
   publishing. This replaces step 4 of the brief above (fast-forward and push after #11621). If #11621 fails the bar: stop and
   report.

**MG4 clean rerun, #11751 (endor `--quiet gpu,cpu`, 23:34; binary a6e3cc43a from #11738, md5 26d1ad1a…): PASS.** Turn 1 (prompt 4575,
189 tokens) and turn 2 (prompt 4633, 128 tokens, reasoning only: content "" on both, finish length) equal on the default and the
`--spec off` server; turn 2 cached 4575 on both. The default server printed its line and speculated over the flow (spec steps
0 -> 75); `--spec off` did not (0). Info, one cold request per server, no warm-up (not a speed measurement): turn-1 TTFT on 2.073 s
vs off 1.617 s here (the default arm ran first in the slot), 1.636 vs 1.652 s in #11738; turn-2 TTFT 0.153 vs 0.135 s here, 0.136
vs 0.136 s in #11738. #11621 measures TTFT with a warm-up and A B B A.

**Coordinator (2026-11-22 ~23:45): MG4 accepted; correction to step 2.** After #11621 passes, merge in whatever origin/main is at
that point, not dabd0d448 (main is now ca3b4e55b, #503 the Metal prefill blend in `inkmix.rs`; rank4, the Metal router in
`inkctrl.rs` and `inkling2_ctrl.metal`, is about to merge). Both touch the Metal path, so on the merged tree rerun md5 64 / 256
(default == `--spec off` == that main) and the full MG6 Metal set. The report includes #11621's TTFT spec on vs off (the cold 2.07 vs
1.62 s of #11751 needs a proper reading). The hold stays: no push, PR or merge until the user's go is relayed.

**#11621 (Mon 05:45) did not measure: aborted in its box check on a parse bug, not on the box.** All three box-check runs decoded
at 76.69 / 76.56 / 76.69 tok/s (bar 65), but the script read the rate only from a `decode overall:` line, which `generate` prints
for long prompts; the 19-token box prompt prints it on the finish line (`| decode 3325 ms (76.69 tok/s)`). The script read "?"
three times and exited 6 before any server started. Fixed in `.worktrees/ink-spec-flip/.scratch/flip_night.sh` (scratch only;
`ink/spec-flip` and its code hash 34525d854 untouched): it falls back to the finish line; checked on the three saved logs (76.69,
76.56, 76.69). The bar and every other step are unchanged. Rerun needs a 75-minute quiet slot on endor.

**Coordinator (2026-11-23 ~05:55): option 1** (the rerun at 07:05 in place of #11653, #11653 after #11812; touch no other lane's
job; `--quiet gpu,cpu`, 75 min; a real box-check failure (< 65 tok/s) stops and reports, no retry; hold stays). **Not achieved:** I
cancelled #11653 before booking 07:05, and fleetq at once started the fp8 lane's waiting #11821 (deadline 09:00), with #11822
behind it, so 07:05 was no longer free without stopping another lane's job. Booked instead: the rerun **#11828 at 11:15** (75 min,
after #11812), the spec-k sweep **#11829 at 12:35** (110 min). Result of the bar around 12:30.

**Coordinator heads-up (2026-11-23 ~06:10):** the CUDA lane found that sampled MTP drafts (`mtp.rs` ~1113, ~1157) may read a full
logits row after a screened head, where only candidate logits are valid; fix on `ink/screen-full-logits`. Keep #11828 as booked;
in the bar report give the t=1.0 acceptance rate separately and say whether the flip ran with the head screen on; if the fix lands
before 11:15, the coordinator may say to merge main in first. Reading of the flip's own runs (#11620 gate logs, binary 34525d854):
the head screen is on ("screened greedy head on", the Metal default since #489) and the device sampler is on; at t=1.0 the DSpark
default proposes nothing (`spec: verify passes=0 proposed=0 accepted=0` at 150 and 5000 prompt tokens), since sampled requests
take plain steps under the default. So the t=1.0 A/B is plain decode on both arms; there is no t=1.0 acceptance rate to report,
and the counters will show it. Inkling-2 has no `[mtp]` head, so the `mtp.rs` sampled-draft path does not run for it.

## 1. Acceptance (Python oracle, #10743 on spark2's CPU)

`tools/inkling2_dspark_ref.py` (modes `taps`, `draft`, `analyze`, `tiny`) runs the target oracle (`inkling2_ref.py`, W4A16,
fp32 math) over fixed token streams. It saves the four DSpark taps per position (the partial entering visits 5, 23, 41, and the
final-norm output), the target argmax and the routing. It then drafts at every anchor exactly as vLLM's decode step does: context
K/V from the taps of positions < t, block = anchor + 7 masks, f_rnn from position t-1, 7 sequential Markov steps.

Streams: vLLM's greedy outputs on the 12 greedy golden prompts (2564 anchors on the target's greedy path) and the two human prose
texts of the long golden (875 anchors). A draft prefix counts as accepted when every draft equals the target's argmax and the text
stays on the target's path. Off-path anchors (where vLLM's W4A4 near-ties differ from the W4A16 oracle) are reported separately.
Raw output: `docs/results/2026-11-21-inkling2-mtp/analysis.{md,json}`.

P(at least j drafts accepted):

| set | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---:|---:|---:|---:|---:|---:|---:|
| greedy golden | 0.722 | 0.521 | 0.384 | 0.293 | 0.230 | 0.189 | 0.152 |
| prose | 0.408 | 0.158 | 0.067 | 0.030 | 0.009 | 0.002 | 0.000 |
| all on-path | 0.642 | 0.428 | 0.303 | 0.226 | 0.174 | 0.141 | 0.113 |

On greedy, the acceptance of position j given j-1 is 0.72, 0.72, 0.74, 0.76, 0.79, 0.82, 0.80. Calibration of the position-1
top-1 probability (greedy, bucket: accept rate): <0.3: 0.17, 0.3-0.5: 0.29, 0.5-0.7: 0.43, 0.7-0.8: 0.53, 0.8-0.9: 0.62,
0.9-0.95: 0.75, >=0.95: 0.95 (1382 of 2564 anchors).

Expert union per visit over R consecutive rows (real routing, 8 of 128 per visit):

| R | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| NVFP4 visits | 8.0 | 12.5 | 16.0 | 18.9 | 21.4 | 23.7 | 25.7 | 27.6 |
| MXFP8 visits | 8.0 | 12.6 | 16.3 | 19.2 | 21.7 | 23.9 | 25.9 | 27.7 |
| independent routing | 8.0 | 15.5 | 22.5 | 29.1 | 35.3 | 41.1 | 46.5 | 51.6 |

Consecutive tokens share experts strongly: at R = 2 the second row adds 4.5 new experts, not 7.5.

Tokens per verify step (accepted + 1) and modelled speedup. The cost model is the m-moe expert timings at the real union, the
m-attn attention per row, the dense bytes at 480 GB/s plus 1.5 ms fixed (14.4 ms plain step), and the native drafter (block 0.52
GB, base logits 0.92 GB, fold 0.1 GB per step). "Cheap" is a 0.5 GB drafter (perf-strategy §6). These are modelled numbers:
the measured ones are in section 3.

| policy (greedy) | mean k | tokens / step | speedup | cheap drafter |
|---|---:|---:|---:|---:|
| fixed k=1 | 1.00 | 1.722 | 1.23 | 1.38 |
| fixed k=2 | 2.00 | 2.242 | 1.46 | 1.64 |
| fixed k=3 | 3.00 | 2.626 | 1.59 | 1.78 |
| gate p1 >= 0.6, kmax 2 | 1.42 | 2.121 | 1.46 | 1.64 |
| gate p1 >= 0.6, kmax 3 | 1.94 | 2.460 | 1.62 | 1.83 |
| gate p1 >= 0.8, kmax 3 | 1.54 | 2.298 | 1.57 | 1.77 |
| gate p1 >= 0.6, kmax 4 | 2.37 | 2.720 | 1.74 | 1.96 |
| gate p1 >= 0.6, kmax 7 | 3.29 | 3.223 | 1.95 | 2.21 |

On prose every policy is 0.99-1.09x modelled (fixed k=3: 0.99; gate 0.6 kmax 3: 1.07). The gate keeps prose from going below 1.

## 2. The Rust drafter

`--spec dflash --spec-draft <Inkling model dir>`: the loader (`intarsia-models/src/load.rs:open_dspark`) sees `mtp_config` +
`mtp.safetensors` and loads the draft as a DFlash-family head with `DflashConfig::dspark`. The engine's DFlash verify, accept
and rollback path is used unchanged. The Inkling refusal of drafters now admits only this draft (MTP and other DFlash drafts are
still refused). Code: `crates/intarsia-core/src/model/dspark.rs`.

- Taps: `ink_pass` copies the partial entering visits 5, 23, 41 and the final-norm output into the DFlash staging buffers
  (`model/inkling2.rs`). This is vLLM's `aux_hidden_states`: `stack[num_blocks]` at those visits, none of which is a pass
  restart, plus the output-norm hidden.
- Context ingest (after every forward, the accepted rows only): taps downloaded, the four-tap softmax(fuse_weight) mix +
  dflare RMSNorm on the host (bf16-rounded), uploaded, `k_proj_ctx` / `v_proj_ctx`, `k_norm_ctx` + neox RoPE (theta 1e6),
  appended to per-layer draft caches. `f_rnn` rows go to a 64-row host ring.
- Block: `[embed_norm(embed(anchor)), mask_embed x 7]` at positions P..P+7. Pre-norm layers with q/k norm + RoPE, block K/V
  appended after the context. Each row is one `AttnDecode` at P+7 (every block row sees all 8 block keys: bidirectional);
  layers 0 and 2 use a window of 1024 + 8 keys. SwiGLU MLP, final norm.
- Head: base logits of the mask rows through the draft's own MXFP4 copy of the target unembed (`INTARSIA_DSPARK_HEAD`),
  device top-64 per row (`Backend::topk_rows`, new on Metal, `kernels/metal/topk.metal`). The rnn_proj hidden / f_rnn parts
  run as two small GEMVs. The rank-256 recurrence runs on the host with vLLM's bf16 rounding points. For each step, the Markov
  bias `fold(latent)` is evaluated on the 64 candidates only. fold = unembed . hidden_out, folded at load: 200058 x 256,
  4.7 s on endor's CPU.
- Gate: stop at the first step whose top-1 probability (softcapped logits over the candidates) is below
  `INTARSIA_DSPARK_TAU` = 0.6. An empty draft is a plain step.
- Opt-ins: `INTARSIA_DSPARK_CAND=0` (exact full-vocab head), `INTARSIA_DSPARK_LAYERS=fp8`, `INTARSIA_DSPARK_DEVICE_TOPK=0`.
- Greedy only: `--spec dflash` never speculates on sampled requests (`engine.rs`, sampled speculation is `SpecMode::Mtp`
  only), so a temperature > 0 request takes plain steps and its distribution is unchanged. The CLI / serve `--spec-draft`
  help says so.

Fable §4.3 (drafter rows sharing the verify's unembed) is circular for this draft. The block's row 0 is the embedding of the
target's argmax at the last accepted verify row, and that argmax exists only after the verify's unembed; attention spreads
row 0 into every mask row from layer 0 on. The separate MXFP4 draft head cuts the read from 0.92 to 0.24 GB per cycle
instead. Emitted tokens are still the bf16 verify argmax.

Tests:
- `crates/intarsia-models/tests/inkling2_dspark.rs`: tiny target + random 3-layer DSpark, `golden/inkling2-tiny-dspark`,
  written by `tools/inkling2_dspark_ref.py tiny`. Greedy ids equal spec off at k 1 / 3 / 7 and max_tokens 1 / 2 / 5 / 40;
  scripted partly-right drafts give every accepted length. Passes on cpu and Metal.
- `crates/intarsia-backend-metal/tests/topk.rs`: `topk_rows` equals `sampler::topk_row` (ties, -inf, NaN, V 200058,
  f32 / bf16).
- Real-model token identity: section 3.

## 3. Metal: token identity and speed (endor, M5 Max)

### Token identity

Every DSpark run below produced the same ids as `--spec off` on all 12 greedy golden prompts x 128 tokens and on both prose
texts x 128. That covers k 1-7, tau 0.4-0.8, mxfp4 / bf16 heads, fp8 layers, the exact head (`CAND=0`) and the gate-off k1.
Runs: #10790 (k3 tau 0.6, k1 tau 0) and #10806 (all of the timing table). CPU backend, real model: 2 greedy prompts x 32,
identical (#10816 on spark2's CPU; 1.21 -> 1.99 tok/s, 3.59 tokens per pass).

Engine acceptance at k1 with the gate off is 71.1%, against the oracle's 72.2% at position 1, so the Rust drafter's numerics
match vLLM's (#10790, bf16 exact head).

### Decode tok/s (#10806, `--quiet gpu,cpu`, `--profile-every 0`, frozen binary c770aaef3)

Greedy golden, 12 prompts x 128 tokens (1501 tokens after each prompt's first):

| policy | tok/s | vs off | tokens / verify pass | acceptance |
|---|---:|---:|---:|---:|
| spec off | 60.21 | 1.00 | - | - |
| k1, tau 0.6 | 63.71 | 1.06 | 1.80 | 79.6% |
| **k2, tau 0.6** | **71.27** | **1.18** | 2.23 | 67.4% |
| k3, tau 0.6 | 68.50 | 1.14 | 2.40 | 56.9% |
| k3, tau 0.8 | 69.65 | 1.16 | 2.55 | 71.7% |
| k4, tau 0.6 | 69.24 | 1.15 | 2.53 | 52.0% |
| k7, tau 0.6 | 63.75 | 1.06 | 2.63 | 42.2% |
| k3, tau 0.6, bf16 draft head | 67.12 | 1.11 | 2.47 | 59.7% |
| k3, tau 0.6, fp8 draft layers | 63.98 | 1.06 | 2.41 | 57.5% |

Prose (the two human texts, 128 tokens each):

| policy | tok/s | vs off | tokens / pass |
|---|---:|---:|---:|
| spec off | 61.55 | 1.00 | - |
| k3, tau 0.6 | 66.00 | 1.07 | 2.33 |
| k7, tau 0.6 | 63.30 | 1.03 | 2.52 |

Drafter profile (k3 tau 0.6, `INTARSIA_DSPARK_PROFILE=1`, ms per cycle, synced; 522 cycles):

| part | ms | where |
|---|---:|---|
| ingest (taps down, host mixes, upload, ctx K/V projections) | 0.94 | host + GPU |
| block pass (3 layers x 8 rows, 24 AttnDecode calls) | 2.64 | GPU |
| head (MXFP4 base GEMM over k rows + device top-64) | 1.34 | GPU |
| Markov step 0 + small downloads | 0.70 | host |

The drafter costs about 5 ms a cycle against a 16.6 ms plain step. That is why the gain stops at about 18% while the oracle's
tokens per pass (2.2-2.6) promised more. Most of it is GPU work (block 2.6 ms, head 1.3 ms), not host syncs (about 1.2 ms of
host time in total). Verify and the k choice are behind it: k2 is the best fixed setting, and k7 pays for 8-row verifies that
mostly reject.

### Decode tok/s after the drafter work (#10874, `--quiet gpu,cpu`, binary 17c67dacf)

Same sets and flags as #10806. 17c67dacf adds the row-exact draft kernels (2-16 rows), the vectorised Markov head and the merge
of ink/metal (rows-exact ConvMix, serve fix). The spec-off baseline also moved (60.21 -> 62.58), so compare within this table.
Every row is token-identical to spec off.

| set, policy | tok/s | vs off | tokens / verify pass | acceptance |
|---|---:|---:|---:|---:|
| greedy, spec off | 62.58 | 1.00 | - | - |
| greedy, k2 tau 0.6 | 79.84 | 1.28 | 2.23 | 67.5% |
| greedy, k3 tau 0.6 | 80.46 | 1.29 | 2.40 | 57.0% |
| **greedy, k3 tau 0.8** | **80.64** | **1.29** | 2.54 | 72.0% |
| prose, spec off | 63.35 | 1.00 | - | - |
| prose, k2 tau 0.6 | 74.46 | 1.18 | 2.16 | 67.8% |

Drafter per cycle (k3 tau 0.6, profile on, 525 cycles): block 1.75 ms (was 2.64), head 1.20 ms (1.34), Markov 0.59 ms (0.70),
ingest 0.75 ms (0.94): about 4.3 ms, down from 5.6. k2 and k3 are now within 1%; tau 0.8 trades fewer drafts for the same speed.

Verify routing is the same as in #10806 (both runs print the same verify kernels: `matmul_bf16_gemv_rows`,
`ink_in_proj_bf16_gemv_rows`, `moe_experts_situ_bf16_rows` / `moe_experts_mxfp8_bf16_rows`, `logits_bf16_gemv_rows`, attention
one `ink_attn_decode_bf16` call per row). From the #10874 totals, with the plain steps (16.0 ms) and the drafter (4.3 ms) taken
out, a verify pass costs about 21 ms at 2.8 rows (k2) and 23 ms at 3.5 rows (k3): about +2.9 ms per extra row (ink-batch:
about 2.3 ms per extra row in its batched step). Per-op attribution at T = 2 / 3 / 4: #10990 (first try #10897 printed no op
table, because `generate --prompt-ids` never printed one; fixed in a3b72495b).

### serve: the draft under admitted kernels only (a3b72495b)

#10874's localeval half failed on the DSpark server: `no kernel for qk_norm on metal/applegpu_g17s (m=256 ...) [admitted kernels
only]`. The plan dry run skipped the DSpark draft, so admission never tested its kernels, and `serve` selects only admitted
ones. `generate` does not require admission, which is why the timing and identity runs never hit this. Fix: the dry run runs
the draft's device calls with its host steps skipped: ingest at 1-16 rows and powers of two up to the store, and the block pass +
head at every draft length, at the start of the context and past the local layers' window. Test
`dspark_runs_with_admitted_kernels_only` (tiny, `PlanOptions { admit, require_admitted }`) fails without the fix with the same
error and passes with it (cpu, spark2 #10975 / #10979).

Spec-off localeval speed-plain (17c67dacf, `20261121-225828-adhoc-endor-intarsia-metal-ink-mtp-off`, decode tok/s median):
63.8 at 146 prompt tokens, 61.9 at 1023, 60.7 at 4110, 57.4 at 16456. The off vs DSpark k3 tau 0.8 rerun is #10991.

### Default-on flip (prepared, not on this branch)

Local branch `ink/mtp-flip`, commit f130b8855 (on a3b72495b): without `--spec`, Inkling-2 on Metal at one rank in process
and `--max-batch` <= 1 runs DSpark k3, and the gate default moves 0.6 -> 0.8. Sampled requests stay plain, and a batching
server has no draft. `--spec off` or `INTARSIA_SPEC_DEFAULT=0` turns it off. docs/tuning.md rows; unit test
`dspark_default_only_where_measured`. It is cherry-picked onto ink/mtp only if #10991 shows on >= off at every length. #10990
checks default-on ids against `--spec off`, plus greedy ids and `plan show` with `INTARSIA_SPEC_DEFAULT=0` against a3b72495b.

### Verify per-op attribution (#10990, attribution only)

One greedy prompt x 64 tokens, gate 0 (every pass verifies k + 1 rows), `--profile-every 1`. Every op is synced and timed, so
the absolute ms are inflated (79 ms per step against 16 ms untimed). Only the shares and growth mean anything. ms per pass:

| op | 1 row (off) | 2 rows (k1) | 3 rows (k2) | 4 rows (k3) |
|---|---:|---:|---:|---:|
| moe_experts (NVFP4 visits) | 11.9 | 16.7 | 20.5 | 23.1 |
| moe_experts_mxfp8 | 1.4 | 2.2 | 2.9 | 3.1 |
| conv_mix | 15.7 | 17.1 | 17.3 | 17.8 |
| ink_in_proj | 10.1 | 11.0 | 11.6 | 11.6 |
| logits (target head + draft GEMVs) | 8.6 | 9.6 | 9.7 | 10.0 |
| ink_attn_decode | 8.1 | 8.6 | 8.4 | 8.7 |
| moe_route | 6.9 | 7.5 | 7.3 | 7.5 |
| matmul (target + draft) | 7.9 | 13.0 | 13.5 | 13.5 |

Going from 1 to 4 rows, the target's ops grow by about 18 ms (profiled). The experts are about 13 ms of that (70%), following
the expert union (8 -> 18.9 experts per visit at 4 rows, section 1). Attention, routing and in_proj barely move. The next lever
is the expert pass at 2-4 rows (a union-fused experts kernel that reads each expert once for all rows), not AttnDecode.

### Flip checks (#10990, flip binary f130b8855) and serve smoke (#10996)

- Default on (no `--spec`): the 12 greedy prompts x 128 and both prose texts give the same ids as `--spec off`. The loader
  prints the DSpark default line (k3; 459 passes, 72.0% acceptance, 2.54 tokens per pass).
- `INTARSIA_SPEC_DEFAULT=0`: no verify passes. Greedy ids md5 439a1865723f2bca3b6508f864e24af1 is the same as `--spec off` on
  a3b72495b, and `plan show` matches a3b72495b.
- serve smoke (a3b72495b, DSpark k3 gate 0.8, admitted kernels only; admission checked 33 kernels, 220 cases): a 1425-token
  greedy request returns HTTP 200 with 13 verify passes. The 16k request was over the 20480 context, so it was not run; the
  16456-token localeval cell in #10991 covers that length.
- Identity at a3b72495b: tiny Metal tests 3 / 3 pass, and greedy k2, k3, k3 bf16 head and prose k3 all match `--spec off`.

### localeval speed-plain, spec off vs DSpark k3 gate 0.8 (#10991, `--quiet gpu,cpu`, a3b72495b, serve --max-batch 1)

Decode and prefill tok/s, median (range):

| prompt tok | decode off | decode DSpark | x | prefill off | prefill DSpark | TTFT off / DSpark s |
|---:|---:|---:|---:|---:|---:|---:|
| 142 | 63.6 (63.6-63.9) | 80.2 (77.3-90.2) | 1.26 | - | - | 0.15 / 0.16 |
| 1021 | 61.6 (61.1-61.7) | 77.5 (70.6-81.2) | 1.26 | 1265 | 1270 | 0.81 / 0.81 |
| 4108 | 60.9 (60.4-60.9) | 87.5 (78.9-111.9) | 1.44 | 1481 | 1376 | 2.78 / 2.98 |
| 16455 | 57.1 (56.2-57.1) | 72.1 (69.9-74.3) | 1.26 | 1466 | 1375 | 11.22 / 11.97 |

Server counters, DSpark arm: 2255 verify passes, 4517 proposed, 3486 accepted (77%, 2.55 tokens per pass). Off arm: 0.

- Decode: DSpark is faster at every length, and its slowest rep beats off's fastest at every length.
- Prefill: DSpark is 7% / 6% slower at 4k / 16k (TTFT +0.2 s / +0.75 s). The draft's context ingest (projections of every
  prompt row into its 3 layers' caches) runs after each prefill chunk.
- Suite status: neither run is a suite result. Off had its 1024 cell rejected for prefill spread. DSpark had its 128 / 1024 /
  4096 cells rejected for decode spread (acceptance varies by content, so per-request decode varies), and its arm was relabelled
  `spec:unknown` because the run was not labelled as a spec lane.
- Flip: not applied. The brief's "on >= off at every length" holds for decode but not for prefill at 4k / 16k, so it goes to the
  coordinator (f130b8855 stays on `ink/mtp-flip`).

### Prompt-time ingest (brief of 2026-11-22)

(1) Requests that can never speculate. A sampled request (temperature > 0, the serve default t1.0) has `spec_on = false`
(`engine.rs`: `spec_on` requires `greedy`), so its draft state is admitted with `active = false`, and
`dflash_after_forward` returns before any ingest. The draft never ingested sampled prompts. A batched server cannot load the
draft at all (`--spec dflash` with `--max-batch` > 1 is refused, and the default needs `--max-batch` <= 1). One cost did remain:
`ink_pass` copied the four taps into the staging buffers on every pass whenever the draft was loaded (4 x t x H bf16 per pass).
These copies now run only when the request drafts.

(2) Greedy. (i) does not apply in full: draft layers 0 and 2 are windowed (1024), but layer 1 attends the whole context, so
every prompt row needs its layer-1 K/V. Instead the ingest no longer goes through the host:
- the taps are one tap-major buffer per pass;
- each layer's mix is `HcPre` with the fixed softmax(fuse_weight) weights, then `RmsNorm` with its `dflare_norm`;
- `f_rnn` goes into a per-sequence device ring, and the draft reads its row there;
- the decode-row stash is row-major `[rows, taps, H]`, so a 1-row pass is one copy and a flush is one `HcPre` over the stash.

Before, the ingest was one host sync per prefill chunk, four tap downloads, the host mixes and three uploads; now it is device
work in the chunk's command stream ((ii)). The new Metal kernel `hc_pre_bf16` serves the existing op `HcPre` (no new op): an
f32 fma chain over the taps, one bf16 rounding, bitwise the host sum (test `hc_pre_matches_host_sum`). The mix now rounds to
bf16 once before the norm, which the host version did not; drafts only propose, so this can move acceptance but never a token.

### Plan: the experts at 2-4 verify rows (brief item 5, not built)

What exists. Verify rows already run union-grouped experts by default (`inkmoe.rs` `Policy::Auto`: union gate+up, then
union down by pair + gather for T >= 2, fused down at T = 1). Each expert's weights are read once per call for all the rows
routed to it; ink-m-moe measured this at 488-493 GB/s (NVFP4) at 4 rows, against about 530-550 GB/s streaming. So the
"union-fused pass" the brief names is mostly in place. The op table agrees: experts grow 11.9 -> 23.1 (x1.94) from 1 to 4
rows while the real union grows 8 -> 18.9 (x2.36), so the cost follows the bytes of the union.

Floor arithmetic, real routing, NVFP4 expert 4.48 MB:

| | 1 row | 4 rows |
|---|---:|---:|
| bytes per visit (8 vs 18.9 experts) | 35.8 MB | 84.7 MB |
| at the measured GB/s (430 / 490) | 83 µs | 173 µs |
| at 550 GB/s | | 154 µs |

Over 44 NVFP4 visits the 4-row pass pays about +4.0 ms for experts, and the 4 MXFP8 visits about +0.7 ms. Reaching streaming
speed would save at most about 0.8 ms of a 4-row pass (3-4%). The union itself (which experts, how many bytes) is fixed by the
routing.

Options, by expected gain per 4-row verify pass:
1. **Measure first** (one default-load job): per-visit µs at T = 1..4 with real routing ids (the ink-m-moe bench used random
   routing: union 29.5 at 4 rows, not 18.9), the gate+up / down split (`set_probe`), and policies union / union-fused.
   Building more is only worth it if this shows the 4-row path well below about 490 GB/s.
2. **Union down fused with the k-order token sum** (a `dn_f` that is union-grouped): removes the gather launch and the `y`
   round trip per visit. About 5-10 µs x 48 visits = 0.25-0.5 ms (1-2%). Bitwise by construction (same per-pair order, token
   sum in k order), checked by `tests/inkmoe.rs`.
3. **Gate+up / down tuned for unions of 12-19 experts** (threadgroup shape, chunks in flight): up to the 0.8 ms streaming
   bound (3-4%).
4. Outside the experts, the per-pass fixed cost is the bigger target at k2-k3: conv_mix (15.7 profiled ms per pass),
   ink_in_proj and the head barely grow with rows, but they are paid on every pass. That is the ink-metal / hostgap lanes'
   work, not this lane's.

Recommendation: run (1); build (2) if the probe shows the gather and `dn_u` above about 10% of the experts time at 4 rows,
else stop here and report. Expected total from (2) + (3): about 1-5% of the verify pass, 1-4% of decode tok/s.

### Adaptive draft length, offline simulation (brief item 3, design only, CPU: spark2 #11150)

`tools/inkling2_dspark_adaptive.py` walks each text as the engine does. From an anchor, a cycle picks k, accepts
min(m(t), k) of the oracle's drafts, and moves on by accepted + 1. Cost model, fitted to the measured Metal runs:
- a plain step is 16.0 ms, each extra verify row +2.9 ms;
- the drafter is 3.8 ms per cycle it runs (block, head, Markov steps, ingest);
- a skipped drafter (k = 0) pays only the ingest, 0.7 ms.

The model reproduces the measured fixed policies on greedy within about 1-2%: k3 tau 0.8 81.7 modelled vs 80.64 measured,
k2 tau 0.6 80.9 vs 79.84. Ids are identical for every policy. Policies:
- **cumulative p1:** the drafter runs every cycle; draft j is kept while the product of drafts 1..j's top-1 probabilities is
  >= c (an estimate of P(all j accepted)). A row pays when P x ms per token (about 12.4) > 2.9 ms, i.e. c of about 0.23.
- **EMA:** a running per-draft acceptance estimate a picks kmax in 0..K by expected tokens per ms; k = 0 skips the drafter,
  with a k = 1 probe after 8 idle cycles.

Greedy golden (12 texts, the model's own greedy path), modelled tok/s:

| policy | tok/s | vs k3 tau 0.8 | tokens / verify pass | rows / pass |
|---|---:|---:|---:|---:|
| spec off | 62.5 | 0.765 | - | 1 |
| fixed k2 tau 0.6 | 80.9 | 0.990 | 2.25 | 2.79 |
| **fixed k3 tau 0.8 (chosen)** | 81.7 | 1.000 | 2.56 | 3.12 |
| fixed k3 tau 0.6 | 84.2 | 1.029 | 2.50 | 3.35 |
| cumulative p1 >= 0.35, kmax 3 | 84.9 | 1.039 | 2.33 | 3.38 |
| cumulative p1 >= 0.5, kmax 3 | 85.2 | 1.042 | 2.42 | 3.25 |
| cumulative p1 >= 0.35, kmax 4 | 86.1 | 1.053 | 2.47 | 3.77 |
| cumulative p1 >= 0.5, kmax 4 | 87.0 | 1.064 | 2.57 | 3.58 |
| EMA lambda 0.1, kmax 3, tau 0.6 | 84.1 | 1.029 | 2.48 | 3.23 |
| EMA lambda 0.1, kmax 4, cumulative p1 >= 0.35 | 85.0 | 1.040 | 2.35 | 3.29 |

Oracle "prose" (2 human texts, teacher-forced): every policy is 0.90-1.04x of k3 tau 0.8, and all are at or below spec off
(59-62 vs 62.5 modelled). The best are EMA lambda 0.1 + cumulative p1 >= 0.25 (62.2) and EMA lambda 0.2 tau 0.6 (61.4).
This set is not what decode sees. It scores the draft on human-written text (P(>=1) 0.408), while decode follows the model's
own greedy continuation. The engine measured 67.8% acceptance and +18% tok/s on its continuations of these prompts (k2
tau 0.6, #10874), and localeval +26% at 1-16k. So the prose rows are a lower bound for text the model did not write, e.g. a
rewrite, not a forecast of decode speed.

Reading:
- The best greedy policy gains about 4-6% over k3 tau 0.8: a per-row cumulative cut (c 0.35-0.5) with kmax 4.
- The EMA alone adds nothing on greedy; on hard text it can skip the drafter, which none of the others do.
- The model's gate sensitivity is uncertain. It says tau 0.6 beats 0.8 by 3%, but on Metal they measured equal (80.46 vs
  80.64). Gains of this size are within the model's error until measured.

**Correction (refit, spark2 #11165 / #11210).** The flat 3.8 ms drafter overrated the policies that draft more (k3 tau 0.6:
84.2 modelled vs 80.46 measured, a 4.6% miss, the size of the gain above). The refit takes the drafter from its profile split:
- 2.5 ms per cycle (block + ingest);
- 0.40 ms per head row (the cap k);
- 0.25 ms per Markov step run.

It fits one parameter: 3.16 ms per extra verify row. Measured points: k2 tau 0.6 80.06 vs 79.84, k3 tau 0.6 81.24 vs 80.46,
k3 tau 0.8 79.38 vs 80.64 (all within 1.6%); k4 / k3 at tau 0.6 1.001 vs 1.011 measured (#10806). Refit, vs k3 tau 0.8:

| policy | greedy tok/s | vs base | oracle prose tok/s | vs base |
|---|---:|---:|---:|---:|
| fixed k3 tau 0.8 (base) | 79.4 | 1.000 | 58.8 | 1.000 |
| fixed k3 tau 0.6 | 81.2 | 1.023 | | |
| fixed k4 tau 0.6 | 81.3 | 1.024 | | |
| cumulative 0.35, kmax 3 | 81.5 | 1.026 | 57.5 | 0.978 |
| cumulative 0.5, kmax 3 | 82.1 | 1.034 | 58.9 | 1.002 |
| cumulative 0.35, kmax 4 | 81.0 | 1.020 | 56.2 | 0.956 |
| cumulative 0.5, kmax 4 | 82.2 | 1.036 | 57.6 | 0.981 |

The modelled gain is +2-4%, no longer +4-6%, which is inside the model's error (the fit misses k3 tau 0.8 by -1.6% and k3
tau 0.6 by +1.0%). The endor sweep #11188 decides. On oracle prose, c 0.35 costs 2-4%; c 0.5 is neutral.

Recommendation, not built (superseded by the 2026-11-22 brief: cut built, sweep queued):
1. Implement the cumulative cut (c, kmax as env knobs, default off): a few lines in `dspark_draft_inner`'s gate.
2. Measure c 0.35 / 0.5 at kmax 3 / 4 against k3 tau 0.8 in one quiet job, with real prose continuations (prose.json).
3. Add the EMA skip only if a workload with low acceptance (rewrites, code edits) turns out to matter.

### Device ingest: gates A and B (85265f9d5)

**Gate A** (#11184, default load, not timing):
- Metal tests: `hc_pre_matches_host_sum` passes, the kernel list pin passes, tiny Metal DSpark 3 / 3; `kernels test --op hc_pre`
  ADMITTED (3 cases, 0 error).
- Ids: equal to spec off on all 12 greedy golden prompts and both prose prompts x 128, for every policy.
- Acceptance against the host-ingest runs:

| policy | device ingest | host ingest |
|---|---:|---:|
| k3 tau 0.8 | 71.8% | 72.0% |
| k2 tau 0.6 | 67.5% | 67.5% |
| k1 tau 0 | 69.4% | 71.1% (earlier run) |
| prose k3 tau 0.8 | 72.7% | |

  k1 tau 0 is 1.7 pp lower, the only gap over 1 pp; the k2 / k3 policies match within 0.2 pp. Known difference, not blocking
  (not the flip's config): the device mix rounds to bf16 once before the norm, which the host mix did not, and k1 tau 0 drafts
  on every step, so it shows the most. The host number is from an earlier run, not interleaved.
- Ingest of a prefill chunk: 2.25 ms (synced).
- serve smoke under admitted kernels only (34 kernels, 221 cases): greedy 1.4k and 9.8k-token requests HTTP 200 with verify
  passes; a t1.0 request HTTP 200 with no change in the spec counters (sampled requests do not draft).
- The long-prompt `generate` runs failed: generate's default context is 4096 (fixed with `--max-seq 20480` in later jobs).

**Gate B** (#11186, quiet). localeval speed-plain, both runs valid (no cells rejected); the DSpark arm verified as
`spec:dspark` from the server counters (1852 passes, 73.8% acceptance, 2.38 tokens per pass). Median (range):

| prompt tok | decode off | decode DSpark | x | prefill off | prefill DSpark | TTFT off / DSpark s |
|---:|---:|---:|---:|---:|---:|---:|
| 144 | 63.7 (61.8-64.2) | 74.4 (72.6-74.7) | 1.17 | - | - | 0.15 / 0.15 |
| 1021 | 61.7 (60.2-62.8) | 76.0 (68.9-82.5) | 1.23 | 1408 (1324-1517) | 1217 (1162-1243) | 0.72 / 0.84 |
| 4109 | 60.6 (60.5-61.0) | 81.6 (69.6-82.5) | 1.35 | 1460 (1438-1623) | 1377 (1369-1390) | 2.81 / 2.98 |
| 16456 | 57.1 (55.6-57.8) | 74.2 (72.3-79.2) | 1.30 | 1432 (1386-1462) | 1402 (1376-1441) | 11.49 / 11.74 |

- Prefill: DSpark -14% / -6% / -2% at 1k / 4k / 16k, against -6 / -7% at 4k / 16k with the host ingest (#10991). Off's own
  1k range is 15% wide.
- Engine-reported prefill at 1k (generate, 3 reps interleaved): greedy off 825-838 ms vs DSpark 832-835; sampled off 831-844
  vs DSpark 833-834, so equal within the spread. The 4k / 16k engine runs failed on the 4096 default context; #11226 reruns
  them.
- Decode at 144 tokens is lower than #10991's (74.4 vs 80.2) with lower acceptance (73.8% vs 77%: localeval's nonce-first
  prompts change content). #11226 separates code from content: ABAB decode, the device- vs host-ingest binary, greedy golden.
- Flip: not yet. The 2% prefill gate is unresolved at 1k (localeval spread) and fails at 4k on localeval; it waits for
  #11226's engine numbers.

### Experts probe (gate A, default load, real routing from the oracle's expert_ids)

NVFP4, µs per call (mean; the bench's GB/s counts the union's bytes):

| T | union | Auto all | gate+up | down (dn_u + gather) | UnionFused all | Fused all |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 8.0 | 90.5 | 53.9 | 33.0 | - | 87.8 |
| 2 | 12.7 | 145.0 | 87.9 | 55.8 | 138.1 | 132.2 |
| 3 | 15.9 | 178.7 | 110.7 | 68.9 | 177.3 | 180.0 |
| 4 | 19.0 | 247.1 (min 230.0) | 142.8 | 89.0 | 225.1 (min 219.8) | 233.0 |

MXFP8 at T = 4: Auto 333.7, UnionFused 330.9, Pair 405.7.

At 4 rows the down phase is 89 of 232-247 µs (36-38%), far above the brief's 10% threshold, and runs at 318 GB/s against
gate+up's 397. Down bytes at 19 experts are 28.4 MB: about 52 µs at 550 GB/s, so up to about 30-37 µs per visit is left, or
1.3-1.6 ms per 4-row verify pass (44 NVFP4 visits, about 6%). Two steps:
1. Auto -> UnionFused at T 3-4: existing kernels, -9% experts time at T = 4 by the means, within noise at T = 3; needs a quiet
   A/B, the bench ran under load.
2. A down kernel that reads each union expert once and keeps the k-order token sum (bitwise): design in the message to the
   coordinator.

### Build brief: union-grouped fused down for the Inkling-2 experts (lane ink-mtp-dn, not started)

Starts only if the quiet probe #11231 (`.worktrees/ink-mtp/.scratch/gate_d_moe_probe.txt`) shows the down phase >= 10% of
experts time at T = 4; otherwise parked.

**Goal.** At 2-4 decode / verify rows (T), the NVFP4 down projection reads each union expert's weights once for all the rows
routed to it, and sums each token's K results in k order in the same launch. Target: down at T = 4 from about 80-89 µs to
about 55-60 µs per visit (floor about 52 µs: 19.0 experts x 4.48 MB / 3 at 550 GB/s). That is about 1.0-1.5 ms per 4-row
verify pass over 44 NVFP4 visits (about 5%); MXFP8 (4 visits) optional.

**Where.**
- Worktree `.worktrees/ink-mtp-dn`, branch `ink/mtp-dn` from `ink/mtp-cut` 5d73d0fa8. Do not push or merge; commit often.
- Kernel: `kernels/metal/inkling2_moe.metal`. The header comment (lines 1-30) is the contract. The existing down kernels are
  `ink_moe_dn_u_*` (down per pair, 4 lanes per row, union leaders via `INK_XR` = 4), `ink_moe_dn_f_*` (one threadgroup holds
  8 output rows of one token and all K slots, then sums in k order) and `ink_moe_gather_*`.
- Launcher: `crates/intarsia-backend-metal/src/inkmoe.rs`. The `SHADERS` list holds the pipeline names; `Policy` (Auto,
  Union, Pair, Fused, UnionFused) is set by `INTARSIA_INK_MOE` and `set_policy`; `DEC_MAX_ROWS` = 16; `set_probe` gives the
  gate+up / down split. Add a policy (e.g. `UnionDownFused`, `INTARSIA_INK_MOE=udf`) that runs the union gate+up and then the
  new down. Default stays `Auto` (flag default off).
- Tests: `crates/intarsia-backend-metal/tests/inkmoe.rs`. Add the policy to `POLICIES` (line 206), so
  `ink_moe_rows_are_bitwise_the_one_row_calls` (T 2..=16, routings random / identical / pooled / invalid ids) and
  `ink_moe_negative_control` cover it.
- Bench: `crates/intarsia-backend-metal/examples/inkmoe_bench.rs`, with `INKMOE_ROUTES` (real routing) and `INKMOE_PROBE=1`.

**Design.**
1. Grid: one threadgroup per block of 8 output rows (H = 2304, so 288 threadgroups), across all T tokens of the call.
2. The threadgroup walks the union of experts, in the same leader order as the union gate+up. For each expert it reads that
   block's down rows once and computes, from L1, every pair (token, k) routed to that expert.
3. Each pair's dot uses exactly `ink_moe_dn_u`'s lane-strided chunk order, fma chain, shuffle-xor reduction (1, 2, 4) and the
   `bf16(dot * gs)` rounding.
4. The y values sit in threadgroup memory: T x K x 8 bf16, 2 KB at T = 16.
5. After a barrier, each token's 8 rows are summed over k = 0..K-1 in k order in f32 and written as bf16, the same as
   `ink_moe_gather` / `dn_f`.
6. Invalid ids (negative or >= E) contribute nothing, as today.

**Contract (must keep).** Rows-exact as the `.metal` header states: a pair's value never depends on T, on the routing of other
tokens, on union or not, or on the kernel. Same chunk order, same fma chain, same reduction, same roundings; the token sum is
always in k order. So the new policy is bitwise every other policy, and a T-row call is bitwise the T one-row calls. No
atomics, no order-dependent reductions.

**Commands** (on endor every GPU run takes a fleetq slot: `fleetq run --host endor --expect MIN --label ...`, timing with
`--quiet gpu,cpu`; compile on bespin, CPU tests on spark2):
- Bitwise test: `cargo test --release -p intarsia-backend-metal --test inkmoe`.
- Admission: `target/release/intarsia kernels test --backend metal --op moe_experts` (and `--op moe_experts_mxfp8` if touched).
- Bench, quiet:
  `INKMOE_PROBE=1 INKMOE_ROUTES=.worktrees/ink-mtp/.scratch/routing_i16.bin:357,296,700 cargo run --release -p intarsia-backend-metal --example inkmoe_bench -- nv 1 2 3 4`
  (the routing file holds i16 expert ids [48 visits, P, 8] for three texts, from the oracle's `expert_ids`).
- Shaders compile with `xcrun -sdk macosx metal -c -std=metal3.1 -O3 -fno-fast-math`; `thread_position_in_threadgroup` needs
  `uint2` when mixed with `uint2` grid arguments.

**Gates.**
1. `tests/inkmoe.rs` bitwise with the new policy, T 2..=16, every routing.
2. Admission ADMITTED.
3. Quiet bench: down at T = 2 / 3 / 4 below the best existing policy by more than the spread, and T = 1 not slower.
4. Model-level: greedy golden x 128 token-identical to spec off, with `INTARSIA_INK_MOE=udf` and DSpark k3 tau 0.8, Metal.
5. Quiet tok/s ABAB, k3 tau 0.8, against `Auto`.

The default flip is a separate decision for the coordinator.

**Probe numbers** (gate A, default load, real routing; µs per call, mean):

| T | union | Auto all | Auto gate+up | Auto down | UnionFused all / down | Fused all / down | Pair all / down |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 8.0 | 90.5 | 53.9 | 33.0 | - | 87.8 / 33.1 | 89.1 / 34.2 |
| 2 | 12.7 | 145.0 | 87.9 | 55.8 | 138.1 / 50.6 | 132.2 / 50.2 | 143.9 / 48.4 |
| 3 | 15.9 | 178.7 | 110.7 | 68.9 | 177.3 / 70.0 | 180.0 / 65.6 | 178.7 / 62.9 |
| 4 | 19.0 | 247.1 (min 230.0) | 142.8 | 89.0 | 225.1 / 80.3 | 233.0 / 79.8 | 234.6 / 80.2 |

Down runs at 318-354 GB/s at T = 4, against gate+up's 379-403. The quiet repeat (#11231) replaces these numbers when it lands.

## Landing as opt-in (lane ink-spec-land, Brief L1)

Branch `ink/spec-land` = origin/main 1f226a2e3 + `ink/mtp` (c6dffe0d6) + `ink/mtp-softcap` (36cbb9536, with `ink/mtp-cut`).
Merge conflicts were textual only: the Metal kernel pin list (`gumbel_argmax_f32` from main next to `hc_pre_bf16`), the Metal
crate's module list (`sync`, `inkctrl` from main next to `topk`), `docs/tuning.md` rows (both sets) and this doc. No defaults
change: DSpark runs only with `--spec dflash --spec-draft <Inkling-2-Mini dir>`; `ink/mtp-flip` (f130b8855) is not included.

### Gate rules (set before any gate ran)

Checkpoint `Inkling-2-Mini` snapshot b85c3cfa6 (its `mtp.safetensors` is the draft). DSpark arm everywhere:
`--spec dflash --spec-draft <same dir>` at its opt-in defaults, k3 and gate 0.8. Amended before any gate ran (coordinator,
2026-11-22: the opt-in default must be what the speed table measures): 840adbca6 moves `INTARSIA_DSPARK_TAU`'s default 0.6 -> 0.8
and gives a DSpark draft its own `--spec-k` default 3 (was the DFlash block's 15, capped at 7); B1 also checks the run's
"--spec-k defaults to 3" line. Every `INTARSIA_*` variable is unset in the job scripts first. main = origin/main 1f226a2e3 built
in the same job from `git archive`, branch = this branch's tip built in the same job. A gate fails on any miss; a failed gate
stops the lane and is reported, never re-ruled.

A. Spec off (the default) is a no-op vs main.
  - A1 (endor, Metal): Inkling greedy `generate --temperature 0` at 64 tokens (`--max-seq 4096`) and 256 tokens
    (`--max-seq 8192 --profile-every 0`), the lighthouse prompt of ink-metal's landing check: stdout md5 branch == main
    (main's known values 65cde8f35a1802e3e2fadb44d94d080d / 61fd59fecb5e0d72ffd08f2c6d07855a).
  - A2 (endor, Metal): `plan show --backend metal --max-seq 8192`, text and `--json`, for Inkling-2-Mini, Qwen3-8B and
    Llama-3.1-8B-Instruct: non-empty and byte-identical branch vs main (6 files).
  - A3 (endor, Metal, branch): every test binary passes with 0 failed and no panic: `intarsia-backend-metal` `--lib` (kernel
    pin list) and `--test` dense, topk, inkmix, inkmoe, rows_exact, screen, inkling2_attn, inkctrl; `intarsia-models` with
    `INTARSIA_INKLING2_BACKEND=metal --test-threads 1`: inkling2, inkling2_batch, inkling2_dspark, inkling2_dspark_cut,
    inkling2_verify_softcap; `intarsia-models` gdn_verify_metal, inkling2_mix_decode_cpu, inkling2_real_decode_ne (see the
    amendment below); `intarsia-cli --features metal --test docs_env_vars`; `kernels test --backend metal --op hc_pre`
    admits its kernel.
  - A4 (bespin, macOS, CPU slot): CI's check steps (`cargo check --workspace --locked`; metal; cuda / hip cfgs `--all-targets`)
    finish without error, and CI's test step (`cargo test --workspace --locked --no-fail-fast` minus metal / cuda / hip) has
    0 failed test binaries.
  - A5 (spark1, CPU slot): CUDA release builds of main and branch; `plan show --backend cuda --max-seq 8192 --mem-budget 30`,
    text and `--json`, for Qwen3.6-35B-A3B-NVFP4 and Qwen3-8B (no weights loaded): byte-identical. HIP: spark1 has no hipcc,
    so as CI does it (`INTARSIA_HIP_CHECK_ONLY=1 cargo check --locked -p intarsia-backend-hip --all-targets` and `-p
    intarsia-cli --features hip`): both finish without error.
B. DSpark k3, gate 0.8 (branch, endor, Metal).
  - B1: `generate --max-tokens 128 --show-ids --profile-every 0` on the 12 greedy golden prompts (`--prompt-ids greedy.json`)
    and the 2 prose texts (`prose.json`): every `ids[` line identical spec off vs DSpark (14 prompts), and the DSpark run's
    `spec:` line shows verify passes and accepted tokens > 0 (not vacuous).
  - B2: `inkling2_verify_softcap` passes on Metal (A3) and cpu (A4): the rows `forward_verify_logits` returns (what a sampled
    verify reads) are bitwise the softcapped decode logits. DSpark itself is greedy-only (sampled speculation exists only for
    `--spec mtp`): in the B3 serve, a t1.0 request returns 200 and leaves `intarsia_spec_steps_total` unchanged.
  - B3: `serve --max-batch 1 --max-seq 20480` with DSpark (admitted kernels only): one ~1.4k-token greedy chat request returns
    HTTP 200 and `intarsia_spec_steps_total` > 0 after it.
C. The PR's own CI (ubuntu check / test, macOS metal check, cuda / hip cfgs, reproducible build): every job green before merge.

Amendment (before #11532 ran, after bespin #11530): A4's test step on bespin (code 8f481dad2) had 7 failed binaries. One is real:
`docs_env_vars` (no rows for `INTARSIA_DSPARK_CUT`, `_DEVICE_TOPK`, `_ROWS_KERNELS`; fixed in e7cfbc65b). The other six are
`#![cfg(target_os = "macos")]` Metal binaries of intarsia-models that cannot run on bespin (no Xcode Metal toolchain there, so no
metallib: "the metallib was not built"; and the script's `CI=0` trips their skip asserts): gdn_verify_metal, inkling2_mix_decode_cpu,
inkling2_real_decode_ne. Ubuntu CI never compiles them. So A4's test step moves to Linux (spark1, CPU slot, `CI` unset, same rule:
0 failed binaries, on the final code), and A3 on endor adds those three binaries (the real decode one with
`INTARSIA_METAL_TEST_MODEL` = the checkpoint). A5 reruns on the final code in the same spark1 job (#11534 ran it on 8f481dad2,
before 840adbca6 touched `load.rs`).

Speed protocol (after A and B pass; branch tip binary, endor, one `--quiet gpu,cpu` slot): `serve --backend metal --max-batch 1
--max-seq 20480` per arm (off, DSpark), localeval `suite --tier adhoc --cells speed-plain` (143 / 1k / 4k / 16k
prompt tokens, 256 forced output tokens, 3 reps each), DSpark at its defaults (k3, gate 0.8); the DSpark arm declared `--lane spec:dspark --drafter dspark`, off `--lane
plain`. Order: one warm-up run (off, `--no-unim`, discarded), then off, DSpark, DSpark, off (ABBA). Reported per length: decode
tok/s, prefill tok/s and TTFT of each run, the arm means (absolute) and DSpark / off ratios, plus the A-to-A and B-to-B drift.

### Gate results

- A4, first pass (bespin #11530, code 8f481dad2): CI's check steps all finished (workspace; metal; cuda and hip cfgs
  `--all-targets`). The workspace test step had 7 failed binaries; see the amendment above. `inkling2_verify_softcap` and the DSpark
  tests passed on cpu there.
- A5 (spark1 #11534, branch 6b62cc05e = code 8f481dad2, main 1f226a2e3): `plan show --backend cuda` byte-identical for the 35B
  NVFP4 (json 15570 B, txt 4017 B) and Qwen3-8B (json 8973 B, txt 1548 B); HIP check-only `-p intarsia-backend-hip --all-targets`
  and `-p intarsia-cli --features hip` both finish. Rerun on the final code: #11540.
- #11532 (endor, default load, code 840adbca6, main 1f226a2e3), 1 failed check:
  - A1 PASS: g64 md5 65cde8f35a1802e3e2fadb44d94d080d and g256 61fd59fecb5e0d72ffd08f2c6d07855a, both equal to main's.
  - A2 PASS: Metal `plan show` byte-identical to main, text and json, for Qwen3-8B (8371 / 1342 B), Llama-3.1-8B (7983 / 1411 B)
    and Inkling-2-Mini (10519 / 2619 B).
  - B1 PASS: ids identical spec off vs DSpark (defaults, "--spec-k defaults to 3" seen) on the 12 greedy golden prompts x 128
    and both prose texts. Greedy golden: 72.67 -> 82.75 tok/s (default load, not a speed number), 478 verify passes,
    acceptance 69.6%, 2.47 tokens per pass (85265f9d5 had 71.8% / 2.56 on the same prompts; greedy acceptance is deterministic,
    so main's numerics moved it; the speed job runs it with `INTARSIA_INK_HEAD_SCREEN=0` too). Prose: 58.68 -> 73.24 tok/s,
    acceptance 73.8%, 2.60 tokens per pass (72.7% before).
  - B3 PASS: serve, admitted kernels only, DSpark: a ~1.4k-token greedy request HTTP 200, verify passes 0 -> 13.
  - B2 serve half PASS: a t1.0 request HTTP 200, spec steps unchanged (13).
  - A3: PASS for metal `--lib` (18), dense (11), topk (1), inkmix (4), inkmoe (2), rows_exact (11), screen (4), inkling2_attn
    (1), inkctrl (2); models inkling2 (11), inkling2_batch (7), inkling2_dspark (3), inkling2_dspark_cut (1), gdn_verify_metal
    (7), inkling2_mix_decode_cpu (2), inkling2_real_decode_ne (4); docs_env_vars (2); `kernels test --op hc_pre` (admitted).
    **FAIL: `inkling2_verify_softcap` on Metal**: "row 0: 81 of 300 verify logits differ from the capped decode logits (max |d|
    9.536743e-7)". The test had run on cpu only before (spark2 #11295 / #11296). The lane stopped here (rule: report, do not
    re-rule); the speed job #11533 aborted on its guard (no pass marker), nothing timed. Diagnosis job: #11553.

### The A3 failure: cause and fix (#11553, coordinator decision (a))

#11553 (endor, default load, diagnosis only) ran `inkling2_verify_softcap` alone:

| run | Metal result |
|---|---|
| branch 840adbca6 | 81 of 300 differ, max 9.5e-7 |
| `INTARSIA_INK_HEAD_SCREEN=0` | 81 of 300 differ (the screen plays no part) |
| `INTARSIA_INK_ROWS_EXACT=0` | 66 of 300 differ |
| 36cbb9536 (ink/mtp-softcap, before the main merge), built from `git archive` | pass |
| cpu, branch and 36cbb9536 | pass |

So main brought it: main's P12 unembed (lane ink-pack12, `head_p12`). A 1-row decode step's head goes through
`Model::head_proj`, which takes the P12 head (`matmul_p12_logits_gemv`), while `Model::verify_head` called `proj` on the bf16
head, i.e. `logits_bf16_gemv_rows` at verify rows: two kernels whose rows are an ulp apart in some logits. Fix 03292acab:
`verify_head` (replicated head) calls `head_proj`, which at a verify call (`exact_rows`) selects P12's `_rows` twin,
`matmul_p12_logits_gemv_rows`, the same shaders as the 1-row P12 head; a model without a P12 head (every non-Inkling model, and
Inkling off Metal or with `INTARSIA_INK_PACK12=0`) runs the same `proj` as before. Re-gate: the full #11532 script again (A1, A2,
B1, B3, A3 with the softcap test, plus the softcap test with P12 off as information), then the speed job.

Main moved meanwhile to 2474c17b9 (#492 ink/batch: `ink_in_proj_gemvx` at 2..16 rows and MoE decode-once, both default-on,
bitwise claimed). Merged as bfaccd447 (conflict: `inkmoe_bench.rs`, both routing modes kept). The re-gate compares against
2474c17b9 (A1 / A2 main built in the same job), and A5 reruns on spark1 against it (#11557). #11540 (A4 Linux test step, A5) ran
on 840adbca6 over the old main.

### Re-gate on the final code (#11555, endor, default load, code bfaccd447 = fix 03292acab + main 2474c17b9): all pass

- A1: g64 65cde8f35a1802e3e2fadb44d94d080d, g256 61fd59fecb5e0d72ffd08f2c6d07855a, equal to main 2474c17b9's (built in the job).
- A2: Metal `plan show` byte-identical to main 2474c17b9, text and json: Qwen3-8B, Llama-3.1-8B, Inkling-2-Mini (same sizes as
  in #11532).
- B1: ids identical spec off vs DSpark on the 12 greedy golden prompts x 128 and both prose texts ("--spec-k defaults to 3"
  seen). Greedy golden 74.41 -> 86.72 tok/s (default load), 478 passes, acceptance 69.6%, 2.47 tokens per pass (unchanged by
  the head fix); prose 73.63 -> 80.08 tok/s, 73.8%, 2.60.
- B3: serve, admitted kernels only (43 kernels, 195 cases on the plan shapes): ~1.4k-token greedy request (1425 prompt tokens)
  HTTP 200, verify passes 0 -> 13 (27 proposed, 21 accepted). B2 serve half: t1.0 request HTTP 200, spec steps unchanged.
- A3: every binary passes, now including `inkling2_verify_softcap` on Metal (bitwise; with `INTARSIA_INK_PACK12=0` it passes
  too): metal `--lib` 18, dense 11, topk 1, inkmix 5, inkmoe 2, rows_exact 11, screen 4, inkling2_attn 1, inkctrl 2; models
  inkling2 11, inkling2_batch 8, inkling2_dspark 3, inkling2_dspark_cut 1, inkling2_verify_softcap 1, gdn_verify_metal 7,
  inkling2_mix_decode_cpu 2, inkling2_real_decode_ne 4; docs_env_vars 2; the merged `inkmoe_bench` example builds;
  `kernels test --op hc_pre` admitted.
- A4 on Linux (spark1 #11540, code 840adbca6 over main 1f226a2e3, `CI` unset): CI's test step minus metal / cuda / hip, 147
  test binaries ok, 0 failed (docs_env_vars, inkling2_dspark, inkling2_dspark_cut, inkling2_verify_softcap among them).
  The same job's A5 at 840adbca6 passed too.
- A5 at the final code (spark1 #11557, branch bfaccd447 vs main 2474c17b9): CUDA `plan show` byte-identical (35B json 15570 B /
  txt 4017 B, 8B json 8973 B / txt 1548 B); HIP check-only, both targets, finish.
- The PR's CI on 4f155e476 (= code bfaccd447): all six jobs green (cargo check linux, cuda cfgs, hip cfgs, macOS metal; cargo
  test linux 16m30s; reproducible build).
- Main moved again to 2c103f913 (#482 graceful-cut: serve / server / supervisor only), merged as 00c49620c (no conflict).
  Recheck #11575 (endor, default load, code 00c49620c vs main 2c103f913): A1 md5 equal (65cde8f3... / 61fd59fe...), A2 Metal
  plan show byte-identical for all three models, B3 ~1.4k greedy HTTP 200 with verify passes 0 -> 13, B2 t1.0 HTTP 200 with
  spec steps unchanged. All pass.
- Main moved to 3f79f70ed (#495 svfflip: `sampled_verify_fast` recipe key, CUDA 35B only; load.rs in the batched loader,
  model.rs doc comments), merged as b6f331152 without conflict. Recheck #11579 (code b6f331152 vs main 3f79f70ed): A1 md5 equal, A2 byte-identical (3 models), B3 HTTP 200 with
  verify passes 0 -> 13, B2 t1.0 HTTP 200 with spec steps unchanged; pushed be12d4946, whose CI (run 45061016361) was all green.
- Main moved to 17b999272 (#494: the device Gumbel sampler default-on for Inkling-2 on Metal, through the recipe), merged without
  conflict. It changes the sampled path B2 exercises under `--spec dflash`, so A1 / A2 / B2 / B3 are rechecked against it before
  the merge (#11580-ish, below in the PR).
- The speed job's first try (#11556) aborted on its own guard: `serve` runs a supervisor whose child holds the port, and the
  guard wanted the supervisor's pid; fixed to accept a descendant, requeued as #11564.

### Speed (#11564, endor M5 Max, one `--quiet gpu,cpu` slot, binary bfaccd447 = this branch on main 2474c17b9)

`serve --backend metal --max-batch 1 --max-seq 20480`, localeval `suite --tier adhoc --cells speed-plain` (256 forced output
tokens, 3 reps per size), warm-up run (off, discarded) then off, DSpark, DSpark, off. DSpark at its opt-in defaults (k3, gate
0.8). The DSpark arms verified as `spec:dspark` from the server counters (passes 1676 / 1721, acceptance 77.5% / 74.4%, 2.54 /
2.47 tokens per pass); off as `plain`. Run ids `20261122-1351*-...-2-off`, `-1354*-3-dspark`, `-1356*-4-dspark`, `-1358*-5-off`
under /opt/code/deepeval/results (unanimis 67d7d2c4, ddcfa518, c7f15a12, b564899c). The last off run rejected its 1k cell on
prefill spread (1392-1731 tok/s); its median is used as is. Medians; the arm value is the mean of its two runs:

| prompt tok | decode off (A1 / A2) | decode DSpark (B1 / B2) | decode x | prefill off | prefill DSpark | prefill DSpark / off | TTFT off s | TTFT DSpark s |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 141 | 67.9 (67.6 / 68.2) | 76.3 (77.7 / 75.0) | 1.12 | - | - | - | 0.18 | 0.19 |
| 1020 | 66.2 (65.8 / 66.6) | 74.1 (72.4 / 75.7) | 1.12 | 1511 | 1661 | 1.10 | 0.68 | 0.61 |
| 4110 | 65.2 (64.8 / 65.5) | 83.5 (84.5 / 82.6) | 1.28 | 1959 | 1883 | 0.96 | 2.10 | 2.18 |
| 16457 | 60.3 (59.5 / 61.1) | 70.2 (68.9 / 71.6) | 1.17 | 2054 | 1986 | 0.97 | 8.03 | 8.29 |

Drift inside the slot: off's two runs agree on decode within 2.5%, but its 16k TTFT moved 8.42 -> 7.64 s (prefill 1954 -> 2154),
more than the DSpark-off gap (+0.25 s); the localeval prefill / TTFT columns are inside that noise at 4k and 16k (and DSpark
"wins" at 1k). Engine-reported prefill in the same slot (`generate`, no HTTP, greedy, 3 reps interleaved):

| prompt tok | off ms (reps) | DSpark ms (reps) | DSpark / off |
|---:|---:|---:|---:|
| 1021 | 593 (596, 593, 590) | 596 (593, 596, 600) | 1.005 |
| 4108 | 1329 (1318, 1329, 1356) | 1330 (1325, 1371, 1330) | 1.001 |
| 16455 | 7476 (6879, 7698, 7476) | 7516 (7591, 7516, 7466) | 1.005 |

Synced ingest profile (`INTARSIA_DSPARK_PROFILE=1`, 11 chunks of >= 100 rows, 1955 rows average): 4.12 ms of draft ingest per
prefill chunk, i.e. about 33 ms over a 16k prompt (8 chunks of 2048), 0.4% of its 7.5 s prefill. On today's main the draft's
prompt ingest costs about 0.5% of prefill at every length (the -6% / -7% of #10991 was the host ingest, gone since 85265f9d5).

Acceptance isolation (greedy golden x 128, same slot): DSpark with the screen on, with `INTARSIA_INK_HEAD_SCREEN=0`, and with
`INTARSIA_INK_PACK12=0`: all 69.6%, 478 passes, 2.47 tokens per pass, ids identical (and equal to spec off with the screen
off). Neither the screen nor P12 moved acceptance; the 71.8% -> 69.6% shift from 85265f9d5 is other main numerics (bit-exact
for the target's own tokens, not for the taps the draft reads). Decode there: screen 84.29, no screen 83.37, P12 off 81.15 tok/s
(spec off without the screen 69.12).



### Default-on proposal (lane ink-spec-land, for the coordinator; not built)

Numbers from #11564 (localeval, one quiet slot, binary bfaccd447) and #11575 / #11555 (gates):

| prompt tok | decode off -> DSpark tok/s | x | TTFT cost, engine (generate) | TTFT, localeval |
|---:|---:|---:|---:|---:|
| 141 | 67.9 -> 76.3 | 1.12 | - | 0.18 -> 0.19 s |
| 1k | 66.2 -> 74.1 | 1.12 | +3 ms (+0.5%) | 0.68 -> 0.61 s |
| 4k | 65.2 -> 83.5 | 1.28 | +1 ms (+0.1%) | 2.10 -> 2.18 s |
| 16k | 60.3 -> 70.2 | 1.17 | +40 ms (+0.5%) | 8.03 -> 8.29 s (off itself moved 8.42 -> 7.64) |

- The TTFT objection that held the flip (#10991: prefill -6% / -7% at 4k / 16k) is gone: it was the host ingest, replaced by the
  device ingest (85265f9d5). The draft ingest now costs 4.1 ms per 2048-row prefill chunk, about 0.5% of prefill.
- The decode gain shrank from #10991's 1.26-1.44x to 1.12-1.28x. Spec-off decode on main got faster (63.6 -> 67.9 at 141), while
  DSpark got about 5% slower in absolute terms at every length (80.2 -> 76.3, 77.5 -> 74.1, 87.5 -> 83.5, 72.1 -> 70.2; different
  slot and day). Off's gain fits main's 1-row work (screened head #489, P12). The screen is worth only about 1% to DSpark
  (84.29 vs 83.37 tok/s with `INTARSIA_INK_HEAD_SCREEN=0`, same slot), since a verify pass runs the full head over its 2-4 rows.
  DSpark's absolute drop is not explained yet. Untested candidates: #492's 2-16-row paths, which are exactly what a verify pass
  runs (revert switches `INTARSIA_INK_PROJ_X=0`, `INTARSIA_INK_MOE_DEC1=0`), and the greedy acceptance drop 71.8% -> 69.6% (not
  from the screen or P12, see the isolation above). One quiet-slot A/B with those two switches would settle it.
- f130b8855's conditions still fit today's main: no `--spec`, Metal, one rank in process, `--max-batch` unset or <= 1, the
  checkpoint's `mtp_config` + `mtp.safetensors` present, `INTARSIA_SPEC_DEFAULT` not 0. Since #492 the recipe's `max_batch = 16` is
  only the cap of `--max-batch`; without the flag serve runs one sequence, so a default server would speculate. Greedy-only holds
  per request (B2: a t1.0 request runs no verify pass and pays no ingest); the device Gumbel sampler in flight (#11554) gates on its
  own and does not see the draft. f130b8855 needs a rebase: `git merge-tree` against this branch shows 4 textual conflicts
  (common.rs, dspark.rs, spec.rs, tuning.md), all from the k3 / 0.8 defaults this PR already lands; what is left of it is
  `dspark_default()` in load.rs and its test. load.rs merges without conflict but would then hold both this PR's k3 block and the
  flip's; keep one.
- Deploy: the drafter is the checkpoint's own `mtp.safetensors` (636 MB, same snapshot, described by `config.json`'s
  `mtp_config`); there is nothing extra to download or convert. A model copy that names files one by one drops it (e.g.
  `scripts/inkling2-spark-copy-model.sh`), and with the flip a missing file silently means no speculation, so the deploy check is
  "the server log shows `the DSpark default of Inkling-2 on metal`". Costs on every Metal start, greedy requests or not: +6.1 s
  load (5.9 s of it the Markov fold [200058 x 256], cacheable) and about 1 GB (0.64 GB draft file + the folded heads; RSS not
  measured).
- Who gains: greedy requests only. Serve's default sampling is t1.0, so chat clients that send a temperature get no speedup.

Draft-ingest TTFT idea (asked for; not built, below the bar): run the last prefill chunk's ingest after the first token's logits
are read back, so the first token does not wait for it (bitwise: drafts never reach the output and the ingest order is unchanged,
only later in the stream). Saves at most one chunk's ingest, 4.1 ms: 0.6% of TTFT at 1k, 0.05% at 16k; about 2-3 h of work. Fully
overlapping every chunk's ingest (a second Metal queue, event-ordered after the chunk's tap writes) could hide up to 33 ms at 16k
(0.4%) if the GPU had idle capacity during prefill GEMMs, which it mostly does not; about a day. Neither is worth building: the
cost they remove is smaller than the TTFT noise of one quiet slot.

## Default-on (Brief L2 / L2a, lane ink-spec-land; branch `ink/spec-flip` from main 22303d704)

What changes:
- `recipes/inkling2.toml` `[tuning.metal] dspark_default_k = 3` (new recipe key, `Recipe::resolve_dspark_default_k`). Without
  `--spec`, Inkling-2 on Metal runs `--spec dflash --spec-draft <model dir> --spec-k 3` (gate 0.8, the default since #493) when:
  one rank in process, `--max-batch` unset or 1, no `--spec-draft` / `--dcoco`, `INTARSIA_SPEC_DEFAULT` not 0. A checkpoint
  without `mtp_config` + `mtp.safetensors`, or a configured prefix store (a drafting request bypasses it; the MTP default's
  rule), skips it with one stderr line, never a load error. `--spec off` or `INTARSIA_SPEC_DEFAULT=0` reverts.
- Sampled requests never draft with DSpark (unchanged) and so keep the device Gumbel sampler (#494).
- L2a, `engine.rs` (every DFlash-family draft): a greedy request whose draft fell out of sync (a request that did not draft ran,
  e.g. a sampled turn) and that can reuse more than `DFLASH_RESYNC_REUSE_MAX` = 128 prefix rows keeps the prefix and runs
  without drafting; before, it re-prefilled the whole prompt to draft. With at most 128 reusable rows (a chat template's
  preamble: a new conversation) it re-prefills from 0 and drafts, as before. The literal rule ("any reusable prefix") would never
  draft again on a server whose conversations share a template preamble, hence the threshold. Output is unchanged either way.
  Coordinator (2026-11-22, L2b): the 128 threshold is approved. Consequence, stated in the PR: on a server whose conversations
  share a system prompt longer than 128 rows, one sampled request turns drafting off for later greedy conversations reusing that
  prompt (prefix reuse and output kept, no speculation speed-up).
  Test: `tests/inkling2_dspark.rs` `dspark_unsynced_draft_keeps_the_prefix_and_skips_drafting` (tiny fixture, cpu / Metal).

Gate rules (set before the gates ran). Job: endor, default load, flip tip vs main 22303d704 built in the same job; every
`INTARSIA_*` unset. Any miss fails the gate and stops the lane (reported, never re-ruled).
- F1: Inkling greedy md5 at 64 / 256 (the A1 prompt): default (no `--spec`) == `--spec off` == main; the default run prints the
  "DSpark default" line and verify passes > 0; `--spec off` does not.
- F2: Metal `plan show` text and json byte-identical to main for Qwen3-8B and Llama-3.1-8B.
- F3: default server (no `--spec`, no `--max-batch`): the ~1.4k greedy request HTTP 200 with verify passes > 0; a t1.0 request
  HTTP 200 with spec steps unchanged.
- F4: `serve --max-batch 4`: no draft loaded, admission line equal to main's, greedy reply text equal to main's.
- F5: a model directory without `mtp.safetensors` (symlinks to the rest): rc 0, exactly one "not applied: no draft" line, no
  speculation, md5 equal to main's at 64.
- F6: t=1.0 seed 7, 128 tokens, a 150-token and a 5000-token prompt: ids identical default vs `--spec off` (both the device
  sampler), no verify pass.
- F7: tests pass: `intarsia-models --lib dspark_default` and `--lib recipe`, `--test recipes_doc`, `--test recipe_gumbel`,
  `intarsia-cli --test docs_env_vars`, and on Metal `inkling2_dspark` (with the L2a test), `inkling2_dspark_cut`,
  `inkling2_verify_softcap`. Before that, bespin (cpu): cargo check workspace + metal CLI, the same cpu tests.
- F8 (L2a): `serve --max-batch 1`, a ~4k conversation: sampled turn (t1.0, 32 tokens), then a greedy turn on it (+ a short
  assistant and user message), then a new short greedy conversation. On the default server the greedy turn's reply equals the
  `--spec off` server's, its TTFT is at most max(1.25 x, +0.05 s) of the `--spec off` server's (prefix reuse level), its cached
  tokens equal, it runs no verify pass; the new conversation does run verify passes and its reply equals `--spec off`'s. Info:
  main with opt-in `--spec dflash` (before the rule) on the same flow.
- Night speed (quiet slot after #11567, `--at 05:45`): box check first (no other intarsia serve / generate; a spec-off greedy
  256-token decode >= 65 tok/s, up to three tries a minute apart). Default server vs `--spec off` server (both without
  `--max-batch`), localeval speed-plain (143 / 1k / 4k / 16k, 256 forced tokens, 3 reps), warm-up then (A B B A) x 3. Bar, on arm
  means of six runs each: decode DSpark / off > 1 at every length; serve TTFT and prefill within 2% of off at 1k, 4k and 16k.
  Also t=1.0 decode (generate, seed 7, 256 tokens, 150 and 5000 prompt tokens, A B B A): default vs off reported. Otherwise stop
  and report.

### Default-on gates, first run (#11620, endor, default load, code 6d70c6fa7 vs main 22303d704): 2 checks failed

- F1 PASS: md5 default == `--spec off` == main at 64 (65cde8f3...) and 256 (61fd59fe...); the default run speculated (57 verify
  passes, 80.7% acceptance, 2.61 tokens per pass), `--spec off` did not.
- F2 PASS: Metal `plan show` byte-identical to main for Qwen3-8B (json 8371 B / txt 1342 B) and Llama-3.1-8B (7983 / 1411 B).
  Info: Inkling's one-sequence plan now differs from main's by 12 lines (the draft's calls), as expected.
- F3 PASS: default server, ~1.4k greedy request HTTP 200 with verify passes 0 -> 13; t1.0 request HTTP 200, steps unchanged.
- F4 FAIL as written ("admission line equal to main's"): main's server line read "30 kernel(s) checked, 113 case(s) run, 0
  cached", the flip's "30 kernel(s) checked, 22 case(s) run, 91 cached". The admission cache persists across servers, and main's
  ran first, so the split follows run order; kernels (30) and total cases (113) are the same. No draft was loaded. Both greedy
  replies were HTTP 200 with an empty `content` (Inkling reasons first and 48 tokens ended inside `reasoning_content`), and the
  `reasoning_content` texts are identical (238 chars). The script never compared them, since the admission check failed first.
- F5 PASS: no `mtp.safetensors`: rc 0, one "not applied: no draft" line, no speculation, md5 equal to main's.
- F6 PASS: t=1.0 seed 7, ids identical default vs `--spec off`, no verify pass, at 150 and 5000 prompt tokens. Decode (default
  load, one run each, on before off): 71.52 vs 73.07 and 67.54 vs 69.00 tok/s, about 2% lower with the default; the night job
  measures it A B B A.
- F7 PASS: dspark_default (1), recipe (16), recipes_doc (3), recipe_gumbel (1), docs_env_vars (2); on Metal inkling2_dspark (4,
  with the L2a test), inkling2_dspark_cut (1), inkling2_verify_softcap (1).
- F8 FAIL as designed, no evidence about the rule. The chat flow reused 0 prompt rows on every server, `--spec off` included
  (cached_tokens 0 at turn 2, 4593 prompt tokens), so the default server's greedy turn had no prefix to keep. With p = 0 it
  drafted (11 verify passes), which is the rule's behaviour. The client also timed only `content` deltas, so TTFT was None
  everywhere. Cause of the 0 reuse: Inkling-2's sliding-window ring slots let a request resume only within 16 rows
  (`INK_RING_MARGIN`) of the cached sequence's end, and the chat template re-renders the history (the turn-1 reply replaced, its
  reasoning dropped), so the common prefix ends ~30 rows before the cached end. Consequence beyond this gate: in serve, Inkling-2
  multi-turn chat reuses the KV prefix only when the new prompt extends the cached tokens almost exactly (e.g. `/v1/completions`
  continuations).
- Lane stopped here, as the rule says. The night job #11621 will not run without the pass marker. A rerun with fixed
  measurement (F4: kernels and total cases, reply = reasoning + content; F8: a `/v1/completions` continuation with an 8-token
  sampled turn) waits on the coordinator.

### Default-on gates, rerun of F4 and F8 (#11634, endor, default load, code 34525d854 vs main 22303d704): all pass

Brief L2c approved both measurement fixes. 34525d854 changed only `crates/intarsia-models/tests/mtp.rs` (the new BackendTuning
field in two struct literals; #496's CI `cargo test` did not compile) and this doc since 6d70c6fa7: the rebuilt flip binary is
byte-identical to #11620's (checked with `cmp` in the job), so #11620's F1-F3 and F5-F7 stand for this code.
- F4 PASS: `--max-batch 4`, no draft loaded; admission main 30 kernels / 113 cases, flip 30 / 113; the greedy reply
  (`reasoning_content` + `content`) identical to main's.
- F8 v2 PASS (`/v1/completions`, `serve --max-batch 1`): turn 1 sampled t1.0, 8 tokens on a 4575-row prompt; turn 2 greedy
  continuation (4584 prompt rows). Default server: 4560 rows cached, no draft (spec steps 0 -> 0), text equal to `--spec off`'s,
  TTFT 0.097 s; `--spec off`: 4560 cached (above the 4000-row void bar), TTFT 0.085 s. Main with opt-in `--spec dflash`, i.e.
  before the L2b rule: 0 cached, 15 spec steps, TTFT 1.864 s. A new short prompt (t3) on the default server drafted (9 steps).
- Summary: 0 failed checks; pass marker `.scratch/flip_pass_34525d854` written, so #11621 (Mon 05:45) runs on this binary.

### Inkling-2 serve re-prefills every normal chat turn (info, Brief L2c; a TTFT lever, separate from the flip, nothing built)

Measured in #11634 on the `--spec off` server (main's behaviour, unchanged by the flip), `/v1/chat/completions`, Metal, endor:

| request | prompt rows | cached rows | TTFT |
|---|---:|---:|---:|
| chat turn 1 (sampled t1.0, ~4.6k-token context, cold) | 4575 | 0 | 1.60 s |
| chat turn 2 (greedy, the history re-rendered by the template) | 4593 | 0 | 1.55 s |
| the same continuation as `/v1/completions` (F8 v2, prompt extends the cached tokens) | 4584 | 4560 | 0.085 s |
| new short chat prompt | 21 | 0 | 0.093 s |

- Turn 2 re-prefills all 4593 rows (about 1.5 s, ~3.0k rows/s) where an exact continuation takes 0.085 s: 18x the TTFT at ~4.6k.
  It grows with the conversation's length, since the whole history is prefilled again each turn.
- Cause, two parts: (1) the chat template re-renders the history (turn 1's reply replaced by its final text, the reasoning
  dropped), so turn 2's tokens leave the cached sequence about 30 rows before its end; (2) Inkling-2's sliding-window layers keep
  ring slots with a 16-row margin (`INK_RING_MARGIN`, `model/inkling2.rs`), so a request can resume only within 16 rows of the
  cached end; anything further back finds the window's rows overwritten and restarts from 0.
- Any OpenAI-chat client gets this on every turn (the coordinator's TTFT question, not the flip's). Levers, not built: snapshot
  the ring state at the end of each prompt (resume at the prompt boundary, which is where re-rendered histories diverge), or a
  larger margin sized to the template's re-render (memory cost per sliding layer).

**Parked 2026-11-23 (coordinator: weekly quota at its limit).** The coordinator cancelled #11828 (the #11621 speed-bar rerun) and
#11829 (spec-k sweep); not rebooked, waiters killed. State: `ink/spec-flip-merge` (merge of main 08ae803e0, code a6e3cc43a; MG1-MG6
pass in #11738 and #11751) is local only, not pushed; `ink/spec-flip` untouched at 9c78ba5e0 (code 34525d854, `flip_pass_34525d854`
valid); the box-check parse fix is in `.worktrees/ink-spec-flip/.scratch/flip_night.sh`. #496 stays HELD (no push, PR or merge
without the user's go). Next: (1) rebook #11828 (75 min, `--quiet gpu,cpu`, from `.worktrees/ink-spec-flip`, `bash .scratch/flip_night.sh`);
(2) then #11829 (spec-k sweep, 110 min, from `.worktrees/ink-spec-k`, `bash .scratch/sweep.sh`); (3) if the bar passes, merge the
then-current origin/main into `ink/spec-flip-merge`, run `.scratch/flipm_post.sh` (md5 64 / 256 + full MG6), report TTFT on vs off,
then the #496 call.

## Brief L4 (coordinator, 2026-11-25 ~08:30), lane ink-spec

- Run order for #496's evidence: speed bar #11853 (rebooked #11828, `--quiet gpu,cpu --expect 75`) first, then the spec-k sweep, then
  the post-merge `flipm_post.sh` check. The coordinator released my queued sweep #11854 on purpose: two cpu correctness gates (#11858
  prefill unit tests, #11859 busy2048 gate A, about 70 min together) were stuck behind it and they unblock two PRs. Resubmitted the sweep
  as #11860 (same command and flags, `--quiet gpu,cpu --expect 110`); it queues behind #11858 / #11859. The bar #11853 runs as it is.
  #496 stays HELD.

## #496 speed bar, run 2026-11-25 (fleetq #11853, endor, `--quiet gpu,cpu`, binary `.worktrees/ink-spec-flip/.scratch/bin/intarsia-34525d854`)

Flip binary only, `localeval suite --cells speed-plain` against `intarsia serve` (default, DSpark k3 tau 0.8) vs `serve --spec off`;
warm-up (discarded) then (A B B A) x 3 = 12 runs, 5 reps per length per run, medians; each run's rows are in
`/opt/code/deepeval/results/20261125-08*-adhoc-endor-intarsia-metal-ink-spec-flip-34525d854-NN-{on,off}/` and the raw log in
`.worktrees/ink-spec-flip/.scratch/night/` (`fleetq_bar.out`; table script `.scratch/night_summary.py`). Box check passed on try 2
(try 1 read 55.7 tok/s, try 2 71.5 against the 65 floor). Block 1 (runs 02-05) ran with the box still slow after that (run 02 on: 61.7-65.5
decode and 2.5 s / 10.1 s TTFT at 4k / 16k, and runs 03-04 off also low); blocks 2 and 3 (runs 06-13) are the clean ones. Both shown.

Blocks 2+3 (runs 06-13, n = 4 on, 4 off), mean of per-run medians:

| prompt | decode off | decode default | ratio | TTFT off | TTFT default | ratio | prefill off | prefill default | ratio |
|---|---|---|---|---|---|---|---|---|---|
| 144 | 72.8 | 83.2 | 1.144 | 0.150 | 0.160 | 1.062 | - | - | - |
| 1k (1022) | 70.6 | 77.0 | 1.091 | 0.547 | 0.543 | 0.993 | 1870 | 1881 | 1.006 |
| 4k (4110) | 69.8 | 90.1 | 1.292 | 1.802 | 1.823 | 1.012 | 2285 | 2257 | 0.988 |
| 16k (16458) | 65.8 | 77.9 | 1.184 | 6.959 | 7.108 | 1.021 | 2365 | 2316 | 0.979 |

All 12 runs (n = 6 + 6): decode ratio 1.122 / 1.085 / 1.241 / 1.153 (144 / 1k / 4k / 16k); TTFT ratio 1.040 / 0.995 / 1.042 / 1.054;
prefill ratio - / 1.008 / 0.972 / 0.962. Per-block decode ratios are above 1.06 at every length in all three blocks. A-to-A (same arm,
blocks 2+3): decode off 1.2-2.5 % spread, default 7.0-17.2 % (the default arm's decode varies with the draft acceptance of each run's
prompts); TTFT within-arm spread 2.6 % (off, 16k) to 3.9 % (default, 16k). Acceptance in the default arm 0.74-0.75 (2679/3571 .. 3233/4287).

Verdict against the bar: decode gain at every length passes (+9 % to +29 %). TTFT / prefill within 2 % passes at 1k and (blocks 2+3)
4k, and misses at 16k by 0.1 points on blocks 2+3 (TTFT +2.1 %, prefill -2.1 %) and by 3.4-3.8 points over all runs; the 144-token
TTFT is +6.2 % (+0.010 s, a 144-token prompt, under the 1024-token prefill-meaningful limit). The 16k miss is inside the same-arm
TTFT noise (2.6-3.9 %), so the cell is a marginal fail, not a clear one.

t=1.0 sampled decode (seed 7, `generate`, 256 tokens, A B B A, `.scratch/night/t1*.err`): default arm tok/s 66.8 / 68.7 (short) and
64.9 / 64.0 (5k) against `--spec off` 70.8 / 70.9 and 66.0 / 65.9: the default arm is 3.9 % / 2.4 % slower on sampled requests, and its
spec counters read 0 verify passes (the default applies to greedy requests only), so the loss is not speculation itself; its cause (drafter resident state, a
per-step check) is not isolated.
