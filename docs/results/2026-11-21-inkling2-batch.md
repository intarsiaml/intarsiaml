# Inkling-2-Mini batched decode (lane ink-batch, 2026-11-21)

Batched decode of 2..=16 sequences in one step (`Model::forward_batch`) for Inkling-2-Mini, CPU reference and Metal.
Branch `ink/batch` from `ink/metal` 1dacac3a8.

## Standing rule (main, 2026-11-22)

A speed number counts as a reference only if its log has the per-row token digests and the plan lines (plan / policy /
P12 / screen). Never restore an old bench source without `--digest` for a reference run.

## Briefs received

- 2026-11-22 ~16:00 (main): merge freeze on main for ~60 min until PR #491 (CUDA mixmoe) lands: no merge or push to
  main; keep running gates and timing; stack any PR as ready to merge; main says when #491 is in, then rebase onto that main
  before merging.

- 2026-11-22 ~15:55 (main), brief 34 closed, agreed: move section 34's logs into the committed raw folder, remove
  .worktrees/ink-batch-34; ink-metal owns the mix chain-split lever. **Brief 35:** this lane owns the MoE expert kernels (gu
  and dn), solo and batched. Solo profile (ink-metal, main 17b999272): MoE experts 4.14 ms per token vs 3.15 floor, 411 GB/s,
  96 dispatches per token; B 16 (doc 32): gu 20.03 vs 17.6, dn 11.14 vs 8.8. (1) Per-kernel trace of solo gu and dn at short
  context: per launch GB/s, threadgroups vs the 40 cores and waves, last-wave fill, rows per expert slice, time in the 8 active
  experts' loads vs reductions / epilogue; explain the 411 GB/s. (2) Bitwise fix, same per-output reduction order (grid reshape
  across the active experts, gu + dn wave fill, one launch for all active experts' slices); timebox 1.5 days; keep only at >= 2%
  of the solo token at short and 5k, or >= 2% of the B 16 step, in quiet ABBA; gates md5 64 / 256 and self-goldens identical,
  batched == solo digests, rows_exact, inkmoe tests. New branch from current main, worktree .worktrees/ink-moe-35; endor rules
  unchanged.

- 2026-11-22 ~15:40 (main), brief 34 change: drop item 2 (mix / blend / norm launches); lane ink-metal owns those kernels
  for solo and rows (its solo profile: conv_mix 234 dispatches per token). Keep going on item 1, in_proj at B 16, only.

- 2026-11-22 ~15:05 (main): report only when brief 34 has a result (a lever built and gated, or closed with numbers); no
  queued / waiting notes.

- 2026-11-22 ~14:40 (main), brief 34: agreed on levers 1-3; keep ink/batch-attn as a local record branch (unpushed), remove
  its worktree. Two gaps the ranking skipped: (1) in_proj 9.03 vs 3.5 ms floor (5.5 ms, the biggest absolute gap): per-call
  kernel trace of gemvx at B16, achieved GB/s, occupancy, threadgroup count; why ~40% of bandwidth while the solo GEMV is near
  the floor: the rows-exact per-row reduction order, or tiling / grid? If a bitwise fix exists (more rows per threadgroup with
  the same per-row dot order, a better grid), build it, timebox half a day, keep only at >= 2% of the B16 step in a quiet ABBA
  with identical digests. (2) mix/blend/norm 3.68 ms over 244 launches vs ~0.3: count launches per class; if merging adjacent
  row ops into one launch is bitwise and saves >= 1 ms per B16 step, build it under the same rule, else record and stop. New
  branch from current main, worktree .worktrees/ink-batch-34. Endor: cargo and git push only in --load cpu slots, ABBA in one
  quiet slot.

- 2026-11-22 (main): new endor rule (found by ink-metal): `git push` runs the repo's pre-push hook, which runs two `cargo test`s,
  so every push from endor is a cargo run: push only inside a fleetq slot, e.g. `fleetq run --host endor --load cpu --expect 10
  -- git push ...`. No reply needed.

- 2026-11-22 ~13:40 (main), brief 33: (lever 3) ROWS_SINGLE head + wo: flip if B16 >= 2% with identical digests, then gates;
  small PR from main 2474c17b9 (has #492), worktree .worktrees/ink-batch-rs. (lever 1) attention at B16 (10.45 vs 5.9 ms): GO, this
  lane owns the batched attention decode kernels now (ink-m-attn and dfuse closed); batched == solo bitwise per row at B 2/8/16;
  the 1-row solo kernel and its path untouched; timebox 1 day; first a one-paragraph plan in doc 32 (where the 4.5 ms goes: KV
  re-reads per row, occupancy, SWA vs full split), build only the part that explains most. (lever 2) MoE: measure the idle union
  threadgroup share first; build leader compaction only if idle >= 1 ms per B16 step, else record and stop. New code branches from
  main 2474c17b9, not ef2598549. Endor: cargo only in --load cpu slots, ABBA timing in one quiet slot.

- 2026-11-22 ~12:35 (main), brief 32: ef2598549 goes to the ink-metal lane, which lands it as a PR (gemvx + decode-once);
  no history rewrite on ink/batch, new commits on top are fine. Profile before building the idle-threadgroup item:
  (1) one B16 real-text step, per-command-buffer GPU trace, warm, early in the slot before the slow state; rank op classes
  by ms per step (MoE gu / dn, in_proj, attention, mix/blend controllers, unembed, gaps). (2) Next to each class its floor:
  bytes / ~546 GB/s (union experts and dense weights read once per step) or FLOPs / peak, whichever is larger. (3) Top 3
  levers with ms saved, bitwise or not, and effort. Under half a day, bitwise and >= 2% of the step: build and gate;
  otherwise report first. The idle union threadgroups item competes like any other lever. Endor: fleetq; cargo only in
  --load cpu slots; timing ABBA in one slot with absolute numbers.

- 2026-11-22 ~12:00 (main, all Metal lanes): on endor every cargo build / check / test runs inside a fleetq slot with
  `--load cpu`; no bare cargo or rustc on endor (a stray rustc, pid 47475, 95% CPU, ran inside prefill's quiet slot #11508 at
  11:51). Endor's GPU is in its slow state now (p8k 2152 vs 2880 at load 1.6): report each arm's absolute ms per step next
  to the deltas. (ink-batch: all its cargo runs are inside its own quiet slots; it has run bare `rustfmt --check` on single
  files, ~0.1 s each; from now on those go through a `--load cpu` slot too.)

- 2026-11-22 ~11:05 (main): gemvx default-on stands. Remove the slice form's kernel and switch from the branch (not opt-in),
  keep its numbers in section 30 (as with rel2 and probe B). Note that the remaining in_proj cost is the bitwise reduction the
  contract requires, so only a non-rows-exact kernel goes lower: parked as (b). Go on MoE, measure first, bound and rule as
  briefed.

- 2026-11-22 ~10:30 (main): gemvx: let #11463 run; if green, default-on is fine (bitwise, -12 us/call) with the revert
  switch and a docs/tuning.md row. Before choosing (a) vs (c), bound (c) from existing traces (<= 20 min, no new job): MoE ms
  share of a B16 step, mean union vs the 128 pairs, ms/step saved if each expert's weights were read once per step instead of
  once per pair. (c) bound >= 2x (a)'s 3.8 ms/step: (c) first, then (a); otherwise (a), then (c) if (a) gives < 2 ms. Drop (b):
  breaking rows-exact costs more than ~3 ms/step buys; it returns only if (a) and (c) both fall short. Record the brief and
  the (c) bound before building.

- 2026-11-22 ~09:20 (main): the Metal 4 shader-count test (`metal4_shaders_are_the_nax_variants_and_not_metal3_shaders`) fails
  on any branch carrying d03224697 (ink/prefill's NAX MoE shaders) until it takes ink/metal's one-line test fix 8e0ff6af6.
  Take it at the next ink/metal merge (or cherry-pick when adding shaders); that failure is not ours. Done: merged ink/metal
  f2c232423 (includes 8e0ff6af6) into ink/batch.

- 2026-11-22 ~09:00 (main, from ink-dfuse): correction to the in_proj reference. At T=1 the stream runs ink-pack12's
  `ink_in_proj_p12_gemv` (p12.rs); P12 is on by default for in_proj and wo up to 4 rows. Its bf16 twin `ink_in_proj_bf16_gemv`
  (inkmix.rs launch_in_proj_gemv / in_proj_rows) is bitwise equal under pack12's contract. The multi-row kernel must match
  that T=1 output bitwise per row. P12 stops at 4 rows, so B8/B16 in_proj likely reads bf16 (~1.33x P12's bytes) on top of
  any per-row re-reading: measure which kernel runs at 8 and 16 rows; a multi-row P12 kernel (weights once for all rows, P12
  bytes) may be the full lever; check p12.rs for an existing multi-row P12 path first. Keep the T=1 kernels and dispatch
  unchanged.

- 2026-11-22 ~08:50 (main): start with in_proj rows; MoE B16 second. Ownership: the B>1 in_proj rows path (T >= 2 decode,
  below the prefill NAX threshold) is ink-batch's; the T=1 path stays ink-dfuse's: do not change its kernel or dispatch.
  Order: (1) measure only: in_proj us per call at 1/8/16 rows on the current tree plus bytes per call; confirm ~6x over floor
  and whether weight reads repeat per row. (2) If confirmed, build a multi-row kernel that reads each weight tile once for all
  rows, rows-exact: each row's output bitwise equal to the T=1 kernel's, same per-row reduction order as the T=1 GEMV. If that
  cannot reach a worthwhile gain, report before building a non-bitwise variant (those stay opt-in; a default flip is the
  user's). (3) Gates: rows_exact (T=1 vs B rows, crossing every dispatch threshold), greedy md5 64/256, inkling2_batch
  digests, admission, then B8/B16 speed with the ABAB rule (warm number primary). MoE B16 trace analysis needs no slot: do
  it while the in_proj slot is queued.

- 2026-11-22 ~07:55 (main): #11393 is the last endor slot spent on the slow state's cause; record its result in section 28
  and return to the batched-decode optimization. If it points to something actionable in software (e.g. a P-state choice our
  dispatch pattern triggers), propose it; else close the topic. Reasoning: the 58.7 ms "fast" B16 shows only in the first ~3
  min after a quiet period; sustained load settles at 73-75 ms, which is what a loaded server sees: ~73-75 ms is the honest
  serving number, the cold one a best case. A BOX OK gate pinned to 61.0 ms would reject most warm runs and stall the queue.
  From now on, for any endor speed claim: (1) interleave arms ABAB (or ABBA) in one slot after a short warm-up; (2) report
  the gain as the per-pair difference plus whether the box was cold or warm (box check = label, not gate); (3) quote the
  warm steady-state number as primary.

- 2026-11-22 ~07:45 (main): #11352 X/Y show the same binary at 61 ms vs 77-90 in #11328: endor has a slow state hitting B16
  hard and B1 barely. Finish #11352 as planned. Then (1) re-label: #11181 consistent with a fast-state run (still unverifiable,
  no digests, but not "the outlier"); fast-state reference = final ~59-61 ms B16 (262-273 tok/s, above 230); slow state ~77-90.
  Place runs by time: #10900, #11057, #11261, #11266, #11297, #11328 slow; #11181, #11345, #11352 fast. (2) Hypothesis: GPU
  clock drops in the slow state (B16 compute-heavy, B1 bandwidth-bound). Add a compute-bound box check to every batched timing
  job: pure-ALU Metal kernel (FMA loop, no memory, ~1 s), time logged next to B1 and a B16 sample; find a non-root way to read
  GPU frequency / performance state (ioreg, IOReport), no sudo. (3) Box check for batched comparisons: B16 within 5% of the
  fast-state reference, else BOX SLOW, no result; keep same-slot ABAB. (4) Record slow-state windows with timestamps.
  Report the ALU-check result.

- 2026-11-22 (main): ink-fp8's quiet #11345 (~07:00-07:30, ink/metal a7cba52f4 + ink/fp8 72900194e, P12 off, screen off;
  box check B1 65.0 tok/s; batched == solo in 4 runs) measured screen off B1 15.39 / 15.53, B8 33.91 / 38.99, B16 63.25 /
  65.40 ms (#11181-like, ~250 tok/s) in the same hour as my 77-90. Before #11352 runs, diff the setups from the logs
  (.worktrees/ink-fp8/.scratch/batchab/ vs #11328): args, prompt type, bench source commit, --digest, env, binary path / sha,
  tree, plan / policy / P12 / screen lines. Name the difference; if --digest, measure its cost (same binary on vs off); if
  ctx / prompts / steps, say which is the honest serving config; make #11352 report both configs side by side. Hold any
  "230 not met" claim until settled; tell main.

- 2026-11-22 (main): one quiet slot (~35 min): NE 3 vs 4 batched plus the final default aede2ff72 as the digest-backed
  batched reference; B 1/8/16, random ids, --digest on every arm, 3 alternating reps, plan / policy / P12 / screen lines
  logged. Section 17's item holds only if NE 4 is within noise of NE 3 at B 8 / 16 and the digests match; if NE 4 costs more,
  report it; do not revert 7eca1b012 without main. ink-metal is told to use #11266 as the batched reference. Then plan
  (measure only): verified B 16 aggregate 194-207 tok/s random / 182-199 real text is under 230; closing it needs ~80 -> 70 ms
  per step at B 16 with the GPU busy 82 of 83 ms. Break the B 16 step into GPU time per kernel class at the final tree,
  re-check the dropped MoE candidate against the real ~80 ms base, propose the top levers with ceilings in ms per step, build
  only those >= 3 ms per step; coordinate with dfuse (its roofline covers B 1).

- 2026-11-22 (main): audit #11181 from its logs (no slot): exact args, env, input type, binary path / hash; --digest per batch vs
  a normal-speed run of the same tree; steps and rounds run; any skip, fallback or different plan. Digests match = same
  computation ~25% faster: find what differed, keep the diagnostic and aim it at #11181's conditions. Digests differ /
  missing or steps / plan differ = #11181 invalid: mark it, list every decision or comparison that used it, cancel the
  queued diagnostic (release if acquired); the box then matches #10956's GPU-bound ~82 ms at B 16.

- 2026-11-22 (main): accepted, the merges are not the B8/B16 cause: no bisect. Finish the toggles and the second ref / final
  runs, then one diagnostic (same slot if --expect covers it, else a fresh quiet slot): ref 8546d143f, B 1 and B 16, 1 rep each.
  (1) Per step, summed command-buffer GPU time (GPUStartTime..GPUEndTime) vs wall: did the GPU's own time grow or the gaps?
  (2) Per rep deltas of vm_stat (pageins, decompressions, swapins), the process's fault counts, and IOAccelerator
  PerformanceStatistics before / after (In use / Alloc system memory, Device Utilization). (3) GPU time grew = GPU slower
  (clocks, bandwidth, GPU memory management); gaps grew = host side. (4) Place #11181 and #11261 (date/time, setup) in time:
  60 ms -> 80-88 -> 80-90 at B16. Report which, with numbers; no reboot or endor change proposals (user's call). Main's
  endor snapshot 06:53-06:56: CPU 90-94% idle, 10 s deltas +938 pageins / +87 decompressions / +4 swapins, swap 21.07 / 21.5 GB,
  compressor ~29 GB, wired ~41 GB, iogpu.wired_limit_mb 0, uptime 7d18h (no reboot since #11181).

- 2026-11-22 (main): endor checked 06:33-06:40: no tensorfold process, kev idle, no thermal / performance warnings, AC high
  power, no other model server; swap nearly full (21.07 / 21.5 GB), which fits the system page-ins but not our slowness
  (0 major faults, slow in low-page-in runs too). (1) Let #11297 finish; its numbers are labelled "box-slow, not speed".
  (2) Control in the same slot if --expect covers it, else a fresh quiet slot: #11261's exact binary, args and input
  type, solo + B16, 1 rep. If it reproduces ~61-65 tok/s solo and 80-88 ms B16, the box is fine and the slowdown is
  #11297's config: find which. If slow too, the box is slow: release and requeue #11297's speed half later. (3) #11309
  (correctness) stays queued either way; hold #11306's timing until the control answers, and if the box is slow requeue
  #11306 rather than run it. Record the control result and this rule here.

- 2026-11-22 (main): batched-regression check is mine. (1) Wait for #11266's synthetic batch_step x3 on c918e9fda: B16
  within 5% of 59.96 ms = no regression, #11261 a bad slot, stop. (2) If confirmed: runtime switches in one binary on the
  final tree, random prompts, #11181 protocol, one quiet slot, pages warm, order alternated, 8546d143f binary as reference
  arm; suspects (a) fp8 screened greedy head, (b) dfuse controller switches at B>1, (c) hostgap host pool / argmax_v4t
  (INTARSIA_HOST_THREADS=1, INTARSIA_METAL_ARGMAX_V4T=0); env names from docs_env_vars. (3) Only if no switch explains it,
  bisect the merges in one slot. Report B8/B16 per arm and the culprit with its switch; fix/revert is main's call. May fold
  into #11297 if long enough, else queue right after. Later from ink-metal: the combined gate on 7e5e6010e (pre-pack12,
  not quiet) logged B 2/5/16 at 23.5 / 27.7 / 74.9 ms, so B16 matches #11261 and any regression predates pack12 (fp8,
  prefill, dfuse or hostgap); consider a B16-only path (screened head, per-row controller loop). ink-metal sends #11266's
  B1/8/16 (and its pack12 on/off arm) when it lands.

- 2026-11-22 (main, priority, CPU only): ink-mtp suspects batched-decode sampler readbacks skip the final logit softcap
  (model.rs batch-head top-k path ~2594-2612: LastTopK vals and LastLogits rows; model/batch.rs; softcap only in head_logits
  and graph_decode_step). (1) Test first: batched rows, top-k values and full logits, bitwise == one-sequence decode's capped
  logits after the same prefill, tiny Inkling cap 30 (shape: ink/mtp-softcap 36cbb9536 tests/inkling2_verify_softcap.rs).
  (2) If it fails, fix the same way, sharing Model::softcap_step from that branch (cherry-pick, no duplicate); batch tests +
  G1 on CPU (bespin or spark2). (3) Report: reproduced yes/no with max |d|, fix commit, exposed serve paths and models.
  rsync -a keeps mtimes: use cp or touch.

- 2026-11-22 (main): the steady auto B16 drop across reps (87.7 / 84.4 / 80.4 ms) looks like warm-up: expert weights may be
  mmap-backed and real text (~63 experts/call vs ~28) pages in never-touched or evicted experts. Check whether weight
  buffers are mmap'd and resident before timing (page-in / fault counts, vm_stat pageins before / after each rep). If so,
  rerun the paired comparison next quiet slot: one full untimed real-text warm rep across all B, then 3 paired reps; report
  pairs within each rep and the mean paired difference. Default-policy decision waits; auto stays.

- 2026-11-22 (main): main killed an orphaned `intarsia serve --port 8131` from this worktree (pids 85506/85510, ppid 1, 11 GB
  RSS, up since ~02:42 EST, no fleetq job) inside slot #11257. (1) Serve job scripts must kill the whole process group
  (set -m or setsid, `kill -- -$PGID` in an EXIT trap) and check the port is closed before exiting. (2) Note next to every
  quiet endor result since ~02:42 (incl. #11181) that it ran with that idle 11 GB server resident; re-measure only if
  borderline.

- 2026-11-22 (main): stop accepted; (a)+(b) approved. (a) batch_step and batch_prof take real-text prompts: fixed,
  deterministic tokenized corpus slice, a different slice per row, same slices every run; record expert union per MoE call
  (mean and max per B); keep random-id mode behind a flag. (b) same quiet job: INTARSIA_INK_MOE=auto vs union-fused at
  B 1/8/16, median of 3, greedy md5 per stream equal across arms. Build and test the bench change now (bench-only). Queue the
  quiet job only after ink-metal's combined gate #11206 passes, on the final ink/metal tree, ~40 min. After it: pick the B16
  candidate from a trace at the realistic union (same rules: >= 6 ms real build, 3-6 report, < 3 drop). If union-fused wins
  at B>1 bitwise, propose flipping Auto with the numbers.

- 2026-11-22 (main): MoE fixed-launch candidate approved. Step 1 profile: queue after ink-metal's combined gate passes, on the
  final ink/metal tree; one quiet slot (~30 min), kernel trace B 2/4/8/12/16, MoE time vs union size. Decide on the real
  (untraced) B16 fixed part: >= 6 ms build fused MoE launches (gate+up and down per layer over union experts in one grid,
  gather folded into down, per-row order unchanged); 3-6 ms report to main first; < 3 ms drop and report in_proj with its
  estimate. Before editing check file overlap with ink-prefill (MoE NVFP4 prefill, #11187) and ink-dfuse (ink_route preload,
  #11190); if the same MoE dispatch files, tell main before writing code. Build gate: rows bitwise vs pre-change tree at
  B 1/8/16 (greedy md5 per stream), Metal suite green, admission OK, quiet B 1/8/16 median of 3 showing the gain. Two
  attempts without gain: stop and name the broken assumption. conc-plain-1k prefill interleave: not mine yet, raise after.

- 2026-11-22 (main): #11181 accepted; NE 3 costs nothing batched, RB16 stays off, B16 slowdown closed as not reproducible;
  skip the paired pre-merge vs merged run. (1) Send ink-metal (af43898d63e5b5190) the exact B=1/8/16 batched-step command for
  its combined post-merge gate (that number becomes the reference). (2) Propose the next batched-decode candidate for B=16
  (~60 ms/step, ~267 tok/s), with a per-kernel breakdown of the B16 step if available and the estimated gain.

- 2026-11-22 (main): move #11174 (NE/RB speed A/B) behind ink-metal's NE fix gate #11179 (fix 7eca1b012, NE 3 fallback at
  D 2304): release it, requeue the same script, flags and log, update the monitor.

- 2026-11-22 (main): bisect names m-mix's NE 4 from D 2048 (25ff7864a) as the G2 decode culprit; NE=3 restores base KL;
  BLEND_RB and InkMixP cleared for accuracy; ink-metal is fixing. Do not requeue the five-arm b8reg; queue ONE trimmed quiet
  job on the merged tip: tip default | tip INTARSIA_INK_MIX_NE=3 | tip NE=3 + BLEND_RB=16, B=1/8/16, 3 reps. Report #11104's
  rtvar correctness when it finishes, then the trimmed table.

- 2026-11-21 (main, after the design note): design approved, proceed. (1) Memory: count the full-attention KV of the 8
  full visits per sequence at max context as well as the rings; `enable_batch` (and the server) must cap B so everything
  fits and refuse cleanly rather than over-allocate. (2) Solo path: single-stream decode tok/s and G2 short / long
  unchanged before vs after the merge (bitwise logits on the short set preferred). (3) MTP later: one row per sequence now;
  do not design out several rows of one sequence (draft verify across streams), do not build it; one line on what it needs.
  (4) Throughput through localeval (local-model-bench) at 1/4/8/16 streams under `fleetq --quiet gpu,cpu`, oMLX's 94 tok/s
  at 16 streams in the same table.
- 2026-11-21 (coordinator): Metal `generate` defaults to `--profile-every 16` (every 16th step waits on 419 command buffers,
  about 4 ms/token of fake gap; real single-stream greedy is about 15.76 ms = 63.4 tok/s). Use `--profile-every 0` for every
  engine-side decode timing, the solo no-regression check included; localeval through serve is unaffected.
- 2026-11-21 (coordinator): endor's quiet-GPU line is 14+ jobs (about 9 h). Correctness runs (#10810 mixed-context
  batched == solo, the G2-bitwise part of #10795) go as default-gpu-load jobs with short honest `--expect` (they share the box
  and backfill); only tok/s (#10773, the tok/s of #10795, #10811) stays `--quiet gpu,cpu`. ink-metal fixed the rows_exact pin
  list at a0751f302: merge ink/metal.
- 2026-11-21 (coordinator, after 173.7 tok/s at 16): (1) do the 9..16-row single-read work now: dense row GEMVs (in_proj, wo,
  unembed) read the weights once for all rows instead of two 8-row chunks, then the experts read each expert's weights once
  per step over the rows routed to it; target 16 streams >= 230 tok/s, every row still bitwise its solo step (the tiny tests
  and #10844's mixed-context check are the gate); new kernels opt-in by row count first; if a 16-row form cannot stay
  bitwise with the 8-row one, ask before changing the contract. (2) Confirm from code that serve / the engine route a single
  live sequence through the solo path, not a B = 1 batched step (B = 1 batch_step 17.5 ms vs 15.86 ms solo); if not, make
  them. (3) Keep #10844 / #10845 / #10811 queued; tiny / cpu tests for new work on spark2 (`--load cpu --mem 30`, copies in
  ~/intarsiaml-ink-batch); endor only for Metal admission and timing.
- 2026-11-21 (coordinator): `intarsia serve` on Metal refused Inkling (require_admitted: no admitted moe_experts kernel; the
  plan-shape case builder dropped `attrs.ink`, so the SiTU kernels were never admitted). Fixed in ink-metal 761a1561b
  (quant/cases.rs, test `situ_experts_case_keeps_the_ink_attrs`). Merge ink/metal at or after 761a1561b now; #10811's script
  must build from that tip at job start, not use a prebuilt binary.
- 2026-11-21 (coordinator, for the 8-vs-16 profile): ink-m-mix measured InkInProj at the real shape: 1 row 69.8 us, 8 rows
  83.3 us, 16 rows 168 us (241 GB/s, arithmetic-limited; the three-Matmul form is no faster at 16 rows, 260 GB/s), about
  +4 ms per step from 8 to 16 rows over 48 visits. Their idea 3, a matrix-unit GEMV for 9..16 rows bitwise equal to the 1-row
  kernel, is my call after the profile: build it only if InkInProj is the biggest share and it can stay bitwise per row.
  ink-metal is merging ink/m-mix 47570539e (InkInProj up to 8 rows on the GEMV instead of NAX): re-merge ink/metal before the
  next batch timing.
- 2026-11-21 (coordinator): ink/metal now has m-mix 47570539e (merge 606fd36e4, tip f55bfe8b7), no default flips: re-merge
  before the opt-in A/B starts.
- 2026-11-21 (coordinator): when #10845 shows solo tok/s base vs branch within noise (<= 1 % slower, or faster), ink/batch at
  that tip merges into ink/metal (ink-metal does the merge); report #10845 with the tip hash as soon as it lands. A solo
  regression above 1 %: find it before anything else.
- 2026-11-21 (coordinator): hypothesis for the ~17 ms not in GPU time: host softcap 0.67 ms per 200058-wide row (ink-hostgap)
  plus a host full-vocab pass ~0.2 ms+, per row, if the batched step downloads B logits rows and softcaps / argmaxes them on the
  host: 16 x ~1 ms. (1) Check from code what happens per row on the host in a greedy batched step. (2) If so, greedy rows on
  the device (the solo path's softcap + argmax per row, one small readback of B ids). (3) Sampled rows keep host logits; the
  per-row softcap on the host pool waits for hostgap's a2f2b87b9. Report before building; record this brief.
- 2026-11-21 (coordinator, rule for all Inkling lanes): this session and every worktree run ON endor; a local cargo
  build / check / test or shader compile loads endor's CPU and skews every quiet timing job there (a possible cause of noisy
  A/Bs). No local builds or tests on endor outside a fleetq slot. macOS compile checks: bespin
  (`fleetq run --host bespin --load cpu --mem 6 --expect <min>`, a `git archive` copy in ~/intarsiaml-ink-batch, toolchain in
  ~/.cargo/bin; no GPU timing or real model there). CPU tests: spark2. GPU work: endor fleetq jobs. Reading, git and editing
  locally are fine.
  Lane audit: every build and test of this lane ran inside an endor fleetq slot except one `cargo check -p intarsia-core`
  at 21:1x, before any timing job of this lane. Every quiet timing job queued now (#10900 A/B, solo A/B, #10811) builds inside
  its own slot before timing, so nothing outside a slot overlaps it.
- 2026-11-21 (coordinator): in ink-metal #10890 `intarsia bench speed` failed at its calibration call ("no tokens received")
  against the Metal serve, while chat and localeval streaming worked; ink-metal is fixing it. #10946 must not end on that
  failure: merge their fix before #10946 builds (hash to follow) or skip the bench step. Done: #10946 now runs localeval
  `conc-plain-1k` first, then the bench as a non-fatal step with a 30 min watchdog. Solo baseline from #10890: at
  `--max-batch 1` `conc-plain-1k` is serialized, 50.8-51.6 tok/s aggregate at 1 / 4 / 16 streams; the `--max-batch 16` run is
  compared against it.
- 2026-11-22 (coordinator): the `bench speed` calibration fix is ink/metal b9dd0d6b4 (crates/intarsia-bench/src/speed.rs only;
  Inkling's first generated token is a block marker the text stream drops, so a max_tokens 1 probe returned usage only and
  counted as "no tokens received"). Cherry-pick it before #10946 builds; keep the non-fatal bench step. Done: cherry-picked as
  6b5ff520f before #10946 started (it builds the tip at job start); the bench step stays non-fatal.
- 2026-11-22 (coordinator, after #10914): ink-metal merges ink/batch bbc0bec00 into ink/metal after #10901, right after
  ink/m-mix 52b337aa9 (both touch the mix shader), and messages the merge hash. (1) Get the 0.6 % back: from that ink/metal
  tip make the row-table branch compile-time (Metal function constant or a separate pipeline variant), so the solo path
  compiles without MX_ROWTAB; shader / pipeline only (ink-dfuse owns the inkmix.rs row kernels for fusion: touch no more
  than needed). (2) Correctness: rows-exact and batched == solo at B 2 / 5 / 16, run on endor in a default-load slot; macOS
  compile checks on bespin; no local builds. (3) Timing: if the two-pass curve job has not started, fold a solo ABAB
  (merged tip vs variant, 6 reps, --profile-every 0) into it; else one short quiet job. (4) Merge criterion: solo >= the
  merged tip and batched output identical; report the hash.
- 2026-11-22 (ink-metal, coordinator): ink/m-mix merged into ink/metal at 331238a4e, ink/batch bbc0bec00 at 45887198b, tip
  8546d143f (docs only); no textual conflicts. m-mix changed `mix_geom` (NE 4 from D 2048) and `BLEND_RB` 16 -> 4, so the
  row-table mix path now runs at NE 4 (md5s differ from before: reduction order). Merge gate #11048 (default load) reruns
  tiny B 2 / 5 / 16, rows_single, rows_exact, real-model batch_step mixed contexts, G2 / G3. Branch the compile-time
  MX_ROWTAB variant from 8546d143f; its solo ABAB's base arm is 8546d143f (not f55bfe8b7).
- 2026-11-22 (coordinator): pid 53955 on 8099 was an orphan of ink-dfuse's finished #10882; killed (with parent 53944),
  8099 free. Keep the port-pick + owner check; no other change.
- 2026-11-22 (coordinator, relaying ink-dfuse): ink-dfuse's hoist switch (INTARSIA_INK_MIX_HOIST) picked x vs ring from the
  row index instead of the in-call index + xoff under row tables; fixed on ink/dfuse ac2371c81 (contiguous path unchanged).
  Run the row-table rows-exact tests and tiny B 2 / 5 / 16 batched == solo with INTARSIA_INK_MIX_HOIST=1 against ink/dfuse
  ac2371c81 in a default-load slot (folded into #11056 if not started, else after ink/dfuse merges into ink/metal); report
  pass / fail to the coordinator and ink-dfuse (a194e917d8f92b542).
- 2026-11-22 (coordinator, on the B 8 / 16 regression A/B): the same m-mix merge (NE 4, BLEND_RB) is the main suspect in
  ink-metal's decode-mode G2 regression on the merged tip (teacher-forced decode rows 2.6e-3 -> 1.1e-2, drift growing row to
  row: looks like 1-row decode writing bad state); their bisect #11142 runs NE=3 / BLEND_RB=16 arms. Send the five-arm B 8 / 16
  table to ink-metal (af43898d63e5b5190) and the coordinator. If ink-metal reverts NE 4 for accuracy: rerun B 8 / 16 on the
  fixed tip first (a batched-only override may be moot). If accuracy is fine at NE 4 and only batched speed needs NE 3 /
  RB 16: approved, a batched-only override in mix_geom / blend_geom, written by ink-dfuse (a194e917d8f92b542, owns the
  inkmix.rs row kernels now; m-mix lane closed): send dfuse the exact lines, solo defaults stay. Solo rtvar: inconclusive,
  stays unmerged; retry in a later quiet slot only if cheap (0.6 % at best).
- 2026-11-22 (coordinator, relaying ink-hostgap): a side worktree whose target/ was cloned (cp -cR, APFS clone) keeps the
  origin's mtimes, so cargo can keep old crates as fresh and build stale code. If a target was cloned: touch every tracked
  file under crates/ before each build (or cargo clean -p intarsia-core -p intarsia-backend-metal), and each arm's log must
  show those crates compiling; for the HOIST=1 check, confirm the hoisted kernel dispatched.
  Done: `cp -cR` here preserves mtimes (checked). Every side target of this lane was cloned (ink-batch/target from ink-metal
  at the start; base-, rtvar-, dfuse-target from it). The queued five-arm A/B and #11104 now touch all tracked files under
  crates/, kernels/, recipes/ of each tree before its build and grep the `Compiling intarsia-*` lines; #11104's dfuse tree
  carries an uncommitted scratch-only `eprintln` that names the first hoisted mix dispatch with and without a row table
  (tests run with --nocapture). Exposure of past results: ink-batch/target was cloned before this lane's first build, from
  a newer ink-metal tip, so the very first tiny runs (before the 21:5x ink/metal merge) may have mixed in ink-metal code
  (they were all rerun after the merge); #11057's rtvar build (its target cloned at job start, 15 s build) is treated as
  unverified: the A/B's arm C rebuilds it from touched sources.
- 2026-11-22 (coordinator, build hazard 2 from pack12 #11066): `cargo test -p intarsia-cli --test <x>` rebuilds
  target/<profile>/intarsia without `--features metal` over the backend binary; later CLI steps fail ("built without the
  Metal backend") and md5 of two empty outputs reads SAME. Rules: build the backend binary last (or tests in their own
  --target-dir); count md5 / self-golden matches only when both runs exit 0 with non-empty output. Reply only if affected.
  Checked: no script of this lane runs `cargo test -p intarsia-cli`; intarsia-models has no metal feature (the backend is a
  plain dependency), so its test builds do not change batch_step. Past matches all had non-empty output and exit 0 (G2:
  EXIT 0 x 4, tensors compared one by one; generate outputs 1025 / 946 bytes). Not affected. Hardened anyway: #11104 compares
  digests only when both runs produced all three, and the timing job's text comparison requires non-empty files.
- 2026-11-22 (coordinator): (1) when #11104 starts, the coordinator runs ink-metal's G2 bisect inside its slot
  (FLEETQ_JOB=11104, ~25 min, 20 GB, correctness only): do not release #11104 early; extra GPU load during it is expected
  (bitwise / correctness tests only). (2) The five-arm regression hunt #11164 hunts the same suspects (m-mix NE 4 / BLEND_RB
  16 -> 4, InkMixP 112 -> 144): when the bisect lands (~04:15) read its table in
  /opt/code/intarsiaml/.worktrees/ink-metal/.scratch/bisect_job.log; if it names the culprit, cut #11164 to the arms it does
  not settle for batched B 8 / 16, or cancel it.

## 1. Design note (contract delta, written before any kernel)

### 1.1 Shape of a batched step

A batched step of B sequences is ONE pass of the visit table over B rows (row r = sequence r's next token), exactly the
loop of a solo B-row pass (`Model::ink_pass`), with two differences:

* every row-local op (`Embedding`, `RmsNorm`, `InkInProj`, `Matmul` wo, the router `Matmul` / `Logits`, `MoeRoute`,
  `MoeExperts` / `MoeExpertsMxfp8`, `AttnResMix`, `LoopBlend`) runs once over the B rows under `exact_rows` (the row-exact
  multi-row kernels that are bitwise the 1-row kernel per row: the property solo 2..=16-row passes already rely on,
  `ink_rows_exact`). The workspace planes are `[B, D]` exactly as for a B-row chunk. **MoE union grouping across
  sequences needs nothing new**: `MoeExperts` sees B rows and does not know or care which sequence a row is from.
* every op that touches per-sequence state, `SconvRes` (69), `InkKvPrologue` (70), `RelSinkAttn` (71), `ConvMix` (74) and
  `InkAttnDecode` (75), runs in a new **row-table form**: one call over the B rows, row r being the 1-row (T = 1) call
  on sequence r's own state at sequence r's own position.

### 1.2 Contract delta (appended, optional; the existing calls are unchanged bit for bit)

The row-table form reuses the existing batched-decode mechanism (`OpCall::rows: Option<RowTab>`, R3a; today defined for
`GdnConv`, `GdnDeltaRule`, `KvAppend`, `AttnDecode`, `QkNormRopeKv`) and extends it to ops 69, 70, 71, 74, 75:

* `call.rows = Some(RowTab { ptrs: u8 [B, stride, 8], rows: B, stride, index, byte_off: 0 })` and `call.pos = Some(i32 [B])`
  (per-row positions; required with a row table). A row-table call has `T = B` rows in every row-indexed tensor
  (`[T, ..]` activations, and row r of each plane of `[N, T, D]` / `[S, T, D]` sources, messages and the `[S, T, D]` mix).
* **Semantics.** Row r of the call is bitwise the T = 1 call at position `pos[r]` (the device-position form of the solo
  step) whose row-indexed tensors are row r of the call's and whose state tensors are sequence r's. No row reads or writes
  another row's state; there are no "in-call" history rows (each row's history is its own ring / cache).
* **State addressing.** The state tensors of the call (the rings and caches, in their output order) describe row 0's
  buffers and are shape / dtype / offset carriers: the j-th state tensor of row r lives at
  `tab[r, index + j] + carrier.offset` (`carrier.offset` = the tensor's byte offset in row 0's buffer; every sequence's
  slots have the same layout). State tensors per op: `SconvRes` / `ConvMix`: the ring (j = 0). `InkKvPrologue`: k ring,
  v ring (j = 0, 1). `RelSinkAttn`: k cache, v cache (j = 0, 1). `InkAttnDecode`: k cache, v cache, k ring, v ring
  (j = 0..4).
* **Row table entries** (`Model::upload_row_table`): per slot two entries, as for `AttnKv` (k, v): an Inkling-2 `Ring`
  slot (`RingKind::InkKv` / `InkConv`) gets `(buffer address, buffer address)`, so both planes of a ring are reached by
  the one entry plus their carrier offsets, and `index = 2 * slot` everywhere. `InkAttnDecode` needs two slots
  (KV, then the K/V conv ring) behind one `index`: the loader guarantees `kvconv_slot == kv_slot + 1` (asserted at load),
  so entries `2*kv .. 2*kv + 4` are k cache, v cache, k ring, v ring. DeepSeek-V4.1's ring kinds and shared slots keep
  their "unsupported" error.
* **Checks** (`OpCall::check`, `inkling2::ops::check`): a row table on 69/70/71/74/75 needs `pos` `i32 [B]`,
  `rows == T`, `index + n_state <= stride`; `InkAttnDecode` keeps its 1..=16 rows. A linear cache's `pos0 + T <= Cr`
  check is the device-position one (skipped, as today with `pos`).
* **Registration.** A row-table kernel is registered `.row_table()` (selected only for calls with a table) and named
  `<base>_rows` so it takes its base kernel's admission (`Registry::effective_admission`). The plan's batch check counts
  a row-table call as B of the solo step's calls (`OpCall::row_weight`), so 69/70/71/74/75 join `plan::PER_ROW_OPS`.
* **Fused vs pair.** Chosen per the 1-row rule (`ink_attn_fused_rule(1, .., ctx)`) over the batch's largest context;
  both forms give the same bits, only time moves.

### 1.3 The 144 per-sequence slots in a batch

Nothing moves: every sequence keeps its own `SequenceState` (48 visits x (KV, K/V conv ring, h conv ring) = 144 slots).
The host writes one row table per step (`u8 [B, 2 * 144, 8]`, 36 KiB at B = 16) and the position vector; a captured
graph of the step can then replay with any set of sequences. Memory: the rings are per sequence and fixed size (about
170 MB per sequence on the real model: 40 SWA rings of 1040 x 1024 x 2 planes bf16, plus 48 x 2 conv rings), held once in
the plan's workspace today; `enable_batch` reserves ring bytes x (B - 1) out of the KV budget like recurrent state, and
shrinks the per-sequence full-attention KV capacity accordingly.

### 1.4 Model / scheduler / engine

* `run_layers(Seqs::Batch)` on an Inkling-2 model runs the visit loop with B rows; the stateful call sites go through one
  helper that, in a batch, swaps in row 0's carriers, the row table and the position vector (no fork of `ink_pass`).
* `enable_batch` accepts `Block::Ink` when every row-table Inkling-2 op the step runs has a registered `row_table` kernel
  (cpu: the decomposition reference below; Metal: the new `_rows` kernels); `row_tables_wanted` gets an Inkling-2 clause.
* Head parity: the solo step softcaps logits on the host (`softcap_logits`); the batched step's `LastLogits` rows and
  device top-k values get the same treatment (argmax is unaffected, the cap is monotone).
* A failed batched pass resets every Inkling-2 state of the batch (half-written rings), as `forward` does for one.
* Several rows of one sequence (batched speculative verify across streams, later): the row-table form would take a
  segment table (row -> (sequence, k-th row of its run)) and the in-call history rule of the solo T-row call within each
  run; nothing in the entry layout or positions above prevents it. Not built.
* Memory (main's note 1): `enable_batch` reserves rings x (B - 1) and divides the remaining KV budget by B x the
  full-visit bytes per token, refusing (`Error::Memory`) when fewer than 16 tokens per sequence would remain, as today.
* Engine / scheduler: nothing Inkling-specific (`engine/batch.rs` drives `forward_batch`); graphs of the batched step stay
  off for Inkling-2 (the solo graph path is not there either).

### 1.5 CPU reference of the row-table form

The cpu backend registers `cpu_f32_<op>_rows` (and `ref_<op>_rows`) for 69/70/71/74/75: the call is decomposed into B
T = 1 sub-calls (row r's slices gathered per plane into temporaries, row r's state tensors as views of the addressed
buffers, `pos0 = pos[r]`), each run by the ordinary kernel, results scattered back. Bitwise the solo row by construction;
it is the reference the Metal `_rows` kernels are tested against. `device_address` on cpu returns the buffer's host
address (owned buffers only); `supports_row_tables` stays false for every other model.

## 2. What was built

| Part | Where | What |
|---|---|---|
| Contract | `crates/intarsia-core/src/inkling2/ops.rs` ("Row tables"), `ops.rs` (`OpCall::check`), `plan.rs` (`PER_ROW_OPS`) | `ROW_TABLE_OPS`, `row_roles` (Row / Planes / State(j) / Fixed per tensor), `row_states`; a row-table call needs `pos` i32 `[B]`, `rows == T`, `index + states <= stride`; the five ops count as `B` solo calls in the plan's batch check. Calls without a table: unchanged code path. |
| Model | `model.rs` (`run_layers` Ink arm), `model/inkling2.rs` (`InkBatch`, `sx` / `rowtab` helpers), `model/batch.rs` | `Seqs::Batch` runs the visit loop over B rows; the 5 stateful call sites go through one helper that adds the row table + positions in a batch (identical call otherwise). `enable_batch` accepts `Block::Ink`, reserves rings x (B - 1); row-table entries `(buf, buf)` for Ink rings; softcap on batched logits / top-k values (solo parity); a failed batched pass resets every state of the batch; batched graphs off for Ink. A batch takes fused `InkAttnDecode` when the backend has no row-table pair (both give the same bits). |
| cpu | `crates/intarsia-backend-cpu/src/rowtab.rs`, `mem.rs` (`Raw` view, `address`), `lib.rs` | `ref_<op>_rows` / `cpu_f32_<op>_rows`: the call run as B one-row sub-calls on the addressed buffers (the reference of the form); registered after the base kernels (admission takes an op's first reference kernel). |
| Metal | `kernels/metal/inkling2_attn.metal` (`ink_attn_dec<GP, RT>`, `ink_attn_comb<RT>`), `inkling2_mix.metal` (`MX_ROWTAB`), `inkattn.rs` (`launch_ink_decode_rows`), `inkmix.rs` (`conv_mix_bf16_rows`), `dense.rs` (`dispatch_ind`) | Templated RT flag on the existing kernels: row r reads its caches / rings through the table (address in the params, resident via the indirect set), its own position, its own T = 1 scratch block; per-row split choice unchanged (grid over the most splits, the combine returns for a row with one split). The non-RT instantiations compile to the same code as before. |
| Recipe | `recipes/inkling2.toml` | `max_batch = 16` (the cap of `--max-batch`; without the flag one sequence at a time, as before). |
| Bench | `crates/intarsia-models/examples/batch_step.rs` | model name from the directory; `--ctx-spread N` (sequence i at ctx + i N). Asserts batched tokens == solo tokens every step. |

## 3. Tests

`crates/intarsia-models/tests/inkling2_batch.rs`, tiny fixture (window 8, rel extent 12, 2-layer blocks, 3 passes), cpu and
`INTARSIA_INKLING2_BACKEND=metal`; logits compared bitwise (softcapped) against the solo decode step of each sequence:

| Test | What | cpu | Metal | Metal, `INTARSIA_INK_SPLIT_KEYS=4` |
|---|---|---|---|---|
| `batch_equals_solo_2_5_16` | 2, 5, 16 sequences at lengths 1..64 (before / at / past the window, ring wrapped many times), 6 steps | pass | pass | pass |
| `batch_argmax_equals_solo` | device argmax rows | pass | pass | pass |
| `batch_membership_changes` | one finishes mid-batch, rejoin, shuffled row order, single-row step | pass | pass | pass |
| `batch_rollback_per_sequence` | two sequences cut back by length (3, 2) while the third continues | pass | pass | pass |
| `batch_failed_pass_resets_every_sequence` | injected failure in a row-table `ConvMix`: every sequence reset to 0, then prefill + batched decode bitwise again | pass | pass | pass |
| `batched_engine_equals_solo_engine` | `BatchedEngine` (4 slots, 9 requests at once, one seeded-sampled) token for token = `LocalEngine`; batched passes ran | pass | pass | pass |

`INTARSIA_INK_SPLIT_KEYS=4` forces several splits per row at tiny contexts, so the row-table combine and rows with different
split counts in one launch are exercised.

Other models (same branch): `intarsia-core` 232 + doc tests, `intarsia-backend-cpu` (admission quick sweep included),
`intarsia-models` `batch`, `batch_recurrent` (15, cpu), `batch_metal`, `inkling2` (8, cpu and Metal), `intarsia-backend-metal`
lib (kernel-list golden gains `ink_attn_decode_bf16_rows` only; `conv_mix_bf16_rows` is an Inkling op, outside the golden) and
its `inkling2_attn` (1), `inkmix` (3), `row_tables` (8), `rows_exact` (10 of 11) tests: pass (#10815). The 11th,
`rows_exact_registrations_are_pinned`, fails identically on the base a88db4620 (#10818): ink/m-mix's
`ink_in_proj_bf16_gemv_rows` is `rows_exact` but not in the pin list; not touched here (integration lane). `plan show --model ~/models/Qwen3-8B --backend metal`:
stdout and stderr byte-identical to a88db4620 (#10785).

## 4. Speed (Metal, endor, real checkpoint)

`batch_step` (`crates/intarsia-models/examples/batch_step.rs`), fleetq #10773 `--quiet gpu,cpu --mem 25`, checkpoint b85c3cfa
(W4A16, bf16), every sequence at context 1024 (+64 decoded), 4 rounds x 16 steps, solo and batched arms alternating;
`profile_every: 0`. Each step asserts the batched tokens equal the solo tokens (held for every B).

| streams | B solo steps ms | 1 batched step ms | speedup | solo agg tok/s | batched agg tok/s | per-stream tok/s |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 17.30 | 17.53 | 0.99x | 57.8 | 57.0 | 57.0 |
| 2 | 36.83 | 23.29 | 1.58x | 54.3 | 85.9 | 43.0 |
| 4 | 71.49 | 32.55 | 2.20x | 56.0 | 122.9 | 30.7 |
| 8 | 146.64 | 58.38 | 2.51x | 54.6 | 137.0 | 17.1 |
| 16 | 292.58 | 92.13 | 3.18x | 54.7 | **173.7** | 10.9 |

oMLX on endor at 16 streams: 94 tok/s aggregate; Fable's estimate for this engine: about 250 (perf-strategy section 7).
Step time grows 5.8 ms from 1 to 2 rows, then about 2.3 ms per row to 8 and 4.2 ms per row from 8 to 16: the 8 -> 16 step
is the one that does not amortise (perf-strategy section 7: the chunked row GEMVs read the weights twice above 8 rows, and
the expert union grows). Engine-path numbers through `intarsia serve --max-batch 16` (`bench speed`, localeval
`conc-plain-1k`): #10811, queued.

## 5. Single-read 9..16-row dense GEMVs (opt-in)

Above 8 rows the row-exact bf16 GEMV (`matmul_bf16_gemv_rows`: Inkling-2's wo) and the logits GEMV (`logits_bf16_gemv_rows`: the
unembed, the router) ran as two chunk launches (`row_chunks`), so their weights came from DRAM twice. The MoE experts were
already read once per step for all rows (`Policy::Auto`: union gate+up and union down above one row) and InkInProj already
takes 1..=16 rows in one dispatch, so these two are the only re-reads.

`<base>_mr8t<T>` (kernels/metal/gemv_multirow.metal, T = 1..8, families 16-byte / 8-byte bf16 and f32-out logits): one launch of
8 + T rows, as an 8-row pass and a T-row pass inside each threadgroup (simdgroup for logits) on the same weight row, the
weights read from DRAM once. Each (weight row, activation row) accumulator is the mirrored 1-row kernel's (same pass helpers,
lane mapping and reduction order). Opt-in `INTARSIA_METAL_ROWS_SINGLE=1` (default off; only 9..=16-row calls of those two
kernels; `with_rows_single` test hook). G-B: `crates/intarsia-backend-metal/tests/rows_single.rs`, 9..=16 rows, wo shape on
both load families, a K off the 16-byte block, logits at N 8195 and 128: every row bitwise the one-row launch and the two-chunk
launch (#10877, pass); `rows_exact` 11 / 11 and the backend lib tests 18 / 18 in the same run.

Expected gain: about 0.9 GB of re-read per step each for wo (48 visits) and the unembed, about 3 ms at 16 rows. A/B on the
batched step: queued (quiet slot).

## 6. Real-checkpoint correctness (#10844, default gpu load)

* **Solo path unchanged**: `intarsia accuracy --backend metal --max-seq 4096 --write-self-golden` on the G2 oracle goldens,
  base a88db4620 vs this branch (bf6eac934 + docs): every tensor of the self-goldens bitwise identical (short 48 / 48,
  long 60 / 60; the files differ only in the `built_unix` / `self_meta` metadata). G2 short KL21 2.380e-3 (gate 4e-3), the
  integration lane's figure; both arms exit 0 on short and long.
* **Batched == solo on the real model at mixed positions**: `batch_step --ctx 600 --ctx-spread 150` (sequence i at
  600 + 150 i: 600..2850, so the 1040-row SWA rings of the later sequences have wrapped while the early ones have not, and the
  full visits read up to 2.9k positions), B = 2, 5, 16, 8 steps: batched tokens == solo tokens every step (asserted).
  Its timings were taken under a shared (non-quiet) slot and are not reported as speed.
* Tiny tests on the same build: cpu 6 / 6, Metal 6 / 6; `rows_exact_registrations_are_pinned` pass after the ink/metal fix.

## 7. Engine path (serve)

First attempt #10811 (22:26) exited 1: `serve --max-batch 16` refused at load, `moe_experts` without an admitted kernel at the
batch phase's shapes (`Batch t=10..`). It ran before the merge that brought ink/metal 761a1561b (the plan-shape SiTU case,
merged at 22:27:45) and on the old script that used a prebuilt binary: the known failure, not a batch-specific one. Log kept
in `.scratch/serve_batch-10811.log` / `serve-10811-prefix/`. Rerun queued with the script that builds the tip at job start; a
default-load admission probe (`serve --max-batch 16` on the tip until `/v1/models` answers, one chat request) runs first.

## 8. Where the 8 -> 16 step goes (#10885, kernel-mode trace, default load: shares only)

`batch_prof --ctx 1024`, `INTARSIA_METAL_PROFILE=kernel` (one command buffer per launch, so absolute times are inflated and
launch-count changes are over-weighted), cut by step marks, median over 10 steps; GPU ms per step:

| op | B 8 | B 16 | delta | launches at 16 |
|---|---:|---:|---:|---:|
| matmul_bf16_gemv_rows (wo, two chunks above 8) | 2.87 | 6.50 | +3.63 | 96 |
| ink_in_proj_bf16_gemv_rows | 8.19 | 11.80 | +3.61 | 48 |
| moe_experts_situ_bf16_rows (gate+up / down / gather) | 29.05 | 33.91 | +4.86 | 132 |
| ink_attn_decode_bf16_rows | 10.59 | 12.08 | +1.48 | 48 |
| logits_bf16_gemv_rows (unembed) | 3.47 | 4.63 | +1.16 | 2 |
| logits_bf16_gemv_rows (router, two chunks above 8) | 1.06 | 1.83 | +0.76 | 96 |
| moe_experts_mxfp8 (the bf16 / mxfp8 visits) | 3.06 | 4.37 | +1.31 | 12 |
| everything else | 9.73 | 9.02 | -0.71 | |
| **sum** | **67.0** | **84.1** | **+17.1** | |

Summed GPU time grows 17 ms while the real step (`batch_step`, quiet) grows 34 ms (58.4 -> 92.1). Before choosing a kernel
lever the command-buffer timeline (GPU busy vs wall per step, B 2 / 8 / 16) is measured, to tell real GPU growth from host
gaps (queued). The coordinator's InkInProj figure (+4 ms) matches this table; it is one of three similar shares, not the
largest (the MoE situ path is).

## 9. A/B of `INTARSIA_METAL_ROWS_SINGLE` (#10900, quiet, tip 76cedfc93)

`batch_step --ctx 1024 --rounds 2 --steps 16 --batches 8,16`, arms alternating off / on, 2 reps; batched step ms
(aggregate tok/s). Batched tokens == solo tokens in every arm (asserted, exit 0), so the one-launch form is also exact on the
real model.

| rep | B 8 off | B 8 on | B 16 off | B 16 on |
|---|---:|---:|---:|---:|
| 1 | 39.18 (204.2) | 38.03 (210.4) | 71.20 (224.7) | 70.33 (227.5) |
| 2 | 39.47 (202.7) | 38.21 (209.4) | 78.45 (204.0) | 78.28 (204.4) |

The flag cannot change a B = 8 step (it only acts on 9..16-row calls), yet B 8 "on" is 1.1-1.3 ms faster in both reps: that is
the run-order bias of this A/B (on always runs second). The B 16 difference (0.9 and 0.2 ms) is inside it. **No measurable
gain; the flag stays opt-in and off.** (Expected about 3 ms; the re-read is mostly served from the SLC, it seems.)

**Bigger finding: the whole curve moved.** Against #10773 (section 4, ink/metal 1dacac3a8-era code), the same harness at this
tip gives solo 61 tok/s (was 57.8), B 8 39.2 ms (was 58.4), B 16 71.2 ms = **224.7 tok/s** (was 92.1 ms = 173.7) in rep 1;
rep 2 of B 16 is 78 ms (204 tok/s), a 10 % spread between reps of the same setting. The likely cause is the merged ink/metal
work since then, chiefly m-mix 47570539e (InkInProj up to 8 rows on the GEMV instead of the matrix units; the 8-row InkInProj
was 8.2 ms per step in #10885); unrelated local CPU load on endor during #10773 (the coordinator's build rule) may also have
inflated it. A full 1 / 2 / 4 / 8 / 16 curve at the tip, two passes, is queued to replace section 4's table.

## 10. Solo path speed: ink/metal f55bfe8b7 vs ink/batch 177cfd2f5 (#10914, quiet)

`intarsia generate --backend metal --max-seq 4096 --max-tokens 256 --temperature 0 --profile-every 0`, both built in the slot,
6 reps per arm alternating order (base/new, new/base, ...); output text identical.

| arm | mean tok/s | median | min | max |
|---|---:|---:|---:|---:|
| base ink/metal f55bfe8b7 | 64.13 | 64.09 | 63.92 | 64.40 |
| ink/batch 177cfd2f5 | 63.73 | 63.64 | 63.47 | 64.11 |

Branch -0.62 % (mean), -0.70 % (median): inside the <= 1 % merge criterion, but consistent (the branch is slower in 5 of 6
pairs), about 0.1 ms per token. Suspects, all in code the solo step runs: the `MX_ROWTAB` runtime branch and the now-mutable
ring pointer in the mix kernel's conv loop, the 32 bytes added to `InkP` / `InkMixP` (set_bytes), and the per-call
`row-table` checks in `ink_pass` / `OpCall::check`. Not chased here (inside the criterion); a compile-time row-table variant of
the mix shader would remove the first two if it matters.

## 11. Host gap of the batched step (#10956, default load: shares only) and the serve probe (#10947)

`batch_prof --ctx 1024`, `INTARSIA_METAL_PROFILE=cb` (command-buffer timeline; the trace costs time itself, so the wall
times are above the untraced run beside each), GPU busy = union of the command buffers inside each step, median of 12:

| B | wall ms | GPU busy ms | host gap ms | command buffers / step |
|---:|---:|---:|---:|---:|
| 2 | 30.91 | 30.45 | 0.46 | 7 |
| 8 | 43.19 | 42.54 | 0.65 | 7 |
| 16 | 83.05 | 82.22 | 0.83 | 7 |

The batched step is GPU-bound: under 1 ms of host gap at every B. So the 17 ms that the kernel-mode trace (#10885) did not
account for between B 8 and B 16 is GPU time the per-launch trace under-counts (one command buffer per launch changes
overlap and cache state), not host work; the coordinator's host-softcap hypothesis is out on this evidence too (greedy rows
never come back as logits, section 8 note). The levers above 8 rows are GPU kernels.

Serve probe #10947 at ink/batch 38e487a5d: `intarsia serve --backend metal --max-batch 16 --max-seq 8192` starts in 62 s
("continuous batching: up to 16 sequences (--max-batch), busy prefill chunk 256") and answers a chat request; the admission
refusal of #10811 is gone with ink/metal 761a1561b.

## 12. #10946 is invalid: it measured another lane's server

#10946's own `intarsia serve` loaded, then failed to bind (`bind 127.0.0.1:8099: Address already in use`, supervisor exit
78). Port 8099 was held by ink-dfuse's `intarsia serve ... --max-batch 1 --port 8099` (pid 53955, running since 23:07), so
the script's readiness probe answered and localeval `conc-plain-1k` (50.8 / 52.1 / 52.6 tok/s at 1 / 4 / 16 streams,
deepeval results/20261122-004010-adhoc-endor-intarsia-metal-ink-batch, unanimis 9be1ea95) measured that `--max-batch 1`
server, not this lane's: **not a `--max-batch 16` result.** (It matches #10890's serialized 50.8-51.6.) Peak RSS in the log
is of the failed start. Fix: the script now takes a free port from 8131 up, and after readiness checks that the listener is
this job's binary (and still the same pid before the bench); rerun queued. The serve probe #10947 (port 8097, "listening on
http://127.0.0.1:8097" in its own log) was its own server and stands.

## 13. Compile-time row-table mix variant (ink/batch-rtvar) timing (#11057, quiet)

Variant: ink/batch-rtvar fc76eb318 (+ 2b740dee7, `batch_step --digest`, example only) on the merged ink/metal tip 8546d143f.
`ink_mix_row<NE, NS, QB, RT>`: the solo instantiations (`ink_mix_row_e*`, RT = false) carry no row-table code, the batched
calls take `ink_mix_row_rt_e*`; host: `mix_row_shader(.., rt)` and the two dispatch sites only. bespin compile check: Rust
compiles; bespin's Metal toolchain cannot build the shaders (empty metallib, `attention.metal` fails there), so the shader
compile is checked on endor (#11104).

**Solo ABAB** (6 reps each, alternating, 256 tokens, `--profile-every 0`; output text identical):

| arm | mean tok/s | median | min | max |
|---|---:|---:|---:|---:|
| merged tip 8546d143f | 62.81 | 62.77 | 61.60 | 63.59 |
| rtvar 2b740dee7 | 62.56 | 62.81 | 61.31 | 63.14 |

Inconclusive: median +0.06 %, mean -0.4 %, and this run's rep spread (3 %, both arms) is several times the 0.6 % being
recovered (#10914's spread was 0.8 %). The whole box was slower in this slot (both arms about 2 % under #10914).

**Curve on rtvar** (2 passes, ctx 1024, batched tokens == solo tokens every step):

| B | pass 1 step ms | agg tok/s | pass 2 step ms | agg tok/s |
|---:|---:|---:|---:|---:|
| 1 | 16.94 | 59.0 | 16.42 | 60.9 |
| 2 | 22.12 | 90.4 | 21.34 | 93.7 |
| 4 | 28.80 | 138.9 | 27.89 | 143.4 |
| 8 | 48.88 | 163.7 | 46.26 | 172.9 |
| 16 | 79.10 | 202.3 | 80.79 | 198.1 |

B 8 is 18-25 % slower than #10900 at ink/batch 76cedfc93 (39.2 ms) and B 16 about 12 % (71.2 ms in #10900 rep 1; rep 2
was 78), while solo moved only about 2-4 %. The batched step got slower between ink/batch 76cedfc93 and the merged tip (or
with the variant). Candidates: m-mix's new defaults (mix_geom NE 4 from D 2048, `BLEND_RB` 16 -> 4) applied to the row-table
mix path, other ink/metal changes in the merge (25 files), or rtvar itself. Quiet A/B queued: ink/batch vs 8546d143f vs
8546d143f with `INTARSIA_INK_MIX_NE=3` / `INTARSIA_INK_BLEND_RB=16` vs rtvar, B 1 / 8 / 16, forward then reversed order.

## 14. Engine path: `intarsia serve --max-batch 16` (#11093, quiet, ink/batch f64368dbb)

Own server checked (port 8131, listener = this job's binary, pid 85510). ink/batch here is the pre-m-mix-merge code (the
#10900 curve: B 16 71-78 ms in batch_step).

**localeval `conc-plain-1k`** (closed loop, ~1k-token prompts, ~254 generated tokens per request, temperature 0, 3 reps per
stream count; deepeval results/20261122-024220-adhoc-endor-intarsia-metal-ink-batch/result.json):

| streams | aggregate decode tok/s (median; reps) | per-stream | TTFT s | `--max-batch 1` (#10946, ink-dfuse server) | oMLX |
|---:|---:|---:|---:|---:|---:|
| 1 | 46.5 (46.5, 46.5, 50.8) | 46.5 | 0.90 | 50.8 | |
| 4 | 84.6 (84.6, 84.6, 84.7) | 21.2 | 1.23 | 52.1 | |
| 16 | **117.7** (117.7, 118.1, 113.6) | 7.4 | 1.68 | 52.6 | 94 |

16 streams: 2.2x the serialized server, 1.25x oMLX. **`intarsia bench speed`** (calibration fix in; 1024-token prompts,
128 generated): single stream decode 60.2-61.3 tok/s (prefill 1.3-1.45k tok/s); concurrency rows (every stream prefills its
prompt each request, TTFT 13.4 s at 16 when 16 prompts arrive together): 2: 73.6, 4: 83.0, 8: 91.6, 16: 93.3 tok/s aggregate.

Why the serve path sits well under batch_step's 200+ at 16: the workload is mostly prefill. At 16 streams localeval ran 80
requests (80 x ~1k = ~82k prompt tokens) in 172 s; at the measured ~1.4k tok/s prefill that is ~55 s of prefill, which the
engine interleaves with decode steps (busy prefill chunk 256), so decode runs at roughly 20.3k tokens / ~117 s = ~173 tok/s
when it runs, and the batch is short of 16 rows while new requests prefill. The next levers on this cell are prefill speed and
how prefill shares steps with a full decode batch, not the batched decode step. Peak RSS not recorded (the `time -l` wrapper
was killed with the server; the script should signal the server pid, not the wrapper). Job exit 1 is that kill, not a failure.

**Orphaned server (found 2026-11-22 by main).** This job's server outlived it: `kill $SP` hit `/usr/bin/time`, not the
`intarsia serve` child (pids 85506/85510, port 8131, 11 GB RSS), which stayed up idle from ~02:42 EST until main killed it
in slot #11257 (~05:00). The serve scripts now start the server in its own process group (`.scratch/serve_lib.sh`:
`set -m`, EXIT trap TERMs the server, then the group, then KILL, then checks the port is closed and the group is empty;
tested with a dummy server under `time -l`).

## 15. Open at hand-off (2026-11-22)

- #11104 (rtvar correctness + ink/dfuse HOIST=1 row-table tests; ink-metal's G2 bisect inside the slot) still running: `.scratch/rtvar_correct.log`.
- #11164 (five-arm B1/8/16 regression A/B) cancelled before it started; requeue with `.scratch/b8reg.sh` (log `.scratch/b8reg.log`).
  The post-merge batched slowdown (B=8 ~39 -> 46-49 ms, B=16 ~71 -> 79-81 ms) is unexplained.
- 230 tok/s at 16 streams (batch_step) not confirmed: 224.7 / 204 pre-merge, 198-202 post-merge.
- `plan show` Qwen3-8B Metal byte-identical last checked at a88db4620 (#10785), before the ink/metal merge.

## 16. rtvar correctness (#11104, `.scratch/rtvar_correct.log`)

`ink/batch-rtvar` 2b740dee7 vs merged base 8546d143f: Metal lib / inkmix / mxfp4_gemv / rows_exact tests 18+3+5+11 pass
(4 ignored); tiny CPU 8+6 pass; tiny Metal 8+6 pass; real model, mixed contexts, batched-token digests at B 2/5/16
363b4c051e0fbbff / b6691f473fb73a82 / 6c18980dd1c6861f on both: BATCHED TOKENS IDENTICAL base == rtvar.

ink/dfuse ac2371c81 in the same job (#11104, exit 0): row-table tests on Metal with INTARSIA_INK_MIX_HOIST=1 and =0 both
pass (tiny inkling2_batch 6/6, inkmix 3/3, tiny state 8/8), and HOIST=1 with forced splits (INTARSIA_INK_SPLIT_KEYS=4) 6/6.

## 17. NE / BLEND_RB speed A/B on the merged tip (#11181, quiet, `.scratch/nerb-11181.log`)

Merged tip 8546d143f (code = ink/metal 1d0892eb7, before the NE fix 7eca1b012), `batch_step --ctx 1024 --rounds 2 --steps 16`,
arms in order fwd / rev / fwd. Batched step ms per rep, median, and aggregate tok/s at the median:

| B | tip (NE 4, RB 4) | NE=3 | NE=3 + BLEND_RB=16 |
|---:|---|---|---|
| 1 | 16.40 / 16.43 / 16.36: **16.40** (61.0) | 16.31 / 16.34 / 16.33: **16.33** (61.2) | 16.29 / 16.41 / 16.62: **16.41** (60.9) |
| 8 | 34.93 / 35.11 / 34.93: **34.93** (229.0) | 35.09 / 34.98 / 34.78: **34.98** (228.7) | 35.04 / 35.14 / 35.27: **35.14** (227.6) |
| 16 | 59.96 / 59.96 / 60.61: **59.96** (266.9) | 59.90 / 59.75 / 59.77: **59.77** (267.7) | 60.39 / 60.41 / 60.50: **60.41** (264.9) |

- NE 4 vs NE 3 costs nothing measurable at B 1/8/16 (within 0.3%); BLEND_RB 16 is ~0.6-0.8% slower at B 8/16, not faster.
  Neither is the batched speed cost, so the NE fix (7eca1b012, NE 3 at D 2304) should not slow the batched step.
- The 71 -> 80 ms B=16 slowdown does not reproduce here: the merged tip runs B 16 at ~60 ms (266.9 tok/s), B 8 at ~35 ms,
  faster than every earlier figure (pre-merge #10900 71 ms; merged #11057 79-81 ms). This job ran without a pre-merge arm, so
  it cannot say what changed; the earlier figures came from other jobs (load, run order, the rtvar build or flags may differ).
  Before trusting the 230 target as met (266.9 here), a paired pre-merge vs merged run in one quiet job is still needed.

- Residency caveat: #11181 (and #11104, correctness only) ran while the orphaned idle `intarsia serve` from #11093 held
  11 GB on endor (section 14). An idle server takes little GPU, and the three arms ran interleaved under the same condition,
  so the A/B conclusion stands; the absolute B 16 figure (59.96 ms) is not borderline against any decision, so no re-measure.

## 18. Batched baseline handed to ink-metal; next B=16 candidate (proposal, 2026-11-22)

**Baseline command** (sent to ink-metal af43898d63e5b5190 for its combined post-merge gate; quiet slot, median of 3 reps;
reference = column 3 batched step ms and column 6 aggregate tok/s):
`cargo build --release -p intarsia-models --example batch_step` then
`INTARSIA_METAL_TEST_MODEL=<Inkling-2-Mini snapshot> ./target/release/examples/batch_step --ctx 1024 --rounds 2 --steps 16 --batches 1,8,16`.
Reference from #11181: B1 16.40 ms / 61.0, B8 34.93 / 229.0, B16 59.96 / 266.9 tok/s.

**B=16 step breakdown.** The only per-kernel trace is #10885 (section 8: pre-merge tree, kernel-mode trace, shares only, traced
sum 84.1 ms). Scaled to today's 60 ms step, assuming the shares held (the m-mix merge changed the mix kernels, which sit in
"everything else"):

| op | share at B 16 | ~ms of 60 | B 8 -> 16 growth (traced) |
|---|---:|---:|---:|
| MoE experts bf16 rows (gate+up / down / gather, 132 launches) | 40% | 24 | +4.9 ms (+17%) |
| ink_attn_decode_bf16_rows | 14% | 8.6 | +1.5 |
| ink_in_proj_bf16_gemv_rows | 14% | 8.4 | +3.6 |
| wo bf16 GEMV rows (two chunks above 8) | 8% | 4.6 | +3.6 |
| unembed + router logits | 8% | 4.6 | +1.9 |
| MoE mxfp8 visits | 5% | 3.1 | +1.3 |
| everything else | 11% | 6.4 | -0.7 |

**Candidate: the fixed part of the MoE expert launches at B 16.** Top-8 of 128 gives an expected union of ~52 experts per layer
at B 8 and ~82 at B 16 (+59%), but traced MoE time grew only 29.1 -> 33.9 ms (+17%). A two-point fit t = a + b * union gives
b ~0.16 ms per union expert per step (~6.7 us per expert per layer, near weight-read speed) and a ~21 ms traced (~15 ms of the
real step) that does not scale with the experts touched: per-launch tail and occupancy over 132 launches with few rows per
expert, plus the gather. Removing it (fewer, fuller launches: gate+up and down per layer over all union experts in one grid,
gather folded into down, per-row reduction order unchanged so rows stay bitwise) is the largest lever left.
Estimate: 3-7 ms per step, so B 16 at 53-57 ms, ~280-300 tok/s (+5-13%).
The fit has only two points from the old tree, so step 1 is one quiet profile job on the final tree: kernel trace at
B 2/4/8/12/16, MoE time against union size. It confirms or kills the fixed part before any kernel is written.
Second candidate: ink_in_proj rows (+3.6 ms from B 8 to 16; gain ~2 ms if it scales like wo would with a single read).
The single-read 16-row GEMV already failed for wo/logits (#10900), so this is ranked lower.

For the localeval `conc-plain-1k` cell (117.7 tok/s at 16), the bigger lever is still prefill and how prefill chunks share
steps with a full decode batch (section 14), not the decode step.

## 19. The MoE candidate's premise does not hold (checked before queuing the profile)

Two facts from the code and the existing #10885 trace (`.scratch/prof/k{8,16}.jsonl`, whose MoE labels carry the distinct
experts `n` and the pairs `k` of each call, `label_moe_routing`):

1. **The proposed launch structure already exists.** `inkmoe.rs` `launch_ink_experts` at T <= 16 already runs gate+up
   union-grouped over all pairs in one grid, then union down, then the gather (3 launches per MoE call, 44 calls per step =
   the 132). Down fused with the k-order token sum (`ink_moe_dn_f`, one launch, no gather) also exists:
   `INTARSIA_INK_MOE=union-fused` (or `fused`), all policies bitwise. `Auto` picks fused only at T = 1. So the code change I
   proposed is mostly an env flag, not a new kernel.
2. **In batch_step / batch_prof the rows share experts far more than real traffic would.** Distinct experts per MoE call:
   B 8 mean 20.9 (median 21, max 29), B 16 mean 23.3 (median 23, max 36). For independent routing, top-8 of 128 would give
   ~52 and ~82. Doubling the rows added only ~2.4 experts per call. The bench prompts are uniform-random token ids, and the
   rows evidently route to almost the same experts (likely they decode the same few tokens). So the +17% MoE growth from
   B 8 to 16 is per-pair work at a nearly constant union. My two-point "fixed part" fit assumed union 52 -> 82 and is void.
   It also means the 266.9 tok/s B 16 figure is optimistic for real prompts, where the union and the MoE weight reads per
   step would be several times larger.

Not queued: the step-1 profile as specified (MoE time vs union size) cannot vary the union with this bench. Reported to main.

## 20. Real-text bench prompts: functional test (#11209, default load next to ink-metal's gate #11206: no speed claims)

ink/batch ff9f65bf9 (bench-only: dcff9e8c4 + ff9f65bf9). Corpus `crates/intarsia-models/examples/data/bench_corpus.txt`
(158 KB frozen snapshot of repo prose, a license, Rust source; 49 909 Inkling-2 tokens, 16 slices of up to 3 119).
Log `.scratch/benchtest-11209.log`, ctx 512, 1 round x 8 steps.

- Deterministic: two text runs give identical digests at B 1 / 4 / 16 (all 16 per-row digests equal; B 16 combined
  889a7108a816c248).
- Rows decode differently on text: 16 distinct per-row digests at B 16. Random ids: at B 16 three rows share one digest
  (14a607d087669a18) and two rows repeat B 1 / B 4 rows' streams: the rows collapse onto the same tokens.
- Expert union per MoE call at B 16 (kernel trace, 44 calls per step): text mean 63.5 (median 65, max 83) vs random 27.7
  (median 27, max 52); independent routing would give ~82. So real text gives ~2.3x the union of the old bench.
- Timings in this shared slot (indicative only): B 16 batched step text 74.8 / 85.6 ms (214 / 187 tok/s) vs random 66.7 ms
  (240 tok/s); B 1 and B 4 equal within noise. The quiet job after #11206 measures it properly.

## 21. Real-text MoE job (#11261, quiet, tree e4ebcfe77 = ink/metal 7e5e6010e + bench; `.scratch/realmoe-11261.log`)

**auto vs union-fused, real text, batched step ms per rep (order fwd / rev / fwd):**

| B | rep 1 auto / uf | rep 2 uf / auto | rep 3 auto / uf | paired diff uf - auto (ms) |
|---:|---|---|---|---|
| 1 | 15.48 / 15.67 | 15.30 / 15.44 | 15.34 / 15.45 | +0.19, -0.14, +0.11 |
| 8 | 53.30 / 51.32 | 53.14 / 51.09 | 47.40 / 51.50 | -1.98, +2.05, +4.10 |
| 16 | 87.72 / 88.29 | 84.28 / 84.37 | 80.41 / 91.28 | +0.57, -0.09, +10.87 |

Per-row digests identical across both arms and all reps (B 1 67843d1dbf0249f5, B 8 74478685fe7299e9, B 16
846a505a4055b0ba): the policies are bitwise as documented. No verdict: auto drifts down rep by rep (87.7 / 84.4 / 80.4 at
B 16), consistent with mmap'd expert weights being paged in on first use (the bench never enables the keep-warm residency
set; real text touches ~63 experts per call). Rerun with an untimed warm rep and page-in counts: #11297. auto stays.

Random-prompt continuity run in the same slot: B 1 15.35 / B 8 42.79 / B 16 74.84 ms (65.1 / 187.0 / 213.8 tok/s), vs
#11181 on 8546d143f 16.40 / 34.93 / 59.96. ink-metal's combined gate on 7e5e6010e (not quiet) logged B 16 74.9 ms, so a B 16
regression may predate pack12; check pending (#11266, then runtime switches).

**Expert union per MoE call on real text (kernel trace, 44 calls per step):**

| B | union mean / median / max | independent routing | MoE GPU ms/step (traced) | all GPU ms/step (traced) |
|---:|---|---:|---:|---:|
| 2 | 14.2 / 15 / 16 | 16 | 8.50 | 21.91 |
| 4 | 24.8 / 25 / 31 | 29 | 18.22 | 38.82 |
| 8 | 41.0 / 42.5 / 55 | 52 | 28.09 | 53.14 |
| 12 | 53.5 / 55 / 71 | 69 | 33.04 | 62.97 |
| 16 | 62.5 / 64 / 83 | 82 | 32.60 | 61.97 |

**MoE candidate: dropped (< 3 ms rule).** MoE time scales with the union: from B 2 to 12, ~0.62 traced ms per extra expert
in the union per step (8.5 -> 33.0 ms as the union goes 14 -> 54), with an intercept of ~0. There is no fixed launch part
to remove; at B 16 MoE is flat (32.6 ms at union 62.5), below that line. Real text's MoE is weight-read work over the
experts actually touched. Fewer launches would not reduce it.

**B 16 per-op on real text (traced, ms/step, B 8 -> B 16):** MoE situ gate+up 15.93 -> 18.23, down 8.65 -> 10.27, gather
0.34 -> 0.27; ink_in_proj 6.36 -> 8.76; ink_attn_decode 8.49 -> 9.78; unembed 2.69 -> 3.62; wo / dense GEMV 2.10 -> 2.70;
MoE mxfp8 3.16 -> 3.82; conv_mix 4.15 -> 3.52; route 0.93 -> 0.73. The traced sum (62 ms) is below the untimed step
(80-88 ms); the per-launch trace under-counts, as in section 11.

**in_proj option (reported per the rule):** ink_in_proj is the next-largest share that grows with B: +2.40 traced ms from
B 8 to 16, about 14% of the traced step. A rows kernel that reads its weights once for all 16 rows could at best recover
that growth: ~2.4 traced, ~3 ms real, so B 16 ~81-85 instead of 84-88 ms (~+3%). The same single-read idea gave nothing for
wo / logits (#10900), so I would not build it without a microbench showing the 8 -> 16 growth is weight reads.

## 22. Batched-decode softcap check (main's priority brief, ink-mtp's suspicion)

**Batched decode applies the cap.** `forward_batch`'s head (`model/batch.rs` ~496-510, since 869be5b87) runs
`softcap_logits` on both the top-k values (`LastTopK`; the cap is monotone, so the device's ids and order stand) and the
full logits rows (`LastLogits`, and top-k rows that fall back to logits on a NaN). The uncapped readbacks ink-mtp found are
in the **verify** paths, not in batched decode: `model.rs` `VerifyRead::Logits` / `VerifyRead::TopK` (~2490-2510),
`forward_verify_batch_run`'s rows (~2694-2716) and `spec_graph.rs` (~425-447). ink/mtp-softcap 36cbb9536 caps those with
`Model::softcap_step`; per its commit, sampled requests do not speculate on Inkling today. No fix is needed on ink/batch.

**Test** (70446e0e7, `inkling2_batch_sampler_rows_are_softcapped`): tiny Inkling-2 (`final_logit_softcapping` 30),
2 / 5 / 16 sequences at mixed positions (before, at and past the SWA window), 3 steps, modes mixed per row (logits, top-k 8,
top-k 40). Every logits row must be bitwise the solo capped logits; every top-k row's values must be bitwise the solo capped
logits of its ids and the k largest of them. Device top-k rows are requested only where the backend has them (Metal; the cpu
backend has no `supports_topk`, so on cpu only logits rows run).
- cpu (bespin #11310): `inkling2_batch` 7/7 pass, max |d| = 0; G1 (`g1a_f32_vs_f64`, `g1b_teacher_forced_bf16`) 2/2 pass.
  Mutation (batch.rs without its two `softcap_logits` calls): the new test FAILS at the logits assertion, so it detects a
  missing cap. (spark2 cannot build this test file: its dev-dependency `intarsia-backend-metal` is macOS-only.)
- Metal tiny (#11309, default load): Metal has no `supports_topk`, so its sampled batched rows are full logits rows (the host
  sampler takes the top-k): the test passes with max |d| = 0, and the mutation copy FAILS. Device top-k is a cpu-backend
  feature (`CpuBackend::arc_topk`) on this model; cuda / hip also have it.
- Device top-k rows on cpu: the test first ran with the plain cpu backend (no top-k); fixed in cd372d2a7 (the softcap test
  wraps `CpuBackend::arc_topk` in the `Failing` test backend, which now forwards `supports_topk` / `topk_rows`). bespin
  #11335-ish run: "device top-k rows tested", 7/7 pass, max |d| = 0, G1 2/2. Mutation with only the top-k softcap removed:
  FAILS at "row 1 (LastTopK(8)): top-k values are not the solo capped logits of their ids". Both row kinds are covered.
- Remote-test hazard hit once: `git archive` sets file mtimes to the commit time, older than a previous run's mutated build,
  so cargo reused a stale mutated `intarsia-core` (every test failed). The bespin script now touches every extracted file.

## 23. #11297: warm rep + paired auto vs union-fused on real text, BOX-SLOW (not speed); control rule

`.scratch/realmoe2-11297.log`, tree e4ebcfe77, quiet slot. The whole box ran ~50% slow: solo decode 45-52 tok/s (61-65 in
#11261, same binary source and args), batched B 16 120-130 ms (80-88). System page-ins per run 5k-57k pages while our
process had 0 major faults and ~1.7 M page reclaims every run (each run is a fresh process, so the warm rep cannot warm the
next run's mappings; reclaims are minor faults on resident pages). Main found no other model server, kev idle, no thermal
warnings, swap 21.07 / 21.5 GB. **Labelled box-slow, not speed.**

| B | rep 1 auto / uf | rep 2 uf / auto | rep 3 auto / uf |
|---:|---|---|---|
| 8 | 64.03 / 74.04 | 62.55 / 66.73 | 69.63 / 64.94 |
| 16 | 130.31 / 128.32 | 127.82 / 125.91 | 123.61 / 123.22 |

(warm rep, untimed: B 8 67.27, B 16 120.92 ms). Paired diff uf - auto at B 16: -2.0, +1.9, -0.4 ms (mean -0.2); at B 8:
+10.0, -4.2, -4.7 (mean +0.4). Digests identical across arms. Even in a slow box there is no policy difference beyond
noise; no flip proposal from this.

**Control rule (main):** a control with #11261's binary (e4ebcfe77 build, kept at `.scratch/ctl/batch_step`), args and
real-text input, solo + B 16, 1 rep. Solo >= 58 tok/s and B 16 <= 92 ms means the box is fine and #11297's config is the
cause (find which). Otherwise the box is slow: requeue #11297's speed half and the toggles later. Combined job
`.scratch/ctl_then_toggle.sh`: the control first, the regression toggles only if the box is fine (#11306 and the separate
control job were released so that the toggles cannot run first). #11309 (tiny Metal softcap, correctness) stays queued.

Control vs #11261, side by side (main asked that the two be visibly identical):

| | #11261 arms (`.scratch/realmoe.sh`) | control (`.scratch/ctl.sh`) |
|---|---|---|
| binary | `.scratch/base-target/.../batch_step` built from e4ebcfe77 | the same tree's build, copied to `.scratch/ctl/batch_step` |
| input | real-text prompts (the bench default since dcff9e8c4; no `--prompts` flag) | real-text prompts (same default, no flag) |
| args | `--ctx 1024 --rounds 2 --steps 16 --batches 1,8,16 --digest` | `--ctx 1024 --rounds 2 --steps 16 --batches 1,16 --digest` |
| env | `INTARSIA_METAL_TEST_MODEL` only (auto arm) | `INTARSIA_METAL_TEST_MODEL` only |

The only difference is the batch list (no B 8). #11297 differed from both by `/usr/bin/time -l`, `INTARSIA_INK_PACK12=0`
(no such switch on e4ebcfe77) and stdout to a file.

## 24. Box-slow control and the B 8 / 16 regression toggles (#11328, quiet, 06:49-~07:25; `.scratch/toggle-11328.log`)

**Control** (#11261's e4ebcfe77 binary, real text, solo + B 16, 1 rep): solo 59.7 tok/s, B 16 90.53 ms: just inside the
rule (>= 58, <= 92), so the toggles ran. Swap 21.07 / 21.5 GB before and after, +4 swap-ins.

**Toggles** (random ids, #11181's args, `INTARSIA_INK_PACK12=0` everywhere; final tree aede2ff72; the screened head is opt-in
on ink/metal and no run logged "screened greedy head on"). Batched step ms:

| arm | B 1 | B 8 | B 16 |
|---|---:|---:|---:|
| ref 8546d143f | 17.08 | 51.17 | 89.73 |
| final default | 16.04 | 48.91 | 82.31 |
| final, dfuse 4 off (ROUTE_FUSED / MSG_PRE / MIX_HOIST / BLEND_NORM_C = 0) | 16.05 | 52.64 | 79.29 |
| final, hostgap off (HOST_THREADS=1, ARGMAX_V4T=0) | 15.57 | 50.30 | 77.96 |
| final, dfuse + hostgap off | 16.54 | 47.34 | 77.03 |
| final default (2) | 15.39 | 46.79 | 79.06 |
| ref 8546d143f (2) | 16.45 | 52.13 | 86.44 |

**Verdict: no code regression.** In one slot the old tree is not faster than the final one; it is ~8% slower at B 16
(ref 86-90 vs final 77-82). No switch recovers #11181's 34.93 / 59.96 ms. Main accepted: no bisect.

**Where #11181 sits in time** (fleetq log and log-file times, 11-22, EST): #10900 ~00:09, ink/batch 76cedfc93: B 16 71.2 ms.
#11057 ~02:xx, merged 8546d143f: 79-81. **#11181 ~03:15-03:49, 8546d143f: 59.96.** #11261 05:52-05:58, e4ebcfe77, random
run: 74.84. #11266 (ink-metal), c918e9fda: 76.20. #11328 06:49+: ref 86-90, final 77-82. So #11181 is the outlier, not the
start of a slowdown: the same 8546d143f code measured 79-81 before it and 86-90 after it. Its slot also had the orphaned
idle 11 GB serve resident (section 14). The GPU-time vs gap diagnostic (queued) is meant to show what differs.

## 25. #11181 audit: INVALID (no digests, unverifiable binary)

Evidence from `.scratch/nerb-11181.log` and `.scratch/nerb.sh`:
- Args: `batch_step --ctx 1024 --rounds 2 --steps 16 --batches 1,8,16`; random-id prompts (8546d143f's own bench, no
  `--prompts`); env per arm only `INTARSIA_METAL_TEST_MODEL` plus the arm's switch (`INTARSIA_INK_MIX_NE=3`,
  `INTARSIA_INK_BLEND_RB=16`). Binary `.scratch/base-target/release/examples/batch_step`, built in the job from `.scratch/base`
  at 8546d143f (`Finished ... in 10.81s` after touching sources).
- **No digests:** the script restores 8546d143f's `batch_step.rs` (`git checkout -- crates/intarsia-models/examples/batch_step.rs`),
  which has no `--digest`, and the log keeps only the table rows (`grep '^\| (1|8|16) |panicked'`). So there is no token
  digest to compare with a normal-speed run of the same tree, and no plan, policy, P12, screen or fallback lines.
- **Binary not recoverable:** `base-target` was rebuilt at e4ebcfe77 (#11261) and aede2ff72 (#11328); there is no hash.
  `.scratch/base` is clean now, but its state at 03:15 (it had served #11057 / #11104 arms) cannot be checked.
- Rounds and steps: 2 x 16 per arm as given. Its only internal check is batch_step's batched == solo token assert (it did not
  fire), which a different but self-consistent kernel set would also pass.
- What stands out: solo per token was normal (B 8 "B solo steps" 131.43 ms = 16.4 ms per token) while batched was ~25% faster
  than every other run of the same code (79-81 before, 86-90 after). All 9 of its runs agree (B 16 59.75-60.61), so it is
  not one noisy run. Something in that slot or build made batched steps faster, and it cannot be identified from the log.

Per main's rule (digests missing), **#11181 is invalid as a speed reference.** The box matches #10956's GPU-bound
~82 ms at B 16 (wall 83.05 vs GPU busy 82.22, 00:21) and the 77-90 ms of #11261 / #11266 / #11328. The endor-state
diagnostic #11350 was released before it started.

**Decisions and comparisons that used #11181's numbers (all to be read against ~77-90 ms at B 16, not 60):**
1. Section 17: "NE 4 / BLEND_RB cost nothing batched" (main accepted: NE 3 fine, RB16 off). Its arms were interleaved in one
   slot, so the relative result may still hold, but it is not verified bitwise. ink-metal's NE fix 7eca1b012 relied on it
   for "no batched cost".
2. Section 17 / main: "the 71 -> 80 ms B 16 slowdown is closed as not reproducible". That rested on #11181's 60 ms; with
   #11181 out, the merged tree's B 16 is ~77-82 ms, consistent with #11057's 79-81.
3. Section 18: the batched baseline sent to ink-metal for its combined post-merge gate (B 1 16.40 / B 8 34.93 / B 16 59.96 ms,
   266.9 tok/s) as "the reference". ink-metal's #11266 (76.20 ms) is the better random-prompt reference.
4. "266.9 tok/s at 16 streams, above the 230 target" (sections 17-18, messages to main). Not established: verified figures
   are 194-207 tok/s random ids (#11328 final) and 182-199 real text (#11261).
5. Section 18's B 16 breakdown scaled to "today's 60 ms step" and the MoE estimate "B 16 53-57 ms, 280-300 tok/s". The MoE
   candidate was dropped on #11261's real-text traces anyway (section 21).
6. The batched-regression hunt (#11266 vs #11181 "+19% / +27%", the #11328 toggles): the comparison base was #11181. The
   toggles themselves (in one slot, ref vs final) are valid and show no code regression.
7. Section 17's residency caveat ("not borderline") assumed #11181 was a valid absolute figure.

## 26. Input from ink-dfuse for the B 16 lever plan (2026-11-22)

ink-dfuse's B 1 roofline: job #11336, `docs/results/2026-11-21-inkling2-dfuse.md` "Job L results", raw
`docs/results/2026-11-21-inkling2-dfuse/raw/L_*` (L_roofline.txt; per-class multi-dispatch microbenches at real shapes,
weights rotated past the system cache, x calls per token). Peak 614 GB/s. Achieved at 1 row: InkInProj P12 532 / 524 GB/s
(bf16 563 / 569), wo P12 507 (bf16 550), unembed P12 576 (bf16 582), NVFP4 experts 438, MXFP8 experts 522, router 69,
controllers 66; attention core latency-bound (23-24 us per call); 0.88 us per dependent dispatch (543 dispatches = 0.48 ms
per step); the classes sum to 12.33 of 13.31 ms. For B 16: attention at ctx 512, 16 rows: fused SWA 123.4 us, full 130.8 us,
and the separate prologue + attention pair (97.2 / 97.7 us per visit) is faster than fused there; ctx 4096, 16 rows: fused
219 us SWA, 561 us full. NVFP4 experts at 16 rows (m-moe's doc): 750-770 us per visit, 484-497 GB/s on a union of ~83
experts. The p12 / MoE / ctrl benches take a rows argument but ran at 1 row only. Nothing of ink-dfuse's is in flight on the
B > 1 row kernels (in_proj rows, ConvMix row kernel, attention decode rows, MoE decode); the dfuse flip 2f952f697 (hoist,
norm_c, route, msg on by default on Metal) is in ink/metal 29e75b835, each with its own `=0`.

## 27. ink-fp8 #11345 vs #11328: setup diff from the logs (no slot)

| | #11345 (ink-fp8, 07:01:53-07:04:39) | #11328 final arms (ink-batch, 06:49:31-06:56:58) |
|---|---|---|
| args | `--ctx 1024 --rounds 2 --steps 16 --batches 1,8,16` | same |
| prompts | random ids (ink/metal's bench default) | same |
| --digest | no (ink/metal's bench has none) | no |
| env | `INTARSIA_INK_PACK12=0 INTARSIA_INK_HEAD_SCREEN=0/1 INTARSIA_INK_HEAD_SCREEN_STATS=1`, `INTARSIA_METAL_TEST_MODEL` | `INTARSIA_INK_PACK12=0`, `INTARSIA_METAL_TEST_MODEL` (+ arm switches) |
| tree | 5d833ef94 = ink/metal a7cba52f4 + ink/fp8 72900194e (screen commits only; screen 0 in the s0 runs) | aede2ff72 (= a7cba52f4 code + one test) |
| binary | `ink-fp8/.scratch/metalfp8/target/.../batch_step`, 16 756 944 B, sha 4f2fa0b8517bc335, incremental build 11.5 s | `.scratch/base-target/.../batch_step`, 16 742 592 B, sha 0e2a7fd6ac490e40, build after touching all sources |
| loader lines | `inkling2: 24 layers, 48 visits, 144 state slots, loaded in 1.6 s` (no P12 / screen lines kept) | not kept (grep of table rows only) |
| cwd | the fp8 tree | `.worktrees/ink-batch` (nothing reads cwd-relative config) |
| B 8 / B 16 batched ms | 33.91 / 63.25 (s0), 38.99 / 65.40 (s0 rep 2) | 46.79-52.64 / 77.03-82.31 |

No difference in args, prompts, digest, env (beyond the screen switches, off), or time of day (they ran 5 minutes apart).
The binaries differ: different trees (screen commits) and builds. No shell `INTARSIA_*` variables and no cargo config are in
play. **Not settled from the logs.** #11352 now starts with both configs side by side, alternating: X = fp8's saved binary +
env (x2), Y = #11328's saved binary + env (x2), Z = this tree's bench with the same args + `--prompts random`, no digest;
then the digest-backed NE arms. The "230 not met" claim is on hold until this is settled.

**27a. Embedded metallibs (07:10).** Carved from the two saved binaries (`MTLB` header, size at +16). Metal-4 lib identical in both (1,752,900 B, sha `f9d9e393bb68cfd5`). Main lib: fp8 #11345 9,203,071 B (`7e2f54d16b093c5e`) vs #11328 9,203,007 B (`119340e9ad7e9551`), 64 B apart, the same 1,064 function names. Consistent with the `screen.metal` diff only (head screen is opt-in and off in both arms). So the GPU code the two runs used is the same apart from an unused kernel; the 63 vs 77-90 ms gap is either host code or box state. The same-slot side-by-side arms X/Y at the head of #11352 decide which.

## 28. Endor slow state: #11352 side-by-side, relabelled runs, ALU box check (#11352 07:14-07:22, #11355 07:22-07:26)

**#11352, one quiet slot, every run ~40 s back to back, B 16 batched step ms (file mtimes):**

| run | time | arm | B 1 tok/s | B 8 ms | B 16 ms |
|---|---|---|---:|---:|---:|
| 1 | 07:15:44 | X fp8 #11345 binary + env | 65.0 | 33.92 | 58.67 |
| 2 | 07:16:21 | Y #11328 binary + env | 63.5 | 34.14 | 61.04 |
| 3 | 07:17:01 | X | 64.1 | 34.36 | 63.78 |
| 4 | 07:17:39 | Y | 64.7 | 34.58 | 63.61 |
| 5 | 07:18:19 | Z this tree (80334ed1a), no digest | 64.9 | 34.11 | 63.30 |
| 6 | 07:18:58 | NE 3 (default), digest | 64.9 | 35.37 | 70.85 |
| 7 | 07:19:38 | NE 4, digest | 63.1 | 40.15 | 72.83 |
| 8 | 07:20:19 | NE 4, digest | 62.5 | 38.32 | 74.38 |
| 9 | 07:20:59 | NE 3, digest | 63.7 | 37.12 | 71.25 |
| 10 | 07:21:40 | NE 3, digest | 64.7 | 41.51 | 77.23 |
| 11 | 07:22:21 | NE 4, digest | 63.8 | 40.62 | 73.68 |

- X and Y: the same code runs the same in one slot (58.7-63.8 ms). The #11328 binary measured 77-90 ms in its own slot, so the
  gap to #11345 was box state, not code.
- The box drifted inside the slot: B 16 rose from 58.7 to 71-77 ms over 7 minutes of back-to-back load while B 1 stayed at
  62.5-65.0 tok/s. `--digest` costs nothing: digests are computed after the timed loop (`batch_step.rs` ~177), so runs 6-11
  are slower from the drift, not from the flag (Z, same binary without digest, ran 63.30 ms 40 s before run 6 at 70.85).
- NE 3 vs NE 4: no speed result (the drift exceeds the arm difference; medians 71.25 vs 73.68 with one inverted pair). The
  digests differ: B 1 identical, B 8 / B 16 rows differ (row 2 of both, e.g. `4aad0881c883b783` NE 3 vs `fff302d90d8827f3`
  NE 4), stable across the reps of each arm. NE 4 is not the same computation as NE 3 on random ids. NE 3 stays the default.

**Relabel (main, 07:45).** Fast state: B 16 ~59-61 ms at the final tree (262-273 tok/s, above 230). Slow state: ~77-90 ms.
By time (11-22, EST): slow #10900 (~00:09, 71.2), #10956 (00:21, ~82), #11057 (~02:xx, 79-81), #11261 (05:52-05:58),
#11266 (05:58-06:08, 76.20), #11297 (06:27-06:35), #11328 (06:49-06:57, 77-90). Fast #11181 (~03:15-03:49, 59.96),
#11345 (07:01-07:04, 63.3-65.4), #11352 runs 1-5 (07:15-07:18, 58.7-63.8). #11181 stays unverifiable (no digests) but is
consistent with a fast-state run, not an outlier. #11352 runs 6-11 (07:19-07:22) drifted to 71-77 within one slot.
Section 25's list of decisions that used #11181 now reads against the fast-state reference, not against 77-90.

**Box check (from now on, every batched timing job).** `.scratch/boxcheck.sh BINARY [env]`: the ALU kernel below, then
`batch_step --batches 1,16 --rounds 1 --steps 16 --prompts random`, per-process GPU time over that sample
(`.scratch/gpuclients.py`, ioreg `AGXDeviceUserClient` `AppUsage.accumulatedGPUTime`, no root), thermal state
(`ProcessInfo.thermalState`) and a verdict: B 16 <= 1.05 x 61.0 ms = BOX OK, else BOX SLOW (no result). Batched comparisons
stay same-slot ABAB.

**ALU check.** `.scratch/alucheck/alucheck.swift`: 2^20 threads, 8 independent fp32 FMA chains each, one store per thread at
the end; self-calibrated to ~160 ms per command buffer, 5 buffers (~0.8 s); GPU time from GPUStartTime..GPUEndTime. GPU
P-state residency over the same window from IOReport ("GPU Stats" / "GPU Performance States", libIOReport via dlopen, no
root). States are reported as P1..P15; the pmgr voltage-states tables do not map 1:1 to 15 states, so no MHz yet.

| check | time | ALU ms (TFLOP/s) | P-states under ALU | thermal | B 1 tok/s | B 16 ms | verdict |
|---|---|---:|---|---|---:|---:|---|
| #11355 start | 07:23:15 | 160.88 (15.64) | P13 94.3%, P12 5.5% | (not logged) | 64.1 | 75.24 | BOX SLOW |
| #11355 end | 07:25:44 | 160.76 (15.57) | P13 94.4%, P12 5.4% | fair | 65.0 | 64.43 | BOX SLOW (5.6% over) |

- **Result: the ALU time does not track the B 16 slowdown.** B 16 moved 75.24 -> 64.43 ms (17%) while the ALU kernel stayed at
  160.8 ms (0.1%) and sat in the same P-state mix both times. So the GPU's clock under a compute load did not drop in the slow
  state. Not yet excluded: the P-state the B 16 workload itself runs at (DVFS follows the load; the ALU kernel saturates and
  may get P13 regardless), and memory-fabric / SLC clocks, which an ALU kernel does not see.
- No other GPU user: per-process GPU time over each B 16 sample and over the whole #11355 job (157 s) was < 40 ms for every
  other process (Herdr, WebKit, BambuStudio, Chrome); `pmset -g therm` records no thermal or performance warning.
- #11355 itself moved within one job too: B 8 text 38.86 ms (07:23) vs B 8 random 33.72, B 16 text cb 70.61, B 16 random
  untraced 62.13 ms. Its by-class tables (shares stable: MoE 52-55%, attention 16-17%, in_proj 12-15%) are usable as
  shares; the absolute ms carry the state.
- Next: #11375 (queued) runs ~9 min of sustained B 16 with a 500 ms P-state trace throughout, a box check every ~80 s and
  `pmset -g therm` before / after, to see whether the B 16 step follows the P-state under the B 16 load itself.

**Slow-state windows seen so far (11-22 EST):** ~00:09-02:xx (#10900, #10956, #11057); 05:52-06:57 (#11261 to #11328,
continuous GPU jobs from many lanes since 05:00); 07:19-07:23 (#11352 runs 6-11 and #11355's start check, after ~5 min of
continuous load in that slot). Fast: ~03:15-03:49, 07:01-07:04, 07:15-07:18, 07:24-07:26 (partly). The 07:15 -> 07:22 drift
under steady load and the recovery by 07:25 look like a load-history (heat) effect rather than a scheduled job; to be
confirmed by #11375's trace.

**28a. #11375 clock test (07:38-07:47, quiet, 9 min back-to-back load, final tree 80334ed1a batch_step sha 0cde56015c6f06e5;
`.scratch/clock.log`, P-state trace `.scratch/clock/pstate.log`, 500 ms).** Box check every ~86 s with a B 16 heater run
(`--rounds 3 --steps 32 --batches 16`) in between:

| check | time | ALU ms med (min-max) | TFLOP/s | P-states under ALU | B 16 sample ms | heater after, B 16 ms |
|---|---|---:|---:|---|---:|---:|
| c1 | 07:38:37 | 161.18 (160.8-161.2) | 15.57 | P13 94%, P12 6% | 74.01 | 70.89 |
| c2 | 07:40:03 | 239.08 (166.9-252.7) | 10.43 | P6 33%, P7 26%, P8 14%, P9-P13 28% | 73.27 | 75.15 |
| c3 | 07:41:29 | 184.45 (147.4-228.2) | 12.11 | P8 26%, P6 20%, P9 17%, rest P10-P13 | 74.16 | 70.99 |
| c4 | 07:42:55 | 228.72 (178.0-252.2) | 11.70 | P8 35%, P9 21%, rest P6-P13 | 83.44 | 74.28 |
| c5 | 07:44:22 | 200.62 (162.8-208.0) | 12.41 | P9 51%, P10 15%, P8 14% | 74.34 | 70.70 |
| c6 | 07:45:49 | 211.46 (166.6-218.3) | 11.86 | P8 44%, P9 26%, P10 12% | 73.07 | 73.33 |

B 1 stayed at 64.0-65.0 tok/s throughout. `thermal` = fair at every check; `pmset -g therm` recorded no warning before or
after; no other process used the GPU (< 25 ms each per sample).

- **The GPU does throttle under sustained load:** from ~1.5 min into the slot the ALU kernel, which held P13 at 07:23, 07:25
  and 07:38, ran at P6-P9 and 25-50% slower (first buffer of each check still ~fast, later buffers slow: a power / thermal
  governor, with ProcessInfo reporting only "fair").
- **But the B 16 step did not follow the ALU time:** c1 had the full ALU clock and B 16 was already slow (74.01 ms); c2 had
  the ALU at 10.4 TFLOP/s and B 16 was the same 73.27. Across the six checks B 16 sat at 73-75 ms (one 83.4) while the ALU
  ran anywhere from 161 to 239 ms. So "slow state = the clock the ALU sees" does not hold.
- Under the batch_step load itself the mean P-state index (0 = P1 ... 14 = P15) swung between ~6.5 and 12.95 within seconds
  in every minute (per-minute means 9.9-11.1). The bench interleaves solo (bandwidth-bound) and batched phases, so at 500 ms
  the trace cannot say which state the B 16 steps ran at, nor whether a fast-state B 16 runs at a higher P-state. That is
  the open question: does the GPU performance controller pick a lower P-state for the B 16 workload in the slow state?
- Next (queued as a measure-only job): B 16 batched-only (`batch_prof --b 16`, a few hundred steps, no solo phases) with a
  100 ms P-state trace and per-step times, at the start of a slot and again after ~5 min of heating, to correlate step time
  with P-state directly. MHz per state is still unmapped (15 states; no pmgr table with 15 entries).

**28b. Speed-claim rule (main, ~07:55), replaces the BOX OK gate of section 28.** Box check stays as a label (cold / warm),
not a gate. Every endor speed claim: short warm-up, then arms ABAB or ABBA in one slot; gain = per-pair difference; state
cold or warm; the warm steady state (B 16 ~73-75 ms at the final tree, ~215-220 tok/s) is the primary number, the cold
~59-61 ms (262-273 tok/s) a best case. #11393 is the last slot spent on the slow state's cause.

**28c. #11393 (08:33-08:40, quiet; B 16 step vs GPU P-state; `.scratch/pstep.log`, `.scratch/pstep/`), the last slot on the
slow state's cause.** Ten batched-only runs (`batch_prof --b 16 --steps 300 --warm 8`, random ids, ctx 1024, new process each,
~30 s apart) with a 100 ms IOReport P-state trace; each step's CLOCK_UPTIME_RAW marks joined to the trace (`.scratch/pscorr.py`).

- Box: start ALU 159.7 ms at P13 94% (B 16 sample 74.48 ms); end ALU 190.8 ms at P8-P11 (B 16 69.83). thermal "fair" both.
- Run medians 67.4-69.2 ms (one 73.8, the run with the lowest mean state 8.72); every run's mean P-state index 8.7-9.9.
- **Per step, B 16 time follows the P-state the GPU runs it at: Pearson r = -0.939 over 3,000 steps.**

| mean P-state index under the step | steps | median step ms |
|---|---:|---:|
| 6-7 | 22 | 85.25 |
| 7-8 | 69 | 78.80 |
| 8-9 | 787 | 75.08 |
| 9-10 | 641 | 69.54 |
| 10-11 | 1,429 | 66.89 |
| 11-12 | 52 | 64.55 |

- The B 16 workload never ran at P12-P13 in this slot, even while the ALU kernel got P13 seconds before. Extrapolating the
  table one or two states up gives ~59-62 ms, the cold "fast" number. So: **fast state = the GPU runs B 16 at P12-P13; slow
  (warm) state = P8-P10.** The B 16 step is clock-sensitive (~2.5-4 ms per state); B 1 (bandwidth-bound) is not.
- Why the GPU picks the lower states: not idle gaps (GPU busy 64.4 of 65.8 ms per step in #11355, so the utilization-driven
  governor sees ~98% busy), and not another process. The system reports thermal state "fair" at every check, and the ALU
  kernel itself drops to P6-P9 after ~1.5 min of load: a power / thermal limit the system applies to the GPU. B 16 draws
  more power per unit of time than the ALU kernel (DRAM traffic + ALU), so it sits lower under the same limit.
- **Actionable in software?** No direct lever: macOS exposes no API to request a GPU P-state, and our dispatch pattern does
  not leave the GPU idle. The indirect lever is energy per step: fewer bytes moved and less wasted ALU at B 16 (MoE 52-55%
  of the step) lowers power, which lets the governor hold a higher state. That is the existing optimization work, so no
  separate item. **Topic closed.** Warm steady state (~68-75 ms B 16) is the primary serving number from here on (28b).

## 29. B 16 levers with ceilings (proposal, measure-only basis: #11355 real text, 07:23-07:26, warm box)

B 16 real-text step at the final tree, GPU busy 69.74 of 70.61 ms wall (`.scratch/brk.log`); class ms scaled to busy. Floors
use dfuse's measured rates (#11336, section 26): NVFP4 experts ~490 GB/s at 16 rows, bf16 GEMV ~560 GB/s, unembed ~580 GB/s;
compute at the ALU check's 15.6 TFLOP/s. Expert bytes: gate+up+down = 3 x 1152 x 2304 params at NVFP4 (4.5 bits) = 4.48 MB,
which reproduces m-moe's 750-770 us per visit at union ~83 (372 MB / 0.76 ms = 490 GB/s). Ceilings are measured minus floor;
they are upper bounds, in warm-state ms.

| class | measured ms | launches | floor (basis) | ceiling | verdict |
|---|---:|---:|---|---:|---|
| MoE experts + route | 37.54 | 192 | 25.1 (44 calls x union 62.5 x 4.48 MB at 490 GB/s) | ~12 | **lever 1** |
| in_proj (q / r / gate / sink / k / v) | 9.80 | 48 | ~1.7 (~0.95 GB bf16 at 560 GB/s; 15 GFLOP = 1.0 ms) | ~8 | **lever 2** |
| attention decode | 11.14 | 52 | dfuse: separate prologue + attn 97 vs fused 123-131 us per visit at ctx 512 | ~0.7 | below 3 ms |
| head (unembed) | 4.09 | 2 | 1.6 (0.92 GB at 580 GB/s) | ~2.5 | below 3 ms |
| mix / blend / norm | 4.10 | 244 | launch-bound (244 x 0.88 us = 0.2 ms dependency) | <3 | below 3 ms |
| dense GEMV (wo etc.) | 3.04 | 96 | ~0.6 (0.34 GB) | ~2.4 | below 3 ms |

- **Lever 1, MoE at real-text unions (~12 ms ceiling).** Real text gives a union of 62.5 experts per call (random ids: 23), so
  ~2 pairs per expert. Each call costs 0.85 ms against 0.57 ms at 490 GB/s for the same union: the union kernels lose ~1/3 at
  low pairs per expert. First step (measure only): MoE time per call against union size and pairs per expert from the
  existing k16text trace labels, to tell per-expert fixed cost from bandwidth. Then a kernel change only if the fit shows >= 3 ms.
- **Lever 2, in_proj rows at B 16 (~8 ms ceiling).** 9.8 ms for ~0.95 GB of bf16 rows is ~6x the bandwidth floor and ~10x
  the compute floor: the B > 1 in_proj kernel does not reuse each weight read across the 16 rows. First step (measure only):
  in_proj us per call at 1 / 8 / 16 rows on one layer (does time grow linearly with rows?). Coordinate with ink-dfuse first:
  their roofline covers B 1, and they have nothing in flight on the B > 1 in_proj rows kernel (section 26).
- Everything else is below the 3 ms bar; none built.
- Gains are reported per the 28b rule: ABAB pairs in one slot after warm-up, warm number primary.

**29a. Corrections from the per-op trace (`.scratch/brk/k16text.jsonl`, traced ms, not scaled to busy).**
- in_proj: one call = the stacked [6552 + 1056 + 1056, 2304] bf16 rows = 39.9 MB (ink_mix_bench `NS`), 48 calls per step
  (not 24 x 20 MB). Per call: 182.0 us at 16 rows, 112.9 us at 8 rows; floor 71 us at 560 GB/s bf16 (53 us at P12's 0.75x
  bytes). So 2.6x over the bf16 floor at 16 rows, not 6x; ceiling 48 x (182 - 71) us = **~5.3 ms** per step (bf16), ~6.2 ms with
  P12 bytes. Still above 3 ms. The launcher already reads each weight row once for all 16 activation rows (R 16, P 1, RB 16:
  one threadgroup per weight row, 16 accumulators per lane); what repeats is the activation load: every weight row's
  threadgroup reloads all 16 x rows (16 x 4.6 KB per 4.6 KB of weights), ~320 MB of cache reads per call. `INTARSIA_INK_PROJ_P`
  (weight rows sharing one x load) and `INTARSIA_INK_PROJ_RB` are same-bits knobs; #11428 (queued, `.scratch/inproj.sh`)
  measures them at 1-16 rows with P12 next to it.
- MoE: per step 44 `moe_experts_situ_bf16_rows` calls (gate+up 415.9 us, down 233.8 us, gather 6.1 us = 655.8 us per call at
  union 62.5, 28.9 ms) plus 4 MXFP8 calls (634.7 + 329.0 + 6.2 = 969.9 us, 3.9 ms). The situ part's floor at 490 GB/s is 25.1
  ms (44 x 62.5 x 4.48 MB), at 580 GB/s 21.2 ms: ceiling **~4-8 ms**, not 12 (section 29 mixed busy-scaled ms with floors).
  gate+up runs at ~450 GB/s, down at ~400 GB/s: the down projection is the weaker half.
- MoE call time vs union size (least-squares over the traced calls, `moe_experts_situ_bf16_rows` gate+up / down):
  B 16 text (union 22-83, mean 62.5): 320.8 + 1.52 n / 178.8 + 0.88 n us; B 16 random (union 13-36, mean 23.1): 470.1 + 0.84 n /
  264.3 + 0.57 n; B 8 text (16-55, mean 41): 194.9 + 2.86 n / 110.4 + 1.50 n. The call time barely depends on the union: B 16
  random (23 experts, 69 MB of gate+up weights) takes *longer* than B 16 text (62.5 experts, 187 MB). The marginal cost per
  expert (~1.5 us) is far below its DRAM time (3 MB at 490 GB/s = 6 us), so at B 16 the situ kernels are not bandwidth-bound;
  their time follows the 128 (row, expert) pairs. Compute floor of one call: 128 pairs x 7.96 M MAC x 2 = 2.04 GFLOP = 131 us
  at 15.6 TFLOP/s, vs 650 us measured: the per-pair work (NVFP4 decode per pair, by the name "situ") dominates. For real text
  the bandwidth floor (~570 us per call at union 62.5) is close to the measured 656 us, so the real-text MoE ceiling stays
  ~4 ms; for workloads with more expert sharing it is much larger. ALU-bound per-pair work also explains why B 16 follows
  the GPU P-state (28c) while B 1 does not. Lever 2 (after in_proj): decode each expert's weights once per call for all its
  pairs (union kernel with shared decode), if the per-pair decode is confirmed as the cost.

## 30. in_proj rows (main's assignment, 2026-11-22)

**Step 1, #11428 (09:18-09:21, quiet, tree 50abd8291; `.scratch/inproj.log`).** `ink_mix_bench proj`, weights rotating past the
system cache, A B B A after a warm-up. Box label: start cold (B 16 61.15 ms, ALU 159.8), end warm (71.91, ALU 192.7).

| rows | default (P 1, RB = R) us/call, S 1 (2 passes) | GB/s | `PROJ_P=2` | `PROJ_RB=8` | P12 (ink_p12_bench) |
|---:|---:|---:|---:|---:|---:|
| 1 | 71.1 / 73.1 | 562 / 547 | | | 57.6 (bf16 72.8) |
| 2 | 74.0 / 76.2 | 540 / 525 | | | |
| 4 | 78.7 / 82.8 | 509 / 483 | | | 86.2 (bf16 96.2) |
| 8 | 109.1 / 109.4 | 368 / 367 | 130.1 / 135.9 | | |
| 16 | 183.2 / 183.0 | 220 | 238.6 / 238.8 | 265.9 / 260.1 | 322.2 (bf16 223.9) |

- Confirmed with a correction: 16 rows cost 2.6x the 1-row time (the bandwidth floor), not 6x. Weight reads do not repeat
  per row (one threadgroup per weight row, all R rows in registers); time grows with rows from 4 up: ~9.5 us per extra row.
- The kernel is ALU-bound at 8-16 rows, not bandwidth-bound: per (weight chunk, row) `ip_dot8` unpacks the 8 weight bf16s
  again (24 ops per 8 MACs), and the epilogue has every one of the 256 threads sum all R rows' 8 partials (R x 8 adds each)
  before thread 0 writes R values. The same-bits knobs that share x loads (P 2) or split rows (RB 8) are slower, as is P12 at 16
  rows (decode per row). A multi-row P12 kernel would not help while the cost is ALU.
- **Step 2 (built, 9ffcfd562): `ink_in_proj_gemvx_r<R>s8u2`, R = 2..=16**, only for the default family at P 1 / RB = R: weight
  chunk unpacked once per chunk for all rows, row r's partials summed by thread r (same additions, same order), which writes
  row r. Same products, same fma chain, same simd_sum, same sg-order sum, so bitwise the R = 1 kernel (and so the T = 1 P12
  kernel, by pack12's contract). The R = 1 instances, the T = 1 dispatch and the P12 path are unchanged.
  `INTARSIA_INK_PROJ_X=0` keeps the old form (docs/tuning.md). The chunk form (> 16 rows, prefill without the NAX library)
  routes its 16-row chunks and tail through it too; `in_proj_chunks_rows_exact` (new) checks it. Estimate (op counts):
  ~184 -> 110-130 us at 16 rows, ~2.5-3.5 ms per B 16 step. #11442 (queued): build, Metal tests, X=0 vs X=1 us/call.

**#11442 (09:59-10:02, quiet, tree 9ffcfd562; `.scratch/projx.log`).** Build clean; Metal tests all pass: inkmix 5/5 (rows-exact
at every R 1..=16, S 1 and 3, and the new chunk-form test at 17 / 40 rows), p12 4/4 (P12 T=1 bitwise bf16), rows_exact 11/11,
kernels table 7/7 (incl. the metal4 count). Box cold at start (B 16 58.07). InkInProj us/call, S 1, order X0 X1 X1 X0:

| rows | X=0 (old) | X=1 (gemvx) | X=1 | X=0 | per-pair diff |
|---:|---:|---:|---:|---:|---:|
| 2 | 71.33 | 70.30 | 71.11 | 70.95 | -1.0 / +0.2 |
| 4 | 72.87 | 72.27 | 73.57 | 71.96 | -0.6 / +1.6 |
| 8 | 82.49 | 80.70 | 80.68 | 91.50 | -1.8 / -10.8 |
| 16 | 162.12 | 151.99 | 151.24 | 173.92 | -10.1 / -22.7 |

- Bitwise and faster, but less than estimated: ~-10 to -23 us per call at 16 rows (~0.5-1.1 ms per B 16 step), within the
  noise at 2-4 rows. The unpack / epilogue work was not the main cost.
- Remaining cost at 16 rows (~150 us vs ~71 floor): the x reads. Every weight row's threadgroup loads all 16 x rows (16 x 4.6 KB
  per 4.6 KB of weights), ~640 MB of cache reads per call, ~4 TB/s at 150 us. Next (401604c38, #11457 queued): gemvx with
  P = 2 / 4 weight rows per threadgroup sharing each x load (`INTARSIA_INK_PROJ_XP`, same bits; the old kernel's P 2 lost to
  re-unpacking and RB passes), measured X0 / XP1 / XP2 / XP4 at 4 / 8 / 16 rows, A B C D D C B A, with inkmix rows-exact at
  each XP.

**#11457 (10:09-10:15, quiet, cold box B 16 58.07; tree 401604c38; `.scratch/projx2.log`).** inkmix rows-exact passes at
`INTARSIA_INK_PROJ_XP` 1, 2 and 4 (5/5 each), p12 4/4, rows_exact 11/11, kernels 7/7. us/call, S 1, A B C D D C B A:

| rows | A X=0 | B gemvx P 1 | C gemvx P 2 | D gemvx P 4 |
|---:|---:|---:|---:|---:|
| 4 | 71.57 / 72.08 | 71.64 / 71.60 | 73.01 / 72.70 | 93.62 / 92.87 |
| 8 | 82.43 / 82.56 | 80.78 / 81.69 | 106.74 / 106.65 | 152.50 / 152.42 |
| 16 | 161.99 / 161.84 | 149.91 / 149.93 | 222.78 / 222.69 | 203.59 / 203.62 |

- gemvx P 1 is -12 us per call at 16 rows (-1 at 8) in both pairs, stable: ~0.6 ms per B 16 step (48 calls) cold. P 2 / 4
  lose badly (register pressure; fewer threadgroups), so P stays 1 (`_XP` stays a measurement knob, default 1).
- Model that fits 4 / 8 / 16 rows: time ~ max(weights from DRAM at ~560 GB/s = 71 us, x re-reads from cache at ~4.2 TB/s):
  x traffic 160 / 320 / 640 MB per call -> 38 / 76 / 152 us. At 16 rows the x re-reads set the time.
- What the bitwise contract fixes: the 256-lane chunk mapping, each lane's chunk order, simd_sum over the same lanes and the
  sg-order sum, per (weight row, row). Within that, the x re-read can only be removed by keeping x close to the lanes across
  weight rows: a two-pass slice form (pass 1: a threadgroup owns one simdgroup's K slice of x for all 16 rows in threadgroup
  memory, 8-16 KB, and loops over many weight rows, writing each simdgroup's simd_sum to a scratch buffer; pass 2 sums the 8
  partials per output in sg order). Same bits by construction; gain uncertain (threadgroup-memory loads instead of cache
  loads, the same simd_sums, +4.4 MB scratch traffic). A non-bitwise B > 1 kernel (simdgroup-matrix GEMM over 16 rows) would
  approach the 71 us floor (~3.8 ms per B 16 step) but changes bits: opt-in only, flip is the user's call.
- Gates for gemvx P 1 default-on queued as #11463 (Metal suite, admission, greedy md5 64 / 256, B 1/8/16 X0 vs X1 A B A B B A
  B A with digests, warm).

**30a. Bound for (c), MoE weights once per expert (from #11355's B 16 text trace, no job).**
- MoE per B 16 text step (traced ms): 44 NVFP4 situ calls x 655.8 us = 28.9 ms, plus 4 MXFP8 calls x 969.9 us = 3.9 ms: 32.7 ms
  of 62.2 traced (53%).
- Per situ call: union 62.5 distinct experts (median 64, max 83) vs 128 (row, expert) pairs: ~2.05 pairs per expert.
- If the per-pair cost became per-expert: 656 us x 62.5 / 128 = 320 us per call, but that is below the call's DRAM floor:
  62.5 experts x 4.48 MB = 280 MB at 490 GB/s (m-moe's measured NVFP4 rate at 16 rows) = 571 us. So the floor binds: saving
  <= (656 - 571) x 44 = **3.7 ms per step**. The MXFP8 calls already run at their floor (8.2 MB per expert; ~513 MB at
  dfuse's 522 GB/s = 983 us vs 970 measured): ~0. At the 614 GB/s roofline instead of 490, the situ floor is 456 us and the
  bound 8.8 ms, but no Inkling NVFP4 kernel has measured above ~500 GB/s.
- **(c) bound ~3.7 ms (up to ~8.8 at roofline) vs (a)'s 3.8 ms: not >= 2x. Order: (a) first, then (c) if (a) gives < 2 ms.**

**#11463 gates for gemvx default-on (10:15-10:31, quiet, tree cd128d1af, every switch unset; `.scratch/gate_projx.log`): green.**
Metal suite: rows_exact 11/11, inkmix 5/5, rows_single 2/2, quick_coverage 1/1, gumbel 2/2, inkling2 11/11, inkling2_batch 8/8
(Metal, real model), core ops ids 7/7, docs_env_vars 2/2. Admission: ink_in_proj 8 admitted, conv_mix 1, ink_attn_decode 1,
moe_experts 4, matmul_p12 4, 0 fail / reject. Greedy md5 64 `65cde8f35a1802e3e2fadb44d94d080d`, 256
`61fd59fecb5e0d72ffd08f2c6d07855a` = ink/metal's. Batched digests identical in all 8 runs, X=0 and X=1 alike (B 1
`6d7e5d70e4d0014a`, B 8 `4eceea81c044812c`, B 16 `65fc922062715df4`, = #11352's NE 3 digests).

Speed, warm box (start B 16 sample 69.01, end 72.97 / ALU 179.8), B 8 / B 16 batched step ms, order 0 1 0 1 1 0 1 0:

| pair | X=0 | X=1 | B 16 diff | B 8 diff |
|---|---|---|---:|---:|
| runs 1 / 2 | 45.30 / 82.09 | 45.31 / 80.88 | -1.21 | +0.01 |
| runs 3 / 4 | 48.65 / 79.17 | 43.63 / 78.25 | -0.92 | -5.02 |
| runs 6 / 5 | 52.46 / 80.56 | 47.96 / 75.71 | -4.85 | -4.50 |
| runs 8 / 7 | 40.62 / 75.01 | 44.74 / 75.17 | +0.16 | +4.12 |

- B 16: mean -1.7 ms, median -1.1 ms per step, 3 of 4 pairs negative; the warm-box noise (+-4 ms) is larger than the effect.
  Consistent with the microbench (-12 us x 48 calls = -0.6 ms cold). Warm B 16 75-82 ms here. **gemvx is default-on**
  (bitwise, all gates green), `INTARSIA_INK_PROJ_X=0` reverts, docs/tuning.md has the row.

**#11466 slice form (10:44-10:58, warm; tree c969909ee; `.scratch/slice.log`): bitwise, slower.** inkmix rows-exact passes
with `INTARSIA_INK_PROJ_SLICE=1` at slice rows 64 / 16 / 200 (5/5 each), rows_exact 11/11. us/call S 1, A B C D D C B A:
gemvx 4 / 8 / 16 rows 75.4 / 126.4 / 192.0 and 75.5 / 115.1 / 195.7; slice (64 rows per group) 96.9 / 145.5 / 266.6 and
95.1 / 151.3 / 245.6; 32 and 128 rows per group the same within noise. So x re-reads were not the cost: staging x once per
slice made it slower (+70 us at 16 rows: the scratch round trip and a second launch, with the same simd_sums). **(a) gives
< 2 ms: stays opt-in (`INTARSIA_INK_PROJ_SLICE`), not default.** What remains per 16-row call is the bitwise reduction itself:
16 simd_sums per (weight row, simdgroup), 1.1 M per call (vs 69 k at 1 row), which the contract fixes. In_proj rows closed
at gemvx; next per main's rule: (c) MoE.

**30b. Slice form removed; in_proj rows closed.** The slice form (c969909ee) is reverted on the branch (kernels
`ink_in_proj_slice1` / `_slice2`, `INTARSIA_INK_PROJ_SLICE` / `_SLICE_ROWS`, the rows kernel's scratch, the tuning rows); its
numbers stay above. The remaining in_proj cost at 16 rows (~150 us cold vs the 71 us DRAM floor) is the reduction the
rows-exact contract requires: per (weight row, row) the same 256-lane split, simd_sum per simdgroup and sg-order sum as the
1-row kernel, 16 x the reductions of a 1-row call. Only a non-rows-exact kernel (one reduction order for all rows, e.g. a
simdgroup-matrix GEMM) goes lower: **(b), parked** (main: batched rows would no longer match solo output; it returns only if
the MoE work also falls short). What stays: gemvx default-on (`INTARSIA_INK_PROJ_X=0` reverts), `INTARSIA_INK_PROJ_XP` (P 2 / 4,
slower, measurement knob).

## 31. MoE at B 16 (c)

Code reading (kernels/metal/inkling2_moe.metal): in union mode the gate+up grid is (I / 16 row blocks, pairs) and the down
grid (H / 32, pairs); a threadgroup works only if its pair leads a run of up to 4 (`INK_XR`) pairs of one expert, else it
returns after `ink_lead` (ballots over the earlier pairs). The leader loads each weight chunk once but decodes it again for
every served pair inside `ink_chunk_dot` (NVFP4 nibbles -> half2, the same for every pair), and the down kernel re-loads and
re-decodes the chunk per pair (`ink_dn_row` per pair). Per element and pair that is ~4 decode ops next to ~1.5 x-extract
ops and 1 fma, which fits "time follows pairs, not union" (29a).

- Step 1 (measure, #11487 queued): `inkmoe_bench` NVFP4 at 8 / 16 rows with `INKMOE_POOL` (new: every token's experts from
  one pool of u per call; unset ~82 at 16 rows, 64, 24, 12), all policies, gate+up / down split.
- Candidate (bitwise, opt-in `INTARSIA_INK_MOE_DEC1=1`, 02857d7ad): decode each chunk once into 16 half2 + scales and run every
  served pair's unchanged fma chain on it (gate+up and down union kernels; down also loads the chunk once). #11489 queued:
  inkmoe rows-exact (every policy, routing incl. pool12 / identical / invalid / duplicates, T 2-16, NVFP4 and MXFP8) with
  DEC1 0 and 1, then us/call DEC1 0 vs 1 at pools unset / 64 / 24, A B B A.

**#11487 (11:22-11:28, warm start; `.scratch/moepool.log`), NVFP4 call us vs union, `inkmoe_bench`, first pass (Auto / Union):**
16 rows: union 83 (pool unset) 1020 / 946 us; union 57 (pool 64) 1011 / 829; union 24 (pool 24) 594 / 690; union 12 (pool 12)
577 / 579. 8 rows: union 51 450 / 454; 42 393 / 389; 23 330 / 329; 12 319 / 319. At 16 rows the call does not get cheaper
than ~580 us however few experts the 128 pairs use: per-pair work, not weight bytes, sets the floor (~4.5 us per pair).
Run-to-run noise at 16 rows is large (Auto and Union are the same configuration at T 16, yet differ by up to 180 us).

**#11489 (11:29-11:35, cold start; `.scratch/dec1.log`), `INTARSIA_INK_MOE_DEC1=1` (decode once for every leader).** inkmoe
rows-exact passes with DEC1 0 and 1 (2/2 each; every policy, routing, T 2-16, NVFP4 + MXFP8), rows_exact 11/11. Auto, whole
call, us, order 0 1 1 0:

| rows / union | DEC1 0 | DEC1 1 | DEC1 1 | DEC1 0 |
|---|---:|---:|---:|---:|
| 8 / 51 | 451.2 | 449.8 | 453.6 | 452.7 |
| 16 / 83 | 738.5 | 723.7 | 725.4 | 749.0 |
| 8 / 42 | 393.0 | 405.5 | 412.5 | 392.8 |
| 16 / 57 | 637.1 | 597.4 | 783.2 | 646.5 |
| 8 / 23 | 330.5 | 272.6 | 272.6 | 333.2 |
| 16 / 24 | 586.0 | 457.4 | 457.4 | 591.7 |
| MXFP8 16 / 57 | 873.9 | 841.6 | | |

- Bitwise. Large gain where experts are shared (16 rows / union 24: -130 us, -22%; 8 / 23: -18%), small at the real-text
  shape (16 / 57-83: -15 to -40 us, inside the noise) and a loss at 8 / 42 (+12-20 us: leaders serving one pair pay the
  decode's registers for nothing). Next (d9c7ff580): `INTARSIA_INK_MOE_DEC1=<n>` decodes once only for leaders serving >= n
  pairs; #11506 queued: rows-exact at n 1 / 2 / 3, bench n 0 / 2 / 3, then batch_step real text and random ids, n 0 vs 2,
  A B A B with digests.

**#11506 (11:36-11:51, warm; tree d9c7ff580; `.scratch/dec2.log`), threshold n = pairs a leader must serve.** inkmoe rows-exact
passes at n 1 / 2 / 3. `inkmoe_bench` Auto, whole call, us, order 0 2 3 3 2 0:

| rows / union | n 0 | n 2 | n 3 | n 3 | n 2 | n 0 |
|---|---:|---:|---:|---:|---:|---:|
| 8 / 51 | 458.8 | 461.9 | 455.0 | 453.8 | 450.5 | 452.0 |
| 16 / 83 | 740.3 | 752.6 | 738.2 | 735.3 | 732.5 | 742.6 |
| 8 / 42 | 391.9 | 384.6 | 393.1 | 395.4 | 386.8 | 392.0 |
| 16 / 57 | 618.7 | 551.5 | 570.2 | 567.2 | 554.2 | 618.4 |
| 8 / 23 | 330.2 | 275.8 | 286.6 | 284.5 | 275.3 | 332.1 |
| 16 / 24 | 585.6 | 462.2 | 469.0 | 467.5 | 460.5 | 584.8 |

n 2 removes the loss at 8 / 42 and is the best everywhere: -65 us (-11%) at the real-text-like 16 / 57, -125 us at 16 / 24.

batch_step, B 8 / B 16 batched ms, digests identical between arms in every run (text B 16 `846a505a4055b0ba`, random
`65fc922062715df4`):

| prompts | pair | n 0 | n 2 | B 16 diff | B 8 diff |
|---|---|---|---|---:|---:|
| text | 1 / 2 | 40.48 / 75.24 | 38.60 / 73.92 | -1.32 | -1.88 |
| text | 3 / 4 | 41.50 / 77.09 | 39.07 / 73.40 | -3.69 | -2.43 |
| random | 5 / 6 | 37.47 / 77.15 | 35.60 / 66.12 | -11.03 | -1.87 |
| random | 7 / 8 | 40.82 / 73.54 | 35.29 / 68.06 | -5.48 | -5.53 |

- **Real text, warm: B 16 -1.3 / -3.7 ms per step (mean -2.5, ~3%), B 8 -1.9 / -2.4 ms.** Random ids (rows share experts):
  B 16 -5.5 / -11.0 ms. Bitwise.
- Default flipped to n = 2 (5f6a7dc90; `INTARSIA_INK_MOE_DEC1=0` reverts, docs/tuning.md row). Gates queued as #11515: Metal
  suite, admission (moe_experts, moe_experts_mxfp8, ink_in_proj, moe_route), greedy md5 64 / 256, B 1/8/16 real text n 0 vs 2
  A B A B B A B A with digests.

**#11515 gates for decode-once default-on (n 2), 12:06-12:26, tree 5f6a7dc90, every switch unset; `.scratch/gate_dec1.log`:
green.** Metal suite: rows_exact 11/11, inkmix 5/5, rows_single 2/2, quick_coverage 1/1, gumbel 2/2, inkling2 11/11,
inkling2_batch 8/8, core ops ids 7/7, docs_env_vars 2/2. Admission: moe_experts 4, moe_experts_mxfp8 2, ink_in_proj 8, moe_route
1 admitted, 0 fail / reject. Greedy md5 64 `65cde8f35a1802e3e2fadb44d94d080d`, 256 `61fd59fecb5e0d72ffd08f2c6d07855a`
(= ink/metal). Digests identical in all 8 runs (text B 1 `67843d1dbf0249f5`, B 8 `74478685fe7299e9`, B 16 `846a505a4055b0ba`).

Real text, box cold at start (B 16 sample 56.35) and slow within minutes (end 64.73, ALU 176.3), batched ms per step:

| pair | n 0 B 8 / B 16 | n 2 B 8 / B 16 | B 16 diff | B 8 diff |
|---|---|---|---:|---:|
| runs 1 / 2 | 51.83 / 84.47 | 50.46 / 82.72 | -1.75 | -1.37 |
| runs 3 / 4 | 44.56 / 80.41 | 42.75 / 74.64 | -5.77 | -1.81 |
| runs 6 / 5 | 43.49 / 77.38 | 41.07 / 82.94 | +5.56 | -2.42 |
| runs 8 / 7 | 44.29 / 84.90 | 41.70 / 78.56 | -6.34 | -2.59 |

- B 8: -1.4 to -2.6 ms in all four pairs (mean -2.0, ~4.5%). B 16: mean -2.1 ms, 3 of 4 pairs negative, the box noise (+-5 ms in
  this slow state) is larger than the effect. With #11506's two text pairs (-1.3, -3.7), six B 16 text pairs: median -2.7,
  mean -2.2 ms. Decode-once stays default-on (n 2).
- Warm real-text B 16 at the tip: ~74-85 ms (190-215 tok/s) in this window; the cold best case ~56 ms.

**Where B 16 stands (warm, real text).** in_proj gemvx ~-0.6-1 ms, MoE decode-once ~-2.2 ms per step; both bitwise and
default-on. Remaining bitwise MoE headroom at union ~62 is ~1-2 ms (the bandwidth floor); in_proj's remaining cost is the
contract's reduction ((b), parked).

## 32. B 16 real-text step profile, floors and levers (brief 32)

**#11542 (13:10-13:13, quiet; tip 3278fbe6d = ef2598549 + doc, built in #11527 `--load cpu`; `.scratch/prof32.log`,
`.scratch/p32/`).** `batch_prof --ctx 1024 --b 16 --steps 12 --warm 3 --prompts text`, one 6-step warm-up, then cb trace, kernel
trace, kernel trace, cb trace, untraced. Box cold at start: warm-up 59.35 ms, cb run 1 63.00 ms (13:11:03); by 13:11:49 the cb run
was 71.94 and the end check slow (ALU 189.5 ms). Kernel shares scaled to the cb busy of the paired run; the table uses the early
pair (kernel run 2 / cb run 1: wall 63.00, GPU busy 62.04, scale 0.875). The late pair (71.94 wall) has the same shares within
0.2 points.

Floors at 546 GB/s. "Per visit" = the bytes each launch must read (48 visits: prelude 6 + core 12 x 3 passes + coda 6; a looped
layer's weights are re-read on each pass because pass p + 1 needs pass p's output). "Once per step" = the brief's floor, every
physical layer's weights (for MoE: the union over a layer's visits) read once. FLOP floor at 15.6 TFLOP/s (the fp32 FMA rate the
ALU check measures; the bitwise kernels use scalar fma).

| class | ms / step | share | floor per visit | floor once per step | FLOP floor | gap to per-visit floor |
|---|---:|---:|---:|---:|---:|---:|
| MoE experts gate+up | 20.03 | 31.8% | 17.6 (situ 44 x 62.3 x 2.99 MB = 15.1; MXFP8 4 x 62 x 5.5 MB = 2.5) | ~12.5 | 2.6 | 2.4 |
| MoE experts down | 11.14 | 17.7% | 8.8 (7.5 + 1.3) | ~6.3 | 1.3 | 2.4 |
| attention decode | 10.45 | 16.6% | 5.9 (KV 16 rows x 1024 x 8 x 128 x 2 x 2 B = 67 MB x 48) | 5.9 | 0.6 | **4.5** |
| in_proj | 9.03 | 14.3% | 3.5 (48 x 39.9 MB) | 1.75 | 2.0 | 5.5 (non-bitwise only, (b)) |
| unembed | 3.74 | 5.9% | 1.69 (922 MB once) | 1.69 | 0.9 | **2.05** |
| mix / blend / norm (244 launches) | 3.68 | 5.8% | ~0.3 (tiny planes) | ~0.3 | ~0 | ~3 (launch-bound) |
| dense GEMV (wo, 96 launches) | 2.86 | 4.5% | 1.25 (48 x 14.2 MB) | 0.62 | 0.6 | 1.6 |
| MoE route + gather | 1.07 | 1.7% | small | | | <1 |
| gaps (wall - busy) | 0.96 | 1.5% | 0 | | | 0.96 |
| **total** | **63.00** | | **~39** | **~29** | | |

- Once-per-step for MoE: the 12 core layers' expert union over 3 passes (estimated ~111 of 128 if the passes' choices were
  independent) instead of 3 x 62. Unreachable without reading a core layer's experts for all 3 passes at once, which the loop's
  data dependence forbids; listed because the brief asks for it.
- The head and wo run as **two 8-row launches each** (`row_chunks` halves 9..16 rows; trace: 2 x `logits_bf16_gemv_rows` t 8,
  1.80 ms each, i.e. each half reads the whole 922 MB head at ~510 GB/s; 96 = 48 x 2 wo halves). The opt-in one-launch form
  (`INTARSIA_METAL_ROWS_SINGLE=1`, `_mr8t<T>` shaders, bitwise, built in this lane, section 9) reads the weights once. Section 9's
  A/B (#10900) saw no gain inside its run-order bias; the trace now says each half pays full DRAM time.

**Top 3 levers (ms per B 16 step, cold numbers):**

| # | lever | ms saved (ceiling) | bitwise | effort | rule |
|---|---|---:|---|---|---|
| 1 | attention decode at the KV floor (10.45 -> ~6) | up to 4.5 (7%) | yes (per-row math stays) | > half day; needs a look at why 2x the KV bytes time (split choice / occupancy at 16 rows); kernel owned by ink-m-attn / ink-dfuse | report first |
| 2 | MoE to the per-visit floor (31.2 -> ~26.4), incl. the idle union threadgroups (non-leaders that ballot and exit; ~half of the 72 x 128 gu / 72 x 128 dn grid) | up to 4.8 (7.6%); idle-TG part unknown, likely <= 1 | yes | idle TGs: leader compaction needs a device-side leader list + indirect dispatch, ~1 day; first a cheap measurement of the idle share | report first; measure the idle share before building |
| 3 | head + wo as one launch (`INTARSIA_METAL_ROWS_SINGLE=1`, exists) | head ~1.7-2.0, wo up to ~1.2 (3-5%) | yes (rows_single tests) | measure + gate only, < 2 h | **measure and gate now** |

Not in the top 3: in_proj (5.5 ms, only non-bitwise, parked as (b)); mix / blend / norm (~3 ms, 244 launches, would need fusion
across controller ops, > 1 day).

Next: #11560 (queued): kernel trace with `INTARSIA_METAL_ROWS_SINGLE=1` (head / wo per launch) and batch_step B 8 / B 16 real
text off / on / on / off with digests, early in a slot. If B 16 gains >= 2% with identical digests: flip default, then gates.

**32a. Plan for lever 1, batched attention decode (brief 33; code on `ink/batch-attn` from main 2474c17b9, worktree
`.worktrees/ink-batch-attn`).** Where the 4.5 ms goes, from the kernel (`ink_attn_dec<GP 3, RT>`, kernels/metal/inkling2_attn.metal)
and the trace: **not KV re-reads** (each row reads only its own sequence's K / V once; the 67 MB per call are distinct bytes, so no
sharing across rows exists), and **not SWA vs full** (at ctx 1024 the window is 1024, so both layer kinds see 1024 keys and 16
splits of 64). It is **latency at limited occupancy**: a call is 8 kv heads x 16 splits x 16 rows = 2048 threadgroups of 128
threads, each a serial chain of q / r staging, 4 barriers, the rel bias (16 dependent fma reads per element), then only 8 key
steps per simdgroup, each a dependent 16-byte K / V load followed by its math (no load in flight while it computes), then the
merge and a partial write. One row alone (the solo call, 128 threadgroups) takes ~20-30 us; 16 rows take 246 us, ~8x one row
for 16x the threadgroups, i.e. the grid runs in roughly 8 waves of latency-bound threadgroups at 272 GB/s. Occupancy is held
down by ~10 KB of threadgroup memory per group and the row-table form's registers (it declares `kcv[16]` / `vcv[16]` for the
T-row case although a row-table row has at most one own key). **Build:** (1) the row-table kernel with its own-key arrays sized
1 (same code path for nnew <= 1, so the same bits) and (2) the key loop pipelined PD = 4 steps deep, ported from ink-d1row's
`ink_attn_body_p` (7d04f4ec2; bitwise the plain loop in 396 cases there; it lost at 1 row, where the bus is idle anyway, but here
each group's chain length times the wave count is the cost). Both as switches of the row-table launcher only; the 1-row solo
kernel and its dispatch untouched. Measure base / (1) / (2) / (1)+(2) per call (kernel trace) and B 16 step ABBA; keep what
explains most. Gate: batched == solo bitwise per row at B 2 / 8 / 16 (inkling2_batch digests + the rows-exact attention test).

**32b. Lever 1 / lever 2 builds (branch `ink/batch-attn` from main 2474c17b9, worktree `.worktrees/ink-batch-attn`).**
- 3eb059670: row-table fused decode variants `ink_attn_dec_rows_g{1,2,3}_{n1,p4,n1p4}` (own-key arrays sized 1; key loop
  pipelined 4 deep, `ink_attn_body_p` ported from ink-d1row 7d04f4ec2 with the row-table's PT instead of `p.t`; the plain
  body differs from d1row's only in that PT parameter), selected by the measurement knob `INTARSIA_INK_ATTN_ROWS`; the 1-row
  decode kernel instances and dispatch are unchanged (template defaults NN 16, PD 0 reproduce them).
- 22eef28e7: `INTARSIA_INK_MOE_DIAG_EXIT=1` (diagnostic: every expert threadgroup returns after the union leader election,
  outputs unwritten) and union-leader counts in `inkmoe_bench`. Idle cost per call ~= (diag-exit call time) x (1 - leaders /
  128); build leader compaction only if that is >= 1 ms per B 16 step (brief 33).
- Jobs: #11563 (`--load gpu,cpu`): build, inkling2_batch (tiny, every row bitwise the solo step, ring wraps) under each
  variant, inkling2_attn, rows_exact, inkmoe, lib kernels; then (quiet) per-variant B 16 attention trace, batch_step B 8 / 16
  text ABBA with digests, and the MoE diag A B B A at pools 64 / unset.

**32c. Lever 3 result, #11560 (13:38-13:47, quiet; tip binaries sha e5d5b7c53e9e4929; `.scratch/rs32.log`): no flip, closed.**
The box was slow throughout (warm-up B 16 82.1 ms, end ALU 190 ms / 10.8 TFLOP/s), so the absolute numbers are the slow state's.
- Trace with `INTARSIA_METAL_ROWS_SINGLE=1`: head one launch 4684.5 us (vs 2 x 1.80 ms = 3.60 ms as halves in #11542), wo 92.1 us
  per launch, 48 / step = 4.42 ms (vs 96 x 34.1 us = 3.27 ms). **The one-launch `_mr8t` form is slower** than the two halves:
  the halves overlap on the GPU, while one launch does the 8-row pass and the tail pass in series inside each threadgroup.
- batch_step text, B 8 / B 16 ms (off / on pairs, order 0 1 1 0 1 0 0 1): B 16 116.56* / 96.84, 95.64 / 97.47, 99.48 / 94.30,
  95.26 / 97.86 (*run 1 cold-start outlier); B 8 63.97 / 62.56, 60.81 / 62.65, 58.84 / 56.16, 60.02 / 60.87. No consistent
  gain. Digests identical in all 8 runs (B 8 `cf7c80eee29f176a`, B 16 `fadddc1af8732357`).
- ROWS_SINGLE stays opt-in and off; no PR (.worktrees/ink-batch-rs not created). The 2-launch halves are the better form here.
- #11563 / #11568 invalid: the shader failed to compile (`rv` declared twice in `ink_attn_dec`: the prefetch registers and the
  prologue's r vector), build.rs only warns, so the binaries had no metallib and every Metal test failed at backend creation
  (error text from short slot #11571). #11568 (the quiet timing job) still ran on that binary: no usable numbers, discarded.
  Fixed in 1d828e206 (registers renamed `pfk` / `pfv`); the build job now stops if build.rs reports no metallib, and the timing
  job starts only if the test job shows no FAILED / panicked. Re-queued as #11574. (The fix was checked with one `xcrun metal -c`
  of that shader on endor outside a slot, a few seconds of one CPU core; no cargo.)

**32d. Lever 1 and lever 2 results (#11574 tests, #11576 quiet, 14:11-14:21; ink/batch-attn 1d828e206; logs
`.worktrees/ink-batch-attn/.scratch/attnA.log` / `attnB.log`).**

Tests (#11574, `--load gpu,cpu`): inkling2_batch 8/8 (tiny, every row bitwise the solo step, B 2 / 5 / 16, ring wraps,
rollback, failed pass) under `INTARSIA_INK_ATTN_ROWS` base, n1, p4 and n1p4; inkling2_attn 1/1, rows_exact 11/11, inkmoe 2/2,
lib kernels 7/7. Real model: digests identical in all 8 batch_step runs across the variants (B 1 `67843d1dbf0249f5`, B 8
`74478685fe7299e9`, B 16 `846a505a4055b0ba`, = ef2598549's).

**Lever 1, attention: neither variant helps; the plan's hypothesis is refuted.** Kernel trace, B 16 real text, attention
decode per call (traced; the same harness for every arm): base median 203.3 us, n1 201.5, p4 260.7, n1p4 257.7. Trimming the
own-key arrays changes nothing (registers were not the limit), and the pipelined key loop is ~28% slower (its 4-deep K / V
register ring costs more occupancy than the extra loads in flight buy). batch_step B 16 (order base n1p4 p4 n1 n1 p4 n1p4 base):
88.84 / 89.56 / 95.97 / 88.45 / 84.37 / 85.16 / 78.33 / 81.50 ms, B 8 49.20 / 53.13 / 51.29 / 48.70 / 48.86 / 48.83 / 51.01
/ 49.42: no variant beats base beyond the slot's drift (slow box, end ALU 164 ms). Both the solo and the batched call run at
~300 GB/s (solo 4.2 MB in ~15 us; batched 67 MB in ~200 us), so the batched form already overlaps the rows; what limits both
is bytes in flight per core = resident threadgroups x the 4 KB a threadgroup has outstanding. Inside the bitwise contract the
per-key dot (16 lanes x 8 elements, shuffle tree), the key-to-simdgroup assignment, the 4-simdgroup merge and the 64-key split
are fixed (all set the rounding), so the free variables are load timing and occupancy; load timing lost here, occupancy has
~10 KB of threadgroup memory that the merge needs. **Recommendation: stop lever 1 within the bitwise contract** (the variants
stay on ink/batch-attn as a measurement knob, not merged). A faster attention needs a different split / reduction shape, i.e.
non-bitwise against the solo kernel, or a solo-kernel change that both paths share (outside this brief).

**Lever 2, idle union threadgroups: below the 1 ms rule; recorded, not built.** `INTARSIA_INK_MOE_DIAG_EXIT=1` (every
threadgroup returns after the leader election), NVFP4 16 rows, A B B A: pool 64 (union 56.9, leaders 59.5 of 128 columns):
50.5 / 50.4 us per call (gate+up 23.3, down 27.6) vs 936 / 624 us normal; pool unset (union 83.2, leaders 83.6): 54.2 / 54.2 us
vs 1052 / 1179 us. Idle share at the real-text union (~62): 50.5 us x (1 - 59.5 / 128) = 27 us per call, x 48 calls = **1.3 ms
per B 16 step as a serial upper bound** (the diag grid runs alone; in a real call the idle groups overlap the leaders' work, so
the real cost is lower), and compaction adds a leader-list pre-kernel per call (~48 x 3-5 us = 0.15-0.25 ms). Net <= ~1.1 ms
and likely ~0.5: not >= 1 ms with any confidence. Recorded and stopped per brief 33.

## 34. Brief 34: in_proj B 16 gap and mix / blend / norm launches (closed, nothing kept)

Branch `ink/batch-34` (from main 3f79f70ed; local record, unpushed, not for merge): c7ad2a823 x-resident in_proj
`ink_in_proj_gemvr_r<R>` behind `INTARSIA_INK_PROJ_R=<groups>` (opt-in), 8d29604fd the same with the next weight row loaded one
iteration ahead; `ink/batch-34-diag` 2cb222440 is a diag-only, non-bitwise tree (ip_dot8w without the x unpack). Logs and scripts in
`docs/results/2026-11-21-inkling2-batch/raw/b34/` (projr.log #11583, projr2.log #11584, diagx.log #11589, diagx2.log #11594;
projr.sh, diagx.sh, diagx2.sh, psoinfo.swift).

**(1) in_proj.** `ink_mix_bench proj` (weights rotated past the system cache), us per call, S 1, default gemvx P 1 RB = R:
R 4 71.1, R 8 80.4, R 16 149.6 (269 GB/s; floor ~73 us at 546 GB/s). R 4 and R 8 sit on the bandwidth floor; the whole gap
is R 8 -> R 16, which grows ~linearly with rows (~9 us per row past 8): fits time = max(73 us, R x 9.2 us).

- x-resident form (x of the R rows read once per threadgroup, chunk tid in registers, the 32 trailing chunks in threadgroup
  memory; a small grid loops over the weight rows): bitwise (inkmix 5/5 at PROJ_R 80 / 320 / 1280, incl. rows_exact 1..16
  and chunks 17 / 40), but slower at every grid: R 16 181-237 us (best 180.7 at 640 groups) vs 149.6; R 8 91-147 vs 80.4.
  With the one-ahead weight prefetch: R 16 180.7-185.3, R 8 98-106. So the x re-reads from cache are not the R 16 limit.
- PSO `maxTotalThreadsPerThreadgroup` is 1024 for gemv r1, gemvx r4 / r8 / r16, gemvr r8 / r16: no register cap visible.
- ALU diag (x unpack removed, wrong values; ABAB A B A B B A B A in one quiet slot, box in the slow state): R 16 base
  277 / 238 / 222 / 239 us vs diag 216 / 210 / 213 / 214 (median -10%); R 8 too noisy (83-169). Removing one of the ~3 ALU ops
  per element and row buys ~10% of in_proj; an f32 x copy would cost x load bytes on top. Ceiling ~10% x 9.03 ms = ~0.9 ms
  per B 16 step (~1.2%), under the 2% bar.
- Closed. Not tested: the per weight row epilogue (16 simd_sums, a barrier, a 16-thread 8-partial sum) as the per-row slope;
  a non-bitwise diag skipping it would show it, but a bitwise fix keeps the same reduction tree.

**(2) mix / blend / norm** (dropped from this lane by main ~15:40, owned by ink-metal; the numbers below stay as input). Kernel traces of real decode (`realprof/k2|k12|k16.jsonl`, median us per launch at B 2 / 12 / 16):
`conv_mix_bf16_rows` (the row kernel, 96 / step) 17.0 / 16.9 / 16.5, flat in B: one threadgroup per row, the cost is its
dependent load / reduction chain, not the launch. The blend launches (#1 46 / step 7.2 / 16.5 / 22.6, #2 11.5 / 20.0 / 23.1,
#3 4.5 / 6.0 / 6.5) grow with B. Launch overhead: 0.88 us per dependent dispatch (ink-dfuse #11336), so all 244 launches
cost ~0.21 ms per step; merging adjacent row ops saves at most that, under the 1 ms bar. Closed, not built. A lever with
more room would split one row's chain over several threadgroups (per-simdgroup partials to device memory, summed in the same
order in a second pass: bitwise possible), ceiling ~half of the 1.6 ms row-kernel time; not in this brief's scope.

## 35. Brief 35: MoE expert kernels (gu, dn), solo and batched

Branch `ink/moe-35` from main 17b999272, worktree `.worktrees/ink-moe-35`. Raw logs and scripts:
`docs/results/2026-11-21-inkling2-batch/raw/b35/`. Harness: ink-dfuse's moevar (`raw/moevar.swift`, `moevar_gen.py`;
the real shader text with substitutions, compiled at run time, random weights, 5 rotating layer copies, GPU time),
extended with an ALU-only diag (every threadgroup reads the same few weight rows, cache-resident), a residency probe
(a device atomic counting threadgroups in flight) and a constant-table decode variant.

### Step 1: why 411-450 GB/s at one row (M1 #11601, M2 #11604, M3 #11606)

Shape per NVFP4 visit at T 1: 8 experts, 35.83 MB (gate+up 23.89, down 11.94). 44 NVFP4 visits + 4 MXFP8 per token.

| | gate+up `ink_moe_gu_nv_a_x` | fused down `ink_moe_dn_f_nv_a_bf16` |
|---|---|---|
| grid | 72 row blocks x 8 experts = 576 TGs x 128 threads | 1 x 288 TGs x 128 threads |
| rows per TG / expert slice | 16 gate + 16 up rows of one expert; 72 TGs per expert | 8 output rows, all 8 experts (simdgroup k % 4: 2 slots in sequence) |
| per lane | 8 lanes per row, 9 chunk pairs (gate, up), 1 chunk in flight | 4 lanes per row, 9 chunks per slot x 2 slots, 1 in flight |
| TGs per core (40 cores) / resident at once | 14.4 / **all 576 in flight** (one wave) | 7.2 / **all 288 in flight** (one wave) |
| shipped us (moevar) | 47.8-49.9 (480-500 GB/s) | 29.1-36.1 (330-410 GB/s) |
| no-math stream of the same bytes (dfuse N, best geometry / own geometry) | 41.8 / 47.7 | 21.6 / 24.4 |
| ALU-only (math, x loads, epilogue; weights cache-resident) | 27.9 (24.1 with one pair per leader) | 15.7 |
| in the real model (kernel trace, commits serialised, 64 tokens) | median 57.8 | median 31.4 |

MXFP8 at one row (4 visits): gate+up 84 us, down 43 us, 510-520 GB/s (inkmoe_bench). Real-model trace 88.6 / 44.2.

- Both kernels run as a single wave (residency probe: every threadgroup of the grid is in flight at once; there is no last
  wave to fill). The time is each lane's serial chain: load, decode + fma, next chunk, with 1 chunk in flight.
- Gate+up is ALU-heavy: its math alone is 58% of its time (27.9 of 47.8 us). It runs at its own geometry's no-math speed
  (47.7) because the decode / fma of one chunk and the next chunk's load do not overlap within a lane, and every lane on the
  chip is in the same phase. More loads in flight or more threadgroups (dfuse N: gu_nb2, gu_sg8, gu_lg16) cost registers or
  add work and do not help; dropping the 4-slot arrays (one pair per leader) takes 2 us.
- The fused down is latency-bound: half the threads of gate+up, twice the dependent chain (2 slots in sequence), math 15.7
  of 29-31 us. 3 chunks ahead takes ~4.5 us; 256 threads (one slot per simdgroup) is slower (33.9), and 3 ahead + 256
  threads (27.8) is slower than 3 ahead alone (26.4).
- So 411-450 GB/s = gate+up at ~490 (ALU-bound chain) + down at ~370 (latency-bound chain) + the dependent dispatch between
  them (~2 us, dfuse N). Perfect stream of the visit: 63.7 us (562 GB/s); shipped 77-81.
- Dead end: decode by a 256-entry constant table (same values, bitwise): ALU-only gate+up 35.0 vs 24.1 us, the shipped
  kernel 48.3-49.4 vs 47.9 (constant-cache lookups with divergent indices cost more than the bit decode).

### Step 2: levers

- **T = 1 (`INTARSIA_INK_MOE_T1`, ink-dfuse fcd05b172, ported onto main):** one pair per union leader in gate+up and 3
  chunks ahead in the fused down; same chunk order and fma chain. moevar visit 72.2-72.8 us vs 80.2-81.7 shipped (M3).
- **T > 1 (`INTARSIA_INK_MOE_DNU3`, new, opt-in):** the by-pair down `ink_moe_dn_u3_*` with 3 chunks ahead (the decode-once
  leaders too). inkmoe_bench down: T 4 random 102 -> 87 us, pool 62 94 / 97 -> 84 / 85; T 8 random 176 / 187 -> 148, pool
  62 150 / 167 -> 142 / 151; T 16 too noisy in M3 (the box went slow mid-run). MXFP8 T 16 pool 62: 283 / 287 -> 289 / 290
  (no gain). Crossed rows-exact sweep (T-row calls on dn_u3, one-row references on the shipped kernels): 7520 launches,
  bitwise.

### G1 (#11615, 15:10-15:25, tree a1a7b0cf2): OFF = defaults, ON = T1 + DNU3

- md5 64 `65cde8f3...` and 256 `61fd59fe...` OFF = ON; inkling2_batch (Metal) with ON 8/8; batch_step digests identical
  in all 8 runs (B 1 `67843d1dbf0249f5`, B 8 `74478685fe7299e9`, B 16 `846a505a4055b0ba`).
- **Solo (generate, 257 tokens, decode tok/s, ABAB):** short OFF 76.12 / 76.22 / 76.04 / 76.37, ON 77.97 / 77.66 / 77.88 /
  77.90: **+2.2%** (every pair). 5k (prompt4k, ~5.2k context) OFF 70.05 / 70.79 / 70.62, ON 72.46 / 72.61 / 72.72:
  **+3.0%** (every pair). Over the 2% bar at both.
- **B 16 (batch_step real text, batched ms/step, A B B A B A A B):** OFF 69.64 / 67.94 / 66.33 / 66.78, ON 65.19 / 67.20 /
  67.23 / 66.25: -1.8% mean, pairs -4.45 / -0.74 / +0.90 / -0.53: under the bar. B 8: OFF 36.71 / 37.36 / 37.44 / 36.47, ON
  37.91 / 37.22 / 37.56 / 37.52 (+1.5%, two pairs +1.1 / +1.2 ms): the microbench gain at 8 rows does not show in the step.
  B 1 column (T = 1 through the batch path): 14.82 / 14.86 / 14.86 / 14.78 -> 14.62 / 14.51 / 14.51 / 14.59.
- Decision by the rule: **T1 default-on** (solo bar met; DNU3 plays no part at T = 1). **DNU3 stays opt-in** (B 16 under
  the bar, B 8 possibly worse).

### G2 (#11624, tree 74be1f7fa: T1 default-on; OFF = `INTARSIA_INK_MOE_T1=0`): green

- rustfmt clean. Metal: inkmoe 2/2 (crossed T1 and DNU3 sweeps), rows_exact 11/11 (4 ignored); models (Metal): inkling2
  11/11, inkling2_batch 8/8; docs_env_vars 2/2.
- Admission `moe_experts`: 4 admitted, 0 fail / reject, OFF and ON.
- md5 64 `65cde8f35a1802e3e2fadb44d94d080d`, 256 `61fd59fecb5e0d72ffd08f2c6d07855a`, OFF = ON (= ink/metal reference).
- G2 / G3 short and long: self-goldens OFF vs ON EQUAL in all four; decode-mode G2 2.928e-3 short / 2.761e-3 long, G3
  8.420e-3 / 1.229e-2 (the same as ink-metal's #11206 and dfuse's job G).
- `plan show` Qwen3-8B: byte-identical to main 17b999272.
- With G1: batched == solo digests (batch_step B 1 / 8 / 16 digests unchanged, inkling2_batch 8/8).

PR: branch `ink/moe-35` (main 17b999272 + T1 port, default-on; DNU3 opt-in). Merge waits for main's #491 freeze, then a
rebase onto that main.
