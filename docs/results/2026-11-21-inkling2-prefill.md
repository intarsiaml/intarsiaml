# Inkling-2-Mini prefill on Metal (lane ink-prefill, endor M5 Max, 2026-11-21)

Branch `ink/prefill` from 1dacac3a8 (ink/metal). Not pushed or merged. Every GPU run had a fleetq slot (timing `--quiet gpu,cpu`).

## Briefs received

1. 2026-11-21 (coordinator, via SendMessage): **both go-aheads approved; keep the order MoE experts first, the looped blend
   second, rel_mode third.**
   - **rel_mode conditions:** (1) `ink.rel_mode` defaults to 0 and mode 0 stays exactly today's contract; (2) a CPU reference
     test asserts the mode-1 pair equals the mode-0 pair bitwise; (3) Metal admission sweep cases for mode 1 at prefill shapes
     (T 17, 512, 2048; SWA and full visits; tau past 8192); (4) the graph picks mode 1 only when the backend admits both calls in
     mode 1, the CUDA lanes need nothing; (5) rerun G3 short/long/4k with mode 1 active on Metal and update the op-list entry
     in the spec doc on this branch.
   - **Blend:** this lane owns the prefill path (T > 16), in its own file plus the one-line hook in `launch_mix`, after
     ink/m-mix lands in ink/metal. m-mix keeps decode (T <= 16). No edits to m-mix's kernel files beyond the hook.
   - Report prefill tok/s at 2k / 8k / 12k next to oMLX's ~1300 when available.
2. 2026-11-21 (coordinator, via SendMessage): **ink/metal is integrated and runs end to end** (tip a88db4620: m-attn, m-moe,
   m-mix up to d99d2c0fc, ink/skel; G1-G3 pass on Metal). Its baseline: 5k-token prefill 1219 tok/s at chunk 256, 2221 tok/s
   at chunk 2048 (~920 ms per 2048 chunk vs this lane's ~700 ms kernel sum).
   - Merge ink/metal a88db4620 into ink/prefill now (unblocks the blend hook).
   - Measure end-to-end prefill at 2k / 8k / 12k, chunk 2048, on the merged tip before and after each kernel switched on (bench
     command in the ink-metal results doc).
   - Report the 920 vs 700 ms split: per-chunk host gap, router / routing, whatever else is missing. Prefill host gaps belong to
     this lane (ink-hostgap is decode only).
   - One line on 7.3 vs 11.9 ms per MoE visit (m-moe's setup: weights cycled from DRAM, union = 128 experts); no more time on it.
3. 2026-11-21 (coordinator, via SendMessage): endor's quiet-GPU line is 14+ jobs (~9 h). Builds, admission and tests go as
   default-load jobs with short honest `--expect` (they backfill); only sweeps and e2e tok/s take `--quiet gpu,cpu`. Merge
   ink/metal at a0751f302 (the rows_exact pin-list test fix).
4. 2026-11-21 (coordinator, via SendMessage): ink-d1row found inkmoe_bench can draw every call's experts from one 128-expert
   pool (~570 MB), so ~16% of expert reads hit the system cache, while in the stream each layer has its own pool (hit rate ~0);
   at 1 row that explains the bench-vs-stream gap fully. Check whether this lane's sweep and the MoE lane's numbers differ in
   pool / rotation setup; before choosing a default, rotate 24 per-layer pools in the bench or confirm with the in-stream
   per-chunk time.
   - Answer: this lane's 2048-row sweep cycles 5 NVFP4 / 3 MXFP8 pools of 573 / 1051 MB (`w[c % layers]`), so a pool comes
     back only after 2.3-3 GB of other weights: no cross-call cache reuse. At 2048 rows the experts are compute / staging-bound
     (a pool is ~0.9 ms of DRAM time of a 7.3 ms call). In-stream (8k prompt, kernel trace) the NVFP4 experts take 8.3 ms per
     visit (gu 5.25, dn 2.79, gather 0.25) vs the bench's 7.3: the stream's routing is less balanced than random. Any MoE
     default is decided on the in-stream per-chunk time (#10884 profiles m128 in-stream), not on the bench.

5. 2026-11-21 (coordinator, via SendMessage): next steps in order.
   1. Merge gate for ink/prefill b178a17f9: localeval speed-plain at 1k / 4k / 16k, base ink/metal vs this branch, own fleetq
      slot via the local-model-bench skill. Pass: prefill / TTFT better at 4k and 16k, decode tok/s within 1% at every length.
      On pass, send the hash to the coordinator (the integration lane merges).
   2. Router microbench crash: find the cause before the merge; harness -> fix or drop the bench; kernel at some shape -> fix
      and add the shape to admission.
   3. Host gap (57 ms / chunk, ~8%): batch the prefill command buffers (first among the speed items).
   4. Router on the accelerators: 14 ms -> ~1 ms per chunk.
   5. MoE (319 of 646 ms): first the shared-unpacked-tile idea (one unpack feeding two row tiles of the same expert); second, as a
      measured experiment, dequant each active expert's weights once per chunk into a bf16 scratch + accelerator matmul. Report
      µs per chunk vs the current kernel for each. Default-on only if bitwise or G2/G3 pass, and faster by the median of 3
      interleaved reps.
   6. rel bias variant: parked; ink/prefill-rel stays unmerged, graph default off.
   Work only in this worktree; non-Metal tests on spark2 (`--load cpu`); timing on endor `--quiet gpu,cpu`; no call or wait
   over 4 min.

6. 2026-11-21 (coordinator, via SendMessage; rule for all Inkling lanes): this session runs on endor, so no local
   cargo build / check / test or shader compiles outside a fleetq slot (they load endor's CPU and skew quiet timing jobs).
   macOS compile checks on bespin (`fleetq run --host bespin --load cpu --mem 6 --expect <min>`, a git-archive copy in
   ~/intarsiaml-<lane>, toolchain in ~/.cargo/bin; no GPU timing or real model there); CPU tests on spark2; GPU work as endor
   fleetq jobs. Reading files, git and editing locally are fine. (Before this rule this lane ran a few local `cargo check` and
   `xcrun metal -c` checks on endor; from here on they go to bespin.)

7. 2026-11-22 (coordinator, via SendMessage; bug found by ink-dfuse): without metal4 (the no-metal4 tiny suite) ink_in_proj has
   no kernel at prefill shapes m 31-63 (NoKernel: that range is NAX-only); ink_route correctly falls back to the pair there.
   Fix: a non-NAX ink_in_proj path for medium m, bitwise with what ink/metal uses today on non-metal4 devices for the
   neighbouring m, or route m 31-63 to the existing GEMM fallback. Check: the tiny inkling2 suite with metal4 forced off passes
   on endor, plus rows-exact; default-load slot, no local builds. After the queued jobs; report the hash.

8. 2026-11-22 (coordinator, via SendMessage): approved with conditions.
   1. Recipe: `[tuning.metal] prefill_chunk = 2048` in recipes/inkling2.toml as its own commit on ink/prefill, only if #11109
      passes all of: G2/G3 short+long at 2048 within the limits (G2 4e-3 / 6e-3, G3 1.05e-2 / 1.65e-2); speed-plain serve 2048
      >= 256 at every length (prefill and decode, beyond spread, or within 1 % for decode); serve log shows "prefill chunk 2048".
      Commit message cites the #11109 numbers. Revert = that one commit.
   2. Merge: #10967 counts as the speed-plain gate for ink/prefill 378dc214e (1k / 4k clean wins, 16k positive in both pairs,
      decode within 1 %). Send ink-metal (af43898d63e5b5190) the hash to merge (378dc214e, or the recipe commit on top once
      #11109 passes).
   3. If #11109 fails a G2/G3 limit at 2048: stop and report the numbers; no other chunk sizes without asking.
   4. Results doc note: every localeval / accuracy number for Inkling on Metal before that commit ran at 256-row chunks.
   Next, in order: prefill cb batching (57 ms / chunk), router NAX, MoE unpack sharing; also the non-metal4 ink_in_proj
   m 31-63 fix (brief 7).

9. 2026-11-22 (coordinator, via SendMessage): command-buffer batching closed; cold-start page-in (~0.2 s) parked (serve
   keep-warm covers it, no residency work). While #10982 / #11105 / #11109 wait: (1) from existing traces (no new endor jobs)
   rank steady-state prefill ops at chunk 2048 (chunks 1+, warm) by ms/chunk next to each op's M5 Max NAX compute floor:
   experts, looped blend, attention, in_proj, router, everything over 5 ms/chunk; (2) pick the op with the largest
   ms-above-floor not covered by router NAX / v65, write a one-paragraph plan (expected ms/chunk saved, bitwise vs
   within-gates), build it on a new branch off the ink/prefill tip, flag default off; (3) compile checks on bespin, CPU tests on
   spark2, GPU only in endor fleetq slots, no local builds on endor; (4) report the ranking and the pick; the coordinator
   answers within the hour on disagreement, otherwise keep building.

10. 2026-11-22 (coordinator, via SendMessage): agreed on all three: router NAX stays reverted (+0.9 % is not worth a
    shared Logits candidate), v65 and dq1 are dead. Next, while #11119 / #11105 wait: (1) MoE NVFP4 diagnosis in this doc
    before any new variant, from traces or counters: (a) rows-per-expert distribution per visit at 2048 (min / median / max,
    experts under 16 rows), (b) per-expert kernel time vs rows (fixed cost), (c) dequant vs MMA share of the inner loop,
    (d) gather / scatter and routing time between expert calls; (2) then propose one variant with expected ms/chunk and why
    v65 / dq1 missed, and send diagnosis + proposal before building; (3) if MoE has nothing above ~20 ms/chunk recoverable,
    take the ConvMix + looped blend fusion (~15 ms) instead, bitwise, flag off. Compile on bespin, GPU only in endor slots.

11. 2026-11-22 (coordinator, via SendMessage): note for reading #11109. ink-metal found a G2 decode-mode regression on
    ink/metal since the m-mix/batch merge 45887198b (decode rows 1.109e-2 short / 9.205e-3 long vs limits 4e-3 / 6e-3, from
    chat_reasoning and chat_system; prefill-mode G2 passes; bisect #11142 running). If the ink/prefill base contains that
    merge, judge the chunk change by 2048 vs 256 on the same binary (prefill-mode G2 and G3 within limits, decode-mode G2 no
    worse than the 256 arm within 5 %), report both arms, and do not stop on the known decode failure alone. If the base
    predates it (decode rows near 2.6e-3), the original limits apply unchanged.
    **This lane:** 45887198b is not an ancestor of 378dc214e (the #11109 binary) or of the ink/prefill tip, so the original
    limits apply to #11109.

12. 2026-11-22 (coordinator, via SendMessage): this lane cherry-picks ed27f4462 onto ink/prefill-merge itself; cargo check +
    the cpu Inkling tests on bespin; then send ink-metal (af43898d63e5b5190) the one new merge hash in place of the old one (ink-metal
    never cherry-picks lane code). The prefill_chunk = 2048 recipe commit goes on top only if #11109 passes G2/G3 at 2048; if
    not, send the hash without it and tell the coordinator. ink-metal holds merges until the bisect lands (~04:15).
    **Done:** ink/prefill-merge 96740cbe2 = 378dc214e + ed27f4462 (cherry-picked). Bespin #11173: cargo check clean;
    cpu tests inkling2 8/8, inkling_chat 7/7, ink_attn_decode 1/1. Sent to ink-metal as the merge hash; the recipe commit
    follows only if #11109 passes.

13. 2026-11-22 (coordinator, via SendMessage): COMMIT the recipe (1k miss is spread, not speed; 4k decode -1.5 % in one pair
    only, chunk size does not touch decode kernels; no rerun). Recipe commit on ink/prefill-merge on top of 96740cbe2 with
    the 1k spread and 4k decode caveats in its message; send ink-metal the new hash in place of 96740cbe2; note peak memory
    at 16k with chunk 2048 if serve logged it.
    **Done:** 05f1c702e (`[tuning.metal] prefill_chunk = 2048`), bespin #11191 recipe tests 15 + 11 pass, cpu inkling2 8/8;
    sent to ink-metal. **Peak memory at 16k / chunk 2048: not logged by serve** (only the 19.53 GB keep-warm residency of
    the weights); ink-metal's combined gate measures it.

14. 2026-11-22 (coordinator, via SendMessage): prologue split default-on APPROVED on ink/prologue-split, conditional: (1) the
    flip in its own commit, `INTARSIA_INK_PROLOGUE_SPLIT=0` reverts it, docs/tuning.md says default on and how to turn it
    off; (2) localeval ABAB on that one binary (=0 vs =1, >= 2 rounds each, fleetq --quiet gpu,cpu); keep only if split-on
    prefill beats off by more than the off-run spread at 4k, 8k and 16k, is not slower at 1k, decode within 1 %, and both
    arms exit 0 with non-empty output; (3) pass: send hash + numbers (ink-metal merges after the current merge); fail: stays
    opt-in, report. Order on endor: #11187 first, the ABAB after, both behind the existing queue. The MoE proposal still goes
    to the coordinator before building.
    **Status:** flip e7cd4931e on ink/prologue-split. The merge-ready branch ink/prologue-split-merge is 297cebadc + the split +
    the counter + the flip = **61980c882**. Bespin #11219: check clean, docs_env_vars 2/2, cpu inkling2 8/8. ABAB queued as
    #11221 behind #11187. The binary builds the tests first and the CLI last (hazard 2). Recipe chunk 2048; 8k comes from p8k
    generate pairs, since speed-plain has no 8k cell.
    Also: docs_env_vars would have failed on the merge (this lane's four variables had no rows). ink/prefill-merge is now
    **297cebadc** with the rows; ink-metal takes it after its combined gate #11206 and keeps this lane's row text.

15. 2026-11-22 (coordinator, via SendMessage): MoE step-1 probe APPROVED with an extra arm and guards. (1) One endor job,
    same binary: (A) normal, (B) bit copy instead of the decode arithmetic, (C) weights constant (no NVFP4 weight loads from
    DRAM; decode ALU, staging, MMA and stores kept); real replayed routing + balanced-128, 3 reps each. (2) Probe bits not
    reachable from serve / generate (cfg / feature for the bench only, or refuse outside the bench); admission never lists them;
    removed or cfg-gated before any merge. (3) Rule: B saves >= 6 us/tile -> build the exact cheaper decode (new file, default
    off, bitwise vs moe_blk_bf16 on real rows), report before any flip; C dominates -> send a one-paragraph weight-reuse idea
    before building; neither -> stop MoE, go to the ConvMix + blend fusion. (4) small-expert GEMV side option parked. Queue
    behind #11221.
    **Status:** probes built (f753bb1c8). kernels/metal/inkling2_moe_probe.metal holds a copy of the gen1 NVFP4 m64 kernel
    with PROBE 0 (unchanged), 1 (B: bit copy) and 2 (C: no weight loads). They are selectable only via `set_moe_probe`, which
    exists only with the `ink-probe` cargo feature (no env var, not a registry kernel). The hook returns None without the
    feature. Rust compiles on bespin with and without the feature. Job #11235 (behind #11221) builds the bench with the
    feature in its own target dir and runs arms a / 0 / 1 / 2 x real / balanced-128 x full / gu / dn, 3 reps.
    Before any merge: drop the probe file and hook, or keep them feature-gated (ink/prefill-merge does not contain them).

16. 2026-11-22 (coordinator, via SendMessage): NAXSKIP guard: gate it behind `ink-probe` and remove the env var, as a separate
    follow-up commit on top of 297cebadc (leave 297cebadc as is; ink-metal has it for the #11206 merge). Before committing,
    grep scripts / .scratch job files / docs for INTARSIA_INK_MOE_NAXSKIP and move live users to the feature build or note
    them gone. Checks: builds with and without ink-probe, docs_env_vars passes, cpu inkling2 passes; one endor smoke inside
    #11235's feature build. Send the hash (small merge after 297cebadc lands). Probe guards for #11235 ok; rule as before.
    **Done:** ink/prefill-naxskip **2455b7abc** (on 297cebadc) adds `ink-probe = []`. `nax_skip()` is 0 without the feature,
    `set_skip` exists only with it, the env var is gone, the docs/tuning.md row is dropped, and inkmoe_bench asserts that
    INKMOE_NAXSKIPS needs the feature. Users grep: no script or job file sets INTARSIA_INK_MOE_NAXSKIP; the lane's job files
    use the bench's INKMOE_NAXSKIPS, and the one still queued (#11235, mp1.sh) builds with the feature. Bespin #11237:
    check with and without the feature clean, docs_env_vars 2/2, cpu inkling2 8/8. The same change is on ink/prefill as
    31f117a00, so #11235 builds it with the feature on endor (the smoke).

Brief 14 ABAB, first attempt (#11221, binary 61980c882):
- Build, bitwise and rows-exact (split default on) pass. All four serves logged "prefill chunk 2048 (recipe inkling2, metal)".
- **The localeval arm never ran**, because of a script bug: the log redirect was relative inside `cd /opt/code/deepeval`, so localeval exited 1 before starting.
- p8k generate pairs (all exit 0, non-empty): split off 2581 / 2708, split on 2851 / 2821 tok/s.
- Rerun with absolute paths: #11272 (psab2.sh).

17. 2026-11-22 (coordinator, via SendMessage): with probe rep 3 matching, cleared under brief 15 to build the ConvMix + looped
    blend fusion (bitwise, flag off) without waiting. Also one measure-only K-step probe of the MoE prefill kernel: 2x and 4x
    the current K tile (fewer barriers / staging rounds), balanced-128, 3 reps each, in the fusion's first quiet slot if it
    fits; default unchanged. Reason: the unexplained ~10 us of 15.2 us per tile is ~60 ms per chunk, more than the fusion's
    ~15. Report fusion ms/chunk with the bitwise result, and K-step us/call per arm.

18. 2026-11-22 (coordinator, via SendMessage): 1k rule for the split. If #11272 pair 2 matches pair 1 (4k / 16k on above
    off's range, decode within 1 %), settle 1k with engine generate timing, not localeval: p1k split off vs on, 3 alternating
    pairs, binary 61980c882, inside a job already queued (#11282 or #11290), no new slot. If the on median p1k prefill is
    >= 0.98x off and the 8k pairs hold, the split goes default-on: send ink-metal the split hash with the numbers. Below
    0.98x it stays opt-in; report. Label the 1k evidence "generate timing" (localeval rejected the cell).
    **Done:** the p1k pairs were appended to #11282 (kp1.sh). The binary is copied to .scratch/bin9k (61980c882). The prompt
    .scratch/p1k.txt is the first half of p2k.txt; generate prints the token count.

19. 2026-11-22 (coordinator, via SendMessage): option (c). Judge 4k by engine generate timing, the same instrument as 1k
    and 8k (bitwise output, real saving: localeval spread does not decide). (1) p4k generate pairs, 3 alternating off / on,
    binary 61980c882 (.scratch/bin9k), next to the p1k pairs: in #11282 if its expected time still covers it, else #11290,
    else a new quiet slot. (2) **Rule, fixed before any numbers: default-on only if BOTH hold: 4k generate median on >=
    1.00x median off, AND 1k generate median on >= 0.98x median off.** Both hold: send ink-metal 61980c882 with the numbers.
    Either fails: stays opt-in, FINAL, no further round. (3) Label the 4k evidence generate timing; keep the failed
    localeval 4k pair 2 in the doc.
    **Placement:** #11282 was queued with --expect 15 and already carries the K-step sweep plus the p1k pairs, so it does not
    cover 6 more runs. The p4k pairs go in #11290 (--expect 35). The p1k pairs stay in #11282. The prompt
    .scratch/p4k.txt is the first half of p8k.txt (~4k tokens; generate prints the count).

20. 2026-11-22 (coordinator, via SendMessage): build the blend piece 1024 gate. **Rule, fixed before any runs: default-on
    only if ALL hold:**
    (1) G2 and G3, short and long, at chunk 2048: numbers identical to piece 256 (it is bitwise); greedy md5 at 64 and 256
    tokens identical.
    (2) Box check first: one default p8k run within 5 % of 2362 tok/s, else BOX SLOW and stop.
    (3) Speed by engine generate timing, 3 pairs per prompt length, order randomized per pair (recorded): 1k, 4k, 8k. Median
    ratio 1024 / 256 >= 1.00x at 4k and 8k, >= 0.98x at 1k.
    All hold: flip in its own commit, INTARSIA_INK_BLEND_PIECE=256 the revert. Any fails: stays opt-in, final, no further
    round. Job script: body in the background, trap TERM and kill the whole descendant tree, check nothing is left at the end
    (an orphan of #11314 contaminated #11315 and #11322).
    Also noted: #11322's 1.6x-slow piece-256 / 1024 traces are most likely the orphaned batch_step of ink-metal's #11314
    (GPU busy outside fleetq until 06:48:20; the 256 trace ran 06:48:07-06:48:21, the 1024 trace 06:48:21-06:48:32), not a
    device state as first written. The p8k pairs all started after 06:48:39. The traces are rerun as #11344.
    **Status:** gate job .scratch/gp1.sh queued after this rule was written. Binary bin10 = de1e0568a, --prefill-chunk 2048.
    - Box check: below 0.95 x 2362 = BOX SLOW, stop. Above +5 % is recorded but does not stop the job: the rule targets a slow box.
    - md5 runs use p4k (3950 tokens -> chunks of 2048 + 1902, so the pieces matter).
    - Both gp1 and bp3 run under .scratch/safe_run.py: own session / process group, the group killed on TERM / INT / HUP,
      leftovers listed and killed at the end. Tested locally with sleep children, on both the leftover path and the TERM path.

21. 2026-11-22 (coordinator, via SendMessage): piece 1024 accepted as opt-in, final (default 256). Next: (b) attention,
    ~40 ms per chunk above its floor (~6 % of a 655 ms chunk). (1) Measure first, per attention call at chunk 2048 and
    4k / 8k context: bytes, FLOPs, achieved rate vs floor, and where the 40 ms goes (QK, softmax, PV, KV loads, launch count,
    occupancy, SWA vs full layers). (2) Propose a build only if a measured or tightly bounded ceiling is >= 2 % end to end at
    4k and 8k, with the bound accounting for scratch footprint. (3) Any build: bitwise or tolerance-justified admission,
    G2 / G3 short and long, md5 64 / 256, then the brief-20 gate shape (box check, generate timing 1k / 4k / 8k in random
    order, rule fixed before the runs). (4) General note in the doc (below). safe_run as before.
22. 2026-11-22 (coordinator, via SendMessage): go ahead with the timing-only probe of B (exact interior-tile path, the
    28 ms bucket). On A: do **not** amend rel_mode 1's bitwise contract. If A is still worth it after B, build it as a NEW
    opt-in mode (e.g. rel_mode 2) with its own tolerance contract against the f64 reference, default off; mode 0 / 1 and the
    md5 64 / 256 references stay untouched. Hold A until B's probe is in, then report B's measured gain plus A's bound on top
    of B (they may overlap) before building anything. Any default flip that changes md5 is the user's call: bring it to the
    coordinator with G2 / G3 numbers.
23. 2026-11-22 (coordinator, via SendMessage): **GO on A as opt-in rel_mode 2**, exactly as scoped: default off, own
    tolerance contract vs the f64 reference, mode 0 / 1 and md5 64 / 256 untouched, G2 / G3 + the brief-20 gate shape before
    anything comes back. Conditions: (1) if the build looks like more than ~a day, send the estimate before going past the
    first kernel; (2) speed gate: prefill tok/s at 4k / 8k / 16k, mode 0 vs mode 2, ABAB inside one endor slot after a
    warm-up, per-pair medians, warm number primary, plus end-to-end TTFT at 4k and 8k; (3) decode unaffected: greedy md5
    64 / 256 with mode 0, decode tok/s within noise; (4) new shaders update the Metal 4 shader-count test in the same commit
    (ink/metal 8e0ff6af6 failed on exactly this); (5) a default flip changes bits: user's call, via the coordinator with
    G2 / G3 and speed numbers. **B:** exact but no gain: remove inkling2_attn_nax_int.metal and INTARSIA_INK_NAX_INT from the
    branch, keep the measurements in the doc. The softmax / exp / rescale bucket (~0.53 ms per SWA visit over floor) is next
    after A; do not start it yet.
24. 2026-11-22 (coordinator, via SendMessage): the Metal 4 shader-count test fails on ink/prefill (d03224697 added
    inkprefill::metal4_shaders() without the test's sum); cherry-pick or merge ink/metal's one-line fix 8e0ff6af6 before
    adding the rel_mode 2 shaders, and note it here. **Done / checked:** ink/prefill already carries its own fix (87e5dc2c6,
    same line); cherry-picking 8e0ff6af6 on top double-counts (126 vs 188), so it was dropped, and
    `metal4_shaders_are_the_nax_variants_and_not_metal3_shaders` passes at the ink/prefill tip (8db176ed5). The rel_mode 2
    branch `ink/prefill-rel2` merged ink/metal 5ef24b165 (contains 8e0ff6af6; the merge kept one copy of the line), and the
    test passes there before and after NAX_R2 joined `inkattn::metal4_shader_names()` (which the sum counts).
25. 2026-11-22 (coordinator, via SendMessage): rel_mode 2: agreed, no flip; leave it on `ink/prefill-rel2`, do **not** merge it
    into ink/prefill or ink/metal (an opt-in mode worth ~1 % that changes output is not worth carrying). **Go on NAX_SG:**
    (1) sweep `INTARSIA_INK_NAX_SG` over 4 / 6 / 8 / 12 (16 if valid) per call at the 4k / 8k / 16k shapes, mode 0, one quiet
    endor slot; (2) exactness of the best value: bitwise vs 12 by construction or only to tolerance; show G2 self-goldens
    byte-identical and greedy md5 64 / 256 identical; (3) speed: end-to-end prefill tok/s and TTFT at 4k / 8k / 16k, best SG vs
    12, warm-up then 3 ABAB pairs; (4) rule: bitwise and median >= +1 % end to end -> flip default-on in its own commit with a
    revert env var, a docs/tuning.md row and a LOG row, send the hash; not bitwise -> report, no flip. Then the softmax /
    rescale bucket, bounded first (share of prefill ms at 8k, best-case saving); build only if the bound is >= 3 %.
26. 2026-11-22 (coordinator, via SendMessage): rerun the NSG 6 vs 12 speed part once, rule as written. (1) Box check first:
    p8k >= 95 % of 2880 tok/s; log the 1-minute load average and the top 5 CPU processes at start and end (endor load was
    26.7 at 11:39 with another job's rustc; the Claude app alone uses ~60 % CPU); if the check fails, wait 10 min inside the
    slot, check once more, then stop and report; no ABAB pairs on a degraded box. (2) Same ABAB rule: flip if the median ratio
    is >= 1.01 at 4k, 8k and 16k; fleetq `--quiet gpu,cpu`, honest `--expect`. (3) After the result, flip or not, remove the
    sweep forms s3 / s4 / s8 / s9 / s16 / s18 / s24; keep s6, s12 and the env var. (4) If it flips, land it with the rest of
    ink/prefill, and report whether ink/prefill has anything else ready for main and what its PR would contain.
27. 2026-11-22 (coordinator, via SendMessage): NSG 6: one more job with a cool-down. One fleetq job on endor
    (`--quiet gpu,cpu`, `--expect 40`): idle up to 20 min inside the slot so the GPU leaves its slow state; a short p8k reading
    every 5 min; as soon as p8k >= 2736, run rg3.sh's ABAB at once, same rule (median ratio >= 1.01 at 4k, 8k, 16k -> flip);
    if p8k never reaches 2736 within 20 min, stop and report the readings, no pairs. Before queueing, read the batch lane's
    slow-state clock test (#11375) for the recovery time and set the idle time from it if it says more than 20 min. If it
    flips, it lands on ink/prefill with the measurement tools; tell the coordinator when ready, it routes the PR.
    **#11375 / #11393 (ink-batch section 28):** the slowdown is a power / thermal governor. The GPU drops from P13 to P6-P9
    after ~1.5 min of sustained load; fast windows came back within ~2-3 min of lighter load (07:22 -> 07:25). Nothing there
    says recovery takes more than 20 min, so the idle cap stays at 20 min, with readings at 0 / 5 / 10 / 15 / 20 min. Main's
    rule 28b for that lane makes the warm steady state the primary serving number. Here the coordinator asked for the cold
    gate explicitly, so this run gates on p8k >= 2736.

## Status

- **Default-on (commit 355791b1a):** the loop blend of prefill chunks (rows > 16, not `exact_rows`) on the accelerators:
  h by the InkInProj NAX GEMM, gate + norm by `ink_blend_gf_nax` + `ink_blend_yn` (kernels/metal/inkling2_blend_nax.metal).
  `INTARSIA_INK_BLEND_NAX=0` restores m-mix's GEMV path. G2 / G3 with it on: see Gates.
- **Opt-in, not faster:** MoE prefill forms nax2 / nax3 (`INTARSIA_INK_MOE_NAX2=<form>`); all bitwise the first generation
  at 2048 rows, none faster in the bench or in the stream (tables below), including the 128-row tiles.
- **Side branch `ink/prefill-rel` (4212f2e45), not merged:** `ink.rel_mode` (approved contract variant): cpu reference +
  bitwise test pass; Metal mode 1 measured slower end to end, so the graph default is off. Tests and admission pass; G3 with mode 1 not run (off by default).

## Floors

- NAX bf16 GEMM peak on endor: 62 TFLOP/s (4096 x 4096, M 2048; docs/design/2026-11-15-mac-roadmap.md). Every NAX prefill term
  below is bf16 x bf16 on the accelerators (the NVFP4 / MXFP8 experts decode to bf16 in threadgroup memory first).
- FLOPs per 2048-row visit: experts 261 GFLOP (16384 pairs x 15.9 MFLOP), InkInProj 81.8, wo 29.0, router 1.2, blend h 3.6 per
  stream; at 62 TFLOP/s the experts' floor is 4.2 ms per visit, InkInProj 1.3, wo 0.47.

## Per-chunk kernel time model (2048 rows, 48 visits, ctx 0), before

Measured per call by the lanes' benches on endor (`--quiet gpu,cpu`), this lane's runs: `.scratch/base.log` (fleetq #10700).

| Term | per call (ms) | calls / chunk | ms / chunk | Notes |
|---|---:|---:|---:|---|
| MoE experts NVFP4 | 7.28 | 44 | 320 | 35.8 TFLOP/s; random routing, 5 weight copies cycled |
| MoE experts MXFP8 | 7.62 | 4 | 30 | 34.3 TFLOP/s |
| Blend (looped visits), m-mix d99d2c0fc | S=3 4.24, S=1 1.79, LoopBlend 3.93 | 24 x 2 | ~130 | extra over the plain ConvMix; h runs as 16-row GEMV passes |
| InkInProj NAX | 1.50 | 48 | 72 | 54 TFLOP/s |
| Prologue + attention NAX (SWA / full) | 0.98 + 0.87 / 1.00 + 1.02 | 40 / 8 | 90 | the prologue writes the 100 MB rel tensor |
| wo NAX | 0.54 | 48 | 26 | 53.6 TFLOP/s |
| ConvMix plain (Q=1 / Q=3) | 0.27 / 0.39 | 96 | ~29 | |
| AttnResMix (loop messages) | 0.22 | 24 | 5 | |
| Router + MoeRoute | pending | 48 | | |
| **Sum** | | | **~700** | ink/metal e2e: ~920 ms per chunk (2221 tok/s at 5k, chunk 2048) |

On 7.3 vs 11.9 ms per MoE visit: m-moe's 2048-row number was taken with the same bench (cycled weight copies, union = all 128
experts) on a shared GPU; this lane's runs at `--quiet gpu,cpu` gave 7.3 ms (NVFP4) / 7.6 ms (MXFP8) on the same code, so the
11.9 ms looks like GPU contention rather than a setup difference (not investigated further).

## Measurements (endor, `--quiet gpu,cpu`)

### End-to-end prefill, chunk 2048 (`intarsia generate --max-tokens 1`, two runs each; fleetq #10779)

| Config | 2k (1921 tok) | 8k (8034) | 12k (12190) |
|---|---:|---:|---:|
| ink/metal baseline (gen1 everything) | 1588 / 1558 | 1753 / 1606 | 1777 / 1776 |
| blend NAX (level 2) | | 1989 / 1902 | |
| MoE nax3_m64_n128_k32 | | 1752 / 1727 | |
| MoE m64_k32_v3 | | 1900 / 2011 (noise: the microbench says slower) | |
| rel_mode 0 (wt-rel binary) | | 1910 / 1703 | 1798 / 1831 |
| rel_mode 1 | | 1733 / 1787 | 1646 / 1733 |

Run-to-run spread at 8k is up to 10%; the interleaved 3-round A/B is #10884. oMLX: ~1300 tok/s marginal prefill.

### Where a 2048-row chunk goes (baseline, 8k prompt, kernel trace / cb trace)

cb trace: 4 chunks in 4014 ms, GPU busy 94%, idle 232 ms (~58 ms per chunk host gap). Kernel trace, GPU ms per chunk:

| Op | ms / chunk |
|---|---:|
| MoE experts NVFP4 (gu 231, dn 123, gather 11) | 366 |
| ConvMix (incl. the looped-visit blend) | 174 |
| RelSinkAttn NAX | 78 |
| InkInProj NAX | 72 |
| InkKvPrologue | 55 |
| MoE experts MXFP8 | 35 |
| wo (matmul_bf16_nax) | 26 |
| router (logits_bf16, 0.31 ms at 3.9 TFLOP/s) | 16 |
| LoopBlend | 9 |
| AttnResMix, MoeRoute, rest | 5 |
| **GPU sum** | **837** (+ ~58 host gap = the ~920 ms ink/metal saw at 5k) |

The 700 ms model missed the blend's real size (ConvMix 174 vs 29 + ~100 estimated), the e2e routing being less balanced
than random (gu 5.25 ms vs 4.84 per visit), attention at 8k context, and the router.

### MoE experts NVFP4, 2048 rows, µs per call (gu only = skip 2, dn only = skip 1)

| Form | full | gu only | dn only | Note |
|---|---:|---:|---:|---|
| gen1 (ink/m-moe) | 7287 | 4845 | 2698 | 35.8 TFLOP/s |
| nax2 m64_k64_v1 (A prefetch) | 7381 | 5408 | 2798 | |
| nax2 m64_k32_v3 (double-buffered) | 8502 | 6884 | 4496 | |
| nax2 m32_k64_v3 | 14257 | | | |
| nax2 v8 (no matmul, probe) | 6348 | | | MMA hides under staging |
| nax2 v28 (loads + epilogue only, probe) | 2510 | | | |
| nax3 m64_n64_k64 (A from device) | 12595 | | | device-tensor A is slow |
| nax3 m64_n128_k32 | 14190 | | | |
MXFP8: gen1 7690, m64_k32_v3 7424 (-3.5%), m64_k64_v1 7598, nax3 9450 / 13327.

Reading: dropping the matmul saves only 0.9 ms, so the kernel is bound by weight decode + threadgroup staging (~3.8 ms)
and loads / epilogue (~2.5 ms), not by the accelerators. With 64-row tiles every expert (mean 128 rows) decodes its weights
twice. The 128-row forms (m128_*, decode once per expert) are much slower (NVFP4 m128_k64_v1 17542, m128_k32_v3 12991;
MXFP8 18005 / 13995 µs; in-stream 1600 / 2000 tok/s at 8k vs 2643): 8 A chunks per thread per step and a 64 x 32 sub-tile per
simdgroup cost more than the halved decode saves. The gen1 kernel stays the default.

### Blend on 2048 rows (ink_mix_bench, µs)

| Call | GEMV (m-mix) | level 1 (NAX h) | level 2 (NAX h + gate) |
|---|---:|---:|---:|
| ConvMix blend S=3 | 4277 | 3165 | 1625 |
| ConvMix blend S=1 | 1783 | 1688 | 1066 |
| LoopBlend S=3 | 4370 | 3030 | 1338 |

### Admission and tests

- `intarsia kernels test --backend metal`: moe_experts / moe_experts_mxfp8 ADMITTED with every form forced (trace confirms
  the form ran; worst case unchanged, 3.472e-3 / 2.099e-4 and 3.378e-3 / 2.453e-4), loop_blend / conv_mix ADMITTED at blend
  levels 0 / 1 / 2 (3.226e-3 / 2.640e-4; trace confirms the NAX blend ran on the 40-row cases).
- `tests/ink_prefill_moe.rs` (2048 rows, real shape, output poisoned with NaN before each run; #10889): every nax2 / nax3 /
  m128 form, NVFP4 and MXFP8, bitwise gen1 (max_abs 0).
- `ink.rel_mode` (wt-rel, #10843): cpu `ink_rel_mode` (mode 1 == mode 0 bitwise over the InkAttnDecode sweep), Metal
  `ink_rel_mode` (prologue + NAX attention, mode 1 == mode 0 bitwise at T 17 / 512 / 2048, SWA and full past 8192),
  `ink_attn_decode`, `inkling2_attn` pass; admission with the appended mode-1 cases: ink_kv_prologue 123 cases ADMITTED,
  rel_sink_attn_bf16_nax 15 ADMITTED, rel_sink_attn_bf16 63 and ink_attn_decode 70 ADMITTED (unchanged).

### Gates with the NAX blend default-on (#10883, bin4 at 355791b1a; trace confirms `blend h nax` + `blend gate nax` ran)

| Gate | 21-way KL scored (bound) | ink/metal baseline | confident top-1 misses |
|---|---:|---:|---:|
| G2 short (oracle) | 2.087e-3 (4.0e-3) | 2.380e-3 | 0 / 363 |
| G2 long (oracle) | 3.239e-3 (6.0e-3) | 3.418e-3 | 1 / 1206 |
| G3 short (vLLM, prefill) | 8.420e-3 (1.05e-2) | 8.391e-3 | 0 / 367 |
| G3 long (vLLM, prefill) | 1.229e-2 (1.65e-2) | 1.248e-2 | 0 / 1211 |

All PASS. Tests: `inkmix` (3, incl. the rows-exact blend at 40 / 300 rows: exact_rows calls keep the GEMV path), `inkmoe` (2),
`ink_prefill_moe` pass.

### Interleaved end-to-end A/B, chunk 2048 (#10884, bin4 = ink/metal a0751f302 + this lane; 2 runs per round)

| Config | 2k | 8k (rounds 1 / 2 / 3) | 12k |
|---|---:|---:|---:|
| `INTARSIA_INK_BLEND_NAX=0` (ink/metal path) | 1796 / 1823 | 1810*, 2457 / 2468, 2208 / 2464, 2356 | 2373 / 2496 |
| **default (NAX blend)** | **1935 / 1952** | **2630, 2595 / 2718, 2609 / 2683, 2626** | **2652 / 2812** |
| MoE m128_k64_v1 | | 1616, 1570 / 1621, 1589 / 1617, 1595 | |
| MoE m128_k32_v3 | | 2022, 1974 / 2018, 2019 / 2025, 1975 | |

(* first run of the job, cold.) Median gain of the NAX blend: 2k +7%, 8k +9% (2440 -> 2640), 12k +12% (2435 -> 2730).
These absolute numbers are ~40% above the #10779 baseline (same commit but ink/metal a88db4620, and a busier box); compare
only within this table. oMLX ~1300, EXO 1750, target 2300-3100.

### Per-chunk split with the default (8k prompt, kernel trace, ms per 2048-row chunk)

cb trace: 4 chunks in 2948 ms (737 ms / chunk), GPU busy 92%, idle 57 ms / chunk.

| Op | ms / chunk |
|---|---:|
| MoE experts NVFP4 | 319 |
| ConvMix (incl. the NAX blend) | 75 |
| RelSinkAttn NAX | 67 |
| InkInProj NAX | 62 |
| InkKvPrologue | 48 |
| MoE experts MXFP8 | 30 |
| wo | 23 |
| router (logits_bf16) | 14 |
| AttnResMix + LoopBlend + MoeRoute + rest | 8 |
| **GPU sum** | **646** |

### localeval speed-plain, base ink/metal e3f92beff vs this branch 378dc214e (#10967, serve --max-batch 1, ABAB)

Brief 5 item 1. `localeval compare` per pair (its rule: under 10 % = noise; "rejected A" = the base run failed the spread check):

| prompt tok | prefill tok/s pair 1 (base -> branch) | pair 2 | decode tok/s pair 1 | pair 2 |
|---:|---|---|---|---|
| ~1k | 1004 -> 1119 (+11.5 %, rejected A) | 1034 -> 1158 (+12.0 %, faster) | 59.1 -> 60.5 | 61.1 -> 61.1 |
| ~4k | 1116 -> 1357 (+21.5 %, faster) | 1226 -> 1258 (+2.6 %, rejected A) | 57.7 -> 60.0 | 59.8 -> 59.5 |
| ~16k | 1193 -> 1304 (+9.4 %, noise) | 1261 -> 1348 (+7.0 %, noise) | 54.6 -> 56.0 | 56.8 -> 56.3 |

Verdict against brief 5 (prefill better at 4k and 16k, decode within 1 %): 4k better in the one valid pair (+21.5 %); 16k
+7 to +9 % in both pairs, under localeval's 10 % noise line; decode within 1 % or better everywhere a pair is valid (worst
-0.9 % at 16k). So: no regression, prefill up, but 16k is not a clean "faster" by localeval's rule.

**Why serve is ~1300 tok/s while `generate --prefill-chunk 2048` is ~2700:** `recipes/inkling2.toml` sets no
`prefill_chunk`, so serve and `intarsia accuracy` run **256-row chunks** (serve-2.log: "prefill chunk 256 (default)"). Every
NAX path in this lane (blend, router, MoE tiles) was tuned and timed at 2048 rows, and G2/G3 above ran at 256. **Every localeval / accuracy number for Inkling on Metal before the recipe commit ran at 256-row chunks (brief 8.4).** Proposal:
`[tuning.metal] prefill_chunk = 2048` in the recipe (as qwen3 / qwen3_5_moe). Check queued as #11109: G2/G3 short+long at
2048, then speed-plain ABAB serve 256 vs 2048 on the same binary.

### The "host gap" is a cold first chunk, not a per-chunk cost (cb trace .scratch/e2e/kw1.cb.jsonl, #10972, 8k prompt)

Idle per 2048-row chunk (span minus GPU busy, chunks split at `embedding_bf16`):

| chunk | command buffers | span ms | busy ms | idle ms |
|---:|---:|---:|---:|---:|
| 0 | 513 | 850.0 | 638.9 | 211.1 |
| 1 | 18 | 657.6 | 657.6 | 0.0 |
| 2 | 18 | 756.7 | 756.7 | 0.0 |
| 3 (1890 rows) | 18 | 733.8 | 733.7 | 0.0 |

The earlier "57 ms per chunk" was this lane's 229 ms prefill idle divided by 4 chunks; it all sits in chunk 0. Of chunk 0's 211 ms,
44 ms is the host committing late, and 167 ms is the GPU starting a committed buffer late. That late start is ~3.7 ms on the
**first** visit of each layer's routed experts (visits 1-18 and 43-48; the second and third core passes and every later chunk
start in 0.1 ms) plus one 17 ms on the first attention. This matches first-touch page-in of each layer's weights, once per
process. Chunks 1-3 run as 128-dispatch buffers with no measurable gap. A warm server (keep-warm on, localeval's per-round
warmup) does not pay it. Batching prefill command buffers therefore has nothing to remove in steady state. Not built;
brief 8's first next item is closed by this measurement. (A cold `generate` pays ~0.2 s once; prefetching the expert weights at load would
remove that and belongs to the residency / keep-warm owner.)

### Router NAX, v65 and dq1 end to end (#10981 checks, #10982 timing; bin6 = 0d226768c)

- Router NAX `logits_bf16_nax`: f64 check passes (2048 rows max_abs 5.0e-6, rel_l2 3.5e-7), admission passes, and G2/G3 with
  it on pass (G2 short 1.807e-3, long 3.248e-3; G3 short 6.410e-3, long 1.163e-2). End to end p8k, 3 interleaved rounds of 2:
  off 2243 / 2639 / 2678 / 2675 / 2667 / 2673, on 2722 / 2697 / 2393 / 2690 / 2712 / 2434 tok/s. Medians 2671 -> 2694 (+0.9 %),
  inside run-to-run noise. **But it adds a candidate to the shared `Op::Logits`:** the lib test
  `kernel_list_is_unchanged_by_core_op_fusion_p1` fails (#10981), and other models' Logits calls with M > 16 would see it.
  That breaks the rule that other models' kernels stay unchanged. **Reverted on ink/prefill (29d90040e)**; the
  shader-count test fix it carried is kept (87e5dc2c6). It was never in the merge hash 378dc214e. Redoing it would need an
  Inkling-only route (a hook in the Inkling router call, not a registry entry).
- v65 (`INTARSIA_INK_MOE_NAX2=m128_k64_v65`, the shared-unpack form): bitwise, admitted, but slower. Bench 11.0 vs 7.3 ms per
  NVFP4 call; e2e p8k 2406 / 2423 / 2350 / 2421 / 2403 / 2429 tok/s vs ~2670 default (-10 %). Dead end.
- dq1 (dequant once to bf16, then GEMM): bitwise, but 14.8 ms per call (decode 9.3 ms of it). Dead end.

### Steady-state ranking vs floors (brief 9; kernel trace .scratch/e2e/profdefault.k.jsonl, chunks 1 + 2 of an 8k prompt, warm)

ms per 2048-row chunk, the mean of chunks 1 and 2 (GPU 651.7 / 660.6 ms; ctx 2k-6k). Floors: NAX bf16 62 TFLOP/s (see Floors);
memory-bound ops at ~550 GB/s DRAM (assumed, not measured on endor; ops whose operands stay in the SLC can beat it).

| Op | ms / chunk | floor ms / chunk | above floor | floor basis | covered by |
|---|---:|---:|---:|---|---|
| MoE experts NVFP4 (44 visits) | 323.2 | 185 | **138** | 261 GFLOP / visit | v65 / dq1 tried: both slower |
| ConvMix incl. the looped blend | 76.5 | n/m | ~15 (fusion bound) | plain 18.6 ms (96 calls); blend 8 x 4 dispatches per looped call, 1.12 ms per call, already at cache speed per dispatch (blend_pair moves 57 MB in 0.043 ms) | - |
| RelSinkAttn NAX | 70.3 | 30 | **40** | SWA 25.8 GFLOP x 40, full ~103 GFLOP x 8 (mean 4k keys); 62 TFLOP/s is optimistic for softmax attention | - |
| InkInProj NAX | 61.6 | 63 | 0 | 81.8 GFLOP / visit | - |
| InkKvPrologue | 47.9 (24.1 split) | 12 | **36** (12 split) | ~140 MB / visit (rel 100.7 MB written, q 12.6, k / v 8.4, pq 14, pk / pv 8.4) | - |
| MoE experts MXFP8 (4 visits) | 30.8 | 17 | 14 | as NVFP4 | v65 / dq1 tried: both slower |
| wo (matmul_bf16_nax) | 22.7 | 22.5 | 0 | 29 GFLOP / visit | - |
| router (logits_bf16) | 15.2 | ~1 | 14 | 1.2 GFLOP / visit | none (router NAX reverted: shared Op::Logits) |
| attn_res_mix + loop_blend + moe_route + rest | 7.7 | - | - | under 5 ms each | - |

**Pick: InkKvPrologue (36 ms above floor; attention's 40 assumes an unreachable 62 TFLOP/s for softmax attention, and its
in-kernel-bias variant already lost).** The prologue runs at 1.0 ms per visit, about 140 GB/s for the ~140 MB it must move.
Per (row, kv head) it runs a 256-thread threadgroup in which only 128 threads do the conv / q / r part. The rel product
(rel[t, h, e] = sum_j r[t, h, j] relproj[j, e], 16 x 1024 per head, 75 % of the bytes) re-reads relproj's 32 KB for every row.
Plan: (1) `ink_kv_prologue_norel_bf16` = the prologue without the rel loop, at 128 threads; (2) `ink_rel_proj_bf16`, one
threadgroup per (kv head, 8-row block). It recomputes r with the prologue's own arithmetic (same lane map, simd_sum, clip, tau
and rA rounding), loads relproj once per block into registers, and writes rel with the same per-element fma order (j ascending).
New file kernels/metal/inkling2_prologue_split.metal, flag `INTARSIA_INK_PROLOGUE_SPLIT=1`, default off, branch
`ink/prologue-split` off the ink/prefill tip. **Bitwise** vs the current prologue: same inputs, the same fp32 chains, only the
threadgroup shape changes. Expected: 1.0 -> ~0.35 ms per visit, ~30 ms per chunk (~4.5 % of 656 ms). If it is not bitwise in
the Metal test, I stop and report rather than argue tolerance.

### NSG 6 vs 12 speed rerun (#11508, rg3.sh, brief 26): box degraded twice, stopped before the pairs; no flip

Box check (p8k at NSG 12, need >= 2736 tok/s = 95 % of 2880), after a 4k warm-up:

| check | time | p8k tok/s | load1 | top CPU processes |
|---|---|---:|---:|---|
| start | 11:51:13 | | 2.19 | **rustc 95 %** (pid 47475, not this job), WebKit 10 % / 10 %, claude 5 % |
| box 1 | 11:51:24 | 2612.7 (fail) | 5.10 | python 173 % (pid 2524), claude 38 % |
| retry after 10 min | 12:01:24 | 2152.5 (fail) | 1.64 | Claude Helper 24 %, claude 14 %, Tailscale 8 % |
| end | 12:01:33 | | 1.54 | idle |

Stopped as the brief says: no ABAB pairs. The second failure came at load 1.6 with nothing heavy on the CPU, so the slowdown
looks like the GPU's state, not CPU contention. The warm-up 4k ran at 2301 tok/s against ~2690 earlier today, the same
symptom as #11499. A rustc (pid 47475) was running at 95 % when this quiet slot started; it was gone by box 1. **NSG 6 stays
opt-in** (`INTARSIA_INK_NAX_SG=6`, bitwise identical to the default 12). The sweep forms are removed (4c18c17e5: s6, s12 and
the env var stay).

### NSG 6 vs 12 gate (#11499, rg2.sh, bin12 640d1c59f): bitwise shown; speed rule FAILS at 8k on a degraded box; no flip

**Exactness: bitwise, as constructed.** G2 self-goldens (`accuracy --write-self-golden`) short and long at NSG 12 and 6: every
tensor byte-identical (short 48 / 48, long 60 / 60); the files differ only in metadata (`built_unix`, the exec command line
in `self_meta`), which is why a whole-file `cmp` differs. Greedy md5 at NSG 6 = NSG 12 = the references (64: a2ab7b3d...,
256: cbd62660...).

**Speed (warm-up, 3 ABAB pairs, prefill tok/s NSG 12 -> 6, TTFT ms):**

| prompt | pair ratios | median | need | TTFT NSG 12 | TTFT NSG 6 |
|---|---|---:|---:|---|---|
| 4k | 1.1937 / 0.9330 / 1.1605 | 1.1605 | 1.01 | 1930 / 1611 / 1863 | 1617 / 1727 / 1605 |
| 8k | 1.0021 / 1.0095 / 0.8629 | 1.0021 | 1.01 | 2930 / 2939 / 2942 | 2923 / 2912 / 3410 |
| 16k | 0.9919 / 1.0192 / 1.0294 | 1.0192 | 1.01 | 6037 / 6152 / 6173 | 6086 / 6036 / 5996 |

The rule as written fails at 8k (1.0021 < 1.01): **no flip.** The run is not a clean measurement, though. The box was in a
degraded state for both arms: 4k prefill 2047-2461 tok/s against 2688-2716 in #11461 80 minutes earlier, decode 40.8-50.5
tok/s against 68.5-70.7, and swings of +-15 % inside pairs (4k pair 2, 8k pair 3). fleetq shows no job overlapping
11:34:31-11:37:44. The previous job #11489 (a build plus tests) ended at 11:34:31, so leftover processes are possible but not
verified, and rg2.sh had no box check. Asked the coordinator whether to rerun once with a box check first (gp1.sh style:
stop if p8k < 95 % of reference).

### Softmax / exp / rescale bucket bound (brief 25; from #11370 E 1, #11468, the blend-gate trace)

Above the 62 TFLOP/s matmul floor with the rel bias removed (E 1): SWA ~0.53 ms per visit x 40 = ~21 ms per chunk, full
visits ~1.2 ms (mean over an 8k prompt's chunks at ctx 0 / 2k / 4k / 6k) x 8 = ~10 ms, so ~31 ms per chunk. At 8k a chunk is
~700 ms end to end (2.79 s / 4), so the bucket is **~4.4 % of 8k prefill**. That is the best case and needs attention at
the full GEMM peak, which flash attention does not reach (the softmax between the two matmuls is not hidden). At a
realistic ~70 % of peak the saving is ~19 ms per chunk, **~2.7 %**. If NSG 6 is flipped it takes ~11 ms of the same
overhead, leaving a best case of ~2.9 % and a realistic ~1.5 %. **Under the >= 3 % build bar on any realistic reading: not
built.**

### NAX attention NSG sweep, mode 0 (brief 25; #11468, 640d1c59f, bench 2048 rows, µs per call, median of 2)

`INTARSIA_INK_NAX_SG`: NSG simdgroups per threadgroup, BQ = 16 x (NSG / 3) query rows; NSG not a multiple of G = 3 leaves
simdgroups idle (4, 8, 16). All nine forms give the same output hash (fixed seed; SWA and full at ctx 0 / 2k / 4k): bitwise.

| visit | 3 | 4 | 6 | 8 | 9 | 12 (default) | 16 | 18 | 24 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SWA ctx 2k | 1204 | 1475 | 1171 | 1402 | 1193 | 1316 | 1377 | 1415 | 1386 |
| SWA ctx 4k | 1218 | 1605 | 1264 | 1469 | 1316 | 1288 | 1446 | 1391 | 1398 |
| SWA ctx 8k | 1185 | 1482 | 1213 | 1449 | 1210 | 1533 | 1475 | 1307 | 1336 |
| SWA ctx 14k | 1217 | 1622 | 1196 | 1452 | 1312 | 1383 | 1405 | 1323 | 1405 |
| full ctx 2k | 2372 | 3220 | 2370 | 3190 | 2599 | 2621 | 2768 | 2789 | 3625 |
| full ctx 4k | 3656 | 4826 | 3476 | 4831 | 4013 | 4729 | 4176 | 4005 | 6198 |
| full ctx 8k | 6804 | 8931 | 6464 | 8348 | 7495 | 6703 | 7266 | 7276 | 12247 |
| full ctx 14k | 10667 | 15145 | 10884 | 13584 | 12430 | 11241 | 13336 | 11962 | 21285 |
| per chunk (40 SWA + 8 full, mean of ctx 2k / 4k / 8k), ms | 82.3 | 106.1 | **81.5** | 101.3 | 87.2 | 92.6 | 95.2 | 92.4 | 113.8 |

Best: NSG 6 (3 is within noise of it); ~11 ms per chunk under 12 (~1.7 % of a 655 ms chunk). The box was noisy (NSG 12 SWA
1.29-1.53 ms here vs 1.14-1.23 earlier), so the end-to-end pairs decide.

**Exactness by construction:** each row's online softmax visits the key tiles in the same absolute order (tiles at multiples
of 32, `full` chosen by kv_len, not by the threadgroup); a smaller BQ only drops key tiles that are fully masked for every row
the threadgroup has, and a fully masked tile is an exact no-op (m unchanged, rescale factor exp2(0) = 1, P = 0, so O x 1 + 0 x V
and l x 1 + 0 are exact). Nothing else in a row's arithmetic depends on NSG. To be shown end to end by G2 self-goldens and md5.

**Rule (brief 25, fixed before the runs; rg2.sh, bin12 = 640d1c59f):** G2 self-goldens short and long at NSG 6 byte-identical
to NSG 12, greedy md5 64 / 256 at NSG 6 identical to NSG 12 (and to a2ab7b3d... / cbd62660...); then warm-up and 3 ABAB pairs
(12, 6) per prompt at 4k / 8k / 16k, `generate --max-tokens 1`. **Flip NSG 6 default-on** if bitwise and the median pair
ratio is >= 1.01 at every length. Else report, no flip.

### rel_mode 2 gate (#11461, bin11 8bef1619a, rule above): quality and decode pass, speed FAILS the bar; stays opt-in

1. Quality (all PASS; mode 0 equals the references exactly):

| gate | limit | mode 0 | mode 2 |
|---|---:|---:|---:|
| G2 short | 4e-3 | 2.087e-3 | 2.101e-3 |
| G2 long | 6e-3 | 3.239e-3 | 3.382e-3 |
| G3 short | 1.05e-2 | 8.420e-3 | 8.503e-3 |
| G3 long | 1.65e-2 | 1.229e-2 | 1.291e-2 |

2. Decode: mode 0 md5 64 / 256 = a2ab7b3d... / cbd62660... (references, unchanged). Mode 2 changes the text (64: a7227430...,
   256: 5b200ba7...), as its contract allows. Decode tok/s, mode 0 vs 2: 64 tokens 70.67 vs 70.61 (-0.1 %); 256 tokens 68.49
   vs 70.63 (+3.1 %, mode 2 faster, just outside the +-3 % band in the favourable direction; decode rows never take mode 2,
   and the two 256-token runs decode different text, so this is noise, not an effect).
3. Speed (warm-up, then ABAB, 3 pairs, prefill tok/s mode 0 -> mode 2; prefill ms = TTFT):

| prompt | pair ratios | median | need | TTFT ms mode 0 | TTFT ms mode 2 |
|---|---|---:|---:|---|---|
| 4k (3950) | 1.0017 / 1.0079 / 1.0087 | 1.0079 | 1.02 | 1470 / 1467 / 1468 | 1467 / 1456 / 1454 |
| 8k (8034) | 1.0086 / 1.0091 / 1.0017 | 1.0086 | 1.02 | 2789 / 2790 / 2788 | 2766 / 2765 / 2783 |
| 16k (16068) | 1.0099 / 1.0337 / 1.0121 | 1.0121 | 1.00 | 5540 / 5740 / 5638 | 5486 / 5553 / 5570 |

**Verdict (brief 25: rel_mode 2 stays on `ink/prefill-rel2`, not merged into ink/prefill or ink/metal):** mode 2 is ~+0.8 % at 4k and 8k and +1.2 % at 16k, consistently the faster arm in all 9 pairs, but below the
2 % bar at 4k and 8k: no flip proposed; `INTARSIA_INK_REL_MODE=2` stays opt-in on `ink/prefill-rel2`. Why the bound
(~3 %) was not reached: the mode-2 attention kernel only breaks even with the device gather on SWA visits, so the gain is the
prologue's skipped rel write (~0.2 ms x 40 visits = ~8 ms per chunk), and the full layers stay in mode 0.

### rel_mode 0 / 1 / 2 x NSG 12 / 6, per call (#11447, bench 2048 rows, µs prologue / NAX attention, median of 2)

| visit | m0 s12 | m1 s12 | m2 s12 | m2 s6 | m0 s6 |
|---|---:|---:|---:|---:|---:|
| SWA ctx 2k | 980 / 1167 | 782 / 3055 | 783 / 1255 | 782 / 1128 | 980 / 1148 |
| SWA ctx 4k | 980 / 1242 | 783 / 3062 | 783 / 1257 | 783 / 1132 | 980 / 1192 |
| SWA ctx 16k | 979 / 1234 | 782 / 3082 | 783 / 1254 | 784 / 1127 | 981 / 1134 |
| full ctx 2k | 997 / 2563 | 798 / 4999 | 797 / 2726 | 797 / 2415 | 998 / 2436 |
| full ctx 4k | 996 / 3659 | 798 / 6639 | 798 / 4245 | 797 / 3812 | 997 / 3440 |
| full ctx 16k | 1070 / 12200 | 797 / 15497 | 797 / 14030 | 797 / 12906 | 1133 / 11784 |

- Mode 1's 16-FMA chain is 2.5x slower on SWA: not a fallback for full visits.
- Mode 2 still loses on full visits at 4k / 16k at either NSG, so the switch applies mode 2 to sliding-window layers only
  (8bef1619a; `INTARSIA_INK_REL2_LAYERS=all` for every layer).
- Side finding, exact and not acted on: NSG 6 (`INTARSIA_INK_NAX_SG=6`) is 2-8 % faster than the default 12 for mode 0 too,
  and mode 2 gains more from it (SWA 1.13 vs mode 0 s12 1.17-1.24 ms). A candidate for after the gate, measured end to end.

### rel_mode 2 gate rule (fixed before the runs; brief 23; binary bin11 = ink/prefill-rel2 8bef1619a, chunk 2048)

Arms: mode 0 (unset) vs `INTARSIA_INK_REL_MODE=2` (sliding-window layers only, the default of that switch; #11447 showed
the mode-2 kernel 10-24 % slower on full visits). One endor slot, `--quiet gpu,cpu`, safe_run.
1. Quality: G2 / G3 short and long with mode 2 under the usual limits (G2 4e-3 / 6e-3, G3 1.05e-2 / 1.65e-2), and the
   mode-0 KL lines unchanged from the references above (2.087e-3 / 3.239e-3 / 8.420e-3 / 1.229e-2).
2. Decode: greedy md5 at 64 / 256 tokens with mode 0 must equal a2ab7b3d0e3d15d0fae1ccf6816ac7dc / cbd62660cd829e0009878cd8c1a3e362;
   decode tok/s of the 256-token runs, mode 2 vs mode 0, within +-3 %.
3. Speed: one warm-up generate, then 4k / 8k / 16k prompts (p4k, p8k, p16k = p8k twice), pairs in ABAB order (0, 2, 0, 2),
   3 pairs per length, `generate --max-tokens 1`; per-pair ratio mode 2 / mode 0, median of the 3. Prefill ms of these runs
   is the end-to-end TTFT. **Worth proposing a flip** if the median ratio is >= 1.02 at 4k and 8k and >= 1.00 at 16k, with 1
   and 2 passing. Otherwise mode 2 stays opt-in.

### rel_mode 2, first kernel (brief 23; branch `ink/prefill-rel2` 36a90c66d = ink/prefill + ink/metal 5ef24b165 + rel_mode 1)

`kernels/metal/inkling2_attn_nax_r2.metal`, `INTARSIA_INK_REL_MODE=2` (default 0). Per key tile, each simdgroup forms its 16
rows' bias over 64 distances as two 16 x 32 x 16 matmuls, r [16 x 16] x relproj, rounds to bf16 into its own 2 KB
threadgroup slice (24 KB at s12) and gathers from it. Contract: mode 1's operands, held to the NAX kernel's BF16_TOL vs the f64
reference (ops.rs doc); mode 0 / 1 unchanged.

Checks (#11436): admission `--op rel_sink_attn` ADMITTED (NAX 22 cases incl. 7 new rel_mode 2: max_abs 2.2e-3 .. 5.5e-3,
rel_l2 1.3e-3 .. 1.6e-3, same range as mode 0's 5.46e-3 / 1.50e-3), `--op ink_kv_prologue` ADMITTED 123 / 123, Metal
`rel_mode_1_pair_is_the_mode_0_pair_bitwise` ok, metal lib tests 18 / 18 (shader count), core lib 257 ok, cpu ink_rel_mode ok.

Per call, µs, median of 4 (bench, 2048 rows), mode 0 vs mode 2, prologue / NAX attention:

| visit | mode 0 | mode 2 |
|---|---:|---:|
| SWA ctx 0 | 980 / 966 | 782 / 932 |
| SWA ctx 2k | 980 / 1253 | 782 / 1253 |
| SWA ctx 4k | 981 / 1275 | 784 / 1258 |
| SWA ctx 16k | 981 / 1313 | 783 / 1257 |
| full ctx 0 | 997 / 1111 | 796 / 1117 |
| full ctx 2k | 1004 / 2466 | 797 / 2722 |
| full ctx 4k | 1030 / 3763 | 799 / 4241 |
| full ctx 16k | 1074 / 12484 | 798 / 15454 |

The prologue saves ~0.2 ms per visit (~9.5 ms per chunk). SWA attention is flat: forming the bias costs about what the
device gather cost. Full visits get 10-24 % slower although most of their tiles need no bias; suspected occupancy from the
24 KB of threadgroup memory. Net at ctx 4k ~ -6 ms per chunk, at 16k a loss. Next probe (#a25): mode 1 / mode 2 at NSG 6
(12 KB) and the mode 1 kernel's cost on full visits (mode 2 could take the mode 1 kernel for full visits: it is bitwise mode 0,
so inside mode 2's tolerance).

### Attention probe B, interior-tile fast path (brief 22; `kernels/metal/inkling2_attn_nax_int.metal`, `INTARSIA_INK_NAX_INT`)

Round 1 (#11391, 7a5230f18): interior = all 16 rows of the simdgroup see every key of the tile and every distance < E.
INT 1 is the exact path, INT 2 also drops the bias on interior tiles (timing only). Bitwise: the bench output hash (fixed
seed, 4 visits, SWA and full at ctx 0 / 2k / 4k) is identical for INT 0 and INT 1 in all 6 configs; INT 2 differs, as
expected. Admission was not run: the CLI binary lacked `--features metal` (fixed in round 2). The box was in its slower
clock state (INT 0 SWA 1.22-1.28 ms vs 1.14 in #11370), but the arms are interleaved (0 1 2 0 1 2, two reps).
NAX µs per call, median of 4:

| visit | INT 0 | INT 1 (exact) | INT 2 (no bias, interior) |
|---|---:|---:|---:|
| SWA ctx 0 | 907 | 860 | 598 |
| SWA ctx 2k | 1233 | 1155 | 793 |
| SWA ctx 4k | 1279 | 1149 | 799 |
| SWA ctx 16k | 1218 | 1155 | 868 |
| full ctx 0 | 1079 | 1069 | 739 |
| full ctx 2k | 2562 | 2574 | 2104 |
| full ctx 4k | 3854 | 4225 | 3577 |
| full ctx 16k | 12244 | 13580 | 12171 |

- Exact path: SWA -5 to -10 % (~0.08-0.13 ms x 40 = 3-5 ms per chunk), but full visits at ctx 4k / 16k get +10 %: their far
  tiles (distance >= E) fail the interior test and pay the extra branch. Net per chunk at 4k: ~0. Round 2 adds the far class.
- The in-kernel bias gather is bigger than the E 1 probe suggested: INT 2 takes 0.36-0.44 ms off each SWA visit
  (~14-18 ms per chunk), so A's in-kernel term is larger than its doc estimate (7.6 ms).

Round 2 (#11423, e0f44dd13): interior tiles also when every distance >= E (no bias), so full visits' far tiles take the fast
path. **Admission** `intarsia kernels test --backend metal -m <Inkling-2-Mini> --kernel rel_sink_attn_bf16_nax`: INT 0 and
INT 1 both ADMITTED, 7 / 7, identical max_abs 5.464e-3, rel_l2 1.500e-3. **Bitwise:** INT 1 hashes equal INT 0's in all
6 configs again. NAX µs per call, median of 4 (spread within an arm is up to +-10 %):

| visit | INT 0 | INT 1 (exact) | INT 2 (no bias, interior) |
|---|---:|---:|---:|
| SWA ctx 0 | 920 | 891 | 604 |
| SWA ctx 2k | 1198 | 1171 | 839 |
| SWA ctx 4k | 1191 | 1199 | 859 |
| SWA ctx 16k | 1231 | 1174 | 893 |
| full ctx 0 | 1145 | 1044 | 717 |
| full ctx 2k | 2584 | 2340 | 1939 |
| full ctx 4k | 4013 | 3536 | 3170 |
| full ctx 16k | 12491 | 12491 | 11105 |

**B verdict:** the exact path saves ~1-4 ms per chunk (SWA 0-5 %, full 0-12 %, inside the noise for SWA), under 1 % end to
end. Below the 2 % bar: no build, no gate; the probe kernel and `INTARSIA_INK_NAX_INT` are removed from the branch (brief 23; source at e0f44dd13). The mask
tests were not the 28 ms: with E 1 the SWA kernel is still ~0.95 ms against a 0.42 ms floor, so that bucket is the softmax /
exp / rescale work around 32-key tiles, which B does not touch.

**A's bound on top of B:** INT 2 shows the in-kernel bias gather costs ~0.33 ms per SWA visit (~13 ms per chunk) and
~0.4 ms per full visit (~3 ms), more than the E 1 estimate. With the prologue's rel write (~22 ms) the removable total is
~38 ms per chunk; A still has to form the bias (16-deep MMA, ~19 % of the QK MMA) and gather it from threadgroup memory,
so the net is ~17-33 ms per chunk, **~2-4 % end to end at 4k / 8k**. B and A overlap only in the few full-visit tiles B
already speeds up.

### Attention per call, brief 21 step 1 (#11370; ink_attn_bench `--prefill`, 2048 rows, 4 layers x 10 reps, measure only)

`INK_ATTN_BENCH_E=1` shrinks the rel tensor from [2048, 24, 1024] (100 MB per call) to [2048, 24, 1]: the prologue no longer
writes it and the NAX kernel's per-score gather is gone (timing only, the numerics are wrong). NAX kernel µs per call;
rep 1 shown, rep 2 E=1 agrees within 10 %, rep 2 E=1024 ran in a slow GPU clock state (SWA 1.29-1.47 ms, prologue up to
1.26 ms) and is not used.

| visit | E 1024 prologue | E 1 prologue | E 1024 NAX attn | E 1 NAX attn | floor at 62 TFLOP/s |
|---|---:|---:|---:|---:|---:|
| SWA ctx 0 | 977 | 500 | 862 | 732 | 310 |
| SWA ctx 2k / 4k / 16k | 979 | 492-555 | 1142-1153 | 927-987 | 416 |
| full ctx 0 | 997 | 543 | 1084 | 918 | 416 |
| full ctx 2k | 1030 | 631 | 2806 | 2396 | 1250 |
| full ctx 4k | 1095 | 552 | 4627 | 3507 | 2080 |
| full ctx 16k | 1144 | 668 | 14805 | 12356 | 7070 |

SWA visits and the prologue match the model trace (SWA 1.16 ms, prologue ~1.0 ms, `.scratch/bp3/k256a.jsonl`). Full visits
do **not**: the bench runs 2.81 / 4.63 ms at ctx 2k / 4k against the trace's 2.34 / 3.61 ms (+20 to 28 %), probably because the
bench cycles 4 distinct K / V + rel operand sets. Full-visit deltas below are therefore discounted. Floor = 4 x rows x keys
seen x 128 x 24 heads / 62 TFLOP/s (QK + PV, no softmax).

Two separate buckets per chunk (40 SWA + 8 full visits, chunk at ctx 2k / 4k):

- **Attention kernel, ~40 ms above floor (the brief's number; trace ~29 ms SWA + ~9 ms full at ctx 2k):**
  - rel gather + footprint: **~10 ms** (SWA 0.19 ms x 40 = 7.6 ms; full a few ms, bench-inflated).
  - **the rest, ~28 ms, is not rel:** with E 1 a SWA visit still takes ~0.95 ms vs its 0.42 ms floor (~27 of 62 TFLOP/s):
    mask / online softmax / rescale around the 32-key tiles. Not split further yet (probe below).
- **Prologue, separate from the 40 ms: the 100 MB rel write costs ~22 ms** (0.98 -> ~0.52 ms per visit x 48).

**Bound for step 2, rel in the kernel (end to end, not GPU-chunk):** removable ~22 (prologue) + ~8 (SWA gather) + ~3 (full,
discounted) = ~33 ms per chunk, minus the in-kernel cost (~3-5 ms, below) = ~26 ms per chunk. Against the current default's
generate wall time (blend gate p256: 4k ~1.64 s for 2 chunks, 8k ~2.99 s for 4): **~3 % at 4k, ~3.5 % at 8k** (~2.5 % against
the older 1980 tok/s 4k runs). Clears 2 %, thinly.

Proposal (sent to the coordinator; not built). Prefill dispatches `ink_attn_nax_s12` (`INTARSIA_INK_NAX_SG` default 12: BQ 64,
3 heads x 4 query sub-tiles). A rel_mode 1 kernel where each simdgroup forms its own bias tile per key tile as one 16-deep
MMA, r [16 rows x 16] x proj [16 x 48 distances] (47 needed), stored bf16 in its own threadgroup slice (16 x 48 x 2 B =
1.5 KB per simdgroup, 18 KB for 12; no threadgroup barriers in the tile loop), then gathers from threadgroup instead of device
memory. Extra MMA ~48 x 16 / (32 x 128) = 19 % of the QK MMA. The rel_mode 1 operands already match (r [T,H,dr] in the rel
slot, proj [dr,E] as the 8th input, 7e4699bf6), **but the approved rel_mode 1 promises mode 1 == mode 0 bitwise; an MMA dot
will not round like the prologue's GEMV**, so this needs a contract amendment: mode 1 within a stated tolerance of the f64
reference, md5 64 / 256 may change, G2 / G3 short and long decide.

Exact alternative with no contract change (to probe first): skip the per-score window / causal / E tests on interior tiles
(every key visible, every distance < E), which is most SWA tiles.

### Prologue split default-on gate (#11272, binary 61980c882, serve recipe chunk 2048, localeval speed-plain, order off / on / off / on)

| prompt tok | prefill pair 1 (off -> on) | pair 2 | decode pair 1 | decode pair 2 |
|---:|---|---|---|---|
| ~1k | 2166 (1871-2460) -> 2140 (1921-2592), both rejected for spread | 1976 (1424-2363, rejected) -> 2067 (1964-2141) | 62.1 -> 62.3 | 61.3 -> 62.0 |
| ~4k | 2428 (2184-2464) -> **2548**, above off's range | 2413 (2409-2415) -> **2400 (2337-2500)**, not above | 60.4 -> 60.9 | 60.6 -> 60.7 |
| ~16k | 2384 (2336-2394) -> **2571**, above | 2449 (2357-2452) -> **2548**, above | 57.5 -> 57.3 | 57.2 -> 57.6 |

8k (engine generate, p8k, 2 alternating pairs; localeval has no 8k cell): off 2759 / 2751, on 2859 / 2853 tok/s (+3.6 %, +3.7 %).
All runs exited 0 with non-empty output. The serve trace shows the split ran in every on arm and in no off arm.

**Against brief 14: not met at 4k.**
- Pair 2's on median (2400) is below off's (2413), let alone above its range.
- 8k (+3.6 %) and 16k (+4 to 8 %) pass, and decode is within 1 % everywhere.
- Brief 18's 1k step was conditional on pair 2 matching pair 1, which it does not. The p1k generate pairs still run in #11282 as extra data.
- **The split stays opt-in** (ink/prologue-split-merge 61980c882 is not sent). Reported to the coordinator.

### Prologue split (#11185, ink/prologue-split 9f925c85e, `INTARSIA_INK_PROLOGUE_SPLIT=1`, default off)

| Check | Result |
|---|---|
| bitwise vs `ink_kv_prologue_bf16` (q, rel, k, v, both rings; T 17 .. 2048, static / dynamic + tau, ring 19 / 1040, pos0 up to 6144, partial 8-row block) | **8 / 8 bitwise**; the launch counter asserts the split ran on the split arm only |
| rows-exact attention suite with the split on | pass |
| lib tests (incl. `kernel_list_is_unchanged_by_core_op_fusion_p1`) | 18 / 18 |
| admission `ink_kv_prologue`, split on | ADMITTED (117 shapes), trace "prologue split" |
| greedy 8 tokens at 8k, off vs on | identical, both runs exit 0, 14 bytes each |
| prologue GPU time per 2048-row chunk (kernel trace, chunks 1 + 2) | **48.0 -> 24.1 ms** (1.00 -> 0.50 ms per visit, both kernels) |
| GPU per chunk | 654.8 -> 631.6 ms (-3.5 %) |
| e2e p8k, 2 interleaved rounds of 2 | off 2635 / 2611 / 2583 / 2730, on 2573 / 2826 / 2709 / 2814 tok/s (medians 2622 -> 2762, +5 %; noisy) |

Saved 24 ms of the 30 ms planned; the rel product now runs at ~0.25-0.3 ms per visit (100 MB written). Because it is bitwise,
G2 / G3 and md5 cannot move; the flip to default-on waits for the coordinator.

### Metal prefill chunk 2048 vs 256 (#11109, binary 378dc214e, brief 8 conditions)

G2 / G3 at `--prefill-chunk 2048` (accuracy; the 256 numbers are the earlier default runs, #10883):

| Gate | 2048 | limit | 256 |
|---|---:|---:|---:|
| G2 short (prefill rows / decode rows) | 2.087e-3 / 2.928e-3 | 4e-3 | 2.380e-3 |
| G2 long | 3.239e-3 | 6e-3 | 3.418e-3 |
| G3 short | 8.420e-3 | 1.05e-2 | 8.391e-3 |
| G3 long | 1.229e-2 | 1.65e-2 | 1.248e-2 |

All pass; serve logs "prefill chunk 2048 (--prefill-chunk)" in both 2048 arms.

localeval speed-plain, serve `--max-batch 1`, same binary, order 256, 2048, 256, 2048 (`localeval compare` per pair):

| prompt tok | prefill tok/s pair 1 (256 -> 2048) | pair 2 | decode pair 1 | decode pair 2 |
|---:|---|---|---|---|
| ~1k | 1266 -> 2055 (+62 %, rejected B: spread) | 1350 -> 2218 (+64 %, rejected B) | 62.2 -> 62.2 | 62.5 -> 62.6 |
| ~4k | 1428 -> 2170 (+52 %, faster) | 1436 -> 2446 (+70 %, faster) | 60.8 -> 59.9 (**-1.5 %**) | 61.1 -> 60.9 (-0.3 %) |
| ~16k | 1456 -> 2455 (+69 %, faster) | 1480 -> 2530 (+71 %, faster) | 57.5 -> 57.7 | 57.9 -> 57.8 |

Against brief 8: G2 / G3 pass; serve log ok; 4k and 16k prefill faster beyond spread in both pairs. Two items are not clean.
- **1k prefill:** both 2048 runs were rejected for spread. Their reps range 2026-2424, every one far above the 256 arm.
- **4k decode, pair 1:** -1.5 % (localeval calls it noise; brief 8 asks for within 1 %). Pair 2 is -0.3 %.
Recipe committed on the coordinator's call (brief 13): ink/prefill-merge 05f1c702e.

### Blend piece (ink/blend-piece de1e0568a, `INTARSIA_INK_BLEND_PIECE`, default 256 = unchanged; #11290 checks, #11322 timing)

Checks (#11290):
- tests/ink_blend_piece.rs: pieces 1024 and 2048 are **bitwise** the 256-row pieces on every output of ConvMix (looped blend) and LoopBlend, at D 2304, 3 streams, 2048 / 1890 / 300 rows. The launch counter confirms the larger pieces ran.
- inkmix rows-exact 3 / 3 at piece 2048.
- Admission at piece 2048: conv_mix ADMITTED (532 shapes), loop_blend ADMITTED (27).
- #11290's timing half is void: it ran generate at the 256-row prefill chunk (this tree has no recipe commit and the script passed no --prefill-chunk), so the pieces never mattered. Rerun as #11322 with --prefill-chunk 2048, same binary.

Timing (#11322, p8k, --prefill-chunk 2048):
- Greedy 8 tokens are identical at pieces 1024 and 2048 vs 256.
- Dispatches per looped ConvMix call: 32 (piece 256), 8 (1024), 5 (2048).
- Kernel trace at piece 2048, chunks 1 + 2: **conv_mix 75.7 -> 58.2 ms per chunk (-17.5 ms, -23 %)** against the piece-256 trace .scratch/e2e/ps0.k.jsonl (same op set). The looped part goes 63.7 -> 45.8. The ops this change does not touch run ~3 % slower in this trace than in ps0 (MoE 323 -> 332), so the device ran slightly slower; the real saving is likely a little above 17.5 ms.
- This job's own piece-256 and piece-1024 traces are **not usable**: every op, untouched ones included, ran ~1.6x slower in those two processes (GPU 1789 / 1030 ms per chunk). Cause: most likely the orphaned batch_step of ink-metal's #11314 (brief 20 note), not a device state; rerun as #11344.
- p8k tok/s, 3 rounds, order 256 / 1024 / 2048 within each round (position bias possible):

| piece | round 1 | round 2 | round 3 | median | vs 256 |
|---:|---:|---:|---:|---:|---:|
| 256 | 2492 | 2362 | 2356 | 2362 | |
| 1024 | 2542 | 2509 | 2521 | 2521 | +6.7 % |
| 2048 | 2527 | 2476 | 2481 | 2481 | +5.0 % |

Trace rerun (#11344, 07:01:24-07:01:53, safe_run, no leftovers; order 2048, 256, 1024, 256), ms per chunk, chunks 1 + 2:

| op | piece 2048 | 256 (a) | 1024 | 256 (b) | ps0 (256, earlier) |
|---|---:|---:|---:|---:|---:|
| moe_experts_situ_bf16 | 367.0 | 323.0 | 354.1 | 323.5 | 323.0 |
| conv_mix_bf16 | **61.6** | 75.5 | **61.8** | 76.0 | 75.7 |
| rel_sink_attn_bf16_nax | 79.6 | 70.3 | 76.7 | 70.3 | 70.3 |
| ink_in_proj_bf16_nax | 70.7 | 61.6 | 67.9 | 61.6 | 61.6 |
| ink_kv_prologue_bf16 | 54.9 | 48.0 | 52.9 | 47.9 | 48.0 |
| **GPU total** | **718.9** | 654.4 | **695.1** | 655.8 | 654.8 |

- **conv_mix -14 ms per chunk at 1024 and 2048**, but every op the change does not touch runs **+9 to 14 % slower** with the larger pieces.
- That pattern holds in both orders: the two piece-256 runs bracket the others and match ps0 exactly. So it comes from the piece, not from the box.
- Net GPU time per chunk is **+40 ms (1024) and +64 ms (2048)** in kernel-trace mode.
- Likely cause: the larger scratch (85 / 170 MB vs 21 MB) evicts the weights from the system-level cache, or adds memory traffic. That is not measured. Kernel-trace mode times each dispatch separately, so it can over-state the effect.
- This contradicts #11322's p8k tok/s (+5 to 7 %, fixed order). The gate #11348 (randomized order, rule fixed in advance) decides.

- Scratch for the blend: ~21 MB (256), ~85 MB (1024), ~170 MB (2048).

Gate (#11348, brief 20; binary de1e0568a, --prefill-chunk 2048, 07:11:45-07:14:43, safe_run: no leftovers):
- (2) Box check: p8k default 2731 tok/s, which is +16 % over the 2362 reference, so not slow. The 2362 came from #11322; clean endor runs since are ~2700.
- (1) G2 / G3 at piece 256 vs 1024: KL lines **identical**. G2 short 2.087e-3, long 3.239e-3; G3 short 8.420e-3, long 1.229e-2.
- (1) Greedy md5 identical at 64 tokens (a2ab7b3d0e3d15d0fae1ccf6816ac7dc) and at 256 tokens (cbd62660cd829e0009878cd8c1a3e362).
- (3) Engine generate timing, tok/s, order per pair as logged:

| length | pair 1 | pair 2 | pair 3 | median 256 | median 1024 | ratio | need |
|---|---|---|---|---:|---:|---:|---|
| 1k (929) | 1024 first: 1123 / 1258 | 256 first: 1300 / 1307 | 1024 first: 1234 / 1326 | 1300 | 1234 | **0.949** | >= 0.98: **FAIL** |
| 4k (3950) | 256 first: 2305 / 2474 | 2427 / 2397 | 2413 / 2506 | 2413 | 2474 | 1.025 | >= 1.00: pass |
| 8k (8034) | 256 first: 2372 / 2739 | 2743 / 2799 | 2738 / 2348 | 2738 | 2739 | 1.000 | >= 1.00: pass |

(pair entries are 256 / 1024.)

**Result: 1k fails, so piece 1024 stays opt-in. This is final under brief 20, with no further round.**
- The default stays 256. INTARSIA_INK_BLEND_PIECE stays as an opt-in knob on ink/blend-piece (bitwise).
- The 8k timings have one ~2350 outlier in each arm, the same size as the first-run-in-process slowdowns seen elsewhere.
- The trace rerun's extra time on untouched ops (+9 to 14 %) fits the 1k loss and the flat 8k result.


- Not decided: the default. Proposal to the coordinator: piece 1024 default-on (best tok/s, half the scratch of 2048), after a gate run (G2 / G3 short and long at chunk 2048 plus an interleaved speed check). Bitwise, so G2 / G3 should not move.

### Prologue split 4k by engine generate timing (#11290, brief 19; binary 61980c882, 3950-token prompt, 3 alternating pairs)

off 1968.3 / 2008.4 / 1979.5, on 1984.5 / 1699.7 / 1891.6 tok/s. All runs exit 0 with non-empty output.
**Median on / off = 1891.6 / 1979.5 = 0.956x: fails the 4k rule (>= 1.00x).** Together with 1k (1.007x, passes) and
localeval 4k pair 2 (2413 -> 2400, failed, above), **the split stays opt-in. This is final under brief 19; no further
round.** ink/prologue-split-merge 61980c882 is not sent to ink-metal. The kernel itself remains bitwise and 24 ms/chunk
faster in the trace (above). 8k (+3.6 %) and 16k (+4 to 8 %) were faster; at 4k end-to-end time does not show the saving.

### MoE K-step probe (#11282, brief 17; inkmoe_bench with ink-probe at 62a8b282c, balanced 128 experts, 3 reps)

Device threadgroup memory limit (MTLDevice.maxThreadgroupMemoryLength on endor): **32768 B**.

| Arm | full µs/call | gu only | dn only | output vs normal (no skip) |
|---|---:|---:|---:|---|
| a normal (K tile 64, row pad 8) | 6191 / 6202 / 6208 | 4065 / 4096 / 4345 | 2394 / 2392 / 2416 | reference |
| 0 probe copy (K 64, pad 8) | 6183 / 6193 / 6303 | 4059 / 4200 / 4449 | 2390 / 2394 / 2449 | bitwise |
| 3 K 64, no pad (pad control) | 6547 / 6557 / 6714 (+6 %) | 4320 / 4471 / 4686 | 2499 / 2503 / 2549 | bitwise |
| **4 K 128, no pad (2x K tile)** | **12282 / 12299 / 12286 (+98 %)** | 8074 / 8065 / 8079 | 4474 / 4482 / 4473 | bitwise |

(The skip-1 arms differ from the reference by design: gate+up is skipped, so down reads whatever h holds.)

- **2x the K tile is twice as slow**, not faster. The pad accounts for 6 % of that, so most of the loss is not pad bank conflicts.
- The likely cause is occupancy: 32768 B per threadgroup is the device limit, so a core holds one threadgroup instead of two or more. That is inference, not measured.
- **4x is not possible:** it would need ~64 KB of threadgroup memory, and 256 does not divide the down's K = 1152.
- Halving the barriers cannot be tested inside 32 KB without halving occupancy. The ~10 µs per tile stays unexplained and the MoE kernel keeps K tile 64. No default change.

### Prologue split 1k by engine generate timing (#11282, brief 18 / 19; binary 61980c882, 929-token prompt, 3 alternating pairs)

off 1387.5 / 1368.8 / 1363.7, on 1344.2 / 1386.1 / 1378.2 tok/s. All runs exit 0 with non-empty output.
**Median on / off = 1378.2 / 1368.8 = 1.007x: passes the 1k rule (>= 0.98x).** Cold single-process generate at 1k includes
the first-chunk page-in, equally in both arms. The 4k pairs (rule >= 1.00x) run in #11290.

### MoE step-1 probe (#11235, inkmoe_bench with `ink-probe`, built in-slot at 16442b2ff; brief 15)

Balanced routing (128 experts x 128 rows, 256 tiles per call), µs per call, mean of 3 reps (each rep's spread under 0.3 %):

| Arm | full | gu only | dn only | full saved vs control | per tile |
|---|---:|---:|---:|---:|---:|
| A normal kernels | 6202 | 4061 | 2392 | | |
| 0 probe copy, unchanged (control) | 6188 | 4059 | 2392 | | |
| B bit copy instead of the NVFP4 decode | 5796 | 3764 | 2304 | 392 | **1.5 µs** |
| C no weight / scale loads from DRAM | 5442 | 3660 | 2044 | 746 | **2.9 µs** |

- The control matches the normal kernels to within 0.2 %, so the probe copy is a fair baseline.
- **Rule result (brief 15): neither.** B saves 1.5 µs per tile, not the 6 µs needed. C is 2.9 µs of the 15.2 µs per-tile cost, so it does not dominate.
- Together B and C explain ~4.5 µs per tile. The remaining ~10 µs is the K loop's structure: 36 steps of 64 over K = 2304 per gate+up tile, each with a threadgroup stage, two barriers and a matmul2d issue. That is inference from the probes, not measured directly.
- **MoE decode work stops here.** Next is the ConvMix + looped blend fusion, as brief 10 says.
- Real replayed routing was not usable: the control averaged 9.3 ms against a 7.2 ms minimum (interference), and B ran slower than normal. Random bit patterns likely make special values in the matmul. The decision uses the balanced runs.
- This #11235 build is also brief 16's endor smoke: the bench built with `ink-probe` on endor (31f117a00 included) and ran.

### Brief 7 fix check (#11105, built in-slot at 92e7e273f, which contains ed27f4462)

| Check | Result |
|---|---|
| tiny inkling2 suite, Metal, metal4 forced off | 8 passed, 0 failed (no NoKernel) |
| tiny inkling2 suite, Metal | 8 passed, 0 failed |
| inkmix rows-exact tests, metal4 off / on | 3 passed / 3 passed |
| admission ink_in_proj, metal4 off | gemv 18 shapes, gemv_rows 24, **gemv_chunks 11 shapes ADMITTED** (max_abs 2.79e-3, rel_l2 1.84e-5) |

### MoE NVFP4 diagnosis (brief 10; #11187 for (a) and (b))

Per visit, warm chunks 1 and 2 of the 8k kernel trace: 7.35 ms. That is the same as the random-routing bench (7.29 ms), so the
real routing costs nothing extra on average.

| Dispatch | ms / visit | ms / chunk (44 visits) | Rate |
|---|---:|---:|---|
| moe_route_f32 (top-k) | 0.020 | 0.95 | |
| moe_group (count, scan, tile table) | 0.019 | 0.85 | |
| gate + up (`#1`) | 4.61 | 202.6 | 174 GFLOP: 38 TFLOP/s (floor 2.8 ms) |
| down (`#2`) | 2.50 | 110.0 | 87 GFLOP: 35 TFLOP/s (floor 1.4 ms) |
| gather / weighted combine (`#3`) | 0.22 | 9.8 | ~85 MB (y 75 MB read, out 9.4 MB written): ~385 GB/s, at DRAM speed |
| router logits (separate op) | 0.32 | 15.2 | |

(a) Rows per expert, real routing (INTARSIA_INK_MOE_DUMP, 8k prompt, chunk 2048, 132 NVFP4 calls at T = 2048):
- Median rows per expert per call: 69. The minimum is 0 (median 4 experts empty per call, up to 54). The maximum is median 1184 and up to 2048: one expert takes every row.
- Pooled percentiles: p5 1, p25 19, p50 65, p75 154, p95 456.
- **Experts under 16 rows: median 27 per call (up to 85).**
- 64-row tiles per call: median 325 against 256 balanced (+27 %); 21 % of tile rows are padding. 128-row tiles would be 203.
- MXFP8 (12 calls): the same shape (median 61, 30 experts under 16 rows, 325 tiles).

(b) Time against rows per expert (inkmoe_bench, NVFP4, 2048 rows unless noted, µs per call; `gu` = gate+up only, `dn` = down
only):

| Routing | full | gu only | dn only | 64-row tiles |
|---|---:|---:|---:|---:|
| random (this run) | 8268 | 5596 | 3031 | ~330 |
| **real (replayed dump)** | **7250** | 5831 | 2690 | 325 |
| balanced, 128 experts x 128 rows | 6202 | 4064 | 2397 | 256 |
| balanced, 96 | 6669 | 4422 | 2505 | 288 |
| balanced, 64 x 256 rows | 6050 | 4008 | 2305 | 256 |
| balanced, 32 x 512 | 5965 | 3972 | 2255 | 256 |
| balanced, 16 x 1024 | 5904 | 3945 | 2214 | 256 |
| balanced, 8 x 2048 | 5868 | 3926 | 2193 | 256 |
| random, T = 256 / 512 / 1024 | 2026 / 3050 / 4192 | | | |

Reading:
- **The cost per expert is not the problem.** 8 experts and 128 experts with the same full tiles differ by only 0.33 ms (2.6 µs per expert).
- **The cost per tile is.** Fit `time = tiles x D + rows x a` to real (325 tiles) and balanced 128 (256 tiles), same 16384 rows: D = 15.2 µs per tile, a = 0.141 µs per row.
- Per call that is 325 x 15.2 = 4.9 ms of 7.25 ms paid once per tile whatever its rows: every tile decodes and streams its expert's 64-column x 2304 weight slab.
- The rows-proportional part (A loads, MMA, epilogue) is only 2.3 ms.
- Real routing costs 1.05 ms per call over balanced. That is 46 ms per chunk, all from the 69 extra partial tiles.
- The T sweep's intercept (~1 ms at T = 256) is the weights streamed once per call (0.57 GB NVFP4).
- Floors for scale: MMA 4.2 ms per call. With D at 0 the call would be ~2.3 ms, bound by the MMA instead.

Why v65 and dq1 missed:
- v65 (two 64-row halves share one decoded stage, 256 threads) halves decodes per full 128 rows. But it raised the rows term: 11.0 ms in the bench.
- dq1 removes the per-tile decode, but writes and re-reads 2 GB of bf16 weights per call (decode 9.3 ms at ~220 GB/s). It moved the cost to memory.
- m128 (one 128-row tile) cut tiles to ~203 but blew registers / staging: 13-17.5 ms.

(c) Inner loop, from the nax2 m64_k64_v1 probes (7.38 ms full, the gen1 schedule within 1 %):
- Dropping the MMA saves 0.9 ms (6.35 ms left), so the accelerators are mostly hidden.
- Loads + epilogue alone take 2.51 ms. That leaves ~3.8 ms for the NVFP4 -> bf16 decode and the threadgroup staging.
- The loads are mostly weight streaming: each 64-row tile re-reads its expert's 64-column weight slab. At ~2.5 tiles per
  expert that is ~1.3 GB of NVFP4 per call, ~2.3 ms at 550 GB/s.
- Decode and staging scale with the same tile count.

So per call: ~3.8 ms decode + staging, ~2.5 ms loads + epilogue, ~0.9 ms exposed MMA, against an MMA floor of 4.2 ms.

(d) Routing, grouping and gather between the expert dispatches: 11.6 ms per chunk (route 0.95, group 0.85, gather 9.8). The
gather runs at DRAM speed. Folding it into the down epilogue would save at most ~10 ms, and not bitwise (the combine order
changes). In the cb-mode trace the gaps between these dispatches are 0.

## General note: scratch footprint (brief 21)

A prefill scratch footprint above ~21 MB measurably slows prefill ops it does not touch. Blend piece 1024 / 2048 (scratch
~85 / 170 MB vs ~21 MB) made MoE, attention, in_proj and the prologue +9 to 14 % slower in a clean, bracketed trace (#11344),
which cancelled the blend's own 14 ms saving end to end (#11348). The cause is likely system-level-cache eviction of the
weights, but that is not measured. **Every new prefill kernel reports its scratch size, and its end-to-end gate is judged
with that scratch in place.**

## Next ideas (ranked)

1. MoE experts (54% of the chunk): the gen1 kernel is staging / decode bound, not MMA bound (no-matmul probe 6.3 of 7.3 ms).
   Try decoding each weight block into threadgroup memory once per BN x BK step shared by two M tiles of the same expert
   (2 x 64 rows, 256 threads), and an fp8/fp4 -> bf16 decode with fewer ALU ops; the 128-thread m128 form is not the way.
2. Host gap: closed. It is a once-per-process first-touch cost in chunk 0, 0 ms in later chunks (section above).
3. Router as a NAX GEMM (logits_bf16 runs at 3.9 TFLOP/s: 14 ms -> ~1 ms per chunk).
4. rel_mode: stage the bias per (tile, distance band) in threadgroup memory instead of 16 FMAs per score; then re-measure
   (branch ink/prefill-rel).
5. Metal prefill chunk 2048 in the recipe (serve and accuracy use 256 today): pending #11109.
