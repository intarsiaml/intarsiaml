# Inkling-2-Mini attention on CUDA (lane ink-cuda-attn, 2026-11-21)

## Briefs received

1. 2026-11-21 (coordinator, via SendMessage): **Approved: route CUDA to the fused op 75 for t <= ATTN_DECODE_MAX_ROWS**,
   with the rule as proposed, accepting the one exception (full 4k at 4 rows). Conditions:
   - make the edit on ink/cuda-attn as its own commit;
   - touch only that condition in `model/inkling2.rs`, so the Metal arm stays byte-identical (dfuse and batch also edit
     that file; keep the diff minimal);
   - gate: the tiny Inkling model on CUDA (spark1) gives bitwise-identical logits / tokens with the rule on and with the
     pair forced, and `plan show` on Metal is unchanged (checkable from the code: the Metal arm is untouched);
   - record this brief and report the hash.
   Done: d536ff099. The tiny-model gate is blocked until the mixmoe ops exist on CUDA.
2. 2026-11-21 (coordinator): **Accepted the op-level `inkling2_attn.rs` gate** as cover for d536ff099 for now. The tiny-model
   on/off gate runs once ink-cuda-mixmoe has its ops; the coordinator forwards the ref. Next lever, my pick with the reason:
   the 4-row full-4k case (131 vs 122 us) or the full-32k 1-row case (736 us).
3. 2026-11-21 (coordinator), in order:
   1. The tiny-model gate moves to ink-cuda-mixmoe: they merge tip fd86495f2, run fused-rule-on vs pair-forced, and
      extend the refusal test. No ref needed here.
   2. Do idea 1 (64 keys per warp step on full-context decode). Target: <= 560 us at full 32k, 1 row. Admission and the
      rows-exact tests must still pass; no case at 512 or 4k may get more than 2 us slower.
   3. Then idea 3 (the combine folded into the tile kernel), if rows-exact stays bitwise.
   4. Idea 2 (contract change): not now; Metal shares the contract, revisit after decode. Idea 4 later.
   Scope: only this worktree; timing runs on a spark1 slot with --quiet gpu,cpu; no tool call or wait over 4 minutes.
4. 2026-11-21 (coordinator, rule for all Inkling lanes): this session and its worktrees run on endor, the Metal timing
   box. No local cargo build / check / test there outside a fleetq slot, since that skews endor's quiet timing jobs.
   CUDA builds and tests stay on spark1 / spark2 through fleetq. Reading files, git and local edits are fine. (Before
   this rule the lane ran `cargo check` / `cargo test --lib` for the CUDA crate and one tiny-model refusal test locally;
   from now on it runs none.)
5. 2026-11-21 (coordinator): **Approved: the staged body restored for window > 0 layers** (a per-layer constant; rows-exact
   and fused == pair stay bitwise). Then, in order:
   1. One cheap diagnostic, a single spark1 quiet job: the same 134 MB read three ways with the read-only kernel:
      (a) one linear stream; (b) today's 264 interleaved K/V row streams; (c) each block one contiguous range of a single
      head, >= 64 KB per block. If (a) and (c) reach ~240+ GB/s and (b) stays ~195, the cause is stream count / pattern:
      rework the decode grid so each block owns a contiguous key range per head, one attempt, measured against 668. If all
      stay ~195: footprint / TLB, record and accept 668.
   2. Then idea 3 (fold the combine into the tile kernel) under the same bitwise rules.
6. 2026-11-22 (coordinator): brief 5 accepted (read-pattern gain kept, fold revert right). (1) Rebuild the spark1 build
   dir from tip 7628e87cb (fleetq, --load cpu) so no stale fold binary is left. (2) Message ink-cuda-mixmoe
   (a4fb1801b5d47a28b) the tip 7628e87cb to merge; it owns the merged CUDA line and the real-model G2/G3. (3) Brief 6, once
   mixmoe's merged tip passes real-model G2/G3 on CUDA: one nsys profile of real-model greedy decode at ctx 1k and 8k (quiet
   fleetq slot, memory budgeted per the Spark OOM rules); report tok/s and the top 5 ops by time with their byte floors;
   attention work again only if it is in the top 3, else send the coordinator the top item with a one-paragraph plan before
   building.
7. 2026-11-22 (coordinator): GO on brief 6 step 3 at 8f7d5cd3a (real-model decode fixed, eager 30.2-30.7 tok/s; vLLM
   NVFP4 37-43). Copy the binary to spark1, run prof_run.sh at 1k / 8k. Report, besides the top-5 ops, the eager launch
   overhead per token: total GPU-idle gaps between kernels, kernel count, mean gap (decides whether CUDA graph decode is
   the next lever).
8. 2026-11-22 (coordinator): after reporting the profile, build CUDA B=1 graph decode for Inkling-2. (1) ink_pass takes
   `dpos: Option<&TensorRef>` and sets call.pos on its 5 positioned call sites (inkling2.rs ~791/816/826/849/924); (2)
   model.rs:4077 passes dpos instead of refusing; (3) check plan.rs decode_position_stable holds for Inkling (no T=1 kernel
   choice depends on kv_len); (4) gates: tiny graph == eager bitwise, real-model greedy 64/256 md5 graph == eager, G2 decode
   within budget, graph tok/s short / 5k vs eager 30.5 / 29.2, nsys of the graph step (remaining gaps, top ops); (5) new
   branch ink/cuda-graph from ink-cuda-mixmoe 8f7d5cd3a, edit only inkling2.rs, model.rs, plan.rs plus tests (mixmoe stays
   out of those); (6) Metal / cpu behaviour identical, tiny cpu G1 on spark2. Estimate half a day; message the hash and
   gate results.
9. 2026-11-22 (coordinator): graph decode accepted (mixmoe merges ink/cuda-graph 5fa5a88bd). Brief 9: port P12 (lossless
   12-bit bf16 packing) to CUDA for the dense GEMVs (in_proj 8.8 ms vs floor ~6.96, wo 2.8 / 2.49, unembed 3.8 / 3.39 at
   1.2k; Metal P12 cut dense bytes 2.214 -> 1.663 GB, bit-exact over 177 tensors, steady step -7.9 %, default-on there).
   (1) Branch ink/cuda-p12 from ink/cuda-graph 5fa5a88bd, merge the ink/pack12 tip (P12 format, loader, Metal op 80
   MatmulP12, the 73 p12 arm, INTARSIA_INK_PACK12); worktree .worktrees/ink-cuda-p12 only, spark1 ~/intarsiaml-ink-cuda-p12
   only. (2) Do not change the format (codes U8 [N,3K/2] + aux U32 side list, per-row exponent base, escapes). (3) New
   files (inkp12.rs + new .cu): MatmulP12 for wo / unembed / small GEMVs, a p12 arm for ink_in_proj (73) with one minimal
   hook in mixmoe's dispatch; tell mixmoe (a4fb1801b5d47a28b) the exact lines before committing. (4) P12 output bitwise
   the CUDA bf16 kernels at rows 1..16 (decode in-register, same reduction order); Metal's crossover (P12 <= 4 rows, bf16
   >= 5) unless measured otherwise; graph-capturable. (5) Gates on spark1: rows-exact P12 vs bf16 rows 1..16, tiny 12
   tests, greedy 64/256 md5 on == off with graphs, G2 decode short / long equal to off. (6) tok/s off vs on at short /
   1.2k / 5k with graphs, interleaved, --quiet gpu,cpu; one nsys of the on step with per-kernel ms and GB/s for in_proj,
   wo, unembed. Default stays off. Report rows-exact, md5, G2, tok/s, per-kernel GB/s, branch hash.
10. 2026-11-22 (coordinator, ahead of the in_proj-fix rerun): (1) coverage per tensor class from nsys: if P12 in_proj is
    still slower than bf16, keep in_proj on bf16 and P12 for wo + unembed only, in the loader's per-op choice (not by
    removing kernels); if faster, keep all three. (2) Default-on on CUDA approved if md5 64/256 identical on vs off, G2 decode
    identical on vs off, rows-exact pass, and tok/s on > off beyond spread at short / 1.2k / 5k with graphs; its own commit,
    INTARSIA_INK_PACK12=0 reverts, same flag semantics as Metal (626adac82 on ink/pack12). (3) Then send the hash to mixmoe
    (a4fb1801b5d47a28b) to merge after its current job, and the final tok/s table + per-kernel us to the coordinator. (4) If
    any condition fails, default stays off; report.
11. 2026-11-22 (coordinator): P12 default-on on CUDA accepted; leave the mixmoe hand-off as set up; don't push. Brief 11:
    (1) rank the decode-step kernels of the 1.2k P12-on nsys (top 10, us/step, %) into this doc; (2) pick the biggest
    candidate not near the roofline (wo / unembed are); for in_proj consider a cheaper unpack only if the estimate is >= 2 %
    of step time; (3) same rules: own commit, off-switch env, bitwise or rows-exact test, G2 decode, greedy md5 non-empty,
    2 interleaved tok/s runs, backend binary built last. Report the table and the pick (with estimated gain) before the first
    job of the brief runs.
12. 2026-11-22 (coordinator): the pick is approved: the P12 in_proj twin with a 2-k-block-deep code prefetch, mma chain
    unchanged (bitwise). Timebox: if two prefetch variants don't get P12 in_proj below bf16 per visit in nsys, stop and move
    to the SWA-tile fallback (#8); no third variation. The table goes to mixmoe (MoE gu + dn = 29.7 % of the step).
13. 2026-11-22 (coordinator): brief 11 closed and accepted. No SWA single-block redesign. P12 in_proj parked until ncu
    (the user has not decided on admin counters). Next: propose the top two CUDA candidates outside mixmoe's lane (MoE gu / dn,
    ink_row, blend GEMV), ranked by estimated end-to-end tok/s gain against vLLM on spark1. Look at (a) CUDA prefill at
    2k / 8k vs vLLM (gap, kernels), (b) porting the Metal MTP / DSpark speculative verify to CUDA (Metal flip gate #11186;
    what cvat PR #464 already gives), (c) a bitwise route for bf16 in_proj from 214 to about 250 GB/s. Per candidate: estimate
    with basis, files touched, overlap with mixmoe. Nothing heavy measured before the pick.
14. 2026-11-22 (coordinator; sent as "item 13", numbered 14 here because 13 is the request it answers): the pick is #1,
    DSpark speculative decode on CUDA; #2 parked. Branch: new worktree .worktrees/ink-cuda-mtp from the ink/mtp tip 85265f9d5
    with fc773d0fc merged; don't wait for ink/mtp -> ink/metal (Metal flip gate #11186). dspark.rs / inkling2.rs / load.rs:
    only the minimum CUDA wiring, every touch listed here (the mtp lane owns them). Milestone 1: an inventory of the draft's
    ops (what Metal's 92-95 compute, shapes, counts per cycle); CUDA kernels in my own files; the CUDA draft forward vs the
    cpu draft reference (logits KL or bitwise, whichever the Metal lane used); CUDA tests green, admission. Milestone 2:
    end-to-end greedy spec at k3 tau 0.8; gate greedy md5 spec == no spec at 64 / 256 and 1.2k, CUDA suite green, acceptance
    reported; speed: quiet localeval speed-plain off vs on (spark1). Rules: MoE at 2..4 rows is mixmoe's (measure its cost
    on the verify path, don't change it); device positions for the draft are mine; stop after two failures on the same
    approach and name the broken assumption; spark1 memory budget (cap compile jobs, honest --mem). Prefill is routed to
    mixmoe.
15. 2026-11-22 (coordinator): inventory accepted (no new op; 92-95 stay with mtp). (1) a new env var goes into
    docs/tuning.md (docs_env_vars); (2) before milestone 2, run the draft forward on the real checkpoint for one real prompt,
    CUDA vs CPU: draft top-1 agreement and KL over about 64 steps (the real hc_pre n, head_dim and window).
16. 2026-11-22 (coordinator): milestone 1 accepted; go on milestone 2. (1) KL tail not blocking; optional cheap replay (CUDA
    draft on the CPU's tap inputs, 10 steps) to say whether the tail comes from the target. (2) First: rows-exact test of the
    CUDA verify-row path at 2 / 3 / 4 rows == 1-row decode bitwise; report any non-rows-exact kernel before fixing. (3) Gates
    add: a sampled t1.0 request runs zero verify passes on CUDA; serve smoke at 1k + 9k, HTTP 200, spec steps > 0. (4) Spec
    under CUDA graphs is milestone 3, not now. (5) load.rs / dspark.rs edits stay small separate commits; CUDA default stays
    off until the coordinator's call.
17. 2026-11-22 (lane ink-cuda-mixmoe, via SendMessage): serve blocker fixed in `ink/cuda-mixmoe` 24ab7961d (situ tol
    2e-3, admission cases seeded from their label, admission cache version 2: first serve load re-runs admission, ~3-4 min);
    merge and rerun servediag. Merged into `ink/cuda-mtp` as 7f2c87acd.
18. 2026-11-22 (coordinator): same fix; merge 24ab7961d into `ink/cuda-attn` (done, c66732975), rerun servediag with the same
    flags, check load / answer / port closed; Metal half unverified until endor #11301.
19. 2026-11-22 (coordinator): localeval speed run first, KL-tail replay after only if slots allow. (1) localeval
    (local-model-bench) in a fleetq slot, --quiet gpu,cpu, honest --expect; ink/cuda-mtp 7f2c87acd, smoke serve flags, spec
    off vs --spec dflash k3 alternating, >= 2 pairs, prompts ~1k / ~4k / ~9k, greedy. (2) Per cell: decode tok/s, prefill
    tok/s, TTFT with spread; acceptance from spec metrics. (3) Compare to vLLM's recorded 37-43 tok/s and the ~50 floor; do not
    take vLLM down, cite the record. (4) A cell localeval rejects for spread: say so, add generate timing, do not drop it.
20. 2026-11-22 (coordinator, addendum to 19): run each arm as it would serve: spec off on the graphed default, spec on on its
    real path; label each arm's mode. If spec on is eager only, add one eager spec-off arm (same prompts, 1 pair) to separate
    the spec gain from the graph gain.
21. 2026-11-22 (coordinator): KL-tail replay on hold (greedy md5 covers greedy; needed before any sampled spec). In order:
    (1) merge e0a83ecb8 (prefill_chunk 2048 in recipes/inkling2.toml [tuning.cuda], ink/cuda-mixmoe) into ink/cuda-mtp,
    build, greedy md5 unchanged. (2) Decode gap profile, measure only: nsys on graphed spec-off decode at ~4k; classes attn
    projections / attn core / MoE experts / router / unembed / mix-conv / rest, each ms per token, bytes, GB/s vs 273, gap;
    top 3 gaps with ceilings; MoE findings to ink-cuda-mixmoe, attention ones mine. (3) Then milestone 3: spec under graphs,
    same gates (verify rows bitwise, greedy md5 spec on vs off, serve smoke). fleetq --quiet gpu,cpu for the profile.
22. 2026-11-22 (coordinator): milestone 3 accepted. Next: graph the DSpark draft pass. Decisions: the verify graph is exact
    and may become default-on, scoped to Inkling on CUDA only (generic INTARSIA_VERIFY_GRAPH default for HIP and other models
    unchanged); not flipped alone. When the draft graph lands, ONE gate with both default-on (env unset) vs both off:
    (1) md5 64/256/1.2k spec on vs off, (2) replay/capture counts, 0 capture failures, (3) serve smoke 1k + 9k, (4) ABAB tok/s
    256/1k/4k, (5) peak GPU memory before vs after. Each switch keeps its own =0 revert, documented in docs/tuning.md. Then
    the attention-side decode gaps (SWA tile occupancy, in_proj bytes).
23. 2026-11-22 (coordinator): draft graph dropped (agreed). Order (a), (b), (d), then (c). (a) Gate now: verify graph
    default-on, Inkling on CUDA only, own =0 revert, peak GPU memory before vs after. (b) Draft weight bytes: a greedy spec
    output equals the target's greedy output whatever the draft computes (verify is bitwise the 1-row decode), so draft
    precision moves acceptance only; fp8 allowed; judge on net tok/s (bytes saved vs acceptance lost), md5 spec on vs off still
    identical; check ink/cuda-p12 (.worktrees/ink-cuda-p12) for an existing CUDA P12 path and reuse it before writing one.
    (d) Attention-side gaps before (c): the bigger pool (2.3 ms projections + 1.2 ms core per ~30 ms token), helps spec-off
    and serving; stay off MoE and mix/conv (ink-cuda-mixmoe). (c) Device Markov step last, only if (b) and (d) are done or
    stalled (2.5 % ceiling, touches host control flow).
24. 2026-11-22 (coordinator): reorder after (a). mixmoe's #11401 trace (hoist on, 4b877086d on ink/cuda-mixmoe) puts InkInProj
    (`ink_mma_gemv`) at 8.7 ms per token, the largest decode kernel on CUDA; it is mine under (d), mixmoe stays off it. New
    order: (a) verify-graph gate, then InkInProj, then draft bytes (b), then the rest of (d), then (c). For InkInProj first
    bytes per token, achieved GB/s, floor; bf16 weights -> a lossless P12 path (check ink/cuda-p12 first); if already near the
    DRAM floor, say so and move to (b).
25. 2026-11-22 (coordinator): GO on the GEMV-form in_proj (P12 GEMV, bitwise its bf16 GEMV twin), opt-in, default off; (b)
    fp8 draft A/B meanwhile is fine. Same form Metal uses at T=1. Conditions: (1) rows-exact is the hard constraint: verify
    rows 2-4 must equal 1-row decode bitwise, so with the switch on the GEMV form (same per-row reduction order) serves every
    decode row count inside the rows-exact range (1..16 or the current threshold), not only T=1; measure us per call at
    1/2/4/8/16 rows; if a per-row GEMV loses to mma at higher rows, try a multi-row GEMV reading each weight tile once for all
    rows while keeping each row's sum order; prefill above the threshold may stay on mma. (2) Gate with the switch on:
    ink_rows / verify rows bitwise; greedy md5 spec on == off (md5 moves, expected); G2/G3 plus decode-mode G2 against the
    current CUDA values (G2s 2.38e-3, G2l 3.56e-3, G3s 6.63e-3, G3l 1.17e-2); ABAB tok/s 256/1k/4k spec off and on; prefill
    tok/s at 2k/8k if the prefill path changes. (3) Default-on moves greedy md5: the user's call, via the coordinator.
26. 2026-11-22 (coordinator, after the in_proj GEMV gate #11456): put the CUDA verify pass's unembed and wo on the P12 rows
    kernels (~0.66 ms/token at g256 spec on, ~2.6 %; mixmoe profile #11462). (1) wo [2304 x 3072], 24 calls per verify pass
    at rows 1 + drafts (up to 8): verify runs bf16 gemv at 78.7 us vs decode P12 49.6 us, because inkling2.rs:1132 takes
    wo_p12 only at t <= DENSE_ROWS_MAX = 4 (a Metal tuning). Measure P12 rows vs bf16 gemv_rows at 5-8 rows first, then a
    CUDA-only limit; Metal's plan show for Inkling must stay identical. (2) Unembed [200058 x 2304], 1 call per verify pass:
    bf16 3.87 ms vs P12 2.92 ms; head_proj (model.rs:5726) already takes P12 to 16 rows with exact_rows, so the verify head
    reaches bf16 another way: check the screened-head path near model.rs:5690 (LogitsGather / ScreenArgmax). (3) Bit contract:
    each verify row == 1-row decode bitwise; prove p12_gemv_rows == bf16_gemv_rows and p12_logits_gemv_rows ==
    logits_bf16_gemv_rows per row with rows-exact tests at 1..8. Gate: md5 spec on == off == current at g64 / g256 / g1.2k,
    verify-graph test bitwise, ABAB tok/s spec on at g256 and g1.2k. Rule: bitwise and >= +1 % -> default-on in its own
    commit with a revert switch and a docs/tuning.md row. Independent of the in_proj GEMV switch. spark1 or spark2 via fleetq.
27. 2026-11-22 (coordinator): keep INTARSIA_INK_IN_PROJ_GEMV opt-in. Before the user: rule out a precision loss (+19 % decode
    KL, 3.73 -> 4.43e-3, is more than a reordered f32 sum should cause). (1) Kernel level, real shape [8664, 2304], 1 row,
    ~20 real decode inputs (dump x from a decode run): max-abs and RMS error vs an f64 reference for today's mma form and the
    GEMV form, P12 and bf16; check accumulator dtype, output rounding, any bf16 intermediate. (2) GEMV error <= mma error ->
    drift (near-tie flips), put to the user as "accept, +8-9 %"; clearly worse -> find and fix the precision step, regate.
    (3) 2-3 lines: which G2 decode rows moved most (prompt, position), and whether they are the drift rows that already put
    CUDA at 3.73e-3 vs Metal 2.59e-3. Then continue brief 26.
28. 2026-11-22 (coordinator): brief 27 accepted (GEMV error <= mma error: the G2 move is drift). INTARSIA_INK_IN_PROJ_GEMV stays
    opt-in; the coordinator puts it to the user as "accept, +8-9 %". Do not flip it.
29. 2026-11-22 (coordinator): CUDA prefill. Ours ~1460 tok/s marginal vs vLLM 2770-4018 (~2x behind); no lane owns it.
    (1) nsys prefill on spark1 at 1k / 4k / 8k, serve's chunk size, current tip. (2) Per chunk, ms by op class (in_proj,
    attention, MoE gu+dn, blend/mix, unembed, launch gaps), each with its floor (bytes / peak BW or FLOPs / peak tensor).
    (3) Rank levers by ms saved; top 3: what to build, estimated gain, bitwise or not. (4) Report before building anything
    over half a day; under half a day, >= 3 % and bitwise: build and gate as usual. fleetq spark1 --quiet gpu,cpu.
30. 2026-11-22 (coordinator): proposal approved. (1) Lever 2a: rel bias computed inside flash, no [T,H,E] table round trip;
    bitwise (same 16-term chain, same bf16 round). Gate: prefill rows bitwise vs today, G2/G3 unchanged, greedy md5 spec off
    and on unchanged, prefill tok/s ABAB 1k / 4k / 8k in warm serve; default-on at >= 3 % with a one-env revert. (2) Lever 2b:
    flash retile (not bitwise). Gate: G3 prefill short / long / 4k within budget, G2 decode within budget (report KL next to
    today's: decode budget is tight). **Speed bar, fixed before the run: >= 5 % prefill at 8k (warm serve); below that it
    stays opt-in and is reported.** (3) Lever 3 (mix row kernels over D): mine after 2, only once mixmoe's PR is merged to
    main (its files; rebase on main first), bitwise only. Lever 1 (block-scaled MoE GEMM) goes to mixmoe.
30a. 2026-11-22 (coordinator): contract option A approved: lift op 75's T <= 16 bound. (1) Contract doc: op 75 at any T
    is bitwise RelSinkAttn(InkKvPrologue(..)); ops 70 / 71 unchanged; the pair stays the fallback and the ink_dump path.
    (2) Graph choice by backend admission only, no new env. Metal and CPU op 75 registrations stay <= 16 rows. Proof:
    `plan show` byte-identical to main for Inkling on Metal and CPU, and for Qwen3-8B and Llama on Metal. (3) Test: op 75
    vs the pair, bitwise on CPU and CUDA, at T = 17, 64, 2048, SWA and full, ctx 0 and > 0. (4) ink_bench timing first;
    below about 0.5 ms saved per visit, stop and report. Note: Metal's prefill prologue pays a similar rel-table write
    (~22 ms per chunk); Metal could later admit op 75 at T > 16 with its own kernel. Not now.
31. 2026-11-22 (coordinator, USER DECISION 6: "do what you would recommend"): in_proj GEMV default-on.
    `INTARSIA_INK_IN_PROJ_GEMV` default-on for Inkling-2 on CUDA, its own commit, `=0` reverts. The user accepted the G2
    short decode drift 3.73e-3 -> 4.43e-3 (limit 4e-3). Reason: the f64 precision test shows no precision loss (RMS 0.289 vs
    0.288 ulp; wrongly rounded 0.008 % vs 0.010 %), only near-tie drift. Gates: G2 long and G3 short / long / 4k within
    limits; re-record greedy md5 64 / 256 / 1.2k as the new CUDA references, `=0` reproduces the old; rows-exact and
    batched == solo hold; quiet ABBA tok/s at 256 and 1.2k, spec off and on, gain beyond spread.
32. 2026-11-22 (coordinator, USER DECISION 5): CUDA spec default-on. Recipe `mtp_default_k` in recipes/inkling2.toml
    `[tuning.cuda]` at the measured best k / tau; revert `INTARSIA_SPEC_DEFAULT=0` or `--spec off`. Missing drafter weights:
    spec off with one stderr line, never a load error. Sampled requests exact under spec or plain steps (same seed, same
    tokens as spec off). A batched server must not change output. Gates: md5 on == off at 64 / 256 / 1.2k; sampled
    same-seed identity; serve smoke (HTTP 200, spec steps > 0); localeval speed-plain off vs on at 142 / 1k / 4k / 16k.
    Bar: decode gain at every length; prefill and TTFT within 2 % at every length (Metal lost 6-7 % prefill to drafter
    ingest: if CUDA does too, stop and report before flipping). Landing: hand both commit hashes to cuda-mixmoe
    (a4fb1801b5d47a28b) for its main PR or a follow-up. Work only in .worktrees/ink-cuda-mtp and the spark1 checkout.
30b. 2026-11-22 (coordinator): 30a go with (a), keeping 31 / 32 clean. (1) Now: op-75 prefill routing behind
    `INTARSIA_INK_PREFILL_FUSED`, default OFF (`=1` opts in), small commit on 86060ae6f, so the 31 / 32 gates measure only
    their own flips (32's prefill / TTFT bar must not include it). (2) After the 31 / 32 gates: e2e gate in a quiet spark1
    slot: warm-serve ABAB 1k / 4k / 8k / 16k, G2 / G3, md5 on == off. (3) Default-on (own commit, `=0` reverts) only if
    prefill >= 1 % faster at 4k and 8k, not slower beyond spread at 1k and 16k, and bitwise; else opt-in, recorded.
35. 2026-11-22 (coordinator): briefs 31 / 32 green. No hash handoff to cuda-mixmoe: ink/cuda-mtp (151 commits ahead of
    main, base 03e493df2, not stacked on #491) lands as its own PR, owned by this lane. (1) Wait for #491 to merge. (2) Landing
    branch from the new main in .worktrees/ink-cuda-mtp-land; merge ink/cuda-mtp (CUDA DSpark, verify P12, softcap, GEMV
    default-on fd56f413e, spec default-on ddb7bb164, batch test builds); resolve conflicts with #491 (same CUDA Inkling files)
    carefully. Op 75 prefill joins as its own commit only if 30b passes its flip bar, else opt-in or out. (3) Gates on the
    merged tree: CUDA greedy md5 64 / 256 / 1.2k = the new references, spec on == off; `INTARSIA_INK_IN_PROJ_GEMV=0` and
    `INTARSIA_SPEC_DEFAULT=0` reproduce the right references; CUDA plan show 35B / 8B unchanged vs main except the intended
    Inkling diff; Metal + CPU plan show Qwen3-8B / Llama byte-identical; Inkling Metal greedy md5 unchanged; HIP check build;
    docs_env_vars; tiny CUDA rows-exact. (4) PR with numbers and both reverts in the body; merge on all-green CI; remove the
    landing worktree. Rules: fleetq slots + earlyoom budget on sparks; cargo / git push on endor only in --load cpu slots.
36. 2026-11-22 (coordinator): #491 merged as main 8c7bb7cb1. Open the CUDA flips PR from the landing branch now (GEMV default-on,
    decision 6, `INTARSIA_INK_IN_PROJ_GEMV=0` reverts; spec default-on, decision 5). Merge main in, rerun the spark md5 / gates
    the merge touches, push in a fleetq CPU slot, let CI run. Do NOT merge yet: #497 (Metal MoE) goes first; on the
    coordinator's word, merge main once more and land.
37. 2026-11-22 17:21 EST (user, via coordinator): "Yes, merge in order": #497, then #498 (this lane: CUDA DSpark spec + in_proj
    GEMV default-on), then #499. When #497 is on main (`gh pr view 497 --json state` at most every 4 min, or the coordinator's
    word): merge main into ink/cuda-mtp-land, rerun the CUDA md5 gates and tests on a Spark via fleetq, wait for green CI, merge
    #498, send the merge commit to the coordinator at once (#499 waits on it). A failing gate or CI: fix and report, no merge.
33. 2026-11-22 (coordinator, rule): this session and every .worktrees checkout live ON endor. Every cargo, rustc or rustfmt
    run on endor goes inside `fleetq run --host endor --load cpu --expect N ...` (or on spark1 / spark2 / bespin in a CPU
    slot). No exceptions for quick checks. (Violated 12:20-12:41 EST with unslotted local cargo test / check runs while
    #11517's quiet timing slot was active.)
34. 2026-11-22 (coordinator, rule): `git push` from endor runs the pre-push hook's two `cargo test`s, so every push is a
    cargo run: push only inside a fleetq slot, e.g. `fleetq run --host endor --load cpu --expect 10 -- git push ...`.
38. 2026-11-22 (coordinator, check A): run `cargo test --features cuda -p intarsia-models --test inkling2_batch` on spark1
    via fleetq on main 576ead89b. If it fails, run it on 0c8f2c037 (pre-#498) too, fresh target dir per tree, confirm the
    binaries differ. Main fails + 0c8f2c037 passes: #498 caused it; find the cause, propose a fix PR (or revert of the
    offending default), no merge. Both fail: pre-existing; record it and say what the test checks. Report at once.
39. 2026-11-22 (coordinator, brief 46, resent): go on the ranking. This lane now owns the CUDA mix/conv row kernels and the
    loop-controller/blend kernels; mixmoe (a4fb180) keeps only expert kernels + repack. Order: (1) levers #1+#2, split the
    single-block mix/conv row kernels across blocks, decode and prefill, each row keeping its reduction order; (2) lever #3,
    P12 in_proj GEMV toward its byte floor; (3) lever #7, unembed P12 GEMV (0.5 day) wherever it fits; (4) lever #4, loop
    controller / blend fusion after 1-3. Hold #5 and #6 (not bitwise); #8 measure-only if a Spark slot is idle. Per lever:
    gates = CUDA md5 64/256/1.2k equal faf40cecb06b / 029e44fede1d / dc28658094c0, spec on == off, tiny CUDA tests incl.
    inkling2_batch. Ship bar >= 2 % decode tok/s at g256 spec off, spec on not slower; for #2 >= 2 % prefill at 8k.
    Opt-out env + tuning row, one PR per lever (#1+#2 may share); ask before merge. Spark slots via fleetq. Worktree
    .worktrees/ink-cuda-attn/. Stop and report if a lever cannot be bitwise; no third variant. #499 is the chat-reuse
    lane's, not this lane's.
40. 2026-11-22 (coordinator): test-fix PR yes. Contents: p12_selection fix (fde495485), the ragged / ConvMix ink_rows cases
    and the results-doc lines, from main 576ead89b. Rerun p12 + ink_rows on the final tree, unset and GEMV=0, push in a
    fleetq slot, open the PR. Merge on green CI once #499 is on main (coordinator's word or `gh pr view 499`); test + docs
    only, so no other approval needed; send the merge commit. Do not touch RmProfilingAdminOnly (a system setting).
41. 2026-11-22 (coordinator, brief 48): lever #5, prefill flash-attention retile (~0.3 s per 8k, ~9 %). Build it OPT-IN
    (env var, default off), since it changes output. Gates: KL against the bf16 / golden reference at short, 4k and 8k under
    the existing quality rule (only KL increases count, against the 5 % limit), plus greedy agreement at 1.2k. Speed: quiet
    prefill A/B at 4k and 8k, >= 2 % to count. Report the numbers; flipping the default is the user's call. Then lever #6
    (SWA attention more blocks, ~2 % decode) on the same terms. Same worktree. If the retile cannot beat 2 % within two
    variants, stop and report.
42. 2026-11-22 (coordinator): #499 is on main as 08ae803e0. The p12 test-fix PR merges on green CI; merge main in first if
    GitHub shows it behind or conflicting. Send the merge commit. Brief 48 continues.
47. 2026-11-23 (coordinator, brief 50; items 43-46 are on branch `ink/cuda-attn`): build lever 1, PDL on the attention decode edges. Branch `ink/cuda-pdlattn` off
    origin/main a4e281264, worktree `.worktrees/ink-cuda-pdlattn/` only, spark1 only. Edges: in_proj -> tile (the tile prefetches
    only old K / V rows and weights before `griddepcontrol.wait`) and tile -> comb -> wo. Reuse the `ink_moe.cu` helpers
    unchanged (ask before changing them; mixmoe may touch them). `INTARSIA_INK_ATTN_PDL=0` reverts; default on (must be
    bitwise); a docs/tuning.md row. The PDL rule (before the wait, read only data from kernels already known finished), with
    each edge's reasoning in a code comment. Do not touch flip branches A / B or lever 3. Gates: (1) all CUDA tests plus a test
    that fails if a pre-wait read touches the current token's K / V row (or why it cannot be tested); (2) bitwise: default and
    `=0` give main's md5s faf40cecb06b / 029e44fede1d / dc28658094c0 and the default 1.2k ref, spec off and on; (3) quiet
    decode A/B vs main, short and 5k, spec off and default, alternated, `--quiet gpu,cpu`; ship bar >= +1 % at both lengths
    spec off; report graphed ms / token and the new idle; (4) under +0.5 %: park with the numbers; no lever 2 (separate
    brief). Report branch, head, numbers; no push (the coordinator opens the PR). Spark memory: 121 GB budget, MAX_JOBS=4,
    10 GB free (15 without earlyoom).

## Summary

CUDA kernels for the three Inkling-2-Mini attention ops on the DGX Spark GB10 (sm_121), branch `ink/cuda-attn`. All three
are ADMITTED against the f64 cpu reference. The decode / verify path (1..16 rows) is rows-exact, and `InkAttnDecode` is
bitwise the unfused pair. Real shapes: 24 q / 8 kv heads (G = 3), head_dim 128, d_rel 16, extent 1024, K 4, window 1024 on a
ring of 1040 rows on SWA layers, a linear cache on full layers.

| kernel (op) | path | what |
|---|---|---|
| `ink_kv_prologue_bf16` (70) | `ink_prologue_k` | one block per (kv head, row), 8 warps. Each warp takes whole "jobs": the conv'd K row (clip), the conv'd V row (unit RMS), q of each of the 3 heads, r of each head. Lane-strided elements, butterfly sum of squares. Then all threads run the rel tail (3 x 1024 elements, a 16-long fma chain each). Conv-ring write in the kernel when T <= Rr - K + 1, else `ink_ring_write_k` runs after it. |
| `rel_sink_attn_bf16` (71), any T | `ink_tile_k<D, false>` + `ink_tile_comb_k`, per 16-row slice | tensor-core (mma.sync m16n8k16 bf16) split-KV over **absolute key tiles shared by every row of the call**: one block per (kv head, tile), so the K / V of a tile is read once for all rows. The M = T x 3 query rows go in m-tiles of 16, with 4 warps per m-tile, one 16-key group each, over 64-key sub-chunks staged with cp.async (double buffer). Rows with one tile finish in the block; the rest merge in tile order in the combine. |
| `ink_attn_decode_bf16` (75) | `ink_tile_k<D, true>` + comb | the same kernel. Each tile block computes the prologue's jobs it needs: q / r of every row, the bias of its sub-chunks (same rel chain), and the conv'd K / V of the call's own keys in its tile. These use the prologue's device functions, so the result is bitwise the pair. |
| `rel_sink_attn_bf16_flash` (71), T > 16, priority 5 | `ink_flash_k<D>` + `ink_kv_write_k` | flash prefill on tensor cores: one block per (kv head, 64 query rows), 12 warps (3 heads x 4 row groups of 16), one pass over the keys, no scratch; the bias loads are issued before the QK^T mma. The cache write runs after it. A separate kernel because it is not bitwise the rows-exact one, the same arrangement as Metal's `rel_sink_attn_bf16_nax` (priority 5, prefill only). |

`rel_sink_attn_bf16` runs a call of more than 16 rows as 16-row slices in order (each slice reads the cache rows the
earlier slices wrote), so it is rows-exact for every T, not only decode / verify. Prefill normally takes the flash kernel.

Diagnostics: `INTARSIA_INK_DECODE=splitkv` routes `rel_sink_attn_bf16` to the scalar split-KV kernel, a straight port of
Metal's `ink_attn_s1` (rows-exact itself, but not bitwise the tile kernel). It is kept for timing comparisons and is the
head_dim 8 path. `INTARSIA_INK_SPLIT_KEYS` sets the tile-size floor (rounded up to a multiple of 64).

### Rows-exact design (tile kernel)

Row t (position p, first visible key lo, vis = p - lo + 1) cuts the keys into absolute tiles of S keys. S = tile_s(vis) is
the smallest 64 x 2^k for which an interval of vis keys meets at most MAXT = 40 tiles. So S = 64 on every SWA visit (17
tiles), and on full layers S grows with the context (1024 at 32k, 33 tiles). The row's partial over tile tau depends on
(p, lo, S, tau) only:
- sub-chunks of 64 keys;
- group g of the m-tile takes keys 16g .. 16g + 15 of each;
- QK^T and PV run on mma with bf16 in and f32 accumulation, P is rounded to bf16, and l sums the rounded p;
- the four groups merge in the order 0, 1, 2, 3.

The combine merges tiles in index order. Whether a row has one tile (finished in the block) or several (combine) also
depends only on its position. Which other rows share the m-tile does not matter, because an mma output element depends only
on its own row of A. Rows of one call can fall into two tile-size classes when the call straddles a size change; the grid
then holds both classes' tiles. With a device position the grid is sized from capacity: window.min(Cr), or Cr. Extra blocks
exit at once.

## Admission (`intarsia kernels test --backend cuda --op ...`, fleetq #10799, commit 5a06044d5)

| op | kernel | cases | fail | max_abs (scaled) | rel_l2 | result |
|---|---|---|---|---|---|---|
| ink_kv_prologue | ink_kv_prologue_bf16 | 117 | 0 | 5.46e-3 | 4.35e-4 | ADMITTED |
| rel_sink_attn | rel_sink_attn_bf16 (tile T <= 16, flash T > 16) | 63 | 0 | 5.92e-3 | 2.24e-3 | ADMITTED |
| ink_attn_decode | ink_attn_decode_bf16 | 70 | 0 | 6.90e-3 | 2.46e-3 | ADMITTED |

The scalar port (commit a87c58fd3, fleetq #10754) was also ADMITTED. Its figures: 117 / 63 / 70 cases, rel_l2 4.3e-4 /
1.2e-4 / 1.0e-4. The tensor-core kernels' rel_l2 is about 20x higher because P is rounded to bf16. That is the usual
flash-attention trade, and it stays well inside BF16_TOL (1.5e-2 abs, 6e-3 rel). The sweep's f32-activation cases are not
registered: the kernels are bf16 only, as on Metal.

Bitwise test `crates/intarsia-backend-cuda/tests/inkling2_attn.rs`: 36 cases. Layers: SWA at pos0 0 / 1020 / 1030 / 2100
(ring wrap); full linear with dynamic taps and tau at 3 / 1018 / 1500 / 3000; full at 8180. Rows T 2 / 4 / 7 / 16. Each case
checks five runs against each other:
- the R-row pair;
- R one-row pairs with host positions;
- R one-row pairs with a device position;
- InkAttnDecode at R rows;
- InkAttnDecode at R x 1 row, with host and with device positions.

The test compares q / rel / k / v / o, both conv rings and both caches. Passes (fleetq #10799).

## Decode timing (fleetq #10963, final build, tile floor 256)

`crates/intarsia-backend-cuda/tests/inkling2_attn_bench.rs` (`INTARSIA_INK_BENCH=1`). The figures are microseconds per
visit, replayed from a captured CUDA graph (eager is 1-5 us higher). 16 visits rotate over their own operands so the
K / V never sits in L2. "KV" is the K + V bytes a visit reads; the floor is KV / 273 GB/s. The faster of pair and fused is
in bold.

| layer | ctx | rows | prologue | attn | pair | fused (op 75) | KV MB | floor us |
|---|---|---|---|---|---|---|---|---|
| SWA | 512 | 1 | 8.2 | 23.9 | 41.6 | **28.7** | 2.1 | 7.7 |
| SWA | 512 | 4 | 8.5 | 34.4 | 44.5 | **37.7** | 2.1 | 7.7 |
| SWA | 512 | 16 | 12.0 | 50.0 | 61.2 | **57.3** | 2.2 | 7.9 |
| SWA | 4k | 1 | 8.2 | 35.1 | 47.4 | **40.8** | 4.2 | 15.4 |
| SWA | 4k | 4 | 8.5 | 46.5 | 53.7 | **47.8** | 4.2 | 15.4 |
| SWA | 4k | 16 | 11.8 | 60.3 | 70.0 | **69.6** | 4.2 | 15.4 |
| full | 512 | 1 | 8.2 | 22.5 | 39.3 | **29.7** | 2.1 | 7.7 |
| full | 512 | 4 | 8.4 | 35.9 | 41.2 | **37.0** | 2.1 | 7.7 |
| full | 512 | 16 | 13.4 | 51.2 | 58.5 | **56.1** | 2.2 | 7.9 |
| full | 4k | 1 | 8.1 | 97.2 | 110.9 | **108.5** | 16.8 | 61.5 |
| full | 4k | 4 | 8.3 | 115.9 | **119.5** | 123.9 | 16.8 | 61.5 |
| full | 4k | 16 | 13.7 | 146.1 | **156.3** | 161.5 | 16.8 | 61.7 |
| full | 32k | 1 | 8.1 | 732.8 | 732.6 | **715.4** | 134.2 | 491.7 |
| full | 32k | 4 | 8.4 | 756.6 | 761.1 | **749.7** | 134.2 | 491.7 |
| full | 32k | 16 | 11.6 | 829.8 | **845.1** | 846.3 | 134.3 | 491.9 |

SWA visits read at most the 1024-key window, so SWA 32k is SWA 4k. On CUDA the fused op 75 is at least as fast as the pair
by 1-13 us at 1-4 rows (SWA 512, 1 row: 28.7 vs 41.6), and within 5 us elsewhere. The pair wins at full 4k with 4 / 16
rows (by 4-5 us, after the last prologue fix) and ties at full 32k with 16 rows. Approved rule, commit d536ff099: on CUDA,
fused for every visit of 1..16 rows.

Tile floor sweep (`INTARSIA_INK_SPLIT_KEYS`, fleetq #10866, before the prologue fix, so compare the attention and fused
columns only):

| case | attn 64 / 128 / 256 | fused 64 / 128 / 256 |
|---|---|---|
| SWA 512, 1 row | 18.9 / 20.0 / 23.6 | 34.2 / 31.6 / 31.0 |
| SWA 4k, 1 row | 36.0 / 35.5 / 32.7 | 52.1 / 53.5 / 40.2 |
| SWA 4k, 16 rows | 81.8 / 71.8 / 63.2 | 128.7 / 112.5 / 85.6 |
| full 4k, 16 rows | 178.4 / 175.3 / 145.4 | 226.2 / 218.2 / 173.7 |
| full 32k, 1 row | 750 / 735 / 745 | 761 / 758 / 745 |

256 is the default. Fewer, longer blocks mean fewer partials to merge, and the fused kernel redoes less per block.

### Launches per visit

| path | launches |
|---|---|
| pair, decode / verify (1..16 rows) | 3: `ink_prologue_k`, `ink_tile_k`, `ink_tile_comb_k` (2 when every row has one tile: host position, keys <= 256) |
| fused op 75 | 2: `ink_tile_k<.., true>`, `ink_tile_comb_k` (+ `ink_kv_write_k` / `ink_ring_write_k` only when T exceeds Cr - window + 1 or Rr - K + 1: never at the real shapes) |
| prefill chunk (2048 rows) | 4: `ink_prologue_k`, `ink_ring_write_k` (T > Rr - K + 1), `ink_flash_k`, `ink_kv_write_k` |
| `rel_sink_attn_bf16` at T > 16 (rows-exact path, not normally selected) | 2 per 16-row slice |

## Prefill timing (2048-row chunk, fleetq #10963)

| layer | ctx before | prologue | attn (flash) | pair | KV MB |
|---|---|---|---|---|---|
| SWA | 0 | 1224.9 | 1221.4 | 2450.1 | 4.2 |
| SWA | 2k | 1307.2 | 1581.3 | 2803.6 | 4.2 |
| SWA | 30k | 1205.5 | 1600.0 | 2837.0 | 4.2 |
| full | 0 | 1171.1 | 1429.2 | 2484.9 | 8.4 |
| full | 2k | 1203.3 | 2293.6 | 3640.1 | 16.8 |
| full | 30k | 1215.8 | 14282.0 | 15631.1 | 134.2 |

The contract's `rel [T,H,E]` costs 2048 x 24 x 1024 x 2 B = 100 MB per chunk. That is 0.37 ms to write plus 0.37 ms to read
at the bandwidth floor, and the bias fetch dominates SWA flash (every key is biased there, since window = E). The flash
kernel against the split-KV port at the same rows (fleetq #10804): 1.25 vs 10.8 ms (SWA ctx 0), 14.3 vs 1093 ms (full 30k).
Full 30k flash: 2048 x 24 x 32768 x 128 x 4 = 0.82 TFLOP in 14.3 ms, about 58 TFLOPS.

## Full-32k decode: what was tried (brief 2)

Pick: full 32k at 1 row (736 us against a 492 us floor) over full 4k at 4 rows (+9 us, verify only). The 32k case costs about
240 us above the floor on every full layer of every long-context token.

Hypothesis tested: wave quantisation (33 tiles x 8 heads = 264 blocks on 48 SMs) or one 70 KB-smem block per SM with
1-deep prefetch. Results, attention microseconds (graph), fleetq #10948:

| setting | full 4k, 1 row | full 32k, 1 row | full 32k, 16 rows |
|---|---|---|---|
| default (at most 40 tiles, 2 stages) | 94.0 | 716 | 842 |
| at most 64 tiles | 98.6 | 746 | 861 |
| at most 96 tiles | 99.7 | 709 | 838 |
| 1 stage (2 blocks / SM) | 96.4 | 704 | 808 |
| 1 stage, at most 96 tiles | 97.9 | 723 | 836 |

All within noise, so neither the tile count nor occupancy is the limit. The defaults stay; the knobs remain as diagnostics.
The kernels sit at about 180-190 GB/s on both the SWA (40 blocks) and the full (264 blocks) visits: about 6-8 us per
32 KB sub-chunk per block, far above DRAM latency. That points at the per-sub-chunk serial path of one block: stage, wait,
three `__syncthreads`, 16 keys of mma per warp. 8B bf16 GEMV decode reaches about 257 GB/s on this box, so about 1.4x is
left. ncu cannot run here (RmProfilingAdminOnly = 1), so the next step needs either profiling rights or the restructure in
idea 1 below.

## Full-32k decode: read pattern (brief 5, commit a5e3ede16)

Diagnostic `.scratch/bw.cu` (fleetq #11016, #11019-#11021): the 134 MB K + V of one full-32k visit
(8 kv heads x 32832 keys x 128 dims, bf16), read and discarded by a read-only kernel. There are 16 rotating copies (the
bench's footprint) unless noted.

| read pattern | us per visit | GB/s |
|---|---|---|
| (a) one linear stream, 768 x 256 threads | 534 | 252 |
| (b) the tile kernel's pattern: 4 warps round-robin over 16-key blocks, 16 B per lane | 637 | 211 |
| (b) with 1 copy (134 MB footprint) | 647 | 208 |
| (c) each block reads its contiguous K then V range, 128 threads | 570-579 | 232-236 |
| b2: (b), but each warp owns a contiguous quarter of the tile | 605 | 222 |
| b3: (b) with 4 16-key blocks in flight | 673-690 | 195-200 |
| b4: (b), but each load instruction covers 2 whole rows (not fragment layout) | 577 | 233 |
| b5: (b) with 32 B loads (`ld.global.v8.b32`) | 607 | 221 |
| b6: 32 B loads + warp-contiguous quarters (fragment-compatible) | 577 | 233 |
| b7: 32 B loads + 64 contiguous keys per warp per 256-key slab | 595 | 226 |

1 copy and 16 copies run at the same speed, so TLB and footprint are not the limit. The access shape is the limit: in
fragment layout, a 16 B load instruction touches 8 rows x 64 B. 32 B loads (8 rows x 128 B) plus long per-warp runs recover
most of the gap to (c).

The one rework attempt is b6 in the direct body (full layers):
- Warp group g owns the tile quarter `tk0 + g S/4 ..`, in key order, so the key-to-warp map stays absolute and the kernel
  stays rows-exact.
- With head_dim >= 64, W = 16 dims per lane and chunk. That is one 32 B load from the cache when both bases are 32-byte
  aligned (`F_AL32`), else two 16 B loads with the same values.
- The bias is staged per step of 4 x 64 keys, one 64-key run per warp.

Gate on fleetq #11036: all ADMITTED, 0 fails, and the 45 rows-exact cases bitwise. Fused op 75, graph us, against the
staged-SWA tip (#11016), one run each:

| case | 1 row | 4 rows | 16 rows |
|---|---|---|---|
| full 32k | 657 -> **613** | 721 -> **649** | 787 -> 784 |
| full 4k | 89.4 -> 87.9 | 104.7 -> 99.6 | 157.8 -> 164.8 |
| full 512 | 27.7 -> 28.2 | 37.1 -> 37.5 | 53.2 -> 56.8 |
| SWA (staged body, unchanged) | 28.5-38.2 -> 30.6-40.6 | | |

Full 32k at 1 row is now 613 us against a read-only ceiling for this pattern of about 577 (b6) and a floor of 492. The
16-row full-layer cases did not gain and may have lost 3-7 us. In those cases ptxas reports a 72-88 B spill for
`ink_tile_k<128>` at 168 registers, and the partial last tile now falls on one warp.

## Combine fold (idea 3, brief 5 step 2): tried, reverted

Prize, from nsys (`--cuda-graph-trace=node`, fleetq #11040; median per visit): `ink_tile_comb_k` takes 2.2-3.7 us at
ctx 512 / 4k with 1-16 rows, 8-10 us at full 32k with 1-4 rows, and 22 us at full 32k with 16 rows. The gap before it is
0.6-1.4 us.

Attempt: the last-arriving block of each (kv head, row) merged that row's G heads in tile order. The merge body was shared
with the kernel, so the bits were the same. Counters were a zero-initialised `__device__` array, reset by the last block.
Gate (fleetq #11054) with the fold on and off: all ADMITTED, and the 45 rows-exact cases bitwise. Fused us, graph, fold on
vs off (2 runs each):

| case | 1 row | 4 rows | 16 rows |
|---|---|---|---|
| SWA 512 | 31.6 vs 28.2 | 57.5 vs 37.9 | 90.3 vs 58.0 |
| full 4k | 95.9 vs 90.9 | 136.3 vs 102.9 | 228.8 vs 165.7 |
| full 32k | 622 vs 609 | 731 vs 655 | 919 vs 806 |

It loses everywhere. All rows of a call end on the same last tile, so a single block per kv head merges T x 3 heads x 128
dims serially from L2. The separate kernel spreads that work over 24 x T blocks. Reverted (the code is not kept). A fold
would need the merge spread over the tile blocks, which is a different design. A cheaper next step is to make
`ink_tile_comb_k` itself faster at 32k / 16 rows (22 us: one thread per dim walks 33 partials twice).

## Real-model decode profile (briefs 6 / 7, binary 8f7d5cd3a, fleetq #11087, spark1)

Greedy, 256 tokens, eager decode (no graph yet), `--profile-every 0`. Prompts of 1206 and 7874 tokens.

| ctx | decode tok/s | prefill tok/s | wall per token | GPU busy | idle gaps | kernels per token | mean gap |
|---|---|---|---|---|---|---|---|
| 1.2k | 30.1 | 938 | 33.5 ms | 32.0 ms | 1.46 ms (4.4 %) | 657 | 2.2 us |
| 7.9k | 29.1 | 1202 | 34.6 ms | 33.1 ms | 1.47 ms (4.3 %) | 657 | 2.2 us |

So the eager launch overhead is about 1.5 ms per token. A graph removes at most that, about 30 -> 31.5 tok/s, plus
whatever the CPU side hides.

Ops per token at 7.9k ctx (nsys, decode window). The floor is bytes / 273 GB/s. There are 48 visits per token (24 layers,
looped twice), 40 SWA and 8 full.

| rank | op (kernels) | ms / token | bytes / token | floor ms | % of floor |
|---|---|---|---|---|---|
| 1 | MoE experts (`ink_moe_gu` + `ink_moe_dn`, top-8 of 128, NVFP4) | 9.35 | 1.72 GB | 6.30 | 67 % |
| 2 | `ink_in_proj` (`ink_mma_gemv`, 135 / 136 blocks: q, gate, r, k, v, sink) | 8.71 | 1.90 GB | 6.96 | 80 % |
| 3 | logits: unembed (`gemv`, 25008 blocks) 3.75 + 48 small `ink_mma_gemv` (1 / 3 blocks, 40 us each) 1.96 | 5.71 | 0.93 GB | 3.40 | 60 % |
| 4 | attention (op 75 tile + combine) | 3.04 (1.2k: 2.06) | 0.44 GB (1.2k: 0.20) | 1.60 (0.72) | 53 % |
| 5 | `matmul` (wo, `gemv` 288 blocks) | 2.83 | 0.68 GB | 2.49 | 88 % |
| 6 | `ink_row` (conv / res mix, one block, 124 / token) | 2.10 | ~0 | ~0 | latency-bound |

At 1.2k ctx the order is the same, with attention fifth. The byte floor of a token is about 5.7 GB, i.e. 20.9 ms or
48 tok/s. About 4.5 ms per token goes to single-block or few-block kernels that move almost no bytes: `ink_row` 2.1, the
small `ink_mma_gemv` 2.0, `moe_route` 0.4. That is three times the launch gaps.

## CUDA B=1 graph decode (brief 8, branch `ink/cuda-graph` 5fa5a88bd on 8f7d5cd3a)

Change, in shared core files and tests only:
- `ink_pass` takes `dpos: Option<&TensorRef>`. Its positioned calls read the device position (`call.pos`): both
  ConvMix, InkKvPrologue, RelSinkAttn and InkAttnDecode (`inkling2::ops::POSITIONED`; SconvRes is not called there).
- `model.rs` passes `dpos` instead of refusing.
- With `dpos = None`, `exec_pos` is `exec`, so Metal and cpu behave identically.
- The plan printed no "decode graph disabled" warning, so no T = 1 kernel choice depends on kv_len.
- New test `inkling2_tiny_graph_decode_is_bitwise_the_eager_decode`.

Gates (spark1, fleetq #11097 and #11106; the checkpoint copied from spark2):

| gate | result |
|---|---|
| tiny, all 12 inkling2 tests, CUDA | ok; graph == eager bitwise over 31 decode steps, 1 capture, 31 replays |
| tiny, all 12 inkling2 tests, cpu (G1a / G1b) | ok (run on spark1, not spark2; both are GB10 aarch64) |
| greedy 64 ("The capital of France is") | ids / text md5 equal: eager, graph, graph again (63 replays) |
| greedy 256 forced | ids / text md5 equal: eager, graph, graph again (255 replays) |
| G2 decode short, graphs on | KL21 3.729e-3 (eager at #11079: 3.729e-3; budget 4e-3), PASS |
| G2 decode long, graphs on | KL21 3.145e-3 (eager 3.145e-3; budget 6e-3), PASS |

Decode tok/s, 256 forced tokens, `--profile-every 0`:

| prompt | eager | graph |
|---|---|---|
| short (g256) | 30.54 | 31.54 |
| 1206 tokens | 29.88 | 31.20 |
| 5039 tokens | 29.33 | 30.50 |

nsys of the graph step at 1.2k ctx: wall 32.5 ms per token, GPU busy 32.1 ms. Idle gaps are 0.39 ms per token (from
1.46 ms eager), with a mean gap of 0.59 us over 657 kernels. The graph took out what launch overhead there was. What is
left is kernel time. The ranking is unchanged: `ink_in_proj`, then MoE gu / dn, unembed, wo, then the latency-bound
`ink_row` (2.07 ms) and the small 1- / 3-block `ink_mma_gemv` calls (1.94 ms).

## P12 on CUDA (briefs 9 / 10, branch `ink/cuda-p12` fc773d0fc)

New files: `kernels/cuda/inkling2_p12.cu`, `kernels/cuda/ik_cuda_abi_inkp12.h`, `crates/intarsia-backend-cuda/src/inkp12.rs`,
`crates/intarsia-backend-cuda/tests/p12.rs`. Each kernel decodes the 12-bit codes to the exact bf16 words in registers (row
base, then escapes patched through a per-lane cursor) and runs its bf16 twin's arithmetic in that twin's order:
- `matmul_p12_gemv` / `matmul_p12_logits_gemv` [`_rows`] mirror `gemv_body`;
- `ink_in_proj_p12_mma` [`_rows`] mirrors `ink_mma_gemv_kernel`.

They register only where the bf16 call runs that twin. Ordinary GEMV is 1..4 rows, the rows twins and in_proj are 1..16. An
ordinary 5..16-row head therefore keeps bf16 (cuBLAS), and P12 on stays bitwise off.

Coverage per tensor class (`p12::pack_in_proj`), nsys of the decode step at 1.2k, µs per call:

| weight | bf16 | P12 | P12 GB/s (packed bytes) |
|---|---|---|---|
| in_proj (135-block grid) | 182 | 197 (194 before the fix) | 154 |
| wo | 59 | 48.7 | 223 |
| unembed | 3756 | 2887 | 241 |

The in_proj P12 kernel is decode-ALU bound. Holding the next escape in a register and prefetching raw codes before the mma did
not recover it. On CUDA, in_proj stays bf16 (the kernel stays registered; `INTARSIA_INK_P12_IN_PROJ=1` packs it), and wo and
the unembed are packed. Default-on on CUDA is commit 0cb867844 (`INTARSIA_INK_PACK12=0` reverts it).

Final gates, fleetq #11169 / #11171 / #11172 (default-on vs `=0`, graphs on):

| gate | result |
|---|---|
| CUDA bitwise / rows-exact (`tests/p12.rs`, real shapes + adversarial, rows 1..16) | 4 / 4 ok |
| tiny, 13 inkling2 tests on CUDA, P12 forced on and off (forced-on: packed logits == bf16 logits) | 13 / 13 both |
| core `quant::p12` unit tests (new default) | 4 / 4 ok |
| greedy 64 / 256, ids + text md5 | identical on vs off (ec9a19c14b2d / a0e6f47bc45f) |
| G2 decode short / long | 3.729e-3 / 3.145e-3 on and off, PASS |

Decode tok/s, 256 forced tokens, 2 interleaved runs each:

| ctx | off | on | gain |
|---|---|---|---|
| short | 31.58 / 31.56 | 33.00 / 32.94 | +4.4 % |
| 1.2k | 31.20 / 31.08 | 32.52 / 32.43 | +4.3 % |
| 5k | 30.45 / 30.46 | 31.89 / 31.74 | +4.5 % |

nsys on, 1.2k: 31.3 ms wall per token (graph decode with P12 off was 32.5).

## Decode step at 1.2k with P12 on (brief 11; nsys #11172, graph decode, 31.27 ms wall / 30.89 ms GPU per token)

Kernels grouped by role (kernel, grid). GB/s is weight or KV bytes over kernel time; the measured read ceiling of this box
is about 252 GB/s (one linear stream).

| # | kernel (role) | us / step | % of step | launches | us each | GB/s | owner |
|---|---|---|---|---|---|---|---|
| 1 | `ink_mma_gemv` 135 / 136 blocks (in_proj, bf16) | 8930 | 28.6 | 48 | 186 | 214 | mixmoe (P12 twin: this lane) |
| 2 | `ink_moe_gu` (experts gate / up) | 6026 | 19.3 | 48 | 125.5 | - | mixmoe |
| 3 | `ink_moe_dn` (experts down) | 3261 | 10.4 | 48 | 67.9 | - | mixmoe |
| 4 | `p12_gemv` 25008 blocks (unembed, P12) | 2887 | 9.2 | 1 | 2887 | 240 | this lane |
| 5 | `p12_gemv` 288 blocks (wo, P12) | 2338 | 7.5 | 48 | 48.7 | 218 | this lane |
| 6 | `ink_row` (conv / res mix, 1 block) | 2074 | 6.6 | 124 | 16.7 | ~0 | mixmoe |
| 7 | `ink_mma_gemv` 1 / 3 blocks (blend controller h) | 1946 | 6.2 | 48 | 40.5 | ~0 | mixmoe (deep kernel on its tip) |
| 8 | `ink_tile_staged_k` (fused SWA attention, 8 x 6 blocks) | 1643 | 5.3 | 40 | 41.1 | 102 | this lane |
| 9 | `ink_blend_gate` | 480 | 1.5 | 48 | 10.0 | - | mixmoe |
| 10 | `moe_route` | 368 | 1.2 | 48 | 7.7 | - | mixmoe |

Then come `ink_tile_k` (full attention, 331 us: 8 x 41.4), `ink_blend_norm` (248), the small `gemv` (220) and
`ink_tile_comb_k` (111).

### Brief 11 pick 1: P12 in_proj prefetch (timeboxed, stopped)

Both variants stayed bitwise (the CUDA P12 test passed 4 / 4 each time). Neither got P12 in_proj below bf16 (186 us per
visit). nsys of the 1.2k decode step with `INTARSIA_INK_P12_IN_PROJ=1`, us per visit:

| variant | P12 in_proj | fleetq |
|---|---|---|
| before (1-block prefetch, decode after the loads land) | 197 | #11157 |
| v1: raw codes + activations two 64-k blocks ahead | 194.6 | #11192 |
| v2: v1 + one 16 B plane-A and one 8 B plane-B load per chunk pair (bf16's load count per step) | 196.7 | #11196 |

Neither more bytes in flight nor fewer load instructions moved it. So the limit is not load issue or memory-level
parallelism as I had guessed. Without ncu (admin only) I can't see what it is. Both variants are reverted (0212b65ff,
c7b7e1b40; the branch content equals fc773d0fc). On CUDA, in_proj stays bf16. Per the timebox I moved to the SWA fallback.

### Brief 11 fallback: more SWA tiles (measured with the existing knob, no code change)

The `INTARSIA_INK_SPLIT_KEYS` tile floor (default 256) at 128 and 64, decode bench, 2 interleaved runs (fleetq on spark1,
`.scratch/split1.log`). Fused op 75, graph us, run 1 / run 2:

| case | floor 256 | floor 128 | floor 64 |
|---|---|---|---|
| SWA 4k (window 1024), 1 row | 42.6 / 42.5 | 53.7 / 53.1 | 55.3 / 56.4 |
| SWA 4k, 4 rows | 53.9 / 54.1 | 70.1 / 68.5 | 71.2 / 70.5 |
| SWA 512, 1 row | 30.8 / 34.3 | 36.0 / 34.1 | 34.5 / 34.9 |
| full 4k, 1 row | 95.8 / 96.5 | 100.8 / 101.5 | 101.1 / 101.5 |
| full 32k, 1 row | 639.9 / 620.2 | 614.5 / 627.9 | 628.7 / 616.7 |

More tiles are slower on the fused SWA path. The attention-only kernel at SWA 4k barely moves either (37.2 -> 39.6 /
39.0). So the SWA visit (4.2 MB of K / V in about 41 us) is not short of blocks. It is fixed per-block and per-visit
cost: q load, prologue jobs, partial write, combine. That needs a different SWA design (one block per head that walks
the whole window, no combine), not a knob. Brief 11 therefore ends without a code change in this lane: the in_proj
variants are reverted and the tile floor stays 256. The remaining levers in the ranking are mixmoe's: MoE gu / dn
(29.7 %), `ink_row` (6.6 %) and the 1- / 3-block blend GEMV (6.2 %).

## Brief 13 proposal (no measurement; sent to the coordinator)

| rank | candidate | estimate | basis | overlap with mixmoe |
|---|---|---|---|---|
| 1 | DSpark speculative verify on CUDA | 1.2-1.4x decode, about 38-45 tok/s | Metal ink/mtp: 1.26-1.44x (#10991), 72-77 % acceptance, 2.54 tokens per pass, +2.9 ms per verify row on a 16 ms step; CUDA verify ops are already rows-exact 2..16 | MoE speed at 2..4 rows decides the gain; the draft's MoE / mix ops |
| 2 | bf16 in_proj deeper prefetch, bitwise | 186 -> about 160 us per visit, -1.2 ms per step, +4 % | the unembed GEMV reaches 240-245 GB/s; risk: the same change did nothing for the P12 twin | total (ink_proj.cu is theirs) |
| - | prefill vs vLLM | not ranked for this lane | CUDA 1159 tok/s at 6.7k vs vLLM 2772-4018 marginal; attention is only about 6-9 % of our prefill | the gap is MoE / in_proj prefill |

Preconditions for 1: ink/mtp is not merged; model.rs refuses drafters on Inkling; the draft's ops (Metal discriminants
92-95) need CUDA kernels; and the draft needs device positions to run graphed. cvat (PR #464) covers only the generic
AttnDecode.

## DSpark draft on CUDA, milestone 1 (brief 14, branch `ink/cuda-mtp` b37737f77)

The draft dispatches only ops that already have CUDA kernels: RmsNorm, Logits, Add, AttnDecode, Embedding, HcPre, QkNorm,
Rope, SiluGlu, plus `Backend::topk_rows`. No new op. Changes:
- `load.rs` admits the DSpark draft on cuda;
- `kernels/cuda/topk.cu` takes bf16 rows (the draft's base logits; it was f32 only, so the draft fell back to a host top-k);
- `INTARSIA_DSPARK_DUMP=1` dumps each draft step's top-64 log-probabilities (diagnostic, `docs/tuning.md` row);
- `INTARSIA_INKLING2_BACKEND=cuda` for the DSpark tiny tests.

| gate (fleetq #11220 / #11241) | result |
|---|---|
| device top-k, f32 and new bf16 rows vs host order (`row_tables.rs`) | ok; no host fallback in the runs below |
| tiny DSpark tests (`inkling2_dspark`), CUDA and cpu | 3 / 3 each |
| real checkpoint, 1206-token prompt, 64 tokens, k3, gate off: acceptance | CUDA 47.4 % (2.42 tokens per pass), CPU 47.4 % (2.38) |
| target greedy ids CUDA vs CPU | identical through generated token 28, then diverge (the target is not bitwise across backends) |
| draft steps matched by (position, step, previous token), all before the divergence | 33 |
| draft top-1 agreement | 32 / 33 (steps 0 / 1 / 2: 11 / 11 / 10 of 11) |
| KL(CPU || CUDA) over the top-64 terms | median 4.5e-3, mean 7.1e-2, max 0.76 (four rows above 0.3, all with the same top-1) |

The outliers have the same top-1 and p1 shifted by 0.04-0.22. The draft reads the target's tap hidden states, which are not
bitwise across backends, so this is expected. I have not separated the draft's own error from that input difference.

CUDA spec decode in that run: 43.97 tok/s over 64 tokens at 1.2k context (eager, gate off), against 32.5 for graph decode
without spec. That is one run, not a gated number.

## DSpark on CUDA, milestone 2 (brief 16, branch `ink/cuda-mtp` b1d84ba5b: ink/mtp + fc773d0fc + mixmoe c6680633c)

| gate (fleetq #11256, #11260, #11273) | result |
|---|---|
| verify rows R = 2 / 3 / 4 == 1-row decode, bitwise after the final softcap (tiny, 4 positions) | CUDA ok, cpu ok |
| tiny DSpark tests | 3 / 3 CUDA, 3 / 3 cpu; device top-k on, no host fallback |
| greedy 64, spec k3 tau 0.8 vs off: ids / text md5 | ec9a19c14b2d / 9914eb30d762 both; acceptance 44.4 %, 2.14 tokens per pass |
| greedy 256 forced | a0e6f47bc45f / bd060e7724b6 both; 46.2 %, 2.11 |
| 1.2k prompt, 256 forced | bf582c11875a / 91582acf6984 both; 77.2 %, 2.80 |
| sampled t1.0 with spec on | 0 verify passes, 0 proposed (sampling never reaches the verify pass) |
| serve smoke 1k + 9k | PASS after mixmoe 24ab7961d (below). First try: serve refused to load, with or without `--spec`: admitted-only load, `moe_experts: 12 shape(s) without a kernel` (m=1 n=1152 k=2304 act bf16 w u8). Server group stopped cleanly, 0 processes left |

The verify rows come back before the final logit softcap (30), while a decode step caps them on readback (`Model::softcap_logits`).
Greedy is unaffected. A sampled verify reading those rows would draw from uncapped logits; that is in the mtp lane's
`verify_head`, sent there by the coordinator.

Decode tok/s, single runs (spec on runs eager, spec off graphed): 64 tokens 34.3 -> 39.0, 256 tokens 34.1 -> 42.1, 1.2k prompt
33.6 -> 51.0.

**Serve after mixmoe 24ab7961d** (`ink/cuda-mtp` 7f2c87acd, build #11317 ok, servediag #11321; serve --backend cuda
--mem-budget 30 --max-seq 12288 --max-batch 1, admitted-only): spec off and `--spec dflash --spec-k 3` both load with no
"no kernel" line, both answer `/v1/models` with `inkling-2-mini`, and both stop with port 8096 closed, 0 serve processes
left. Serve smoke with spec (fleetq #11323, same flags + `--spec dflash --spec-k 3`): 1k prompt (1206 tokens) HTTP 200 in 2.7 s,
9k prompt (9198 tokens) HTTP 200 in 8.6 s, 64 completion tokens each; metrics `spec_steps 38, proposed 92, accepted 69`
(75 %); port closed, 0 serve processes left. The job exited 141: SIGPIPE from the script's final `grep | head` log scan
under `pipefail`, after every check had printed. It is not a serve failure. The Metal half of 24ab7961d is not verified here
(endor #11301).

Earlier (before 24ab7961d): mixmoe's admission fix 65daa3e7a is in this tree (merged at c6680633c), and serve still refuses with the same message
with spec off too. So it is not a spec issue. My assumption that 65daa3e7a was the missing piece is wrong. Two runs failed the
same way, so I stopped here and reported. Not fixed.

## DSpark on CUDA: localeval speed (briefs 19-20)

Setup: fleetq #11325, spark1, `--quiet gpu,cpu`. Branch `ink/cuda-mtp` 7f2c87acd. `serve --backend cuda --mem-budget 30
--max-seq 12288 --max-batch 1`. localeval 0.4.0 `speed`, temperature 0, 128 forced tokens, 3 reps + 1 warmup per cell, a
nonce first in each prompt. Arms in order A1 B2 A3 B4 C5:
- graph: spec off, CUDA graphs on (the serving default).
- spec: `--spec dflash --spec-k 3`. Its real path is eager; spec under graphs is milestone 3.
- eager: spec off, `--no-graphs`. One arm, to separate the spec gain from the graph gain.

Acceptance comes from `/metrics` deltas over each cell: `accepted_tokens / proposed_tokens`, warmup call included.
Values are median (min-max) over 3 reps. localeval rejected no cell (`rejected` null on all 15).

| ctx (prompt tok) | arm (mode) | decode tok/s | prefill tok/s | TTFT s | decode spread | acceptance |
|---|---|---:|---:|---:|---:|---:|
| 1k (1034-1038) | graph A1 | 34.2 (33.8-34.5) | 1224 | 0.84 | 2.1 % | - |
| | graph A3 | 34.3 (34.2-34.4) | 1201 | 0.86 | 0.6 % | - |
| | spec B2 (eager) | 41.9 (40.0-46.1) | 1223 | 0.85 | 14.5 % | 51.0 % (171/335) |
| | spec B4 (eager) | 40.8 (39.7-48.2) | 1236 | 0.84 | 20.9 % | 50.4 % (173/343) |
| | eager C5 | 32.9 (32.9-33.2) | 1210 | 0.86 | 0.9 % | - |
| 4k (4124-4127) | graph A1 | 33.5 (33.4-33.5) | 1380 | 2.99 | 0.3 % | - |
| | graph A3 | 33.3 (33.1-33.8) | 1357 | 3.04 | 2.1 % | - |
| | spec B2 (eager) | 47.3 (44.9-57.9) | 1385 | 2.98 | 27.5 % | 63.3 % (209/330) |
| | spec B4 (eager) | 49.6 (44.8-49.7) | 1381 | 2.99 | 9.9 % | 63.1 % (198/314) |
| | eager C5 | 32.3 (31.9-32.3) | 1381 | 2.98 | 1.2 % | - |
| 9k (9251-9274) | graph A1 | 32.6 (32.1-32.6) | 1449 | 6.39 | 1.5 % | - |
| | graph A3 | 32.7 (32.2-33.0) | 1452 | 6.37 | 2.6 % | - |
| | spec B2 (eager) | 49.2 (44.4-55.7) | 1448 | 6.39 | 23.0 % | 69.0 % (214/310) |
| | spec B4 (eager) | 45.2 (44.4-51.8) | 1442 | 6.43 | 16.3 % | 62.5 % (207/331) |
| | eager C5 | 31.3 (30.7-31.7) | 1448 | 6.39 | 3.0 % | - |

Reading, as the mean of each pair:
- **Spec vs graph:** spec on beats graphed spec off by +21 % at 1k (41.4 vs 34.3), +45 % at 4k (48.5 vs 33.4) and +44 % at
  9k (47.2 vs 32.7).
- **Spec gain alone:** against the eager spec-off arm the gain is +26 / +50 / +51 %.
- **Graph gain alone:** graphs add only +4 % (1k 34.3 vs 32.9, 4k 33.4 vs 32.3, 9k 32.7 vs 31.3).
- **What spec under graphs could add:** about that 4 %, if the graph gain carries over.
- **Prefill and TTFT:** the same in every arm. Spec costs no prefill.
- **Spread:** the spec arms spread 10-28 % because acceptance varies with each rep's random text. The graph and eager arms
  stay within 3 %.

Against the brief's figures (vLLM 37-43 tok/s decode on this box; I did not take vLLM down or re-measure it), and the ~50
tok/s native-weights floor:
- **Spec off (32.6-34.3):** below vLLM.
- **Spec on at 1k (~41):** inside vLLM's range.
- **Spec on at 4k / 9k (45-50):** above vLLM's range and at the floor. The best pair mean is 48.5 at 4k.

I did not check which vLLM configuration the 37-43 figure came from. The spark2 vLLM recipe in unanimis runs dspark with 7
draft tokens.

Raw data:
- localeval runs: `/opt/code/deepeval/results/20261122-063325-adhoc-ink-cuda-graph-a1-graph-1024` through
  `20261122-064403-adhoc-ink-cuda-eager-a5-eager-9216` (15 runs).
- unanimis records, cell order: 2a49f575, 0fe3814c, 789cdb9a, 0181ab13, bf451f7c, ce15dc51, f15d2500, 80a2721e, f972d2ed,
  2d609c2b, 9befad89, f3a3edba, 477d88ec, b3edca5d, d0960414.
- Scripts: `ink-cuda-mtp/.scratch/{speedrun.sh,arm.sh}`.
- After the run spark1 had 0 serve processes and port 8096 closed. The script's "2 left" line was `pgrep -f` matching its
  own ssh command line.

## Chunk 2048 merge and the graphed decode gap profile (brief 21)

**Merge** e0a83ecb8 into `ink/cuda-mtp`: 64fbf0ecf. Build #11342 ok. md5 gate (fleetq #11349, same job as the profile):
- **64 tokens:** ec9a19c14b2d / 9914eb30d762, unchanged; spec on == off, 44.4 %.
- **256 tokens:** a0e6f47bc45f / bd060e7724b6, unchanged; spec on == off, 46.2 %.
- **1.2k prompt:** changed to 32ec7a414c3a / 56903b843b73 (was bf582c11875a / 91582acf6984), the same in spec off and on;
  81.1 % acceptance, 53.3 tok/s with spec on vs 33.5 off. The 1206-token prompt is now one prefill chunk instead of several.
  That is the expected sum-order change of a prefill chunk; spec exactness is unaffected. The ink-cuda-mixmoe lane's G2 / G3
  accuracy gates cover this change (their doc, "Prefill chunk sweep").

**Profile** (graphed, spec off): `generate` with a 4116-token prompt, 128 tokens, `nsys --cuda-graph-trace=node`. 32.75 tok/s
under nsys, 33.17 tok/s plain. Over 123 tokens (first 4 skipped): 30.45 ms wall, 30.06 ms GPU busy, 657 kernels per token.
The GPU is never idle inside the graph, so the whole gap is kernel time.

Visits per token: 48. Layers 0-5 and 18-23 run once; layers 6-17 run 3 passes. 40 visits are SWA (window 1024) and 8 are
full: layers 5, 11, 17, 23.

How bytes are counted, per token:
- in_proj is bf16 (`ink_mma_gemv`; P12 lost here in brief 11), about 8600 rows x 2304, 39.6 MB a visit.
- wo is P12, 10.6 MB a visit. Unembed is P12, 0.692 GB.
- Experts: top 8 of 128. NVFP4 expert 4.48 MB with scales (44 visits); MXFP8 expert 8.21 MB (layers 20-23, 4 visits).
- KV is bf16, 8 heads x 128 dims, K + V: 4.19 MB per SWA visit and 17.1 MB per full visit at 4.1k. This assumes each pass
  keeps its own cache.
- Controller bytes are an estimate from the kernel grids.

| class | kernels | ms / token | MB / token | GB/s (of 273) | floor ms | gap ms | owner |
|---|---|---:|---:|---:|---:|---:|---|
| attention projections | `ink_mma_gemv` 48 (in_proj, 184 us), `p12_gemv` grid 288 48 (wo, 49 us) | 11.18 | 2413 | 216 (79 %) | 8.84 | **2.34** | in_proj: mixmoe kernel, P12 in_proj: mine (brief 11) |
| attention core | `ink_tile_staged_k` 40 (SWA, 37 us, grid 8x6), `ink_tile_k` 8 (full, 90 us), `ink_tile_comb_k` 48 | 2.33 | 305 | 131 (48 %) | 1.12 | **1.21** | mine |
| MoE experts | `ink_moe_gu` 48 (126 us), `ink_moe_dn` 48 (68 us) | 9.35 | 1839 | 197 (72 %) | 6.74 | **2.61** | mixmoe |
| router | `gemv` grid 16 48 (4.6 us), `moe_route` 48 (7.6 us, one warp) | 0.59 | 28 | 48 | 0.10 | 0.48 | mixmoe |
| unembed | `p12_gemv` grid 25008 x1 | 2.93 | 692 | 236 (87 %) | 2.53 | 0.39 | - |
| mix / conv | `ink_row_kernel` 124 (16.9 us, **grid 1 block 768**: ConvMix 96, AttnResMix 26, ...) | 2.09 | < 30 | - | ~0.1 | **~2.0** | mixmoe |
| loop controllers / blend | `ink_mma_gemv_deep` 2x24 (17-19 us, grids 3 and 1), `ink_blend_gate` 2x24, `ink_blend_norm` 2x24 | 1.58 | ~100 (est.) | ~63 | ~0.37 | ~1.2 | mixmoe |
| rest | rmsnorm, embedding, argmax | 0.02 | - | - | - | - | - |
| **total** | | **30.06** | **~5400** | 180 | **19.8 (50.5 tok/s)** | **10.3** | |

Top 3 gaps. The ceiling is the decode rate with that gap closed to its floor, from 33.3 tok/s:
1. **MoE experts, 2.6 ms; ceiling 36.4 tok/s** (mixmoe). `gu` runs at 189 GB/s, `dn` at 175 GB/s. At 85 % of peak the
   class would take 7.9 ms (-1.4 ms).
2. **Attention projections, 2.3 ms; ceiling 36.0 tok/s.** in_proj reads bf16 at 215 GB/s, already 79 %. Its floor with P12
   bytes would be 5.2 ms instead of 7.0, but brief 11's P12 in_proj was compute-bound (197 us vs 186 us bf16). The bytes are
   the gap, not the bandwidth: an in_proj P12 or fp8 form that runs at bf16 speed is worth up to 1.8 ms.
3. **mix / conv row kernels, 2.0 ms; ceiling 35.7 tok/s** (mixmoe). 124 launches a token, each one block on one SM for
   17 us, on a few KB of data. They are latency-bound; split over D across blocks they should take 2-3 us each.

Next after those:
- Attention core, 1.2 ms (mine). The SWA tile kernel launches only 48 blocks (8 kv heads x 6 tiles) and reads 4.2 MB in
  37 us, at 113 GB/s. Brief 11's smaller key tiles were slower, so the fix is more heads or queries per block, not more
  splits.
- Controllers / blend, ~1.2 ms. These are tiny GEMVs on 1-3 blocks.

Spec under graphs (milestone 3) recovers the eager-to-graph 4 % on the spec arm, not this gap.
Raw: spark1 `~/intarsiaml-ink-cuda-mtp/out/prof/{g4k.nsys-rep,cls.txt,wbytes.txt}`; scripts `ink-cuda-mtp/.scratch/
{prof4k.sh,cls.py,wbytes.py}`.

## Milestone 3: spec under graphs (brief 21 step 3)

What is graphed: the **verify pass**, which is the target model's q = k + 1 rows. The existing captured verify-only pass
(`Model::verify_replay`, `INTARSIA_VERIFY_GRAPH=1`, until now HIP only) is opened to CUDA. The DSpark draft still runs eagerly.
The default stays **off**; flipping it is the coordinator's call.

Code on `ink/cuda-mtp`:
- 6e102ef35: `verify_graph_ok` allows `cuda` next to `hip`.
- 33863d8b9: Inkling-2 passes the per-row check. The check is about the generic AttnPrefill vs per-row AttnDecode choice,
  which Inkling never runs: its verify attention is the rows-exact `InkAttnDecode`, which reads the device position
  (`inkling2::ops::POSITIONED`). On CUDA the flag is set only when an MTP head is loaded, so the first change alone never
  replayed: 0 replays, speed equal to eager.
- Test commits eaa6322ff, d3143a77b, db0f02be9, 9510d41a3: the new test `inkling2_tiny_verify_graph_replays_equal_decode_bitwise`.
- Tuning row in `docs/tuning.md`: 01d5f1fa1.

Before editing I checked four things (advisor):
- the DSpark verify goes through `forward_verify_run` (`engine::spec_verify`);
- tap collection is device-only (`ink_copy` into the fixed `feat` buffers, `inkling2.rs:904-931`);
- the positioned ops read `ws.pos` through `dpos`;
- every pass of the real runs replays (graph replays above the eager baseline equal the verify pass count).

| gate (`INTARSIA_VERIFY_GRAPH=1`) | result |
|---|---|
| verify rows R = 2 / 3 / 4 == 1-row decode, fresh state per case (#11392) | CUDA ok, cpu ok |
| new test: one state, R-row passes back to back from pos 5, ONE recording per R replayed at every position, == eager 1-row decode bitwise (#11403) | ok: R = 2 17 passes, R = 3 11, R = 4 8; 1 capture and every pass replayed for each R |
| greedy md5 spec on (graphed verify) vs off (#11389) | 64 ec9a19c14b2d, 256 a0e6f47bc45f, 1.2k 32ec7a414c3a: identical; replays over the eager baseline +21 / +108 / +74 = the verify passes; 0 capture failures |
| serve smoke 1k + 9k with the env (#11392) | HTTP 200 (2.0 s, 4.7 s), spec steps 39, proposed 96, accepted 67; port closed, 0 processes left |

The first two runs of the new test were wrong: the tests load with `profile_every` 1 (a profiled pass runs eagerly), and
`set_graphs(false)` drops the recordings. Both passed bitwise on the eager fallback with 0 replays. The test now asserts the
capture and replay counts, so that cannot pass silently again.

A/B with `generate`, spec on, 256 forced tokens, fixed prompts, `INTARSIA_VERIFY_GRAPH` 0 / 1 alternating, 2 pairs (#11392):

| prompt | eager verify tok/s (pair 1, 2) | graphed verify tok/s (pair 1, 2) | gain | acceptance |
|---|---:|---:|---:|---:|
| hash map (P2) | 42.16, 42.21 | 42.82, 42.95 | +1.7 % | 46.2 % |
| 1k (1206 tok) | 53.98, 53.75 | 54.92, 54.91 | +1.9 % | 81.1 % |
| 4k (4116 tok) | 50.62, 50.98 | 51.94, 51.85 | +2.2 % | 73.6 % |

Ids are identical between the arms in every pair, and pairs agree within 0.4 %. The gain is half of the 4 % the graph
saves on a plain decode step, because the DSpark draft pass is still eager. Graphing the draft too is the rest: the
draft's host steps already skip under the recorder (`dflash.rs:788-863`); it needs its own capture keyed like the verify
pass.

Not yet re-measured with localeval; the spec arm's 10-28 % spread there would hide a 2 % change.

## Draft graph: measured ceiling 0.7 %, not built (brief 22)

Measured before building. Spec-on decode, verify graph on, 1k prompt, 256 tokens, k3: 54.7 tok/s, 74 verify passes,
81 % acceptance (fleetq #11404).

- **`INTARSIA_DSPARK_PROFILE=1` (synced phases, per draft, median of 74):** block 2.71 ms, head 1.10 ms, first Markov step
  0.35 ms, ingest 4.00 ms; cycle 52.9 ms. These synced numbers overstate launch cost: each sync makes the queue drain.
- **nsys `--cuda-graph-trace=node`, same run:** the GPU is busy **97 %** of the decode window (4575 of 4735 ms). Idle is
  159.5 ms:
  - 117 ms in 118 gaps over 100 us: host syncs (the Markov loop's downloads, the verify readback);
  - 37 ms in under-5 us gaps, graphed and eager alike.
- **Idle right before the eager kernels**, the most a draft graph could remove: 33.3 ms of launch-size gaps (13612) and
  53.6 ms of host-sync gaps (18, which a graph does not touch). **Ceiling 0.70 %** of the window.
- **The eager draft is GPU work, not launch overhead:** 421 ms of GPU time (8.9 % of the window).
  - `gemv_kernel` 233 ms: 29 calls a draft at 80 us. These are the bf16 draft weights at 8 rows: q / k / v / o /
    gate / up / down x 3 layers, plus the ingest's k / v.
  - MXFP4 head `mxfp4_gemv_mr` 110 ms (1.10 ms a draft).
  - `topk` 26 ms, `splitkv` attention 27 ms.

So the draft graph is not built. It would need its own capture keyed by sequence and window phase, with device positions
for the draft's RoPE / KvAppend / AttnDecode and the f_rnn ring slot, for under 0.7 %. What would actually help, measured:
1. **Draft GEMV bytes:** 233 ms is 4.9 % of the window. P12 (lossless, 0.75x bytes) or fp8 draft weights would save 1.2-2.5 %.
2. **Host syncs:** 117 ms is 2.5 %. The Markov loop downloads `hw` / `fw` / top-k and runs on the host; a device Markov
   step would remove most of it.
3. **Verify graph default-on:** gate it now on its own (brief 22's gate, minus the draft switch).

## Verify graph default-on for Inkling-2 on CUDA: gate (brief 22 / 23 a)

Change: `ink/cuda-mtp` 427652506. `INTARSIA_VERIFY_GRAPH` unset = on for Inkling-2 on CUDA and off everywhere else;
`0` = off, `1` = on (as before). The `docs/tuning.md` row says so. The test runs unless `=0`.
Gate: fleetq #11418, one job, default (env unset) vs `INTARSIA_VERIFY_GRAPH=0`.

| check | default (on) | `=0` (off) |
|---|---|---|
| verify rows R = 2 / 3 / 4 bitwise, CUDA + cpu | ok | - |
| verify-graph test (one recording per R replayed at 8-17 positions, bitwise) | ok: 1 capture, 17 / 11 / 8 replays | - |
| md5 64 / 256 / 1.2k, spec on == off | ec9a19c14b2d / a0e6f47bc45f / 32ec7a414c3a, equal | same hashes, equal |
| graph replays in the spec-on runs (64 / 256 / 1.2k) | 39 / 135 / 101 (captures 4) = the plain steps + 21 / 108 / 74 verify passes | 18 / 27 / 27 (captures 1) |
| capture failures | 0 | 0 |
| serve smoke | 1k 200 in 2.0 s, 9k 200 in 4.8 s, spec steps 39; port closed, 0 left | 1k 200 in 2.3 s, 9k 200 in 4.8 s, spec steps 39; port closed, 0 left |

ABAB tok/s (generate, spec on, 256 forced tokens, ids identical across arms in every pair):

| prompt | default pair 1, 2 | `=0` pair 1, 2 | gain |
|---|---:|---:|---:|
| 256 (hash map) | 42.92, 42.77 | 41.89, 42.02 | +2.1 % |
| 1k | 54.68, 54.56 | 53.67, 53.57 | +1.9 % |
| 4k | 51.85, 51.80 | 50.76, 50.49 | +2.3 % |

Peak GPU memory per run (nvidia-smi used_memory of the process, 0.5 s samples):
- default: 21238-21366 MiB;
- `=0`: 21232-21570 MiB.

No measurable difference; peak RSS is 19.8-20.2 GB in both arms. The 3 extra verify recordings (q = 2, 3, 4) are lost in
the noise.

## InkInProj (brief 24): bytes, GB/s, floor; why P12 does not pay in the mma form

From the 4k decode profile:
- InkInProj is bf16 (`ink_mma_gemv`), 48 visits a token at 39.6-39.9 MB: **1.90 GB / token, 8.84 ms, 215 GB/s (79 % of 273)**.
- Floor: bf16 6.97 ms; P12 bytes (0.75x + aux) about 5.2 ms.

ink_bench (spark1 #11420, 4 rotating weight copies, synced batches) at the real shape [8664, 2304]:

| kernel | T=1 | T=3 | T=4 | note |
|---|---:|---:|---:|---|
| `ink_in_proj_bf16_mma` (today) | 202 us | 186-198 us | 184-198 us | 540 warps (16 rows each), 198-217 GB/s |
| `ink_in_proj_p12_mma` direct | 191 us | 204 us | 204 us | 147-157 GB/s of codes |
| `ink_in_proj_p12_mma` staged cp.async, 4 stages (3b0b5dffe, reverted) | 208 us | 213 us | 213 us | bitwise, slower |
| `matmul_bf16_gemv_rows` (one warp per row, 8664 warps) | **171.5 us** | 175 us | 175 us | **228-233 GB/s** |

Two attempts at the P12 mma kernel's latency (brief 11's 2-ahead prefetch and wide loads; this 4-stage cp.async
pipeline) did not help, so the assumption that it is short of bytes in flight is wrong. The mma tiling has 540 warps
for 48 SMs, and each lane decodes 144 chunks in one dependent chain at about 11 warps an SM. The decode latency is not
hidden, and P12's decode only lengthens that chain. The GEMV form (one warp per weight row) has 16x the warps.
- bf16 GEMV already beats the mma form by 12-15 % at 1-4 rows.
- Its P12 twin (`p12_gemv_kernel`, bitwise the bf16 GEMV) reads codes at 218 GB/s on wo.

**Decision needed (numerics):** running in_proj as a GEMV (bf16, or P12 lossless on top) changes the in_proj summation
order. Outputs stay rows-exact and P12 stays bitwise its bf16 twin, but the target's greedy md5 moves (like the chunk-2048
change), so it needs the G2 / G3 accuracy gates. Estimated gain per token at 4k:
- GEMV bf16: about 48 x (184 - 172) us = 0.6 ms (2 %);
- GEMV P12 at the wo kernel's 218 GB/s of codes: 48 x (184 - 137) us = 2.3 ms (about 7.5 %).

Before any of this: check the GEMV form at the verify / batch row counts (5-16), where the mma form reads the weights once.

## Draft weight bytes (brief 23 b): `INTARSIA_DSPARK_LAYERS=fp8` A/B, mixed

Setup: fleetq #11434-era job (`dslab` step), spec on, verify graph default-on, 256 forced tokens. Arms alternate bf16 /
fp8, 2 pairs. Output ids are identical in every cell, as expected: the draft only proposes.

| prompt | bf16 tok/s (pair 1, 2) | fp8 tok/s (pair 1, 2) | change | verify passes bf16 -> fp8 | acceptance bf16 -> fp8 |
|---|---:|---:|---:|---:|---:|
| 256 (hash map) | 42.35, 42.90 | 44.31, 44.22 | **+3.8 %** | 108 -> 108 | 46.2 -> 47.8 % |
| 1k | 54.85, 54.90 | 53.04, 53.24 | **-3.1 %** | 74 -> 73 | 81.1 -> 82.3 % |
| 4k | 51.88, 51.85 | 52.45, 52.46 | +1.1 % | 82 -> 83 | 73.6 -> 74.1 % |

Pairs agree within 1.3 %. Acceptance does not drop with fp8: it rises slightly, so the draft keeps its quality. The speed is
mixed. At 1k the fp8 arm runs one verify pass fewer and is still 3 % slower, so the fp8 draft path costs more time per pass
there. My guess is the fp8 rows kernel at the draft's ingest row counts; that is not measured. Net: no default change.
fp8 stays opt-in. A P12 (lossless) draft path is not written yet; it would leave acceptance unchanged and save 25 % of the
233 ms of draft GEMV bytes, about 1.2 %. It is parked behind the InkInProj GEMV work (brief 25).

## InkInProj GEMV form (brief 25): kernel results

Commit `af0e27cd3` on `ink/cuda-mtp`, opt-in `INTARSIA_INK_IN_PROJ_GEMV=1`:
- `ink_in_proj_{bf16,p12}_gemv[_rows]`: one warp per stacked weight row, M = 1..16 activation rows. Each row's sum has the
  same lane / chunk / warp-sum order whatever M, so the kernel is rows-exact.
- The P12 kernel is bitwise its bf16 twin.
- Under the switch, in_proj is packed to P12 and the P12 copy is used up to 16 rows (default: 4).

Test `in_proj_gemv_twins_rows_exact` (fleetq #11445): ok. It covers the real shape and an adversarial small shape, S = 1 / 3,
R = 1..16. P12 == bf16 GEMV bitwise, and rows 0 and R-1 of an R-row call == the 1-row call. The other 4 P12 tests still pass.

ink_bench at [8664, 2304], S = 1 (#11445, µs per call, median of 200):

| rows | bf16 mma (today) | bf16 GEMV | P12 mma | **P12 GEMV** | P12 GEMV GB/s of codes |
|---:|---:|---:|---:|---:|---:|
| 1 | 201.5 | 181.9 | 202.3 | **140.7** | 213 |
| 2 | 190.2 | 185.7 | 197.1 | **148.2** | 202 |
| 4 | 186.6 | 181.7 | 198.2 | **146.2** | 205 |
| 8 | 191.1 | 179.8 | 204.5 | **147.2** | 204 |
| 16 | 187.1 | 192.9 | 205.9 | **179.7** | 167 |

S = 3 (the loop passes with three activation planes) agrees within 3 %.

P12 GEMV is the fastest at every row count: -30 % at 1 row and -4 % at 16 rows, against the bf16 mma in use today. No multi-row
GEMV is needed. Estimated decode gain: 48 visits x about 58 µs = 2.8 ms per token at 1 row, about 9 % at 4k.

## InkInProj GEMV form (brief 25): gate, switch on vs off (fleetq #11456)

The binary was built from `e777c0525`, not `af0e27cd3`: the brief-26 build ran in a queue gap first. The extra commit adds
only `INTARSIA_INK_VERIFY_P12`, which is off when unset. Arms: `INTARSIA_INK_IN_PROJ_GEMV=1` vs `0`.

**Bitwise and md5 (switch on):**
- verify rows R = 2 / 3 / 4: ok on CUDA and cpu.
- Verify-graph test: 1 capture, 17 / 11 / 8 replays, bitwise.
- Greedy md5, spec on == spec off at all three, with 0 capture failures. The md5s move, as expected (a different sum order):

| prompt | ids md5 (switch on) | text md5 (switch on) | ids md5 today |
|---|---|---|---|
| g64 | faf40cecb06b | e17b9c1663db | ec9a19c14b2d |
| g256 | 029e44fede1d | 42d568670a9e | a0e6f47bc45f |
| g1.2k | dc28658094c0 | 053cf3a0ce0f | 32ec7a414c3a |

**Accuracy (KL21; the harness's static floor is 3e-2, the lane budgets are from the mixmoe doc):**

| gate | switch off | switch on | lane budget | verdict |
|---|---:|---:|---:|---|
| G2 short, prefill | 2.376e-3 | 2.376e-3 | 4e-3 | same bits: prefill does not use the GEMV form |
| G2 long, prefill | 3.124e-3 | 3.124e-3 | 6e-3 | same |
| G3 short | 6.630e-3 | 6.630e-3 | 1.05e-2 | same |
| G3 long | 1.173e-2 | 1.173e-2 | 1.65e-2 | same |
| **G2 decode short** | 3.729e-3 | **4.428e-3** | **4e-3** | **over the lane budget** (still under the harness floor) |
| G2 decode long | 3.177e-3 | 3.763e-3 | 6e-3 | pass |

Decode KL rises by about 0.6-0.7e-3 on both prompts, and short crosses its 4e-3 budget (headroom was 0.27e-3).

**ABAB tok/s, 256 forced tokens (mean of 2 pairs; pairs agree within 0.5 %):**

| prompt | spec off: off -> on | spec on: off -> on | prefill tok/s |
|---|---:|---:|---|
| g256 (hash map) | 33.85 -> 36.92 (**+9.1 %**) | 42.72 -> 45.44 (+6.4 %; acceptance 46.2 -> 50.2 %, different text) | equal |
| 1k | 33.45 -> 36.19 (**+8.2 %**) | 54.58 -> 55.60 (+1.9 %; acceptance 81.1 -> 77.4 %) | equal |
| 4k | 32.78 -> 35.40 (**+8.0 %**) | 51.38 -> 54.08 (+5.3 %; acceptance 73.6 -> 70.4 %) | equal |

The spec-on gains mix the kernel gain with acceptance on different text. The spec-off numbers are the clean kernel effect.
Peak GPU memory: 20192 -> 20874 MiB (+0.67 GB for the P12 in_proj copy). Spec on: 21.24-21.49 -> 21.92-22.25 GB.

Not default-on: greedy output moves, and G2 decode short is over its 4e-3 budget. Both are the user's call through the coordinator.

## Verify pass: wo and unembed on the P12 rows kernels (brief 26)

Code on `ink/cuda-mtp`:
- `e777c0525` (switch), `19ef2e95f` (default on; `INTARSIA_INK_VERIFY_P12=0` reverts; `docs/tuning.md` row).
- wo: a CUDA verify pass takes `wo_p12` up to 16 rows. Before, it took it only up to 4 (`DENSE_ROWS_MAX`, a Metal tuning). Metal
  and every non-verify call are unchanged: the wider limit applies only when `self.verifying()` and the backend is `cuda`.
- Unembed: the non-screened verify head (`verify_head`, model.rs) called `self.proj` on the bf16 head. It now tries the P12
  head's rows kernel first, with `exact_rows`. The screened path (`INTARSIA_INK_HEAD_SCREEN`, LogitsGather on the bf16 head) is
  untouched.

**Bit contract** (fleetq #11482, all P12 tests pass):
- `matmul_p12_bitwise_bf16_wo_and_small`: wo [2304 x 3072] at 1..16 rows, P12 rows kernel == bf16 rows GEMV, and rows-exact.
- `matmul_p12_bitwise_bf16_unembed`: now also 1..8 rows at [200058 x 2304], the same checks.

**ink_bench vhead** (µs per call, exact rows; P12 faster at every verify row count, so no lower cap for wo):

| rows | wo bf16 rows | wo P12 rows | unembed bf16 rows | unembed P12 rows |
|---:|---:|---:|---:|---:|
| 1 | 65.1 | 48.7 | 3896 | 2963 |
| 2 | 63.1 | 49.5 | 3877 | 2953 |
| 4 | 62.6 | 50.5 | 3887 | 2973 |
| 5 | 63.6 | 48.7 | 3901 | 2963 |
| 8 | 63.1 | 51.5 | 3922 | 2928 |

**Gate** (#11494, switch = 1):
- Verify-graph test: bitwise, 1 capture, every pass replayed.
- md5 spec on == spec off == today: g64 ec9a19c14b2d, g256 a0e6f47bc45f, g1.2k 32ec7a414c3a; 0 capture failures.
- ABAB spec on (2 pairs, pairs within 0.2 %):
  - g256: 43.08 -> 43.66 tok/s (**+1.3 %**);
  - g1.2k: 55.07 -> 55.68 (**+1.1 %**).

The coordinator estimated about 2.6 %. The measured gain is smaller, but it meets the >= 1 % rule, so the switch is default-on.
Default-on check (build of `19ef2e95f`, env unset, fleetq #11511):
- verify-graph test bitwise (17 / 11 / 8 replays);
- md5 unchanged, spec on == off: ec9a19c14b2d / a0e6f47bc45f / 32ec7a414c3a;
- 0 capture failures;
- spec on 43.66 tok/s at g256 and 55.41 at g1.2k, matching the switch-on arm.

## InkInProj GEMV precision check (brief 27): no precision loss, the G2 move is drift

**Code.** Both forms read bf16 x and bf16 weights (or P12 codes decoded to the exact bf16 bits), accumulate in f32 and round
once to bf16 on output. Neither has a bf16 intermediate.
- mma: `mma.sync` m16n8k16 bf16 x bf16 -> f32 accumulators.
- GEMV: f32 FMA per lane over 8-element chunks, then an f32 warp sum.

**Measurement** (`inkling2_in_proj_precision`, fleetq #11505). Real model on CUDA with today's kernels; G2 short prompt 0,
12-token prefill, then 20 one-token decode steps. Every step's in_proj inputs at layers 0-5 (5 SWA, 1 full; N = 8600 / 8664,
K = 2304) go through all four kernels at 1 row, against an f64 reference over the same bf16 inputs. 1,033,280 outputs per kernel:

| kernel | max abs err | RMS err | RMS err (bf16 ulp) | max err (ulp) | outputs not correctly rounded |
|---|---:|---:|---:|---:|---:|
| bf16 mma (today) | 3.120e-2 | 2.941e-3 | 0.288 | 3.78 | 104 (0.010 %) |
| P12 mma | same bits as bf16 mma | | | | |
| bf16 GEMV | 3.120e-2 | 2.941e-3 | 0.289 | 18.58 | **87 (0.008 %)** |
| P12 GEMV | same bits as bf16 GEMV | | | | |

Both forms are correctly rounded on 99.99 % of outputs; the RMS error is the bf16 output rounding itself (0.29 ulp). GEMV has
fewer misrounded outputs (87 vs 104). Its max of 18.58 ulp is one output near cancellation: its own ulp is tiny, and the absolute
error is inside the same 3.1e-2 max. By the brief's rule, GEMV error <= mma error. So the decode KL move (3.73 -> 4.43e-3) comes
from about 0.01 % of in_proj outputs rounding to the other neighbouring bf16. That flips near-ties downstream, which is drift,
not a precision loss.

**Which G2 decode rows moved** (per prompt, 16 decode rows each, KL21 off -> on):
- Up: chat_reasoning 6.6e-3 -> 11.3e-3, chat_multiturn 3.4e-3 -> 6.0e-3, chat_system 12.3e-3 -> 13.9e-3, story 1.3e-3 -> 1.8e-3.
- Down: chat_short 4.3e-3 -> 0.75e-3, science 1.6e-3 -> 1.2e-3.
- Both directions is a drift signature. chat_system's worst row (0.139 -> 0.143) dominates both arms, so it was already a drift
  row at 3.73e-3. I do not have Metal's per-prompt table, so I cannot say whether these are the rows behind CUDA 3.73e-3 vs
  Metal 2.59e-3.

## CUDA prefill profile (brief 29)

Tree: `ink/cuda-mtp` 19ef2e95f (recipe chunk 2048 on CUDA). spark1, `--quiet gpu,cpu`. fleetq #11514 (generate + nsys) and
#11516 (serve, warm).

**First: the 1460 tok/s figure is stale.** It came from my localeval run #11325, which predates the chunk-2048 merge
(e0a83ecb8). Now:

| path | 1k (1206) | 4k (4116) | 8k (8139) | marginal 4k -> 8k |
|---|---:|---:|---:|---:|
| serve, warm (`--max-batch 1`, `max_tokens 1`, HTTP wall) | 0.5 s | 1.6 s | 3.0 s | **2870 tok/s** |
| generate, cold process (plain) | 0.75 s | 1.89 s | 3.33 s | 2790 tok/s |

The serve gauge `intarsia_last_prefill_tokens_per_second` reads 2712 at 8k. That puts us at the bottom of vLLM's
2770-4018 range, not 2x behind. A cold `generate` pays about 0.5 s once: 13-22 host gaps over 50 us in the first prefill,
which looks like lazy module loading and first-call setup. It is not in a warm serve.

**Per chunk at 8k** (4 chunks of about 2048 rows; nsys kernel time / 4; GPU busy 746 ms per chunk, idle in launch gaps under
5 ms per chunk when warm). Floors use the measured practical bf16 GEMM rate on this box, cuBLAS in_proj at 2048 rows =
81.8 GFLOP in 0.74 ms = 110 TFLOP/s, and 273 GB/s for bytes:

| class | kernels | ms / chunk | % | floor ms / chunk | gap ms | owner |
|---|---|---:|---:|---:|---:|---|
| MoE gu + dn | `ink_moe_mma` (2 per visit, 3.9 ms each), `ink_mma_sum`, route, tiles | 393 | 53 % | ~114 (12.5 TFLOP at 110 TF/s; NVFP4 bytes 27.5 GB -> 101) | **~280** | mixmoe |
| attention | `ink_flash_k` 1.79 ms, `ink_prologue_k` 1.24 ms, kv / ring writes | 147 | 20 % | ~25 (1.85 TFLOP -> 17; rel table 0.1 GB per visit write + read) | **~120** | mine |
| blend / mix | `ink_row_kernel` 460 calls at 180 us, `ink_blend_gate`, `ink_mma_gemv`, `ink_blend_norm` | 135 | 18 % | ~16 (activation planes, ~4.5 GB) | **~120** | mixmoe |
| in_proj | cuBLAS nvjet 192x160 / 128x224 | 36 | 5 % | 36 | ~0 | - |
| wo + other GEMMs | cuBLAS `Kernel2` (192x at 420 us, unidentified op, 3 TF/s), nvjet 128x192 / 128x144 | 35 | 5 % | ~13 | ~20 | ? |
| unembed | `p12_gemv` (last row only) | 0.7 | 0 % | 0.7 | 0 | - |
| launch gaps (warm) | | < 5 | < 1 % | 0 | < 5 | - |

**Levers, ranked by ms saved per 8k prefill (3.0 s):**
1. **MoE grouped GEMM, about 1.1 s (37 %), mixmoe's.** `ink_moe_mma` reaches 32 TFLOP/s against cuBLAS's 110 on the same
   box. What to build: the block-scaled tensor-core path for the expert weights (sm_121 `mma ... kind::mxf4nvf4` / mxf8f6f4)
   instead of a bf16 dequant into the mma, or at least a tile and pipeline that reaches the bf16 rate. Estimated
   prefill gain: +30-50 %. Not bitwise (another sum order). Multi-day.
2. **Attention, about 0.48 s (16 %), mine.**
   - The prologue materialises the rel-bias table [T, H, E] (E = window 1024 on SWA, rel_extent on full layers):
     1.24 ms per visit, about 81 GB/s on its writes. The flash kernel then reads it back.
   - What to build: compute the rel bias inside the flash kernel from r (dr = 16) and the rel position rows, with the same
     16-term chain and bf16 rounding as `ink_rel_g`, so it can stay bitwise. Also retile flash, which runs at about 5 % of
     the GEMM rate.
   - Estimated gain: +12-15 %. The fused rel can be bitwise; a flash retile is not. 1-2 days.
3. **Blend / mix row kernels, about 0.48 s (16 %), mixmoe's.** The same single-block `ink_row_kernel` shape as in decode:
   460 launches per chunk at 180 us. What to build: spread the row ops over D across blocks, or fuse them with the
   neighbouring blend. Estimated gain: +12-15 %. Bitwise if each element's reduction is kept. About 1 day.

Not built: none of the top 3 is under half a day, and items 1 and 3 belong to mixmoe. The one cheap candidate is splitting
the prologue's rel tail differently (bitwise). Its ceiling is about 150 ms (5 %) at 8k, but I have no per-layer-kind split
yet, so its gain is unmeasured.

## Briefs 31 / 32: in_proj GEMV and DSpark spec default-on on CUDA (fleetq #11558, branch `ink/cuda-mtp`)

Commits: fd56f413e (31: `in_proj_gemv(backend)`, default on for CUDA only, `INTARSIA_INK_IN_PROJ_GEMV=0` reverts; Metal
unchanged), ddb7bb164 (32: `[tuning.cuda] mtp_default_k = 3` in recipes/inkling2.toml read by `dspark_default` in
`intarsia-models/src/load.rs`; `Recipe::resolve_draft_default` for a recipe without `[mtp]`; unit test
`dspark_default_only_where_the_recipe_sets_it`). The gate build also carries 30a with its routing off (fe6ae38fa) and the
softcap commit b5b435f1a (cherry-pick of 36cbb9536).

**Brief 31, greedy md5 (ids), spec off == spec on in every row:**

| | 64 | 256 | 1.2k |
|---|---|---|---|
| default (GEMV on): new CUDA references | faf40cecb06b | 029e44fede1d | dc28658094c0 |
| `INTARSIA_INK_IN_PROJ_GEMV=0` | ec9a19c14b2d | a0e6f47bc45f | 32ec7a414c3a (the old references) |

**Brief 31, ABBA decode tok/s** (generate, 256 forced tokens, pair 1 = on, off; pair 2 = off, on):

| | GEMV on (p1 / p2) | GEMV off (p1 / p2) | gain |
|---|---|---|---|
| spec off, g256 | 36.80 / 36.65 | 33.95 / 33.93 | +8.3 % |
| spec off, g1.2k | 36.06 / 35.94 | 33.42 / 33.36 | +7.8 % |
| spec on, g256 | 46.19 / 46.16 | 43.43 / 43.41 | +6.3 % (acceptance 50.2 vs 46.2 %: another text) |
| spec on, g1.2k | 56.62 / 56.41 | 55.23 / 55.45 | +2.0 % (acceptance 77.4 vs 81.1 %) |

**Brief 31, accuracy** (21-way KL, all PASS; default = GEMV on vs `=0`):

| golden | GEMV on | GEMV off |
|---|---:|---:|
| G2 short prefill | 2.376e-3 | 2.376e-3 |
| G2 long prefill | 3.124e-3 | 3.124e-3 |
| G2 short decode | 4.428e-3 | 3.729e-3 (the accepted drift, limit 4e-3) |
| G2 long decode | 3.763e-3 | 3.177e-3 (within 4e-3) |
| G3 short / long / 4k | 6.630e-3 / 1.173e-2 / 9.671e-3 | the same |

G3 and the G2 prefill rows are identical: prefill chunks run the mma form either way (GEMV is 1..16 rows only).

Spread within an arm is under 0.5 %, so every gain is beyond it. Prefill unchanged (1550-1592 tok/s at 1.2k in all arms).
GPU peak +680 MB with GEMV on (the P12 in_proj copy is packed: 20874 vs 20192 MiB spec off).

**Brief 32:**
- Default (no `--spec`) vs `--spec off`: ids and text identical at 64 / 256 / 1.2k. Decode 38.24 vs 37.18, 46.16 vs 36.88,
  56.69 vs 36.06 tok/s. Verify passes 18 / 107 / 77. The loader prints the default line.
- Sampled t1.0 seed 7, 64 tokens: default == `--spec off` (ids 93994764f351, same text), no verify pass.
- Serve with the default, `--max-batch 1`: HTTP 200 at 1k (1.8 s) and 9k (4.7 s); spec steps 42, proposed 105, accepted 72.
- A batching server (`--max-batch 4`) refuses Inkling-2 on CUDA at load, as it did before brief 32: "batched decode of
  Inkling-2 needs the row-table kernels ... (the cuda backend registers none)". So no batched output exists to change.
- Follow-up job (#11591): repeat of default vs off: identical ids again at 64 / 256 / 1.2k and sampled.
  Model dir without `mtp.safetensors`: exit 0, one line "--spec dflash default of recipe inkling2 on cuda not applied: no
  DSpark draft in this checkpoint ...; --spec off".
- Tiny CUDA suite `inkling2` (rows-exact, verify rows == decode, graph replays == decode, chunked == prefill): 15 / 15 pass
  with the default and with `INTARSIA_INK_IN_PROJ_GEMV=0`, after e89a96c14 (the verify tests capped the rows a second time
  once b5b435f1a caps them in the model).
- `inkling2_verify_softcap` on CUDA: pass in all four arms (default, `VERIFY_P12=0`, `PACK12=1`, both).
- `inkling2_batch` (batched == solo) does not apply on CUDA: Inkling-2 has no CUDA row-table kernels and batching is refused
  at load. The CUDA arm was removed again (8f5c9a31c); the Linux build fix stays.
- localeval speed-plain (#11592, serve `--max-batch 1`, 128 out tokens, ABBA = off, default, default, off):

| prompt | TTFT off (a1 / a4) | TTFT default (a2 / a3) | prefill tok/s off | prefill tok/s default | decode off | decode default |
|---:|---:|---:|---:|---:|---:|---:|
| 142 | 0.24 / 0.23 | 0.23 / 0.24 | 571 / 566 | 568 / 563 | 37.1 / 37.2 | 48.7 / 53.9 |
| 1k | 0.49 / 0.51 | 0.47 / 0.48 | 2129 / 2043 | 2179 / 2147 | 36.9 / 37.1 | 45.5 / 47.3 |
| 4k | 1.67 / 1.68 | 1.68 / 1.66 | 2476 / 2463 | 2460 / 2486 | 36.1 / 35.8 | 53.9 / 51.3 |
| 16k | 6.30 / 6.28 | 6.26 / 6.28 | 2610 / 2615 | 2627 / 2619 | 33.7 / 33.9 | 47.4 / 50.9 |

  **Brief 32 bar met:** decode gains at every length (+31-45 % at 142, +23-28 % at 1k, +43-51 % at 4k, +40-50 % at 16k).
  Prefill and TTFT are within 2 % at every length; the means differ by under 1 %. Unlike Metal, there is no drafter-ingest
  prefill loss.

## Op 75 on prefill chunks: the rel bias inside flash (briefs 30 / 30a, branch `ink/cuda-mtp`)

What changed (86060ae6f, plus the tile skip and register-staged relproj, uncommitted at the time of the v2 / v3 runs):
- Contract (30a): `InkAttnDecode` (op 75) is `RelSinkAttn(InkKvPrologue(..))` bit for bit at any T >= 1; the core check takes
  T > 16. Ops 70 / 71 are unchanged; the pair stays the fallback and the dump path. cpu registers op 75 at 1..=16 rows only,
  Metal already did.
- CUDA `ink_attn_prefill_bf16` (op 75 at T > 16, G = 3): the prologue writes q / k / v and r `[T,H,dr]` to scratch instead
  of the rel table (`F_R_OUT`), then `ink_flash_k<D, RIN>` forms each bias as `ink_rel_g` does (chain from 0, j ascending,
  `__fmaf_rn`, then the bf16 round) from r in shared memory and the relproj columns of the sub-chunk's distances
  (`[FBR + CH][16]` f32, loaded into registers one sub-chunk ahead). A 16-key step whose smallest distance is >= E skips it.
- Graph: a prefill chunk takes op 75 where the backend admits it (CUDA), else the pair.

Checks:
- CUDA `op75_prefill_bitwise_pair`: 18 cases (T = 17 / 64 / 2048; SWA ring 1040 at pos0 0 / 1500 / 3001; full at 0 / 1500
  / 5003) bitwise the pair (prologue + `rel_sink_attn_bf16_flash`): output, conv rings, K / V caches (#11523, #11526).
- cpu: `fused_decode_attention_is_the_pair_bitwise` now 82 cases with 12 real-shape prefill cases (T = 17 / 64 / 2048, SWA
  and full, ctx 0 and > 0), all bitwise. (Run on endor without a slot, see rule 33.)
- `plan show` text + json, base d0e247bb5 vs tip, Inkling on Metal and cpu, Qwen3-8B and Llama-3.1-8B on Metal: identical
  except the load-time lines of Inkling on Metal (#11524).

Per visit, 2048-row chunk, us, graph replay (`inkling2_attn_bench`, INTARSIA_INK_BENCH_PREFILL=1):

| layer ctx | pair (#11538) | op 75 v1 (#11523) | op 75 v2 (#11526) | op 75 v3 (#11538, a46f77429) | saved v3 |
|---|---:|---:|---:|---:|---:|
| SWA 0 | 2509 | 1990 | 1982 | 1916 | 593 |
| SWA 2k | 2833 | 2395 | 2432 | 2377 | 456 |
| SWA 30k | 2731 | 2487 | 2375 | 2346 | 385 |
| full 0 | 2626 | 2075 | 2056 | 2092 | 534 |
| full 2k | 3624 | 3307 | 3400 | 3209 | 415 |
| full 30k | 15644 | 17733 | 18720 | 15817 | -173 |

- v1: relproj staged with plain loads each sub-chunk. v2: staged through registers one sub-chunk ahead (full 30k got
  worse). v3: a 16-key step whose smallest distance is >= E skips the r loads and the dot; that fixed full 30k (r was loaded
  for every step). ptxas: D = 128 RIN 164 registers, no spills.
- Pair column run to run: 2404-2509 (SWA 0), 3601-3624 (full 2k); single runs, about 4 % noise.
- **Against the 0.5 ms bar:** v3 saves 0.39-0.59 ms per visit at the prefill contexts (mean 0.48 over the five cells up to
  SWA 30k / full 2k), about 20 % of the attention visit; full layers at long context break even. So it sits at the bar,
  not clearly above it. Reported to main before any end-to-end gate.

### Brief 30b end-to-end (#11593, #11616, `INTARSIA_INK_PREFILL_FUSED` unset vs `=1`, tip 8f5c9a31c)

- Greedy ids (generate, 128 forced tokens) are identical, fused off vs on and spec off vs on, at hm / 1.2k / 4k / 9k.
- G2 / G3 with `=1`: every KL is bitwise the unfused run (2.376e-3, 3.124e-3, 4.428e-3, 3.763e-3, 6.630e-3, 1.173e-2,
  9.671e-3), all PASS.
- Cold generate prefill, fused vs unfused: 1.2k 742-748 vs 770-772 ms (+3.5 %), 4k 1843-1848 vs 1894 ms (+2.7 %),
  9k 3628-3629 vs 3719-3745 ms (+2.8 %). 146 tokens: equal.
- Warm serve ABAB, ms timer (#11632, 6 requests per arm and size, A = unfused, B = fused):

| prompt | A s (min-max) | B s (min-max) | gain |
|---:|---:|---:|---:|
| 1k | 0.490-0.493 | 0.469-0.474 | +4.1 % |
| 4k | 1.607-1.616 | 1.555-1.562 | +3.3 % |
| 8k | 3.013-3.037 | 2.919-2.937 | +3.4 % |
| 16k | 6.106-6.137 | 5.912-6.034 | +2.7 % |

  **Bar met** (>= 1 % at 4k and 8k, not slower at 1k and 16k, bitwise). Default-on as its own commit, `INTARSIA_INK_PREFILL_FUSED=0`
  reverts (on ink/cuda-mtp; hash in the landing section).

## Next ideas (ranked)

1. **Decode tile kernel: more keys per warp per barrier.** Each warp takes 64 keys per sub-chunk (4 mma k-groups)
   instead of 16, giving 256-key sub-chunks with a quarter of the barriers. Or load K / V fragments straight from
   global, with no shared-memory staging, for M <= 48 rows. Target: full 32k from 716 to about 520 us.
2. **Contract idea for main: attention computes the bias from r and relproj.** It would drop the `rel [T,H,E]` tensor:
   100 MB per 2048-row chunk written and read, about 0.75 ms at the floor. The prologue's rel tail is most of its
   1.6 ms per chunk, and the bias fetch dominates SWA flash. Not changed here: it is a contract change.
3. **Faster tile combine** (the fold was tried and lost, see above): `ink_tile_comb_k` costs 2-4 us at short ctx and
   22 us at full 32k with 16 rows. Use a warp per (row, head) over the partials, or PDL so it overlaps the tile kernel's tail.
4. **Flash prefill:** stage the bias slice with cp.async per sub-chunk; 128-row query blocks for full layers (14.3 ms at
   30k is about 58 TFLOPS).
5. Full 4k at 4 rows (fused 9 us slower than the pair): the fused blocks each redo the call's q / r jobs; share them
   through a 1-block pre-pass if verify rows matter.

## Landing PR #498 (branch `ink/cuda-mtp-land`, brief 35-37)

Merge of ink/cuda-mtp onto #491 (main 8c7bb7cb1). Conflicts and their resolution:
- `model.rs` verify head: takes main's `head_proj`, the P12 rows head at `exact_rows` on every backend. Main's screen option B
  had already dropped the screened verify path. `INTARSIA_INK_VERIFY_P12` now governs wo only (doc and tuning row updated).
- The duplicate `softcap_step` is dropped.
- `inkling2.rs` keeps both env helpers.
- The tests take main's softcap text and keep the CUDA arm of `inkling2_verify_softcap`.
- `load.rs`: the recipe DSpark default keeps its own k (main's `DSPARK_K_DEFAULT` block skips it).
- `docs/tuning.md` gains rows for the bench and precision diagnostic variables, so `docs_env_vars` passes.

Gates on the merged tree:
- spark1 #11657 at 644a3a1c2:
  - CUDA md5, default: faf40cecb06b / 029e44fede1d / dc28658094c0, spec on == off.
  - `IN_PROJ_GEMV=0`: ec9a19c14b2d / a0e6f47bc45f / 32ec7a414c3a, spec on == off.
  - The default equals `--spec off` at 64 / 256 / 1.2k and on a sampled request.
  - `INTARSIA_SPEC_DEFAULT=0`: the default's ids, no verify pass. A model dir without the draft: plain decode with one line.
  - 9k prefill: identical ids with fused off and on (3720 vs 3607 ms).
  - Tiny CUDA `inkling2` 17/17. `inkling2_verify_softcap` passes in all 4 arms.
  - CUDA plan show for the 35B and 8B, text and json: byte-identical to main.
  - HIP check-only, both targets: pass.
  - The `inkling2_batch` CUDA arm failure is a stale test binary from 449c11a16: the landing tree has no CUDA arm and the gate did
    not rebuild that binary.
- endor #11656: Metal and cpu plan show for Qwen3-8B and Llama byte-identical. Inkling Metal ids at g64 / g256 identical to
  main (fa9f4b4a5701 / 550cd932b33f).
- endor #11652: cpu tests pass: models lib 80, inkling2 17, softcap, inkling2_batch 8, ink_attn_decode, core lib 264. docs_env_vars
  passes after c371d8242.
- At 64 output tokens the spec default decodes 4 % slower (37.8 vs 39.4 tok/s, acceptance 44 %). From 256 tokens up it is faster.
- Post-#497 merge (51503dd85, main 0c8f2c037), spark1 #11683:
  - CUDA md5, default: faf40cecb06b / 029e44fede1d / dc28658094c0. `GEMV=0`: ec9a19c14b2d / a0e6f47bc45f / 32ec7a414c3a.
    Spec on == off in both.
  - Tiny CUDA 17/17 with PACK12 on and off. `inkling2_verify_softcap` passes in 4 arms.
  - The push (#11684) waits in the endor queue behind the quiet mlx-serve bench #11677. The coordinator said to keep it there.
- **#498 merged** on 2026-11-22 as main 576ead89b (head 51503dd85, all CI green). The landing worktree is removed.

## Cleanup and next levers (brief 38, 2026-11-22)

Cleanup:
- Removed: the landing worktree, the remote `ink/cuda-mtp-land` branch, spark1 `~/intarsiaml-ink-cuda-mtp`, and everything
  under spark1 `~/intarsiaml-ink-cuda-attn` except the model.
- Kept: spark1 `~/intarsiaml-ink-cuda-attn/model` (19 GB; the ink-cuda-graph and ink-cuda-p12 gate scripts point at it).
- Kept: worktrees `.worktrees/ink-cuda-attn`, branch `ink/cuda-attn`. Its results-doc lines after 51503dd85 are not on main.
- Kept: worktree `.worktrees/ink-cuda-mtp`, branch merged; its untracked `.scratch` holds the gate, bench and profile scripts.

Next CUDA levers, MoE experts excluded. The estimates come from the nsys tables above (decode at 1.2k-7.9k ctx, prefill per
8k), not from new runs.
(Correction after brief 46: the #1 and #4 estimates came from profiles taken before `INTARSIA_INK_ROW_HOIST` and the
controller split-K were on by default. On main the row kernel is 1.25 ms a token and the blend class 1.27 ms, per #11719.)

| rank | lever | saves | bitwise | days |
|---|---|---|---|---|
| 1 | Decode: split the `ink_row` mix / conv kernels across blocks. Today 124 single-block launches per token at 17 us. Keep each row's reduction order | ~1.5-2.0 ms / token (~6 %) | yes, if reductions are kept per row | 1-2 |
| 2 | Prefill: the same row kernels at 2048 rows (460 calls at 180 us per chunk) | ~0.45 s per 8k (~15 %) | yes, as 1 | shared with 1 |
| 3 | Decode: P12 in_proj GEMV to its byte floor. 141-150 us per visit vs ~110 us for 30 MB | ~1.5 ms / token (~5 %) | yes (P12 twin) | 1-2 |
| 4 | Decode: fuse the loop controllers / blend (`ink_mma_gemv` on 1-3 blocks, blend_gate, blend_norm; 2x24 each) | ~1.0-1.2 ms / token | yes, if dot order is kept | 1-2 |
| 5 | Prefill: flash retile (lever 2b). Flash runs at about 5 % of the GEMM rate; attention is ~100 ms per chunk after op 75 vs a floor of ~25 | ~0.3 s per 8k (~10 %) | no | 2-3 |
| 6 | Decode: occupancy of the SWA attention tile kernel (48 blocks at 113 GB/s, ~1.2 ms / token) | ~0.5 ms / token | no, if the split changes | 1 |
| 7 | Decode: unembed P12 GEMV, 2.93 ms vs a 2.53 ms floor | ~0.4 ms / token | yes | 0.5 |
| 8 | Prefill: identify cuBLAS `Kernel2` (192 calls at 420 us, 3 TF/s per chunk) | up to ~80 ms per 8k | depends | 0.5 to measure |

## inkling2_batch on main (brief 38 check A, fleetq #11705)

Main 576ead89b built in its own src and target dir on spark1; test binary md5 c2318879f0ed. Result: no failure on main, so
no fix or revert.
- **The literal command** (`--features cuda`) does not start: `intarsia-models` has no `cuda` feature.
- **Default backend (CPU), CI=1:** 8/8 pass in 12 s.
- **INTARSIA_INKLING2_BACKEND=cuda:** every test panics at `inkling2_batch.rs:44`, by design. The test refuses CUDA because
  CUDA has no Inkling-2 row-table kernels and refuses batching. The CUDA arm was removed in 8f5c9a31c, which landed with #498.
- **The landing-gate panics** (`:80` / `:104`) came from the gate running this test with BACKEND=cuda on the landing tree
  before 8f5c9a31c. CUDA refused that arm's `enable_batch` / `forward`.
- **0c8f2c037 was not run;** the brief asked for it only if main failed.
- **What the test checks:** on the tiny fixture, every row of a batched `forward_batch` step is bitwise the solo step of its
  sequence (logits, softcapped sampler rows, tokens). Cases: 2, 5 and 16 sequences around the SWA window, the rel extent and
  the attn-res blocks; membership changes; per-sequence rollback; failed-pass reset; the P12 plan; batched engine == solo.
- **Lever gates** run it on its own backend.

## Levers 1 + 2 (brief 46): row kernel split and prefetch, stopped

Both approaches are bitwise, but neither reaches the ship bar, so neither ships and there is no third variant. The branch
keeps only a 20-row ConvMix / SconvRes case in `ink_rows`, which exercises the deferred ring write at the real ring.

**Approach 1, split** (commits 39def9d19, a384f3dc3; removed in a18d25935).
- How it works: 2-8 blocks per row as a thread-block cluster. Each thread keeps the elements of the one-block layout, and the
  warp partials are summed in warp order through distributed shared memory.
- Bitwise: `ink_rows` passes at splits 2-8, decode and prefill, against one-block rows (fleetq #11709 / #11713).
- Speed (`ink_bench mix`, #11709, us per call): flat to slower in decode, and slower at 2048 rows by +2 % (split 2) to
  +340 % (split 8).
- Why: each thread's dependent chain (loads, butterflies, block sums, the second reduction) is unchanged. The one-block
  kernel is latency-bound per thread, not short of SMs.
- Registers are not the cause: ptxas shows the production instance (NE 3, one block) at 63 registers with no spills.

**Approach 2, L2 prefetch of all input rows at block start** (a384f3dc3, a18d25935; reverted).
- Bitwise: `ink_rows` 8/8, with the 20 / 300-row calls (prefetch) against the 1-row calls (none).
- md5: 64 / 256 / 1.2k = faf40cecb06b / 029e44fede1d / dc28658094c0, spec off and default (#11714).
- `ink_bench mix` (#11713), prefetch on vs off:

| call | rows | off us | on us |
|---|---:|---:|---:|
| conv_mix N7 Q3 norm | 2048 | 997-1196 | 774 |
| conv_mix N4 Q1 norm | 2048 | 547 | 450-461 |
| attn_res_mix N4 Q1 | 2048 | 373-519 | 297 |
| sconv_res | 2048 | 234 | 212 |
| blend calls (S 1 / 3) | 2048 | flat | flat |
| conv_mix N4 Q1 norm | 1 / 16 | 10.2 | 12.3 |
| conv_mix N7 Q3 norm | 1 / 16 | 16.4 | 18.5 |

- Model, 8139-token prompt, spec off, 3 alternating pairs (#11714):

| pair | default (prefetch above 16 rows) ms | `INTARSIA_INK_ROW_PREFETCH=0` ms |
|---|---:|---:|
| 1 | 3178 | 3180 |
| 2 | 3163 | 3180 |
| 3 | 3159 | 3171 |

- Result: +0.4 % at 8k, under the 2 % bar. The faster mix calls are a small part of the chunk, and the blend calls dominate the
  row kernel time there and do not move. Decode would lose about 2 us a call.

**Gates on the prefetch tree** (#11714):
- md5 as above.
- `inkling2_batch` 8/8 on its own backend.
- Tiny CUDA `inkling2`: 15 pass and 2 skip by design ("no screened-head kernels on CUDA"). They show as failures only because
  the gate sets CI=1; the earlier 17/17 counts ran without CI.

What would move the decode row kernel (1.25 ms a token: 124 x 10.1 us in #11719; the 2.1 ms / 17 us of the ranking predate `INTARSIA_INK_ROW_HOIST`) is a shorter per-thread chain, not more blocks:
- fuse the two reductions of a ConvMix (the mix statistics and the norm) into fewer block barriers;
- or overlap the row kernel with the next kernel (PDL).
Both change code structure, not just tuning. A fused reduction order would not be bitwise. PDL is, but it is the mixmoe
lane's repack + PDL work (#11700).

## Lever 3 (brief 46): P12 in_proj GEMV toward its floor, stopped; lever 7 bounded below the bar

Two bitwise approaches on `ink_in_proj_p12_gemv[_rows]`. Each keeps every row's lane, chunk and add order and its warp sum,
so P12 stays bitwise its bf16 twin and the model's bits do not move.
- `INTARSIA_INK_IN_PROJ_RPW=2|4`: 2 or 4 adjacent weight rows per warp. The rows' code loads are in flight together, and
  rows of one segment share the activation load. Commit e0279e43a.
- `INTARSIA_INK_IN_PROJ_PERSIST=1`: a grid of the resident blocks only, each warp striding over its row groups. Commit
  7228a5862.
- Both were default off; reverted after the measurement (RPW 2 / 4 also make 16-row calls 35-75 % slower). The [41, 23, 25]
  twin case stays: it exercises the default kernel.

Tests (fleetq #11717, #11718): p12 5/5 with unset, `GEMV=0`, `RPW=2 + PERSIST=1` and `PERSIST=1`. The twin test gained a
[41, 23, 25] case, which puts segment edges inside a row pair or quad and leaves a ragged tail.

`p12_selection` was stale on main. It still expected `ink_in_proj_p12_mma`, but the GEMV default-on of brief 31 (#498)
selects `ink_in_proj_p12_gemv`. Fixed to follow `p12::in_proj_gemv("cuda")` in fde495485.

ink_bench at [8664, 2304], us per call; ranges are the two reps and S = 1 / 3 (#11718):

| rows | default | RPW 2 | PERSIST | RPW 2 + PERSIST |
|---:|---:|---:|---:|---:|
| 1 | 144-148 | 139-143 | 148-158 | 139-152 |
| 2 | 148-151 | 140-143 | 140-157 | 140-155 |
| 4 | 147-148 | 147-149 | 142-158 | 146-154 |
| 16 | 181-183 | 250-257 | 161-164 | 237-250 |

**Result:**
- The best choice per row count saves about 6 us at 1 row, nothing at 4 rows (spec verify, k 3), and 18 us at 16 rows.
- At 48 visits a token that is 0.29 ms, about 1.2 % at g256 spec off (about 25 ms a token), and 0 % with spec on.
- That is under the 2 % bar, so no PR. The rest of the gap to the floor (146 vs about 120 us at the 252 GB/s streaming peak of
  `bw.cu`) is not in wave tails or loads in flight.
- The P12 code layout fixes the loads: an 8-byte plane-A and a 4-byte plane-B word per 8 elements and lane. A wider per-lane
  chunk would change the lane-to-element map and so the sum order (md5).

**Lever 7 (unembed `p12_gemv`) is bounded below the bar:**
- It runs at 2.93 ms a token, 236 GB/s (87 % of 273).
- At the 252 GB/s practical peak it would take 2.75 ms. The most it can give is 0.18 ms, about 0.7 %. Not attempted.

## Lever 4 (brief 46): loop controller / blend, scoped, one bitwise try, no gain

Fresh nsys of graphed decode (fleetq #11719; main 576ead89b kernels; hash-map prompt, 256 tokens, spec off):
- 38.4 tok/s: 26.05 ms wall a token, 25.68 ms busy, 657 kernels a token.
- MoE gu + dn 8.86 ms; in_proj (P12 GEMV) 6.67 ms; unembed 2.96 ms; wo 2.35 ms; SWA attention tile 1.31 ms; row kernel 1.25 ms
  (124 x 10.1 us).
- Blend / controller class, 1.27 ms a token:

| kernel | streams | us per call | ms / token |
|---|---|---:|---:|
| `ink_ctl_kernel` | S = 3 | 14.55 | 0.35 |
| `ink_ctl_kernel` | S = 1 | 8.34 | 0.20 |
| `ink_blend_gate_kernel` | S = 3 | 13.77 | 0.33 |
| `ink_blend_gate_kernel` | S = 1 | 6.13 | 0.15 |
| `ink_blend_norm_kernel` | S = 1 / 3 | 5.1 | 0.25 |

Byte floors: S = 3 moves 1.77 MB through each of ctl and gate, about 7 us at 252 GB/s. Taking both to their floors and folding
the norm into the gate's last block would save at most about 0.8 ms (3 %). That bound needs latency fixes I cannot see:
ncu is admin-only on spark1 (`RmProfilingAdminOnly: 1`).

The try: `#pragma unroll` on the gate's gain / forget load loop and on the controller fold's partial loads. Same order, so
bitwise; `ink_rows` ok. In-graph times did not move (#11721: ctl 14.63 / 8.33 us, gate 14.13 / 6.26 us), so the compiler was
already overlapping those loads. Reverted.

The norm fold alone is worth at most 0.25 ms (1 %), under the bar, so it was not built. The SWA attention tile (33 us a call
at 271 positions, 1.31 ms a token) is a fixed in-block latency chain, not context-bound: its blocks already skip empty tiles.
More warps for the in-block prologue would be bitwise but worth about 0.6 %.

## Lever 8 (brief 46, measure only): the prefill's cuBLAS `Kernel2` is wo

nsys `-t cuda,cublas` of the 8139-token prefill, one output token (fleetq #11726, spark1 idle).
- **Calls:** `Kernel2` has grid 128 x 2, block 256, and runs 192 times (48 visits x 4 chunks).
- **Driver call:** it is launched through `cuLaunchKernel`, as cuBLAS does.
- **Neighbours:** every instance sits between the attention kernels (`ink_prologue_k`, `ink_ring_write_k`, `ink_flash_k`,
  `ink_kv_write_k`) and the next sublayer's `ink_row_kernel`. So it is the wo projection: [2048 x 3072] by [3072 x 2304], bf16.
- **Cost:** 404.9 us a call and 77.7 ms per 8k prefill (2.3 % of 3385 ms). That is 2.9e10 FLOP in 405 us, about 72 TFLOP/s.
- **Correction:** the earlier class table put this kernel at 3 TF/s; that figure was wrong.
- **Ceiling:** at the 110 TFLOP/s cuBLAS reaches on in_proj at 2048 rows, wo would take about 264 us. The most it can save
  is about 27 ms per 8k (0.8 %).

## PDL on the attention decode edges (brief 50, branch `ink/cuda-pdlattn` off main a4e281264)

**What it does** (`INTARSIA_INK_ATTN_PDL`, default on; `0` / `off` / `false` gives the old plain launches; `launch` gives PDL
without the prefetches, a diagnostic). Op 75 decode (1..16 rows), graphed or eager. New header `kernels/cuda/ink_pdl.cuh`:
the same two griddepcontrol instructions as `ink_moe.cu`'s file-local helpers, which stay unchanged.
- **in_proj -> tile:** the in_proj GEMV triggers its dependents at block start. The tile, launched as a programmatic
  dependent, issues `prefetch.global.L2` of its block's old K / V cache rows (positions below the call's first), then
  `griddepcontrol.wait`.
- **tile -> combine -> wo:** combine waits at once and triggers wo. wo (`matmul_p12_gemv[_rows]` only; the logits path
  stays a plain launch) prefetches its own P12 weight row into L2, then waits.
- **The rule:** before the wait, a kernel reads only data finished before the step's kernels started (the position word,
  old cache rows, weights). The cache rows and weights are read through L2 prefetch hints, which return no data. Every block
  waits before it returns, so completion stays transitive along the chain. The reasoning per edge is in the comments at
  `tile_pdl_prelude`, `ink_tile_comb_k` and `p12_gemv_kernel`.

**Gate 1, tests** (fleetq #11803, #11807; spark1; build db49be53e):
- `pdl_prefetch_rows_stay_below_pos0` (new, in `inkling2_attn.rs`) passes. It runs `ik_ink_attn_pdl_selftest`, which drives
  the kernels' own `tile_geo` / `tile_blk` / `tile_pdl_rows` over 3 windows, 4 tile floors, 3 tile caps, rows 1..16 and 400
  positions up to 40k. A violation is a prefetched row at or past pos0, outside the block's keys, or (ring) sharing a slot
  with the call's rows.
- **It can fail:** the same file with the range end moved to pos0 (one sed line) reports 231096 violations; unmodified, 0.
  It is a range test because a prefetch hint cannot change a value, so no bitwise test could see a wrong range.
- **All intarsia-backend-cuda tests, `--no-fail-fast`, PDL on:** 44 result lines, 41 ok; 194 passed, 4 failed, 26 ignored.
  The 4 failures are main's: this diff touches no kernel table or pin, and `git grep` finds no `ik_ink_attn_prefill_bf16` in
  main's ABI headers (main's suite itself was not run). CI evidently does not run these host-side tests:
  - `kernels::tests::every_launcher_is_declared_in_the_abi_header`, `op_prefix_matches_launcher_name` and
    `quick_coverage::every_cuda_kernel_has_a_quick_case`, all three on `ink_attn_prefill_bf16`. That kernel came in with my
    86060ae6f (brief 30a): it is not in the ABI header, is not named after op 75, and has no quick case.
  - `rows_registry::exact_rows_calls_pick_the_rows_kernels_only`: the pin table lacks mixmoe's `matmul_mxfp4_gemv_ink_rows`
    (5..=8 rows at k = 160).
- **With `INTARSIA_INK_ATTN_PDL=0`:** `inkling2_attn` 3/3, `ink_rows` 9/9, `p12` 5/5, `rows_exact` 12/12, `verify_attn_rows`
  3/3, `graph_pos` 5/5.
- **Tiny inkling2 on CUDA, default and `=0`:** 15/15 (the 2 screened-head tests are skipped: no CUDA kernels by design);
  `inkling2_batch` 8/8.

**Gate 2, bits** (#11803). The md5 of 64 / 256 / 1.2k is faf40cecb06b / 029e44fede1d / dc28658094c0, main's references,
under the default and `=0`, spec off and spec default. The A/B runs below also give one token md5 per length across all
arms and reps: short 029e44fede1d, 5k 34e3b3e6cebb.

**Gate 3, quiet decode A/B** (#11806, `--quiet gpu,cpu`; 3 reps; the arm order rotates per cell).
- M = main's decode code: the B build 900230f6a (main a4e281264 + the prefill-only flash bias) with
  `INTARSIA_INK_FLASH_BIAS=0`, which gives main's bits.
- P = this branch, default. Z = this branch with `=0`. L = `launch` (PDL without the prefetches).

Decode tok/s, mean (min-max), change vs M:

| length | spec | M | P | Z | L |
|---|---|---:|---:|---:|---:|
| short (15 + 256) | off | 39.87 (39.82-39.89) | **40.17 (+0.75 %)** | 39.82 (-0.12 %) | 40.01 (+0.36 %) |
| short | default | 49.34 | 49.48 (+0.28 %) | 49.28 (-0.13 %) | 49.49 (+0.29 %) |
| 5k (5039 + 128) | off | 37.74 (37.67-37.77) | **38.19 (+1.19 %)** | 37.70 (-0.10 %) | 37.75 (+0.04 %) |
| 5k | default | 50.19 (49.64-50.56) | 50.79 (+1.20 %) | 50.44 (+0.50 %) | 50.16 (-0.06 %) |

Graphed decode, nsys, spec off (wall and idle per token; idle = wall - busy):

| | short P | short Z | 5k P | 5k Z |
|---|---:|---:|---:|---:|
| wall ms / token | 25.27 | 25.52 | 26.39 | 26.91 |
| idle ms / token | 0.36 | 0.39 | 0.37 | 0.38 |
| kernels / token | 657 | 657 | 657 | 657 |

**Result: in the band the brief left open.** Spec off: +0.75 % short, +1.19 % at 5k. That is above the +0.5 % park line
and under the +1 % ship bar at short. P's and M's ranges do not overlap, and Z is within 0.12 % of M. The spec-default
cells are noise-limited (Z vs M +0.50 % at 5k, M's range 1.8 %), so they are not evidence either way.

**What limits it** (fleetq #11808 `cuobjdump -res-usage`, #11809 device properties).
- The GB10 has 48 SMs, a 64K register file per SM and a 24 MB L2. `ink_tile_staged_k<128, fused>` holds 162 registers x
  384 threads and `ink_tile_k` 168: one tile block fills an SM's register file.
- **5k:** most of the 6 SWA tile slots per kv head have keys, and the full-layer tile runs 34 slots. Live tile blocks cover
  the SMs, so wo and combine cannot be resident beside them. PDL overlaps the tile's uneven tail and the launch gaps, and the
  old-row prefetch carries most of the gain (L alone is +0.04 %, P +1.19 %).
- **Short (positions 15-270, tile 256):** only 1-2 of the 6 slots per kv head have keys, so 8-16 of 48 tile blocks are live
  and the rest return after the wait.
  - Combine and wo do become resident early and spin at the wait. Their kernel times stretch over the tile: combine 1.0 ->
    41.5 us, wo 48.8 -> 78.1 us a call.
  - wo's weight prefetch is issued about 33 us before its wait returns, yet wo still runs about 44 us after it, against
    48.8 us plain.
  - The 10.6 MB of wo weights fits the 24 MB L2, so capacity alone does not explain it. **Unexplained.** Candidates:
    prefetch hints dropped or throttled, wo's 10.6 MB needing about 39 us of DRAM against the tile's 33 us, or L2
    partition effects.
  - Capping the tile's registers would therefore not move short context.

**Follow-ups, outside this brief:**
- (a) Find out why the wo prefetch does not pay at short context: a microbench of wo after a prefetch with an idle gap, and
  `cp.async.bulk.prefetch.L2` instead of per-line hints.
- (b) Only for 5k: cap the tile's registers so one wo block fits beside it. Bitwise, but a tile slowdown risk.
## Main's CUDA host tests (brief 51, branch `ink/cuda-hosttests`)

Main a4e281264 had 4 red intarsia-backend-cuda host tests (CI does not run them). Fixed on the test, registry and ABI side.
No kernel code changes.
- `ink_attn_prefill_bf16` (op 75 above 16 rows, 86060ae6f) is renamed `ink_attn_decode_prefill_bf16`, after its op like its
  siblings, and declared in `ik_cuda_abi_ink.h`. Its launcher `ik_ink_attn_decode_prefill_bf16` is an alias of
  `ik_ink_attn_decode_bf16`, which takes T > 16 itself.
- The op 75 quick sweep gains a 40-row prefill chunk at G = 3 (window 64, ring 80), appended so the earlier cases keep
  their data.
- Once the first name was declared, the ABI test reached two more missing names, both from this lane's GEMV in_proj:
  `ink_in_proj_bf16_gemv_rows` and `ink_in_proj_p12_gemv_rows`. Both are now declared in `ik_cuda_abi_inkp12.h` as aliases
  of their launchers. A scan of every registry name against the headers finds none missing now.
- `rows_registry` ROWS_PINS: at k = 160, 5..=8 rows now pick mixmoe's `matmul_mxfp4_gemv_ink_rows` (table entry only).

Gate (fleetq #11810, spark1, 8f332fba1):
- All intarsia-backend-cuda tests, `--no-fail-fast`: 44 of 44 result groups ok; 197 passed, 0 failed, 26 ignored.
- Op 75 admission: quick, `ink_attn_decode_bf16` 16 cases and `ink_attn_decode_prefill_bf16` 1 case; full, 70 and 12
  cases. All ADMITTED.
- md5 64 / 256 / 1.2k: faf40cecb06b / 029e44fede1d / dc28658094c0, spec off and default, main's references.

## The screened greedy head on CUDA (brief 54 part 2, branch `ink/cuda-p12dec` off main 352acba93)

Sizing and choice: brief 54 part 1 (on `ink/cuda-attn`). The unembed P12 GEMV already runs at the streaming peak, so the
lever moves fewer bytes. An fp8 screen copy of 0.46 GB replaces reading the 0.69 GB P12 head, and only candidate rows get
rescored.

**Commits.**
- `73e0c8afb`: the kernels, opt-in.
- `7bdb0e71f`: a test fix.
- `24c3ed1a5`: the tuning row.
- `ce492db75`: the **CUDA default on**, a separate commit with a LOG row whose go is PENDING. `INTARSIA_INK_HEAD_SCREEN=0`
  reverts.

**What was built.**
- `kernels/cuda/ink_screen.cu`: `head_screen_f32` and `screen_argmax_f32`, the contract's exact f32 operations, bitwise the
  cpu reference. One 1024-thread block per row.
- `logits_gather_p12`: the serving P12 head kernel instance `p12_gemv_kernel<1, float>` in an index mode. The pointer is
  null in every serving call, and the gather launches on the head's grid, so a gathered logit is the head's.
- `crates/intarsia-backend-cuda/src/inkscreen.rs`: registration and dispatch.
- Changes in `model.rs::screened_argmax`:
  - The CUDA head names are accepted.
  - On CUDA the screen GEMV must be `matmul_fp8_gemv[_rows]` and runs with `exact_rows`.
  - The first-use probe is skipped inside a graph capture, because CUDA's decode graph records the first head.
- `INTARSIA_INK_HEAD_SCREEN_STATS=1` counts fallbacks on the device, graph replays included.
- `graph_step.rs`: the solo decode recording is keyed by (state, head, full logits). A `LastLogits` step records its head
  under `ScreenSuspend`, so it never replays the screened head, which leaves only its candidates' logits in `ws.logits`.
  Without this, default-on failed `inkling2_tiny_graph_decode_is_bitwise_the_eager_decode` (fleetq #11832). Regate
  #11838 on 53c4bd664: tiny inkling2 17/17, `inkling2_batch` 8/8, and md5 at 64 / 256 / 1.2k equal to main
  (`faf40cecb06b` / `029e44fede1d` / `dc28658094c0`) with screen on and off, spec off and default.

**Gate 1, the bound under CUDA's arithmetic.** The bound needs the screen values to come from a tree of correctly rounded
f32 adds and fmas over exact products, of depth at most D + 1, plus one scale multiply. CUDA's CUDA-core fp8 GEMV is exactly
that, so the bound was not widened:
- e4m3 decodes exactly (e4m3 to half to f32);
- in W8A16 the bf16 activations are staged unchanged, so each e4m3 × bf16 product is exact in f32;
- each lane runs 16-term fmaf chains, `acc += s`, then a 5-level warp tree: depth 16 + K/512 + 5;
- no fast math.

The tensor-core forms are refused, because their accumulation is not correctly rounded. The argument is written in the
header of `ink_screen.cu`.

**Gate 2, near ties** (`tests/inkling2_screen.rs`; both tests pass, #11827):
- 40 near-tied top rows (exact copies and one-ulp variants) against a cap of 8: the count exceeds the cap, the fallback
  gather writes all 4096 logits bitwise the serving head's, and the token is the full head's argmax.
- +inf bounds give a non-finite row and the full head's token.
- 3 near-tied rows within the cap take the candidate path. Every gathered slot is bitwise the head's logit for that row,
  every maximiser is a candidate, and the token matches.
- HeadScreen is bitwise `screen_row` in every case.

**Gate 3, fallbacks and cost on real prompts** (#11825, #11827):

| run | screened rows | fallbacks | mean candidates (path) | max |
|---|---:|---:|---:|---:|
| 5k prompt, 256 tokens | 256 | 0 | 2.02 | 33 |
| hash map, 2048 tokens | 2048 | 5 | 1.26 | 1398 |
| 1.2k prompt, 2048 tokens | 2048 | 0 | 1.18 | 32 |
| story, 2048 tokens | 2048 | 4 | 1.55 | 881 |

Screen cost per token from nsys (graphed, spec off; the same at short and 5k):

| kernel | µs / token |
|---|---:|
| fp8 GEMV (`fp8_gemv_rows_full_kernel`) | 1951 |
| `head_screen_kernel` (single block) | 166 |
| gather | 31.5 |
| ScreenArgmax | ~2 |
| **total** | **~2150** |

Against 2965 µs for the full head, about 0.8 ms per token is saved. A fallback step costs one full head on top (about 3 ms);
at 9 in 6144 steps that averages about 4 µs a token. nsys wall per token: 25.23 ms (screen off, #11823) against 24.54 ms
(screen on) at short.

**Gate 4, token identity.**
- md5 64 / 256 / 1.2k = faf40cecb06b / 029e44fede1d / dc28658094c0 (main's). That holds with the screen off, on, and on with
  cap 1 (which forces fallbacks), spec off and default.
- 3 × 2048-token greedy runs, screen on against off: identical ids (919ca9aaabc7, f0404a7b8833, 9b06e86d3aad).

**Gate 5, full head elsewhere.** The tiny `inkling2_tiny_screened_head_one_row_only` test now runs on CUDA and passes. It
checks:
- screen on equals off;
- forced fallbacks are counted;
- a 5-row verify does not call the screen;
- `LastLogits` (sampled) steps do not call the screen (assertion added).

`inkling2_tiny_screen_keeps_batching` also runs on CUDA and passes.

**Gate 6, tests.**
- All intarsia-backend-cuda tests: 199 passed. The one failure was my test's wrong assumption that the maximum is row 1000
  (a one-ulp-up row can win), fixed in `7bdb0e71f` and passing.
- The 3 ops are ADMITTED: head_screen 18 cases, logits_gather 6, screen_argmax 18.
- Tiny inkling2 on CUDA 17/17, batch 8/8, p12 5/5 (the head kernel's signature changed, its bits did not).
- Core screen tests 6/6 and `docs_env_vars` 2/2 (local).

**Speed, quiet A/B** (#11827, 3 reps, order alternated; decode tok/s, mean (min-max)):

| length | spec | screen off | screen on | change |
|---|---|---:|---:|---:|
| short (15 + 256) | off | 40.12 (40.06-40.16) | 41.42 (41.31-41.50) | **+3.23 %** |
| 5k (5039 + 128) | off | 38.33 (38.30-38.37) | 39.38 (38.81-39.69) | **+2.74 %** |
| short | default | 49.44 | 50.11 | +1.36 % |
| 5k | default | 50.86 | 51.23 | +0.73 % |

The ship bar (+1 % spec off, no regression) is met. Spec default gains less because verify steps keep the full head.

**Next:** HeadScreen in one block costs 166 µs a token. A multi-block form (per-block max and count, then an ordered write)
would take about 15 µs: about -0.15 ms a token (+0.6 %), bitwise.
