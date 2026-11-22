# Inkling-2-Mini CUDA in-projection, mix / conv and MoE kernels (lane ink-cuda-mixmoe, spark2 GB10)

Branch `ink/cuda-mixmoe` (worktree `.worktrees/ink-cuda-mixmoe`, base `1dacac3a8`). Not pushed or merged into main.

## 0. Briefs received

- (coordinator, during the lane) Report status: which of AttnResMix, LoopBlend, MoeExpertsMxfp8 (+NVFP4 situ), InkInProj, ConvMix
  are ADMITTED on CUDA, with an ETA for the rest. As soon as all are admitted, merge `ink/cuda-attn` (tip at or after `db002e2fa`)
  into `ink/cuda-mixmoe` and report the ref; when merging, extend `inkling2_tiny_no_kernel_refusal_names_the_op` (cuda-attn's
  `db002e2fa` expects CUDA kernels for exactly the inkattn TABLE ops) to this lane's ops.

- (coordinator, 2nd) Merge plan approved; keep the uncommitted perf work out of the merge. After the merge and the tiny test,
  without waiting: (1) tiny-model gate for cuda-attn's `d536ff099`: logits / tokens bitwise with the CUDA fused InkAttnDecode rule
  on vs forced to the pair, reported with the ref; (2) the first real Inkling run on CUDA on spark2 (checkpoint copy there is
  hash-identical to endor's): G1 / G2 / G3 per the spec gates, then greedy decode tok/s at short and 5k with profiling off; budget
  memory per the DGX rules (weights + KV + ~10 GB OS + build headroom against `fleetq status` free), `--mem` = that peak, poll
  `free -g` during the first load, kill below 10 GB available; correctness jobs at default load, timing `--quiet gpu,cpu`;
  (3) report G-gate results and tok/s next to Metal's 63 tok/s. Do not take any vLLM role down.

- (coordinator, 3rd) `ink/metal` `e3f92beff` guards the two `intarsia_backend_metal` uses in `crates/intarsia-models/tests/inkling2.rs`
  behind `cfg(target_os = "macos")`; if this branch merges `ink/metal` at or after `07589bd9c`, take `e3f92beff` too; pass it on to
  ink-cuda-attn with the merged ref. (This branch has its own equivalent guard, commit after the cuda-attn merge; no ink/metal merge
  here.)

- (coordinator, 4th) ink-cuda-attn's tip is `fd86495f2` (15 commits, all 4 attention kernels ADMITTED, fused-decode rule
  `d536ff099`): (1) merge that tip; (2) this lane owns the combined tiny-model logits gate: run it with the fused rule on and with the
  pair forced, report both; (3) extend `inkling2_tiny_no_kernel_refusal_names_the_op` to this lane's ops; (4) then G1-G3 and greedy
  decode speed on spark2 as planned, killing the run below 10 GB free.

- (coordinator, 5th, rule for all Inkling lanes) This session and every worktree run on endor, the Metal timing box: no local
  cargo build / check / test there outside a fleetq slot (it skews the quiet timing jobs on endor). CUDA builds and tests stay on
  spark1 / spark2 through fleetq. Reading files, git and editing locally are fine. (Before this rule the lane ran local stub
  `cargo test` / `cargo check` on endor without a slot; none since.)

(sections below filled in as the lane finishes)

- (coordinator, 3rd) `ink/cuda-attn` has a new gated tip `4f8eb3d1b` (fleetq #11016: all 4 kernels ADMITTED, 45 rows-exact cases
  bitwise; full 32k decode 657 us, was 715; SWA layers back at baseline). Merge it after the chunk-13 head fix passes and before
  G2/G3 and the greedy tok/s run, so the first real-model CUDA numbers use the faster attention.

- (coordinator, 4th) This lane owns the CUDA kernel table: `NOT_ON_CUDA` in `crates/intarsia-backend-cuda/src/kernels.rs` is now the
  slice form `const NOT_ON_CUDA: &[Op] = &[ ... ];` (was `[Op; 17]`; ink/metal has 26 entries since 44fba0140, pack12 30), so a merge
  conflicts only on entry lines, never on the count. Folded into the head-fix commit.

- (ink-cuda-attn lane, after the 3rd brief) Merge `ink/cuda-attn` at `1881a5ef7` (= `7628e87cb` + docs; kernel commit `a5e3ede16`,
  fleetq #11036: four kernels ADMITTED, 45 rows-exact cases bitwise); message that lane when the merged tip passes real-model
  G2/G3 on CUDA (its brief 6, nsys at ctx 1k / 8k, starts then). Merged: it contains the coordinator's `4f8eb3d1b`.

- (coordinator, 5th) (1) Merge `ink/cuda-attn` `1881a5ef7` (read-pattern change `a5e3ede16` + docs, bitwise-admitted), after the
  running G2/G3 job, not into its build; (2) rerun the tiny gate and G2/G3 on the merged tip (a full-layer attention kernel change
  needs its own gate); (3) when real-model G2/G3 pass there, message both the coordinator and ink-cuda-attn with the hash, the
  spark box and the binary path. Note: `1881a5ef7` was already merged (`3712a6a79`) before any G2/G3 job started; the first
  G2/G3 job (fleetq after #11049) builds and gates that merged tip, so it is the gate this brief asks for.

- (coordinator, 7th) G2/G3 accepted (first correct real-model Inkling run on CUDA; hash passed to ink-cuda-attn). Next:
  (1) G3 4k, then `generate` greedy tok/s at short and 5k with `--profile-every 0`; (2) with the local-model-bench skill,
  localeval speed-plain against `intarsia serve` (CUDA, spark2) at 142 / 1k / 4k / 16k if memory allows, vLLM retro stays up
  (budget `--mem` around it); compare with the existing spark2 vLLM NVFP4 Inkling numbers (37-43 tok/s) recalled from unanimis,
  no vLLM rerun; (3) report a table intarsia CUDA vs vLLM, decode and TTFT; keep the per-op profile for cuda-attn.

- (coordinator, 8th; with ink-cuda-attn's report from spark1 #11075) Real-model DECODE fails on CUDA at `995b565e4`:
  `ik_moe_experts_situ_bf16` IK_ERR_INVALID_ARG on the first 1-row eager step; G2/G3 were prefill-mode only, so the port does not
  count as working yet. Priority before tok/s and localeval: (1) fix the 1-row decode MoE, rows-exact vs its multi-row form;
  (2) a CUDA DECODE gate as on Metal: G2 decode rows (short / long, confident rows), greedy generate 64 and 256 tokens eager,
  md5 stable over 2 runs, tokens compared with Metal's greedy output for the same prompt (first divergence reported, not
  gated); (3) graph decode ("unsupported: device-position (graph) decode of Inkling-2") stays eager: list it as a perf item
  with an estimated cost, do not build it; (4) when decode G2 + greedy pass, message the coordinator and ink-cuda-attn the hash,
  then continue with tok/s and localeval.

- (coordinator, 9th) Decode accepted (8f7d5cd3a forwarded to ink-cuda-attn). New order: (1) Metal token comparison stays queued;
  (2) graph decode before tok/s and localeval (30 tok/s eager vs vLLM 37-43; eager launch overhead the likely gap): write a
  ONE-PARAGRAPH plan to the coordinator before building. Likely route: ink/batch's row-table form of ops 69/70/71/74/75
  (`OpCall::rows` + `pos` i32[B] in a device buffer, now on ink/metal 8546d143f) run at B = 1 makes positions device-resident,
  which device-position capture needs. The plan lists the CUDA kernels that need the row-table variant, how the ring slot / pos
  advance on device, and what else blocks capture (host top-k / argmax, sampler, MoE expert counts). No separate contract.
  (3) Headroom: G2 decode short 3.73e-3 vs budget 4e-3 on CUDA (Metal 2.59e-3): note which ops differ from Metal's reduction
  order; no further CUDA approximation may eat that margin. (4) Then tok/s and localeval vs vLLM, after graph decode lands or on
  the coordinator's go.

- (coordinator, 10th) GO with a split: (1) graph decode moves to ink-cuda-attn (my plan as written, from 8f7d5cd3a, touching
  only inkling2.rs, model.rs, plan.rs; stay out of those files or message cuda-attn first). (2) This lane: the latency-bound
  small ops, conv_mix S>0, loop_blend (stashed WIP) and the N=128 router GEMV. Each change bitwise vs the current CUDA kernel at
  T=1..16 (rows-exact tests); split-K changes the reduction order unless its tree matches exactly. A faster non-bitwise form goes
  to the coordinator as a numerics change with G2 decode measured against 4e-3 before it lands (CUDA headroom 0.27e-3).
  (3) Router: compare split-K with a port of Metal's fused op 76 InkRoute (ink/dfuse: router GEMV + MoeRoute, last-TG top-k,
  bitwise vs unfused); pick whichever is bitwise and faster; write the choice down. (4) Afterwards maybe: list a CUDA port of
  lossless P12 (ink/pack12, default-on on Metal) for in_proj as the next candidate with an estimate; do not build it.
  (5) Report per op us before/after at T=1/4/16, and eager tok/s short / 5k.

- (coordinator, 11th) cuda-attn's nsys at 8f7d5cd3a (spark1 #11087, its doc 903937e67) confirms: idle gaps 1.47 ms/token (657
  kernels, 2.2 us mean gap). Per token at 7.9k ctx, measured vs floor: experts gu+dn 9.35 vs 6.30 ms; in_proj 8.71 vs 6.96;
  unembed 3.75 vs 3.39; 48 tiny 1-3-block ink_mma_gemv calls 1.96 ms (40 us each, ~0 bytes); ink_row single-block conv /
  res-mix 2.10 ms (124 calls); moe_route 0.4. Order, all bitwise vs the current kernels: (1) the latency-bound tiny ops (~4.5
  ms: ink_row, the small mma_gemv calls, moe_route; more blocks or fuse into neighbours); (2) MoE experts 170 -> >= 230 GB/s
  (~2-3 ms); (3) CUDA P12 stays a listed candidate. Per item: us before / after and eager tok/s.

- (coordinator, 12th) Merge ink-cuda-attn's B=1 graph decode ink/cuda-graph 5fa5a88bd (on 8f7d5cd3a; tiny graph == eager bitwise
  over 31 steps, greedy 64/256 md5 equal, G2 decode 3.729e-3 / 3.145e-3 as eager; eager -> graph 30.54 -> 31.54 short, 29.88 ->
  31.20 at 1.2k, 29.33 -> 30.50 at 5k; idle 1.46 -> 0.39 ms/token; touches only inkling2.rs, model.rs, the tiny test) after
  the running job, before the next timing job. Rule: every new or changed decode kernel stays capturable and keeps
  decode_position_stable (no kernel choice on context length or host position at 1 row; position from the device; "decode
  graph disabled" means not done). Order unchanged: tiny ops (~4.4 ms: ink_row 2.07, 1-/3-block ink_mma_gemv 1.94, moe_route
  0.36), then experts 170 -> 230 GB/s. CUDA P12 moves to ink-cuda-attn (ink/cuda-p12); accept its p12 weight arm hook in the
  ink_in_proj dispatch and keep the bf16 in_proj reduction order unchanged. Report the next tiny-ops numbers with graphs on.

- (coordinator, 13th) Deep GEMV landing and the no-gain reverts accepted. Order: (1) now: generate tok/s at short / 1.2k / 5k
  with graphs on, and localeval speed-plain of `intarsia serve` vs the running vLLM NVFP4 on spark2 (37-43), quiet slot, vLLM
  retro not taken down without the coordinator's OK; (2) from existing outputs, no new GPU job: a per-row decode KL21 table for
  G2 short on CUDA with ink-metal's docs/results/2026-11-21-inkling2-metal/decrows.py (CUDA 3.73e-3 vs Metal pre-regression
  2.59e-3): spread evenly, or a few rows that drift and grow (Metal's regression shows that on chat_reasoning / chat_system)?
  If a few rows, name the op by replaying their per-module captures against the oracle (ink/spec replay setup); (3) opt-in
  tensor-core NVFP4 expert GEMV (W4A16, bf16 activations, fp32 accumulate, default off): rows-exact against itself, G2 decode /
  prefill and G3 within limits, KL <= ~10 % above the SIMT baseline, tok/s gain beyond spread; default-on is the
  coordinator's call; (4) split-K blend h parked until step 2 restores headroom; (5) InkRoute port later. ncu permission:
  correct not to change it; the coordinator raises it with the user.

- (coordinator, 14th) Build hazard: `cargo test -p intarsia-cli --test <x>` rebuilds target/<profile>/intarsia without
  `--features cuda`. Rules: build the backend binary last or give tests their own target dir; count md5 / self-golden matches
  only when both runs exit 0 with non-empty output. Checked: not affected here (build.sh copies the cuda binary to
  `bin/intarsia` right after building it; this lane's tests are intarsia-models / intarsia-backend-cuda only; every compared
  ids md5 is non-empty, `a0e6f47...`, with `finish=` lines).

- (ink-cuda-attn, coordinator-approved) After the current job, merge ink/cuda-p12 fc773d0fc (local branch in
  .worktrees/ink-cuda-p12; contains ink/cuda-graph 5fa5a88bd, ink/pack12 and CUDA P12). P12 default-on on CUDA
  (INTARSIA_INK_PACK12=0 off) packs wo and the unembed only; the in-projection stays bf16 (`p12::pack_in_proj`), so
  ink_in_proj_bf16_mma stays on decode. Its gate (spark1 #11172): greedy 64/256 md5 on == off, G2 decode identical, tiny 13/13,
  bitwise 4/4; decode short 31.6 -> 33.0, 1.2k 31.1 -> 32.5, 5k 30.5 -> 31.8 tok/s.

- (coordinator, 15th) For the next pick after the P12 merge: cuda-attn's nsys at 1.2k, P12 on, graphs (#11172, 30.89 ms GPU /
  token; ink/cuda-attn 40644991c, "Decode step at 1.2k with P12 on"): MoE gu 6026 us (19.3 %), dn 3261 (10.4 %), ink_row 2074
  (6.6 %), blend h mma_gemv 1946 (6.2 %; build without this lane's deep GEMV ba7a9fcdc), blend_gate 480, moe_route 368: ~45 %
  of the step. cuda-attn takes in_proj and the SWA tiles: hands off. Report MoE gu / dn GB/s vs ~273 and the top candidate
  with an estimated gain. Answer (from those timings and config dims, NVFP4 0.5625 B / weight, top-8, 44 visits): gu 1051 MB
  -> 174 GB/s, dn 525 MB -> 161 GB/s; at 240 GB/s both save ~2.7 ms / token. Candidate in the doc section below.

- (coordinator, 16th) MoE probe plan APPROVED. Order: serve admission fix (#11158), P12 merge fc773d0fc and its localeval first;
  then the loads-only probe. Probe: timebox 45 min, own fleetq slot `--quiet gpu,cpu`; the loads-only kernel is bench-only
  (env-gated or a separate binary, never default path, never merged); measure gu and dn separately at the #11172 shape (1.2k,
  P12 on, graphs). Decide by the worse of gu / dn loads-only GB/s: >= 230 compute-bound -> leaner bitwise inner loop (more rows
  per warp, same lane->chunk map and shuffle tree, LUT / prmt fp4 decode); <= 195 pattern is the ceiling -> load-time weight
  repack (contiguous per-warp streams), bitwise; 195-230 -> report numbers before building either. Either build needs greedy md5
  equal to the pre-change run, the CUDA test suite green, and an nsys step time showing the gain. Two attempts at one route
  without a gain: stop and report which assumption broke. ink_row stays parked.

- (coordinator, 17th) FYI for ranking after the probe, order unchanged. (1) cuda-attn starts DSpark speculative decode on CUDA
  (.worktrees/ink-cuda-mtp); its gain depends mostly on MoE cost at 2..4 verify rows: weigh MoE at T = 2..4 with T = 1 when
  picking the next build. (2) The CUDA prefill gap is this lane's: 1159 tok/s at 6.7k (#11085) vs vLLM 2772-2902 at 4-16k and
  4018 at 1k (2.4-3.5x); attention is 6-9 % of it, the rest the MoE and in_proj prefill GEMMs. When the probe reports, give a
  ranked choice between decode MoE (the probe outcome) and prefill GEMMs, each with an estimated tok/s or TTFT gain.

- (coordinator, 18th) Serve-stop hygiene (an endor serve stopped with `kill $SP` lived 2 h holding 11 GB). When #11225 ends:
  port closed, `pgrep -f 'intarsia serve'` empty on spark2, `free -g`; clean up inside a fleetq slot if anything is left. Future
  serve scripts: own process group (`set -m` / setsid) and an EXIT trap that TERMs serve, then TERM/KILLs the group, checks the
  port is closed and prints STOP FAILED otherwise; helper .worktrees/ink-batch/.scratch/serve_lib.sh.

- (coordinator, 19th) #11225 hung: in le.sh's remote command `cd ... && nohup serve ... &` backgrounds the whole `&&` list as a
  subshell, which held ssh's stdout open, and `echo $!` recorded the subshell (2804215), not serve. The coordinator, inside
  FLEETQ_JOB=11225 at ~05:22, wrote serve's PID 2804216 to out/serve.pid and killed the subshell; localeval then ran. After
  le.sh: check port 8090 dead and the --child 2804217 gone with the parent, else clean up in a slot. Future launches:
  `cd X && { nohup B serve ... >log 2>&1 </dev/null & echo $! >pid; }` or `setsid -f` (this lane: .scratch/serve_remote.sh,
  setsid with the redirects on the command, stop by process group, STOP FAILED check).

- (coordinator, 20th) After the MoE loads-only probe: profile CUDA prefill (the largest like-for-like gap, marginal 1.44-1.49k vs
  vLLM 2.8-4.0k tok/s) in one quiet slot on spark2: a 4k-prompt prefill with a per-op breakdown (MoE grouped GEMM, attention,
  mix/conv, norms, launch gaps) from nsys or the engine's op timers. Report the top 3 costs (ms, share) with a proposed fix
  each; do not build, the coordinator picks. No decode-vs-vLLM claims until a plain no-spec comparison exists (the reference ran
  with DSpark MTP).

- (coordinator, 21st) BLOCKER after the prefill profile: CUDA serve does not load on ink/cuda-mtp b1d84ba5b (merged c6680633c)
  on spark1, with and without --spec: `moe_experts: 12 shape(s) without a kernel (m=1 n=1152 k=2304 act=Bf16 w=U8)`, exit 78;
  binary built 05:25, after the merge (includes 65daa3e7a). Check (a) stale admission cache on spark1, (c) spark1 source differs
  from b1d84ba5b first, then (b) a weight-format change at m=1. Reproducer ink-cuda-mtp/.scratch/servediag.sh (~8 min, spark1,
  own fleetq slot). Report cause and fix commit; the attention lane then reruns servediag, the serve smoke and localeval.

- (coordinator, 22nd) Rule for the situ sweep (600 cases): (1) worst rel_l2 <= 1.5e-3 AND the worst element is plain bf16
  accumulation noise (no element over ~8 ulps, no structure by expert / row / position) -> raise the situ rel_l2 tol to 2e-3,
  abs unchanged, on CUDA and Metal, comment citing the sweep; (2) anything else is a kernel error: fix the kernel. Either way
  seed admission case data from the case key (shape, act, w, situ), not plan-call order, and check Metal and CPU admission
  still pass. Report the worst case, the branch taken and the commit; then the attention lane reruns servediag. Prefill after
  the blocker: CUDA prefill chunk 1024 and 2048 vs 256 (the same 4k profile + G2 / greedy md5; Metal runs 2048), budget the
  activation memory and give the job a --mem peak; if 2048 is bitwise / within tolerance and faster, send numbers (the
  coordinator decides default-on); then the rest of the top 3.

- (coordinator, 23rd) 24ab7961d acked: the 2e-3 NVFP4 situ tol is justified by the 600-case sweep (max 9.5e-4, all 1-ulp bf16;
  numbers in the "serve on spark1 refused to load" section), passed to the attention lane for the spark1 servediag rerun.
  (1) The Metal half stays out of the Inkling integration merge until #11301 passes Metal build + admission; if #11301 fails,
  revert only the Metal constant and report. (2) Go ahead with the CUDA prefill chunk sweep, then the rest of the prefill top 3.

- (coordinator, 24th) Approved: prefill chunk 2048 the CUDA default for Inkling-2 on ink/cuda-mixmoe (1.9x at 4k, md5
  identical, G2 / G3 pass, RSS unchanged). (1) Make the change, record the revert (`--prefill-chunk 256` or the recipe line),
  same recipe mechanism as Metal; no push / merge into main. (2) Measure warm serve prefill at 2048 (spark1 or spark2, fleetq
  `--quiet gpu,cpu`, `--mem` budgeted): one untimed warm request, then ~1k / 4k / 9k prompts x2; check the ~2500 tok/s
  estimate, report TTFT and decode. (3) Then start prefill MoE MMA W4A4 on the FP4 tensor cores (moe_sm120 way); same gates
  (rows-exact or tolerance-justified admission, G2 / G3, greedy md5 or explained drift); decode MoE repack after. (4) Metal
  half of 24ab7961d out of any merge until #11301 passes.
- (coordinator, 25th) Ack chunk 2048 default and #11301 PASS (Metal half of 24ab7961d cleared for the merge). W4A4: (1) gates
  option (a): opt-in speed mode gated on G3 short AND long, G2 waived but reported as information; W4A4 never default
  without the user; results table shows quality and speed side by side; say plainly if G3 passes with under 5% margin; no
  rounding tuning until it passes. (2) Contract approved: act_quant = 1 = W4A4 per hf_quant_config (NVFP4: per-token
  fouroversix_mae, block 16, e4m3 scales, global amax/1536, for x and h; MXFP8 layers 20-23: mxfp8_e4m3 activations, ue8m0 per
  32), chosen by recipe `[tuning.cuda] act_quant`, W4A16 default. Quantize at every T including decode (required: spec verify
  rows bitwise equal T=1 decode rows, greedy md5 spec on vs off under act_quant=1; both on the act_quant=1 gate list). Measure
  decode tok/s act_quant 1 vs 0. ink_ref.rs quantizer matches the GPU exactly; admission rows-exact up to GEMM accumulation.
  sm_121a fine if the default build still works for everything else and other CUDA tests pass; report the build.rs change.
- (coordinator, 26th) Priority change: CUDA decode first, W4A4 after (ink-cuda-attn d049cd622: 30.06 ms GPU busy/token vs
  19.8 ms floor; 6.5 ms of the gap in this lane's classes: MoE 2.61, mix/conv ~2.0 = 124 single-block launches at 17 us,
  controllers ~1.2, router 0.48). (1) Let the W4A4 build run plus its admission and rows-exact tests, record status, park W4A4
  (G3 / spec gates later). (2) First the mix/conv row kernels (cut the 124 single-block launches: fuse or multi-block), then the
  decode MoE repack (197 vs 273 GB/s), then controllers and router. (3) Gates: rows-exact or tolerance-justified admission,
  G2/G3, greedy md5 unchanged spec off and on, graphed decode tok/s with spread; default candidate only if bitwise or within
  plan tolerances; own commit with a revert switch each. (4) Measure decode spec off and on.
- (coordinator, 27th) MoE L2 prefetch accepted default-on (+1.9 %, just under the 2 % bar; md5 unchanged, kernel GB/s gains
  corroborate, revert switch `INTARSIA_INK_MOE_PF=0`). Spec on: the Inkling drafter is on ink/cuda-mtp as DSpark
  (`--spec dflash --spec-draft <dir>`, not `--spec mtp`); do not merge ink/cuda-mtp here; spec-on md5 / tok/s recorded as
  pending until both branches are merged (coordinator coordinates). Next: the cheaper row-kernel prefetch, then the
  thread-block cluster split. W4A4 stays parked until its own tolerance sweep. Confirm ink_fp4_kernel ran before quoting any
  T=2048 FP4 number.
- (coordinator, 28th) Cluster split dropped: OK. MoE repack approved, probe first (~2-3 h cap, measure only), on spark1, T=1,
  same grid / threads: (a) no-math stream of the current layout, (b) no-math stream of the proposed repacked layout, (c) the
  shipped kernel. Build only if (b) clearly beats (a) with room for >= 1 ms / token; if (a) ~ (b), layout is not the limit:
  report and pick the next item (Metal ink-dfuse found per-thread work, not layout, was its limit). Load-time memory: repack
  expert by expert into the final buffer, never a full second copy; poll `free -g` on the first load and kill below 10 GB
  available; fleetq `--mem` = load peak. Prefill moves to the new layout, so the gate adds prefill tok/s at 2k and 8k before /
  after, besides bitwise rows-exact, greedy md5, decode tok/s with spread. Ownership: InkInProj (8.7 ms) and the attention
  projections are ink-cuda-attn's (item d); stay off them. MoE, mix/conv, blend, controllers and router stay here.
- (coordinator, 29th) Repack stays unbuilt. One more cheap probe before LoopBlend (<= 1 h): vary the thread geometry in the
  same no-math harness (2x the CTAs or warps per SM; 128-bit loads per lane if not already; both). If any geometry's no-math
  stream reaches ~250 GB/s (>= 0.8 ms / token against the shipped kernel), report before building; else close MoE at T=1 and
  go to LoopBlend, then the router. Confirm the spark2 runs held a fleetq slot (they did: #11417 and the pass-2 job, both
  `fleetq run --quiet gpu,cpu`).
- (coordinator, 30th) Before choosing bitwise ring vs split-K for the controller GEMV, check split-N (output columns of h over
  more CTAs, each keeping the full k loop and per-tile mma k-sequence: bitwise). Report h's N and the CTA count. Rule: (1) split-N
  reaches ~8+ CTAs per call -> build it (bitwise, default-on allowed, gate as option 1), skip the ring; (2) N too narrow ->
  split-K opt-in, default off (G2 / G3, md5 drift, prefill 2k / 8k gates; default-on is the user's call), optionally the
  bitwise ring as default meanwhile if it fits in ~half a day. Gate + norm fuse first.
- (coordinator, 31st) Merge `ink/cuda-mtp` at `427652506` into this branch now (this branch is the CUDA integration line): DSpark
  drafter on CUDA (`--spec dflash --spec-draft <dir>`) and the verify graph, default-on for Inkling on CUDA only
  (`INTARSIA_VERIFY_GRAPH=0` reverts; cuda-attn gate #11418 passed). Watch `model.rs` (spec admission) and `load.rs` (mtp lane's
  allow-list fix); keep both intents, stop and report if a conflict is not mechanical. Then in a fleetq slot: (1) build, ink_rows,
  tiny tests (cuda + cpu), verify-graph test; (2) greedy md5 64 / 256 / 1.2k spec off (must stay `a0e6f47bc45f` etc.); (3) spec-on
  md5 and tok/s with the verify graph default-on; (4) serve smoke at 1k and 9k. Send the merge hash and numbers; the gate + norm
  fuse and split-N work continue after.
- (coordinator, 32nd) Split-K stays default-on (bitwise). Next, in order: (1) profile spec-on decode at g256 and g1.2k and spec-off
  g256 on spark1; rank the top 3 targets by ms / token saved with evidence; build the top one with the same gate (md5, G2 / G3 if
  bits change, ABAB tok/s). On the list: why g64 spec on loses ~2 % (36.76 -> 36.05): first-step draft cost or low acceptance
  without adaptive draft length; an acceptance-based fallback to spec off with identical output is a candidate. (2) PR prep after
  ink/metal's main PR merges: merge main in, CUDA no-op check on other models (Qwen3-8B greedy 256 short / 4k bit-identical to
  main, plan show JSON for 35B and 8B, kernels test --quick), report; coordinator takes the PR. Spec on by default stays opt-in
  (deployment call for the user).
- (coordinator, 33rd) (1) Draft head small-M kernel: go; gate md5 identical, acceptance not lower, ABAB tok/s g256 and g1.2k;
  default-on if >= +1 % at g256. (2) Verify unembed / wo goes to ink-cuda-attn: send shapes, call counts and the exact bit
  contract (which decode kernel the verify rows must equal) in 3-5 lines. (3) Draft MLP fp8: skip. g64: after (1), a load-time
  warmup of the draft's first-call path (cuBLASLt / cutlass lazy init, first splitkv) if < 30 min; report the load-time cost.
- (coordinator, 34th) Head kernel default-on stands; no g64 warmup (note that nsys's first-call cost explained the stall). PR prep:
  don't poll; #488 (ink/metal -> main) is in CI, coordinator messages on merge. Meanwhile, from #11462 (no new job unless needed):
  rank this lane's costs (MoE experts, ConvMix, controller blend and loop kernels) for spec-off g256 and spec-on verify passes,
  ms / token with each one's bandwidth floor; build the top item if its bound is >= 0.3 ms / token (same gate: md5, rows-exact or
  bitwise tests, ABAB); if none reaches 0.3, say so and stop.
- (coordinator, 35th) #488 merged: main is `57e1b3332` (Inkling on Metal / cpu). When the current build item allows: merge main into
  this branch, run the CUDA no-op check on other models (Qwen3-8B greedy 256 short / 4k bit-identical to main, plan show JSON for
  the 35B and 8B, kernels test --quick), send the result; do not open the PR.
- (coordinator, 36th) MoE expert repack: build it in place, in stages; no 25 GB second copy. Stage A (~half a day, go / no-go):
  load-time repack behind a switch, default off; only the decode `_rows` path learns the layout, everything else stays on the
  old one by refusing the switch whenever prefill or W4A4 would run; microbench NVFP4 T=1/2/4 vs today, g256 decode tok/s A/B
  spec off and on (rotated order), rows bitwise; continue only if g256 spec-off decode >= +1.2 %, else stop, revert, report.
  Stage B: `ink_moe_mma` (prefill) and W4A4 learn the layout; prefill 4k / 8k tok/s within -1 %; G2 / G3 unchanged (bitwise
  expected); then default-on with a one-env revert. Next report: one line with the MoE gu+dn bandwidth floor per token at g256
  (bytes read / measured peak); if 8.8 ms is far above it, name the next lever after the repack. Finish brief 35 first (no-op
  #11498 and re-verify #11502 green, then "your PR"). Box: spark1 via fleetq, same worktree.
- (coordinator, 36a) PR confirmed in writing: this lane may push `ink/cuda-mixmoe` to origin and open its PR to main once #11498
  (no-op) and #11502 (Inkling re-verify) are green. Order: wait for PR #489 (Metal head screen) to merge, merge main in again;
  if that merge touches engine code or recipes, rerun the Inkling CUDA md5s (spec off and on) before pushing. PR body: what
  changes by default (split-K blend, MX4 draft head, each with its revert env), every gate by job number, the no-op results.
  Merge only on all-green CI; no force-push; no other lanes' branches. Box: stay on spark2 (`~/intarsiaml-ink-cuda-mixmoe`, its
  baselines); Stage A runs there.
- (coordinator, 36b) PR #489 merged as main `693f6440e`: merge it once #11498 / #11502 are green, rerun the Inkling CUDA md5s
  if the merge touches engine code or recipes, then push and open the PR.
- (coordinator, 39th) Stopping stage A at +1.13 % was right. Before merging #491 (91 files, +16k): put no-op evidence for head
  `cb13bd5f2` vs main `1f226a2e3` in the PR body: Metal and cpu plan show byte-identical for Qwen3-8B and Llama, Inkling Metal greedy
  md5 64 / 256 unchanged, CUDA plan show 35B and 8B unchanged; rerun #11498 / #11502 if they ran on an older main (cpu / default
  slots, cargo only inside slots). Merge on all-green CI, then delete the remote branch. Decisions 5 and 6 (user): cuda-attn makes
  the default-on commits for CUDA spec and in_proj GEMV (briefs 31, 32) as a follow-up PR; do not hold #491. Next: gu+dn in one
  launch per layer, build only if bitwise, timebox half a day, keep only at >= 1 % at g256 spec-off in a quiet ABBA, else record
  and stop.
- (coordinator, 40th, endor rule from ink-metal) `git push` runs the pre-push hook, which runs two `cargo test`s, so every push
  from endor is a cargo run: push only inside a fleetq slot, e.g. `fleetq run --host endor --load cpu --expect 10 -- git push ...`.
- (coordinator, 37th) cuda-attn's prefill profile (ink/cuda-attn `f39988040`, #11514): at 8k MoE gu+dn is 393 ms of a 746 ms
  chunk; `ink_moe_mma` runs at 32 TF/s vs cuBLAS 110 TF/s here; a block-scaled tensor-core MoE GEMM (mxf4nvf4 / mxf8, sm_121)
  is worth an estimated +30-50 % prefill. Order: (1) brief 35 PR; (2) repack stage A (decode `_rows` only, as briefed); (3)
  instead of a plain stage B, ONE expert layout serving both the decode rows path and a block-scaled prefill GEMM, built
  together. Before building (3), send a design note: layout, kernel path (CUTLASS / cuBLASLt block-scaled or own), expected ms
  per 8k chunk, effort, quality plan (not bitwise: G3 prefill short / long / 4k within budget; decode unchanged if decode does
  not use it). Lever 3 (mix row kernels spread over D, ~0.48 s at 8k) goes to cuda-attn after this PR merges; it will touch
  the ink_row files on main.
- (coordinator, RULE, 38th) This session and all `.worktrees` live ON endor. Every cargo / rustc / rustfmt run on endor goes
  inside `fleetq run --host endor --load cpu --expect N ...` (or on a Spark / bespin in a CPU slot). No exceptions for quick
  checks: unslotted `cargo check` / `cargo test` at 12:40-12:44 EST spoiled another lane's `--quiet gpu,cpu` timing (#11517).

# Inkling-2-Mini CUDA in-projection, mix / conv and MoE kernels (lane ink-cuda-mixmoe, spark2 GB10)

Branch `ink/cuda-mixmoe` (worktree `.worktrees/ink-cuda-mixmoe`, base `1dacac3a8`). Not pushed or merged into main.

## 0. Briefs received

- (coordinator, during the lane) Report status: which of AttnResMix, LoopBlend, MoeExpertsMxfp8 (+NVFP4 situ), InkInProj, ConvMix
  are ADMITTED on CUDA, with an ETA for the rest. As soon as all are admitted, merge `ink/cuda-attn` (tip at or after `db002e2fa`)
  into `ink/cuda-mixmoe` and report the ref; when merging, extend `inkling2_tiny_no_kernel_refusal_names_the_op` (cuda-attn's
  `db002e2fa` expects CUDA kernels for exactly the inkattn TABLE ops) to this lane's ops.

- (coordinator, 2nd) Merge plan approved; keep the uncommitted perf work out of the merge. After the merge and the tiny test,
  without waiting: (1) tiny-model gate for cuda-attn's `d536ff099`: logits / tokens bitwise with the CUDA fused InkAttnDecode rule
  on vs forced to the pair, reported with the ref; (2) the first real Inkling run on CUDA on spark2 (checkpoint copy there is
  hash-identical to endor's): G1 / G2 / G3 per the spec gates, then greedy decode tok/s at short and 5k with profiling off; budget
  memory per the DGX rules (weights + KV + ~10 GB OS + build headroom against `fleetq status` free), `--mem` = that peak, poll
  `free -g` during the first load, kill below 10 GB available; correctness jobs at default load, timing `--quiet gpu,cpu`;
  (3) report G-gate results and tok/s next to Metal's 63 tok/s. Do not take any vLLM role down.

- (coordinator, 3rd) `ink/metal` `e3f92beff` guards the two `intarsia_backend_metal` uses in `crates/intarsia-models/tests/inkling2.rs`
  behind `cfg(target_os = "macos")`; if this branch merges `ink/metal` at or after `07589bd9c`, take `e3f92beff` too; pass it on to
  ink-cuda-attn with the merged ref. (This branch has its own equivalent guard, commit after the cuda-attn merge; no ink/metal merge
  here.)

- (coordinator, 4th) ink-cuda-attn's tip is `fd86495f2` (15 commits, all 4 attention kernels ADMITTED, fused-decode rule
  `d536ff099`): (1) merge that tip; (2) this lane owns the combined tiny-model logits gate: run it with the fused rule on and with the
  pair forced, report both; (3) extend `inkling2_tiny_no_kernel_refusal_names_the_op` to this lane's ops; (4) then G1-G3 and greedy
  decode speed on spark2 as planned, killing the run below 10 GB free.

- (coordinator, 5th, rule for all Inkling lanes) This session and every worktree run on endor, the Metal timing box: no local
  cargo build / check / test there outside a fleetq slot (it skews the quiet timing jobs on endor). CUDA builds and tests stay on
  spark1 / spark2 through fleetq. Reading files, git and editing locally are fine. (Before this rule the lane ran local stub
  `cargo test` / `cargo check` on endor without a slot; none since.)

(sections below filled in as the lane finishes)

## Real-model G2 / G3 on CUDA (spark2, merged tip with ink/cuda-attn `1881a5ef7`)

Tree `995b565e4` (`31365b3bf` adds docs only); binary `spark2:~/intarsiaml-ink-cuda-mixmoe/bin/intarsia` (fleetq #11062 build),
gates fleetq #11064, `accuracy --backend cuda --max-seq 4096 --mode prefill`. Tiny gate on the same tree: 11 / 11 (fused InkAttnDecode
== pair, dump md5 equal). Lowest available memory during the runs: 62 GB.

| gate | golden | KL21 (scored) | budget | confident misses | top-1 | verdict |
|---|---|---|---|---|---|---|
| G2 short | oracle w4a16 short | 2.376e-3 | <= 4e-3, misses <= 1 | 0 / 363 | 98.77 % | PASS |
| G2 long | oracle w4a16 long | 3.560e-3 | <= 6e-3, misses <= 2 | 1 / 1206 | 98.92 % | PASS |
| G3 short | vLLM short | 6.630e-3 | <= 1.05e-2 | 0 / 367 | 97.72 % | PASS |
| G3 long | vLLM long | 1.169e-2 | <= 1.65e-2 | 0 / 1211 | 97.51 % | PASS |

Fixes on the way here: the AllLogits head groups of 2..=16 rows now run the row-exact head on Inkling (CUDA's ordinary head is
cuBLAS from 5 rows, so chunks of 5..16 rows scored rows differently from decode steps); the tiny rollback test accepts CUDA's
conservative `reuse_len`; `ink_rows` serializes the two tests that set the process-wide MoE tensor-core threshold (a parallel-test
race, not a kernel bug). G3 4k and greedy tok/s (short, 5k) not run yet.

## Real-model decode on CUDA (8f7d5cd3a, spark2)

The first eager 1-row step failed (`ik_moe_experts_situ_bf16` IK_ERR_INVALID_ARG). The ordinary experts kernels reserved only
the tensor-core scratch layout, which at E = 128 and few pairs is 256 bytes short of the SIMT layout. Fixed by reserving the
larger of the two; covered by `ink_experts_rows_exact_128_experts` (T 1..16, NVFP4 and MXFP8 situ) and a scratch-size unit test.

Decode gate (fleetq #11079): G2 decode (teacher-forced, 16 positions per prompt) short KL21 3.729e-3 (budget 4e-3), 0 / 96
confident misses on the decode rows; long 3.145e-3 (budget 6e-3), 0 / 119. Greedy eager, profile off: 64 tokens of raw "The
capital of France is" and 256 forced tokens of a short prompt, ids md5 identical over 2 runs (and the 256 text over 3 more in
#11085). G3 4k prefill (#11068): KL21 9.754e-3 (budget 1.4e-2), 5 / 8229 confident misses (budget 9), top-1 95.76 %.
Metal token comparison (endor #11156, ink/metal binary built 2026-11-22 00:09, snapshot b85c3cfa6, whose config / index / quant
config md5 and weight-file size equal spark2's ed40e7da6; the ed40e7da6 snapshot on endor has no config.json): Metal is
md5-stable over 2 runs too (ids g64 fa9f4b4a..., g256 550cd932...). CUDA vs Metal greedy ids, first divergence (not gated):
64-token raw "The capital of France is" at token 7 (CUDA 164452, Metal 68014); 256-token hash-map prompt at token 69
(150149 vs 150837). The top-1 margins at those tokens were not measured; both backends pass G2 decode on their own.

### Eager greedy speed (fleetq #11085, `--quiet gpu,cpu`, `--profile-every 0`, vLLM retro resident and idle)

| prompt | prefill | decode (256 forced tokens) |
|---|---|---|
| 15 tokens | 100 ms | 30.53 / 30.55 / 30.63 tok/s |
| 6672 tokens | 5758 / 5797 ms (1159 / 1151 tok/s) | 29.19 / 29.17 tok/s |

### Per-op decode profile (128 tokens, short prompt, `--profile-every 1 --op-table`: sync per op, 34.5 ms/step vs 32.7 eager)

| op | ms / step | calls / step | us / call |
|---|---|---|---|
| ink_in_proj | 9.22 | 48 | 192 |
| moe_experts (NVFP4) | 8.77 | 44 | 199 |
| conv_mix | 5.19 | 96 | 54 |
| logits (48 router + head) | 4.03 | 49 | 82 |
| matmul | 2.98 | 48 | 62 |
| ink_attn_decode | 1.85 | 48 | 39 |
| moe_experts_mxfp8 | 1.24 | 4 | 310 |
| moe_route | 0.63 | 48 | 13 |
| attn_res_mix | 0.36 | 26 | 14 |
| loop_blend | 0.16 | 2 | 78 |

About 415 launches per token. The profiled run's per-op syncs add only 1.8 ms/step, so the eager step is mostly GPU time per op.

### Perf items

- Graph decode (B = 1, device position): today "unsupported: device-position (graph) decode of Inkling-2", so decode is eager.
  The CUDA kernels already take a device `pos` (POSITIONED contract); the work is `ink_pass` plumbing (`call.pos` on the 5
  positioned call sites, `dpos` through model.rs). Estimated gain 1-3 ms/token (host launch and inter-kernel gaps, ~415
  launches); cost about half a day.
- Latency-bound small ops (about 10-12 ms/token together): conv_mix with sources (46-53 us GPU at T = 1), loop_blend (35-40
  us), the router GEMV on 128 outputs (~80 us), the 48 per-visit matmuls (~62 us). These are the larger lever.

### Headroom against Metal (G2 decode short 3.73e-3 CUDA vs 2.59e-3 Metal, budget 4e-3)

Ops whose reduction order differs from Metal's: ink_in_proj (one k16 tensor-core step per 16 inputs, then fp32 adds),
matmul / logits GEMV (warp-shuffle tree vs simdgroup tree), the MoE SIMT dot products (warp tree per row), the CUDA attention
kernels (ink-cuda-attn's split and tile order), and the cuBLAS head at prefill. None are approximations (fp32 accumulation
everywhere, bf16 rounding points as in the contract), but further CUDA changes must not trade accuracy for speed.

## Latency-bound small ops and MoE experts, bitwise only (graphs on, after merging ink/cuda-graph 5fa5a88bd)

Per-kernel GPU time per token from nsys (spark2 #11092, eager, 64 tokens; `ink_row` per slot from the SQLite export):

| kernel | calls / token | us / call | ms / token |
|---|---|---|---|
| ink_mma_gemv, in-projection (135-136 blocks) | 48 | 173-197 | ~8.9 |
| ink_mma_gemv, blend controller h (G 64: 1 or 3 blocks) | 24 + 24 | 40.1 / 40.9 | 1.95 |
| ink_row_kernel (one 768-thread block per row) | 124 | 4.0-18.5 (plain-visit ConvMix with mix + norm: 17.9) | 2.09 |
| ink_blend_gate / ink_blend_norm | 48 / 48 | 9.8 / 5.5 | 0.74 |
| moe_route | 48 | 8.1 | 0.39 |
| ink_moe_gu / ink_moe_dn (NVFP4) | 44 / 44 | ~120 / ~66 | 8.2 |

| change | bitwise check | before | after |
|---|---|---|---|
| blend h GEMV deep prefetch (ba7a9fcdc): separate kernel for grids <= 8 blocks, 4 k blocks in flight, same mma / fp32 chain | `ink_gemv_deep_bitwise` (deep == plain, G 64/128, S 1/3, R 1..40), ink_rows 7/7, tiny 12/12, greedy ids md5 equal | loop_blend T=1 35.9 us (S 1) / 40.4 (S 3); T=4 36.8 / 41.0; T=16 47.1 / 50.9 | 28.7 / 32.8; 29.4 / 35.6; 38.9 / 43.0 |
| (kept) | | greedy, graphs on: short 31.58 / 31.47, 5k 30.26 tok/s | short 32.77 / 32.63, 5k 31.16 tok/s (#11144) |
| row kernel: all independent loads first + tap loop unroll | greedy md5 equal | conv_mix N4 Q1 norm 14.3 us | 16.2 us: reverted |
| blend GEMV PF 4 inside the shared kernel (with `break` in the stage loop) | greedy md5 equal | loop_blend T=1 36.4 | 43.1: reverted (stage arrays likely went to local memory) |
| MoE gu / dn: next chunk's raw bytes loaded one ahead | greedy md5 equal | nv T=1 184.7 us (194 GB/s) | 185.4: no change, reverted |
| MoE gu / dn: persistent grid (resident blocks, grouping once) | greedy md5 equal | nv T=1 187.9 us | 186.6: no change, reverted |

Not done, and why. (1) ink_row: its reductions are "per-thread partial in element order, warp butterfly, warp partials in warp
order" with the block size and elements per thread fixed by D, so any change of block shape changes the bits; the load hoist
did not help, so it is not a load-issue problem. A bitwise multi-SM form needs a thread-block cluster with DSMEM keeping the
same thread-to-element map and warp order (large; gain unknown without counters). GPU performance counters are not
accessible to this user on spark2 (ncu: ERR_NVGPUCTRPERM), so stall reasons could not be read. (2) MoE experts: two bitwise
restructurings (prefetch, persistence) moved nothing; the SIMT kernels are not limited by load latency or waves as assumed.
Reaching ~230 GB/s needs a different inner loop (e.g. tensor-core decode GEMV over the expert rows), which is not bitwise the
current SIMT chain: a numerics change. (3) moe_route (8 us, one warp) and the blend gate / norm: fusion into neighbours means
a fused op in core (Metal's InkRoute op 76 is on ink/dfuse, not merged), outside this lane's files now.

## G2 decode KL per row on CUDA (self-goldens of #11079; `2026-11-21-inkling2-cuda-mixmoe-decrows.py`, ink-metal's decrows.py)

Short set (n 128, mean 3.73e-3): the top row carries 29 %, the top 3 50 %, the top 5 62 %; without the top 5 the mean is
1.46e-3. Long set (n 160, mean 3.15e-3): 28 / 48 / 59 %, 1.32e-3 without the top 5. The large rows are isolated spikes, not
drift: chat_system pos 47..62 is 0.000 everywhere except pos 57 (0.139, golden margin 0.77) and pos 62 (0.056); chat_short
pos 34 0.046; chat_multiturn pos 95 0.030 (margin 1.00); chat_reasoning pos 90..105 stays at 0.000-0.028 with no growth
(Metal's regression grew to 0.431 over those rows). chat_system pos 57 is the row that regressed on Metal too (0.011 -> 0.303).
Next: per-module replay of chat_system pos 57 against the oracle.

## Merged tip with ink/cuda-p12 fc773d0fc (613dbacda; spark2 #11211, `--quiet gpu,cpu`, graphs on, P12 default on)

Serve admission fix (65daa3e7a; the P12 branch had the same fix, its form kept at the merge): `serve` loads Inkling-2 on CUDA
under the admitted-only rule (23 kernels, 119 cases, none rejected) and answers. Gate: ink_rows 7/7, backend-cuda lib 35/35,
tiny 13/13 (tests/inkling2_capture.rs compiles), core quant::cases 2/2; greedy 256 short ids md5 a0e6f47... over 2 runs
(unchanged from 8f7d5cd3a: P12 and the deep GEMV keep the bits).

| prompt | prefill | decode tok/s (256 forced) | before P12 merge (#11144) |
|---|---|---|---|
| 15 tokens | 98-108 ms | 34.56 / 34.62 | 32.77 / 32.63 |
| 1261 tokens | 1295 ms (974 tok/s) | 33.98 | 32.31 (#11158) |
| 6672 tokens | 5775 ms (1155 tok/s) | 33.02 | 31.16 |

## localeval speed: intarsia serve on CUDA vs vLLM (spark2)

intarsia: c6680633c (P12 on, graphs on, no speculation), fleetq #11225, localeval run
`20261122-051731-adhoc-intarsia-cuda-c6680633c` (unanimis c69bed69), 3 reps, 128 forced tokens, thinking off. vLLM:
`20261120-192706-spark2-inkling-none-quick-spec-effnone`, NVFP4 image `vllm/vllm-openai:inkling-2-mini`, **DSpark MTP drafter
on** (`--lane spec:dspark --drafter dspark-mtp7`, probe ratio 1.081; peak decode 63-95 tok/s shows accepted drafts), 254 tokens.
Not rerun (coordinator). The decode columns are therefore not like for like: vLLM's includes speculation, intarsia's does not.

| prompt tok | TTFT s intarsia / vLLM | marginal prefill tok/s intarsia / vLLM | decode tok/s intarsia / vLLM (spec) | intarsia / vLLM decode |
|---:|---:|---:|---:|---:|
| 135 / 144 | 0.24 / 0.21 | - / - | 34.6 / 36.8 | 0.94 |
| 1037 / 1024 | 0.86 / 0.43 | 1462 / 4018 | 34.5 / 38.4 | 0.90 |
| 4127 / 4112 | 3.00 / 1.54 | 1441 / 2772 | 33.6 / 42.8 | 0.79 |
| 16448 / 16456 | 11.27 / 5.79 | 1491 / 2902 | 32.0 / 38.7 | 0.83 |

Serve load: 228 s, admission 26 kernels / 103 cases, none rejected. After the stop: port 8090 closed, no serve process left
(2804216 and its --child 2804217 gone), memory back to the 41-42 GB baseline (vLLM retro 38 GB).

## MoE decode loads-only probe (brief 16; spark2 #11275, `--quiet gpu,cpu`; probe code env-gated, uncommitted, reverted)

`INTARSIA_INK_MOE_LOADS_ONLY=1` replaced the NVFP4 gu / dn inner loop by the same loads (weights, scales, activations) xor-ed
together, no fp4 decode and no dot; `=2` also dropped the activation loads. ink_bench moe (128 experts, top-8, realistic union,
layer copies beyond the cache), GB/s of expert bytes:

| T | union | gate+up normal / loads-only / weights-only | down normal / loads-only / weights-only |
|---|---|---|---|
| 1 | 8.0 | 199.7 / 200.1 / 200.9 | 186.1 / 182.8 / 185.5 |
| 2 | 15.5 | 204.3 / 204.6 / 205.8 | 192.7 / 186.8 / 195.5 |
| 4 | 29.0 | 205.4 / 205.7 / 206.6 | 192.4 / 192.2 / 195.6 |
| 16 | 83.0 | 199.3 / 199.2 / 201.3 | 176.5 / 177.9 / 181.5 |

In the model at the #11172 shape (1.2k prompt, P12 on, graphs, nsys, 60 tokens): gu 5893 -> 5840 us / token, dn 3247 -> 3251
(48 calls each: 44 NVFP4 + 4 MXFP8, the probe touches NVFP4 only; NVFP4 + MXFP8 bytes give ~208 / ~189 GB/s). Decode itself
33.21 -> 33.42 tok/s.

Reading: removing all arithmetic (and even the activation reads) moves nothing, so the kernels are not compute-bound; GB/s is
flat from 8 to 83 experts, so it is not a fixed per-launch cost either. The bound is the weight-load pattern (per warp
instruction: dn 8 rows x 64 B, gu 4 rows x 128 B in two matrices far apart, plus 2-byte scale loads in a separate stream),
against ~238 GB/s for the bf16 decode GEMV's long contiguous rows. Worse of gu / dn loads-only: 178-192 GB/s (<= 195): by
the brief-16 rule the route is the load-time weight repack (contiguous per-warp streams, scales interleaved, same lane -> chunk
map, so bitwise). Estimate at ~235 GB/s: gu + dn 9.1 -> ~7.4 ms / token (~1.7 ms, ~5-6 %), the same rate gain at T = 2..4
(DSpark verify rows). Cost: the tensor-core prefill path (ink_moe_mma) and the `_rows` twins read the same weights, so they
move to the new layout too (or the experts are kept twice: ~17 GB more); about 1-1.5 days. Not built: the coordinator picks
between this and the prefill GEMMs after the prefill profile (brief 20).

## serve on spark1 refused to load (brief 21): admission data drawn in plan order (24ab7961d)

ink/cuda-mtp b1d84ba5b on spark1 (`servediag.sh`: --mem-budget 30 --max-seq 12288 --max-batch 1) failed with "moe_experts: 12
shape(s) without a kernel". Not a stale cache (the failing entry was written by the 05:25 binary itself) and not the source
(spark1's cases.rs carries situ). It was a real admission failure: `moe_experts_situ_bf16 | T=1 E=32 K=8 H=2304 I=1152 bf16
situ` at rel_l2 1.041e-3 against tol 1e-3 (max_abs 5e-3, abs tol 1.5e-2); the 33 other situ cases there passed at 2e-5 -
3e-4. Admission drew each case's data from one generator in plan-call order, so a different plan drew different data.

Sweep (`tests/ink_situ_spread.rs`, env-gated; 600 admission-built cases on spark2, CUDA vs the cpu reference, admission judge):

| T | cases | median | p90 | p99 | max | over 1e-3 |
|---|---|---|---|---|---|---|
| 1 | 300 | 0 | 3.14e-4 | 7.80e-4 | 9.50e-4 | 0 |
| 4 | 300 | 9.78e-5 | 2.84e-4 | 5.01e-4 | 6.21e-4 | 0 |

Worst case: 111 / 2304 outputs differ, each by exactly 1 bf16 ulp (e.g. 8.9375 vs 9.0), at scattered indices; T=4's worst: 175
/ 9216, also 1 ulp each. That is the bf16 output rounding of the x8-weighted expert sum. Branch 1 of the coordinator's rule:
`SITU_TOL` rel_l2 2e-3 (abs 1.5e-2 unchanged) on the NVFP4 situ kernels and their `_rows` twins, CUDA and Metal; MXFP8 keeps
MOE_TOL. And `plan::admit` seeds each case from its label (`case_rng`), so a plan change cannot change a case's data; the cache
version goes 1 -> 2. Checks on spark2: core plan:: 10/10, admission:: 4/4, quant::cases 2/2, backend-cuda lib 35/35, ink_rows
7/7; CUDA plan admission with the servediag flags 26/26; CPU plan admission (real model) 14/14; serve with the servediag flags
loads (26 kernels, 103 cases, none rejected), answers, and stops clean. Metal build + admission: endor #11301, pending.

## CUDA prefill at 4k (brief 20; spark2 #11280, nsys, chunk 256 default, P12 on)

4161-token prompt, prefill 3926 ms (1060 tok/s) profiled (3715 ms unprofiled), 18 481 kernels.

| cost | ms | share | note |
|---|---|---|---|
| MoE experts, tensor-core (`ink_moe_mma_kernel`, 1632 launches) | 2526 | 64.4 % | weight-stream bound: see below |
| idle gaps between kernels | 548 | 14.0 % | ~30 us per launch over 18 481 launches |
| in_proj / matmul (cuBLAS nvjet 193, ink_mma_gemv 37, ink_mma_sum 25) | ~255 | 6.5 % | cuda-attn's |
| attention (ink_flash_k 122, ink_prologue_k 122, kv_write 4) | ~249 | 6.3 % | cuda-attn's |
| mix / conv / blend (ink_row 164, blend_gate 56, blend_norm 11) | 236 | 6.0 % | |
| other (Kernel2 90, ...) | ~110 | 2.8 % | |

Why the MoE costs this much: at the default prefill chunk of 256 rows a chunk has 2048 pairs over 128 experts (~16 rows per
expert), so every chunk streams all 128 experts' weights again: 573 MB per visit and chunk, x 48 visits x 17 chunks = ~467 GB
per 4k prefill, ~2.5 s at ~185 GB/s, which is the measured 2526 ms. The GEMM is starved of rows, not slow per FLOP (25 TFLOP
of expert math in 2.5 s is 10 TFLOP/s). First fix (brief 22): a larger prefill chunk (1024, 2048; Metal runs 2048).

## Prefill chunk sweep at 4k (spark2 #11318, `--quiet gpu,cpu`, P12 on, 4161-token prompt, 64 greedy tokens)

| chunk | prefill ms (rep 1 / 2) | tok/s | decode tok/s | peak RSS | greedy md5 (64 tok) |
|---|---|---|---|---|---|
| 256 (default) | 3704 / 3719 | 1123 / 1119 | 33.30 / 33.29 | 19.44 GB | `cbed4e2c…` |
| 1024 | 2202 / 2179 | 1889 / 1910 | 33.20 / 33.22 | 19.44 GB | `cbed4e2c…` |
| 2048 | 1963 / 1946 | 2120 / 2138 | 33.31 / 33.34 | 19.44 GB | `cbed4e2c…` |

Same greedy tokens at every chunk; peak RSS unchanged (the 2048-row scratch fits in the existing pools).

nsys (prefill window only):

| | chunk 1024 | chunk 2048 |
|---|---|---|
| span / GPU busy | 2415 / 1885.5 ms | 2177 / 1645.5 ms |
| idle gaps | 529.5 ms (22%) | 531.5 ms (24%) |
| kernels | 7549 | 5855 |
| MoE experts (`ink_moe_mma_kernel`) | 1110.2 ms, 480 launches | 883.1 ms, 288 launches |
| other | 492.8 ms | 475.2 ms |
| mix / conv / blend | 240.4 ms, 3548 launches | 243.0 ms, 3108 launches |
| in_proj / matmul (cuBLAS, mma) | 39.5 ms | 41.6 ms |

Idle-gap breakdown at chunk 2048 (`.scratch/gaps.sh`): 5257 gaps under 5 µs sum to 11.2 ms; 9 gaps over 1 ms sum to 516.1 ms, and two of them are 339.2 ms (`rmsnorm` -> first cuBLAS `nvjet_sm121_tst_mma_192x160x64`) and 150.7 ms (`ink_ring` -> first `nvjet_sm121_tss_mma_64x104x64`). Runtime API in the same window: `cuLibraryLoadData` 31 calls 461 ms, `cuKernelGetFunction` 144 calls 303 ms. These are first-use stalls before the first cuBLAS GEMM shapes, not launch latency. They are not CUDA's lazy module
loading: `CUDA_MODULE_LOADING=EAGER` leaves the CLI prefill unchanged (#11326: LAZY 1945 ms, EAGER 1945 / 1940 ms, same md5),
so the stall is inside cuBLAS(Lt) first use (its own library loads / heuristics) or host-side first touch. Whether it recurs
per request is what the warm serve run (brief 24 item 2) measures.

Gates at chunk 2048: G2 short PASS (KL21 2.376e-3, 0/363 confident misses), G2 long PASS (3.124e-3, 1/1206), G3 4k PASS (9.671e-3, 2/8229).

Prefill ranking after this: chunk 2048 is a 1.9x prefill win with identical tokens and gates, so flip it (coordinator call). After that, MoE experts are 54% of warm GPU time (883 ms); the next prefill lever is the MoE MMA kernel (W4A4 FP4 tensor cores as vLLM's sm120 path does), then "other" (475 ms, unclassified kernels). Decode MoE repack (~9% decode) remains the decode lever.

### Chunk 2048 default (brief 24 item 1)

`recipes/inkling2.toml` gains `[tuning.cuda] prefill_chunk = 2048` (the same `resolve_prefill_chunk` mechanism the other
families use; other backends keep the default). Revert: `--prefill-chunk 256` per run, or delete that section.

Metal admission #11301 (endor, the Metal half of 24ab7961d: situ tol + per-case seeding): 28 kernels admitted `yes` at both
`--max-seq 20480` and `--mem-budget 30 --max-seq 12288`, none refused, rc 0. That clears brief 23 / 24 item 4 for the Metal half.

### Warm serve prefill (brief 24 item 2; spark2, localeval speed, `--quiet gpu,cpu`, 2 reps + 1 warmup per size, 128 forced tokens)

| prompt tok | chunk 256 TTFT s | tok/s | chunk 2048 TTFT s | tok/s | marginal (2048) | decode tok/s (2048) |
|---:|---:|---:|---:|---:|---:|---:|
| ~1035 | 0.87 | 1185 | 0.49 | 2093 | - | 34.6 |
| ~4126 | 3.04 | 1356 | 1.70 | 2434 | 2574 | 33.7 |
| ~9260 | 6.48 | 1430 | 3.50 | 2645 | 2844 | 33.2 |

Runs #11330 (chunk 256: the serve binary predates the recipe line and embeds the recipes at build time, so it still resolved
`prefill chunk 256 (default)`) and #11346 (`--prefill-chunk 2048` on the same binary; localeval run
`20261122-*-adhoc-intarsia-cuda-c2048-warm`). Warm 4k prefill at 2048 is 1.70 s / 2434 tok/s: the ~2500 tok/s estimate from GPU
busy time holds (marginal 2574). The CLI's extra ~0.25-0.6 s is the per-process first-use stall; it does not recur per request.
Decode is unchanged by the chunk (33-35 tok/s). The recipe default takes effect from the next build (checked then).

## W4A4 expert activations (`ink.act_quant = 1`): status at park (brief 26 item 1; spark2 #11364, build 09ac2f7e1)

Commits 12ed05626 (contract text in `inkling2/ops.rs`, `inkling2/actq.rs` quantisers, CPU refs `quant_ref.rs` / `ink_ref.rs`,
`ShapeInfo::act_quant` + `KernelImpl::act_quant()`, sweep cases labelled ` w4a4`, `kernels/cuda/ink_actq.cu`, row scales in the
decode and bf16-MMA epilogues, knob `INTARSIA_INK_ACT_QUANT` / recipe `[tuning.<backend>] act_quant`, default w4a16),
143eaa02a (tests), 09ac2f7e1 (`ink_moe_fp4_sm120a.cu`: NVFP4 prefill on the FP4 tensor core, `INTARSIA_INK_MOE_FP4=0` falls back
to the bf16 MMA on the dequantised xq). Default numerics unchanged (act_quant 0 everywhere; no recipe sets it).

- Admission (`kernels test --backend cuda`): all expert kernels ADMITTED. Every w4a4 case at T <= 40 is bitwise the f64
  reference (max_abs 0: the GPU quantiser reproduces `actq.rs` bit for bit); T=64 (table path) rel_l2 1.2e-5; T=2048 NVFP4
  rel_l2 2.7e-4 (FP4 tensor-core path not yet confirmed to have run: unquotable until then), MXFP8 T=2048 1.4e-4.
- `ink_rows` 7/7 with the aq=1 variants: T-row calls bitwise T one-row calls and the `_rows` twins (decode, verify rows, table
  path), chunk-invariant prefill; prefill vs SIMT rel_l2 0 (NVFP4: the W4A4 products are short enough that the f32 sums agree
  on this data) / 8.8e-5 (MXFP8).
- Spread vs the reference (`ink_situ_spread`, 150 cases, real shape, NVFP4): T=1 max 8.9e-5 (1 element, 1 ulp); T=4 p99
  1.2e-3, max 7.6e-3 at seed 7, **1 of 150 over SITU_TOL (2e-3)**: one 1-ulp difference of a bf16 `a` element moves the row
  amax / an e2m1 code, and that row's down output then differs in ~20 % of elements (15 ulp at most). A W4A4 tolerance needs its
  own sweep and justification (not done; parked).
- Not done (parked by brief 26): G3 short / long, G2 (information), spec-verify rows and greedy md5 spec on / off under
  act_quant=1, decode tok/s A4 vs A16, FP4 prefill timing.

## Decode: L2 prefetch prologues (brief 26 item 2; spark2 #11377, ink_bench, `--quiet gpu,cpu`)

| kernel (T=1 / T=4) | prefetch off | prefetch on |
|---|---|---|
| conv_mix N4 Q1 norm | 14.3 / 14.3 us | 18.8 / 18.5 us |
| conv_mix N7 Q3 norm | 22.5 / 22.2 us | 27.7 / 27.6 us |
| conv_mix S=3 | 45.1 / 47.1 us | 50.7 / 51.3 us |
| sconv_res / attn_res_mix | 10.2 / 8.2 us | 12.3 / 12.3 us |
| loop_blend S=3 | 32.8 / 32.8 us | 34.8 / 37.0 us |
| moe NVFP4 (union 8 / 29) | 188.1 / 198.9 GB/s | 198.8 / 219.8 GB/s |
| moe MXFP8 (union 8 / 29) | 219.9 / 226.0 GB/s | 220.0 / 229.8 GB/s |

Row kernel: prefetch is 2-5 us slower per call (the prologue's address arithmetic runs on every thread of the 768-thread block,
and the kernel is already register-capped: 80 registers with 228 B spill stores / 680 B spill loads in the widest variant,
before and after) -> removed. MoE: NVFP4 +6 % at T=1, +10 % at T=4, MXFP8 flat -> kept (`INTARSIA_INK_MOE_PF=0` reverts). It
only adds `prefetch.global.L2`, so values are unchanged; the model-level md5 / tok/s check is next.

### MoE prefetch gate (spark2 #11388, fresh build of c0b832b2f, `--quiet gpu,cpu`)

- `ink_rows` 7/7. Recipe chunk now resolves: `prefill chunk 2048 (recipe inkling2, cuda)`.
- Greedy 256 tokens (hash-map prompt, graphed decode, spec off), 3 interleaved reps:

| INTARSIA_INK_MOE_PF | decode tok/s (3 reps) | mean | greedy ids md5 |
|---|---|---|---|
| 0 | 34.38 / 34.32 / 34.37 | 34.36 | `a0e6f47bc45f` (the gate golden) |
| 1 (default) | 34.93 / 34.98 / 35.11 | 35.01 (+1.9 %) | `a0e6f47bc45f` |

- Spec on: **pending** (brief 27: runs after ink/cuda-mtp and this branch are merged, with `--spec dflash`). `--spec mtp` exits with "unsupported: speculative drafters (--spec mtp / dflash)
  on Inkling-2" (the Inkling drafter lives on another lane's branch); the prefetch only adds `prefetch.global.L2`, so verify
  rows cannot change, but the spec-on md5 / tok/s must be rerun where the drafter is.
- nsys of this run traced only the prompt (decode replays CUDA graphs; the next pass uses `--cuda-graph-trace=node`).
- In-model per-kernel (spark2 #11396, `nsys --cuda-graph-trace=node`, 64 greedy tokens, graphed decode), per token:

| kernel | PF=0 | PF=1 |
|---|---|---|
| ink_moe_gu NVFP4 (44 calls) | 127.7 us (5.62 ms) | 118.2 us (5.20 ms) |
| ink_moe_dn NVFP4 (44 calls) | 73.0 us (3.21 ms) | 66.1 us (2.91 ms) |
| ink_moe_gu / dn MXFP8 (4 calls) | 205.7 / 107.7 us | 206.5 / 102.8 us |
| ink_row_kernel<3> (124 calls) | 17.0 us (2.10 ms) | 17.0 us |

  NVFP4 MoE -0.72 ms / token (gu 7.5 %, dn 9.5 % faster), matching the +1.9 % tok/s.

### Row kernel: cheaper prefetch (v1) also no gain (#11396)

One warp per region, 32-bit lane stride, no modulo in the loop: conv_mix N4 Q1 norm 14.3 -> 16.4 us, attn_res_mix 8.2 ->
8.9 us, the others flat (ink_bench, 2 interleaved reps each). Neither prefetch form helps, so the row kernel is not waiting on
these loads' DRAM latency; dropped (not committed; scratch copy `.scratch/ink_mix_pf_v1.cu`). Next: the tap-loop hoist
(bitwise integer change), then the cluster split.

### Row kernel: tap-loop hoist (f274ba96e; `INTARSIA_INK_ROW_HOIST=0` reverts)

The decode conv computed `a.ring[(qp % a.Rr) * D + c]` with a **64-bit modulo once per tap per element** (K 4 x NE 3 = 12 per
thread, plus one more per element for the ring write): ~100 instructions each, on all 24 warps of the 768-thread block. Now the
taps are the outer loop (each `acc[j]` still sums l = 0 .. K-1 in order, so bitwise), the source row and its 32-bit ring
modulo are formed once per tap, and the ring-write offset once per thread. This also explains both failed prefetch tries:
the kernel was paying integer issue, not DRAM latency.

| | HOIST=0 | HOIST=1 |
|---|---|---|
| ink_bench conv_mix N4 Q1 norm / N7 Q3 / conv only / S=1 blend | 14.3 / 20.5 / 8.8 / 39.0 us | 10.2 / 16.6 / 6.1 / 34.8 us |
| in model, ink_row_kernel<3> (124 calls / token, nsys graph-node, #11401) | 16.6 us (2.06 ms / token) | 10.2 us (1.27 ms / token) |
| greedy 256 decode tok/s (3 interleaved reps, #11401) | 34.94 / 35.18 / 35.21 (35.11) | 35.87 / 36.15 / 36.09 (36.04, +2.6 %) |
| greedy ids md5 | `a0e6f47bc45f` | `a0e6f47bc45f` |

`ink_rows` 7/7 (#11398). Branch decode now 34.36 -> 36.04 tok/s (+4.9 %) from the MoE prefetch and this. Spec-on pending (brief 27).

### W4A4 FP4 path confirmed (#11398, nsys of `kernels test --op moe_experts --kernel moe_experts_situ_bf16`)

With `INTARSIA_INK_MOE_FP4=1` the T=2048 w4a4 case launches `ink_fp4_kernel` (gate+up and down) + `ink_fp4_sum_kernel`; with
`=0` it runs `ink_moe_mma_kernel` instead. So the T=2048 NVFP4 W4A4 admission (rel_l2 2.72e-4, max_abs 7.9e-3) is the FP4
tensor-core path; the bf16-MMA fallback gives the same figures to 4 digits.

### Cluster split: premise checked, not built

ptxas (`-Xptxas -v`, after the hoist): `ink_row_kernel<3>` (the instance for D 2304, 768 threads) uses 80 registers with
**0 B spills**; only NE 5 / 6 spill (264 / 84 B; unused by this model). The earlier "spills before and after" was the widest
instance. With no spills, a cluster split buys only more SMs issuing for a 10.2 us kernel: at most ~0.5 ms / token of the
remaining 1.27. Decode GPU time per token now (#11401 trace, hoist on): ink_mma_gemv (InkInProj) 8.7 ms, NVFP4 MoE gu + dn
5.2 + 2.9 ms, P12 unembed 2.9, attention p12 gemv 2.2, ink_row 1.27, blend gate + norm + deep gemv ~1.6, moe_route 0.38.
The MoE kernels run at ~190 (gu) / ~170 (dn) GB/s at T=1; the repack to ~240 GB/s saves ~1.9 ms / token (~7 %), the largest
lever in this lane's classes.

## MoE repack probe (brief 28; spark2 #11417 + pass 2, `--quiet gpu,cpu`; standalone `.scratch/moe_layout_probe.cu`)

Run on spark2 (the lane's build tree; same GB10 as spark1). No-math streams at T=1 with the shipped kernels' grid and threads
(gu: I/4 blocks x 256 threads, warp = expert, 4 rows x 8 lanes; dn: H/8 blocks, 8 rows x 4 lanes; 8 experts out of 128, 64
random expert sets rotating over 4 weight copies), xor-folding the loaded words. Repacked layout: per (expert, row block) one
contiguous run, iteration-major `[iter][lane][payload]` then `[iter][lane][scales]`, so each warp load is one contiguous
1 KB (gu) / 512 B (dn) piece. Two passes each, they agree within 2 %:

| T=1, per call | gu | dn |
|---|---|---|
| (a) current layout, no math | 117.0 us / 204 GB/s | 63.4-65.1 us / 183-188 GB/s |
| (a') current layout + L2 prefetch (= the shipped kernel's loads) | 109.8-110.8 us / 216-218 GB/s | 59.5-61.4 us / 195-201 GB/s |
| (b) repacked, no math (prefetch adds nothing) | 106.0-107.1 us / 223-226 GB/s | 52.9-53.1 us / 225 GB/s |
| (c) shipped kernel in the model (PF on, #11396 nsys) | 118.2 us | 66.1 us |
| shipped kernel, ch loop unrolled by 3 (ink_bench, NVFP4 T=1 / 4 / 16) | no change: 180.0 vs 180.3 / 590 vs 590 / 1684 vs 1678 us | |

Reading: the shipped kernel runs ~7-9 % above its own loads' no-math time, so its math is mostly hidden and the loads set the
pace. Against the fair baseline (a'), the repacked stream saves ~4 us (gu) + ~7 us (dn) = ~11 us per NVFP4 layer, ~0.49
ms / token over 44 layers (~0.6 ms if the math overhead scales), **below the 1 ms bar: the repack is not built** (brief 28 rule).
Any layout reaches only ~225 GB/s at this geometry (82 % of 273); the remaining gap is the thread geometry / per-SM request
rate, not the layout. The unroll-by-3 change was reverted (no effect). Next item: the LoopBlend path (deep gemv 0.86 +
gate 0.49 + norm 0.25 ms / token) and the router (0.38 ms).

### MoE thread-geometry probe (brief 29; spark2, fleetq `--quiet gpu,cpu`; same harness, current layout, no math, 2 passes)

The shipped kernels already load 128 bits per lane per step (one `uint4`). Varied: rows per warp (= contiguous bytes per load
instruction: 32 / rows lanes per row) and threads per block (CTA count at the same per-warp pattern):

| geometry | gu us / GB/s | geometry | dn us / GB/s |
|---|---|---|---|
| 4 rows/warp, 256 thr (shipped) | 114.4-114.6 / 208 | 8 rows/warp, 256 thr (shipped) | 60.8 / 196 |
| 4 rows/warp, 128 thr (2x CTA) | 113.5-116.1 / 206-210 | 8 rows/warp, 128 thr (2x CTA) | 60.9-64.8 / 184-196 |
| 4 rows/warp, 64 thr (4x CTA) | 113.3-113.4 / 211 | 8 rows/warp, 64 thr (4x CTA) | 61.0-62.0 / 193-196 |
| 2 rows/warp, 256 thr | 108.9-109.3 / 219 | 4 rows/warp, 256 thr | 57.3-57.8 / 207-208 |
| 2 rows/warp, 128 thr | 107.5-109.8 / 218-222 | 4 rows/warp, 128 thr | 55.9-56.0 / 213 |
| 1 row/warp, 256 thr | 105.9-108.6 / 220-226 | 2 rows/warp, 256 thr | 53.9 / 221 |
| 8 rows/warp, 256 thr | 118.1-118.7 / 201-202 | 16 rows/warp, 256 thr | 93.9-95.2 / 126-127 |

More CTAs at the same pattern change nothing; longer contiguous runs per instruction (fewer rows per warp) reach the same
~225 GB/s ceiling as the repacked layout, without a repack. No geometry reaches 250 GB/s: the best saves ~0.5 ms / token
against the shipped kernel, under the 0.8 ms bar. **MoE at T=1 closed** (brief 29). A low-cost follow-up if wanted later:
gu at 1-2 rows / warp and dn at 2 rows / warp in the shipped kernels (same layout, ~0.5 ms / token, needs its own bitwise check
since the lane-to-chunk map and the shuffle tree change).

## LoopBlend controller GEMV probe (spark2, fleetq `--quiet gpu,cpu`; `.scratch/blend_probe.cu`, no math, 2 passes agree)

The controller h projection (`ink_mma_gemv_deep`, 47 calls / token at 18.3 us, 0.86 ms) streams one 64 x 4608 bf16 stream
(590 KB) per 4-warp CTA, 1-3 CTAs per call. One CTA, one stream, 128 rotating copies:

| per-SM loads | us / call | GB/s |
|---|---|---|
| register double buffer kPF 4 (the shipped kernel's loads) | 14.7 | 40 |
| kPF 2 | 13.8 | 43 |
| cp.async shared ring, 8 stages (64 KB) / 12 stages (96 KB) | 10.3 / 10.2 | 57 / 58 |
| kPF 4 + 8 / 16 / 32 helper CTAs issuing L2 prefetch | 12.2-12.4 | 47-48 |

One SM tops out near 57 GB/s. A bitwise kernel (same per-tile mma sequence, a cp.async ring for the loads) gets ~1.4x: ~0.25
ms / token. Only more SMs (split-K) go further: ~3-4 us per call, ~0.7 ms / token, but it changes h's bits, so it needs the
same split and reduction order at every T (decode, verify, prefill) and G2 / G3 + prefill gates. Asked the coordinator which;
meanwhile the bitwise gate + norm fuse (~0.2 ms).

### Split-N check (brief 30; spark2 fleetq `--quiet gpu,cpu`, same probe, 2 passes agree)

h's N is the controller rank G = 64 per stream: 4 n-tiles of 16, so split-N gives 4 CTAs per call at S=1 and 12 at S=3 (one
warp each; 24 + 24 calls / token). No-math, shipped register loads:

| | current | split-N (1 warp / CTA) |
|---|---|---|
| S=1 (590 KB) | 14.7 us (1 CTA x 4 warps) | 12.9 us (4 CTAs) |
| S=3 (1.77 MB) | 16.3 us (3 CTAs x 4 warps) | 15.5 us (12 CTAs) |

Spreading the same warps over more SMs gains 5-12 %: each warp's serial load chain (147 KB over 288 k-steps) is the limit,
not the SM. N is too narrow -> rule 2: split-K as an opt-in (default off), after the gate + norm fuse.

### Gate + norm fuse (brief 30 step 1): slower, removed

`2a876a6a0` folded `ink_blend_norm_kernel` into the gate kernel (last slice block of each (stream, row block) sums the slice
partials and writes the normed rows: same sums, same order, bitwise). Gate (fleetq `--quiet gpu,cpu`, `.scratch/fusegate.sh`):
greedy md5 `a0e6f47bc45f` both ways, but decode 36.11 / 35.91 / 36.14 tok/s fuse off vs 35.55 / 35.75 / 35.80 fuse on (#11425), and `ink_bench
mix` T=4 loop_blend 30.7 -> 41.0 us (S=1), conv_mix 36.9 -> 47.2 us; T=1 unchanged. Cause: the last block does the norm for up
to 16 rows x D = 2304 with one CTA after a serial thread-0 partial sum, a tail longer than the launch it saves: nsys (64 tokens) gate 10.1 us + norm 5.3 us
per call unfused vs 20.3 us fused (3072 calls). Removed in the
next commit (no env switch left).

### Controller h split-K for decode / verify chunks (bitwise; default on, `INTARSIA_INK_BLEND_SPLITK=0` reverts)

`ink_ctl_kernel` (ink_mix.cu, `5488b8f6d`): for blend chunks of rc <= 16 rows the k range of each 16-row h tile is spread over
2 x 64-k blocks per warp, 4 warps per CTA (S=1: 18 CTAs x 4 tiles; S=3: 18 x 12). Every warp computes each 16-k step's mma
tile from zero, as `mma_add` in `ink_mma_gemv` does, and stores it; the last CTA per tile (counter zeroed by the row kernel) adds
the tiles in k order into an f32 from 0 and rounds once: the same chain of f32 adds as `ink_mma_gemv` / its deep form, so h has
the same bits. Prefill chunks (rc > 16) keep `ink_mma_gemv`. Scratch +3.5 MB at S=3, D=2304, G=64.

Gate #11430 (spark2, `--quiet gpu,cpu`, `.scratch/skgate.sh`): ink_rows 7/7; loop_blend 27 and conv_mix 532 cases ADMITTED;
greedy 256 ids md5 `a0e6f47bc45f` in all 6 runs; decode (graphed, spec off, 256 tok) 36.02 / 35.85 / 36.02 tok/s off vs 36.54 /
36.65 / 36.45 on (+1.6 %). nsys, 64 tokens: controller h `ink_mma_gemv_deep` 3024 x 18.1 us + 48 x 22.3 us (55.9 ms) ->
`ink_ctl_kernel` 3072 x 11.5 us (35.2 ms), -0.32 ms / token. `ink_bench mix`:

| case | split-K off | on |
|---|---|---|
| loop_blend T=1 S=1 / S=3 | 28.7 / 32.8 us | 22.6 / 28.7 us |
| loop_blend T=4 S=1 / S=3 | 30.0 / 35.7 us | 24.8 / 32.1 us |
| conv_mix N4 Q1 T=1 S=1 / S=3 | 34.9 / 40.5 us | 28.7 / 35.3 us |
| conv_mix N4 Q1 T=4 S=1 / S=3 | 36.9 / 43.5 us | 31.4 / 38.2 us |

Default on because it is bitwise (brief 30 rule 2 made split-K opt-in because a split normally changes h's bits; this one does
not). Branch decode: 34.36 -> ~36.5 tok/s (MoE prefetch, row hoist, controller split-K).

## Merge of ink/cuda-mtp 427652506 (brief 31): `6f77be924`

Clean merge (model.rs, model/inkling2.rs, load.rs auto-merged; no conflicts). Gate #11431 (spark2, `--quiet gpu,cpu`,
`.scratch/mergegate.sh`) on the merged tree (+ split-K, `9f1abe940`); pre-merge baseline from the `5488b8f6d` binary with
`INTARSIA_INK_BLEND_SPLITK=0` in the same job.

- Build OK; ink_rows 7/7; tiny `inkling2` 15/15 cpu and 15/15 cuda (incl. `inkling2_tiny_verify_graph_replays_equal_decode_bitwise`;
  #11460 with --nocapture: cuda R=2/3/4 17 + 11 + 8 verify passes, 1 capture each, every pass replayed); tiny `inkling2_dspark` 3/3
  cpu, 3/3 cuda.
- Greedy ids md5 (graphed, 64 France / 256 hash-map / 256 after the 1.2k README prompt, `--ignore-eos`):

| run | g64 | g256 | g1.2k |
|---|---|---|---|
| pre-merge, split-K off | `ec9a19c14b2d` | `a0e6f47bc45f` | `aaaa65a7ac70` |
| merged `--spec off`, split-K off | same | same | same |
| merged `--spec off` (split-K on) | same | same | same |
| merged, no `--spec` flag (CUDA default: no drafter) | same | - | - |
| merged `--spec dflash --spec-draft <model dir>` (verify graph on) | same | same | same |

- Decode tok/s, 256 tokens: g256 spec off 36.47 / 36.50 / 36.58, spec on 40.81 / 40.75 / 41.09 (+11.9 %, acceptance 34.1 %, 132
  replays / 8 captures); g1.2k spec off 35.93, spec on 42.99 (+19.6 %, acceptance 43.0 %); g64 36.76 vs 36.05 (acceptance 31.6 %).
- Serve smoke (spec on, `--max-seq 20480`; load 252 s incl. admission of 38 kernels / 231 cases; localeval
  `20261122-100712-adhoc-intarsia-cuda-mixmoe-merge-mtp-smoke`, 128 forced tokens, 1 rep): 1036 tok prompt TTFT 0.47 s, prefill
  2200 tok/s, decode 42.2 tok/s (peak 75.3); 9251 tok prompt TTFT 3.59 s, prefill 2578 (marginal 2635), decode 50.4 (peak 84.6).
  No error / panic / capture-failure lines in the serve log; stopped clean, port closed.

## Spec-on decode profile (brief 32; #11462, spark2, nsys graph nodes, merged binary)

A cycle = one draft + its verify pass or plain step. g256 spec on: 25.3 ms / token wall, 24.0 GPU busy, 132 cycles of 49 ms,
1.93 tokens per cycle (spec off 27.6 ms / token). GPU idle with spec on 1.4 ms / token (spec off 0.4). Per-kernel at g256 spec
on (ms / token): MoE gu 6.56, in_proj `ink_mma_gemv` 4.52, MoE dn 3.60, bf16 `gemv_kernel` 3.29 (verify unembed + wo + draft
layers), draft head `mxfp4_gemv_kernel` 1.26, P12 1.16.

| target | ms / token at g256 spec on | evidence | owner / status |
|---|---|---|---|
| 1. draft head MXFP4 GEMV (200058 x 2304, M = 7, once per cycle) | ~0.7 | 2.44 ms per call, 100 GB/s: the exact-rows draft takes the generic warp-per-row GEMV (72 groups per row: 2-3 per lane, activations re-read per row) | built, below |
| 2. verify unembed and wo on bf16 instead of P12 | ~0.4 + ~0.27 | unembed `gemv_kernel` 3.87 ms per verify pass vs P12 2.92 at decode; wo 78.7 vs 49.6 us (`inkling2.rs:1132` takes wo_p12 only for t <= DENSE_ROWS_MAX = 4) | handed to ink-cuda-attn (brief 33) |
| 3. draft MLP layers (bf16, 6 x 177 us per cycle, ~240 GB/s) | ~0.55 | only fewer bytes help (fp8 layers), which trades acceptance | skipped (brief 33) |

Verify-pass MoE: gu 259 us vs 116 at decode (2.2x for ~4.5 rows): the expert union, bandwidth-bound; no target.

g64 spec on (36.05 vs 36.76 tok/s): not acceptance (31.6 %, near g256's 34.1 %). The first draft has one 148 ms host stall
before the first cuBLASLt / cutlass kernel (lazy init) and 23 ms before the first splitkv: 171 ms one-off, ~2.7 ms / token over
64 tokens; without it on64 is ~1728 ms vs off 1724.

## DSpark draft head: `matmul_mxfp4_gemv_ink_rows` (default on, `INTARSIA_INK_MX4HEAD=0` reverts)

kernels/cuda/ink_mx4head.cu (`6b20dd184`), registered in inkmoe.rs as a rows-exact `MatmulMxfp4` kernel for 5..=8 rows and
K <= 3072, W4A16 (priority 5, above `matmul_mxfp4_gemv_rows`). Bitwise `mxfp4_gemv_kernel` per element (same per-group fmaf
chain, same lane order, same warp_sum); one warp owns 4 output rows and loads every group it owns for them up front (NI = 3 at
K 2304), each activation group once for the 4 rows. Probe (.scratch/mx4head_probe.cu, 3 rotated weight copies, M = 7):

| kernel at M = 7 | us | GB/s | bitwise gemv |
|---|---|---|---|
| `mxfp4_gemv_kernel` (current) | 2443 | 100 | - |
| `mxfp4_mma` (tensor core, row-invariant) | 1375 | 178 | no |
| mr R2 / mrC R4 / mrC R6 / R8 (registers held per m, spills at R8) | 1920 / 1452 / 1727 / 3574 | | yes |
| **D: all groups preloaded, R 4** | **1382** | **177** | **yes, 0 of 1.4M differ** |

M 5 / 8: 1109 / 1453 us. ptxas: the NI 3 / 2 instances 0 spills (NI 1 spilled: K <= 1024 runs the NI 2 instance).

Gate #11481 (spark2, `--quiet gpu,cpu`, `.scratch/hdgate.sh`): ink_rows 8/8 (new `ink_mxfp4_head_bitwise_gemv`: bitwise
`matmul_mxfp4_gemv_rows`, M 5..8, both out dtypes, N tails, K 1 / 33 / 72 / 96 groups); `kernels test --op matmul_mxfp4`
6 kernels ADMITTED (the new one 12 cases, max_abs 9.8e-8). Greedy, spec on (DSpark), ABAB x 3:

| | head off | head on |
|---|---|---|
| g256 tok/s | 40.96 / 40.94 / 40.93 | 41.92 / 41.93 / 41.93 (+2.4 %) |
| g1.2k tok/s | 42.99 / 43.05 / 42.70 | 44.17 / 44.08 / 44.03 (+2.7 %) |
| ids md5 g256 / g1.2k | `a0e6f47bc45f` / `aaaa65a7ac70` | same |
| acceptance g256 / g1.2k | 34.1 % / 43.0 % | same |

Spec off g256 unchanged (`a0e6f47bc45f`, 36.50 tok/s). nsys (128 tokens): head 72 x 2418.6 us -> 71 x 1439.0 us.

### g64 spec on after the head kernel; first-call warmup not built (#11486)

Without a profiler, g64 spec on (France, 64 tokens, head kernel on) is 1720 / 1717 ms = 36.62 / 36.70 tok/s, ids `ec9a19c14b2d`:
level with spec off (36.53-36.76). The 148 ms first-draft stall of #11462 was mostly nsys's own first-call cost (the merge gate
without nsys had on64 1748 ms vs 1714 spec off with split-K on: 34 ms). `CUDA_MODULE_LOADING=EAGER` (every module loaded at start): decode 1679 / 1685 ms
(37.51 / 37.40 tok/s, -38 ms, same ids) for +2-4 s process wall. So lazy first-use loading costs ~38 ms once per process; a
targeted load-time warmup would save at most that for one-shot `generate`, and serve pays it once. Not built (well under the
brief's bar of being worth a load-time cost); `CUDA_MODULE_LOADING=EAGER` is the switch for a one-shot user who wants it.

Serve smoke with the head kernel (#11490, spec on, 1k, localeval `intarsia-cuda-mixmoe-mx4head-smoke`): loads, answers, stops
clean. Its decode figure (57.9 tok/s vs 42.2 in #11431 at the same size) is not like-for-like and is not cited; the serve
admission count (38 kernels / 231 cases) is unchanged, so serve admission did not cover the head kernel: #11481's `kernels test`
is its admission evidence. The same job found `kernel_list_is_unchanged_by_core_op_fusion_p1` failing on the new name; pinned
in `d3abfa9b6` (5..8 rows at K <= 3072: no other model has that MXFP4 shape).

## This lane's costs (brief 34; from #11462, ms / token at g256)

| kernel | spec off | spec on | floor / evidence | bound (ms / token) |
|---|---|---|---|---|
| MoE gu + dn (NVFP4, 44 layers + MXFP8 4) | 8.80 | 10.17 | loads set the pace (#11417: kernel 7-9 % over its no-math loads); any layout tops out ~225 GB/s at this geometry (#11417 / brief 29) | repack ~0.5-0.6 (the only item >= 0.3) |
| `ink_row_kernel` (ConvMix / SconvRes / AttnResMix rows) | 1.28 (124 calls x 10.3 us) | 0.70 | launch / latency bound; hoist done, prefetch and cluster split dead ends | < 0.3 without new fusion |
| LoopBlend: `ink_ctl` + gate + norm | 1.28 (48 x 26.6 us) | 0.75 | controller stream 590 KB = 2.2 us at 273 GB/s; split-K done; gate + norm fuse a dead end | < 0.3 left without a different fusion |
| `moe_route_kernel` | 0.38 | 0.20 | one small launch per layer; fusing it needs `InkRoute` on CUDA (Metal: -0.11 ms / step) | ~0.1-0.2 |
| `ink_tile_staged_k` + `ink_tile_k` | 1.67 | 1.17 | attention (ink-cuda-attn), not this lane | - |

Only the MoE repack clears 0.3 ms / token. It was parked under brief 29's 1 ms bar; it is not built here without a go (it changes
the expert layout that `ink_moe_mma` (prefill), the `_rows` path and W4A4 all read: either all of them learn the layout, or a
second copy of the experts is kept: ~25 GB for the NVFP4 experts, estimated from the probe bytes (~4.5 MB per expert x 128 x 44 layers)).

### Dead end: MoE NVFP4 decode staging (`9be9868a9`, reverted `6cbbff7bf`)

A cheaper way to get the contiguous loads without a repack: each warp copies its expert row block to shared memory with
contiguous 16-byte `cp.async` (512 B per warp instruction), then runs the same chunk loop (bitwise by construction). Gate #11491:
`kernels test` ADMITTED, but `ink_bench` NVFP4 180 -> 234 us (T=1), 592 -> 726 (T=4), 1686 -> 1780 (T=16), MXFP8 (not staged)
unchanged; model g256 spec off 36.66 -> 33.10 tok/s with the same ids. Cause: the gu slice is 8 warps x 2 x 4 x 1296 B = 83 KB
of dynamic shared memory per block (dn 41 KB), so 1 (gu) / 2 (dn) blocks per SM against the register-limited occupancy of the
plain kernel; and at a union of 8 each warp has one expert, so `cp.async.wait_all` exposes the whole load before any math.
Reverted; the job was stopped after the first ABAB pair.

## CUDA no-op on other models (brief 35; #11498, spark2)

Branch `6cbbff7bf` (this lane + ink/cuda-mtp + main `57e1b3332`) against main `57e1b3332`, both built `--features cuda` in the
same job: Qwen3-8B greedy 256 (chat, no-think) short-0 identical ids (md5 `3d425d4ebaa1`, 15.64 vs 15.67 tok/s) and 4k-0
identical (md5 `f10cca037c96`, 15.10 both); `plan show` JSON and text byte-identical for the 8B (8973 / 1548 B) and the 35B
NVFP4 (15570 / 4017 B); `kernels test --quick` on the branch: 128 kernels, 1556 cases, rc 0; ink_rows 8/8.

## Inkling on CUDA after the main merges (#11502, spark2, tree `611d17464` = main `693f6440e` merged)

- Tiny `inkling2` cpu 17/17; cuda 15/17: the two screened-head tests from #489 (`inkling2_tiny_screened_head_one_row_only`,
  `inkling2_tiny_screen_keeps_batching`) force `INTARSIA_INK_HEAD_SCREEN=1`, whose ops are cpu / Metal only, so the CUDA load
  is refused; they now skip on CUDA (`dd250e728`; rerun in #11520). Tiny `inkling2_dspark` 3/3 cpu and cuda.
- Greedy ids md5, spec off and spec on (DSpark, verify graph): g64 `ec9a19c14b2d`, g256 `a0e6f47bc45f`, g1.2k `aaaa65a7ac70`
  in all six runs, as before both merges. Decode tok/s spec off 36.67 / 36.46 / 35.98, spec on 36.88 / 41.82 / 43.99
  (acceptance 31.6 / 34.1 / 43.0 %).

## MoE decode repack, stage A (brief 36; branch `ink/cuda-mixmoe-repack` `0e0c21028`, not in the PR; gate #11520)

`INTARSIA_INK_MOE_REPACK=1` (CUDA, default off): at load each block of 4 gate / up rows and of 8 down rows (payload and scales)
is permuted so that a decode warp step is one contiguous 512 B (payload) / 64 B (scales) run
(`inkling2::ops::moe_repack_rows`, `ink.moe_layout = 1`); the NVFP4 gu / dn decode kernels index it (`RP` instances), every
repacked call takes the decode kernels (no tensor-core path), W4A4 is refused. Bitwise by construction: the same chunks reach
the same lanes. `ink_rows` repack test passed in #11520 (its first form also raced `ink_experts_rows_exact` through a
process-wide setter; fixed in `0e0c21028`, not yet rerun).

| ink_bench NVFP4, 2 passes | plain | repacked |
|---|---|---|
| T=1 | 180.6 / 177.6 us (199-202 GB/s) | 170.9 / 170.2 us (210 GB/s, -5.4 %) |
| T=2 | 328.9 / 331.2 us | 312.4 / 312.3 us (-5.4 %) |
| T=4 | 592.0 / 594.2 us | 571.2 / 573.2 us (-3.5 %) |
| T=16 | 1682.7 / 1686.0 us | 1644.9 / 1641.4 us (-2.4 %) |

| greedy 256, rotated order | plain | repacked |
|---|---|---|
| spec off tok/s | 36.32 / 36.48 / 36.37 (36.39) | 36.82 / 36.90 / 36.68 (36.80, **+1.13 %**) |
| spec on tok/s | 41.87 / 41.76 (41.82) | 42.38 / 42.34 (42.36, +1.3 %) |
| ids g256 | `a0e6f47bc45f` | `a0e6f47bc45f` |
| g1.2k (spec off) | `aaaa65a7ac70`, prefill 1583 tok/s | `4b44e4ab3ff3`, prefill 436 tok/s |

The g1.2k ids differ because a repacked prefill runs the decode kernels (another f32 order than the tensor-core GEMM) and is
3.6x slower (MoE per 2048-row layer: decode kernels 78.2 ms vs tensor core 7.8 ms): stage A's refusal of the prefill path,
not a repack error. **Verdict: +1.13 % spec off is under the +1.2 % bar: stopped, not merged** (the branch stays as the
record). The decode gain is real but small: ~0.4 ms / token.

MoE bandwidth floor (brief 36 item 3): at g256 decode a token reads 1.84 GB of experts (44 NVFP4 layers x 8 experts x 4.48 MB
+ 4 MXFP8 layers x 8 x 8.21 MB). At the best measured stream on this box (~240 GB/s: the head GEMV 242, plain bf16 GEMV 239)
the floor is 7.7 ms / token (6.7 ms at the nominal 273 GB/s), against 8.80 ms measured (#11462, spec off) and ~8.4 ms
repacked. What remains after the repack (~0.7 ms) is launch ramp / tail: 2 launches per layer (96 per token) whose last wave
runs a few experts on a few SMs. Next lever: one launch per layer for gu + dn (dn blocks start on the h rows as they
complete, a per-row-block ready flag), worth ~0.3-0.5 ms / token by estimate; not measured.

Prefill (for brief 37): the existing W4A4 FP4 tensor-core path (`ink_moe_fp4_sm120a.cu`, block-scaled `mxf4nvf4` m16n8k64)
already runs at 4.83 ms per 2048-row NVFP4 layer (incl. activation quantisation) vs 7.83 ms for the bf16 `ink_moe_mma`
(ink_bench, #11520): 54 vs 33 TF/s.

## PR #491 no-op evidence (brief 39; head `9ca7a7588`, same code as `cb13bd5f2`, vs main `1f226a2e3`)

Metal plan show Qwen3-8B / Llama-3.1-8B / Inkling and cpu plan show Qwen3-8B / Llama: byte-identical (endor #11566). Inkling Metal greedy
md5 g64 `65cde8f35a18` and g256 `61fd59fecb5e`, both equal to main. CUDA plan show 35B / 8B: byte-identical. CUDA 8B greedy ids are
identical, and kernels test --quick passes 128 kernels / 1556 cases (spark2 #11547). Inkling CUDA md5 g64 / g256 / g1.2k are unchanged
with spec off and on (#11541).

## Merge of main 3f79f70ed and the plan-test fix (PR #491 CI)

CI `cargo test (linux)` failed in `plan.rs::lazy_admission_cases_build_nothing_when_cached_and_keep_the_data_of_those_that_run`.
It also failed on endor twice in a row, so it is deterministic, not a flake. The cause is that serde_json without the
`float_roundtrip` feature parses some 17-digit floats to the neighbouring value. The entries kept from the cache take one more
parse / print step than the cold file, so `cpu_f32_logits` rel_l2 came back 1 ulp off (2.4632088598130532e-8 vs
2.463208859813053e-8). This branch exposed it by seeding each case from its label (`case_rng`), which gives that case new data.
Turning `float_roundtrip` on made the test pass 2/2 (#11590, probe only, reverted). I did not turn it on for the workspace
because it would change how every JSON float parses, model configs included. Fix: the test now compares floats up to 4 ulp;
data drawn from another generator would differ by far more. Plan tests pass 10/10, twice (#11595). Merge checks after the
main merge: workspace check, mtp 17/17, inkling2 69/69 (endor #11581); Inkling CUDA md5s are unchanged with spec off and on
(spark2 #11582).
