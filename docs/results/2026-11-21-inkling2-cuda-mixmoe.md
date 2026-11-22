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
