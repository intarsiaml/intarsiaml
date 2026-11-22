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
Metal token comparison: queued on endor.

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
