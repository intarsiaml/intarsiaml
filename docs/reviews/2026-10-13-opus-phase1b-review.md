# Phase 1b adversarial code review (Opus 5.5, 2026-10-13)

I found 10 real problems. Most of them are in the accuracy gate and the A/B procedure, not in the kernels. Everything below comes from reading the uncommitted tree: I ran nothing on a GPU, and the CUDA Rust path has never been built or run.

**Code checked, no bug found:**
- **Grouped MoE kernel (`moe_grouped.cu`):**
  - The tile-table bound `ceil(P/BM)+min(P,E)` is correct, and experts with zero tokens get no tiles.
  - Partial tiles are zero-filled, and their epilogue skips rows at or past `rows`.
  - Duplicate and out-of-range ids are dropped consistently in grouping and in the reduce.
  - The scatter is deterministic per pair even though the atomic cursor reorders pairs.
  - The cp.async commit-group, raw-ring and A-ring reuse are race-free (checked for SPU=1 and SPU=2).
  - The gate/up row mapping and `nrow()` agree with the SiLU-GLU epilogue.
- **Decode MoE kernel (`moe_decode.cu`):** rounding points match `moe.cu`, and a test checks bit-equality against it.
- **Split-KV attention:** the max-grid launch with `nsplit_eff` computed on the device partitions keys the same way as the host-length version.
- **Capture safety:** the decode step does no host transfers, allocations or syncs while being captured. Recurrent state is restored by in-place copy, so captured buffer addresses stay valid.

## Findings, most severe first

**1. [HIGH, accuracy gate] The "before" baseline is not recorded from the pre-change code, and nothing records which code produced it.**
- **Where:**
  - `docs/results/2026-10-13-phase1b.md:100-102` (procedure).
  - `crates/intarsia-cli/src/cmd/accuracy/baseline.rs:89-101` (RunConfig).
  - `crates/intarsia-cli/src/cmd/accuracy.rs:1222-1232` and `:1293` (`tool_version = CARGO_PKG_VERSION`).
  - `crates/intarsia-backend-cuda/src/device.rs:268-271`.
- **Problem:**
  - The baseline is recorded from the new tree with `INTARSIA_CUDA_DISABLE` and `--no-graphs`. Several changes cannot be disabled and so sit in both runs:
    - the device-`pos` rope, kv_append and attn_decode kernels;
    - the split-KV decode grid, now always launched at maximum size, with `nsplit_eff` in the combine;
    - the `lm_head` and quant-scheme loader changes.
  - The baseline stores no git revision or dirty state, no disable list, no graphs flag, and no kernel resolved per op.
  - Unknown names in `INTARSIA_CUDA_DISABLE` are ignored silently.
- **Failure scenarios:**
  - An off-by-one in the device-side `nsplit_eff` at some lengths is present in both runs, so only the 3e-2 floor is left to catch it.
  - A typo in the disable list (e.g. `moe_expert_mma_bf16`) records a "before" baseline with the new kernels, and "after" passes trivially.
- **Fix:**
  - Build HEAD (1d4718b) in a worktree and record the baseline from it.
  - Add git rev and dirty flag, graphs on/off, the disable list and the per-op kernel set (`model.resolved.ops`) to RunConfig. Refuse to compare on a mismatch, or print the kernel diff.
  - Reject unknown names in `INTARSIA_CUDA_DISABLE`.

**2. [HIGH, accuracy gate] The gate is far too coarse for a change on the same weights.**
- **Where:** `baseline.rs:150-167` (`judge`).
- **Problem:**
  - Only per-mode aggregates are gated, against vLLM running an fp8 KV cache. The baseline mean KL of about 6.5e-3 is dominated by that reference's noise, so the x1.15 margin allows about 1e-3 nats of real regression.
  - `max_kl` and the per-prompt stats are stored but not gated; per-prompt values are only printed after a failure.
  - The tail check uses the signed mean, so positions that lose and gain tail mass cancel.
  - There is no way to make a golden from our own engine (no self-golden), which would allow a direct old-vs-new comparison.
- **Failure scenario:** a grouped-MoE bug that doubles the KL of one of 10 prompts (for example the `chat_system` prompt at 90.6% top-1) raises the mean by about 10% and passes.
- **Fix:**
  - Add a "write golden from this engine" mode (full logits, or top-k plus tail).
  - Gate the new kernels and graphs against the HEAD eager output at mean KL ≤ ~1e-4, max KL bounded, and top-1 ≥ 99.5%.
  - Also gate each prompt (≤ x1.15 + eps), gate max KL, and gate the mean of |tail excess|.

**3. [HIGH, tolerances] Kernel tolerances are 15-30x wider than the change they are meant to catch.**
- **Where:**
  - `crates/intarsia-backend-cuda/src/quantmoe.rs:27` (`TOL` rel_l2 6e-3, max_abs 1.5e-2 × scale), shared by `moe_experts_mma_bf16`, `matmul_*_gemm` and `moe_experts_decode_bf16`.
  - `crates/intarsia-backend-cuda/tests/quantmoe.rs:45` (`rel_l2 < 1e-2`).
- **Problem:**
  - The measured difference from the old kernel is 3.5e-4. Together with finding 2, a systematic grouped-kernel error 15x larger (a dropped rounding point, a wrong scale on one tile class) passes admission, the unit test and the model gate.
  - `docs/quant-moe.md` says the `*_gemm` outputs are identical to `*_mma` ("rel_l2 0 in all 30 combinations"), and the results doc says "rel_l2 up to 3.5e-4 against the old kernel" for the "MoE and GEMM kernels". These contradict each other.
  - No test asserts that equality. Nothing in `crates/*/tests` references `matmul_*_gemm`; only the admission sweep does.
- **Fix:**
  - Add bitwise tests of `*_gemm` against `*_mma`, mirroring `moe_experts_decode_matches_old_kernel`.
  - Compare `moe_experts_mma_bf16` with `moe_experts_bf16` at rel_l2 ≤ 1e-3.
  - Give the new kernels their own TOL at about 3x the measured value.

**4. [MED-HIGH, graph / A/B validity] Nothing checks that the served A/B actually replays graphs.**
- **Where:**
  - `crates/intarsia-core/src/model/graph_step.rs:63-68` (an unstable plan returns `usable=false` with no message) and `:109-112` (a failed capture writes one stderr line, then the process stays eager for good).
  - `crates/intarsia-server/src/metrics.rs` (does not export `graph_replays` or `graph_captures`).
  - Results doc `:123-126`: the gate rule points at `generate` output, but the A/B runs through `serve`.
- **Failure scenario:** the never-linked CUDA capture fails on the first decode (for example cuBLAS under stream capture, or a launcher returning an error). The server decodes eagerly, and the A/B reports roughly Phase 1a numbers as the Phase 1b result: "graphs don't help", or a FAIL blamed on kernels.
- **Fix:**
  - Export `intarsia_graph_replays_total` and `intarsia_graph_captures_total`.
  - Warn at load when `decode_position_stable` is false, naming the differing ops.
  - Add `--require-graphs`, which makes a failed capture a hard error.
  - In the procedure, require the replay delta to be about `(gen-1) × reps` per size.

**5. [MED, benchmark validity] Profiled eager steps bias both comparisons.**
- **Where:** `crates/intarsia-cli/src/common.rs:118` (GPU default `profile_every=16`); `crates/intarsia-core/src/model.rs:657-658` (graph mode profiles every 128th engine step, counting prefill chunks too).
- **Problem:**
  - A profiled step runs eagerly with two synchronisations around each of its 844 ops.
  - In `--no-graphs` mode it happens every 16th step versus every 128th with graphs, so the eager-vs-graph speed-up is inflated.
  - Against vLLM, roughly one slow step per 128-token request costs about 1%, which is the whole gate margin (76 vs 76-78).
- **Fix:** run every A/B arm with `--profile-every 0`, and report the as-shipped setting separately.

**6. [MED, benchmark validity] The A/B procedure is not controlled enough for a margin of 0-2 tok/s.**
- **Where:** results doc `:107-116`.
- **Problems:**
  - vLLM in-flight is checked only before each run, not sampled during it. Phase 1a finding 7 asked for sampling during both runs; `guidance` is a shared fleet role.
  - The arms run back to back (ours, then vLLM), once, with no ABAB interleave and no spread reported.
  - vLLM's speculative-decoding and KV settings are not recorded with the run.
  - The "before" arm uses the disable list but keeps graphs and the new split-KV launch, so it is neither Phase 1a nor a clean kernel-only A/B.
- **Fix:**
  - Poll `num_requests_running` and `num_requests_waiting` at about 1 Hz during each run and discard any run that sees a value above 0.
  - Interleave ABAB at least 3 times and report min/median/max.
  - Store vLLM's version and `cache_config_info` in the result JSON.
  - Define "before" as the HEAD build (or disable list plus `--no-graphs`).

**7. [MED, ops risk] The procedure repeats the memory co-tenancy suspected of hanging spark1.**
- **Where:** results doc `:86-116`; `deploy/localeval/serve-and-bench.sh:30-31`.
- **Problem:**
  - One lease with `--mem 40` covers all of: a `cargo build --release`, accuracy runs at `--max-seq 16384` on the 35B, and our 38 GB server.
  - All of it sits beside `guidance`, which reserves 64 GB (`gpu_memory_utilization 0.5`) and uses native KV offloading, on 128 GB of unified memory.
  - `serve-and-bench.sh`'s `stop()` still kills only the supervisor PID and returns at once (Phase 1a finding 7, unfixed), so the slot can be released while the child still holds about 38 GB.
- **Failure scenario:** a second hang mid-run, or a fleetq release while memory is still held.
- **Fix:**
  - Measure the peak RSS of the 16k accuracy run first (spark2 or `--mem-budget`) and set `--mem` to the true peak.
  - Build outside the lease.
  - Make `stop()` wait for the PID and then `kill -9` before returning.

**8. [MED, graph coverage] Replay after a mid-sequence GatedDeltaNet restore is untested.**
- **Where:** `crates/intarsia-models/tests/graphs.rs:237-256`, which covers only `reset` (truncate to 0).
- **Problem:** `serve`'s prefix-reuse path truncates to a nonzero snapshot length (`state.rs:180-210`), then decodes through the captured graph. Neither the CPU fixture nor `accuracy --mode reuse` covers this (reuse has run only on an attention-only model). The fixture also has no MoE or quantised ops, so `moe_topk` and `moe_experts_decode` have never been inside a capture.
- **Fix:** add a `graphs.rs` case: prefill A, snapshot at a chunk boundary, decode, truncate to the snapshot, prefill B, then graph decode. It must be bitwise equal to eager. Then run the 256-step greedy, graph vs `--no-graphs`, on the real 35B.

**9. [LOW-MED, claims] The replica figures behind "graphs reach the bar" are for a different MoE kernel set.**
- **Where:** `kernels/cuda_checks/decode_replica.cu:187` launches `ik_moe_experts_bf16` (4 launches). The engine selects `moe_experts_decode_bf16` (priority 10, m ≤ 1).
- **Problem:** the 13.15 ms / 76 tok/s graph figure and the 0.57 ms side-stream figure, which results doc `:76-77` uses to argue graph replay reaches the bar, were not measured on the engine's decode kernel set. The capture of the new decode kernel was not exercised either.
- **Fix:** rerun the replica with `ik_moe_experts_decode_bf16` and state which kernel set the figures describe.

**10. [LOW, claims and golden validity]**
- **(a) The M=40 head case.** `crates/intarsia-core/src/quant/cases.rs:185` says the M=40 head case is what accuracy runs select (`*_gemm`). But `head_logits` is capped at `LOGIT_ROWS=16` (`model.rs:35`, `:1021`), and `*_gemm` needs M ≥ 33, so the model never runs that path for the head. The case is harmless but the claim is false; fix the comment.
- **(b) Possible prefix-cache hits in the golden.** `tools/golden_vllm.py` re-sends prompt plus continuation to a server with `enable_prefix_caching=True` and adds no nonce or `cache_salt`. Whether vLLM 0.26 reads cached fp8 KV blocks for `prompt_logprobs` requests is unverified. Add a per-request salt or nonce, or record that the cache was bypassed.

## The 3 most valuable next steps

1. **Make the gate able to see a regression, then run the acceptance on spark1 once it is power-cycled.** Concretely:
   - fix findings 1-4 (self-golden from the HEAD eager run, a tight same-weights gate, bitwise gemm tests, a replay counter checked in the A/B);
   - run `kernels test`, the 256-step graph-vs-eager greedy and the accuracy runs;
   - then the interleaved A/B with `--profile-every 0` and vLLM in-flight sampled during each run.

   None of this needs new kernel work, and it decides the Phase 1 gate.
2. **Close the remaining decode gap inside the engine's captured graph.**
   - Port the replica's second-stream branches (shared expert, GDN z/b/a projections, worth about 0.57 ms). Every kernel currently shares one `self.scratch` buffer (`model.rs:583`), so branches that run concurrently need their own scratch regions.
   - Write the fused GDN decode kernel: conv, gating, delta rule and gated norm in one pass over the f32 state, which today moves 2.5x the necessary state bytes, about 0.86 ms.
3. **MTP speculative decoding from the checkpoint's `mtp.*` head, but only after step 1.**
   - Split-KV already accepts T ≤ 16 for verify.
   - Verification needs one GDN snapshot (or a cheap per-position state checkpoint) per draft token, and a graph per verify width.
   - Gate it on the self-golden, so that speculation cannot quietly cost accuracy.

The files that matter most:
- `/opt/code/intarsiaml/crates/intarsia-cli/src/cmd/accuracy/baseline.rs`
- `/opt/code/intarsiaml/crates/intarsia-cli/src/cmd/accuracy.rs`
- `/opt/code/intarsiaml/crates/intarsia-core/src/model/graph_step.rs`
- `/opt/code/intarsiaml/crates/intarsia-core/src/model.rs`
- `/opt/code/intarsiaml/crates/intarsia-backend-cuda/src/quantmoe.rs`
- `/opt/code/intarsiaml/crates/intarsia-backend-cuda/tests/quantmoe.rs`
- `/opt/code/intarsiaml/crates/intarsia-server/src/metrics.rs`
- `/opt/code/intarsiaml/docs/results/2026-10-13-phase1b.md`
