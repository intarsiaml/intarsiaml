# Phase 2a adversarial code review (Opus 5.5, 2026-10-13)

Read-only review run as the last stage of the Phase 2a workflow. Verbatim.

**Phase 2a HIP review (worktree /opt/code/intarsiaml-hip, branch phase2a/hip, uncommitted diff plus untracked files). 12 findings, most severe first.**

1. **The GEMV tuning knob can silently skip the launch.** `kernels/hip/matmul.hip:113` vs `:131-134`.
   - Problem: the parser accepts R∈{1,2,4} and U∈{2,4,8,16}, but `gemv_vec` only instantiates (2,4), (1,2/4/8/16), (2,2/8) and (4,2/4/8). For `INTARSIA_HIP_GEMV=2,16` or `4,16` no case matches, nothing is launched, and `ik_launch_status()` still returns IK_OK.
   - Failure scenario: someone runs `serve` with `INTARSIA_HIP_GEMV=2,16` during a tuning sweep. Every decode matmul and logits GEMV leaves stale data in its output and the model emits garbage. The admission cache key (`Backend::fingerprint()`) does not include the env var, so a cached ADMITTED result does not catch it.
   - Fix: restrict the parser to the instantiated set (or instantiate the missing ones), make an unmatched config return IK_ERR_INVALID_ARG, and put the env knobs into the fingerprint.

2. **Admission cannot see causal leaks or split-boundary off-by-one errors.** `crates/intarsia-backend-hip/tests/attention.rs:30-33`, `kernels.rs:44` (BF16_TOL 1.5e-2 / 6e-3).
   - Problem: the KV cache is `cap = p + t + 3`. The slots past the causal edge hold random data from the same distribution as valid keys, and the result is graded against a 6e-3 rel-L2 budget.
   - Failure scenario: one extra key leaking past `pos0 + qrow` (flash `need_mask` boundary, or the split-KV `end`/`visible` clamp) at kv_len 4k shifts the output by roughly 1/kv_len. That passes easily. The flash kernel already uses 2.4e-3 of the 6e-3 rel-L2 budget (results doc §1, 40% of it), so there is little headroom left for spotting real errors.
   - Fix: fill every K/V slot at or beyond `pos0 + t + 1` with poison (K = ±1e4, V = NaN or 1e4) in both the attention sweep and the core admission sweep. Add exact chunk-edge cases (kv_len = k·64 ± 1, and q-block starts at a BK multiple ± 1). Then any leak fails at tolerance 0.

3. **§9.2 bandwidth numbers are cache-hot, not DRAM.** `crates/intarsia-backend-hip/tests/quantmoe.rs:74-93`; `docs/results/2026-10-13-phase2a-hip.md:187-205`, `:191`.
   - Problem: `bench()` relaunches the same buffers every rep. `matmul_bench.rs` instead rotates over at least 768 MB of copies.
   - The fp8 qkv weights (16.8 MB), z, o_proj and the moe_experts T=1..16 working sets all fit in the 64 MB Infinity Cache.
   - The NVFP4 lm_head at 672 GB/s is above what I believe the R9700 DRAM spec is (about 640 GB/s; not checked against a datasheet). The doc's explanation that "627 is a lower bound" is therefore wrong: that number is cache-assisted.
   - Failure scenario: the "% of 627" column and the gap analysis built on it (§9.6 items 2 and 5: fp8 GEMV at 56-71%, MoE T=1 at 37%) are not streaming numbers, so tuning effort lands on the wrong kernels.
   - Fix: use the same rotation as `matmul_bench` (≥768 MB of distinct weight copies) and rerun before quoting % of peak.

4. **§8 radiance downtime log is incomplete and its timings are wrong.** `docs/results/2026-10-13-phase2a-hip.md:141-148`.
   - Problem: the doc lists five takedowns, "about 16 minutes" in total, with `/v1/models` "15 s after every restart". The fleetq log shows 13 takedowns: #1389, 1393, 1395, 1399, 1401 plus the 35B bursts #1411, 1414, 1415, 1420, 1423, 1430, 1432, 1433.
   - Measured from acquire to "is back", that is about 18 min for the five listed plus about 12 min for the 35B bursts, so about 30 min in total.
   - The gap from finish to "is back" was up to 43 s (#1399), not 15 s.
   - In one case the lease was released before radiance was back: #1412 started on both cards at 02:03:43, and radiance-rank0 came back at 02:03:47 after #1411.
   - The 35B bursts appear nowhere in the doc.
   - Fix: regenerate §8 from `fleetq log`. In `/tmp/burst.sh`, poll `/v1/models` until it answers before `fleetq release`.

5. **MoE results now depend on batch size, unlike CUDA.** `kernels/hip/moe.hip:526-560` (WMMA switch at `:540`); `kernels/hip/moe_wmma.h`.
   - Problem: CUDA `moe.cu` has no MMA path, so a token's MoE output there is independent of T. HIP uses three paths with different accumulation orders: p1 for P≤16, grouped GEMV for T<16, and bf16 WMMA for T≥16.
   - Failure scenario:
     - The same prompt gives different hidden states depending on `--prefill-chunk` or on a remainder chunk of fewer than 16 tokens.
     - Prefix-cache reuse versus recompute differs.
     - Multi-row verify with 16 or more rows differs from serial decode.
     - Greedy outputs differ from the CUDA backend.
   - The kernel is registered as Deterministic, which is true run to run, but nothing in the registry records the loss of batch invariance. Dense GEMV vs GEMM paths already have this property, so the new part is MoE.
   - Fix: document it in the registry and quant-moe.md, or flag `moe_experts` as not batch-invariant so `exact` mode / speculation refuses it. Longer term, make the GEMV kernels accumulate in the WMMA k-permuted order.

6. **Results are not tied to code, and part of the final tree was only smoke-tested.**
   - Problem:
     - All four bench JSONs have `"fingerprint": {}`, and nothing records a git SHA. The rsync excludes `.git`, and the tree is uncommitted.
     - No accuracy or admission JSON or log is committed; the evidence lives only in `/tmp/burst{2..5}.log` on the Mac.
     - `device.rs`, `ffi.rs` and `ik_runtime.{h,hip}` changed at 02:30:54 (trace/event code), after the last burst that ran `kernels test`, `cargo test` and `accuracy` (#1430, `b3.sh`). Bursts 6 and 7 ran only generate, serve and bench.
   - I checked this the other way too. Burst 5 did rerun accuracy and admission after the `moe_wmma.h` change, so "final kernels" in §9.3 is backed. The 02:38 edit to `moe.hip` only changed a comment.
   - Fix:
     - Have the server expose engine version, git SHA and the backend fingerprint, and have `bench speed` record them.
     - Commit the accuracy and admission JSON next to the bench JSON.
     - Rerun `cargo test -p intarsia-backend-hip` on the final tree.

7. **Build reproducibility.** `crates/intarsia-backend-hip/build.rs:55-59`; `deploy/tatooine/Dockerfile.hip-dev:3`; `dev.sh` (persistent `intarsia-cargo-target` volume).
   - Problem: there is no rerun trigger on the hipcc binary or the ROCm version, and the target volume survives image rebuilds. The base image is pulled by tag (`rocm/dev-ubuntu-24.04:7.2.4`), not digest, and apt is unpinned.
   - Failure scenario: after a ROCm image bump, cargo reuses the old `libintarsia_hip_kernels.a`. The runtime fingerprint changes (new rt/drv), so re-admission passes, but the result describes kernels built by the old compiler.
   - Fix:
     - Emit `rerun-if-changed` for the hipcc path and `/opt/rocm/.info/version`.
     - Bake the hipcc version into `IK_HIP_ARCH` or the pack fingerprint.
     - Pin the base image by digest.

8. **The quant/MoE tolerance is far too loose for f32 outputs.** `crates/intarsia-backend-hip/src/quantmoe.rs:23` (same in CUDA `quantmoe.rs:23`).
   - Problem: one TOL of 1.5e-2 / 6e-3 also grades f32 outputs, notably the NVFP4 and fp8 lm_head logits. That is 15-30x looser than `F32_TOL` (1e-3 / 2e-4).
   - Failure scenario: a scale-indexing or K-tail bug in `matmul_nvfp4_gemv` / `_mma` that perturbs logits by about 1e-2 is admitted, and only shows up as small top-1 or KL drift.
   - Fix: when `out == F32`, use F32_TOL or a scale-relative tolerance. Fix both backends.

9. **INTARSIA_HIP_TRACE does not measure per-kernel GPU time.** `crates/intarsia-backend-hip/src/device.rs:100-118`, `:417`.
   - Problem: one event is recorded after each launch and the time between consecutive events is charged to the later kernel.
     - Host launch gaps and idle bubbles count as kernel time.
     - The first kernel after every synchronize is never counted.
     - Events from all streams are mixed in one list.
   - Failure scenario: §9.5 (doc `:238-250`) describes this as "per-kernel GPU time". Its "mostly 6-8 us = dispatch bound" and "about 4 ms is 700 small launches" are inferences, and the method cannot tell short kernels apart from dispatch gaps.
   - Fix: record one event before and one after each launch (or use rocprofv3 kernel timestamps), and keep per-stream lists. Until then, call it "inter-launch wall time".

10. **Some vector accesses rely on unchecked alignment.** `kernels/hip/moe_wmma.h:101`, against `moe.hip:490,492` (weights checked to 8 bytes only); `kernels/hip/matmul.hip:280-286` (uint4/float4 output stores; out alignment is never checked, unlike `dq_gemm`'s `vec_out`).
    - Failure scenario: an expert-weight or output view at an offset that is a multiple of 8 but not 16 produces misaligned 16 B global accesses. gfx12 normally tolerates this (unaligned mode), but it is slower and not guaranteed under other SH_MEM settings.
    - Fix: check 16-byte alignment of `gate_up_w`, `down_w` and `out`, or fall back to scalar paths.

11. **The TensorFold credit was dropped in the HIP ports.** `kernels/hip/ik_quant.h:1-9`, `fp8.hip:10`, `nvfp4.hip:8`.
    - Problem: `kernels/cuda/ik_quant.cuh:5-6` and `nvfp4.cu:8` credit TensorFold (MIT, `LICENSES/TensorFold-MIT.txt`). The HIP files call themselves ports or counterparts with "the same numerics contract" but carry no credit.
    - Radiance is fine as it stands. No code was ported, and `docs/backends/hip.md:37-38` correctly defers `LICENSES/radiance-NOTICE` until the first ported file.
    - Fix: carry the TensorFold attribution line into the three HIP headers.

12. **Several doc and comment claims are wrong.**
    - `results.md:178` says "The CUDA version returned early before a block barrier". CUDA uses `__syncwarp()` (`kernels/cuda/moe.cu:92`), which is correct with that early return, so nothing was fixed.
    - `moe_wmma.h:18` says "a WMMA k extent of 16 is one NVFP4 scale block". Under the k permutation, the two lane halves read different scale blocks. The code is right; the comment is not.
    - `kernels.rs:5-6` says the quant, MoE and GDN launchers are "not ported yet", which contradicts the table.
    - `attention_splitkv.hip:14` says the split depends "on kv_len only". It also depends on rows and n_kv_heads.
    - `results.md:235` compares against a 141 tok/s reference that was measured on Qwen3.5-35B-A3B (`amd-r9700-lessons.md:13`), not Qwen3.6.

I checked these by hand and found no errors:
- The e4m3, e2m1 and fp16-dot2 bit math.
- The gfx12 WMMA A/B/C layouts in the GEMM, flash, dq_gemm and moe_wmma kernels, including the k-slot permutation and the scale-byte indexing.
- The LDS double-buffer barrier ordering.
- The early-return and barrier uniformity.
- The flash causal mask.
- The split-KV scratch bound.
- The wave32 shuffle widths.

The 8B decode KL of 9.6e-4 is not a HIP regression: the CUDA run in Phase 1a scored 1.115e-3 and 9.81e-4.

**Three most valuable next steps for the R9700**

1. **HIP-graph decode capture plus radiance's one-shot all-reduce with fused RMSNorm, as the TP=2 enabler.**
   - The 35B decode step is 10.4 ms against about 3.5 ms of weight bytes. Fix item 9 first to confirm that most of the gap really is launch overhead.
   - TP=2 over 28 GB/s P2P with tiny per-layer payloads needs a latency-optimized one-shot kernel, not a ring, and only pays off once launches are captured.
   - Port it under `kernels/packs/radiance` with LICENSE and NOTICE, pinned by commit, admitted against the cpu chain `all_reduce → rmsnorm`.
   - Measure TP=2 decode on the 8B (16.4 GB, fits split) before the 35B.

2. **Prefill on the native fp8 path, plus chunked GDN.**
   - Use `v_wmma_f32_16x16x16_fp8_fp8` for W8A8: the weights are already e4m3, and the activations are quantised exactly as `xprep` does now. Today the fp8 qkv path runs at 7.5 TFLOPS and is 40% of prefill.
   - Replace the sequential `gdn_delta_rule` (11% of prefill, 400 us per layer at T=256) with a chunked form modelled on radiance `r4d_gdn_recurrent_update` / fla.
   - We are at 2-3k tok/s against radiance's 14.9k. Admission needs the saturating-activation cases from finding 2's poison-style tests.

3. **MXFP4 last, as a new grouped kernel rather than a port.**
   - While decode is launch-bound, cutting bytes per token buys little.
   - `radiance-port-plan.md` §3.6 says radiance has no MXFP4 routed-expert GEMM; only dense `r4d_gemm_mxfp4a8_*` exists. MXFP4 MoE therefore means a new grouped kernel (reusing the `moe_wmma.h` register-A design with E8M0 scales), plus deciding whether the GGUF MXFP4 experts are bit-exact.
   - Do it after steps 1 and 2. First fix the quantmoe bench rotation (finding 3), so the MXFP4 gain is measured against real DRAM numbers.
