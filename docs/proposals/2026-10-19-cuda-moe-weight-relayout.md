# Proposal: contiguous-unit re-layout of the CUDA MoE expert weights (decision needed)

Date 2026-10-19. Scope: spark1 (GB10, sm_121), Qwen3.6-35B-A3B NVFP4, CUDA. No code changed, no new measurements. Every
figure comes from **[W]** `docs/results/2026-10-18-cuda-moe-w4a16.md` (raw logs in its `raw/` directory), tagged with the
section. "Derived" means arithmetic on recorded figures; "not measured" means no recorded number exists.

## 1. Problem and expected gain

Every grouped MoE kernel (v1, W4A16 v2, opt-in FP4) reads 64 bytes of each weight row per pipeline unit. The DRAM probe gives
172-176 GB/s for 64 B runs against 258-260 GB/s for 512 B and 1024 B runs [W §2, `raw/base/stream-runs.out`]. A layer call
with all experts active streams 453 MB of weights and scales: 2.6 ms at 175 GB/s against 1.7 ms at 260 GB/s [W §2].

**The simulation numbers are kernel time reductions.** The report gives microseconds going down with a minus sign:
T=1024 3000 -> 2533 us (-15.6 %), T=2048 3598 -> 3194 us (-11.2 %), T=4096 5356 -> 5188 us (-3.1 %) [W §5]. They are medians
of 3 rounds of the whole `moe_experts_w4a16_bf16` call (E=256, K=8, H=2048, I=512), timing only: the simulated kernel gives
wrong results, so this is an upper bound, not a working kernel [W §5; `raw/final/microbench-and-tests.out`]. The microbench
ran with vLLM up, so only relative numbers count [W §4]. The report concludes "not worth the cost" [W §5, §12].

**End-to-end estimate at T=2048 (the serve prefill chunk):**
- Kernel gain: 3598 -> 3194 us = 404 us per layer call = 11.2 % [W §5].
- MoE share of prefill time: derived from the report, as its own §9 does. v2 at T=2048 is 3592 us [W §4] x 40 layers
  = 144 ms of "about 410 ms per chunk" [W §9], so about 35 %.
- Estimate: 11.2 % x 35 % = **3.9 %** less prefill time; equivalently 404 us x 40 = 16.2 ms of 410 ms. That is about
  **+4.1 % tok/s**, so the 4958 tok/s of the 4096-token A/B cell [W §9] becomes about 5160, or 79 % of vLLM's 6572 instead
  of 75 % [W §1] (derived).
- The gain is about the size of the A/B noise: the 4096 base cell varied 4512 to 4744 tok/s between rounds, 2 reps per size
  [W §9, §12]. A real kernel would need more rounds to confirm 4 %.
- **At T=4096 the sim gains only 3.1 %.** The report's finding: there, activation (A) re-reads, not the weight stream,
  are the likely limit, about 1.6 GB of bf16 A re-reads per call against 453 MB of weights. This is inference; a variant
  with A traffic removed at 4096 is "not measured" [W §2]. A larger chunk would not make the re-layout pay more.
- Effect of the new layout on decode kernels: not measured.

## 2. Option 1: second copy, only the prefill kernel moves

- Memory: the report says **+16 GB** (40 layers x 402 MB) [W §5]. I checked it: 402 MB is payload only (268 + 134 MB);
  the 50 MB of scales per layer [W §2] are extra, so with scales it is 40 x 453 MB = **18.1 GB** (derived). The sim stored
  scales in the unit too [W §5], so a real copy would need them.
- Headroom: the box has 128 GB (task statement). vLLM's footprint and the resident size of the 35B in intarsia are
  **not measured** in [W], so the free margin is not known. What [W] records: v2 itself adds 0 GB [W §10], and dev jobs beside
  vLLM ran at `fleetq --mem 4..16` [W §11, §13]. A 16-18 GB copy is at least as large as the biggest slot used there, which
  is why it must be measured against vLLM before anyone commits to it.
- Moves: only `moe_experts_w4a16_bf16`; decode, v1 and FP4 keep the old layout [W §5]. Bitwise-to-v1 stays testable.
- Cost: the memory risk, a load-time permute, and accounting for the extra bytes.

## 3. Option 2: one new layout, 0 GB extra

The loader `load_experts` in `crates/intarsia-core/src/quant/moe.rs` is backend-agnostic: it stacks the checkpoint
tensors into `[E, 2I, H/2]` for every backend, and the ABI documents that layout (`kernels/abi/ik_abi.h`). CUDA consumers:
decode (`moe_decode.cu`, `moe_experts_decode_bf16` and `_v2`), v1 mma (`moe_grouped.cu`), opt-in FP4
(`moe_nvfp4_mma_sm120a.cu`), W4A16 v2 (`moe_w4a16.cu`), and the base `moe_experts_bf16` (`moe.cu`).

Files that would change (grep of `crates/intarsia-core`, `crates/intarsia-backend-*`, `kernels/`):
- Loader and contract: `crates/intarsia-core/src/quant/moe.rs`, `kernels/abi/ik_abi.h`, `kernels/cuda/ik_cuda_abi.h`.
- CUDA kernels: `kernels/cuda/{moe.cu, moe_decode.cu, moe_grouped.cu, moe_nvfp4_mma_sm120a.cu, moe_w4a16.cu}`.
  `moe_group.cuh` stays: it does expert grouping and the weighted reduce only, no weight addressing.
- CUDA crate: `src/{quantmoe.rs, device_quantmoe.rs, kernels.rs}`; tests `tests/{moe_w4a16.rs, moe_nvfp4.rs,
  moe_nvfp4_cpu.rs, quantmoe.rs, fused_decode.rs}`; mutation scripts `scripts/cuda-moe-w4a16/mutate.py`,
  `scripts/cuda-moe-nvfp4/mutate.py`.
- Oracle and admission: `crates/intarsia-core/src/quant/{reference.rs, cases.rs}` and
  `crates/intarsia-backend-cpu/src/quant_ref.rs` use row-major tensors. They stay unchanged only if the permute is private
  to the CUDA backend at upload and admission gets a hook for it.
- **HIP shares the loader and pack format**: `kernels/hip/moe.hip`, `crates/intarsia-backend-hip/src/{ffi.rs, quantmoe.rs,
  device.rs}`, `tests/fused_decode.rs` change if the layout enters the shared contract. Any edit of `quant/moe.rs` needs
  a HIP check (the W4A16 change needed none: "Shared code touched: none" [W §12]).
- **Metal**: shares the loader, but its registry has no quantised or MoE kernels ("not ported yet",
  `crates/intarsia-backend-metal/src/kernels.rs`), so nothing there reads the packed format. Whether Metal falls back to the
  CPU oracle for MoE: not checked.

Re-validation for each moved kernel (decode, v1, FP4, and v2), following the checks used for v2:
- Bitwise tests: v2 == v1 byte for byte on random shapes, skewed routing and the real layer shape [W §7]; decode, v1 and FP4
  need the same against their own old output.
- Admission sweep `intarsia kernels test --backend cuda --op moe_experts` (v1 7 cases, v2 1 case) [W §7b].
- KL21 vs the vLLM goldens, prefix reuse and greedy ids: needs vLLM down, 10.9 min last time [W §7, §11]. Decode must be
  re-run too (81.0 / 78.1 / 70.8 tok/s at 1024 / 4096 / 16384 today [W §1]).
- Mutation testing: v2's 23 mutants were all killed [W §8] and must be re-aimed at the new addressing; FP4 and decode likewise.

Risk: a layout built for 4 KB / 8 KB prefill units may slow decode (not measured).

## 4. Option 3: do nothing

Lost: about 3.9 % of prefill time (about 200 tok/s at 4096, derived) and 4 points of the vLLM gap. Kept: no new memory, no
new risk to decode, FP4 or HIP, and the shipped +7.6 / +7.7 % prefill [W §1]. The report advises "do not start there"
[W §12]. The rest of the prefill outside this kernel "was not investigated" [W §12], so other levers are unranked here.

## 5. Recommendation

**Option 3, do nothing now;** revisit Option 1 if a free-memory margin of at least 20 GB beside vLLM is measured and a
correct-result kernel confirms the roughly 11 % kernel gain at T=2048 (the simulation is timing only).

## 6. Effort (hours; my estimates, no hours are recorded in [W])

| Option | Kernel + loader | Validation (bitwise, admission, KL21, mutants, A/B) | Total |
|---|---|---|---|
| 1 duplicate | 8-12 | 6-8 (2 vLLM windows) | 14-20 |
| 2 single layout | 20-30 (5 kernels, HIP decision) | 15-20 (all kernels, decode regression) | 35-50 |
| 3 nothing | 0 | 0 | 0 (1 to record the decision) |
