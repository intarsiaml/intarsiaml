# Proposal: core `Op` fusions for decode (P1-P5 from HIP round 5), with a recommended order (decision needed)

Date 2026-10-19. Base `origin/main` `456e6e7d`. Scope: shared core (`crates/intarsia-core`) plus the three backends
(`crates/intarsia-backend-{cuda,hip,metal}`). **Nothing is implemented and nothing was measured for this page.** Every
figure is read from a recorded document and tagged with its source and section. "Derived" means arithmetic on recorded
figures, "not measured" means no recorded number exists, "estimate" means the author's judgment (effort hours only).

Sources:

* **[R5]** `docs/results/2026-10-19-hip-round5.md` (sections: Decisions after review, Headline, Results vs base, Profile,
  What I built / Core-op proposals, What I tried and dropped, Accuracy, Mutations, Admission, Caveats, Radiance downtime).
* **[R4]** `docs/results/2026-10-19-hip-round4.md` (numbered sections, e.g. §2 Diagnosis, §12 PR caveats, §13 downtime).
* **[trace]** `docs/results/2026-10-19-hip-round5/raw/prof/decode128-r5.txt` (the profile [R5] cites, v3 build, 35B, 157 steps).
* **[C17]** `docs/results/2026-10-17-cuda-norm-gemv.md` §1 and `docs/results/2026-10-17-cuda-norm-gemv/raw/prof-dev/steps.txt`
  (CUDA decode step profile; "steps.txt" below).
* **[N17]** `docs/results/2026-10-17-cuda-noop-norm-gemv.md`, **[C14]** `docs/results/2026-10-14-core-changes-cuda-hip-check.md`
  (precedent no-op checks of core `Op` additions), **[CP3]** `docs/results/2026-10-19-cuda-prefill-3.md` (CUDA admission counts).
* Code facts are given as `path:line` on `456e6e7d` (lines are approximate to a few lines).

## 0. Summary

Setting: Qwen3.6-35B-A3B NVFP4 decode on one R9700 (gfx1201), HIP graph replay. A launch boundary inside the replay costs about
3 us (p50 3.0 us) [R4 §2; R5 Profile], so [R5] prices each removed launch at ~3 us of a 7.62 ms (profiled) step.

**That price is an upper bound.** [R5] itself lets us calibrate it (section 1): the one-launch MoE v3 removed 40 launches and
moved decode by +0.8 % (145.7 -> 146.9 tok/s), which is **1.2 to 1.4 us per removed launch** on the unprofiled step, not 3 us.
The table gives both columns. All figures are on the v3 build (488 launches/step) unless stated, because [R5]'s P1-P5 counts
come from the v3 trace.

| | fuses | new `Op` (suggested) | launches/step | [R5] estimate (3 us/launch, % of 7.62 ms) | calibrated (1.4 us/launch, derived) | cross-block wait? |
|---|---|---|---|---|---|---|
| P1 | `MoeTopk` into `MoeExperts` | `MoeRouted` | -40 | ~120 us (1.6 %) | ~56 us (+0.8 % tok/s) | no (recompute variant) |
| P2 | shared `MatmulNvfp4Glu` + shared-down `MatmulNvfp4` + `MoeCombine` | `SharedExpert` | -80 | ~240 us (3.1 %) | ~112 us (+1.7 %) | yes (glu -> down) |
| P2a (author's split of P2) | shared-down `MatmulNvfp4` + `MoeCombine` only | `SharedDownCombine` | -40 | not in [R5]; 40 x 3 us = 120 us (derived) | ~56 us (+0.8 %) | **no** |
| P2' | P1 + P2: topk, experts, shared expert, combine as one grid | `MoeBlock` | -160 | ~480 us (6 %) | ~224 us (+3.4 % = ~151.9 tok/s from 146.9) | yes (two edges) |
| P3 | `GdnConv` into `GdnDeltaRule` | `GdnConvDelta` | -30 | ~90 us (1.2 %) | ~42 us (+0.6 %) | no, but a conv-state write hazard |
| P4 | `GateSplit` into `QkNormRopeKv`; `SigmoidMul` into the o_proj GEMV | `GateSplitQkNormRopeKv`, `SigmoidMulMatmulFp8` | -20 (-10 each) | ~60 us (0.8 %) | ~28 us (+0.4 %) | no |
| P5 | attention input `RmsNormAdd` into the k/v `MatmulFp8Pair` | `RmsNormAddMatmulFp8Pair` | -10 | ~30 us (0.4 %) | ~14 us (+0.2 %) | no |
| P2'+P3+P4+P5 | | | -220 | ~0.66 ms ([R5] says "~8.5 %"; 660/7622 = 8.7 %, derived) | ~308 us (+4.7 %); ~264 us (+4.0 %) at 1.2 us/launch | |

Sources of the columns: launches and the [R5] estimate are `[R5 What I built / Core-op proposals]` (the table there); the P2a row and
the whole "calibrated" column are derived here (section 1). tok/s gain is `1/(1 - saved/step) - 1` on the unprofiled v3 step of
6.81 ms (= 1000/146.9, derived).

**Recommendation in one line:** land the three sync-free folds (P5, P2a, P4b) as one core change while a HIP-only task kills
`v3-nofence`; take P2' only after v3 is default-on and a microbench shows the top-k and shared-expert work do not eat the saving;
do not build P1 on its own. Details in section 11.

**Scope note on P2'.** The brief lists "router" in P2'. [R5]'s P2' (-160) covers topk, experts, shared expert and combine
(5 launches per MoE layer -> 1 = -4 x 40 layers). The router **logit GEMV is not in it**: it is already folded with the
residual norm and the shared-gate logit into `RmsNormAddMatmulPair` (40 of its 69 launches per step, `[R5 Profile]`). Folding it
in would add a third dependency (the logits and the normalised activation are needed by every block before top-k can start),
so it would be one more grid-wide wait; the extra -40 launches would be derived (40 x the same per-launch price), not recorded.
This page keeps [R5]'s scope.

## 1. Evidence and calibration

**Baselines [R5 Profile, Headline].** Default build (v2 experts, two launches): 528 launches per step. Opt-in v3
(`INTARSIA_HIP_MOE_V3=1`, one launch): 488. [R5]'s P1-P5 numbers are counted on the 488 trace. On the default build P2' would also
absorb v2's second experts launch: -200 vs 528 (derived: 160 + the 40 v3 already removes). That extra 40 is v3's own measured +0.8 %.

**Per-launch price, three readings.**

| reading | numbers | per removed launch |
|---|---|---|
| boundary gap in replay [R4 §2] | p10/p50/p90 2.92 / 3.00 / 3.16 us; idle = launches x ~3.2 us (1.834 ms / 578) | ~3.0 us |
| v3, profiled idle [R5 Profile, trace] | idle 1.681 -> 1.569 ms for -40 launches | 2.8 us (derived) |
| v3, profiled wall [R5 Profile] | wall 7.749 -> 7.622 ms (kernel time 6.068 -> 6.053 ms) | 3.2 us (derived) |
| **v3, unprofiled A/B** [R5 Results vs base] | 128 tokens 145.7 -> 146.9 tok/s: step 6.863 -> 6.807 ms | **1.4 us (derived)** |
| | 16k tokens 128.2 -> 129.0 tok/s: step 7.800 -> 7.752 ms | **1.2 us (derived)** |

The profiled step (7.62 ms) is longer than the unprofiled one (6.81 ms at 128 tokens, derived from tok/s), and the unprofiled
A/B moved by less than half of what the profile predicted (+0.8 % measured vs 127 us / 7622 us = +1.7 %, derived). The reason is
not recorded ("not measured"). The proposal therefore uses 1.2 to 1.4 us as the planning price and [R5]'s 3 us as the ceiling. Both
columns are percentages of step time, **not** tok/s gains; tok/s gains are given where stated. Round 4 shows the same pattern: its
50 removed launches were "about half" of a +3.8 % gain, the other half kernel time [R4 §1].

**What the removed launches cost in kernel time (trace, v3 build, per step, avg per launch).** These are the numbers a fused
kernel has to absorb without getting slower:

| kernel | n/step | us/step | avg us |
|---|---|---|---|
| `moe_experts_p1_v3_kernel<4>` | 40 | 1476.9 | 36.9 |
| `nvfp4_gemv_v2_kernel<1, bf16, false>` (40/step: by count the shared-down; not matched to the call site by name) | 40 | 473.7 | 11.8 |
| `nvfp4_glu_kernel` (shared gate/up) | 40 | 172.1 | 4.3 |
| `moe_topk_v2_kernel` | 40 | 156.2 | 3.9 |
| `moe_combine_kernel<bf16>` | 40 | 62.6 | 1.6 |
| `gdn_conv_kernel<true>` / `gdn_delta_rows_v2_kernel<128,2,2>` | 30 / 30 | 61.2 / 191.0 | 2.0 / 6.4 |
| `gate_split_kernel` / `sigmoid_mul_kernel` | 10 / 10 | 10.0 / 10.5 | 1.0 / 1.0 |
| `rmsnorm_kernel<true,true>` (10 attention layers + the final norm) | 11 | 23.3 | 2.1 |

Two consequences. (1) `moe_topk_v2_kernel` takes 3.9 us per launch, more than the 1.2-1.4 us a removed launch is worth; recomputing
top-k inside every role of P1/P2' puts part of that latency on the critical path (the expert ids are needed to address the weight
loads), so P1 can be a net loss. Not measured. (2) The four small MoE kernels P2' absorbs sum to 864.6 us/step (156.2 + 172.1 +
473.7 + 62.6, derived); that is an upper bound on any kernel-time overlap gain (shared expert running beside the routed experts
in one grid), also not measured.

**CUDA and Metal baselines.** CUDA decode on spark1 is **625 launches per step** (724 -> 625 after the norm folds) [C17 §1 table;
`raw/prof-dev/steps.txt`: "launches/step 625 span 12261.3 us kernel sum 12196.6 us idle inside the step 64.6 us (0.5%)"].
Verified; the later CUDA reports (`2026-10-18-cuda-moe-w4a16.md`, `2026-10-19-cuda-prefill-3.md`) are prefill/MoE-prefill work and
restate no decode launch count, so 625 is the last recorded one. Metal registers no quantised, MoE or GatedDeltaNet op
(`crates/intarsia-backend-metal/src/kernels.rs:307-308`: "not ported yet, so the registry reports `NoKernel` for them").

## 2. Common mechanism: how a fused op is tried and what falls back (applies to every proposal)

**The pattern (verified in code).** `ExecRunner::run_fused` (`crates/intarsia-core/src/model.rs:150`) calls `Model::try_fused`
(`model.rs:1381`), which runs the op only if `fused_ok` (`model.rs:1399`) says so: fusion not disabled (`tp.world > 1` or
`INTARSIA_NO_FUSION=1` turn it off), the call passes `OpCall::check`, and `registry.select` finds an admitted kernel (in
`forward_verify` the call is checked with `exact_rows`). Otherwise it returns `Ok(false)`, runs nothing, and the caller runs the
unfused ops. The `OpRunner::run_fused` default is `Ok(false)` (`crates/intarsia-core/src/quant/moe.rs:134`), so runners that do not
override it (e.g. `crates/intarsia-backend-cpu/tests/quant_hf.rs`) never see a fused op.

**Precedent, `Op::QkNormRopeKv` (HIP round 4).** The call site is `Model::qkv_fold` (`model.rs:~1445-1500`): it builds the
`OpCall`, `if !self.fused_ok(&call) { return Ok(false) }`, and the caller then runs the old q-norm / rope / append chain. The op
was added in core first with HIP "UNIMPLEMENTED"; HIP then registered `qk_norm_rope_kv_bf16` and removed it from the list, "so no
core change" at that point [R4 §4.2]. The fallback was **measured**: `INTARSIA_HIP_DISABLE=...qk_norm_rope_kv_bf16` makes `fused_ok`
find no kernel, and the 4408-token prefill (842 / 838 ms), decode (135.7 / 136.0 tok/s) and 256 greedy ids equal base [R4 §12,
W8]. Core-op additions with no CUDA/Metal kernel were also checked bit-for-bit on CUDA [N17] and CUDA+HIP [C14].

**Conditions this proposal must keep for "no-op for a backend without the kernel".**

1. Every new call site is try-then-unfused, with the unfused path untouched and its intermediate buffers still allocated
   (`MoeScratch` etc.). Resident memory is then unchanged.
2. **Batched decode.** `check_batch_ops` (`crates/intarsia-core/src/plan.rs:282`) fails the plan load (`Error::Unsupported`) if the
   solo decode step and a batched step contain different op multisets ("the two would fuse differently"). A fused op that is
   selectable at 1 row but not at the batch's row count would therefore **fail the load instead of falling back**. Every HIP kernel
   registered for a new op must accept all row counts (like `qk_norm_rope_kv_bf16`, "at every row count" [R4 §4]); v3's
   `T*K <= 16` limit [R5 What I built, 2] must become an internal dispatch to the unfused kernels, as v3 already does for prefill
   shapes (`kernels/hip/moe.hip:1228` comment), not a `when` refusal. Batched decode on HIP "was not exercised end to end" [R4 §12].
3. **The CPU backend is not a no-op backend.** It registers `ref_<op>` and `cpu_f32_<op>` for every `Op::ALL`
   (`crates/intarsia-backend-cpu/src/lib.rs:156`), so models run on it take the fused path. `fused_ref.rs` defines each fused op as
   "the composition of the ops it replaces, including the output rounding of every intermediate tensor"
   (`crates/intarsia-backend-cpu/src/fused_ref.rs:1-5`), so outputs stay identical by construction, but tests that count ops
   in a recorded plan change (see the shared-file list).

**Shared files touched by *any* new `Op` (checked by grep on `456e6e7d`).** Listed once; each proposal below lists only what
differs. The first group is needed before any backend work, because tests iterate `Op::ALL`.

| file | what changes |
|---|---|
| `crates/intarsia-core/src/ops.rs` | enum (`~:87`), `Op::ALL: [Op; 30]` (`~:108`, length), `as_str` (`~:127`), `arity` (`~:167`), the doc table at the top (`:20-50`), `check`/`shape` dispatch (`~:463-465`, `~:557-559`), a `check_*` unit test per op (`~:624`) |
| `crates/intarsia-core/src/quant/ops.rs` | `check` (`:20`), `shape` (`:251`), scratch size fns (`:318-337`): MoE / NVFP4 / FP8 ops (P1, P2, P2a, P2', P5, P4b) |
| `crates/intarsia-core/src/gdn.rs` | `check` (`:65`), `shape` (`:144`): `GateSplit` / `SigmoidMul` / `GdnConv` / `GdnDeltaRule` family (P3, P4a) |
| `crates/intarsia-core/src/quant/cases.rs` | admission cases: a case builder per op (like `rmsnorm_add_matmul_pair_case` `:156`), `sweep` (`:261`), `label` (`:468`), `case_from_call` (`:553`), the op list in its test (`~:620`) |
| `crates/intarsia-core/src/admission.rs` | the same for the non-quant ops: case generation (`~:346`, `~:537-607`), arity table (`~:823`), label (`~:862`), expected-output (`~:1012-1080`) (P3, P4a) |
| `crates/intarsia-backend-cpu/src/fused_ref.rs`, `reference.rs` (`~:58-66`), `fast.rs` (`~:18-24`) | the composition oracle and the dispatch lists |
| `crates/intarsia-backend-hip/src/kernels.rs` | `UNIMPLEMENTED` (`:79`), asserted over `Op::ALL` at `~:267-269`: an unimplemented new op must be listed, an implemented one removed |
| `crates/intarsia-backend-cuda/src/kernels.rs` | `names_unique_and_ops_covered` (`~:411-415`) fails for any op CUDA lacks unless exempted (today `Sample`, `QkNormRopeKv`). [N17] records this exact one-line change for the norm folds |
| `crates/intarsia-backend-cpu/tests/admission.rs` (`:15`), `tests/ops.rs` (`:286`) | require every op to have an admitted CPU kernel with >0 cases and `cpu_f32` == `ref` on the first 6 cases per op: **a sweep case per new op is mandatory** |
| `crates/intarsia-models/tests/graphs.rs` (`~:290-296`), `batch_metal.rs` (`~:68-72`) | assert op counts of the recorded plan on the CPU backend; counts change when the CPU backend takes a new fused op |
| model call sites | per proposal below |

Metal has no `Op::ALL` loop; no Metal source edit is expected (to be confirmed by the Mac build). HIP backend files that every HIP
implementation touches (list from [R5 Files]): `crates/intarsia-backend-hip/src/{kernels,quantmoe,ffi,device}.rs`,
`kernels/hip/*.hip` + `ik_hip_abi.h`, `crates/intarsia-backend-hip/tests/*`, `docs/backends/hip.md`,
`scripts/hip-decode/{mutate.py,remote.sh}`, `docs/results/<date>-*`.

**Staging.** Land the core change (ops, checks, CPU oracle, cases, call sites, table-test edits in the CUDA and HIP crates) as its
own PR with **no kernel anywhere**, exactly like [N17]/[C14]: it must be a no-op on CUDA, HIP and Metal. A second PR registers the
HIP kernels. This keeps the CUDA/Metal check (section 9) to one window per core PR.

---

## 3. P1 `MoeRouted`: top-k folded into the experts

**a. What fuses.** `Op::MoeTopk` (logits `[T,E]` -> `topk_w`, `topk_id`) into `Op::MoeExperts`. New op `MoeRouted`: inputs
`n`, `logits`, `gate_up_w`, `gate_up_s`, `down_w`, `down_s`, `gscales` (7), output `routed` (or `out` without a shared expert), attrs
`top_k`, `norm_topk`. The top-k results are consumed only by `MoeExperts` (`quant/moe.rs`, `forward_folded`), so nothing else needs
them. Two variants in [R5]: every role recomputes the 256-logit top-k, or role 0 publishes it.

**b. Gain.** -40 launches; [R5] ~120 us (1.6 %); calibrated ~56 us (+0.8 % tok/s), both derived from the launch count. Works on
v2 (two-launch) or v3 (one-launch); on v2 the recompute variant needs no wait because the launch boundary orders everything. **Risk to
the number:** `moe_topk_v2_kernel` takes 3.9 us per launch (section 1), longer than the saving; if the in-role recompute costs a
large part of that, the net is zero or negative. Not measured. A microbench of the in-block top-k (one wave per token, 8 rounds of
select-and-shuffle, `kernels/hip/moe.hip:100-135`) must precede any commitment.

**c. Shared files.** The common list, `quant/ops.rs` and `quant/cases.rs` (a `MoeRouted` case = `experts_case` + `topk_case` with
ties), and the call site `MoeWeights::forward_folded` (`crates/intarsia-core/src/quant/moe.rs:229-320`): replace the
`MoeTopk` + `MoeExperts` pair by `run.run_fused(Op::MoeRouted, ...)`, falling back to the two `run.run` calls on `false`. CPU oracle:
`fused_ref.rs` composition `MoeTopk` -> `MoeExperts` with the top-k outputs as temporaries.

**d. Backend impact.** If a backend lacks the op: `run_fused` -> `false` -> the existing two `run.run` calls, i.e. no change
(mechanism in section 2). CUDA: its MoE chain per layer is topk, `moe_dec_gate_up`, `moe_dec_down_v2`, `nvfp4_glu`, nvfp4 GEMV,
`moe_combine` (40 launches each per step, `raw/prof-dev/steps.txt`); P1 would remove 40 of 625. CUDA idle is 64.6 us for the whole
step (0.5 %), i.e. ~0.10 us per launch (derived), so the launch-count gain is ~4 us; any real CUDA gain would be kernel time
(CUDA `moe_topk_kernel` 3.5 us avg, 139.1 us/step) and is not measured. Metal: nothing to gain (no MoE kernels).

**e. Check plan.** HIP: (1) new `tests/moe_routed.rs`: bitwise vs the launch chain `moe_topk` -> `moe_experts` on the shapes of
`tests/moe_v3.rs` (11 shapes + duplicate/invalid ids), ties in the logits (the topk tie rule: lowest expert wins), 200 back-to-back
launches, 20 graph replays; (2) `intarsia kernels test` full sweep (not `--quick`, which fails on base too [R5 Admission]): baseline 46
kernels / 808 cases (35B), 46 / 821 (8B) plus the new kernel's cases, existing counts unchanged; (3) KL21 on the four 35B/8B goldens
identical to [R5 Accuracy] (35B short 7.417e-3 / 1.172e-2; long 4.356e-2 / 9.273e-3 with the pre-existing prefill floor FAIL; 8B
2.335e-4 / 3.626e-4 and 3.966e-4 / 3.573e-4), prefix reuse PASS; (4) 256 greedy ids after the 4408-token prompt identical (`ids4k`);
(5) mutants in `scripts/hip-decode/mutate.py`: top-k tie rule flipped, normalisation dropped, one role reads another token's logits,
`K` off by one, plus (publish variant only) the acquire-fence mutant; all non-equivalent ones killed; (6) interleaved A/B x3 in one
window at 128 / 1024 / 4096 / 16384 tokens, `--require-graphs`, launches/step from the profile 488 -> 448 (v3) or 528 -> 488 (v2).
CUDA / Metal: section 9. **Slots:** section 10.

**f. Risk.** Recompute variant: no cross-block wait; risk is the latency above and the batch-consistency rule (kernel must accept all
row counts). Publish variant: adds an acquire-wait on a producer role, inheriting `v3-nofence` (section 8).

**g. Effort (estimate).** Core + CPU oracle + cases + call site 5 h; HIP kernel (recompute) 8 h; microbench 2 h; tests and mutants
5 h; CUDA/Metal no-op checks 4 h (shared with other core PRs). About 24 h, plus 2 GPU windows.

---

## 4. P2 `SharedExpert` and P2a `SharedDownCombine`: the shared expert and the combine

**a. What fuses.** P2 [R5]: `MatmulNvfp4Glu` (shared gate+up+SiLU-GLU) + the shared-down `MatmulNvfp4` + `MoeCombine` as one op
`SharedExpert`. Inputs `n`, the six glu parts, the three down parts, `routed`, `sl` (the shared-gate logit, already produced by
`RmsNormAddMatmulPair`/`MatmulPair`), output `out`. The combine arithmetic is
`out = bf16(routed + bf16(bf16(sigmoid(sl)) * so))` (`kernels/hip/moe.hip:1019-1030`); `so` and `routed` are bf16 tensors, so a
fused kernel must round them exactly where the unfused chain does.

**Author's split (not in [R5]).** The glu -> down edge needs every one of the shared intermediate's outputs in every down row-block,
i.e. a grid-wide wait. The down -> combine edge does not: `combine` is element-wise on the down GEMV's output row, so it can be the
epilogue of the down kernel with **no cross-block dependency** (`routed` and `sl` come from earlier launches). So:

* **P2a `SharedDownCombine`** (`MatmulNvfp4` + `MoeCombine`): -40 launches, sync-free. Checked in the kernel: in
  `nvfp4_gemv_kernel` (`kernels/hip/nvfp4.hip:19-58`; the v2 kernel the trace shows is documented bit-identical to it, `:~60`) one
  warp owns one output row, loops over all of K itself, reduces with `ik_warp_sum` and lane 0 stores `s * g`; there is no split-K, no
  atomic on `out` and no second pass. The template flag `false` in the trace name is `XS` (activation staging in LDS,
  `INTARSIA_HIP_XSTAGE`, default off, `:242-250`). So a row is complete inside one warp and the combine can run on lane 0 right
  after the reduction. The recorded time of the combine kernel is
  1.6 us avg (62.6 us/step) and the shared-down kernel 11.8 us avg [trace]; the epilogue keeps the second and absorbs the first.
* **P2b** (glu + down, the cross-block part): the other -40 of [R5]'s -80; only worth building inside P2'.

**b. Gain.** P2: -80 launches, [R5] ~240 us (3.1 %), calibrated ~112 us (+1.7 %). P2a: -40 launches, 120 us gap-only, calibrated
~56 us (+0.8 %), all derived.

**c. Shared files.** Common list, `quant/ops.rs`, `quant/cases.rs`; call site `forward_folded` (`quant/moe.rs:~296-322`, the shared
glu / down / combine part). P2a's call needs `routed`, `so` is no longer materialised; the unfused fallback reads the same `ws.sh_*`
buffers as today. CPU oracle: composition `MatmulNvfp4` -> `MoeCombine` (P2a), `MatmulNvfp4Glu` -> `MatmulNvfp4` -> `MoeCombine` (P2).

**d. Backend impact.** Fallback as section 2 (falls back to the glu / down / combine calls in the existing order). CUDA: shared-down
(40) and combine (40) are separate launches there too [steps.txt]; P2a would remove 40 of 625, ~4 us of idle (0.10 us each, derived);
the two kernels' recorded time is 164.7 + 41.7 us/step (steps.txt), the only place a CUDA gain could come from; not measured.
Metal: nothing (no NVFP4/MoE kernels).

**e. Check plan.** HIP: bitwise vs the unfused chain on the shared-expert shapes of the 35B (`tests/fused_decode.rs` is the pattern
for `MatmulNvfp4Glu`), including bf16 rounding of `so` and of the sigmoid (mutants: sigmoid not rounded, product rounded after the
add, wrong token's `sl`); admission, KL21 (identical digits as in P1), ids, A/B as in P1; P2 adds the sentinel stress test of
section 8 for the glu -> down edge. CUDA / Metal: section 9.

**f. Risk.** P2a: low (row-local epilogue, pattern of `RmsNormAddMatmulPair`). P2: the glu -> down wait is the same class as v3's
gate_up -> down wait and inherits the unkilled `v3-nofence` mutant (section 8).

**g. Effort (estimate).** P2a: core 5 h + HIP kernel 8 h + tests/mutants 4 h + checks 4 h = ~21 h, 1-2 GPU windows. P2 (with wait):
core 6 h + HIP 20 h + tests incl. stress 14 h + checks 5 h = ~45 h.

---

## 5. P2' `MoeBlock`: the whole MoE block after the router logits in one launch

**a. What fuses.** `MoeTopk`, `MoeExperts`, shared `MatmulNvfp4Glu`, shared-down `MatmulNvfp4`, `MoeCombine` (P1 + P2). New op
`MoeBlock`: inputs `n`, `logits`, `sl`, the five expert tensors, the nine shared parts (17 inputs), output `out`. 5 launches per MoE
layer -> 1 (-4 x 40 = -160 on the v3 build, [R5]; -200 on the default build, derived). Not included: the router GEMV (section 0 scope
note).

How it maps onto v3's mechanism ([R5 What I built, 2]; `kernels/hip/moe.hip:769-990`): roles come from an atomic ticket, not
`blockIdx`, so a waiting role only waits on roles that are already running and the wait cannot deadlock for any dispatch order; the
producers fence + add 1 to a per-(device, stream) counter, the consumers' thread 0 spins with acquire at agent scope. P2' needs a
**chain of three role classes** (top-k and shared glu -> gate_up and shared down inputs -> down + combine) with the ticket order
following the dependency order; the same deadlock argument holds for any acyclic chain. [R5] calls the shared expert "a 9th,
sigmoid-gated expert" - that is a scheduling idea only: the shared expert is computed by `nvfp4_glu_kernel` / `nvfp4_gemv_kernel`
(`kernels/hip/nvfp4.hip:21,170`), the routed ones by `moe_gate_up_p1_v2_kernel` / `moe_down_reduce_p1_v2_kernel`
(`moe.hip:577,668`). For bitwise equality the shared roles must carry the **shared kernels' arithmetic**, not reuse the experts'
code; whether the two are bit-identical is not checked here ("not measured").

**b. Gain.** -160 launches. [R5]: ~480 us (6 %). Calibrated: ~224 us, +3.4 % tok/s (146.9 -> ~151.9 at 128 tokens, derived);
at 1.2 us/launch +2.9 %. Possible upside beyond the launch price: overlap of the shared expert (and top-k) with the routed experts,
bounded by 864.6 us/step (section 1); not measured. Possible loss: the top-k and shared-expert latency on the critical path.

**c. Shared files.** Common list + `quant/ops.rs`, `quant/cases.rs` (a `MoeBlock` case builder combining `experts_case`,
`topk_case`, `combine_case`, `matmul_nvfp4_glu_case`), call site `forward_folded` (`quant/moe.rs:229-320`): after the router /
pair logits, `run.run_fused(Op::MoeBlock, ...)` when the weights are NVFP4 with a shared expert (the same conditions as today's
`glu_done`), else the existing sequence. Scratch: `quant/ops.rs:337` `moe_experts_decode_scratch`-style declaration (h plus the
shared intermediate). Lines to read first: `quant/moe.rs:260-320` and the OpRunner doc at `:121-136`.

**d. Backend impact.** Falls back to the existing five calls (section 2). The batch rule applies strongly: the HIP kernel must serve
every row count (v3's `T*K <= 16` becomes an internal dispatch to the v2 / unfused kernels). CUDA: its chain is 6 launches per MoE
layer (topk, gate_up, down, glu, shared-down, combine; 40 each per step, steps.txt), so a CUDA `MoeBlock` would remove up to -200 of
625; CUDA idle is 64.6 us/step, so the launch-count gain is ~21 us (0.17 % of 12.26 ms, derived). The four absorbed small kernels
cost 669.1 us/step on CUDA (139.1 + 323.6 + 164.7 + 41.7, derived) - an upper bound on a kernel-time gain, not measured. [C17]'s own
-99 launches moved the step by -127 us but idle only by -10 us (the rest was kernel time), the same warning. Metal: nothing.

**e. Check plan.**

* HIP bitwise: new `tests/moe_block.rs` against the unfused chain (the five launches), the shapes of `moe_v3.rs`, duplicates / invalid
  ids, ties, 200 back-to-back launches with changing activations on one scratch, 20 graph replays, and the skew seam extended to every
  producer role (R5's `INTARSIA_HIP_MOE_V3_SKEW` is only the last gate_up role).
* **Sentinel stress test** from [R5 Decisions] (skew per role, h and the shared intermediate pre-filled with NaN / `0x7fc0`
  before every launch, consumers' inputs checked for the sentinel, many launches, several CU-occupancy settings) - mandatory before
  any default-on decision.
* Admission: full sweep, 46 / 808 (35B) and 46 / 821 (8B) baselines + the new kernel's cases. KL21 and prefix reuse as in P1; 256
  greedy ids after the 4408-token prompt; mutants (role order, ticket reset, fence, wait count, shared-vs-routed pointer swap, sigmoid
  rounding, combine order); A/B x3 in one window at four context sizes; profile launches/step 488 -> 328 (v3) or 528 -> 328.
* Rollback test: `INTARSIA_HIP_DISABLE=<kernel>` gives the base numbers (as [R4 §12]).
* CUDA / Metal: section 9.

**f. Risk (the main one: cross-block ordering).** Round 5 kept `moe_experts_bf16_v3` opt-in because the mutant `v3-nofence`
(acquire fence before the h reads dropped) **survives**: it is not equivalent (without the fence the HIP memory model allows a
stale read) and no test makes the stale read happen; a missed acquire in a spin-wait "risks a hang or a stale read for only +0.8 %"
[R5 Decisions, Mutations]. The same survivor was seen on the dropped split-KV v2 (`skv-nofence`). P2' multiplies the number of
wait edges (shared glu -> shared down, top-k -> expert roles, gate_up -> down), each with a producer fence and a consumer acquire,
so it inherits that unkilled-mutant class once per edge; and it needs the stress test above to exist first. Other risks: the
counter pool is 16 slots per (device, stream), a 17th stream falls back (correct, slower) [R5]; 1 KB resident counters exist
already [R5 Memory]; the top-k latency and arithmetic-family equality described above; plan-load failure if the kernel is not
selectable at all row counts. **P2' first conflicts with the current opt-in decision for v3** [R5 Decisions]: shipping a wider
version of the same mechanism as default would reverse that decision, so P2' must stay opt-in until `v3-nofence` is killed.

**g. Effort (estimate).** Core + CPU oracle + cases + call site 8 h; HIP kernel 30 h; stress test and mutants 20 h; no-op checks 6 h;
plus killing `v3-nofence` first (~10 h, section 8). About 64 h (+10 h), 4-5 GPU windows (R5 used 5 windows for v3 + flash v2 + f8 [R5
Radiance downtime]).

---

## 6. P3 `GdnConvDelta`: conv folded into the delta rule

**a. What fuses.** `GdnConv` (causal conv + SiLU, in-place conv state `[C, K-1]` f32) into `GdnDeltaRule` (decode and prefill call
the same op; the conv output feeds only the delta rule). New op `GdnConvDelta`: inputs `qkv`, conv taps, `a`, `b`, `A_log`,
`dt_bias`, outputs `core`, delta state, conv state (in place). The model calls both with plain `self.exec`
(`crates/intarsia-core/src/model/gdn_block.rs:185-198`), no `try_fused` yet.

**b. Gain.** -30 launches; [R5] ~90 us (1.2 %); calibrated ~42 us (+0.6 %). The conv kernel is 2.0 us avg (61.2 us/step) and its
q/k part would be recomputed by every value-row block of a head (`gdn_delta_rows_v2` runs "one wave = 16 value rows of one head"
[R4 §4]), so the saved kernel time is partly given back; not measured.

**c. Shared files.** Common list with `gdn.rs` and `admission.rs` instead of the `quant/*` pair; call site `gdn_block.rs:~185-198`
(`try_fused` then the two `exec`s); CPU oracle composition `GdnConv` -> `GdnDeltaRule` (`crates/intarsia-backend-cpu/src/fused_ref.rs`
+ `gdn_ref.rs`). No `PER_ROW_OPS` change (the GDN ops are not in it).

**d. Backend impact.** Fallback as section 2. CUDA: `gdn_conv_kernel` and `gdn_conv_state_kernel` are two launches there, 30 each
per step [steps.txt], and `gdn_delta_rows_kernel` 22.7 us avg; a CUDA fusion would remove up to 60 launches (~6 us of idle, derived)
and needs the same state-hazard solution. Metal: no GDN kernels, nothing.

**e. Check plan.** HIP bitwise: new test modeled on `tests/gdn_v2.rs` (35B layout, T = 1 .. 1000, PD 1 / 2 / 4, output rows and
**both** states compared byte for byte) against the launch chain; admission sweep (`gdn_delta_rule_bf16_v2` has 8 cases [R4 §9]);
KL21, ids, A/B as in P1; mutants: conv state not written, written before a sibling block reads it, wrong tap, the 9 `gdn2-*`
mutants re-aimed at the fused kernel. CUDA / Metal: section 9.

**f. Risk.** No grid-wide wait is needed for the outputs, but there is a **read-after-write hazard on the in-place conv state**:
several delta-rule blocks of one key head need the same q/k channels, so they all read the same `state[c]`, and the new state must
be written only after every one of them has read the old one. The conv kernel documents the same problem today: its multi-tile form
needs "a second kernel once every tile is done reading the old one", and only the single-tile decode form writes the state itself
(`kernels/hip/gdn.hip:24-29`). Options: a designated writer after a cross-block wait (inherits the `v3-nofence` class), a ping-pong
state buffer (new resident memory and a step-parity input that must be graph-replayable), or a state layout change. This needs a
design decision before any code; the 1.2 % [R5] / 0.6 % calibrated does not justify a wait-based solution.

**g. Effort (estimate).** Core 5 h + design of the state hazard 6 h + HIP 14 h + tests/mutants 8 h + checks 4 h = ~37 h.

---

## 7. P4 `GateSplitQkNormRopeKv` and `SigmoidMulMatmulFp8`; P5 `RmsNormAddMatmulFp8Pair`

These are row-local folds; no cross-block wait.

### P4a `GateSplitQkNormRopeKv` (-10)

**a.** `GateSplit` (gated q `[T,N,2D]` -> q, gate) into the fused attention prologue: q is read from the interleaved gated row, and
the gate half is copied out (it is read later by the `SigmoidMul`). A **new op**, not a new attr on `QkNormRopeKv`, so the existing op
(HIP and Metal kernels, 240 + 60 bitwise cases [R4 §4.3]) and its arity `(5, 3)` (`ops.rs:~194`) stay untouched.
**b.** -10 launches (one per attention layer); part of P4's ~60 us ([R5]) / ~28 us calibrated.
**c.** Common list with `gdn.rs` / `admission.rs`; **`plan.rs:277` `PER_ROW_OPS` and `ops.rs:~295` (the device-position check list)
must include the new op**, because it contains the per-row cache append: forgetting it makes `check_batch_ops` mis-count batched
steps. Call site `Model::qkv_fold` (`model.rs:~1445-1500`) + the `GateSplit` exec at `model.rs:~1509`; the plan tests in
`crates/intarsia-models/tests/graphs.rs:~290-296` and `batch_metal.rs:~68-72` change.
**d.** Fallback as section 2. CUDA does not register `QkNormRopeKv` at all (`crates/intarsia-backend-cuda/src/kernels.rs:411`), so
its attention prologue is still the 6-launch chain (rope 20 + kv_append 20 + rope_pass 20 + norms per step [steps.txt]); P4a is
meaningless there until `QkNormRopeKv` is ported (HIP: 578 -> 528 launches, -50 [R4 §1]). Metal registers `QkNormRopeKv` but not
`GateSplit` (dense, un-gated models only), so nothing.
**e./f.** Bitwise vs `gate_split` -> `qk_norm_rope_kv_bf16` chain (extend `tests/qk_rope_kv.rs`: gated layouts, positions 0 / 37 / 4090,
bf16 and fp8 caches, host and device position); re-run the 10 `qkr-*` mutants on the new kernel. Plus the common HIP gates of P1 e(2)-(6): full admission sweep, KL21 identical to
[R5 Accuracy], 256 greedy ids (`ids4k`), A/B x3 (launches/step 488 -> 478 on v3), rollback by `INTARSIA_HIP_DISABLE`. CUDA / Metal:
section 9. Risk: low-medium, because it edits
the most heavily mutation-tested prologue (10 `qkr-*` mutants, all killed [R4 §8]) and the batch accounting.
**g.** ~16 h core+HIP+tests, ~4 h checks.

### P4b `SigmoidMulMatmulFp8` (-10)

**a.** `SigmoidMul` (`attn_out * sigmoid(gate)`, bf16) into the input load of the o_proj fp8 GEMV; the product tensor (`ws.q[qc]`, only
read by o_proj) disappears from the fused path. Inputs `attn_out`, `gate`, fp8 weight + scales, output `out`. Bitwise requires the
sigmoid-mul expression and its bf16 rounding of `sigmoid_mul_kernel` (`kernels/hip/elementwise.hip`) to run on the load path; each o_proj
block (grid g128 in the trace) loads the whole input row, so the sigmoid-mul is recomputed per block (cost not measured; the
standalone kernel is 1.0 us avg).
**b.** -10 launches; with P4a -20 ([R5] ~60 us, calibrated ~28 us).
**c.** Common list with `quant/ops.rs` / `quant/cases.rs` (it carries an fp8 GEMV, like `gated_rmsnorm_matmul_fp8_case` `:239`);
call site `model.rs:~1642` (the `SigmoidMul` exec before `proj(&o_in, &a.wo, ...)`).
**d.** Fallback as section 2. CUDA has `sigmoid_mul_kernel` 10/step (0.8 us avg) and an fp8 GEMV; launch-count gain ~1 us of idle
(derived), kernel-time gain not measured. Metal: nothing (no fp8).
**e./f.** Bitwise vs `sigmoid_mul` -> `fp8_gemv`; mutants: sigmoid not rounded, gate of the wrong row, product rounded twice. Plus the common HIP gates of P1 e(2)-(6) (full admission
sweep, KL21 identical to [R5 Accuracy], `ids4k`, A/B x3, rollback). CUDA / Metal: section 9. Risk low.
**g.** ~12 h core+HIP+tests, ~3 h checks (shares the core PR's window).

### P5 `RmsNormAddMatmulFp8Pair` (-10)

**a.** The attention layers' input norm (`norm_in`, `model.rs:~1303-1316`: `RmsNormAdd` or `RmsNorm`, no fused op to try today [R5]) into
the k/v `MatmulFp8Pair` (`Model::kv_pair`, `model.rs:1425`), the fp8 twin of `RmsNormAddMatmulPair`. Inputs `blk`, `res`, norm weight, k and v fp8 weights
+ scales; outputs `n`, `sum`, k, v. It must store `n` (and `sum`) because the q projection still reads `n`, exactly like the existing
op ("also stores `n` and `sum` for the block's other readers", `quant/moe.rs:141`).
**b.** -10 launches; [R5] ~30 us (0.4 %); calibrated ~14 us (+0.2 %). The norm kernel is 2.1 us avg (10 of its 11 launches).
**c.** Common list with `quant/*`. Unlike the MoE and GDN blocks, the attention block does not take a `NormFold` today (`norm_in` is a
closure run before the block), so P5 needs that plumbing in `model.rs` too (the `NormFold` type is in `quant/moe.rs:141-190`).
**d.** Fallback as section 2 (norm runs unfused, as now). CUDA has both parts (`rmsnorm_add_pair_kernel`, `fp8_gemv_pair_kernel`);
launch-count gain ~1 us of idle (derived). Metal: nothing.
**e./f.** Bitwise vs the `RmsNormAdd` -> `MatmulFp8Pair` chain, including the stored `n` and `sum`; admission; mutants: `n` not stored,
`sum` from the wrong operand, `eps`, zero-centred weight offset. Plus the common HIP gates of P1 e(2)-(6): full admission sweep, KL21 identical to [R5 Accuracy],
`ids4k`, A/B x3 (launches/step 488 -> 478 on v3), rollback by `INTARSIA_HIP_DISABLE`; CUDA / Metal: section 9. Risk low (precedent `RmsNormAddMatmulPair`, `GatedRmsnormMatmulFp8`).
**g.** Core 6 h (with the `NormFold` plumbing) + HIP 6 h + tests 4 h + checks 3 h = ~19 h.

---

## 8. Cross-block synchronisation: what is at stake

* **Which proposals need a grid-wide wait.** P2 (glu -> down), P2' (two or three edges), P1 only in its "publish" variant. P2a, P3
  (outputs), P4a, P4b, P5 do not; P3 has the state hazard of section 6 instead.
* **Why it matters.** Round 5 shipped v3 opt-in because `v3-nofence` is unkilled [R5 Decisions, Mutations]: no test produces the
  stale read, so we cannot show the acquire is both necessary and correct, on hardware where the stale-read window is
  CU-occupancy-dependent. Since v3 itself is only +0.8 % (145.7 -> 146.9 tok/s) [R5 Headline], that risk was not worth it; P2' raises
  the benefit (calibrated +3.4 %) but also the number of edges.
* **Pre-requisite task (HIP only, no shared file).** Build the test [R5 Decisions] describes ("a stress run where the gate_up roles delay
  their h stores (skew per role, not only the last one), the h scratch is pre-filled with a sentinel (NaN / 0x7fc0 bf16) before
  every launch, and the down roles' inputs are checked for the sentinel ... over many launches and several CU-occupancy settings"),
  and show it kills `v3-nofence`. If it does, v3 can be reconsidered as default (+0.8 % already measured) and P2' has a test
  template. If it cannot be killed, P2' and P2 should not be built: the wait-free variants (P2a and the others) stay.
* **Deadlock.** v3's ticket rule makes the waits deadlock-free for any dispatch order [R5 What I built, 2]; for a longer chain the
  tickets must be issued in dependency order. This is an argument from the mechanism, not a test result.

---

## 9. CUDA and Metal: no-op checks (all proposals)

The core PR changes only core tables, the model call sites, the CPU oracle, and test tables in the CUDA and HIP crates. No kernel,
no `.cu`, no `.metal`, no CUDA/Metal source other than the one-line CUDA table-test exemption [N17].

**CUDA (spark1, GB10).** Following [N17] exactly:

* `git diff <base> <head> -- crates/intarsia-backend-cuda kernels/cuda` is only the table-test edit.
* 35B (Qwen3.6-35B-A3B-NVFP4) and 8B (Qwen3-8B): 256 greedy ids, graph and `--no-graphs`, byte-identical to the base build; 35B-short
  accuracy vs the vLLM golden (sha `b6b51e1b1699c66c`) the same numbers as base (97.99 % / 6.271e-3 prefill, 98.12 % / 1.556e-2 decode in
  [N17]); self-golden dumps with `scripts/cuda-decode/cmp_dump.py` 0 of 60 differ; 8B fp32 goldens PASS.
* **Admission sweep counts unchanged:** `intarsia kernels test --backend cuda --model <35B>` (or the plan admission at `serve` load)
  before == after on the same tree. Last recorded: 35 kernels / 501 cases base, 36 / 505 with the prefill-3 kernel [CP3 Admission];
  re-measure on current main first, because the counts move with merges. New sweep cases are appended only (the rule [CP3] followed)
  so existing cases' data and counts do not move.
* `plan show` for the 35B decode plan: same op table and 625 launches per step before and after (a cheap exact no-op proof; `plan show`
  prints `calls_per_decode_step`, `crates/intarsia-cli/src/cmd/plan.rs:124`).
* Decode A/B, 2 interleaved rounds, within noise ([N17]: -0.3 % / -0.2 %, no mechanism for a cost since replay does not run the model layer).
* `cargo test -p intarsia-backend-cuda` is cheap insurance; [N17] skipped it because the sources were unchanged, and here one test table changes.

**Metal (endor, M5 Max) and the Mac.**

* `cargo test --workspace`, `cargo check --workspace`, `cargo check -p intarsia-cli --features hip` on the Mac (the three checks [R5 Files]
  records: 96 test binaries ok at the last run, `raw/logs/mac-checks.log`), plus `--features metal` (proposed here, not recorded in [R5]).
  The CUDA feature build is not assumed to compile on the Mac; it is built on spark1 in the CUDA window below. **`cargo test --workspace` is where the CPU-backend plan-count tests
  and the CPU admission tests (section 2, 3.) fail first if a case or count was forgotten.**
* `cargo test -p intarsia-backend-metal --release` in an `--exclusive` endor slot (precedent: fleetq job #2167 in
  `docs/results/2026-10-16-metal-forward-hash.md`), and Qwen3-8B BF16 `serve --backend metal` / goldens: the 8B is dense, so none of the new ops is
  reachable; expect the op multiset of `plan show` unchanged and the forward hashes / goldens identical.
* Admission sweep counts on Metal: not recorded in the documents read for this page; record before / after on the same tree.
* Metal gets no gain from P1-P5 (section 1) and cannot until the MoE, NVFP4, FP8 and GDN ops are ported.

**Goldens identical** means: every golden digit equals the base's (the fused CPU path is the composition oracle, so there is no
numeric change by construction).

---

## 10. Fleet slots and downtime

* **HIP checks (tatooine, R9700, radiance stopped):** [R4 §13] used 9 windows, 43.3 min total, each under 20 min (cap 60); [R5 Radiance
  downtime] used 5 windows, 44.34 min (cap 45). Plan per proposal (estimate, 1 window = 7 to 12 min of GPU work as in [R5]):
  P5, P4a, P4b, P2a: 2 windows each if done separately, ~3 windows total if batched (tests + KL/ids, A/B x3, mutants); P1, P3: 2 to 3;
  P2': 4 to 5 (stress test and mutant reruns dominate). The slot is an `--exclusive` fleetq lease on tatooine that takes the radiance
  service down (confirm the role name with `fleetq status`; not recorded in [R4]/[R5]); CPU builds and mutant builds use a separate
  non-exclusive CPU slot, as [R5 Mutations] did ("built in a CPU slot").
* **CUDA no-op (spark1):** one window of about 15 min: [N17] recorded 14.8 min for the equivalent check (`window.sh 15`,
  `--exclusive` lease with `--takes-down guidance`, vLLM stopped, restored and waited for `/v1/models`), builds under
  `fleetq run --host spark1 --mem 16`.
* **Metal (endor):** one `--exclusive` slot for `cargo test -p intarsia-backend-metal --release` (precedent #2167; downtime not recorded)
  plus an optional serve identity run ([`2026-10-17-metal-relcheck-main.md`]: one exclusive slot, `--mem 24 --expect 150`).
* **Batching:** because the CUDA and Metal checks cost a window per *core PR*, group the sync-free ops into one core PR (section 11).

---

## 11. Recommended order

| step | what | why | gate |
|---|---|---|---|
| 0 | **Kill `v3-nofence`** (HIP-only: stress test of [R5 Decisions]); then decide v3 default-on | prerequisite for any wait-based fusion; v3 alone is already a measured +0.8 % | mutant killed on hardware; else stop at the wait-free proposals |
| 1 | **Core PR 1: P5 + P2a + P4b** (3 new ops, 60 launches), then the HIP kernels | all row-local, no wait, precedents exist (`RmsNormAddMatmulPair`, `GatedRmsnormMatmulFp8`); one CUDA window and one Metal run for three ops | bitwise, KL21 identical to [R5 Accuracy], A/B x3 shows the calibrated gain (~84 us, +1.2 %, derived) |
| 2 | **Core PR 2: P4a + P3** | both edit already-validated kernels and need a design decision (batch accounting; conv-state hazard) | design note on the conv state before code |
| 3 | **P2'** (subsumes P1 and P2b) | the largest estimate (6 % [R5], 3.4 % calibrated) but the longest, riskiest and the one that conflicts with v3's opt-in decision | step 0 passed; microbench of in-block top-k and shared-expert overlap says the critical path does not grow |
| - | **P1 alone: do not build** | calibrated +0.8 % at best, `moe_topk_v2` (3.9 us) is longer than the price of a launch, and P2' contains it | only if the microbench shows the in-role top-k is far cheaper than 3.9 us |

"P2' first" is rejected for now because it needs step 0, the biggest effort (~64 h estimate) and the most checks, for an estimate
that halves when calibrated. "P1 + P3 first as low risk" is also rejected: P1 has the latency problem above and P3 the state hazard;
the actually low-risk set is P5 + P2a + P4b.

Expected result if everything lands (derived): -220 launches vs the v3 build, ~308 us of a 6.81 ms step, about +4.7 % decode (~153.9
tok/s from 146.9 at 128 tokens; +4.0 % at 1.2 us/launch); [R5]'s gap-only figure for the same set is ~8.5 %. The first step alone
(-60 launches) is ~84 us, +1.2 %. These stack on v3; on the default build, v3's +0.8 % comes first.

**One line:** do P5 + P2a + P4b first as one sync-free core change (about +1.2 % decode, no cross-block risk) while killing
`v3-nofence` in parallel, and take P2' (about +3.4 % calibrated, 6 % by [R5]'s rule) only after that test exists; do not build P1 alone.

## 12. Not measured / open

* Why the profiled gain of v3 (+1.7 % predicted) exceeds the unprofiled one (+0.8 %) [R5]; an unprofiled launch-count sweep would
  settle the per-launch price (only v3's 40 launches and round 4's 50 are recorded).
* Cost of recomputing top-k inside a role; cost of recomputing the sigmoid-mul or conv per block; overlap gain of the shared expert.
* Whether the shared kernels' arithmetic (`nvfp4.hip`) is bit-identical to the routed experts' (`moe.hip`) code, which decides whether
  the "9th expert" can share code.
* Batched-decode behaviour of any new op on HIP ("not exercised end to end" [R4 §12]); TP > 1 (fusion is off there by `fused_ok`).
* Metal admission counts; CUDA decode launch count after 2026-10-17 (no later report restates it).
* All effort hours are the author's estimates.
