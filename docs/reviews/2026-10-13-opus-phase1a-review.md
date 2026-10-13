# Phase 1a adversarial code review (Opus 5.5, 2026-10-13)

Read-only review run as the last stage of the Phase 1a workflow. Verbatim.

I found 10 real problems. None of them is a numerical bug in the kernels: the weak spots are the accuracy harness, the reference it compares against, the test evidence and the claims in the results doc. I read the code and checked fleetq and the live vLLM metrics, but ran nothing on the GPU.

**Areas checked with no bug found:**
- NVFP4 decoding is right: the fp4-to-e4m3 lookup table, nibble order, block scale then `weight_scale_2`, and gate and up each using their own global scale.
- FP8 W8A8 static quantisation is right (divide by `input_scale`, round, saturate).
- MoE routing is right: top-k on bf16 router logits, softmax over the selected k (same as renormalising), weights rounded to bf16, `sigmoid(shared_expert_gate) * shared` added at HF's rounding points, and expert grouping is deterministic.
- GatedDeltaNet matches HF `Qwen3_5MoeGatedDeltaNet`: gating, l2norm eps 1e-6, q scale 1/sqrt(dk), causal conv and its carried state, value head h reading key head h/(Hv/Hk), and the gated norm without the 1+w offset.
- Gated attention (per-head [q|gate] split, q_norm after the split, sigmoid on the output), partial rotary (64 of 256 dims, inv_freq over 64) and rope_theta 1e7 are right.
- The split-KV merge and the flash-attention causal mask are right.
- A failed forward resets recurrent state, and snapshot restore is correct; there is also a CPU test against a tiny HF hybrid model.
- The web UI has no XSS: `md()` escapes `&<>` before adding tags and never builds attributes, and everything else uses `textContent`.

## Findings, most severe first

**1. [HIGH, harness] The top-20 KL ignores probability our model puts outside vLLM's top-20, and is not the lower bound its comment claims.**
- **Where:** `crates/intarsia-cli/src/cmd/accuracy.rs:284-287` (comment), `:358-367` (computation).
- **What's wrong:** Both sides are renormalised over the reference's top-20. By the chain rule, full KL ≥ P(S)·KL over the support, so the renormalised value can be up to 1/P(S) times the full KL. More importantly, it is blind to probability that moves outside the support.
- **Failure scenario:** a bug in the NVFP4 `lm_head` (say zeroed block scales) raises thousands of rare-token logits from about -10 to about 0. Most of the probability goes to tokens outside the top-20, but the renormalised KL stays near 0. Top-1 survives wherever the top logit is above 0, and top-1 is not gated anyway, so the model passes.
- **Fix:** vLLM reports full-vocabulary log-probs, so add a tail bucket. The reference tail is `1 - Σexp(lp_ref)`, our tail is `1 - Σsoftmax(ours)[S]`, and KL is taken over the 21-way partition, which is a true lower bound. Also report our probability mass on S, and gate top-1.

**2. [HIGH, harness] The 35B is gated with the loose 4-bit threshold, though both sides run the same quantised weights.**
- **Where:** `accuracy.rs:409-416`, `weight_class` at `:445-475`; results doc lines 113-115.
- **What's wrong:** The 3e-2 threshold in ARCHITECTURE §6 allows for quantisation error against a BF16 reference. Here both sides run the same NVFP4/FP8 checkpoint, so only implementation noise should remain. No noise floor was measured.
- **Failure scenario:** we measured 6.5e-3, already 6.5x the bf16 threshold. A real bug adding up to about 2e-2 KL passes, and top-1 is not gated. The `chat_system` prompt sits at 90.6% top-1, which is unexplained and currently invisible to the gate.
- **Fix:** add an "implementation" class for references built from the same checkpoint. Calibrate it from vLLM against itself (different `max-num-batched-tokens` or `enforce-eager`, KV dtype matched), and gate top-1 at 99% or more.

**3. [MED-HIGH, reference and claims] vLLM uses an FP8 KV cache; we use bf16, and the golden does not record this.**
- **Evidence:** the live `guidance` `/metrics` shows `vllm:cache_config_info{cache_dtype="fp8", enable_prefix_caching="True"}`. The checkpoint's `hf_quant_config.json` has `kv_cache_quant_algo: FP8`. Our `recipes/qwen3_5_moe.toml:62` sets `kv_dtype = "bf16"`.
- **What's wrong:**
  - The golden metadata in `tools/golden_vllm.py:98-110` records no vLLM version, flags or cache settings.
  - The reference is a shared, changeable fleet role.
  - The results doc (lines 129-130, 184-185) calls it "the same checkpoint" with no caveat.
  - It justifies the W8A8 default as "what vLLM runs", but did not apply the same reasoning to the KV cache.
- **Failure scenario:** part of the 6.5e-3 KL is reference noise, so model error and reference error cannot be separated. In the speed comparison vLLM also reads half the KV bytes.
- **Fix:** rebuild the golden with a vLLM started with `--kv-cache-dtype auto`, or support FP8 KV with the checkpoint's k/v scales. Store `cache_config_info`, the vLLM version and the launch flags in the golden metadata.

**4. [MED-HIGH, coverage] The golden never goes past about 700 tokens or 16 decode steps, and the real-model reuse path is never compared.**
- **Where:** `tools/prompts/golden*.jsonl` (longest is `long_prose_700`); `engine.rs:536` (`forward_logits_decode` resets the state).
- **What's wrong:** these are untested:
  - positions beyond about 700 under partial rope with theta 1e7;
  - GatedDeltaNet state carried over thousands of tokens;
  - many prefill chunks with recurrent snapshots;
  - restoring a snapshot on CUDA with the 40-layer model;
  - the MoE block at model level outside the real-checkpoint run.

  The server runs at 8192 tokens and the benchmark at 4096.
- **Fix:** add 4k and 16k prompts to the vLLM golden. Add a CUDA test that runs prompt A, then A+B through the engine with reuse, and checks bit-equality against a cold run of A+B. Add the tiny random MoE model test against HF that the doc already lists as missing.

**5. [MED, claims] The results doc grades against a gate that does not exist, and the verdict contradicts its own section 2.2.**
- **Where:** results doc lines 3-10.
- **What's wrong:**
  - It cites "ARCHITECTURE.md 13, Phase 1a", but §13 has no Phase 1a row.
  - The actual Phase 1 gate (§13, line 322) is "correct and ≥ vLLM single-stream (86-90 tok/s)". At 63-65 tok/s that gate fails.
  - Line 10 says "no regression except one gate margin", while section 2.2 says the decode KL is "reported as a failure, not a margin".
  - "No regression" cannot be shown for that row, because no baseline decode run exists.
- **Fix:** state "Phase 1 gate: FAIL (speed); Qwen3-8B decode KL: FAIL", and treat 1a as an informal milestone.

**6. [MED, evidence] The GPU tests were not run, and they report "pass" when skipped.**
- **Where:** `crates/intarsia-backend-cuda/tests/quantmoe.rs:4-5` (`#![cfg(ik_cuda)]`, "skips (passes) without one"); the same in `attention.rs`; `crates/intarsia-models/tests/hybrid_real.rs:91` skips without `INTARSIA_QWEN36_DIR`; results doc lines 41-43.
- **What's wrong:** spark1 ran only `cargo check --tests`. The Mac's "cargo test, 0 failed" therefore covers no CUDA code, and kernel correctness rests on the admission sweeps alone.
- **Fix:** run `cargo test --release --features cuda --workspace` on spark1 with `INTARSIA_QWEN36_DIR` set, inside the lease, and record the result. Report skips as skipped, or fail them when `CI=1`.

**7. [MED, fleetq and measurement] The benchmark lease was not exclusive, and the server-stop script releases the slot too early.**
- **Lease:**
  - `fleetq show 1374` reads "40 GB uses spark1-vllm", with no `--exclusive`.
  - The fleetq README pairs speed benches with `--exclusive`, which means "no untracked requests may be in flight".
  - vLLM's in-flight count was sampled only while vLLM was being measured (doc lines 136-138), not during our three runs (about 00:41-00:53).
  - Other users of `guidance` could have shared memory bandwidth with our 35B numbers.
- **Script:** in `deploy/localeval/serve-and-bench.sh:30-31`, `stop()` kills the PID it saved, which is the supervisor. It returns at once, but the supervisor gives the child up to 30 s (`supervisor.rs:66-79`, `:104`). fleetq then releases the slot while about 38 GB may still be held.
- **Fix:** take speed-bench leases with `--exclusive` and sample vLLM's gauges during both runs. In `stop()`, wait for the PID to exit, then send `kill -9`, before returning.

**8. [LOW-MED, loading] A missing `lm_head` still silently falls back to the embedding table.**
- **Where:** `crates/intarsia-core/src/model.rs:257-260`. This was raised in the Phase 0 review (#4) and is still unfixed.
- **What's wrong:** with `tie_word_embeddings=false` and the `lm_head` tensor renamed or missing, the `(_, Some(_))` branch uses the embeddings as the head.
- **Failure scenario:** a re-exported or compressed-tensors checkpoint produces garbage logits from `serve`, which runs no accuracy check.
- **Fix:** return an error when `!tie_embeddings` and the head tensor is absent.

**9. [LOW-MED, quant format] The per-layer quantisation scheme is never read from `hf_quant_config.json`.**
- **Where:** `crates/intarsia-core/src/quant/nvfp4.rs:8-9`. `input_scale` is ignored unconditionally, and the format is detected from tensor names alone.
- **Failure scenario:** a ModelOpt checkpoint quantised as NVFP4 W4A4 (`quant_algo: "NVFP4"`) loads and silently runs as W4A16, which differs from its declared scheme and from vLLM. `kv_cache_quant_algo` is ignored the same way.
- **Fix:** parse `quantized_layers` and `kv_cache_quant_algo`. Reject unsupported schemes, or log any deviation in the plan.

**10. [LOW, claims] The decode byte floor leaves out the recurrent state and the KV cache.**
- **Where:** results doc lines 163-166.
- **What's wrong:**
  - Each decode step reads and writes the f32 GatedDeltaNet state: 30 layers × 2 × 2.1 MB ≈ 126 MB per token.
  - At 4k context the bf16 KV reads add about 84 MB more.
  - A routed expert is 1.77 MB with its scales, not 1.67 MB.
- **Effect:** the floor is about 2.35 GB, or 10.6 ms, rather than 2.2 GB. The percentages against the floor shift slightly; the conclusions stand.

## Three most valuable next steps toward beating vLLM on spark1

1. **A tensor-core grouped MoE GEMM for prefill.** Prefill is 5-7x behind and `moe_experts` takes 63% of it. Process all of an expert's tokens in one pass, instead of the current chunks of 8 pairs that re-read the weights. Dequantise each weight tile once to bf16 and use `mma` (or sm_121's block-scaled fp4 MMA), and apply the same dequantise-once change to the FP8 and NVFP4 `*_mma` kernels.

2. **CUDA graphs plus fused MoE and GatedDeltaNet blocks for decode.** Decode now makes 844 launches per token in 15.5 ms; the floor is about 10.6 ms and vLLM takes about 13 ms.
   - Make `pos0` and `kv_len` live in device memory so one graph can be replayed every step.
   - Fuse router, top-k, gate_up, down, reduce and combine into one or two kernels.
   - Fuse conv, delta rule and gated norm per GatedDeltaNet layer.
   - Target about 150 launches.

3. **MTP speculative decoding from the checkpoint's own `mtp.*` head (kept in bf16).** Add a verify path that respects the recurrent state, reusing the existing snapshot code with one snapshot per draft position. vLLM's 86-90 tok/s figure relies on speculation, so beating it single-stream needs this. Do it only after findings 1-4 are fixed (tail-bucket KL, calibrated gate, long golden, matched KV dtype), so the speed work cannot quietly cost accuracy.

The files that matter most:
- `/opt/code/intarsiaml/crates/intarsia-cli/src/cmd/accuracy.rs`
- `/opt/code/intarsiaml/tools/golden_vllm.py`
- `/opt/code/intarsiaml/recipes/qwen3_5_moe.toml`
- `/opt/code/intarsiaml/docs/results/2026-10-12-phase1a.md`
- `/opt/code/intarsiaml/deploy/localeval/serve-and-bench.sh`
- `/opt/code/intarsiaml/crates/intarsia-core/src/model.rs`
- `/opt/code/intarsiaml/crates/intarsia-core/src/quant/nvfp4.rs`
- `/opt/code/intarsiaml/crates/intarsia-backend-cuda/tests/quantmoe.rs`
