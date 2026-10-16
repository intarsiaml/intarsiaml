# Phase 0 adversarial code review (Opus 5.5, 2026-10-12)

Read-only review run as the last stage of the Phase 0 workflow. Verbatim.

Adversarial review: IntarsiaML Phase 0 (working tree on arch/plan), read-only

I found no bugs in the kernels themselves. Ten real problems sit in the accuracy harness, the gates, the server and the fleet process, ranked below with three Phase 1 next steps after them.

These parts look suspicious but are correct, so they are not listed:
- **Kernels:** attention (causal `visible = pos0+t+1`, GQA mapping, online softmax, block reductions with syncs), rope (rotate-half in fp64, safe in place), kv_append (bf16/fp8), rmsnorm_add (the residual is rounded to bf16 before the norm), the decode matrix-vector kernel (GEMV) and argmax.
- **Model wiring:** qk_norm runs before rope, and kv_append runs before attention.
- **Prefix reuse:** the last sampled token is never committed, so `st.tokens` always matches what is in the KV cache.
- **OpenAI streams:** the `include_usage` framing, `[DONE]`, and `cached_tokens` on streams are all right.
- **No silent fallback to the reference kernels on the GPU:** `Registry::new` filters by backend, and the cuda backend registers no reference kernels.
- **The generic decoder (`model.rs`) has no Qwen-specific code.**

## Findings, most severe first

**1. [HIGH, harness] The golden-logits gate never runs the decode path.**
- **Where:** `tools/golden.py:24`, `crates/intarsia-cli/src/common.rs:30`, `crates/intarsia-cli/src/cmd/accuracy.rs:264`, `docs/results/2026-10-12-phase0.md:150`.
- **What is wrong:**
  - Golden prompts are capped at 8..256 tokens, and the default prefill chunk is 256 tokens.
  - So every golden position is computed in one prefill chunk starting at position 0, with every layer matmul on cuBLAS (8 or more rows).
  - None of this gets an HF comparison: `matmul_bf16_gemv` (1 to 4 rows, the hottest decode kernel), `attn_decode_bf16`, rope / kv_append / prefill attention at positions above 0, and multi-chunk prefill.
  - The admission tests use 0.6B shapes only, so the 8B GEMV shapes (K=4096 and K=12288) are verified by nothing quantitative.
  - The results doc says (line 150) that the K=12288 down projection is "covered only by the end-to-end accuracy run". That is false: that run never calls the GEMV.
- **Failure scenario:** a GEMV bug at K=12288 (wrong tail handling, a precision loss) passes gate (b). Yet every served token and every speed number comes from that kernel. Only the coarse perplexity check and the sanity prompts would notice.
- **Fix:**
  - Add a teacher-forced decode mode to `intarsia accuracy`: prefill n−k tokens, then feed the golden ids one at a time with `HeadMode::LastLogits`, comparing each row against the golden.
  - Also run the gate with `--prefill-chunk 64`, and add prompts longer than 256 tokens.
  - Add the recipe's real shapes to the admission sweep.

**2. [HIGH, harness] NaN logits are scored as a perfect match.**
- **Where:** `accuracy.rs:201` (and `lse`, `top_k` at `:140`).
- **What is wrong:**
  - `lse` folds with `f64::max`, which ignores NaN, so any NaN in our row makes the log-normaliser `ol` NaN, and then the KL value is NaN.
  - `kl.max(0.0)` turns NaN into 0.0.
  - `top_k` skips NaN entries, so top-1 can still "agree".
  - An all-NaN row empties `ot`, and `ot[0]` panics instead of failing cleanly.
  - On the device, argmax returns token 0 when every value is NaN, so the engine never reports an error either.
- **Failure scenario:** a lm_head tile that writes NaN into a few vocab entries scores KL = 0 and top-1 agreement on every affected position. The gate passes.
- **Fix:**
  - Any non-finite value in our row should make that position fail, with KL counted as infinity.
  - Assert finite logits in the engine when running in debug or accuracy mode.
  - Add a NaN case to the unit tests.

**3. [HIGH, gates and claims] The gates are looser than ARCHITECTURE §6, and verdict (a) is overstated.**
- **Where:** `accuracy.rs:30`, `:242`, `:204`, `:375`; `crates/intarsia-backend-cpu/src/lib.rs:115,121`; results doc lines 4-7, 40-48.
- **What is wrong:**
  - §6 sets BF16 vs HF at top-1 ≥ 99.5% and mean KL ≤ 1e-3. The code defaults to top-1 99.0%, and to KL 5e-3 on any non-cpu backend, which is the FP8 threshold. Thresholds are keyed by backend, not by quant.
  - The top-1 rule at `:204` counts fp16 ties as agreement.
  - "Greedy-token match length" is not computed.
  - The quick gate has 3 prompts (§6 says 5), never checks that the model "answers a known fact", and uses a fixed perplexity limit of 40 instead of a per-family range.
  - `--backend cpu` does not run the reference ("cpu-ref") kernels: `cpu_f32_*` (fast.rs, priority 0) outranks `ref_*` (priority -100). So the "cpu-ref matches HF" runs actually exercised fast.rs.
  - Verdict (a) therefore fails §6 three ways: 99.08% is below 99.5%, the bf16 golden fails the 1e-3 KL limit, and the run was not on cpu-ref.
  - The vLLM smoke comparison, listed in the gate row, was not run.
- **Failure scenario:** a BF16 CUDA build with mean KL 4e-3 (four times the plan's limit) reports PASS.
- **Fix:**
  - Key the defaults on weight and KV dtype (BF16: 99.5% / 1e-3), and let a recipe override them.
  - Add a `--reference` flag that forces the `ref_*` kernels.
  - Reword verdict (a) as "not met under §6", or amend §6 deliberately.

**4. [MED-HIGH, model loading] Config and weights can be misloaded silently.**
- **Where:** `crates/intarsia-core/src/recipe.rs:209-231` (line 228); `crates/intarsia-core/src/model.rs:186`.
- **What is wrong:**
  - `rope_theta`, `rms_norm_eps` and `max_position_embeddings` fall back silently to 1e4, 1e-6 and 4096.
  - The `tools` venv has transformers 5.19.0, whose `Qwen3Config` uses `rope_parameters`. A checkpoint saved by it can lack a top-level `rope_theta`, and the loader then rotates with theta 1e4.
  - `rope_scaling` (YaRN, which Qwen documents for long context), `attention_bias`, `use_sliding_window` and `hidden_act` are never read.
  - At `model.rs:186`, `lm_head` is replaced by the embedding table whenever `tied_to` is set, even when `tie_word_embeddings=false`. A missing or renamed `lm_head` therefore loads garbage instead of failing.
  - Checkpoint tensors the recipe does not map (biases, for example) are not reported.
- **Failure scenario:** a YaRN-patched Qwen3 `config.json` loads cleanly but produces wrong logits at every position.
- **Fix:**
  - Fail when `rope_scaling` or `rope_parameters` holds a type that is not implemented, or when `attention_bias` is true.
  - Read `rope_parameters.rope_theta`.
  - Error out instead of tying when `tie_embeddings` is false.
  - List unconsumed tensors at load.

**5. [MED, design drift] Admission and shape predicates are not enforced when the model loads.**
- **Where:** `crates/intarsia-core/src/registry.rs:200`; `model.rs:305,405`.
- **What is wrong:**
  - The serve registry starts with every kernel Untested and `require_admitted=false`, and nothing ever calls `apply()` on it.
  - Kernels are selected per call inside `exec`; there is no plan step that checks them up front.
- **Failure scenario:** a kernel that fails admission on this host is still selected. `--kv-dtype fp8_e4m3` passes load and then hits NoKernel at the first attention call.
- **Fix:**
  - Resolve every (op, shape) of the decode step and the prefill chunk at load, so a missing kernel fails there.
  - Run admission on the recipe's real shapes (cached per host fingerprint), then set `require_admitted=true` for `serve`.

**6. [MED, fleetq] The localeval run took an exclusive slot on the wrong host, and the documented recipe outlives its slot.**
- **Where:** results doc line 130; `docs/benchmarking.md:66-72`; `crates/intarsia-core/src/memplan.rs:39`.
- **What is wrong:**
  - `fleetq status` shows job #1313 ("localeval speed adhoc intarsia-phase0") as "done on endor exclusive". The run went through an ssh tunnel to `127.0.0.1:18090`, so localeval ignored `FLEETQ_JOB=1312` and locked endor exclusively while it actually loaded spark1.
  - The documented flow starts `intarsia serve` on spark1 "in another shell, or nohup", then runs `fleetq release $JOB` with no step that stops the server, so about 17 GB stays in use after the slot is gone.
  - The bench line at `:71` has no `FLEETQ_JOB=` prefix.
  - The KV budget is `kv_fraction` × the box's free memory, not the leased `--mem`. A large `--max-seq` can take far more than the lease.
- **Fix:**
  - Point localeval at `http://spark-4e77:8090` (or pass the host) so the slot matches the box under load.
  - In the doc, stop the server before `fleetq release`, and use `fleetq run ... -- ssh spark1 'intarsia serve ...'` instead.
  - Add `--mem-budget` and clamp the KV cache to it.

**7. [MED, concurrency] A dead worker or a vanished client is not handled.**
- **Where:** `crates/intarsia-core/src/engine.rs:363-364`; `crates/intarsia-server/src/lib.rs:110-112`; `crates/intarsia-server/src/api.rs:140-144`.
- **What is wrong:**
  - A panic in `generate_one` kills the only worker thread for good. After that, `submit` returns 503 forever.
  - `/health` always answers HTTP 200. It reports "stalled" only if `last_step_ms > 0`, so a worker that dies on its first request reports "ok" indefinitely.
  - Cancellation is noticed only when sending the first token fails. A queued request whose client has already left still runs its full prefill (about 8 s at 4k tokens).
  - The FIFO queue is unbounded, and each request gets its own OS bridge thread.
  - Default `max_tokens` is unbounded, so one request can hold the single worker for about 10 minutes.
- **Fix:**
  - Catch panics in the worker and turn them into a failed engine state; return 503 from `/health` when stalled or dead.
  - Add a per-job cancel flag that is checked between prefill chunks.
  - Bound the queue (429 when full), and bring the supervisor/watchdog from §4 forward.

**8. [MED-LOW, Anthropic API] Usage counts cached tokens twice, and `stop_reason` is wrong for stop sequences.**
- **Where:** `crates/intarsia-server/src/anthropic.rs:161,172,199`; `:102`.
- **What is wrong:**
  - `input_tokens` is the full prompt while `cache_read_input_tokens` is also reported. In Anthropic's convention, `input_tokens` excludes cache reads (total = input + cache_creation + cache_read), so clients double-count whenever a prefix is reused.
  - A matched stop sequence reports `end_turn` with `stop_sequence: null` (disclosed in the results doc, but still a contract violation).
- **Fix:**
  - Report `input_tokens = prompt - cached`.
  - Carry the matched stop string out of `FinishReason`.

**9. [LOW-MED, bench and claims] The speed benchmark measures the host-sampling path, and `min_tokens` was never tested end to end.**
- **Where:** `crates/intarsia-bench/src/speed.rs:157-164`; `engine.rs:522,531,574`; `crates/intarsia-core/src/sampler.rs:51`; results doc line 137.
- **What is wrong:**
  - The bench sends `ignore_eos` together with `min_tokens = max_tokens`. That makes `need_logits` true for every token, so each step downloads about 600 KB of full-vocabulary logits and runs the CPU argmax, not the on-device argmax that normal greedy requests use.
  - With `ignore_eos` set, `min_tokens` does nothing, so "ignore_eos and min_tokens work end to end" is unsupported for `min_tokens`.
  - Stop strings can end generation before `min_tokens` (vLLM skips stop-string checks below `min_tokens`).
  - Sampling with temperature above 0 sorts all ~152k logits on the host for every token.
- **Fix:**
  - Use logits only when masking actually applies (`min_tokens` without `ignore_eos`).
  - Gate stop strings on `min_tokens`.
  - Use partial selection for top-p.
  - Add an end-to-end `min_tokens`-only test.

**10. [LOW, telemetry] The per-op profiler misattributes time to `embedding`.**
- **Where:** `model.rs:406-409`; results doc line 159.
- **What is wrong:**
  - Prefill chunks that are not the last one end without a sync, and pageable `cudaMemcpyAsync` H2D does not synchronise the device.
  - When the next chunk is profiled, its first op (Embedding) starts the timer and then waits for all the previous chunk's queued work.
  - That explains the "54 ms embedding, 34%" figure, so the doc's "Unverified" can be marked resolved.
- **Fix:** synchronise before `t0` (or at chunk start) when profiling.

## Phase 1: the three most valuable next steps

1. **Close the accuracy holes before porting any NVFP4 GEMV, GDN or verify kernel.** Those are exactly the decode-path kernels the current gate cannot see. That means teacher-forced decode goldens, the NaN-safe comparison, dtype-keyed §6 thresholds, about 64 prompts including some over 256 tokens and multi-chunk runs, and admission on the recipe's real shapes enforced at serve load (`require_admitted`).
2. **Add a real plan compiler step and a workspace argument to the kernel ABI.** Every kernel would be resolved when the model loads (`intarsia plan show`), unsupported config fields would be rejected, and scratch buffers would become possible. That unblocks split-KV decode attention (25% lost at 4k context), the TensorFold ports and CUDA graphs.
3. **Make the server trustworthy for localeval and fleetq under load.** That means the supervisor/watchdog, cancel tokens checked between prefill chunks, a bounded queue, vLLM-compatible in-flight gauges for localeval's idle check, `--mem-budget` tied to the lease, and benchmarks addressed to the real host. Only then is the paged-KV and continuous-batching work measurable.

Main files reviewed:
- `/opt/code/intarsiaml/crates/intarsia-cli/src/cmd/accuracy.rs`
- `/opt/code/intarsiaml/tools/golden.py`
- `/opt/code/intarsiaml/crates/intarsia-core/src/{model.rs,engine.rs,registry.rs,recipe.rs,memplan.rs,sampler.rs,admission.rs}`
- `/opt/code/intarsiaml/crates/intarsia-server/src/{api.rs,openai.rs,anthropic.rs,lib.rs}`
- `/opt/code/intarsiaml/kernels/cuda/{attention.cu,elementwise.cu,norm.cu,matmul.cu,argmax.cu}`
- `/opt/code/intarsiaml/docs/benchmarking.md`
- `/opt/code/intarsiaml/docs/results/2026-10-12-phase0.md`
