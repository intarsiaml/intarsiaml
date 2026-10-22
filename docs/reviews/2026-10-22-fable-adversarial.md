# Independent adversarial review of intarsiaml, 2026-10-22

> **Independent review by Fable 5.1, run 2026-10-21 23:18 EDT.** Committed verbatim below the Response table; the
> original text is unchanged. The Response table maps each ranked finding to the lane that addresses it and is the only
> part added after the run.

## Response

Updated 2026-10-22. "In progress" means a lane has been assigned and is working; it does not mean the finding is closed.

| # | sev | finding (short) | addressed by | status |
|---|---|---|---|---|
| 1 | critical | 35B accuracy has no independent reference; gates cannot see a 1 % error | fx1 | in progress |
| 2 | high | Gates redefined after failing (35b-long mask, Metal decode gate, HIP tp=2 waiver, self-vs-ref-noise) | fx2 | in progress |
| 3 | high | "Top 3" is single-stream vs at most two comparators; CB Metal-only; tp=2 vs vLLM is 2 GPUs vs 1 | fx3 | in progress |
| 4 | high | New-model path refuses most of the catalogue at recipe load; no graph IR | fx4 | in progress |
| 5 | high | Hybrid prefix reuse has no CI test; fleet gate skips the failing case at default chunk | fx5 | in progress |
| 6 | medium | R9700 and Metal comparators handicapped or not clock-matched | fx6 | in progress |
| 7 | medium | ARCHITECTURE.md describes components that do not exist | fx7 (status markers in docs/ARCHITECTURE.md, diagram legend) | in progress |
| 8 | medium | CUDA and HIP baselines/envelopes not committed | folded into fx1 | in progress |
| 9 | medium | Quant format enum across 21 files; 20+ accuracy gate flags | none yet | deferred (user decision) |
| 10 | medium | PR #70 MoE multi-tile path and four ops untested; no reuse-mode run on Metal 35B | PR #70 review fixes, fx10 (tests, reuse run) | PR #92 |
| 11 | low | CI excludes GPU crates, skips tests; `INTARSIA_CUDA_FAST_PREFILL` semantics flipped | fx12 | in progress |
| 12 | low | tp_remote `Truncate`/`Snapshot` errors deferred; `truncate` not atomic across ranks | fx12 | in progress |

Finding 9 is a design question (how far to refactor the quant format enum and the accuracy flags), so it waits for the user.

---

## Review text

Reviewer: Claude Fable 5.1, independent of the build sessions. Base: `origin/main` f9087322 (PR #69), open PR #70
(`lane-m35/metal-moe`). Everything below was read from `origin/main` (`git show origin/main:<path>`), the PR #70
diff, and the committed docs. No fleet machine was touched. "Verified" means I read the code or data myself;
"suspected" means the evidence is a doc statement I could not cross-check against code or raw data.

Requirements referenced by number: 1 modular, 2 easy to use, 3 accurate, 4 fast (top 3 per platform),
5 reproducible, 6 APIs, 7 UI, 8 metrics, 9 diagrams, 10 Spark, 11 R9700, 12 tp=X, 13 Intel Arc.

---

## Summary verdict

The engine is further along than most one-week-old inference engines, and the documentation discipline (raw logs,
downtime logs, mutants, no-op checks) is unusually good. The headline problems are not kernel bugs; they are:

1. The flagship model (Qwen3.6-35B-A3B NVFP4) has **no independent accuracy reference**. Its only golden is vLLM,
   which the team itself measured as non-deterministic at temperature 0 and 1.6-1.8x noisier than the gate can
   resolve; all 35B accuracy gates on main are therefore either a loose static floor (3e-2, 2.1x above where the
   engine sits) or advisory. The team's own data shows a 1 % partial-scale bug on half the model passes every
   shipped 35B budget. Requirement 3 is not demonstrated for the model requirements 4, 10, 11 and 12 are measured on.
2. **Gates were redefined after they failed** three times this week (35b-long row mask, Metal decode floor and
   clock normalisation, HIP tp=2 waiver of the self-vs-ref-noise check). Each redefinition has a defensible
   rationale, but each was decided on the same day as the failure, by the same people, and each turned a FAIL
   into a PASS without any new engine evidence.
3. **"Top 3" is vacuous as measured.** Every platform has at most two live comparators, all comparisons are
   single-stream, the serving path for the 35B on CUDA and HIP is single-sequence (continuous batching is Metal-only
   and excludes MoE), and the comparator on R9700 is a handicapped llama.cpp build. The CUDA tp=2 headline
   compares two GPUs against vLLM on one.
4. **Architecture docs describe a system that does not exist yet** (paged KV, radix prefix cache, tuning DB,
   autotuner, kernel packs, Vulkan/SYCL "behind the same trait"). What exists is a clean op table and registry
   with a hard-coded Qwen-shaped decoder; a new model outside that shape (Llama 3.x rope scaling, Qwen2.5 attention
   bias, Gemma/Mistral sliding window, DeepSeek MLA) is refused at recipe load. Requirement 1 holds for Qwen3-family
   models only.

---

## A. Architecture (requirements 1, 2, 5, 13)

### A.1 What is actually there (verified)

* **Op table**: `crates/intarsia-core/src/ops.rs:66` `enum Op`, 38 ops (`Op::ALL` at :114). Adding an op touches
  six places in `ops.rs` (enum, ALL, as_str, arity, check, ShapeInfo::from_call) plus a CPU reference, plus one
  launcher per backend. No code generation; the order of `ALL` is pinned by a test (:897).
* **Backend trait**: `crates/intarsia-core/src/backend.rs:165`, 23 methods, 9 required (`name, info, alloc,
  upload, download, copy, kernels, launch, synchronize`), 14 defaulted. `launch(kernel_name, &OpCall, stream)` is a
  string-keyed dispatch, not a method per op. Graph methods default to `Err(Unsupported)` (:289-295). This is the
  right shape for a new backend; it matches ARCHITECTURE 3.1 in spirit.
* **Registry**: `registry.rs:63` `KernelImpl` with op/backend/arch/dtypes/shape predicate/exactness/priority;
  `Registry::select` (:302) fails loudly with `NoKernel` (:303-317), surfaced at plan time (`plan.rs:270, :630`). No
  silent dequant-to-bf16 fallback on any GPU backend. Good.
* **Quant formats**: `quant/mod.rs:213` `trait QuantFormat` (5 methods), hard-coded `formats()` array of 5 at
  :262-268. Adding a format touches about 11 sites (`docs/adding-a-model.md:330-341`), of which `ops.rs` alone is
  6, and the `Nvfp4|Mxfp4|Exl3|Fp8` identifiers appear in 21 files outside the backend crates (ops.rs 75 hits,
  quant/cases.rs 62, quant/scheme.rs 29, quant/ops.rs 28, cmd/accuracy.rs 28). The architecture doc's promise
  "implement the trait + register matmul kernels that accept it" is not what the code requires.
* **Model architectures**: there is no graph IR. `model.rs:89-125` `enum Block { Attn, Gdn, Ffn, Moe }`, loader
  branches on block kind at :463 (attn, only `gqa`), :507 (gdn), :560 (moe), :577 (ffn, only dense/silu_glu).
  `recipe.rs:445-475` hard-rejects `hidden_act != silu`, `attention_bias`, `use_sliding_window`,
  `layer_types` containing `sliding_attention`, and any `rope_type` other than default (yarn, linear, dynamic,
  llama3, longrope). So: Llama-2 shape works with a recipe; Llama-3.1+, Qwen2.5, Gemma, Mistral, GLM, DeepSeek,
  Phi do not without core code.
* **Continuous batching is Metal-only and excludes MoE**: `crates/intarsia-models/src/load.rs:273-282` returns
  `Unsupported("--max-batch N needs the Metal backend ...")` for cuda/hip/cpu; ADR-0010 states "MoE models out of
  scope" and "one KV buffer slot, no arena, no paging". There is no paged KV and no radix prefix cache
  (`state.rs` has `AttnKvState`, `RecurrentState`, `SequenceState`; no block table). ARCHITECTURE §4 "KV: paged
  blocks; radix-tree prefix cache" describes a plan, not the code.
* **No tuning DB, no autotuner, no kernel packs**: there is no `tuning/` directory on main; `kernels/` holds native
  CUDA/HIP/Metal sources (licences for radiance/TensorFold/exllamav3 are present in `LICENSES/`, so code was ported,
  not vendored as packs). Kernel choice is by static priority plus admission, which is fine, but ARCHITECTURE
  3.2 "tuning DB keyed by host fingerprint, committed lockfile" is unimplemented and is also the mechanism the doc
  relies on for requirement 5.
* **Backend enumeration leaks**: about 10 sites in the CLI (`common.rs:167-215` open_backend_with, :219-235,
  :340, :376, :424; `qualify.rs:365` whitelist, :1904 attn-simt only for cuda|hip, :624; `doctor.rs:146-151`;
  `onboard.rs:660`; `accuracy/envelope.rs:247-257`). `intarsia-cli/src/lib.rs:6-7` has a `compile_error!` because
  cuda and hip export the same `ik_*` C symbols. A fourth GPU backend using the C ABI would hit the same clash.
* **TP**: one `trait Comm` (`comm.rs:72`) with `all_reduce_sum`, `all_reduce_sum_into`, `all_gather`; a
  backend-neutral `HostCommGroup` (:197), NCCL for CUDA, P2P for HIP. `capture_safe` is NCCL-only, so graphs under
  TP are CUDA-only. `tp_remote.rs` is a TCP control channel; the CLI forces `--backend cuda` for `--tp-remote`
  (`common.rs:376, :424`). Clean abstraction; the "tp=X" requirement is really tp=2 (`load.rs` rejects world != 2
  for remote; HIP P2P is 2-card).
* **onboard / qualify**: `onboard.rs` (1353 lines) drives `qualify.rs` (2261 lines). Quick tier measured at 3m07s
  to 3m14s for Qwen3-8B bf16 on an M5 Max with a cached golden; in-depth 6m25s total
  (`docs/results/2026-10-21-o1-onboard.md`). That meets the "1-2 h / 2-4 h" bar for a model the recipe already
  covers. For anything MoE, hybrid or quantized `qualify.rs:1326-1331` refuses to make a golden and demands a vLLM
  server URL or a golden file; the 4k golden is never auto-made (:1275). So the quick tier for the model class that
  matters (requirements 10-12) depends on a competitor's engine being installed and serving the same checkpoint.

### A.2 Weakest abstractions (ranked)

1. **Model = hard-coded Rust + recipe-time rejection list.** A new architecture is a core change in `model.rs`,
   `ops.rs`, cpu reference and possibly `state.rs` (`docs/adding-a-model.md:346-348`). The rejection list in
   `recipe.rs:445-475` is honest, but it means requirement 1's "basic test within 1-2 h" is false for most of the
   current open-weight catalogue. Rope scaling in particular (yarn/llama3) is a two-function change that blocks
   every long-context Llama/Qwen2.5 variant.
2. **Quant format is an enum smeared across 21 files**, with `ops.rs` carrying one `Op` per (format, fusion)
   combination (`MatmulFp8`, `MatmulNvfp4`, `MatmulMxfp4`, `MatmulExl3`, `MatmulFp8Pair`, `MatmulNvfp4Glu`, ...).
   Each new format multiplies the fused-op surface. The admission oracle (CPU `dequant_ref` + reference matmul)
   is the one part that scales.
3. **Accuracy harness is the largest single piece of logic** (`cmd/accuracy.rs` ~3400 lines plus `refnoise.rs`,
   `selfgolden.rs`, `envelope.rs`, `turns.rs`, `baseline.rs`; `docs/accuracy.md` is 803 lines). It has at least 20
   gate-shaping flags (`--gate-class`, `--min-top1`, `--max-mean-kl`, `--max-tail-excess`, `--ref-noise[-factor]
   [-prompt-factor]`, `--no-ref-noise`, `--baseline` + 7 `--margin-*`, `--noise-envelope` + `--envelope-k`,
   `--self-golden` + 5 `--self-max-*` + `--self-prompt-factor`, `--self-vs-ref-noise`, `--reuse-*` x6, `--pre-change`,
   `--allow-*` x4). This is the opposite of requirement 2. More importantly, every flag is a way to make a gate pass.
4. **Backend-specific behaviour in shared code by name string** (`batch.rs:56 name() != "cpu"`, `load.rs:273-282`
   and :329 Metal checks, `load.rs` spec Metal-only). The `Backend` trait already has capability methods
   (`fusion_under_tp`, `token_feedback`, `avoid_single_row_prefill`); batching and speculation should be the same.
5. **C ABI is CUDA-stream-shaped and symbol-colliding** (`kernels/abi/ik_abi.h`: `void* stream` documented as
   cudaStream_t/hipStream_t; identical `ik_*` names in cuda and hip). Metal avoided it with a Rust-side launcher.
   For Intel Arc the honest path is Metal's (pure Rust FFI to a SYCL/Level-Zero shim with its own symbol prefix),
   which means the "same trait" claim is true but the "same ABI" claim is not.

### A.3 Intel Arc (requirement 13) specifically

* No FP8/FP4 numerics exist outside CUDA/HIP (Metal gets them only in PR #70, software-decoded). An Arc backend would
  run bf16 until it has software-dequant GEMMs; the plan would fail loudly (`NoKernel`), which is correct.
* Warp/wave assumptions live only in kernels (HIP `ik_hip.h:20` `#error` on wave64; ~18 CUDA and ~20 HIP files use
  warp-size shuffles; mma/wmma/mfma in ~11/12 files). Nothing ports to XMX/`joint_matrix` without a rewrite, which is
  expected; the registry/admission design makes that incremental.
* TP on Arc would work through `HostCommGroup` (host-bounced, not capture-safe). Fine for a first cut.
* The CLI whitelists (`qualify.rs:365`, `common.rs:167-215`, `doctor.rs:146-151`, `onboard.rs:660`) need edits; the
  `attn-simt` perturbation (`qualify.rs:1904`) would silently not apply, so the envelope gate on Arc would be
  weaker than on CUDA/HIP without anyone noticing.

### A.4 Does the design deliver requirements 1-5?

| req | verdict | one line |
|---|---|---|
| 1 modular | partial | op/registry/backend seams are good; model and quant seams are not; CB/spec are backend-string-gated |
| 2 easy to use | partial | `onboard` is good; the accuracy gate surface (20+ flags, 4 verdict kinds) is not |
| 3 accurate | not demonstrated for the 35B | see D |
| 4 fast | measured single-stream only, n=2 comparators | see B |
| 5 reproducible | partial | `--locked` + toolchain pin + stamps are good; CUDA binary not reproducible (B1), no tuning DB, baselines for CUDA/HIP are not committed (`golden/baselines/` holds 6 Metal 8B files only) |

---

## B. Claims vs evidence

Scope: `docs/results/2026-10-21-standings.md`, `-cuda-round7.md`, `-35b-tp2-cuda.md`, `-35b-tp2-hip.md`,
`-tp2-cuda-graphs.md`, `-g3-metal.md`, `-mg-metal-gate-clock.md`, `-a2-ref-noise-gate.md`, `-m3-accuracy-debt.md`,
`-q1-spark-quality.md`, `-a3-gsm8k-parity.md`, `2026-10-22-mp-metal-prefill.md`.

Methodology that is sound and I will not relitigate: in-engine A/B x3 with interleaved rounds and reversed order
(round 7, G1b, G1c, G2b, G3); same-day vLLM pairing on Spark; downtime logs; bitwise no-op checks on the other
backends before merge; mutants on new logic; `compute-sanitizer` on the NCCL graph path. That is better than most
published engine comparisons.

### B.1 "Spark: #1 of 2 on decode at 1k/4k, #2 of 2 on prefill" (standings §1, §2.1)

* Pairing: intarsia round-6 default (10-19) vs vLLM 10-19 same evening; round-7 opt-in vs vLLM 10-21. Same-day
  pairs, warm servers, 3 reps, median. Sound as far as it goes.
* Overstated: (a) n=2 engines, so "#1" and "#2" are "ahead" and "behind"; "top 3" is satisfied by existing.
  (b) Decode lead is +5.7 % / +2.1 % / -4.2 % with no stated Spark noise band; the Q1 same-GPU run the same day
  read 75.7 vs 76.4 at 1k (behind) and 81.9 vs 76.7 at 4k (ahead), so the 1k lead is inside noise. (c) The prefill
  deficit (-15.3 % at 16k) is the robust finding and is buried under the decode rank. (d) The intarsia row carries
  no quality number for the engine actually served on Spark that day (standings §4 admits it).
* Verified (docs + raw file names). Not overstated on prefill; overstated on decode rank wording.

### B.2 "CUDA round 7: +10-13 % prefill, +20 % with chunk 4096; vLLM parity at 16k" (round7 headline)

* The +10-13 % in-engine A/B is clean (3 rounds, spreads of 10-30 tok/s on 4.5-7k). Verified from the table.
* "Parity with vLLM at 16k" is opt-in `lt` + opt-in `--prefill-chunk 4096` against vLLM at its defaults
  (`max_num_batched_tokens` not raised; vLLM was not re-tuned). One side tuned, the other not. Suspected handicap
  of the comparator; the doc does not state vLLM's chunk setting.
* The same build **fails the prefix-reuse gate** (`chat_reuse_4k` continue 3/32) and moves 35b-short KL by 23 %
  vs vLLM. The doc says so, and the user kept it opt-in (LOG 2026-10-21 11:23). The standings table nevertheless
  lists the opt-in rows beside the default as if they were candidate standings. Mildly overstated.

### B.3 "tp=2 graphs + fusion decodes +22 / +23 / +18 % faster than vLLM" (tp2-cuda-graphs §1, standings)

* This compares intarsia on **two** GB10s (spark1+spark2 over RoCE) against vLLM on **one** GB10. vLLM tp=2 was
  "not run on the fleet" (G1b §4). The doc itself concedes aggregate throughput is below two tp=1 servers because
  batched decode refuses world > 1. The tp=2-vs-tp=1 numbers (+12-21 % decode, +22-25 % prefill) are sound; the
  "vs vLLM" row is apples to oranges and should not appear without a vLLM tp=2 arm. Verified (doc). Overstated.
* Also: the window-2 vLLM column is a reused same-day run, not re-measured in the window (doc §6).

### B.4 "R9700: #1 of 2 live, prefill 2.3-3.2x, decode 1.35-1.42x llama.cpp" (standings §2.2)

* Comparator: llama.cpp **Vulkan** build `0.4.0-dev 10864` with `MXFP4_MOE` GGUF (different quant), while
  (a) a newer v0.5.0 Vulkan build "exists but was not used", (b) the llama.cpp HIP build could not load its
  libraries in the dev image, (c) hipfire MQ4R (second-hand 204 tok/s decode vs intarsia 160) was not run,
  (d) vLLM-ROCm image is gone. The one live comparator is the slowest configuration of the slowest engine.
* Unexplained anomaly in the same section: intarsia took 2.8-3.0 s per MMLU call against llama.cpp's 0.37 s
  (8x), and its MMLU run was cut by the 600 s cap. For a 160 tok/s engine on 5-token answers that is a serving-path
  problem (TTFT? template? thinking?) that nobody chased. "quality: tie" rests on n=225 with that caveat.
* Prefill margin probably survives any comparator; decode "#1" does not survive hipfire. Verified (doc). Overstated
  on decode rank; comparator handicapped.

### B.5 "Metal: #1 of 3 on prefill and on decode at every length" (standings §2.3, MP headline)

* Noise: GPU shared with a desktop running Zoom, load average 18-28, single cells +-10 %, intarsia's own 16k
  prefill moved 8 % between arms, llama.cpp decode spread 17 %. Decode margins over llama.cpp are +0 to +8 %.
* G3 §5b (verified, line 196): llama.cpp ran ~170 MHz lower than the intarsia arms; applying intarsia's own measured
  clock slope (0.014 tok/s per MHz) "would put intarsia near 0.98x llama.cpp at 4k". The team wrote this and then
  the standings still claim "#1 of 3 on decode at every length".
* Quality column: "invalid" (0.0), later diagnosed as the harness's `response_format` being ignored by intarsia
  (standings §10) -- i.e. a real server feature gap (no `response_format` handling) presented as a harness fault.
* Prefill lead (1.5-1.7x llama.cpp, 1.1-1.7x mlx-lm) is robust. Decode "#1" is not established. Overstated.

### B.6 "35b-long accuracy PASS" (tp2-cuda, tp2-hip, m35 PR #70, a2)

* On 10-20 every 35B build FAILed the 35b-long floor on all rows (4.3-4.7e-2 vs 3.0e-2). On 10-21 10:43 the floor
  was redefined to "scored rows" (rows whose target is outside system/user turns, `turns.rs`), which reads 1.42e-2,
  and every build PASSes (a2 §2.1 table, verified: "logged FAIL / main today PASS / bound by static floor").
* The rationale (serving never predicts user-turn tokens; vLLM's own re-run moves those rows by 0.10-0.12 KL) is
  reasonable. But: (a) the change was decided the same morning the failure was analysed, (b) the unexplained
  systematic offset remains: on the scored rows intarsia sits 1.6-1.7x further from the vLLM golden than vLLM
  sits from itself (1.36-1.42e-2 vs 8.4-8.7e-3), on every backend, including the f64 CPU oracle, and the
  fp8-KV hypothesis was tested in A3 and found to explain only a small part ("KL CPU||vLLM(bf16 KV) 0.100 vs
  CPU||vLLM 0.174 at the last row; vLLM-CUDA = +0.103 digit mass"), (c) the noise-aware gate at factor 1.0 would
  fail every 35B build (a2 §1: "no 35B build would pass at factor 1.0"), so the static floor is what passes them.
* Verified. The PASS is real under the current definition; the definition was moved to make it so, and the
  residual offset vs the only reference is unexplained and larger than the reference's noise.

### B.7 "GSM8K gap is not in intarsia" (a3) -- partly overstated

* The paired replay is excellent work (1319 questions, McNemar). The claim that intarsia is "well ahead" on
  last-number scoring is a different metric than the one localeval reports and that the standings use.
* The CPU-oracle arbitration covers 24 prompts and, as the doc admits, shares intarsia's model definition
  (recipe, scale interpretation, GDN). It shows intarsia's two backends agree; it does not show vLLM is wrong.
  Combined with B.6, the honest statement is "intarsia and vLLM disagree on this checkpoint by more than vLLM's own
  noise, in a direction that changes GSM8K behaviour, and we do not know which is right".

### B.8 Metal decode gate (G3, MG) -- gate redefined after failing

* 10-15: 87.7 % of a flat 40.6 tok/s floor, PASS. 10-21: 82.9-84.5 %, FAIL on an unchanged binary (G3 showed the
  old binary also reads 33.8 on the day: verified, code did not regress). Same day the gate became (1) a context-aware
  floor (39.0 at 4k instead of 40.6), (2) clock-normalised by an mlx reference workload, (3) with a state window that
  makes out-of-window runs STATE-INVALID rather than FAIL, and (4) competitor arms "reported, not admitted".
* MG §Limits: the normalisation baseline is "circular for this run" (median of the same run's samples) and the
  rescue of a degraded run "is shown only on synthetic data in the unit tests".
* Each piece is defensible; together they guarantee the gate cannot fail on the machine that runs it (a loaded
  endor is STATE-INVALID, a quiet endor passes). The ratio gate (>= 0.90x mlx-lm, >= 1.00x llama.cpp) is the only
  part that still bites, and B.5 shows it is not clock-matched.

### B.9 Other things that were quietly loosened this week (verified in LOG.md / docs)

* 35b-short baseline gate: ADVISORY (ADR-0008), `scripts/phase1b-acceptance.sh:464-503` prints "ADVISORY FAIL
  (reported, not blocking)".
* `--self-golden --self-vs-ref-noise`: max-position KL, top-1 flips and tail mass "reported only"
  (`refnoise.rs:645-670` `apply_to_self_checks` sets `gated = false`); default self-golden budgets (1e-4) were
  "unreachable" for any real kernel change on 35B (a2 §1), so they were replaced by "3x each prompt's own vLLM
  noise", which the same doc calls "a loose per-prompt bound" (tightest prompt at 0.31-0.63 of its limit).
* HIP tp=2: short-set KL 1.65x vLLM's run-to-run bar (G2b "FAIL by the literal bar"); user decision 2026-10-21
  (MG §Decisions 5): "HIP tp=2 need not pass `--self-vs-ref-noise` against tp=1".
* Qwen3.8-27B W4A4 prefill: FAIL -> PASS via the ref-noise term (a2 §1); decode stays FAIL at 1.39x and the
  follow-up is deferred.
* `rescore_cuda_round7_from_the_committed_table` (accuracy.rs tests) asserts the committed round-7 numbers pass at
  factor 3.0. That is a test of a decision, not of code: it will fail if anyone tightens the factor, which makes the
  factor harder to tighten.

---

## D. Accuracy methodology: can a real regression pass?

Yes, on the 35B, by a wide margin. Evidence, all verified in docs or code:

1. **Static floor headroom.** 4-bit class floor 3.0e-2 on scored-row KL21; engine sits at 1.36-1.46e-2 across
   CUDA tp=1/tp=2, HIP tp=1/tp=2, Metal (PR #70). A regression that doubles the engine's distance from the reference
   passes. For scale, the entire tp=2 reordering effect is 8e-3 and the `lt` kernel swap 7.6e-3.
2. **The 1 % partial-scale mutant passes.** G1b §2 finding 4 and G2b self-golden table: `INTARSIA_TP_TEST_BUG=scale:1.01`
   on rank 1 "passes the 35B budget and reads the same as the clean tp=2" on both CUDA and HIP, on short and long.
   The only thing that catches it is a Mac f32 unit test on a synthetic hybrid fixture
   (`tp_hybrid::hybrid_moe_tp2_catches_a_one_percent_partial_scale_bug`), which covers the TP sharding code path
   and nothing else (not a kernel, not a scale-loading bug on tp=1).
3. **Baseline-relative gates are not available on CUDA/HIP.** `golden/baselines/` on main holds six Metal 8B files.
   `qualify.rs:1847`: "no baseline for backend ... KL, tail, prefix-reuse gates only, top-1 not gated". The
   envelope/baseline machinery that could catch a 2e-3 drift exists but is exercised only through ad-hoc
   per-lane scripts with uncommitted baselines on the boxes.
4. **35b-short baseline is advisory**; `--self-vs-ref-noise` ungates everything except mean KL and non-finite rows;
   per-prompt limit is 3x a single vLLM re-run that itself moves 1.3x between re-runs (a2 §2.2).
5. **Prefix-reuse gate**: FAILs only when `rows.is_empty()` (`accuracy.rs:2094-2100`); prompts whose split does not
   cross a chunk boundary are SKIPPED and do not fail. At the HIP default chunk 4096 the 4k-prompt `continue` cases
   are skipped (M3 §2.8), so "default passes" is partly "default does not run it". The 12k prompts do cross and do
   run, so the gate is not vacuous, but the case that failed at chunk 1024 is the one the default never sees. The
   hybrid recipes carry no measured `reuse_logit_tol` (accuracy.md says it "must be measured once on a known-good
   build"); the built-in 0.25 nat / 1e-2 values were calibrated on dense bf16 8B.
6. **The reference itself**: vLLM `guidance` is non-deterministic at T=0 (A3 §3: 126/200 identical replies between
   two runs; 71 vs 66 exact), runs fp8 KV + prefix caching + `max-num-seqs 2`, and `prompt_logprobs` on a cached
   prefix "is unverified". Every 35B gate inherits this. `tools/golden.py` cannot load the NVFP4 checkpoint, so no
   HF reference exists; the f64 CPU oracle exists (A3 used it for 24 prompts; M3 for one 4k prompt in 25 min) but is
   not wired into any gate.
7. **Tests that cannot fail / are not run**: 22 `#[ignore]` tests and 34 env-gated (`INTARSIA_TEST_REAL_MODEL`,
   `INTARSIA_REQUIRE_GPU`) tests on main; CI excludes all three GPU crates and skips
   `bad_decode_row_fails_only_its_request` (a batched-decode fault-isolation test that "fails 3 of 3 runs on the
   runner, passes on endor" -- which is the signature of a real race, not a test race, until shown otherwise).

What would make the gate able to catch a regression on the 35B (ordered by cost):

* Wire the f64 CPU oracle into `qualify` for the 35B (one short prompt set, prefill-only, ~30 min on endor) and gate
  **engine-vs-oracle** at the bf16 class floor on scored rows. The oracle shares the model definition, so it will not
  catch a recipe error, but it will catch every kernel, scale and TP regression at 1e-3 resolution instead of 3e-2.
* Commit a CUDA and a HIP baseline + envelope for the 35B goldens to `golden/baselines/` and make `qualify` refuse to
  PASS the 35B without one (today it downgrades to KL-only with a note).
* Make the 1 % partial-scale mutant a required-kill in the TP gate on GPU (it is a one-env-var run), not only on the
  Mac fixture.
* Keep the row mask, but report the all-rows number next to it in every table and in the verdict JSON so the
  redefinition stays visible.
* Measure `reuse_logit_tol` once on a known-good 35B build and put it in `recipes/qwen3_5_moe.toml`; run the
  `continue` case at chunk min(default, 1024) in `qualify` so the default is not vacuous.

---

## C. Correctness risk in code merged since 2026-10-15 and in PR #70

### C.1 What is sound (one line each, verified)

* No `unsafe`, no `async`, no `RwLock` in `intarsia-core/src`; the solo engine serialises requests under one mutex.
* Sampler RNG is per request (`engine.rs:918`, `sampler.rs:87-101`); greedy never touches it; unseeded requests
  seed from `SystemTime`, so run-to-run determinism needs an explicit `seed` (document it).
* Prefix reuse never hands a running slot to another request (`scheduler.rs:634`); a failed request truncates its
  slot to 0 so it is never a reuse candidate (`scheduler.rs:564, :587`); `Model::forward` zeroes recurrent state on
  error (`model.rs:1069-1071`).
* tp=2: rank 0 alone samples; ids travel inside `Forward` (`tp_remote.rs:200-204`), so there is no separate
  broadcast to desync; per-step length checks ("tensor-parallel ranks diverged", `model.rs:1109, :1187`); failures
  `comm.abort()` so the other rank cannot hang (`tp_remote.rs:135, :143`); graph failure at world > 1 is a hard
  error, not a silent eager fallback (`graph_step.rs:171-177`); NCCL landing buffer is reserved at load
  (`Comm::reserve`) and the `no-reserve` mutant was killed by `compute-sanitizer` (G1c §2).
* Token feedback (T1) is scoped out of TP, recurrent, graphs, batching, structured output and speculation
  (`feedback.rs:74-84`, `engine.rs:1283-1288`); the stop-one-step-late rollback is mutant-tested (8/8).
* CUDA graphs under TP include the all-reduces in the captured graph (not split), gated on `Comm::capture_safe`
  (NCCL only), with rank 0's replay decision carried as `FWD_GRAPH`.

### C.2 Findings (verified unless marked)

1. **Hybrid prefix reuse is only tested end to end on the fleet, never in CI.** `crates/intarsia-models/tests/engine.rs:243-294`
   drive reuse through the engine on an attention-only toy model (aligned restore `== 0.0`, unaligned `< 1e-3`);
   `hybrid.rs:176` tests `SequenceState::snapshot/restorable_len/truncate` directly, not through `Seq::new`
   (`engine.rs:899-904`) and the prefill snapshot schedule (`engine.rs:1124-1143`). The one path where a wrong
   `restorable_len` would silently corrupt output (GDN state restored at a length that does not match the KV) has
   no CI test on a hybrid model. The CLI `--mode reuse` gate covers it on GPU, but see D.5 for what the default
   chunk skips. Severity: high (requirement 3 on the hybrid flagship).
2. **Unaligned recurrent restore is tolerance-judged, not exact, and the tolerance was calibrated on a dense 8B.**
   `state.rs:200-244` restores only at an exact snapshot length; the engine then re-prefills the rest, which on the
   35B moves the stream by KL up to 8.5e-2 / 1.2 nats (M3 §2.8) against a built-in tie rule of 0.25 nat / 1e-2 KL.
   The rule's "rounding control" (re-chunk the cold run) reads 0 on HIP because HIP prefill is chunk-invariant, so a
   0.0037-nat near-tie counted as a FAIL at chunk 1024 and the same mechanism could let a real divergence pass at
   another split. Severity: medium.
3. **Remote-rank `Truncate`/`Snapshot`/`NewState` are fire-and-forget; errors surface only on the next `Forward`**
   (`tp_remote.rs:273-306`, `pending`). Rank 0 proceeds with a rolled-back state while rank 1 may still hold the old
   one; the next step aborts the group, which is safe, but every such error is reported as a forward failure with the
   wrong message context. Severity: low; fix is a reply on `Truncate` (one round trip per reuse, negligible).
4. **`SequenceState::truncate` order** (`state.rs:204-209`): in-process peers are truncated before the primary can
   fail at the snapshot lookup (:219), leaving ranks at different lengths. Reachable only if snapshot sets diverge
   between ranks, which `snapshot()` prevents by recursing into peers. Severity: low; cheap to make atomic (look up
   the snapshot first).
5. **`SequenceState::new` has no overflow check on `n_kv_heads * cap * head_dim * dtype`** (`state.rs:97`). With
   `--max-seq` taken from the CLI this is a u64 on 64-bit hosts, so practically unreachable; note only.
6. **Defaults that changed numerics this week** (from `git log --since=2026-10-15`):
   * HIP `matmul_fp8_mma_f8` became the default fp8 prefill GEMM (8624fec1): 35b-short KL21 7.071e-3 -> 7.417e-3
     (worse), long 4.583e-2 -> 4.356e-2 (better); user-decided. Everything else on main that changed a default
     (HIP MoeBlock on, CUDA `gdn_delta_rule_bf16_split`, HIP flash v2 4-head, HIP split trigger 950 us, HIP split-KV
     8-wave made default then reverted, chunk 4096 on HIP for the 35B, Metal chunk 2048, Metal token feedback,
     CUDA graphs at tp>1) is claimed bitwise and was no-op checked. Two opt-in paths are known non-bitwise and are
     candidates for default-on in open PRs #59/#63/#64 (TP fusion) -- those PRs should carry the same KL tables.
   * `INTARSIA_CUDA_FAST_PREFILL` parsing changed from "any value" to `== "1"` (9aa73f1b): a script setting it to
     `0` to *enable* it (the old semantics) now silently disables it. Severity: low, but it is exactly the class of
     silent flag drift the build stamp was meant to prevent; grep the fleet scripts.
7. **CI cannot catch a GPU or batching regression.** `.github/workflows/ci.yml:54` excludes the metal, cuda and hip
   crates, runs with `CI=0` so every `INTARSIA_REQUIRE_GPU`/`INTARSIA_TEST_REAL_MODEL` test skips, and skips
   `bad_decode_row_fails_only_its_request` by name. I read that test (`batch_engine.rs:277-305`): the failing
   assertion is `(m.failed, m.finished) == (1, 3)` read immediately after `collect()` returns, i.e. before the
   scheduler thread has necessarily bumped `finished` -- a metrics-ordering race in the test, not a fault-isolation
   bug (the fault-isolation assertions above it passed). Fix: wait on the counter with a timeout, then drop the
   `--skip`. Severity: low for the engine, medium for process (a skipped test in a 6-day-old CI is how real races
   get hidden later).
8. **`rescore_cuda_round7_from_the_committed_table`** pins the gate's factor to the data (see B.9). Severity: low.

### C.3 PR #70 (`lane-m35/metal-moe`): Metal kernels for the 35B

What the PR shows: nine new Metal ops, admission 0 FAIL (39 kernels, 688 cases), `cargo test -p
intarsia-backend-metal` 125/0, accuracy vs the vLLM goldens PASS (short prefill 5.78e-3, long 1.43e-2, decode
1.12e-2 / 1.14e-2; numbers in line with CUDA/HIP), one coherent greedy sample. No core change. What it does not show
(verified from the PR doc and diff): no `--mode reuse` run (GDN snapshot/restore on a brand-new backend is untested),
no prefix-reuse or 256-greedy-ids comparison against CUDA, no speed number, no Metal no-op check of the *8B* path
after linking four new `.metal` files into the one metallib (symbol-prefix discipline is asserted in prose), and the
tests bind weights only at 8 mod 16 alignment (the real shard layout) with unaligned weights refused at launch.

Kernel-level findings from the independent read of the diff: see C.4.

### C.4 PR #70 kernel review (delegated read of `kernels/metal/{moe,nvfp4,fp8,gdn}.metal` and the Metal crate;
relayed through the coordinator, not re-verified by me except where noted)

No kernel correctness bug was found. The NVFP4 e2m1 decode and scale application, the fp8 e4m3 decode with static
activation scale, the top-k semantics (ties to lowest index, softmax over the selected set), the expert-grouping
barriers and the bf16 rounding points of the MoE path, and the delta-rule recurrence were judged consistent with the
CUDA/HIP kernels and the CPU reference. Five findings, all about coverage or plumbing:

1. **Medium (verified by mutation in the review): `tests/moe.rs` at T=300 uses uniform routing**, so every expert
   gets the same token count and the multi-tile branch of `moe_tile` (an expert with more than one 32-row tile) is
   never exercised by a cargo test; the admission sweep covers it only if its random routing happens to produce
   it. I confirmed the test runs T in {1,2,3,300} and asserts run-to-run bitwise and 8-mod-16 invariance
   (diff lines 108-128 of `tests/moe.rs`), but could not confirm the routing distribution from the diff alone.
   Fix: add a skewed-routing case (one expert with >= 3 tiles, one with 0 tokens) compared against the CPU
   reference at the same tolerance as the uniform case.
2. **Medium: `moe_topk`, `moe_combine`, `gate_split` and `sigmoid_mul` have no cargo test; GDN is tested only at
   K=4 and dk=128.** Admission covers them on random data, but the pin tests in `kernels.rs` do not. Fix: one test
   each with a hand-built expected value (top-k ties, k = experts, zero logits; conv at K=2 and 8; delta rule at
   dk=64 and 256 with a chunk boundary mid-sequence).
3. **Low: the scale-offset and f32-output MoE variants are never run** by a test. Fix: extend the T=3 case to both.
4. **Low: the `moe_topk` registration predicate does not assert `top_k <= experts`.** The engine never asks for
   more, so this is a defence-in-depth gap only. Fix: add the predicate; the plan then refuses loudly.
5. **Low: MoE output is not batch-invariant across the `T*K <= 16` decode/grouped split** (the per-pair GEMV and the
   grouped simdgroup-matrix path round differently). This is documented for CUDA/HIP as well and is why `--max-batch`
   is Metal-only and MoE-excluded (ADR-0010), but it means the 35B on Metal cannot join continuous batching without
   row-exact kernels, and the prefill/decode near-tie behaviour seen in B.2 (`chat_reuse_4k` 3/32) will recur
   here. Fix: none needed now; document it in the PR and in ADR-0010's Metal row.

Merge recommendation for PR #70: mergeable as an opt-in port once (1) and (2) are added and a `--mode reuse` run on
the long golden is attached (GDN snapshot/restore has never run on this backend). No core change, so the CUDA/HIP
no-op checks are docs-only.

---

## Ranked findings (the summary the caller asked for; each has evidence, requirement, fix, acceptance)

| # | sev | status | finding |
|---|---|---|---|
| 1 | critical | verified | 35B accuracy has no independent reference and the shipped gates cannot see a 1 % error (D.1-D.6, B.6) |
| 2 | high | verified | Gates redefined after failing, same day, same people: 35b-long row mask, Metal decode gate, HIP tp=2 waiver, self-vs-ref-noise ungating (B.6, B.8, B.9) |
| 3 | high | verified | "Top 3" is measured single-stream against at most two comparators; CB is Metal-only and excludes MoE; tp=2-vs-vLLM is 2 GPUs vs 1 (B.1, B.3, A.1) |
| 4 | high | verified | New-model path refuses most of the catalogue at recipe load (rope scaling, attention bias, sliding window, non-silu); no graph IR (A.2.1) |
| 5 | high | verified | Hybrid prefix reuse end to end has no CI test; the fleet gate skips the failing case at the default chunk; tolerance calibrated on a dense 8B (C.2.1-2, D.5) |
| 6 | medium | verified | R9700 and Metal comparators are handicapped or not clock-matched; Metal decode "#1" is ~0.98x llama.cpp clock-matched by the team's own slope; R9700 llama.cpp is an old Vulkan build and an 8x per-call latency anomaly is unexplained (B.4, B.5) |
| 7 | medium | verified | ARCHITECTURE.md describes paged KV, radix cache, tuning DB, autotuner, kernel packs and a `intarsia-quant` crate that do not exist; requirement 5 leans on the tuning DB (A.1) |
| 8 | medium | verified | Baselines/envelopes for CUDA and HIP are not committed; `golden/baselines/` holds Metal 8B only; qualify downgrades to KL-only silently on those backends (D.3) |
| 9 | medium | verified | Quant format is an enum across 21 files with one `Op` per (format, fusion); adding a format is ~11 sites; accuracy harness has 20+ gate-shaping flags (A.2.2-3) |
| 10 | medium | reported | PR #70 MoE multi-tile path and four ops untested by cargo tests; no reuse-mode run on Metal 35B (C.3, C.4) |
| 11 | low | verified | CI excludes all GPU crates, runs `CI=0`, skips one batching test by name; 22 `#[ignore]`, 34 env-gated tests; `INTARSIA_CUDA_FAST_PREFILL` semantics flipped silently (C.2.6-7) |
| 12 | low | verified | tp_remote `Truncate`/`Snapshot` errors are deferred to the next `Forward`; `truncate` is not atomic across ranks (C.2.3-4) |

Fixes with acceptance criteria are in the sections referenced; the summary returned to the caller repeats them.
