# g4f: the full Gemma-4-26B-A4B-NVFP4 through `intarsia onboard` quick tier on CUDA

Lane g4f (follow-up of lane g4, docs/results/2026-10-22-g4-gemma4-onboard.md), branch `lane-g4/gemma4` = PR #101 (draft), fleetq owner
`claude-f31ccf91-g4f`, lease #4260 on spark2 (64 GB, `--quiet gpu,cpu`, NOT exclusive; spark2's vLLM service was up and idle,
GPU 0 % busy at the start). Raw logs and goldens: `docs/results/2026-10-22-g4f-gemma4-full/raw/`. The CUDA binary of every run here was built from merge
commit 16c2205f (later commits only add HIP `UNIMPLEMENTED`, tools, docs and LOG rows). Clock: 19:42 start, 20:35 this
text (T = EDT 2026-10-22).

## Verdict

**The full 26B loads, runs and answers correctly on CUDA, and `intarsia onboard --quick-only` ran end to end in 7 min 01 s.
The quick tier verdict is FAIL (gate and acc-short), and neither failure is an engine defect; no floor was changed, PR #101 stays
draft.**

| Stage (CUDA, GB10, spark2) | Verdict | Wall | Number |
|---|---|---|---|
| doctor / recipe | PASS | 1 s / 0 s | `builtin:gemma4`, 41/41 tensor templates, quant nvfp4 |
| smoke (load + one raw prompt) | PASS | 28 s | |
| plan | PASS | 72 s | 21 ops resolved, model loaded in 72 s (cold page cache) |
| kernels | PASS | 84 s | 71 kernels, 1425 cases |
| gate (ppl <= 40, raw sanity prompts) | **FAIL** | 38 s | ppl 533.5 (HF bf16 520, vLLM 402), 4 of 5 sanity prompts repeat |
| acc-short vs HF bf16 golden | **FAIL** | 46 s | prefill 21-way KL 2.95e-1 (floor 3.0e-2), top-1 83.0 %; decode KL 3.35e-1, top-1 80.6 % |
| acc-format / acc-oracle | SKIP / MISSING-BASELINE | | no format reference; no committed f64 CPU oracle for this model |
| bench 128 / 1024 / 4096 | PASS | 153 s | see Speed |

Total `onboard` 7 min 01 s (peak RSS 18.3 GB). From the start of this lane to the quick verdict: 21 minutes of wall clock (19:42 to
20:03), which holds the origin/main merge (about 9 min), a 21 s warm CUDA build, the HF reference golden (4 min: load + dequant 126 s,
forward 13 s) and the 7 min `onboard`. The next 50 minutes were diagnosis (below), the part a generic onboarding flow does not
yet do for you.

## Why the gate fails: the model, not the engine

The quick gate runs raw text (no chat template). Gemma-4-26B-A4B is an instruction model and is bad at raw text in every
implementation tried, on identical token ids, no `<bos>` (the checkpoint's tokenizer.json adds none):

| Perplexity on `tools/prompts/ppl.txt` (1023 tokens) | no `<bos>` | `<bos>` first |
|---|---|---|
| engine (`accuracy --quick`, 1166 tokens) | 533.5 | not run |
| HF transformers 5.6.2, bf16 (`tools/gemma4_slice/hf_sanity.py`) | 520.0 | 7758 |
| vLLM 0.32.0rc1 (W4A4, `tools/vllm_ppl.py`) | 402.1 | 6676 |

HF and vLLM also repeat on the raw sanity prompts ("The capital of France is" -> " a capital of France is a capital ..."). In
chat form the same engine is fine (`raw/chat-greedy.log`, greedy, temperature 0, `--chat`):

- "What is the capital of France? Answer in one word." -> `Paris` (HF bf16: `Paris`)
- "Name the three primary colors." -> "The three primary colors depend on whether you are referring to light or pigment: 1. Additive ... Red, Green, and Blue (RGB). 2. Subtractive ... Red, Yellow, and"
- "Write a Python function that returns the n-th Fibonacci number." -> "There are several ways to implement this ... ### 1. The Efficient Approach (Iterative) ..."

The raw smoke prompt in the quick tier ("a로.\n- - - -") and the raw sanity completions in `raw/onb/qualify/logs/gate.out` are what
the base model's raw mode produces here. I tried a recipe `add_bos` key (prepend `bos_token_id` to raw text): it makes raw text
worse (above), so it was reverted, not shipped. The gate needs a chat-form variant for instruction-only models (for example the
perplexity text and sanity prompts wrapped in the model's chat template); that changes the gate's input, so it is left for the
coordinator (LOG row 2026-10-22 "Gemma-4 quick gate"). The ppl <= 40 floor itself was not touched.

## Why acc-short fails: bf16 noise of a router-sensitive MoE, measured

Reference ids are the 10 prompts of `golden.jsonl` + `golden_long.jsonl` (1468 prefill positions, 160 decode rows), `--template hf`.
Three references of the same checkpoint, all on identical ids (`tools/golden_cmp.py`, `intarsia accuracy`, `intarsia ref-noise`):

| Reference | How | Engine vs it: prefill 21-way KL / top-1 | HF bf16 vs it |
|---|---|---|---|
| HF bf16 (the golden the quick tier uses) | GPU, NVFP4 experts dequantised to bf16 in memory, `tools/gemma4_nvfp4_golden.py` | **2.95e-1 / 83.0 %** (decode 3.35e-1 / 80.6 %) | |
| HF f32 compute | same weights; non-expert tensors f32, experts bf16-dequantised and cast per expert (`--dtype float32`, ~60 GB) | **2.38e-1 / 86.7 %** (`accuracy`; decode 2.02e-1 / 89.4 %) | 2.54e-1 / 83.9 % (`golden_cmp`, same pair) |
| vLLM (nightly 3f1bcfd8, W4A4) | prompt_logprobs on the HF ids (`tools/golden_vllm.py --ids-from`) | 7.9e-1 / 73.0 % | vs HF bf16 5.5e-1 / 75.3 %; vs f32 6.3e-1 / 73.1 % |

The two f32 comparisons use different estimators (`accuracy` scores the engine's full-vocabulary logits against the f32 top-20;
`ref-noise` and `golden_cmp` compare two top-20 lists), so the like-for-like pair is `intarsia ref-noise --golden f32 --other X
--method precision_spread` for both X (`raw/ref-noise-f32-matched.txt`; X = the engine's own top-20 from `accuracy
--write-self-golden`): **engine 2.08e-1, HF bf16 2.36e-1** (`golden_cmp`: 2.39e-1 against 2.54e-1; top-1 86.7 % against 83.9 %).

Reading: **the engine is no farther from the f32 reference than HF's own bf16 forward is** (0.88 times its distance under one
estimator), and engine-versus-bf16-golden (2.95e-1) is what two independent bf16 draws of this model give: 1.25 times the measured
reference noise of 2.36e-1 (or 1.33 times 2.23e-1 in the direction `ref-noise --golden bf16 --other f32`; the sidecar written from
that run, `raw/ref_noise/`, makes the floor 2.23e-1 and the run still FAILs, "KL / ref noise = 1.326", `raw/acc-noise.log`). The
cause is the router: top-8 of 128 experts per layer over 30 layers flips on near ties under bf16 rounding, and the off-template raw
prompts (story 1.1 to 1.5, code 0.49 to 0.59, long_prose_300 0.43 to 0.49) are the most sensitive; chat prompts are 0.04 to 0.26
(engine vs f32: chat_system 3.7e-2, chat_multiturn 7.1e-2). vLLM is farther from everything because it runs the declared W4A4
(per-expert `input_scale` activation quantisation), which the engine does not (LOG 2026-10-22, W4A4 as W4A16). The 2-layer
real-weight slice of lane g4 (KL 3.5e-3) did not see this because two layers cannot flip a router often. I did not lower the 3.0e-2
floor and did not change the ref-noise factor (a factor of 1.33 would pass; that is the user's decision).

Limits: the f32 reference still rounds the experts' weights to bf16 (relative 2^-9, deterministic, not router-flipping) and uses the
bf16-rounded embedding scale 53.0 (as the engine does); it is not an exact fp32 forward of the dequantised NVFP4 weights, and no
second independent f32 implementation exists. `acc-oracle` (f64 CPU oracle) was not attempted. Multi-chunk prefill is untested (longest golden prompt 700 tokens < chunk 1024).

## Speed (onboard `bench`, quick tier; slot: non-exclusive `--quiet gpu,cpu` lease on spark2, nothing else of this lane running)

| tokens | decode tok/s (eager, "% of byte floor") | prefill tok/s |
|---|---|---|
| 128 | 34.3 (274 %) | 634 |
| 1024 | 31.1 (248 %) | 536 |
| 4096 | 24.6 (196 %) | 366 |

A second `onboard --quick-only` run of the same binary against the f32 golden (`raw/onb-f32/`, same lease, 20:35-20:42; its
`gate` / `acc-short` verdicts are FAIL as above, acc-short KL 2.38e-1 prefill / 2.02e-1 decode) measured 34.5 / 31.3 / 24.7
decode and 630 / 534 / 372 prefill tok/s: the two samples agree within 2 %.

These are NOT tuned numbers and not exclusive-slot numbers: Gemma layers are not CUDA-graph captured (decode "ran eagerly"), the
MoE prefill uses the generic GELU kernel (`moe_intermediate_size` 704 is not a multiple of 128, so the mma / w4a16 GELU kernels do
not apply) and head_dim 512 global attention is a SIMT kernel. The "% of byte floor" column is computed with a `--gb-per-token`
estimate and the report itself flags it as not a plain-decode floor. They say "works", not "fast".

## Checks

- Merge of origin/main (a92267b0) into the branch: conflicts in `ops.rs` (`Op::ALL` 42, GeluGlu appended after main's RopeScaled /
  AddBias), `fast.rs` (CPU rope: proportional rope and `RopeScaled` combined), `recipe.rs` (BUILTIN_RECIPES), `tuning.md`, LOG.md
  (resolver). Merge commit 16c2205f.
- CPU tests on spark2 (`cargo test -p intarsia-core -p intarsia-models -p intarsia-cli`, debug): 448 passed, 0 failed (includes
  gemma4_tiny, recipes_doc, docs_env_vars). Full workspace (`--exclude intarsia-backend-metal`, which does not compile off macOS):
  706 passed, 0 failed (119 test binaries, `raw/cargo-test-workspace-summary.txt`). **One real fix from this**: `cargo test -p intarsia-backend-hip` failed on the PR branch
  (`names_unique_and_ops_covered`: `gelu_glu` neither in the HIP kernel table nor in `UNIMPLEMENTED`); `Op::GeluGlu` is now listed
  unimplemented (HIP and Metal still do not run Gemma-4; their source is otherwise unchanged by this PR, Metal not compiled here).
- Qwen3-8B no-op on CUDA, FULL 36 layers (spark2, `Qwen/Qwen3-8B`, `--max-seq 1024`): binary of origin/main a92267b0 versus this
  branch: `plan show` identical (apart from the load-time line), 256 greedy ids identical (`raw/noop/`).
- Not run: `kernels test` full sweep, in-depth tier, 4k goldens, TP, HIP/Metal builds.

## Timeline (EDT)

| Time | Event |
|---|---|
| 19:42 | start; `fleetq status`: spark2 78 GB free |
| 19:48 | lease #4260 (spark2, 64 GB, expect 150) granted immediately |
| 19:49-19:51 | merge origin/main, resolve 5 conflicts, commit |
| 19:50 | CUDA release build (warm target): 21 s |
| 19:51-19:54 | HF bf16 golden of the full model (3 tries: LUT on CPU, triton needs Python.h -> `TORCH_DISABLE_NATIVE_JIT=1`; load+dequant 126 s, forward 13 s) |
| 19:56-20:03 | `onboard --quick-only` on the full model: 7 min 01 s, verdict FAIL (gate, acc-short) |
| 20:03-20:25 | diagnosis: HF raw-text ppl 520 (same as engine), BOS test, vLLM golden + ppl, f32-compute HF golden (~4 min), ref-noise record |
| 20:25-20:35 | Qwen3-8B no-op, workspace tests, chat greedy, this doc |

## Next

1. A chat-form quick gate for instruction-only models (ppl text and sanity prompts through the chat template). Until then every
   such model fails `gate` on raw text.
2. Decide how a router-sensitive MoE is judged: the f32 reference (`--dtype float32`) and its noise record
   (`raw/ref_noise/`) are now reproducible in ~4 min on one Spark; whether the existing `ref-noise-factor` rule should apply at
   1.33 (two independent bf16 draws) is the user's call.
3. Speed work, only after (1) and (2): CUDA-graph capture of the Gemma layer, a 704-wide mma MoE kernel, a tensor-core head_dim 512
   attention.
4. `acc-oracle`: commit an f64 CPU oracle for the model (needs a host with RAM for the CPU reference at 26B).
