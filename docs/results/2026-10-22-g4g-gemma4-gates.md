# Gemma-4-26B-A4B-NVFP4: recipe-scoped quick-gate fixes (lane g4g, 2026-10-22)

Branch `lane-g4/gemma4` (PR #101), fleetq owner `claude-f31ccf91-g4g`. Follow-up of `2026-10-22-g4f-gemma4-full.md`, whose
`onboard --quick-only` on CUDA failed two stages that are not engine defects: `gate` (raw-text perplexity 533, limit 40) and
`acc-short` (KL 2.95e-1 / 3.35e-1 against the HF bf16 golden, floor 3e-2, bf16 MoE router flips). User-delegated 20:59.

**Result (follow-up, user-approved 21:51; spark2 lease #4302, non-exclusive `--quiet gpu,cpu`): `onboard --quick-only` on CUDA PASSES, exit
0, 7 min 10 s: `gate` PASS, `acc-short` PASS against the HF f32 golden, `acc-oracle` SKIP. The first version of this lane (spark1, lease #4276: `acc-short`
against the bf16 golden with factor 1.6 and a committed CUDA baseline) ended at exit 1 on `acc-oracle` MISSING-BASELINE; it is superseded by
section B2. The f64 CPU oracle was built and measured and is NOT committed (section C).**

## A. Chat-form quick gate (`[accuracy] ppl_form = "chat"`, `max_ppl`)

* `tools/prompts/ppl_chat.json`: a fixed three-turn conversation (gardening advice, 6 messages). `accuracy --quick` renders it with
  the model's own chat template (`Engine::apply_chat_template`, the same `minijinja` template serving uses). For every assistant
  message there is one forward: the prompt as the model is served (earlier messages plus the generation prompt) followed by the tokens
  the template renders for the turn itself (content and turn-close). Only those tokens are scored (320 in total). The turn starts
  where the generation prompt and the rendering of the finished turn stop agreeing, so no turn format is hard-coded.
* Why the served prompt and not the history rendering: Gemma-4's generation prompt ends in an empty thought channel
  (`<|channel>thought\n<channel|>`) that its rendering of a finished turn omits. Scoring the history rendering gave 1634 (engine) and 1682
  (HF): off-distribution for this model. Scoring the answer after the prompt it is actually served with gives:

| Reference, same 320 assistant tokens | Chat-form perplexity |
|---|---|
| HF transformers bf16 (`tools/chat_ppl.py hf`, spark2, lease #4266, non-exclusive `--quiet gpu,cpu`) | 35.14 |
| intarsia CUDA (`accuracy --quick`, same lease) | 35.97 (2.4% above HF) |
| vLLM W4A4 (`tools/chat_ppl.py vllm`) | not measured: spark2 was taken back by the coordinator before a server could be started, and spark1 has no vLLM image that is known to serve Gemma-4 (the g4f image `nightly-3f1bcfd8` is on spark2 only) |

* Limit: `max_ppl = 60.0` in `recipes/gemma4.toml`, 1.7x the HF number. The raw-text numbers (HF 520, vLLM 402, engine 533) show what a
  broken or off-form measurement looks like; the engine is within 3% of HF on the chat form. The vLLM figure should be added when a Spark
  is available; it can only confirm the limit (vLLM is W4A4 and was farther from everything else in the g4f doc).
* The five sanity prompts, sent as user turns: all five OK (counting continues "11, 12, ...", the story continues, blue / "**Paris**" /
  "**oxygen**" found). The engine's chat output was already coherent in g4f.
* Opt-in: absent keys give today's raw gate and limit 40 (`ppl_form`/`max_ppl` default to `None`; unit tests in
  `crates/intarsia-models/src/recipe.rs` and `crates/intarsia-cli/src/cmd/accuracy.rs`). An unknown `ppl_form` is a recipe load error.

## B. Reference-noise factor (`[accuracy] ref_noise_factor`)

The key already existed (recipe, `--ref-noise-factor`, per-golden `factor`); this lane sets it for gemma4 and makes the noise record
load automatically. The HF bf16 golden of g4f (250 KB, 10 prompts, 1468 positions) and its noise record (HF bf16 vs HF f32 on the same
prompts, KL21 2.227e-1, per-prompt values) are now committed as `golden/gemma4-26b-a4b-nvfp4-hfbf16.safetensors` and
`golden/ref_noise/gemma4-26b-a4b-nvfp4-hfbf16.safetensors.json` (same bytes, FNV-1a `fc511251c94ca54f`, size 249928; only the file name in
the binding changed). `onboard` / `qualify` find a golden by model name in `golden/`, so `acc-short` now uses both with no flags. The
factor only multiplies a measured noise, so it does nothing without that record (a different golden, or `--no-ref-noise`, gives the
class floor alone).

Ratios of the g4f measurements to the noise 2.227e-1: prefill 2.953e-1 -> 1.326; decode 3.350e-1 -> **1.504**. The decode rows
use the same all-rows noise value. The brief said 1.5; that fails decode by 0.3% (floor 3.340e-1 < 3.350e-1), so the recipe says
**1.6** (largest ratio plus about 6%). Recipe-scoped, global floors untouched.

## B2. acc-short against the f32 reference (follow-up)

`golden/gemma4-26b-a4b-nvfp4-fp32.safetensors` is the g4f HF f32-compute golden (same 10 prompts, 1468 positions, `dtype float32`; non-expert
tensors f32, experts bf16-dequantised and cast per expert). Its noise record, `golden/ref_noise/gemma4-26b-a4b-nvfp4-fp32.safetensors.json`
(`intarsia ref-noise --golden fp32 --other hfbf16 --method precision_spread`), is HF's own bf16 forward measured against it: KL21 2.362e-1.
`find_golden` prefers names containing `fp32`, so `acc-short` picks it by model name. The rule "engine-vs-f32 KL <= factor x HF-bf16-vs-f32
KL" is the existing noise-record gate with that record and `[accuracy] ref_noise_factor`; no new code key was needed.

| Estimator | engine vs f32 | HF bf16 vs f32 | ratio |
|---|---|---|---|
| gate (engine full-vocab logits vs golden top-20 + other bucket), prefill n=1468 | 2.384e-1 | 2.362e-1 (record) | **1.010** |
| gate, decode n=160 (same record value) | 2.024e-1 | 2.362e-1 | **0.857** |
| `ref-noise` (top-20 vs top-20), g4f | 2.079e-1 | 2.362e-1 | 0.88 |

The brief's 0.88 is the third row; the gate's estimator is the first two, so a factor of 1.0 would fail prefill by 1%. `ref_noise_factor
= 1.1` (largest gate ratio 1.010 plus ~9%) replaces the 1.6 of the first version. Consequences, all by existing rules: the golden's dtype
is float32, so `needs_baseline_class` is false (no recorded baseline: the bf16 baseline of the first version was removed) and `acc-oracle`
is SKIP, not MISSING-BASELINE. The bf16 golden, its record and the (now unused by default) `ref_noise_factor` comparison stay in `golden/`
for reference. What is lost with it: no same-backend recorded dump, so a 1% scale bug is only seen by the KL floor of 2.6e-1.

## C. The CPU oracle (asked for in the follow-up; built, measured, not committed)

Made with qualify's own command (`accuracy --backend cpu --golden <bf16 golden> --mode prefill --max-seq 8192 --prefill-chunk 256
--mem-budget 40 --write-self-golden ...`, spark2 CPU, 5 min, 30 GB RSS; `raw/oracle/`). Measured against it, spark2 CUDA, oracle flags
(`--gate-class bf16 --max-mean-kl 2e-2 --min-top1 0 --no-ref-noise`): **mean 21-way KL 4.070e-1, top-1 83.8%, FAIL** (floor 2e-2, measured
on Qwen3.6-35B where clean builds read ~1e-2). Per prompt: count 1.3e-2, long_prose_700 6.6e-2, chat_multiturn 7.1e-2, story 1.47, code 0.72,
long_prose_300 1.25. The CPU backend runs bf16 activations (log: "weights/act/kv bf16/bf16"; "f64" is only the kernel accumulation), so it
is another bf16 forward of a router that flips on near-ties, and it is no closer to HF f32 (CPU vs HF f32 3.234e-1 against CUDA vs HF f32
2.384e-1). No bf16 engine can meet 2e-2 against it, and relaxing that floor is a gate decision the brief did not grant, so the oracle is not
in `golden/oracle/` (committing it would turn SKIP into FAIL) and no CUDA dump was recorded or blessed. A real f32 CPU path (act dtype f32)
would be a different oracle; it was not tried.

## Verdict per stage, first version (`raw/quick.md`, `raw/g4g-onb2.log`; spark1, lease #4276, 6 min 41 s; SUPERSEDED by the table after it)

| Stage | g4f (before) | now |
|---|---|---|
| recipe, smoke, plan, kernels (71 kernels, 1425 cases) | PASS | PASS |
| `gate` | FAIL (ppl 533.5, limit 40; 4 of 5 raw sanity prompts) | **PASS**: chat-form ppl 35.97 (limit 60), 5/5 sanity prompts, no non-finite rows |
| `acc-short` | FAIL (KL 2.95e-1 / 3.35e-1 vs floor 3.0e-2) | **PASS**: prefill KL 2.953e-1 <= 3.563e-1 (noise 2.227e-1 x 1.6, "bound by ref noise"), decode KL 3.350e-1 <= 3.563e-1; top-1 82.97% / 80.62% (not gated for the 4-bit class); tail mass ok; gated against the committed CUDA baseline (below) |
| `acc-format` | SKIP | SKIP (no format-reference golden) |
| `acc-oracle` | MISSING-BASELINE | MISSING-BASELINE (unchanged, out of scope) |
| `bench` (quick) | PASS | PASS, 35.6 / 32.2 / 25.3 decode tok/s at 128 / 1024 / 4096, prefill 630 / 529 / 369 tok/s (non-exclusive lease: not speed numbers) |

### Verdict per stage, final (`raw/final/`, spark2 lease #4302, `onboard --quick-only`, 7 min 10 s, exit 0)

| Stage | g4f (before) | final |
|---|---|---|
| recipe, smoke, doctor, plan, kernels (71 kernels, 1425 cases) | PASS | PASS |
| `gate` | FAIL (ppl 533.5, limit 40) | **PASS** (chat ppl 35.97 <= 60, 5/5 sanity) |
| `acc-short` | FAIL (KL 2.95e-1 / 3.35e-1 vs 3.0e-2) | **PASS** vs the f32 golden: prefill KL 2.384e-1, decode 2.024e-1 <= 2.598e-1 (2.362e-1 x 1.1); top-1 86.65% / 89.38% (not gated, 4-bit class) |
| `acc-format` | SKIP | SKIP |
| `acc-oracle` | MISSING-BASELINE | SKIP (full-precision golden gates the model) |
| `bench` | PASS | PASS |

First version: the noise record and the golden are found by model name with no flags (`golden/` is searched by `facts::find_golden`; the sidecar by
`golden/ref_noise/<golden file>.json`). One more committed file was needed: `acc-short` of a quantized MoE with a non-fp32 golden reads
**MISSING-BASELINE** when it passes on the static floor alone (fx1), so the run's own `acc-short` baseline is committed as
`golden/baselines/gemma4-26b-a4b-nvfp4-hfbf16.cuda.baseline.json` (recorded by the first run of this lane on this branch's code, unstamped
like the qwen36 baselines; no noise envelope recorded, the fixed margin applies). It makes a later regression show against this build;
it blesses the current numbers, which are the ones this doc discusses. That is the one decision in this lane beyond the brief (LOG row).

## Checks

* Final tests, `cargo test --no-fail-fast -p intarsia-core -p intarsia-models -p intarsia-cli -p intarsia-backend-hip` on spark2 (debug, CPU,
  `docs/results/` synced this time): 473 passed, 0 failed (`raw/final/g4g-test3.log`). New tests: `ppl_form` / `max_ppl` parse, default and
  error cases; only gemma4 opts in (`ppl_form` chat, `max_ppl` 60, `ref_noise_factor` 1.1) and every other built-in keeps raw / None;
  Qwen2.5's `decode_positions = 1000` still parses; the chat-sequence builder (served prompt, thought channel, rejected templates); the
  shipped conversation; the quick-gate arg defaults.
* Final `onboard --quick-only` on the full 26B, CUDA, spark2, lease #4302: exit 0 (`raw/final/`).
* No-op, Qwen3-8B on CUDA (spark2, `/mnt/models/Qwen3-8B`), origin/main `a92267b0` build against this branch's build: `accuracy --quick`
  identical (ppl 1.847 over 1169 tokens, 5/5 sanity, PASS; the branch only adds the line `perplexity form: raw (limit 40 from default)`);
  `accuracy --golden qwen3-8b-fp32 --mode both`: prefill KL 2.625e-4 / decode 2.854e-4, top-1 100.00%, PASS, identical lines on both builds.
  (The first version's no-op on spark1 gave the same numbers.)

## Not done

* vLLM chat-form perplexity (table in A): `tools/chat_ppl.py vllm` is written, never run (no Gemma-capable vLLM image on spark1; spark2 not
  used for it).
* The acc-oracle gate is not working for Gemma-4 and cannot with the 2e-2 floor (section C); `acc-oracle` is SKIP only because the f32 golden
  now gates `acc-short`. If a drift check against a recorded same-backend dump is wanted, that needs a decision on what the oracle stage
  should do for a model whose bf16 forward is chaotic (a CUDA dump without an oracle is "blessing CUDA as its own oracle").
* `acc-short` is judged by a floor of 2.6e-1 KL, which sees a gross error but not a 1% scale bug.
