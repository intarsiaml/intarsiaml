# NAB2: the LFM2 short-conv block, a new block type from weights to onboard PASS (2026-10-27)

Lane `lane-nab2/lfm2-shortconv`. Question: is the hard half of requirement 1 true, "a new model comes out, basic test in 1-2 h, in-depth in 2-4 h, adding
new modules is easy", on a model that needs a block the library does not have? Model: `LiquidAI/LFM2.5-1.2B-Instruct` (`Lfm2ForCausalLM`, dense bf16, 2.35 GB, 16
layers: 10 gated short-conv layers, 6 GQA layers with QK-norm, SwiGLU FFN), downloaded with approval to the HF cache at 09:21 EDT (stopwatch start). Hardware: endor,
Apple M5 Max, every run through fleetq. Raw files: `docs/results/raw-nab2/`.

**Result: yes, with caveats stated below.** Basic PASS 22 minutes after the weights landed, in-depth PASS 32 minutes, all merge gates green. The targets are 1-2 h and 2-4 h.

## What was built

* `Op::ShortConv` (appended to `Op::ALL`): `bcx [T,3C]` (the `in_proj` output, rows `[B | C | x]`), taps `[C,K]` -> `y [T,C] = C * causalconv(B * x)`, depthwise,
  no activation, no bias, plus the conv state `[C,K-1]` f32 in place (the last `K-1` values of `B * x`). Reference: `crates/intarsia-core/src/gdn.rs`
  (`short_conv`, sharing `causal_conv` with the GDN conv; the SiLU is the only difference), cpu dispatch in `gdn_ref.rs`, admission sweep in `admission.rs`.
* `Block::Conv` (`crates/intarsia-core/src/model/short_conv.rs`): solo/chunk, packed-segment and batched-row entry points; the state is a `SlotSpec::Recurrent` slot, the
  same mechanism as GatedDeltaNet's, so snapshots, restore, reset and speculative rollback needed no new code.
* Metal: `kernels/metal/gdn.metal` `gdn_conv_t<K, SHORT>` (the GDN conv kernel with a template switch: gated read of `B * x`, activation off, `C` gate on the
  output) and `short_conv_state` for the multi-tile state pass; kernels `short_conv_bf16_k2..k8`, launcher in `crates/intarsia-backend-metal/src/gdn.rs`.
* Recipe `recipes/lfm2.toml` (scaffolded by `intarsia recipe new`, 10 TODOs, completed by hand): `[layer.conv]`, `[config]` renames (`norm_eps`, `tie_embedding`,
  `conv_L_cache`), final norm = `model.embedding_norm`, QK-norm = `q_layernorm`/`k_layernorm`, `w1/w3/w2` FFN names. `ModelDims` gains `short_conv` and the FFN width rule
  `block_auto_adjust_ff_dim` (`int(2*ff/3)`, `block_ffn_dim_multiplier`, `block_multiple_of`, as HF `Lfm2MLP`).
* Fixes the model needed: chat templates with `{% generation %}` (LFM2's) did not compile in minijinja (the tags are comments now); `recipe new` coverage ignored the
  recipe's `tie_embeddings` key and had no `conv` arm; `kernels test --model` read config keys with the HF defaults and died on `rms_norm_eps`.
* CUDA and HIP: no kernel. `ShortConv` is in cuda `NOT_ON_CUDA` and hip `UNIMPLEMENTED`; a load fails with `NoKernel` naming `short_conv`
  (`tests/lfm2.rs`, a wrapped backend; CUDA/HIP themselves were not run, no box in this lane).
* Tests: `tests/lfm2.rs` (10 tests on a tiny random HF `Lfm2ForCausalLM` fixture, `tools/gen_lfm2_fixture.py`: dims, layer kinds, prefill, chunked prefill, decode, snapshot restore,
  failed-pass reset, batched == solo, refusal naming the op, CUDA/HIP tables), `gdn.rs` unit test (literal HF formula and chunk invariance), Metal `short_conv_chunked_prefill_equals_one_call_and_the_reference`,
  cpu admission over `ShortConv`, `real_lfm2_batched_decode_is_bitwise_the_solo_step`, docs tables (`recipes.md`, `tuning.md`).

## Stage timing (EDT, from `date` and the tool timestamps; stopwatch 09:21)

| Stage | Start | End | Wall | Target |
|---|---|---|---|---|
| Read the NAB drill doc, `adding-a-model`, GDN block and kernels, HF `modeling_lfm2.py` | 09:21 | 09:27 | 6 min | |
| `recipe new` (not covered, 10 TODOs) | 09:27 | 09:28 | 1 min | |
| Op + f64 reference + cpu dispatch + sweep, `Block::Conv`, recipe, FFN width, chat-template fix: **first correct greedy text (cpu)** | 09:28 | 09:33 | 5 min | |
| Metal kernel from the GDN conv template, **ADMITTED** (20 cases) | 09:33 | 09:34 | 1 min | |
| 6 planted mutants, all REJECTED (script `mutants.py`, 6 rebuilds) | 09:34 | 09:36 | 2 min | |
| `onboard` attempt 1: "recipe needs a human" (coverage bugs); attempt 2: `kernels` stage dies on `rms_norm_eps`; fixes | 09:36 | 09:40 | 4 min | |
| `onboard` attempt 3: **quick PASS** (1 m 47 s; whole run 2 m 33 s with in-depth, long goldens skipped) | 09:40 | 09:43 | 3 min | **basic 1-2 h: reached at +22 min** |
| Long + 4k fp32 goldens by hand (HF on CPU, 41 s and ~9 min incl. the bf16 spread pass) | 09:43 | 09:52 | 9 min | |
| `onboard --redo in-depth` with all goldens: **in-depth PASS** (11 pass, 0 fail, 4 skip, 39 s) | 09:52 | 09:53 | 1 min | **in-depth 2-4 h: reached at +32 min** |
| localeval speed sweep (128 / 1k / 4k) and MMLU `--limit 10` | 09:54 | 09:57 | 3 min | |
| MMLU sanity check against HF (114 prompts, engine vs HF bf16) | 09:58 | 10:03 | 5 min | |
| Gates: `plan show` A/B, chunk invariance, batched == solo on the real model | 10:03 | 10:06 | 3 min | |
| `cargo test --workspace --release` (1074 passed, 1 failed, fixed, re-run green) | 10:06 | 10:12 | ~6 min | |
| Docs, LOG row, PR, CI | | | see PR | |

Early stage boundaries (before 09:33) are estimated from the order of tool calls; later ones are clocked. The rows add up to less than the wall clock because of unlogged reading.

## Gates

| Gate | Result |
|---|---|
| `onboard` PASS on Metal, quick and in-depth, auto-made short golden | PASS. Quick: prefill top-1 99.55 % (KL 5.88e-4), decode top-1 99.22 % (KL 6.97e-4) vs the HF fp32 golden; the calibrated floor (#203) applied. In-depth with long and 4k goldens: long 99.31 % / 98.41 %, 4k 98.80 % / 99.48 % (KL 2e-4 to 6e-4); skipped: acc-format (bf16), acc-oracle, ab, tp (no input given) |
| `kernels test` ShortConv admitted, >= 3 planted mutants caught | ADMITTED (20 cases, max abs 5.1e-5, rel_l2 2.2e-6); 6 of 6 mutants REJECTED: SiLU left on (rel_l2 0.85), `C` gate taken from `B`, product `B*C` instead of `B*x`, reversed taps, state from `x` instead of `B*x`, multi-tile state pass skipped (12 of 20 cases fail) |
| batched == solo for the new block | PASS on the real model, Metal: 4550 (row, step) pairs bitwise identical, N = 2..8, 130 steps each, mixed lengths, plus decode beside prefill chunks; also on the cpu tiny fixture |
| chunk-size invariance | PASS: 1711-token prompt, 64 greedy ids identical for default (256), `--prefill-chunk 512` and `--prefill-chunk 100` (sha 82ab9ccd... all three) |
| `plan show` byte-identical, Qwen3-8B and Qwen3.5-4B, Metal | PASS: `cmp` equal against the binary built from unmodified main (sha-256 prefixes 8e3bdf8d / e95239f9) |
| CUDA and HIP refuse at load naming the op | PASS in the unit test (`NoKernel` containing `short_conv`, `no kernel`); not run on a CUDA or HIP box |
| Mac `cargo test --workspace` | PASS after one fix: 1074 passed, 26 ignored; the one failure was `docs_env_vars` (`INTARSIA_LFM2_DIR` had no row in `docs/tuning.md`), row added, target re-run green |
| CI | see the PR |

## Speed and accuracy (Metal, localeval 0.4.0, `intarsia serve`, max-seq 6144, 3 reps, 128 forced tokens)

| prompt tok | TTFT s | prefill tok/s | marginal prefill | decode tok/s |
|---:|---:|---:|---:|---:|
| 126 | 0.01 | 8418 (overhead-bound) | - | 196.2 |
| 1026 | 0.06 | 16327 | 18750 | 195.5 |
| 4091 | 0.25 | 16579 | 16658 | 188.8 |

Run `20261027-095421-adhoc-nab2-lfm2-metal` (unanimis f23aae2b). `onboard`'s own bench read decode 217 / 194 / 182 tok/s at 128 / 1k / 4k (83 % / 74 % / 70 % of the 262 tok/s byte floor)
(a `generate` of a 1711-token prompt prefilled at 12 k tok/s). Not tuned: the decode is the generic path (eager, one conv launch per layer, no fusion of the conv with the projections).

MMLU `--limit 10` (570 questions, 3 min): **0.314** (run `20261027-095430-adhoc-nab2-lfm2-metal-mmlu10`, unanimis 50edfbc3). That looks low for a 1.2B instruct model, so it was checked: the
server falls back from `response_format` to a prompt, and the model answers `B` for 455 of 570 questions, which is the model's behaviour in this format, not the engine's. Same 114 prompts (2 per subject, chat
template, "answer with only the letter", greedy 6 tokens) through HF transformers bf16 on CPU and through the engine: **96.5 % identical answers** (110/114; accuracy 39.5 % engine, 41.2 % HF, both
dominated by `D`). The engine is as good as the reference; the score is not a quality claim about the model.

## Friction list, ranked by what it cost here

1. **`recipe new`'s coverage and the config-key renames live in code that does not read the recipe** (about 7 min and two failed onboards, one of them 69 s in). Coverage had no arm for a new block
   and ignored `[config] tie_embeddings`, so a fully covered model was reported "needs a human" with exit 3; `kernels test --model` called `ConfigKeys::default()` and died on `rms_norm_eps`.
   Fix: one `Recipe::dims(&cfg)` and one `Recipe::tensor_templates()` that every command uses, and a test that greps for `ConfigKeys::default()` outside tests.
2. **Adding an op touches about 12 sites, five of them hand-counted** (`Op::ALL` length, the "appended after X" asserts counted from the end, `NOT_ON_CUDA` length, `PER_ROW_OPS` length,
   Metal `MAIN_KERNEL_LIST` pin, plus the exhaustive matches in shape/check/sweep/label/case). The compiler finds the matches, nothing finds a stale count until the test runs. Fix: derive these from
   one list (`Op::ALL` as a slice, the order test as `ALL[i]` pairs, kernel pins from the registry).
3. **A block is wired in six places across `model.rs` and `model/batch.rs`** (load, run, trace name, tail check, `batch_capable`, `has_hybrid_blocks`, and the workspace allocation, which exists twice, once for the
   decode twin, with a boolean per block kind). Fix: a `Block` trait (`load`, `ws_bytes`, `run_*`, `needs_recurrent_batch`) so a block is one file plus one registry line.
4. **The long and 4k goldens are still manual** (9 min of HF on CPU for 1.2B; the previous lane lost 22 min on a 2B). Until they exist the in-depth tier skips `acc-long` / `acc-4k` and still says PASS.
   Fix: `onboard` makes them when `--golden-dir` is writable and the model is under 3 B parameters, or runs `golden.py --device mps`.
5. **Mutant testing of a kernel is a script you write** (`mutants.py`, 60 lines, 6 rebuilds). Fix: `kernels test --mutants FILE` that applies declared patches next to the kernel and requires REJECTED.
6. **Chat template compatibility is found by loading** (`{% generation %}` was a hard load error in minijinja). Fix: `recipe new` renders the template once with a two-turn conversation and reports the error as a TODO.
7. **Hidden config arithmetic** (`block_auto_adjust_ff_dim`: the config says 12288, the weights are 8192 wide) is only visible in the HF source. It was read first and handled before the first run; if skipped the loader's shape check
   would fail naming the tensor, which is correct but late. Fix: `recipe new` prints the first MLP tensor shape next to the config's `intermediate_size` when they differ.
8. **`fleetq guard` hook fires on heredoc text** (the word `serve` inside a Python file written via heredoc blocked the call, twice). Fix: match the command word, not the command text.

## What this does not show

* The previous lane (NAB) had already written down the short-conv semantics and the friction list; step 1 here started from that, and the GDN conv kernel and recurrent-slot machinery already existed. A block
  with no sibling in the tree (an attention variant, an SSM scan) would take longer; the 6-place wiring (item 3) would be the same.
* 1.2B dense bf16: no quantisation format to add, goldens in minutes. The 24B-A2B MoE of the same family (MLX 4/6/8-bit on endor) also needs the MoE block with 64 routed experts and no shared expert, which
  `recipes/qwen3_5_moe.toml` does not cover; not attempted.
* CUDA and HIP run nothing here: they refuse. Porting the kernel is the Metal change plus the same template in `kernels/cuda` and `kernels/hip`.
* Decode speed is the unfused generic path; no tuning, no comparison against another engine (mlx-lm, llama.cpp), no GSM8K or IFEval.
