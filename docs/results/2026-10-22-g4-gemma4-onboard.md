# g4: Gemma-4-26B-A4B-NVFP4 through `intarsia onboard` (no existing recipe)

Lane g4, branch `lane-g4/gemma4`, fleetq owner `claude-f31ccf91-g4`. T0 = 09:11:56 EDT 2026-10-22 (weights already on spark1:
`~/.cache/huggingface/hub/models--nvidia--Gemma-4-26B-A4B-NVFP4`, 2 shards, 18.8 GB). Hard stop 11:30.

## Verdict

**Requirement 1 is met for the code, not yet for the clock.** A model with no recipe went from "`recipe new` exits 3 with 10 TODOs" to a
recipe-covered, loading, generating, admission-clean, HF-matching model in about 1 h 10 min of wall clock (09:12 to 10:23), by
writing new engine code (one layer block, 4 op attributes, 1 op, 9 CUDA kernels, a loader). The 2 h quick / 4 h full targets are
not measurable on the full 26B: **no GPU lease big enough for it (>= 20 GB) was grantable before the hard stop**, so there is no
"weights present to quick numbers" number for the real model. What was run instead, all on real weights:

| Check | Result |
|---|---|
| `recipe new` before / `onboard` recipe stage after | 10 TODOs, exit 3 / `builtin:gemma4 covers the model (41/41 tensor templates resolve, quant nvfp4)`, PASS in 0 s |
| Tiny random 6-layer Gemma-4 (NVFP4 experts, window 8) vs HF transformers 5.6.2, CPU backend | top-1 5/5, 21/21, 39/40 vs HF-f32; mean row rel-L2 to HF-f32 1.7e-2 / 2.1e-2 / 1.1e-1, where HF-bf16 itself is 5.4e-2 / 5.5e-2 / 1.2e-1 from HF-f32 (`tests/gemma4_tiny.rs`, runs in CI) |
| 2-layer slice of the REAL weights (layer 0 sliding + layer 5 global, 3.2 GB) vs HF, CUDA, prefill, 248 positions | final norm scaled 2^-5 (see below): top-1 95.2%, mean 21-way KL 3.5e-3, gate PASS; CPU backend 95.2% / 3.8e-3. Multi-chunk prefill (`--prefill-chunk 31`) + decode mode: prefill 96.0% / 3.2e-3, decode 96.9% / 3.2e-3, PASS |
| `intarsia onboard --quick-only` on that slice, CUDA, spark2 | recipe PASS (0 s), smoke PASS (8 s), kernels admission PASS (71 kernels, 1425 cases, 0 rejected), quick tier 3 min 34 s; verdict FAIL only on the perplexity / repeat-sanity gates, which a 2-layer model cannot pass (`raw/onboard-2L-slice/`) |
| 6-layer real-weight slice, CUDA vs CPU `generate` (12 greedy ids) | identical ids |
| `intarsia kernels test --backend cuda` | 71 kernels, 1425 cases (full sweep, spark2), 505 (quick, spark1), 0 rejected |
| Qwen3-8B no-op (CUDA, first 12 layers, 256 greedy ids + `plan show`) vs merge-base `acba5987` | plan identical, 256 ids identical (`raw/qwen3-noop/`). Deviation: 12 layers, not the full model, because only <= 12 GB leases were obtainable |

Why the final norm was scaled in the slice comparison: the real `model.language_model.norm.weight` has mean 29 (max 588), so after
only two layers the pre-softcap logits have std 140 and 24% sit at the 30.0 soft-cap; the specified comparison is then vacuous
(top-1 2.0%, KL 5.5e-8 PASS, meaningless). Scaling that one weight by 2^-5 (exact in bf16) in BOTH the HF and the engine slice
brings the logits to std 4.4 and makes the comparison real. Not run: the full 26B against any reference (an HF reference needs
~52 GB dequantised, vLLM on the Sparks cannot be asked: it serves other models), sliding-window behaviour at the real window of 1024
against HF (the tiny model covers window 8; the kernel tests cover window 1024 on 1100 keys).

## What was added

* `recipes/gemma4.toml`: one block per layer (`gemma`), prefix, `norm_offset = 0.0`, tied head, `[defaults] w4a4_as_w4a16 = true`.
* `crates/intarsia-core/src/model/gemma.rs` (new): the Gemma-4 decoder layer (sandwich norms, per-layer-type attention with
  K = V on global layers, parallel dense GeGLU MLP + 128-expert MoE, `layer_scalar`), `Block::Gemma`, `GemmaDims` from `config.json`
  (`rope_parameters` per layer type, `global_head_dim`, soft-cap, embedding scale), final soft-cap on host logits.
* Folded at load time, no op: `router.scale * hidden^-0.5` into the router norm weight, `per_expert_scale` into each expert's
  down-projection global scale (`MoeWeights::load_gemma`).
* Op surface (all behind the new `.ext()` kernel flag, so a backend that ignores an attribute cannot run it silently wrong):
  `Op::GeluGlu`, `OpAttrs::window`, `rope_prop`, `gelu` (MoeExperts), and `scale` on `Add` / `Embedding`.
* CPU reference + fast kernels for all of them; CUDA: `gelu_glu`, `rope_prop`, `add_scaled`, `embedding_scaled`, attention with
  window and head_dim up to 512 (SIMT, plus a split-KV window variant for D <= 256), MoE GELU epilogues in 4 kernels + the generic one.
  Kernels-test cases for each. Not done: HIP and Metal (they keep refusing Gemma through `.ext()`), fused variants, captured CUDA
  graphs for Gemma layers (eager decode), tensor parallelism, speculative decoding, the vision tower.
* `recipe new` / `onboard`: reads `hidden_activation`, knows the `gemma` block's tensors (`coverage`), and a recipe can now accept a
  W4A4-declared NVFP4 checkpoint (`[defaults] w4a4_as_w4a16`), without which the loader and `onboard` both refuse it.

## What the architecture needs versus what the engine has

Source of truth: `transformers 5.6.2` `modeling_gemma4.py` (no Gemma-4 knowledge exists in the repo; the wheel was the only spec) and the
checkpoint's `config.json`. The 10 `TODO(recipe-new)` items of lane `on` (`recipe-new-baseline.txt`) are a subset of the gap:

| # | Feature | Where it shows up | Engine had | TODO of `recipe new` |
|---|---|---|---|---|
| 1 | RMSNorm is `x * w` (no `+1`) | `Gemma4RMSNorm` | `norm_offset` | #1 (placeholder 1.0 was WRONG; real value 0.0) |
| 2 | Sliding-window attention (1024) on 25 of 30 layers, full on 5 | `layer_types` | none | #2, #3 |
| 3 | Per-layer-type head_dim (256 / `global_head_dim` 512), KV heads (8 / `num_global_key_value_heads` 2) | attention | one `head_dim`, one `n_kv_heads` for the model | none |
| 4 | `attention_k_eq_v`: global layers have no `v_proj`; V = unit-RMSNorm(k_proj x) | attention | none | none |
| 5 | `v_norm` (RMSNorm without scale) on every layer, `q_norm`/`k_norm` plain | attention | `QkNorm` with weight (ones weight works) | none |
| 6 | Attention scale 1.0 (no 1/sqrt(d)) | `self.scaling = 1.0` | `scale` attr | none |
| 7 | "proportional" RoPE on global layers (theta 1e6, partial 0.25 of 512: pairs (j, j+256) j<64, exponent 2j/512, zero tail) | rope | `rotary_dim` partial rope (different pairing and exponent: silently wrong) | none (`recipe new` flags rope types other than `default`, but the `rope_parameters` per-layer-type dict was not read) |
| 8 | Dense GeGLU MLP and a 128-expert top-8 MoE in PARALLEL in every layer, outputs summed | `enable_moe_block` | FFN or MoE, serial | #4, #6 (1542 unclaimed tensors) |
| 9 | Router: unit RMSNorm of the raw residual x `scale` x hidden^-0.5 -> `proj` -> softmax over all experts -> top-8 -> renorm -> x `per_expert_scale`; no shared expert | `Gemma4TextRouter` | qwen-style router | none |
| 10 | Sandwich norms: `post_attention_layernorm`, `pre_feedforward_layernorm{,_2}`, `post_feedforward_layernorm{,_1,_2}` | decoder layer | pre-norms only | #6 (partly) |
| 11 | `layer_scalar` (30 distinct values 0.07..0.82) multiplies the whole hidden state after each layer | decoder layer | none | none |
| 12 | Embedding scale sqrt(2816) (rounded to bf16 first: 53.0) | `Gemma4TextScaledWordEmbedding` | none | none |
| 13 | Final logit soft-cap 30 (`30 tanh(x/30)`) | `final_logit_softcapping` | none | none |
| 14 | GELU-tanh (`gelu_pytorch_tanh`) in the dense MLP and in the experts | `hidden_activation` | silu only (SiLU hard-coded in 5 CUDA MoE kernels) | `recipe new` read `hidden_act`, missed `hidden_activation`, assumed silu |
| 15 | Tied embeddings | `tie_word_embeddings` | yes | handled |
| 16 | NVFP4 declared W4A4 (`input_scale` per expert); only the per-expert gate/up/down are quantised, attention, dense MLP, router, embeddings are bf16 | `hf_quant_config.json` | W4A16 NVFP4 | #7, #8, #9, #10 (the same issue four times) |
| 17 | `embed_vision.*`, `vision_tower.*` | multimodal wrapper | ignored | #5 |

The 10 TODOs map to: #1 -> row 1; #2, #3 -> rows 2-4 (one new attention kind); #4, #6 -> rows 8-11; #5 -> row 17 (vision only: ignore); #7-#10 -> row 16.
Rows 3, 4, 5, 6, 7, 9, 11, 12, 13, 14 have no TODO at all.

Silent errors in the scaffold (it exited 3 as designed, but the file it wrote is wrong beyond the markers; `recipe-new-baseline-scaffold.toml`):
`gated = true` copied from the closest recipe with only a "verify" comment, `[layer.ffn]` pointed at `experts.0`, `full_attention` layers mapped to `["ffn"]` only,
`act = "silu_glu"`, `norm_offset = 1.0` (placeholder from another family).

## Timeline (EDT, 2026-10-22)

| Time | Event |
|---|---|
| 09:11:56 | T0. Task start; weights present on spark1. |
| 09:13 | queued an endor `cargo build` (fleetq #4143): 6th in line, ~14 min expected; cancelled at 09:19 in favour of building on spark1 inside one long non-exclusive lease (#4154, queued behind exclusive #4142). |
| 09:14 | scp'd config/index; read config. 5 min later: `intarsia recipe new` (borrowed another lane's release binary) -> exit 1, 10 TODOs. |
| 09:16-09:18 | fetched the transformers 5.6.2 wheel to read `modeling_gemma4.py` (the only spec). |
| 09:20 | feature map above; read engine (`model.rs`, ops, kernels). Gate checks: CUDA attention supports head_dim 64/128/256 only (`kMaxD = 256`, `flash_ok`, `splitkv_ok`), so global layers (512) need a new kernel; SiLU is hard-coded in all CUDA MoE kernels; the 30 `layer_scalar` values are not 1.0, so the op is needed. |
| 09:22 | op interface committed (`GeluGlu`, `window` / `rope_prop` / `gelu` attrs, `scale` on Add/Embedding, `.ext()` kernel gate); three worker agents started (CPU reference, CUDA kernels, HF tiny-model golden). |
| 09:19-09:35 | **Queue friction.** endor: 13 jobs deep behind a stale (74 min, no heartbeat) job of another owner, so the Mac cannot build. spark1: our long non-exclusive lease (#4154) and a backfill-sized one (#4166/#4168, `--expect 14`) both sat behind an exclusive job (#4142) that itself waited for a 19 GB network copy (#4162, ours: the HF-golden worker) and then for 45 GB that the box did not have free (37 GB reported); the `--expect` backfill that `fleetq status` advertises did not admit our job. Moved the dev builds to spark2 (lease #4169, first in line behind one exclusive job). Lost about 20 minutes of compile feedback. |
| 09:25 | tiny random-weight Gemma-4 + HF transformers reference (bf16 and f32) produced by a worker in 12 min; the HF reference found two things the modeling file does not make obvious: `inv_freq` must stay f32 (a plain `.to(bfloat16)` on the model corrupts it) and random weights make bf16 vs f32 diverge by router flips (so the engine is judged against HF's own bf16/f32 spread). |
| 09:38 | First compile of the whole Gemma-4 block code (`cargo check`, spark2): clean on the first try after the CPU-reference ops landed. |
| 09:41 | **Tiny model, CPU backend vs HF transformers 5.6.2** (`crates/intarsia-models/tests/gemma4_tiny.rs`, 6 layers, 8 experts NVFP4, window 8): top-1 agreement with HF-f32 5/5, 21/21, 39/40; mean per-row relative L2 to HF-f32 1.7e-2 / 2.1e-2 / 1.1e-1, where HF's own bf16 forward is 5.4e-2 / 5.5e-2 / 1.2e-1 from its f32 forward. The engine's bf16 forward is therefore as close to HF-f32 as HF-bf16 is. First failure on the way: the loader refused bf16 experts (the engine only loads NVFP4 experts), so the tiny checkpoint was regenerated with ModelOpt NVFP4 experts (+12 min worker time). |
| 09:44 | Weights copied spark1 -> spark2 (18.8 GB, 2 min 42 s at 116 MB/s) because spark1 had no lease for us; this copy is not part of an onboarding clock but of the queue's. |
| 09:53 | **First CUDA build of the whole branch** (spark2, `cargo build --release --features cuda`, 22 s with a warm cache). |
| 09:58 | `plan show` on a 6-layer slice of the REAL checkpoint on CUDA (spark2): loads in 5.9 s, resolves every op to a kernel on the first try (`embedding_scaled`, `rope_prop`, `attn_*_ext`, `moe_experts_decode_v2_gelu`, `gelu_glu`, `add_scaled`). |
| 10:00 | `generate` on that slice, CUDA and CPU reference: the same 12 greedy ids on both (`raw/gemma4-6L-gen-*.txt`). The text is garbage (a 6-of-30-layer model) and the top logits sit at exactly 30.0 (the soft-cap), which is not evidence either way. |
| 09:55-10:05 | **GPU model-size leases unobtainable.** Both Sparks were gated by another lane's chain (#4172 stale, #4174 multi-host, #4171 exclusive): `fleetq when` says 11:57 (spark1) / 13:27 (spark2) for a 20-26 GB lease. Only leases <= 12 GB are admitted (backfill). The full 26B-A4B (18.8 GB of weights) does not fit that, so the full-model onboard, the 4k goldens and the in-depth tier could not run in the lane's window. |

## Where the time went, and the friction (the main deliverable)

Engineering clock, 09:12 to 10:23, with the fleet queue excluded: spec reading 10 min; interface design + 3 parallel workers (CPU
reference, CUDA kernels, HF tiny model) 10 min to launch, 25-60 min to finish each; own layer block + loader + recipe 20 min; first
compile of everything 09:38 (clean after the CPU ops landed); first tiny-model match 09:41; real weights loading and generating on
CUDA 09:58; HF match on the real-weight slice 10:14; `onboard` stages on it 10:23. Fleet queue: roughly 40 of those minutes were
spent waiting for or re-arranging leases, and the full-model run never got one.

Top friction points, in order of cost:

1. **Fleet capacity, not the engine.** Two Sparks, ~15 lanes. A 26 GB non-exclusive lease needed an exclusive-job chain
   (#4171 -> #4172 stale -> #4174 two-host) to clear: estimated start 11:57 / 13:27 at 09:55, still ~11:05 at 10:25. `fleetq`'s
   backfill admits only a job that leaves the waiting job's memory free, so leases <= 12 GB went through at once and a 19 GB model
   never did. Everything that needs the model resident (HF golden, the 4k golden, in-depth tier, the quick tier on the real model)
   is blocked behind that. Fix for next time: reserve a model-sized slot for an onboarding lane up front (`fleetq reserve`), and
   keep a <= 12 GB slice workflow (below) so the engineering loop never waits.
2. **The recipe format cannot describe a new layer structure; no spec exists in the repo.** `recipe new` produced a scaffold that is
   wrong beyond its 10 TODOs (it kept `gated = true`, pointed `ffn` at `experts.0`, assumed silu because it read `hidden_act` and
   the key is `hidden_activation`, and silently dropped what it could not name: `layer_scalar`, the 5 extra norms, the router
   parameters, the per-layer-type rope dict, `attention_k_eq_v`, the embedding scale, the soft-cap). Only transformers'
   `modeling_gemma4.py` (fetched from the pinned wheel) told the truth. The fix here was a new Rust block, not recipe keys; the
   features that were "just a number" (scale, window, rope variant, activation) were cheap, the structure was not.
3. **No cheap reference for a big model.** An HF golden of the full 26B needs the NVFP4 experts dequantised to bf16 (~52 GB of RAM or
   disk) and `tools/golden.py` has no such path; vLLM on the fleet serves other models. What worked: slice the real checkpoint to 2
   layers (`tools/gemma4_slice/`, 3 GB, loads in 5 s), dequantise for HF, produce a golden in 7 s. The catch it exposed: the real
   final-norm weight makes a truncated model saturate the soft-cap, so a "PASS" is vacuous unless the norm is rescaled (see
   Verdict). The checkpoint's `NVFP4` W4A4 declaration was the other surprise: the engine's own policy refuses it, and nothing in the
   docs said how to waive it (now `w4a4_as_w4a16`).

Smaller: `cargo test -p intarsia-cli` overwrote the `--features cuda` binary in the same target dir (the next `accuracy` run said "built
without the cuda feature"); tiny-model "bf16 vs f32" disagrees through router flips unless judged against HF's own spread; HF
`inv_freq` must stay f32 when the model is cast to bf16; the Gemma tokenizer has no post-processor BOS (chat template adds it, a raw
`generate --prompt` does not).

## What would make the next model faster

* A `fleetq reserve` for onboarding with a model-sized memory claim, taken before the lane starts (item 1).
* `tools/slice_checkpoint.py` generalised (layers to keep, tensor-name regex, NVFP4 dequantise for HF) so every new model gets a
  <= 12 GB real-weight comparison within minutes, and `intarsia onboard --golden-from <slice golden>` already works with it.
* `recipe new` should (a) read `hidden_activation`, (b) list every per-layer tensor name it cannot place with its shape (it listed
  1542 experts but not `layer_scalar` / the five norms individually), (c) print the transformers modeling-file path of the pinned
  version, (d) treat a W4A4 declaration as a recipe decision, not a dead end.
* A tiny-model generator per architecture family from the HF config (`tools/gemma4_tiny.py` is 300 lines): it caught every structural
  bug before any GPU or big weight was touched, and doubles as a CI test.
* Op-attribute + `.ext()` gating made the CPU, CUDA and test work parallelisable across three agents with no integration surprises;
  keep that split (interface commit first, then CPU reference / kernels / reference data in parallel).

## Not done / open

* The real 26B has not run through `onboard`, the 4k and in-depth tiers, nor been compared to any reference at full depth: no lease.
* Real sliding window (1024) vs HF; speed (SIMT D=512 attention, eager decode, Gemma layers not graph-captured); HIP and Metal.
* `moe_intermediate_size` 704 is not a multiple of 128: the mma / w4a16 GELU MoE kernels do not apply to it, prefill with T >= 2
  uses the slow generic `moe_experts_gelu_bf16` (correct, untuned). The 2-layer slice's quick-tier speed numbers (~7000 tok/s prefill) are
  for 2 layers and mean nothing for the model.
