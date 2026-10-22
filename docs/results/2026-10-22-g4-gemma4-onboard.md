# g4: Gemma-4-26B-A4B-NVFP4 through `intarsia onboard` (no existing recipe)

Status: IN PROGRESS (this file is rewritten at the end of the lane; the timeline below is appended as it happens).
Lane g4, branch `lane-g4/gemma4`, fleetq owner `claude-f31ccf91-g4`. Hard stop 11:30 EDT. T0 = 09:11:56 EDT (weights already on spark1:
`~/.cache/huggingface/hub/models--nvidia--Gemma-4-26B-A4B-NVFP4`, 2 shards, 18.8 GB, nothing downloaded).

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
