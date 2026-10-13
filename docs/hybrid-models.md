# Hybrid models: GatedDeltaNet + gated attention (Qwen3.5 / Qwen3.6)

Recipes: `recipes/qwen3_5.toml` (dense FFN), `recipes/qwen3_5_moe.toml` (MoE FFN). The decoder is generic: a recipe
sets `layer.pattern_by_layer = "layer_types"` and `[layer.by_type]` maps each config.json layer type to a block
pattern (`linear_attention = ["gdn", "moe"]`, `full_attention = ["attn", "moe"]`). Unknown layer types fail the load.

## Semantics (spec: HF `modeling_qwen3_5_moe.py`, checked by `tools/gen_hybrid_fixture.py` fixtures)

* RMSNorm is zero-centred: scale = `1 + w` (`model.norm_offset = 1.0`, op attr `w_offset`). The gated norm inside
  GatedDeltaNet uses the plain weight.
* Full attention: `q_proj` emits `[q | gate]` per head (`attn.gated = true`, op `gate_split`); qk-norm, partial rope
  (`rotary_dim = partial_rotary_factor * head_dim`, op attr `rotary_dim`; the tail of each head is untouched),
  attention, then `attn_out * sigmoid(gate)` (op `sigmoid_mul`) before `o_proj`.
* GatedDeltaNet: `in_proj_qkv -> gdn_conv (causal depthwise conv + SiLU, K-1 history) -> gdn_delta_rule`,
  then `rmsnorm(core) * silu(z)` and `out_proj`. Value head `h` reads key head `h / (Hv/Hk)`.
  Math: `core/src/gdn.rs` (f64 reference, doc comment lists the exact formulas).

## State

Each GatedDeltaNet layer owns one `SlotSpec::Recurrent` slot: f32 conv state `[C, K-1]`, then (256-byte aligned) the
delta-rule state `[Hv, dv, dk]`. Qwen3.6-35B-A3B: 30 layers x (8192*3*4 + 32*128*128*4) B = ~2.1 MB per layer, ~63 MB
per sequence.

Recurrent state cannot be shortened, so prefix reuse works through snapshots: the engine snapshots after each prefill
chunk and one token before the end of the prompt (`SequenceState::snapshot`, at most `RECURRENT_SNAPSHOTS` = 4 kept),
and reuses the longest prefix that is the held sequence itself, a snapshot, or empty (`restorable_len`). A failed
forward resets the sequence (the state would otherwise match no token prefix).

## Kernels

`kernels/cuda/gdn.cu`: `gdn_conv_bf16` (tiled over tokens), `gdn_delta_rule_bf16` (sequential over tokens; one
thread per state row for dk <= 128, one warp per row for dk = 256). A chunked parallel prefill form is not done yet.
CPU: the f64 reference in `intarsia_core::gdn` backs both `ref_*` and `cpu_f32_*`.
