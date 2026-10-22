# Recipe reference

A recipe is a TOML file that maps one model family onto the library's blocks: which blocks make up a layer, which
checkpoint tensor feeds each one, and a few serving defaults. Most new models need only a recipe, no code
([adding-a-model.md](adding-a-model.md) is the workflow; this page is the key-by-key reference).

The schema lives in [`crates/intarsia-core/src/recipe.rs`](../crates/intarsia-core/src/recipe.rs). Two tests keep this
page honest: every key the schema accepts must appear below in backticks, and every `toml` block here that starts with
`[model]` must parse and validate as a recipe (`crates/intarsia-models/tests/recipes_doc.rs`).

## Getting a recipe

```sh
intarsia recipe new --model /models/NewModel            # covered already? prints "covered: builtin:qwen3 ..." and exits 0
intarsia recipe new --model /models/NewModel --out recipes/newmodel.toml   # else writes a scaffold with TODO(recipe-new) lines
intarsia onboard --model /models/NewModel --backend hip # recipe, smoke generate, qualify; exit 3 = a TODO needs a human
intarsia plan show --model /models/NewModel --backend hip [--recipe FILE] [--json]   # what the loader resolved
```

`recipe new` fills every field it can infer from `config.json` and the safetensors headers, with a comment naming the
source, and leaves `# TODO(recipe-new): ...` where a person must decide (an unsupported activation, rope type, bias, or
block kind). Delete the marker once you have checked the line. A recipe is picked by `match.architectures`; pass a
specific one with `--recipe FILE`, or a directory of them with `--recipe-dir DIR` (searched before the built-ins, which
live in `recipes/*.toml` and are compiled in by `crates/intarsia-models/src/recipe.rs`).

**Errors you will meet.** An unknown key is rejected with the list of valid ones (`unknown field `kv_dtyp`, expected
one of `kv_dtype`, `max_batch`, `fp8_act``) in every table with a fixed set of keys. Three places still accept names
they do not know: `[layer.norms]` (a map: a misspelled `pre_ffn` is a norm that is never applied), `[layer.by_type]`
(a map from layer type to block list) and extra `[layer.<block>]` tables (`other`; only `[layer.moe]` and `[layer.gemma]` are read, the rest
fail at build). A recipe in `--recipe-dir` that does not parse is skipped when choosing by architecture, and the "no
recipe matches" error now lists each skipped file with its parse error, so a typo shows up there instead of as a silent
miss. `intarsia plan show` is the quickest way to see a load error with the model's real `config.json`.

## Tables and keys

Types: *string*, *int*, *float*, *bool*, *list* (of strings), *table*. "Required" keys have no default and a recipe
without them does not parse. A tensor name may contain `{i}` for the layer index; the loader appends `.weight` when the
checkpoint has the name only with that suffix.

### `[model]` (required)

| Key | Type | Default | What it controls |
|---|---|---|---|
| `match.architectures` | list | required | Recipe is chosen when one of these equals an entry of `config.json` `architectures` (`"Qwen3ForCausalLM"`). First match wins; `--recipe-dir` recipes are searched before built-ins. |
| `family` | string | required | Name shown in logs, `plan show` and tuning sources (`recipe qwen3_5_moe, cuda`). Also the file stem by convention. |
| `chat_template` | string | `"hf"` | Where the chat template comes from. `"hf"` = the model's own Jinja template via minijinja (`chat_template.jinja`, then `tokenizer_config.json`). |
| `reasoning_parser` | string | none | Server-side splitting of reasoning from the answer. `"think-tags"` splits on `<think>...</think>`. Set, it also turns thinking on in the chat template; unset, the model is not asked to think. |
| `tool_parser` | string | none | Tool-call dialect the server parses from the output. `"hermes"` is the dialect the server recognises. |
| `norm_offset` | float | `0.0` | Added to every RMSNorm weight in f32 before use (`x * (1 + w)` for Qwen3.5/3.6 is `1.0`). Applies to pre-block norms, the final norm and q/k norms, not the gated norm inside GatedDeltaNet. Getting it wrong gives fluent nonsense. |

### `[layer]` (required)

| Key | Type | Default | What it controls |
|---|---|---|---|
| `pattern` | list | required, non-empty | Blocks applied in order inside every layer, e.g. `["attn", "ffn"]`. Each name selects the table `[layer.<name>]`. Blocks: `attn`, `gdn`, `ffn`, `moe`. |
| `pattern_by_layer` | string | none | Hybrid models: the `config.json` key holding a per-layer type list. Must equal `[config] layer_types` (default `"layer_types"`). When set, layer *i* uses `by_type[layer_types[i]]` instead of `pattern`. |
| `by_type` | table of lists | empty | Needs `pattern_by_layer`. One block list per layer type, e.g. `linear_attention = ["gdn", "moe"]`. A layer type with no entry is a load error. |
| `norms` | table of strings | empty | `pre_<block> = "<norm tensor prefix>"`: the norm applied before that block (`pre_attn`, `pre_ffn`, `pre_gdn`, `pre_moe`). |

### `[layer.attn]` (grouped-query attention)

| Key | Type | Default | What it controls |
|---|---|---|---|
| `kind` | string | required | `"gqa"`. Others (MLA, sliding window, Mamba) are not implemented. |
| `q`, `k`, `v`, `o` | string | required | Projection tensor prefixes. |
| `q_norm`, `k_norm` | string | none (no qk-norm) | Per-head RMSNorm on q / k before rope (Qwen3 has it; Llama and Qwen2 do not). |
| `rope` | string | `"default"` | Rotary variant. `"default"` is neox / rotate-half with `rope_theta` from `config.json` over the first `partial_rotary_factor * head_dim` values of each head. The scaling type (`rope_type`) is read from `config.json`, not the recipe; only the unscaled embedding is supported on `main`. |
| `gated` | bool | `false` | `q` projects to `[n_heads, 2 * head_dim]`: the query plus a per-head output gate (`attn_out * sigmoid(gate)` before `o`), interleaved per head (HF `attn_output_gate`, Qwen3.5/3.6). |

Projection biases (`q_bias`, `k_bias`, `v_bias`, Qwen2 style) and `llama3` / `yarn` rope scaling are added by draft PR
#77 (lane fx4); until it merges, a checkpoint with biases or scaled rope is refused at load with a message naming the
missing feature.

### `[layer.gdn]` (Gated DeltaNet linear attention: Qwen3-Next, Qwen3.5, Qwen3.6)

| Key | Type | Default | What it controls |
|---|---|---|---|
| `kind` | string | required | `"gated_delta_net"`. |
| `in_qkv` | string | required | Fused q/k/v projection `[2*Hk*dk + Hv*dv, hidden]`. |
| `in_z` | string | required | Output-gate projection `[Hv*dv, hidden]`. |
| `in_b` | string | required | Beta projection `[Hv, hidden]`. |
| `in_a` | string | required | Decay projection `[Hv, hidden]`. |
| `conv` | string | required | Depthwise causal conv taps `[C, 1, K]` or `[C, K]`. |
| `a_log`, `dt_bias` | string | required | Decay parameters. |
| `norm` | string | required | Gated RMSNorm weight `[dv]` (plain weight, never offset by `norm_offset`). |
| `out` | string | required | Output projection. |

### `[layer.ffn]` (dense MLP)

| Key | Type | Default | What it controls |
|---|---|---|---|
| `kind` | string | required | `"dense"`. |
| `act` | string | required | `"silu_glu"` (SwiGLU). Other activations (GELU, ...) are not implemented; `recipe new` leaves a TODO. |
| `gate`, `up`, `down` | string | required | Projection tensor prefixes. |

### `[layer.moe]` (sparse mixture of experts, with shared expert)

| Key | Type | Default | What it controls |
|---|---|---|---|
| `kind` | string | required | `"qwen3_5_moe"` (the only MoE layout implemented: routed experts plus a gated shared expert). |
| `prefix` | string | required | Tensor prefix of the MoE block, e.g. `"model.language_model.layers.{i}.mlp"`; the loader finds router, experts and shared expert under it. Expert counts and widths come from `config.json`. |

### `[layer.gemma]` (Gemma 4 decoder layer, one block per layer)

| Key | Type | Default | What it controls |
|---|---|---|---|
| `kind` | string | required | `"gemma4"` (the only layout implemented). |
| `prefix` | string | required | Tensor prefix of the whole layer, e.g. `"model.language_model.layers.{i}"`. Everything under it (attention, dense MLP, router, experts, the five norms, `layer_scalar`) is found by the fixed Gemma 4 names. Head sizes, KV heads, sliding window, rope tables, soft-cap and expert counts come from `config.json` (`ModelDims::gemma`, `layer_types`). Use `pattern = ["gemma"]` and `[layer.norms] pre_gemma = <input_layernorm>`. |

### `[top]` (required)

| Key | Type | Default | What it controls |
|---|---|---|---|
| `embed` | string | required | Token embedding tensor. |
| `final_norm` | string | required | Final norm tensor. |
| `lm_head.tensor` | string | required | Output projection tensor. |
| `lm_head.tied_to` | string | none | Tensor to reuse when `lm_head.tensor` is absent from the checkpoint (or `tie_word_embeddings` is set). A missing head without this is a load error. |

### `[defaults]`

| Key | Type | Default | What it controls |
|---|---|---|---|
| `kv_dtype` | string | `"bf16"` | KV cache dtype: `bf16`, `f16`, `f32` or `fp8_e4m3`. Override per run with `--kv-dtype`. Anything but `bf16` changes numerics ([tuning.md](tuning.md)). |
| `max_batch` | int | `1` | Most sequences a run may decode together. `--max-batch N` above it is refused; without the flag the value is clamped to what the engine implements (continuous batching is Metal only, up to 8). |
| `fp8_act` | string | `"static"` | FP8 checkpoints: `"static"` quantises activations with the checkpoint's `input_scale` (W8A8), `"none"` keeps them 16-bit (W8A16). No effect on other formats. See [quant-moe.md](quant-moe.md). Override with `--fp8-act`. |

### `[tuning]` and `[tuning.<backend>]`

| Key | Type | Default | What it controls |
|---|---|---|---|
| `prefill_chunk` | int, at least 1 | `256` (code default) | Tokens per prefill forward pass, for every backend. Override with `--prefill-chunk`. |
| `[tuning.<backend>] prefill_chunk` | int, at least 1 | the base value | Same, for one backend: `cpu`, `cuda`, `hip` or `metal`. A backend without a table uses the base value. |

Order of resolution: `--prefill-chunk`, then `[tuning.<backend>]`, then `[tuning]`, then 256; `plan show` prints which
one won. The chunk is a numerics knob, so commit a value only with the measurement behind it in a comment
([tuning.md](tuning.md)).

### `[config]` (rename `config.json` keys)

Every key is a string naming the `config.json` field to read (looked up in `text_config` first for multimodal
wrappers). Omit the table unless the family names a field differently from Hugging Face's usual.

| Key | Default |
|---|---|
| `hidden_size` | `"hidden_size"` |
| `num_layers` | `"num_hidden_layers"` |
| `num_heads` | `"num_attention_heads"` |
| `num_kv_heads` | `"num_key_value_heads"` |
| `head_dim` | `"head_dim"` (falls back to `hidden / heads` when the config has none) |
| `intermediate_size` | `"intermediate_size"` |
| `vocab_size` | `"vocab_size"` |
| `rms_eps` | `"rms_norm_eps"` |
| `rope_theta` | `"rope_theta"` |
| `tie_embeddings` | `"tie_word_embeddings"` |
| `max_positions` | `"max_position_embeddings"` |
| `layer_types` | `"layer_types"` (hybrid models) |
| `linear_k_heads`, `linear_v_heads` | `"linear_num_key_heads"`, `"linear_num_value_heads"` |
| `linear_k_head_dim`, `linear_v_head_dim` | `"linear_key_head_dim"`, `"linear_value_head_dim"` |
| `linear_conv_kernel` | `"linear_conv_kernel_dim"` |

### `[accuracy]` (read only by `intarsia accuracy`; no effect on loading or serving)

| Key | Type | Default | What it controls |
|---|---|---|---|
| `chat_turns` | string | none | Turn format of the model's chat template (`"chatml"`): which rows of a chat prompt the model is asked to predict; the KL floor skips system and user turns. Fallback for a golden whose metadata names no known format. |
| `reuse_logit_tol` | float | built-in constant | Prefix-reuse judge: rounding scale in nats of log-probability over the top-5 tokens within which a warm and a cold distribution count as one. Measure once on a known-good build, never derive per run. |
| `reuse_kl_tol` | float | built-in constant | Prefix-reuse judge: largest KL(cold ‖ warm) in nats that still counts as rounding. |
| `ref_noise_factor` | float | `1.0` | Factor on the measured reference noise of the golden for the KL floor. A per-golden value wins. |
| `ref_noise_prompt_factor` | float | `3.0` | `--self-golden --self-vs-ref-noise`: how many times the reference noise the worst prompt's mean KL may be. A per-golden value wins. |

## Worked example 1: a dense model (Qwen3-8B)

Pure GQA plus SwiGLU, qk-norm, one block pattern for every layer. This is `recipes/qwen3.toml` without the tuning
comments.

```toml
[model]
match.architectures = ["Qwen3ForCausalLM"]
family = "qwen3"
chat_template = "hf"
reasoning_parser = "think-tags"
tool_parser = "hermes"

[layer]
pattern = ["attn", "ffn"]

[layer.attn]
kind = "gqa"
q = "model.layers.{i}.self_attn.q_proj"
k = "model.layers.{i}.self_attn.k_proj"
v = "model.layers.{i}.self_attn.v_proj"
o = "model.layers.{i}.self_attn.o_proj"
q_norm = "model.layers.{i}.self_attn.q_norm"
k_norm = "model.layers.{i}.self_attn.k_norm"
rope = "default"

[layer.ffn]
kind = "dense"
act = "silu_glu"
gate = "model.layers.{i}.mlp.gate_proj"
up = "model.layers.{i}.mlp.up_proj"
down = "model.layers.{i}.mlp.down_proj"

[layer.norms]
pre_attn = "model.layers.{i}.input_layernorm"
pre_ffn = "model.layers.{i}.post_attention_layernorm"

[top]
embed = "model.embed_tokens"
final_norm = "model.norm"
lm_head = { tensor = "lm_head", tied_to = "model.embed_tokens" }

[defaults]
kv_dtype = "bf16"
max_batch = 8

[tuning.metal]
prefill_chunk = 2048   # measured: docs/results/2026-10-14-metal-prefill-p3-final.md

[accuracy]
chat_turns = "chatml"
```

To adapt it to a similar family (Llama, Mistral): change `match.architectures` and `family`, delete `q_norm` / `k_norm`
if the checkpoint has no such tensors, check the tensor prefixes with `intarsia recipe new --model <dir>` (it lists
what it found), and leave `[tuning]` out until you have measured something.

## Worked example 2: a hybrid MoE model (Qwen3.6-35B-A3B)

Two layer types chosen per layer from `config.json`, a gated attention block, a Gated DeltaNet block, a sparse MoE
feed-forward, a zero-centred norm, and a multimodal wrapper whose text weights sit under `model.language_model.*`.
Condensed from `recipes/qwen3_5_moe.toml`.

```toml
[model]
match.architectures = ["Qwen3_5MoeForConditionalGeneration", "Qwen3_5MoeForCausalLM"]
family = "qwen3_5_moe"
chat_template = "hf"
reasoning_parser = "think-tags"
tool_parser = "hermes"
norm_offset = 1.0                 # x * (1 + w)

[layer]
pattern = ["attn", "moe"]         # used only when a layer has no type
pattern_by_layer = "layer_types"  # config.json key with the per-layer list

[layer.by_type]
full_attention = ["attn", "moe"]
linear_attention = ["gdn", "moe"]

[layer.attn]
kind = "gqa"
gated = true                      # q_proj = [q | output gate] per head
q = "model.language_model.layers.{i}.self_attn.q_proj"
k = "model.language_model.layers.{i}.self_attn.k_proj"
v = "model.language_model.layers.{i}.self_attn.v_proj"
o = "model.language_model.layers.{i}.self_attn.o_proj"
q_norm = "model.language_model.layers.{i}.self_attn.q_norm"
k_norm = "model.language_model.layers.{i}.self_attn.k_norm"
rope = "default"

[layer.gdn]
kind = "gated_delta_net"
in_qkv = "model.language_model.layers.{i}.linear_attn.in_proj_qkv"
in_z = "model.language_model.layers.{i}.linear_attn.in_proj_z"
in_b = "model.language_model.layers.{i}.linear_attn.in_proj_b"
in_a = "model.language_model.layers.{i}.linear_attn.in_proj_a"
conv = "model.language_model.layers.{i}.linear_attn.conv1d"
a_log = "model.language_model.layers.{i}.linear_attn.A_log"
dt_bias = "model.language_model.layers.{i}.linear_attn.dt_bias"
norm = "model.language_model.layers.{i}.linear_attn.norm"
out = "model.language_model.layers.{i}.linear_attn.out_proj"

[layer.moe]
kind = "qwen3_5_moe"
prefix = "model.language_model.layers.{i}.mlp"

[layer.norms]
pre_attn = "model.language_model.layers.{i}.input_layernorm"
pre_gdn = "model.language_model.layers.{i}.input_layernorm"
pre_moe = "model.language_model.layers.{i}.post_attention_layernorm"

[top]
embed = "model.language_model.embed_tokens"
final_norm = "model.language_model.norm"
lm_head = { tensor = "lm_head", tied_to = "model.language_model.embed_tokens" }

[defaults]
kv_dtype = "bf16"
max_batch = 32                    # declared; the engine clamps to what it implements

[tuning.cuda]
prefill_chunk = 2048              # docs/results/2026-10-13-prefill-cuda.md

[tuning.hip]
prefill_chunk = 4096              # docs/results/2026-10-17-hip-prefill.md

[accuracy]
chat_turns = "chatml"
```

The vision-language sibling (`recipes/qwen3_5_vl.toml`) is the same recipe with a different `match.architectures` and
tensor prefix: recipes do not inherit from each other, copy the file and change the prefix.

## Worked example 3: a quantized checkpoint

The quantisation scheme is **not** chosen in the recipe. The loader detects it from the checkpoint (`quantization_config`
in `config.json`, `hf_quant_config.json`, and the scale tensors next to each weight: `weight_scale`, `input_scale`,
`weight_scale_2`, ...), so an FP8, NVFP4 or MXFP4 build of a dense family reuses the same tensor names and the same
recipe. `intarsia recipe new` reports the detected scheme (`covered: ... quant: nvfp4+fp8`) when the engine implements it and writes a
TODO when it does not. What a recipe adds for a quantized model is a default for the activation mode and, if you
measured one, a different prefill chunk. This one pins W8A16 for an FP8 build of the dense family above; it is selected
only with `--recipe`, because its `match.architectures` equals the built-in recipe's.

```toml
[model]
match.architectures = ["Qwen3ForCausalLM"]
family = "qwen3_fp8_w8a16"
reasoning_parser = "think-tags"
tool_parser = "hermes"

[layer]
pattern = ["attn", "ffn"]

[layer.attn]
kind = "gqa"
q = "model.layers.{i}.self_attn.q_proj"
k = "model.layers.{i}.self_attn.k_proj"
v = "model.layers.{i}.self_attn.v_proj"
o = "model.layers.{i}.self_attn.o_proj"
q_norm = "model.layers.{i}.self_attn.q_norm"
k_norm = "model.layers.{i}.self_attn.k_norm"

[layer.ffn]
kind = "dense"
act = "silu_glu"
gate = "model.layers.{i}.mlp.gate_proj"
up = "model.layers.{i}.mlp.up_proj"
down = "model.layers.{i}.mlp.down_proj"

[layer.norms]
pre_attn = "model.layers.{i}.input_layernorm"
pre_ffn = "model.layers.{i}.post_attention_layernorm"

[top]
embed = "model.embed_tokens"
final_norm = "model.norm"
lm_head = { tensor = "lm_head", tied_to = "model.embed_tokens" }

[defaults]
fp8_act = "none"        # W8A16: activations stay 16-bit; "static" (default) is W8A8 with the checkpoint's input_scale
```

W8A16 versus W8A8 changes numerics (see the measurements in [quant-moe.md](quant-moe.md)), so measure it with `intarsia accuracy` before you change a default. Every quantized checkpoint needs
its own golden check; [accuracy.md](accuracy.md) says which thresholds apply per format (BF16 top-1 99.5% / KL 1e-3,
FP8 KL 5e-3, 4-bit KL 3e-2). Per-run overrides without editing the recipe: `--fp8-act`, `--mxfp4-act`, `--kv-dtype`.

## Checklist for a new recipe

1. `intarsia recipe new --model <dir>`; resolve every `TODO(recipe-new)`. Do not guess a rope, activation or bias
   setting: nonsense output with a plausible tokenizer almost always means one of them is wrong.
2. `intarsia plan show --model <dir> --backend <b>`: loads the model, lists the kernel chosen for every op, and names an
   op with no kernel.
3. `intarsia generate --model <dir> --backend cpu --prompt "The capital of France is" --max-tokens 16`.
4. `intarsia onboard --model <dir> --backend <b> [--golden-from <url>]` for the quick and in-depth gates.
5. Commit the recipe to `recipes/`, add it to `BUILTIN_RECIPES` in `crates/intarsia-models/src/recipe.rs`, and tune
   with [tuning.md](tuning.md) only after the accuracy gates pass.
