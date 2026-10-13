# Adding a model

Most models need only a recipe: a TOML file that maps the checkpoint's tensor names onto library
blocks. A new block type is needed only for a new mechanism (see the end).

## 1. Write the recipe

Start from `recipes/qwen3.toml` and copy it to `recipes/<family>.toml`. The pieces:

```toml
[model]
match.architectures = ["Qwen3ForCausalLM"]   # config.json "architectures" selects the recipe
family = "qwen3"
chat_template = "hf"                          # the model's own jinja template
reasoning_parser = "think-tags"               # parsers the server applies (optional)
tool_parser = "hermes"

[layer]
pattern = ["attn", "ffn"]                     # blocks applied in order inside every layer

[layer.attn]                                  # one table per block named in `pattern`
kind = "gqa"
q = "model.layers.{i}.self_attn.q_proj"       # {i} = layer index; ".weight" is appended if needed
k = "..."  v = "..."  o = "..."
q_norm = "..."  k_norm = "..."                # present = per-head qk RMSNorm before rope
rope = "default"                              # neox / rotate-half, theta from config.json

[layer.ffn]
kind = "dense"
act = "silu_glu"
gate = "..."  up = "..."  down = "..."

[layer.norms]                                 # pre_<block> = the norm applied before that block
pre_attn = "model.layers.{i}.input_layernorm"
pre_ffn = "model.layers.{i}.post_attention_layernorm"

[top]
embed = "model.embed_tokens"
final_norm = "model.norm"
lm_head = { tensor = "lm_head", tied_to = "model.embed_tokens" }   # tied when the tensor is absent

[defaults]
kv_dtype = "bf16"                             # bf16 | f16 | f32 | fp8_e4m3 (fp8 only on cpu-ref for now)
```

Dimensions are read from `config.json` with the Hugging Face key names (`hidden_size`,
`num_hidden_layers`, `num_attention_heads`, `num_key_value_heads`, `head_dim`, `intermediate_size`,
`vocab_size`, `rms_norm_eps`, `rope_theta`, `tie_word_embeddings`, `max_position_embeddings`).
`head_dim` is taken from the config when present (Qwen3 sets 128 with hidden 1024 / 16 heads), and
only falls back to `hidden / heads`. A `[config]` table can rename keys for families that differ.

Recipes are matched by architecture from `--recipe-dir` directories first and then the built-ins
(`crates/intarsia-models/src/recipe.rs`, `BUILTIN_RECIPES`). To try a recipe without installing it:
`--recipe path/to/recipe.toml`. A recipe that names a block this build does not implement fails at
load with `block kind 'x' is not implemented in this build`, not with garbage output.

## 2. Try it: `intarsia generate`

```sh
cargo build --release -p intarsia-cli
target/release/intarsia generate --model <hf-dir> --backend cpu \
    --prompt "The capital of France is" --max-tokens 16
```

Raw completion applies no chat template. `--chat` wraps the prompt as a user message with the
model's template (`--no-think` renders `enable_thinking=false`). The template is taken from
`chat_template.jinja` when present, else from `tokenizer_config.json` (the order transformers uses);
override with `--chat-template-from-config` or `--chat-template-file`. The run prints the
planned context length (from measured free memory), the recipe and template sources, and the per-op time split.

Check a few prompts for sane continuations (counting, a story start, a factual sentence). Unmapped
or misnamed tensors fail loudly at load; wrong semantics (rope variant, norm offset) show up as
fluent-looking nonsense, which is what step 3 catches.

## 3. Prove it: `intarsia accuracy`

`intarsia accuracy` compares full-vocabulary log-probs against a Hugging Face BF16 golden file
produced by `tools/golden.py` (the accuracy tooling; top-1 agreement, mean/max KL, top-5 overlap; thresholds in
ARCHITECTURE section 6: BF16 top-1 99.5% and KL 1e-3, FP8 KL 5e-3, 4-bit KL 3e-2). Run it on cpu-ref
first (it is the oracle), then on each GPU backend. `Engine::forward_logits_each` is the API it uses.

For a quick numerical sanity check without a golden file, compare the top logits of the last prompt
position with transformers: `intarsia generate ... --show-logits 5`.

## 4. When you do need code

* New quantization format: implement `QuantFormat` (ARCHITECTURE 3.4) and register matmul kernels
  that accept it; the oracle path (`dequant_ref` then matmul) works as soon as `load` and
  `dequant_ref` exist.
* New block kind (MLA, sliding window, GDN, Mamba, MoE): add the block to the decoder in
  `crates/intarsia-core/src/model.rs`, its op(s) to `ops.rs` with a cpu-ref reference, and a
  `SlotSpec` if it carries state (`Recurrent` for GDN/Mamba). Then `intarsia kernels test` covers it.
* New kernel for an existing op: `kernels/README.md`.
