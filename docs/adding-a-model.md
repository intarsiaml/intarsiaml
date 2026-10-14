# Adding a model

A newly downloaded model goes through two commands:

```sh
# 1. does an existing recipe cover it? if not, scaffold one (seconds, reads config.json and the tensor names only)
intarsia recipe new --model /models/NewModel

# 2. qualify it: quick numbers in minutes, then the in-depth stages in the same run
intarsia qualify --model /models/NewModel --backend hip
```

Most models need only a recipe, a TOML file that maps the checkpoint's tensor names onto library
blocks. A new block type is needed only for a new mechanism ([When you need code](#when-you-need-code)).

## 1. `intarsia recipe new`

```sh
intarsia recipe new --model <hf dir> [--out recipes/<family>.toml] [--recipe-dir DIR]... [--force]
```

It reads `config.json` (with `text_config` of multimodal wrappers), `quantization_config` (in `config.json` or
`text_config`), `hf_quant_config.json`, and the tensor names from `model.safetensors.index.json` or the
safetensors headers (never the tensor data), then matches the architecture to the closest existing recipe and
block types.

* **An existing recipe covers the model** (architecture matches, every tensor template of the recipe resolves in
  the checkpoint, the config passes `check_supported`, the quantisation is one the engine implements): it prints
  `covered: builtin:qwen3 already handles this model (25/25 tensor templates resolve, quant: bf16); no new recipe
  needed`, exits 0 and writes nothing (unless you pass `--out`). Measured on the checkpoints this repo is tested
  with: Qwen3-8B and Qwen3-Reranker-0.6B are covered by `builtin:qwen3`, Qwen3.6-35B-A3B-NVFP4 by
  `builtin:qwen3_5_moe` (61/61 templates, quant `nvfp4+fp8`;
  `docs/results/2026-10-13-qualify/recipe-new-tatooine.txt`).
* **No recipe covers it**: it writes the scaffold to `--out` (default `recipes/<family>.toml`, refuses to
  overwrite without `--force`). Every inferred field carries a comment with its source (`# from config.json
  architectures`, `# from tensors: model.layers.0.self_attn.q_norm.weight`, `# from text_config.layer_types`).
  Anything it cannot map becomes `# TODO(recipe-new): <what, why, what to do>` where the field belongs: an unknown
  block (MLA, sliding window, Mamba, an unclaimed tensor group), an activation other than `silu`, `attention_bias`,
  a rope type other than `default` (yarn, linear, llama3, longrope), an unknown quantisation scheme (the
  `quant_method` / `quant_algo` string is named), recipe tensors missing from the checkpoint. The numbered list is
  printed to stderr and the command **exits non-zero**. Resolve every TODO, then run step 2.

The hand-written reference of the recipe format follows; `recipe new` produces exactly this.

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

[tuning]                                      # optional: per-backend performance defaults
prefill_chunk = 256                           # base value, every backend (omit: 256)
[tuning.cuda]                                 # override for one backend (cpu | cuda | hip)
prefill_chunk = 2048
```

Dimensions are read from `config.json` with the Hugging Face key names (`hidden_size`,
`num_hidden_layers`, `num_attention_heads`, `num_key_value_heads`, `head_dim`, `intermediate_size`,
`vocab_size`, `rms_norm_eps`, `rope_theta`, `tie_word_embeddings`, `max_position_embeddings`).
`head_dim` is taken from the config when present (Qwen3 sets 128 with hidden 1024 / 16 heads), and
only falls back to `hidden / heads`. A `[config]` table can rename keys for families that differ.
Hybrid models (GatedDeltaNet, MoE) add `pattern_by_layer` / `[layer.by_type]`, `[layer.gdn]`, `[layer.moe]`
and `norm_offset`; see `recipes/qwen3_5.toml` and `recipes/qwen3_5_moe.toml`.

`[tuning]` sets the prefill chunk (tokens per forward pass) a model loads with. Resolution order: the explicit
`--prefill-chunk` flag, then `[tuning.<backend>] prefill_chunk` for the running backend, then `[tuning] prefill_chunk`,
then 256; a backend without a table falls back to the base value. Load prints the result and its source (`prefill chunk
2048 (recipe qwen3_5_moe, cuda)`), `intarsia plan show --json` carries `prefill_chunk` and `prefill_chunk_source`, and
`accuracy` records both in baselines and self-goldens. Only tune a chunk with measurements behind it and cite them in a
comment: `recipes/qwen3_5_moe.toml` sets 2048 on CUDA only (`docs/results/2026-10-13-prefill-cuda.md`), because the
chunk changes rounding (prefix-reuse and baseline results are chunk-sensitive) and the 35B at 2048 has not been shown
to fit a 32 GB R9700. A baseline recorded at another chunk is refused with a message naming the recipe default;
pass `--prefill-chunk N` to run at the baseline's.

Recipes are matched by architecture from `--recipe-dir` directories first and then the built-ins
(`crates/intarsia-models/src/recipe.rs`, `BUILTIN_RECIPES`: add the new file there so the binary ships it).
To try a recipe without installing it pass `--recipe path/to/recipe.toml` (also to `qualify`). A recipe that
names a block this build does not implement fails at load with `block kind 'x' is not implemented in this
build`, not with garbage output.

## 2. `intarsia qualify`

```sh
intarsia qualify --model <dir> --backend <cpu|cuda|hip|metal> [--tp N] [--reference <url-or-hf-dir>]
                 [--golden FILE] [--golden-long FILE] [--out reports/<model>/<stamp>/]
                 [--quick-only] [--resume] [--push-metrics <pushgateway url>] [--on-quick-done '<cmd>']
```

One run, two sections, back to back. Every stage runs this same binary as a subprocess (a stage cannot leak GPU
memory into the next), records its wall time, verdict, provenance stamp and artifacts in
`<out>/state.json`, and `--resume` skips the stages that finished (a stage that ended in `ERROR` is run again).

**Exit code**: `0` every gate passed, `1` a gate failed, `2` the pipeline itself failed (a tool crashed or timed
out, bad arguments, or `--max-minutes` ran out before the end). A stage has one of four verdicts: `PASS`,
`FAIL` (the model did not meet the gate), `SKIP` (nothing to run or compare, the reason says why; read the
Skipped list before trusting a PASS), `ERROR` (the pipeline failed).

When the quick section ends, `quick-report.md` and `quick.json` are written, `QUICK DONE: <path>` is printed,
`--push-metrics` pushes the headline numbers and `--on-quick-done '<cmd>'` runs in the background with the report
path as `$1` (also `$INTARSIA_QUICK_REPORT`, `$INTARSIA_QUICK_JSON`, `$INTARSIA_REPORT_DIR`), so the quick numbers
can be published while the in-depth stages run. A failing hook is a warning in the report, not an exit code.
At the end `full-report.md` / `full.json` are written and the metrics pushed again. The layout of the reports is
[qualify-report-template.md](qualify-report-template.md): the headline first (model, backend, decode and prefill
tok/s with % of the byte floor, accuracy verdicts, failures with reasons), then a per-stage table.

### Stages and expected wall times

Measured on tatooine (2 x Radeon AI PRO R9700, `--backend hip`, Qwen3-8B bf16, 16.4 GB; run 3 in
`docs/results/2026-10-13-qualify/qwen3-8b-hip/`, stage times from `state.json`, `--tp 2` given). A cold weight
load from disk took ~50 s (about 20 GB/min); a warm load, with the files in the page cache, ~5 s. For a bigger
checkpoint the first load and each server start scale with its size; the accuracy and bench stages scale with the
number of prompts and generated tokens, not with the model.

| Section | Stage | What it runs | Gate | 8B on R9700 |
|---|---|---|---|---|
| quick | `doctor` | `doctor --out`: host, GPU, driver, bandwidth, known pitfalls | no check `fail`, backend opens | 0.5 s |
| quick | `recipe` | `recipe new`'s analysis: recipe, tensor templates, quantisation | an existing recipe covers the model | <0.1 s |
| quick | `plan` | `plan show --admit`: load the model, kernel per op, admission | loads, every op admitted | 50 s (cold load) |
| quick | `kernels` | `kernels test --model` on the model's shapes; a load error lists the missing kernels | all admitted | 41 s (skipped on cpu) |
| quick | `gate` | `accuracy --quick`: 5 sanity prompts, perplexity, non-finite rows | all ok | 9 s |
| quick | `acc-short` | `accuracy --golden` (prefill + teacher-forced decode) with the class thresholds | floor gate | 8 s |
| quick | `bench` | `serve` + `bench speed` at 128 / 1024 / 4096, byte floor | every size measured | 80 s |
| | | **quick section** | | **3 min 8 s** |
| full | `acc-long` | `accuracy --mode all` on the long golden: prefill, decode, prefix reuse | floor gate + reuse gate | 29 s |
| full | `envelope` | noise envelope (`chunk`, plus `attn-simt` on GPUs) and the baseline; then a self-check gate | the pair gates its own build | 24 s |
| full | `bench-long` | `serve` + `bench speed` at 1024 / 4096 / 16384 | every size measured | 83 s |
| full | `ab` | `localeval speed` against `--reference` and against our server | both ran | skipped (no `--reference`) |
| full | `tp` | with `--tp N`: tp plan, `bench speed`, accuracy against the short golden | accuracy gate | 71 s (tp 2) |
| | | **in-depth section** | | **3 min 26 s** (6 min 34 s in total) |

The numbers it produced for that run: decode 37.7 / 37.1 / 35.9 tok/s at 128 / 1024 / 4096 tokens (91 / 90 / 87 %
of the 41.4 tok/s byte floor), 32.2 tok/s at 16.8k (78 %), prefill 2.0k to 3.5k tok/s, accuracy against the HF
fp32 golden top-1 100.00 % / KL 2.3e-4 (short, prefill) and 99.58 % / 4.0e-4 (long), tp 2 decode 58.1 tok/s at 128
tokens. On the Mac, `--backend cpu --smoke` (Qwen3-Reranker-0.6B) takes 45 s for the quick and 2 min for the whole
pipeline (`docs/results/2026-10-13-qualify/mac-cpu-smoke/`).

What each stage needs and how it decides:

* **Goldens** (`acc-short`, `acc-long`, `envelope`, `tp`): `--golden` / `--golden-long`, else found by model name
  in `--golden-dir` (default `./golden`, `/models/golden`, `$INTARSIA_GOLDEN_DIR`; an fp32 file is preferred
  over others, `<name>-long...` is the long set), else made: from `--reference http://...` (a vLLM-style server:
  `tools/golden_vllm.py`, top-20 `prompt_logprobs`) or, for a small dense bf16 model (at most `--make-golden-max-gb`
  5 GB of weights, no MoE / hybrid / quantisation), an HF fp32 golden with `tools/golden.py` (needs `uv` or a
  python with torch and transformers). Otherwise the stage is **skipped with the reason**; a model that no one can
  make a golden for needs `--golden` or `--reference`. The short golden cannot exercise prefix reuse (the reuse
  gate needs a split of at least one prefill chunk, 256 tokens), so reuse is gated on the long golden, and there only one prompt
  (`long_prose_700`, continue and restore) is long enough to exercise it: a thin check. A stress run with
  `--prefill-chunk 16` on the short golden of Qwen3-8B (hip) failed the old stream-agreement criterion once
  (`science`, `continue`: the streams diverged at token 9, 9/32 agreeing;
  `docs/results/2026-10-13-qualify/run2-prefix-reuse-fail/`). It was investigated in
  `docs/results/2026-10-13-prefix-reuse.md`: not an engine bug but a rounding tie (the decode kernels' KV rows differ
  from the prefill kernels' in the last bit; the two candidate tokens were 0.0003 nats apart), and the reuse gate now
  judges a divergence on the logits. `qualify` still does not include the chunk-16 stress stage. ADR-0007's 35B (CUDA)
  FAIL-then-PASS (`continue` diverging at token 8) has the same signature; it was not re-run, so the same cause is likely,
  not shown.
* **Byte floor** (`bench`, `bench-long`, `tp`): decode is memory bound, so bandwidth / bytes read per token is a
  ceiling on single-stream tok/s. Bytes per token come from the checkpoint's tensor sizes (routed experts scaled by
  `top_k / experts`, the input embedding left out when `lm_head` is a separate tensor; for `--tp` the busiest
  rank's weights from the tp plan), bandwidth from `--bandwidth-gbs`, else a table by GPU arch (gfx1201 627 GB/s
  as in `docs/results/2026-10-13-phase2a-hip.md`, GB10 221, Apple M5 Max `applegpu_g17s` 614), else `doctor`'s measured device copy. The report names
  where both came from; above 100 % means a wrong input. Override with `--gb-per-token` / `--bandwidth-gbs`.
* **Speed is measured, not gated**: there is no threshold in `qualify`; a bench stage fails only when a size
  produced no result. Servers run with `--profile-every 0` (profiled steps bias throughput, see
  `docs/benchmarking.md`). The bench speaks chat completions: a model whose chat template ignores the user message
  (reranker templates do) cannot size prompts, and the stage is skipped with that reason; pass
  `--chat-template-file` with a plain ChatML template to bench it anyway.
* **A/B** (`ab`) runs `uv run --project /opt/code/deepeval localeval speed` once against our server and once
  against `--reference`, same sizes, same `--thinking off`. It never uses an ad-hoc script, localeval takes its own
  fleetq slot (inside a lease you hold `FLEETQ_JOB` makes it carry on), and `--no-unim` is passed through. It is
  one sequential round, not the interleaved rounds of `scripts/phase1b-acceptance.sh`; a ratio inside the noise
  (decode about 5 %) is not a result. It is skipped when there is no `--reference` URL, no localeval checkout
  (`--localeval-dir`) or no `uv`.
* **Regression gating artifacts** (`envelope`): `baselines/acc-short.baseline.json` and `baselines/envelope.json`
  in the report directory. Copy both next to the golden (`golden/baselines/`) and gate a later change with
  `intarsia accuracy --model <dir> --golden <file> --baseline <baseline> --noise-envelope <envelope>`
  (`docs/accuracy.md`, "Noise envelope"). A build without `INTARSIA_BUILD_STAMP` is recorded as `unstamped`:
  fine for gating the same binary, refused as a pre-change "before". Build with
  `INTARSIA_BUILD_STAMP="$(scripts/tree-stamp.sh)"` on a clean tree for a stamped one.
* **tp variant** (`tp`, only with `--tp N`): the model sharded over N devices; the other stages use one device.
  Models that do not fit one device (or GatedDeltaNet / MoE, not sharded yet) fail `plan` and need `--tp`
  support first.

### Running it

* **Inside a fleet lease.** `qualify` loads GPUs and starts servers; on a fleet box take the slot first
  (`fleetq run --host <box> --mem <GB> -- ...`, or `fleetq acquire` and prefix the commands with `FLEETQ_JOB`),
  with the lease's memory as `--mem-budget`. On tatooine the production `radiance` engine must be taken down for
  the run (`--takes-down radiance`) and restored before the lease is released; the build and the GPU run go through
  `deploy/tatooine/dev.sh`. Never run `qualify` beside a server that holds the memory it needs.
* **Time boxes.** `--max-minutes N` stops starting stages after N minutes (exit 2, the state is saved) and
  `--stage-timeout-min N` kills a stage's subprocess; continue later with `--resume --out <dir>`. A lease window of
  40 minutes: `--max-minutes 28 --stage-timeout-min 9`.
* **Smoke test** (cpu, any machine): `--backend cpu --smoke` uses tiny sizes and one repetition; the goldens
  of the repository's own small models are in `golden/`.
* **Metrics**: `--push-metrics http://host:9091` PUTs `intarsia_qualify_*` gauges (decode / prefill tok/s and % of
  floor by prompt size, accuracy top-1 and KL by stage and mode, stage pass flags and seconds, section seconds,
  exit code) to `/metrics/job/intarsia_qualify/model/<m>/backend/<b>`.

### Reading a result

Fix in this order: `recipe` (write or complete the recipe), `plan` (a missing kernel is named, with the
kernels the backend does register next to it; a new kernel is `kernels/README.md`), `gate` (nonsense output means
a wrong rope / norm / activation in the recipe), `acc-short` (the first numbers that say a recipe is right),
`kernels`, then the in-depth stages. Accuracy numbers against an HF bf16 golden are only as good as that
reference; docs/accuracy.md explains why fp32 goldens gate dense models and how the floor and the baseline gates
differ. Failure reasons in the report are the first line of the stage; the full output is in `logs/<stage>.out`
and `.err`.

## 3. By hand: `generate` and `accuracy`

The same checks, one at a time, when you are building the recipe:

```sh
cargo build --release -p intarsia-cli
target/release/intarsia generate --model <hf-dir> --backend cpu --prompt "The capital of France is" --max-tokens 16
target/release/intarsia accuracy --model <hf-dir> --backend cpu --quick
target/release/intarsia accuracy --model <hf-dir> --backend cpu --golden golden/<name>.safetensors
```

Raw completion applies no chat template. `--chat` wraps the prompt as a user message with the
model's template (`--no-think` renders `enable_thinking=false`). The template is taken from
`chat_template.jinja` when present, else from `tokenizer_config.json` (the order transformers uses);
override with `--chat-template-from-config` or `--chat-template-file`. The run prints the
planned context length (from measured free memory), the recipe and template sources, and the per-op time split.

Unmapped or misnamed tensors fail loudly at load; wrong semantics (rope variant, norm offset) show up as
fluent-looking nonsense, which the quick gate and the golden comparison catch. `intarsia accuracy` compares
full-vocabulary log-probs against a Hugging Face golden file produced by `tools/golden.py` (top-1 agreement,
mean/max KL, top-5 overlap; thresholds in ARCHITECTURE section 6: BF16 top-1 99.5% and KL 1e-3, FP8 KL 5e-3,
4-bit KL 3e-2). Run it on cpu-ref first (it is the oracle), then on each GPU backend.
`Engine::forward_logits_each` is the API it uses. For a quick numerical sanity check without a golden file,
compare the top logits of the last prompt position with transformers: `intarsia generate ... --show-logits 5`.

## When you need code

* New quantization format: implement `QuantFormat` (ARCHITECTURE 3.4) and register matmul kernels
  that accept it; the oracle path (`dequant_ref` then matmul) works as soon as `load` and
  `dequant_ref` exist.
* New block kind (MLA, sliding window, GDN, Mamba, MoE): add the block to the decoder in
  `crates/intarsia-core/src/model.rs`, its op(s) to `ops.rs` with a cpu-ref reference, and a
  `SlotSpec` if it carries state (`Recurrent` for GDN/Mamba). Then `intarsia kernels test` covers it.
* New kernel for an existing op: `kernels/README.md`.

`recipe new` names these cases in its TODO list; `qualify`'s `plan` stage names the ops that have no kernel.
