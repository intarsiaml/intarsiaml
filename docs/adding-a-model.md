# Adding a model

A newly downloaded model goes through one command:

```sh
intarsia onboard --model /models/NewModel --backend hip [--golden-from http://host:8000/v1] [--out reports/NewModel/onboard-1]
```

It chains the steps below (recipe, smoke generate, `qualify`'s quick tier, then its in-depth tier), **publishes the
quick numbers as soon as the quick tier is done** (`quick.md`, `quick.json`, `quick.prom`) while the in-depth tier
keeps running, and can be killed and run again with the same `--out`: finished stages are not repeated. The individual
commands ([`recipe new`](#1-intarsia-recipe-new), [`qualify`](#2-intarsia-qualify), [by hand](#3-by-hand-generate-and-accuracy)) stay
available for one step at a time.

Most models need only a recipe, a TOML file that maps the checkpoint's tensor names onto library
blocks. A new block type is needed only for a new mechanism ([When you need code](#when-you-need-code)).

## 0. `intarsia onboard`

```sh
intarsia onboard --model <hf dir> --backend <cpu|cuda|hip|metal> [--golden-from <vllm url | hf dir | golden file>]
                 [--out DIR] [--quick-only] [--fail-fast] [--redo STAGE]... [--push-metrics URL] [--on-quick-done CMD]
                 [--recipe FILE] [--recipe-dir DIR]... [--device N] [--tp N] [--mem-budget GiB] [-- <qualify flags>]
```

Six stages, in this order (`STAGES` in `crates/intarsia-cli/src/cmd/onboard.rs`; a new stage is one entry there):

| Stage | What it runs | Notes |
|---|---|---|
| `recipe` | `recipe new`'s analysis | A covering recipe is used as is. Otherwise a scaffold goes to `<out>/recipe.toml` and every later stage gets `--recipe` of it. While `TODO(recipe-new)` lines remain, onboard prints them numbered and **stops with exit 3**; fix them, delete the markers and run the same command again. A scaffold with no TODOs goes on (check it into `recipes/` afterwards). |
| `smoke` | `intarsia generate`, 16 tokens of "The capital of France is" | Load plus one completion. Crash or empty output stops the run (the quick report is still written). |
| `quick` | `intarsia qualify --quick-only` on `<out>/qualify/` | doctor, plan (load), kernels, gate, accuracy against the golden, speed snapshot 128 / 1024 / 4096 with the byte floor (`intarsia bench`). |
| `publish-quick` | report rendering | `quick.md`, `quick.json`, `quick.prom` (+ `--on-quick-done`). Runs **before** the in-depth tier starts, also after a failed or stopped run. |
| `in-depth` | `intarsia qualify` on the same directory | Long-context accuracy, 4k golden, noise envelope, speed sweep 1k / 4k / 16k, A/B against `--golden-from` when it is a server, `tp` with `--tp`. Quick stages are skipped (qualify's own `--resume`). |
| `publish-in-depth` | report rendering | `in-depth.md`, `in-depth.json`, `in-depth.prom`. Quick + in-depth stages, so it is the full report. |

`--golden-from` becomes `--reference` (an `http(s)://` URL of a vLLM-style server, or an HF model directory) or
`--golden` (a `.safetensors` file); without it qualify looks for a golden by model name and makes one only for small
checkpoints (see [qualify](#2-intarsia-qualify)). The accuracy, speed and gate logic is qualify's: onboard runs it and
renders its `state.json` with qualify's report code, so every number is the one `qualify` documents.

**What is published.** Each report carries the onboard verdict (a stopped recipe or a crashed smoke test is an
`ERROR`, never "PASS (0 stages)"), qualify's headline and per-stage tables, and a **stage timeline** table: start,
end and duration in UTC for each onboard stage and each qualify stage inside it. `*.json` has the same under
`onboard.timeline` plus `published_unix`; `*.prom` is Prometheus text exposition (qualify's headline gauges plus
`intarsia_onboard_stage_seconds`, `intarsia_onboard_stage_end_timestamp_seconds` and
`intarsia_onboard_published_timestamp_seconds`), ready for a node-exporter textfile directory; `--push-metrics URL`
also PUTs the headline to a Pushgateway when each tier is done. Files are written atomically, `quick.md` last, so its
mtime is the moment the quick numbers went public. `--on-quick-done 'cmd'` runs in the background with the path of
`quick.md` as `$1`.

**Resume.** `<out>/onboard-state.json` holds one record per stage (verdict, start, end). Run the same command with the same
`--out`: a stage that finished (PASS or FAIL: the gate ran) is skipped; an ERROR stage and the stage that was running
when the process died run again, and `qualify` then skips its own finished stages, so a kill in the middle of the
in-depth tier costs only the stage that was running. `--redo quick` runs a stage again even though it finished (its
report is rewritten with it). A SKIP inside qualify (no golden) is not retried by onboard; supply the golden and
`--redo quick`. SIGTERM / SIGINT of onboard stops the whole chain (qualify, its model server and the stage
subprocess), so a killed run holds no GPU memory. Exit code: `0` all gates passed, `1` a gate failed, `2` the pipeline
failed or stopped early, `3` the recipe needs a human.

Lease the box first (`fleetq run --host <box> --mem <GB> -- intarsia onboard ...`, or an exclusive lease for numbers
you will compare).

**Measured timings (Qwen3.6-35B-A3B-NVFP4, a recipe that already exists; 2026-10-22, lane `on`,
`docs/results/2026-10-22-on-onboard.md`).** Targets: quick numbers within 2 h and the in-depth tier within 4 h of the
weights being on the box.

| Box | Goldens | Quick numbers published | Whole pass | Notes |
|---|---|---|---|---|
| spark1 (CUDA, GB10), `--mem-budget 38`, vLLM up | made from the vLLM server in 29 s (short + long + 4k) | 7 min 24 s after start (smoke 1 m 33 s with the weights cold) | **15 min 24 s** | all 14 stages PASS, 3 skipped (`acc-format`, `ab`, `tp`) |
| tatooine (HIP, one R9700), `--mem-budget 29` | same goldens, copied | 1 min 24 s | **2 min 47 s** | `acc-4k` FAILs the prefix-reuse gate (a real engine finding, not an onboarding fault) |
| tatooine, cached canonical goldens (no 4k) | cached | 1 min 23 s | 3 min 59 s | PASS |

Build (cached dependencies) 44 s on spark1, 6 s for the CLI crate on tatooine. The wall clock of a real onboarding is
dominated by what the pipeline cannot do for you: the box queue (here 4.5 h on the Sparks), a recipe that needs a
human (`TODO(recipe-new)`), and a new kernel. A model whose recipe exists is far inside both targets.

**Gotchas found by running it on CUDA and HIP.**

* **Neither spark1 nor the tatooine dev image can make a golden.** `qualify` makes one with `uv run` or `python3` plus numpy
  and safetensors; spark1 has neither `uv` nor numpy, and the HIP dev image has no numpy. Make the goldens on a machine
  that has `uv` (`scripts/onboard/make-goldens.sh <dir> <model name> <vLLM url> <served model>`: short, long and 4k set,
  29 s for the 35B) and give the directory with `-- --golden-dir DIR`. The files are found by model name. **With `--golden-from URL` and no golden the accuracy stages are SKIPPED and the run can still report PASS** (a SKIP
  is not a failure): read the Skipped list of `quick.md` before trusting a PASS.
* **The 4k golden is never made by `qualify`.** Without one, `acc-4k` is SKIPPED and the in-depth tier is shorter than a
  complete one (the 4k golden is where the prefix-reuse gate bites at a 2048 or 4096 chunk). `make-goldens.sh` makes it.
* **A vLLM server may listen on one address only.** spark1's vLLM listens on its Tailscale address, not `localhost`: use
  `--golden-from http://<that address>:8000/v1` when onboard runs on the box.
* **A cached golden whose file name matches a tracked `golden/ref_noise/*.json` sidecar must be the same file.** On
  tatooine `/models/golden/qwen36-35b-a3b-nvfp4-vllm-top20.safetensors` is an older file with another hash, and
  `acc-short` ERRORs with `ref_noise was measured on golden hash ...`. Use the golden of the repository (`golden/`) or a
  freshly made one with another name. The report used to show only a warning line above that error; it now starts at
  the `Error:` line.
* **Build the binary from a clean, committed tree.** The baseline written by `envelope` records the tree it came from, and a
  modified tree makes `envelope` ERROR (`the baseline was recorded from a modified tree`).
* **The long golden may hold a prompt longer than the fixed accuracy context.** The 35B long golden has a 12k-token prompt
  and the old fixed `--max-seq 8192` failed `acc-long`. `qualify` now sizes the context from the golden's longest prompt.
* **A box shared through fleetq is the slow part.** Take one lease that covers the golden generation, the build and the
  onboard run (`fleetq acquire ... --wait`, then `FLEETQ_JOB=<id>` on each command), so a failed step does not send you to the
  back of the line. Launch the waiting `fleetq` client detached (`start_new_session`); a client that dies is dropped from the
  queue after three minutes.

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
  block (MLA, sliding window, Mamba, an unclaimed tensor group), an activation other than `silu`, an output-projection
  bias (`o_proj.bias`, config `attention_bias = true`), a rope type other than `default`, `llama3` or `yarn` (linear,
  dynamic, longrope), an unknown quantisation scheme (the
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
q_bias = "model.layers.{i}.self_attn.q_proj.bias"   # present = projection bias, added right after the projection
k_bias = "..."  v_bias = "..."                # (Qwen2 / Qwen2.5 set these without any config key; each one optional)
rope = "default"                              # neox / rotate-half, theta from config.json; llama3 / yarn scaling is read from
                                              # config.json (rope_parameters / rope_scaling), nothing to set here

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
[tuning.cuda]                                 # override for one backend (cpu | cuda | hip | metal)
prefill_chunk = 2048
```

Dimensions are read from `config.json` with the Hugging Face key names (`hidden_size`,
`num_hidden_layers`, `num_attention_heads`, `num_key_value_heads`, `head_dim`, `intermediate_size`,
`vocab_size`, `rms_norm_eps`, `rope_theta`, `tie_word_embeddings`, `max_position_embeddings`).
`head_dim` is taken from the config when present (Qwen3 sets 128 with hidden 1024 / 16 heads), and
only falls back to `hidden / heads`. A `[config]` table can rename keys for families that differ.
Hybrid models (GatedDeltaNet, MoE) add `pattern_by_layer` / `[layer.by_type]`, `[layer.gdn]`, `[layer.moe]`
and `norm_offset`; see `recipes/qwen3_5.toml` and `recipes/qwen3_5_moe.toml`.

### What the attention block expresses without code

| Feature | Where it comes from | Built-in recipe | Backends |
|---|---|---|---|
| GQA, `head_dim` from the config, partial rotary (`partial_rotary_factor`) | `config.json` | `qwen3`, `qwen3_5*` | all |
| per-head qk RMSNorm (`q_norm` / `k_norm`) | recipe | `qwen3` | all |
| q / k / v projection bias (`q_bias` / `k_bias` / `v_bias`) | recipe (Qwen2 sets them without a config key) | `qwen2` | cpu, Metal (op `add_bias`)<sup>1</sup> |
| `rope_type` `llama3` (Llama 3.1 / 3.2 / 3.3) and `yarn` | `config.json` `rope_parameters` / `rope_scaling` | `llama` | cpu, Metal (op `rope_scaled`)<sup>1</sup> |
| gated query (`attn_output_gate`) | recipe (`gated = true`) | `qwen3_5*` | all |

<sup>1</sup> CUDA and HIP do not register `rope_scaled` / `add_bias` yet: a model that needs one fails at load with
`NoKernel` (named in the plan stage); it never runs the unscaled rope or drops a bias.

How scaling works: the loader computes the whole frequency table once (`crates/intarsia-core/src/rope.rs`, the
transformers formulas for `llama3` and `yarn`, including YaRN's attention factor on cos / sin), uploads it, and the
attention block calls `rope_scaled` instead of `rope`. A model without scaling emits exactly the calls it did before, so its
plan and numerics do not change. Scaled models also skip the fused qk-norm + rope + KV-append prologue (it computes
plain frequencies). A biased projection runs as the plain matmul followed by `add_bias` (in place, bf16 + bf16 bias; two
bf16 roundings: after the projection and after the add). Measured on Qwen2.5-7B this is closer to the fp32 reference than
rounding once through an f32-output matmul (prefill top-1 99.30% vs 98.9%), and keeps the fast NAX prefill GEMM. Still rejected, by name, at `recipe new` and at load: `linear` / `dynamic` / `longrope` rope
types, sliding-window attention, an activation other than `silu`, an `o_proj` bias (HF Llama's `attention_bias = true`), a
checkpoint that carries a projection bias the recipe does not map.

`[tuning]` sets the prefill chunk (tokens per forward pass) a model loads with. Resolution order: the explicit
`--prefill-chunk` flag, then `[tuning.<backend>] prefill_chunk` for the running backend, then `[tuning] prefill_chunk`,
then 256; a backend without a table falls back to the base value. Load prints the result and its source (`prefill chunk
2048 (recipe qwen3_5_moe, cuda)`), `intarsia plan show --json` carries `prefill_chunk` and `prefill_chunk_source`, and
`accuracy` records both in baselines and self-goldens. Only tune a chunk with measurements behind it and cite them in a
comment: `recipes/qwen3_5_moe.toml` sets 2048 on CUDA (`docs/results/2026-10-13-prefill-cuda.md`) and 4096 on HIP
(`docs/results/2026-10-17-hip-prefill.md`; it fits a 32 GB R9700 at `--max-seq` 17408), `recipes/qwen3.toml` sets 2048 on Metal and HIP and 4096 on CUDA (`docs/results/2026-10-22-ck-cuda8b-chunk.md`), because the
chunk changes rounding (prefix-reuse and baseline results are chunk-sensitive). (Updated 2026-10-21: this paragraph used to say
CUDA only and that 2048 on the 35B was not shown to fit an R9700.) A baseline recorded at another chunk is refused with a message naming the recipe default;
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
`<out>/state.json`, and `--resume` skips the stages that finished (`PASS` or `FAIL`); a stage that ended in `SKIP`
or `ERROR` is run again, so a run that skipped the golden stages can be completed with `--resume --golden-dir DIR`. A stage that
FAILed is kept as FAIL by `--resume`, and so are the stages it blocked: after fixing the engine, start a new `--out`.

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
| quick | `acc-short` | `accuracy --golden` (prefill + teacher-forced decode) with the class thresholds; a MoE / hybrid / quantized model with a reference-server golden is also gated against the backend's committed baseline + envelope (`golden/baselines/`) and reads **MISSING-BASELINE** (exit 1) without one that applies | floor gate (+ baseline gate) | 8 s |
| quick | `acc-oracle` | `accuracy --golden oracle/<name>-cpu-oracle...` (prefill): the engine against the f64 CPU oracle, for a MoE / hybrid / quantized model (no oracle: MISSING-BASELINE with the `accuracy --backend cpu ... --write-self-golden` command that makes it, to commit under `golden/oracle/`), then a drift check against the recorded same-backend dump `golden/baselines/<oracle>.<backend>[-tpN].selfgolden.safetensors` | oracle KL on scored rows <= `--oracle-max-kl` (2e-2) and drift mean KL21 <= 1e-4; **MISSING-BASELINE** without the dump. Re-record the dumps when a change is meant to move the numerics (docs/results/2026-10-22-fx1-oracle-gate.md) | ~20-80 s (35B) |
| quick | `bench` | `serve` + `bench speed` at 128 / 1024 / 4096, byte floor | every size measured | 80 s |
| | | **quick section** | | **3 min 8 s** |
| full | `acc-long` | `accuracy --mode all` on the long golden: prefill, decode, prefix reuse | floor gate + reuse gate | 29 s |
| full | `acc-4k` | `accuracy --mode all` on the 4k golden (`<name>-4k...`, if there is one): prefill across chunks, decode, prefix reuse past 2 chunks of 2048; with it `acc-long` runs `--mode both` | KL floor + reuse gate + recorded baseline/envelope (no absolute top-1 floor) | skipped without a 4k golden |
| full | `envelope` | noise envelope (`chunk`, plus `attn-simt` on GPUs) and the baseline; then a self-check gate | the pair gates its own build | 24 s |
| full | `bench-long` | `serve` + `bench speed` at 1024 / 4096 / 16384 | every size measured | 83 s |
| full | `ab` | `localeval speed` against `--reference` and against our server | both ran | skipped (no `--reference`) |
| full | `tp` | with `--tp N`: tp plan, `bench speed`, accuracy against the short golden and, when there is one, the CPU oracle | accuracy gate + oracle floor | 71 s (tp 2) |
| | | **in-depth section** | | **3 min 26 s** (6 min 34 s in total) |

The numbers it produced for that run: decode 37.7 / 37.1 / 35.9 tok/s at 128 / 1024 / 4096 tokens (91 / 90 / 87 %
of the 41.4 tok/s byte floor), 32.2 tok/s at 16.8k (78 %), prefill 2.0k to 3.5k tok/s, accuracy against the HF
fp32 golden top-1 100.00 % / KL 2.3e-4 (short, prefill) and 99.58 % / 4.0e-4 (long), tp 2 decode 58.1 tok/s at 128
tokens. On the Mac, `--backend cpu --smoke` (Qwen3-Reranker-0.6B) takes 45 s for the quick and 2 min for the whole
pipeline (`docs/results/2026-10-13-qualify/mac-cpu-smoke/`).

What each stage needs and how it decides:

* **Goldens** (`acc-short`, `acc-long`, `acc-4k`, `envelope`, `tp`): `--golden` / `--golden-long` / `--golden-4k`, else found by model name
  in `--golden-dir` (default `./golden`, `/models/golden`, `$INTARSIA_GOLDEN_DIR`; an fp32 file is preferred
  over others, `<name>-long...` is the long set), else made: from `--reference http://...` (a vLLM-style server:
  `tools/golden_vllm.py`, top-20 `prompt_logprobs`) or, for a small dense bf16 model (at most `--make-golden-max-gb`
  5 GB of weights, no MoE / hybrid / quantisation), an HF fp32 golden with `tools/golden.py` (needs `uv` or a
  python with torch and transformers). Otherwise the stage is **skipped with the reason**; a model that no one can
  make a golden for needs `--golden` or `--reference`. A reference server must implement vLLM's `prompt_logprobs`
  (radiance 1.9.0 ignores it and returns no rows). For an AMD Quark MXFP4 checkpoint with no reference server and no
  bf16 base, `tools/golden_quark_mxfp4.py` makes the golden from HF's own Qwen3.5 decoder layers, one layer in memory at a
  time (~8.5 min and ~6 GB for Qwen3.8-27B on endor's CPU), with Quark's activation QDQ (`--act dynamic`, the declared
  W4A4) or without (`--act none`, W4A16, to compare with `--mxfp4-act none`); see
  docs/results/2026-10-21-m1-qwen38-drill.md. A W4A4 checkpoint (activations quantised at run time) does not pass
  the absolute class floor against any golden that is not bit-identical: rounding activations to fp4 amplifies
  last-bit differences, so the reference does not reproduce itself (Qwen3.8-27B: HF bf16 vs HF f32, both W4A4, 21-way
  KL 0.197 against a floor of 0.03). Gate the weight path in W4A16 instead (`--mxfp4-act none` against an `--act none`
  golden: 8.3e-4 there), and quote W4A4 against the reference's own spread (a second golden with `--dtype float32`). The short golden cannot exercise prefix reuse (the reuse
  gate needs a split of at least one prefill chunk, 256 tokens), so reuse is gated on the long golden, and there only one prompt
  (`long_prose_700`, continue and restore) is long enough to exercise it: a thin check, and none at all at a prefill
  chunk above ~350 (the long golden's gate prints SKIPPED, which fails). A **4k golden** (`<name>-4k...`, built with
  `tools/golden.py --prompts tools/prompts/golden_4k.jsonl`, never made by `qualify`; its `chat_reuse_4k` prompt splits at
  4447 tokens) takes the reuse gate over in `acc-4k`; when one is found, `acc-long` runs `--mode both` and `acc-4k` runs
  `--mode all`, which keeps a prefill chunk of 2048 (Metal) qualifiable. `acc-4k` has **no absolute top-1 floor** (no build reaches 99.5% on 4k prompts: the misses are near-ties of the fp32 reference); it gates KL (class floor), tail mass, non-finite rows, prefix reuse, and top-1 / KL against the backend's recorded baseline and noise envelope (`golden/baselines/<golden>.<backend>.baseline.json` / `.envelope.json`, recorded at the chunk the run uses), and with no matching baseline it is KL-only with a note. Without a 4k golden nothing changes. A stress run with
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
  `dequant_ref` exist. The checklist the MXFP4 drill (docs/results/2026-10-21-m1-qwen38-drill.md) followed:
  `quant/<fmt>.rs` (format, detect, load, quantize, test helpers) and its id in `quant/mod.rs` (`formats()`, `op()`,
  `dequant()`); a new `Op` appended to `Op::ALL` with its arity, `quant::ops::check`/`shape` arms and admission cases in
  `quant/cases.rs`; the oracle math in `quant/reference.rs` and the cpu dispatch (`intarsia-backend-cpu`
  `quant_ref.rs`, `reference.rs`, `fast.rs`); the declared scheme in `quant/scheme.rs` (`Algo`, `check_linear`) and
  where the loader reads it (`intarsia-models` `load.rs`); `recipe new`'s classifier (`cmd/recipe.rs` `Cls`); then per
  backend a launcher in `kernels/<backend>/`, its ABI declaration, ffi, dispatch and table entry. Test the reference
  against an independent decode of real tensor slices (`tools/gen_mxfp4_fixture.py`), then `intarsia kernels test`.
  Supported today: bf16, FP8 (per-tensor), NVFP4 (ModelOpt W4A16), MXFP4 (Quark OCP MX, W4A4 by default as declared,
  `--mxfp4-act none` for W4A16). Dense FFN, attention and GatedDeltaNet projections take any of them.
* A multimodal wrapper of a known text family (`*ForConditionalGeneration`, text weights under
  `model.language_model.*`) needs a copy of the family's recipe with the prefix changed (`recipes/qwen3_5_vl.toml`):
  recipes have no prefix parameter or inheritance. `recipe new` then writes the scaffold to the first free
  `recipes/<family>_<n>.toml`, not over the family's existing recipe.
* New block kind (MLA, sliding window, GDN, Mamba, MoE): add the block to the decoder in
  `crates/intarsia-core/src/model.rs`, its op(s) to `ops.rs` with a cpu-ref reference, and a
  `SlotSpec` if it carries state (`Recurrent` for GDN/Mamba). Then `intarsia kernels test` covers it.
* New kernel for an existing op: `kernels/README.md`.

`recipe new` names these cases in its TODO list; `qualify`'s `plan` stage names the ops that have no kernel.
