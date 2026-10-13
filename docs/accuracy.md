# Accuracy gates

Implements ARCHITECTURE section 6, levels 2 and 3 (admission, level 1, is `intarsia kernels test` and runs
on the model's own shapes at load, see `kernels/README.md`).

## Golden logits (model level)

Two reference sources, two file formats; `intarsia accuracy` reads both.

### HF transformers: `tools/golden.py` (format `intarsia-golden-v1`, full vocabulary)

Runs the Hugging Face `transformers` model in **bf16** (one forward per prompt, whole sequence at once),
upcasts the logits to fp32, takes `log_softmax` over the vocab and stores it as fp16.

```
cd tools
uv run golden.py --model <HF model dir> --out ../golden/<name>.safetensors [--device cpu|cuda]
                 [--prompts a.jsonl b.jsonl] [--dtype float32] [--topk 20]
```

On a GPU box without a Python env, run it inside the vLLM image (torch + transformers + safetensors are
there), through `fleetq`, with `golden.py`, `goldenlib.py` and `prompts/` copied next to each other
(`~/intarsiaml-golden/` on spark1, see `run.sh` there). `golden/*.safetensors` is git-ignored.

| tensor | dtype | shape | content |
|---|---|---|---|
| `ids_<i>` | I32 | `[n_i]` | token ids of prompt i |
| `logprobs_<i>` | F16 | `[n_i, vocab]` | row t = `log_softmax(logits_t)` of the fp32-upcast reference logits, i.e. the distribution over token t+1 |

`__metadata__` (all strings): `format`, `model`, `torch`, `transformers`, `device`, `dtype`, `template`,
`vocab`, and `prompts` (JSON list of `{name, kind, n_tokens, text}`). `vocab` is the lm_head width, which can
exceed the tokenizer vocab; our engine must produce the same width. fp16 rounds log-probs (relative 5e-4);
the checker renormalises each reference row before computing KL to remove the rounding bias.

A full-vocabulary row is about 300 KB, so a golden costs `positions x vocab x 2` bytes: the default prompt
set (1434 positions) is 435 MB at vocab 151,669. Keep it out of nearly full disks (spark1) or use top-k.

### Top-k goldens: `tools/golden_vllm.py`, `golden.py --topk K` (format `intarsia-golden-topk-v1`)

For reference servers that can only report the top-k log-probs (vLLM's `prompt_logprobs`), and to keep goldens
small. `golden_vllm.py` builds one from any OpenAI-compatible vLLM server:

```
python3 tools/golden_vllm.py --base-url http://spark-4e77:8000/v1 --model guidance \
    --out golden/qwen36-35b-a3b-nvfp4-vllm-top20.safetensors        # inside `fleetq run --uses guidance`
```

It tokenizes each prompt with the server's `/tokenize`, then sends one `/v1/completions` request with the ids
plus one dummy token, `max_tokens=1` and `prompt_logprobs=20`. vLLM reports, for prompt position j >= 1, the
top-20 of the distribution that predicts token j, so row t (the distribution after `ids[..=t]`) is
`prompt_logprobs[t+1]`; the dummy last token exists only to obtain the row for the final real position
(causal attention: it cannot change earlier rows). Entries outside the top-20 that vLLM adds for the actual
prompt token are dropped. `golden_topk.py` converts a full golden to top-k (used to test the comparator).

| tensor | dtype | shape | content |
|---|---|---|---|
| `ids_<i>` | I32 | `[n_i]` | token ids of prompt i |
| `topk_ids_<i>` | I32 | `[n_i, K]` | the K most likely token ids of row t, best first; -1 pads |
| `topk_logprobs_<i>` | F32 | `[n_i, K]` | their log-probs under the **full-vocabulary** softmax as the source reports them; -1e30 pads |

`__metadata__` as above plus `topk`, `source` (`vllm`), `server`, `served_model`.

**Provenance metadata** (all strings; older goldens lack it and `intarsia accuracy` says "not recorded"):
`engine` (`hf-transformers` / `vllm`), `engine_version`, `model`, `model_revision` (HF snapshot hash when the
path has one), `kv_dtype` (the reference's KV cache dtype: for vLLM the `cache_dtype` of
`vllm:cache_config_info`, for HF the activation dtype), `cache_config_info` (all labels of that metric),
`max_model_len`, `launch_flags` (vLLM's are not observable through the API; `golden_vllm.py --note` records
what you know), `template`, `add_special_tokens`, `sampling` (prompt_logprobs, temperature, how the
continuation was made), `prompt_files`, `skipped` (prompts that did not fit `max_model_len`), `built_unix`.
Per prompt, `prompts` also carries `n_prompt`, `continuation` and `reuse_split` (below). `intarsia accuracy`
prints this block and **warns when the reference's KV dtype differs from ours** (aliases are normalised: `fp8`
= `fp8_e4m3`, `bfloat16` = `bf16`; vLLM's `auto` counts as a 16-bit cache). The live `guidance` vLLM runs an
**fp8 KV cache** (`cache_dtype="fp8"`, prefix caching on) while we run bf16 (the recipe's `kv_dtype`): part of every
KL against that reference is the reference's own KV quantisation noise, and it cannot be separated from our error
until a reference with `--kv-cache-dtype auto` exists. That warning is printed on every 35B run.

### How a top-k golden is compared (tail bucket)

A top-k reference is compared with three numbers, none of which is the old one alone:

* `KL(supp)`: KL over the reference's top-k support with both sides renormalised there (the number reported
  before). It is **blind to probability mass that moves outside the top-k**: if a bug lifts thousands of rare-token
  logits from -10 to 0, most of our mass leaves the support while this KL stays near 0 (unit test
  `mass_leaving_the_top_k_is_invisible_...`).
* `KL(21-way)`: KL over the partition {the k reference ids, everything else}. vLLM reports full-vocabulary
  log-probs, so the reference's "other" bucket is `1 - sum(top-k)` and ours is our softmax mass outside the same
  ids (summed directly). By the chain rule this is the 2-way {support, tail} KL plus `P_ref(support)` times
  `KL(supp)`: a true lower bound of the full-vocabulary KL, and the tightest one available. **This is the gated KL
  for top-k goldens.** For full-vocabulary goldens the same partition is built from the reference's top-20 (so
  numbers are comparable), while the gated KL there is the exact full-vocabulary one.
* `tail ref` / `tail us`: the mean probability each side puts outside the support, and the gate on
  `|mean(tail us - tail ref)|` (`--max-tail-excess`, default 0.02, all classes). The reference tail is read from
  f32 log-probs, so it is resolved to about 1e-6.

Top-1 is our full-vocabulary argmax against the reference's best id; top-5 overlap compares our top-5 with the
reference's top-5. The top-k files are small (0.25 MB for 10 prompts, 6 MB for the long set), so they live in
the repo copy on every box; spark1 keeps them in `~/intarsiaml-golden/`.

### Prompt set

`tools/prompts/golden.jsonl` (8 prompts of 18-90 tokens: counting, story, code, science, and four chat prompts)
plus `tools/prompts/golden_long.jsonl` (two prose prompts of 300 and 700 tokens cut from `ppl.txt`, so the
default 256-token prefill chunk is crossed twice and three times). One JSON object per line:
`{"name", "kind": "raw", "text"}`, `{"name", "kind": "raw", "file": "ppl.txt", "skip_tokens", "max_tokens"}`
or `{"name", "kind": "chat", "messages": [...]}`. Chat prompts are rendered with a fixed ChatML text
(`--template chatml`), independent of the model's own template, so the token ids are stable. Tokenisation uses
`add_special_tokens=True`; prompts must be 8-16384 tokens. Shared code: `tools/goldenlib.py`.

**Long sets** (review finding 4: nothing used to go past 700 tokens or 16 decode steps):
`tools/prompts/golden_4k.jsonl` (a 4000-token docs prompt, a 4000-token code prompt, and a 4.5k-token two-turn
chat) and `golden_12k.jsonl` (12000-token docs and code prompts). Their text comes from frozen corpora
(`corpus_docs.txt`: the repo's docs, `corpus_code.txt`: two Rust sources; the ids are in the golden, so editing
the repo later changes nothing). Every prompt has `"continue": 64`: the golden tool appends 64 tokens of the
**reference's own greedy continuation**, so the teacher-forced decode window (`--decode-positions 64`) is on the
model's distribution rather than on arbitrary document text. The chat prompt has `"reuse_after_message": 2`:
the first two messages plus the assistant header are "turn 1" and the golden records that token count as
`reuse_split` for `--mode reuse`. Chat messages may take `content_file`/`skip_chars`/`max_chars` instead of
`content`.

```
# vLLM (35B), inside the lease; the server must have max_model_len >= 12064+1 or the prompt is skipped, loudly
python3 tools/golden_vllm.py --base-url http://spark-4e77:8000/v1 --model guidance \
    --prompts tools/prompts/golden_4k.jsonl tools/prompts/golden_12k.jsonl \
    --out golden/qwen36-35b-a3b-nvfp4-vllm-long-top20.safetensors
# HF (8B), top-20 only (a full-vocabulary 4k golden is over a GB per prompt)
docker run ... golden.py --model <8B> --device cuda --topk 20 --prompts prompts/golden_4k.jsonl --out ...
```

## `intarsia accuracy`

```
intarsia accuracy --model DIR --backend cpu|cuda [--golden FILE]      # golden gate, both modes
intarsia accuracy --model DIR --backend cpu|cuda --quick               # day-0 gate
```

`--golden` defaults to `golden/<model name>.safetensors` (the model name is the lower-cased HF repo name
when the dir is an HF cache snapshot, else the dir name).

### Modes (`--mode prefill|decode|both|reuse|all`, default both)

* **prefill**: `Engine::forward_logits_each` over the stored ids (state reset per prompt, all-position
  logits): the prefill kernels (`AttnPrefill`, GEMM / chunked matmul), across chunk boundaries when the prompt
  is longer than `--prefill-chunk` (default 256; the tool prints how many prompts cross a boundary and warns
  when none does, then try `--prefill-chunk 64`).
* **decode (teacher-forced)**: `Engine::forward_logits_decode`: prefill `n - k` tokens, then feed the golden
  ids one at a time (`T = 1`) and compare the logits row of each step (`k = --decode-positions`, default 16,
  clamped to `n - 1`). This is the path serving uses (`AttnDecode`, GEMV matmul and logits), which a full
  forward never touches.

* **reuse** (`--mode reuse`, part of `all`): the prefix-reuse path of the real model through `Engine::generate`,
  what `serve` runs (logits do not travel through that path, so streams are compared). Per prompt whose split
  is at least one prefill chunk: *continue*: turn 1 = `ids[..s]` + a real 16-token reply, turn 2 = turn 1 + reply
  + the rest of the prompt (the engine must reuse what it holds, `cached >= s`); *restore*: turn 1 = `ids[..s]`
  + a different tail, turn 2 = `ids`, so the cache is truncated (attention) or a recurrent snapshot taken at a
  prefill chunk boundary is restored (GatedDeltaNet), `0 < cached <= s`. A cold run is forced by serving an
  unrelated prompt in between (`cached == 0` is checked) and the two greedy streams (`--reuse-gen` 32 tokens)
  must agree on `--reuse-min-match` (16) leading tokens; the first divergence is printed. `s` is the golden's
  `reuse_split`, else half of the prompt. A reuse run that has nothing to run on prints SKIPPED and fails.

Per prompt and overall: top-1 agreement (strict: our argmax equals the reference argmax; positions that differ
only because the fp16 reference scores two tokens identically are counted as disagreement and reported
separately), mean and max KL(ref || ours) in nats (f64), top-5 overlap, the **greedy-match length** and the
number of non-finite rows. Greedy-match length is the number of leading positions, in order, where our argmax
equals the reference argmax, i.e. how many tokens greedy decoding would track the reference before the first
divergence; it is informational (no gate), and in prefill mode it starts at the first prompt token, where
distributions are flat, so the decode-mode figure is the meaningful one.

**Any NaN or infinity anywhere in one of our logits rows fails that position** (KL counts as infinity, top-1 as
a miss) and the gate fails whatever the other numbers say. A NaN in the golden is an error, not a pass.

### Gates

Default thresholds follow the weaker of the weight dtype and the KV dtype (`--gate-class bf16|fp8|4bit`
overrides the detection; weights are classified from `quantization_config` in `config.json` and
`hf_quant_config.json`, none = bf16 class):

| class | top-1 | mean KL(ref \|\| ours) |
|---|---|---|
| bf16 (vs HF bf16) | >= 99.5% | <= 1e-3 |
| fp8 (weights or `--kv-dtype fp8_e4m3`) | reported, not gated | <= 5e-3 |
| 4-bit (NVFP4 / MXFP4) | reported, not gated | <= 3e-2 |

ARCHITECTURE section 6 sets a top-1 threshold only for BF16, so none is invented for the other classes;
`--min-top1` / `--max-mean-kl` override any of it. The exit status is non-zero when a gate fails. The "4-bit
... and benchmark deltas within noise" part of section 6 needs the task-level suites, not this tool.

The KL threshold applies to the **gated KL**: the 21-way KL for a top-k golden, the full-vocabulary KL for a full
golden. Next to it the floor gates `|tail excess| <= --max-tail-excess` (0.02) and "no non-finite row". These are
**absolute floors** and loose for what they guard against: the 4-bit class allows a quantisation error against a
different reference, so a speed change that adds 2e-2 KL on the same weights passes it. Use a baseline for that.

### Baseline-relative gate (`--write-baseline`, `--baseline`)

```
intarsia accuracy ... --golden G --write-baseline golden/baselines/NAME.json   # record this run
intarsia accuracy ... --golden G --baseline golden/baselines/NAME.json         # no worse than that, plus margins
```

A baseline stores, per mode and per prompt: positions, top-1, mean KL, 21-way KL, top-5, tail masses, max KL,
greedy match; the identity of the golden (file name, size, FNV-1a 64 of its bytes, prompt and position counts,
its provenance metadata); and the run configuration (model, backend, device, KV dtype, activation dtype, fp8
mode, prefill chunk, decode positions, gate class). It is written even when the absolute floor fails in that run
(it then records `floor_pass: false`). The gate then requires, per mode, with these default margins (flags in
brackets):

| metric | allowed regression |
|---|---|
| top-1 | -0.3 pt (`--margin-top1`), but never less than one position (100/n pt: one flip must not fail a 160-position mode) |
| mean KL, 21-way KL | x1.15 (`--margin-kl`) and at least +1e-6 absolute (a baseline near zero is not brittle) |
| top-5 | -0.5 pt (`--margin-top5`), at least five overlap units (100/n pt) |
| `\|tail excess\|` | may grow by 0.002 (`--margin-tail`) over the baseline's |
| non-finite rows | none |

It **refuses to compare** (before running anything) when the golden differs (hash, size, prompt or position
count: a regenerated golden needs a new baseline) or when the KV dtype, backend, activation dtype, fp8 mode,
prefill chunk or decode positions differ (`--allow-config-mismatch` downgrades the configuration part to a
warning; the golden must always match). A different device name is a warning only. The output keeps the two
verdicts apart: `floor verdict` (class thresholds) and `baseline verdict`, and when the baseline gate fails it
lists the prompts that moved most. Record a baseline **before** changing kernels, and verify it in a second
process (the engine is deterministic run to run, so a run passes its own baseline with equal numbers).
Baselines live in `golden/baselines/` (tracked, unlike the large `*.safetensors`).

### `--reference`

`--reference` makes every op run its `ref_*` kernel (the f64 oracle) instead of the fast cpu kernels (before
this flag `--backend cpu` ran `cpu_f32_*` and the "cpu-ref matches HF" claim was about those). It is slow;
use it on short goldens, or with `--decode-positions`. On a backend with no reference kernels (cuda) the load
fails.

### Quick gate (`--quick`)

* Perplexity over `tools/prompts/ppl.txt` (about 1.2k tokens of public-domain English, embedded in the
  binary; override with `--ppl-file`); must be <= `--max-ppl` (default 40), and no logits row may be non-finite.
* Five greedy sanity prompts (counting, story, a completion, and two fact prompts: "The capital of France is" must answer Paris and "...two hydrogen atoms and one" oxygen, within the first 16 tokens), `--sanity-tokens` (48) each: output must be
  non-blank and no 4-gram of generated token ids may occur more than 3 times (fact prompts are exempt from the repeat check).

## Reading the numbers: the reference has its own noise

HF bf16 is not ground truth. On Qwen3-Reranker-0.6B (the Mac test model, reranker-tuned, distributions unusually
flat on chat prompts) the HF bf16 forward itself differs from the HF fp32 forward by mean KL 8.7e-4 and 97.9%
top-1 on the short prompt set, so the 1e-3 bf16-vs-HF gate leaves little margin on small models. To separate
"our engine is wrong" from "the bf16 reference is noisy", generate a second golden with `golden.py --dtype
float32`; an engine that is right sits clearly closer to fp32 than the bf16 reference does.

Measured 2026-10-12 with the strict scoring above, Mac, cpu fast kernels, 10 prompts (1434 positions; decode
mode = last 16 positions of each prompt, 160 positions), activations bf16:

| model | golden | mode | top-1 | mean KL | max KL | gate (bf16 class) |
|---|---|---|---|---|---|---|
| Qwen3-Reranker-0.6B | HF bf16 | prefill | 96.79% | 2.38e-3 | 4.9e-2 | FAIL |
| Qwen3-Reranker-0.6B | HF bf16 | decode | 97.50% | 2.39e-3 | 4.7e-2 | FAIL |
| Qwen3-Reranker-0.6B | HF fp32 | prefill | 98.40% | 9.66e-4 | 3.6e-2 | FAIL (top-1) |
| Qwen3-Reranker-0.6B | HF fp32 | decode | 99.38% | 9.11e-4 | 2.2e-2 | FAIL (top-1) |
| Qwen3-8B, cuda (spark1) | HF bf16 (cuda) | prefill | 99.44% | 7.75e-4 | 5.3e-2 | FAIL (top-1: 8 of 1434 positions short of 99.5%) |
| Qwen3-8B, cuda (spark1) | HF bf16 (cuda) | decode | 100.00% | 8.69e-4 | 1.8e-2 | PASS |

The earlier "99.08% / 1.25e-3" figures counted fp16 ties as agreement and used 8 short prompts; the numbers
above are not comparable with them. Results are identical for `--prefill-chunk` 64, 256 and 1024 (chunking is
not the cause). The Qwen3-8B rows are the CUDA backend (admitted kernels: cuBLAS prefill GEMM, GEMV decode) against an HF bf16 golden produced on the same box; the prefill gate misses the 99.5% top-1 default by 0.06 percentage points while mean KL passes, and the defaults were not loosened. Judge new backends mainly by how they compare
with `cpu` on the same golden file.

## Where the goldens live

* Mac `golden/` (git-ignored `*.safetensors`): `qwen3-reranker-0.6b-long.safetensors` (HF bf16),
  `...-long-fp32.safetensors` (HF fp32), the older 8-prompt `qwen3-reranker-0.6b*.safetensors`, and the two vLLM
  top-20 goldens of the 35B (small): `qwen36-35b-a3b-nvfp4-vllm-top20.safetensors` (10 short prompts, rebuilt
  2026-10-13 with provenance metadata) and `qwen36-35b-a3b-nvfp4-vllm-long-top20.safetensors` (4k and 12k prompts +
  64 greedy continuation tokens each, vLLM 0.26.1rc1 `guidance`, fp8 KV cache).
* spark1: `~/intarsiaml-golden/` (outside the repo copy on purpose: syncing the repo with `rsync --delete`
  would remove it): the two vLLM goldens above, `qwen3-8b-full.safetensors` (8 short + 2 long prompts, full
  vocabulary, HF bf16, with metadata, 435 MB), `qwen3-8b-long.safetensors` / `qwen3-8b.safetensors` (older, no
  metadata), and, to be built there, the 8B long golden `qwen3-8b-4k-top20.safetensors` (`golden.py --topk 20 --prompts
  prompts/golden_4k.jsonl` in the `vllm/vllm-openai:v0.28.0-aarch64-cu129` image; `run8b.sh` in that directory has the command;
  NOT built yet: spark1 went offline during that run).
  Pass them with `--golden`.
* `golden/baselines/*.json` (tracked): recorded runs, named after golden and model; see the baseline section.
