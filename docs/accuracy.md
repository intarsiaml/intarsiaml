# Accuracy gates

Implements ARCHITECTURE section 6, levels 2 and 3 (admission, level 1, is `intarsia kernels test`).

## Golden logits (model level)

`tools/golden.py` runs the Hugging Face `transformers` model in **bf16** (one forward per prompt, whole
sequence at once), upcasts the logits to fp32, takes `log_softmax` over the vocab and stores it as fp16.

```
cd tools
uv run golden.py --model <HF model dir> --out ../golden/<name>.safetensors [--device cpu|cuda]
```

On a GPU box without a Python env, run it inside the vLLM image (torch + transformers + safetensors are
there), e.g. `docker run --gpus all --entrypoint python3 -v <hf cache>:/hf:ro -v <repo>:/work
vllm/vllm-openai:v0.28.0-aarch64-cu129 /work/tools/golden.py --model /hf/hub/... --out /work/golden/x.safetensors
--device cuda` (through `fleetq`). `golden/*.safetensors` is git-ignored (about 130 MB for 8 prompts).

### Prompt set

`tools/prompts/golden.jsonl`, one JSON object per line: `{"name", "kind": "raw", "text"}` or
`{"name", "kind": "chat", "messages": [...]}`. Eight prompts of 18-90 tokens: counting, story, code, science
(raw) and four chat prompts (short, system prompt, multi-turn, reasoning). Chat prompts are rendered with a
fixed ChatML text by default (`--template chatml`), independent of the model's own template, so the token ids
are stable; `--template hf` uses `tokenizer.apply_chat_template`. Tokenisation uses
`add_special_tokens=True`. Prompts must be 8-256 tokens.

### File format (`intarsia-golden-v1`)

A standard safetensors file:

| tensor | dtype | shape | content |
|---|---|---|---|
| `ids_<i>` | I32 | `[n_i]` | token ids of prompt i |
| `logprobs_<i>` | F16 | `[n_i, vocab]` | row t = `log_softmax(logits_t)` of the fp32-upcast reference logits, i.e. the distribution over token t+1 |

`__metadata__` (all strings): `format`, `model`, `torch`, `transformers`, `device`, `dtype`, `template`,
`vocab`, and `prompts` (JSON list of `{name, kind, n_tokens, text}`). `vocab` is the lm_head width, which can
exceed the tokenizer vocab; our engine must produce the same width.

fp16 rounds log-probs (relative 5e-4). The checker renormalises each reference row before computing KL to
remove the rounding bias.

## `intarsia accuracy`

```
intarsia accuracy --model DIR --backend cpu|cuda [--golden FILE]      # golden gate
intarsia accuracy --model DIR --backend cpu|cuda --quick               # day-0 gate
```

`--golden` defaults to `golden/<model name>.safetensors` (the model name is the lower-cased HF repo name
when the dir is an HF cache snapshot, else the dir name). The tool runs `Engine::forward_logits_each` on the
stored ids (state reset per prompt, all-position logits) and reports per prompt and overall:

* top-1 agreement (percent of positions where our argmax equals the reference argmax; a tie in the fp16
  reference counts as agreement),
* mean and max KL(ref || ours) in nats, computed in f64 from the full distributions,
* top-5 overlap (average |top5(ref) and top5(ours)| / 5, percent).

The exit status is non-zero when a gate fails.

| flag | default | meaning |
|---|---|---|
| `--min-top1` | 99.0 | minimum overall top-1 agreement, percent |
| `--max-mean-kl` | 1e-3 on `cpu`, 5e-3 otherwise | maximum overall mean KL |

### Quick gate (`--quick`)

* Perplexity over `tools/prompts/ppl.txt` (about 1.2k tokens of public-domain English, embedded in the
  binary; override with `--ppl-file`); must be <= `--max-ppl` (default 40).
* Three greedy sanity prompts (counting, story, a completion), `--sanity-tokens` (48) each: output must be
  non-blank and no 4-gram of generated token ids may occur more than 3 times.

## Reading the numbers: the reference has its own noise

HF bf16 is not ground truth. On Qwen3-Reranker-0.6B (the Mac test model) the HF bf16 forward itself differs
from the HF fp32 forward by mean KL 8.7e-4 and 97.9% top-1 on this prompt set, so the 1e-3 bf16-vs-HF gate
leaves little margin on small models. To separate "our engine is wrong" from "the bf16 reference is noisy",
generate a second golden with `golden.py --dtype float32` and compare against that as well; an engine that
is right sits clearly closer to fp32 than the bf16 reference does.

| model / backend | golden | top-1 | mean KL | max KL |
|---|---|---|---|---|
| Qwen3-Reranker-0.6B, cpu-ref | HF bf16 | 99.08% | 1.25e-3 | 4.9e-2 |
| Qwen3-Reranker-0.6B, cpu-ref | HF fp32 | 99.08% | 5.5e-4 | 3.6e-2 |
| Qwen3-8B (spark1), cpu-ref | HF bf16 (cuda) | 99.77% | 6.2e-4 | 2.1e-2 |

The first row misses the 1e-3 mean-KL gate by noise in the reference (see above), the second passes it. The
Mac test model is reranker-tuned and its distributions are unusually flat on chat prompts; judge new
backends mainly by how they compare with `cpu` on the same golden file.

## Where the goldens live

* Mac: `golden/qwen3-reranker-0.6b.safetensors` (git-ignored).
* spark1: `~/intarsiaml-golden/qwen3-8b.safetensors` (outside the repo copy on purpose: syncing the repo with
  `rsync --delete` would remove it). Pass it with `--golden`.
