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

Comparison against a top-k golden is stated honestly as what it is: KL is computed **over the reference's
top-k support**, with both distributions renormalised over those k ids (a lower bound of the full-vocabulary
KL, blind to mass the reference puts outside its top-k); top-1 is our full-vocabulary argmax against the
reference's best id; top-5 overlap compares our top-5 with the reference's top-5. The files are small
(0.25 MB for 10 prompts), so they live in the repo copy on every box; the NVFP4 one is
`~/intarsiaml-golden/qwen36-35b-a3b-nvfp4-vllm-top20.safetensors` on spark1.

### Prompt set

`tools/prompts/golden.jsonl` (8 prompts of 18-90 tokens: counting, story, code, science, and four chat prompts)
plus `tools/prompts/golden_long.jsonl` (two prose prompts of 300 and 700 tokens cut from `ppl.txt`, so the
default 256-token prefill chunk is crossed twice and three times). One JSON object per line:
`{"name", "kind": "raw", "text"}`, `{"name", "kind": "raw", "file": "ppl.txt", "skip_tokens", "max_tokens"}`
or `{"name", "kind": "chat", "messages": [...]}`. Chat prompts are rendered with a fixed ChatML text
(`--template chatml`), independent of the model's own template, so the token ids are stable. Tokenisation uses
`add_special_tokens=True`; prompts must be 8-4096 tokens. Shared code: `tools/goldenlib.py`.

## `intarsia accuracy`

```
intarsia accuracy --model DIR --backend cpu|cuda [--golden FILE]      # golden gate, both modes
intarsia accuracy --model DIR --backend cpu|cuda --quick               # day-0 gate
```

`--golden` defaults to `golden/<model name>.safetensors` (the model name is the lower-cased HF repo name
when the dir is an HF cache snapshot, else the dir name).

### Modes (`--mode prefill|decode|both`, default both)

* **prefill**: `Engine::forward_logits_each` over the stored ids (state reset per prompt, all-position
  logits): the prefill kernels (`AttnPrefill`, GEMM / chunked matmul), across chunk boundaries when the prompt
  is longer than `--prefill-chunk` (default 256; the tool prints how many prompts cross a boundary and warns
  when none does, then try `--prefill-chunk 64`).
* **decode (teacher-forced)**: `Engine::forward_logits_decode`: prefill `n - k` tokens, then feed the golden
  ids one at a time (`T = 1`) and compare the logits row of each step (`k = --decode-positions`, default 16,
  clamped to `n - 1`). This is the path serving uses (`AttnDecode`, GEMV matmul and logits), which a full
  forward never touches.

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

* Mac: `golden/qwen3-reranker-0.6b-long.safetensors` (HF bf16), `...-long-fp32.safetensors`
  (HF fp32) and the older 8-prompt `qwen3-reranker-0.6b*.safetensors` (git-ignored).
* spark1: `~/intarsiaml-golden/` (outside the repo copy on purpose: syncing the repo with `rsync --delete`
  would remove it): `qwen3-8b.safetensors` (8 prompts), `qwen3-8b-long.safetensors`, and
  `qwen36-35b-a3b-nvfp4-vllm-top20.safetensors`. Pass them with `--golden`.
