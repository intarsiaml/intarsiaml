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

**Prefix caching in the golden request.** `golden_vllm.py` re-sends prompt plus continuation to a server that runs with
`enable_prefix_caching=True` (the `guidance` vLLM). Whether a `prompt_logprobs` request reads cached fp8 KV blocks of
an earlier request is unverified for vLLM 0.26 (Phase 1b review finding 10b), and the goldens built so far recorded
nothing about it. `--cache-salt SALT` sends vLLM's per-request `cache_salt` (`SALT-<prompt index>-<n>`) so a request
cannot share blocks; it is recorded in the `sampling` metadata, and without it the metadata says the cache was not
controlled. The flag has not been run against a live server; rebuild the goldens with it before relying on a
KV-noise-free comparison.

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
  is longer than `--prefill-chunk` (default: the recipe's value for the backend, else 256; the tool prints how many prompts cross a boundary and warns
  when none does, then try `--prefill-chunk 64`).
* **decode (teacher-forced)**: `Engine::forward_logits_decode`: prefill `n - k` tokens, then feed the golden
  ids one at a time (`T = 1`) and compare the logits row of each step (`k = --decode-positions`, default 16,
  clamped to `n - 1`). This is the path serving uses (`AttnDecode`, GEMV matmul and logits), which a full
  forward never touches.

* **reuse** (`--mode reuse`, part of `all`): the prefix-reuse path of the real model through `Engine::generate`,
  what `serve` runs (the greedy streams are compared; the logits are recorded only to judge a divergence). Per
  prompt whose split is at least one prefill chunk: *continue*: turn 1 = `ids[..s]` + a real 16-token reply, turn 2 =
  turn 1 + reply + the rest of the prompt (the engine must reuse what it holds, `cached >= s`); *restore*: turn 1 =
  `ids[..s]` + a different tail, turn 2 = `ids`, so the cache is truncated (attention) or a recurrent snapshot taken
  at a prefill chunk boundary is restored (GatedDeltaNet), `0 < cached <= s`; *aligned*: *restore* with `s` rounded
  down to a multiple of the prefill chunk. A cold run is forced by serving an unrelated
  prompt in between (`cached == 0` is checked) and the two greedy streams (`--reuse-gen` 32 tokens) are compared.
  `s` is the golden's `reuse_split`, else half of the prompt. A reuse run that has nothing to run on prints SKIPPED
  and fails. `--reuse-prompt a,b` runs only those prompts.
  * Identical streams pass. In *aligned* the warm run prefills exactly the chunks of the cold run, so on a backend
    whose cold run repeats itself bit for bit (checked from the recorded logits when the streams differ) the streams
    must be identical (`--reuse-exact`, default on).
  * A diverging pair passes if it is a rounding tie, judged on the logits (`judge_streams`), or if it agrees on
    `--reuse-min-match` leading tokens as before: the logit criterion can only rescue, never fail what passed. Both runs are repeated with the logits row of every
    step recorded; rows `0..d` share one context (`d` the first divergence), so any difference between them is
    numerical. The divergence is a *rounding tie* when every shared row agrees within `--reuse-logit-tol` (0.25 nats of
    log-probability over the top-5 tokens, raised to twice what re-chunking the cold run alone moves them) and
    `--reuse-kl-tol` (KL(cold || warm) <= 1e-2): the two top candidates were that close and the last bit of the KV
    rows decided. The tolerances are absolute and calibrated on bf16 Qwen3-8B (not scaled by the model class: a 4-bit or hybrid
    model may need larger ones, which is why the leading-token rule stays as the other way through). A broken restore
    (wrong position, stale row, off-by-one chunk) moves the logits by far more (measured in the report) and fails; the unit tests `prefix_reuse_*` in `crates/intarsia-models/tests/engine.rs` pin the
    engine's bookkeeping bitwise. A divergence that does not reproduce when the logits are recorded is accepted with
    that note (the run is not repeatable).
  * Why not "N identical leading tokens" (`--reuse-min-match`, still the fallback for an engine that cannot record
    logits): a *continue* run reuses KV rows the decode kernels wrote (the reply), which a cold run prefills with
    the GEMM / prefill-attention kernels; the two differ in the last bf16 bit (KL about 3e-4, the size of the golden
    gate's decode-against-prefill difference), and a greedy stream flips whenever two candidates are closer than that,
    at any token (`docs/results/2026-10-13-prefix-reuse.md`).
  * `--reuse-diag` prints, per case, the logit-level comparison (bitwise rows, argmax flips, KL) of the engine's
    warm / cold runs against `Engine::forward_script` replays: repeatability, engine against replay, warm against
    cold, cold at other chunkings, warm with the reply prefilled instead of decoded.

Per prompt and overall: top-1 agreement (strict: our argmax equals the reference argmax; positions that differ
only because the fp16 reference scores two tokens identically are counted as disagreement and reported
separately), mean and max KL(ref || ours) in nats (f64), top-5 overlap, the **greedy-match length** and the
number of non-finite rows. Greedy-match length is the number of leading positions, in order, where our argmax
equals the reference argmax, i.e. how many tokens greedy decoding would track the reference before the first
divergence; it is informational (no gate), and in prefill mode it starts at the first prompt token, where
distributions are flat, so the decode-mode figure is the meaningful one.

**Any NaN or infinity anywhere in one of our logits rows fails that position** (KL counts as infinity, top-1 as
a miss) and the gate fails whatever the other numbers say. A NaN in the golden is an error, not a pass.

`INTARSIA_ACC_POS_DUMP=<file>` (development aid, any mode) appends one tab-separated line per compared position
(`mode prompt pos token top1_agrees kl21 tail_ref tail_ours`), to see where along a prompt a difference sits (chunk
boundaries, special tokens, document regions). Run it once with `--golden` and once with `--self-golden` to locate the
positions where two kernels or two chunk sizes disagree.

### Which golden to gate against

* **Dense models (no experts, no linear-attention layers: Qwen3-8B, the 0.6B Mac model): the HF fp32 golden**
  (`golden.py --dtype float32`), gate class bf16, plus the baseline gate with its fixed margins. The HF bf16 golden carries
  about as much noise as the engine under test (HF bf16 against HF fp32: mean KL 5.6e-4 on the 8 short Qwen3-8B prompts,
  6.3e-4 with the two long ones, against a floor of 1e-3), so a bf16 golden passes or fails the floor on its own noise.
  Against the fp32 goldens the CUDA engine measures 2.9e-4 / 4.0e-4 (prefill) and passes both modes (table in "Reading the
  numbers"). Dense models are also quiet: old and new build differ by a mean KL of about 1e-10 on the 8B, which is what the
  fixed per-prompt margins were calibrated on. The 8B goldens are `qwen3-8b-fp32.safetensors` (8 short prompts) and
  `qwen3-8b-long-fp32.safetensors` (those plus `long_prose_300/700`); `scripts/phase1b-acceptance.sh` gates its `8b-short`
  and `8b-long` cases against them with `--gate-class bf16`.
* **MoE / hybrid / quantised models (the 35B NVFP4): the vLLM top-20 golden** (the only reference that runs the
  checkpoint), the class floor, a baseline from the pre-change build **and a noise envelope** for the per-prompt part
  ("Noise envelope" below).

### Gates

**Advisory gate (ADR-0008).** For Qwen3.6-35B-A3B-NVFP4 the `35b-short` baseline gate against the vLLM golden is advisory:
`scripts/phase1b-acceptance.sh` reports its FAIL as `ADVISORY FAIL (ADR-0008)` and does not fail the run. The failures were
measured as noise (`docs/results/2026-10-14-35b-accuracy.md`), and no gate tried resolves a 5 % attention bug, so loosening
it would not catch bugs. All other gates, including the `35b-short` floor, `35b-long` and the kernel-level tests, stay blocking.

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

A baseline stores, per mode and per prompt: positions, top-1, mean KL, 21-way KL, top-5, tail masses, the mean
per-position |tail difference|, the largest single-position 21-way KL, greedy match; the identity of the golden (file
name, size, FNV-1a 64 of its bytes, prompt and position counts, its provenance metadata); the run configuration (model,
backend, device, KV dtype, activation dtype, fp8 mode, prefill chunk, decode positions, gate class); **the code that
produced it** (below); and how it was executed (command line, kernel disable list, resolved kernel per op). It is written
even when the absolute floor fails in that run (it then records `floor_pass: false`). The gate then requires, per mode,
with these default margins (flags in brackets):

| metric | allowed regression |
|---|---|
| top-1 | -0.3 pt (`--margin-top1`), but never less than one position (100/n pt: one flip must not fail a 160-position mode) |
| mean KL, 21-way KL | x1.15 (`--margin-kl`) and at least +1e-6 absolute (a baseline near zero is not brittle) |
| top-5 | -0.5 pt (`--margin-top5`), at least five overlap units (100/n pt) |
| `\|tail excess\|` (signed mean) | may grow by 0.002 (`--margin-tail`) over the baseline's |
| mean `\|tail difference\|` per position | may grow by 0.002 (`--margin-tail`): the signed mean lets positions that lose and gain tail mass cancel |
| largest single-position 21-way KL | x2.5 (`--margin-max-kl`), at least +1e-3 (a maximum is a noisy statistic) |
| **each prompt**: mean 21-way KL | x1.75 (`--margin-prompt-kl`), at least +1e-5; with `--noise-envelope` at least baseline + k x the prompt's measured noise |
| **each prompt**: top-1 | -1 pt, but at least one position; and each prompt's `\|tail difference\|` +0.002; same widening with an envelope |
| non-finite rows | none |

The per-prompt rows exist because the aggregate hides one prompt that got worse: with 10 prompts, one doubling its KL moves
the mean by about 10% and passes the x1.15 gate (unit test `one_prompt_doubling_fails_although_the_overall_mean_passes`).
The x2.5 / x1.75 factors are the smallest that let a legitimate bf16 reordering through on the one model that could be
measured here and still fail a 2x larger change; see "Choosing the self-golden budgets" for the data and its limits.
**On MoE and hybrid models these fixed per-prompt margins are inside the noise** (the 35B moves one prompt by KL 0.13-0.17
between two equally correct builds); the aggregate checks were kept hard by design, but the same experiment measured them inside the noise on `35b-long` too (docs/results/2026-10-13-phase1b.md 10.0). The per-prompt part is calibrated with a noise envelope
("Noise envelope" below). Without one the gate keeps the fixed margins and warns when `config.json` shows experts or
linear-attention layers.

It **refuses to compare** (before running anything) when the golden differs (hash, size, prompt or position count: a
regenerated golden needs a new baseline), when the KV dtype, backend, activation dtype, fp8 mode, prefill chunk, decode
positions or kernel disable list differ (`--allow-config-mismatch` downgrades the configuration part to a warning; the
golden must always match), and when the code stamp is not a legitimate "before" (next section). A different device name
is a warning only. The output keeps the verdicts apart: `floor verdict` (class thresholds) and `baseline verdict`, and
for the per-prompt part it prints every prompt and check with baseline value, this run, the limit and where the limit came
from (`fixed` or `envelope`), plus the envelope delta and the `k` that would have been needed. Verify a baseline in a
second process (the engine is deterministic run to run, so a run passes its own baseline with equal numbers; that needs
`--allow-same-code`). Baselines live in `golden/baselines/` (tracked, unlike the large `*.safetensors`).

### Noise envelope: per-prompt limits that follow the noise (`--write-noise-envelope`, `--noise-envelope`)

**Why.** The fixed per-prompt margins (KL x1.75, top-1 -1 pt or one position, `|tail diff|` +0.002) were set on a dense
model, where an equally correct build moves a prompt by almost nothing. On the MoE / hybrid 35B they sit inside the noise:
the same build with SIMT instead of flash attention, or with prefill chunk 256 instead of 1024, moves one prompt's 21-way
KL by 0.13-0.17 (`chat_reuse_4k`) and the others by 0.005-0.012, and the old build fails its own baseline gate when only
the attention kernel changes (docs/results/2026-10-13-phase1b.md, sections 8-10). A limit smaller than the spread of
correct builds measures that spread, not a defect. The 8B dense model is quiet (old vs new mean KL about 1e-10), so it
does not need this.

**Record it** (the same build as the baseline, a second time, with a declared equally correct perturbation):

```
intarsia accuracy ... --golden G --write-baseline B.json                                  # run 1, unperturbed
intarsia accuracy ... --golden G --noise-ref B.json --noise-perturb attn-simt --write-noise-envelope E.json
intarsia accuracy ... --golden G --noise-ref B.json --noise-perturb chunk=64  --write-noise-envelope E.json   # extends E.json
```

`--noise-perturb` is `chunk` (the baseline's prefill chunk / 4, at least 16), `chunk=N`, `attn-simt` (cuda / hip: sets
`INTARSIA_CUDA_DISABLE` / `INTARSIA_HIP_DISABLE` to the flash and split-KV attention kernels, the SIMT ones remain) or
`external:TEXT` (you set other flags or environment yourself, e.g. `INTARSIA_CPU_FAULT=matmul:4.7e-3` on the cpu backend,
and say what in TEXT). The run is refused unless it is the same golden, the same configuration except the declared
perturbation, the same build as the one that recorded `B.json` (code stamp equal), and it really changed something. For
each prompt and mode the envelope keeps both runs' numbers and the delta the per-prompt gate looks at: |top-1 difference|
in points, |mean 21-way KL difference|, |mean `|tail diff|` difference|; the largest single-position KL difference is
recorded and printed but no check uses it (there is no per-prompt max-KL check). A second perturbation into the same file
makes the per-prompt maximum count. The exit status of an envelope run does not depend on its own gates; it also prints
the perturbed run judged by the fixed margins, which is the quickest way to see that a margin is inside the noise.
A perturbation that does not reach a prompt (a prefill chunk change leaves every prompt shorter than the chunk
bit-identical) gives delta 0 for it: that is "not measured", not "quiet", and the writer and the gate both list those
prompts; the fixed margin applies to them.

**Use it**: `intarsia accuracy ... --golden G --baseline B.json --noise-envelope E.json [--envelope-k 2]`. The envelope is
refused unless it was measured against this golden, this baseline run configuration and the build that recorded `B.json`.
Per prompt and check:

```
limit = baseline value -/+ max(fixed margin, k x envelope delta of that prompt)        k = 2 by default
```

The verdict prints for every prompt and check the baseline value, this run, the limit, its source (`fixed`, or `envelope`
when k x delta was wider), the envelope delta, and the `k needed` for that check to pass, followed by the number of
failing prompts at k = 0, 1, 2, 3, 5, 10. **The aggregate (all-prompt) checks keep their fixed margins and stay hard**; the
envelope never touches them. Without `--noise-envelope` the per-prompt checks behave as before and the run warns when
`config.json` shows experts or linear-attention layers (a dense model gets no warning).

**Choosing k.** `k = 2` was fixed before the 35B re-evaluation; it says "a new build may move a prompt twice as far as
two correct builds of the old code moved it". It is the low end of what is defensible, and the numbers show where:

* The new build's difference to the baseline and the envelope's difference are two draws of the same noise (they share the
  baseline run), so even a correct build exceeds `k x delta` sometimes. Monte Carlo of one upper-limit check (normal noise;
  Laplace and t3 are slightly worse), fraction of correct builds that fail it, by number m of real perturbations in the
  envelope (the maximum is used):

  | m | k = 1 | k = 2 | k = 3 | k = 5 | k = 10 |
  |---|---|---|---|---|---|
  | 1 | 25 % | 13.6 % | 9.1 % | 5.5 % | 2.8 % |
  | 2 | 16.8 % | 5.5 % | 2.6 % | 1.0 % | 0.2 % |
  | 3 | 12.5 % | 2.7 % | 0.9 % | 0.2 % | 0 |
  | 4 | 10.0 % | 1.5 % | 0.4 % | 0.1 % | 0 |

  A gate has about 2 modes x 10 prompts x 3 checks, so with one or two perturbations k = 2 flags some prompt of a correct
  build most of the time.
* Measured on the Mac (Qwen3-Reranker-0.6B, cpu, `INTARSIA_CPU_FAULT=matmul:p` as the equally correct perturbation, p =
  4.7e-3 is the size of the old/new MoE kernel difference; 10 prompts x prefill and decode; the envelope's chunk 64
  perturbation moved nothing, 19 of 20 prompts bit-identical): correct builds of other p / other random draws failed
  1 prompt with m = 1 (1 of 1 trials) and with m = 2 (2 of 2: one prompt each in `code` + `chat_short`, `chat_system` + an
  aggregate max-KL); with m = 4 and m = 7 one of 4 trials failed (`chat_system`, a single position KL 0.15, an outlier no
  sample of the envelope had). Logs, envelopes and the driver scripts: `docs/results/2026-10-13-phase1b-acceptance/gate-calibration/mac/` (the scripts hard-code the scratch directory they ran in; the Monte Carlo is `scripts/mc.py`, run with `uv run --with numpy`).
* A bug is far outside: attention output x1.05 failed 7/6 aggregate checks and 10 of 10 prompts in both modes; x1.01 failed
  mean KL, tail-bucket KL and top-5 in prefill, mean KL in decode, and 3 + 1 prompts with the envelope (3 + 3 with fixed
  margins only: on this dense model the envelope removes none of them); `matmul:1.0` (every output one ulp off, 14x the
  rel_l2 of the envelope's perturbation) failed 9 + 7 prompts and the aggregates; x1.002 is not distinguishable from a
  correct build (the same prompt `code` that a correct build failed). 2x the envelope's rel_l2 (`matmul:2e-2`) failed one
  prompt per mode and the prefill aggregate top-1: the resolution of the gate is about a factor of 2 in rel_l2.

So: treat k = 2 as a pre-registered default, not as a calibrated false-positive rate. A prompt that fails with `k needed`
below about 3 is inside the spread of correct builds; one with `k needed` above 5 is not. For a hard per-prompt verdict
record at least 4 perturbations or raise `--envelope-k`; with one or two (all the 35B window could afford) read the
`k needed` column next to the verdict.

### Which code produced a baseline (provenance)

Review finding 1: the "before" baseline used to be recorded from the new tree with kernels switched off, and nothing
said which code made it. Switching off kernels cannot switch off the device-position rope / kv_append / attention
kernels, the split-KV grid or the loader changes, and a typo in `INTARSIA_CUDA_DISABLE` silently left the new kernel on.
Now:

* A build stamps itself. `INTARSIA_BUILD_STAMP` (the output of `scripts/tree-stamp.sh`: `head=<sha> code-tree=<hash>
  dirty=yes|no`, plus `overlay=<paths>`) is passed to `cargo build` and read with `option_env!`. The build boxes get the
  tree without `.git`, so the binary cannot ask git itself; `scripts/phase1b-acceptance.sh` computes the stamp on the Mac.
  A binary built without it is "unstamped" and its baselines and self-goldens are refused.
* A baseline and a self-golden store that stamp. The gate (`--pre-change SHA`, `--revlist FILE`) accepts a baseline only if
  its head is the declared pre-change commit or an ancestor of the code under test (`git rev-list HEAD` of that tree,
  written where the history is), the two code trees differ (a self-comparison proves nothing: `--allow-same-code` for
  reproducibility checks), and the baseline's tree is clean. The one allowed exception is the **harness overlay**: the
  pre-change engine has no baseline or self-golden support, so "before" is a build of the pre-change commit plus only the files
  in `HARNESS_OVERLAY_PATHS` (`accuracy.rs` and its three submodules, which read engine output and never change it). Its
  stamp says `dirty=yes overlay=...` and the gate accepts exactly that. The build under test must be clean (`--allow-dirty`
  for development).
* Kernel disable list: a baseline recorded with `INTARSIA_CUDA_DISABLE` / `INTARSIA_HIP_DISABLE` set is refused (it is not
  the pre-change code). The variables themselves now reject names that match no registered kernel when the device opens.
* Printed, never refused: `--no-graphs`, the resolved kernel per op, differences from the recorded run. If nothing differs
  the run says so ("a kernel or graph change cannot show up in this comparison").

### Same-weights regression gate: `--write-self-golden` / `--self-golden`

Review finding 2: against a vLLM reference with an fp8 KV cache the 35B baseline mean KL is about 6.5e-3, mostly the
reference's own noise, so "no worse than x1.15" allows about 1e-3 nats of real regression. When the weights are the
same the right reference is the previous build's own distribution:

```
# the pre-change build (4963b43 + harness overlay), once
intarsia accuracy --model M --backend cuda --golden G --write-baseline B.json --write-self-golden S.safetensors
# the new build
intarsia accuracy --model M --backend cuda --golden G --baseline B.json --pre-change 4963b43 --revlist revlist   # floor, baseline, per prompt
intarsia accuracy --model M --backend cuda --self-golden S.safetensors --pre-change 4963b43 --revlist revlist    # same weights
```

The dump keeps, for every prompt of G, the top-64 log-probs (full-vocabulary normalised, so the tail mass is known) of the
**prefill** path (every position) and of the **decode** path (the teacher-forced window, which is what graph replay and
the decode MoE kernel changed). It is a `intarsia-golden-topk-v1` file for the prefill rows (`--golden S` works), with
`dec_pos_i`, `dec_topk_ids_i`, `dec_topk_logprobs_i` for the decode rows and `self_golden=1`, the code stamp and the run
configuration in the metadata. Each path is compared with the same path of the old build, never prefill with decode, with
the same comparator as everything else (top-1, KL on the top-64 support, 21-way KL with the tail bucket). Budgets, all
flags, and the output line `SELF-STATS ...` the acceptance script reads:

| check | flag | default |
|---|---|---|
| mean 21-way KL(old \|\| new) | `--self-max-kl` | 1e-4 |
| largest single-position 21-way KL | `--self-max-pos-kl` | 5e-3 |
| top-1 flips, percent of positions (at least one position) | `--self-max-flip-pct` | 0.5 |
| mean per-position `\|tail difference\|` | `--self-max-tail` | 5e-4 |
| worst prompt's mean KL | `--self-prompt-factor` x `--self-max-kl` | 5 x |
| non-finite rows | | none |

`--self-golden` needs the same model, backend, KV and activation dtype, fp8 mode, prefill chunk and decode positions
(`--allow-config-mismatch` to downgrade). It cannot be combined with `--golden`, `--baseline`, `--write-baseline` or
`--mode reuse`.

### Choosing the self-golden budgets

**Method.** The question a budget answers is "how far does a legitimate change move the logits, and how far a bug".
The prompt asked for the budget to come from the run-to-run noise of one build on the Mac cpu backend with
Qwen3-Reranker-0.6B. Measured 2026-10-13 (golden `qwen3-reranker-0.6b-long`, 10 prompts, 1434 prefill and 160 decode
positions, old build = 4963b43 + harness overlay writing the dump, new build = this tree reading it):

| run against the old build's dump | prefill mean KL21 / flips of 1434 | decode mean KL21 / flips of 160 |
|---|---|---|
| new build, same flags (run-to-run) | 2.5e-14 / 0 | 1.5e-14 / 0 |
| `--prefill-chunk 64`, `1024`; `--kv-dtype f32` | identical to the row above (bitwise) | identical |
| `--kv-dtype f16` | 2.7e-4 / 19 | 4.7e-4 / 2 |
| `--reference` (f64 oracle kernels) | 1.3e-3 / 26 | 1.1e-3 / 2 |

1. **Run-to-run noise on one build is exactly zero** (1e-14 is the f32 rounding of the stored log-probs): the engine is
   bit-reproducible on the cpu, so no budget can be derived from it, and a budget set as a multiple of it would reject any
   legitimate reordering. (On GPU it is probably also zero; the acceptance script checks it: the new build against its own
   dump from a second process, budgets 1e-9 / 1e-6 / 0 flips.)
2. **The smallest non-zero perturbation already saturates this model.** `INTARSIA_CPU_FAULT=<op>:<p>`
   (`crates/intarsia-backend-cpu/src/fault.rs`, off by default, for gate-sensitivity checks only) moves each bf16 output
   element of the named ops one ulp up or down with probability `p`, the output change a kernel with another summation
   order produces. `p = 4.7e-3` is rel_l2 3.5e-4, the size measured between the old and the grouped MoE kernel (unit test
   `flip_rate_to_rel_l2`); rel_l2 is about `5.1e-3 * sqrt(p)`.

   | `matmul` flip rate p (rel_l2 of each matmul output) | prefill mean KL21 / flips | decode mean KL21 / flips |
   |---|---|---|
   | 1e-9 (a handful of flips in about 1e9 outputs) | 9.2e-4 / 24 | 4.9e-4 / 2 |
   | 1e-6 | 1.0e-3 / 27 | 1.1e-3 / 2 |
   | 1e-4 | 1.3e-3 / 31 | 1.2e-3 / 3 |
   | 4.7e-3 (3.5e-4) | 1.2e-3 / 27 | 9.2e-4 / 3 |
   | 2e-2 (7e-4) | 1.4e-3 / 33 | 1.2e-3 / 3 |
   | 0.2 (2.3e-3) | 2.0e-3 / 45 | 1.8e-3 / 1 |
   | 1.0 (5.1e-3) | 3.6e-3 / 50 | 3.8e-3 / 4 |
   | attention only, p = 1 | 1.4e-3 / 32 | 1.3e-3 / 3 |

   One flipped ulp in the long prompts moves the whole prompt's KL by about 1e-3: the response is flat from a handful of
   flips up to rel_l2 1e-3 and only then grows (the 0.6B reranker's distributions are flat and an early token's change
   propagates to every later position). **On this model the noise floor of any non-zero change is about 1e-3 mean KL and 2%
   top-1 flips**, ten times the review's 1e-4 target. The Mac proxy therefore cannot calibrate a 35B budget (sharper
   distributions, NVFP4 weights, 40 layers of MoE): it only shows the shape of the problem.
3. **What the budget must do is decided on the target model**, so the defaults are the review's targets (1e-4 / 5e-3 /
   0.5% / 5e-4, prompt x5) and the acceptance run derives the budgets it applies as `max(default, 3 x floor)`, where the floor
   is measured in the same run on the real weights: the pre-change build against its own dump with the SIMT attention
   kernels instead of flash / split-KV (`INTARSIA_CUDA_DISABLE=attn_decode_bf16_splitkv,attn_prefill_bf16_splitkv,attn_prefill_bf16_flash`),
   a numerically equivalent reordering. If the floor is above the defaults the gate is less sensitive on that model and the
   log says by how much; if it is below, the defaults apply. The rule is fixed in the script header before any number
   exists.

**Sensitivity check (the gate can see a change the old gates cannot).** Old build baseline `B0` and dump `S0`, new build
with `INTARSIA_CPU_FAULT=matmul:1e-9` (about one flipped element in 1e9 matmul outputs): the baseline gate (floor, baseline,
per prompt) PASSES (KL against HF moves from 2.380e-3 to 2.463e-3 prefill, within x1.15) while the self-golden gate FAILS
(mean KL21 9.2e-4 against 1e-4). With `matmul:4.7e-3` (a legitimate-size reordering of every matmul) the baseline gate
passes with the shipped margins and fails at `matmul:2e-2` (rel_l2 7e-4) on the per-prompt KL, the quantity the old
aggregate gate could not see. Those margins (x2.5 max KL, x1.75 per prompt) are the ones that separate the two on this
model; they were x1.5 / x1.3 in the first draft and failed the legitimate-size case. They are untested on the 35B.

### TP self-golden budget (tp=2 against tp=1): measured, not shipped

**Question.** `--tp 2` against a `--tp 1` self-golden of the same weights (Qwen3-8B BF16, 2x R9700) fails the default budgets
by 5-6x (mean KL 5.9e-4 prefill, 5.1e-4 decode against 1e-4). Is that a defect, or the size of an equally correct
reordering, and can a TP budget be derived from measured noise rather than set by hand?

**Method** (all on the long fp32 prompt set, 10 prompts, 1434 prefill and 160 decode positions; raw logs in `docs/results/2026-10-14-tp2-budget-logs/`,
table generator `scripts/tp2-budget-summarize.py`, results `docs/results/2026-10-14-tp2-budget.md`). `S0` is the tp=1 default dump. Every other configuration is run
against `S0` with the default budget flags, and only its `SELF-STATS` line is read.

1. **Spread of equivalent tp=1 variants against `S0`**: run twice (second process), device 1 instead of 0,
   `INTARSIA_NO_FUSION=1`, `--prefill-chunk 1024`, `--prefill-chunk 64`, `INTARSIA_HIP_DISABLE` of the flash / split-KV
   attention kernels (SIMT attention), of the decode gemv kernels (`matmul_bf16_gemv,logits_bf16_gemv`), and the combination.
   Run-to-run, device 1, `INTARSIA_NO_FUSION=1` (the run reports the same resolved kernels: the fused kernels do not
   apply to this model, so fusion is not a difference between tp=1 and tp=2 here) and chunk 1024 (2 of 10 prompts span
   more than one chunk of 256, none of 1024) are bitwise identical to `S0` (mean KL 9e-11, the rounding of the stored
   log-probs). The variants that move the logits are SIMT attention, the gemv switch and, a little, chunk 64:

   | tp=1 variant vs `S0` | prefill mean KL / max pos KL / flips of 1434 / tail / worst prompt | decode mean KL / max pos KL / flips of 160 / tail / worst prompt |
   |---|---|---|
   | chunk 64 | 1.2e-6 / 3.3e-4 / 0 / 5.2e-7 / 2.0e-5 | 4.7e-5 / 2.9e-3 / 0 / 3.6e-6 / 4.7e-4 |
   | SIMT attention (= with no-fusion, chunk 64) | **6.07e-4 / 4.9e-2 / 6** / 1.6e-4 / 7.7e-4 | **3.54e-4** / 6.2e-3 / 0 / 2.0e-4 / 7.6e-4 |
   | decode gemv off | 9e-11 (identical) | 2.89e-4 / **8.9e-3** / 0 / **2.1e-4** / **1.15e-3** |
   | **max over the variants** | 6.07e-4 / 4.9e-2 / 6 (0.42%) | 3.54e-4 / 8.9e-3 / 0 / 2.1e-4 / 1.15e-3 |

2. **tp=2 against `S0`** (f32 partials, P2P): prefill 5.90e-4 / 5.0e-2 / 5 flips / 1.5e-4 / 9.8e-4, decode 5.07e-4 / 1.5e-2 /
   0 / 2.1e-4 / 2.0e-3. The P2P and host communicators, and a second run, give the same numbers to the last digit; bf16
   partials, chunk 64 and SIMT attention under tp=2 land at 5.6-5.9e-4 prefill (5-7 flips) and 3.8-5.1e-4 decode.
3. **Distance to the fp32 golden** (what the floor measures; full-vocabulary KL, prefill / decode): tp=1 variants
   3.97e-4 - 4.22e-4 / 2.70e-4 - 3.57e-4, tp=2 4.19e-4 / 3.45e-4: **inside the spread of the tp=1 variants**. The top-1
   agreement is 99.51% for tp=2 against 99.58% for every tp=1 variant (one position; the tp=2 variants span 99.51 - 99.86%). On
   the short golden's prompts (434 / 128 positions) tp=2 (3.90e-4 prefill) is outside the three tp=1 runs there
   (2.34e-4 - 3.28e-4); the budget below was derived on the long set only.
4. **Conclusion on the numbers**: tp=2 against tp=1 is the same kind of perturbation as the attention-kernel switch. The
   prefill figures are inside the tp=1 spread, the decode figures (5.1e-4 against 3.5e-4) outside it by 1.4x and inside it at
   k = 2. The default 1e-4 budget was chosen for single-kernel changes and cannot apply to TP.

**Derived TP budget** (budget = max over the tp=1 variants x k, with k = 2 = `envelope::DEFAULT_K`, not touched):

| check | max over tp=1 variants (which) | x 2 | tp=2 observed |
|---|---|---|---|
| mean 21-way KL | 6.07e-4 (SIMT attention, prefill) | 1.2e-3 | 5.9e-4 / 5.1e-4 |
| largest single-position KL | 4.9e-2 (SIMT attention, prefill) | 9.8e-2 | 5.0e-2 / 1.5e-2 |
| top-1 flips (prefill) | 6 of 1434 (SIMT attention) | 12 = 0.84% | 5 |
| mean \|tail diff\| | 2.1e-4 (gemv off, decode) | 4.2e-4 | 1.5e-4 / 2.1e-4 |
| worst prompt mean KL | 1.15e-3 (gemv off, decode) | 2.3e-3 (x1.9 the mean-KL budget) | 9.8e-4 / 2.0e-3 |

tp=2 passes all five with margin (the tightest, worst prompt in decode, at 86% of its limit).

**It was not shipped, because it does not catch a 1% partial-scale bug.** `INTARSIA_TP_TEST_BUG=scale:F[@RANK]`
(test-only, `crates/intarsia-core/src/model.rs`; printed in the self-golden's exec diff as a fault) multiplies one rank's
partial by `F` before every all-reduce (rank 1 by default). Against `S0`, judged by the derived budget above:

| bug | prefill mean KL / max pos KL / flips | decode mean KL / max pos KL | verdict at the derived budget | fp32 floor (long golden) |
|---|---|---|---|---|
| none (tp=2) | 5.9e-4 / 5.0e-2 / 5 | 5.1e-4 / 1.5e-2 | pass | pass |
| 1.001, rank 1 | 6.0e-4 / **1.1e-1** / 8 | 4.3e-4 / 6.3e-3 | **fails on max pos KL, from one position** | pass |
| 1.003, rank 1 | 5.4e-4 / 6.4e-2 / 5 | 5.2e-4 / 9.1e-3 | pass | pass |
| 1.005, rank 1 | 5.5e-4 / 4.9e-2 / 8 | 5.2e-4 / 1.2e-2 | pass | not run |
| **1.01, rank 1** | 9.3e-4 / **2.2e-1** / 7 | 7.0e-4 / 1.9e-2 | **fails, on prefill max pos KL alone** | pass (top-1 99.51%, KL 6.7e-4) |
| **1.01, rank 0** | 7.4e-4 / 7.2e-2 / 5 | 5.8e-4 / 1.1e-2 | **pass** | not run |
| 1.01, rank 1, short golden's prompts | 4.3e-4 / 4.5e-2 / 0 | 5.8e-4 / 1.1e-2 | **pass** | pass |
| 1.02, rank 1 | **1.44e-3** / 1.7e-1 / 7 | **1.32e-3** / 4.0e-2 | fails (mean KL, both modes) | **fails** (KL 1.17e-3, top-1 99.37%) |
| 1.03 / 1.05 / 1.1 | 2.5e-3 / 5.4e-3 / 1.9e-2 | 2.0e-3 / 3.8e-3 / 1.4e-2 | fails | fails |

A 1% bug is about a 0.5% change of each block output, 5.7e-3 relative in the f32 logits of the Mac model
(`crates/intarsia-models/tests/tp_test_bug.rs`) but only 1.6x the bf16 reordering noise in mean KL here. The derived budget
fails the rank-1 realisation on a single heavy-tailed statistic (the largest KL of 1434 positions; it also fails a 0.1%
bug, and reads 1.6e-1 for clean tp=2 with bf16 partials against the tp=2 default dump) and passes the rank-0 realisation
and the short golden's prompts. The mean KL, flips and worst-prompt checks only see the bug from 2% (long golden; on the
short golden's 128 decode positions it fails at 2% by 1.23e-3 against 1.21e-3) or 3%. No single mean-KL budget separates
a 1% bug from clean tp=2 either: the rank-0 bug's decode figure (5.8e-4) is below clean tp=2's prefill figure (5.9e-4),
and its prefill figure (7.4e-4) is 1.2x the SIMT-attention spread, k = 1.2 instead of the envelope's 2.

**What guards TP instead.** (a) the Mac f32 test `tp2_matches_tp1_prefill_and_decode_f32` (`crates/intarsia-models/tests/tp.rs`,
summation order 2.5e-7 against its 1e-5 bound) and `tp_test_bug.rs` (the same comparison with `scale:1.01` gives 5.7e-3:
23000x the clean figure): it checks the sharding and the reduction exactly, in `cargo test --workspace`; (b) the exact
all-reduce tests on the GPU (`crates/intarsia-backend-hip/tests/comm.rs`); (c) the fp32-golden floor on both prompt sets
(`--gate-class bf16`), which fails a partial-scale bug from 2%. Together they leave a gap between 1% and 2% on the GPU gates
that only (a) closes. The default self-golden budget is unchanged, so `--tp 2` against a tp=1 self-golden keeps failing it;
do not use that comparison as a TP gate.

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

Qwen3-8B against fp32 goldens, measured 2026-10-13 on spark1 (same prompts, same CUDA bf16 build; HF fp32 goldens
`qwen3-8b-fp32.safetensors` = 8 short prompts, `qwen3-8b-long-fp32.safetensors` = those plus `long_prose_300/700`; HF bf16
against HF fp32 is mean full-vocabulary KL 5.6e-4 short, 6.3e-4 long, so the bf16 golden carries about as much noise as
the engine):

| model | golden | mode | top-1 | mean KL | max KL | gate (bf16 class) |
|---|---|---|---|---|---|---|
| Qwen3-8B, cuda (spark1) | HF fp32, 8 short prompts | prefill | 100.00% | 2.90e-4 | 1.5e-2 | PASS |
| Qwen3-8B, cuda (spark1) | HF fp32, 8 short prompts | decode | 100.00% | 3.24e-4 | 6.8e-3 | PASS |
| Qwen3-8B, cuda (spark1) | HF bf16, 8 short prompts | prefill | 100.00% | 6.38e-4 | 2.0e-2 | PASS |
| Qwen3-8B, cuda (spark1) | HF bf16, 8 short prompts | decode | 100.00% | 1.12e-3 | 1.5e-2 | FAIL (mean KL 1.0e-3) |
| Qwen3-8B, cuda (spark1) | HF fp32, 10 prompts | prefill | 99.65% | 4.00e-4 | 2.1e-2 | PASS |
| Qwen3-8B, cuda (spark1) | HF fp32, 10 prompts | decode | 100.00% | 3.33e-4 | 6.8e-3 | PASS |
| Qwen3-8B, cuda (spark1) | HF bf16, 10 prompts | prefill | 99.58% | 8.57e-4 | 1.1e-1 | PASS |
| Qwen3-8B, cuda (spark1) | HF bf16, 10 prompts | decode | 100.00% | 9.81e-4 | 1.5e-2 | PASS |

Against fp32 the engine's KL (2.9e-4 short, 4.0e-4 long, prefill) is below the HF bf16 golden's own distance to fp32
(5.6e-4, 6.3e-4), and the bf16 golden's decode row is the only FAIL: that failure is mostly the reference's noise. Raw logs: `docs/results/2026-10-13-phase1b-acceptance/long-context/logs/log_8b_*.txt`.

The earlier "99.08% / 1.25e-3" figures counted fp16 ties as agreement and used 8 short prompts; the numbers
above are not comparable with them. Results are identical for `--prefill-chunk` 64, 256 and 1024 (chunking is
not the cause). The Qwen3-8B rows are the CUDA backend (admitted kernels: cuBLAS prefill GEMM, GEMV decode) against an HF bf16 golden produced on the same box; the prefill gate misses the 99.5% top-1 default by 0.06 percentage points while mean KL passes, and the defaults were not loosened. Judge new backends mainly by how they compare
with `cpu` on the same golden file.

## Where the goldens live

* Mac `golden/` (git-ignored `*.safetensors`) and `tatooine:/data/models/intarsia/golden/`: `qwen3-8b-fp32.safetensors`
  and `qwen3-8b-long-fp32.safetensors` (HF fp32, Qwen3-8B, built 2026-10-13 on spark1 with `golden.py --dtype float32`;
  the bf16 `qwen3-8b.safetensors` / `qwen3-8b-long.safetensors` stay on spark1 and tatooine), `qwen3-reranker-0.6b-long.safetensors` (HF bf16),
  `...-long-fp32.safetensors` (HF fp32), the older 8-prompt `qwen3-reranker-0.6b*.safetensors`, and the two vLLM
  top-20 goldens of the 35B (small): `qwen36-35b-a3b-nvfp4-vllm-top20.safetensors` (10 short prompts, rebuilt
  2026-10-13 with provenance metadata) and `qwen36-35b-a3b-nvfp4-vllm-long-top20.safetensors` (4k and 12k prompts +
  64 greedy continuation tokens each, vLLM 0.26.1rc1 `guidance`, fp8 KV cache).
* spark1: `~/intarsiaml-golden/` (outside the repo copy on purpose: syncing the repo with `rsync --delete`
  would remove it): the two vLLM goldens above, the two HF fp32 goldens `qwen3-8b-fp32.safetensors` / `qwen3-8b-long-fp32.safetensors` (the default for the 8B, pushed and sha256-checked by `scripts/phase1b-acceptance.sh`), `qwen3-8b-full.safetensors` (8 short + 2 long prompts, full
  vocabulary, HF bf16, with metadata, 435 MB), `qwen3-8b-long.safetensors` / `qwen3-8b.safetensors` (older, no
  metadata), and, to be built there, the 8B long golden `qwen3-8b-4k-top20.safetensors` (`golden.py --topk 20 --prompts
  prompts/golden_4k.jsonl` in the `vllm/vllm-openai:v0.28.0-aarch64-cu129` image; `run8b.sh` in that directory has the command;
  NOT built yet: spark1 went offline during that run).
  Pass them with `--golden`.
* `golden/baselines/*.json` (tracked): recorded runs, named after golden and model; see the baseline section.
