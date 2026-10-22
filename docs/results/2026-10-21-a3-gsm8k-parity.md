# A3: GSM8K gap intarsia vs vLLM on spark1 (Qwen3.6-35B-A3B-NVFP4, tp=1)

**Verdict: the cause is not in intarsia. No engine change.** The Q1 gap (exact match 0.240 vs 0.281) comes from two things together:

1. **Scoring (localeval / deepeval).** With `--structured none`, deepeval's GSM8K prompt ends with "Let's think step-by-step." and then "Make sure to output only the numerical answer.", and the scorer is exact match on the whole reply. A reply that works the problem and ends with the right number scores 0. On the first token the model is close to a tie between "write the number" and "start working".
2. **The two engines resolve that tie differently.** At the first generated token vLLM puts more mass on a digit than intarsia does. intarsia's CPU backend (f32 accumulation, f64 NVFP4/MoE/GDN math, the same weights) agrees with intarsia CUDA and not with vLLM. vLLM with a bf16 KV cache still disagrees, so the fp8 KV cache is not the reason. Caveat: the CPU backend shares intarsia's model definition (recipe, scale interpretation, GDN). So this shows that intarsia's two backends agree. It does not prove that vLLM is the one in error. No independent fp32 reference exists for this checkpoint: `tools/golden.py` cannot load it (2026-10-14-35b-accuracy.md, implementation table).

So intarsia writes out its working more often (and is then almost always right), and the exact-match scorer counts those replies as wrong. If you score the last number in the reply, intarsia is well **ahead** of vLLM on the same prompts (section 5).

**Full paired replay, all 1319 questions** (same prompts and requests as localeval; intarsia is the Q1 binary):

| metric | intarsia | vLLM `guidance` | paired diff (95% CI) | McNemar (intarsia-only / vLLM-only right) |
|---|---|---|---|---|
| exact (localeval's GSM8K score) | 0.2403 | 0.2828 | −0.043 [−0.058, −0.027] | 31 / 87, p = 2.5e-7 |
| **last number = gold** | **0.6808** | **0.5762** | **+0.105 [+0.083, +0.127]** | **187 / 49, p = 2.6e-19** |

- The replay reproduces Q1: intarsia's exact score is identical to Q1's (0.2403), vLLM's is 0.2828 vs Q1's 0.2813, and there are 603 vs 400 long replies (Q1: 603 vs 390).
- Replies longer than 20 tokens: intarsia 603, vLLM 400. In 238 questions intarsia goes long where vLLM stays short; the reverse happens in 35.
- When a reply is long, its last number is right for 579/603 (intarsia) and 385/400 (vLLM).
- finish_reason: `stop` on all replies except `length` on 6 (intarsia) and 4 (vLLM), at max_tokens 1024.

**On localeval's own score the gap persists**, and the paired test makes it tighter. It is explained above and is not an intarsia defect, so step 5 (re-run after a fix) does not apply: no fix was made. Last-number scoring is supporting evidence that intarsia's maths is not worse; it does not close the localeval gap. MMLU was not re-run, because nothing in intarsia changed and Q1 found MMLU equal. The first 200 rows of `*-full.jsonl` are the run-1 replies from section 3.

## 1. Setup
- spark1 (GB10), fleetq #3276 (non-exclusive, 40 GB). vLLM `guidance` stayed up the whole time. No role was taken down; downtime 0.
- intarsia: `~/intarsia-tp2/bin/g1c-base-intarsia` (the binary Q1 used, main 0294b778), `serve --backend cuda --max-seq 20480 --mem-budget 38`, bf16 KV. For the logit dumps I also used a build of origin/main 957e3da8 (`main-*` below).
- vLLM `guidance`: image `nightly-65b7662d3`, `--quantization modelopt_fp4 --kv-cache-dtype fp8 --enable-prefix-caching --max-num-seqs 2 --reasoning-parser qwen3`.
- Fleet: the vLLM requests (the replays, both goldens and the 1319 run, about 2000 requests) went out under #3276, which did not declare `--uses guidance`. #3314 with `--uses` was queued behind an unrelated job and was cancelled. The direct id check ran under a slot with `--uses guidance`.
- Second reference vLLM `a3ref` (my own slot, port 8095, removed afterwards): same image and env, with `--kv-cache-dtype auto` (bf16), `--no-enable-prefix-caching`, `--max-num-seqs 1`.
- Replay (`raw/replay.py`): the GSM8K prompts built exactly the way localeval 0.2.1 builds them (`make_benchmark("gsm8k")`, 3-shot, CoT on, plus the confinement line), sent through localeval's own `Client.chat`. That gives the same body: streamed, temperature 0, `max_tokens` 1024, `chat_template_kwargs.enable_thinking=false`. The script records the full reply, finish_reason and usage, and scores with deepeval's `Scorer.exact_match_score`.

## 2. Suspects ruled out (first 200 questions)
| suspect | evidence | result |
|---|---|---|
| prompt tokenisation / chat template | `usage.prompt_tokens` equal on all 200. vLLM `/tokenize` of the messages with `enable_thinking=false` gives ids identical to the plain ChatML render `<|im_start|>user\n…<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n` (tail ids `…248046 198 248045 74455 198 248068 271 248069 271`). Proxy without a GPU: the argmax at the last row of intarsia's dump (taken on vLLM's ids, `accuracy` prefill path) is the first token of intarsia's chat reply for 187/200 questions. The direct check, sending vLLM's `/tokenize` ids to intarsia's `/v1/completions`: ID_CHECK_RESULT | same |
| stop handling | finish_reason `stop` on 200/200 on both. No `length` cut-offs. generation_config eos `[248046, 248044]` | same |
| max_tokens default | localeval sends 1024 explicitly. Neither server's default applies | same |
| default sampling from generation_config (top_k 20, top_p 0.95, temp 1.0) | both servers are greedy at temperature 0. intarsia gave 200/200 identical replies over two runs | not a factor |
| fp8 KV cache | intarsia with `--kv-dtype fp8_e4m3`: 74 long replies (bf16: 80, vLLM: 45). vLLM with bf16 KV: 57 long, first-token digit mass 0.685 vs 0.696 with fp8 (paired diff 0.011 ± 0.009) | small part at most |
| intarsia fp8 activation mode | `--fp8-act static` (the default) and `none` give first-token digit mass 0.652 and 0.639 on 24 prompts. vLLM: 0.771 | not a factor |

## 3. Replay on the first 200 questions
| run | exact | last number | replies > 20 tokens | long replies whose last number is right |
|---|---|---|---|---|
| intarsia run 1 | 65 | 143 | 80 | 78 |
| intarsia run 2 | 65 (200/200 identical to run 1) | 143 | 80 | 78 |
| vLLM `guidance` run 1 | 71 | 116 | 45 | 45 |
| vLLM `guidance` run 2 | 66 (only 126/200 identical to run 1) | 119 | 53 | 53 |
| vLLM bf16 KV, no prefix cache | 67 | 123 | 57 | 56 |

- **vLLM is not deterministic at temperature 0 here.** Two runs scored 71 and 66 exact; the second run is level with intarsia (65 vs 66).
- **intarsia replies at length more often, and that is systematic.** intarsia long while vLLM short: 41 (run 1), 34 (run 2), 30 (bf16-KV vLLM). The reverse: 6, 7, 7. vLLM against itself: 12 vs 20.
- **The two engines first differ on the first generated token**, e.g. q3: intarsia "James runs 3 sprints…" vs vLLM "540", and q47: intarsia "John bought blue ties…" vs vLLM "400". On those questions vLLM's own top-2 at the first token are close: q3 'James' 0.33 / '3' 0.29 / '5' 0.25; q19 '6' 0.39 / 'Mar' 0.39. Its argmax for q3 and q47 also changes between `/v1/completions` and `/v1/chat/completions` on the same ids.

## 4. Logits at the first generated token (last prompt row)
Top-20 log-probs per position: `tools/golden_vllm.py` for vLLM, `intarsia accuracy --mode prefill --write-self-golden` for intarsia (`raw/lastrow.py`, `raw/klall.py`). "Digit mass" is the probability that the first token starts with a digit, `$` or `-`.

| implementation | digit mass, 200 prompts | digit mass, the 24-prompt subset |
|---|---|---|
| vLLM `guidance` (fp8 KV) | 0.696 | 0.771 |
| vLLM bf16 KV | 0.685 | — |
| intarsia CUDA, g1c binary | 0.593 | — |
| intarsia CUDA, main 957e3da8 | — | 0.652 |
| **intarsia CPU backend (high-precision reference, endor)** | — | **0.643** |

Paired differences on the 24-prompt subset (mean ± se):
- CPU − CUDA = −0.009 ± 0.017.
- CPU − vLLM = −0.128 ± 0.027.
- CPU − vLLM(bf16 KV) = −0.110 ± 0.021.

On 200 prompts, vLLM − CUDA = +0.103 ± 0.010.

Mean 21-way KL, top-20 support plus tail, over the 24 prompts:

| KL21 | all rows | last row |
|---|---|---|
| CPU ‖ CUDA | **0.026** | **0.049** |
| CPU ‖ vLLM(bf16 KV) | 0.039 | 0.100 |
| CPU ‖ vLLM | 0.045 | 0.174 |

Over 200 prompts: vLLM ‖ vLLM(bf16 KV) is 0.035, and vLLM ‖ CUDA is 0.041.

intarsia's high-precision CPU backend sides with intarsia CUDA: against the g1c binary used in the replay, CPU − CUDA digit mass = −0.009 ± 0.017, and KL21 0.026 over all rows, 0.049 at the last row. The KV dtype explains only a small part of vLLM's difference. What is left is a difference between vLLM's implementation and intarsia's model definition. It is not isolated further, and the CPU arbiter covers 24 prompts only.

The absolute KL on these few-shot maths prompts is high for every pair, vLLM against vLLM included. That makes `intarsia accuracy` against the `guidance` golden FAIL its 3e-2 floor here (4.3e-2). It is a property of this prompt set, not a regression: the CPU against CUDA KL is the smallest of all the pairs.

## 5. Scoring that counts the maths
Two metrics, with the paired difference intarsia − vLLM:
- **exact:** deepeval's exact match on the whole reply.
- **last number:** the last number in the reply equals the gold answer.

On the first 200: exact −0.030 [−0.072, +0.012], McNemar p = 0.16; last number **+0.135 [+0.077, +0.193]**, p = 1.5e-5.

## 6. What this means and what to change (not done; outside intarsia)
- For accuracy comparisons between engines, the GSM8K exact-match score with `--structured none` and thinking off measures which reply format wins a near-tie at the first token. It does not measure maths accuracy. Options in localeval (deepeval harness, not this repo):
  - score the last number in the reply;
  - or run `--structured json_schema`, so deepeval's `NumberSchema` path forces a JSON number (this also suppresses the chain of thought);
  - or turn off CoT for GSM8K (`/opt/code/deepeval/src/localeval/quality.py:63` sets `enable_cot=True`), so the prompt stops asking for both.
  - Either needs the user's OK, since it changes localeval's GSM8K numbers for every engine.
- vLLM `guidance` is not deterministic at temperature 0. Suspected causes, not isolated: prefix caching, fp8 KV, batching with other clients (max-num-seqs 2). A single vLLM run carries about ±2.5 points of run-to-run noise on 200 questions.

## Runs and raw files (`2026-10-21-a3-gsm8k-parity/raw/`)
- `replay.py`, `cmp.py`, `lenient.py`, `paired.py`, `lastrow.py`, `klall.py`, `probe_vllm.py`, `probe_chat.py`: the scripts.
- `g1c-200*.jsonl`, `vllm-200*.jsonl`, `vref-200.jsonl`, `g1cfp8-200.jsonl`, `g1c-full.jsonl`, `vllm-full.jsonl`: per-question replies.
- `acc-*.log`: intarsia accuracy runs (CUDA on spark1, CPU on endor, fleetq #3335). The `.safetensors` dumps (16 MB each) stay in the lane scratchpad and are not committed.
