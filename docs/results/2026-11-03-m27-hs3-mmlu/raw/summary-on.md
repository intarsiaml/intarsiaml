# localeval: adhoc — 2026-11-03T14:53

- model: `intarsia` on ? (?), url `http://127.0.0.1:18471/v1`
- thinking: off; run kind: quality; label: m27-hs3on
- harness: localeval 0.4.0, deepeval 4.2.6, runner endor.kuepker.haus, git a43b846
- command: `/opt/code/deepeval/.venv/bin/localeval quality --url http://127.0.0.1:18471/v1 --model intarsia --thinking off --structured none --bench mmlu --limit 25 --label m27-hs3on --recipe /opt/code/intarsiaml/.worktrees/m27-hs3/.scratch/intarsia serve --backend metal --model /Users/josef/models/Qwen3.8-27B-Quark-AWQ-MXFP4 --served-model-name intarsia --max-seq 20000 --profile-every 0 --no-supervisor --no-ui --host 127.0.0.1 --port 18471`
- result file: `results/20261103-145358-adhoc-m27-hs3on/result.json`

## Launch recipe

from inline command (secrets redacted):

````
/opt/code/intarsiaml/.worktrees/m27-hs3/.scratch/intarsia serve --backend metal --model /Users/josef/models/Qwen3.8-27B-Quark-AWQ-MXFP4 --served-model-name intarsia --max-seq 20000 --profile-every 0 --no-supervisor --no-ui --host 127.0.0.1 --port 18471
````

## Benchmarks (deepeval)

| benchmark | score | n | TTFT s | prefill tok/s | decode tok/s | calls | errors | wall s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| mmlu | 0.818 | 1425 | 0.79 | 854 | - | 1425 | 0 | 1417.3 |
- mmlu: 1425 replies kept; 0 truncated (score 0); thinking verified off; mean output 2.0 tokens

Benchmark speed is incidental (short, varying prompts); prefill '-' means no call had 1024+ uncached tokens. Quote the sweep for speed.

## Method

Streamed OpenAI-compatible calls, temperature 0. TTFT = request start to first content/reasoning token; prefill tok/s = uncached prompt_tokens / TTFT; decode tok/s = (completion_tokens - 1) / (last - first token), counts from server usage. Speed sweep: fresh nonce first in every prompt (no prefix-cache hits), ignore_eos+min_tokens forces the full length, 1 discarded warmup, median of reps. Other clients' in-flight requests checked before every sweep call. Quality-run speed is incidental (varying prompts), not a sweep.
