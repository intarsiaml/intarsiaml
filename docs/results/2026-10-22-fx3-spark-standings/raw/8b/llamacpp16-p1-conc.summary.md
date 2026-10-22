# localeval: adhoc — 2026-10-22T04:42

- model: `llamacpp` on ? (?), url `http://spark-fa00:8400/v1`
- thinking: off; run kind: speed; label: fx3-8b-llamacpp16-p1-conc
- harness: localeval 0.3.0, deepeval 4.2.6, runner endor.kuepker.haus, git 1f24e53
- command: `/opt/code/deepeval/.venv/bin/localeval speed --url http://spark-fa00:8400/v1 --model llamacpp --thinking off --label fx3-8b-llamacpp16-p1-conc --sizes 1024 --gen 128 --reps 1 --concurrency 1,4,16`
- result file: `results/20261022-044232-adhoc-fx3-8b-llamacpp16-p1-conc/result.json`

## Warnings

- INCOMPLETE: RuntimeError: adhoc: calibration failed: HTTP 400: {"error":{"code":400,"message":"request (4462 tokens) exceeds the available context size (2560 tokens), try increasing it","type":"exceed_context_size_error","n_prompt_tokens":4462,"n_ctx":2560}}

## Method

Streamed OpenAI-compatible calls, temperature 0. TTFT = request start to first content/reasoning token; prefill tok/s = uncached prompt_tokens / TTFT; decode tok/s = (completion_tokens - 1) / (last - first token), counts from server usage. Speed sweep: fresh nonce first in every prompt (no prefix-cache hits), ignore_eos+min_tokens forces the full length, 1 discarded warmup, median of reps. Other clients' in-flight requests checked before every sweep call. Quality-run speed is incidental (varying prompts), not a sweep.
