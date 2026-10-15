1792054264.888	[localeval] calibrated: 22.1 tokens/sentence, 52 base tokens, ctx=8192
1792054271.873	[localeval]    128 tok rep 1/3: prompt=114 ttft=0.05s prefill=2154 tok/s decode=36.8 tok/s peak=36.9 tok/s out=256
1792054278.905	[localeval]    128 tok rep 2/3: prompt=113 ttft=0.05s prefill=2118 tok/s decode=36.5 tok/s peak=36.8 tok/s out=256
1792054285.871	[localeval]    128 tok rep 3/3: prompt=113 ttft=0.05s prefill=2094 tok/s decode=36.9 tok/s peak=37.0 tok/s out=256
1792054293.113	[localeval]   1024 tok rep 1/3: prompt=991 ttft=0.27s prefill=3686 tok/s decode=36.6 tok/s peak=36.8 tok/s out=256
1792054300.355	[localeval]   1024 tok rep 2/3: prompt=991 ttft=0.27s prefill=3686 tok/s decode=36.6 tok/s peak=36.7 tok/s out=256
1792054307.630	[localeval]   1024 tok rep 3/3: prompt=986 ttft=0.27s prefill=3621 tok/s decode=36.4 tok/s peak=36.6 tok/s out=256
1792054316.163	[localeval]   4096 tok rep 1/3: prompt=4089 ttft=1.24s prefill=3294 tok/s decode=35.0 tok/s peak=35.5 tok/s out=256
1792054325.123	[localeval]   4096 tok rep 2/3: prompt=4091 ttft=1.68s prefill=2433 tok/s decode=35.0 tok/s peak=35.2 tok/s out=256
1792054334.264	[localeval]   4096 tok rep 3/3: prompt=4091 ttft=1.86s prefill=2197 tok/s decode=35.0 tok/s peak=35.4 tok/s out=256
1792054334.264	# localeval: adhoc — 2026-10-15T04:51
1792054334.264	
1792054334.264	- model: `intarsia-metal` on ? (?), url `http://127.0.0.1:8093/v1`
1792054334.264	- thinking: off; run kind: speed; label: clk-safe-r1
1792054334.264	- harness: localeval 0.2.1, deepeval 4.2.6, runner endor.kuepker.haus, git 69c7a84
1792054334.264	- command: `/opt/code/deepeval/.venv/bin/localeval speed --url http://127.0.0.1:8093/v1 --model intarsia-metal --sizes 128,1024,4096 --gen 256 --reps 3 --thinking off --label clk-safe-r1`
1792054334.264	- result file: `results/20261015-045103-adhoc-clk-safe-r1/result.json`
1792054334.264	
1792054334.264	## Speed sweep
1792054334.264	
1792054334.264	served `intarsia-metal`, fingerprint `None`, ctx 8192, 3 reps, 256 forced output tokens
1792054334.264	
1792054334.264	| prompt tok | TTFT s | prefill tok/s | marginal prefill | decode tok/s | peak decode | server prefill | server decode | out tok |
1792054334.264	|---:|---:|---:|---:|---:|---:|---:|---:|---:|
1792054334.264	| 113 | 0.05 | 2118 (2094–2154) * | - | 36.8 (36.5–36.9) | 36.9 (36.8–37.0) | - | - | 256 |
1792054334.265	| 991 | 0.27 | 3686 (3621–3686) * | 4046 | 36.6 (36.4–36.6) | 36.7 (36.6–36.8) | - | - | 256 |
1792054334.265	| 4091 | 1.68 | 2433 (2197–3294) | 2196 | 35.0 (35.0–35.0) | 35.4 (35.2–35.5) | - | - | 256 |
1792054334.265	
1792054334.265	\* under 1024 uncached prompt tokens: TTFT is per-request overhead, not prefill throughput.
1792054334.265	
1792054334.265	Marginal prefill = Δprompt tokens / ΔTTFT from the previous row; fixed request overhead cancels, so it is the cleanest prefill throughput figure.
1792054334.265	
1792054334.265	Peak decode = fastest 32-token stretch of each call (burst speed, e.g. a run of accepted speculative drafts); median and range over reps. '-' = older run or output too short.
1792054334.265	
1792054334.265	
1792054334.265	## Method
1792054334.265	
1792054334.265	Streamed OpenAI-compatible calls, temperature 0. TTFT = request start to first content/reasoning token; prefill tok/s = uncached prompt_tokens / TTFT; decode tok/s = (completion_tokens - 1) / (last - first token), counts from server usage. Speed sweep: fresh nonce first in every prompt (no prefix-cache hits), ignore_eos+min_tokens forces the full length, 1 discarded warmup, median of reps. Other clients' in-flight requests checked before every sweep call. Quality-run speed is incidental (varying prompts), not a sweep.
1792054334.265	
1792054334.265	[localeval] saved /opt/code/deepeval/results/20261015-045103-adhoc-clk-safe-r1/result.json
1792054335.294	[localeval] logged to unanimis: 18e19429-f145-4190-82f5-78909f5d4278
