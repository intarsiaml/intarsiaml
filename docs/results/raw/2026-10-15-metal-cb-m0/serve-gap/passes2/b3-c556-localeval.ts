1792096466.467	[localeval] calibrated: 22.0 tokens/sentence, 53 base tokens, ctx=8192
1792096473.618	[localeval]    128 tok rep 1/3: prompt=111 ttft=0.05s prefill=2037 tok/s decode=35.9 tok/s peak=36.6 tok/s out=256
1792096480.648	[localeval]    128 tok rep 2/3: prompt=114 ttft=0.05s prefill=2115 tok/s decode=36.5 tok/s peak=36.8 tok/s out=256
1792096487.692	[localeval]    128 tok rep 3/3: prompt=113 ttft=0.05s prefill=2088 tok/s decode=36.5 tok/s peak=36.9 tok/s out=256
1792096494.975	[localeval]   1024 tok rep 1/3: prompt=989 ttft=0.25s prefill=3896 tok/s decode=36.3 tok/s peak=36.6 tok/s out=256
1792096502.243	[localeval]   1024 tok rep 2/3: prompt=992 ttft=0.25s prefill=3895 tok/s decode=36.4 tok/s peak=36.7 tok/s out=256
1792096509.558	[localeval]   1024 tok rep 3/3: prompt=990 ttft=0.25s prefill=3890 tok/s decode=36.1 tok/s peak=36.5 tok/s out=256
1792096518.149	[localeval]   4096 tok rep 1/3: prompt=4092 ttft=1.28s prefill=3195 tok/s decode=34.9 tok/s peak=35.5 tok/s out=256
1792096526.618	[localeval]   4096 tok rep 2/3: prompt=4085 ttft=1.08s prefill=3776 tok/s decode=34.5 tok/s peak=35.5 tok/s out=256
1792096535.185	[localeval]   4096 tok rep 3/3: prompt=4089 ttft=1.27s prefill=3223 tok/s decode=34.9 tok/s peak=35.3 tok/s out=256
1792096535.186	# localeval: adhoc — 2026-10-15T16:34
1792096535.186	
1792096535.186	- model: `intarsia-metal` on ? (?), url `http://127.0.0.1:8093/v1`
1792096535.186	- thinking: off; run kind: speed; label: cbm0-gap-b3-c556
1792096535.186	- harness: localeval 0.2.1, deepeval 4.2.6, runner endor.kuepker.haus, git 69c7a84
1792096535.186	- command: `/opt/code/deepeval/.venv/bin/localeval speed --url http://127.0.0.1:8093/v1 --model intarsia-metal --sizes 128,1024,4096 --gen 256 --reps 3 --thinking off --label cbm0-gap-b3-c556`
1792096535.186	- result file: `results/20261015-163425-adhoc-cbm0-gap-b3-c556/result.json`
1792096535.186	
1792096535.186	## Speed sweep
1792096535.186	
1792096535.186	served `intarsia-metal`, fingerprint `None`, ctx 8192, 3 reps, 256 forced output tokens
1792096535.186	
1792096535.186	| prompt tok | TTFT s | prefill tok/s | marginal prefill | decode tok/s | peak decode | server prefill | server decode | out tok |
1792096535.186	|---:|---:|---:|---:|---:|---:|---:|---:|---:|
1792096535.186	| 113 | 0.05 | 2088 (2037–2115) * | - | 36.5 (35.9–36.5) | 36.8 (36.6–36.9) | - | - | 256 |
1792096535.186	| 990 | 0.25 | 3895 (3890–3896) * | 4363 | 36.3 (36.1–36.4) | 36.6 (36.5–36.7) | - | - | 256 |
1792096535.186	| 4089 | 1.27 | 3223 (3195–3776) | 3056 | 34.9 (34.5–34.9) | 35.5 (35.3–35.5) | - | - | 256 |
1792096535.186	
1792096535.186	\* under 1024 uncached prompt tokens: TTFT is per-request overhead, not prefill throughput.
1792096535.186	
1792096535.186	Marginal prefill = Δprompt tokens / ΔTTFT from the previous row; fixed request overhead cancels, so it is the cleanest prefill throughput figure.
1792096535.186	
1792096535.186	Peak decode = fastest 32-token stretch of each call (burst speed, e.g. a run of accepted speculative drafts); median and range over reps. '-' = older run or output too short.
1792096535.186	
1792096535.186	
1792096535.186	## Method
1792096535.186	
1792096535.186	Streamed OpenAI-compatible calls, temperature 0. TTFT = request start to first content/reasoning token; prefill tok/s = uncached prompt_tokens / TTFT; decode tok/s = (completion_tokens - 1) / (last - first token), counts from server usage. Speed sweep: fresh nonce first in every prompt (no prefix-cache hits), ignore_eos+min_tokens forces the full length, 1 discarded warmup, median of reps. Other clients' in-flight requests checked before every sweep call. Quality-run speed is incidental (varying prompts), not a sweep.
1792096535.186	
1792096535.186	[localeval] saved /opt/code/deepeval/results/20261015-163425-adhoc-cbm0-gap-b3-c556/result.json
1792096535.919	[localeval] logged to unanimis: d59a519e-3f5c-4dd0-b293-0ab1a4e0f071
