1792096231.365	[localeval] calibrated: 22.1 tokens/sentence, 51 base tokens, ctx=8192
1792096238.358	[localeval]    128 tok rep 1/3: prompt=112 ttft=0.05s prefill=2139 tok/s decode=36.7 tok/s peak=37.0 tok/s out=256
1792096245.395	[localeval]    128 tok rep 2/3: prompt=112 ttft=0.05s prefill=2058 tok/s decode=36.5 tok/s peak=36.8 tok/s out=256
1792096252.378	[localeval]    128 tok rep 3/3: prompt=109 ttft=0.05s prefill=2038 tok/s decode=36.8 tok/s peak=37.0 tok/s out=256
1792096259.605	[localeval]   1024 tok rep 1/3: prompt=990 ttft=0.25s prefill=3908 tok/s decode=36.6 tok/s peak=36.8 tok/s out=256
1792096266.825	[localeval]   1024 tok rep 2/3: prompt=990 ttft=0.25s prefill=3908 tok/s decode=36.6 tok/s peak=36.7 tok/s out=256
1792096274.047	[localeval]   1024 tok rep 3/3: prompt=988 ttft=0.25s prefill=3903 tok/s decode=36.6 tok/s peak=36.7 tok/s out=256
1792096282.299	[localeval]   4096 tok rep 1/3: prompt=4089 ttft=1.08s prefill=3786 tok/s decode=35.6 tok/s peak=35.7 tok/s out=256
1792096290.602	[localeval]   4096 tok rep 2/3: prompt=4085 ttft=1.12s prefill=3656 tok/s decode=35.5 tok/s peak=35.6 tok/s out=256
1792096298.942	[localeval]   4096 tok rep 3/3: prompt=4088 ttft=1.14s prefill=3585 tok/s decode=35.4 tok/s peak=35.6 tok/s out=256
1792096298.943	# localeval: adhoc — 2026-10-15T16:30
1792096298.943	
1792096298.943	- model: `intarsia-metal` on ? (?), url `http://127.0.0.1:8093/v1`
1792096298.943	- thinking: off; run kind: speed; label: cbm0-gap-b1-c556
1792096298.943	- harness: localeval 0.2.1, deepeval 4.2.6, runner endor.kuepker.haus, git 69c7a84
1792096298.943	- command: `/opt/code/deepeval/.venv/bin/localeval speed --url http://127.0.0.1:8093/v1 --model intarsia-metal --sizes 128,1024,4096 --gen 256 --reps 3 --thinking off --label cbm0-gap-b1-c556`
1792096298.943	- result file: `results/20261015-163029-adhoc-cbm0-gap-b1-c556/result.json`
1792096298.943	
1792096298.943	## Speed sweep
1792096298.943	
1792096298.943	served `intarsia-metal`, fingerprint `None`, ctx 8192, 3 reps, 256 forced output tokens
1792096298.943	
1792096298.943	| prompt tok | TTFT s | prefill tok/s | marginal prefill | decode tok/s | peak decode | server prefill | server decode | out tok |
1792096298.943	|---:|---:|---:|---:|---:|---:|---:|---:|---:|
1792096298.943	| 112 | 0.05 | 2058 (2038–2139) * | - | 36.7 (36.5–36.8) | 37.0 (36.8–37.0) | - | - | 256 |
1792096298.943	| 990 | 0.25 | 3908 (3903–3908) * | 4390 | 36.6 (36.6–36.6) | 36.7 (36.7–36.8) | - | - | 256 |
1792096298.943	| 4088 | 1.12 | 3656 (3585–3786) | 3586 | 35.5 (35.4–35.6) | 35.6 (35.6–35.7) | - | - | 256 |
1792096298.943	
1792096298.943	\* under 1024 uncached prompt tokens: TTFT is per-request overhead, not prefill throughput.
1792096298.943	
1792096298.943	Marginal prefill = Δprompt tokens / ΔTTFT from the previous row; fixed request overhead cancels, so it is the cleanest prefill throughput figure.
1792096298.943	
1792096298.943	Peak decode = fastest 32-token stretch of each call (burst speed, e.g. a run of accepted speculative drafts); median and range over reps. '-' = older run or output too short.
1792096298.943	
1792096298.943	
1792096298.943	## Method
1792096298.943	
1792096298.943	Streamed OpenAI-compatible calls, temperature 0. TTFT = request start to first content/reasoning token; prefill tok/s = uncached prompt_tokens / TTFT; decode tok/s = (completion_tokens - 1) / (last - first token), counts from server usage. Speed sweep: fresh nonce first in every prompt (no prefix-cache hits), ignore_eos+min_tokens forces the full length, 1 discarded warmup, median of reps. Other clients' in-flight requests checked before every sweep call. Quality-run speed is incidental (varying prompts), not a sweep.
1792096298.943	
1792096298.943	[localeval] saved /opt/code/deepeval/results/20261015-163029-adhoc-cbm0-gap-b1-c556/result.json
1792096299.668	[localeval] logged to unanimis: 93497c80-98a6-4c84-88a6-99eb1b8bfc0b
