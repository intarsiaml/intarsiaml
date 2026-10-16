1792096584.514	[localeval] calibrated: 22.0 tokens/sentence, 53 base tokens, ctx=8192
1792096591.597	[localeval]    128 tok rep 1/3: prompt=116 ttft=0.05s prefill=2013 tok/s decode=36.3 tok/s peak=36.7 tok/s out=256
1792096598.760	[localeval]    128 tok rep 2/3: prompt=112 ttft=0.05s prefill=2028 tok/s decode=35.9 tok/s peak=36.7 tok/s out=256
1792096605.874	[localeval]    128 tok rep 3/3: prompt=110 ttft=0.05s prefill=2058 tok/s decode=36.1 tok/s peak=36.7 tok/s out=256
1792096613.154	[localeval]   1024 tok rep 1/3: prompt=991 ttft=0.25s prefill=3905 tok/s decode=36.3 tok/s peak=36.5 tok/s out=256
1792096620.601	[localeval]   1024 tok rep 2/3: prompt=990 ttft=0.25s prefill=3879 tok/s decode=35.5 tok/s peak=36.3 tok/s out=256
1792096628.117	[localeval]   1024 tok rep 3/3: prompt=988 ttft=0.25s prefill=3886 tok/s decode=35.1 tok/s peak=36.0 tok/s out=256
1792096636.805	[localeval]   4096 tok rep 1/3: prompt=4084 ttft=1.31s prefill=3110 tok/s decode=34.6 tok/s peak=35.4 tok/s out=256
1792096645.156	[localeval]   4096 tok rep 2/3: prompt=4085 ttft=1.07s prefill=3828 tok/s decode=35.0 tok/s peak=35.4 tok/s out=256
1792096653.721	[localeval]   4096 tok rep 3/3: prompt=4087 ttft=1.27s prefill=3204 tok/s decode=35.0 tok/s peak=35.5 tok/s out=256
1792096653.722	# localeval: adhoc — 2026-10-15T16:36
1792096653.722	
1792096653.722	- model: `intarsia-metal` on ? (?), url `http://127.0.0.1:8093/v1`
1792096653.722	- thinking: off; run kind: speed; label: cbm0-gap-b4-56ec
1792096653.722	- harness: localeval 0.2.1, deepeval 4.2.6, runner endor.kuepker.haus, git 69c7a84
1792096653.722	- command: `/opt/code/deepeval/.venv/bin/localeval speed --url http://127.0.0.1:8093/v1 --model intarsia-metal --sizes 128,1024,4096 --gen 256 --reps 3 --thinking off --label cbm0-gap-b4-56ec`
1792096653.722	- result file: `results/20261015-163623-adhoc-cbm0-gap-b4-56ec/result.json`
1792096653.722	
1792096653.722	## Speed sweep
1792096653.722	
1792096653.722	served `intarsia-metal`, fingerprint `None`, ctx 8192, 3 reps, 256 forced output tokens
1792096653.722	
1792096653.722	| prompt tok | TTFT s | prefill tok/s | marginal prefill | decode tok/s | peak decode | server prefill | server decode | out tok |
1792096653.722	|---:|---:|---:|---:|---:|---:|---:|---:|---:|
1792096653.722	| 112 | 0.05 | 2028 (2013–2058) * | - | 36.1 (35.9–36.3) | 36.7 (36.7–36.7) | - | - | 256 |
1792096653.722	| 990 | 0.25 | 3886 (3879–3905) * | 4390 | 35.5 (35.1–36.3) | 36.3 (36.0–36.5) | - | - | 256 |
1792096653.722	| 4085 | 1.27 | 3204 (3110–3828) | 3031 | 35.0 (34.6–35.0) | 35.4 (35.4–35.5) | - | - | 256 |
1792096653.722	
1792096653.722	\* under 1024 uncached prompt tokens: TTFT is per-request overhead, not prefill throughput.
1792096653.722	
1792096653.722	Marginal prefill = Δprompt tokens / ΔTTFT from the previous row; fixed request overhead cancels, so it is the cleanest prefill throughput figure.
1792096653.722	
1792096653.722	Peak decode = fastest 32-token stretch of each call (burst speed, e.g. a run of accepted speculative drafts); median and range over reps. '-' = older run or output too short.
1792096653.722	
1792096653.722	
1792096653.722	## Method
1792096653.722	
1792096653.722	Streamed OpenAI-compatible calls, temperature 0. TTFT = request start to first content/reasoning token; prefill tok/s = uncached prompt_tokens / TTFT; decode tok/s = (completion_tokens - 1) / (last - first token), counts from server usage. Speed sweep: fresh nonce first in every prompt (no prefix-cache hits), ignore_eos+min_tokens forces the full length, 1 discarded warmup, median of reps. Other clients' in-flight requests checked before every sweep call. Quality-run speed is incidental (varying prompts), not a sweep.
1792096653.722	
1792096653.722	[localeval] saved /opt/code/deepeval/results/20261015-163623-adhoc-cbm0-gap-b4-56ec/result.json
1792096654.548	[localeval] logged to unanimis: 98b3a168-50a9-4c83-9528-82bdc715d553
