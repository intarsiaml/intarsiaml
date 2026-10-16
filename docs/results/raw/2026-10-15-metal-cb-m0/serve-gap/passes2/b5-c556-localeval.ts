1792096703.169	[localeval] calibrated: 22.1 tokens/sentence, 54 base tokens, ctx=8192
1792096710.217	[localeval]    128 tok rep 1/3: prompt=113 ttft=0.05s prefill=2008 tok/s decode=36.5 tok/s peak=36.7 tok/s out=256
1792096717.266	[localeval]    128 tok rep 2/3: prompt=113 ttft=0.05s prefill=2068 tok/s decode=36.4 tok/s peak=36.7 tok/s out=256
1792096724.379	[localeval]    128 tok rep 3/3: prompt=113 ttft=0.05s prefill=2105 tok/s decode=36.1 tok/s peak=36.9 tok/s out=256
1792096731.694	[localeval]   1024 tok rep 1/3: prompt=992 ttft=0.25s prefill=3909 tok/s decode=36.1 tok/s peak=36.5 tok/s out=256
1792096738.991	[localeval]   1024 tok rep 2/3: prompt=991 ttft=0.25s prefill=3899 tok/s decode=36.2 tok/s peak=36.6 tok/s out=256
1792096746.277	[localeval]   1024 tok rep 3/3: prompt=990 ttft=0.25s prefill=3909 tok/s decode=36.3 tok/s peak=36.6 tok/s out=256
1792096754.820	[localeval]   4096 tok rep 1/3: prompt=4091 ttft=1.26s prefill=3235 tok/s decode=35.0 tok/s peak=35.4 tok/s out=256
1792096763.091	[localeval]   4096 tok rep 2/3: prompt=4086 ttft=1.04s prefill=3916 tok/s decode=35.3 tok/s peak=35.6 tok/s out=256
1792096771.556	[localeval]   4096 tok rep 3/3: prompt=4086 ttft=1.23s prefill=3317 tok/s decode=35.3 tok/s peak=35.5 tok/s out=256
1792096771.557	# localeval: adhoc — 2026-10-15T16:38
1792096771.557	
1792096771.557	- model: `intarsia-metal` on ? (?), url `http://127.0.0.1:8093/v1`
1792096771.557	- thinking: off; run kind: speed; label: cbm0-gap-b5-c556
1792096771.557	- harness: localeval 0.2.1, deepeval 4.2.6, runner endor.kuepker.haus, git 69c7a84
1792096771.557	- command: `/opt/code/deepeval/.venv/bin/localeval speed --url http://127.0.0.1:8093/v1 --model intarsia-metal --sizes 128,1024,4096 --gen 256 --reps 3 --thinking off --label cbm0-gap-b5-c556`
1792096771.557	- result file: `results/20261015-163821-adhoc-cbm0-gap-b5-c556/result.json`
1792096771.557	
1792096771.557	## Speed sweep
1792096771.557	
1792096771.557	served `intarsia-metal`, fingerprint `None`, ctx 8192, 3 reps, 256 forced output tokens
1792096771.557	
1792096771.557	| prompt tok | TTFT s | prefill tok/s | marginal prefill | decode tok/s | peak decode | server prefill | server decode | out tok |
1792096771.557	|---:|---:|---:|---:|---:|---:|---:|---:|---:|
1792096771.557	| 113 | 0.05 | 2068 (2008–2105) * | - | 36.4 (36.1–36.5) | 36.7 (36.7–36.9) | - | - | 256 |
1792096771.557	| 991 | 0.25 | 3909 (3899–3909) * | 4390 | 36.2 (36.1–36.3) | 36.6 (36.5–36.6) | - | - | 256 |
1792096771.557	| 4086 | 1.23 | 3317 (3235–3916) | 3165 | 35.3 (35.0–35.3) | 35.5 (35.4–35.6) | - | - | 256 |
1792096771.557	
1792096771.557	\* under 1024 uncached prompt tokens: TTFT is per-request overhead, not prefill throughput.
1792096771.557	
1792096771.557	Marginal prefill = Δprompt tokens / ΔTTFT from the previous row; fixed request overhead cancels, so it is the cleanest prefill throughput figure.
1792096771.557	
1792096771.557	Peak decode = fastest 32-token stretch of each call (burst speed, e.g. a run of accepted speculative drafts); median and range over reps. '-' = older run or output too short.
1792096771.557	
1792096771.557	
1792096771.557	## Method
1792096771.557	
1792096771.557	Streamed OpenAI-compatible calls, temperature 0. TTFT = request start to first content/reasoning token; prefill tok/s = uncached prompt_tokens / TTFT; decode tok/s = (completion_tokens - 1) / (last - first token), counts from server usage. Speed sweep: fresh nonce first in every prompt (no prefix-cache hits), ignore_eos+min_tokens forces the full length, 1 discarded warmup, median of reps. Other clients' in-flight requests checked before every sweep call. Quality-run speed is incidental (varying prompts), not a sweep.
1792096771.557	
1792096771.557	[localeval] saved /opt/code/deepeval/results/20261015-163821-adhoc-cbm0-gap-b5-c556/result.json
1792096772.268	[localeval] logged to unanimis: 3a8c2706-a2ac-49d1-9d24-d7e4401edb32
