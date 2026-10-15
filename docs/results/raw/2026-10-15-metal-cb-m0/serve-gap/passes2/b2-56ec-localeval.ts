1792096348.351	[localeval] calibrated: 22.1 tokens/sentence, 52 base tokens, ctx=8192
1792096355.363	[localeval]    128 tok rep 1/3: prompt=111 ttft=0.05s prefill=2098 tok/s decode=36.6 tok/s peak=36.8 tok/s out=256
1792096362.323	[localeval]    128 tok rep 2/3: prompt=113 ttft=0.05s prefill=2107 tok/s decode=36.9 tok/s peak=37.0 tok/s out=256
1792096369.344	[localeval]    128 tok rep 3/3: prompt=115 ttft=0.05s prefill=2149 tok/s decode=36.6 tok/s peak=36.9 tok/s out=256
1792096376.584	[localeval]   1024 tok rep 1/3: prompt=991 ttft=0.25s prefill=3894 tok/s decode=36.5 tok/s peak=36.7 tok/s out=256
1792096383.876	[localeval]   1024 tok rep 2/3: prompt=990 ttft=0.25s prefill=3894 tok/s decode=36.2 tok/s peak=36.6 tok/s out=256
1792096391.123	[localeval]   1024 tok rep 3/3: prompt=988 ttft=0.25s prefill=3885 tok/s decode=36.5 tok/s peak=36.6 tok/s out=256
1792096399.768	[localeval]   4096 tok rep 1/3: prompt=4085 ttft=1.23s prefill=3312 tok/s decode=34.4 tok/s peak=35.7 tok/s out=256
1792096408.524	[localeval]   4096 tok rep 2/3: prompt=4084 ttft=1.12s prefill=3638 tok/s decode=33.4 tok/s peak=34.7 tok/s out=256
1792096417.058	[localeval]   4096 tok rep 3/3: prompt=4085 ttft=1.29s prefill=3171 tok/s decode=35.2 tok/s peak=35.4 tok/s out=256
1792096417.058	# localeval: adhoc — 2026-10-15T16:32
1792096417.058	
1792096417.058	- model: `intarsia-metal` on ? (?), url `http://127.0.0.1:8093/v1`
1792096417.058	- thinking: off; run kind: speed; label: cbm0-gap-b2-56ec
1792096417.058	- harness: localeval 0.2.1, deepeval 4.2.6, runner endor.kuepker.haus, git 69c7a84
1792096417.058	- command: `/opt/code/deepeval/.venv/bin/localeval speed --url http://127.0.0.1:8093/v1 --model intarsia-metal --sizes 128,1024,4096 --gen 256 --reps 3 --thinking off --label cbm0-gap-b2-56ec`
1792096417.058	- result file: `results/20261015-163226-adhoc-cbm0-gap-b2-56ec/result.json`
1792096417.058	
1792096417.058	## Speed sweep
1792096417.058	
1792096417.058	served `intarsia-metal`, fingerprint `None`, ctx 8192, 3 reps, 256 forced output tokens
1792096417.058	
1792096417.058	| prompt tok | TTFT s | prefill tok/s | marginal prefill | decode tok/s | peak decode | server prefill | server decode | out tok |
1792096417.058	|---:|---:|---:|---:|---:|---:|---:|---:|---:|
1792096417.058	| 113 | 0.05 | 2107 (2098–2149) * | - | 36.6 (36.6–36.9) | 36.9 (36.8–37.0) | - | - | 256 |
1792096417.058	| 990 | 0.25 | 3894 (3885–3894) * | 4342 | 36.5 (36.2–36.5) | 36.6 (36.6–36.7) | - | - | 256 |
1792096417.058	| 4085 | 1.23 | 3312 (3171–3638) | 3161 | 34.4 (33.4–35.2) | 35.4 (34.7–35.7) | - | - | 256 |
1792096417.058	
1792096417.058	\* under 1024 uncached prompt tokens: TTFT is per-request overhead, not prefill throughput.
1792096417.058	
1792096417.058	Marginal prefill = Δprompt tokens / ΔTTFT from the previous row; fixed request overhead cancels, so it is the cleanest prefill throughput figure.
1792096417.058	
1792096417.058	Peak decode = fastest 32-token stretch of each call (burst speed, e.g. a run of accepted speculative drafts); median and range over reps. '-' = older run or output too short.
1792096417.058	
1792096417.058	
1792096417.058	## Method
1792096417.058	
1792096417.058	Streamed OpenAI-compatible calls, temperature 0. TTFT = request start to first content/reasoning token; prefill tok/s = uncached prompt_tokens / TTFT; decode tok/s = (completion_tokens - 1) / (last - first token), counts from server usage. Speed sweep: fresh nonce first in every prompt (no prefix-cache hits), ignore_eos+min_tokens forces the full length, 1 discarded warmup, median of reps. Other clients' in-flight requests checked before every sweep call. Quality-run speed is incidental (varying prompts), not a sweep.
1792096417.058	
1792096417.058	[localeval] saved /opt/code/deepeval/results/20261015-163226-adhoc-cbm0-gap-b2-56ec/result.json
1792096417.843	[localeval] logged to unanimis: 9230db55-63a7-4849-93a4-f700b317779e
