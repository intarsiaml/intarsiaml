1792081774.769	[localeval] calibrated: 22.0 tokens/sentence, 55 base tokens, ctx=8192
1792081781.564	[localeval]    128 tok rep 1/3: prompt=113 ttft=0.05s prefill=2143 tok/s decode=37.8 tok/s peak=42.8 tok/s out=256
1792081788.566	[localeval]    128 tok rep 2/3: prompt=112 ttft=0.05s prefill=2075 tok/s decode=36.7 tok/s peak=39.0 tok/s out=256
1792081795.406	[localeval]    128 tok rep 3/3: prompt=112 ttft=0.05s prefill=2100 tok/s decode=37.6 tok/s peak=39.3 tok/s out=256
1792081802.327	[localeval]   1024 tok rep 1/3: prompt=990 ttft=0.25s prefill=3905 tok/s decode=38.2 tok/s peak=44.4 tok/s out=256
1792081809.195	[localeval]   1024 tok rep 2/3: prompt=993 ttft=0.25s prefill=3906 tok/s decode=38.6 tok/s peak=49.1 tok/s out=256
1792081815.802	[localeval]   1024 tok rep 3/3: prompt=989 ttft=0.25s prefill=3881 tok/s decode=40.1 tok/s peak=54.0 tok/s out=256
1792081823.221	[localeval]   4096 tok rep 1/3: prompt=4086 ttft=1.09s prefill=3738 tok/s decode=40.3 tok/s peak=54.9 tok/s out=256
1792081830.833	[localeval]   4096 tok rep 2/3: prompt=4087 ttft=1.11s prefill=3665 tok/s decode=39.2 tok/s peak=46.5 tok/s out=256
1792081838.647	[localeval]   4096 tok rep 3/3: prompt=4084 ttft=1.30s prefill=3145 tok/s decode=39.1 tok/s peak=48.1 tok/s out=256
1792081838.648	# localeval: adhoc — 2026-10-15T12:29
1792081838.648	
1792081838.648	- model: `intarsia-metal` on ? (?), url `http://127.0.0.1:8093/v1`
1792081838.648	- thinking: off; run kind: speed; label: p3-on
1792081838.648	- harness: localeval 0.2.1, deepeval 4.2.6, runner endor.kuepker.haus, git 69c7a84
1792081838.648	- command: `/opt/code/deepeval/.venv/bin/localeval speed --url http://127.0.0.1:8093/v1 --model intarsia-metal --sizes 128,1024,4096 --gen 256 --reps 3 --thinking off --label p3-on`
1792081838.648	- result file: `results/20261015-122933-adhoc-p3-on/result.json`
1792081838.648	
1792081838.648	## Speed sweep
1792081838.648	
1792081838.648	served `intarsia-metal`, fingerprint `None`, ctx 8192, 3 reps, 256 forced output tokens
1792081838.648	
1792081838.648	| prompt tok | TTFT s | prefill tok/s | marginal prefill | decode tok/s | peak decode | server prefill | server decode | out tok |
1792081838.648	|---:|---:|---:|---:|---:|---:|---:|---:|---:|
1792081838.648	| 112 | 0.05 | 2100 (2075–2143) * | - | 37.6 (36.7–37.8) | 39.3 (39.0–42.8) | - | - | 256 |
1792081838.648	| 990 | 0.25 | 3905 (3881–3906) * | 4346 | 38.6 (38.2–40.1) | 49.1 (44.4–54.0) | - | - | 256 |
1792081838.648	| 4086 | 1.11 | 3665 (3145–3738) | 3596 | 39.2 (39.1–40.3) | 48.1 (46.5–54.9) | - | - | 256 |
1792081838.648	
1792081838.648	\* under 1024 uncached prompt tokens: TTFT is per-request overhead, not prefill throughput.
1792081838.648	
1792081838.648	Marginal prefill = Δprompt tokens / ΔTTFT from the previous row; fixed request overhead cancels, so it is the cleanest prefill throughput figure.
1792081838.648	
1792081838.648	Peak decode = fastest 32-token stretch of each call (burst speed, e.g. a run of accepted speculative drafts); median and range over reps. '-' = older run or output too short.
1792081838.648	
1792081838.648	
1792081838.648	## Method
1792081838.648	
1792081838.648	Streamed OpenAI-compatible calls, temperature 0. TTFT = request start to first content/reasoning token; prefill tok/s = uncached prompt_tokens / TTFT; decode tok/s = (completion_tokens - 1) / (last - first token), counts from server usage. Speed sweep: fresh nonce first in every prompt (no prefix-cache hits), ignore_eos+min_tokens forces the full length, 1 discarded warmup, median of reps. Other clients' in-flight requests checked before every sweep call. Quality-run speed is incidental (varying prompts), not a sweep.
1792081838.649	
1792081838.649	[localeval] saved /opt/code/deepeval/results/20261015-122933-adhoc-p3-on/result.json
1792081839.433	[localeval] logged to unanimis: faa98abc-ffe8-4747-b1d6-0e6885fb7797
