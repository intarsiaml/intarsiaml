1792081889.191	[localeval] calibrated: 22.1 tokens/sentence, 52 base tokens, ctx=8192
1792081896.341	[localeval]    128 tok rep 1/3: prompt=114 ttft=0.05s prefill=2149 tok/s decode=35.9 tok/s peak=36.6 tok/s out=256
1792081903.395	[localeval]    128 tok rep 2/3: prompt=114 ttft=0.05s prefill=2065 tok/s decode=36.4 tok/s peak=36.8 tok/s out=256
1792081910.453	[localeval]    128 tok rep 3/3: prompt=113 ttft=0.05s prefill=2115 tok/s decode=36.4 tok/s peak=36.9 tok/s out=256
1792081917.715	[localeval]   1024 tok rep 1/3: prompt=992 ttft=0.25s prefill=3895 tok/s decode=36.4 tok/s peak=36.7 tok/s out=256
1792081925.054	[localeval]   1024 tok rep 2/3: prompt=992 ttft=0.25s prefill=3904 tok/s decode=36.0 tok/s peak=36.6 tok/s out=256
1792081932.453	[localeval]   1024 tok rep 3/3: prompt=990 ttft=0.26s prefill=3863 tok/s decode=35.7 tok/s peak=36.2 tok/s out=256
1792081940.847	[localeval]   4096 tok rep 1/3: prompt=4090 ttft=1.06s prefill=3867 tok/s decode=34.8 tok/s peak=35.5 tok/s out=256
1792081949.164	[localeval]   4096 tok rep 2/3: prompt=4083 ttft=1.06s prefill=3848 tok/s decode=35.1 tok/s peak=35.5 tok/s out=256
1792081957.586	[localeval]   4096 tok rep 3/3: prompt=4090 ttft=1.08s prefill=3798 tok/s decode=34.7 tok/s peak=35.5 tok/s out=256
1792081957.586	# localeval: adhoc — 2026-10-15T12:31
1792081957.587	
1792081957.587	- model: `intarsia-metal` on ? (?), url `http://127.0.0.1:8093/v1`
1792081957.587	- thinking: off; run kind: speed; label: p4-off
1792081957.587	- harness: localeval 0.2.1, deepeval 4.2.6, runner endor.kuepker.haus, git 69c7a84
1792081957.587	- command: `/opt/code/deepeval/.venv/bin/localeval speed --url http://127.0.0.1:8093/v1 --model intarsia-metal --sizes 128,1024,4096 --gen 256 --reps 3 --thinking off --label p4-off`
1792081957.587	- result file: `results/20261015-123127-adhoc-p4-off/result.json`
1792081957.587	
1792081957.587	## Speed sweep
1792081957.587	
1792081957.587	served `intarsia-metal`, fingerprint `None`, ctx 8192, 3 reps, 256 forced output tokens
1792081957.587	
1792081957.587	| prompt tok | TTFT s | prefill tok/s | marginal prefill | decode tok/s | peak decode | server prefill | server decode | out tok |
1792081957.587	|---:|---:|---:|---:|---:|---:|---:|---:|---:|
1792081957.587	| 114 | 0.05 | 2115 (2065–2149) * | - | 36.4 (35.9–36.4) | 36.8 (36.6–36.9) | - | - | 256 |
1792081957.587	| 992 | 0.25 | 3895 (3863–3904) * | 4368 | 36.0 (35.7–36.4) | 36.6 (36.2–36.7) | - | - | 256 |
1792081957.587	| 4090 | 1.06 | 3848 (3798–3867) | 3839 | 34.8 (34.7–35.1) | 35.5 (35.5–35.5) | - | - | 256 |
1792081957.587	
1792081957.587	\* under 1024 uncached prompt tokens: TTFT is per-request overhead, not prefill throughput.
1792081957.587	
1792081957.587	Marginal prefill = Δprompt tokens / ΔTTFT from the previous row; fixed request overhead cancels, so it is the cleanest prefill throughput figure.
1792081957.587	
1792081957.587	Peak decode = fastest 32-token stretch of each call (burst speed, e.g. a run of accepted speculative drafts); median and range over reps. '-' = older run or output too short.
1792081957.587	
1792081957.587	
1792081957.587	## Method
1792081957.587	
1792081957.587	Streamed OpenAI-compatible calls, temperature 0. TTFT = request start to first content/reasoning token; prefill tok/s = uncached prompt_tokens / TTFT; decode tok/s = (completion_tokens - 1) / (last - first token), counts from server usage. Speed sweep: fresh nonce first in every prompt (no prefix-cache hits), ignore_eos+min_tokens forces the full length, 1 discarded warmup, median of reps. Other clients' in-flight requests checked before every sweep call. Quality-run speed is incidental (varying prompts), not a sweep.
1792081957.587	
1792081957.587	[localeval] saved /opt/code/deepeval/results/20261015-123127-adhoc-p4-off/result.json
1792081958.253	[localeval] logged to unanimis: 40a1a11f-9837-444f-be55-609d5d31cd82
