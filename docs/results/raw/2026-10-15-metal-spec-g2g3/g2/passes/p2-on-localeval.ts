1792081650.797	[localeval] calibrated: 22.1 tokens/sentence, 49 base tokens, ctx=8192
1792081657.866	[localeval]    128 tok rep 1/3: prompt=135 ttft=0.05s prefill=2401 tok/s decode=36.4 tok/s peak=38.7 tok/s out=256
1792081664.621	[localeval]    128 tok rep 2/3: prompt=137 ttft=0.06s prefill=2416 tok/s decode=38.1 tok/s peak=46.0 tok/s out=256
1792081671.438	[localeval]    128 tok rep 3/3: prompt=137 ttft=0.06s prefill=2309 tok/s decode=37.7 tok/s peak=42.9 tok/s out=256
1792081678.355	[localeval]   1024 tok rep 1/3: prompt=992 ttft=0.25s prefill=3876 tok/s decode=38.3 tok/s peak=44.3 tok/s out=256
1792081685.227	[localeval]   1024 tok rep 2/3: prompt=992 ttft=0.25s prefill=3905 tok/s decode=38.5 tok/s peak=43.8 tok/s out=256
1792081692.031	[localeval]   1024 tok rep 3/3: prompt=991 ttft=0.25s prefill=3905 tok/s decode=38.9 tok/s peak=45.3 tok/s out=256
1792081699.742	[localeval]   4096 tok rep 1/3: prompt=4088 ttft=1.11s prefill=3685 tok/s decode=38.6 tok/s peak=46.3 tok/s out=256
1792081707.385	[localeval]   4096 tok rep 2/3: prompt=4090 ttft=1.15s prefill=3546 tok/s decode=39.3 tok/s peak=46.6 tok/s out=256
1792081715.013	[localeval]   4096 tok rep 3/3: prompt=4089 ttft=1.17s prefill=3483 tok/s decode=39.5 tok/s peak=46.4 tok/s out=256
1792081715.014	# localeval: adhoc — 2026-10-15T12:27
1792081715.014	
1792081715.014	- model: `intarsia-metal` on ? (?), url `http://127.0.0.1:8093/v1`
1792081715.014	- thinking: off; run kind: speed; label: p2-on
1792081715.014	- harness: localeval 0.2.1, deepeval 4.2.6, runner endor.kuepker.haus, git 69c7a84
1792081715.014	- command: `/opt/code/deepeval/.venv/bin/localeval speed --url http://127.0.0.1:8093/v1 --model intarsia-metal --sizes 128,1024,4096 --gen 256 --reps 3 --thinking off --label p2-on`
1792081715.014	- result file: `results/20261015-122729-adhoc-p2-on/result.json`
1792081715.014	
1792081715.014	## Speed sweep
1792081715.014	
1792081715.014	served `intarsia-metal`, fingerprint `None`, ctx 8192, 3 reps, 256 forced output tokens
1792081715.014	
1792081715.014	| prompt tok | TTFT s | prefill tok/s | marginal prefill | decode tok/s | peak decode | server prefill | server decode | out tok |
1792081715.014	|---:|---:|---:|---:|---:|---:|---:|---:|---:|
1792081715.014	| 137 | 0.06 | 2401 (2309–2416) * | - | 37.7 (36.4–38.1) | 42.9 (38.7–46.0) | - | - | 256 |
1792081715.014	| 992 | 0.25 | 3905 (3876–3905) * | 4318 | 38.5 (38.3–38.9) | 44.3 (43.8–45.3) | - | - | 256 |
1792081715.014	| 4089 | 1.15 | 3546 (3483–3685) | 3445 | 39.3 (38.6–39.5) | 46.4 (46.3–46.6) | - | - | 256 |
1792081715.014	
1792081715.014	\* under 1024 uncached prompt tokens: TTFT is per-request overhead, not prefill throughput.
1792081715.014	
1792081715.014	Marginal prefill = Δprompt tokens / ΔTTFT from the previous row; fixed request overhead cancels, so it is the cleanest prefill throughput figure.
1792081715.014	
1792081715.014	Peak decode = fastest 32-token stretch of each call (burst speed, e.g. a run of accepted speculative drafts); median and range over reps. '-' = older run or output too short.
1792081715.014	
1792081715.014	
1792081715.014	## Method
1792081715.014	
1792081715.014	Streamed OpenAI-compatible calls, temperature 0. TTFT = request start to first content/reasoning token; prefill tok/s = uncached prompt_tokens / TTFT; decode tok/s = (completion_tokens - 1) / (last - first token), counts from server usage. Speed sweep: fresh nonce first in every prompt (no prefix-cache hits), ignore_eos+min_tokens forces the full length, 1 discarded warmup, median of reps. Other clients' in-flight requests checked before every sweep call. Quality-run speed is incidental (varying prompts), not a sweep.
1792081715.014	
1792081715.014	[localeval] saved /opt/code/deepeval/results/20261015-122729-adhoc-p2-on/result.json
1792081715.768	[localeval] logged to unanimis: 242cf760-1b79-4c7a-82cf-499d33f530f3
