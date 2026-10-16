1792054956.391	[localeval] calibrated: 22.0 tokens/sentence, 51 base tokens, ctx=8192
1792054963.380	[localeval]    128 tok rep 1/3: prompt=112 ttft=0.05s prefill=2127 tok/s decode=36.8 tok/s peak=36.9 tok/s out=256
1792054970.384	[localeval]    128 tok rep 2/3: prompt=115 ttft=0.05s prefill=2154 tok/s decode=36.7 tok/s peak=36.8 tok/s out=256
1792054977.383	[localeval]    128 tok rep 3/3: prompt=112 ttft=0.05s prefill=2073 tok/s decode=36.7 tok/s peak=37.0 tok/s out=256
1792054984.614	[localeval]   1024 tok rep 1/3: prompt=989 ttft=0.25s prefill=3901 tok/s decode=36.5 tok/s peak=36.7 tok/s out=256
1792054991.844	[localeval]   1024 tok rep 2/3: prompt=986 ttft=0.25s prefill=3895 tok/s decode=36.5 tok/s peak=36.8 tok/s out=256
1792054999.099	[localeval]   1024 tok rep 3/3: prompt=989 ttft=0.25s prefill=3899 tok/s decode=36.4 tok/s peak=36.8 tok/s out=256
1792055007.508	[localeval]   4096 tok rep 1/3: prompt=4112 ttft=1.24s prefill=3310 tok/s decode=35.6 tok/s peak=35.7 tok/s out=256
1792055015.923	[localeval]   4096 tok rep 2/3: prompt=4110 ttft=1.25s prefill=3296 tok/s decode=35.6 tok/s peak=35.7 tok/s out=256
1792055024.341	[localeval]   4096 tok rep 3/3: prompt=4109 ttft=1.25s prefill=3280 tok/s decode=35.6 tok/s peak=35.7 tok/s out=256
1792055024.342	# localeval: adhoc — 2026-10-15T05:02
1792055024.342	
1792055024.342	- model: `intarsia-metal` on ? (?), url `http://127.0.0.1:8093/v1`
1792055024.342	- thinking: off; run kind: speed; label: clk-cand-r2
1792055024.342	- harness: localeval 0.2.1, deepeval 4.2.6, runner endor.kuepker.haus, git 69c7a84
1792055024.342	- command: `/opt/code/deepeval/.venv/bin/localeval speed --url http://127.0.0.1:8093/v1 --model intarsia-metal --sizes 128,1024,4096 --gen 256 --reps 3 --thinking off --label clk-cand-r2`
1792055024.342	- result file: `results/20261015-050234-adhoc-clk-cand-r2/result.json`
1792055024.342	
1792055024.342	## Speed sweep
1792055024.342	
1792055024.342	served `intarsia-metal`, fingerprint `None`, ctx 8192, 3 reps, 256 forced output tokens
1792055024.342	
1792055024.342	| prompt tok | TTFT s | prefill tok/s | marginal prefill | decode tok/s | peak decode | server prefill | server decode | out tok |
1792055024.342	|---:|---:|---:|---:|---:|---:|---:|---:|---:|
1792055024.342	| 112 | 0.05 | 2127 (2073–2154) * | - | 36.7 (36.7–36.8) | 36.9 (36.8–37.0) | - | - | 256 |
1792055024.342	| 989 | 0.25 | 3899 (3895–3901) * | 4385 | 36.5 (36.4–36.5) | 36.8 (36.7–36.8) | - | - | 256 |
1792055024.342	| 4110 | 1.25 | 3296 (3280–3310) | 3143 | 35.6 (35.6–35.6) | 35.7 (35.7–35.7) | - | - | 256 |
1792055024.342	
1792055024.342	\* under 1024 uncached prompt tokens: TTFT is per-request overhead, not prefill throughput.
1792055024.342	
1792055024.342	Marginal prefill = Δprompt tokens / ΔTTFT from the previous row; fixed request overhead cancels, so it is the cleanest prefill throughput figure.
1792055024.342	
1792055024.342	Peak decode = fastest 32-token stretch of each call (burst speed, e.g. a run of accepted speculative drafts); median and range over reps. '-' = older run or output too short.
1792055024.342	
1792055024.342	
1792055024.342	## Method
1792055024.342	
1792055024.342	Streamed OpenAI-compatible calls, temperature 0. TTFT = request start to first content/reasoning token; prefill tok/s = uncached prompt_tokens / TTFT; decode tok/s = (completion_tokens - 1) / (last - first token), counts from server usage. Speed sweep: fresh nonce first in every prompt (no prefix-cache hits), ignore_eos+min_tokens forces the full length, 1 discarded warmup, median of reps. Other clients' in-flight requests checked before every sweep call. Quality-run speed is incidental (varying prompts), not a sweep.
1792055024.342	
1792055024.342	[localeval] saved /opt/code/deepeval/results/20261015-050234-adhoc-clk-cand-r2/result.json
1792055025.038	[localeval] logged to unanimis: 6566995c-1da7-4c2f-8c8d-1a6a61a6c6c7
