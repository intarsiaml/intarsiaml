1792054379.098	[localeval] calibrated: 22.1 tokens/sentence, 48 base tokens, ctx=8192
1792054386.154	[localeval]    128 tok rep 1/3: prompt=135 ttft=0.05s prefill=2402 tok/s decode=36.4 tok/s peak=36.8 tok/s out=256
1792054393.127	[localeval]    128 tok rep 2/3: prompt=135 ttft=0.05s prefill=2371 tok/s decode=36.9 tok/s peak=37.0 tok/s out=256
1792054400.122	[localeval]    128 tok rep 3/3: prompt=136 ttft=0.05s prefill=2404 tok/s decode=36.7 tok/s peak=37.0 tok/s out=256
1792054407.384	[localeval]   1024 tok rep 1/3: prompt=992 ttft=0.25s prefill=3913 tok/s decode=36.4 tok/s peak=36.7 tok/s out=256
1792054414.613	[localeval]   1024 tok rep 2/3: prompt=991 ttft=0.25s prefill=3904 tok/s decode=36.6 tok/s peak=36.7 tok/s out=256
1792054421.845	[localeval]   1024 tok rep 3/3: prompt=990 ttft=0.25s prefill=3905 tok/s decode=36.5 tok/s peak=36.8 tok/s out=256
1792054430.076	[localeval]   4096 tok rep 1/3: prompt=4091 ttft=1.07s prefill=3832 tok/s decode=35.6 tok/s peak=35.8 tok/s out=256
1792054438.373	[localeval]   4096 tok rep 2/3: prompt=4089 ttft=1.09s prefill=3763 tok/s decode=35.4 tok/s peak=35.7 tok/s out=256
1792054446.667	[localeval]   4096 tok rep 3/3: prompt=4087 ttft=1.09s prefill=3733 tok/s decode=35.4 tok/s peak=35.6 tok/s out=256
1792054446.668	# localeval: adhoc — 2026-10-15T04:52
1792054446.668	
1792054446.668	- model: `intarsia-metal` on ? (?), url `http://127.0.0.1:8093/v1`
1792054446.668	- thinking: off; run kind: speed; label: clk-cand-r1
1792054446.668	- harness: localeval 0.2.1, deepeval 4.2.6, runner endor.kuepker.haus, git 69c7a84
1792054446.668	- command: `/opt/code/deepeval/.venv/bin/localeval speed --url http://127.0.0.1:8093/v1 --model intarsia-metal --sizes 128,1024,4096 --gen 256 --reps 3 --thinking off --label clk-cand-r1`
1792054446.668	- result file: `results/20261015-045257-adhoc-clk-cand-r1/result.json`
1792054446.668	
1792054446.668	## Speed sweep
1792054446.668	
1792054446.668	served `intarsia-metal`, fingerprint `None`, ctx 8192, 3 reps, 256 forced output tokens
1792054446.668	
1792054446.668	| prompt tok | TTFT s | prefill tok/s | marginal prefill | decode tok/s | peak decode | server prefill | server decode | out tok |
1792054446.668	|---:|---:|---:|---:|---:|---:|---:|---:|---:|
1792054446.668	| 135 | 0.06 | 2402 (2371–2404) * | - | 36.7 (36.4–36.9) | 37.0 (36.8–37.0) | - | - | 256 |
1792054446.668	| 991 | 0.25 | 3905 (3904–3913) * | 4345 | 36.5 (36.4–36.6) | 36.7 (36.7–36.8) | - | - | 256 |
1792054446.668	| 4089 | 1.08 | 3763 (3733–3832) | 3719 | 35.4 (35.4–35.6) | 35.7 (35.6–35.8) | - | - | 256 |
1792054446.668	
1792054446.668	\* under 1024 uncached prompt tokens: TTFT is per-request overhead, not prefill throughput.
1792054446.668	
1792054446.668	Marginal prefill = Δprompt tokens / ΔTTFT from the previous row; fixed request overhead cancels, so it is the cleanest prefill throughput figure.
1792054446.668	
1792054446.668	Peak decode = fastest 32-token stretch of each call (burst speed, e.g. a run of accepted speculative drafts); median and range over reps. '-' = older run or output too short.
1792054446.668	
1792054446.668	
1792054446.668	## Method
1792054446.668	
1792054446.668	Streamed OpenAI-compatible calls, temperature 0. TTFT = request start to first content/reasoning token; prefill tok/s = uncached prompt_tokens / TTFT; decode tok/s = (completion_tokens - 1) / (last - first token), counts from server usage. Speed sweep: fresh nonce first in every prompt (no prefix-cache hits), ignore_eos+min_tokens forces the full length, 1 discarded warmup, median of reps. Other clients' in-flight requests checked before every sweep call. Quality-run speed is incidental (varying prompts), not a sweep.
1792054446.668	
1792054446.668	[localeval] saved /opt/code/deepeval/results/20261015-045257-adhoc-clk-cand-r1/result.json
1792054447.358	[localeval] logged to unanimis: 3841c235-9a46-4acb-ba7f-98452a4c6724
