1792055068.873	[localeval] calibrated: 22.0 tokens/sentence, 53 base tokens, ctx=8192
1792055075.885	[localeval]    128 tok rep 1/3: prompt=114 ttft=0.05s prefill=2020 tok/s decode=36.7 tok/s peak=36.9 tok/s out=256
1792055082.885	[localeval]    128 tok rep 2/3: prompt=111 ttft=0.05s prefill=2058 tok/s decode=36.7 tok/s peak=37.0 tok/s out=256
1792055089.843	[localeval]    128 tok rep 3/3: prompt=112 ttft=0.05s prefill=2081 tok/s decode=36.9 tok/s peak=37.0 tok/s out=256
1792055097.095	[localeval]   1024 tok rep 1/3: prompt=991 ttft=0.25s prefill=3914 tok/s decode=36.4 tok/s peak=36.8 tok/s out=256
1792055104.327	[localeval]   1024 tok rep 2/3: prompt=990 ttft=0.25s prefill=3902 tok/s decode=36.5 tok/s peak=36.7 tok/s out=256
1792055111.554	[localeval]   1024 tok rep 3/3: prompt=992 ttft=0.25s prefill=3912 tok/s decode=36.6 tok/s peak=36.8 tok/s out=256
1792055119.750	[localeval]   4096 tok rep 1/3: prompt=4085 ttft=1.04s prefill=3909 tok/s decode=35.7 tok/s peak=35.7 tok/s out=256
1792055127.976	[localeval]   4096 tok rep 2/3: prompt=4086 ttft=1.05s prefill=3880 tok/s decode=35.5 tok/s peak=35.7 tok/s out=256
1792055136.196	[localeval]   4096 tok rep 3/3: prompt=4086 ttft=1.06s prefill=3851 tok/s decode=35.6 tok/s peak=35.7 tok/s out=256
1792055136.197	# localeval: adhoc — 2026-10-15T05:04
1792055136.197	
1792055136.197	- model: `intarsia-metal` on ? (?), url `http://127.0.0.1:8093/v1`
1792055136.197	- thinking: off; run kind: speed; label: clk-cand-r3
1792055136.197	- harness: localeval 0.2.1, deepeval 4.2.6, runner endor.kuepker.haus, git 69c7a84
1792055136.197	- command: `/opt/code/deepeval/.venv/bin/localeval speed --url http://127.0.0.1:8093/v1 --model intarsia-metal --sizes 128,1024,4096 --gen 256 --reps 3 --thinking off --label clk-cand-r3`
1792055136.197	- result file: `results/20261015-050427-adhoc-clk-cand-r3/result.json`
1792055136.197	
1792055136.197	## Speed sweep
1792055136.197	
1792055136.197	served `intarsia-metal`, fingerprint `None`, ctx 8192, 3 reps, 256 forced output tokens
1792055136.197	
1792055136.197	| prompt tok | TTFT s | prefill tok/s | marginal prefill | decode tok/s | peak decode | server prefill | server decode | out tok |
1792055136.197	|---:|---:|---:|---:|---:|---:|---:|---:|---:|
1792055136.197	| 112 | 0.05 | 2058 (2020–2081) * | - | 36.7 (36.7–36.9) | 37.0 (36.9–37.0) | - | - | 256 |
1792055136.197	| 991 | 0.25 | 3912 (3902–3914) * | 4373 | 36.5 (36.4–36.6) | 36.8 (36.7–36.8) | - | - | 256 |
1792055136.197	| 4086 | 1.05 | 3880 (3851–3909) | 3869 | 35.6 (35.5–35.7) | 35.7 (35.7–35.7) | - | - | 256 |
1792055136.197	
1792055136.197	\* under 1024 uncached prompt tokens: TTFT is per-request overhead, not prefill throughput.
1792055136.197	
1792055136.197	Marginal prefill = Δprompt tokens / ΔTTFT from the previous row; fixed request overhead cancels, so it is the cleanest prefill throughput figure.
1792055136.197	
1792055136.197	Peak decode = fastest 32-token stretch of each call (burst speed, e.g. a run of accepted speculative drafts); median and range over reps. '-' = older run or output too short.
1792055136.197	
1792055136.197	
1792055136.197	## Method
1792055136.197	
1792055136.197	Streamed OpenAI-compatible calls, temperature 0. TTFT = request start to first content/reasoning token; prefill tok/s = uncached prompt_tokens / TTFT; decode tok/s = (completion_tokens - 1) / (last - first token), counts from server usage. Speed sweep: fresh nonce first in every prompt (no prefix-cache hits), ignore_eos+min_tokens forces the full length, 1 discarded warmup, median of reps. Other clients' in-flight requests checked before every sweep call. Quality-run speed is incidental (varying prompts), not a sweep.
1792055136.197	
1792055136.197	[localeval] saved /opt/code/deepeval/results/20261015-050427-adhoc-clk-cand-r3/result.json
1792055136.894	[localeval] logged to unanimis: 62445ee6-8870-416b-903f-67bcf5d16203
