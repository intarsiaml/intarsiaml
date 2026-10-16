1792055150.679	[localeval] calibrated: 22.1 tokens/sentence, 53 base tokens, ctx=8192
1792055158.004	[localeval]    128 tok rep 1/3: prompt=111 ttft=0.09s prefill=1258 tok/s decode=35.2 tok/s peak=35.5 tok/s out=256
1792055165.309	[localeval]    128 tok rep 2/3: prompt=114 ttft=0.09s prefill=1235 tok/s decode=35.4 tok/s peak=35.5 tok/s out=256
1792055172.601	[localeval]    128 tok rep 3/3: prompt=113 ttft=0.09s prefill=1211 tok/s decode=35.4 tok/s peak=35.5 tok/s out=256
1792055180.266	[localeval]   1024 tok rep 1/3: prompt=989 ttft=0.39s prefill=2552 tok/s decode=35.0 tok/s peak=35.2 tok/s out=256
1792055187.933	[localeval]   1024 tok rep 2/3: prompt=990 ttft=0.39s prefill=2555 tok/s decode=35.0 tok/s peak=35.2 tok/s out=256
1792055195.616	[localeval]   1024 tok rep 3/3: prompt=993 ttft=0.39s prefill=2554 tok/s decode=35.0 tok/s peak=35.1 tok/s out=256
1792055204.835	[localeval]   4096 tok rep 1/3: prompt=4088 ttft=1.64s prefill=2485 tok/s decode=33.7 tok/s peak=34.1 tok/s out=256
1792055213.974	[localeval]   4096 tok rep 2/3: prompt=4091 ttft=1.64s prefill=2498 tok/s decode=34.0 tok/s peak=34.1 tok/s out=256
1792055223.176	[localeval]   4096 tok rep 3/3: prompt=4089 ttft=1.64s prefill=2492 tok/s decode=33.7 tok/s peak=34.0 tok/s out=256
1792055223.176	# localeval: adhoc — 2026-10-15T05:05
1792055223.176	
1792055223.176	- model: `/Users/josef/models/Qwen3-8B-GGUF/Qwen3-8B-BF16.gguf` on ? (?), url `http://127.0.0.1:8091/v1`
1792055223.176	- thinking: off; run kind: speed; label: clk-llamacpp-r3
1792055223.176	- harness: localeval 0.2.1, deepeval 4.2.6, runner endor.kuepker.haus, git 69c7a84
1792055223.176	- command: `/opt/code/deepeval/.venv/bin/localeval speed --url http://127.0.0.1:8091/v1 --model /Users/josef/models/Qwen3-8B-GGUF/Qwen3-8B-BF16.gguf --sizes 128,1024,4096 --gen 256 --reps 3 --thinking off --label clk-llamacpp-r3`
1792055223.176	- result file: `results/20261015-050548-adhoc-clk-llamacpp-r3/result.json`
1792055223.176	
1792055223.176	## Speed sweep
1792055223.176	
1792055223.176	served `/Users/josef/models/Qwen3-8B-GGUF/Qwen3-8B-BF16.gguf`, fingerprint `b11429-d81235049`, ctx 8192, 3 reps, 256 forced output tokens
1792055223.176	
1792055223.176	| prompt tok | TTFT s | prefill tok/s | marginal prefill | decode tok/s | peak decode | server prefill | server decode | out tok |
1792055223.176	|---:|---:|---:|---:|---:|---:|---:|---:|---:|
1792055223.176	| 113 | 0.09 | 1235 (1211–1258) * | - | 35.4 (35.2–35.4) | 35.5 (35.5–35.5) | 1288 | 35.4 | 256 |
1792055223.176	| 990 | 0.39 | 2554 (2552–2555) * | 2963 | 35.0 (35.0–35.0) | 35.2 (35.1–35.2) | 2578 | 35.0 | 256 |
1792055223.176	| 4089 | 1.64 | 2492 (2485–2498) | 2469 | 33.7 (33.7–34.0) | 34.1 (34.0–34.1) | 2500 | 33.7 | 256 |
1792055223.176	
1792055223.176	\* under 1024 uncached prompt tokens: TTFT is per-request overhead, not prefill throughput.
1792055223.176	
1792055223.176	Marginal prefill = Δprompt tokens / ΔTTFT from the previous row; fixed request overhead cancels, so it is the cleanest prefill throughput figure.
1792055223.176	
1792055223.176	Peak decode = fastest 32-token stretch of each call (burst speed, e.g. a run of accepted speculative drafts); median and range over reps. '-' = older run or output too short.
1792055223.176	
1792055223.176	
1792055223.176	## Method
1792055223.176	
1792055223.176	Streamed OpenAI-compatible calls, temperature 0. TTFT = request start to first content/reasoning token; prefill tok/s = uncached prompt_tokens / TTFT; decode tok/s = (completion_tokens - 1) / (last - first token), counts from server usage. Speed sweep: fresh nonce first in every prompt (no prefix-cache hits), ignore_eos+min_tokens forces the full length, 1 discarded warmup, median of reps. Other clients' in-flight requests checked before every sweep call. Quality-run speed is incidental (varying prompts), not a sweep.
1792055223.176	
1792055223.176	[localeval] saved /opt/code/deepeval/results/20261015-050548-adhoc-clk-llamacpp-r3/result.json
1792055223.878	[localeval] logged to unanimis: 3b4357be-09d6-418f-8091-40eea2fc751f
