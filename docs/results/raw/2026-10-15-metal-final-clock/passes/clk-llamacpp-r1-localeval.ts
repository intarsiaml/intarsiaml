1792054550.490	[localeval] calibrated: 22.1 tokens/sentence, 54 base tokens, ctx=8192
1792054557.823	[localeval]    128 tok rep 1/3: prompt=117 ttft=0.09s prefill=1317 tok/s decode=35.2 tok/s peak=35.5 tok/s out=256
1792054565.126	[localeval]    128 tok rep 2/3: prompt=112 ttft=0.09s prefill=1264 tok/s decode=35.4 tok/s peak=35.5 tok/s out=256
1792054572.413	[localeval]    128 tok rep 3/3: prompt=115 ttft=0.09s prefill=1232 tok/s decode=35.4 tok/s peak=35.5 tok/s out=256
1792054580.070	[localeval]   1024 tok rep 1/3: prompt=989 ttft=0.38s prefill=2559 tok/s decode=35.1 tok/s peak=35.2 tok/s out=256
1792054587.720	[localeval]   1024 tok rep 2/3: prompt=991 ttft=0.39s prefill=2557 tok/s decode=35.1 tok/s peak=35.2 tok/s out=256
1792054595.389	[localeval]   1024 tok rep 3/3: prompt=991 ttft=0.39s prefill=2551 tok/s decode=35.0 tok/s peak=35.2 tok/s out=256
1792054604.520	[localeval]   4096 tok rep 1/3: prompt=4087 ttft=1.64s prefill=2491 tok/s decode=34.0 tok/s peak=34.2 tok/s out=256
1792054613.725	[localeval]   4096 tok rep 2/3: prompt=4089 ttft=1.63s prefill=2501 tok/s decode=33.7 tok/s peak=34.1 tok/s out=256
1792054622.901	[localeval]   4096 tok rep 3/3: prompt=4088 ttft=1.64s prefill=2495 tok/s decode=33.8 tok/s peak=34.0 tok/s out=256
1792054622.901	# localeval: adhoc — 2026-10-15T04:55
1792054622.901	
1792054622.901	- model: `/Users/josef/models/Qwen3-8B-GGUF/Qwen3-8B-BF16.gguf` on ? (?), url `http://127.0.0.1:8091/v1`
1792054622.901	- thinking: off; run kind: speed; label: clk-llamacpp-r1
1792054622.901	- harness: localeval 0.2.1, deepeval 4.2.6, runner endor.kuepker.haus, git 69c7a84
1792054622.901	- command: `/opt/code/deepeval/.venv/bin/localeval speed --url http://127.0.0.1:8091/v1 --model /Users/josef/models/Qwen3-8B-GGUF/Qwen3-8B-BF16.gguf --sizes 128,1024,4096 --gen 256 --reps 3 --thinking off --label clk-llamacpp-r1`
1792054622.901	- result file: `results/20261015-045548-adhoc-clk-llamacpp-r1/result.json`
1792054622.901	
1792054622.901	## Speed sweep
1792054622.901	
1792054622.901	served `/Users/josef/models/Qwen3-8B-GGUF/Qwen3-8B-BF16.gguf`, fingerprint `b11429-d81235049`, ctx 8192, 3 reps, 256 forced output tokens
1792054622.901	
1792054622.901	| prompt tok | TTFT s | prefill tok/s | marginal prefill | decode tok/s | peak decode | server prefill | server decode | out tok |
1792054622.901	|---:|---:|---:|---:|---:|---:|---:|---:|---:|
1792054622.901	| 115 | 0.09 | 1264 (1232–1317) * | - | 35.4 (35.2–35.4) | 35.5 (35.5–35.5) | 1306 | 35.4 | 256 |
1792054622.901	| 991 | 0.39 | 2557 (2551–2559) * | 2950 | 35.1 (35.0–35.1) | 35.2 (35.2–35.2) | 2581 | 35.1 | 256 |
1792054622.901	| 4088 | 1.64 | 2495 (2491–2501) | 2472 | 33.8 (33.7–34.0) | 34.1 (34.0–34.2) | 2504 | 33.8 | 256 |
1792054622.901	
1792054622.901	\* under 1024 uncached prompt tokens: TTFT is per-request overhead, not prefill throughput.
1792054622.901	
1792054622.901	Marginal prefill = Δprompt tokens / ΔTTFT from the previous row; fixed request overhead cancels, so it is the cleanest prefill throughput figure.
1792054622.901	
1792054622.901	Peak decode = fastest 32-token stretch of each call (burst speed, e.g. a run of accepted speculative drafts); median and range over reps. '-' = older run or output too short.
1792054622.901	
1792054622.901	
1792054622.901	## Method
1792054622.901	
1792054622.901	Streamed OpenAI-compatible calls, temperature 0. TTFT = request start to first content/reasoning token; prefill tok/s = uncached prompt_tokens / TTFT; decode tok/s = (completion_tokens - 1) / (last - first token), counts from server usage. Speed sweep: fresh nonce first in every prompt (no prefix-cache hits), ignore_eos+min_tokens forces the full length, 1 discarded warmup, median of reps. Other clients' in-flight requests checked before every sweep call. Quality-run speed is incidental (varying prompts), not a sweep.
1792054622.901	
1792054622.901	[localeval] saved /opt/code/deepeval/results/20261015-045548-adhoc-clk-llamacpp-r1/result.json
1792054623.608	[localeval] logged to unanimis: 41c0c3ec-f65c-4d92-91bc-006bb6f25867
