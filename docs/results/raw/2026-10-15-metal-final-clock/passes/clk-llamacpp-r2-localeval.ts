1792054725.657	[localeval] calibrated: 22.1 tokens/sentence, 55 base tokens, ctx=8192
1792054732.981	[localeval]    128 tok rep 1/3: prompt=114 ttft=0.09s prefill=1289 tok/s decode=35.3 tok/s peak=35.5 tok/s out=256
1792054740.353	[localeval]    128 tok rep 2/3: prompt=114 ttft=0.09s prefill=1271 tok/s decode=35.0 tok/s peak=35.4 tok/s out=256
1792054747.643	[localeval]    128 tok rep 3/3: prompt=116 ttft=0.09s prefill=1231 tok/s decode=35.4 tok/s peak=35.5 tok/s out=256
1792054755.306	[localeval]   1024 tok rep 1/3: prompt=990 ttft=0.39s prefill=2560 tok/s decode=35.0 tok/s peak=35.1 tok/s out=256
1792054762.969	[localeval]   1024 tok rep 2/3: prompt=992 ttft=0.39s prefill=2560 tok/s decode=35.0 tok/s peak=35.2 tok/s out=256
1792054770.634	[localeval]   1024 tok rep 3/3: prompt=991 ttft=0.39s prefill=2558 tok/s decode=35.0 tok/s peak=35.1 tok/s out=256
1792054779.768	[localeval]   4096 tok rep 1/3: prompt=4086 ttft=1.64s prefill=2486 tok/s decode=34.0 tok/s peak=34.1 tok/s out=256
1792054788.897	[localeval]   4096 tok rep 2/3: prompt=4089 ttft=1.64s prefill=2497 tok/s decode=34.0 tok/s peak=34.2 tok/s out=256
1792054798.119	[localeval]   4096 tok rep 3/3: prompt=4088 ttft=1.64s prefill=2495 tok/s decode=33.6 tok/s peak=34.0 tok/s out=256
1792054798.120	# localeval: adhoc — 2026-10-15T04:58
1792054798.120	
1792054798.120	- model: `/Users/josef/models/Qwen3-8B-GGUF/Qwen3-8B-BF16.gguf` on ? (?), url `http://127.0.0.1:8091/v1`
1792054798.120	- thinking: off; run kind: speed; label: clk-llamacpp-r2
1792054798.120	- harness: localeval 0.2.1, deepeval 4.2.6, runner endor.kuepker.haus, git 69c7a84
1792054798.120	- command: `/opt/code/deepeval/.venv/bin/localeval speed --url http://127.0.0.1:8091/v1 --model /Users/josef/models/Qwen3-8B-GGUF/Qwen3-8B-BF16.gguf --sizes 128,1024,4096 --gen 256 --reps 3 --thinking off --label clk-llamacpp-r2`
1792054798.120	- result file: `results/20261015-045843-adhoc-clk-llamacpp-r2/result.json`
1792054798.120	
1792054798.120	## Speed sweep
1792054798.120	
1792054798.120	served `/Users/josef/models/Qwen3-8B-GGUF/Qwen3-8B-BF16.gguf`, fingerprint `b11429-d81235049`, ctx 8192, 3 reps, 256 forced output tokens
1792054798.120	
1792054798.120	| prompt tok | TTFT s | prefill tok/s | marginal prefill | decode tok/s | peak decode | server prefill | server decode | out tok |
1792054798.120	|---:|---:|---:|---:|---:|---:|---:|---:|---:|
1792054798.120	| 114 | 0.09 | 1271 (1231–1289) * | - | 35.2 (35.0–35.4) | 35.5 (35.4–35.5) | 1313 | 35.2 | 256 |
1792054798.120	| 991 | 0.39 | 2560 (2558–2560) * | 2963 | 35.0 (35.0–35.0) | 35.1 (35.1–35.2) | 2584 | 35.0 | 256 |
1792054798.120	| 4088 | 1.64 | 2495 (2486–2497) | 2470 | 34.0 (33.6–34.0) | 34.1 (34.0–34.2) | 2503 | 34.0 | 256 |
1792054798.120	
1792054798.120	\* under 1024 uncached prompt tokens: TTFT is per-request overhead, not prefill throughput.
1792054798.120	
1792054798.120	Marginal prefill = Δprompt tokens / ΔTTFT from the previous row; fixed request overhead cancels, so it is the cleanest prefill throughput figure.
1792054798.120	
1792054798.120	Peak decode = fastest 32-token stretch of each call (burst speed, e.g. a run of accepted speculative drafts); median and range over reps. '-' = older run or output too short.
1792054798.120	
1792054798.120	
1792054798.120	## Method
1792054798.120	
1792054798.120	Streamed OpenAI-compatible calls, temperature 0. TTFT = request start to first content/reasoning token; prefill tok/s = uncached prompt_tokens / TTFT; decode tok/s = (completion_tokens - 1) / (last - first token), counts from server usage. Speed sweep: fresh nonce first in every prompt (no prefix-cache hits), ignore_eos+min_tokens forces the full length, 1 discarded warmup, median of reps. Other clients' in-flight requests checked before every sweep call. Quality-run speed is incidental (varying prompts), not a sweep.
1792054798.120	
1792054798.120	[localeval] saved /opt/code/deepeval/results/20261015-045843-adhoc-clk-llamacpp-r2/result.json
1792054798.816	[localeval] logged to unanimis: e705555c-d9dd-4f4e-9966-e3ed5046693e
