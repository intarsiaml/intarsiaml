1792054844.246	[localeval] calibrated: 22.1 tokens/sentence, 54 base tokens, ctx=8192
1792054851.236	[localeval]    128 tok rep 1/3: prompt=113 ttft=0.05s prefill=2143 tok/s decode=36.7 tok/s peak=36.9 tok/s out=256
1792054858.295	[localeval]    128 tok rep 2/3: prompt=113 ttft=0.05s prefill=2097 tok/s decode=36.4 tok/s peak=36.8 tok/s out=256
1792054865.277	[localeval]    128 tok rep 3/3: prompt=109 ttft=0.05s prefill=2030 tok/s decode=36.8 tok/s peak=37.0 tok/s out=256
1792054872.510	[localeval]   1024 tok rep 1/3: prompt=986 ttft=0.27s prefill=3678 tok/s decode=36.6 tok/s peak=36.8 tok/s out=256
1792054879.743	[localeval]   1024 tok rep 2/3: prompt=989 ttft=0.27s prefill=3679 tok/s decode=36.6 tok/s peak=36.7 tok/s out=256
1792054886.979	[localeval]   1024 tok rep 3/3: prompt=991 ttft=0.27s prefill=3688 tok/s decode=36.6 tok/s peak=36.8 tok/s out=256
1792054895.295	[localeval]   4096 tok rep 1/3: prompt=4087 ttft=1.15s prefill=3540 tok/s decode=35.6 tok/s peak=35.7 tok/s out=256
1792054903.604	[localeval]   4096 tok rep 2/3: prompt=4085 ttft=1.15s prefill=3542 tok/s decode=35.6 tok/s peak=35.7 tok/s out=256
1792054911.920	[localeval]   4096 tok rep 3/3: prompt=4088 ttft=1.15s prefill=3543 tok/s decode=35.6 tok/s peak=35.7 tok/s out=256
1792054911.921	# localeval: adhoc — 2026-10-15T05:00
1792054911.921	
1792054911.921	- model: `intarsia-metal` on ? (?), url `http://127.0.0.1:8093/v1`
1792054911.921	- thinking: off; run kind: speed; label: clk-safe-r2
1792054911.921	- harness: localeval 0.2.1, deepeval 4.2.6, runner endor.kuepker.haus, git 69c7a84
1792054911.921	- command: `/opt/code/deepeval/.venv/bin/localeval speed --url http://127.0.0.1:8093/v1 --model intarsia-metal --sizes 128,1024,4096 --gen 256 --reps 3 --thinking off --label clk-safe-r2`
1792054911.921	- result file: `results/20261015-050042-adhoc-clk-safe-r2/result.json`
1792054911.921	
1792054911.921	## Speed sweep
1792054911.921	
1792054911.921	served `intarsia-metal`, fingerprint `None`, ctx 8192, 3 reps, 256 forced output tokens
1792054911.921	
1792054911.921	| prompt tok | TTFT s | prefill tok/s | marginal prefill | decode tok/s | peak decode | server prefill | server decode | out tok |
1792054911.921	|---:|---:|---:|---:|---:|---:|---:|---:|---:|
1792054911.921	| 113 | 0.05 | 2097 (2030–2143) * | - | 36.7 (36.4–36.8) | 36.9 (36.8–37.0) | - | - | 256 |
1792054911.921	| 989 | 0.27 | 3679 (3678–3688) * | 4056 | 36.6 (36.6–36.6) | 36.8 (36.7–36.8) | - | - | 256 |
1792054911.921	| 4087 | 1.15 | 3542 (3540–3543) | 3501 | 35.6 (35.6–35.6) | 35.7 (35.7–35.7) | - | - | 256 |
1792054911.921	
1792054911.921	\* under 1024 uncached prompt tokens: TTFT is per-request overhead, not prefill throughput.
1792054911.921	
1792054911.921	Marginal prefill = Δprompt tokens / ΔTTFT from the previous row; fixed request overhead cancels, so it is the cleanest prefill throughput figure.
1792054911.921	
1792054911.921	Peak decode = fastest 32-token stretch of each call (burst speed, e.g. a run of accepted speculative drafts); median and range over reps. '-' = older run or output too short.
1792054911.921	
1792054911.921	
1792054911.921	## Method
1792054911.921	
1792054911.921	Streamed OpenAI-compatible calls, temperature 0. TTFT = request start to first content/reasoning token; prefill tok/s = uncached prompt_tokens / TTFT; decode tok/s = (completion_tokens - 1) / (last - first token), counts from server usage. Speed sweep: fresh nonce first in every prompt (no prefix-cache hits), ignore_eos+min_tokens forces the full length, 1 discarded warmup, median of reps. Other clients' in-flight requests checked before every sweep call. Quality-run speed is incidental (varying prompts), not a sweep.
1792054911.921	
1792054911.921	[localeval] saved /opt/code/deepeval/results/20261015-050042-adhoc-clk-safe-r2/result.json
1792054912.620	[localeval] logged to unanimis: 4510ebeb-4bc4-4aed-b882-1ef7391419d4
