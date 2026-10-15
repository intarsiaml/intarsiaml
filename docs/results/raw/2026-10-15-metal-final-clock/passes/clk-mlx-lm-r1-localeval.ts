1792054460.715	[localeval] calibrated: 22.1 tokens/sentence, 51 base tokens, ctx=None
1792054468.360	[localeval]    128 tok rep 1/3: prompt=111 ttft=0.24s prefill=440 tok/s decode=34.5 tok/s peak=41.3 tok/s out=256
1792054475.983	[localeval]    128 tok rep 2/3: prompt=111 ttft=0.22s prefill=487 tok/s decode=34.5 tok/s peak=41.4 tok/s out=256
1792054483.605	[localeval]    128 tok rep 3/3: prompt=117 ttft=0.22s prefill=514 tok/s decode=34.4 tok/s peak=48.4 tok/s out=256
1792054491.569	[localeval]   1024 tok rep 1/3: prompt=991 ttft=0.43s prefill=2315 tok/s decode=33.8 tok/s peak=40.6 tok/s out=256
1792054499.548	[localeval]   1024 tok rep 2/3: prompt=992 ttft=0.42s prefill=2366 tok/s decode=33.7 tok/s peak=39.1 tok/s out=256
1792054507.496	[localeval]   1024 tok rep 3/3: prompt=990 ttft=0.43s prefill=2310 tok/s decode=33.9 tok/s peak=40.4 tok/s out=256
1792054516.624	[localeval]   4096 tok rep 1/3: prompt=4091 ttft=1.32s prefill=3093 tok/s decode=32.7 tok/s peak=39.7 tok/s out=256
1792054525.775	[localeval]   4096 tok rep 2/3: prompt=4087 ttft=1.34s prefill=3053 tok/s decode=32.6 tok/s peak=40.1 tok/s out=256
1792054534.934	[localeval]   4096 tok rep 3/3: prompt=4090 ttft=1.34s prefill=3058 tok/s decode=32.6 tok/s peak=38.3 tok/s out=256
1792054534.934	# localeval: adhoc — 2026-10-15T04:54
1792054534.934	
1792054534.934	- model: `/Users/josef/models/Qwen3-8B` on ? (?), url `http://127.0.0.1:8092/v1`
1792054534.934	- thinking: off; run kind: speed; label: clk-mlx-lm-r1
1792054534.934	- harness: localeval 0.2.1, deepeval 4.2.6, runner endor.kuepker.haus, git 69c7a84
1792054534.934	- command: `/opt/code/deepeval/.venv/bin/localeval speed --url http://127.0.0.1:8092/v1 --model /Users/josef/models/Qwen3-8B --sizes 128,1024,4096 --gen 256 --reps 3 --thinking off --label clk-mlx-lm-r1`
1792054534.934	- result file: `results/20261015-045418-adhoc-clk-mlx-lm-r1/result.json`
1792054534.934	
1792054534.934	## Speed sweep
1792054534.934	
1792054534.934	served `/Users/josef/models/Qwen3-8B`, fingerprint `0.31.3-0.32.1-macOS-27.0-arm64-arm-64bit-applegpu_g17s`, ctx ?, 3 reps, 256 forced output tokens
1792054534.934	
1792054534.934	| prompt tok | TTFT s | prefill tok/s | marginal prefill | decode tok/s | peak decode | server prefill | server decode | out tok |
1792054534.934	|---:|---:|---:|---:|---:|---:|---:|---:|---:|
1792054534.934	| 111 | 0.22 | 487 (440–514) * | - | 34.5 (34.4–34.5) | 41.4 (41.3–48.4) | - | - | 256 |
1792054534.934	| 991 | 0.43 | 2315 (2310–2366) * | 4272 | 33.8 (33.7–33.9) | 40.4 (39.1–40.6) | - | - | 256 |
1792054534.935	| 4090 | 1.34 | 3058 (3053–3093) | 3406 | 32.6 (32.6–32.7) | 39.7 (38.3–40.1) | - | - | 256 |
1792054534.935	
1792054534.935	\* under 1024 uncached prompt tokens: TTFT is per-request overhead, not prefill throughput.
1792054534.935	
1792054534.935	Marginal prefill = Δprompt tokens / ΔTTFT from the previous row; fixed request overhead cancels, so it is the cleanest prefill throughput figure.
1792054534.935	
1792054534.935	Peak decode = fastest 32-token stretch of each call (burst speed, e.g. a run of accepted speculative drafts); median and range over reps. '-' = older run or output too short.
1792054534.935	
1792054534.935	
1792054534.935	## Warnings
1792054534.935	
1792054534.935	- server exposes no in-flight counters (/metrics, /slots); could not verify the server was idle, so other clients may have contended this run
1792054534.935	
1792054534.935	## Method
1792054534.935	
1792054534.935	Streamed OpenAI-compatible calls, temperature 0. TTFT = request start to first content/reasoning token; prefill tok/s = uncached prompt_tokens / TTFT; decode tok/s = (completion_tokens - 1) / (last - first token), counts from server usage. Speed sweep: fresh nonce first in every prompt (no prefix-cache hits), ignore_eos+min_tokens forces the full length, 1 discarded warmup, median of reps. Other clients' in-flight requests checked before every sweep call. Quality-run speed is incidental (varying prompts), not a sweep.
1792054534.935	
1792054534.935	[localeval] saved /opt/code/deepeval/results/20261015-045418-adhoc-clk-mlx-lm-r1/result.json
1792054535.634	[localeval] logged to unanimis: 57a3766e-9ca0-448e-a7fe-11dfe6ca4162
