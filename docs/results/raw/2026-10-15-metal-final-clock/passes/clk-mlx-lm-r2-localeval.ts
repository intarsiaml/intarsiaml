1792054636.926	[localeval] calibrated: 22.1 tokens/sentence, 52 base tokens, ctx=None
1792054644.566	[localeval]    128 tok rep 1/3: prompt=114 ttft=0.24s prefill=468 tok/s decode=34.4 tok/s peak=39.0 tok/s out=256
1792054652.190	[localeval]    128 tok rep 2/3: prompt=115 ttft=0.22s prefill=503 tok/s decode=34.4 tok/s peak=43.7 tok/s out=256
1792054659.817	[localeval]    128 tok rep 3/3: prompt=116 ttft=0.22s prefill=504 tok/s decode=34.4 tok/s peak=40.4 tok/s out=256
1792054667.766	[localeval]   1024 tok rep 1/3: prompt=990 ttft=0.43s prefill=2314 tok/s decode=33.9 tok/s peak=40.3 tok/s out=256
1792054675.745	[localeval]   1024 tok rep 2/3: prompt=992 ttft=0.43s prefill=2317 tok/s decode=33.8 tok/s peak=40.0 tok/s out=256
1792054683.710	[localeval]   1024 tok rep 3/3: prompt=990 ttft=0.43s prefill=2293 tok/s decode=33.8 tok/s peak=39.3 tok/s out=256
1792054692.832	[localeval]   4096 tok rep 1/3: prompt=4086 ttft=1.31s prefill=3112 tok/s decode=32.7 tok/s peak=44.4 tok/s out=256
1792054701.970	[localeval]   4096 tok rep 2/3: prompt=4089 ttft=1.32s prefill=3087 tok/s decode=32.6 tok/s peak=38.8 tok/s out=256
1792054711.120	[localeval]   4096 tok rep 3/3: prompt=4090 ttft=1.32s prefill=3092 tok/s decode=32.6 tok/s peak=39.3 tok/s out=256
1792054711.121	# localeval: adhoc — 2026-10-15T04:57
1792054711.121	
1792054711.121	- model: `/Users/josef/models/Qwen3-8B` on ? (?), url `http://127.0.0.1:8092/v1`
1792054711.121	- thinking: off; run kind: speed; label: clk-mlx-lm-r2
1792054711.121	- harness: localeval 0.2.1, deepeval 4.2.6, runner endor.kuepker.haus, git 69c7a84
1792054711.121	- command: `/opt/code/deepeval/.venv/bin/localeval speed --url http://127.0.0.1:8092/v1 --model /Users/josef/models/Qwen3-8B --sizes 128,1024,4096 --gen 256 --reps 3 --thinking off --label clk-mlx-lm-r2`
1792054711.121	- result file: `results/20261015-045714-adhoc-clk-mlx-lm-r2/result.json`
1792054711.121	
1792054711.121	## Speed sweep
1792054711.121	
1792054711.121	served `/Users/josef/models/Qwen3-8B`, fingerprint `0.31.3-0.32.1-macOS-27.0-arm64-arm-64bit-applegpu_g17s`, ctx ?, 3 reps, 256 forced output tokens
1792054711.121	
1792054711.121	| prompt tok | TTFT s | prefill tok/s | marginal prefill | decode tok/s | peak decode | server prefill | server decode | out tok |
1792054711.121	|---:|---:|---:|---:|---:|---:|---:|---:|---:|
1792054711.121	| 115 | 0.22 | 503 (468–504) * | - | 34.4 (34.4–34.4) | 40.4 (39.0–43.7) | - | - | 256 |
1792054711.121	| 990 | 0.43 | 2314 (2293–2317) * | 4268 | 33.8 (33.8–33.9) | 40.0 (39.3–40.3) | - | - | 256 |
1792054711.121	| 4089 | 1.32 | 3092 (3087–3112) | 3459 | 32.6 (32.6–32.7) | 39.3 (38.8–44.4) | - | - | 256 |
1792054711.121	
1792054711.121	\* under 1024 uncached prompt tokens: TTFT is per-request overhead, not prefill throughput.
1792054711.121	
1792054711.121	Marginal prefill = Δprompt tokens / ΔTTFT from the previous row; fixed request overhead cancels, so it is the cleanest prefill throughput figure.
1792054711.121	
1792054711.121	Peak decode = fastest 32-token stretch of each call (burst speed, e.g. a run of accepted speculative drafts); median and range over reps. '-' = older run or output too short.
1792054711.121	
1792054711.121	
1792054711.121	## Warnings
1792054711.121	
1792054711.121	- server exposes no in-flight counters (/metrics, /slots); could not verify the server was idle, so other clients may have contended this run
1792054711.121	
1792054711.121	## Method
1792054711.121	
1792054711.121	Streamed OpenAI-compatible calls, temperature 0. TTFT = request start to first content/reasoning token; prefill tok/s = uncached prompt_tokens / TTFT; decode tok/s = (completion_tokens - 1) / (last - first token), counts from server usage. Speed sweep: fresh nonce first in every prompt (no prefix-cache hits), ignore_eos+min_tokens forces the full length, 1 discarded warmup, median of reps. Other clients' in-flight requests checked before every sweep call. Quality-run speed is incidental (varying prompts), not a sweep.
1792054711.121	
1792054711.121	[localeval] saved /opt/code/deepeval/results/20261015-045714-adhoc-clk-mlx-lm-r2/result.json
1792054711.827	[localeval] logged to unanimis: 4625be0f-2540-4a09-afca-056e46497ca0
