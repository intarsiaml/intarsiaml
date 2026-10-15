1792055349.086	[localeval] calibrated: 22.1 tokens/sentence, 51 base tokens, ctx=None
1792055356.710	[localeval]    128 tok rep 1/3: prompt=114 ttft=0.23s prefill=470 tok/s decode=34.5 tok/s peak=41.5 tok/s out=256
1792055364.314	[localeval]    128 tok rep 2/3: prompt=115 ttft=0.22s prefill=508 tok/s decode=34.5 tok/s peak=39.7 tok/s out=256
1792055371.929	[localeval]    128 tok rep 3/3: prompt=111 ttft=0.22s prefill=479 tok/s decode=34.5 tok/s peak=41.5 tok/s out=256
1792055379.870	[localeval]   1024 tok rep 1/3: prompt=990 ttft=0.43s prefill=2293 tok/s decode=34.0 tok/s peak=40.4 tok/s out=256
1792055387.818	[localeval]   1024 tok rep 2/3: prompt=990 ttft=0.43s prefill=2303 tok/s decode=33.9 tok/s peak=39.1 tok/s out=256
1792055395.781	[localeval]   1024 tok rep 3/3: prompt=990 ttft=0.42s prefill=2359 tok/s decode=33.8 tok/s peak=40.9 tok/s out=256
1792055404.914	[localeval]   4096 tok rep 1/3: prompt=4089 ttft=1.32s prefill=3094 tok/s decode=32.6 tok/s peak=41.8 tok/s out=256
1792055414.033	[localeval]   4096 tok rep 2/3: prompt=4087 ttft=1.32s prefill=3103 tok/s decode=32.7 tok/s peak=40.1 tok/s out=256
1792055423.148	[localeval]   4096 tok rep 3/3: prompt=4089 ttft=1.31s prefill=3108 tok/s decode=32.7 tok/s peak=43.7 tok/s out=256
1792055423.148	# localeval: adhoc — 2026-10-15T05:09
1792055423.148	
1792055423.148	- model: `/Users/josef/models/Qwen3-8B` on ? (?), url `http://127.0.0.1:8092/v1`
1792055423.148	- thinking: off; run kind: speed; label: clk-mlx-lm-r3
1792055423.148	- harness: localeval 0.2.1, deepeval 4.2.6, runner endor.kuepker.haus, git 69c7a84
1792055423.148	- command: `/opt/code/deepeval/.venv/bin/localeval speed --url http://127.0.0.1:8092/v1 --model /Users/josef/models/Qwen3-8B --sizes 128,1024,4096 --gen 256 --reps 3 --thinking off --label clk-mlx-lm-r3`
1792055423.148	- result file: `results/20261015-050907-adhoc-clk-mlx-lm-r3/result.json`
1792055423.148	
1792055423.148	## Speed sweep
1792055423.148	
1792055423.148	served `/Users/josef/models/Qwen3-8B`, fingerprint `0.31.3-0.32.1-macOS-27.0-arm64-arm-64bit-applegpu_g17s`, ctx ?, 3 reps, 256 forced output tokens
1792055423.148	
1792055423.148	| prompt tok | TTFT s | prefill tok/s | marginal prefill | decode tok/s | peak decode | server prefill | server decode | out tok |
1792055423.148	|---:|---:|---:|---:|---:|---:|---:|---:|---:|
1792055423.148	| 114 | 0.22 | 479 (470–508) * | - | 34.5 (34.5–34.5) | 41.5 (39.7–41.5) | - | - | 256 |
1792055423.148	| 990 | 0.43 | 2303 (2293–2359) * | 4191 | 33.9 (33.8–34.0) | 40.4 (39.1–40.9) | - | - | 256 |
1792055423.148	| 4089 | 1.32 | 3103 (3094–3108) | 3490 | 32.7 (32.6–32.7) | 41.8 (40.1–43.7) | - | - | 256 |
1792055423.148	
1792055423.148	\* under 1024 uncached prompt tokens: TTFT is per-request overhead, not prefill throughput.
1792055423.148	
1792055423.148	Marginal prefill = Δprompt tokens / ΔTTFT from the previous row; fixed request overhead cancels, so it is the cleanest prefill throughput figure.
1792055423.148	
1792055423.148	Peak decode = fastest 32-token stretch of each call (burst speed, e.g. a run of accepted speculative drafts); median and range over reps. '-' = older run or output too short.
1792055423.148	
1792055423.148	
1792055423.148	## Warnings
1792055423.148	
1792055423.148	- server exposes no in-flight counters (/metrics, /slots); could not verify the server was idle, so other clients may have contended this run
1792055423.148	
1792055423.148	## Method
1792055423.148	
1792055423.148	Streamed OpenAI-compatible calls, temperature 0. TTFT = request start to first content/reasoning token; prefill tok/s = uncached prompt_tokens / TTFT; decode tok/s = (completion_tokens - 1) / (last - first token), counts from server usage. Speed sweep: fresh nonce first in every prompt (no prefix-cache hits), ignore_eos+min_tokens forces the full length, 1 discarded warmup, median of reps. Other clients' in-flight requests checked before every sweep call. Quality-run speed is incidental (varying prompts), not a sweep.
1792055423.149	
1792055423.149	[localeval] saved /opt/code/deepeval/results/20261015-050907-adhoc-clk-mlx-lm-r3/result.json
1792055423.829	[localeval] logged to unanimis: 30121c82-2c54-4a59-864e-411d84678833
