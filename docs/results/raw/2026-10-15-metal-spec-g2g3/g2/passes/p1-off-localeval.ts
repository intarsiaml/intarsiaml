1792081507.643	[localeval] calibrated: 22.1 tokens/sentence, 51 base tokens, ctx=8192
1792081514.791	[localeval]    128 tok rep 1/3: prompt=112 ttft=0.06s prefill=1928 tok/s decode=36.0 tok/s peak=36.5 tok/s out=256
1792081521.938	[localeval]    128 tok rep 2/3: prompt=113 ttft=0.05s prefill=2054 tok/s decode=36.0 tok/s peak=36.6 tok/s out=256
1792081529.177	[localeval]    128 tok rep 3/3: prompt=112 ttft=0.05s prefill=2072 tok/s decode=35.5 tok/s peak=36.4 tok/s out=256
1792081536.493	[localeval]   1024 tok rep 1/3: prompt=989 ttft=0.25s prefill=3888 tok/s decode=36.1 tok/s peak=36.4 tok/s out=256
1792081543.843	[localeval]   1024 tok rep 2/3: prompt=991 ttft=0.25s prefill=3893 tok/s decode=35.9 tok/s peak=36.5 tok/s out=256
1792081551.130	[localeval]   1024 tok rep 3/3: prompt=988 ttft=0.26s prefill=3785 tok/s decode=36.3 tok/s peak=36.5 tok/s out=256
1792081559.466	[localeval]   4096 tok rep 1/3: prompt=4092 ttft=1.04s prefill=3925 tok/s decode=35.0 tok/s peak=35.6 tok/s out=256
1792081567.751	[localeval]   4096 tok rep 2/3: prompt=4087 ttft=1.05s prefill=3875 tok/s decode=35.3 tok/s peak=35.6 tok/s out=256
1792081576.051	[localeval]   4096 tok rep 3/3: prompt=4089 ttft=1.08s prefill=3788 tok/s decode=35.3 tok/s peak=35.5 tok/s out=256
1792081576.052	# localeval: adhoc — 2026-10-15T12:25
1792081576.052	
1792081576.052	- model: `intarsia-metal` on ? (?), url `http://127.0.0.1:8093/v1`
1792081576.052	- thinking: off; run kind: speed; label: p1-off
1792081576.052	- harness: localeval 0.2.1, deepeval 4.2.6, runner endor.kuepker.haus, git 69c7a84
1792081576.052	- command: `/opt/code/deepeval/.venv/bin/localeval speed --url http://127.0.0.1:8093/v1 --model intarsia-metal --sizes 128,1024,4096 --gen 256 --reps 3 --thinking off --label p1-off`
1792081576.052	- result file: `results/20261015-122505-adhoc-p1-off/result.json`
1792081576.052	
1792081576.052	## Speed sweep
1792081576.052	
1792081576.052	served `intarsia-metal`, fingerprint `None`, ctx 8192, 3 reps, 256 forced output tokens
1792081576.052	
1792081576.052	| prompt tok | TTFT s | prefill tok/s | marginal prefill | decode tok/s | peak decode | server prefill | server decode | out tok |
1792081576.052	|---:|---:|---:|---:|---:|---:|---:|---:|---:|
1792081576.052	| 112 | 0.05 | 2054 (1928–2072) * | - | 36.0 (35.5–36.0) | 36.5 (36.4–36.6) | - | - | 256 |
1792081576.052	| 989 | 0.25 | 3888 (3785–3893) * | 4385 | 36.1 (35.9–36.3) | 36.5 (36.4–36.5) | - | - | 256 |
1792081576.052	| 4089 | 1.05 | 3875 (3788–3925) | 3875 | 35.3 (35.0–35.3) | 35.6 (35.5–35.6) | - | - | 256 |
1792081576.052	
1792081576.052	\* under 1024 uncached prompt tokens: TTFT is per-request overhead, not prefill throughput.
1792081576.052	
1792081576.052	Marginal prefill = Δprompt tokens / ΔTTFT from the previous row; fixed request overhead cancels, so it is the cleanest prefill throughput figure.
1792081576.052	
1792081576.052	Peak decode = fastest 32-token stretch of each call (burst speed, e.g. a run of accepted speculative drafts); median and range over reps. '-' = older run or output too short.
1792081576.052	
1792081576.052	
1792081576.052	## Method
1792081576.052	
1792081576.052	Streamed OpenAI-compatible calls, temperature 0. TTFT = request start to first content/reasoning token; prefill tok/s = uncached prompt_tokens / TTFT; decode tok/s = (completion_tokens - 1) / (last - first token), counts from server usage. Speed sweep: fresh nonce first in every prompt (no prefix-cache hits), ignore_eos+min_tokens forces the full length, 1 discarded warmup, median of reps. Other clients' in-flight requests checked before every sweep call. Quality-run speed is incidental (varying prompts), not a sweep.
1792081576.052	
1792081576.052	[localeval] saved /opt/code/deepeval/results/20261015-122505-adhoc-p1-off/result.json
1792081576.784	[localeval] logged to unanimis: 26d01aa2-a5ad-4bb7-bb51-f5909709297d
