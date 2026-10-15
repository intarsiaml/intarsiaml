1792055267.778	[localeval] calibrated: 22.1 tokens/sentence, 51 base tokens, ctx=8192
1792055274.772	[localeval]    128 tok rep 1/3: prompt=116 ttft=0.05s prefill=2187 tok/s decode=36.7 tok/s peak=36.9 tok/s out=256
1792055281.811	[localeval]    128 tok rep 2/3: prompt=109 ttft=0.05s prefill=1984 tok/s decode=36.5 tok/s peak=36.8 tok/s out=256
1792055288.766	[localeval]    128 tok rep 3/3: prompt=112 ttft=0.05s prefill=2080 tok/s decode=36.9 tok/s peak=37.0 tok/s out=256
1792055296.004	[localeval]   1024 tok rep 1/3: prompt=989 ttft=0.27s prefill=3685 tok/s decode=36.6 tok/s peak=36.7 tok/s out=256
1792055303.238	[localeval]   1024 tok rep 2/3: prompt=993 ttft=0.27s prefill=3699 tok/s decode=36.6 tok/s peak=36.8 tok/s out=256
1792055310.477	[localeval]   1024 tok rep 3/3: prompt=989 ttft=0.27s prefill=3691 tok/s decode=36.6 tok/s peak=36.8 tok/s out=256
1792055318.783	[localeval]   4096 tok rep 1/3: prompt=4083 ttft=1.15s prefill=3541 tok/s decode=35.7 tok/s peak=35.8 tok/s out=256
1792055327.109	[localeval]   4096 tok rep 2/3: prompt=4085 ttft=1.15s prefill=3539 tok/s decode=35.6 tok/s peak=35.7 tok/s out=256
1792055335.433	[localeval]   4096 tok rep 3/3: prompt=4089 ttft=1.15s prefill=3538 tok/s decode=35.6 tok/s peak=35.7 tok/s out=256
1792055335.434	# localeval: adhoc — 2026-10-15T05:07
1792055335.434	
1792055335.434	- model: `intarsia-metal` on ? (?), url `http://127.0.0.1:8093/v1`
1792055335.434	- thinking: off; run kind: speed; label: clk-safe-r3
1792055335.434	- harness: localeval 0.2.1, deepeval 4.2.6, runner endor.kuepker.haus, git 69c7a84
1792055335.434	- command: `/opt/code/deepeval/.venv/bin/localeval speed --url http://127.0.0.1:8093/v1 --model intarsia-metal --sizes 128,1024,4096 --gen 256 --reps 3 --thinking off --label clk-safe-r3`
1792055335.434	- result file: `results/20261015-050746-adhoc-clk-safe-r3/result.json`
1792055335.434	
1792055335.434	## Speed sweep
1792055335.434	
1792055335.434	served `intarsia-metal`, fingerprint `None`, ctx 8192, 3 reps, 256 forced output tokens
1792055335.434	
1792055335.434	| prompt tok | TTFT s | prefill tok/s | marginal prefill | decode tok/s | peak decode | server prefill | server decode | out tok |
1792055335.434	|---:|---:|---:|---:|---:|---:|---:|---:|---:|
1792055335.434	| 112 | 0.05 | 2080 (1984–2187) * | - | 36.7 (36.5–36.9) | 36.9 (36.8–37.0) | - | - | 256 |
1792055335.434	| 989 | 0.27 | 3691 (3685–3699) * | 4060 | 36.6 (36.6–36.6) | 36.8 (36.7–36.8) | - | - | 256 |
1792055335.434	| 4085 | 1.15 | 3539 (3538–3541) | 3494 | 35.6 (35.6–35.7) | 35.7 (35.7–35.8) | - | - | 256 |
1792055335.434	
1792055335.434	\* under 1024 uncached prompt tokens: TTFT is per-request overhead, not prefill throughput.
1792055335.434	
1792055335.434	Marginal prefill = Δprompt tokens / ΔTTFT from the previous row; fixed request overhead cancels, so it is the cleanest prefill throughput figure.
1792055335.434	
1792055335.434	Peak decode = fastest 32-token stretch of each call (burst speed, e.g. a run of accepted speculative drafts); median and range over reps. '-' = older run or output too short.
1792055335.434	
1792055335.434	
1792055335.434	## Method
1792055335.434	
1792055335.434	Streamed OpenAI-compatible calls, temperature 0. TTFT = request start to first content/reasoning token; prefill tok/s = uncached prompt_tokens / TTFT; decode tok/s = (completion_tokens - 1) / (last - first token), counts from server usage. Speed sweep: fresh nonce first in every prompt (no prefix-cache hits), ignore_eos+min_tokens forces the full length, 1 discarded warmup, median of reps. Other clients' in-flight requests checked before every sweep call. Quality-run speed is incidental (varying prompts), not a sweep.
1792055335.434	
1792055335.434	[localeval] saved /opt/code/deepeval/results/20261015-050746-adhoc-clk-safe-r3/result.json
1792055336.129	[localeval] logged to unanimis: 5d8e038f-cda8-474f-89cf-ec2d67b3cca8
