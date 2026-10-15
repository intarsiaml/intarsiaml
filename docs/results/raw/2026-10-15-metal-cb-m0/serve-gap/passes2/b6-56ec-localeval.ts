1792096821.061	[localeval] calibrated: 22.1 tokens/sentence, 53 base tokens, ctx=8192
1792096828.140	[localeval]    128 tok rep 1/3: prompt=114 ttft=0.06s prefill=1918 tok/s decode=36.3 tok/s peak=36.6 tok/s out=256
1792096835.230	[localeval]    128 tok rep 2/3: prompt=110 ttft=0.05s prefill=2013 tok/s decode=36.2 tok/s peak=36.5 tok/s out=256
1792096842.305	[localeval]    128 tok rep 3/3: prompt=114 ttft=0.05s prefill=2112 tok/s decode=36.3 tok/s peak=36.9 tok/s out=256
1792096849.613	[localeval]   1024 tok rep 1/3: prompt=992 ttft=0.25s prefill=3900 tok/s decode=36.2 tok/s peak=36.5 tok/s out=256
1792096856.879	[localeval]   1024 tok rep 2/3: prompt=988 ttft=0.25s prefill=3895 tok/s decode=36.4 tok/s peak=36.5 tok/s out=256
1792096864.163	[localeval]   1024 tok rep 3/3: prompt=990 ttft=0.25s prefill=3904 tok/s decode=36.3 tok/s peak=36.4 tok/s out=256
1792096872.581	[localeval]   4096 tok rep 1/3: prompt=4089 ttft=1.23s prefill=3328 tok/s decode=35.5 tok/s peak=35.7 tok/s out=256
1792096880.959	[localeval]   4096 tok rep 2/3: prompt=4090 ttft=1.04s prefill=3916 tok/s decode=34.8 tok/s peak=35.3 tok/s out=256
1792096889.701	[localeval]   4096 tok rep 3/3: prompt=4086 ttft=1.30s prefill=3132 tok/s decode=34.3 tok/s peak=35.0 tok/s out=256
1792096889.702	# localeval: adhoc — 2026-10-15T16:40
1792096889.702	
1792096889.702	- model: `intarsia-metal` on ? (?), url `http://127.0.0.1:8093/v1`
1792096889.702	- thinking: off; run kind: speed; label: cbm0-gap-b6-56ec
1792096889.702	- harness: localeval 0.2.1, deepeval 4.2.6, runner endor.kuepker.haus, git 69c7a84
1792096889.702	- command: `/opt/code/deepeval/.venv/bin/localeval speed --url http://127.0.0.1:8093/v1 --model intarsia-metal --sizes 128,1024,4096 --gen 256 --reps 3 --thinking off --label cbm0-gap-b6-56ec`
1792096889.702	- result file: `results/20261015-164019-adhoc-cbm0-gap-b6-56ec/result.json`
1792096889.702	
1792096889.702	## Speed sweep
1792096889.702	
1792096889.702	served `intarsia-metal`, fingerprint `None`, ctx 8192, 3 reps, 256 forced output tokens
1792096889.702	
1792096889.702	| prompt tok | TTFT s | prefill tok/s | marginal prefill | decode tok/s | peak decode | server prefill | server decode | out tok |
1792096889.702	|---:|---:|---:|---:|---:|---:|---:|---:|---:|
1792096889.702	| 114 | 0.05 | 2013 (1918–2112) * | - | 36.3 (36.2–36.3) | 36.6 (36.5–36.9) | - | - | 256 |
1792096889.702	| 990 | 0.25 | 3900 (3895–3904) * | 4358 | 36.3 (36.1–36.4) | 36.5 (36.4–36.5) | - | - | 256 |
1792096889.702	| 4089 | 1.23 | 3328 (3132–3916) | 3182 | 34.8 (34.3–35.5) | 35.3 (35.0–35.7) | - | - | 256 |
1792096889.702	
1792096889.702	\* under 1024 uncached prompt tokens: TTFT is per-request overhead, not prefill throughput.
1792096889.702	
1792096889.702	Marginal prefill = Δprompt tokens / ΔTTFT from the previous row; fixed request overhead cancels, so it is the cleanest prefill throughput figure.
1792096889.702	
1792096889.702	Peak decode = fastest 32-token stretch of each call (burst speed, e.g. a run of accepted speculative drafts); median and range over reps. '-' = older run or output too short.
1792096889.702	
1792096889.702	
1792096889.702	## Method
1792096889.702	
1792096889.702	Streamed OpenAI-compatible calls, temperature 0. TTFT = request start to first content/reasoning token; prefill tok/s = uncached prompt_tokens / TTFT; decode tok/s = (completion_tokens - 1) / (last - first token), counts from server usage. Speed sweep: fresh nonce first in every prompt (no prefix-cache hits), ignore_eos+min_tokens forces the full length, 1 discarded warmup, median of reps. Other clients' in-flight requests checked before every sweep call. Quality-run speed is incidental (varying prompts), not a sweep.
1792096889.702	
1792096889.702	[localeval] saved /opt/code/deepeval/results/20261015-164019-adhoc-cbm0-gap-b6-56ec/result.json
1792096890.575	[localeval] logged to unanimis: 21d3d895-9688-48fc-b9e5-66d6a42a4c04
