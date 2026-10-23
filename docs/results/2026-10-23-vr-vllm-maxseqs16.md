# vLLM 35B NVFP4 with --max-num-seqs 16 on the GB10 (lane vr, 2026-10-23)

Question: the old vLLM concurrency number (70 / 105 / 105 at 1 / 4 / 16 streams) came from the production server with
`--max-num-seqs 2`, so it was a cap. This is vLLM given 16 sequences. Measurement only; no engine code changed.

## Setup
- Box spark2 (spark-fa00), fleetq job #4561 (`--exclusive --takes-down retro`, owner claude-f31ccf91-vr), 12:54 to 13:04 EDT.
- Config = the retro launcher (`spark2:~/missioncntrl-guidance-run.sh`, image `missioncntrl/vllm-openai:65b7662d`, vLLM 0.26.1rc1.dev602)
  with ONE change, `--max-num-seqs 2` to `16`. Run as a separate container `vr-vllm` on the same host:port; the production container
  `missioncntrl-guidance` was only stopped and then started again (not recreated).
- Flags: `serve nvidia/Qwen3.6-35B-A3B-NVFP4 --host 100.96.95.89 --port 8000 --served-model-name guidance capcom --quantization modelopt_fp4
  --max-model-len 131072 --max-num-seqs 16 --max-num-batched-tokens 8192 --kv-cache-dtype fp8 --kv-cache-memory-bytes 8589934592
  --gpu-memory-utilization 0.32 --enable-prefix-caching --enable-prompt-tokens-details --enable-chunked-prefill --enable-auto-tool-choice
  --tool-call-parser qwen3_xml --reasoning-parser qwen3`; env `HF_HUB_OFFLINE=1 VLLM_USE_FLASHINFER_MOE_FP4=0
  VLLM_DISABLED_KERNELS=FlashInferFP8ScaledMMLinearKernel`. gpu-memory-utilization was NOT raised (started first try; 8 GiB fp8 KV = 759,837 tokens).
- Workload as fx3: `localeval speed --sizes 1024 --gen 128 --reps 1 --concurrency 1,4,16 --thinking off`, 990-token nonce-first prompts, one burst per
  visit, 3 visits on one server start (fx3 restarted per visit; here the 3 visits are back to back, so the range is run-to-run noise only). Plus
  `--sizes 4096 --concurrency 4` once.

## Results (aggregate decode tok/s; TTFT of the median stream, s)
| streams | vLLM seqs=16 (this run, median of 3) | range | TTFT s | vLLM seqs=2 (old) | SGLang | IntarsiaML #117 |
|---:|---:|---|---:|---:|---:|---:|
| 1 | 69.8 | 69.6-71.2 | 0.18 | 70 | 67 | 73.4 |
| 4 | 160.0 | 157.9-161.4 | 0.60 | 105 | 155 / 157.5 | 123.3 |
| 16 | 294.7 | 292.7-295.6 | 1.62 | 105 | 217 / 219.9 | 166.6 |
| 4 @ 4k prompt | 104.7 (1 visit) | | 2.10 | not measured | not measured | 83.2 |

Old TTFT (fx3): vLLM seqs=2 0.20 / 1.5 / 8.8; SGLang 0.20 / 0.62 / 2.0. Per-stream decode here: 77.1 / 48.4-49.8 / 23.8-24.2.

Reading: with 16 sequences allowed vLLM is the concurrency leader on this box, 294.7 at 16 streams (+34 % over SGLang 219.9, +77 % over
IntarsiaML #117), and level with SGLang at 4 streams (160 vs 157.5, within noise). The earlier "SGLang is the concurrency leader" came from the
seqs=2 cap. Caveats: SGLang numbers are from fx3's separate vLLM-down window and not re-run here; this run is the retro build (nightly 65b7662d)
rather than fx3's spark1 build (`nightly-aarch64`), same vLLM version string; retro uses a 131072 context and an 8 GiB KV pool, fx3's spark1 server 32768.

## Run ids (localeval, logged to unanimis)
- p1 `20261023-125924-adhoc-vr-35b-vllm-seqs16-p1-conc` (unanimis 20e01785-7bdd-438a-8018-d1f83969c6cf)
- p2 `20261023-125940-adhoc-vr-35b-vllm-seqs16-p2-conc`
- p3 `20261023-125957-adhoc-vr-35b-vllm-seqs16-p3-conc`
- 4k@4 `20261023-130013-adhoc-vr-35b-vllm-seqs16-4k4` (unanimis 2cdd7d8c-abf3-469d-9dcc-4f97b6f6da82)

## Retro downtime
Stopped 12:54:15, healthy again 13:04:05 (590 s: 285 s for the test server to start, the rest for the production container to reload). After restore,
`/v1/models` listed `guidance` (max_model_len 131072) and a short completion returned "OK".

Raw: `docs/results/2026-10-23-vr-vllm-maxseqs16/raw/` (run.sh, run.log, conc-*.out, vllm-server.log).
