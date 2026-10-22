# Metal task quality: intarsia vs mlx-lm vs llama.cpp (lane mq, 2026-10-22)

Measurement only. No engine changed. Base `origin/main` 25db14d9 (intarsia built `--features metal`, `CARGO_TARGET_DIR=/opt/code/wt-mq-target`). Host endor (M5 Max). Raw outputs and result JSON: `docs/results/2026-10-22-mq-metal-quality/raw/`; driver `scripts/mq-metal-quality/arm.sh`; CI math `scripts/mq-metal-quality/ci.py`.

## Result

Scores, greedy (temperature 0), 95% Wilson CI. MMLU n = 1425 (25 questions x 57 subjects, localeval `--limit 25`); GSM8K n = 25. GSM8K score is localeval's primary `gsm8k_flexible`.

| model / weights | engine | MMLU (n=1425) | GSM8K flexible (n=25) |
|---|---|---|---|
| Qwen3-8B BF16 safetensors | intarsia Metal | 0.673 [0.648, 0.697] | 0.92 [0.75, 0.98] |
| Qwen3-8B BF16 safetensors | mlx-lm 0.31.3 | 0.679 [0.654, 0.702] | 0.92 [0.75, 0.98] |
| Qwen3-8B BF16 GGUF | llama.cpp (llama-server) | 0.676 [0.652, 0.700] | 0.88 [0.70, 0.96] |
| Qwen3.6-35B-A3B NVFP4 | intarsia Metal (only engine) | 0.840 [0.820, 0.858] | 0.64 [0.45, 0.80] |

**Reading.**
1. Qwen3-8B BF16: all three engines' CIs overlap on both tasks. No quality difference is claimed in either direction. The MMLU spread (0.673 to 0.679) is 0.6 points on n=1425, far inside the CI half-width of about 2.4 points. GSM8K at n=25 has CIs 20-25 points wide and can only rule out a gross failure.
2. Qwen3.6-35B-A3B: intarsia only (see below). It is a different model from the 8B rows, so the 8B-versus-35B gap is not an engine effect. Its GSM8K figure (strict 0.40) is a harness-format property (see Settings), not a regression claim; the CI [0.45, 0.80] is wide.
3. The 8B MMLU level (0.68) is low for Qwen3-8B because of the harness (0-shot, thinking off, text prompt with answer parsing), identical on every engine; only the cross-engine comparison is meaningful.
4. `gsm8k_strict` is 0.000 on all three 8B arms (answer-format property under `--structured none`), so it carries no information; flexible is the reported figure.

Incidental speed inside these runs (short varying prompts, non-exclusive slots, other lanes' jobs on endor at the same time; NOT a speed result, use the exclusive sweeps in the standings docs): MMLU decode tok/s on 8B: intarsia 35.5, mlx-lm 21.1, llama.cpp 17.0.

## Settings (all arms identical)

Per-arm localeval command (0.3.0, deepeval 4.2.6, deepeval git 1f24e53):

`uv run localeval quality --url http://127.0.0.1:<port>/v1 --model <id> --thinking off --structured none --bench mmlu,gsm8k --limit 25 --label mq-<arm>`

- These are the flags of fx3's concurrently running Spark quality jobs (fleetq #3979/#3980: `--thinking off --structured none --bench mmlu,gsm8k --limit 25`, read from the live process command line), and the same as the earlier q1 lane. localeval has no seed option: temperature is fixed at 0.0, dataset subsets are deterministic, and `--max-tokens` is the default 1024 (thinking off). I did not retrieve fx3's final numbers (its quality jobs had not finished), so no numeric cross-platform comparison is made here.
- Chat template: every engine applies the model's own template with thinking disabled. mlx-lm: `--chat-template-args '{"enable_thinking":false}'`; llama.cpp: `--jinja` (GGUF-embedded template) plus localeval's thinking-off request field; intarsia: its built-in Qwen3 template with the same request field. I did not diff the rendered prompts across engines; a template difference would likely show as a score difference, and none was seen at this n.
- Servers: intarsia `serve --backend metal --max-seq 8192 --profile-every 0 --no-supervisor --no-ui`; mlx-lm `mlx_lm.server --prompt-cache-size 1`; llama.cpp server `-ngl 999 -fa on -c 8192 --cache-ram 0 --jinja`. Each server was started and stopped inside its own fleetq slot (non-exclusive, `--mem` 24 for 8B, 40 for 35B); nothing else on endor was touched.
- Weights: Qwen3-8B BF16 as in the standings (`docs/results/2026-10-22-fx6-standings.md` section 2.2: intarsia and mlx-lm on `/Users/josef/models/Qwen3-8B`, llama.cpp on `Qwen3-8B-BF16.gguf`).

## 35B comparator: none available

The only complete Qwen3.6-35B-A3B checkpoint on endor that intarsia loads is the NVFP4 one (compressed-tensors: NVFP4 experts plus FP8 W8A8 projections; see the LOG row of 2026-10-21 for lane M35). mlx-lm needs MLX-format weights and llama.cpp needs GGUF; neither format of this checkpoint exists on the box, and `ref-qwen36-mlx4` (MLX affine 4-bit) is a different quant, not the same weights. I decided from the checkpoint formats and did not attempt a load. So the 35B row is intarsia only.

## Runs

| arm | fleetq job | localeval run id | wall s (MMLU / GSM8K) |
|---|---|---|---|
| mlx-lm 8B | #4023 | `20261022-065854-adhoc-mq-mlx8` | 1710 / 161 |
| llama.cpp 8B | #4024 | `20261022-065857-adhoc-mq-llama8` | 1678 / 213 |
| intarsia 35B | #4025 | `20261022-073038-adhoc-mq-intarsia35` | 2549 / 37 |
| intarsia 8B | #4112 | `20261022-084653-adhoc-mq-intarsia8` | 780 / 98 |
| build (intarsia Metal) | #3997 | | |

Earlier attempts #4000-#4003 and #4009 failed within seconds (a relative output path in `arm.sh` after a `cd`, fixed to an absolute path before the runs above) and produced no scores. #4026 was cancelled and resubmitted as #4112 with `--expect 50` to fit before another lane's exclusive reservation.

## Limits

- GSM8K n=25: CIs 20-35 points wide; only gross failures are detectable. MMLU n=1425 resolves about 5 points between engines.
- MMLU k is derived from the 3-decimal score (error under 1 question); Wilson interval, unpaired. Per-question pairing (McNemar) would be tighter and was not done.
- Non-exclusive slots: quality is unaffected by contention; the incidental timings are.
- The harness answer parsing is text-based (`--structured none`); absolute levels are harness properties.
