#!/bin/sh
set -e
cd /opt/code/intarsiaml/.worktrees/nab
date
/opt/code/intarsiaml/tools/.venv/bin/python tools/golden.py --model /opt/code/intarsiaml/.worktrees/nab/.work/qwen3.5-2b-mlx4-dequant --out .work/goldens/Qwen3.5-2B-MLX-4bit-long-fp32.safetensors --device cpu --dtype float32 --prompts tools/prompts/golden.jsonl tools/prompts/golden_long.jsonl
date
/opt/code/intarsiaml/tools/.venv/bin/python tools/golden.py --model /opt/code/intarsiaml/.worktrees/nab/.work/qwen3.5-2b-mlx4-dequant --out .work/goldens/Qwen3.5-2B-MLX-4bit-4k-fp32-top20.safetensors --device cpu --dtype float32 --topk 20 --prompts tools/prompts/golden_4k.jsonl
date
