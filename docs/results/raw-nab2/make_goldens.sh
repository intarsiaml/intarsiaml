#!/bin/sh
# long + 4k fp32 goldens of LFM2.5-1.2B-Instruct from HF transformers on CPU (tools/golden.py)
cd /opt/code/intarsiaml/.worktrees/nab2/tools || exit 1
PY=/opt/code/intarsiaml/tools/.venv/bin/python
D=../raw/nab2/goldens
$PY golden.py --model /Users/josef/.cache/huggingface/hub/models--LiquidAI--LFM2.5-1.2B-Instruct/snapshots/0f604ada3f766f9f257460c4c9f0b5d6f69d431b --out $D/lfm2.5-1.2b-instruct-long-fp32.safetensors --device cpu --dtype float32 --prompts prompts/golden.jsonl prompts/golden_long.jsonl || exit 1
$PY golden.py --model /Users/josef/.cache/huggingface/hub/models--LiquidAI--LFM2.5-1.2B-Instruct/snapshots/0f604ada3f766f9f257460c4c9f0b5d6f69d431b --out $D/lfm2.5-1.2b-instruct-4k-fp32-top20.safetensors --device cpu --dtype float32 --topk 20 --prompts prompts/golden_4k.jsonl
