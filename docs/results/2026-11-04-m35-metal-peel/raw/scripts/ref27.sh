#!/bin/zsh
cd /opt/code/intarsiaml/.worktrees/m35-peel
/opt/code/intarsiaml/tools/.venv/bin/python -I tools/golden_quark_mxfp4.py --model /Users/josef/models/Qwen3.8-27B-Quark-AWQ-MXFP4 --out .scratch/acc27/ref-f32.safetensors --prompts .scratch/acc27/prompts.jsonl --act none --quark-site /nonexistent --dtype float32 --template chatml 2>&1 | grep -vE "falling back"
