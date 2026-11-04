#!/bin/zsh
# HF f32 and bf16 forward (W4A16, one full-sequence forward per stream) on prompt + the Metal solo stream's 200 ids.
[ -n "${FLEETQ_JOB:-}" ] || exit 2
cd /opt/code/intarsiaml/.worktrees/m27-flip
PY=/opt/code/intarsiaml/tools/.venv/bin/python
O=$PWD/.scratch/logs/oracle; M=/Users/josef/models/Qwen3.8-27B-Quark-AWQ-MXFP4
for dt in float32 bfloat16; do
  $PY -I .scratch/hf_ref.py $O/hf-seqs.json --model $M --out $O/hf-$dt.safetensors --act none --quark-site /nonexistent --dtype $dt 2>&1 | grep -vE "falling back" | tail -9
done
