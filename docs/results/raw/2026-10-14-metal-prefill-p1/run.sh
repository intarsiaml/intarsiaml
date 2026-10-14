#!/bin/zsh
# metal prefill P1 spike: the measurement burst. Run inside a fleetq slot (FLEETQ_JOB set by the caller):
#   fleetq acquire --host endor --exclusive --mem 24 --label "metal P1 NAX gemm spike: final bench" --wait   # id N
#   FLEETQ_JOB=N docs/results/raw/2026-10-14-metal-prefill-p1/run.sh
#   fleetq release N
set -u
R=${R:-/opt/code/intarsiaml/.claude/worktrees/agent-a1247931a45f1aeb7/docs/results/raw/2026-10-14-metal-prefill-p1}
BIN=${BIN:-/opt/code/intarsiaml/.claude/worktrees/agent-a1247931a45f1aeb7/target/release/examples/gemm_nax}
PY=/Users/josef/.local/share/uv/tools/mlx-lm/bin/python
idle() { for i in 1 2 3; do ioreg -r -c IOAccelerator -d 1 | grep -o '"Device Utilization %"=[0-9]*'; sleep 1; done; }

echo "== idle before" | tee $R/gpu-idle.txt
sleep 5; idle | tee -a $R/gpu-idle.txt
pmset -g therm >> $R/gpu-idle.txt 2>&1

# 1. MLX before
$PY $R/mlx_gemm_chain.py > $R/mlx_gemm_chain-before.jsonl 2> $R/mlx_gemm_chain-before.err
# 2. errors vs the f64 reference (all tile configurations are bitwise equal; the hash column shows it)
NAX_M=205,256,973,1024,2048 $BIN err > $R/err.txt 2>&1
# 3. microbench, same weights every dispatch (what the MLX bench does) and a rotating pool of distinct weights
sleep 5
NAX_ROUNDS=15 $BIN bench > $R/bench-same-weights.txt 2>&1
NAX_ROUNDS=15 NAX_POOL_MIB=3072 NAX_ONLY=t128x64w4,t64x64w4 $BIN bench > $R/bench-pool3g.txt 2>&1
# 4. MLX after
echo "== idle after" >> $R/gpu-idle.txt
idle | tee -a $R/gpu-idle.txt
$PY $R/mlx_gemm_chain.py > $R/mlx_gemm_chain-after.jsonl 2> $R/mlx_gemm_chain-after.err
echo done
