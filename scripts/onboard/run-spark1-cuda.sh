#!/usr/bin/env bash
# Lane `on`: one back-to-back `intarsia onboard` pass on spark1 (CUDA) with a golden MADE from the vLLM
# reference server (fresh --golden-dir, so the golden-generation path runs).
# Run INSIDE a fleetq slot: FLEETQ_JOB=<id> scripts/onboard/run-spark1-cuda.sh [tree] [out-name]
# (vLLM stays up and is the reference at http://localhost:8000/v1 on the box.)
set -euo pipefail
: "${FLEETQ_JOB:?run inside a fleetq slot}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TREE=${1:-on}
OUTN=${2:-onboard-35b-cuda}
HOST=${HOST:-spark1}
M35=/home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5
RPATH='export PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH;'
rsh() { ssh -n -o BatchMode=yes "$HOST" "$RPATH export FLEETQ_JOB=$FLEETQ_JOB; $*"; }
T0=$(date +%s)
echo "== build $(date -u +%T)"
"$REPO/scripts/cuda-decode/build.sh" "$TREE"
T1=$(date +%s); echo "build_seconds=$((T1 - T0))"
OUT='$HOME/onboard-runs/'$OUTN
rsh "rm -rf $OUT; mkdir -p $OUT/golden-fresh"
echo "== onboard start $(date -u +%T)"
rsh "cd \$HOME/intarsia-decfuse/$TREE && date -u +%FT%TZ > $OUT/start.txt && /usr/bin/time -v target/acc/release/intarsia onboard --model $M35 --backend cuda --golden-from http://localhost:8000/v1 --mem-budget 38 --out $OUT/report -- --golden-dir $OUT/golden-fresh --no-unim > $OUT/driver.log 2>&1; echo exit=\$? >> $OUT/driver.log; date -u +%FT%TZ > $OUT/end.txt; tail -30 $OUT/driver.log"
T2=$(date +%s); echo "onboard_wall_seconds=$((T2 - T1))"
