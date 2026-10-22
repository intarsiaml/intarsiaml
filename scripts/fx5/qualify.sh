#!/usr/bin/env bash
# Lane FX5, spark1: `qualify` with only the acc-long stage on the 35B long golden, on the final tree (recipe tolerance compiled in).
#   fleetq run --host spark1 --mem 45 --expect 45 --label "fx5 qualify acc-long 35B, final tree (vLLM up)" -- scripts/fx5/qualify.sh
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
S="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$S/../.." && pwd)"; cd "$REPO"
HOST=${HOST:-spark1}
RAW=${RAW:-$REPO/docs/results/2026-10-22-fx5-hybrid-reuse-ci/raw/final}
mkdir -p "$RAW"
R='~/intarsia-decfuse'
RPATH='export PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH;'
rsh() { ssh -n -o BatchMode=yes "$HOST" "$RPATH export FLEETQ_JOB=$FLEETQ_JOB; $*"; }
M35=/home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5
G='$HOME/intarsiaml-golden/qwen36-35b-a3b-nvfp4-vllm-long-top20.safetensors'
BIN="$R/fx5/target/acc/release/intarsia"
OUT="$R/out/fx5q"
"$REPO/scripts/cuda-decode/build.sh" fx5 2>&1 | tail -3 | tee "$RAW/00-build.out"
rsh "rm -rf $OUT; mkdir -p $OUT"
skips=(); for s in kernels gate acc-short acc-format bench acc-4k envelope bench-long ab tp; do skips+=(--skip $s); done
rsh "cd $R/fx5 && $BIN qualify -m $M35 --backend cuda --golden-long $G --out $OUT/qualify --mem-budget 40 --stage-timeout-min 120 ${skips[*]} > $OUT/qualify.log 2>&1; echo exit \$? >> $OUT/qualify.log"
rsync -a "$HOST:intarsia-decfuse/out/fx5q/" "$RAW/"
tail -12 "$RAW/qualify.log" | cut -c1-300
python3 - "$RAW/qualify/full.json" <<'PY'
import json,sys
j=json.load(open(sys.argv[1]))
for s in j.get("stages",[]):
    if s.get("id")=="acc-long": print("acc-long", s.get("verdict"), json.dumps((s.get("metrics") or {}).get("reuse")))
PY
