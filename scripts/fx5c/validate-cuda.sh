#!/usr/bin/env bash
# Lane FX5c, spark1: the 35B qualify acc-long stage (reuse at chunk 1024 and the --mode all run at the plan chunk 2048) with the
# oracle-relative continue rule, then mutation and fallback runs on long_docs_4k. The tree must already be built there
# (`fleetq run --host spark1 --mem 4 -- scripts/cuda-decode/build.sh fx5c`).
#   fleetq run --host spark1 --mem 45 --expect 60 --label "fx5c: qualify acc-long 35B CUDA, oracle-relative continue (non-exclusive, numerics only)" -- scripts/fx5c/validate-cuda.sh
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "$REPO"
HOST=${HOST:-spark1}
RAW=${RAW:-$REPO/docs/results/2026-10-22-fx5c-oracle-gate/raw/cuda}
mkdir -p "$RAW"
R='~/intarsia-decfuse'
RPATH='export PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH;'
rsh() { ssh -n -o BatchMode=yes "$HOST" "$RPATH export FLEETQ_JOB=$FLEETQ_JOB; $*"; }
M35=/home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5
G='$HOME/intarsiaml-golden/qwen36-35b-a3b-nvfp4-vllm-long-top20.safetensors'
BIN="$R/fx5c/target/acc/release/intarsia"
OUT="$R/out/fx5c/val"
rsh "rm -rf $OUT; mkdir -p $OUT ~/intarsiaml-golden/oracle"
rsync -a "$REPO"/golden/oracle/*.continue-oracle.safetensors "$HOST:intarsiaml-golden/oracle/"
rsh "ls -la ~/intarsiaml-golden/oracle/ | grep continue"
skips=(); for s in kernels gate acc-short acc-format bench acc-4k envelope bench-long ab tp; do skips+=(--skip $s); done
rsh "cd $R/fx5c && $BIN qualify -m $M35 --backend cuda --golden-long $G --out $OUT/qualify --mem-budget 40 --stage-timeout-min 120 ${skips[*]} > $OUT/qualify.log 2>&1; echo exit \$? >> $OUT/qualify.log"
rsync -a "$HOST:intarsia-decfuse/out/fx5c/val/" "$RAW/"
tail -8 "$RAW/qualify.log" | cut -c1-300
python3 - "$RAW/qualify/full.json" <<'PY'
import json,sys
j=json.load(open(sys.argv[1]))
for s in j.get("stages",[]):
    if s.get("id")=="acc-long": print("acc-long", s.get("verdict"), json.dumps((s.get("metrics") or {}).get("reuse")))
PY
# mutation (a deliberate error on the warm rows only) and the absolute fallback, long_docs_4k, chunk 1024
A="$BIN accuracy --model $M35 --backend cuda --golden $G --mode reuse --max-seq 8192 --mem-budget 40 --allow-dirty --prefill-chunk 1024 --reuse-prompt long_docs_4k"
for m in "x1.2 scale:1.2" "x1.1 scale:1.1" "x1.01 scale:1.01"; do set -- $m
  rsh "cd $R/fx5c && INTARSIA_REUSE_TEST_BUG=$2 $A > $OUT/mut-$1.log 2>&1; echo exit \$? >> $OUT/mut-$1.log"
done
rsh "cd $R/fx5c && $A --no-reuse-oracle > $OUT/noorc.log 2>&1; echo exit \$? >> $OUT/noorc.log"
rsync -a "$HOST:intarsia-decfuse/out/fx5c/val/" "$RAW/"
grep -a -e 'continue judged' -e '^exit' -e 'prefix-reuse' "$RAW"/mut-*.log "$RAW"/noorc.log | cut -c1-300
