#!/usr/bin/env bash
# Lane FX5c, spark2 (or HOST): (1) Qwen3-8B reuse gate on its 4k golden with the 35B continue-oracle files on the lookup path: must read as before,
# with no oracle line; (2) the 35B `qualify` acc-long stage end to end (qualify -> accuracy subprocess -> oracle/ beside the golden -> report parsing).
# Needs the tree built there (validate-cuda.sh BUILD=1 did it).
#   fleetq run --host spark2 --mem 38 --expect 70 --label "..." -- env HOST=spark2 scripts/fx5c/validate-extra.sh
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "$REPO"
HOST=${HOST:-spark2}
RAW=${RAW:-$REPO/docs/results/2026-10-22-fx5c-oracle-gate/raw/extra}
mkdir -p "$RAW"
R='~/intarsia-decfuse'
RPATH='export PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH;'
rsh() { ssh -n -o BatchMode=yes "$HOST" "$RPATH export FLEETQ_JOB=$FLEETQ_JOB; $*"; }
BIN="$R/fx5c/target/acc/release/intarsia"
OUT="$R/out/fx5c/extra"
HOST=$HOST "$REPO/scripts/cuda-decode/build.sh" fx5c 2>&1 | tail -3     # the gate code changed since the last build there
rsh "rm -rf $OUT; mkdir -p $OUT ~/intarsiaml-golden"
rsync -a /opt/code/intarsiaml/golden/qwen3-8b-4k-fp32-top20.safetensors "$HOST:intarsiaml-golden/"
M8=$(ssh -n "$HOST" 'ls -d ~/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/* | head -1')
echo "8B model: $M8"
rsh "ls ~/intarsiaml-golden/oracle | grep -c continue"
rsh "cd $R/fx5c && $BIN accuracy --model $M8 --backend cuda --golden ~/intarsiaml-golden/qwen3-8b-4k-fp32-top20.safetensors --mode reuse --max-seq 8192 --mem-budget 30 --allow-dirty --prefill-chunk 1024 --reuse-require-all > $OUT/reuse-8b.log 2>&1; echo exit \$? >> $OUT/reuse-8b.log"
rsh "cd $R/fx5c && $BIN accuracy --model $M8 --backend cuda --golden ~/intarsiaml-golden/qwen3-8b-4k-fp32-top20.safetensors --mode reuse --max-seq 8192 --mem-budget 30 --allow-dirty --prefill-chunk 1024 --reuse-require-all --no-reuse-oracle > $OUT/reuse-8b-noorc.log 2>&1; echo exit \$? >> $OUT/reuse-8b-noorc.log"
rsync -a "$HOST:intarsia-decfuse/out/fx5c/extra/" "$RAW/"
grep -a -e 'continue' -e 'oracle' -e '^exit' -e 'prefix-reuse' "$RAW/reuse-8b.log" | cut -c1-260 | tail -12
echo "=== CUDA direct reuse gate, chunks 1024 + 2048, mutation, fallback (validate-cuda.sh)"
HOST=$HOST RAW="$REPO/docs/results/2026-10-22-fx5c-oracle-gate/raw/cuda" "$REPO/scripts/fx5c/validate-cuda.sh" 2>&1 | tail -30
[[ "${QUALIFY:-1}" == 1 ]] || exit 0
echo "=== qualify acc-long 35B"
M35=/home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5
G='$HOME/intarsiaml-golden/qwen36-35b-a3b-nvfp4-vllm-long-top20.safetensors'
skips=(); for s in kernels gate acc-short acc-format bench acc-4k envelope bench-long ab tp; do skips+=(--skip $s); done
rsh "cd $R/fx5c && $BIN qualify -m $M35 --backend cuda --golden-long $G --out $OUT/qualify --mem-budget 40 --stage-timeout-min 120 ${skips[*]} > $OUT/qualify.log 2>&1; echo exit \$? >> $OUT/qualify.log"
rsync -a "$HOST:intarsia-decfuse/out/fx5c/extra/" "$RAW/"
tail -8 "$RAW/qualify.log" | cut -c1-300
python3 - "$RAW/qualify/full.json" <<'PY'
import json,sys
j=json.load(open(sys.argv[1]))
for s in j.get("stages",[]):
    if s.get("id")=="acc-long": print("acc-long", s.get("verdict"), json.dumps((s.get("metrics") or {}).get("reuse")))
PY
grep -a "continue judged" "$RAW"/qualify/logs/*reuse*.out 2>/dev/null | cut -c1-300
