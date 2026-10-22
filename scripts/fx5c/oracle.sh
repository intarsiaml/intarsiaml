#!/usr/bin/env bash
# Lane FX5c: the f64 CPU oracle (`accuracy --backend cpu`, fx1's recipe: chunk 256, top-64 self-golden) on the teacher-forced
# `continue` sequence. INPUT is the one-prompt golden of make_oracle_input.py. HOST=spark1 uses the binary built there by
# cuda-dump.sh (its cpu backend is the same code); HOST=endor uses /opt/code/target-fx5c/release/intarsia.
#   fleetq run --host spark1 --mem 45 --expect 60 --load cpu --label "fx5c: CPU oracle, 4112-token continue sequence" -- scripts/fx5c/oracle.sh INPUT
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
INPUT=${1:?input golden}
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "$REPO"
HOST=${HOST:-spark1}
RAW=${RAW:-$REPO/docs/results/2026-10-22-fx5c-reuse-vs-oracle/raw}
OO=${ORACLE_OUT:-oracle}
mkdir -p "$RAW/$OO"
if [[ $HOST == endor ]]; then
  M=/Users/josef/models/Qwen3.6-35B-A3B-NVFP4
  /usr/bin/time -l /opt/code/target-fx5c/release/intarsia accuracy --model $M --backend cpu --golden "$INPUT" --mode prefill --prefill-chunk 256 --mem-budget 40 --max-seq 8192 --allow-dirty --write-self-golden "$RAW/$OO/oracle.dump.safetensors" > "$RAW/$OO/oracle.log" 2>&1 < /dev/null
  echo "exit $?" >> "$RAW/$OO/oracle.log"
else
  RPATH='export PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH;'
  rsh() { ssh -n -o BatchMode=yes "$HOST" "$RPATH export FLEETQ_JOB=$FLEETQ_JOB; $*"; }
  R='~/intarsia-decfuse'
  M=/home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5
  rsh "mkdir -p $R/out/fx5c/$OO"
  rsync -a "$INPUT" "$HOST:intarsia-decfuse/out/fx5c/$OO/input.safetensors"
  rsh "cd $R/fx5c && /usr/bin/time -v $R/fx5c/target/acc/release/intarsia accuracy --model $M --backend cpu --golden $R/out/fx5c/$OO/input.safetensors --mode prefill --prefill-chunk 256 --mem-budget 40 --max-seq 8192 --allow-dirty --write-self-golden $R/out/fx5c/$OO/oracle.dump.safetensors > $R/out/fx5c/$OO/oracle.log 2>&1; echo exit \$? >> $R/out/fx5c/$OO/oracle.log"
  rsync -a "$HOST:intarsia-decfuse/out/fx5c/$OO/oracle.log" "$HOST:intarsia-decfuse/out/fx5c/$OO/oracle.dump.safetensors" "$RAW/$OO/"
fi
tail -5 "$RAW/$OO/oracle.log"
