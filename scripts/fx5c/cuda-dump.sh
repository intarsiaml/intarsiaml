#!/usr/bin/env bash
# PROMPT (default long_docs_4k), OUT (subdirectory name), DUMPS (where the large f32 rows go, out of the repo).
# Lane FX5c: build the CLI (with --reuse-dump / --reuse-seq) on spark1 and dump the teacher-forced `continue` rows of the 35B,
# long_docs_4k, chunk 1024 (fx5b's diagnostic). Without SEQ the engine's own reply and cold stream are used and written to
# seq.json; with SEQ=<seq.json> that sequence is forced.
#   fleetq run --host spark1 --mem 45 --expect 45m --label "fx5c: build + 35B continue rows dump (non-exclusive, numerics only)" -- scripts/fx5c/cuda-dump.sh
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "$REPO"
HOST=${HOST:-spark1}; NAME=fx5c; R='~/intarsia-decfuse'
OUT=${OUT:-cuda}                       # subdirectory of $RAW and of the box's out dir
RAW=${RAW:-$REPO/docs/results/2026-10-22-fx5c-reuse-vs-oracle/raw}
mkdir -p "$RAW/$OUT"
RPATH='export PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH;'
rsh() { ssh -n -o BatchMode=yes "$HOST" "$RPATH export FLEETQ_JOB=$FLEETQ_JOB; $*"; }
M35=/home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5
G='$HOME/intarsiaml-golden/qwen36-35b-a3b-nvfp4-vllm-long-top20.safetensors'
BIN="$R/$NAME/target/acc/release/intarsia"
if [[ "${BUILD:-1}" == 1 ]]; then
  rsh "mkdir -p $R/$NAME; test -d $R/$NAME/target || cp -a $R/fx5/target $R/$NAME/target"
  "$REPO/scripts/cuda-decode/build.sh" $NAME 2>&1 | tail -8
fi
SEQARG=""
if [[ -n "${SEQ:-}" ]]; then
  rsh "mkdir -p $R/out/fx5c/$OUT"; rsync -a "$SEQ" "$HOST:intarsia-decfuse/out/fx5c/$OUT/seq-in.json"; SEQARG="--reuse-seq $R/out/fx5c/$OUT/seq-in.json"
fi
rsh "mkdir -p $R/out/fx5c/$OUT; cd $R/$NAME && $BIN accuracy --model $M35 --backend cuda --golden $G --mode reuse --max-seq 8192 --mem-budget 40 --allow-dirty --reuse-diag --reuse-prompt ${PROMPT:-long_docs_4k} --prefill-chunk 1024 --reuse-dump $R/out/fx5c/$OUT/dump $SEQARG > $R/out/fx5c/$OUT/run.log 2>&1; echo exit \$? >> $R/out/fx5c/$OUT/run.log"
rsync -a --exclude '*.f32' "$HOST:intarsia-decfuse/out/fx5c/$OUT/" "$RAW/$OUT/"
if [[ -n "${DUMPS:-}" ]]; then mkdir -p "$DUMPS/$OUT"; rsync -a --include '*.f32' --include 'seq.json' --exclude '*' "$HOST:intarsia-decfuse/out/fx5c/$OUT/dump/" "$DUMPS/$OUT/"; fi
tail -15 "$RAW/$OUT/run.log"
