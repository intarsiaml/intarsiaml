#!/usr/bin/env bash
# Lane FX5c: the same teacher-forced `continue` rows on Metal (endor), forcing the sequence CUDA generated (SEQ). Needs the build of metal-build.sh.
#   fleetq run --host endor --mem 30 --expect 30 --label "fx5c: Metal 35B continue rows dump (non-exclusive, numerics only)" -- scripts/fx5c/metal-dump.sh SEQ.json
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
SEQ=${1:?seq.json}
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "$REPO"
RAW=${RAW:-$REPO/docs/results/2026-10-22-fx5c-reuse-vs-oracle/raw}
DUMPS=${DUMPS:?directory for the f32 dumps (kept out of the repo)}
mkdir -p "$RAW/metal" "$DUMPS/metal"
BIN=${BIN:-/opt/code/target-fx5c/release/intarsia}
"$BIN" accuracy --model $HOME/models/Qwen3.6-35B-A3B-NVFP4 --backend metal --golden /opt/code/intarsiaml/golden/qwen36-35b-a3b-nvfp4-vllm-long-top20.safetensors --mode reuse \
  --max-seq 8192 --mem-budget 30 --allow-dirty --reuse-diag --reuse-prompt long_docs_4k --prefill-chunk 1024 --reuse-dump "$DUMPS/metal" --reuse-seq "$SEQ" > "$RAW/metal/run.log" 2>&1 < /dev/null
echo "exit $?" >> "$RAW/metal/run.log"
cp "$DUMPS/metal/seq.json" "$RAW/metal/seq.json" 2>/dev/null
tail -8 "$RAW/metal/run.log"
