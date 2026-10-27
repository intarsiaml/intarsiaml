#!/usr/bin/env bash
# Lane cgv: nsys decode profiles on spark1, inside a fleetq slot (FLEETQ_JOB set). Usage: profs.sh <name>:<tree>[:extra generate args, spaces as underscores] ...
# Writes raw/prof/<name>.{log,gen.err,rank.txt,shapes.txt} (scripts/cgv/rank_decode.py, shapes.py); the trace csv is deleted after.
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq}"
D=$(cd "$(dirname "$0")" && pwd); R=$D/../../docs/results/2026-10-27-cgv-cuda-mxfp4-gemv/raw/prof; mkdir -p "$R"
for spec in "$@"; do
  IFS=: read -r name tree xargs <<< "$spec"; xargs=${xargs//_/ }
  ssh spark1 "FLEETQ_JOB=$FLEETQ_JOB bash -s -- $tree prof-$name $xargs" < "$D/prof-decode.sh" > "$R/$name.log" 2>&1
  scp -q spark1:intarsiaml-cgv/out/prof-$name/trace_cuda_gpu_trace.csv "$R/$name.trace.csv" 2>/dev/null
  scp -q spark1:intarsiaml-cgv/out/prof-$name/gen.err "$R/$name.gen.err" 2>/dev/null
  W=$(grep -aoE "decode [0-9]+ ms" "$R/$name.gen.err" | tail -1 | grep -oE "[0-9]+" | awk '{print $1/1000}')
  python3 "$D/rank_decode.py" "$R/$name.trace.csv" "$W" > "$R/$name.rank.txt" 2>&1
  python3 "$D/shapes.py" "$R/$name.trace.csv" "$W" > "$R/$name.shapes.txt" 2>&1
  rm -f "$R/$name.trace.csv"   # 30-50 MB each; the rank and shape tables are the record
  grep -aE "finish=|spec:" "$R/$name.gen.err" | cut -c1-200
done
