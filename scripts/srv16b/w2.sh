#!/usr/bin/env bash
# Lane srv16b GPU window 2: exactness gates of the default (cap 3072 + half rule) and the 3-round interleaved A/B
# main (base) vs #235's 8192-row pass (dev, INTARSIA_PACK_ROWS=0 INTARSIA_PACK_HALF=0) vs the default (dev).
#   fleetq run --host spark1 --quiet gpu,cpu --mem 50 --expect 45 --label "srv16b: W2 gates + A/B" -- scripts/srv16b/w2.sh
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq}"
H="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RAW=/opt/code/intarsiaml/.worktrees/srv16b/docs/results/2026-10-29-srv16b-pack-latency/raw
scp -q "$H/lat.py" spark1:intarsiaml-srv16b/lat.py
date +%T
"$H/batch-real.sh" packed > "$RAW/w2-batch-real-packed.txt" 2>&1; tail -3 "$RAW/w2-batch-real-packed.txt"
"$H/batch-real.sh" run > "$RAW/w2-batch-real.txt" 2>&1; tail -3 "$RAW/w2-batch-real.txt"
"$H/test.sh" run --test rows_exact > "$RAW/w2-rows-exact.txt" 2>&1; tail -3 "$RAW/w2-rows-exact.txt"
date +%T
IDENT_OUT=ident-w2 "$H/ident.sh" base:base dev:dev; echo "ident rc=$?"
IDENT_OUT=ident-w2-mix IDENT_MIX=1 "$H/ident.sh" base:base dev:dev; echo "ident mix rc=$?"
date +%T
X=--max-batch_16
"$H/le-window.sh" w2 3 base:base::$X p8k:dev:INTARSIA_PACK_ROWS=0,INTARSIA_PACK_HALF=0:$X dev:dev::$X
date +%T
