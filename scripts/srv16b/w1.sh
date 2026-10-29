#!/usr/bin/env bash
# Lane srv16b GPU window 1 (exploratory, one round): main vs #235's pack (8192 rows) vs row caps and the half-queue rule, localeval
# 1/4/8/16 streams + lat.py bursts per arm.
#   fleetq run --host spark1 --quiet gpu,cpu --mem 50 --expect 40 --label "srv16b: W1 pack sweep" -- scripts/srv16b/w1.sh
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq}"
H="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
scp -q "$H/lat.py" spark1:intarsiaml-srv16b/lat.py
X=--max-batch_16
"$H/le-window.sh" w1 1 base:base::$X p8k:dev::$X r2k:dev:INTARSIA_PACK_ROWS=2048:$X r3k:dev:INTARSIA_PACK_ROWS=3072:$X \
  r4k:dev:INTARSIA_PACK_ROWS=4096:$X h:dev:INTARSIA_PACK_HALF=1:$X h3k:dev:INTARSIA_PACK_HALF=1,INTARSIA_PACK_ROWS=3072:$X \
  h4k:dev:INTARSIA_PACK_HALF=1,INTARSIA_PACK_ROWS=4096:$X
date +%T
