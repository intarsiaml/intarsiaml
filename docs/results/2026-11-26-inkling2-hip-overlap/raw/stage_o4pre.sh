#!/usr/bin/env bash
# Pre-stage of window O4 (lane ink-hip-overlap) on card 0: main5 (origin/main a4c8f3b19) at the recipe's default serve config
# (`--max-batch 16`, nothing else): its admission prime, the context line and KV capacity (the default-budget check), then main5's
# greedy md5 (spec off) at short / 4k x 64 / 256. Card 1 (stage_o4bg.sh) does the same for o4 at the same time.
set -uo pipefail
. "$HOME/intarsiaml-ink-hip-bpf/.scratch/stage_lib.sh"
L=$REPO/logs/o4; mkdir -p "$L"
CAPT=${PRE_CAP:-170}
SENV="" WAITS=150 srv 0 def-main5 main5 --max-batch 16
stop 0
md5s 0 main5 "--spec off"
stop 0
echo "[pre] done $(now)"
