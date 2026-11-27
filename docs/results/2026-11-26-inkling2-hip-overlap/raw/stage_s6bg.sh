#!/usr/bin/env bash
# Card 1 of window S6 (lane ink-hip-overlap2, order BA): first prime main6's batched admission (its first start; the marker
# logs/s6/main6-primed only once the server answered, which card 0's B visit waits for before A starts), then Inkling-2 greedy md5 on
# o6 (`--spec off` and the DSpark default, short / 4k x 64 / 256), then main6 `--spec off` while time lasts.
set -uo pipefail
. "$HOME/intarsiaml-ink-hip-bpf/.scratch/stage_lib.sh"
. "$HOME/intarsiaml-ink-hip-bpf/.scratch/stage_lib2.sh"
L=$REPO/logs/s6; mkdir -p "$L"; rm -f "$L/main6-primed"
WAITS=150 srv 1 prime-main6 main6 --max-batch 16 --max-seq 4608 --kv-fraction 0.9
[ "$UP" != 0 ] && { grep -aE "admission on plan" "$L/prime-main6.log"; touch "$L/main6-primed"; }
stop 1
md5s 1 o6 "--spec off"; md5s 1 o6 ""; md5s 1 main6 "--spec off"
stop 1
echo "[bg] done $(now)"
