#!/usr/bin/env bash
# Card 1 of window W0 (lane ink-hip-overlap2): prime main6's batched admission and check its 16 real prompts at once against W4's
# solo replies (main's own changes since W4 must not move them); then, once card 0's W0P server (o6) is up (o6 primed), the poisoned
# run on o6 with overlap on (INTARSIA_HIP_POISON=ff, INTARSIA_OVERLAP=1): 16 at once and 4 x 4.
set -uo pipefail
. "$HOME/intarsiaml-ink-hip-bpf/.scratch/stage_lib.sh"
. "$HOME/intarsiaml-ink-hip-bpf/.scratch/stage_lib2.sh"
L=$REPO/logs/w0; mkdir -p "$L"
WAITS=150 srv 1 w0-main6 main6 --max-batch 16 --max-seq 4608 --kv-fraction 0.9
[ "$UP" != 0 ] && bw w0-main6 conc
stop 1
for i in $(seq 1 120); do grep -q "loaded ink" "$REPO/logs/serve/W0P.log" 2>/dev/null && break; sleep 1; done
echo "[bg] W0P up: $(grep -ac 'loaded ink' "$REPO/logs/serve/W0P.log" 2>/dev/null) $(now)"
if (( $(left) > 50 )); then
  SENV="INTARSIA_HIP_POISON=ff INTARSIA_OVERLAP=1" WAITS=$(( $(left) - 35 )) srv 1 poison-o6 o6 --max-batch 16 --max-seq 4608 --kv-fraction 0.9
  [ "$UP" != 0 ] && { bw poison-o6 conc conc:4:0 conc:4:1 conc:4:2 conc:4:3; curl -sf -m 5 "http://$IP:8133/metrics" | grep -E '^intarsia_batch_overlap_steps_total '; }
fi
stop 1
echo "[bg] done $(now)"
