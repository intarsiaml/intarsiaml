#!/usr/bin/env bash
# Window S4 (lane ink-hip-overlap; the coordinator's extra window, checks only, no speed), ON tatooine under window.sh:
#   card 0: the merged head o5 at the speed config (--max-batch 16 --max-seq 4608 --kv-fraction 0.9, overlap on by default): its
#           first start runs and caches o5's admission (a new binary, empty cache); then the real prompts 8 (wave 0) and 16 at once;
#           then o5 with INTARSIA_OVERLAP=0 (the opt-in fallback), the same two modes. Replies compared offline with W4's solo replies.
#   card 1: the solo server (no --max-batch, --spec off: W4's config) of o4, then (after o5's admission is cached) of o5: the 16 real
#           prompts one at a time; then hip tiny inkling2_batch / inkling2_packed on the merged tree if time is left.
set -uo pipefail
export BG_CAP=${BG_CAP:-225}
. "$HOME/intarsiaml-ink-hip-bpf/.scratch/stage_lib.sh"
L=$REPO/logs/s4; mkdir -p "$L"; rm -f "$L/o5-primed"
end_metrics() { curl -sf -m 5 "http://$IP:8133/metrics" > "$L/$1-metrics-end.txt" 2>/dev/null; echo "[$1] $(grep -E '^intarsia_batch_overlap_steps_total ' "$L/$1-metrics-end.txt")"; }
card0() {
  SENV="" WAITS=170 srv 0 def-o5 o5 --max-batch 16 --max-seq 4608 --kv-fraction 0.9
  touch "$L/o5-primed"
  [ "$UP" != 0 ] && { bw def-o5 conc:8:0 conc; end_metrics def-o5; }
  stop 0
  (( $(left) < 60 )) && { echo "[off-o5] SKIP ($(left) s left)"; return; }
  SENV="INTARSIA_OVERLAP=0" WAITS=45 srv 0 off-o5 o5 --max-batch 16 --max-seq 4608 --kv-fraction 0.9
  [ "$UP" != 0 ] && { bw off-o5 conc:8:0 conc; end_metrics off-o5; }
  stop 0
}
card1() {
  SENV="" WAITS=100 srv 1 solo-o4 o4 --max-seq 4608 --spec off
  [ "$UP" != 0 ] && bw solo-o4 solo
  stop 1
  local i
  for i in $(seq 1 200); do [ -f "$L/o5-primed" ] && break; (( $(left) < 70 )) && break; sleep 1; done
  (( $(left) < 70 )) && { echo "[solo-o5] SKIP ($(left) s left)"; return; }
  SENV="" WAITS=60 srv 1 solo-o5 o5 --max-seq 4608 --spec off
  [ "$UP" != 0 ] && bw solo-o5 solo
  stop 1
  tiny 1 o5 inkling2_batch inkling2_packed
}
echo "[s4] start $(now)"
card0 > "$L/card0.out" 2>&1 &
P0=$!
card1 > "$L/card1.out" 2>&1 &
P1=$!
wait $P0 $P1
stop 0; stop 1
echo "[s4] card0:"; cat "$L/card0.out"
echo "[s4] card1:"; cat "$L/card1.out"
echo "[s4] done $(now)"
