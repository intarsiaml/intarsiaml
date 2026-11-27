#!/usr/bin/env bash
# Window S5 (lane ink-hip-overlap; checks only, within the brief's ~15 min), ON tatooine under window.sh: gate 2's solo server, which
# S2 and S4 missed (a solo serve admits its own P12 plan shapes, ~100 s uncached, longer than their waits): the solo server (no
# --max-batch, --spec off: W4's config) of o5 (the merged head) on card 0 and of o4 on card 1 at once, each the 16 real prompts one at
# a time (serve_bitwise.py solo), compared offline with W4's solo replies.
set -uo pipefail
export BG_CAP=${BG_CAP:-200}
. "$HOME/intarsiaml-ink-hip-bpf/.scratch/stage_lib.sh"
L=$REPO/logs/s5; mkdir -p "$L"
one() {  # card binary
  SENV="" WAITS=$(( $(left) - 45 )) srv "$1" solo-$2 "$2" --max-seq 4608 --spec off
  [ "$UP" != 0 ] && bw solo-$2 solo
  stop "$1"
}
echo "[s5] start $(now)"
one 0 o5 > "$L/card0.out" 2>&1 &
P0=$!
one 1 o4 > "$L/card1.out" 2>&1 &
P1=$!
wait $P0 $P1
stop 0; stop 1
echo "[s5] card0:"; cat "$L/card0.out"
echo "[s5] card1:"; cat "$L/card1.out"
echo "[s5] done $(now)"
