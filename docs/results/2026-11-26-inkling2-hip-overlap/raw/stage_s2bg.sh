#!/usr/bin/env bash
# Background stage of window S2 (lane ink-hip-overlap) on card 1 (revised 21:10 by the resumed agent: S1 closed the md5 / G2 / poison
# items): the merged head o5 (1b35c6741 = o4 + origin/main aee06201c, #535) greedy md5 `--spec off` vs main (W3), the solo server (no
# --max-batch) of o4, o5's md5 with DSpark (its default), then o5's solo server: the 16 real prompts one at a time (serve_bitwise.py
# solo), compared offline with W4's solo replies.
set -uo pipefail
. "$HOME/intarsiaml-ink-hip-bpf/.scratch/stage_lib.sh"
L=$REPO/logs/s2; mkdir -p "$L"
solo() {
  local b=$1
  (( $(left) < 60 )) && { echo "[solo-$b] SKIP ($(left) s left)"; return; }
  SENV="" WAITS=40 srv 1 solo-$b $b --max-seq 4608
  [ "$UP" != 0 ] && bw solo-$b solo
  stop 1
}
md5s 1 o5 "--spec off"
solo o4
md5s 1 o5 ""
solo o5
echo "[bg] done $(now)"
