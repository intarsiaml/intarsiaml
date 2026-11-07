#!/usr/bin/env bash
# Lane c27: one quiet slot, several ab.sh invocations (ids gate, then speed rounds). Run from the Mac:
#   fleetq run --host spark1 --quiet gpu,cpu --mem 50 --expect 28 --label "..." -- scripts/c27/win.sh <out dir name> [ids|speed ...]
# Env passed through to ab.sh on spark1: ARMS, ARM_<name>, SIZES, ROUNDS, NTOK, IDSP.
set -uo pipefail
S="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; REPO="$(cd "$S/../.." && pwd)"
OUT=${1:?out dir name}; shift
L=$REPO/docs/results/2026-11-07-cuda27-decode/raw/$OUT; mkdir -p "$L"
FWD=""; for v in ARMS SIZES ROUNDS NTOK IDSP $(compgen -e | grep '^ARM_'); do [ -n "${!v:-}" ] && FWD="$FWD $v='${!v}'"; done
for step in "$@"; do
  ids=0; [ $step = ids ] && ids=1
  ssh -o BatchMode=yes spark1 "IDS=$ids O=\$HOME/intarsia-c27/out/$OUT/$step $FWD bash -s" < "$S/ab.sh" 2>&1 | tee -a "$L/$step.log"
done
rsync -a --exclude 'p-*.txt' spark1:intarsia-c27/out/$OUT/ "$L/" 2>/dev/null
