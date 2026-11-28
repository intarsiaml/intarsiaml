#!/usr/bin/env bash
# Mac side, inside `fleetq run --host tatooine-r9700 --load cpu` (lane ink-hip-table): for each NAME=SHA, export exactly that commit
# (git archive into .scratch/src-NAME), sync it to tatooine's lane folder (--delete; logs/ and .scratch/ kept), lay this lane's raw/
# scripts over it (docs only), and build the hip CLI there as .scratch/bin/intarsia-NAME (build.sh touches every source first).
set -uo pipefail
W=/opt/code/intarsiaml/.worktrees/ink-hip-table
L=intarsiaml-ink-hip-table; R=docs/results/2026-11-27-inkling2-hip-table/raw
ssh -o BatchMode=yes tatooine "mkdir -p ~/$L/.scratch/bin ~/$L/logs" < /dev/null || exit 2
for pair in "$@"; do
  NAME=${pair%%=*}; SHA=${pair#*=}
  X=$W/.scratch/src-$NAME; rm -rf "$X"; mkdir -p "$X"
  git -C "$W" archive "$SHA" | tar -x -C "$X" || exit 3
  rsync -a --delete --exclude /target --exclude /.git --exclude /.worktrees --exclude /logs --exclude /.scratch "$X/" tatooine:$L/ < /dev/null || exit 4
  ssh -o BatchMode=yes tatooine "mkdir -p ~/$L/$R" < /dev/null || exit 5
  rsync -a "$W/$R/" tatooine:$L/$R/ < /dev/null || exit 5
  full=$(git -C "$W" rev-parse "$SHA")
  echo "[prep] $NAME = $full $(date +%T)"
  ssh -o BatchMode=yes tatooine "echo '$full ($NAME, git archive)' > ~/$L/.scratch/bin/intarsia-$NAME.commit && bash ~/$L/$R/build.sh $NAME" < /dev/null || { echo "[prep] build $NAME failed"; exit 6; }
done
echo "[prep] done $(date +%T)"
