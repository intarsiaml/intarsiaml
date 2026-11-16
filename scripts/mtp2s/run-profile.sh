#!/usr/bin/env bash
# mtp2s 2026-11-16 tasks 1 + 2 in ONE fleetq slot: the draft-split trace run (draft.sh) and nsys arms (prof.sh) with seg / iter / cmp tables.
#   fleetq run --host spark2 --quiet gpu,cpu --mem 50 --expect 25 --label "mtp2s: draft split + nsys" -- scripts/mtp2s/run-profile.sh
# env: BIN (remote binary, default ~/intarsia-mtp2s/bin/base), ARMS (default "F2 FM FK1"), RAW (local dir for the text results)
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
H=spark2; REPO="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
RAW=${RAW:-$REPO/docs/results/2026-11-16-mtp2s/raw/profile}; mkdir -p "$RAW"
BIN=${BIN:-/home/nvidia/intarsia-mtp2s/bin/base}; ARMS=${ARMS:-F2 FM FK1}
rsh() { ssh -n -o BatchMode=yes $H "export FLEETQ_JOB=$FLEETQ_JOB; $*"; }
ssh -o BatchMode=yes $H 'mkdir -p ~/intarsia-mtp2s/src/scripts/mtp2s' && scp -q -o BatchMode=yes "$REPO"/scripts/mtp2s/*.sh "$REPO"/scripts/mtp2s/*.py $H:intarsia-mtp2s/src/scripts/mtp2s/
rsh "free -g | sed -n 2p; systemctl is-active earlyoom"
echo "== draft split trace"; rsh "env BIN=$BIN NAME=d GEN=256 bash ~/intarsia-mtp2s/src/scripts/mtp2s/draft.sh" | tee "$RAW/draft-run.txt"
echo "== nsys arms $ARMS"; rsh "env BIN=$BIN GEN=256 bash ~/intarsia-mtp2s/src/scripts/mtp2s/prof.sh $ARMS" | tee "$RAW/prof-run.txt"
for a in $ARMS; do
  rsh "cd ~/intarsia-mtp2s/src/scripts/mtp2s && python3 seg.py ~/intarsia-mtp2s/out/prof/$a.sqlite --names > ~/intarsia-mtp2s/out/prof/seg-$a.txt 2>&1; python3 iter.py ~/intarsia-mtp2s/out/prof/$a.sqlite ${VERIFY_KERNELS:-1464} > ~/intarsia-mtp2s/out/prof/iter-$a.txt 2>&1; python3 cmp.py ~/intarsia-mtp2s/out/prof/$a.sqlite > ~/intarsia-mtp2s/out/prof/cmp-$a.txt 2>&1"
done
mkdir -p "$RAW/prof" "$RAW/draft"
ssh -o BatchMode=yes $H "cd ~/intarsia-mtp2s/out/prof && tar cf - seg-*.txt iter-*.txt cmp-*.txt gen-*.txt metrics*-*.txt serve-*.log warm-*.txt 2>/dev/null" | tar xf - -C "$RAW/prof"
ssh -o BatchMode=yes $H "cd ~/intarsia-mtp2s/out/draft && gzip -c serve-d.log; " > "$RAW/draft/serve-d.log.gz"
ssh -o BatchMode=yes $H "cd ~/intarsia-mtp2s/out/draft && cat gen-d.txt" > "$RAW/draft/gen-d.txt"
for a in $ARMS; do rsh "gzip -c ~/intarsia-mtp2s/out/prof/$a.sqlite" > "$RAW/prof/$a.sqlite.gz"; done
rsh "free -g | sed -n 2p; df -h ~ | tail -1"
echo "== done"
