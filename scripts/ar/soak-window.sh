#!/usr/bin/env bash
# Lane ar: merge-gate stages (scripts/ar/stages-soak.sh) in one radiance-down window, driven FROM THE MAC:
#   FLEETQ_JOB=<id> W=g1 STAGES="kt" scripts/ar/soak-window.sh      (STAGES default "kt ref soak ids"; N, REF, REFN, DEFN set the process counts)
# Fetches ~/intarsiaml-ars/.ars/out/<W>/ into docs/results/2026-10-28-det-tp2-nondeterminism/raw/ars-<W>/ ; radiance restored by guard.sh.
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq}" "${W:?W=<window>}"
CAP=${CAP:-3000}
M27=/models/radiance/hf-cache/hub/models--amd--Qwen3.8-27B-Quark-AWQ-MXFP4/snapshots/5233554c5fa56afda40150556b95573c2d7d29c0
M35=/models/intarsia/Qwen3.6-35B-A3B-NVFP4
IMG=intarsia-hip-dev:rocm7.2.4-rust1.90.0
ORD=("d3 d1 d5" "d5 d1 d3" "d1 d5 d3" "d3 d5 d1")
# ARMS="d3 d1": two arms, order alternating per round
[[ -n ${ARMS:-} ]] && { rev=$(echo $ARMS | awk '{for (i=NF;i>=1;i--) printf "%s ", $i}'); ORD=("$ARMS" "$rev" "$ARMS" "$rev"); }
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; REPO="$(cd "$HERE/../.." && pwd)"
OUT=$REPO/docs/results/2026-10-28-det-tp2-nondeterminism/raw/ars-$W; mkdir -p "$OUT"
T0=$(date +%s); left() { echo $(( CAP - ($(date +%s) - T0) )); }
say() { echo "$(date +%T) $*" | tee -a "$OUT/steps.log"; }
SSH="ssh -o BatchMode=yes tatooine"
$SSH 'mkdir -p ~/intarsiaml-ars/.ars' < /dev/null; scp -q "$HERE/guard.sh" tatooine:intarsiaml-ars/.ars/
FIFO=$(mktemp -u "$REPO/.worktrees/ars-guard.XXXXXX" 2>/dev/null || mktemp -u "$OUT/guard.XXXXXX"); mkfifo "$FIFO"
$SSH 'bash ~/intarsiaml-ars/.ars/guard.sh' > "$OUT/guard.log" 2>&1 < "$FIFO" &
GUARD=$!
exec 9> "$FIFO"
for i in $(seq 1 30); do grep -q "radiance stopped" "$OUT/guard.log" 2>/dev/null && break; sleep 1; done
say "guard: $(grep -h 'radiance stopped' "$OUT/guard.log")"
cleanup() { $SSH 'docker rm -f ars-serve' >/dev/null 2>&1 < /dev/null; exec 9>&-; wait $GUARD; rm -f "$FIFO"; say "guard: $(grep -h 'radiance answers\|DID NOT' "$OUT/guard.log")"
  start=$(grep -m1 'radiance stopped' "$OUT/guard.log" | grep -oE '[0-9]{2}:[0-9]{2}:[0-9]{2}'); back=$(grep -m1 'radiance answers' "$OUT/guard.log" | grep -oE '[0-9]{2}:[0-9]{2}:[0-9]{2}' | head -1)
  echo "$(date +%F) fleetq #$FLEETQ_JOB ars-$W radiance stopped $start answers ${back:-NEVER} (guard.log in raw/ars-$W)" >> "$OUT/../downtime.log"; }
trap cleanup EXIT
scp -q "$HERE/stages-soak.sh" tatooine:intarsiaml-ars/.ars/
$SSH "rm -rf ~/intarsiaml-ars/.ars/out/$W; N=${N:-250} REF=${REF:-30} REFN=${REFN:-3} DEFN=${DEFN:-2} BIN=${BIN:-ars-intarsia} STAGES='${STAGES:-kt ref soak ids}' timeout $CAP bash ~/intarsiaml-ars/.ars/stages-soak.sh ~/intarsiaml-ars/.ars/out/$W" < /dev/null 2>&1 | tee "$OUT/stages.log"
rsync -a "tatooine:intarsiaml-ars/.ars/out/$W/" "$OUT/" 2>/dev/null
say "stages done"
