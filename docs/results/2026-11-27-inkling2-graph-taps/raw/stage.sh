#!/usr/bin/env bash
# Lane ink-plain-tax, ON tatooine inside window.sh, card 1 only: the HIP tiny test of the decode-graph key, on the test-only build (red:
# 418c122bd's code, expected to FAIL) and on the fix (green, expected to PASS); then the whole inkling2_dflash2 suite on green.
set -uo pipefail
REPO=$HOME/intarsiaml-ink-plain-tax
cd "$REPO"
export INTARSIA_REPO=$REPO INTARSIA_TARGET_VOL=intarsia-cargo-target-ink-plain-tax INTARSIA_CONTAINER_NAME=ink-plain-tax-run HIP_VISIBLE_DEVICES=1
DEV="$REPO/deploy/tatooine/dev.sh"
T=decode_graph_recorded_without_drafting_is_not_replayed_by_a_drafting_state
O=$REPO/logs/stage; mkdir -p "$O"
for b in red green; do
  echo "[stage] $b $(cat .scratch/bin/t-$b.commit) $(md5sum .scratch/bin/t-$b | cut -c1-32) $(date +%T)"
  timeout 90 $DEV env INTARSIA_INKLING2_BACKEND=hip /work/.scratch/bin/t-$b --exact $T --nocapture --test-threads=1 < /dev/null > "$O/$b.txt" 2>&1
  rc=$?
  echo "[stage] $b rc $rc | $(grep -aE '^test result' "$O/$b.txt" | tail -1) | skipped: $(grep -ac 'skipped' "$O/$b.txt")"
  grep -aE 'graphs enabled|panicked|is not the|vacuous' "$O/$b.txt" | cut -c1-300
done
echo "[stage] green full inkling2_dflash2 suite on hip $(date +%T)"
timeout 120 $DEV env INTARSIA_INKLING2_BACKEND=hip /work/.scratch/bin/t-green --test-threads=1 < /dev/null > "$O/green-all.txt" 2>&1
echo "[stage] green-all rc $? | $(grep -aE '^test result' "$O/green-all.txt" | tail -1)"
grep -aE '^test .* FAILED|panicked' "$O/green-all.txt" | head -10 | cut -c1-300
echo "[stage] end $(date +%T)"
