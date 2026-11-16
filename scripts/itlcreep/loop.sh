#!/usr/bin/env bash
# itlcreep: replay loop, run on whichever machine is the client (the Spark itself, or the Mac). Env:
#   TAG URL SEED0 MIN OUT PLAN SCRIPT PY (command prefix, e.g. "uv run python") [ONLY=0|1 worker] [IGNORE=1] [LOCALEVAL_SRC]
# One replay = 8 conversations x 6 turns at K=2 (scripts/soak2 traffic: mixed greedy / T=1.0), fresh seed each, no new replay after MIN minutes.
set -uo pipefail
: "${TAG:?}" "${URL:?}" "${SEED0:?}" "${MIN:?}" "${OUT:?}" "${PLAN:?}" "${SCRIPT:?}" "${PY:?}"
mkdir -p "$OUT"
end=$(( $(date +%s) + MIN*60 )); seed=$SEED0; fails=0
echo "[loop $TAG] start $(date +%T) end $(date -r $end +%T 2>/dev/null || date -d @$end +%T) url $URL only=${ONLY:-both} ignore=${IGNORE:-0}"
while (( $(date +%s) < end )); do
  perl -e 'alarm 1500; exec @ARGV' $PY "$SCRIPT" --url "$URL" --model soak2 --arm "$TAG$seed" --order interleaved --seed "$seed" --streams 2 \
    --plan "$PLAN" --trace "$OUT/trace-$TAG$seed.json" --out "$OUT" --temp-mode mixed ${ONLY:+--only-worker $ONLY} ${IGNORE:+--ignore-busy} > "$OUT/$TAG$seed.out" 2>&1
  if [[ ! -s $OUT/$TAG$seed-interleaved-seed$seed-k2.summary.json ]]; then fails=$((fails+1)); echo "[loop $TAG] seed $seed: no summary ($fails)"; (( fails >= 2 )) && { echo "[loop $TAG] ABORT"; exit 3; }
  else fails=0; echo "[loop $TAG] seed $seed done $(date +%T): $(grep -h 'totals' "$OUT/$TAG$seed.out" | tail -1 | cut -c1-200)"; fi
  seed=$((seed+1))
done
echo "[loop $TAG] done $(date +%T), $((seed-SEED0)) replays"
