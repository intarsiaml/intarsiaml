#!/usr/bin/env bash
# Lane mtp-snap-gpu: prefill / decode speed, main vs PR #395, Qwen3.6-35B-A3B-NVFP4 on one R9700, `serve --spec mtp` (macro prefill 8192, chunk 4096: recipe defaults).
# Driven FROM THE MAC under one fleetq job that takes radiance down:
#   fleetq run --host tatooine --takes-down radiance --expect 40 --mem 30 --quiet gpu,cpu --label "intarsiaml mtp-snap-gpu: speed" -- scripts/mtp-snap-gpu/speed-window.sh
# Rounds interleaved (main pr / pr main / main pr), a fresh server per arm, `localeval speed` over SIZES, then the server goes away. localeval runs inside this
# job's slot (FLEETQ_JOB is inherited). CAP: seconds; arms that no longer fit are skipped.
set -uo pipefail
SIZES=${SIZES:-1024,4096,16384}; ROUNDS=${ROUNDS:-3}; GEN=${GEN:-64}; CAP=${CAP:-2400}; REPS=${REPS:-1}; ARMS=${ARMS:-"main pr"}
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; REPO="$(cd "$HERE/../.." && pwd)"
OUT=${OUT:-$REPO/docs/results/2026-11-16-mtp-snap/raw/gpu/speed-$(date +%H%M)}
. "$HERE/win.sh"
EXTRA=""; SENV=""
for r in $(seq 1 "$ROUNDS"); do
  order="$ARMS"
  (( r % 2 == 0 )) && order=$(echo $ARMS | awk '{for (i=NF;i>=1;i--) printf "%s ", $i}')
  for arm in $order; do
    tag="$arm-r$r"
    if (( $(left) < 330 )); then say "SKIPPED round $r arm $arm (time left $(left)s)"; continue; fi
    if ! serve "$arm" "$tag"; then say "round $r arm $arm: server did not come up"; drop; continue; fi
    [[ $r == 1 ]] && curl -s -m 10 "http://tatooine:$PORT/v1/intarsia/plan" > "$OUT/plan-$tag.json" 2>&1 < /dev/null
    RECIPE="HIP_VISIBLE_DEVICES=0 intarsia serve --backend hip --model $MODEL --served-model-name $NAME --spec mtp --profile-every 0 --mem-budget 29 --max-seq $MAXSEQ --no-supervisor (tree mtpsnap-gpu, bin $arm, docker $IMG)"
    ( cd /opt/code/deepeval && uv run localeval speed --url "http://tatooine:$PORT/v1" --model "$NAME" --label "mtpsnap-$arm-r$r" \
        --sizes "$SIZES" --gen "$GEN" --reps "$REPS" --thinking off --wait-idle 60 --recipe "$RECIPE" ) \
      > "$OUT/localeval-$arm-r$r.log" 2>&1 < /dev/null
    say "round $r arm $arm: localeval exit $? $(grep -aE 'run_id|results/' "$OUT/localeval-$arm-r$r.log" | tail -2 | tr '\n' '|')"
    curl -s -m 10 "http://tatooine:$PORT/metrics" 2>/dev/null < /dev/null | grep -aE '^intarsia_(spec_(steps|proposed_tokens|accepted_tokens)_total|tokens_cached_total)' > "$OUT/metrics-$arm-r$r.txt"
    drop
  done
done
say "done, time left $(left)s"
