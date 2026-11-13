#!/usr/bin/env bash
# ab35c shared helpers (sourced): ab35b's helpers with this lane's output dir and tool smoke (adds --thinking default).
REPO="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
OUT=$REPO/docs/results/2026-11-13-ab35c-tools-store-sampled/raw; mkdir -p "$OUT"
S=$REPO/scripts/ab35b; SC=$REPO/scripts/ab35c
HN=spark-fa00
# replay <arm> <url> <order> <seed>: ab35b multi-turn replay (8 conversations x 6 turns), plan/trace shared by seed
replay() {
  echo "== $1 replay $3 seed $4 $(date +%T)"
  ( cd /opt/code/deepeval && uv run python "$S/replay.py" --url "$2" --model guidance --arm "$1" --order "$3" --seed "$4" \
      --plan "$OUT/replay-plan.json" --trace "$OUT/replay-trace-seed$4.json" --out "$OUT" ) > "$OUT/$1-replay-$3-seed$4.out" 2>&1
  tail -12 "$OUT/$1-replay-$3-seed$4.out"
}
sampled() { # sampled <arm> <url> [metrics-url] [sizes]
  echo "== $1 sampled $(date +%T)"
  ( cd /opt/code/deepeval && uv run python "$S/sampled.py" --url "$2" --model guidance --arm "$1" --out "$OUT" ${3:+--metrics "$3"} ${4:+--sizes "$4"} ) > "$OUT/$1-sampled.out" 2>&1
  tail -6 "$OUT/$1-sampled.out"
}
tools() { # tools <arm> <url> <off|default> [reps] [stream-reps]
  echo "== $1 tools thinking=$3 $(date +%T)"
  local sfx=""; [[ $3 == default ]] && sfx="-think"
  python3 "$SC/toolsmoke.py" --url "$2" --model guidance --arm "$1" --thinking "$3" --reps "${4:-5}" --stream-reps "${5:-2}" --out "$OUT" > "$OUT/$1-tools$sfx.out" 2>&1
  tail -6 "$OUT/$1-tools$sfx.out"
}
# replay2 <arm> <url> <order> <seed> [streams=2]: K-concurrent-stream replay (ab35c/replay2.py), same plan/trace as replay()
replay2() {
  local k="${5:-2}"
  echo "== $1 replay2 k=$k $3 seed $4 $(date +%T)"
  ( cd /opt/code/deepeval && uv run python "$SC/replay2.py" --url "$2" --model guidance --arm "$1" --order "$3" --seed "$4" --streams "$k" \
      --plan "$OUT/replay-plan.json" --trace "$OUT/replay-trace-seed$4.json" --out "$OUT" ) > "$OUT/$1-replay2-$3-seed$4.out" 2>&1
  tail -14 "$OUT/$1-replay2-$3-seed$4.out"
}
