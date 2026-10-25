#!/usr/bin/env bash
# Lane hg: steady-state decode of the base (origin/main) and dev (branch) serve builds, 1 / 4 / 16 streams, ABBA order.
# Runs on endor inside the tatooine fleetq job that took radiance down; each arm is a fresh `win.sh up`.
set -uo pipefail
H=$(cd "$(dirname "$0")" && pwd); OUT=${OUT:-$H/../../docs/results/2026-10-25-hg-hip-serve-gap/raw/steady}; mkdir -p "$OUT"
i=0
for arm in hg-base hg hg hg-base; do
  i=$((i + 1))
  ssh tatooine "bash ~/hg-win.sh up $arm steady$i" < /dev/null | tail -1
  for n in 1 4 16; do
    python3 "$H/stall.py" --url http://tatooine:8294/v1 --streams $n --gen 256 --inject "" --out "$OUT/$i-$arm-$n.json" > /dev/null
    python3 -c "import json;d=json.load(open('$OUT/$i-$arm-$n.json'));print('$i $arm streams $n steady', d['steady_agg_tok_s'], 'tok/s, ITL p50', d['steady_itl_ms']['p50'], 'p99', d['steady_itl_ms']['p99'])"
  done
  ssh tatooine "bash ~/hg-win.sh down steady$i" < /dev/null | tail -1
done
