#!/usr/bin/env bash
# Lane hg: base (origin/main) against the packed-prefill build, same fleetq slot, with localeval. Runs on endor inside the tatooine
# job that holds the slot (FLEETQ_JOB set), radiance down. Arms: base, pk (knob unset), pk+pack (INTARSIA_PACK_PREFILL=1), visited
# base pk pkpack pkpack pk base. Per visit: 2 rounds of localeval speed (1k: 1,4,16 streams; 4k: 4 streams), then stall.py steady ITL at 1/4/16.
set -uo pipefail
H=$(cd "$(dirname "$0")" && pwd); OUT=${OUT:-$H/../../docs/results/2026-10-25-hg-hip-serve-gap/raw/ab}; mkdir -p "$OUT"
LE=/opt/code/deepeval; URL=http://tatooine:8294/v1
i=0
for arm in ${ARMS:-base pk pkpack pkpack pk base}; do
  i=$((i + 1))
  case $arm in base) tree=hg-base; env="";; pk) tree=hgpk; env="";; pkpack) tree=hgpk; env="INTARSIA_PACK_PREFILL=1";; esac
  ssh tatooine "bash ~/hg-win.sh up $tree v$i-$arm $env" < /dev/null | tail -1
  for r in 1 2; do
    (cd $LE && uv run localeval speed --url $URL --model intarsia --thinking off --sizes 1024 --gen 128 --reps 1 --concurrency 1,4,16 \
      --label hg-v$i-$arm-1k-r$r --no-unim > "$OUT/v$i-$arm-1k-r$r.txt" 2>&1; cp "$(ls -dt $LE/results/*hg-v$i-$arm-1k-r$r | head -1)/result.json" "$OUT/v$i-$arm-1k-r$r.json")
    (cd $LE && uv run localeval speed --url $URL --model intarsia --thinking off --sizes 4096 --gen 128 --reps 1 --concurrency 4 \
      --label hg-v$i-$arm-4k-r$r --no-unim > "$OUT/v$i-$arm-4k-r$r.txt" 2>&1; cp "$(ls -dt $LE/results/*hg-v$i-$arm-4k-r$r | head -1)/result.json" "$OUT/v$i-$arm-4k-r$r.json")
  done
  for n in 1 4 16; do python3 "$H/stall.py" --url $URL --streams $n --gen 640 --inject "" --out "$OUT/v$i-$arm-itl$n.json" > /dev/null; done
  ssh tatooine "bash ~/hg-win.sh down v$i-$arm" < /dev/null | tail -1
  echo "visit $i $arm done $(date +%T)"
done
