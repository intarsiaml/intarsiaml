#!/usr/bin/env bash
# Lane htk GPU window 2, run on endor inside a fleetq job on tatooine that takes radiance down (<= 30 min): the served A/B of
# INTARSIA_DEVICE_SAMPLING=0 (host sampling) against =1 (device top-k) on one 35B server (tp1, --max-batch 16), same binary.
# Visits: off on [on off, if time allows]. Per visit: equal.py (first off and first on only; token ids of seeded requests), localeval speed
# sampled at 1 stream (size sweep) and 4, 16 streams, and a greedy 1-stream sweep. Radiance is restarted at the end whatever happens.
#   fleetq run --host tatooine --takes-down radiance --mem 30 --quiet gpu,cpu --expect 28 --label "HTK window 2" -- bash scripts/htk/window2.sh
set -uo pipefail
H=$(cd "$(dirname "$0")" && pwd); OUT=${OUT:-$H/../../docs/results/2026-10-27-htk-hip-topk/raw}; mkdir -p "$OUT"
LE=/opt/code/deepeval; export LOCALEVAL_TARGETS=$H/targets.toml
T0=$(date +%s); el() { echo $(( $(date +%s) - T0 )); }
trap 'ssh tatooine "bash ~/intarsiaml-htk/src/scripts/htk/win.sh down w2-last; bash ~/intarsiaml-htk/src/scripts/htk/win.sh restore" < /dev/null' EXIT
i=0
for arm in ${ARMS:-off on on off}; do
  i=$((i + 1))
  if [ $i -ge 3 ] && [ "$(el)" -gt "${LATE:-840}" ]; then echo "[w2] t+$(el) s: skipping visit $i $arm (late)"; continue; fi
  v=$([ $arm = on ] && echo 1 || echo 0)
  ssh tatooine "bash ~/intarsiaml-htk/src/scripts/htk/win.sh up v$i-$arm INTARSIA_DEVICE_SAMPLING=$v" < /dev/null | tail -2
  if [ $i -le 2 ]; then python3 "$H/equal.py" run --url http://tatooine:8296/v1 --out "$OUT/ids-$arm.json" > "$OUT/ids-$arm.txt" 2>&1; tail -1 "$OUT/ids-$arm.txt"; fi
  for t in htk-sampled htk-greedy; do
    extra=""; [ $t = htk-sampled ] && extra="--concurrency 4,16"
    (cd $LE && uv run localeval speed $t --thinking off --sizes 1024 --gen 128 --reps 2 $extra --label htk-v$i-$arm-$t --no-unim \
       --recipe "intarsia serve --backend hip --model Qwen3.6-35B-A3B-NVFP4 --max-seq 8192 --max-batch 16; INTARSIA_DEVICE_SAMPLING=$v" > "$OUT/v$i-$arm-$t.txt" 2>&1
     cp "$(ls -dt $LE/results/*htk-v$i-$arm-$t | head -1)/result.json" "$OUT/v$i-$arm-$t.json")
  done
  ssh tatooine "bash ~/intarsiaml-htk/src/scripts/htk/win.sh down v$i-$arm" < /dev/null | tail -1
  echo "[w2] visit $i $arm done t+$(el) s"
done
python3 "$H/equal.py" compare "$OUT/ids-off.json" "$OUT/ids-on.json" | tee "$OUT/ids-compare.txt" | tail -3
