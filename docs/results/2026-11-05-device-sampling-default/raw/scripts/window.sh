#!/usr/bin/env bash
# Lane lane-ds, spark1 window: served A/B of the new default (device top-k sampling) against INTARSIA_DEVICE_SAMPLING=0, one 35B NVFP4 server per
# visit (tp1, --max-batch 16), same binary (~/intarsiaml-ds/src/target/release/intarsia), order on off on off (ABAB, A = default).
# Per visit: free the page cache, start the server, equal.py (seeded requests, 7 settings x 1/4/16 streams; first visit of each arm only),
# localeval speed ds-sampled (1024-token prompts, 128 forced tokens, 4 and 16 streams), ds-nok (16 streams), stop the server.
#   fleetq run --host spark1 --quiet gpu,cpu --mem 46 --expect 70 --label "lane-ds: device sampling default, equality + ABAB" -- bash window.sh
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
H=$(cd "$(dirname "$0")" && pwd); OUT=${OUT:-$H/..}; mkdir -p "$OUT"
HOST=spark1; HN=spark-4e77; PORT=8197; LE=/opt/code/deepeval; export LOCALEVAL_TARGETS=$H/targets.toml
BIN='~/intarsiaml-ds/src/target/release/intarsia'
M35=/home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5
rsh() { ssh -n -o BatchMode=yes "$HOST" "$@"; }
SPID=""
stop_server() { [[ -n "$SPID" ]] && { rsh "kill $SPID 2>/dev/null; for i in \$(seq 1 30); do kill -0 $SPID 2>/dev/null || exit 0; sleep 1; done; kill -9 $SPID 2>/dev/null; sleep 2"; SPID=""; }; }
trap stop_server EXIT INT TERM
i=0
for arm in ${ARMS:-on off on off}; do
  i=$((i + 1)); v=$([ $arm = on ] && echo default || echo 0)
  echo "== visit $i $arm $(date +%T)"
  # evict the page cache of other models (the loader reads MemFree), as the earlier spark1 windows did
  rsh 'sync; sudo -n sh -c "echo 1 > /proc/sys/vm/drop_caches"' >/dev/null 2>&1 || true
  envs=""; [ $arm = off ] && envs="INTARSIA_DEVICE_SAMPLING=0"
  SPID=$(rsh "mkdir -p ~/intarsiaml-ds/out; nohup env $envs $BIN serve --backend cuda --model $M35 --port $PORT --host 0.0.0.0 --served-model-name intarsia --max-seq 8192 --mem-budget 40 --profile-every 0 --max-batch 16 > ~/intarsiaml-ds/out/serve-v$i-$arm.log 2>&1 < /dev/null & echo \$!")
  up=0; for _ in $(seq 1 150); do curl -sf "http://$HN:$PORT/health" >/dev/null && { up=1; break; }; sleep 2; done
  if [[ $up != 1 ]]; then echo "server visit $i did not come up"; rsh "tail -5 ~/intarsiaml-ds/out/serve-v$i-$arm.log"; stop_server; exit 1; fi
  echo "v$i-$arm: MemAvailable $(rsh "awk '/MemAvailable/ {print int(\$2/1024)}' /proc/meminfo") MB with the server up" | tee -a "$OUT/mem.txt"
  rsh "grep -ac 'device' ~/intarsiaml-ds/out/serve-v$i-$arm.log; head -c 0 /dev/null" > /dev/null
  if [ $i -le 2 ]; then python3 "$H/../../../../../scripts/htk/equal.py" run --url "http://$HN:$PORT/v1" --out "$OUT/ids-$arm.json" > "$OUT/ids-$arm.txt" 2>&1; tail -1 "$OUT/ids-$arm.txt"; fi
  for t in ds-sampled ds-nok; do
    conc=4,16; [ $t = ds-nok ] && conc=16
    (cd $LE && uv run localeval speed $t --thinking off --sizes 1024 --gen 128 --reps 2 --concurrency $conc --label ds-v$i-$arm-$t \
       --recipe "intarsia serve --backend cuda --model Qwen3.6-35B-A3B-NVFP4 --max-seq 8192 --mem-budget 40 --max-batch 16; INTARSIA_DEVICE_SAMPLING=$v" > "$OUT/v$i-$arm-$t.txt" 2>&1
     cp "$(ls -dt $LE/results/*ds-v$i-$arm-$t | head -1)/result.json" "$OUT/v$i-$arm-$t.json" 2>/dev/null)
    grep -aE 'streams: aggregate|aggregate' "$OUT/v$i-$arm-$t.txt" | tail -3
  done
  cp_log() { rsh "cat ~/intarsiaml-ds/out/serve-v$i-$arm.log" > "$OUT/serve-v$i-$arm.log" 2>/dev/null; }; cp_log
  stop_server
  echo "== visit $i done $(date +%T)"
done
python3 "$H/../../../../../scripts/htk/equal.py" compare "$OUT/ids-on.json" "$OUT/ids-off.json" | tee "$OUT/ids-compare.txt" | tail -3
