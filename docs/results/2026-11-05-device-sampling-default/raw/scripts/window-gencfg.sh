#!/usr/bin/env bash
# Lane lane-ds, spark1 window 2: served A/B of the checkpoint sampling defaults (generation_config.json: temperature 1.0, top_p 0.95, top_k 20) against `--generation-config none`, one 35B NVFP4 server per
# visit (tp1, --max-batch 16), one binary built here from the worktree, order auto none auto none (ABAB, A = default).
# Per visit: free the page cache, start the server, equal.py (first visit of each arm only), localeval speed ds-nok (requests that send temperature 1.0 and
# no top_p / top_k; 1024-token prompts, 128 forced tokens, 4 and 16 streams), stop the server.
#   fleetq run --host spark1 --quiet gpu,cpu --mem 46 --expect 70 --label "lane-ds: generation-config defaults ABAB" -- bash window-gencfg.sh
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
bash "$H/build.sh" 2>&1 | tail -4 || exit 1
i=0
for arm in ${ARMS:-on off on off}; do
  i=$((i + 1)); v=$([ $arm = on ] && echo auto || echo none)
  echo "== visit $i $arm $(date +%T)"
  # evict the page cache of other models (the loader reads MemFree), as the earlier spark1 windows did
  rsh 'sync; sudo -n sh -c "echo 1 > /proc/sys/vm/drop_caches"' >/dev/null 2>&1 || true
  envs=""; gc=""; [ $arm = off ] && gc="--generation-config none"
  SPID=$(rsh "mkdir -p ~/intarsiaml-ds/out; nohup env $envs $BIN serve --backend cuda --model $M35 --port $PORT --host 0.0.0.0 --served-model-name intarsia --max-seq 8192 --mem-budget 40 --profile-every 0 --max-batch 16 $gc > ~/intarsiaml-ds/out/serve-v$i-$arm.log 2>&1 < /dev/null & echo \$!")
  up=0; for _ in $(seq 1 150); do curl -sf "http://$HN:$PORT/health" >/dev/null && { up=1; break; }; sleep 2; done
  if [[ $up != 1 ]]; then echo "server visit $i did not come up"; rsh "tail -5 ~/intarsiaml-ds/out/serve-v$i-$arm.log"; stop_server; exit 1; fi
  echo "v$i-$arm: MemAvailable $(rsh "awk '/MemAvailable/ {print int(\$2/1024)}' /proc/meminfo") MB with the server up" | tee -a "$OUT/gc-mem.txt"
  rsh "grep -a 'sampling defaults' ~/intarsiaml-ds/out/serve-v$i-$arm.log" | tee -a "$OUT/gencfg-startup.txt"
  curl -s "http://$HN:$PORT/metrics" | grep -a "^intarsia_sampling_defaults" | tee -a "$OUT/gencfg-startup.txt"
  if [ $i -le 2 ]; then python3 "$H/../../../../../scripts/htk/equal.py" run --url "http://$HN:$PORT/v1" --out "$OUT/gc-ids-$arm.json" > "$OUT/gc-ids-$arm.txt" 2>&1; tail -1 "$OUT/gc-ids-$arm.txt"; fi
  for t in ds-nok; do
    conc=4,16
    (cd $LE && uv run localeval speed $t --thinking off --sizes 1024 --gen 128 --reps 2 --concurrency $conc --label gc-v$i-$arm-$t \
       --recipe "intarsia serve --backend cuda --model Qwen3.6-35B-A3B-NVFP4 --max-seq 8192 --mem-budget 40 --max-batch 16; --generation-config $v" > "$OUT/gc-v$i-$arm-$t.txt" 2>&1
     cp "$(ls -dt $LE/results/*gc-v$i-$arm-$t | head -1)/result.json" "$OUT/gc-v$i-$arm-$t.json" 2>/dev/null)
    grep -aE 'streams: aggregate|aggregate' "$OUT/gc-v$i-$arm-$t.txt" | tail -3
  done
  cp_log() { rsh "cat ~/intarsiaml-ds/out/serve-v$i-$arm.log" > "$OUT/gc-serve-v$i-$arm.log" 2>/dev/null; }; cp_log
  stop_server
  echo "== visit $i done $(date +%T)"
done
python3 "$H/../../../../../scripts/htk/equal.py" compare "$OUT/gc-ids-on.json" "$OUT/gc-ids-off.json" | tee "$OUT/gc-ids-compare.txt" | tail -3
