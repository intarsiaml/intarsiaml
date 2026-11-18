#!/bin/zsh
# copyflipm memory cost of the default, inside a fleetq slot on endor. The default reserves 15 per-row recurrent-state buffers instead of 2
# (and runs 16-row verify passes). (1) a 600-token copy3k `generate` in which chains DO fire: peak memory footprint, default vs INTARSIA_SPEC_COPY=0;
# (2) `serve --max-batch 1`: the KV / state bytes and token capacity the server reports in /metrics, and its resident size, default vs =0.
[ -n "${FLEETQ_JOB:-}" ] || { echo "not inside a fleetq slot" >&2; exit 2; }
W=${0:A:h:h:h}; RAW=$W/docs/results/2026-11-18-copyflipm/raw/mem; mkdir -p $RAW; rm -f $RAW/*
NEW=$W/target-cf/release/intarsia; M=$HOME/models/Qwen3.6-35B-A3B-NVFP4; CELLS=$W/target-cf/cells; PORT=8398
for r in 1 2; do for arm in zero def; do
  E=(X=1); [ $arm = zero ] && E=(INTARSIA_SPEC_COPY=0)
  env $E INTARSIA_SPEC_TRACE=1 /usr/bin/time -l $NEW generate -m $M --backend metal --max-seq 8192 --spec mtp --spec-k 2 --chat --no-think --temperature 0 --profile-every 0 -p "$(cat $CELLS/copy3k.txt)" --max-tokens 600 < /dev/null > /dev/null 2> $RAW/gen-$arm-$r.err
  echo "generate $arm r$r: chains $(grep -ac 'spec: copy pass' $RAW/gen-$arm-$r.err); $(grep -aE 'maximum resident|peak memory footprint' $RAW/gen-$arm-$r.err | tr -s ' ' | tr '\n' ';')"
done; done
for arm in zero def; do
  E=(X=1); [ $arm = zero ] && E=(INTARSIA_SPEC_COPY=0)
  env $E $NEW serve --backend metal --model $M --model-id m --max-batch 1 --max-seq 8192 --spec mtp --spec-k 2 --profile-every 0 --no-supervisor --no-ui --host 127.0.0.1 --port $PORT > $RAW/serve-$arm.log 2>&1 < /dev/null &
  pid=$!; up=0
  for i in $(seq 1 100); do kill -0 $pid 2>/dev/null || break; curl -sf -m 3 http://127.0.0.1:$PORT/health > /dev/null 2>&1 && { up=1; break; }; sleep 3; done
  if [ $up = 1 ]; then curl -s http://127.0.0.1:$PORT/metrics | grep -aE '^intarsia_(kv_|.*memory|.*bytes)' | sed 's/{[^}]*}//' > $RAW/metrics-$arm.txt
    echo "serve $arm: rss $(ps -o rss= -p $pid | tr -d ' ') KiB; $(grep -aE 'kv_(bytes_total|tokens_capacity)' $RAW/metrics-$arm.txt | tr '\n' ';')"; else echo "serve $arm: did not come up"; fi
  kill $pid 2>/dev/null; wait $pid 2>/dev/null
done
diff $RAW/metrics-zero.txt $RAW/metrics-def.txt > $RAW/metrics.diff && echo "metrics: identical" || { echo "metrics differ:"; head -10 $RAW/metrics.diff; }
