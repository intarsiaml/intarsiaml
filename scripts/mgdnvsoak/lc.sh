#!/bin/zsh
# mgdnvsoak run 2: long context, served `--max-batch 1` at the model's default recipe (greedy, thinking off), one server per arm; per arm the 16K then the 32K prompt, 1024 tokens.
#   env MODEL=<dir> MT=27b|35b R=<round> [CELLS="long16k long32k"] scripts/mgdnvsoak/lc.sh   (inside a fleetq slot); arm order reversed on even rounds
source ${0:A:h}/lib.sh
O=$RAW/lc-${MT:?MT}; mkdir -p $O; BIN=${BIN:-$NEW_BIN}; MODEL=${MODEL:?MODEL}; r=${R:?R}; C=$S/cell2.py; CELLS=${CELLS:-"long16k long32k"}
[ -f $CELLS_DIR/long16k.json ] || python3 $S/longprompts.py $CELLS_DIR ${CH16:-52000} ${CH32:-104000}
stop() { [ -n "${TP:-}" ] && { curl -sf $URL/metrics > $1.metrics.txt 2>/dev/null; kill -INT $TP 2>/dev/null; for i in {1..60}; do kill -0 $TP 2>/dev/null || break; sleep 1; done; kill -9 $TP 2>/dev/null; wait $TP 2>/dev/null; TP=; }; }
trap 'stop $O/aborted' EXIT
order=(off:X=1 on:INTARSIA_METAL_GDN_VERIFY_FUSE=1); [ $((r % 2)) = 0 ] && order=(${(Oa)order})
for spec in $order; do
  a=${spec%%:*}; e=${spec#*:}
  echo "== $MT round $r arm $a $(cool) $(date +%T)"
  env $e $BIN serve --backend metal --model $MODEL --served-model-name m --max-batch 1 --max-seq 36864 --profile-every 0 --no-supervisor --no-ui --host 127.0.0.1 --port $PORT > $O/serve-$a-r$r.out 2> $O/serve-$a-r$r.err < /dev/null &
  TP=$!; ok=0
  for i in {1..400}; do curl -sf -m 2 $URL/health >/dev/null 2>&1 && { ok=1; break; }; kill -0 $TP 2>/dev/null || break; sleep 1; done
  [ $ok = 1 ] || { echo "SERVER FAILED $a: $(tail -3 $O/serve-$a-r$r.err)"; kill $TP 2>/dev/null; TP=; exit 1; }
  python3 $S/cell2.py $PORT $CELLS_DIR/chat-short.json 1 warm$r > /dev/null 2>&1
  for c in ${=CELLS}; do
    t0=$(th); python3 $C $PORT $CELLS_DIR/$c.json 1 c$r --out $O/txt-$a-r$r >> $O/$a-r$r.jsonl 2>&1; echo "$a r$r $c thermal $t0->$(th): $(tail -1 $O/$a-r$r.jsonl | cut -c1-200)"
  done
  curl -s $URL/metrics | grep -E '^intarsia_(spec_)' | sed 's/{[^}]*}//' | tr '\n' ' ' > $O/metrics-$a-r$r.txt; echo "$a r$r metrics: $(cat $O/metrics-$a-r$r.txt)"
  stop $O/serve-$a-r$r
done
echo "lc done"
