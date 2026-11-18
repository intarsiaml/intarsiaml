#!/bin/zsh
# mgdnvsoak speed A/B on the 27B (default recipe: MTP k2 from the recipe, no --spec flag), `serve --max-batch 1`; one round = both arms (order reversed on even rounds), REPS reps per cell.
#   env R=<round> scripts/mgdnvsoak/ab27.sh   (inside a fleetq slot)
source ${0:A:h}/lib.sh
O=$RAW/ab27; mkdir -p $O; BIN=${BIN:-$NEW_BIN}; r=${R:?R}; REPS=${REPS:-3}; CELLS=${CELLS:-"chat-short code-gen reason agent-edit"}; C=$W/scripts/mcopy/cell.py
stop() { [ -n "${TP:-}" ] && { curl -sf $URL/metrics > $1.metrics.txt 2>/dev/null; kill -INT $TP 2>/dev/null; for i in {1..60}; do kill -0 $TP 2>/dev/null || break; sleep 1; done; kill -9 $TP 2>/dev/null; wait $TP 2>/dev/null; TP=; }; }
trap 'stop $O/aborted' EXIT
order=(off:X=1 on:INTARSIA_METAL_GDN_VERIFY_FUSE=1); [ $((r % 2)) = 0 ] && order=(${(Oa)order})
for spec in $order; do
  a=${spec%%:*}; e=${spec#*:}
  echo "== round $r arm $a $(cool) $(date +%T)"
  env $e $BIN serve --backend metal --model $M27 --served-model-name m --max-batch 1 --max-seq 8192 --profile-every 0 --no-supervisor --no-ui --host 127.0.0.1 --port $PORT > $O/serve-$a-r$r.out 2> $O/serve-$a-r$r.err < /dev/null &
  TP=$!; ok=0
  for i in {1..400}; do curl -sf -m 2 $URL/health >/dev/null 2>&1 && { ok=1; break; }; kill -0 $TP 2>/dev/null || break; sleep 1; done
  [ $ok = 1 ] || { echo "SERVER FAILED $a: $(tail -3 $O/serve-$a-r$r.err)"; kill $TP 2>/dev/null; TP=; exit 1; }
  python3 $C $PORT $CELLS_DIR/chat-short.json 2 warm$r > /dev/null 2>&1
  for c in ${=CELLS}; do
    t0=$(th); python3 $C $PORT $CELLS_DIR/$c.json $REPS c$r --out $O/txt-$a-r$r >> $O/$a-r$r.jsonl 2>&1; echo "$a r$r $c thermal $t0->$(th): $(tail -1 $O/$a-r$r.jsonl | cut -c1-170)"
  done
  curl -s $URL/metrics | grep -E '^intarsia_(spec_)' | sed 's/{[^}]*}//' | tr '\n' ' ' > $O/metrics-$a-r$r.txt; echo "$a r$r metrics: $(cat $O/metrics-$a-r$r.txt)"
  stop $O/serve-$a-r$r
done
echo "ab27 done"
