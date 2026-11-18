#!/bin/zsh
# run 4: continuous mixed load (greedy + sampled, thinking on/off, 8 clients) on the 27B `serve --max-batch 16` batched MTP, knob ON, DUR seconds, RSS / footprint every 60 s. Inside a fleetq slot.
source ${0:A:h}/lib.sh
O=$RAW/soak; mkdir -p $O; DUR=${DUR:-1200}; NC=${NC:-8}
stop() { [ -n "${TP:-}" ] && { curl -sf $URL/metrics > $1.metrics.txt 2>/dev/null; kill -INT $TP 2>/dev/null; for i in {1..60}; do kill -0 $TP 2>/dev/null || break; sleep 1; done; kill -9 $TP 2>/dev/null; wait $TP 2>/dev/null; TP=; }; }
trap 'stop $O/aborted' EXIT
echo "== soak start $(date +%T) thermal $(th)"
env INTARSIA_BATCHED_MTP=$NC INTARSIA_BATCHED_VERIFY=1 INTARSIA_METAL_GDN_VERIFY_FUSE=1 $NEW_BIN serve --backend metal --model $M27 --served-model-name m --max-batch 16 --max-seq 8192 --spec mtp --spec-k 2 --spec-off-at 9 --profile-every 0 --no-supervisor --no-ui --host 127.0.0.1 --port $PORT > $O/serve.out 2> $O/serve.err < /dev/null &
TP=$!; ok=0
for i in {1..400}; do curl -sf -m 2 $URL/health >/dev/null 2>&1 && { ok=1; break; }; kill -0 $TP 2>/dev/null || break; sleep 1; done
[ $ok = 1 ] || { echo "SERVER FAILED: $(tail -3 $O/serve.err)"; exit 1; }
zsh $S/mon.sh $TP $O/mon.tsv & MP=$!
python3 $S/soak.py $PORT $CELLS_DIR $DUR $NC $O/soak.jsonl
kill $MP 2>/dev/null
curl -s $URL/metrics | grep -E '^intarsia_(spec_|batch)' | sed 's/{[^}]*}//' > $O/metrics-end.txt
kill -0 $TP 2>/dev/null && echo "server alive at end" || echo "SERVER DIED"
stop $O/serve
echo "soak finished $(date +%T)"
