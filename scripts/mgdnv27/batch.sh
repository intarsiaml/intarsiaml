#!/bin/zsh
# mgdnv27 batched 35B: `serve --max-batch MB --spec mtp --spec-k 2` with INTARSIA_BATCHED_MTP=$BMTP INTARSIA_BATCHED_VERIFY=1 (off by default on Metal; needed for batched MTP verify),
# BMTP (default N) = INTARSIA_BATCHED_MTP: sequences draft only while at most BMTP decode (the recipe value is 1 on Metal; 2 gave no drafting at 4 or 8 streams). N concurrent requests, knob off / on (order reversed on even rounds). One round per call: env MB=16 N=8 R=1 [SAMPLED=1] scripts/mgdnv27/batch.sh
source ${0:A:h}/lib.sh
BIN=${BIN:-$NEW_BIN}; MB=${MB:?MB}; N=${N:?N}; r=${R:?R}; MAXTOK=${MAXTOK:-300}; BMTP=${BMTP:-$N}; O=$RAW/batch${SAMPLED:+-sampled}-mb$MB-bmtp$BMTP; mkdir -p $O
X=(); [ -n "${SAMPLED:-}" ] && X=(--temp 0.7 --seed 1234)
stop() { [ -n "${TP:-}" ] && { curl -sf $URL/metrics > $1.metrics.txt 2>/dev/null; kill -INT $TP 2>/dev/null; for i in {1..60}; do kill -0 $TP 2>/dev/null || break; sleep 1; done; kill -9 $TP 2>/dev/null; wait $TP 2>/dev/null; TP=; }; }
trap 'stop $O/aborted' EXIT
order=(off:0 on:1); [ $((r % 2)) = 0 ] && order=(${(Oa)order})
for spec in $order; do
  a=${spec%%:*}; k=${spec#*:}
  echo "== mb$MB n$N round $r arm $a $(cool) $(date +%T)"
  env INTARSIA_BATCHED_MTP=$BMTP INTARSIA_BATCHED_VERIFY=1 INTARSIA_METAL_GDN_VERIFY_FUSE=$k $BIN serve --backend metal --model $M35 --served-model-name m --max-batch $MB --max-seq 8192 --spec mtp --spec-k 2 --spec-off-at 9 --profile-every 0 --no-supervisor --no-ui --host 127.0.0.1 --port $PORT > $O/serve-$a-r$r.out 2> $O/serve-$a-r$r.err < /dev/null &
  TP=$!; ok=0
  for i in {1..400}; do curl -sf -m 2 $URL/health >/dev/null 2>&1 && { ok=1; break; }; kill -0 $TP 2>/dev/null || break; sleep 1; done
  [ $ok = 1 ] || { echo "SERVER FAILED $a: $(tail -3 $O/serve-$a-r$r.err)"; kill $TP 2>/dev/null; TP=; exit 1; }
  python3 $S/conc.py $PORT $CELLS_DIR 2 warm$r $MAXTOK > /dev/null 2>&1
  for rep in 1 2; do python3 $S/conc.py $PORT $CELLS_DIR $N w$rep $MAXTOK $X >> $O/$a-r$r.jsonl 2>&1; echo "$a r$r rep$rep: $(tail -1 $O/$a-r$r.jsonl)"; done
  curl -s $URL/metrics | grep -E '^intarsia_(spec_|batch)' | sed 's/{[^}]*}//' | tr '\n' ' ' > $O/metrics-$a-r$r.txt; echo "$a r$r metrics: $(cat $O/metrics-$a-r$r.txt | cut -c1-400)"
  stop $O/serve-$a-r$r
done
echo "batch done"
