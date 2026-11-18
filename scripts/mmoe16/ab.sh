#!/bin/zsh
# mmoe16 gate (c): served A/B on Metal, `serve --max-batch 1` (single-sequence engine), arms differ by env only (same binary), fresh server per arm and round,
# the arm order reversed on even rounds (ABBA across rounds), cells in a fixed order, REPS reps each; thermal level logged before / after every cell.
#   fleetq run --host endor --quiet gpu,cpu --mem 40 --expect 20 --label "mmoe16: A/B round 1" -- env W=ab1 R0=1 ROUNDS=1 ARMS="base:X=1 n20:INTARSIA_SPEC_COPY=20" scripts/mmoe16/ab.sh
# ARMS "name:ENV=val[+ENV2=val]" (values may hold commas: INTARSIA_SPEC_COPY=20,15; an arm named `off` serves --spec off); CELLS (default: the seven of prompts.py);
# R0 = number of the first round (the order is reversed when the round number is even), ROUNDS (1), REPS (3).
TAG=${W:?W}; source ${0:A:h}/lib.sh
O=$RAW/$TAG; mkdir -p $O; BIN=${BIN:-$NEW_BIN}; R0=${R0:-1}; ROUNDS=${ROUNDS:-1}; REPS=${REPS:-3}
CELLS=${CELLS:-"copy3k agent-edit agent-batch json-xform chat-short reason code-gen"}; ARMS=${ARMS:?ARMS}; C=$W/scripts/mcopy/cell.py
stop() { [ -n "${TP:-}" ] && { curl -sf $URL/metrics > $1.metrics.txt 2>/dev/null; kill -INT $TP 2>/dev/null; for i in {1..60}; do kill -0 $TP 2>/dev/null || break; sleep 1; done; kill -9 $TP 2>/dev/null; wait $TP 2>/dev/null; gzip -f $1.err 2>/dev/null; TP=; }; }
trap 'stop $O/aborted' EXIT
for r in $(seq $R0 $((R0 + ROUNDS - 1))); do
  order=(${=ARMS}); [ $((r % 2)) = 0 ] && order=(${(Oa)order})
  for spec in $order; do
    a=${spec%%:*}; e=${spec#*:}; SA=(--spec mtp --spec-k 2); [ $a = off ] && SA=(--spec off)
    echo "== round $r arm $a env $e $(cool) $(date +%T)"
    env ${=e//+/ } $BIN serve --backend metal --model $M --served-model-name m --max-batch 1 --max-seq 8192 $SA --profile-every 0 --no-supervisor --no-ui \
      --host 127.0.0.1 --port $PORT > $O/serve-$a-r$r.out 2> $O/serve-$a-r$r.err < /dev/null &
    TP=$!; ok=0
    for i in {1..300}; do curl -sf -m 2 $URL/health >/dev/null 2>&1 && { ok=1; break; }; kill -0 $TP 2>/dev/null || break; sleep 1; done
    [ $ok = 1 ] || { echo "SERVER FAILED $a: $(tail -3 $O/serve-$a-r$r.err)"; kill $TP 2>/dev/null; TP=; exit 1; }
    python3 $C $PORT $CELLS_DIR/chat-short.json 2 warm$r > /dev/null 2>&1
    for c in ${=CELLS}; do
      t0=$(th); python3 $C $PORT $CELLS_DIR/$c.json $REPS c$r --out $O/txt-$a-r$r >> $O/$a-r$r.jsonl 2>&1; echo "$a r$r $c thermal $t0->$(th): $(tail -1 $O/$a-r$r.jsonl | cut -c1-170)"
    done
    curl -s $URL/metrics | grep -E '^intarsia_(spec_(steps|proposed|accepted|copy))' | sed 's/{[^}]*}//' | tr '\n' ' ' > $O/metrics-$a-r$r.txt; echo "$a r$r metrics: $(cat $O/metrics-$a-r$r.txt)"
    stop $O/serve-$a-r$r
  done
done
echo "ab done"
