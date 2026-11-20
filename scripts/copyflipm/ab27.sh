#!/bin/zsh
# copyflipm served speed A/B on the Qwen3.8-27B (Metal): `serve --max-batch 1` WITHOUT --spec (the recipe's MTP default applies), arms differ by env only
# (same binary, the #460 build): `def` = INTARSIA_SPEC_COPY unset (copy drafts on by default), `zero` = INTARSIA_SPEC_COPY=0. A fresh server per arm and round,
# the arm order reversed on even rounds (ABBA across rounds), cells in a fixed order, REPS reps each; thermal level logged before / after every cell.
# Harness of scripts/copymetal/ab.sh. Inside a fleetq slot on endor (see round27.sh).
# R0 = number of the first round (the order is reversed when the round number is even), ROUNDS (1), REPS (3), CELLS (default: the seven of prompts.py).
[ -n "${FLEETQ_JOB:-}" ] || { echo "not inside a fleetq slot" >&2; exit 2; }
W=${0:A:h:h:h}; RAW=$W/docs/results/2026-11-18-copyflipm/raw; BIN=$W/target-cf/release/intarsia; CELLS_DIR=$W/target-cf/cells
M=$HOME/models/Qwen3.8-27B-Quark-AWQ-MXFP4; PORT=${PORT:-18727}; URL=http://127.0.0.1:$PORT
th() { notifyutil -g com.apple.system.thermalpressurelevel 2>&1 | awk '{print $2}'; }
cool() { local w=0; while [ "$(th)" != 0 ] && [ $w -lt ${COOL:-120} ]; do sleep 5; w=$((w+5)); done; echo "cooled ${w}s thermal $(th)"; }
mkdir -p $CELLS_DIR; [ -f $CELLS_DIR/copy3k.json ] || python3 $W/scripts/mcopy/prompts.py $CELLS_DIR > /dev/null
O=$RAW/ab27; mkdir -p $O; R0=${R0:-1}; ROUNDS=${ROUNDS:-1}; REPS=${REPS:-3}
CELLS=${CELLS:-"copy3k agent-edit agent-batch json-xform chat-short reason code-gen"}; ARMS=${ARMS:-"zero:INTARSIA_SPEC_COPY=0 def:X=1"}; C=$W/scripts/mcopy/cell.py
stop() { [ -n "${TP:-}" ] && { curl -sf $URL/metrics > $1.metrics.txt 2>/dev/null; kill -INT $TP 2>/dev/null; for i in {1..60}; do kill -0 $TP 2>/dev/null || break; sleep 1; done; kill -9 $TP 2>/dev/null; wait $TP 2>/dev/null; gzip -f $1.err 2>/dev/null; TP=; }; }
trap 'stop $O/aborted' EXIT
for r in $(seq $R0 $((R0 + ROUNDS - 1))); do
  order=(${=ARMS}); [ $((r % 2)) = 0 ] && order=(${(Oa)order})
  for spec in $order; do
    a=${spec%%:*}; e=${spec#*:}
    echo "== round $r arm $a env $e $(cool) $(date +%T)"
    env ${=e//+/ } $BIN serve --backend metal --model $M --served-model-name m --max-batch 1 --max-seq 8192 --profile-every 0 --no-supervisor --no-ui \
      --host 127.0.0.1 --port $PORT > $O/serve-$a-r$r.out 2> $O/serve-$a-r$r.err < /dev/null &
    TP=$!; ok=0
    for i in {1..300}; do curl -sf -m 2 $URL/health >/dev/null 2>&1 && { ok=1; break; }; kill -0 $TP 2>/dev/null || break; sleep 1; done
    [ $ok = 1 ] || { echo "SERVER FAILED $a: $(tail -3 $O/serve-$a-r$r.err)"; kill $TP 2>/dev/null; TP=; exit 1; }
    echo "$a r$r startup: $(grep -aE 'speculative decoding|copy drafts' $O/serve-$a-r$r.err | cut -c1-120 | tr '\n' '|')"
    python3 $C $PORT $CELLS_DIR/chat-short.json 2 warm$r > /dev/null 2>&1
    for c in ${=CELLS}; do
      t0=$(th); python3 $C $PORT $CELLS_DIR/$c.json $REPS c$r --out $O/txt-$a-r$r >> $O/$a-r$r.jsonl 2>&1; echo "$a r$r $c thermal $t0->$(th): $(tail -1 $O/$a-r$r.jsonl | cut -c1-170)"
    done
    curl -s $URL/metrics | grep -E '^intarsia_(spec_(steps|proposed|accepted|copy))' | sed 's/{[^}]*}//' | tr '\n' ' ' > $O/metrics-$a-r$r.txt; echo "$a r$r metrics: $(cat $O/metrics-$a-r$r.txt)"
    stop $O/serve-$a-r$r
  done
done
echo "ab done"
