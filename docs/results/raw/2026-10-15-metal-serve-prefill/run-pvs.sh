#!/bin/zsh
# Plain `generate` (fresh process) vs serve-path 4k prefill, interleaved in time inside ONE fleetq slot.
# usage: run-pvs.sh LABEL BIN ITERS
# Per iteration: plain generate; sleep 5; server request after idle; server request back-to-back.
LABEL=$1; BIN=$2; ITERS=$3
W=/opt/code/intarsiaml/.claude/worktrees/agent-ad08dc8ef0ba4269b
OUT=$W/docs/results/raw/2026-10-15-metal-serve-prefill
MODEL=/Users/josef/models/Qwen3-8B
PROMPT=$OUT/../2026-10-14-metal-phase-b/prompt-4k.txt
export INTARSIA_CPU_FREE_MEM_GB=60
J=$(fleetq acquire --host endor --exclusive --mem 40 --expect 30 --label "metal serve-prefill $LABEL" --wait --json | python3 -c 'import sys,json; print(json.load(sys.stdin)["id"])')
echo "fleetq job $J" | tee -a $OUT/fleetq-jobs.txt
SPID=""
cleanup() { [[ -n "$SPID" ]] && kill $SPID 2>/dev/null; pkill -f "macmon pipe"; fleetq release $J; }
trap cleanup EXIT
export FLEETQ_JOB=$J
pgrep -fl zoom.us && echo "ZOOM RUNNING"
echo "date $(date -Iseconds) bin=$BIN" | tee $OUT/$LABEL.meta
$BIN serve --model $MODEL --backend metal --device 0 --host 127.0.0.1 --port 8093 --served-model-name intarsia-metal --max-seq 8192 --profile-every 0 --no-supervisor --no-ui > $OUT/$LABEL-server.out 2> $OUT/$LABEL-server.err &
SPID=$!
for i in {1..120}; do curl -sf http://127.0.0.1:8093/health >/dev/null 2>&1 && break; sleep 1; done
(macmon pipe -i 100 | python3 -I $OUT/macmon_log.py > $OUT/$LABEL-macmon.tsv) &
python3 -I $OUT/one.py 8093 intarsia-metal $PROMPT > /dev/null
python3 -I $OUT/one.py 8093 intarsia-metal $PROMPT > /dev/null
echo -e "iter\tkind\tprefill_ms\tttft_ms\tt0_epoch\twall" > $OUT/$LABEL.tsv
for i in {1..$ITERS}; do
  sleep 5
  t0=$(python3 -c 'import time; print("%.3f"%time.time())')
  $BIN generate --model $MODEL --backend metal --max-seq 8192 --temperature 0 --max-tokens 1 --profile-every 0 --prompt "$(cat $PROMPT)" > /dev/null 2> $OUT/$LABEL-gen-$i.err
  ms=$(grep -o 'prefill [0-9]* ms' $OUT/$LABEL-gen-$i.err | tail -1 | grep -o '[0-9]*')
  echo -e "$i\tplain\t$ms\t\t$t0\t$(date +%T)" >> $OUT/$LABEL.tsv
  sleep 5
  read a b c <<< "$(python3 -I $OUT/one.py 8093 intarsia-metal $PROMPT)"
  echo -e "$i\tserve_idle5\t$b\t$a\t$c\t$(date +%T)" >> $OUT/$LABEL.tsv
  read a b c <<< "$(python3 -I $OUT/one.py 8093 intarsia-metal $PROMPT)"
  echo -e "$i\tserve_b2b\t$b\t$a\t$c\t$(date +%T)" >> $OUT/$LABEL.tsv
  rm -f $OUT/$LABEL-gen-$i.err.tmp
done
pkill -f "macmon pipe"; kill $SPID; wait $SPID 2>/dev/null; SPID=""
echo done
