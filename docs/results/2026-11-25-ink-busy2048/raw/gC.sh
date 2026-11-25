#!/bin/bash
# ink-busy2048 gate C (quiet gpu,cpu): the section 48 A/B, 256 vs 2048, with `--max-batch 16` on the new binary (tree = busy2048 HEAD) for every arm.
# Arm 0 is a discarded warm-up (2048 default). Then ABBA: 256 (--busy-chunk 256), 2048 (no flag: the recipe default), 2048, 256.
# Per arm: own server, 2 warm-up requests, /metrics before / after, localeval conc-plain-1k (1 / 4 / 16 streams, 3 reps), peak RSS sampled
# every 2 s from the server pid (ps rss). Box check before / after.
set -m
. /opt/code/intarsiaml/.worktrees/ink-batch/.scratch/b2048/common.sh
. $W/.scratch/serve_lib.sh
B=$O/bin/new/intarsia; BS=$O/bin/main/batch_step; TREE=$(git -C $N rev-parse --short HEAD)
echo "tree $TREE; binary md5 $(md5 -q $B)"
[ -x $W/.scratch/boxcheck.sh ] && run bash $W/.scratch/boxcheck.sh $BS 2>&1 | tail -2
n=-1
for arm in 2048 256 2048 2048 256; do
  n=$((n+1)); A=$O/arm$n-bc$arm; mkdir -p $A
  x=""; [ $arm = 256 ] && x="--busy-chunk 256"
  PORT=8171; while lsof -nP -iTCP:$PORT -sTCP:LISTEN >/dev/null 2>&1; do PORT=$((PORT+1)); done; SRV_PORT=$PORT
  echo "== arm $n busy chunk $arm ($([ -n "$x" ] && echo "flag $x" || echo "default, no flag")) port $PORT $(date +%T)"
  start_group $A/serve.log $A/serve.err $B serve --model $CK --backend metal --host 127.0.0.1 --port $PORT --max-batch 16 --max-seq 8192 --served-model-name inkling-2-mini $x
  k=0; up=1
  until curl -sf http://127.0.0.1:$PORT/v1/models > /dev/null; do
    sleep 1; k=$((k+1))
    kill -0 $SRV_PGID 2>/dev/null || { echo "server died"; tail -5 $A/serve.err; up=0; break; }
    [ $k -gt 600 ] && { echo "server not up"; up=0; break; }
  done
  [ $up = 1 ] || { stop_group; continue; }
  Lp=$(lsof -t -nP -iTCP:$PORT -sTCP:LISTEN | head -1)
  ps -o command= -p $Lp | grep -q "^$B serve" || { echo "port $PORT answers another server"; stop_group; continue; }
  grep -h 'busy prefill chunk' $A/serve.err | tail -1
  ( while kill -0 $Lp 2>/dev/null; do ps -o rss= -p $Lp >> $A/rss.txt 2>/dev/null; sleep 2; done ) &
  SAMP=$!
  for w in 1 2; do
    curl -s -o /dev/null http://127.0.0.1:$PORT/v1/chat/completions -H 'Content-Type: application/json' \
      -d '{"model":"inkling-2-mini","messages":[{"role":"user","content":"Count from one to twenty."}],"max_tokens":64,"temperature":0}'
  done
  curl -s http://127.0.0.1:$PORT/metrics > $A/metrics0.txt
  (cd /opt/code/deepeval && exec uv run localeval suite --url http://127.0.0.1:$PORT/v1 --model inkling-2-mini --tier adhoc --cells conc-plain-1k --thinking default --label endor-intarsia-metal-busy2048-arm$n-$arm --tokenizer-repo thinkingmachines/Inkling-2-Mini --max-context 8192 --max-num-seqs 16 --lane plain --recipe "endor: intarsia serve --backend metal --max-batch 16 --max-seq 8192 $x (ink/busy2048 $TREE, ABBA arm $n)" > $A/localeval.log 2>&1) &
  CP=$!; wait $CP; rc=$?; CP=""
  echo "localeval exit $rc"; grep -iE 'streams|aggregate|ttft|result.json' $A/localeval.log | tail -8
  curl -s http://127.0.0.1:$PORT/metrics > $A/metrics1.txt
  [ "$(lsof -t -nP -iTCP:$PORT -sTCP:LISTEN | head -1)" = "$Lp" ] || echo "server on $PORT changed"
  stop_group
  kill $SAMP 2>/dev/null; wait $SAMP 2>/dev/null
  echo "peak rss arm $n: $(sort -n $A/rss.txt | tail -1) KiB over $(wc -l < $A/rss.txt | tr -d ' ') samples"
done
[ -x $W/.scratch/boxcheck.sh ] && run bash $W/.scratch/boxcheck.sh $BS 2>&1 | tail -2
echo "gate C done"
