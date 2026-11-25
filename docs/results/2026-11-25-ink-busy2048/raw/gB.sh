#!/bin/bash
# ink-busy2048 gate B (default load, correctness only): (c) greedy md5 at 64 / 256 tokens main vs new (lone prompt), (c2) the busy prefill chunk each binary
# resolves at startup for Qwen3-8B / Llama-3.1-8B / Inkling-2-Mini (default and --busy-chunk 256), (d) fresh-server output check at the default (2048)
# and at --busy-chunk 256 against solo, short (~1.2k tokens, one chunk) and long (~2.5k tokens, two 2048 chunks / ten 256 chunks) prompts, plus /metrics deltas.
set -m
. /opt/code/intarsiaml/.worktrees/ink-batch/.scratch/b2048/common.sh
. $W/.scratch/serve_lib.sh
cd $N
echo "binary md5 main $(md5 -q $O/bin/main/intarsia) new $(md5 -q $O/bin/new/intarsia)"
python3 -I $O/mkprompts2.py $O/prompt4k.txt $O/pd
P="Write a short story about a lighthouse keeper who finds a message in a bottle. It"
echo "== (c) greedy md5 $(date +%T)"
for v in main new; do
  B=$O/bin/$v/intarsia; D=$O/md5-$v; mkdir -p $D
  run $B generate --model $CK --backend metal --max-seq 4096 --prompt "$P" --max-tokens 64 --temperature 0 > $D/g64.out 2> $D/g64.err
  run $B generate --model $CK --backend metal --max-seq 8192 --temperature 0 --profile-every 0 --prompt "$P" --max-tokens 256 > $D/g256.out 2> $D/g256.err
  for n in 64 256; do if [ -s $D/g$n.out ]; then echo "md5 $v $n $(md5 -q $D/g$n.out)"; else echo "md5 $v $n INVALID (empty)"; fi; done
done
cmp $O/md5-main/g64.out $O/md5-new/g64.out && echo "md5 64: main == new"
cmp $O/md5-main/g256.out $O/md5-new/g256.out && echo "md5 256: main == new"
up() { # $1 = binary, $2 = dir, $3 = model, $4 = served name, rest = extra serve args; sets PORT
  local B=$1 D=$2 M=$3 NM=$4; shift 4; mkdir -p $D
  PORT=8161; while lsof -nP -iTCP:$PORT -sTCP:LISTEN >/dev/null 2>&1; do PORT=$((PORT+1)); done; SRV_PORT=$PORT
  start_group $D/serve.log $D/serve.err $B serve --model $M --backend metal --host 127.0.0.1 --port $PORT --served-model-name $NM "$@"
  local k=0
  until curl -sf http://127.0.0.1:$PORT/v1/models > /dev/null; do
    sleep 1; k=$((k+1)); kill -0 $SRV_PGID 2>/dev/null || { echo "server died"; tail -5 $D/serve.err; return 1; }
    [ $k -gt 600 ] && { echo "server not up"; return 1; }
  done
  ps -o command= -p $(lsof -t -nP -iTCP:$PORT -sTCP:LISTEN | head -1) | grep -q "^$B serve" || { echo "port $PORT answers another server"; return 1; }
  echo "  $(grep -h 'busy prefill chunk' $D/serve.err | tail -1)"
}
echo "== (c2) resolved busy chunk at startup $(date +%T)"
for v in main new; do
  for spec in "qwen3-8b:$HOME/models/Qwen3-8B" "llama31-8b:$HOME/models/Llama-3.1-8B-Instruct" "ink:$CK"; do
    nm=${spec%%:*}; md=${spec#*:}
    for bc in default 256; do
      x=""; [ $bc = 256 ] && x="--busy-chunk 256"
      echo "startup $v $nm busy-chunk=$bc:"
      up $O/bin/$v/intarsia $O/su-$v-$nm-$bc $md m --max-batch 4 --max-seq 4096 $x
      stop_group
    done
  done
done
CMPR=$O/oc3.py
M() { curl -s http://127.0.0.1:$PORT/metrics | grep -E '^intarsia_batch_(cached_tokens_total|prefill_chunks_total|prefill_tokens_total|requests_finished_total) ' | awk '{printf "%s=%s ", substr($1,16), $2}'; echo; }
echo "== (d) output check, new binary $(date +%T)"
for set in short long; do
  S=$O/oc-solo-$set; echo "-- solo reference ($set), server --busy-chunk 256 $(date +%T)"
  if up $O/bin/new/intarsia $S $CK inkling-2-mini --max-batch 16 --max-seq 8192 --busy-chunk 256; then run python3 -I $CMPR $PORT $S $O/pd/$set solo; M; fi
  stop_group
  for arm in default 256; do
    [ $arm = 256 ] && [ $set = short ] && continue
    x=""; [ $arm = 256 ] && x="--busy-chunk 256"
    A=$O/oc-$arm-$set; echo "-- concurrent ($set), fresh server, busy chunk $arm $(date +%T)"
    up $O/bin/new/intarsia $A $CK inkling-2-mini --max-batch 16 --max-seq 8192 $x || { stop_group; continue; }
    echo "before: $(M)"
    run python3 -I $CMPR $PORT $A $O/pd/$set conc $S/solo.json
    echo "after:  $(M)"
    stop_group
  done
done
echo "gate B done"
