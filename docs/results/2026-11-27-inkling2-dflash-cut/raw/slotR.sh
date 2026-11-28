#!/bin/bash
# endor EXCLUSIVE fleetq slot R, lane ink-dflash-cut: the ready gates before asking whether k7 + INTARSIA_DFLASH_CUT=0.3 becomes the Metal
# default (lane ink-metal-dflash2-ready's / ink-metal-solo's S3-S5 on this binary). Arms (lib.sh): off, dfl5 (the default), cut7.
#   R1 sampled: generate seed 7, 32 tokens, T1.0 k0 (w100) and T0.6 k0 (w60), off vs cut7 (cut7 traced): ids equal, no drafting.
#   R2 memory: generate --max-seq 4194304 (1048576 fallback) --kv-fraction 0.25, 64 tokens of c1, dfl5 cut7 dfl5 cut7: peak, context.
#   R3 serve --max-batch 1: cut7 vs off, 2 chats x 3 turns on the chat endpoint + 2 x 3 token-id extensions: replies and cached_tokens.
set -u
: "${FLEETQ_JOB:?run under fleetq}"
. $(dirname $0)/lib.sh
O=$R/R; mkdir -p $O; T0=$(date +%s)
echo "slot R start $(date +%T) binary md5 $(md5 -q $B) git $(git -C $W rev-parse --short HEAD)"
G() { env $(envs $1) /usr/bin/time -l $B generate -m $M --backend metal --profile-every 0 --show-ids "${@:2}" $(flags $1); }
echo "== R1: sampled, seed 7, 32 tokens, --ignore-eos (off vs cut7; cut7 traced)"
for arm in off cut7; do
  E=""; [ $arm = cut7 ] && E="INTARSIA_SPEC_TRACE=1"
  for cfg in "1.0 0 1.0 100" "0.6 0 1.0 60"; do
    set -- $cfg; f=$O/s-$arm-T$1-k$2
    env $E $(envs $arm) /usr/bin/time -l $B generate -m $M --backend metal --max-seq 4096 --profile-every 0 --show-ids --max-tokens 32 --ignore-eos \
      --temperature $1 --top-k $2 --top-p $3 --seed 7 --prompt-ids $R/w$4.json $(flags $arm) > $f.out 2> $f.err
    echo "$arm T$1 k$2 n$4 rc $? t+$(( $(date +%s) - T0 ))s | ids lines $(grep -ac '^ids\[' $f.err) | drafted-pass lines $(grep -acE '^spec: .*token [0-9]+ pos [0-9]+ drafted' $f.err) | $(grep -aE '^spec: verify' $f.err | tr '\n' ' ' | cut -c1-200)"
  done
done
python3 $R/seedcmp.py $O off cut7
echo "== R2: memory / KV capacity"
n=0
for arm in dfl5 cut7 dfl5 cut7; do
  n=$((n+1)); f=$O/mem$n-$arm
  for MS in 4194304 1048576; do
    G $arm --max-seq $MS --kv-fraction 0.25 --max-tokens 64 --ignore-eos --prompt-ids $R/c1.json > $f.out 2> $f.err
    rc=$?; [ $rc = 0 ] && break
  done
  echo "mem$n $arm rc $rc max-seq $MS | $(grep -aE '^loaded ' $f.err | head -1 | cut -c1-260) | $(grep -aE 'dflash: cut' $f.err | head -1) | $(grep -aE 'peak memory footprint' $f.err | tr '\n' ' ')"
  sleep 3
done
echo "== R3: serve --max-batch 1, cut7 vs --spec off, multi-turn"
i=0
for arm in cut7 off; do
  i=$((i+1)); port=$((8430+i)); U=http://127.0.0.1:$port/v1; name=$arm-start1
  [ $(( $(date +%s) - T0 )) -ge 1500 ] && { echo "slot R: time box before $name"; break; }
  t=$(date +%s)
  serve_start $name $arm $port "--max-batch 1"
  if python3 $R/client.py wait $U 420; then
    echo "$name up after $(( $(date +%s) - t )) s | $(grep -aE '^loaded ' $O/s-$name.err | head -1 | cut -c1-200) | $(grep -aE 'dflash: cut|^--spec dflash' $O/s-$name.err | tr '\n' '|' | cut -c1-300)"
    python3 $R/client.py chat $U $O/chat-$name.json $R/b4.json $R/abba3.json
  else
    echo "$name NOT UP within 420 s"
  fi
  serve_stop
  echo "$name stopped t+$(( $(date +%s) - T0 ))s | $(grep -aE '^spec: verify' $O/s-$name.err | tr '\n' '|' | cut -c1-300)"
  sleep 3
done
python3 $R/client.py cmpchat $O/chat-cut7-start1.json $O/chat-off-start1.json
echo "slot R end $(date +%T) t+$(( $(date +%s) - T0 ))s"
