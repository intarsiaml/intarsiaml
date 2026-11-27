#!/bin/bash
# endor EXCLUSIVE fleetq slot F, lane ink-metal-dflash2-ready (coordinator check 5): `serve --max-batch 1` with the DFlash2 draft (dfl5) and with
# --spec off, each started twice (cold, warm); multi-turn chats (chat endpoint and token-id extension), per-turn ids and cached_tokens; start times.
set -u
: "${FLEETQ_JOB:?run under fleetq}"
. $(dirname $0)/lib.sh
O=$S/F; mkdir -p $O; T0=$(date +%s)
echo "slot F start $(date +%T) binary md5 $(md5 -q $B) git $(git -C $W rev-parse --short HEAD)"
macmon pipe -i 250 > $O/macmon.jsonl 2>/dev/null &
i=0
for spec in dfl5:1 dfl5:2 off:1 off:2; do
  arm=${spec%%:*}; n=${spec#*:}; i=$((i+1)); port=$((8300+i)); U=http://127.0.0.1:$port/v1; name=$arm-start$n
  [ $(( $(date +%s) - T0 )) -ge 780 ] && { echo "slot F: time box before $name"; break; }
  t=$(date +%s.%N)
  serve_start $name $arm $port "--max-batch 1" ""
  if python3 $R/client.py wait $U 420; then
    echo "$name up after $(python3 -c "import time; print(round(time.time()-$t,1))") s | $(grep -aE '^loaded ' $O/s-$name.err | head -1 | cut -c1-200) | admission: $(grep -aiE 'admi' $O/s-$name.err | head -3 | tr '\n' '|' | cut -c1-300)"
    python3 $R/client.py chat $U $O/chat-$name.json $S/prompts/b4.json $S/prompts/abba3.json
  else
    echo "$name NOT UP within 420 s"
  fi
  serve_stop
  echo "$name stopped t+$(( $(date +%s) - T0 ))s"
  sleep 3
done
for n in 1 2; do python3 $R/client.py cmpchat $O/chat-dfl5-start$n.json $O/chat-off-start$n.json; done
python3 $R/client.py cmpchat $O/chat-dfl5-start1.json $O/chat-dfl5-start2.json
echo "slot F end $(date +%T) t+$(( $(date +%s) - T0 ))s"
