#!/bin/bash
# endor EXCLUSIVE fleetq slot H, lane ink-metal-flip: G7 (added after slot G, pre-registered before this slot): `serve` (one sequence, the
# default) with the flip default (both drafts) vs --spec off, each started twice; multi-turn chats (chat endpoint and token-id extension,
# prefix reuse), per-turn ids and cached_tokens. Adapted from lane ink-metal-dflash2-ready's slotF.sh (#549 check 5, DFlash2 alone).
set -u
: "${FLEETQ_JOB:?run under fleetq}"
. $(dirname $0)/lib.sh
O=$R/H; mkdir -p $O; T0=$(date +%s)
echo "slot H start $(date +%T) binary md5 $(md5 -q $B) commit $(cat $B.commit) git $(git -C $W rev-parse --short HEAD)"
macmon pipe -i 250 > $S/macmon-H.jsonl 2>/dev/null &
i=0
for spec in flip:1 flip:2 off:1 off:2; do
  arm=${spec%%:*}; n=${spec#*:}; i=$((i+1)); port=$((8400+i)); U=http://127.0.0.1:$port/v1; name=$arm-start$n
  [ $(( $(date +%s) - T0 )) -ge 1200 ] && { echo "slot H: time box before $name"; break; }
  t=$(date +%s)
  serve_start $name $arm $port "--max-batch 1"
  if python3 $R/client.py wait $U 420; then
    echo "$name up after $(( $(date +%s) - t )) s | $(grep -aE '^loaded ' $O/s-$name.err | head -1 | cut -c1-200) | $(grep -aE 'DFlash2 drafter|second draft|DSpark default' $O/s-$name.err | tr '\n' '|' | cut -c1-400)"
    python3 $R/client.py chat $U $O/chat-$name.json $R/b4.json $R/abba3.json
  else
    echo "$name NOT UP within 420 s"
  fi
  serve_stop
  echo "$name stopped t+$(( $(date +%s) - T0 ))s | $(grep -aE '^spec: verify|switched drafts' $O/s-$name.err | tr '\n' '|' | cut -c1-300)"
  sleep 3
done
for n in 1 2; do python3 $R/client.py cmpchat $O/chat-flip-start$n.json $O/chat-off-start$n.json; done
python3 $R/client.py cmpchat $O/chat-flip-start1.json $O/chat-flip-start2.json
echo "slot H end $(date +%T) t+$(( $(date +%s) - T0 ))s"
