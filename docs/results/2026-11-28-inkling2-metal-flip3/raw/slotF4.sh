#!/bin/bash
# endor EXCLUSIVE fleetq slot F4, lane ink-metal-flip3: serve with the new Metal defaults. `intarsia serve --max-batch 1`, the multi-turn chat set
# (b4.json + abba3.json, client.py chat), arms new / rev / off; per-turn ids must be equal new = rev = off, and new's stderr must show both default notes.
set -u
: "${FLEETQ_JOB:?run under fleetq}"
. $(dirname $0)/lib.sh
O=$R/F4; mkdir -p $O; T0=$(date +%s)
echo "slot F4 start $(date +%T) binary md5 $(md5 -q $B) commit $(cat $B.commit)"
i=0
for arm in new rev off; do
  i=$((i+1)); port=$((8460+i)); U=http://127.0.0.1:$port/v1; name=mb1-$arm
  t=$(date +%s)
  serve_start $name $arm $port "--max-batch 1"
  if python3 $R/client.py wait $U 420; then
    echo "$name up after $(( $(date +%s) - t )) s"
    python3 $R/client.py chat $U $O/chat-$name.json $R/b4.json $R/abba3.json
    echo "$name chat done t+$(( $(date +%s) - T0 ))s"
  else
    echo "$name NOT UP within 420 s"
  fi
  serve_stop; echo "$name stopped t+$(( $(date +%s) - T0 ))s"
  grep -aE '^dflash: candidates|^dflash: DFlash2 draft head|^dflash: cut|^spec' $O/s-$name.err | sort | uniq -c | cut -c1-200
  sleep 3
done
echo "== new vs rev"; python3 $R/client.py cmpchat $O/chat-mb1-new.json $O/chat-mb1-rev.json
echo "== new vs off"; python3 $R/client.py cmpchat $O/chat-mb1-new.json $O/chat-mb1-off.json
echo "slot F4 end $(date +%T) t+$(( $(date +%s) - T0 ))s"
