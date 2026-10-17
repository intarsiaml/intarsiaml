#!/bin/zsh
# usage: end_sample.sh OUTDIR  -- after every server of the slot is stopped: 5 s idle GPU sample (macmon), ps top 12, leftover servers, swap.
O=$1; mkdir -p $O
macmon pipe -i 1000 > $O/macmon-end-idle.jsonl 2>/dev/null & P=$!
sleep 5; kill $P 2>/dev/null
{ echo "--- end-idle $(date '+%H:%M:%S')"
  python3 -c "
import json,sys
for l in open('$O/macmon-end-idle.jsonl'):
    try: d=json.loads(l)
    except Exception: continue
    print('gpu_usage',d['gpu_usage'],'gpu_power',round(d['gpu_power'],2),'sys_power',round(d['sys_power'],1))"
  ps -axo pid,pcpu,rss,etime,comm | sort -k2 -nr | head -12 | cut -c1-150
  sysctl vm.swapusage
  echo "leftover servers: $(pgrep -fl 'intarsia.* serve' || echo none)"
  echo "tracked: $(pgrep -fl 'kev|tensorfold' | head -5 | tr '\n' ';')"
} >> $O/hygiene-end-idle.txt 2>&1
