#!/bin/zsh
# mgdnvsoak: sample RSS, footprint (incl. GPU-resident), swap and thermal of PID every 60 s into $2 until PID exits. mon.sh <pid> <out.tsv>
echo "time\trss_mb\tfootprint\tswap_used\tthermal" > $2
while kill -0 $1 2>/dev/null; do
  rss=$(ps -o rss= -p $1 | awk '{printf "%d", $1/1024}'); fp=$(footprint -p $1 2>/dev/null | grep -i -m1 'Footprint:' | sed 's/.*Footprint: *//'); sw=$(sysctl -n vm.swapusage | awk '{print $6}')
  echo "$(date +%T)\t$rss\t$fp\t$sw\t$(notifyutil -g com.apple.system.thermalpressurelevel 2>&1 | awk '{print $2}')" >> $2; sleep 60
done
