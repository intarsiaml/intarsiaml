#!/bin/zsh
# usage: psample.sh OUTFILE SECONDS : every 3 s, epoch + top-5 CPU consumers (excluding ps itself) + load average
OUTF=$1; END=$(( $(date +%s) + $2 ))
while (( $(date +%s) < END )); do
  echo "T $(python3 -c 'import time;print("%.1f"%time.time())') load $(sysctl -n vm.loadavg)" >> $OUTF
  ps -Ao pcpu,pid,comm -r | sed -n '2,7p' | sed -E 's#/[^ ]*/([^/ ]+)$#\1#' >> $OUTF
  sleep 3
done
