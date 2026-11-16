#!/usr/bin/env bash
# usage: memwatch.sh <pgrep -f pattern> <outfile>: waits up to 10 min for a process matching the pattern, then logs MemAvailable (MB)
# every 2 s while one runs and kills it (-9) under 10 GB (earlyoom is the backstop). Write the pattern so it does not match this
# script's own command line (e.g. 'dep[s]/mtp_store_cuda').
pat=$1; f=$2; : > "$f"
for i in $(seq 1 300); do pgrep -f "$pat" >/dev/null && break; sleep 2; done
while pgrep -f "$pat" >/dev/null; do
  a=$(awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo); echo "$(date +%T) $a" >> "$f"
  if [ "$a" -lt 10240 ]; then pkill -9 -f "$pat"; echo "KILLED low mem $a" >> "$f"; fi
  sleep 2
done
