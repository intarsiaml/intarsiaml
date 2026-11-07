#!/usr/bin/env bash
# Lane mt2: on a Spark, kill this lane's processes at once when MemAvailable drops below 10 GB (a GB10 swaps instead of OOM-killing).
# Started by scripts/mt2/xg.sh as ~/intarsia-mt2/memwatch.sh <log dir>; stopped by it on exit.
while true; do
  a=$(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo)
  if [ "$a" -lt 10 ]; then echo "$(date +%T) MemAvailable $a GB: killing intarsia-mt2 processes" >> "$1/memwatch.log"; pkill -KILL -f '[i]ntarsia-mt2/bin/'; fi
  sleep 1
done
