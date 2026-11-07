#!/usr/bin/env bash
# lane q2s: on a Spark, kill this lane's processes at once when MemAvailable drops below 10 GB (a GB10 swaps instead of OOM-killing).
# Started by scripts/q2s/xg.sh as ~/intarsia-q2s/memwatch.sh <log dir>; stopped by it on exit.
while true; do
  a=$(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo)
  if [ "$a" -lt 10 ]; then echo "$(date +%T) MemAvailable $a GB: killing intarsia-q2s processes" >> "$1/memwatch.log"; pkill -KILL -f '[i]ntarsia-mt2/bin/'; fi
  sleep 1
done
