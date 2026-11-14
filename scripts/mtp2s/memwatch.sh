#!/usr/bin/env bash
# usage: memwatch.sh <pid> <name>: log MemAvailable every 2 s; kill -9 <pid> if it drops under 10 GB
pid=$1; name=$2; f=$HOME/intarsia-mtp2s/out/mem-$name.txt
while kill -0 "$pid" 2>/dev/null; do
  a=$(awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo); echo "$(date +%T) $a" >> "$f"
  if [ "$a" -lt 10240 ]; then kill -9 "$pid"; echo "KILLED low mem $a" >> "$f"; fi
  sleep 2
done
