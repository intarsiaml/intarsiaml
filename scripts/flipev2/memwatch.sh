#!/usr/bin/env bash
# usage: memwatch.sh <pid> <name>: log MemAvailable every 2 s; kill -9 <pid> (and its children) if it drops under 15 GB
pid=$1; name=$2; f=$HOME/intarsiaml-flipev2/out/mem-$name.txt
while kill -0 "$pid" 2>/dev/null; do
  a=$(awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo); echo "$(date +%T) $a" >> "$f"
  if [ "$a" -lt 15360 ]; then pkill -9 -P "$pid"; kill -9 "$pid"; echo "KILLED low mem $a" >> "$f"; fi
  sleep 2
done
