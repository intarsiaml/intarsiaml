#!/usr/bin/env bash
# usage: memwatch.sh <pid> <outfile>: log MemAvailable (MB) every 2 s; kill -9 <pid> below 15 GB (the 2026-11-19 lane rule)
pid=$1; f=$2
while kill -0 "$pid" 2>/dev/null; do
  a=$(awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo); echo "$(date +%T) $a" >> "$f"
  if [ "$a" -lt 15360 ]; then kill -9 "$pid"; echo "KILLED low mem $a" >> "$f"; fi
  sleep 2
done
