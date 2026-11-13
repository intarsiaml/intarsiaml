#!/usr/bin/env bash
# fp8-decode: while the server pid in $1 lives, log MemAvailable (GB), the GPU's SM clock / power / temperature / utilisation
# and the server processes' cumulative minor page faults to $2 every 2 s; kill -9 the server below 10 GB available.
while kill -0 "$(cat "$1")" 2>/dev/null; do
  a=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)
  g=$(nvidia-smi --query-gpu=clocks.sm,power.draw,temperature.gpu,utilization.gpu --format=csv,noheader 2>/dev/null | tr -d ' ')
  f=$(ps -o min_flt= -p "$(pgrep -d, -f '[s]erve --backend cuda.*--port 8291')" 2>/dev/null | awk '{s+=$1} END {print "minflt=" s+0}')
  echo "$(date +%T.%N | cut -c1-12) $a $g $f" >> "$2"
  if [ "$a" -lt 10 ]; then kill -9 "$(cat "$1")"; echo KILLED >> "$2"; fi
  sleep 2
done
