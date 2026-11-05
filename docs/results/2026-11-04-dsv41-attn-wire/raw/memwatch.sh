#!/bin/bash
# log MemAvailable every 5 s; under 10 GB: flag ABORT and kill this job's build and test processes
d=$HOME/intarsia-dsv41-attnwire
while true; do
  a=$(awk '/MemAvailable/ {printf "%d", $2/1048576}' /proc/meminfo)
  echo "$(date +%T) avail ${a} GB" >> "$d/logs/mem.log"
  if [ "$a" -lt 10 ]; then
    echo "$(date +%T) AVAILABLE UNDER 10 GB: killing this job's processes" >> "$d/logs/mem.log"
    touch "$d/ABORT"
    "$d/killmine.sh"
  fi
  sleep 5
done
