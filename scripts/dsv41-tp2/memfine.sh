#!/bin/bash
# One line a second on this box: unix time, then GiB of MemAvailable and its parts, RSS of our ranks, and at low memory the top
# processes. Kills our ranks under $1 GiB (run-full.sh's watcher does the same over ssh, every ~6 s). Started by run-full.sh.
KILL=${1:-5}; PAT=${2:?pkill -f pattern of our ranks}
while true; do
  m=$(awk '/^(MemAvailable|MemFree|Cached|Shmem|AnonPages|Mapped|Unevictable|Slab|PageTables):/{printf "%s=%.2f ", substr($1,1,length($1)-1), $2/1048576}' /proc/meminfo)
  r=$(ps -C intarsia -o rss= | awk '{s+=$1} END{printf "%.2f", s/1048576}')
  a=${m#*MemAvailable=}; a=${a%% *}
  echo "$(date +%s) $m rss=$r"
  if awk "BEGIN{exit !($a < 7.5)}"; then ps -eo rss,comm --sort=-rss | head -6 | awk '{printf "  top %.2f %s\n", $1/1048576, $2}'; fi
  if awk "BEGIN{exit !($a < $KILL)}"; then pkill -KILL -f "$PAT"; echo "KILLED under $KILL GiB"; fi
  sleep 1
done
