#!/bin/bash
# usage: rsswatch.sh LIMIT_GB cmd...   (stops cmd and its children if their RSS exceeds LIMIT_GB)
lim=$(( $1 * 1048576 )); shift
"$@" & pid=$!
( while kill -0 $pid 2>/dev/null; do
    kids=$(ps -o pid= --ppid $pid 2>/dev/null)
    r=0; for p in $pid $kids; do v=$(ps -o rss= -p $p 2>/dev/null | tr -d ' '); r=$(( r + ${v:-0} )); done
    if [ "$r" -gt "$lim" ]; then echo "[rsswatch] RSS ${r} kB over the limit, stopping" >&2; kill $kids $pid; fi
    sleep 1; done ) &
wait $pid
