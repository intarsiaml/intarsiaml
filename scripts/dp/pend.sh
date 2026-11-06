#!/usr/bin/env bash
# Runs ON tatooine (ssh tatooine bash -s < pend.sh): prints the smallest number of seconds left on any pending "sleep N; <start radiance>" timer a
# lane left behind (nothing if none). Read-only.
ps -eo etimes=,args= | awk '/sleep [0-9]+; *systemctl --user st[a]rt radiance/ && !/awk/ { n = $0; sub(/.*sleep /, "", n); sub(/;.*/, "", n); r = n - $1; if (r > 0) print r }' | sort -n | head -1
