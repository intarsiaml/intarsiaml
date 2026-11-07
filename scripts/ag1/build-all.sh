#!/usr/bin/env bash
# Lane ag1: copy of scripts/dt1 (lane dt1), repointed to ~/intarsiaml-ag1, volume intarsia-cargo-ag1, CLI ag1-intarsia, container ag1-serve, output docs/results/2026-11-07-spec-auto-graph/raw.
# Lane dh (from the Mac, inside ONE fleetq cpu job): sync and build each "<rev>:<bin>" pair in order into ~/intarsiaml-ag1 (the last one
# synced stays the tree the windows run from), e.g.
#   fleetq run --host tatooine --load cpu --expect 60 --label "ag1 build" -- scripts/ag1/build-all.sh 3f46da4f7:ag1-intarsia-base HEAD:ag1-intarsia
# HTESTS="win_splitkv attention" also compiles those HIP GPU test binaries for the last pair.
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq}"
S="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
n=$#; i=0
for pair in "$@"; do
  i=$((i + 1)); rev=${pair%%:*}; bin=${pair#*:}; mark=""
  [[ $bin == *:* ]] && { mark=${bin#*:}; bin=${bin%%:*}; }
  stamp=$("$S/sync.sh" "$rev" | tail -1) || { echo "sync $rev failed"; exit 2; }
  ht=""; [[ $i == "$n" ]] && ht=${HTESTS:-}
  echo "== $rev -> $bin ($stamp) $(date +%T)"
  ssh -o BatchMode=yes tatooine "mkdir -p ~/intarsiaml-ag1/.ag1 && FLEETQ_JOB=$FLEETQ_JOB MTESTS='${MTESTS:-}' TREES='dev=$stamp' BIN=$bin HTESTS='$ht' bash -s" < "$S/build.sh" || exit 3
  # "<rev>:<bin>:<marker>": the built CLI must contain <marker> (a string only that rev has), or the build was stale
  if [[ -n $mark ]]; then
    n=$(ssh -o BatchMode=yes tatooine "grep -c '$mark' ~/intarsiaml-ag1/$bin" < /dev/null)
    if [[ ${n:-0} == 0 ]]; then echo "STALE: $bin lacks '$mark'"; exit 4; fi
    echo "$bin has '$mark'"
  fi
done
scp -q "$S"/lib.sh "$S"/guard.sh "$S"/stages-*.sh tatooine:intarsiaml-ag1/.ag1/
