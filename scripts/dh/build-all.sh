#!/usr/bin/env bash
# Lane dh (from the Mac, inside ONE fleetq cpu job): sync and build each "<rev>:<bin>" pair in order into ~/intarsiaml-dh (the last one
# synced stays the tree the windows run from), e.g.
#   fleetq run --host tatooine --load cpu --expect 60 --label "dh build" -- scripts/dh/build-all.sh 982ec8b27:dh-intarsia-base HEAD:dh-intarsia
# HIPTEST=1 also compiles the HIP crate's GPU tests (and MTESTS models tests) for the last pair.
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq}"
S="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
n=$#; i=0
for pair in "$@"; do
  i=$((i + 1)); rev=${pair%%:*}; bin=${pair#*:}; mark=""
  [[ $bin == *:* ]] && { mark=${bin#*:}; bin=${bin%%:*}; }
  stamp=$("$S/sync.sh" "$rev" | tail -1) || { echo "sync $rev failed"; exit 2; }
  ht=0; [[ $i == "$n" ]] && ht=${HIPTEST:-0}
  echo "== $rev -> $bin ($stamp) $(date +%T)"
  ssh -o BatchMode=yes tatooine "mkdir -p ~/intarsiaml-dh/.dh && FLEETQ_JOB=$FLEETQ_JOB TREES='dev=$stamp' BIN=$bin HIPTEST=$ht MTESTS='${MTESTS:-}' bash -s" < "$S/build.sh" || exit 3
  # "<rev>:<bin>:<marker>": the built CLI must contain <marker> (a string only that rev has), or the build was stale
  if [[ -n $mark ]]; then
    n=$(ssh -o BatchMode=yes tatooine "grep -c '$mark' ~/intarsiaml-dh/$bin" < /dev/null)
    if [[ ${n:-0} == 0 ]]; then echo "STALE: $bin lacks '$mark'"; exit 4; fi
    echo "$bin has '$mark'"
  fi
done
scp -q "$S"/lib.sh "$S"/guard.sh "$S"/stages-*.sh tatooine:intarsiaml-dh/.dh/
