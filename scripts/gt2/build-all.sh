#!/usr/bin/env bash
# Lane gt2: copy of scripts/ag1 repointed to ~/intarsiaml-gt2, volume intarsia-cargo-gt2, CLI gt2-intarsia, container gt2-serve, output docs/results/2026-11-07-spec-graph-tp2/raw.
# Lane dh (from the Mac, inside ONE fleetq cpu job): sync and build each "<rev>:<bin>" pair in order into ~/intarsiaml-gt2 (the last one
# synced stays the tree the windows run from), e.g.
#   fleetq run --host tatooine --load cpu --expect 60 --label "gt2 build" -- scripts/gt2/build-all.sh 3f46da4f7:gt2-intarsia-base HEAD:gt2-intarsia
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
  ssh -o BatchMode=yes tatooine "mkdir -p ~/intarsiaml-gt2/.gt2 && FLEETQ_JOB=$FLEETQ_JOB MTESTS='${MTESTS:-}' TREES='dev=$stamp' BIN=$bin HTESTS='$ht' bash -s" < "$S/build.sh" || exit 3
  # "<rev>:<bin>:<marker>": the built CLI must contain <marker> (a string only that rev has), or the build was stale
  if [[ -n $mark ]]; then
    n=$(ssh -o BatchMode=yes tatooine "grep -c '$mark' ~/intarsiaml-gt2/$bin" < /dev/null)
    if [[ ${n:-0} == 0 ]]; then echo "STALE: $bin lacks '$mark'"; exit 4; fi
    echo "$bin has '$mark'"
  fi
done
scp -q "$S"/lib.sh "$S"/guard.sh "$S"/stages-*.sh tatooine:intarsiaml-gt2/.gt2/
