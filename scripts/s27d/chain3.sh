#!/usr/bin/env bash
# Lane s27d window 3 (inside a fleetq quiet slot): 4 more interleaved 1k rounds base vs dev (default flags), then one 4k pair.
#   fleetq run --host spark1 --quiet gpu,cpu --mem 50 --expect 40 --label "s27d: w3 gate rounds base vs dev" -- scripts/s27d/chain3.sh
H=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SIZES=1024 REPS=1 $H/le-window.sh w3 4 "base:base" "dev:dev"
SIZES=4096 REPS=1 $H/le-window.sh w3-4k 1 "base:base" "dev:dev"
