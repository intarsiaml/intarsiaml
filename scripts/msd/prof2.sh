#!/usr/bin/env bash
# Lane msd: attribution window: nsys profile of a 16- and an 8-stream burst, one fleetq slot.
#   fleetq run --host spark1 --quiet gpu,cpu --mem 45 --expect 25 --label "msd: nsys B=16,8" -- scripts/msd/prof2.sh <bin> <name>
set -uo pipefail
H="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RANK_ARGS="${RANK_ARGS:-5}" WIN="4 11" N=16 "$H/prof.sh" "$1" "$2-b16"
RANK_ARGS="${RANK_ARGS:-5}" WIN="3 7" N=8 "$H/prof.sh" "$1" "$2-b8"
