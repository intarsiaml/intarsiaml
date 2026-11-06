#!/usr/bin/env bash
# Lane pf: build (scripts/cpf/build.sh all dev) then every GPU stage (scripts/cpf/gpu.sh) in ONE spark1 slot:
#   fleetq run --host spark1 --quiet gpu,cpu --mem 60 --expect 190 --label "pf: build + CUDA pack gates" -- scripts/cpf/all.sh
set -uo pipefail
H="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
"$H/build.sh" all dev || { echo "BUILD FAILED"; exit 1; }
"$H/gpu.sh" "$@"
