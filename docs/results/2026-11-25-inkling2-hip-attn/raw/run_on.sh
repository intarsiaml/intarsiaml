#!/usr/bin/env bash
# Run one command in the HIP dev image on card 0 for a given tree and target volume (ON tatooine, inside window.sh or a fleetq
# build job). Usage: run_on.sh <repo dir> <target volume> <container name> <command> [args]
set -euo pipefail
cd "$1"
export INTARSIA_REPO=$PWD INTARSIA_TARGET_VOL=$2 INTARSIA_CONTAINER_NAME=$3
shift 3
export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-0}
exec deploy/tatooine/dev.sh "$@" < /dev/null
