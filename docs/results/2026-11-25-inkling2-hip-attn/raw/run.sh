#!/usr/bin/env bash
# Run one command in the HIP dev image on card 0 (ON tatooine, inside window.sh): the lane's repo, its own cargo target volume,
# container name `ink-hip-attn-run` (window.sh removes it on exit). Usage: run.sh <command> [args]; the binary is /target/release/intarsia.
set -euo pipefail
cd "$HOME/intarsiaml-ink-hip-attn"
export INTARSIA_REPO=$PWD INTARSIA_TARGET_VOL=intarsia-cargo-target-ink-hip-attn INTARSIA_CONTAINER_NAME=ink-hip-attn-run
export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-0}
exec deploy/tatooine/dev.sh "$@" < /dev/null
