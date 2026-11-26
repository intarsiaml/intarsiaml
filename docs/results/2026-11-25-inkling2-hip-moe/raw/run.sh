#!/usr/bin/env bash
# Run one command in the HIP dev image on card 0 (ON tatooine): lane ink-hip-moe's repo, its own cargo target volume, container
# name `ink-hip-moe-run` (window.sh removes it on exit). Copy of the lead's run.sh.
set -euo pipefail
cd "$HOME/intarsiaml-ink-hip-moe"
export INTARSIA_REPO=$PWD INTARSIA_TARGET_VOL=intarsia-cargo-target-ink-hip-moe INTARSIA_CONTAINER_NAME=${INK_CONTAINER:-ink-hip-moe-run}
export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-0}
exec deploy/tatooine/dev.sh "$@" < /dev/null
