#!/usr/bin/env bash
# ON tatooine (inside window.sh): one command in the HIP dev image, this lane's repo and target volume. CARD (default 0) picks the
# GPU; the container is ink-hip-pf2-tiny-c<CARD> (window.sh removes both on exit). Binaries: /work/.scratch/bin/intarsia-<name>.
set -euo pipefail
cd "$HOME/intarsiaml-ink-hip-pf2-tiny"
CARD=${CARD:-0}
export INTARSIA_REPO=$PWD INTARSIA_TARGET_VOL=intarsia-cargo-target-ink-hip-pf2-tiny INTARSIA_CONTAINER_NAME=ink-hip-pf2-tiny-c$CARD
export HIP_VISIBLE_DEVICES=$CARD
exec deploy/tatooine/dev.sh "$@" < /dev/null
