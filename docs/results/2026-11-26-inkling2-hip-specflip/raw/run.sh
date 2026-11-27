#!/usr/bin/env bash
# ON tatooine (inside window.sh): one command in the HIP dev image on card $CARD (default 1), this lane's repo and target volume, container
# ink-hip-specflip-run (window.sh removes it on exit). Binaries: /work/.scratch/bin/intarsia-<name>.
set -euo pipefail
cd "$HOME/intarsiaml-ink-hip-specflip"
export INTARSIA_REPO=$PWD INTARSIA_TARGET_VOL=intarsia-cargo-target-ink-hip-specflip INTARSIA_CONTAINER_NAME=ink-hip-specflip-run
export HIP_VISIBLE_DEVICES=${CARD:-1}
# one container per card when two stages run at once
[ -n "${CNAME:-}" ] && export INTARSIA_CONTAINER_NAME=$CNAME
exec deploy/tatooine/dev.sh "$@" < /dev/null
