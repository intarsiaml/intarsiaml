#!/usr/bin/env bash
# ON tatooine (inside window.sh): one command in the HIP dev image on card 0, this lane's repo and target volume, container
# ink-dflash2-run (window.sh removes it on exit). Binaries: /work/.scratch/bin/intarsia-<name>.
set -euo pipefail
cd "$HOME/intarsiaml-ink-dflash2"
export INTARSIA_REPO=$PWD INTARSIA_TARGET_VOL=intarsia-cargo-target-ink-dflash2 INTARSIA_CONTAINER_NAME=ink-dflash2-run
export HIP_VISIBLE_DEVICES=0
exec deploy/tatooine/dev.sh "$@" < /dev/null
