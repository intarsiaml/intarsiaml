#!/usr/bin/env bash
# ON tatooine (inside window.sh): one command in the HIP dev image on card 0, this lane's repo and target volume, container
# ink-hip-i8-run (window.sh removes it on exit). Binaries: /work/.scratch/bin/intarsia-<name>, /target/release/intarsia.
set -euo pipefail
cd "$HOME/intarsiaml-ink-hip-i8"
export INTARSIA_REPO=$PWD INTARSIA_TARGET_VOL=intarsia-cargo-target-ink-hip-i8 INTARSIA_CONTAINER_NAME=ink-hip-i8-run
export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-0}
exec deploy/tatooine/dev.sh "$@" < /dev/null
