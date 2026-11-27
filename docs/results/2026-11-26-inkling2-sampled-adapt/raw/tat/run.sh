#!/usr/bin/env bash
# ON tatooine (inside window.sh): one command in the HIP dev image on card 0, this lane's repo and target volume, container
# ink-adapt-run (window.sh removes it on exit). Binaries: /work/.scratch/bin/intarsia-<name>, /target/release/intarsia.
set -euo pipefail
cd "$HOME/intarsiaml-ink-sampled-adapt"
export INTARSIA_REPO=${RUN_REPO:-$PWD} INTARSIA_TARGET_VOL=intarsia-cargo-target-ink-sampled-adapt INTARSIA_CONTAINER_NAME=${INK_CONTAINER:-ink-adapt-run}
export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-0}
exec deploy/tatooine/dev.sh "$@" < /dev/null
