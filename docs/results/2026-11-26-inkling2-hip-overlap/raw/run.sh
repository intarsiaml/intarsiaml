#!/usr/bin/env bash
# ON tatooine (inside window.sh): one command in the HIP dev image on card 0, this lane's repo and target volume, container
# ink-hip-bpf-run (window.sh removes it on exit). Binaries: /work/.scratch/bin/intarsia-<name>, /target/release/intarsia.
set -euo pipefail
cd "${INK_REPO:-$HOME/intarsiaml-ink-hip-bpf}"
# INK_REPO / INK_VOL: another source tree and target volume (main's tests: .scratch/main-src, intarsia-cargo-target-ink-hip-bpf-main)
# INK_CONTAINER: another container name (a second card's run beside the first, e.g. ink-hip-bpf-run2; window.sh removes both)
export INTARSIA_REPO=$PWD INTARSIA_TARGET_VOL=${INK_VOL:-intarsia-cargo-target-ink-hip-bpf} INTARSIA_CONTAINER_NAME=${INK_CONTAINER:-ink-hip-bpf-run}
export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-0}
exec deploy/tatooine/dev.sh "$@" < /dev/null
