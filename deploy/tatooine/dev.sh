#!/usr/bin/env bash
# Run a command in the intarsia HIP dev image on tatooine (run ON tatooine).
#   deploy/tatooine/dev.sh cargo check --workspace
#   HIP_VISIBLE_DEVICES=1 deploy/tatooine/dev.sh ./run-something
# GPU use requires a fleetq lease taking radiance down (see docs/results/2026-10-13-r9700-prep.md).
set -euo pipefail
IMAGE="${INTARSIA_IMAGE:-intarsia-hip-dev:rocm7.2.4-rust1.90.0}"
REPO="${INTARSIA_REPO:-$HOME/intarsiaml}"
MODELS="${INTARSIA_MODELS:-/data/models/intarsia}"
TARGET_VOL="${INTARSIA_TARGET_VOL:-intarsia-cargo-target}"
ENVS=(-e GPU_MAX_HW_QUEUES=1 -e CARGO_TARGET_DIR=/target)
[[ -n "${HIP_VISIBLE_DEVICES:-}" ]] && ENVS+=(-e "HIP_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES")
[[ -n "${FLEETQ_JOB:-}" ]] && ENVS+=(-e "FLEETQ_JOB=$FLEETQ_JOB")
# Named volumes are created root-owned; make them writable for --user once.
for v in "$TARGET_VOL" "${TARGET_VOL}-registry"; do
  if ! docker volume inspect "$v" >/dev/null 2>&1; then
    docker volume create "$v" >/dev/null
    docker run --rm -v "$v":/v --user 0 "$IMAGE" chmod 0777 /v
  fi
done
TTY=(); [[ -t 0 && -t 1 ]] && TTY=(-t)
exec docker run --rm -i "${TTY[@]}" \
  --device /dev/kfd --device /dev/dri \
  --group-add "$(stat -c %g /dev/kfd)" --group-add "$(stat -c %g /dev/dri/renderD128)" --group-add video \
  --ipc host --security-opt seccomp=unconfined \
  --user "$(id -u):$(id -g)" -e HOME=/tmp \
  "${ENVS[@]}" \
  -v "$REPO":/work \
  -v "$MODELS":/models:ro \
  -v "$TARGET_VOL":/target \
  -v "${TARGET_VOL}-registry":/opt/cargo/registry \
  -w /work "$IMAGE" "${@:-bash}"
