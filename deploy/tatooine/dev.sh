#!/usr/bin/env bash
# Run a command in the intarsia HIP dev image on tatooine (run ON tatooine).
#   deploy/tatooine/dev.sh cargo check --workspace
#   HIP_VISIBLE_DEVICES=1 deploy/tatooine/dev.sh ./run-something
#   deploy/tatooine/dev.sh --target-vol     # name of this image's cargo target volume
# GPU use requires a fleetq lease taking radiance down (see docs/results/2026-10-13-r9700-prep.md).
set -euo pipefail
IMAGE="${INTARSIA_IMAGE:-intarsia-hip-dev:rocm7.2.4-rust1.90.0}"
REPO="${INTARSIA_REPO:-$HOME/intarsiaml}"
MODELS="${INTARSIA_MODELS:-/data/models/intarsia}"
# The cargo target dir is per image build: a volume that outlived an image rebuild would let cargo reuse kernel
# libraries compiled by the previous hipcc. INTARSIA_TARGET_VOL overrides; INTARSIA_CLEAN=1 wipes this image's target.
IMAGE_ID="$(docker image inspect -f '{{.Id}}' "$IMAGE" | sed 's/^sha256://' | cut -c1-12)"
TARGET_VOL="${INTARSIA_TARGET_VOL:-intarsia-cargo-target-$IMAGE_ID}"
REGISTRY_VOL="${INTARSIA_REGISTRY_VOL:-intarsia-cargo-target-registry}"  # crates.io download cache, shared across images
if [[ "${1:-}" == --target-vol ]]; then echo "$TARGET_VOL"; exit 0; fi
if [[ "${INTARSIA_CLEAN:-0}" == 1 ]]; then docker volume rm -f "$TARGET_VOL" >/dev/null; fi
ENVS=(-e GPU_MAX_HW_QUEUES=1 -e CARGO_TARGET_DIR=/target)
[[ -n "${HIP_VISIBLE_DEVICES:-}" ]] && ENVS+=(-e "HIP_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES")
[[ -n "${FLEETQ_JOB:-}" ]] && ENVS+=(-e "FLEETQ_JOB=$FLEETQ_JOB")
# engine tuning knobs (INTARSIA_HIP_* kernel knobs, INTARSIA_TP_* tensor-parallel knobs) reach the container
for v in $(compgen -e | grep -E '^INTARSIA_(HIP|TP)_' || true); do ENVS+=(-e "$v=${!v}"); done
# Named volumes are created root-owned; make them writable for --user once.
for v in "$TARGET_VOL" "$REGISTRY_VOL"; do
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
  -v "$REGISTRY_VOL":/opt/cargo/registry \
  -w /work "$IMAGE" "${@:-bash}"
