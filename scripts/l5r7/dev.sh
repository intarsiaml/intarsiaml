#!/usr/bin/env bash
# deploy/tatooine/dev.sh plus the 27B checkpoint's HF cache mounted at /hf (the Quark-AWQ-MXFP4 snapshot is not under /data/models/intarsia).
# Run ON tatooine. Same environment variables as deploy/tatooine/dev.sh (INTARSIA_REPO, INTARSIA_TARGET_VOL, HIP_VISIBLE_DEVICES, INTARSIA_HIP_*).
set -euo pipefail
IMAGE="${INTARSIA_IMAGE:-intarsia-hip-dev:rocm7.2.4-rust1.90.0}"
REPO="${INTARSIA_REPO:-$HOME/intarsiaml}"
MODELS="${INTARSIA_MODELS:-/data/models/intarsia}"
TARGET_VOL="${INTARSIA_TARGET_VOL:?}"
REGISTRY_VOL="${INTARSIA_REGISTRY_VOL:-intarsia-cargo-target-registry}"
ENVS=(-e GPU_MAX_HW_QUEUES=1 -e CARGO_TARGET_DIR=/target)
[[ -n "${HIP_VISIBLE_DEVICES:-}" ]] && ENVS+=(-e "HIP_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES")
ENVS+=(-e "INTARSIA_HOSTNAME=$(hostname)")
[[ -n "${FLEETQ_JOB:-}" ]] && ENVS+=(-e "FLEETQ_JOB=$FLEETQ_JOB")
for v in $(compgen -e | grep -E '^INTARSIA_(HIP|TP|BATCH|QWEN)_' || true); do ENVS+=(-e "$v=${!v}"); done
for v in "$TARGET_VOL" "$REGISTRY_VOL"; do
  if ! docker volume inspect "$v" >/dev/null 2>&1; then
    docker volume create "$v" >/dev/null
    docker run --rm -v "$v":/v --user 0 "$IMAGE" chmod 0777 /v
  fi
done
exec docker run --rm -i \
  --device /dev/kfd --device /dev/dri \
  --group-add "$(stat -c %g /dev/kfd)" --group-add "$(stat -c %g /dev/dri/renderD128)" --group-add video \
  --ipc host --security-opt seccomp=unconfined \
  --user "$(id -u):$(id -g)" -e HOME=/tmp \
  "${ENVS[@]}" \
  -v "$REPO":/work \
  -v "$MODELS":/models:ro \
  -v /data/models/radiance/hf-cache/hub:/hf:ro \
  -v "$TARGET_VOL":/target \
  -v "$REGISTRY_VOL":/opt/cargo/registry \
  -w /work "$IMAGE" "${@:-bash}"
