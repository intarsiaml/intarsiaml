#!/usr/bin/env bash
# ON tatooine, `fleetq run --host tatooine-r9700 --load cpu` (lane ink-sampled-adapt): the branch CLI + two test binaries as `ad1`, then
# main aee06201c (.scratch/main-src) as `main6`.
set -uo pipefail
T=$HOME/intarsiaml-ink-sampled-adapt/.scratch/tat
echo "[build] branch $(date +%T)"
MTESTS="inkling2_sampled_real inkling2_dspark_sampled_adapt" bash $T/build.sh ${BR:-ad3} || exit $?
if [ -z "${SKIP_MAIN:-}" ]; then
  echo "[build] main $(date +%T)"
  bash $T/build.sh ${MAINNAME:-main6} .scratch/main-src || exit $?
fi
echo "[build] done $(date +%T)"
md5sum $HOME/intarsiaml-ink-sampled-adapt/.scratch/bin/*
