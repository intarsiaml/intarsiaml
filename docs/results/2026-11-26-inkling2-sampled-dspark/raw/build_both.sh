#!/usr/bin/env bash
# ON tatooine, --load cpu slot: the branch CLI + test binaries, then main's CLI + its inkling2 tiny test (lane ink-sampled).
set -uo pipefail
T=$HOME/intarsiaml-ink-sampled/.scratch/tat
echo "[build] branch $(date +%T)"
MTESTS="${MTESTS-inkling2 inkling2_dspark inkling2_dspark_cut inkling2_verify_softcap inkling2_dspark_sampled inkling2_dspark_sampled_gumbel inkling2_sampled_real}" bash $T/build.sh ${BR:-br2} || exit $?
if [ -z "${NOMAIN:-}" ]; then
  echo "[build] main $(date +%T)"
  MTESTS="inkling2" bash $T/build.sh main .scratch/main-src || exit $?
fi
echo "[build] done $(date +%T)"
