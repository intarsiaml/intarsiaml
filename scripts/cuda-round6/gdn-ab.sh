#!/usr/bin/env bash
# GDN recurrence timing (tests/gdn_timing.rs, 35B shapes) on spark1 inside the caller's slot: the hoisted prep on r6-dev
# (INTARSIA_GDN_HOIST=0 / 1, ROUNDS rounds) and, with EXP=1, the bisect variants of tree r6-exp (INTARSIA_GDN_EXP: 1 = no
# output dot-product chain, 2 = no per-token prep / raw loads after the first token, 3 = both, 4 = no raw loads only,
# 8 = no prep only; timing only, wrong results). DEV=0 / EXP=0 skip a part; DEV_ENVS / EXP_ENVS override the env lists.
#   fleetq run --host spark1 --mem 8 --label "..." -- scripts/cuda-round6/gdn-ab.sh
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
S="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[[ "${DEV:-1}" == 1 ]] && TREE=r6-dev NOBUILD=1 STEPS=gdn ROUNDS=${ROUNDS:-2} GDN_ENVS="${DEV_ENVS:-INTARSIA_GDN_RING=0;INTARSIA_GDN_RING=1}" "$S/dev.sh"
if [[ "${EXP:-1}" == 1 ]]; then
  TREE=r6-exp NOBUILD=1 STEPS=gdn ROUNDS=1 GDN_ENVS="${EXP_ENVS:-INTARSIA_GDN_EXP=0;INTARSIA_GDN_EXP=1;INTARSIA_GDN_EXP=2;INTARSIA_GDN_EXP=3}" "$S/dev.sh"
fi
