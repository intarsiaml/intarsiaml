#!/usr/bin/env bash
LANE="$HOME/intarsiaml-ink-hip-pfflip"
echo "[phaseA2] build branch $(date +%T)"
TESTS="ink_moe ink_attn" LIBTESTS=1 bash "$LANE/.scratch/build.sh" pf || exit $?
echo "[phaseA2] done $(date +%T)"
