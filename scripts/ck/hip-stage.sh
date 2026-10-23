#!/usr/bin/env bash
# Runs ON tatooine inside og-burst.sh (radiance already down): one stage of lane ck on the R9700s.
#   hip-stage.sh acc|dec|speed35|speed8 <outdir under ~/ck-hip>
#   acc   : 35B on GPU 0 and Qwen3-8B on GPU 1 AT THE SAME TIME (numerics only, timing is irrelevant), 3 chunks each
#   speed35 / speed8 : one GPU (0), that model, a fresh serve per chunk, no drift-control repeat (window cap)
set -uo pipefail
STAGE=${1:?acc|dec|speed35|speed8}; OUT=${2:?outdir}
mkdir -p "$OUT"
C=$HOME/ck-hip/scripts/ck/chunk-job.sh
case $STAGE in
  acc)
    MODELS_RUN=35b RS_GPU=0 bash "$C" hip "$OUT/g0" acc < /dev/null > "$OUT/g0.log" 2>&1 &
    p0=$!
    MODELS_RUN=8b RS_GPU=1 bash "$C" hip "$OUT/g1" acc < /dev/null > "$OUT/g1.log" 2>&1 &
    p1=$!
    wait $p0; echo "gpu0 (35B) exit $?"; wait $p1; echo "gpu1 (8B) exit $?"
    cat "$OUT/g0/steps.log" "$OUT/g1/steps.log" ;;
  dec)
    MODELS_RUN=35b RS_GPU=0 bash "$C" hip "$OUT/g0" "gen accdec" < /dev/null > "$OUT/g0.log" 2>&1 &
    p0=$!
    MODELS_RUN=8b RS_GPU=1 bash "$C" hip "$OUT/g1" "gen accdec" < /dev/null > "$OUT/g1.log" 2>&1 &
    p1=$!
    wait $p0; echo "gpu0 (35B) exit $?"; wait $p1; echo "gpu1 (8B) exit $?"
    cat "$OUT/g0/steps.log" "$OUT/g1/steps.log" ;;
  speed35|speed8)
    SPEED_CHUNKS="2048 4096 8192" MODELS_RUN=${STAGE#speed}b RS_GPU=0 bash "$C" hip "$OUT/$STAGE" speed < /dev/null 2>&1 ;;
esac
