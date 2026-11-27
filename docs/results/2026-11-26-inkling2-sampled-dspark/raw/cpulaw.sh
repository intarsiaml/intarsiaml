#!/usr/bin/env bash
# ON tatooine, --load cpu slot (lane ink-sampled): build the lane's fixture test (br3: law split per configuration, binned cells) and run
# the three fixture law tests on the cpu backend (INTARSIA_INKLING2_BACKEND unset), in parallel test threads.
set -uo pipefail
L=$HOME/intarsiaml-ink-sampled; T=$L/.scratch/tat; OUT=$L/logs/cpulaw; mkdir -p $OUT
NOMAIN=1 BR=br3 MTESTS="inkling2_dspark_sampled" bash $T/build_both.sh > $OUT/build.log 2>&1 || { tail -5 $OUT/build.log; exit 1; }
grep -E "copied|exit" $OUT/build.log
echo "[cpulaw] run $(date +%T)"
INK_CONTAINER=ink-sampled-cpu bash $T/run.sh env INTARSIA_LAW_N=${LAW_N:-1500} /work/.scratch/bin/t-br3-inkling2_dspark_sampled --ignored law --nocapture > $OUT/law.txt 2>&1
echo "[cpulaw] rc $? $(date +%T)"
grep -aE "chi2|Fisher|test result|panicked|^test " $OUT/law.txt
