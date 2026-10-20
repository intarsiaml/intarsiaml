#!/usr/bin/env bash
# SASS of one kernel of a built tree on spark1 (CPU only), op counts of its hottest loop body to stdout, full listing kept
# in ~/intarsia-decfuse/out/r6sass/<obj>-<tag>.sass:   fleetq run --host spark1 --mem 0 -- scripts/cuda-round6/sass.sh <tree> <obj> <function substring> [tag]
set -euo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
TREE=${1:?tree}; OBJ=${2:?object, e.g. gdn.o}; FN=${3:?function substring}; TAG=${4:-$TREE}
ssh -n -o BatchMode=yes spark1 "export PATH=/usr/local/cuda/bin:\$PATH FLEETQ_JOB=$FLEETQ_JOB; mkdir -p ~/intarsia-decfuse/out/r6sass; cd ~/intarsia-decfuse/$TREE && d=\$(ls -td target/acc/release/build/intarsia-backend-cuda-*/out | head -1) && f=~/intarsia-decfuse/out/r6sass/${OBJ%.o}-$TAG.sass && cuobjdump -sass \$d/$OBJ | awk -v fn='$FN' '/Function :/{p = index(\$0, fn) > 0} p' > \$f && echo \"\$(wc -l < \$f) lines\" && grep -oE '^ +/\\*[0-9a-f]{4}\\*/ +[A-Z0-9_.]+' \$f | awk '{print \$2}' | sed -E 's/\\..*//' | sort | uniq -c | sort -rn | head -30"
