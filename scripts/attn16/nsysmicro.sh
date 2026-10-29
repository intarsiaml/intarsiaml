#!/usr/bin/env bash
# Lane attn16 short check: nsys kernel durations of timing_attn_rows (M = 16) per env variant (each "A=1,B=2" in $VARS).
#   fleetq run --host spark2 --quiet gpu,cpu --mem 8 --expect 3 --label "attn16: nsys micro" -- scripts/attn16/nsysmicro.sh [bin]
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
B=${1:-dev}
for v in ${VARS:--}; do
  ssh -n -o BatchMode=yes spark2 "export FLEETQ_JOB=$FLEETQ_JOB; cd ~/intarsiaml-attn16/src/crates/intarsia-backend-cuda; mkdir -p ~/intarsiaml-attn16/out
    [ '$v' != - ] && export ${v//,/ }; ATTNT_M=16 ATTNT_POS=${ATTNT_POS:-1100} nsys profile -t cuda --force-overwrite true -o ~/intarsiaml-attn16/out/m ~/intarsiaml-attn16/bin/row_tables-$B --ignored --exact timing_attn_rows --test-threads=1 --nocapture >/dev/null 2>&1
    nsys stats -r cuda_gpu_kern_sum -f csv ~/intarsiaml-attn16/out/m.nsys-rep 2>/dev/null | grep -E 'splitkv|combine' | awk -F, '{printf \"%s calls avg %.1f us med %.1f  %s\n\", \$3, \$4/1000, \$5/1000, substr(\$9,1,50)}' | sed 's/^/[$v] /'
    rm -f ~/intarsiaml-attn16/out/m.*"
done
