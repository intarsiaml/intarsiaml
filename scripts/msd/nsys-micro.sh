#!/usr/bin/env bash
# Lane msd: nsys kernel summary of one (routing, T) case of the timing_moe_rows micro test (spark1); extra env after the name.
#   fleetq run --host spark1 --quiet gpu,cpu --mem 10 --expect 6 --label "msd: nsys micro" -- scripts/msd/nsys-micro.sh <routing> <T> <outname> [ENV=V ...]
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
HOST=${HOST:-spark1}; RT=$1; T=$2; N=$3; shift 3
ssh -n -o BatchMode=yes "$HOST" "export PATH=/usr/local/cuda/bin:\$PATH FLEETQ_JOB=$FLEETQ_JOB INTARSIA_REQUIRE_GPU=1 MSD_ROUTING=$RT MSD_T=$T $*; cd ~/intarsiaml-msd2/src && X=\$(ls -t target/release/deps/rows_exact-* | grep -v '\.d\$' | head -1); nsys profile -t cuda --force-overwrite true -o /tmp/msd2-micro-$N \$X --ignored --nocapture timing_moe_rows > /dev/null 2>&1; nsys stats -r cuda_gpu_kern_sum -f csv /tmp/msd2-micro-$N.nsys-rep 2>/dev/null | awk -F, 'NR>1{printf \"%s calls avg_us %.1f  %s\\n\", \$3, \$4/1000, substr(\$9,1,70)}' | grep -E 'moe_' | head -12; rm -f /tmp/msd2-micro-$N.*"
