#!/usr/bin/env bash
# Lane mp: nsys kernel summary of a bench binary built by dev.sh. Usage (in a fleetq slot): nsys.sh <host> <variant> <T> [skew] [extra nsys args]
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq}"
HOST=$1; V=$2; T=$3; SKEW=${4:-0}; shift 4 2>/dev/null || shift $#
TAG=${ONLY:-prod}
ssh -n -o BatchMode=yes "$HOST" "export FLEETQ_JOB=$FLEETQ_JOB PATH=/usr/local/cuda/bin:\$PATH; cd ~/mp; rm -f prof_${V}_$TAG.*; \
  ONLY=${ONLY:-prod} nsys profile -t cuda $* --force-overwrite true -o prof_${V}_$TAG bin/bench_$V $T 6 $SKEW > prof_${V}_$TAG.out 2>&1; \
  nsys stats -r cuda_gpu_kern_sum -f csv prof_${V}_$TAG.nsys-rep 2>/dev/null | python3 -c '
import csv,sys
rows=list(csv.reader(l for l in sys.stdin if l.strip() and not l.startswith(\"Generating\") and not l.startswith(\"Processing\")))
h=rows[0]; ti=h.index(\"Total Time (ns)\"); ni=h.index(\"Instances\"); na=h.index(\"Name\")
for r in rows[1:]:
    print(\"%10.1f us/inst  x%-4s %s\" % (int(r[ti])/int(r[ni])/1e3, r[ni], r[na][:110]))
'"
