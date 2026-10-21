#!/usr/bin/env bash
# Read-only: where spark1's memory is (fleetq reports memory it does not track). No GPU work.
#   fleetq run --host spark1 --mem 0 --label "..." -- scripts/cuda-round7/memcheck.sh
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
ssh -o BatchMode=yes "${HOST:-spark1}" "FLEETQ_JOB=$FLEETQ_JOB bash -s" <<'EOF'
free -g
grep -E 'MemAvailable|^Cached|Shmem:|SReclaimable|Unevictable|Mlocked' /proc/meminfo
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
ps -eo pid,user,rss,etime,args --sort=-rss | head -12 | cut -c1-200
docker stats --no-stream --format '{{.Name}} {{.MemUsage}}' 2>/dev/null
df -h /dev/shm | tail -1
EOF
