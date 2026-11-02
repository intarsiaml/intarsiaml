#!/usr/bin/env bash
# Lane k4fp8 (from lane hfp8): microbench window, driven FROM THE MAC under one fleetq job that holds both R9700 cards and takes radiance down
# (guard.sh of lane hprof stops radiance and restores it when this script ends or dies):
#   RUNS="base:INTARSIA_HIP_FP8_ONEPASS_MIN=99 new:INTARSIA_HIP_FP8_ONEPASS_MIN=1" BIN=bench-a OUT=<dir> \
#     fleetq run --host tatooine-r9700,tatooine-r9700-2 --takes-down radiance --quiet gpu,cpu --expect 10 --mem 30 \
#       --label "K4FP8 fp8 rows microbench" -- scripts/k4fp8/mbwin.sh
# RUNS: space separated tag[@binary]:ENV=v;ENV2=v2 (env pairs separated by ';', so values may hold commas); each runs k4fp8-out/$BIN once on card 0 with
# the BENCH_* variables of this script's environment passed through (BENCH_SHAPES, BENCH_MS, BENCH_MODE, BENCH_REPS, BENCH_ACTQ).
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
BIN=${BIN:-bench-a}; TREE=${TREE:-k4fp8}; CAP=${CAP:-540}
IMG=intarsia-hip-dev:rocm7.2.4-rust1.90.0
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; REPO="$(cd "$HERE/../.." && pwd)"
OUT=${OUT:?OUT dir}; mkdir -p "$OUT"
T0=$(date +%s); left() { echo $(( CAP - ($(date +%s) - T0) )); }
say() { echo "$(date +%T) $*" | tee -a "$OUT/steps.log"; }
SSH="ssh -o BatchMode=yes tatooine"
FIFO="$OUT/guard.fifo"; mkfifo "$FIFO"
$SSH 'mkdir -p ~/hprof' < /dev/null; scp -q "$REPO/scripts/hprof/guard.sh" tatooine:hprof/guard.sh
$SSH 'bash ~/hprof/guard.sh' > "$OUT/guard.log" 2>&1 < "$FIFO" &
GUARD=$!
exec 9> "$FIFO"
for i in $(seq 1 30); do grep -q "radiance stopped" "$OUT/guard.log" 2>/dev/null && break; sleep 1; done
say "guard: $(grep -h 'radiance stopped' "$OUT/guard.log")"
cleanup() {
  exec 9>&-; wait $GUARD; rm -f "$FIFO"
  say "guard: $(grep -h 'stopping radiance\|radiance answers\|DID NOT' "$OUT/guard.log" | tr '\n' '|')"
  echo "radiance down: $(grep -h 'stopping radiance' "$OUT/guard.log" | sed 's/.*radiance //') -> $(grep -h 'radiance answers\|DID NOT' "$OUT/guard.log" | sed 's/.*\] //')  (fleetq job $FLEETQ_JOB)" >> "$OUT/downtime.log"
}
trap cleanup EXIT

benv=""; for v in $(compgen -e | grep -E '^BENCH_'); do benv="$benv -e $v=${!v}"; done
for run in ${RUNS:?RUNS}; do
  tag=${run%%:*}; envs=""; bin=$BIN
  if [[ $tag == *@* ]]; then bin=${tag#*@}; tag=${tag%%@*}; fi
  if [[ $run == *:* ]]; then for kv in $(echo "${run#*:}" | tr ';' ' '); do envs="$envs -e $kv"; done; fi
  if (( $(left) < 60 )); then say "SKIPPED $tag (time left $(left)s)"; continue; fi
  say "$tag: start ($envs)"
  $SSH "timeout 230 docker run --rm --device /dev/kfd --device /dev/dri --group-add \$(stat -c %g /dev/kfd) --group-add \$(stat -c %g /dev/dri/renderD128) --group-add video --ipc host \
    --security-opt seccomp=unconfined --user \$(id -u):\$(id -g) -e HOME=/tmp -e GPU_MAX_HW_QUEUES=1 -e HIP_VISIBLE_DEVICES=0 $benv $envs \
    -v \$HOME/intarsiaml-tp2-$TREE:/work -w /work $IMG /work/k4fp8-out/$bin" < /dev/null > "$OUT/$tag.txt" 2>&1
  say "$tag: exit $? ($(grep -c . "$OUT/$tag.txt") lines)"
done
say "done, time left $(left)s"
