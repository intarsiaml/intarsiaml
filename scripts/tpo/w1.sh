#!/usr/bin/env bash
# Lane TPO window 1, ON tatooine inside ONE fleetq slot taking radiance down (both cards): 4k prefill profile on main at tp2.
# sections: M (stream-overlap microbench, queues 1/2/4) T (plain timing: recipe default, f32 partials, no-comm) O (engine op table) P (rocprofv3 kernel trace)
# env: BIN=tpo-base PROMPT=4k-0 O=<out dir>
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
SECTIONS=${*:-M T O P}; want() { [[ " $SECTIONS " == *" $1 "* ]]; }
BIN=${BIN:-tpo-base}; PROMPT=${PROMPT:-4k-0}
O=${O:-$HOME/intarsiaml-tpo-logs/w1}; P_OUT=${P_OUT:-p}; mkdir -p "$O"
trap restore EXIT
systemctl --user stop radiance; sleep 3; echo "[tpo] radiance $(systemctl --user is-active radiance) $(now)"
P=$(cat $R/scripts/cmtp/ids-prompts/$PROMPT.txt)
GEN="generate -m $M27 --backend hip --tp 2 --devices 0,1 --mem-budget 29 --max-seq 8192 --spec off --ignore-eos --max-tokens 1"
if want M; then
  for q in 1 2 4; do
    gpu 200 sh -c "cd /work/scripts/tpo && hipcc -O2 --offload-arch=gfx1201 overlap.hip -o /tmp/overlap 2>/dev/null && HIP_VISIBLE_DEVICES=0 GPU_MAX_HW_QUEUES=$q /tmp/overlap" > "$O/overlap-q$q.txt" 2>&1
    cat "$O/overlap-q$q.txt"
  done
fi
if want T; then
  for arm in rec f32 nocomm; do
    envs=""; case $arm in f32) envs="INTARSIA_TP_PARTIAL=f32" ;; nocomm) envs="INTARSIA_TP_NO_COMM=1" ;; esac
    for rep in 1 2 3; do
      gpu 230 env $envs /work/$BIN $GEN --profile-every 0 -p "$P" > "$O/t-$arm-$rep.out" 2> "$O/t-$arm-$rep.err"
      echo "T $arm rep$rep: $(grep -a 'prefill ' "$O/t-$arm-$rep.err" | tail -1 | cut -c1-160)"
    done
  done
fi
if want O; then
  gpu 230 /work/$BIN $GEN --profile-every 1 --op-table -p "$P" > "$O/o-rec.out" 2> "$O/o-rec.err"
  echo "O rec: $(grep -a 'prefill ' "$O/o-rec.err" | tail -1 | cut -c1-160)"
fi
if want P; then
  d=/work/tpo-prof; rm -rf "$R/tpo-prof"; mkdir -p "$R/tpo-prof"
  gpup 235 rocprofv3 --kernel-trace --output-format csv -d $d -o run -- env ${ENVS:-A=1} /work/$BIN $GEN --profile-every 0 -p "$P" > "$O/$P_OUT.out" 2> "$O/$P_OUT.err"
  csv=$(find "$R/tpo-prof" -name '*kernel_trace.csv' | head -1)
  if [[ -n "$csv" ]]; then gzip -c "$csv" > "$O/$P_OUT.csv.gz"; fi
  rm -rf "$R/tpo-prof"
  echo "P: $(grep -a 'prefill ' "$O/$P_OUT.err" | tail -1 | cut -c1-160) csv=$(ls -s $O/$P_OUT.csv.gz 2>/dev/null || echo NO)"
fi
echo "[tpo] w1 done $(now)"
