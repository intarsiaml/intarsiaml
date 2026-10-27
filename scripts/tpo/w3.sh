#!/usr/bin/env bash
# Lane TPO generic quick window, ON tatooine inside ONE fleetq slot taking radiance down (both cards).
# sections: M (stream-overlap microbench, queues 1/2/4) T (plain timing of ARMS, 3 reps, --max-tokens 1) I (48 greedy tokens per arm, md5 of the output)
# env: ARMS="name:bin:ENV=V,ENV=V:arg,arg ..." (bin under ~/intarsiaml-tpo; envs and extra generate args comma separated, may be empty) PROMPT=4k-0 REPS=3 O=<out dir>
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
SECTIONS=${*:-M T I}; want() { [[ " $SECTIONS " == *" $1 "* ]]; }
PROMPT=${PROMPT:-4k-0}; REPS=${REPS:-3}
O=${O:-$HOME/intarsiaml-tpo-logs/w3}; mkdir -p "$O"
trap restore EXIT
systemctl --user stop radiance; sleep 3; echo "[tpo] radiance $(systemctl --user is-active radiance) $(now)"
P=$(cat $R/scripts/cmtp/ids-prompts/$PROMPT.txt)
GEN="generate -m $M27 --backend hip --tp 2 --devices 0,1 --mem-budget 29 --max-seq 8192 --spec off --ignore-eos"
if want M; then
  gpu 200 sh -c "cd /work/scripts/tpo && hipcc -O2 --offload-arch=gfx1201 overlap.hip -o /tmp/overlap 2>&1 | grep -a error; for q in 1 2 4; do GPU_MAX_HW_QUEUES=\$q /tmp/overlap; done" > "$O/overlap.txt" 2>&1
  cat "$O/overlap.txt"
fi
for arm in ${ARMS:-}; do
  IFS=: read -r name bin envs args <<< "$arm"; envs=${envs//,/ }; args=${args//,/ }
  if want T; then
    for rep in $(seq 1 $REPS); do
      gpu 230 env $envs /work/$bin $GEN $args --max-tokens 1 --profile-every 0 -p "$P" > "$O/t-$name-$rep.out" 2> "$O/t-$name-$rep.err"
      echo "T $name rep$rep: $(grep -a 'prefill ' "$O/t-$name-$rep.err" | tail -1 | cut -c1-160)"
    done
  fi
  if want I; then
    gpu 230 env $envs /work/$bin $GEN $args --max-tokens 48 --profile-every 0 -p "$P" > "$O/i-$name.out" 2> "$O/i-$name.err"
    echo "I $name: md5 $(md5sum < "$O/i-$name.out" | cut -c1-12) $(grep -a 'prefill ' "$O/i-$name.err" | tail -1 | cut -c1-120)"
  fi
done
echo "[tpo] w3 done $(now)"
