#!/usr/bin/env bash
# Lane TPO generic quick window, ON tatooine inside ONE fleetq slot taking radiance down (both cards).
# sections: M (stream-overlap microbench, queues 1/2/4) T (plain timing of ARMS, 3 reps, --max-tokens 1) I (48 greedy tokens per arm, md5 of the output)
# env: ARMS="name:bin:ENV=V,ENV=V:arg,arg ..." (bin under ~/intarsiaml-tpo; envs and extra generate args comma separated, may be empty) PROMPT=4k-0 REPS=3 O=<out dir>
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
SECTIONS=${*:-M T I}; want() { [[ " $SECTIONS " == *" $1 "* ]]; }
PROMPT=${PROMPT:-4k-0}; REPS=${REPS:-3}
O=${O:-$HOME/intarsiaml-tpo-logs/w3}; mkdir -p "$O"
if [[ -z "${KEEPDOWN:-}" ]]; then trap restore EXIT; fi
systemctl --user stop radiance; sleep 3; echo "[tpo] radiance $(systemctl --user is-active radiance) $(now)"
P=$(cat $R/scripts/cmtp/ids-prompts/$PROMPT.txt)
GEN="generate -m $M27 --backend hip --tp 2 --devices 0,1 --mem-budget 29 --max-seq 8192 --spec off --ignore-eos"
if want M; then
  gpu 200 sh -c "cd /work/scripts/tpo && hipcc -O2 --offload-arch=gfx1201 overlap.hip -o /tmp/overlap 2>&1 | grep -a error; for q in 1 2 4; do GPU_MAX_HW_QUEUES=\$q /tmp/overlap; done" > "$O/overlap.txt" 2>&1
  cat "$O/overlap.txt"
fi
for arm in ${ARMS:-}; do
  IFS=: read -r name bin envs args <<< "$arm"; envs=${envs//,/ }; args=${args//,/ }
  gen=$GEN; if [[ " $args " == *" SPECON "* ]]; then gen=${GEN/--spec off/}; args=${args/SPECON/}; fi
  if want T; then
    for rep in $(seq 1 $REPS); do
      gpu 230 env $envs /work/$bin $gen $args --max-tokens 1 --profile-every 0 -p "$P" > "$O/t-$name-$rep.out" 2> "$O/t-$name-$rep.err"
      echo "T $name rep$rep: $(grep -a 'prefill ' "$O/t-$name-$rep.err" | tail -1 | cut -c1-160)"
    done
  fi
  if want I; then
    gpu 230 env $envs /work/$bin $gen $args --max-tokens 48 --profile-every 0 -p "$P" > "$O/i-$name.out" 2> "$O/i-$name.err"
    echo "I $name: md5 $(md5sum < "$O/i-$name.out" | cut -c1-12) $(grep -a 'prefill ' "$O/i-$name.err" | tail -1 | cut -c1-120)"
  fi
done
if want G; then
  # `plan show` of the 8B and the 35B, base binary on card 0, ${GBIN:-tpo-dev3} on card 1, in parallel; text must be byte-identical
  for m in 8B:/models/Qwen3-8B 35B:/models/Qwen3.6-35B-A3B-NVFP4; do
    tag=${m%%:*}; path=${m#*:}
    G=0 MODELS=/data/models/intarsia gpu 150 /work/tpo-base plan show --backend hip --model $path --max-seq 8192 > "$O/plan-$tag-base.txt" 2> "$O/plan-$tag-base.err" &
    G=1 MODELS=/data/models/intarsia gpu 150 env ${GENVS:-A=1} /work/${GBIN:-tpo-dev3} plan show --backend hip --model $path --max-seq 8192 > "$O/plan-$tag-dev.txt" 2> "$O/plan-$tag-dev.err" &
    wait
    if cmp -s "$O/plan-$tag-base.txt" "$O/plan-$tag-dev.txt" && [ -s "$O/plan-$tag-base.txt" ]; then echo "G plan $tag: byte-identical ($(wc -c < "$O/plan-$tag-base.txt") B)"; else echo "G plan $tag: DIFFER or empty"; fi
  done
fi
if want R; then
  # interleaved rounds (REPS_R runs per arm and round, order reversed every other round)
  read -ra arms <<< "$ARMS"
  for r in $(seq 1 ${ROUNDS:-6}); do
    order=("${arms[@]}")
    if (( r % 2 == 0 )); then order=(); for ((k=${#arms[@]}-1; k>=0; k--)); do order+=("${arms[k]}"); done; fi
    for arm in "${order[@]}"; do
      IFS=: read -r name bin envs args <<< "$arm"; envs=${envs//,/ }; args=${args//,/ }
      gpu 230 env $envs /work/$bin $GEN $args --max-tokens 1 --profile-every 0 -p "$P" > "$O/r-$name-$r.out" 2> "$O/r-$name-$r.err"
      echo "R $name r$r: $(grep -a 'prefill ' "$O/r-$name-$r.err" | tail -1 | cut -c1-110)"
    done
  done
fi
echo "[tpo] w3 done $(now)"
