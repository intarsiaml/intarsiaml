#!/bin/zsh
# copyflipm memory cost of the default, inside a fleetq slot on endor: peak memory footprint of a short `generate` with the default (copy
# drafts on, 15 per-row recurrent-state buffers) and with INTARSIA_SPEC_COPY=0 (2).
[ -n "${FLEETQ_JOB:-}" ] || { echo "not inside a fleetq slot" >&2; exit 2; }
W=${0:A:h:h:h}; RAW=$W/docs/results/2026-11-18-copyflipm/raw/mem; mkdir -p $RAW
NEW=$W/target-cf/release/intarsia; M=$HOME/models/Qwen3.6-35B-A3B-NVFP4; CELLS=$W/target-cf/cells
for r in 1 2; do for arm in zero def; do
  E=(X=1); [ $arm = zero ] && E=(INTARSIA_SPEC_COPY=0)
  env $E /usr/bin/time -l $NEW generate -m $M --backend metal --max-seq 8192 --spec mtp --spec-k 2 --chat --no-think --temperature 0 --profile-every 0 -p "$(cat $CELLS/code-gen.txt)" --max-tokens 64 < /dev/null > /dev/null 2> $RAW/$arm-$r.err
  echo "$arm r$r: $(grep -aE 'maximum resident|peak memory footprint' $RAW/$arm-$r.err | tr -s ' ' | tr '\n' ';')"
done; done
