#!/usr/bin/env bash
# Background stage (lane ink-hip-bpf) on card 1 during a serve window (serve_window.sh BG=): G2 short / long (accuracy --mode
# prefill, oracle W4A16 goldens) for main (80719483a) and the branch, then admission of the touched ops on the branch binary.
# Ends by itself within BG_CAP seconds (default 200).
set -uo pipefail
REPO=$HOME/intarsiaml-ink-hip-bpf
S=$REPO/.scratch
M=/models/Inkling-2-Mini
MA=${MA:-main}
BR=${BR:-b5}
now() { date '+%F %T'; }
end=$(( $(date +%s) + ${BG_CAP:-200} ))
left() { echo $(( end - $(date +%s) )); }
run1() { HIP_VISIBLE_DEVICES=1 INK_CONTAINER=ink-hip-bpf-run2 timeout $(( $(left) > 5 ? $(left) - 5 : 1 )) bash "$S/run.sh" "$@"; }
for b in $BR $MA; do
  for g in short long; do
    (( $(left) < 25 )) && { echo "G2 $g $b SKIP ($(left) s left)"; continue; }
    echo "G2 $g $b [$(now)] $(run1 /work/.scratch/bin/intarsia-$b accuracy -m $M --backend hip --max-seq 4096 --mode prefill \
      --golden golden/oracle/inkling2-mini-oracle-w4a16-$g-top20.safetensors 2>&1 | grep -aE "prefill floor|floor verdict|Error|panicked" | cut -c1-300 | tr '\n' ' ')"
  done
done
for op in conv_mix moe_experts moe_experts_mxfp8; do
  (( $(left) < 20 )) && { echo "ADMIT $op SKIP ($(left) s left)"; continue; }
  echo "ADMIT $op [$(now)]"
  run1 /work/.scratch/bin/intarsia-$BR kernels test --backend hip --op $op 2>&1 | grep -aE "ADMITTED|REJECT|FAIL|NO CASES|Error|panicked" | head -8 | sed "s/^/ADMIT $op: /"
done
docker rm -f ink-hip-bpf-run2 >/dev/null 2>&1
echo "[bg] done $(now)"
