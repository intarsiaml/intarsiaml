#!/usr/bin/env bash
# Lane spark35: the KL arms (intarsia accuracy, self-golden against the base arm's own dump) on $HOST. Source of the pass rule:
# docs/results/2026-11-05-spark35-numerics.md "Pass rules". Arms (name:binary:env:extra accuracy args, spaces as _):
#   base  : bf16 KV, f32 state, chunk 2048 (writes the reference dumps)
#   noise : base with --prefill-chunk 1024 (the same-window benign perturbation: bf16 vs bf16, another chunk schedule)
#   fp8   : --kv-dtype fp8                         (D)
#   cemu  : INTARSIA_CUDA_GDN_STATE_BF16_EMU=1      (C, emulated bf16 state)
#   cdemu : both                                   (C+D)
#   creal : the real bf16 state layout (INTARSIA_GDN_STATE=bf16): must equal cemu (same values, half the bytes); cdreal: with fp8 KV
# Two carriers (scripts/spark35num/mkcarrier.py): std (5 prompts 3.9k..65k: prefill, all positions, + a 2048-token teacher-forced decode
# window) and dec8k (16k, 32k: an 8192-token decode window). Needs build.sh to have made bin/dev; carriers in $L/carriers.
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
ARMS=${ARMS:-"base noise fp8 cemu cdemu"}
LOOSE="--allow-config-mismatch --allow-unstamped --allow-same-code --self-max-kl 1 --self-max-pos-kl 100 --self-max-flip-pct 100 --self-max-tail 1"
spec() { case $1 in
  base) echo "dev||" ;; noise) echo "dev||--prefill-chunk 1024" ;; fp8) echo "dev||--kv-dtype fp8" ;;
  cemu) echo "dev|INTARSIA_CUDA_GDN_STATE_BF16_EMU=1|" ;; cdemu) echo "dev|INTARSIA_CUDA_GDN_STATE_BF16_EMU=1|--kv-dtype fp8" ;;
  creal) echo "dev|INTARSIA_GDN_STATE=bf16|" ;; cdreal) echo "dev|INTARSIA_GDN_STATE=bf16|--kv-dtype fp8" ;; esac; }
run_acc() { # run_acc <set std|dec8k> <arm>
  local set=$1 arm=$2 bin envs extra gold mode dp ms sg
  IFS='|' read -r bin envs extra <<< "$(spec "$arm")"
  case $set in std) gold=std; mode=both; dp=2048; ms=70000 ;; dec8k) gold=dec8k; mode=decode; dp=8192; ms=34000 ;; esac
  if [[ $arm == base ]]; then sg="--write-self-golden $L/out/dump-$set-base.st --allow-unstamped"; else sg="--self-golden $L/out/dump-$set-base.st $LOOSE"; fi
  echo "== kl $set $arm $(date +%T)"
  rsh "cd $L/out && rm -f pos-$set-$arm.tsv && env $envs INTARSIA_ACC_POS_DUMP=$L/out/pos-$set-$arm.tsv $L/bin/$bin accuracy --model $M35 --backend cuda --golden $L/carriers/$gold.safetensors --mode $mode --decode-positions $dp --max-seq $ms --mem-budget ${MEMB:-45} $extra $sg > $L/out/acc-$set-$arm.log 2>&1; echo exit \$?; grep -a -E 'SELF-STATS|^wrote self-golden|^Error' $L/out/acc-$set-$arm.log | cut -c1-300"
  rsh "cd $L/out && gzip -kf pos-$set-$arm.tsv; ls -la acc-$set-$arm.log pos-$set-$arm.tsv.gz | awk '{print \$5, \$9}'"
  scp -q -o BatchMode=yes "$HOST:intarsiaml-s35/out/acc-$set-$arm.log" "$HOST:intarsiaml-s35/out/pos-$set-$arm.tsv.gz" "$RES/" 2>&1 | tail -1
}
for arm in $ARMS; do
  fits 14 "kl $arm" || continue
  run_acc std $arm
  [[ ${SKIP_DEC8K:-0} = 1 || $arm = creal || $arm = cdreal ]] || run_acc dec8k $arm
done
echo "== kl done $(date +%T)"
