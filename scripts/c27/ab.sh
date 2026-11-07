#!/usr/bin/env bash
# Lane c27, ON spark1 inside ONE fleetq quiet slot: paired ABAB decode A/B with `generate` (greedy, --ignore-eos, corpus prompts), per visit one fresh process.
#   fleetq run --host spark1 --quiet gpu,cpu --mem 50 --expect 28 --label "..." -- ssh spark1 'ARMS="a b" bash -s' < scripts/c27/ab.sh
# Env: ARMS "name ..." with ARM_<name>="<tree>|<ENV=.. ENV2=..>|<generate spec args>" (default: base = src tree, quant = dev tree + INTARSIA_MTP_QUANT=fp8, both --spec mtp --spec-k 2)
#      SIZES (4k), ROUNDS (3), NTOK (192), O (out dir). Round r runs the arms in reversed order against round r-1.
#      IDS=1: instead run the ids gate: `--spec off` on the first arm's tree vs every arm, prompts IDSP (short-0 4k-1), 256 ids.
set -uo pipefail
export PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH
R=$HOME/intarsia-c27; M=$R/models/Qwen3.8-27B-Quark-AWQ-MXFP4; O=${O:-$R/out/ab}; mkdir -p $O
CORP=$R/src/tools/prompts/corpus_docs.txt; IDP=$R/src/scripts/cmx/ids-prompts
ARMS=${ARMS:-"base quant"}
ARM_base=${ARM_base:-"src||--spec mtp --spec-k 2"}
ARM_quant=${ARM_quant:-"dev|INTARSIA_MTP_QUANT=fp8|--spec mtp --spec-k 2"}
chars() { case $1 in 1k) echo 3300;; 4k) echo 14035;; 16k) echo 54000;; esac; }
arm() { eval echo "\$ARM_$1"; }
vg() { curl -s -m 5 http://100.120.88.104:8000/metrics | grep -a '^vllm:generation_tokens_total' | awk '{print $2}'; }
run() { # run <tag> <arm> <prompt-file> <max-seq> <ntok> <extra generate args>
  local tag=$1 a=$2 pf=$3 ms=$4 nt=$5; shift 5
  IFS='|' read -r tree envs spec <<< "$(arm $a)"
  env ${envs:-A=1} $R/target-$tree/release/intarsia generate -m $M --backend cuda --mem-budget 45 --max-seq $ms $spec --chat --no-think --temperature 0 \
    --profile-every 0 --ignore-eos --show-ids "$@" -p "$(cat $pf)" --max-tokens $nt < /dev/null > $O/$tag.out 2> $O/$tag.err
  grep -a '^ids:' $O/$tag.err | head -1 > $O/$tag.ids
  echo "$tag: $(grep -a 'finish=' $O/$tag.err | tail -1 | grep -oE 'prefill [0-9]+ ms \([0-9.]+ tok/s\) \| decode [0-9]+ ms \([0-9.]+ tok/s\)') | $(grep -a 'spec:' $O/$tag.err | tail -1 | cut -c1-160)"
}
echo "free GB: $(free -g | awk '/Mem:/ {print $7}')  vllm gen tokens before: $(vg)  $(date +%T)"
if [ "${IDS:-0}" = 1 ]; then
  first=${ARMS%% *}; IFS='|' read -r t0 e0 s0 <<< "$(arm $first)"
  for p in ${IDSP:-short-0 4k-1}; do
    ms=4096; [ ${p#4k} != $p ] && ms=8192; [ ${p#16k} != $p ] && ms=20480
    pf=$IDP/$p.txt; [ -f $pf ] || pf=$R/src/scripts/cmtp/ids-prompts/$p.txt
    ARM_off="$t0|$e0|--spec off"
    run ids-$p-off off $pf $ms 256
    for a in $ARMS; do
      run ids-$p-$a $a $pf $ms 256
      cmp -s <(cut -d: -f2- $O/ids-$p-off.ids) <(cut -d: -f2- $O/ids-$p-$a.ids) && [ -s $O/ids-$p-$a.ids ] && echo "IDS $p: $a == off ($(wc -c < $O/ids-$p-$a.ids) B, md5 $(md5sum < $O/ids-$p-$a.ids | cut -c1-8))" || echo "IDS $p: $a DIFFERS from off"
    done
  done
else
  for sz in ${SIZES:-4k}; do
    head -c $(chars $sz) $CORP > $O/p-$sz.txt
    ms=8192; [ $sz = 16k ] && ms=20480
    for r in $(seq 1 ${ROUNDS:-3}); do
      order="$ARMS"; [ $((r % 2)) = 0 ] && order=$(echo $ARMS | tr ' ' '\n' | tac | tr '\n' ' ')
      for a in $order; do run r$r-$sz-$a $a $O/p-$sz.txt $ms ${NTOK:-192}; done
    done
  done
fi
echo "vllm gen tokens after: $(vg)  $(date +%T)"
