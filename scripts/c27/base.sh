#!/usr/bin/env bash
# Lane c27, ON spark1 inside a fleetq quiet slot: decode baseline of the 27B (CUDA tp1) with `generate`: sizes 1k/4k/16k x {plain, mtp k2}, then nsys profiles.
#   fleetq run --host spark1 --quiet gpu,cpu --mem 50 --expect 25 --label "c27 baseline" -- ssh spark1 'TREE=src bash -s' < scripts/c27/base.sh
# Env: TREE (src), MODE list, SIZES (1k 4k 16k), NTOK (192), NSYS (1: also profile 4k plain + mtp), ENVX (extra env for the binary), O (out dir)
set -uo pipefail
export PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH
R=$HOME/intarsia-c27; T=${TREE:-src}; B=$R/target-$T/release/intarsia; M=$R/models/Qwen3.8-27B-Quark-AWQ-MXFP4
O=${O:-$R/out/base}; mkdir -p $O
CORP=$R/$T/tools/prompts/corpus_docs.txt
chars() { case $1 in 1k) echo 3300;; 4k) echo 14035;; 16k) echo 54000;; esac; }
gen() { # gen <size> <mode> <tag> [pre-cmd...]
  local sz=$1 mode=$2 tag=$3 spec; shift 3
  case $mode in plain) spec="--spec off";; mtp) spec="--spec mtp --spec-k 2";; mtp3) spec="--spec mtp --spec-k 3";; esac
  head -c $(chars $sz) $CORP > $O/p-$sz.txt
  env ${ENVX:-A=1} "$@" $B generate -m $M --backend cuda --mem-budget 45 --max-seq $([ $sz = 16k ] && echo 20480 || echo 8192) $spec --chat --no-think --temperature 0 \
    --profile-every 0 --ignore-eos --show-ids -p "$(cat $O/p-$sz.txt)" --max-tokens ${NTOK:-192} < /dev/null > $O/g-$sz-$mode-$tag.out 2> $O/g-$sz-$mode-$tag.err
  echo "$sz $mode $tag: $(grep -a 'finish=' $O/g-$sz-$mode-$tag.err | tail -1 | cut -c1-200)"
}
echo "free GB: $(free -g | awk '/Mem:/ {print $7}')  $(date +%T)"
vg() { curl -s -m 5 http://100.120.88.104:8000/metrics | grep -a '^vllm:generation_tokens_total' | awk '{print $2}'; }
echo "vllm gen tokens before: $(vg)"
for sz in ${SIZES:-1k 4k 16k}; do for mode in ${MODES:-plain mtp mtp3}; do [ $sz = 16k ] && [ $mode = mtp3 ] && continue; gen $sz $mode a; free -g | awk '/Mem:/ {print "  avail " $7}'; done; done
if [ "${NSYS:-1}" = 1 ]; then
  nsz=${NSZ:-4k}
  for mode in ${NMODES:-plain mtp}; do
    spec="--spec off"; [ $mode = mtp ] && spec="--spec mtp --spec-k 2"
    head -c $(chars $nsz) $CORP > $O/p-$nsz.txt
    env ${ENVX:-A=1} nsys profile -t cuda --cuda-graph-trace=node --force-overwrite true -o $O/nsys-$mode-$nsz $B generate -m $M --backend cuda --mem-budget 45 --max-seq $([ $nsz = 16k ] && echo 20480 || echo 8192) $spec \
      --chat --no-think --temperature 0 --ignore-eos --profile-every 0 -p "$(cat $O/p-$nsz.txt)" --max-tokens ${NTOK:-192} < /dev/null > $O/n-$mode-$nsz.out 2> $O/n-$mode-$nsz.err
    nsys stats -r cuda_gpu_trace --format csv --output $O/tr-$mode-$nsz $O/nsys-$mode-$nsz.nsys-rep > /dev/null 2>&1
    echo "nsys $mode $nsz: $(grep -a 'finish=' $O/n-$mode-$nsz.err | tail -1 | cut -c1-200)"; ls $O | grep "tr-$mode-$nsz"
  done
fi
echo "vllm gen tokens after: $(vg)"
echo "done $(date +%T)"
