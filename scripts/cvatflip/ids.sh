#!/usr/bin/env bash
# cvatflip gate (a) on spark1 (35B NVFP4, CUDA, tp=1, `generate`): greedy ids per arm for each prompt; every arm must equal `off`.
# Arms (ARMS, default "off def zero"): off = --spec off; def = --spec mtp --spec-k 2, knob unset (the new default: on); zero = the same with
# INTARSIA_CUDA_VERIFY_ATTN_ROWS=0; defcopy=<n>[,<len>] / zerocopy=<n>[,<len>] = the same with INTARSIA_SPEC_COPY=<n>[,<len>] (the copy passes are counted from the trace).
# INTARSIA_SPEC_TRACE=1 is on in every spec arm. Prompts: names of
# $R/cells/<name>.txt (scripts/mcopy/prompts.py) or of scripts/cmtp/ids-prompts (short-0, 4k-0, 16k-0: 256 ids, --ignore-eos).
#   fleetq run --host spark1 --quiet gpu,cpu --mem 50 --expect 40 --label "cvatflip ids" -- ssh spark1 'ARMS="off def zero" bash -s chat-short code-gen' < scripts/cvatflip/ids.sh
set -uo pipefail
export PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH
R=$HOME/intarsiaml-cvatflip
H=/home/nvidia/.cache/huggingface/hub
M35=$(ls -d $H/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/* | head -1)
BIN=${BIN:-$R/bin/intarsia}; O=${O:-$R/out/ids}; mkdir -p "$O"; fails=0
python3 $R/src/scripts/mcopy/prompts.py $R/cells > /dev/null
for n in "$@"; do
  if [ -f $R/cells/$n.txt ]; then P=$R/cells/$n.txt; IG=""
    case $n in copy3k) MS=8192; NT=3800 ;; json-xform) MS=8192; NT=2600 ;; *) MS=4096; NT=500 ;; esac
  else
    P=$R/src/scripts/cmtp/ids-prompts/$n.txt; IG="--ignore-eos"; NT=256
    case $n in short*) MS=4096 ;; 4k*) MS=8192 ;; *) MS=20480 ;; esac
  fi
  NT=${NTOK:-$NT}
  for arm in ${ARMS:-off def zero}; do
    E=""; case $arm in off) A="--spec off" ;; def) A="--spec mtp --spec-k 2"; E="INTARSIA_SPEC_TRACE=1" ;; zero) A="--spec mtp --spec-k 2"; E="INTARSIA_CUDA_VERIFY_ATTN_ROWS=0 INTARSIA_SPEC_TRACE=1" ;;
      defcopy=*) A="--spec mtp --spec-k 2"; E="INTARSIA_SPEC_COPY=${arm#defcopy=} INTARSIA_SPEC_TRACE=1" ;; zerocopy=*) A="--spec mtp --spec-k 2"; E="INTARSIA_CUDA_VERIFY_ATTN_ROWS=0 INTARSIA_SPEC_COPY=${arm#zerocopy=} INTARSIA_SPEC_TRACE=1" ;; esac
    f=$O/gen-$n-${arm//[=,]/_}
    env $E $BIN generate -m $M35 --backend cuda --mem-budget 40 --max-seq $MS $A --chat --no-think --temperature 0 --profile-every 0 $IG --show-ids \
      -p "$(cat $P)" --max-tokens $NT < /dev/null > $f.out 2> $f.err
    grep -a '^ids:' $f.err > $f.ids
  done
  for arm in ${ARMS:-off def zero}; do
    [ "$arm" = off ] && continue
    f=$O/gen-$n-${arm//[=,]/_}; extra=""
    cp=$(grep -ac '^spec: copy token' $f.err); acc=$(grep -a '^spec: copy token' $f.err | sed 's/.*accepted \([0-9]*\) emits.*/\1/' | sort -n | uniq -c | tr '\n' ' ')
    [[ $arm == defcopy=* || $arm == zerocopy=* ]] && extra=" copy_passes=$cp accepted_hist[count acc]=$acc"
    # per-step host work of the MTP path (trace on): captured-graph steps, eager MTP draft steps (the passes that are not graph steps), mean graph step and mean host gap before it (ms)
    gs=$(grep -ac '^spec: mtp graph step' $f.err); es=$(( $(grep -ac '^spec: token' $f.err) - gs ))
    if [ "$gs$es" != 00 ]; then extra="$extra graph_steps=$gs eager_steps=$es $(grep -a '^spec: mtp graph step' $f.err | awk '{s+=$5;g+=$8;n++} END{if(n) printf "mean_step=%.3f mean_gap=%.3f", s/n, g/n}')"; fi
    extra="$extra $(grep -a 'verify attention rows' $f.err | sort | uniq -c | tr '\n' ';')"
    st=$(grep -a '^spec:' $f.err | grep -av 'spec: copy token\|spec: token\|spec: mtp\|spec: cost' | tail -1 | cut -c1-110)
    if cmp -s $O/gen-$n-off.ids $f.ids && [ -s $O/gen-$n-off.ids ]; then echo "PASS ids $n $arm: $(tr ',' '\n' < $f.ids | wc -l) ids identical |$extra | $st"; else echo "FAIL ids $n $arm: differ/empty |$extra | $st"; fails=$((fails+1)); fi
  done
done
echo "ids fails=$fails"
