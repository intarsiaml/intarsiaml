#!/usr/bin/env bash
# Lane lmhead-fp8-default, CUDA gates ON spark1 (no-timing parts), inside a quiet slot:
#   fleetq run --host spark1 --quiet gpu,cpu --mem 50 --expect 60 --label "..." -- ssh spark1 'bash -s plan' < scripts/lmhead/cgates.sh      (then: ids)
# plan: `plan show` (text, JSON) of the 27B base / head / head+off, and of the 8B and 35B base / head: 8B and 35B byte-identical, 27B head+off == base,
#       27B base vs head differ by the head lines only (diff kept).
# ids35: the 35B, variable unset, head vs base (short and ~3.5k prompts).
# ids : 256 greedy ids of the 27B (default flags = MTP k2 draft head 65536 from the recipe), short and ~3.5k prompts: head run 1, head run 2 (deterministic?),
#       head+off (== base?), base.
set -uo pipefail
export PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH
R=$HOME/lmh-cuda; O=${O:-$R/out}; mkdir -p $O
HB=$R/target-head/release/intarsia; BB=$R/target-base/release/intarsia
H=/home/nvidia/.cache/huggingface/hub
M27=$R/models/Qwen3.8-27B-Quark-AWQ-MXFP4
M35=$(ls -d $H/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/* | head -1); M8=$(ls -d $H/models--Qwen--Qwen3-8B/snapshots/* | head -1)
PD=$R/src/scripts/tp2-hip/noop/prompts
[ -z "${INTARSIA_LMHEAD_QUANT:-}" ] || { echo "INTARSIA_LMHEAD_QUANT is set"; exit 2; }
run() { # run <arm> args...   arms: base, head, head-off
  local a=$1; shift
  case $a in base) $BB "$@" ;; head) $HB "$@" ;; head-off) INTARSIA_LMHEAD_QUANT=off $HB "$@" ;; esac
}
case "${1:?plan|ids}" in
plan)
  plan() { run $1 plan show --backend cuda --model $3 --max-seq 8192 --mem-budget 45 --json < /dev/null > $O/plan-$2-$1.json 2> $O/plan-$2-$1.json.err
           run $1 plan show --backend cuda --model $3 --max-seq 8192 --mem-budget 45 < /dev/null > $O/plan-$2-$1.txt 2> $O/plan-$2-$1.txt.err; }
  for a in base head; do plan $a 8 $M8; plan $a 35 $M35; done
  for a in base head head-off; do plan $a 27 $M27; done
  for n in 8 35; do for k in json txt; do
    cmp -s $O/plan-$n-base.$k $O/plan-$n-head.$k && [ -s $O/plan-$n-head.$k ] && echo "plan $n $k: byte-identical ($(wc -c < $O/plan-$n-head.$k) B)" || echo "plan $n $k: DIFFER or empty"; done; done
  for k in json txt; do
    cmp -s $O/plan-27-head-off.$k $O/plan-27-base.$k && [ -s $O/plan-27-base.$k ] && echo "plan 27 $k: head+off == base (byte-identical)" || echo "plan 27 $k: head+off vs base DIFFER or empty"
    diff $O/plan-27-base.$k $O/plan-27-head.$k > $O/plan-27-base-vs-head.$k.diff; echo "plan 27 $k: base vs head: $(grep -c '^[<>]' $O/plan-27-base-vs-head.$k.diff) changed lines"
  done
  ls -la $O/plan-27-*.err | awk '{print $5, $9}'; head -c 300 $O/plan-27-head.txt.err ;;
ids)
  ids() { # ids <arm> <tag> <prompt>
    local ms=8192
    run $1 generate -m $M27 --backend cuda --mem-budget 45 --max-seq $ms --chat --no-think --temperature 0 --profile-every 0 --ignore-eos --show-ids \
      -p "$(cat $PD/$3.txt)" --max-tokens 256 < /dev/null > $O/gen-$2-$3.out 2> $O/gen-$2-$3.err
    grep -a '^ids:' $O/gen-$2-$3.err > $O/gen-$2-$3.ids; echo "[ids] $2 $3: $(tr ',' '\n' < $O/gen-$2-$3.ids | wc -l) ids $(date +%T)"
  }
  for p in short 4k; do ids head head1 $p; ids head head2 $p; ids head-off off $p; ids base base $p; done
  grep -ah "lm_head quantised\|speculative decoding\|--spec mtp" $O/gen-head1-short.err | head -3 | cut -c1-200
  for p in short 4k; do
    c() { cmp -s $O/gen-$1-$p.ids $O/gen-$2-$p.ids && [ -s $O/gen-$1-$p.ids ] && echo "ids $p: $1 == $2 ($(tr ',' '\n' < $O/gen-$1-$p.ids | wc -l) ids, sha $(shasum $O/gen-$1-$p.ids | cut -c1-8))" || echo "ids $p: $1 vs $2 DIFFER or empty"; }
    c head1 head2; c off base; c head1 base
  done ;;
ids35)
  # 35B (NVFP4 head, no lm_head_quant in its recipe): the variable unset, head vs base must be identical.
  ids() { local ms=8192
    run $1 generate -m $M35 --backend cuda --mem-budget 45 --max-seq $ms --chat --no-think --temperature 0 --profile-every 0 --ignore-eos --show-ids \
      -p "$(cat $PD/$3.txt)" --max-tokens 256 < /dev/null > $O/gen35-$2-$3.out 2> $O/gen35-$2-$3.err
    grep -a '^ids:' $O/gen35-$2-$3.err > $O/gen35-$2-$3.ids; echo "[ids35] $2 $3: $(tr ',' '\n' < $O/gen35-$2-$3.ids | wc -l) ids $(date +%T)"
  }
  for p in short 4k; do ids head head $p; ids base base $p; done
  for p in short 4k; do
    cmp -s $O/gen35-head-$p.ids $O/gen35-base-$p.ids && [ -s $O/gen35-head-$p.ids ] && echo "ids35 $p: head == base ($(tr ',' '\n' < $O/gen35-head-$p.ids | wc -l) ids, sha $(shasum $O/gen35-head-$p.ids | cut -c1-8))" || echo "ids35 $p: head vs base DIFFER or empty"
  done ;;
esac
