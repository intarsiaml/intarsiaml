#!/usr/bin/env bash
# Lane tc (2026-10-22, docs/results/2026-10-22-tc-tail-chunk.md): GPU stages of the tail-merge A/B, run ON tatooine inside a
# radiance-down window opened by tc-window.sh. Three arms, one binary each (built beforehand in CPU slots):
#   base  ~/intarsiaml-tcbase  volume intarsia-cargo-target-tcbase   origin/main
#   off   ~/intarsiaml-tc      volume intarsia-cargo-target-tc       the lane branch, INTARSIA_PREFILL_TAIL_MERGE unset
#   on    ~/intarsiaml-tc      volume intarsia-cargo-target-tc       the lane branch, INTARSIA_PREFILL_TAIL_MERGE=$TM
# The knob is exported INSIDE the container command (dev.sh forwards only INTARSIA_HIP_* / INTARSIA_TP_*).
#   tc-stages.sh <stage>...   stages: ids acc
# Needs bench-out/tc/prompt{16654,4408}.txt and the 4k golden in both trees, and (acc) /models/golden/*. Output: $OUTD (default
# ~/intarsiaml-tc/bench-out/tc/out). Every inner command has stdin from /dev/null.
set -uo pipefail
TM=${TM:-1024}
BUDGET=${BUDGET:-600}  # seconds from start; stages / runs that need more than what is left are skipped and logged
OUTD=${OUTD:-$HOME/intarsiaml-tc/bench-out/tc/out}
mkdir -p "$OUTD"
D=$HOME/intarsiaml-tc/deploy/tatooine/dev.sh
M35=/models/Qwen3.6-35B-A3B-NVFP4
GOLD=/models/golden
G_LONG=qwen36-35b-a3b-nvfp4-vllm-long-top20.safetensors
G_4K=Qwen3.6-35B-A3B-NVFP4-vllm-4k-top20.safetensors  # copied into bench-out/tc of both trees by the caller
T0=$(date +%s)
left() { echo $((BUDGET - ($(date +%s) - T0))); }
need() { if (($(left) < $1)); then echo "SKIPPED $2: $(left) s left, needs $1"; return 1; fi; return 0; }
ts() { date +%T; }

# arm_set <arm>: REPO VOL ENVX (the export line put in front of the command inside the container)
arm_set() {
  case $1 in
    base) REPO=$HOME/intarsiaml-tcbase; VOL=intarsia-cargo-target-tcbase; ENVX="" ;;
    off) REPO=$HOME/intarsiaml-tc; VOL=intarsia-cargo-target-tc; ENVX="" ;;
    on) REPO=$HOME/intarsiaml-tc; VOL=intarsia-cargo-target-tc; ENVX="export INTARSIA_PREFILL_TAIL_MERGE=$TM;" ;;
    *) echo "bad arm $1"; return 1 ;;
  esac
}
# gen <arm> <prompt N|file> <max tokens> <out prefix>: one generate with --show-ids, ids in <out prefix>.ids
gen() {
  local arm=$1 n=$2 mt=$3 out=$4 gpu=${5:-0}
  arm_set "$arm" || return 1
  mkdir -p "$REPO/bench-out/tc"
  printf '%s\n' "$ENVX /target/release/intarsia generate -m $M35 --backend hip --mem-budget 29 --max-seq 18432 --ignore-eos --show-ids --profile-every 0 -p \"\$(cat /work/bench-out/tc/prompt$n.txt)\" --max-tokens $mt" > "$REPO/bench-out/tc/gen-$arm-$n.sh"
  HIP_VISIBLE_DEVICES=$gpu INTARSIA_REPO=$REPO INTARSIA_TARGET_VOL=$VOL bash "$D" bash -c "bash /work/bench-out/tc/gen-$arm-$n.sh" < /dev/null 2> "$out.err" > "$out.out"
  grep -a '^ids:' "$out.err" > "$out.ids"
  grep -aE "prompt:|finish=|steps=|rror|panick" "$out.err" | sed "s/^/$arm $n: /"
}

stage_ids() { # 256 greedy ids after the 16654 (localeval's 16k) and 4408 (4k + 312 tail) prompts: base vs off vs on
  local n a
  for n in 16654 4408; do
    need 90 "ids $n" || continue
    echo "--- ids $n $(ts)"
    for a in base off on; do gen $a $n 256 "$OUTD/ids-$a-$n"; done
    echo "ids $n: $(tr -cd ',' < "$OUTD/ids-base-$n.ids" | wc -c) commas in base"
    for a in off on; do
      if [ -s "$OUTD/ids-$a-$n.ids" ] && cmp -s "$OUTD/ids-base-$n.ids" "$OUTD/ids-$a-$n.ids"; then
        echo "IDS $n: base == $a"
      else
        echo "IDS $n: base != $a ($(cmp "$OUTD/ids-base-$n.ids" "$OUTD/ids-$a-$n.ids" 2>&1 | head -1))"
      fi
    done
  done
}

stage_pf() { # prefill time of ONE prompt (generate --max-tokens 1, a fresh process per run), PFREPS reps per arm and size, the arm order reversed on even reps
  local n rep a order
  for n in ${PFSIZES:-16654 4408 4089}; do
    for rep in $(seq 1 ${PFREPS:-5}); do
      need 60 "pf $n rep $rep" || return 0
      order="base off on"; [ $((rep % 2)) -eq 0 ] && order="on off base"
      for a in $order; do
        echo "pf $n rep $rep $a: $(gen $a $n 1 "$OUTD/pf-$a-$n-r$rep" | grep -a 'finish=' | sed 's/.*prefill \([0-9]*\) ms.*/\1 ms/')"
      done
    done
  done
}

# accrun <arm> <gpu> <tag> <extra accuracy args...>: one accuracy run, log in $OUTD/acc-<tag>.log
accrun() {
  local arm=$1 gpu=$2 tag=$3; shift 3
  arm_set "$arm" || return 1
  printf '%s\n' "$ENVX /target/release/intarsia accuracy --backend hip --model $M35 --mem-budget 29 --allow-unstamped --allow-dirty --prefill-chunk 4096 $*" > "$REPO/bench-out/tc/acc-$tag.sh"
  HIP_VISIBLE_DEVICES=$gpu INTARSIA_REPO=$REPO INTARSIA_TARGET_VOL=$VOL bash "$D" bash -c "bash /work/bench-out/tc/acc-$tag.sh" < /dev/null > "$OUTD/acc-$tag.log" 2>&1
  echo "acc $tag ($arm, gpu $gpu): exit $? $(ts)"
}
# stage_acc: the 4k golden (prompts 4064 / 4064 / 4631) is copied into bench-out/tc of both trees by the caller; the long golden (5 prompts, 4064 .. 12064) is the one in /models/golden
# base writes a self-golden per golden (GPU 0 and 1 in parallel); off and on compare against it (parallel again).
stage_acc() {
  local g mode=${ACC_MODE:-both}
  need 240 "acc base" || return 0
  echo "--- acc base self-goldens $(ts)"
  accrun base 0 base-long --golden $GOLD/$G_LONG --mode $mode --decode-positions 64 --max-seq 16384 --write-self-golden /work/bench-out/tc/S-long.safetensors &
  accrun base 1 base-4k --golden /work/bench-out/tc/$G_4K --mode $mode --decode-positions 64 --max-seq 8192 --write-self-golden /work/bench-out/tc/S-4k.safetensors &
  wait
  need 240 "acc off/on" || return 0
  echo "--- acc off/on vs self-golden $(ts)"
  for g in long 4k; do
    ms=16384; [ $g = 4k ] && ms=8192
    # the self-golden is read from the base tree's bench-out (mounted at /work for the base arm); copy it next to the dev tree first
    cp "$HOME/intarsiaml-tcbase/bench-out/tc/S-$g.safetensors" "$HOME/intarsiaml-tc/bench-out/tc/S-$g.safetensors"
    accrun off 0 off-$g --self-golden /work/bench-out/tc/S-$g.safetensors --mode $mode --decode-positions 64 --max-seq $ms &
    accrun on 1 on-$g --self-golden /work/bench-out/tc/S-$g.safetensors --mode $mode --decode-positions 64 --max-seq $ms &
    wait
  done
}

for s in "$@"; do
  echo "################ stage $s start $(ts) (budget left $(left) s)"
  "stage_$s"
  echo "################ stage $s end $(ts)"
done
