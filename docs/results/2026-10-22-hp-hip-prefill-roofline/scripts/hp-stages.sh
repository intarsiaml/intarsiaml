#!/usr/bin/env bash
# Lane hp (docs/results/2026-10-22-hp-hip-prefill-roofline.md): GPU stages run ON tatooine inside a radiance-down window
# opened by hp-window.sh. GPU 0 only. Trees and cargo target volumes (built beforehand in CPU slots):
#   base  ~/intarsiaml-hpbase  volume intarsia-cargo-target-hp-base   (origin/main)
#   dev   ~/intarsiaml-hp      volume intarsia-cargo-target-hp-dev    (lane branch)
#   test  ~/intarsiaml-hp      volume intarsia-cargo-target-hp-test   (cargo test --no-run of the lane branch)
#   hp-stages.sh <stage>[:arg] ...
# Every inner command has stdin from /dev/null (a docker run would eat the rest of this script).
set -uo pipefail
export HIP_VISIBLE_DEVICES=0
DEV=$HOME/intarsiaml-hp
BASE=$HOME/intarsiaml-hpbase
OUT=$DEV/bench-out/hp/out
mkdir -p "$OUT"
for t in "$DEV" "$BASE"; do mkdir -p "$t/bench-out/hp" && cp -n "$HOME"/hp-prompts/prompt*.txt "$t/bench-out/hp/" 2>/dev/null; done
M35=/models/Qwen3.6-35B-A3B-NVFP4
M8=/models/Qwen3-8B
G=/models/golden
PROFIMG=intarsia-hip-prof:rocm7.2.4-rust1.90.0
ts() { date +%T; }
T0=$(date +%s)
BUDGET=${BUDGET:-1500}
left() { echo $((BUDGET - ($(date +%s) - T0))); }
need() { if (($(left) < $1)); then echo "SKIPPED $2: $(left) s left, needs $1"; return 1; fi; return 0; }
tree() { [ "$1" = base ] && echo "$BASE" || echo "$DEV"; }
vol() { echo "intarsia-cargo-target-hp-$1"; }
# dv <base|dev|test> <cmd...>: dev.sh in that tree with that volume
dv() { local w=$1; shift; INTARSIA_REPO=$(tree "$w") INTARSIA_TARGET_VOL=$(vol "$w") bash "$DEV/deploy/tatooine/dev.sh" "$@" < /dev/null; }
# mkgen <tree> <tag> <prompt N> <max tokens> <ENV assignments or x> [generate args]: bench-out/hp/gen-<tag>.sh in that tree
mkgen() {
  local w=$1 tag=$2 n=$3 mt=$4 envv=$5; shift 5
  local ex=""; [ "$envv" != x ] && ex="export $envv;"
  printf '%s\n' "$ex /target/release/intarsia generate -m $M35 --backend hip --mem-budget 29 --max-seq 18432 --ignore-eos $* -p \"\$(cat /work/bench-out/hp/prompt$n.txt)\" --max-tokens $mt" > "$(tree "$w")/bench-out/hp/gen-$tag.sh"
}
run() { dv "$1" bash -c "bash /work/bench-out/hp/gen-$2.sh" 2>&1 | grep -aE "prompt:|finish=|rror|panick"; }

stage_ceiling() {  # the matrix ceilings (ceiling.hip), in the ROCm image that has hipBLASLt
  need 240 ceiling || return 0
  echo "--- ceiling $(ts)"
  docker run --rm --device /dev/kfd --device /dev/dri --group-add "$(stat -c %g /dev/kfd)" --group-add "$(stat -c %g /dev/dri/renderD128)" \
    --group-add video --security-opt seccomp=unconfined -e HIP_VISIBLE_DEVICES=0 -v "$DEV":/work -w /work --entrypoint bash \
    fx6-rocm-llama:7.2.4 -c "timeout 220 ./bench-out-hp-ceiling /opt/rocm/lib/libhipblaslt.so.1" < /dev/null 2>&1 | tee "$OUT/ceiling.txt"
}
stage_bin() {  # bin:<file in ~/intarsiaml-hp>: a standalone microbenchmark binary (dev image)
  need 200 "bin $1" || return 0
  echo "--- bin $1 $(ts)"
  INTARSIA_REPO=$DEV bash "$DEV/deploy/tatooine/dev.sh" bash -c "timeout 190 /work/$1" < /dev/null 2>&1 | tee "$OUT/bin-$1.txt"
}
# prof:<base|dev>[:ENV=v,...]: engine op table (--profile-every 1 --op-table) and rocprofv3 kernel traces at 4096 and 16384
stage_prof() {
  local w=${1%%:*} envv=x tag; [[ "$1" == *:* ]] && envv=${1#*:} && envv=${envv//,/ }
  tag=$w; [ "$envv" != x ] && tag="$w-$(echo "$envv" | tr -c 'A-Za-z0-9=' '_' | sed 's/INTARSIA_HIP_//g')"
  local n
  for n in 4096 16384; do
    need 120 "optab $tag $n" || continue
    echo "--- op table $tag $n $(ts)"
    mkgen "$w" "optab-$tag-$n" $n 1 "$envv" --profile-every 1 --op-table
    dv "$w" bash -c "bash /work/bench-out/hp/gen-optab-$tag-$n.sh" > "$OUT/optab-$tag-$n.out" 2> "$OUT/optab-$tag-$n.err"
    grep -aE "prompt:|finish=" "$OUT/optab-$tag-$n.err"
  done
  for n in 4096 16384; do
    need 150 "rocprof $tag $n" || continue
    echo "--- rocprofv3 $tag $n $(ts)"
    mkgen "$w" "tb-$tag-$n" $n 1 "$envv" --profile-every 0
    rm -rf "$OUT/prof/$tag-$n"; mkdir -p "$OUT/prof/$tag-$n"
    INTARSIA_IMAGE=$PROFIMG dv "$w" bash -c "timeout -s KILL 300 rocprofv3 --kernel-trace --output-format csv -d /work/bench-out/hp/prof-tmp/$tag-$n -o run -- bash /work/bench-out/hp/gen-tb-$tag-$n.sh" 2>&1 | grep -aE "prompt:|finish=|rror|panick"
    local csv; csv=$(find "$(tree "$w")/bench-out/hp/prof-tmp/$tag-$n" -name '*kernel_trace.csv' | head -1)
    [ -n "$csv" ] && cp "$csv" "$OUT/prof/$tag-$n/run_kernel_trace.csv" && python3 -I "$DEV/scripts/hip-decode/analyze_prefill.py" "$csv" > "$OUT/prof/$tag-$n/table.txt" && head -24 "$OUT/prof/$tag-$n/table.txt"
    rm -rf "$(tree "$w")/bench-out/hp/prof-tmp/$tag-$n"
  done
}
# pf:<base|dev>:<reps>[:ENV=v,...]: unprofiled prefill-only timing (generate --max-tokens 1) at 4096 and 16384, reps each
stage_pf() {
  local w=${1%%:*} rest=${1#*:} envv=x; local reps=${rest%%:*}; [[ "$rest" == *:* ]] && envv=${rest#*:} && envv=${envv//,/ }
  local n r
  for r in $(seq 1 "$reps"); do
    for n in 4096 16384; do
      need 90 "pf $w $n" || continue
      mkgen "$w" "pf-$w-$n" $n 1 "$envv" --profile-every 0
      echo "pf $w [$envv] $n r$r: $(run "$w" "pf-$w-$n" | tr '\n' ' ')"
    done
  done | tee -a "$OUT/pf.txt"
}
# ids:<base|dev>:<prompt N>[:ENV=v,...]: greedy 256 ids
stage_ids() {
  local w=${1%%:*} rest=${1#*:} envv=x; local n=${rest%%:*}; [[ "$rest" == *:* ]] && envv=${rest#*:} && envv=${envv//,/ }
  need 150 "ids $w $n" || return 0
  local tag="$w-$n"; [ "$envv" != x ] && tag="$tag-$(echo "$envv" | tr -c 'A-Za-z0-9=' '_' | sed 's/INTARSIA_HIP_//g')"
  mkgen "$w" "ids-$tag" $n 256 "$envv" --profile-every 0 --show-ids
  dv "$w" bash -c "bash /work/bench-out/hp/gen-ids-$tag.sh" > "$OUT/ids-$tag.out" 2> "$OUT/ids-$tag.err"
  grep -a '^ids:' "$OUT/ids-$tag.err" > "$OUT/ids-$tag.ids"
  echo "ids $tag: $(grep -aE "finish=" "$OUT/ids-$tag.err" | tail -1) commas $(tr -cd ',' < "$OUT/ids-$tag.ids" | wc -c) sha $(sha256sum < "$OUT/ids-$tag.ids" | cut -c1-16)"
}
# ids8:<base|dev>: Qwen3-8B greedy 256 ids after the 4096 prompt (the no-op check)
stage_ids8() {
  local w=$1
  need 150 "ids8 $w" || return 0
  mkgen "$w" "ids8-$w" 4096 256 x --profile-every 0 --show-ids
  sed -i "s#-m $M35 --backend hip --mem-budget 29#-m $M8 --backend hip#" "$(tree "$w")/bench-out/hp/gen-ids8-$w.sh"
  dv "$w" bash -c "bash /work/bench-out/hp/gen-ids8-$w.sh" > "$OUT/ids8-$w.out" 2> "$OUT/ids8-$w.err"
  grep -a '^ids:' "$OUT/ids8-$w.err" > "$OUT/ids8-$w.ids"
  echo "ids8 $w: $(grep -aE "prompt:|finish=" "$OUT/ids8-$w.err" | tr '\n' ' ') commas $(tr -cd ',' < "$OUT/ids8-$w.ids" | wc -c) sha $(sha256sum < "$OUT/ids8-$w.ids" | cut -c1-16)"
}
# test:<filter>: the lane's hip crate tests (test volume, built with --no-run beforehand), single-threaded
stage_test() {  # test:<cargo test args, commas for spaces> e.g. test:--test,gdn_v2  (or test:--lib)
  need 300 "test $1" || return 0
  local args=${1//,/ }
  echo "--- cargo test hip $args $(ts)"
  dv test bash -c "timeout 280 cargo test --release -p intarsia-backend-hip $args -- --test-threads 1" 2>&1 | grep -aE "^test |test result|FAIL|panick|error" | tail -40 | tee -a "$OUT/test.txt"
}
# qual:<base|dev>:<35|8>[:ENV]: qualify quick
stage_qual() {
  local w=${1%%:*} rest=${1#*:} envv=x; local m=${rest%%:*}; [[ "$rest" == *:* ]] && envv=${rest#*:} && envv=${envv//,/ }
  need 600 "qual $w $m" || return 0
  local md=$M35 mb="--mem-budget 29"; [ "$m" = 8 ] && md=$M8 && mb=""
  local ex=""; [ "$envv" != x ] && ex="export $envv;"
  echo "--- qualify $w $m [$envv] $(ts)"
  dv "$w" bash -c "$ex /target/release/intarsia qualify --model $md --backend hip --quick-only --gen 128 --reps 5 --no-unim --out /work/bench-out/hp/qual-$w-$m --golden-dir $G $mb" > "$OUT/qual-$w-$m.txt" 2>&1
  grep -aiE "pass|fail|verdict|kl" "$OUT/qual-$w-$m.txt" | tail -30
}
# acc:<base|dev>[:ENV]: 35B accuracy vs the short golden (KL21)
stage_acc() {
  local w=${1%%:*} envv=x; [[ "$1" == *:* ]] && envv=${1#*:} && envv=${envv//,/ }
  need 400 "acc $w" || return 0
  local ex=""; [ "$envv" != x ] && ex="export $envv;"
  local tag=$w; [ "$envv" != x ] && tag="$w-$(echo "$envv" | tr -c 'A-Za-z0-9=' '_' | sed 's/INTARSIA_HIP_//g')"
  echo "--- accuracy 35B short $tag $(ts)"
  dv "$w" bash -c "$ex /target/release/intarsia accuracy --model $M35 --backend hip --mem-budget 29 --golden $G/qwen36-35b-a3b-nvfp4-vllm-top20-meta.safetensors --mode both --max-seq 4096 --allow-unstamped --allow-dirty" > "$OUT/acc-$tag.txt" 2>&1
  tail -22 "$OUT/acc-$tag.txt"
}

for st in "$@"; do
  name=${st%%:*}; arg=""; [[ "$st" == *:* ]] && arg=${st#*:}
  echo "=== $st [$(ts), $(left) s left]"
  "stage_$name" "$arg"
  if [ "$name" = test ] && grep -q "test result: FAILED\|error: test failed\|could not compile" "$OUT/test.txt" 2>/dev/null; then
    echo "ABORT: a test failed; the remaining stages are skipped"
    touch "$OUT/ABORT"
    break
  fi
done
echo "=== stages done [$(ts)]"
