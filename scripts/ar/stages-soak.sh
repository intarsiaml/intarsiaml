#!/usr/bin/env bash
# Lane ar (ON tatooine, radiance down): merge gates for the new INTARSIA_HIP_AR_DRAIN default, after scripts/det/lib.sh (tree ~/intarsiaml-ars,
# CLI /work/ars-intarsia). Stages (STAGES, default "kt ref soak ids"):
#   kt   : `kernels test --backend hip --quick`
#   ref  : REF fresh tapped processes with INTARSIA_HIP_AR_DRAIN=3 (the rollback = main's behaviour), majority = main's output
#   soak : N fresh tapped tp=2 processes of short-5 (--spec mtp --spec-k 2, 16 tokens) with the environment NOT setting the knob
#   ids  : 256 greedy tokens of short-5, default vs =3, tp=2, ids compared
#   tests: the HIP comm and ar_norm integration tests (prebuilt by build.sh with TESTS=1) under INTARSIA_HIP_AR_DRAIN=1, =7 and =7 with
#          INTARSIA_HIP_AR_LL_MAX=2097152 (every test on the LL path); the tests with an explicit mode (LL vs fence bitwise, delayed-rank
#          stress, tag-check negative control) run in each pass
#   rate : `all_reduce_rate` (us per call by size) under modes 1, 3, 7 and 7 with LL up to 2 MiB
#   plan : `plan show` of the 27B and 35B at tp=2 (all-reduce calls and bytes per decode step, fused with the rmsnorm)
#   kt7  : kernels test --quick with the B binary (ars-intarsia-b) under INTARSIA_HIP_AR_DRAIN=7
#   soak7: like soak, with the B binary and, with INTARSIA_HIP_AR_DRAIN=7
#   ids18: the det lane's 18 prompts (short/4k/16k x 0-5), plain decode (--spec off) at tp=2, 256 greedy tokens: REFN (3) fresh runs with
#          INTARSIA_HIP_AR_DRAIN=3 (main's behaviour; their majority is the reference) and DEFN (2) fresh runs with the default; the verdict is
#          scripts/ar/ids18_verdict.py on the fetched ids files
# E="K=V" sets a process's environment (loop / g); TAP=1 writes the per-row layer hashes of every process to $O/<tag>-<i>.tap.
set -uo pipefail
export O=${1:?out dir}
mkdir -p "$O"
export INTARSIA_MODELS=/data/models
M27=/models/radiance/hf-cache/hub/models--amd--Qwen3.8-27B-Quark-AWQ-MXFP4/snapshots/5233554c5fa56afda40150556b95573c2d7d29c0
TREE=$HOME/intarsiaml-ars
BIN=${BIN:-ars-intarsia}
log() { echo "$(date +%T) $*" | tee -a "$O/steps.log"; }
DEV=$TREE/deploy/tatooine/dev.sh
# g <tag> <max tokens> [intarsia args]: one greedy generation of prompt $PR (default short-5), ids to $O/<tag>.ids
g() {
  local tag=$1 mt=$2; shift 2
  INTARSIA_REPO=$TREE HIP_VISIBLE_DEVICES=0,1 timeout 900 "$DEV" env ${E:-} "/work/$BIN" generate -m "$M27" --backend hip --mem-budget 29 --max-seq 20480 \
    --chat --no-think --profile-every 0 --ignore-eos --show-ids -p "$(cat "$TREE/scripts/dfh/ids-prompts/${PR:-short-5}.txt")" --max-tokens "$mt" "$@" > "$O/$tag.out" 2> "$O/$tag.err" < /dev/null
  local rc=$?; echo "rc=$rc $(grep -a '^ids' "$O/$tag.err")" > "$O/$tag.ids"; log "$tag exit $rc${E:+ [$E]}"
}
# loop <tag> <n> [intarsia args]: n fresh tp=2 processes in ONE dev.sh container; ids line (with exit code) of every process appended to $O/<tag>.ids, TAP=1 writes $O/<tag>-<i>.tap
loop() {
  local tag=$1 n=$2; shift 2
  local o=${O/#$TREE//work}
  INTARSIA_REPO=$TREE HIP_VISIBLE_DEVICES=0,1 timeout 3300 "$DEV" bash -c '
    tag=$1 n=$2 o=$3 E=$4 tap=$5 mt=$6; shift 6
    for i in $(seq 1 "$n"); do
      t=; [[ $tap == 1 ]] && t=INTARSIA_DET_TAP=$o/$tag-$i.tap
      env $E $t '"/work/$BIN"' generate -m "'"$M27"'" --backend hip --tp 2 --devices 0,1 --mem-budget 29 --max-seq 20480 --chat --no-think --profile-every 0 \
        --ignore-eos --show-ids -p "$(cat /work/scripts/dfh/ids-prompts/short-5.txt)" --max-tokens $mt "$@" > /dev/null 2> "$o/$tag.tmp"
      echo "rc=$? $(grep -a "^ids" "$o/$tag.tmp")" >> "$o/$tag.ids"
    done; rm -f "$o/$tag.tmp"' _ "$tag" "$n" "$o" "${E:-}" "${TAP:-0}" "${MT:-16}" "$@" < /dev/null
  log "$tag: $(wc -l < "$O/$tag.ids") processes; ids-line counts $(cut -d' ' -f2- "$O/$tag.ids" | sort | uniq -c | sort -rn | awk '{print $1}' | tr '\n' ' ')"
}
vram() { local m=0 v c; for c in /sys/class/drm/card[0-9]/device/mem_info_vram_used; do v=$(cat "$c" 2>/dev/null || echo 0); (( v > m )) && m=$v; done; echo $m; }
for i in 1 2 3 4 5 6; do (( $(vram) < 1073741824 )) && break; sleep 10; done
(( $(vram) < 1073741824 )) || { log "CONTENDED: $(( $(vram) / 1048576 )) MiB in use on a card: NO VERDICT"; exit 2; }
M="--spec mtp --spec-k 2"
for st in ${STAGES:-kt ref soak ids}; do
  case $st in
    plan) # how many all-reduces per decode step, of which sizes, and how many fused with the rmsnorm (no run, but it loads the weights)
          for mm in "27b:$M27" "35b:/models/intarsia/Qwen3.6-35B-A3B-NVFP4"; do
            INTARSIA_REPO=$TREE HIP_VISIBLE_DEVICES=0,1 timeout 600 "$DEV" "/work/ars-intarsia-b" plan show --backend hip --model "${mm#*:}" --tp 2 --devices 0,1 \
              --mem-budget 29 --max-seq 8192 > "$O/plan-${mm%%:*}.txt" 2> "$O/plan-${mm%%:*}.err" < /dev/null
            log "plan ${mm%%:*}: exit $? $(grep -a -E 'all-reduce|fused' "$O/plan-${mm%%:*}.txt" | tr '\n' '|' | cut -c1-260)"
          done ;;
    kt7) BIN=ars-intarsia-b INTARSIA_REPO=$TREE HIP_VISIBLE_DEVICES=0,1 INTARSIA_HIP_AR_DRAIN=7 timeout 1500 "$DEV" "/work/ars-intarsia-b" kernels test --backend hip --quick > "$O/kernels-test-d7.log" 2>&1 < /dev/null
        log "kernels test --backend hip --quick (B binary, drain 7): exit $? $(tail -3 "$O/kernels-test-d7.log" | tr '\n' '|' | cut -c1-300)" ;;
    kt) INTARSIA_REPO=$TREE HIP_VISIBLE_DEVICES=0,1 timeout 1500 "$DEV" "/work/$BIN" kernels test --backend hip --quick > "$O/kernels-test.log" 2>&1 < /dev/null
        log "kernels test --backend hip --quick: exit $? $(tail -3 "$O/kernels-test.log" | tr '\n' '|' | cut -c1-300)" ;;
    ref) TAP=1 E="INTARSIA_HIP_AR_DRAIN=3" loop ref3 "${REF:-30}" $M ;;
    soak) TAP=1 loop soak "${N:-250}" $M ;;
    soak7) BIN=ars-intarsia-b TAP=1 E="INTARSIA_HIP_AR_DRAIN=7" loop soak7 "${N:-250}" $M ;;
    tests) for v in "1:" "7:" "7:2097152"; do d=${v%%:*}; m=${v#*:}
             INTARSIA_TARGET_VOL=intarsia-cargo-ars INTARSIA_REPO=$TREE/.ars/btree HIP_VISIBLE_DEVICES=0,1 INTARSIA_HIP_AR_DRAIN=$d INTARSIA_HIP_AR_LL_MAX=$m timeout 1500 \
               "$DEV" cargo test --release -p intarsia-backend-hip --test comm --test ar_norm -- --test-threads=1 > "$O/tests-d$d-${m:-default}.log" 2>&1 < /dev/null
             log "tests drain=$d ll_max=${m:-default}: exit $? $(grep -a 'test result' "$O/tests-d$d-${m:-default}.log" | tr '\n' ' ' | cut -c1-200)"
           done ;;
    rate) for v in "1:" "3:" "7:" "7:2097152"; do d=${v%%:*}; m=${v#*:}
             INTARSIA_TARGET_VOL=intarsia-cargo-ars INTARSIA_REPO=$TREE/.ars/btree HIP_VISIBLE_DEVICES=0,1 INTARSIA_HIP_AR_DRAIN=$d INTARSIA_HIP_AR_LL_MAX=$m timeout 900 \
               "$DEV" cargo test --release -p intarsia-backend-hip --test comm rate -- --ignored --nocapture --test-threads=1 > "$O/rate-d$d-${m:-default}.log" 2>&1 < /dev/null
             log "rate drain=$d ll_max=${m:-default}: exit $?"
           done ;;
    ids) E="" g ids-default 256 --tp 2 --devices 0,1 $M; E="INTARSIA_HIP_AR_DRAIN=3" g ids-d3 256 --tp 2 --devices 0,1 $M
         cmp -s "$O/ids-default.ids" "$O/ids-d3.ids" && log "ids default == d3" || log "ids default DIFFERS from d3" ;;
    ids18) for kp in short 4k 16k; do for i in 0 1 2 3 4 5; do n=$kp-$i
             for r in $(seq 1 "${REFN:-3}"); do PR=$n E="INTARSIA_HIP_AR_DRAIN=3" g "ref-$n-$r" 256 --spec off --tp 2 --devices 0,1; done
             for r in $(seq 1 "${DEFN:-2}"); do BIN=${DEFBIN:-$BIN} PR=$n E="${DEFE:-}" g "def-$n-$r" 256 --spec off --tp 2 --devices 0,1; done
           done; done ;;
  esac
done
log "soak stages done"
