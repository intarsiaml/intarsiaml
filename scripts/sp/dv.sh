#!/usr/bin/env bash
# Lane sp, `--mtp-draft-vocab 65536` as the Qwen3.8-27B recipe default: HIP gates on tatooine (27B MXFP4, natural-v1 128 / 1k, 256 greedy ids).
#   dv.sh prep   inside `fleetq run --load cpu,disk` (radiance stays up): copy a warm cargo volume, build tree ~/intarsiaml-dv (release, hip)
#   dv.sh run    inside `fleetq run --takes-down radiance ...`: base = main (tree ~/intarsiaml-w0, volume intarsia-cargo-target-w0, built by w0.sh prep),
#                dev = the branch (tree ~/intarsiaml-dv). Radiance is restored in the trap. Each check prints PASS or FAIL.
#   1. spec off: `plan show` json and text of the 27B byte-identical, base vs dev; 256 ids identical
#   2. dev MTP k=2 with no flag (the recipe default, stderr says so) ids == spec off, and acceptance counts == dev with explicit 65536
#   3. dev `--mtp-draft-vocab 0` ids == spec off and acceptance counts == base MTP k=2 (the full head is main's path)
set -uo pipefail
TD=$HOME/intarsiaml-dv; VD=intarsia-cargo-target-dv; TB=$HOME/intarsiaml-w0; VB=intarsia-cargo-target-w0; O=$HOME/sp-logs/dv
H=/data/models/radiance/hf-cache/hub
M27=/models/models--amd--Qwen3.8-27B-Quark-AWQ-MXFP4/snapshots/5233554c5fa56afda40150556b95573c2d7d29c0
case "${1:-}" in
prep)
  cd ~
  if ! docker volume inspect $VD >/dev/null 2>&1; then
    docker volume create $VD >/dev/null
    docker run --rm -v intarsia-cargo-target-hgbase:/s:ro -v $VD:/d alpine sh -c 'cp -a /s/. /d/ && chmod 0777 /d'
  fi
  INTARSIA_REPO=$TD INTARSIA_TARGET_VOL=$VD $TD/deploy/tatooine/dev.sh cargo build --release --locked --features hip -p intarsia-cli 2>&1 | tail -4
  echo "== stamp: source $(cat $TD/HEAD.txt 2>/dev/null); binary $(docker run --rm -v $VD:/t:ro alpine sh -c 'ls -l /t/release/intarsia' | awk '{print $5, $6, $7, $8}'); main base: $(cat $TB/HEAD.txt 2>/dev/null)"
  echo PREP DONE ;;
run)
  mkdir -p $O
  restore() { systemctl --user start radiance; for i in $(seq 1 36); do curl -s -m 3 http://localhost:8080/v1/models | grep -q '"data":\[' && break; sleep 5; done
    echo "[dv] radiance $(systemctl --user is-active radiance) after $((i * 5)) s"; }
  trap restore EXIT
  echo "dev source: $(cat $TD/HEAD.txt); base source: $(cat $TB/HEAD.txt)"
  systemctl --user stop radiance; sleep 3
  dev() { # dev <tree> <vol> <card> -- cmd...
    t=$1; v=$2; c=$3; shift 4
    INTARSIA_MODELS=$H INTARSIA_REPO=$t INTARSIA_TARGET_VOL=$v HIP_VISIBLE_DEVICES=$c timeout 900 $t/deploy/tatooine/dev.sh "$@"
  }
  gen() { # gen <side b|d> <card> <tag> <size> -- <generate args>
    side=$1; card=$2; tag=$3; n=$4; shift 5
    if [ $side = b ]; then t=$TB; v=$VB; else t=$TD; v=$VD; fi
    dev $t $v $card bash -c "/target/release/intarsia generate -m $M27 --backend hip --mem-budget 29 --max-seq 20480 --profile-every 0 --ignore-eos --show-ids --max-tokens 256 $* -p \"\$(cat /work/sp-out/n$n.txt)\"" \
      > $O/$tag-n$n.out 2> $O/$tag-n$n.err < /dev/null
    grep -a '^ids:' $O/$tag-n$n.err > $O/$tag-n$n.ids
    echo "$tag n$n md5=$(md5sum < $O/$tag-n$n.ids | cut -c1-8) | $(grep -a 'finish=' $O/$tag-n$n.err | tail -1 | sed 's/.*| decode/decode/' | cut -c1-60) | $(grep -a '^spec:' $O/$tag-n$n.err | tail -1 | cut -c1-110)"
  }
  ( for side in b d; do for f in "" "--json"; do
      dev $([ $side = b ] && echo "$TB $VB" || echo "$TD $VD") 1 -- /target/release/intarsia plan show $f -m $M27 --backend hip --max-seq 8192 > $O/plan-$side${f:+-json}.txt 2>&1
    done; done ) &
  ( for n in 128 1024; do gen b 0 b-off $n -- --spec off; gen b 0 b-mtp $n -- --spec mtp --spec-k 2; done ) > $O/card0.log 2>&1 &
  ( for n in 128 1024; do gen d 1 d-off $n -- --spec off; gen d 1 d-def $n -- --spec mtp --spec-k 2
      gen d 1 d-v0 $n -- --spec mtp --spec-k 2 --mtp-draft-vocab 0; gen d 1 d-v65536 $n -- --spec mtp --spec-k 2 --mtp-draft-vocab 65536; done ) > $O/card1.log 2>&1 &
  wait
  echo "== card 0 (main)"; cat $O/card0.log; echo "== card 1 (branch)"; cat $O/card1.log
  fails=0; chk() { if [ "$1" = 0 ]; then echo "PASS $2"; else echo "FAIL $2"; fails=$((fails+1)); fi; }
  md() { grep -a "^$1 n$2 " $O/card$3.log | sed -E 's/.*md5=([0-9a-f]+).*/\1/'; }
  acc() { grep -a "^$1 n$2 " $O/card$3.log | sed -E 's/.*(passes=[0-9]+ proposed=[0-9]+ accepted=[0-9]+).*/\1/'; }
  cmp -s $O/plan-b.txt $O/plan-d.txt && [ -s $O/plan-b.txt ]; chk $? "27B spec off: plan show text byte-identical to main ($(wc -c < $O/plan-b.txt) B)"
  cmp -s $O/plan-b-json.txt $O/plan-d-json.txt && [ -s $O/plan-b-json.txt ]; chk $? "27B spec off: plan show json byte-identical to main ($(wc -c < $O/plan-b-json.txt) B)"
  for n in 128 1024; do
    [ -n "$(md b-off $n 0)" ] && [ "$(md b-off $n 0)" = "$(md d-off $n 1)" ]; chk $? "n$n: spec-off ids == main's"
    [ "$(md d-def $n 1)" = "$(md d-off $n 1)" ]; chk $? "n$n: MTP k=2 with the recipe default (no flag) ids == spec off"
    [ "$(md d-v65536 $n 1)" = "$(md d-off $n 1)" ]; chk $? "n$n: MTP k=2 --mtp-draft-vocab 65536 ids == spec off"
    [ "$(md d-v0 $n 1)" = "$(md d-off $n 1)" ]; chk $? "n$n: MTP k=2 --mtp-draft-vocab 0 ids == spec off"
    a=$(acc d-def $n 1); b=$(acc d-v65536 $n 1); [ -n "$a" ] && [ "$a" = "$b" ]; chk $? "n$n: default acceptance == explicit 65536 ($a)"
    a=$(acc d-v0 $n 1); b=$(acc b-mtp $n 0); [ -n "$a" ] && [ "$a" = "$b" ]; chk $? "n$n: --mtp-draft-vocab 0 acceptance == main's full head ($a)"
    grep -aq "recipe qwen3_5_vl, hip" $O/d-def-n$n.err; chk $? "n$n: the load log names the recipe default (recipe qwen3_5_vl, hip)"
    ! grep -aq "recipe qwen3_5_vl" $O/d-v0-n$n.err; chk $? "n$n: no recipe-default line with --mtp-draft-vocab 0"
  done
  echo "[dv] done: $fails failing check(s)" ;;
*) echo "usage: dv.sh prep|run"; exit 2 ;;
esac
