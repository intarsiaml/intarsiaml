#!/usr/bin/env bash
# Lane sp window 0 (batched-MTP design, no new code): solo Qwen3.8-27B MTP on current main.
#   w0.sh prep   ON tatooine inside `fleetq run --load cpu,disk` (radiance stays up): copy a warm cargo volume, build tree ~/intarsiaml-w0 (release, hip)
#   w0.sh run    ON tatooine inside `fleetq run --takes-down radiance ...`: the window; radiance is restored in the trap
# card 0: spec off and MTP k=2 (INTARSIA_SPEC_TRACE=1: ms per pass) at 128/1k/4k/16k; card 1: k=1, k=2 repeat, k=3 at 1k/4k + plan show (state bytes).
set -uo pipefail
T=$HOME/intarsiaml-w0; VOL=intarsia-cargo-target-w0; O=$HOME/sp-logs/w0
H=/data/models/radiance/hf-cache/hub
M27=/models/models--amd--Qwen3.8-27B-Quark-AWQ-MXFP4/snapshots/5233554c5fa56afda40150556b95573c2d7d29c0
case "${1:-}" in
prep)
  cd ~
  if ! docker volume inspect $VOL >/dev/null 2>&1; then
    docker volume create $VOL >/dev/null
    docker run --rm -v intarsia-cargo-target-hgbase:/s:ro -v $VOL:/d alpine sh -c 'cp -a /s/. /d/ && chmod 0777 /d'
  fi
  INTARSIA_REPO=$T INTARSIA_TARGET_VOL=$VOL $T/deploy/tatooine/dev.sh cargo build --release --locked --features hip -p intarsia-cli 2>&1 | tail -4
  echo "== stamp: source $(cat $T/HEAD.txt 2>/dev/null); binary $(docker run --rm -v $VOL:/t:ro alpine sh -c 'ls -l /t/release/intarsia' | awk '{print $5, $6, $7, $8}')"
  echo PREP DONE ;;
run)
  mkdir -p $O
  restore() { systemctl --user start radiance; for i in $(seq 1 36); do curl -s -m 3 http://localhost:8080/v1/models | grep -q '"data":\[' && break; sleep 5; done
    echo "[w0] radiance $(systemctl --user is-active radiance) after $((i * 5)) s"; }
  trap restore EXIT
  echo "source: $(cat $T/HEAD.txt)"
  systemctl --user stop radiance; sleep 3
  gen() { # gen <card> <tag> <size> [env KEY=VAL ...] -- <generate args>
    card=$1; tag=$2; n=$3; shift 3; envs=(); while [ "$1" != "--" ]; do envs+=("$1"); shift; done; shift
    INTARSIA_MODELS=$H INTARSIA_REPO=$T INTARSIA_TARGET_VOL=$VOL HIP_VISIBLE_DEVICES=$card timeout 900 $T/deploy/tatooine/dev.sh bash -c \
      "env ${envs[*]} /target/release/intarsia generate -m $M27 --backend hip --mem-budget 29 --max-seq 20480 --profile-every 0 --ignore-eos --show-ids --max-tokens 256 $* -p \"\$(cat /work/sp-out/n$n.txt)\"" \
      > $O/$tag-n$n.out 2> $O/$tag-n$n.err < /dev/null
    grep -a '^ids:' $O/$tag-n$n.err > $O/$tag-n$n.ids
    echo "$tag n$n md5=$(md5sum < $O/$tag-n$n.ids | cut -c1-8) | $(grep -a 'finish=' $O/$tag-n$n.err | tail -1 | sed 's/.*| decode/decode/' | cut -c1-80) | $(grep -a '^spec:' $O/$tag-n$n.err | tail -1 | cut -c1-130) | $(grep -a '^steps=' $O/$tag-n$n.err | tail -1 | cut -c1-90)"
  }
  ( for n in 128 1024 4096 16384; do gen 0 off $n INTARSIA_X=1 -- --spec off; done
    for n in 128 1024 4096 16384; do gen 0 k2 $n INTARSIA_SPEC_TRACE=1 -- --spec mtp --spec-k 2; done ) > $O/card0.log 2>&1 &
  ( for n in 1024 4096; do gen 1 k1 $n INTARSIA_SPEC_TRACE=1 -- --spec mtp --spec-k 1; gen 1 k2b $n INTARSIA_X=1 -- --spec mtp --spec-k 2; gen 1 k3 $n INTARSIA_SPEC_TRACE=1 -- --spec mtp --spec-k 3; done
    INTARSIA_MODELS=$H INTARSIA_REPO=$T INTARSIA_TARGET_VOL=$VOL HIP_VISIBLE_DEVICES=1 $T/deploy/tatooine/dev.sh /target/release/intarsia plan show --json -m $M27 --backend hip --max-batch 4 --spec mtp --spec-k 2 --max-seq 8192 > $O/plan-b4-mtp.json 2>&1
    INTARSIA_MODELS=$H INTARSIA_REPO=$T INTARSIA_TARGET_VOL=$VOL HIP_VISIBLE_DEVICES=1 $T/deploy/tatooine/dev.sh /target/release/intarsia plan show --json -m $M27 --backend hip --max-batch 4 --max-seq 8192 > $O/plan-b4.json 2>&1 ) > $O/card1.log 2>&1 &
  wait
  echo "== card 0"; cat $O/card0.log; echo "== card 1"; cat $O/card1.log
  ls -l $O/plan-b4*.json
  echo "[w0] done" ;;
*) echo "usage: w0.sh prep|run"; exit 2 ;;
esac
