#!/usr/bin/env bash
# Lane sp final HIP gate window, ON tatooine inside a fleetq job that takes radiance down (stop below, restore in the trap):
# the default path of the final head (tree sp, card 1) against main (tree sp-base, card 0), in parallel, ~15 min.
#   A: spec-off no-op, 8B and 35B, short and 4k: plan show json/text byte-identical, 256 greedy ids identical (hg/noop.sh)
#   B: MTP k=2 on the 35B and the 27B at 128 and 1k (natural-v1): ids == spec off, ids == main's MTP, and the acceptance
#      counts (passes / proposed / accepted) == main's. Prints PASS or FAIL per check.
set -uo pipefail
O=$HOME/sp-logs/win11; mkdir -p "$O" $HOME/sp-ids
S=$HOME/intarsiaml-sp/scripts/sp
D=$HOME/intarsiaml-sp/deploy/tatooine/dev.sh
M8=/models/Qwen3-8B; M35=/models/Qwen3.6-35B-A3B-NVFP4
H=/data/models/radiance/hf-cache/hub
M27=/models/models--amd--Qwen3.8-27B-Quark-AWQ-MXFP4/snapshots/5233554c5fa56afda40150556b95573c2d7d29c0
trap 'bash $HOME/sp-win.sh restore' EXIT
echo "source head: $(cat $HOME/intarsiaml-sp/HEAD.txt 2>/dev/null); binary cost-gate strings: $(docker run --rm -v intarsia-cargo-target-sp:/t:ro alpine sh -c 'grep -ac intarsia_spec_costgate /t/release/intarsia'), built $(docker run --rm -v intarsia-cargo-target-sp:/t:ro alpine sh -c 'ls -l /t/release/intarsia' | awk '{print $6, $7, $8}')"
systemctl --user stop radiance; sleep 3
noop() { # noop <tag> <tree> <vol> <gpu>
  RUN="env INTARSIA_REPO=$HOME/intarsiaml-$2 INTARSIA_TARGET_VOL=$3 HIP_VISIBLE_DEVICES=$4 $D /target/release/intarsia" \
    bash $S/../hg/noop.sh run "$O" "$1" hip $M8 $M35 $S/../hg/prompts < /dev/null
}
echo "[win11] A: no-op"
noop base sp-base intarsia-cargo-target-spbase 0 > "$O/noop-base.log" 2>&1 &
noop dev sp intarsia-cargo-target-sp 1 > "$O/noop-dev.log" 2>&1 &
wait
bash $S/../hg/noop.sh cmp "$O" base dev | tee "$O/noop-cmp.txt"
grep -q DIFFER "$O/noop-cmp.txt" && echo "FAIL spec-off no-op against main" || echo "PASS spec-off no-op against main (plan json/text + 256 ids, 8B and 35B, short and 4k)"
echo "[win11] B: MTP"
mtp() { # mtp <tree> <gpu> <prefix>
  t=$1; g=$2; p=$3
  export HIP_VISIBLE_DEVICES=$g
  bash $S/ids.sh $t $p-35-off 128,1024 --spec off
  bash $S/ids.sh $t $p-35-mtp 128,1024 --spec mtp --spec-k 2
  MODELS=$H MODEL=$M27 bash $S/ids.sh $t $p-27-off 128,1024 --spec off
  MODELS=$H MODEL=$M27 bash $S/ids.sh $t $p-27-mtp 128,1024 --spec mtp --spec-k 2
}
( mtp sp-base 0 b ) > "$O/mtp-base.log" 2>&1 &
( mtp sp 1 d ) > "$O/mtp-dev.log" 2>&1 &
wait
echo "== main (card 0)"; cat "$O/mtp-base.log"; echo "== final head (card 1)"; cat "$O/mtp-dev.log"
fails=0
acc() { grep -a "^$1 n$2 " "$O/mtp-$3.log" | sed -E 's/.*(passes=[0-9]+ proposed=[0-9]+ accepted=[0-9]+).*/\1/'; }
md() { grep -a "^$1 n$2 " "$O/mtp-$3.log" | sed -E 's/.*md5=([0-9a-f]+).*/\1/'; }
for m in 35 27; do for n in 128 1024; do
  [ -n "$(md d-$m-mtp $n dev)" ] && [ "$(md d-$m-mtp $n dev)" = "$(md d-$m-off $n dev)" ] && echo "PASS ${m}B n$n: MTP k=2 ids == spec off" || { echo "FAIL ${m}B n$n: MTP k=2 ids == spec off"; fails=$((fails+1)); }
  [ "$(md d-$m-mtp $n dev)" = "$(md b-$m-mtp $n base)" ] && echo "PASS ${m}B n$n: MTP k=2 ids == main's MTP" || { echo "FAIL ${m}B n$n: MTP k=2 ids == main's MTP"; fails=$((fails+1)); }
  a=$(acc d-$m-mtp $n dev); b=$(acc b-$m-mtp $n base)
  [ -n "$a" ] && [ "$a" = "$b" ] && echo "PASS ${m}B n$n: acceptance equal to main ($a)" || { echo "FAIL ${m}B n$n: acceptance $a against main's $b"; fails=$((fails+1)); }
done; done
cp $HOME/sp-ids/[bd]-3[5]-*.ids $HOME/sp-ids/[bd]-27-*.ids $O/ 2>/dev/null
echo "[win11] done: $fails failing MTP check(s)"
