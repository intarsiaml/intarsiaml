#!/usr/bin/env bash
# Lane sp follow-up window, ON tatooine inside a fleetq job that takes radiance down (stop below, restore in the trap).
# main (tree sp-base, card 0) against the follow-up branch (tree sp, card 1), in parallel:
#   A: spec-off no-op, 8B and 35B, short and 4k: plan show json/text and 256 greedy ids (noop.sh)
#   B: MTP k=2 on the 35B (natural-v1 128/1k/4k) and the 27B (4k/16k): spec off, captured, eager (INTARSIA_SPEC_GRAPH=0);
#      dev also --mtp-draft-vocab 65536 (35B 128/1k, 27B 4k). md5 of the 256 ids, acceptance, decode tok/s.
set -uo pipefail
O=$HOME/sp-logs/win8; mkdir -p "$O" $HOME/sp-ids
S=$HOME/intarsiaml-sp/scripts/sp
D=$HOME/intarsiaml-sp/deploy/tatooine/dev.sh
M8=/models/Qwen3-8B; M35=/models/Qwen3.6-35B-A3B-NVFP4
H=/data/models/radiance/hf-cache/hub
M27=/models/models--amd--Qwen3.8-27B-Quark-AWQ-MXFP4/snapshots/5233554c5fa56afda40150556b95573c2d7d29c0
trap 'bash $HOME/sp-win.sh restore' EXIT
systemctl --user stop radiance; sleep 3
noop() { # noop <tag> <tree> <vol> <gpu>
  RUN="env INTARSIA_REPO=$HOME/intarsiaml-$2 INTARSIA_TARGET_VOL=$3 HIP_VISIBLE_DEVICES=$4 $D /target/release/intarsia" \
    bash $S/../hg/noop.sh run "$O" "$1" hip $M8 $M35 $S/../hg/prompts < /dev/null
}
echo "[win8] A: no-op"
noop base sp-base intarsia-cargo-target-spbase 0 > "$O/noop-base.log" 2>&1 &
noop dev sp intarsia-cargo-target-sp 1 > "$O/noop-dev.log" 2>&1 &
wait
cat "$O/noop-base.log" "$O/noop-dev.log"
bash $S/../hg/noop.sh cmp "$O" base dev | tee "$O/noop-cmp.txt"
echo "[win8] B: MTP"
mtp() { # mtp <tree> <gpu> <tag-prefix> [vocab: 1 to include the draft-vocab runs]
  t=$1; g=$2; p=$3
  export HIP_VISIBLE_DEVICES=$g
  bash $S/ids.sh $t $p-35-off 128,1024,4096 --spec off
  bash $S/ids.sh $t $p-35-cap 128,1024,4096 --spec mtp --spec-k 2
  INTARSIA_SPEC_GRAPH=0 bash $S/ids.sh $t $p-35-eager 128,1024 --spec mtp --spec-k 2
  [ "${4:-0}" = 1 ] && bash $S/ids.sh $t $p-35-v65536 128,1024 --spec mtp --spec-k 2 --mtp-draft-vocab 65536
  MODELS=$H MODEL=$M27 bash $S/ids.sh $t $p-27-off 4096,16384 --spec off
  MODELS=$H MODEL=$M27 bash $S/ids.sh $t $p-27-cap 4096,16384 --spec mtp --spec-k 2
  MODELS=$H MODEL=$M27 INTARSIA_SPEC_GRAPH=0 bash $S/ids.sh $t $p-27-eager 4096,16384 --spec mtp --spec-k 2
  [ "${4:-0}" = 1 ] && MODELS=$H MODEL=$M27 bash $S/ids.sh $t $p-27-v65536 4096 --spec mtp --spec-k 2 --mtp-draft-vocab 65536
}
( mtp sp-base 0 b 0 ) > "$O/mtp-base.log" 2>&1 &
( mtp sp 1 d 1 ) > "$O/mtp-dev.log" 2>&1 &
wait
echo "== main (card 0)"; cat "$O/mtp-base.log"; echo "== follow-up branch (card 1)"; cat "$O/mtp-dev.log"
cp $HOME/sp-ids/[bd]-3[5]-*.ids $HOME/sp-ids/[bd]-27-*.ids $O/ 2>/dev/null
echo "[win8] done"
