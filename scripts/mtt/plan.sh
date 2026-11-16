#!/bin/zsh
# Lane mtt gate 2: `intarsia plan show --backend metal` byte-identical between main (BASE binary) and the branch, 8B / 27B / 35B x
# default / --spec off / --spec mtp, without and with admission (`--admit`: the path the lane changed). usage (fleetq slot): OUT=dir plan.sh
set -u
[ -n "${FLEETQ_JOB:-}" ] || { echo "not inside a fleetq slot" >&2; exit 2; }
W=${W:-/opt/code/intarsiaml/.worktrees/mtt}; NEW=$W/target/metal/release/intarsia; BASE=$W/target/base/release/intarsia; R=${OUT:?}; mkdir -p $R
ok=1
for m in Qwen3-8B Qwen3.8-27B-Quark-AWQ-MXFP4 Qwen3.6-35B-A3B-NVFP4; do
  for s in default off mtp; do
    for adm in plain admit; do
      FL=(); [ $s != default ] && FL=(--spec $s); [ $adm = admit ] && FL+=(--admit)
      t=$m-$s-$adm
      $BASE plan show --backend metal --model $HOME/models/$m --max-seq 40000 $FL > $R/$t.base.txt 2> $R/$t.base.err; eb=$?
      $NEW plan show --backend metal --model $HOME/models/$m --max-seq 40000 $FL > $R/$t.new.txt 2> $R/$t.new.err; en=$?
      if [ $eb = $en ] && cmp -s $R/$t.base.txt $R/$t.new.txt; then v=identical; else v=DIFFERENT; ok=0; fi
      echo "GATE plan $t: exit $eb/$en, $(wc -c < $R/$t.new.txt | tr -d ' ') bytes, $v$([ $eb != 0 ] && echo " ($(tail -1 $R/$t.new.err | cut -c1-90))")"
    done
  done
done
[ $ok = 1 ] && echo "GATE plan: PASS" || echo "GATE plan: FAIL"
