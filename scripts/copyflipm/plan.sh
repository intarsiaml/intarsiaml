#!/bin/zsh
# copyflipm plan gates, inside a fleetq slot on endor: `plan show --json` of the new binary against the base (origin/main at the branch point).
# (`plan show` has no --max-batch: the mb8 side of the gate is serve.sh, which compares the served plan lines.)
[ -n "${FLEETQ_JOB:-}" ] || { echo "not inside a fleetq slot" >&2; exit 2; }
W=${0:A:h:h:h}; RAW=$W/docs/results/2026-11-18-copyflipm/raw; mkdir -p $RAW/plan; rm -f $RAW/plan/*
BASE=$W/target-cf-base/intarsia-base; NEW=$W/target-cf/release/intarsia; M=$HOME/models/Qwen3.6-35B-A3B-NVFP4
# show NAME BIN "ENV=.." backend args...: stdout (the plan json), exit code, stderr with the load timings blanked
show() { local n=$1 b=$2 e=$3 be=$4; shift 4; env ${=e:-X=1} $b plan show --model $M --backend $be "$@" --max-seq 8192 --json < /dev/null > $RAW/plan/$n.json 2> $RAW/plan/$n.err.raw; echo $? > $RAW/plan/$n.exit
  sed -E 's/[0-9]+\.[0-9]+ (s|ms)/T/g' $RAW/plan/$n.err.raw > $RAW/plan/$n.err; rm $RAW/plan/$n.err.raw
  echo "$n: exit $(cat $RAW/plan/$n.exit) $(wc -c < $RAW/plan/$n.json | tr -d ' ') bytes sha $(shasum -a 256 < $RAW/plan/$n.json | cut -c1-16)"; }
MTP=(--spec mtp --spec-k 2)
show base-mb1 $BASE "" metal $MTP;                  show new-mb1 $NEW "" metal $MTP
show base-mb1-c20 $BASE INTARSIA_SPEC_COPY=20 metal $MTP
show new-mb1-c0 $NEW INTARSIA_SPEC_COPY=0 metal $MTP; show new-mb1-cempty $NEW INTARSIA_SPEC_COPY= metal $MTP
show base-mb1-c20-7 $BASE INTARSIA_SPEC_COPY=20,7 metal $MTP; show new-mb1-c20-7 $NEW INTARSIA_SPEC_COPY=20,7 metal $MTP
show base-off $BASE "" metal --spec off;            show new-off $NEW "" metal --spec off
show base-off-c20 $BASE INTARSIA_SPEC_COPY=20 metal --spec off; show new-off-c20 $NEW INTARSIA_SPEC_COPY=20 metal --spec off
show base-cpu $BASE "" cpu $MTP;                    show new-cpu $NEW "" cpu $MTP
show base-cpu-c20 $BASE INTARSIA_SPEC_COPY=20 cpu $MTP; show new-cpu-c20 $NEW INTARSIA_SPEC_COPY=20 cpu $MTP
# no --spec at all (the recipe's MTP default applies on Metal at tp 1: the most common config the default reaches), the 35B and the Qwen3.8-27B
show base-ns $BASE "" metal;                        show new-ns $NEW "" metal
show base-ns-c20 $BASE INTARSIA_SPEC_COPY=20 metal; show new-ns-c0 $NEW INTARSIA_SPEC_COPY=0 metal
show base-ns-sd0 $BASE INTARSIA_SPEC_DEFAULT=0 metal; show new-ns-sd0 $NEW INTARSIA_SPEC_DEFAULT=0 metal
M=$HOME/models/Qwen3.8-27B-Quark-AWQ-MXFP4
show base-27-ns $BASE "" metal;                     show new-27-ns $NEW "" metal
show base-27-ns-c20 $BASE INTARSIA_SPEC_COPY=20 metal; show new-27-ns-c20 $NEW INTARSIA_SPEC_COPY=20 metal; show new-27-ns-c0 $NEW INTARSIA_SPEC_COPY=0 metal
show base-27-off $BASE "" metal --spec off;         show new-27-off $NEW "" metal --spec off
M=$HOME/models/Qwen3.6-35B-A3B-NVFP4
pf=0
# chk A B same|diff LABEL: stdout, normalised stderr and exit code all equal / not all equal
chk() { local s=1; for x in json err exit; do cmp -s $RAW/plan/$1.$x $RAW/plan/$2.$x || s=0; done
  if [ $s = 1 ]; then [ "$3" = same ] && echo "PASS plan $4: $1 == $2 (exit $(cat $RAW/plan/$1.exit))" || { echo "FAIL plan $4: $1 == $2 (should differ)"; pf=$((pf+1)); }
  else [ "$3" = diff ] && echo "PASS plan $4: $1 != $2 (as intended)" || { echo "FAIL plan $4: $1 != $2"; pf=$((pf+1)); }; fi; }
# the default's plan equals base's explicit =20 plan on the json (stderr says nothing of copy at load): compare json and exit only
cj() { cmp -s $RAW/plan/$1.json $RAW/plan/$2.json && cmp -s $RAW/plan/$1.exit $RAW/plan/$2.exit && echo "PASS plan $3: $1 json == $2 json" || { echo "FAIL plan $3: $1 json != $2 json"; pf=$((pf+1)); }; }
cj base-mb1-c20 new-mb1 "(a) default mb1 == base =20"
chk base-mb1 new-mb1 diff "(a') default differs from base unset"
chk base-mb1 new-mb1-c0 same "(b) =0 == base unset (byte-identical)"
chk base-mb1 new-mb1-cempty same "(b') empty == base unset"
chk base-mb1-c20-7 new-mb1-c20-7 same "explicit =20,7 unchanged"
chk base-off new-off same "(d) --spec off unchanged"
chk base-off-c20 new-off-c20 same "--spec off with explicit =20 unchanged"
chk base-cpu new-cpu same "(e) cpu unset unchanged"
chk base-cpu-c20 new-cpu-c20 same "cpu explicit =20 unchanged"
cj base-ns-c20 new-ns "(f) no --spec (recipe MTP default), 35B: default == base =20"
chk base-ns new-ns diff "(f2) no --spec, 35B: default differs from base unset"
chk base-ns new-ns-c0 same "(f3) no --spec, 35B: =0 == base unset"
chk base-ns-sd0 new-ns-sd0 same "(g) INTARSIA_SPEC_DEFAULT=0, 35B: unchanged"
chk base-27-ns new-27-ns same "(h) no --spec, Qwen3.8-27B: unset byte-identical to base unset (the recipe does not opt in)"
chk base-27-ns-c20 new-27-ns-c20 same "(h2) 27B explicit =20 unchanged"
chk base-27-ns new-27-ns-c0 same "(h3) no --spec, 27B: =0 == base unset"
chk base-27-off new-27-off same "(h4) --spec off, 27B: unchanged"
echo "plan fails=$pf; cpu exit base/new $(cat $RAW/plan/base-cpu.exit)/$(cat $RAW/plan/new-cpu.exit)"
