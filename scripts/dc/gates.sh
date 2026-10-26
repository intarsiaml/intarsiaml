#!/usr/bin/env bash
# Lane dc gates on spark1 (CUDA). Sources: ~/intarsiaml-dc-src (this worktree rsynced), base = origin/main tree ~/intarsiaml-dc-base.
#   gates.sh buildbase   (fleetq --load cpu)  clone-less export of origin/main to ~/intarsiaml-dc-base and build its CLI
#   gates.sh g0          (fleetq --quiet gpu,cpu)  flags off: dev vs base plan show (json+text) and 256 greedy ids, 8B short and 4k
#   gates.sh g3 <ids...> (fleetq --quiet gpu,cpu)  dev --spec off vs --spec dflash greedy ids, same prompts; prints spec: stats
set -uo pipefail
export PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH
H=/home/nvidia/.cache/huggingface/hub
M8=$(ls -d $H/models--Qwen--Qwen3-8B/snapshots/* | head -1)
DR=$(ls -d $H/models--TD-Labs--Qwen3-8B-DCOCO/snapshots/* | head -1)
SRC=$HOME/intarsiaml-dc-src; BASE=$HOME/intarsiaml-dc-base; O=$HOME/intarsiaml-dc-out
BIN_DEV=$HOME/intarsiaml-dc-target/release/intarsia; BIN_BASE=$HOME/intarsiaml-dc-target-base/release/intarsia
mkdir -p $O
md() { md5sum "$1" | cut -c1-8; }
case "${1:?}" in
buildbase)
  cd $BASE && CARGO_TARGET_DIR=$HOME/intarsiaml-dc-target-base CARGO_BUILD_JOBS=8 cargo build --release --features cuda -p intarsia-cli 2>&1 | grep -E '^error|Finished' | tail -3 ;;
builddev)
  cd $SRC && CARGO_TARGET_DIR=$HOME/intarsiaml-dc-target CARGO_BUILD_JOBS=8 cargo build --release --features cuda -p intarsia-cli 2>&1 | grep -E '^error|Finished' | tail -3 ;;
g0)
  PD=$SRC/scripts/hg/prompts; fails=0
  for t in base dev; do
    if [ $t = base ]; then B=$BIN_BASE; else B=$BIN_DEV; fi
    for p in short 4k; do
      $B generate -m $M8 --backend cuda --mem-budget 30 --max-seq 8192 --chat --no-think --temperature 0 --profile-every 0 --ignore-eos --show-ids \
        -p "$(cat $PD/$p.txt)" --max-tokens 256 < /dev/null > /dev/null 2> $O/g0-$p-$t.err
      grep -a '^ids:' $O/g0-$p-$t.err > $O/g0-$p-$t.ids
    done
    $B plan show --backend cuda --model $M8 --max-seq 8192 --mem-budget 30 --json < /dev/null > $O/g0-plan-$t.json 2> /dev/null
    $B plan show --backend cuda --model $M8 --max-seq 8192 --mem-budget 30 < /dev/null > $O/g0-plan-$t.txt 2> /dev/null
  done
  for f in short.ids 4k.ids; do :; done
  for p in short 4k; do cmp -s $O/g0-$p-base.ids $O/g0-$p-dev.ids && [ -s $O/g0-$p-dev.ids ] && echo "PASS g0 ids $p identical ($(tr ',' '\n' < $O/g0-$p-dev.ids | wc -l) ids, md5 $(md $O/g0-$p-dev.ids))" || { echo "FAIL g0 ids $p"; fails=$((fails+1)); }; done
  for k in json txt; do cmp -s $O/g0-plan-base.$k $O/g0-plan-dev.$k && [ -s $O/g0-plan-dev.$k ] && echo "PASS g0 plan $k byte-identical ($(wc -c < $O/g0-plan-dev.$k) B)" || { echo "FAIL g0 plan $k"; fails=$((fails+1)); }; done
  echo "g0 fails=$fails" ;;
g3)
  shift; fails=0
  for id in "$@"; do
    f=$SRC/scripts/dc/prompts/$id.txt; [ -f "$f" ] || f=$SRC/scripts/hg/prompts/$id.txt
    for mode in off dflash; do
      if [ $mode = off ] && [ "${MODE:-dflash}" != dflash ] && [ -s $O/g3-dflash-$id-off.err ]; then cp $O/g3-dflash-$id-off.err $O/g3-ngram-$id-off.err; cp $O/g3-dflash-$id-off.ids $O/g3-ngram-$id-off.ids; continue; fi
      if [ $mode = off ]; then A="--spec off"; elif [ "${MODE:-dflash}" = ngram ]; then A="--spec ngram --spec-k ${K:-4}"; elif [ "${MODE:-dflash}" = dcoco ] || [ "${MODE:-dflash}" = dcoco0 ]; then A="--spec dflash --spec-draft $DR --dcoco"; else A="--spec dflash --spec-draft $DR"; fi
      env ${GATE:+INTARSIA_SPEC_GATE=$GATE} ${ZERO:+INTARSIA_DCOCO_ZERO=1} $BIN_DEV generate -m $M8 --backend cuda --mem-budget 30 --max-seq 8192 $A --chat --no-think --temperature 0 --profile-every 0 --ignore-eos --show-ids \
        -p "$(cat $f)" --max-tokens ${NTOK:-192} < /dev/null > /dev/null 2> $O/g3-${MODE:-dflash}-$id-$mode.err
      grep -a '^ids:' $O/g3-${MODE:-dflash}-$id-$mode.err > $O/g3-${MODE:-dflash}-$id-$mode.ids
    done
    st=$(grep -a '^spec:' $O/g3-${MODE:-dflash}-$id-dflash.err | tail -1 | cut -c1-160); fin=$(grep -a 'finish=' $O/g3-${MODE:-dflash}-$id-dflash.err | tail -1 | sed 's/.*| decode/decode/' | cut -c1-60); fo=$(grep -a 'finish=' $O/g3-${MODE:-dflash}-$id-off.err | tail -1 | sed 's/.*| decode/decode/' | cut -c1-60)
    if cmp -s $O/g3-${MODE:-dflash}-$id-off.ids $O/g3-${MODE:-dflash}-$id-dflash.ids && [ -s $O/g3-${MODE:-dflash}-$id-off.ids ]; then echo "PASS g3 $id ids identical ($(tr ',' '\n' < $O/g3-${MODE:-dflash}-$id-off.ids | wc -l) ids) | off: $fo | dflash: $fin | $st"; else echo "FAIL g3 $id ids differ | $st"; fails=$((fails+1)); fi
  done
  echo "g3 fails=$fails" ;;
mtp35)
  # 35B NVFP4 --spec mtp (k=2) vs --spec off, greedy ids, long raw-text prompts from the repo docs; BIN_BASE = main.
  # usage: gates.sh mtp35 <tokens>... (env NP prompts per length, default 4; BIN overrides the binary)
  shift; fails=0; NP=${NP:-4}; B=${BIN:-$BIN_BASE}
  M35=$(ls -d $H/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/* | head -1)
  for T in "$@"; do
    case $T in 4096) MS=8192 ;; 16384) MS=20480 ;; *) MS=$((T + 4096)) ;; esac
    for k in $(seq 0 $((NP - 1))); do
      f=$O/p35-$T-$k.txt
      python3 - "$SRC/docs" $T $k > $f <<'PY'
import sys, glob
d, T, k = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
files = sorted(glob.glob(d + "/results/*.md")) + sorted(glob.glob(d + "/*.md"))
want = int(T * 3.3); out = []; n = 0; i = (k * 37 + (T // 4096) * 11) % len(files)
while n < want:
    t = open(files[i % len(files)], errors="ignore").read()
    out.append(t); n += len(t); i += 1
print("".join(out)[:want], end="")
PY
      for mode in off mtp; do
        if [ $mode = off ]; then A="--spec off"; else A="--spec mtp --spec-k 2"; fi
        env ${GATE:+INTARSIA_SPEC_GATE=$GATE} $B generate -m $M35 --backend cuda --mem-budget ${MEMB:-34} --max-seq $MS --profile-every 0 --ignore-eos --show-ids --max-tokens ${NTOK:-128} $A \
          -p "$(cat $f)" < /dev/null > /dev/null 2> $O/m35-$T-$k-$mode.err
        grep -a '^ids:' $O/m35-$T-$k-$mode.err > $O/m35-$T-$k-$mode.ids
      done
      pt=$(grep -a -o 'prompt[ _a-z]*[=:] *[0-9]*' $O/m35-$T-$k-mtp.err | head -1); st=$(grep -a '^spec:' $O/m35-$T-$k-mtp.err | tail -1 | cut -c1-150)
      fo=$(grep -a 'finish=' $O/m35-$T-$k-off.err | tail -1 | sed 's/.*| decode/decode/' | cut -c1-50); fm=$(grep -a 'finish=' $O/m35-$T-$k-mtp.err | tail -1 | sed 's/.*| decode/decode/' | cut -c1-50)
      if cmp -s $O/m35-$T-$k-off.ids $O/m35-$T-$k-mtp.ids && [ -s $O/m35-$T-$k-off.ids ]; then echo "PASS mtp35 T=$T p$k ($pt, $(wc -c < $f) chars) ids identical ($(tr ',' '\n' < $O/m35-$T-$k-off.ids | wc -l)) | off: $fo | mtp: $fm | $st"; else echo "FAIL mtp35 T=$T p$k ($pt) ids differ or empty | $st"; fails=$((fails+1)); fi
    done
  done
  echo "mtp35 fails=$fails" ;;
m35probe)
  # why does --spec mtp fail to load at --max-seq 8192? args: <max-seq>...
  shift; M35=$(ls -d $H/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/* | head -1)
  for MS in "$@"; do
    echo "== max-seq $MS"; $BIN_DEV generate -m $M35 --backend cuda --mem-budget 34 --max-seq $MS --profile-every 0 --max-tokens 4 --spec mtp --spec-k 2 -p "Hello" < /dev/null 2>&1 >/dev/null | grep -aE "Error|Caused|probe|loaded|invalid" | cut -c1-300
  done ;;
prosediag)
  # why does the gate not recover the prose loss? off vs dflash (gate on) vs dflash (INTARSIA_SPEC_COSTGATE=0), trace on. args: prompt ids
  shift
  for id in "$@"; do
    f=$SRC/scripts/dc/prompts/$id.txt
    for cfg in off gate nogate; do
      case $cfg in off) A="--spec off"; E="" ;; gate) A="--spec dflash --spec-draft $DR"; E="INTARSIA_SPEC_TRACE=1" ;; nogate) A="--spec dflash --spec-draft $DR"; E="INTARSIA_SPEC_TRACE=1 INTARSIA_SPEC_COSTGATE=0" ;; esac
      env $E $BIN_DEV generate -m $M8 --backend cuda --mem-budget 30 --max-seq 8192 $A --chat --no-think --temperature 0 --profile-every 0 --ignore-eos --show-ids -p "$(cat $f)" --max-tokens 256 < /dev/null > /dev/null 2> $O/pd-$id-$cfg.err
      echo "$id $cfg: $(grep -a 'finish=' $O/pd-$id-$cfg.err | tail -1 | sed 's/.*| decode/decode/' | cut -c1-60) | $(grep -a '^spec:' $O/pd-$id-$cfg.err | grep -v 'cost sample\|gate holds' | tail -1 | cut -c1-140)"
      grep -a -c 'cost sample plain' $O/pd-$id-$cfg.err | sed "s/^/  plain samples: /"
      grep -a 'cost sample' $O/pd-$id-$cfg.err | awk '{k=$4; n[k]++; t[k]+=$5; ms[k]+=$7} END {for (k in n) printf "  %s: %d samples, %.2f tokens/sample, %.1f ms/sample\n", k, n[k], t[k]/n[k], ms[k]/n[k]}'
      grep -a -c 'holds speculation off' $O/pd-$id-$cfg.err | sed "s/^/  gate-off notes: /"
    done
  done ;;
esac
