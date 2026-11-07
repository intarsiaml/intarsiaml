#!/usr/bin/env bash
# Lane d2r (docs/results/2026-11-07-dense-mtp-remote-tp2.md): the Qwen3.8-27B (MXFP4 W4A16) on CUDA, `--spec mtp` at tp=2 across the two Sparks
# (rank 0 = spark1 `generate --tp-remote`, rank 1 = spark2 `tp-worker`, NCCL over CX-7 port f1), against tp=1 on spark1. vLLM stays UP on both
# boxes. Same `generate` method as lane c27 (scripts/c27/tp2.sh, ab.sh). Runs HERE (the Mac) as the one command of ONE fleetq job holding both:
#   STEPS="ids speed" fleetq run --host spark1,spark2 --mem spark1=50,spark2=30 --quiet gpu,cpu --expect 75 --label "..." -- scripts/d2r/xg.sh
# (speed skips itself when an ids comparison DIFFERS)
# Steps: ids    256 greedy ids, prompts IDSP (short-0 4k-1; scripts/cmx/ids-prompts), tp2 `--spec off` vs tp2 `--spec mtp --spec-k 2` (rule a); tp1 MTP k2
#               recorded for its acceptance (the ids cannot see a bad head: every emitted token is a verify row's argmax)
#        speed  decode tok/s at SIZES (1k 4k 16k), ARMS (tp1m tp2off tp2m), ROUNDS (2; even rounds reversed = ABAB), NTOK (192) tokens
# A memory watcher on each box kills this lane's processes if MemAvailable drops below 10 GB. Every process started here is killed on exit.
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
S="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; REPO="$(cd "$S/../.." && pwd)"
OUT=${OUT:-$(date +%H%M)}
L=$REPO/docs/results/2026-11-07-dense-mtp-remote-tp2/raw/$OUT; mkdir -p "$L"
P='$HOME/intarsia-d2r'; M="$P/models/Qwen3.8-27B-Quark-AWQ-MXFP4"; BIN="$P/bin/${BINNAME:-dev}-intarsia"
CORP=$P/dev/tools/prompts/corpus_docs.txt; IDP=$P/dev/scripts/cmx/ids-prompts
WIP=10.100.152.3; WPORT=${WPORT:-29571}
NCCLENV="INTARSIA_NCCL_LIB=\$HOME/intarsia-nccl/libnccl.so.2 NCCL_SOCKET_IFNAME=enp1s0f1np1 NCCL_IB_HCA=rocep1s0f1,roceP2p1s0f1 NCCL_IB_GID_INDEX=3 NCCL_DEBUG=WARN NCCL_MAX_NCHANNELS=8"
want() { [[ -z "${STEPS:-}" || " $STEPS " == *" $1 "* ]]; }
log() { echo "$(date +%T) $*" | tee -a "$L/steps.log"; }
r() { local h=$1; shift; ssh -n -o BatchMode=yes "$h" "export FLEETQ_JOB=$FLEETQ_JOB; $*"; }
XO="$P/out/$OUT"
kill_lane() { for h in spark1 spark2; do r $h "pkill -TERM -f '[i]ntarsia-d2r/bin/'; for i in \$(seq 1 20); do pgrep -f '[i]ntarsia-d2r/bin/' >/dev/null || break; sleep 1; done; pkill -KILL -f '[i]ntarsia-d2r/bin/'; true"; done; }
cleanup() {
  local rc=$?
  log "cleanup (exit $rc)"
  kill_lane
  for h in spark1 spark2; do r $h "pkill -f '[i]ntarsia-d2r/memwatch'; true"; done
  rsync -a --exclude 'p-*.txt' "spark1:intarsia-d2r/out/$OUT/" "$L/spark1/" 2>/dev/null; rsync -a "spark2:intarsia-d2r/out/$OUT/" "$L/spark2/" 2>/dev/null
  trap - EXIT INT TERM; exit $rc
}
trap cleanup EXIT INT TERM
for h in spark1 spark2; do
  r $h "mkdir -p $XO; echo $h: \$(strings $BIN | grep -m1 -o 'head=[0-9a-f]* code-tree=[0-9a-f]* dirty=[a-z]*'); free -g | sed -n 2p" | tee -a "$L/steps.log"
  scp -q "$S/memwatch.sh" $h:intarsia-d2r/memwatch.sh && r $h "nohup bash $P/memwatch.sh $XO > /dev/null 2>&1 < /dev/null &"
done
sleep 1; for h in spark1 spark2; do r $h "pgrep -f '[i]ntarsia-d2r/memwatch' >/dev/null && echo $h memwatch up || echo $h MEMWATCH MISSING" | tee -a "$L/steps.log"; done
chars() { case $1 in 1k) echo 3300;; 4k) echo 14035;; 16k) echo 54000;; esac; }
# visit <tag> <arm: tp1m|tp2off|tp2m|tp1off> <prompt file on spark1> <max-seq> <ntok>: one fresh process group per visit
visit() {
  local tag=$1 arm=$2 pf=$3 ms=$4 nt=$5 spec="--spec off" tp=1
  case $arm in tp1m|tp2m) spec="--spec mtp --spec-k 2";; esac
  case $arm in tp2*) tp=2;; esac
  local gen="-m $M --backend cuda --max-seq $ms $spec --profile-every 0"
  if [[ $tp == 2 ]]; then
    r spark2 "nohup bash -c 'env $NCCLENV $BIN tp-worker --listen $WIP:$WPORT -m $M --backend cuda --tp 2 --max-seq $ms --mem-budget 24 --profile-every 0 $spec > $XO/worker-$tag.log 2>&1' > /dev/null 2>&1 < /dev/null &"
    sleep 3
    gen="$gen --tp 2 --mem-budget 24 --tp-remote $WIP:$WPORT"
  else
    gen="$gen --mem-budget 45"
  fi
  r spark1 "env $NCCLENV timeout 900 $BIN generate $gen --chat --no-think --temperature 0 --ignore-eos --show-ids -p \"\$(cat $pf)\" --max-tokens $nt < /dev/null > $XO/$tag.out 2> $XO/$tag.err; echo rc=\$? > $XO/$tag.rc; grep -a '^ids:' $XO/$tag.err | head -1 > $XO/$tag.ids"
  [[ $tp == 2 ]] && r spark2 "for i in \$(seq 1 30); do pgrep -f '[t]p-worker --listen $WIP:$WPORT' >/dev/null || exit 0; sleep 1; done; pkill -TERM -f '[t]p-worker --listen $WIP:$WPORT'; sleep 3; pkill -KILL -f '[t]p-worker --listen $WIP:$WPORT'; true"
  log "$tag: $(r spark1 "cat $XO/$tag.rc; grep -a 'finish=' $XO/$tag.err | tail -1 | grep -oE 'prefill [0-9]+ ms \([0-9.]+ tok/s\) \| decode [0-9]+ ms \([0-9.]+ tok/s\)'; grep -a 'spec:' $XO/$tag.err | tail -1 | cut -c1-160; grep -a -m2 -iE 'unsupported|^error|panicked' $XO/$tag.err | cut -c1-200; echo avail \$(free -g | awk '/Mem:/ {print \$7}')" | tr '\n' '|') worker: $([[ $tp == 2 ]] && r spark2 "grep -a -E 'decode graph|serving' $XO/worker-$tag.log | tail -2 | cut -c1-140 | tr '\n' '|'")"
}
if want ids; then
  for p in ${IDSP:-short-0 4k-1}; do
    ms=4096; [[ $p == 4k* ]] && ms=8192
    pf=$IDP/$p.txt
    for a in tp1m tp2off tp2m; do visit ids-$p-$a $a $pf $ms 256; done
    h0=$(r spark1 "cut -d: -f2- $XO/ids-$p-tp2off.ids | md5sum | cut -c1-12"); h1=$(r spark1 "cut -d: -f2- $XO/ids-$p-tp2m.ids | md5sum | cut -c1-12")
    n1=$(r spark1 "wc -c < $XO/ids-$p-tp2m.ids")
    if [[ $h0 == "$h1" && ${n1:-0} -gt 100 ]]; then log "IDS $p: tp2m == tp2off ($n1 B, md5 $h0)"; else log "IDS $p: tp2m DIFFERS from tp2off or empty (off $h0, mtp $h1, $n1 B)"; fi
  done
fi
if want speed && grep -q "IDS .*DIFFERS" "$L/steps.log"; then log "speed skipped: rule (a) failed"; elif want speed; then
  ARMS=${ARMS:-tp1m tp2off tp2m}
  for sz in ${SIZES:-1k 4k 16k}; do
    ms=8192; [[ $sz == 16k ]] && ms=20480
    r spark1 "head -c $(chars $sz) $CORP > $XO/p-$sz.txt"
    for rd in $(seq ${ROUND_FROM:-1} ${ROUNDS:-2}); do
      order="$ARMS"; [[ $((rd % 2)) == 0 ]] && order=$(echo $ARMS | tr ' ' '\n' | awk '{a[NR]=$0} END {for (i=NR; i>=1; i--) print a[i]}' | tr '\n' ' ')
      for a in $order; do visit r$rd-$sz-$a $a $XO/p-$sz.txt $ms ${NTOK:-192}; done
    done
  done
fi
log done
