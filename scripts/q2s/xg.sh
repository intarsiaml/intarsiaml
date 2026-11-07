#!/usr/bin/env bash
# Lane q2s (docs/results/2026-11-07-27b-tp2-mtp-quant.md): the Qwen3.8-27B (MXFP4 W4A16) on CUDA, `--spec mtp` at tp=2 across the two Sparks
# (rank 0 = spark1 `generate --tp-remote`, rank 1 = spark2 `tp-worker`, NCCL over CX-7 port f1), against tp=1 on spark1. vLLM stays UP on both
# boxes. Same `generate` method as lane c27 (scripts/c27/tp2.sh, ab.sh). Runs HERE (the Mac) as the one command of ONE fleetq job holding both:
#   STEPS="ids speed" fleetq run --host spark1,spark2 --mem spark1=50,spark2=30 --quiet gpu,cpu --expect 75 --label "..." -- scripts/q2s/xg.sh
# (speed skips itself when an ids comparison DIFFERS)
# Arm names: t<tp>_off (plain), t<tp>_<q>_<k> (--spec mtp --spec-k k, INTARSIA_MTP_QUANT=<q>, q = u (unset) | fp8 | mix | mxfp4).
# Steps: ids      256 greedy ids, prompts IDSP (short-0 4k-1), IDARMS (default t2_off t2_fp8_2 t2_mix_2); each arm's md5 against the first arm
#        speed    decode tok/s at SIZES (1k 4k 16k), ARMS, ROUNDS (2; even rounds reversed = ABAB), NTOK (192) tokens, greedy
#        sampled  tp1, SARMS (default t1_u_2 t1_fp8_2 t1_mix_2) x SPROMPTS (short-0 short-1 short-2 mid-0 mid-1), 256 tokens,
#                 temperature 0.7 top-p 0.95 seed 1234, spec acceptance and tokens per verify pass from the `spec:` line
# A memory watcher on each box kills this lane's processes if MemAvailable drops below 10 GB. Every process started here is killed on exit.
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
S="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; REPO="$(cd "$S/../.." && pwd)"
OUT=${OUT:-$(date +%H%M)}
L=$REPO/docs/results/2026-11-07-27b-tp2-mtp-quant/raw/$OUT; mkdir -p "$L"
P='$HOME/intarsia-q2s'; M="$P/models/Qwen3.8-27B-Quark-AWQ-MXFP4"; BIN="$P/bin/${BINNAME:-dev}-intarsia"
CORP=$P/dev/tools/prompts/corpus_docs.txt; IDP=$P/dev/scripts/cmx/ids-prompts
WIP=10.100.152.3; WPORT=${WPORT:-29571}
NCCLENV="INTARSIA_NCCL_LIB=\$HOME/intarsia-nccl/libnccl.so.2 NCCL_SOCKET_IFNAME=enp1s0f1np1 NCCL_IB_HCA=rocep1s0f1,roceP2p1s0f1 NCCL_IB_GID_INDEX=3 NCCL_DEBUG=WARN NCCL_MAX_NCHANNELS=8"
want() { [[ -z "${STEPS:-}" || " $STEPS " == *" $1 "* ]]; }
log() { echo "$(date +%T) $*" | tee -a "$L/steps.log"; }
r() { local h=$1; shift; ssh -n -o BatchMode=yes "$h" "export FLEETQ_JOB=$FLEETQ_JOB; $*"; }
XO="$P/out/$OUT"
kill_lane() { for h in spark1 spark2; do r $h "pkill -TERM -f '[i]ntarsia-q2s/bin/'; for i in \$(seq 1 20); do pgrep -f '[i]ntarsia-q2s/bin/' >/dev/null || break; sleep 1; done; pkill -KILL -f '[i]ntarsia-q2s/bin/'; true"; done; }
cleanup() {
  local rc=$?
  log "cleanup (exit $rc)"
  kill_lane
  for h in spark1 spark2; do r $h "pkill -f '[i]ntarsia-q2s/memwatch'; true"; done
  rsync -a --exclude 'p-*.txt' "spark1:intarsia-q2s/out/$OUT/" "$L/spark1/" 2>/dev/null; rsync -a "spark2:intarsia-q2s/out/$OUT/" "$L/spark2/" 2>/dev/null
  trap - EXIT INT TERM; exit $rc
}
trap cleanup EXIT INT TERM
for h in spark1 spark2; do
  r $h "mkdir -p $XO; echo $h: \$(strings $BIN | grep -m1 -o 'head=[0-9a-f]* code-tree=[0-9a-f]* dirty=[a-z]*'); free -g | sed -n 2p" | tee -a "$L/steps.log"
  scp -q "$S/memwatch.sh" $h:intarsia-q2s/memwatch.sh && r $h "nohup bash $P/memwatch.sh $XO > /dev/null 2>&1 < /dev/null &"
done
sleep 1; for h in spark1 spark2; do r $h "pgrep -f '[i]ntarsia-q2s/memwatch' >/dev/null && echo $h memwatch up || echo $h MEMWATCH MISSING" | tee -a "$L/steps.log"; done
chars() { case $1 in 1k) echo 3300;; 4k) echo 14035;; 16k) echo 54000;; esac; }
# visit <tag> <arm> <prompt file on spark1> <max-seq> <ntok> [extra generate flags]: one fresh process group per visit
visit() {
  local tag=$1 arm=$2 pf=$3 ms=$4 nt=$5 xf=${6:-} spec="--spec off" tp=1 qe=""
  [[ $arm == t2* ]] && tp=2
  if [[ $arm != *_off ]]; then
    local q=$(echo $arm | cut -d_ -f2) k=$(echo $arm | cut -d_ -f3)
    spec="--spec mtp --spec-k $k"; [[ $q != u ]] && qe="INTARSIA_MTP_QUANT=$q"
  fi
  local gen="-m $M --backend cuda --max-seq $ms $spec --profile-every 0"
  if [[ $tp == 2 ]]; then
    r spark2 "nohup bash -c 'env $NCCLENV $qe $BIN tp-worker --listen $WIP:$WPORT -m $M --backend cuda --tp 2 --max-seq $ms --mem-budget 24 --profile-every 0 $spec > $XO/worker-$tag.log 2>&1' > /dev/null 2>&1 < /dev/null &"
    sleep 3
    gen="$gen --tp 2 --mem-budget 24 --tp-remote $WIP:$WPORT"
  else
    gen="$gen --mem-budget 45"
  fi
  r spark1 "env $NCCLENV $qe timeout 900 $BIN generate $gen --chat --no-think ${xf:---temperature 0} --ignore-eos --show-ids -p \"\$(cat $pf)\" --max-tokens $nt < /dev/null > $XO/$tag.out 2> $XO/$tag.err; echo rc=\$? > $XO/$tag.rc; grep -a '^ids:' $XO/$tag.err | head -1 > $XO/$tag.ids"
  [[ $tp == 2 ]] && r spark2 "for i in \$(seq 1 30); do pgrep -f '[t]p-worker --listen $WIP:$WPORT' >/dev/null || exit 0; sleep 1; done; pkill -TERM -f '[t]p-worker --listen $WIP:$WPORT'; sleep 3; pkill -KILL -f '[t]p-worker --listen $WIP:$WPORT'; true"
  log "$tag: $(r spark1 "cat $XO/$tag.rc; grep -a 'finish=' $XO/$tag.err | tail -1 | grep -oE 'prefill [0-9]+ ms \([0-9.]+ tok/s\) \| decode [0-9]+ ms \([0-9.]+ tok/s\)'; grep -a 'spec:' $XO/$tag.err | tail -1 | cut -c1-160; grep -a -m2 -iE 'unsupported|^error|panicked' $XO/$tag.err | cut -c1-200; echo avail \$(free -g | awk '/Mem:/ {print \$7}')" | tr '\n' '|') worker: $([[ $tp == 2 ]] && r spark2 "grep -a -E 'decode graph|serving' $XO/worker-$tag.log | tail -2 | cut -c1-140 | tr '\n' '|'")"
}
if want ids; then
  for p in ${IDSP:-short-0 4k-1}; do
    ms=4096; [[ $p == 4k* ]] && ms=8192
    pf=$IDP/$p.txt; first=""; h0=""
    for a in ${IDARMS:-t2_off t2_fp8_2 t2_mix_2}; do
      visit ids-$p-$a $a $pf $ms 256
      h=$(r spark1 "cut -d: -f2- $XO/ids-$p-$a.ids | md5sum | cut -c1-12"); n=$(r spark1 "wc -c < $XO/ids-$p-$a.ids")
      if [[ -z $first ]]; then first=$a; h0=$h; log "IDS $p: reference $a ($n B, md5 $h0)"
      elif [[ $h == "$h0" && ${n:-0} -gt 100 ]]; then log "IDS $p: $a == $first ($n B, md5 $h)"; else log "IDS $p: $a DIFFERS from $first or empty (ref $h0, got $h, $n B)"; fi
    done
  done
fi
if want speed; then
  ARMS=${ARMS:-t2_u_2 t2_fp8_2 t2_mix_2}
  for sz in ${SIZES:-1k 4k 16k}; do
    ms=8192; [[ $sz == 16k ]] && ms=20480
    r spark1 "head -c $(chars $sz) $CORP > $XO/p-$sz.txt"
    for rd in $(seq ${ROUND_FROM:-1} ${ROUNDS:-2}); do
      order="$ARMS"; [[ $((rd % 2)) == 0 ]] && order=$(echo $ARMS | tr ' ' '\n' | awk '{a[NR]=$0} END {for (i=NR; i>=1; i--) print a[i]}' | tr '\n' ' ')
      for a in $order; do visit r$rd-$sz-$a $a $XO/p-$sz.txt $ms ${NTOK:-192}; done
    done
  done
fi
if want sampled; then
  for p in ${SPROMPTS:-short-0 short-1 short-2 mid-0 mid-1}; do
    for a in ${SARMS:-t1_u_2 t1_fp8_2 t1_mix_2}; do visit smp-$p-$a $a $IDP/$p.txt 4096 256 "--temperature 0.7 --top-p 0.95 --seed ${SEED:-1234}"; done
  done
fi
log done
