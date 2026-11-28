#!/bin/bash
# endor EXCLUSIVE fleetq slot S, lane ink-metal-solo: gates S1-S5 of docs/results/2026-11-27-inkling2-metal-solo.md on one binary (the
# branch). Arms: solo = no --spec (the new default: DFlash2 alone on Metal); both = INTARSIA_SPEC_SAMPLED_DFLASH=1 (both drafts load, as
# main's default did); off = --spec off; dfl5 = explicit DFlash2 alone (#549's arm). Adapted from lane ink-metal-flip's slotG.sh / slotH.sh.
set -u
: "${FLEETQ_JOB:?run under fleetq}"
. $(dirname $0)/lib.sh
flags() { case $1 in off) echo "--spec off" ;; solo|both) echo "" ;; dfl5) echo "--spec dflash --spec-draft $D --spec-k 5" ;; esac; }
envs() { case $1 in both) echo "INTARSIA_SPEC_SAMPLED_DFLASH=1" ;; *) echo "" ;; esac; }
O=$R/S; mkdir -p $O; T0=$(date +%s)
echo "slot S start $(date +%T) binary md5 $(md5 -q $B) commit $(cat $B.commit) git $(git -C $W rev-parse --short HEAD)"
macmon pipe -i 250 > $S/macmon-S.jsonl 2>/dev/null &
G() { env $(envs $1) /usr/bin/time -l $B generate -m $M --backend metal --profile-every 0 --show-ids "${@:2}" $(flags $1); }
echo "== S1 + S2: greedy, generate, code + len4k + len16k, 256 tokens, --ignore-eos"
for arm in solo both off dfl5; do
  G $arm --max-seq 20480 --max-tokens 256 --ignore-eos --prompt-ids $R/g3.json > $O/g-$arm.out 2> $O/g-$arm.err
  echo "g-$arm rc $? t+$(( $(date +%s) - T0 ))s | $(grep -aE '^decode\[' $O/g-$arm.err | tr '\n' ' ' | cut -c1-300) | $(grep -aE '^spec: verify' $O/g-$arm.err | tr '\n' ' ' | cut -c1-400) | $(grep -aE 'peak memory footprint' $O/g-$arm.err)"
  echo "g-$arm load: $(grep -aE 'DFlash2 drafter|second draft|DSpark default|^--spec dflash|^dflash: |^loaded ' $O/g-$arm.err | tr '\n' '|' | cut -c1-900)"
done
python3 $R/md5.py $O/g-off.err $O/g-solo.err $O/g-both.err $O/g-dfl5.err
echo "== S3: sampled, generate, seed 7, 32 tokens, --ignore-eos (off vs solo; solo traced)"
for arm in off solo; do
  E=""; [ $arm = solo ] && E="INTARSIA_SPEC_TRACE=1"
  for cfg in "1.0 0 1.0 100" "0.6 0 1.0 60"; do
    set -- $cfg; f=$O/s-$arm-T$1-k$2
    env $E /usr/bin/time -l $B generate -m $M --backend metal --max-seq 4096 --profile-every 0 --show-ids --max-tokens 32 --ignore-eos \
      --temperature $1 --top-k $2 --top-p $3 --seed 7 --prompt-ids $R/w$4.json $(flags $arm) > $f.out 2> $f.err
    echo "$arm T$1 k$2 n$4 rc $? t+$(( $(date +%s) - T0 ))s | ids lines $(grep -ac '^ids\[' $f.err) | $(grep -aE '^spec: verify' $f.err | tr '\n' ' ' | cut -c1-200)"
  done
done
python3 $R/seedcmp.py $O off solo
echo "== S4: memory / KV capacity, generate --max-seq 4194304 (1048576 fallback) --kv-fraction 0.25, 64 tokens of the code prompt"
n=0
for arm in solo both solo both; do
  n=$((n+1)); f=$O/mem$n-$arm
  for MS in 4194304 1048576; do
    G $arm --max-seq $MS --kv-fraction 0.25 --max-tokens 64 --ignore-eos --prompt-ids $R/c1.json > $f.out 2> $f.err
    rc=$?; [ $rc = 0 ] && break
  done
  echo "mem$n $arm rc $rc max-seq $MS | $(grep -aE '^loaded ' $f.err | head -1 | cut -c1-260) | $(grep -aiE 'context|kv cache' $f.err | head -2 | tr '\n' ' ' | cut -c1-260) | $(grep -aE 'peak memory footprint|maximum resident' $f.err | tr '\n' ' ')"
  sleep 3
done
echo "== S5: serve --max-batch 1, solo default vs --spec off, multi-turn (chat endpoint + token-id extensions), prefix reuse"
i=0
for spec in solo:1 off:1; do
  arm=${spec%%:*}; n=${spec#*:}; i=$((i+1)); port=$((8410+i)); U=http://127.0.0.1:$port/v1; name=$arm-start$n
  [ $(( $(date +%s) - T0 )) -ge 1500 ] && { echo "slot S: time box before $name"; break; }
  t=$(date +%s)
  serve_start $name $arm $port "--max-batch 1"
  if python3 $R/client.py wait $U 420; then
    echo "$name up after $(( $(date +%s) - t )) s | $(grep -aE '^loaded ' $O/s-$name.err | head -1 | cut -c1-200) | $(grep -aE 'DFlash2 drafter|second draft|DSpark default|^--spec dflash' $O/s-$name.err | tr '\n' '|' | cut -c1-500)"
    python3 $R/client.py chat $U $O/chat-$name.json $R/b4.json $R/abba3.json
  else
    echo "$name NOT UP within 420 s"
  fi
  serve_stop
  echo "$name stopped t+$(( $(date +%s) - T0 ))s | $(grep -aE '^spec: verify|switched drafts' $O/s-$name.err | tr '\n' '|' | cut -c1-300)"
  sleep 3
done
python3 $R/client.py cmpchat $O/chat-solo-start1.json $O/chat-off-start1.json
echo "slot S end $(date +%T) t+$(( $(date +%s) - T0 ))s"
