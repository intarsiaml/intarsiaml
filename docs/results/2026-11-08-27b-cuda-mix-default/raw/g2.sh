#!/usr/bin/env bash
# lane mxd gate 2, run on the box inside one fleetq job: 27B tp1 ids (default / INTARSIA_MTP_QUANT=off / --spec off) and a quick 4k decode default vs off.
set -uo pipefail
R=$HOME/intarsia-mxd; O=$R/out/g2; mkdir -p $O
BIN=$R/target-dev/release/intarsia
M=$R/models/Qwen3.8-27B-Quark-AWQ-MXFP4
IDP=$R/dev/scripts/cmx/ids-prompts; CORP=$R/dev/tools/prompts/corpus_docs.txt
export PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH
( while true; do a=$(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo); if [ "$a" -lt 10 ]; then echo "$(date +%T) MemAvailable $a GB: killing" >> $O/memwatch.log; pkill -KILL -f '[i]ntarsia-mxd/target-'; fi; sleep 1; done ) &
MW=$!
trap 'kill $MW 2>/dev/null; pkill -TERM -f "[i]ntarsia-mxd/target-" 2>/dev/null; true' EXIT
echo "stamp: $(strings $BIN | grep -m1 -o 'mxd-dev [0-9a-f]*')  avail $(free -g | awk '/Mem:/ {print $7}') GB"
# visit <tag> <arm: def|off|spec0> <prompt file> <max-seq> <ntok>
visit() {
  local tag=$1 arm=$2 pf=$3 ms=$4 nt=$5 env="" spec=""
  case $arm in off) env="INTARSIA_MTP_QUANT=off";; spec0) spec="--spec off";; esac
  env $env timeout 900 $BIN generate -m $M --backend cuda --max-seq $ms $spec --profile-every 0 --mem-budget 45 --chat --no-think --temperature 0 --ignore-eos --show-ids -p "$(cat $pf)" --max-tokens $nt < /dev/null > $O/$tag.out 2> $O/$tag.err
  echo "rc=$?" > $O/$tag.rc
  grep -a '^ids:' $O/$tag.err | head -1 > $O/$tag.ids
  echo "$tag: $(cat $O/$tag.rc) quantised-lines=$(grep -a -c 'mtp head quantised' $O/$tag.err) $(grep -a 'mtp head quantised' $O/$tag.err | head -1 | cut -c1-150) | $(grep -a 'finish=' $O/$tag.err | tail -1 | grep -oE 'prefill [0-9]+ ms \([0-9.]+ tok/s\) \| decode [0-9]+ ms \([0-9.]+ tok/s\)') | $(grep -a 'spec:' $O/$tag.err | tail -1 | cut -c1-120) | avail $(free -g | awk '/Mem:/ {print $7}')"
}
for p in short-0 4k-1; do
  ms=4096; [[ $p == 4k* ]] && ms=8192
  declare -A H
  for a in spec0 def off; do
    visit ids-$p-$a $a $IDP/$p.txt $ms 256
    H[$a]=$(cut -d: -f2- $O/ids-$p-$a.ids | md5sum | cut -c1-12); n=$(wc -c < $O/ids-$p-$a.ids)
    echo "IDS $p $a: md5 ${H[$a]} ($n B)"
  done
  for a in def off; do [[ ${H[$a]} == ${H[spec0]} ]] && echo "IDS $p: $a == spec off  PASS" || echo "IDS $p: $a DIFFERS from spec off  FAIL"; done
done
head -c 14035 $CORP > $O/p-4k.txt
i=0
for a in def off off def; do i=$((i+1)); visit speed4k-$i-$a $a $O/p-4k.txt 8192 192; done
echo "== done $(date +%T)"
