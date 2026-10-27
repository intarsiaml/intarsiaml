#!/usr/bin/env bash
# G4X same-weights accuracy gates (docs/accuracy.md "Same-weights regression gate"; run INSIDE a fleetq slot, --quiet gpu,cpu, --mem >= 34).
#   selfgold.sh write <binary>                 the pre-change build records replay-out/acc/base.baseline.json + base.self.safetensors
#   selfgold.sh check <name> <binary> [ENV=V]  a new build against both: baseline (x1.15 margins) and self-golden vs reference noise
set -u
R=$HOME/intarsiaml-g4x; M=$HOME/models/Gemma-4-26B-A4B-NVFP4; G=$R/golden/gemma4-26b-a4b-nvfp4-fp32.safetensors; A=$R/replay-out/acc
mode=$1; shift
gb=$(( $(awk '/MemAvailable/ {print $2}' /proc/meminfo) / 1024 / 1024 )); echo "MemAvailable ${gb} GB"; (( gb >= 45 )) || { echo "ABORT: memory"; exit 1; }
mkdir -p $A
COMMON=(--model $M --backend cuda --max-seq 8192 --mem-budget 30 --prefill-chunk 1024 --allow-dirty --allow-unstamped --allow-same-code)
PRE=(--pre-change ${PRE_CHANGE:-2859c1a89} --revlist $R/replay-out/revlist.txt)  # revlist: `git rev-list <pre-change>`, scp'd next to this script
if [ $mode = write ]; then
  bin=$R/replay-out/$1
  $bin accuracy "${COMMON[@]}" --golden $G --mode both --write-baseline $A/base.baseline.json --write-self-golden $A/base.self.safetensors > $A/base.write.log 2>&1
  echo "write exit $?"; grep -aE "golden gate|^Error" $A/base.write.log | head -3
else
  name=$1; bin=$R/replay-out/$2; shift 2
  ENVS=("$@"); [ ${#ENVS[@]} -eq 0 ] && ENVS=(INTARSIA_X=0)
  env "${ENVS[@]}" $bin accuracy "${COMMON[@]}" --golden $G --mode both --baseline $A/base.baseline.json "${PRE[@]}" > $A/$name.baseline.log 2>&1
  echo "$name baseline exit $?"; grep -aE "baseline verdict|golden gate|^Error|baseline gate|exceeds|worse" $A/$name.baseline.log | cut -c1-300 | head -12
  env "${ENVS[@]}" $bin accuracy "${COMMON[@]}" --self-golden $A/base.self.safetensors --self-vs-ref-noise --mode both "${PRE[@]}" > $A/$name.self.log 2>&1
  echo "$name self-golden exit $?"; grep -aE "SELF-STATS|self-golden gate|^Error|threshold|tightest|FAIL" $A/$name.self.log | cut -c1-300 | head -12
fi
