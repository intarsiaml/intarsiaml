#!/usr/bin/env bash
# runs on spark1 under the exclusive lease. usage: remote.sh <jobfile>
# jobfile lines: name|bin(new|old)|disable-list|case(short|long)|mode|kind|extra
#  kind: plain | record | self:<dump-name> | base:<baseline-name>
export PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH
R=$HOME/intarsia-acc; GD=$HOME/intarsiaml-golden; B=$R/bis; mkdir -p $B
M=/home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5
PROV="--pre-change 4963b43 --revlist $R/revlist --allow-same-code --allow-dirty --allow-config-mismatch"
LOOSE="--self-max-kl 1 --self-max-pos-kl 10 --self-max-flip-pct 100 --self-max-tail 1"
while IFS='|' read -r name bin dis cs mode kind extra; do
  [[ -z "$name" || "$name" == \#* ]] && continue
  if [[ $cs == short ]]; then G=$GD/qwen36-35b-a3b-nvfp4-vllm-top20.safetensors; F="--decode-positions 16 --max-seq 4096 --mem-budget 60"
  else G=$GD/qwen36-35b-a3b-nvfp4-vllm-long-top20.safetensors; F="--decode-positions 64 --max-seq 16384 --mem-budget 70"; fi
  BIN=$R/$bin/target/acc/release/intarsia
  avail=$(awk '/MemAvailable/ {print int($2/1024/1024)}' /proc/meminfo)
  if (( avail < 60 )); then echo "ABORT: only ${avail} GB available"; exit 3; fi
  case $kind in
    plain)  X="--golden $G --mode $mode $F";;
    record) X="--golden $G --mode $mode $F --write-baseline $B/$name.baseline.json --write-self-golden $B/$name.self.safetensors";;
    self:*) X="--self-golden $B/${kind#self:}.self.safetensors --mode $mode $F $PROV $LOOSE";;
    base:*) X="--golden $G --mode $mode $F --baseline $B/${kind#base:}.baseline.json $PROV";;
  esac
  echo "[$(date +%T)] $name: $bin dis=[$dis] $cs $mode $kind $extra"
  env INTARSIA_CUDA_DISABLE="$dis" $BIN accuracy --model $M --backend cuda $X $extra > $B/$name.log 2>&1 < /dev/null
  echo "[$(date +%T)] $name exit $?"
done < "$1"
