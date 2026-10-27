#!/bin/bash
# KL gates for the GDN prep form (bitwise claim): inside a fleetq slot. usage: kl.sh BIN OUT
[ -n "${FLEETQ_JOB:-}" ] || { echo "not inside a fleetq slot" >&2; exit 2; }
BIN=$1; OUT=$2; mkdir -p $OUT
M=$HOME/models/Qwen3.6-35B-A3B-NVFP4; G=/opt/code/intarsiaml/golden
run() { # name golden maxseq decode
  local n=$1 g=$2 ms=$3 dp=$4
  echo "== $n off $(date +%T)"
  INTARSIA_METAL_GDN_PRE=0 $BIN accuracy --model $M --backend metal --golden $G/$g --mode both --decode-positions $dp --max-seq $ms --allow-unstamped \
     --write-self-golden $OUT/$n.off.selfgolden.safetensors --write-verdict $OUT/$n.off.verdict.json > $OUT/$n.off.log 2>&1; echo "exit $?"
  echo "== $n on $(date +%T)"
  $BIN accuracy --model $M --backend metal --golden $G/$g --mode both --decode-positions $dp --max-seq $ms --allow-unstamped \
     --write-self-golden $OUT/$n.on.selfgolden.safetensors --write-verdict $OUT/$n.on.verdict.json > $OUT/$n.on.log 2>&1; echo "exit $?"
  echo "== $n on vs off self-golden $(date +%T)"
  $BIN accuracy --model $M --backend metal --golden $G/$g --mode both --decode-positions $dp --max-seq $ms --allow-unstamped --allow-same-code \
     --self-golden $OUT/$n.off.selfgolden.safetensors > $OUT/$n.self.log 2>&1; echo "exit $?"
  grep -a "SELF-STATS" $OUT/$n.self.log
}
run short qwen36-35b-a3b-nvfp4-vllm-top20.safetensors 4096 16
run long qwen36-35b-a3b-nvfp4-vllm-long-top20.safetensors 16384 16
run 4k qwen36-35b-a3b-nvfp4-vllm-4k-top20.safetensors 8192 64
echo done
