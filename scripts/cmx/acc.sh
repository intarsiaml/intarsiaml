#!/usr/bin/env bash
# Lane cmx: KL gate against the 27B W4A16 golden (HF Quark emulation, bf16; docs/results/2026-10-25-ac-27b-accuracy-gate.md section 2), ON spark1,
# inside a fleetq quiet slot:  fleetq run --host spark1 --quiet gpu,cpu --mem 50 --expect 25 --label "cmx: KL" -- ssh spark1 'bash -s' < scripts/cmx/acc.sh
# Arms (ARMS): base, dev, dev0 (dev with INTARSIA_CUDA_MXFP4_MMA=0). `--mode both`: prefill (chunk 256, so every chunk is M = 256 and takes the
# GEMM) and teacher-forced single-token decode of the last 16 positions of each prompt. Static floor 0.03 (no baseline: there is none for CUDA).
set -uo pipefail
export PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH
R=$HOME/intarsiaml-cmx; M=${MODEL:-$HOME/intarsiaml-s27d/models/Qwen3.8-27B-MXFP4}; G=${GOLDEN:-$HOME/intarsiaml-od/goldens/qwen3.8-27b-mxfp4-hfquark-w4a16-top20.safetensors}
O=${O:-$R/out/acc}; mkdir -p $O
for arm in ${ARMS:-base dev dev0}; do
  case $arm in base) tree=base; env="" ;; dev) tree=dev; env="" ;; dev0) tree=dev; env="INTARSIA_CUDA_MXFP4_MMA=0" ;; esac
  echo "== $arm $(date +%T)"
  env $env $R/trees/$tree/target/release/intarsia accuracy --backend cuda --model $M --mxfp4-act none --golden $G --mode both --decode-positions 16 \
    --max-seq 4096 --mem-budget 45 > $O/acc-$arm.out 2> $O/acc-$arm.err < /dev/null
  echo "rc=$?"; grep -aE "KL|top-1|PASS|FAIL|floor" $O/acc-$arm.out | head -12 | cut -c1-220
done
