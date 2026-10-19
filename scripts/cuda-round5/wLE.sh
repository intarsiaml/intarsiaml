#!/usr/bin/env bash
# Round-5 window 2, spark1 side (INSIDE scripts/cuda-decode/window.sh, driven by scripts/cuda-prefill4/wC-localeval.sh with
# ARMS="base dev"): localeval servers only, base = tree r5-base (origin/main), dev = tree r5-f8 default (bf16 KV cache).
set -u
D=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
echo "== localeval server base (r5-base) $(date +%T)"
TREE=r5-base OUTNAME=r5C bash $D/cuda-prefill4/wC-serve.sh "base:INTARSIA_X=0"
echo "== localeval server dev (r5-f8) $(date +%T)"
TREE=r5-f8 OUTNAME=r5C bash $D/cuda-prefill4/wC-serve.sh "dev:INTARSIA_X=0"
