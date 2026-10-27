#!/usr/bin/env bash
# Lane cgv: ON spark1 inside a fleetq slot: the 27B verify-rows probe (tests/verify_rows_real.rs) in three configurations, to separate the GEMM from the
# multi-row verify attention that the probe uses unless INTARSIA_CUDA_VERIFY_ATTN_PER_ROW is set (the engine sets it for --spec mtp):
#   a. dev, INTARSIA_CUDA_VERIFY_ATTN_PER_ROW=1 (what `--spec mtp` runs)     b. dev, INTARSIA_CUDA_MXFP4_GEMV_MR=0 (the pre-change matmul path), probe attention
#   c. dev, GEMV_MR=0 and ATTN_PER_ROW=1
export PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH
R=$HOME/intarsiaml-cgv; M=${MODEL:-$HOME/intarsiaml-cgv/models/Qwen3.8-27B-MXFP4}; O=${O:-$R/out/gates}; mkdir -p $O
cd $R/trees/${TREE:-dev2}
for v in "a:INTARSIA_CUDA_VERIFY_ATTN_PER_ROW=1" "b:INTARSIA_CUDA_MXFP4_GEMV_MR=0" "c:INTARSIA_CUDA_MXFP4_GEMV_MR=0 INTARSIA_CUDA_VERIFY_ATTN_PER_ROW=1"; do
  n=${v%%:*}; e=${v#*:}
  echo "== variant $n ($e) $(date +%T)"
  env $e INTARSIA_VERIFY_DIR=$M INTARSIA_VERIFY_ROWS=4 INTARSIA_VERIFY_LENS=${LENS:-17,130,1024,4000} INTARSIA_REQUIRE_GPU=1 cargo test --release -p intarsia-models --test verify_rows_real -- --nocapture > $O/verify-rows-$n.log 2>&1
  echo "rc=$?"; grep -aE "^DIFF|^len=|PROBE_RESULT|test result" $O/verify-rows-$n.log | cut -c1-200
done
echo "== done $(date +%T)"
