#!/bin/zsh
# metal phase C prefill profile; run inside a fleetq slot (FLEETQ_JOB set by the caller).
R=/opt/code/intarsiaml/docs/results/raw/2026-10-14-metal-phase-c
BIN=/opt/code/intarsiaml/target/release/intarsia
M=/Users/josef/models/Qwen3-8B
mkdir -p $R/plain $R/optable $R/trace $R/chunk
gen() { # label prompt-file extra-args...
  local L=$1 P=$2; shift 2
  $BIN generate --model $M --backend metal --max-seq 8192 --temperature 0 --max-tokens 1 --prompt "$(cat $R/prompt-$P.txt)" "$@"
}
# 0. page-in / pipeline warm-up (discarded)
gen warm 4k --profile-every 0 > $R/plain/warm.out 2> $R/plain/warm.err
# 1. plain prefill speed, chunk 256 (default), 2 runs each
for P in 1k 4k; do for i in 1 2; do gen plain $P --profile-every 0 > $R/plain/$P-c256-r$i.out 2> $R/plain/$P-c256-r$i.err; done; done
# 2. chunk sweep at 4k (plain)
for C in 128 512 1024 2048; do gen chunk 4k --profile-every 0 --prefill-chunk $C > $R/chunk/4k-c$C.out 2> $R/chunk/4k-c$C.err; done
gen chunk 1k --profile-every 0 --prefill-chunk 1024 > $R/chunk/1k-c1024.out 2> $R/chunk/1k-c1024.err
# 3. engine op table (host-synchronised per op)
for P in 1k 4k; do gen optable $P --profile-every 1 --op-table > $R/optable/$P-c256.out 2> $R/optable/$P-c256.err; done
# 4. GPU timestamp traces: kernel (one cb per launch) and cb (stream as normal)
for P in 1k 4k; do
  for MODE in kernel cb; do
    INTARSIA_METAL_PROFILE=$MODE:$R/trace/$MODE-$P-c256.jsonl gen trace $P --profile-every 0 > $R/trace/$MODE-$P-c256.out 2> $R/trace/$MODE-$P-c256.err
  done
done
for MODE in kernel cb; do
  INTARSIA_METAL_PROFILE=$MODE:$R/trace/$MODE-4k-c1024.jsonl gen trace 4k --profile-every 0 --prefill-chunk 1024 > $R/trace/$MODE-4k-c1024.out 2> $R/trace/$MODE-4k-c1024.err
done
# 5. MLX reference GEMM / SDPA
/Users/josef/.local/share/uv/tools/mlx-lm/bin/python $R/mlx_gemm_bench.py > $R/mlx_gemm_bench.jsonl 2> $R/mlx_gemm_bench.err
gzip -f $R/trace/*.jsonl
echo done
