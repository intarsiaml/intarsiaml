#!/usr/bin/env bash
# HIP decode perf work (perf/hip-decode): GPU stages. Runs ON tatooine, inside a fleetq lease with radiance down
# (see burst.sh, which stops radiance, runs this, restarts radiance and waits for it). One R9700 (HIP_VISIBLE_DEVICES=0).
#   scripts/hip-decode/remote.sh prof:TAG bench35:TAG bench8:TAG acc35 acc8 kernels test graphproof ...
# Needs the release build from `deploy/tatooine/dev.sh cargo build --release -p intarsia-cli --features hip` (CPU slot, no
# lease). Profiling runs in the derived image intarsia-hip-prof (deploy/tatooine/Dockerfile.hip-prof) against the same
# cargo target volume as the dev image.
set -uo pipefail
export INTARSIA_REPO="${INTARSIA_REPO:-$HOME/intarsiaml-hipperf}" HIP_VISIBLE_DEVICES="${RS_GPU:-0}"  # RS_GPU=1: the second R9700 (tests in parallel)
cd "$INTARSIA_REPO"
D=deploy/tatooine/dev.sh
B=/target/release/intarsia
M35=/models/Qwen3.6-35B-A3B-NVFP4
M8=/models/Qwen3-8B
G=/models/golden
IMG=intarsia-hip-dev:rocm7.2.4-rust1.90.0
OUTD=$INTARSIA_REPO/bench-out
ts() { date +%T; }
trap 'docker rm -f intarsia-serve >/dev/null 2>&1' EXIT
mkdir -p "$OUTD"

TARGET_VOL() { $D --target-vol; }

stage_profng() { PROF_EXTRA="--no-graphs" stage_prof "${1:-base}-eager"; }
stage_prof() { # prof <tag>: rocprofv3 kernel trace of a 160-token 35B decode + the unprofiled rate
  local tag=${1:-base}
  local gen="${PROF_PRE:-} $B generate -m $M35 --backend hip --mem-budget 29 --max-seq 4096 --chat --no-think --profile-every 0 --ignore-eos ${PROF_EXTRA:-} -p 'Write a long story about a robot who learns to paint.' --max-tokens 160"
  echo "--- unprofiled"
  $D bash -c "$gen" 2>&1 | grep -E "finish=|steps=|WARNING"
  echo "--- rocprofv3 (kernel trace)"
  rm -rf "$OUTD/prof/$tag"; mkdir -p "$OUTD/prof/$tag"
  local tv; tv=$(TARGET_VOL)   # (computed first: inside the env-prefix list it would see INTARSIA_IMAGE)
  INTARSIA_IMAGE=intarsia-hip-prof:rocm7.2.4-rust1.90.0 INTARSIA_TARGET_VOL=$tv \
    $D bash -c "rocprofv3 --kernel-trace --output-format csv -d /work/bench-out/prof/$tag -o run -- $gen" 2>&1 | grep -E "finish=|steps=|rror"
  find "$OUTD/prof/$tag" -name '*kernel_trace.csv' | head -2
  local csv; csv=$(find "$OUTD/prof/$tag" -name '*kernel_trace.csv' | head -1)
  [ -n "$csv" ] && python3 "$INTARSIA_REPO/scripts/hip-decode/analyze.py" "$csv" > "$OUTD/prof/$tag/table.txt" && cat "$OUTD/prof/$tag/table.txt"
}

serve_bench() { # serve_bench <name> <model dir> <gb-per-token> <tag> [extra serve args]
  local name=$1 model=$2 gb=$3 tag=$4; shift 4
  local port=8090
  docker rm -f intarsia-serve >/dev/null 2>&1
  echo "--- serve $name [$(ts)]"
  docker run -d --name intarsia-serve --network host \
    --device /dev/kfd --device /dev/dri --group-add "$(stat -c %g /dev/kfd)" --group-add "$(stat -c %g /dev/dri/renderD128)" --group-add video \
    --ipc host --security-opt seccomp=unconfined --user "$(id -u):$(id -g)" -e HOME=/tmp -e GPU_MAX_HW_QUEUES=1 -e HIP_VISIBLE_DEVICES=0 \
    ${SERVE_ENV:-} \
    -v "$INTARSIA_REPO":/work -v /data/models/intarsia:/models:ro -v "$(TARGET_VOL)":/target -w /work \
    $IMG $B serve --backend hip --model "$model" --profile-every "${PE:-0}" "$@" \
    --host 0.0.0.0 --port $port --served-model-name "$name" --max-seq ${MAXSEQ:-8192} >/dev/null
  local up=0 i
  for i in $(seq 1 200); do
    if curl -sf -m 2 http://127.0.0.1:$port/health >/dev/null 2>&1; then up=1; break; fi
    docker ps -q -f name=intarsia-serve | grep -q . || break  # the server exited (e.g. a binary without the hip feature): stop waiting
    sleep 3
  done
  if [ "$up" != 1 ]; then echo "serve did not come up"; docker logs intarsia-serve 2>&1 | tail -30; docker rm -f intarsia-serve >/dev/null; return 1; fi
  echo "up after ~$((i*3))s"
  docker exec intarsia-serve $B bench speed --url http://127.0.0.1:$port --model "$name" \
    --sizes ${SIZES:-128,1024,4096} --gen ${GEN:-128} --reps ${REPS:-5} --gb-per-token "$gb" --bandwidth-gbs 627 \
    --label "$tag" --out-dir /work/bench-out --thinking off 2>&1 | tail -22
  curl -s http://127.0.0.1:$port/metrics | grep -iE "graph|eager" | head
  docker rm -f intarsia-serve >/dev/null
}

# bench35[:tag] / bench8[:tag]: graphs on (the default), per-op profiling off (--profile-every 0, so every decode step is a graph replay)
# bench35ng / bench8ng: --no-graphs (eager). bench35pe16 / bench8pe16: the server's default profiling (every 16th step eager).
stage_bench35() { serve_bench qwen36-35b-intarsia-hip $M35 2.2 "hipdecode-35b-${1:-base}" --mem-budget 29; }
stage_bench8() { serve_bench qwen3-8b-intarsia-hip $M8 16.38 "hipdecode-8b-${1:-base}"; }
stage_bench35ng() { serve_bench qwen36-35b-intarsia-hip $M35 2.2 "hipdecode-35b-${1:-base}-eager" --mem-budget 29 --no-graphs; }
stage_bench8ng() { serve_bench qwen3-8b-intarsia-hip $M8 16.38 "hipdecode-8b-${1:-base}-eager" --no-graphs; }
stage_bench35pe16() { PE=16 serve_bench qwen36-35b-intarsia-hip $M35 2.2 "hipdecode-35b-${1:-base}-pe16" --mem-budget 29; }
stage_bench8pe16() { PE=16 serve_bench qwen3-8b-intarsia-hip $M8 16.38 "hipdecode-8b-${1:-base}-pe16"; }

# Prefill (perf/hip-prefill). pfprof:<tag>[:chunk] = rocprofv3 kernel trace of ONE prefill of a ~4k-token prompt (35B, max-tokens 1),
# per-kernel table by analyze_prefill.py. The prompt is scripts/hip-decode/prompt4k.txt.
stage_pfprof() {
  local tag=${1%%:*} chunk="" ex=""   # arg is <tag>[:<chunk>] (the stage runner passes everything after the first colon)
  [[ "$1" == *:* ]] && chunk=${1#*:}
  [ -n "$chunk" ] && ex="--prefill-chunk $chunk"
  printf '%s\n' "${PROF_PRE:-} $B generate -m $M35 --backend hip --mem-budget 29 --max-seq 8192 --profile-every 0 --ignore-eos $ex -p \"\$(cat /work/scripts/hip-decode/prompt4k.txt)\" --max-tokens 1" > "$OUTD/pfgen.sh"
  local gen="bash /work/bench-out/pfgen.sh"
  echo "--- unprofiled (one prefill, includes the cold start)"
  $D bash -c "$gen" 2>&1 | grep -aE "prompt:|finish="
  rm -rf "$OUTD/prof/$tag"; mkdir -p "$OUTD/prof/$tag"
  local tv; tv=$(TARGET_VOL)
  INTARSIA_IMAGE=intarsia-hip-prof:rocm7.2.4-rust1.90.0 INTARSIA_TARGET_VOL=$tv \
    $D bash -c "rocprofv3 --kernel-trace --output-format csv -d /work/bench-out/prof/$tag -o run -- $gen" 2>&1 | grep -aE "prompt:|finish=|rror"
  local csv; csv=$(find "$OUTD/prof/$tag" -name '*kernel_trace.csv' | head -1)
  [ -n "$csv" ] && python3 "$INTARSIA_REPO/scripts/hip-decode/analyze_prefill.py" "$csv" > "$OUTD/prof/$tag/table.txt" && cat "$OUTD/prof/$tag/table.txt"
}
# chunk35:<chunk> / chunk8:<chunk>: serve with --prefill-chunk, prefill speed at 128/1024/4096 (gen 16, 3 reps)
stage_chunk35() { GEN=16 REPS=3 serve_bench qwen36-35b-intarsia-hip $M35 2.2 "pfchunk-35b-${1}" --mem-budget 29 --prefill-chunk "${1}"; }
stage_chunk8() { GEN=16 REPS=3 serve_bench qwen3-8b-intarsia-hip $M8 16.38 "pfchunk-8b-${1}" --prefill-chunk "${1}"; }
# f8test: the e4m3 WMMA GEMM against the cpu reference, M-invariance, bf16-path equivalence (tests/fp8_f8.rs)
stage_f8test() {
  $D cargo test --release -p intarsia-backend-hip --test fp8_f8 -- --nocapture --test-threads=1 2>&1 \
    | grep -E "^test |test result|FAILED|panicked|rel_l2|differ|f8 vs" | cut -c1-300 | tail -30
}
# alltests: every test of the HIP crate (admission, determinism, fused decode, fp8_f8, ...), one thread
stage_alltests() {
  $D cargo test --release -p intarsia-backend-hip --no-fail-fast -- --test-threads=1 2>&1 | grep -E "test result|FAILED|panicked|Running|failed" | cut -c1-200 | tail -60
}
# accpert: 35B short golden, prefill + decode, for the default build (f8 off), the e4m3 GEMM on, and two other correct kernels that move
# the rounding (SIMT attention instead of flash / split-KV, GEMV-style MoE instead of the WMMA projections): the spread to read the
# e4m3 GEMM's delta against. ACC_TAIL=4: the OVERALL lines and verdicts only; full logs in bench-out/acc/.
stage_accpert() {
  local m=qwen36-35b-a3b-nvfp4-vllm-top20-meta.safetensors
  ACC_TAG=pert-base ACC_TAIL=4 acc $M35 $m --max-seq 4096
  INTARSIA_HIP_FP8_F8=1 ACC_TAG=pert-f8 ACC_TAIL=4 acc $M35 $m --max-seq 4096
  INTARSIA_HIP_DISABLE=attn_prefill_bf16_flash,attn_prefill_bf16_splitkv ACC_TAG=pert-simt ACC_TAIL=4 acc $M35 $m --max-seq 4096
  INTARSIA_HIP_MOE_WMMA=0 ACC_TAG=pert-moegemv ACC_TAIL=4 acc $M35 $m --max-seq 4096
  INTARSIA_HIP_FP8_F8=1 INTARSIA_HIP_DISABLE=attn_prefill_bf16_flash,attn_prefill_bf16_splitkv ACC_TAG=pert-f8simt ACC_TAIL=4 acc $M35 $m --max-seq 4096
}
# f8 variants of the long golden: acc35lf8:<chunk>
stage_acc35lf8() { INTARSIA_HIP_FP8_F8=1 ACC_TAG="f8c$1" ACC_MODE=all ACC_TAIL=40 acc $M35 qwen36-35b-a3b-nvfp4-vllm-long-top20.safetensors --decode-positions 64 --max-seq 16384 --prefill-chunk "$1"; }
# kf8: the admission sweep on the 35B's shapes, the lines of the fp8 kernels and the verdict
stage_kf8() { $D $B kernels test --backend hip --model $M35 2>&1 | grep -E "fp8|kernels,|FAIL|REJECT"; }
# pfbench: prefill-shaped timings (fp8 mma vs mma_f8, routed experts) for INTARSIA_HIP_MOE_Z = 0 (full grid), 2 (default), 4, 1
stage_pfbench() {
  local z
  for z in 0 2 4 1; do
    echo "=== INTARSIA_HIP_MOE_Z=$z"
    INTARSIA_HIP_MOE_Z=$z $D cargo test --release -p intarsia-backend-hip --test quantmoe -- --ignored --nocapture bench_prefill_shapes 2>&1 \
      | grep -E "^env|^fp8|^moe|panicked|FAILED" | cut -c1-220
  done
}
# chunk35l:<chunk>: does that chunk fit at 16k context (max-seq 17408, one 16384-token prompt)
stage_chunk35l() { MAXSEQ=17408 SIZES=16384 GEN=16 REPS=2 serve_bench qwen36-35b-intarsia-hip $M35 2.2 "pfchunk-35b-16k-${1}" --mem-budget 29 --prefill-chunk "${1}"; }
# pf35[:tag] / pf8[:tag]: A/B speed, prefill 128/1024/4096/16384 and decode 128 (the serve's own recipe chunk)
stage_pf35() { export SERVE_ENV=""; [[ "${1:-}" == f8* ]] && export SERVE_ENV="-e INTARSIA_HIP_FP8_F8=1"; MAXSEQ=17408 SIZES=128,1024,4096,16384 serve_bench qwen36-35b-intarsia-hip $M35 2.2 "pf-35b-${1:-base}" --mem-budget 29; }
stage_pf8() { export SERVE_ENV=""; [[ "${1:-}" == f8* ]] && export SERVE_ENV="-e INTARSIA_HIP_FP8_F8=1"; MAXSEQ=17408 SIZES=128,1024,4096,16384 serve_bench qwen3-8b-intarsia-hip $M8 16.38 "pf-8b-${1:-base}"; }

gen35() { # gen35 <tag> [extra args]: 256-token greedy decode of the same prompt; writes ids + the summary lines to a file
  local tag=$1; shift
  $D env ${GEN_ENV:-X=0} $B generate -m $M35 --backend hip --mem-budget 29 --max-seq 4096 --chat --no-think --profile-every 0 --ignore-eos --show-ids \
    -p 'Write a long story about a robot who learns to paint.' --max-tokens 256 "$@" > "$OUTD/gen35-$tag.out" 2> "$OUTD/gen35-$tag.err"   # (stdout is the text, stderr the ids and the summary)
  grep -a '^ids:' "$OUTD/gen35-$tag.err" > "$OUTD/gen35-$tag.ids"
  grep -aE "finish=|steps=" "$OUTD/gen35-$tag.err"
}

# graph replay vs eager on the real 35B: 256 greedy tokens each, the ids must be identical (and the counters say what ran)
stage_graphproof() {
  echo "--- eager (--no-graphs)"; gen35 eager --no-graphs
  echo "--- graph"; gen35 graph
  echo "ids in each file: $(tr -cd ',' < "$OUTD/gen35-eager.ids" | wc -c) / $(tr -cd ',' < "$OUTD/gen35-graph.ids" | wc -c) commas"
  if [ -s "$OUTD/gen35-eager.ids" ] && cmp -s "$OUTD/gen35-eager.ids" "$OUTD/gen35-graph.ids"; then
    echo "GRAPH == EAGER: the 256 generated token ids are identical"
  else
    echo "GRAPH != EAGER: token ids DIFFER"; cmp "$OUTD/gen35-eager.ids" "$OUTD/gen35-graph.ids" | head -3
  fi
}

# INTARSIA_NO_FUSION=1 (the container only forwards INTARSIA_HIP_* / INTARSIA_TP_*, so the variable is set with `env` inside it):
# the unfused chain must still run, with the launch count of the base tree's own NO_FUSION run, and give the ids of the fused run
stage_nofusion() {
  echo "--- NO_FUSION generate (256 tokens)"; GEN_ENV="INTARSIA_NO_FUSION=1" gen35 nofusion
  if [ -s "$OUTD/gen35-graph.ids" ] && cmp -s "$OUTD/gen35-nofusion.ids" "$OUTD/gen35-graph.ids"; then
    echo "NO_FUSION == FUSED: the 256 generated token ids are identical"
  else
    echo "NO_FUSION != FUSED (or no fused run in this tree): ids DIFFER or are missing"; cmp "$OUTD/gen35-nofusion.ids" "$OUTD/gen35-graph.ids" | head -3
  fi
  echo "--- NO_FUSION profile, this tree"; PROF_PRE="env INTARSIA_NO_FUSION=1" stage_prof nofusion
  echo "--- NO_FUSION profile, base tree"
  ( export INTARSIA_REPO="${BASE_REPO:-$HOME/intarsiaml-fp8gemv-base}" INTARSIA_TARGET_VOL="${BASE_VOL:-intarsia-cargo-target-fp8gemv-base}"
    cd "$INTARSIA_REPO" && OUTD=$INTARSIA_REPO/bench-out && mkdir -p "$OUTD" && PROF_PRE="env INTARSIA_NO_FUSION=1" stage_prof nofusion )
}

stage_envsweep() { # decode rate of a 256-token generation under CLR env knobs, graph and eager
  local kv g
  for kv in "X=0" "HIP_FORCE_DEV_KERNARG=0" "HIP_FORCE_DEV_KERNARG=1" "ROC_USE_FGS_KERNARG=0" "DEBUG_CLR_MAX_BATCH_SIZE=1" "GPU_STREAMOPS_CP_WAIT=1"; do
    for g in "" "--no-graphs"; do
      printf '%-32s %-12s ' "$kv" "${g:-graph}"
      env "$kv" $D bash -c "export $kv; $B generate -m $M35 --backend hip --mem-budget 29 --max-seq 4096 --chat --no-think --profile-every 0 --ignore-eos $g -p 'Write a long story about a robot who learns to paint.' --max-tokens 200" 2>&1 | grep -aoE "decode [0-9]+ ms \([0-9.]+ tok/s\)"
    done
  done
}

stage_test() { # cargo tests of the hip crate (kernel tests, device_pos, graph proof)
  $D cargo test --release -p intarsia-backend-hip -- --nocapture 2>&1 | grep -E "^test |test result|FAILED|panicked" | tail -60
}

stage_fusedtest() { # the fused decode kernels against the unfused launches (bitwise) and the cpu reference, plus the GPU admission sweep
  $D cargo test --release -p intarsia-backend-hip --test fused_decode --test fp8_pair_shapes --test decode_v2 --test admission -- --nocapture --test-threads=1 2>&1 \
    | grep -E "^test |test result|FAILED|panicked|differs|rel_l2" | cut -c1-400 | tail -40
}

stage_gtest() { # the graph tests alone, single-threaded, with the full output
  $D cargo test --release -p intarsia-backend-hip --test device_pos -- --nocapture --test-threads=1 2>&1 | grep -vE "^warning|^\s*(\||=|-->)" | tail -40
}

stage_v2test() { # v2 kernels vs v1 (bitwise) and the cpu reference; the fp8 staging variants 1 (default), 2, 3 each in their own process
  for q in 1 2 3 x; do
    echo "--- INTARSIA_HIP_FP8_Q=$q (x: default Q with INTARSIA_HIP_XSTAGE=1)"
    INTARSIA_HIP_XSTAGE=$([ $q = x ] && echo 1 || echo 0) INTARSIA_HIP_FP8_Q=$([ $q = x ] && echo 2 || echo $q) $D cargo test --release -p intarsia-backend-hip --test decode_v2 --test fused_decode -- --nocapture --test-threads=1 2>&1 \
      | grep -vE "^warning|^\s*(\||=|-->)|^$|Compiling|Finished|Running" | cut -c1-300 | tail -40
  done
}

stage_sweep() { # DRAM-cold v1 vs v2 at the real decode shapes, across the fp8 row / staging knobs
  local rows q
  for rows in 2; do
    INTARSIA_HIP_FP8_ROWS=$rows $D cargo test --release -p intarsia-backend-hip --test quantmoe -- --ignored --nocapture bench_decode_variants 2>&1 \
      | grep -E "^env|^nvfp4|^fp8|^moe" | cut -c1-120
  done
  for pre in 1 2; do
    echo "=== MOE_PRE=$pre"
    INTARSIA_HIP_MOE_PRE=$pre $D cargo test --release -p intarsia-backend-hip --test quantmoe -- --ignored --nocapture bench_decode_variants 2>&1 \
      | grep -E "^moe" | cut -c1-120
  done
  echo "=== XSTAGE=1"
  INTARSIA_HIP_XSTAGE=1 $D cargo test --release -p intarsia-backend-hip --test quantmoe -- --ignored --nocapture bench_decode_variants 2>&1 \
    | grep -E "^env|^nvfp4|^moe" | cut -c1-120
  for q in 3; do
    INTARSIA_HIP_FP8_Q=$q $D cargo test --release -p intarsia-backend-hip --test quantmoe -- --ignored --nocapture bench_decode_variants 2>&1 \
      | grep -E "^env|^fp8" | cut -c1-120
  done
}

stage_fp8sweep() { # fp8 GEMV at the 35B decode shapes: v2 singles, a size scale (fixed cost / streaming rate), the pair block shapes; DRAM-cold
  $D cargo test --release -p intarsia-backend-hip --test quantmoe -- --ignored --nocapture bench_fp8_sweep 2>&1 | grep -E "^(case|o_proj|attn|k or|qkv|z |scale|pair)|panicked|error" | cut -c1-110
}

stage_mutate() {
  # each mutation of a v2 kernel / fusion must make decode_v2 FAIL (kernel-level accuracy tests have teeth)
  local MUT="${INTARSIA_MUT_DIR:-$HOME/intarsiaml-hipperf2-mut}"
  local m
  for m in ${MUTATIONS:-conv-state nvfp4-iteration topk-tiebreak rope-tail moe-down fp8-scale fp8-convert pair-block fp8-pair-scale glu-scale glu-round gated-order gated-round pair-nb0 pair-grid norm-skip-sum norm-skip-normed norm-sum-round norm-woff norm-order norm-pair-block norm-sync gn-round gn-woff gn-order gn-group gn-weight gn-silu f8-scale f8-kperm f8-actq f8-ktail f8-w16-always f8-w16-never f8-noexit moe-z-step}; do
    rm -rf "$MUT"; mkdir -p "$MUT"
    rsync -a --exclude target --exclude bench-out --exclude prof-out "$INTARSIA_REPO/" "$MUT/"
    python3 "$INTARSIA_REPO/scripts/hip-decode/mutate.py" "$MUT" $m || continue
    printf '%-18s ' "$m"
    INTARSIA_REPO="$MUT" deploy/tatooine/dev.sh cargo test --release -p intarsia-backend-hip --test "$(python3 "$INTARSIA_REPO/scripts/hip-decode/mutate.py" x --test-of $m)" 2>&1 \
      | grep -E "^test result|panicked at|error(\[|:)" | head -3 | cut -c1-200 | tr '\n' ' '
    echo
  done
  rm -rf "$MUT"
  # the mutated builds shared the cargo target volume: make the real sources newer than its outputs, so the next build of
  # this tree recompiles the kernels instead of trusting mtimes (run the build before any later stage)
  find "$INTARSIA_REPO/kernels" "$INTARSIA_REPO/crates" -type f -exec touch {} +
  echo "NOTE: rebuild this tree (scripts/hip-decode/build.sh) before running binaries from its target volume"
}

stage_kernels() {
  $D $B kernels test --backend hip --model $M35 --quick 2>&1 | tail -12
  $D $B kernels test --backend hip --model $M8 2>&1 | tail -6
}

acc() { # acc <model> <golden> [args]
  local m=$1 g=$2; shift 2
  echo "--- accuracy $g $*"
  mkdir -p "$OUTD/acc"
  $D $B accuracy --backend hip --model "$m" --golden "$G/$g" --mode ${ACC_MODE:-both} --mem-budget 29 --allow-unstamped --allow-dirty "$@" 2>&1 | tee "$OUTD/acc/${ACC_TAG:-acc}-${g%%.*}.log" | tail -${ACC_TAIL:-22}
}
# Prefill chunk (perf/hip-prefill): acc35c:<chunk> short golden, acc35l:<chunk> long golden with the prefix-reuse gate (--mode all),
# acc8c:<chunk> the two 8B fp32 goldens. Full logs in bench-out/acc/.
stage_acc35c() { ACC_TAG="c$1" acc $M35 qwen36-35b-a3b-nvfp4-vllm-top20-meta.safetensors --max-seq 4096 --prefill-chunk "$1"; }
stage_acc35l() { ACC_TAG="c$1" ACC_MODE=all ACC_TAIL=40 acc $M35 qwen36-35b-a3b-nvfp4-vllm-long-top20.safetensors --decode-positions 64 --max-seq 16384 --prefill-chunk "$1"; }
stage_acc8c() { ACC_TAG="c$1" acc $M8 qwen3-8b-fp32.safetensors --prefill-chunk "$1"; ACC_TAG="c$1" acc $M8 qwen3-8b-long-fp32.safetensors --prefill-chunk "$1"; }
stage_acc35() { acc $M35 qwen36-35b-a3b-nvfp4-vllm-top20-meta.safetensors --max-seq 4096; }  # sha b6b51e1b1699c66c: the golden the CUDA numbers use
stage_acc8() {
  acc $M8 qwen3-8b-fp32.safetensors
  acc $M8 qwen3-8b-long-fp32.safetensors
}

# Round 3 (perf/hip-round3, docs/results/2026-10-19-hip-round3.md).
# prof4k:<tag>: rocprofv3 kernel trace of a 160-token 35B decode after the ~4.4k-token prompt (decode at 4k context), per-step table
stage_prof4k() {
  local tag=${1:-base}
  printf '%s\n' "$B generate -m $M35 --backend hip --mem-budget 29 --max-seq 8192 --profile-every 0 --ignore-eos -p \"\$(cat /work/scripts/hip-decode/prompt4k.txt)\" --max-tokens 160" > "$OUTD/dec4k.sh"
  echo "--- unprofiled"
  $D bash -c "bash /work/bench-out/dec4k.sh" 2>&1 | grep -aE "prompt:|finish=|steps=|WARNING"
  rm -rf "$OUTD/prof/$tag"; mkdir -p "$OUTD/prof/$tag"
  local tv; tv=$(TARGET_VOL)
  INTARSIA_IMAGE=intarsia-hip-prof:rocm7.2.4-rust1.90.0 INTARSIA_TARGET_VOL=$tv \
    $D bash -c "rocprofv3 --kernel-trace --output-format csv -d /work/bench-out/prof/$tag -o run -- bash /work/bench-out/dec4k.sh" 2>&1 | grep -aE "finish=|rror"
  local csv; csv=$(find "$OUTD/prof/$tag" -name '*kernel_trace.csv' | head -1)
  [ -n "$csv" ] && python3 "$INTARSIA_REPO/scripts/hip-decode/analyze.py" "$csv" > "$OUTD/prof/$tag/table.txt" && cat "$OUTD/prof/$tag/table.txt"
}
# ids4k: greedy ids of 256 tokens after the ~4.4k-token prompt (two prefill chunks), this tree and the base tree; must be identical
gen35l() { # gen35l <outfile prefix>
  printf '%s\n' "$B generate -m $M35 --backend hip --mem-budget 29 --max-seq 8192 --profile-every 0 --ignore-eos --show-ids -p \"\$(cat /work/scripts/hip-decode/prompt4k.txt)\" --max-tokens 256" > "$OUTD/ids4k.sh"
  $D bash -c "bash /work/bench-out/ids4k.sh" > "$1.out" 2> "$1.err"
  grep -a '^ids:' "$1.err" > "$1.ids"
  grep -aE "prompt:|finish=" "$1.err"
}
stage_ids4k() {
  rm -f "$OUTD"/ids4k-*   # never compare against a stale copy
  # IDS_NEW_ENV="NAME=value": set for this tree's run only (e.g. INTARSIA_HIP_DISABLE of a kernel the base does not have)
  echo "--- this tree ${IDS_NEW_ENV:-}"; ( [ -n "${IDS_NEW_ENV:-}" ] && export "${IDS_NEW_ENV?}"; gen35l "$OUTD/ids4k-new" )
  echo "--- base tree"
  local outd_new=${OUTD_NEW:-$OUTD}
  ( export INTARSIA_REPO="${BASE_REPO:-$HOME/intarsiaml-fp8gemv-base}" INTARSIA_TARGET_VOL="${BASE_VOL:-intarsia-cargo-target-fp8gemv-base}"
    cd "$INTARSIA_REPO" && OUTD=$INTARSIA_REPO/bench-out && mkdir -p "$OUTD" && gen35l "$OUTD/ids4k-base" && cp "$OUTD/ids4k-base.ids" "$outd_new/ids4k-base.ids" )
  echo "ids: $(tr -cd ',' < "$OUTD/ids4k-new.ids" | wc -c) / $(tr -cd ',' < "$OUTD/ids4k-base.ids" | wc -c) commas"
  if [ -s "$OUTD/ids4k-new.ids" ] && cmp -s "$OUTD/ids4k-new.ids" "$OUTD/ids4k-base.ids"; then
    echo "IDS4K IDENTICAL: the 256 greedy ids after the 4k prompt match the base tree"
  else
    echo "IDS4K DIFFER"; cmp "$OUTD/ids4k-new.ids" "$OUTD/ids4k-base.ids" | head -3
  fi
}
# pfb: bench_prefill_shapes once (default env): fp8 mma / mma_v2 / mma_f8, W8A8 and weight-only; MoE experts at 32 and 64 pairs
stage_pfb() {
  $D cargo test --release -p intarsia-backend-hip --test quantmoe -- --ignored --nocapture bench_prefill_shapes 2>&1 \
    | grep -E "^env|^fp8|^moe|panicked|FAILED" | cut -c1-260
}
# r3test: the new kernels' bitwise tests, the MoE accuracy tests (both tile sizes), the f8 tests (they compare against matmul_fp8_mma)
stage_r3test() {
  $D cargo test --release -p intarsia-backend-hip --no-fail-fast --test fp8_mma_v2 --test moe_pairs --test quantmoe --test fp8_f8 -- --nocapture --test-threads=1 2>&1 \
    | grep -E "^test |test result|FAILED|panicked|differ|v2 |worst" | cut -c1-300 | tail -40
}
stage_admission() {
  $D cargo test --release -p intarsia-backend-hip --test admission -- --nocapture --test-threads=1 2>&1 \
    | grep -E "^test |test result|FAILED|panicked|REJECT|FAIL|fp8_mma|moe_experts|repeat checks" | cut -c1-220 | tail -30
}
# Mutations of round 3, built in a CPU slot (mutbuild, no GPU) and run in a GPU window (mutrun): the kernel library is static, so
# each mutant's test binary, copied to /target/r3mut/<name>, is self-contained and the window only runs them.
R3MUT="v2-wsel v2-perm v2-whi v2-xstride v2-quant-half v2-noquant v2-nodecode pairs-sred pairs-xrow pairs-out moe-z-step f8-actq"
stage_mutbuild() {
  local MUT="${INTARSIA_MUT_DIR:-$HOME/intarsiaml-r3-mut}" m t exe
  for m in ${MUTATIONS:-$R3MUT}; do
    rm -rf "$MUT"; mkdir -p "$MUT"
    rsync -a --exclude target --exclude bench-out --exclude prof-out "$INTARSIA_REPO/" "$MUT/"
    python3 "$INTARSIA_REPO/scripts/hip-decode/mutate.py" "$MUT" "$m" || continue
    t=$(python3 "$INTARSIA_REPO/scripts/hip-decode/mutate.py" x --test-of "$m")
    exe=$(INTARSIA_REPO="$MUT" $D cargo test --release -p intarsia-backend-hip --test "$t" --no-run --message-format=json 2>/dev/null \
      | python3 -c 'import json,sys
for l in sys.stdin:
    try: d = json.loads(l)
    except Exception: continue
    if d.get("reason") == "compiler-artifact" and d.get("executable") and d["target"]["kind"] == ["test"]: print(d["executable"])' | tail -1)
    if [ -z "$exe" ]; then echo "$m: BUILD FAILED"; continue; fi
    INTARSIA_REPO="$MUT" $D bash -c "mkdir -p /target/r3mut && cp '$exe' /target/r3mut/$m"
    echo "$m: test $t built ($exe)"
  done
  rm -rf "$MUT"
  find "$INTARSIA_REPO/kernels" "$INTARSIA_REPO/crates" -type f -exec touch {} +
  echo "NOTE: rebuild this tree (scripts/hip-decode/build.sh) before running binaries from its target volume"
}
stage_mutrun() {
  local m
  for m in ${MUTATIONS:-$R3MUT}; do
    printf '%-16s ' "$m"
    # (a mutant that hangs a kernel is killed by the timeout: exit 124, no "test result" line)
    $D bash -c "timeout 120 /target/r3mut/$m --test-threads=1; echo exit=\$?" 2>&1 | grep -aE "^test result|panicked at|^exit=" | head -3 | cut -c1-200 | tr '\n' ' '
    echo
  done
}

# Round 4 (perf/hip-round4, docs/results/2026-10-19-hip-round4.md).
# r4test: the new kernels' bitwise tests (gdn_v2 once per prefetch depth 1 / 2 / 4), and the tests of the kernels and paths
# they sit next to (device position + graph replay, fused decode, attention)
stage_r4test() {
  local c
  for c in 1 2 4; do
    INTARSIA_HIP_GDN_PD=$c $D cargo test --release -p intarsia-backend-hip --test gdn_v2 -- --nocapture --test-threads=1 2>&1 \
      | grep -E "^test |test result|FAILED|panicked|differ|bitwise" | cut -c1-300
  done
  $D cargo test --release -p intarsia-backend-hip --no-fail-fast --test qk_rope_kv --test device_pos --test fused_decode --test attention -- --nocapture --test-threads=1 2>&1 \
    | grep -E "^test |test result|FAILED|panicked|differ|bitwise" | cut -c1-300 | tail -40
}
# k35: the full admission sweep on the 35B's shapes (the quick sweep has no dk = 128 GDN case and no flash-sized prefill,
# so it leaves gdn_delta_rule_bf16_v2 and attn_prefill_bf16_flash untested and fails, on the base tree too); k35q: the quick one
stage_k35() { $D $B kernels test --backend hip --model $M35 2>&1 | grep -aE "gdn|qk_norm|rope|kv_append|kernels,|FAIL|REJECT|SKIPPED|rror"; }
stage_k35q() { $D $B kernels test --backend hip --model $M35 --quick 2>&1 | grep -aE "kernels,|FAIL|REJECT|SKIPPED|rror"; }
# ids4kx: ids4k with this tree's delta rule v2 disabled (the fused prologue alone)
stage_ids4kx() { IDS_NEW_ENV="INTARSIA_HIP_DISABLE=gdn_delta_rule_bf16_v2" stage_ids4k; }
# acc35f8: the 35B short golden with the opt-in W8A8 GEMM (INTARSIA_HIP_FP8_F8=1); the long one is acc35lf8:<chunk>
stage_acc35f8() { INTARSIA_HIP_FP8_F8=1 ACC_TAG=f8 stage_acc35; }
# ids4koff: ids4k with both round-4 kernels disabled (the rollback lever: the model falls back to the unfused ops)
stage_ids4koff() { IDS_NEW_ENV="INTARSIA_HIP_DISABLE=gdn_delta_rule_bf16_v2,qk_norm_rope_kv_bf16" stage_ids4k; }
# gdnbench: the delta-rule kernels at the 35B's decode and prefill shapes (us per launch), per prefetch depth
stage_gdnbench() {
  local c
  for c in 1 2 4; do
    echo "=== INTARSIA_HIP_GDN_PD=$c"
    INTARSIA_HIP_GDN_PD=$c $D cargo test --release -p intarsia-backend-hip --test gdn_v2 -- --ignored --nocapture gdn_v2_timing 2>&1 \
      | grep -E "^gdn|panicked|FAILED" | cut -c1-260
  done
}

# plan35: the resolved plan of the 35B (kernel per op, the workspace size) at the recipe chunk
stage_plan35() { $D $B plan show --backend hip --model $M35 --mem-budget 29 --max-seq 17408 2>&1 | grep -aE "scratch|fp8|moe_experts|rror" | cut -c1-200; }

stage_qualify() { # qualify <model-dir> <tag>
  local m=$1 tag=$2
  $D $B qualify --model "$m" --backend hip --quick-only --gen 128 --reps 5 --no-unim --out /work/bench-out/qualify-$tag --golden-dir $G --mem-budget 29 2>&1 | tail -40
}
stage_qual35() { stage_qualify $M35 "35b-${1:-base}"; }
stage_qual8() { stage_qualify $M8 "8b-${1:-base}"; }

# Round 5 (perf/hip-round5, docs/results/2026-10-19-hip-round5.md).
# pf8o:<tag>: 8B prefill on prompts around and just over the 4096-token recipe chunk (targets 4000 / 4160 / 4400; bench speed
# records each rep's prompt_tokens), gen 16. A tag containing "off" disables the fused attention prologue for that server.
stage_pf8o() {
  export SERVE_ENV=""; [[ "${1:-}" == *off* ]] && export SERVE_ENV="-e INTARSIA_HIP_DISABLE=qk_norm_rope_kv_bf16"
  MAXSEQ=17408 SIZES=4000,4160,4400 GEN=16 REPS=5 serve_bench qwen3-8b-intarsia-hip $M8 16.38 "pf8o-${1:-base}"
}
# profenv:<tag>:<NAME=value>: the decode profile (prof) with one INTARSIA_HIP_* knob set (dev.sh forwards it)
stage_profenv() { local tag=${1%%:*} kv=${1#*:}; ( export "$kv"; echo "--- $kv"; stage_prof "$tag" ); }
# r5test: round 5's new-kernel tests (moe_v3) and the tests of the paths next to them, then the f8 tests (now the default GEMM)
stage_r5test() {
  timeout 400 $D cargo test --release -p intarsia-backend-hip --no-fail-fast --test moe_v3 --test decode_v2 --test device_pos --test fused_decode \
    --test fp8_f8 --test fp8_mma_v2 --test moe_pairs -- --nocapture --test-threads=1 2>&1 \
    | grep -aE -A2 "^test |test result|FAILED|panicked|differ|bitwise" | cut -c1-300 | tail -70
  # flash v2 at every query-heads-per-block value (read once per process), then the attention sweeps (cpu reference, poison)
  for hpb in 1 2 4; do
    echo "--- flash_v2 INTARSIA_HIP_FLASH_HPB=$hpb"
    INTARSIA_HIP_FLASH_HPB=$hpb timeout 300 $D cargo test --release -p intarsia-backend-hip --test flash_v2 -- --nocapture 2>&1 \
      | grep -aE "^test |test result|panicked|differs|bitwise v1 on|HPB=" | cut -c1-300
  done
  timeout 400 $D cargo test --release -p intarsia-backend-hip --test attention -- --test-threads=1 2>&1 | grep -aE "^test |test result|panicked" | cut -c1-300
}
# pfprofenv:<tag>:<NAME=value>: the 4408-token prefill profile (pfprof) with one INTARSIA_HIP_* knob set
stage_pfprofenv() { local tag=${1%%:*} kv=${1#*:}; ( export "$kv"; echo "--- $kv"; stage_pfprof "$tag" ); }
# pf8x:<tag>: 8B A/B in one server: prefill 1024 / 4160 (just over the 4096 recipe chunk) / 16384 and decode, 3 reps, gen 64
stage_pf8x() { export SERVE_ENV=""; MAXSEQ=17408 SIZES=1024,4160,16384 GEN=64 REPS=3 serve_bench qwen3-8b-intarsia-hip $M8 16.38 "pf8x-${1:-base}"; }
# acc35f8off: the 35B short golden under the f8 kill switch (INTARSIA_HIP_FP8_F8=0, must reproduce the v2 numbers)
stage_acc35f8off() { INTARSIA_HIP_FP8_F8=0 ACC_TAG=f8off stage_acc35; }
# round-5 GPU mutants (built by mutbuild in a CPU slot, run by mutrun in a GPU window): MUTATIONS="$R5MUT"
# (v3-ticket hangs the kernel by design of the mutation: last, so a wedged queue cannot disturb the others)
R5MUT="v3-noreset v3-early v3-nowait v3-hrow v3-gurow v3-nofence fl2-head fl2-kvh fl2-row fl2-stride v3-ticket"
stage_mutbuild5() { MUTATIONS="$R5MUT" stage_mutbuild; }
stage_mutrun5() { MUTATIONS="$R5MUT" stage_mutrun; }
# plan8 / plan35: the resolved plan (which kernel serves each op) of the 8B / 35B, every line, to see whether the fp8 GEMM appears
stage_plan8() { $D $B plan show --backend hip --model $M8 --max-seq 17408 2>&1 | grep -aE "matmul|scratch|rror" | cut -c1-200 | head -40; }
stage_plan35f() { $D $B plan show --backend hip --model $M35 --mem-budget 29 --max-seq 17408 2>&1 | grep -aE "matmul_fp8|moe_experts|scratch|rror" | cut -c1-200 | head -40; }

# Round 6 (perf/hip-round6, docs/results/2026-10-19-hip-round6.md).
# pf35env:<tag>[:NAME=value] / pf8env:<tag>[:NAME=value]: the pf35 / pf8x A/B runs with one INTARSIA_HIP_* knob set in the server
stage_pf35env() {
  local tag=${1%%:*} kv=""; [[ "$1" == *:* ]] && kv=${1#*:}
  export SERVE_ENV=""; [ -n "$kv" ] && export SERVE_ENV="-e $kv"
  echo "--- serve env: ${kv:-none}"
  MAXSEQ=17408 SIZES=128,1024,4096,16384 serve_bench qwen36-35b-intarsia-hip $M35 2.2 "pf-35b-$tag" --mem-budget 29
}
stage_pf8env() {
  local tag=${1%%:*} kv=""; [[ "$1" == *:* ]] && kv=${1#*:}
  export SERVE_ENV=""; [ -n "$kv" ] && export SERVE_ENV="-e $kv"
  echo "--- serve env: ${kv:-none}"
  MAXSEQ=17408 SIZES=1024,4160,16384 GEN=64 REPS=3 serve_bench qwen3-8b-intarsia-hip $M8 16.38 "pf8x-$tag"
}
# round-6 mutants (mutate.py; built by mutbuild6 in a CPU slot, run by mutrun6 in a GPU window)
R6MUT="mw-jn mw-full mw-two mw-lastsync mw-def64 mw-def32 mw-tile mw-range hpb-def hpb-two fl2-head fl2-kvh fl2-row fl2-stride"
stage_mutbuild6() { MUTATIONS="$R6MUT" stage_mutbuild; }
stage_mutrun6() { MUTATIONS="$R6MUT" stage_mutrun; }
# bg:<name>:<stage>,<stage>,...: run those stages on the second R9700 (RS_GPU=1) in the background, log bench-out/<name>.log;
# bgwait: wait for it and print its log (both inside the same window, so the GPU-1 work counts against the same downtime)
stage_bg() {
  local name=${1%%:*} st=${1#*:}
  ( RS_GPU=1 bash "$INTARSIA_REPO/scripts/hip-decode/remote.sh" ${st//,/ } > "$OUTD/$name.log" 2>&1 ) &
  echo "$! $name" > "$OUTD/bg.pid"; echo "bg $name on GPU 1: pid $! stages ${st//,/ }"
}
stage_bgwait() {
  local pid name; read -r pid name < "$OUTD/bg.pid"
  while kill -0 "$pid" 2>/dev/null; do sleep 2; done
  echo "--- bg $name done $(ts)"; cat "$OUTD/$name.log"
}
# r6bench: MoE projection variants (INTARSIA_HIP_MOE_MW 0..7, bench_prefill_shapes; the sweep switch must be INTARSIA_HIP_* to reach dev.sh)
stage_r6bench() {
  INTARSIA_HIP_MW_SWEEP=1 timeout 300 $D cargo test --release -p intarsia-backend-hip --test quantmoe -- --ignored --nocapture bench_prefill_shapes 2>&1 \
    | grep -aE "^env|^moe|panicked|FAILED" | cut -c1-260
}
# r6test: round 6's bitwise tests (MoE variants at both tiles, delta rule v2 / pipelined, flash v2 at HPB 2 / 4) and the
# accuracy tests of the kernels they replace
stage_r6test() {
  timeout 300 $D cargo test --release -p intarsia-backend-hip --no-fail-fast --test moe_pairs --test quantmoe --test round6_selection -- --nocapture --test-threads=1 2>&1 \
    | grep -aE "^test |test result|FAILED|panicked|differ|changed" | cut -c1-300 | tail -40
  local hpb
  for hpb in 1 2 4; do
    echo "--- flash_v2 INTARSIA_HIP_FLASH_HPB=$hpb"
    INTARSIA_HIP_FLASH_HPB=$hpb timeout 300 $D cargo test --release -p intarsia-backend-hip --test flash_v2 -- --nocapture 2>&1 \
      | grep -aE "^test |test result|panicked|differs|bitwise v1 on|HPB=" | cut -c1-300
  done
}

for s in "$@"; do
  name=${s%%:*}; arg=""; [[ "$s" == *:* ]] && arg=${s#*:}
  echo "################ stage $s start $(ts)"
  if [[ "$name" == *@base ]]; then
    # A/B: the same stage on the origin/main tree and its own cargo target volume (BASE_REPO / BASE_VOL), for interleaved runs
    ( export INTARSIA_REPO="${BASE_REPO:-$HOME/intarsiaml-fp8gemv-base}" INTARSIA_TARGET_VOL="${BASE_VOL:-intarsia-cargo-target-fp8gemv-base}"
      cd "$INTARSIA_REPO" && OUTD=$INTARSIA_REPO/bench-out && mkdir -p "$OUTD" && "stage_${name%@base}" $arg )
  else
    "stage_$name" $arg
  fi
  echo "################ stage $s end $(ts)"
done
