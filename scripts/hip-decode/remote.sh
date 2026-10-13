#!/usr/bin/env bash
# HIP decode perf work (perf/hip-decode): GPU stages. Runs ON tatooine, inside a fleetq lease with radiance down
# (see burst.sh, which stops radiance, runs this, restarts radiance and waits for it). One R9700 (HIP_VISIBLE_DEVICES=0).
#   scripts/hip-decode/remote.sh prof:TAG bench35:TAG bench8:TAG acc35 acc8 kernels test graphproof ...
# Needs the release build from `deploy/tatooine/dev.sh cargo build --release -p intarsia-cli --features hip` (CPU slot, no
# lease). Profiling runs in the derived image intarsia-hip-prof (deploy/tatooine/Dockerfile.hip-prof) against the same
# cargo target volume as the dev image.
set -uo pipefail
export INTARSIA_REPO="${INTARSIA_REPO:-$HOME/intarsiaml-hipperf}" HIP_VISIBLE_DEVICES=0
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
  local gen="$B generate -m $M35 --backend hip --mem-budget 29 --max-seq 4096 --chat --no-think --profile-every 0 --ignore-eos ${PROF_EXTRA:-} -p 'Write a long story about a robot who learns to paint.' --max-tokens 160"
  echo "--- unprofiled"
  $D bash -c "$gen" 2>&1 | grep -E "finish=|steps="
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
    --host 0.0.0.0 --port $port --served-model-name "$name" --max-seq 8192 >/dev/null
  local up=0 i
  for i in $(seq 1 200); do
    if curl -sf -m 2 http://127.0.0.1:$port/health >/dev/null 2>&1; then up=1; break; fi
    docker ps -q -f name=intarsia-serve | grep -q . || break  # the server exited (e.g. a binary without the hip feature): stop waiting
    sleep 3
  done
  if [ "$up" != 1 ]; then echo "serve did not come up"; docker logs intarsia-serve 2>&1 | tail -30; docker rm -f intarsia-serve >/dev/null; return 1; fi
  echo "up after ~$((i*3))s"
  docker exec intarsia-serve $B bench speed --url http://127.0.0.1:$port --model "$name" \
    --sizes 128,1024,4096 --gen 128 --reps 5 --gb-per-token "$gb" --bandwidth-gbs 627 \
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

gen35() { # gen35 <tag> [extra args]: 256-token greedy decode of the same prompt; writes ids + the summary lines to a file
  local tag=$1; shift
  $D $B generate -m $M35 --backend hip --mem-budget 29 --max-seq 4096 --chat --no-think --profile-every 0 --ignore-eos --show-ids \
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
  $D cargo test --release -p intarsia-backend-hip --test fused_decode --test admission -- --nocapture --test-threads=1 2>&1 \
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

stage_mutate() {
  # each mutation of a v2 kernel / fusion must make decode_v2 FAIL (kernel-level accuracy tests have teeth)
  local MUT="${INTARSIA_MUT_DIR:-$HOME/intarsiaml-hipperf2-mut}"
  local m
  for m in ${MUTATIONS:-conv-state nvfp4-iteration topk-tiebreak rope-tail moe-down fp8-scale fp8-convert pair-block fp8-pair-scale glu-scale glu-round gated-order gated-round}; do
    rm -rf "$MUT"; mkdir -p "$MUT"
    rsync -a --exclude target --exclude bench-out --exclude prof-out "$INTARSIA_REPO/" "$MUT/"
    python3 "$INTARSIA_REPO/scripts/hip-decode/mutate.py" "$MUT" $m || continue
    printf '%-18s ' "$m"
    INTARSIA_REPO="$MUT" deploy/tatooine/dev.sh cargo test --release -p intarsia-backend-hip --test "$(python3 "$INTARSIA_REPO/scripts/hip-decode/mutate.py" x --test-of $m)" 2>&1 \
      | grep -E "^test result|panicked at|error(\[|:)" | head -3 | cut -c1-200 | tr '\n' ' '
    echo
  done
  rm -rf "$MUT"
}

stage_kernels() {
  $D $B kernels test --backend hip --model $M35 --quick 2>&1 | tail -12
  $D $B kernels test --backend hip --model $M8 2>&1 | tail -6
}

acc() { # acc <model> <golden> [args]
  local m=$1 g=$2; shift 2
  echo "--- accuracy $g $*"
  $D $B accuracy --backend hip --model "$m" --golden "$G/$g" --mode both --mem-budget 29 --allow-unstamped --allow-dirty "$@" 2>&1 | tail -22
}
stage_acc35() { acc $M35 qwen36-35b-a3b-nvfp4-vllm-top20.safetensors --max-seq 4096; }
stage_acc8() {
  acc $M8 qwen3-8b-fp32.safetensors
  acc $M8 qwen3-8b-long-fp32.safetensors
}

stage_qualify() { # qualify <model-dir> <tag>
  local m=$1 tag=$2
  $D $B qualify --model "$m" --backend hip --quick-only --gen 128 --reps 5 --no-unim --out /work/bench-out/qualify-$tag --golden-dir $G --mem-budget 29 2>&1 | tail -40
}
stage_qual35() { stage_qualify $M35 "35b-${1:-base}"; }
stage_qual8() { stage_qualify $M8 "8b-${1:-base}"; }

for s in "$@"; do
  name=${s%%:*}; arg=""; [[ "$s" == *:* ]] && arg=${s#*:}
  echo "################ stage $s start $(ts)"
  "stage_$name" $arg
  echo "################ stage $s end $(ts)"
done
