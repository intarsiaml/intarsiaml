#!/usr/bin/env bash
# Lane mx: helpers for the stages scripts (ON tatooine, sourced; after T2's lib.sh). Trees ~/intarsiaml-mx-<name>, CLI at
# /work/mx-intarsia inside the HIP dev image (deploy/tatooine/dev.sh mounts the tree at /work, one models dir at /models).
# Every run: stdout -> $O/<tag>.out, stderr -> $O/<tag>.err. Environment for a run: E="K=V K2=V2" (passed with `env`).
# GPU: card 0 only unless G=<ids>.
O=${O:?out dir}
HUB=/data/models/radiance/hf-cache/hub; MI=/data/models/intarsia
M27=/models/models--amd--Qwen3.8-27B-Quark-AWQ-MXFP4/snapshots/5233554c5fa56afda40150556b95573c2d7d29c0
M35=/models/Qwen3.6-35B-A3B-NVFP4; M8=/models/Qwen3-8B
G27=/work/p1-golden/qwen3.8-27b-mxfp4-hfquark-top20.safetensors
N27=/work/golden/ref_noise/qwen3.8-27b-mxfp4-hfquark-top20.safetensors.json
SHORT='Write a long story about a robot who learns to paint.'
G=${G:-0}
T() { echo "$HOME/intarsiaml-mx-$1"; }
V() { echo "intarsia-cargo-target-mx-$1"; }
mdir() { case $1 in "$M27") echo "$HUB" ;; *) echo "$MI" ;; esac; }
log() { echo "$(date +%T) $*" | tee -a "$O/steps.log"; }
T0=${T0:-$(date +%s)}
BUDGET=${BUDGET:-2400}
left() { echo $((BUDGET - ($(date +%s) - T0))); }
need() { if (($(left) < $1)); then log "SKIPPED $2: $(left) s left, needs $1"; return 1; fi; return 0; }
# run <tree> <timeout> <tag> <models dir> <intarsia args...>
run() {
  local tree=$1 to=$2 tag=$3 md=$4; shift 4
  # shellcheck disable=SC2086
  INTARSIA_REPO=$(T "$tree") INTARSIA_TARGET_VOL=$(V "$tree") INTARSIA_MODELS=$md HIP_VISIBLE_DEVICES=$G timeout "$to" "$(T "$tree")/deploy/tatooine/dev.sh" \
    env ${E:-} /work/mx-intarsia "$@" > "$O/$tag.out" 2> "$O/$tag.err" < /dev/null
  local rc=$?
  log "$tag exit $rc${E:+ [$E]}: $(grep -ahE '^golden gate|self-golden gate|SELF-STATS|ref-noise|verdict|^Error|panicked|prompt:|finish=' "$O/$tag.out" "$O/$tag.err" | cut -c1-240 | tr '\n' '|')"
  return $rc
}
# pf <tree> <tag> <model> <ntok> [args]: prefill-only timing (generate --max-tokens 1) on the hp prompt of ntok tokens
pf() {
  local tree=$1 tag=$2 m=$3 n=$4; shift 4
  run "$tree" 300 "pf-$tag" "$(mdir "$m")" generate -m "$m" --backend hip --mem-budget 29 --max-seq 18432 --profile-every 0 \
    --ignore-eos -p "$(cat "$(T "$tree")/mx-out/prompt$n.txt")" --max-tokens 1 "$@"
}
# ids <tree> <tag> <model> <short|4k|16k> [args]: 256 greedy ids -> $O/ids-<tag>.ids (P1/T2 flags, so md5s compare: 27B tp1 main
# short 2086b6a4, 4k ef88d237 at T2's head)
ids() {
  local tree=$1 tag=$2 m=$3 kind=$4; shift 4
  local p="$SHORT"; [[ $kind == 4k ]] && p="$(cat "$(T "$tree")/scripts/hip-decode/prompt4k.txt")"
  [[ $kind == 16k ]] && p="$(cat "$(T "$tree")/mx-out/prompt16384.txt")"
  run "$tree" 300 "ids-$tag" "$(mdir "$m")" generate -m "$m" --backend hip --mem-budget 29 --max-seq 18432 \
    --profile-every 0 --ignore-eos --show-ids -p "$p" --max-tokens 256 "$@"
  grep -a '^ids:' "$O/ids-$tag.err" > "$O/ids-$tag.ids"
  log "ids $tag md5 $(md5sum < "$O/ids-$tag.ids" | cut -c1-8)"
}
same() { if cmp -s "$O/$1" "$O/$2" && [[ -s "$O/$1" ]]; then log "SAME $3 ($1 == $2)"; else log "DIFF $3 ($1 != $2)"; fi; }
# optab <tree> <tag> <model> <ntok>: engine op table of one prefill (--profile-every 1 --op-table)
optab() {
  local tree=$1 tag=$2 m=$3 n=$4; shift 4
  run "$tree" 300 "optab-$tag" "$(mdir "$m")" generate -m "$m" --backend hip --mem-budget 29 --max-seq 18432 --profile-every 1 \
    --op-table --ignore-eos -p "$(cat "$(T "$tree")/mx-out/prompt$n.txt")" --max-tokens 1 "$@"
}
# prof <tree> <tag> <model> <ntok>: rocprofv3 kernel trace of one prefill -> $O/prof-<tag>.csv + table
prof() {
  local tree=$1 tag=$2 m=$3 n=$4; shift 4
  local d=/work/mx-out/prof/$tag
  rm -rf "$(T "$tree")/mx-out/prof/$tag"; mkdir -p "$(T "$tree")/mx-out/prof/$tag"
  # shellcheck disable=SC2086
  INTARSIA_IMAGE=intarsia-hip-prof:rocm7.2.4-rust1.90.0 INTARSIA_TARGET_VOL=$(V "$tree") INTARSIA_REPO=$(T "$tree") \
    INTARSIA_MODELS=$(mdir "$m") HIP_VISIBLE_DEVICES=$G timeout -s KILL 300 "$(T "$tree")/deploy/tatooine/dev.sh" env ${E:-} \
    rocprofv3 --kernel-trace --output-format csv -d $d -o run -- /work/mx-intarsia generate -m "$m" --backend hip --mem-budget 29 \
    --max-seq 18432 --profile-every 0 --ignore-eos -p "$(cat "$(T "$tree")/mx-out/prompt$n.txt")" --max-tokens 1 "$@" > "$O/prof-$tag.log" 2>&1 < /dev/null
  local csv; csv=$(find "$(T "$tree")/mx-out/prof/$tag" -name '*kernel_trace.csv' | head -1)
  if [[ -n "$csv" ]]; then
    cp "$csv" "$O/prof-$tag.csv"
    python3 -I "$(T "$tree")/scripts/hip-decode/analyze_prefill.py" "$csv" > "$O/prof-$tag.txt" 2>&1
    rm -rf "$(T "$tree")/mx-out/prof/$tag"
  else echo "no trace" > "$O/prof-$tag.txt"; fi
  log "prof $tag${E:+ [$E]}: $(grep -aE 'finish=|prompt:' "$O/prof-$tag.log" | head -2 | tr '\n' '|') $(head -1 "$O/prof-$tag.txt")"
}
# hipbin <tree> <src in ~/mx> <tag> [args]: compile a standalone HIP program in the dev image and run it on card $G
hipbin() {
  local tree=$1 src=$2 tag=$3; shift 3
  cp "$HOME/mx/$src" "$(T "$tree")/mx-out/$src"
  INTARSIA_REPO=$(T "$tree") INTARSIA_TARGET_VOL=$(V "$tree") HIP_VISIBLE_DEVICES=$G timeout 200 "$(T "$tree")/deploy/tatooine/dev.sh" \
    bash -c "cd /work/mx-out && hipcc --offload-arch=gfx1201 -O3 -o ${src%.hip} $src && ./${src%.hip} $*" > "$O/bin-$tag.txt" 2>&1 < /dev/null
  log "bin $tag exit $?: $(grep -aE 'RESULT|rror' "$O/bin-$tag.txt" | head -3 | tr '\n' '|')"
}
# ceiling <tree>: hipBLASLt / WMMA-peak ceilings at the 27B shapes (scripts/mx/ceiling27.hip; built in the dev image, run in
# the ROCm image that ships hipBLASLt, as lane hp did)
ceiling() {
  local tree=$1 t; t=$(T "$tree")
  cp "$HOME/mx/ceiling27.hip" "$t/mx-out/"
  INTARSIA_REPO=$t INTARSIA_TARGET_VOL=$(V "$tree") timeout 200 "$t/deploy/tatooine/dev.sh" bash -c \
    "hipcc -O3 --offload-arch=gfx1201 -I kernels/hip mx-out/ceiling27.hip kernels/hip/ik_runtime.hip -ldl -o mx-out/ceiling27" > "$O/ceiling-build.txt" 2>&1 < /dev/null
  docker run --rm --device /dev/kfd --device /dev/dri --group-add "$(stat -c %g /dev/kfd)" --group-add "$(stat -c %g /dev/dri/renderD128)" \
    --group-add video --security-opt seccomp=unconfined -e HIP_VISIBLE_DEVICES=$G -v "$t":/work -w /work --entrypoint bash \
    fx6-rocm-llama:7.2.4 -c "timeout 220 ./mx-out/ceiling27 /opt/rocm/lib/libhipblaslt.so.1" < /dev/null > "$O/ceiling.txt" 2>&1
  log "ceiling exit $?: $(grep -aE 'peak|mlp down ' "$O/ceiling.txt" | head -4 | tr '\n' '|')"
}
# hiptest <tree> <tag> <cargo test args...>: the HIP crate's GPU tests from the ...-hiptest volume (built by build.sh HIPTEST=1)
hiptest() {
  local tree=$1 tag=$2; shift 2
  INTARSIA_REPO=$(T "$tree") INTARSIA_TARGET_VOL=$(V "$tree")-hiptest HIP_VISIBLE_DEVICES=$G timeout 600 "$(T "$tree")/deploy/tatooine/dev.sh" \
    env ${E:-} cargo test --release -p intarsia-backend-hip "$@" > "$O/test-$tag.txt" 2>&1 < /dev/null
  log "test $tag exit $?: $(grep -aE '^test result|FAILED|panicked' "$O/test-$tag.txt" | sort | uniq -c | tr '\n' '|' | cut -c1-300)"
}
# mtest <tree> <tag> <cargo test args...>: intarsia-models GPU tests from the ...-mtest volume (build.sh MTEST=1), HIP backend
mtest() {
  local tree=$1 tag=$2; shift 2
  INTARSIA_REPO=$(T "$tree") INTARSIA_TARGET_VOL=$(V "$tree")-mtest INTARSIA_MODELS=$MI HIP_VISIBLE_DEVICES=$G timeout 1200 "$(T "$tree")/deploy/tatooine/dev.sh" \
    env INTARSIA_BATCH_TEST_BACKEND=hip INTARSIA_REQUIRE_GPU=1 INTARSIA_QWEN36_DIR=$M35 ${E:-} cargo test --release -p intarsia-models "$@" > "$O/mtest-$tag.txt" 2>&1 < /dev/null
  log "mtest $tag exit $?: $(grep -aE '^test result|FAILED|panicked|SKIPPED' "$O/mtest-$tag.txt" | sort | uniq -c | tr '\n' '|' | cut -c1-300)"
}
# mxbench <tree> <tag> [M] [reps]: isolated GEMM timing mma vs f8 at the 27B shapes (scripts/mx/mxbench.hip)
mxbench() {
  local tree=$1 tag=$2; shift 2
  cp "$HOME/mx/mxbench.hip" "$(T "$tree")/mx-out/"
  INTARSIA_REPO=$(T "$tree") INTARSIA_TARGET_VOL=$(V "$tree") HIP_VISIBLE_DEVICES=$G timeout 300 "$(T "$tree")/deploy/tatooine/dev.sh" \
    bash -c "hipcc -O3 --offload-arch=gfx1201 -I kernels/hip -I kernels/abi -Rpass-analysis=kernel-resource-usage mx-out/mxbench.hip kernels/hip/mxfp4.hip kernels/hip/ik_runtime.hip -o mx-out/mxbench > mx-out/mxbench-build-$tag.txt 2>&1; env ${E:-} ./mx-out/mxbench $*" > "$O/mxbench-$tag.txt" 2>&1 < /dev/null
  cp "$(T "$tree")/mx-out/mxbench-build-$tag.txt" "$O/" 2>/dev/null
  log "mxbench $tag exit $?: $(grep -aE 'whole-model|rror' "$O/mxbench-$tag.txt" | head -2 | tr '\n' '|')"
}
