#!/usr/bin/env bash
# Lane T2: helpers for the stages scripts (ON tatooine, sourced; after H1's lib.sh). Trees: ~/intarsiaml-t2-<name>, CLI at
# /work/t2-intarsia inside the HIP dev image (deploy/tatooine/dev.sh mounts the tree at /work and one models dir at /models).
# Every run: stdout -> $O/<tag>.out, stderr -> $O/<tag>.err. Environment for a run: E="K=V K2=V2" (passed with `env` inside the
# container, so any variable reaches it).
O=${O:?out dir}
# Qwen3.8-27B MXFP4 lives in radiance's HF cache (read-only mount); the 35B / 8B in /data/models/intarsia
HUB=/data/models/radiance/hf-cache/hub; MI=/data/models/intarsia
M27=/models/models--amd--Qwen3.8-27B-Quark-AWQ-MXFP4/snapshots/5233554c5fa56afda40150556b95573c2d7d29c0
M35=/models/Qwen3.6-35B-A3B-NVFP4; M8=/models/Qwen3-8B
G27=/work/p1-golden/qwen3.8-27b-mxfp4-hfquark-top20.safetensors
N27=/work/golden/ref_noise/qwen3.8-27b-mxfp4-hfquark-top20.safetensors.json
SHORT='Write a long story about a robot who learns to paint.'
TP2="--tp 2 --devices 0,1"
T() { echo "$HOME/intarsiaml-t2-$1"; }
mdir() { case $1 in "$M27") echo "$HUB" ;; *) echo "$MI" ;; esac; }
log() { echo "$(date +%T) $*" | tee -a "$O/steps.log"; }
# run <tree> <gpus> <timeout> <tag> <models dir> <intarsia args...>
run() {
  local tree=$1 gpus=$2 to=$3 tag=$4 md=$5; shift 5
  mkdir -p "$(T "$tree")/t2-out"
  # shellcheck disable=SC2086
  INTARSIA_REPO=$(T "$tree") INTARSIA_MODELS=$md HIP_VISIBLE_DEVICES=$gpus timeout "$to" "$(T "$tree")/deploy/tatooine/dev.sh" \
    env ${E:-} /work/t2-intarsia "$@" > "$O/$tag.out" 2> "$O/$tag.err" < /dev/null
  local rc=$?
  log "$tag exit $rc${E:+ [$E]}: $(grep -ahE '^golden gate|self-golden gate|SELF-STATS|ref-noise|verdict|^Error|tensor-parallel rank 1: decode graph|graph replays=|finish=' "$O/$tag.out" "$O/$tag.err" | cut -c1-220 | tr '\n' '|')"
  return $rc
}
# ids <tree> <gpus> <tag> <model> <short|4k> [args]: 256 greedy ids -> $O/ids-<tag>.ids. P1's exact flags (raw prompt, no chat
# template), so the tp=1 27B md5s compare with P1's (2086b6a4 short, ef88d237 4k).
ids() {
  local tree=$1 gpus=$2 tag=$3 m=$4 kind=$5; shift 5
  local p="$SHORT"; [[ $kind == 4k ]] && p="$(cat "$(T "$tree")/scripts/hip-decode/prompt4k.txt")"
  run "$tree" "$gpus" 300 "ids-$tag" "$(mdir "$m")" generate -m "$m" --backend hip --mem-budget 29 --max-seq 8192 \
    --profile-every 0 --ignore-eos --show-ids -p "$p" --max-tokens 256 "$@"
  grep -a '^ids:' "$O/ids-$tag.err" > "$O/ids-$tag.ids"
  log "ids $tag md5 $(md5sum < "$O/ids-$tag.ids" | cut -c1-8)"
}
same() { # same <a> <b> <what>: cmp two files in $O
  if cmp -s "$O/$1" "$O/$2" && [[ -s "$O/$1" ]]; then log "SAME $3 ($1 == $2)"; else log "DIFF $3 ($1 != $2)"; fi
}
# firstdiff <a.ids> <b.ids> <what>: index of the first differing greedy id
firstdiff() {
  python3 - "$O/$1" "$O/$2" "$3" <<'PY' | tee -a "$O/steps.log"
import sys, re
def ids(p):
    t = open(p).read()
    return [int(x) for x in re.findall(r"-?\d+", t.split("ids:", 1)[1])] if "ids:" in t else []
a, b = ids(sys.argv[1]), ids(sys.argv[2])
d = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None)
print(f"FIRSTDIFF {sys.argv[3]}: {len(a)} / {len(b)} ids, first divergence at {d if d is not None else 'none (identical)'}")
PY
}
# acc <tree> <gpus> <tag> [args]: 27B accuracy against P1's W4A4 golden with the ref-noise sidecar, dump -> $O/acc-<tag>.safetensors
acc() {
  local tree=$1 gpus=$2 tag=$3; shift 3
  run "$tree" "$gpus" 600 "acc-$tag" "$HUB" accuracy --backend hip --model $M27 --mem-budget 29 --golden $G27 --mode both \
    --ref-noise $N27 --write-self-golden /work/t2-out/acc-$tag.safetensors --allow-unstamped "$@"
  mv "$(T "$tree")/t2-out/acc-$tag.safetensors" "$O/" 2>/dev/null
}
# self <tree> <gpus> <tag> <dump in $O> [args]: change vs base (--self-golden against the dump)
self() {
  local tree=$1 gpus=$2 tag=$3 d=$4; shift 4
  cp "$O/$d" "$(T "$tree")/t2-out/$d"
  run "$tree" "$gpus" 600 "self-$tag" "$HUB" accuracy --backend hip --model $M27 --mem-budget 29 --mode both \
    --self-golden /work/t2-out/$d --pre-change ${PRE:-d3f89b7} --allow-same-code --allow-config-mismatch --allow-unstamped "$@"
}
# prof <tag> <gpus> [args]: rocprofv3 kernel trace of a 128-token 27B decode after a ~1k-token prompt (P1's 1k: the first 3700
# bytes of prompt4k.txt), per-rank step table by prof2.py -> $O/prof-<tag>.txt (the trace stays in the tree's t2-out/prof/)
prof() {
  local tag=$1 gpus=$2; shift 2
  local d=/work/t2-out/prof/$tag p; p="$(head -c 3700 "$(T head)/scripts/hip-decode/prompt4k.txt")"
  rm -rf "$(T head)/t2-out/prof/$tag"; mkdir -p "$(T head)/t2-out/prof/$tag"
  # shellcheck disable=SC2086
  INTARSIA_IMAGE=intarsia-hip-prof:rocm7.2.4-rust1.90.0 INTARSIA_TARGET_VOL=intarsia-cargo-target-t2-head INTARSIA_REPO=$(T head) \
    INTARSIA_MODELS=$HUB HIP_VISIBLE_DEVICES=$gpus timeout -s KILL 400 "$(T head)/deploy/tatooine/dev.sh" env ${E:-} \
    rocprofv3 --kernel-trace --output-format csv -d $d -o run -- /work/t2-intarsia generate -m $M27 --backend hip --mem-budget 29 \
    --max-seq 8192 --profile-every 0 --ignore-eos -p "$p" --max-tokens 128 "$@" > "$O/prof-$tag.log" 2>&1 < /dev/null
  local csv; csv=$(find "$(T head)/t2-out/prof/$tag" -name '*kernel_trace.csv' | head -1)
  if [[ -n "$csv" ]]; then python3 "$HOME/t2/prof2.py" "$csv" > "$O/prof-$tag.txt" 2>&1; else echo "no trace" > "$O/prof-$tag.txt"; fi
  log "prof $tag${E:+ [$E]}: $(grep -aE 'finish=' "$O/prof-$tag.log" | head -1 | grep -oE 'decode [0-9]+ ms \([0-9.]+ tok/s\)') | $(head -1 "$O/prof-$tag.txt")"
}
