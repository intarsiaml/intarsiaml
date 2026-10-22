#!/usr/bin/env bash
# Lane HS flip stages (ON tatooine, radiance down; run by window.sh with a hard timeout). Tree hs-flip = the branch with fusion
# on by default at HIP tp=2 (no INTARSIA_TP_FUSION in the environment unless a stage sets it).
#  1. tp=2 ids (256 greedy, chat, thinking off): default == explicit =1 == H1's fusion-on reference (35B short + 4k, 8B short);
#     =0 == H1's graphs-only reference (the opt-out still works). The default run's decode tok/s shows it ran fused.
#  2. tp=1: plan show and ids unchanged against H1's window-7 tp=1 outputs (fusion acts only at world > 1).
#  3. serve soak on the DEFAULT path (no env var; container hs-serve, both GPUs): tools/hs_soak.py, ~14 min of mixed load.
set -uo pipefail
export O=${1:?out dir}
source "$HOME/hs/lib.sh"
REF=$HOME/hs/ref
IMG=intarsia-hip-dev:rocm7.2.4-rust1.90.0
TREE=$(T hs-flip)
CID=hs-serve
cmpr() { # cmpr <file in $O> <reference file> <what>
  if cmp -s "$O/$1" "$2"; then log "SAME $3"; else log "DIFF $3"; fi
}
tps() { grep -ahoE '[0-9.]+ tok/s' "$O/gen-$1.err" | tail -1; }
vram() { for c in /sys/class/drm/card[0-9]*/device; do t=$(cat $c/mem_info_vram_total 2>/dev/null || echo 0); [ "$t" -gt 20000000000 ] && echo -n "$(basename $(dirname $c))=$(( $(cat $c/mem_info_vram_used) / 1048576 ))MiB "; done; }
START=$(date '+%F %T')
log "vram idle: $(vram)"
run hs-flip 0,1 60 version --version

# 1. tp=2 ids
for k in short 4k; do
  gen hs-flip 0,1 tp2-default-$k $M35 $k $TP2
  E="INTARSIA_TP_FUSION=1" gen hs-flip 0,1 tp2-on-$k $M35 $k $TP2
  E="INTARSIA_TP_FUSION=0" gen hs-flip 0,1 tp2-off-$k $M35 $k $TP2
  same gen-tp2-default-$k.ids gen-tp2-on-$k.ids "35B tp=2 $k: default vs explicit INTARSIA_TP_FUSION=1"
  cmpr gen-tp2-default-$k.ids "$REF/gen-head-35b-tp2gf-$k.ids" "35B tp=2 $k: default vs H1 fusion-on reference"
  cmpr gen-tp2-off-$k.ids "$REF/gen-head-35b-tp2g-$k.ids" "35B tp=2 $k: INTARSIA_TP_FUSION=0 vs H1 graphs-only reference"
  log "35B tp=2 $k decode: default $(tps tp2-default-$k), =1 $(tps tp2-on-$k), =0 $(tps tp2-off-$k)"
done
gen hs-flip 0,1 tp2-default-8b $M8 short $TP2
E="INTARSIA_TP_FUSION=0" gen hs-flip 0,1 tp2-off-8b $M8 short $TP2
cmpr gen-tp2-default-8b.ids "$REF/gen-head-8b-tp2gf-short.ids" "8B tp=2 short: default vs H1 fusion-on reference"
cmpr gen-tp2-off-8b.ids "$REF/gen-head-8b-tp2g-short.ids" "8B tp=2 short: INTARSIA_TP_FUSION=0 vs H1 graphs-only reference"
log "8B tp=2 short decode: default $(tps tp2-default-8b), =0 $(tps tp2-off-8b)"

# 2. tp=1 unchanged
plan hs-flip 0 tp1-35b $M35
cmpr plan-tp1-35b.out "$REF/plan-head-35b.out" "35B tp=1 plan show vs H1 window-7 head"
gen hs-flip 0 tp1-35b-short $M35 short
cmpr gen-tp1-35b-short.ids "$REF/gen-head-35b-short.ids" "35B tp=1 short ids vs H1 window-7 head"

# 3. serve soak on the default path
docker rm -f $CID > /dev/null 2>&1
docker run -d --name $CID --network host \
  --device /dev/kfd --device /dev/dri --group-add "$(stat -c %g /dev/kfd)" --group-add "$(stat -c %g /dev/dri/renderD128)" --group-add video \
  --ipc host --security-opt seccomp=unconfined --user "$(id -u):$(id -g)" -e HOME=/tmp -e GPU_MAX_HW_QUEUES=1 \
  -e INTARSIA_HOSTNAME="$(hostname)" \
  -v "$TREE":/work -v /data/models/intarsia:/models:ro -w /work "$IMG" \
  /work/hs-intarsia serve --backend hip --model $M35 --tp 2 --devices 0,1 --profile-every 0 --mem-budget 29 --no-supervisor \
  --host 0.0.0.0 --port 8090 --served-model-name qwen36-35b-hs --max-seq 8192 > "$O/serve.cid" 2> "$O/serve-start.err" < /dev/null
log "serve container started (default path, no INTARSIA_TP_FUSION): $(cut -c1-12 "$O/serve.cid")"
python3 "$HOME/hs/tools/hs_soak.py" lane --url http://127.0.0.1:8090 --out "$O/soak" --prompt4k "$HOME/hs/prompt4k.txt" \
  --ref-short "$REF/gen-head-35b-tp2gf-short.ids" --ref-4k "$REF/gen-head-35b-tp2gf-4k.ids" --serve-pat "hs-intarsia serve" \
  --budget-s "${BUDGET_S:-965}" --soak-s "${SOAK_S:-840}" --tail-s "${TAIL_S:-75}" > "$O/soak.log" 2>&1 < /dev/null
rc=$?
log "soak rc=$rc (0 pass, 1 criterion failed, 2 abort): $(grep -a 'lane:' "$O/soak.log" | tail -1 | cut -c1-300)"
log "vram under serve: $(vram)"
docker logs $CID > "$O/serve.log" 2>&1
docker inspect -f '{{.State.Status}} exit={{.State.ExitCode}} restarts={{.RestartCount}} oomkilled={{.State.OOMKilled}}' $CID > "$O/container_state.txt" 2>&1
docker stop -t 20 $CID > /dev/null 2>&1
docker logs $CID > "$O/serve-final.log" 2>&1
docker rm -f $CID > /dev/null 2>&1
journalctl -k --since "$START" --no-pager 2>&1 | grep -iE 'amdgpu|kfd|gpu|reset|timeout|fault|hang|ring' > "$O/kernel_gpu.log"
grep -aiE '\bnan\b|non-finite|panic|assert|IK_ERR|hip error|illegal|fatal' "$O/serve-final.log" > "$O/serve-suspect.txt"
log "serve log suspect lines: $(wc -l < "$O/serve-suspect.txt"); kernel gpu lines: $(wc -l < "$O/kernel_gpu.log"); container: $(cat "$O/container_state.txt")"
log "vram after stop: $(vram)"
log "done"
