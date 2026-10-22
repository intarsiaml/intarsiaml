#!/usr/bin/env bash
# Lane HS soak stages (ON tatooine, radiance down; run by window.sh with a hard timeout). PR #59 head, tp=2, fusion on:
#  1. pre: CLI generate short / 4k with INTARSIA_TP_FUSION=1, cmp against H1's fusion-on reference ids (window 7).
#  2. serve (container hs-serve, both GPUs, INTARSIA_TP_FUSION=1) and tools/hs_soak.py: start sentinels, solo reference,
#     ~15 min of mixed load at B = 1 / 4 / 8, drain, second reference, end sentinels, memory + poll sampling.
#  3. stop serve, keep its log + container state + kernel log, then post: the same two generates, cmp against the reference.
set -uo pipefail
export O=${1:?out dir}
source "$HOME/hs/lib.sh"
REF=$HOME/hs/ref
IMG=intarsia-hip-dev:rocm7.2.4-rust1.90.0
TREE=$(T head)
CID=hs-serve
cmpref() { # cmpref <gen tag> <short|4k>
  if cmp -s "$O/gen-$1.ids" "$REF/gen-head-35b-tp2gf-$2.ids"; then log "SAME as H1 fusion-on reference: $1 ($2)"; else log "DIFF from H1 fusion-on reference: $1 ($2)"; fi
}
vram() { for c in /sys/class/drm/card[0-9]*/device; do t=$(cat $c/mem_info_vram_total 2>/dev/null || echo 0); [ "$t" -gt 20000000000 ] && echo -n "$(basename $(dirname $c))=$(( $(cat $c/mem_info_vram_used) / 1048576 ))MiB "; done; }
START=$(date '+%F %T')
log "vram idle: $(vram)"
run head 0,1 60 version --version
E="INTARSIA_TP_FUSION=1" gen head 0,1 pre-short $M35 short $TP2; cmpref pre-short short
E="INTARSIA_TP_FUSION=1" gen head 0,1 pre-4k $M35 4k $TP2; cmpref pre-4k 4k
docker rm -f $CID > /dev/null 2>&1
docker run -d --name $CID --network host \
  --device /dev/kfd --device /dev/dri --group-add "$(stat -c %g /dev/kfd)" --group-add "$(stat -c %g /dev/dri/renderD128)" --group-add video \
  --ipc host --security-opt seccomp=unconfined --user "$(id -u):$(id -g)" -e HOME=/tmp -e GPU_MAX_HW_QUEUES=1 \
  -e INTARSIA_HOSTNAME="$(hostname)" -e INTARSIA_TP_FUSION=1 \
  -v "$TREE":/work -v /data/models/intarsia:/models:ro -w /work "$IMG" \
  /work/h1-intarsia serve --backend hip --model $M35 --tp 2 --devices 0,1 --profile-every 0 --mem-budget 29 --no-supervisor \
  --host 0.0.0.0 --port 8090 --served-model-name qwen36-35b-hs --max-seq 8192 > "$O/serve.cid" 2> "$O/serve-start.err" < /dev/null
log "serve container started: $(cut -c1-12 "$O/serve.cid")"
python3 "$HOME/hs/tools/hs_soak.py" lane --url http://127.0.0.1:8090 --out "$O/soak" --prompt4k "$HOME/hs/prompt4k.txt" \
  --ref-short "$REF/gen-head-35b-tp2gf-short.ids" --ref-4k "$REF/gen-head-35b-tp2gf-4k.ids" \
  --budget-s "${BUDGET_S:-910}" --soak-s "${SOAK_S:-760}" > "$O/soak.log" 2>&1 < /dev/null
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
E="INTARSIA_TP_FUSION=1" gen head 0,1 post-short $M35 short $TP2; cmpref post-short short
E="INTARSIA_TP_FUSION=1" gen head 0,1 post-4k $M35 4k $TP2; cmpref post-4k 4k
log "done"
