#!/usr/bin/env bash
# Lane f185 (ON tatooine, radiance down, run by window.sh): HIP checks of the review of PR #185. Binaries: dev = this branch, base = fac1d6541 (main before #185).
#  (a) Qwen3.8-27B tp=2, the shipped MTP default (no --spec), short / 4k / 16k prompt, dev vs base: the 256 greedy ids, the `spec:` stats line and the per-pass
#      trace (INTARSIA_SPEC_TRACE) must be identical.
#  (b) Qwen3.6-35B-A3B NVFP4 tp=1, explicit --spec mtp --spec-k 2, one ~16k prompt, dev (card 0) in parallel with dev --spec off (card 1): dev loads, the
#      "macro prefill passes=" line appears, ids == --spec off.
#  (c) optional, only while time is left: dev --spec off on the 27B (reference ids for (a)), base --spec mtp on the 35B (pre-#185 stats for (b)).
set -uo pipefail
export O=${1:?out dir}
R=$HOME/intarsiaml-f185; D=$R/work/deploy/tatooine/dev.sh
export INTARSIA_MODELS=/data/models   # dev.sh mounts it at /models (the default mounts only /data/models/intarsia)
M27=/models/radiance/hf-cache/hub/models--amd--Qwen3.8-27B-Quark-AWQ-MXFP4/snapshots/5233554c5fa56afda40150556b95573c2d7d29c0
M35=/models/intarsia/Qwen3.6-35B-A3B-NVFP4
P=$R/work/scripts/mpf/ids-prompts
OPT_AFTER=${OPT_AFTER:-1000}   # no optional run starts after this many seconds
log() { echo "$(date +%T) $*" | tee -a "$O/steps.log"; }
vram() { local m=0 v c; for c in /sys/class/drm/card[0-9]/device/mem_info_vram_used; do v=$(cat "$c" 2>/dev/null || echo 0); (( v > m )) && m=$v; done; echo $m; }
clear() { local i; for i in 1 2 3 4 5 6; do (( $(vram) < 1073741824 )) && return 0; sleep 10; done; log "CONTENDED: $(( $(vram) / 1048576 )) MiB in use on a card: NO VERDICT"; return 1; }
# g <tag> <gpus> <bin> <model> <maxseq> <prompt name> [args]; E="K=V ..." for the environment of the run
g() {
  local tag=$1 gpus=$2 bin=$3 m=$4 ms=$5 n=$6; shift 6
  INTARSIA_REPO=$R/work INTARSIA_TARGET_VOL=intarsia-cargo-f185 HIP_VISIBLE_DEVICES=$gpus timeout 400 "$D" env ${E:-} /work/.bin/$bin generate -m "$m" --backend hip \
    --mem-budget 29 --max-seq "$ms" --chat --no-think --profile-every 0 --ignore-eos --show-ids -p "$(cat "$P/$n.txt")" --max-tokens 256 "$@" \
    > "$O/$tag.out" 2> "$O/$tag.err" < /dev/null
  local rc=$?
  grep -a '^ids:' "$O/$tag.err" > "$O/$tag.ids"
  grep -a '^spec: token' "$O/$tag.err" > "$O/$tag.trace"
  log "$tag exit $rc: $(grep -aE '^spec: verify|macro prefill|default of recipe|^Error|error:' "$O/$tag.err" | tail -4 | cut -c1-170 | tr '\n' '|')"
  return $rc
}
same() { if [ -s "$O/$1" ] && cmp -s "$O/$1" "$O/$2"; then log "SAME $3"; else log "DIFF $3 ($1 vs $2)"; fi; }
statl() { grep -a '^spec: verify' "$O/$1"; }
samespec() { if [ -n "$(statl $1)" ] && [ "$(statl $1)" = "$(statl $2)" ]; then log "SAME spec stats $3: $(statl $1)"; else log "DIFF spec stats $3: [$(statl $1)] vs [$(statl $2)]"; fi; }
sametrace() { # first divergence of the per-pass traces
  if [ -s "$O/$1" ] && cmp -s "$O/$1" "$O/$2"; then log "SAME per-pass trace $3 ($(wc -l < "$O/$1") passes)"
  else log "DIFF per-pass trace $3: first difference at line $(cmp "$O/$1" "$O/$2" 2>&1 | grep -oE 'line [0-9]+' | head -1) of $(wc -l < "$O/$1") / $(wc -l < "$O/$2")"; fi
}
for f in "${M27/#\/models//data/models}/config.json" "${M35/#\/models//data/models}/config.json" "$R/work/.bin/base" "$R/work/.bin/dev" "$P/16k-0.txt"; do
  [ -e "$f" ] || { log "PREFLIGHT: $f missing: NO VERDICT"; exit 3; }
done
clear || exit 2
log "stages start, VRAM in use $(( $(vram) / 1048576 )) MiB"
if [[ " ${STAGES_SEL:-a b c} " == *" a "* ]]; then
  for n in short-0 4k-0 16k-0; do
    case $n in short*) ms=4096 ;; 4k*) ms=8192 ;; *) ms=20480 ;; esac
    E="INTARSIA_SPEC_TRACE=1" g a-$n-base 0,1 base "$M27" $ms $n --tp 2 --devices 0,1
    E="INTARSIA_SPEC_TRACE=1" g a-$n-dev 0,1 dev "$M27" $ms $n --tp 2 --devices 0,1
    same a-$n-dev.ids a-$n-base.ids "(a) $n: dev default ids == base default ids"
    samespec a-$n-dev.err a-$n-base.err "(a) $n: dev vs base default"
    sametrace a-$n-dev.trace a-$n-base.trace "(a) $n: dev vs base default"
  done
fi
if [[ " ${STAGES_SEL:-a b c} " == *" b "* ]]; then
  ( E="INTARSIA_SPEC_TRACE=1" g b-16k-dev 0 dev "$M35" 20480 16k-0 --spec mtp --spec-k 2 ) &
  ( g b-16k-off 1 dev "$M35" 20480 16k-0 --spec off ) &
  wait
  same b-16k-dev.ids b-16k-off.ids "(b) 35B tp1 16k: dev --spec mtp ids == dev --spec off"
  log "(b) macro passes: off=[$(grep -a 'macro prefill' "$O/b-16k-off.err")] dev-mtp=[$(grep -a 'macro prefill' "$O/b-16k-dev.err")]"
fi
if [[ " ${STAGES_SEL:-a b c} " == *" c "* ]]; then
  if (( SECONDS < OPT_AFTER )); then
    E="INTARSIA_SPEC_TRACE=1" g b-16k-base 0 base "$M35" 20480 16k-0 --spec mtp --spec-k 2
    same b-16k-base.ids b-16k-off.ids "(c) 35B tp1 16k: base --spec mtp ids == dev --spec off"
    samespec b-16k-dev.err b-16k-base.err "(c) 35B tp1 16k: dev vs base --spec mtp"
    sametrace b-16k-dev.trace b-16k-base.trace "(c) 35B tp1 16k: dev vs base --spec mtp"
  fi
  for n in short-0 4k-0 16k-0; do
    case $n in short*) ms=4096 ;; 4k*) ms=8192 ;; *) ms=20480 ;; esac
    if (( SECONDS < OPT_AFTER )); then
      g a-$n-off 0,1 dev "$M27" $ms $n --tp 2 --devices 0,1 --spec off
      same a-$n-dev.ids a-$n-off.ids "(c) $n: 27B dev default ids == dev --spec off"
    else log "SKIP a-$n-off (past ${OPT_AFTER} s)"; fi
  done
fi
log "stages done (${SECONDS} s)"
