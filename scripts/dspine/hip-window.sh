#!/usr/bin/env bash
# DSpine draft (HIP, tatooine, tp=1 on card 0): bridge kernel tests, then greedy ids vs --spec off with tokens a pass and tok/s.
# Run ON tatooine inside a fleetq slot that takes radiance down:
#   fleetq run --host tatooine --expect 40 --takes-down radiance --label "dspine HIP window" -- ssh tatooine "ARMS='...' bash -s" < scripts/dspine/hip-window.sh
# Radiance is restarted on exit (trap), whatever happens.
set -uo pipefail
TREE=$HOME/intarsiaml-dsp; BIN=${BIN:-dsp-intarsia}; O=$TREE/.dsp/${TAG:-w1}; mkdir -p "$O"
M=/models/radiance/hf-cache/hub/models--amd--Qwen3.8-27B-Quark-AWQ-MXFP4/snapshots/5233554c5fa56afda40150556b95573c2d7d29c0
DS=/models/intarsia/Qwen3.8-27B-DSpine-b16; DFL2=/models/tensorfold/Qwen3.8-27B-DFlash2
ARMS=${ARMS:-off ds7 ds15 df7}; PROMPTS=${PROMPTS:-short 4k}; TOKENS=${TOKENS:-256}
SHORT='Write a long story about a robot who learns to paint.'
log() { echo "$(date +%T) $*" | tee -a "$O/steps.log"; }
dev() { INTARSIA_MODELS=/data/models INTARSIA_REPO=$TREE INTARSIA_TARGET_VOL=intarsia-cargo-dt1 HIP_VISIBLE_DEVICES=0 timeout "$1" "$TREE/deploy/tatooine/dev.sh" "${@:2}" < /dev/null; }
restore() {
  log "starting radiance"; systemctl --user start radiance < /dev/null
  for i in $(seq 1 60); do curl -s -m 3 http://localhost:8080/v1/models | grep -q '"data":\[' && { log "radiance answers"; return; }; sleep 5; done
  log "RADIANCE NOT ANSWERING after 300 s"
}
trap restore EXIT
log "stopping radiance"; systemctl --user stop radiance < /dev/null
for i in $(seq 1 30); do
  m=0; for c in /sys/class/drm/card[0-9]/device/mem_info_vram_used; do v=$(cat "$c" 2>/dev/null || echo 0); (( v > m )) && m=$v; done
  (( m < 1073741824 )) && break; sleep 5
done
log "vram max $((m / 1048576)) MiB"
for op in bridge_norm bridge_code bridge_gate bridge_write; do
  dev 300 /work/$BIN kernels test --backend hip --op $op > "$O/ktest-$op.log" 2>&1
  log "ktest $op exit $?: $(tail -3 "$O/ktest-$op.log" | cut -c1-160 | tr '\n' '|')"
done
for p in $PROMPTS; do
  P="$SHORT"; [[ $p == 4k ]] && P="$(cat "$TREE/scripts/hip-decode/prompt4k.txt")"
  for a in $ARMS; do
    case $a in
      off) args=(--spec off) ;;
      mtp2) args=(--spec mtp --spec-k 2) ;;
      dsd) args=(--spec dflash --spec-draft $DS) ;;
      ds*) args=(--spec dflash --spec-k ${a#ds} --spec-draft $DS) ;;
      df*) args=(--spec dflash --spec-k ${a#df} --spec-draft $DFL2) ;;
    esac
    dev 600 /work/$BIN generate -m $M --backend hip --mem-budget 29 --max-seq 20480 --chat --no-think \
      --profile-every 0 --ignore-eos --show-ids -p "$P" --max-tokens $TOKENS "${args[@]}" > "$O/gen-$p-$a.out" 2> "$O/gen-$p-$a.err"
    rc=$?; grep -a '^ids:' "$O/gen-$p-$a.err" > "$O/gen-$p-$a.ids"
    log "gen $p $a exit $rc: $(grep -ahE '^Error|error:|tok/s|spec|accept' "$O/gen-$p-$a.err" | tail -3 | cut -c1-200 | tr '\n' '|')"
    if [[ $a != off ]]; then
      if cmp -s "$O/gen-$p-off.ids" "$O/gen-$p-$a.ids" && [[ -s $O/gen-$p-$a.ids ]]; then log "SAME $p $a"; else log "DIFF $p $a"; fi
    fi
  done
done
