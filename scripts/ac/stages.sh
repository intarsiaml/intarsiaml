#!/usr/bin/env bash
# Lane ac: ONE radiance-down window on tatooine (ON tatooine, called by window.sh). Qwen3.8-27B MXFP4, tp=1, GPU 0 only.
# Question: is the 27B accuracy gate failure an engine fault, or a golden / flag mismatch? And, if the reference is the problem,
# the engine baseline + noise envelope the 27B gate should use (docs/results/2026-10-25-ac-27b-accuracy-gate.md).
#   0. control: the W4A16 golden with the DEFAULT flags (activations dynamic = W4A4), the way tp2's w12 ran it. Expect FAIL.
#   1. W4A16 golden with `--mxfp4-act none` (the flag that matches the golden). Expect PASS. Records baseline + dump.
#   2. envelope (same build, prefill chunk /4), 3. gate against baseline + envelope.
#   4-6. the same for the W4A4 golden (dynamic activations, the checkpoint's declared mode), with the hash-bound ref-noise sidecar.
set -uo pipefail
export O=${1:?out dir}
CAP=${CAP:-900}; T0=$(date +%s)
REPO=$HOME/intarsiaml-ac
PC=${PC:?PC=<head sha7 of the build>}
[[ -r $HOME/ac-logs/pc ]] && PC=$(cat "$HOME/ac-logs/pc")   # the build job writes the head it built (the queued window may carry an older PC)
log() { echo "$(date +%T) $*" | tee -a "$O/steps.log"; }
left() { echo $(( CAP - ($(date +%s) - T0) )); }
M27=/models/models--amd--Qwen3.8-27B-Quark-AWQ-MXFP4/snapshots/5233554c5fa56afda40150556b95573c2d7d29c0
G16=/work/p1-golden/qwen3.8-27b-mxfp4-hfquark-w4a16-top20.safetensors
G4=/work/p1-golden/qwen3.8-27b-mxfp4-hfquark-top20.safetensors
N4=/work/golden/ref_noise/qwen3.8-27b-mxfp4-hfquark-top20.safetensors.json
export INTARSIA_MODELS=/data/models/radiance/hf-cache/hub
# run <timeout> <tag> <accuracy args...>: one accuracy run in the dev image on GPU 0; stdout/stderr -> $O/<tag>.{out,err}
run() {
  local to=$1 tag=$2; shift 2
  INTARSIA_REPO=$REPO INTARSIA_TARGET_VOL=intarsia-cargo-target-ac HIP_VISIBLE_DEVICES=0 timeout "$to" \
    "$REPO/deploy/tatooine/dev.sh" /work/ac-intarsia accuracy --backend hip --model "$M27" --mem-budget 29 --max-seq 4096 "$@" \
    > "$O/$tag.out" 2> "$O/$tag.err" < /dev/null
  local rc=$?
  log "$tag exit $rc: $(grep -ahE 'golden gate|floor verdict|baseline verdict|^Error' "$O/$tag.out" "$O/$tag.err" | cut -c1-200 | tr '\n' '|')"
  log "  $tag act: $(grep -ahE 'mxfp4 activations|this build' "$O/$tag.err" | cut -c1-160 | tr '\n' '|')"
  return $rc
}
mkdir -p "$REPO/ac-out"
log "window start, CAP ${CAP}s, pre-change $PC"
# 0. control (mismatched flags, as w12 ran it)
run 240 ctl-w4a16-golden-default-flags --golden $G16 --mode both
# 1-3. W4A16
A16="--mxfp4-act none"
B16=/work/ac-out/qwen3.8-27b-mxfp4-hfquark-w4a16-top20.hip.baseline.json
E16=/work/ac-out/qwen3.8-27b-mxfp4-hfquark-w4a16-top20.hip.envelope.json
run 300 a16-record $A16 --golden $G16 --mode both --decode-positions 16 --write-baseline $B16 --write-self-golden /work/ac-out/a16.safetensors
run 300 a16-envelope $A16 --golden $G16 --mode both --decode-positions 16 --noise-ref $B16 --noise-perturb chunk --write-noise-envelope $E16 --allow-same-code
run 300 a16-gate $A16 --golden $G16 --mode both --decode-positions 16 --baseline $B16 --noise-envelope $E16 --pre-change "$PC" --allow-same-code
# 4-6. W4A4 (declared mode)
B4=/work/ac-out/qwen3.8-27b-mxfp4-hfquark-top20.hip.baseline.json
E4=/work/ac-out/qwen3.8-27b-mxfp4-hfquark-top20.hip.envelope.json
if (( $(left) > 400 )); then
  run 300 a4-record --golden $G4 --mode both --decode-positions 16 --ref-noise $N4 --write-baseline $B4 --write-self-golden /work/ac-out/a4.safetensors
  run 300 a4-envelope --golden $G4 --mode both --decode-positions 16 --noise-ref $B4 --noise-perturb chunk --write-noise-envelope $E4 --allow-same-code
  run 300 a4-gate --golden $G4 --mode both --decode-positions 16 --ref-noise $N4 --baseline $B4 --noise-envelope $E4 --pre-change "$PC" --allow-same-code
else log "SKIPPED W4A4 (time left $(left)s)"; fi
log "done, time left $(left)s"
