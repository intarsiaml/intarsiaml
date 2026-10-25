#!/usr/bin/env bash
# Lane tp2 window W=w19 (ON tatooine, radiance down, window.sh with SNAP=batch STAGES=w17): repro rate of the intermittent first-step mismatch of
# the 27B batched step at tp=2 (w15: N=4, step 0, every logit ~8 % off; w16: the same full test passed), and the candidate fix.
#  - the test real_35b_tp2_batch_first_steps_repro loads the 27B afresh per rep and runs INTARSIA_REPRO_STEPS batched steps per size 2..16 against the
#    solo steps, counting mismatches (REPRO failures f/reps). Each size's first step records its graphs, which is where the w15 failure was.
#  V0 baseline; V1 INTARSIA_TP_BATCH_PREPARE=1 (the peers' graphs are recorded one after the other on the main thread, before rank 0 replays and before
#  the peer threads run: no capture overlaps another rank's launches); V2 batched graphs off (INTARSIA_BATCH_GRAPHS=0); V3 HIP_LAUNCH_BLOCKING=1.
#  First the 27B accuracy of w16 that failed on a missing golden: tp=1, tp=2 twice with dumps, tp=2 against the golden alone (hard gate), run to run.
set -uo pipefail
export O=${1:?out dir}
source "$HOME/tp2/lib.sh"
D=${D:-batch}; PC=${PC:-none}  # the build head (the queued window was submitted with an older PC)
CAP=${CAP:-1750}; T0=$(date +%s)
left() { echo $(( CAP - ($(date +%s) - T0) )); }
need() { if (( $(left) > $1 )); then return 0; fi; log "SKIPPED (time left $(left)s < $1): $2"; return 1; }
vram() { local m=0 v c; for c in /sys/class/drm/card[0-9]/device/mem_info_vram_used; do v=$(cat "$c" 2>/dev/null || echo 0); (( v > m )) && m=$v; done; echo $m; }
clear() { local i; for i in 1 2 3 4 5 6; do (( $(vram) < 1073741824 )) && return 0; sleep 10; done
  log "CONTENDED before '$1': $(( $(vram) / 1048576 )) MiB in use on a card: NO VERDICT"; return 1; }
M27=/models/models--amd--Qwen3.8-27B-Quark-AWQ-MXFP4/snapshots/5233554c5fa56afda40150556b95573c2d7d29c0
G27=/work/p1-golden/qwen3.8-27b-mxfp4-hfquark-w4a16-top20.safetensors
export INTARSIA_MODELS=/data/models/radiance/hf-cache/hub
log "window start, CAP ${CAP}s, pre-change $PC, VRAM in use $(( $(vram) / 1048576 )) MiB, golden $(ls "$(T $D)/p1-golden" 2>&1 | head -1)"
mkdir -p "$(T $D)/tp2-out"
FLOOR="--self-max-kl 1e-9 --self-max-pos-kl 1e-6 --self-max-flip-pct 0 --self-max-tail 5e-8 --pre-change $PC"
# every container started by a step is removed when the step ends (a timeout kills the docker client, not the container)
sweep() { local keep=$1 c; for c in $(docker ps -q); do grep -q "$c" <<< "$keep" || { docker rm -f "$c" > /dev/null 2>&1 && log "removed leftover container $c"; }; done; }
repro() { # repro <tag> <timeout> <test> <env...>
  local tag=$1 to=$2 test=$3; shift 3
  local keep; keep=$(docker ps -q)
  INTARSIA_REPO=$(T $D) INTARSIA_TARGET_VOL=intarsia-cargo-target-tp2-$D-hiptest HIP_VISIBLE_DEVICES=0,1 timeout -k 10 "$to" \
    "$(T $D)/deploy/tatooine/dev.sh" env INTARSIA_BATCH_TEST_BACKEND=hip INTARSIA_REQUIRE_GPU=1 INTARSIA_QWEN36_DIR="$M27" INTARSIA_BATCH_TEST_TP=2 "$@" \
    cargo test --release -p intarsia-models --test batch_cuda_real -- --test-threads=1 --nocapture --ignored --exact $test \
    > "$O/repro-$tag.log" 2>&1 < /dev/null
  local rc=$?
  sweep "$keep"
  log "repro $tag exit $rc [$*]: $(grep -aE '^REPRO failures' "$O/repro-$tag.log" | tail -1) | $(grep -aE '^REPRO rep .*FAIL' "$O/repro-$tag.log" | head -3 | cut -c1-200 | tr '\n' '|')"
}
FS=real_35b_tp2_batch_first_steps_repro; SO=real_35b_tp2_solo_fresh_states_repro
# w18: the size-loop mismatch reproduced in EVERY variant at rep 0..3 (graphs on, prepare, batched graphs off, launch blocking), always at the first
# batched step of fresh sequences ("step 0"), and the SOLO logit of the failing row differed between runs (seq 7 ctx 221: 2.473 in four runs, 2.125 in one;
# seq 12 ctx 27: -1.43 vs 0.377): the solo step of a fresh state is wrong too, so the batched step is not (only) the culprit. This window takes parts out.
variant() { need 250 "$1" && clear "$1" && repro "$@"; }
# E1 everything eager (no solo graphs, no batched graphs) -- fails => not a graph problem
variant e1-size-loop-no-graphs 240 $FS INTARSIA_REPRO_REPS=12 INTARSIA_REPRO_STEPS=2 INTARSIA_REPRO_NO_GRAPHS=1
# E2 no batched step at all: 60 fresh states, first decode step bit-identical? -- fails => the solo step of a fresh state at tp=2 (production relevant)
variant e2-solo-only 240 $SO INTARSIA_REPRO_REPS=3 INTARSIA_REPRO_STATES=60
# E3 solo + one batched step between every other state
variant e3-solo-plus-batch 240 $SO INTARSIA_REPRO_REPS=3 INTARSIA_REPRO_STATES=60 INTARSIA_REPRO_BATCH_BETWEEN=1
# E4 the same, all eager
variant e4-solo-plus-batch-no-graphs 240 $SO INTARSIA_REPRO_REPS=3 INTARSIA_REPRO_STATES=60 INTARSIA_REPRO_BATCH_BETWEEN=1 INTARSIA_REPRO_NO_GRAPHS=1
# E5 fusion off (a different collective sequence)
variant e5-size-loop-fusion0 240 $FS INTARSIA_REPRO_REPS=12 INTARSIA_REPRO_STEPS=2 INTARSIA_TP_FUSION=0
# E6 solo-only, eager
variant e6-solo-only-no-graphs 240 $SO INTARSIA_REPRO_REPS=3 INTARSIA_REPRO_STATES=60 INTARSIA_REPRO_NO_GRAPHS=1
log "done, time left $(left)s"
