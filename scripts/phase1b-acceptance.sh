#!/usr/bin/env bash
# Phase 1b acceptance run on spark1: kernels, accuracy against baselines taken from the PRE-CHANGE build, graph replay
# against eager, and an interleaved A/B against vLLM. One script, run from the Mac (endor), stages in order:
#
#   preflight  Mac only. Stamp the tree under test, build the "before" tree (PRE_CHANGE + the accuracy harness files
#              only), write the revision list the gate checks ancestry against.        (no fleet box touched)
#   build      spark1, small shared fleetq slot, vLLM stays up. Sync both trees, build both, compile the CUDA tests.
#   lease      spark1, ONE exclusive slot that takes `guidance` down. Stops the vLLM container, waits for its memory.
#   kernels    `intarsia kernels test --backend cuda` must say ADMITTED; the CUDA kernel tests.
#   accuracy   per model/golden: the OLD build records baseline + self-golden, the NEW build is gated against both
#              (floor, baseline, per-prompt, same-weights self-golden with budgets from a measured noise floor).
#   graphs     256-step greedy decode on the real 35B, graph replay vs --no-graphs (must be identical), replays > 0.
#   ab         ROUNDS interleaved rounds. Each round: vLLM down, ours-new and ours-old (alternating order), then
#              vLLM up and vLLM. The container is stopped and started between arms; ours never runs beside it.
#   verdict    the decision rule below, applied to the saved JSON. Exit 0 only on PASS.
#   (always)   trap: stop our server (wait, then kill -9), start the vLLM container, wait until healthy, release the slots.
#
# Why vLLM is stopped: the 01:50 hang of spark1 happened with an 8B HF golden job next to the 64 GB vLLM reservation
# (docs/results/2026-10-13-phase1b.md). Nothing else runs on the box during the lease: no HF job, no golden job, no build.
#
# DECISION RULE (written before any number exists; do not edit after a run):
#   * A round of an arm is VALID when (ours-new only) graph replays > 0 and eager decode steps == 0 and profiled
#     steps == 0 for every measured size, AND the in-flight poller (1 Hz) never saw running > 1 or waiting > 0 on
#     the server under test (the bench is the only client). Invalid rounds are repeated, at most 2 extra rounds.
#   * Every arm runs with --profile-every 0 (profiled eager steps are ~100x slower and bias both sides); the
#     as-shipped profiling default is reported as one separate round only when AB_SHIPPED=1.
#   * Per size S and round r: d(S,r) = median decode tok/s of ours-new minus that of vLLM in the SAME round
#     (median of REPS measured calls after one discarded warmup call).
#   * PASS(S) when d(S,r) >= 0 in every valid round; MARGINAL(S) when the median of d is >= 0 but some round is < 0;
#     FAIL(S) when the median of d is < 0. Fewer than 3 valid rounds: INCONCLUSIVE(S).
#   * Phase 1 speed gate: PASS when PASS at 128, 1024 and 4096; FAIL when any FAIL; otherwise NOT DECIDED (report the
#     min/median/max of d per size). The accuracy stages must also pass for the run to exit 0.
#   * Self-golden budgets = max(DEFAULT, 3 x the noise floor measured in this run): the "noise" run is the OLD build
#     against its own dump with the SIMT attention kernels instead of flash/split-KV (a numerically equivalent
#     reordering on the real weights). DEFAULT: mean KL(21-way) 1e-4, max position KL 5e-3, top-1 flips 0.5%, mean
#     |tail diff| 5e-4, prompt KL x5 (docs/accuracy.md, "Choosing the self-golden budgets").
#
# Required environment (no defaults on purpose):
#   VLLM_CONTAINER   docker container of spark1's guidance vLLM (find it with `ssh spark1 docker ps`; it is not recorded
#                    in this repo, and the script restarts it with `docker start`, so no launch flags are guessed)
#   S1_35B           Qwen3.6-35B-A3B NVFP4 model dir on spark1
# Optional: S1_8B, PRE_CHANGE (4963b43), ROUNDS (3), REPS (5), GEN (128), SIZES (128,1024,4096), HOST (spark1),
#   VLLM_HOST (127.0.0.1: address the vLLM container is bound to; spark1 binds its tailnet IP, see `docker inspect`),
#   MACHINE (spark-4e77), MEM_LEASE_GB (100), MODEL_GB_35B (38), MODEL_GB_8B (17), STAGES (comma list, default all),
#   AB_SHIPPED (0), ALLOW_DIRTY (0: development only, the tree under test is then not stamped clean),
#   DRY_RUN (1: stub ssh/rsync/fleetq/docker, log what would run), FAIL_AT=<stage> (DRY_RUN: force that stage to fail
#   and show that the trap restores vLLM), WORK (artefact directory on the Mac).

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HOST=${HOST:-spark1}
MACHINE=${MACHINE:-spark-4e77}
PRE_CHANGE=${PRE_CHANGE:-4963b43}
ROUNDS=${ROUNDS:-3}
REPS=${REPS:-5}
GEN=${GEN:-128}
SIZES=${SIZES:-128,1024,4096}
MEM_LEASE_GB=${MEM_LEASE_GB:-100}
BUILD_JOBS=${BUILD_JOBS:-8}               # cargo -j on spark1 while vLLM is up: bounds rustc/nvcc memory
MODEL_GB_35B=${MODEL_GB_35B:-38}
MODEL_GB_8B=${MODEL_GB_8B:-17}
STAGES=${STAGES:-all}
AB_SHIPPED=${AB_SHIPPED:-0}
ALLOW_DIRTY=${ALLOW_DIRTY:-0}
DRY_RUN=${DRY_RUN:-0}
FAIL_AT=${FAIL_AT:-}
S1_8B=${S1_8B:-/home/nvidia/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218}
TS=$(date +%Y%m%d-%H%M%S)
WORK=${WORK:-$REPO/target/phase1b-acceptance/$TS}
R='~/intarsia-acc'                  # remote root (expanded by the remote shell)
GD='~/intarsiaml-golden'            # goldens live outside the synced tree (see docs/accuracy.md)
NEWBIN="$R/new/target/acc/release/intarsia"
OLDBIN="$R/old/target/acc/release/intarsia"
SERVE_PORT=8090
VLLM_PORT=8000
VLLM_HOST=${VLLM_HOST:-127.0.0.1}   # address the vLLM container is bound to (spark1: the tailnet IP; see `docker inspect`, --host)
DRYLOG="$WORK/dry-run.log"
mkdir -p "$WORK"

log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*" >&2; }
die() { log "ABORT: $*"; exit 1; }
have_stage() { [[ "$STAGES" == all || ",$STAGES," == *",$1,"* ]]; }

# ---------------------------------------------------------------------------------------------------------------------
# Everything that touches the fleet goes through these four wrappers, so DRY_RUN can stub it and the trap test is real.
# ---------------------------------------------------------------------------------------------------------------------
dry() { printf '%s\n' "$*" >> "$DRYLOG"; }
rsh() {                                        # run a command line on the box
  if [[ $DRY_RUN == 1 ]]; then dry "ssh $HOST: $*"; return 0; fi
  ssh -n -o BatchMode=yes "$HOST" "$@"
}
rsh_stdin() {                                  # run a command on the box with our stdin (to write a file)
  if [[ $DRY_RUN == 1 ]]; then dry "ssh $HOST (stdin): $*"; cat >/dev/null; return 0; fi
  ssh -o BatchMode=yes "$HOST" "$@"
}
rsh_out() {                                    # same, stdout wanted (dry run: canned answers for the probes we make)
  if [[ $DRY_RUN == 1 ]]; then
    dry "ssh $HOST (out): $*"
    case "$*" in
      *MemAvailable*) echo "${FAKE_MEM_KB:-120000000}" ;;
      *SELF-STATS*) echo "SELF-STATS mode=prefill positions=100 mean_kl21=2e-5 max_kl21=1e-3 flips=0 flip_pct=0 tail_abs=1e-5 worst_prompt_kl21=3e-5" ;;
      *) echo "" ;;
    esac
    return 0
  fi
  ssh -n -o BatchMode=yes "$HOST" "$@"
}
rsync_to() {                                   # rsync_to <local dir> <remote dir>
  if [[ $DRY_RUN == 1 ]]; then dry "rsync $1 -> $HOST:$2"; return 0; fi
  rsync -a --delete --exclude target --exclude .git --exclude golden --exclude tools/.venv --exclude .claude "$1/" "$HOST:$2/"
}
fq() {                                         # fleetq
  if [[ $DRY_RUN == 1 ]]; then
    dry "fleetq $*"
    [[ "$1" == acquire ]] && echo "9999"
    return 0
  fi
  fleetq "$@"
}
docker_remote() {                              # docker on the box, as the box's user
  if [[ $DRY_RUN == 1 ]]; then dry "docker $*"; return 0; fi
  ssh -n -o BatchMode=yes "$HOST" docker "$@"
}

# ---------------------------------------------------------------------------------------------------------------------
# State and the trap. Order matters: our server first (its memory), then vLLM back up and healthy, then the slots.
# ---------------------------------------------------------------------------------------------------------------------
BUILD_JOB=""; LEASE_JOB=""; VLLM_STOPPED=0; SERVER_UP=0; STATUS=0
unset FLEETQ_JOB

stop_server() {
  [[ $SERVER_UP == 1 ]] || return 0
  log "stopping our server (wait for the pid, then kill -9)"
  rsh "pid=\$(cat /tmp/intarsia-acc-serve.pid 2>/dev/null || true); if [ -n \"\$pid\" ]; then kill \$pid 2>/dev/null || true; for i in \$(seq 1 60); do kill -0 \$pid 2>/dev/null || break; sleep 1; done; kill -9 \$pid 2>/dev/null || true; fi; for i in \$(seq 1 30); do curl -sf http://127.0.0.1:$SERVE_PORT/health >/dev/null 2>&1 || break; sleep 1; done; rm -f /tmp/intarsia-acc-serve.pid" || true
  SERVER_UP=0
}

vllm_healthy() { rsh_out "curl -sf http://$VLLM_HOST:$VLLM_PORT/health >/dev/null && curl -sf http://$VLLM_HOST:$VLLM_PORT/v1/models | grep -q ." >/dev/null 2>&1; }

start_vllm() {
  [[ $VLLM_STOPPED == 1 ]] || return 0
  log "starting the vLLM container $VLLM_CONTAINER and waiting for it to serve"
  FLEETQ_JOB="$LEASE_JOB" docker_remote start "$VLLM_CONTAINER" >/dev/null || log "WARNING: docker start failed"
  if [[ $DRY_RUN == 1 ]]; then VLLM_STOPPED=0; return 0; fi
  local i
  for i in $(seq 1 180); do                    # up to 15 minutes: 35B NVFP4 load
    if vllm_healthy; then VLLM_STOPPED=0; log "vLLM is serving again"; return 0; fi
    sleep 5
  done
  log "WARNING: vLLM did not answer /health within 15 min; the slot is released anyway, tell the owner of guidance"
}

cleanup() {
  local rc=$?
  trap - EXIT INT TERM
  set +e
  [[ $rc -ne 0 ]] && log "exiting with status $rc, restoring the box"
  stop_server
  start_vllm
  for j in "$LEASE_JOB" "$BUILD_JOB"; do
    [[ -n "$j" ]] && { log "releasing fleetq job $j"; fq release "$j" >/dev/null || log "WARNING: release of $j failed"; }
  done
  [[ -d "$WORK/old-wt" ]] && git -C "$REPO" worktree remove --force "$WORK/old-wt" >/dev/null 2>&1
  [[ $rc -eq 0 ]] && rc=$STATUS
  log "done (status $rc); artefacts in $WORK"
  exit $rc
}
trap cleanup EXIT INT TERM

maybe_fail() { [[ $DRY_RUN == 1 && "$FAIL_AT" == "$1" ]] && { log "DRY_RUN: forced failure at $1"; return 1; }; return 0; }

# memory guard: abort if MemAvailable < model + 20 GB (the hang looked like memory pressure on the unified memory)
memcheck() {                                   # memcheck <model GB> <what>
  local need=$(( $1 + 20 )) kb gb
  kb=$(rsh_out "awk '/MemAvailable/ {print \$2}' /proc/meminfo")
  gb=$(( ${kb:-0} / 1024 / 1024 ))
  log "memory: ${gb} GB available, $2 needs $1 GB + 20 GB headroom"
  (( gb >= need )) || die "MemAvailable ${gb} GB < $need GB (model $1 GB + 20 GB) before $2; something else holds the memory, not starting"
}

wait_mem_freed() {                             # after docker stop: the 64 GB must actually come back
  local i
  [[ $DRY_RUN == 1 ]] && return 0
  for i in $(seq 1 60); do
    local gb=$(( $(rsh_out "awk '/MemAvailable/ {print \$2}' /proc/meminfo") / 1024 / 1024 ))
    (( gb >= 70 )) && { log "memory freed: ${gb} GB available"; return 0; }
    sleep 5
  done
  die "MemAvailable did not recover after stopping vLLM"
}

# ---------------------------------------------------------------------------------------------------------------------
# preflight (Mac)
# ---------------------------------------------------------------------------------------------------------------------
stage_preflight() {
  log "== preflight: stamp the tree under test, build the pre-change tree"
  cd "$REPO"
  command -v python3 >/dev/null || die "python3 is needed for the verdict"
  git cat-file -e "$PRE_CHANGE^{commit}" || die "$PRE_CHANGE is not a commit here"
  git merge-base --is-ancestor "$PRE_CHANGE" HEAD || die "$PRE_CHANGE is not an ancestor of HEAD"
  local new_stamp old_stamp
  new_stamp=$(scripts/tree-stamp.sh)
  log "tree under test: $new_stamp"
  if [[ $new_stamp == *dirty=yes* && $ALLOW_DIRTY != 1 ]]; then
    die "the working tree differs from HEAD; commit first (ALLOW_DIRTY=1 only for development, the results then carry dirty=yes)"
  fi
  git rev-list HEAD > "$WORK/revlist"
  # the new tree to ship: HEAD itself (committed content), or the working tree in development mode
  rm -rf "$WORK/new" "$WORK/old"; mkdir -p "$WORK/new" "$WORK/old"
  if [[ $ALLOW_DIRTY == 1 ]]; then
    rsync -a --exclude target --exclude .git --exclude golden --exclude tools/.venv --exclude .claude "$REPO/" "$WORK/new/"
  else
    git archive HEAD | tar -x -C "$WORK/new"
  fi
  # the pre-change tree: PRE_CHANGE with only the accuracy harness files copied over it (it has no baseline or
  # self-golden support of its own). A temporary worktree gives tree-stamp.sh a git repo to look at.
  local wt="$WORK/old-wt" f
  git worktree add --detach "$wt" "$PRE_CHANGE" >/dev/null
  OVERLAY=()                                    # (bash 3.2 on the Mac has no mapfile)
  while IFS= read -r f; do OVERLAY+=("$f"); done < <(sed -n '/HARNESS_OVERLAY_PATHS: &\[&str\] = &\[/,/\];/p' crates/intarsia-cli/src/cmd/accuracy/provenance.rs | grep -o '"crates/[^"]*"' | tr -d '"')
  (( ${#OVERLAY[@]} >= 3 )) || { git worktree remove --force "$wt"; die "could not read HARNESS_OVERLAY_PATHS from provenance.rs"; }
  for f in "${OVERLAY[@]}"; do
    mkdir -p "$wt/$(dirname "$f")"
    if [[ $ALLOW_DIRTY == 1 ]]; then cp "$f" "$wt/$f"; else git show "HEAD:$f" > "$wt/$f"; fi
  done
  local dirty_paths overlay_list p
  dirty_paths=$(git -C "$wt" status --porcelain --untracked-files=all | awk '{print $2}' | sort)
  for p in $dirty_paths; do
    printf '%s\n' "${OVERLAY[@]}" | grep -qx "$p" || { git worktree remove --force "$wt"; die "the pre-change tree differs in $p, which is not a declared harness file"; }
  done
  overlay_list=$(printf '%s\n' $dirty_paths | paste -sd, -)
  old_stamp="$(cd "$wt" && "$REPO/scripts/tree-stamp.sh") overlay=$overlay_list"
  rsync -a --exclude .git "$wt/" "$WORK/old/"
  git worktree remove --force "$wt"
  printf '%s\n' "$new_stamp" > "$WORK/new.stamp"; printf '%s\n' "$old_stamp" > "$WORK/old.stamp"
  log "pre-change tree: $old_stamp"
  [[ -n "${VLLM_CONTAINER:-}" || $DRY_RUN == 1 ]] || die "VLLM_CONTAINER is not set (the guidance vLLM container on $HOST)"
  [[ -n "${S1_35B:-}" || $DRY_RUN == 1 ]] || die "S1_35B is not set (35B model dir on $HOST)"
}
load_stamps() { NEW_STAMP=$(cat "$WORK/new.stamp"); OLD_STAMP=$(cat "$WORK/old.stamp"); }

# ---------------------------------------------------------------------------------------------------------------------
# build (small shared slot; vLLM stays up; nothing here loads the GPU)
# ---------------------------------------------------------------------------------------------------------------------
push_goldens() {                               # the small vLLM top-k goldens; the big full-vocabulary ones are already on the box
  local f
  for f in "$REPO"/golden/qwen36-35b-a3b-nvfp4-vllm-top20.safetensors "$REPO"/golden/qwen36-35b-a3b-nvfp4-vllm-long-top20.safetensors; do
    [[ -f "$f" || $DRY_RUN == 1 ]] || continue
    if [[ $DRY_RUN == 1 ]]; then dry "rsync $f -> $HOST:$GD/"; else rsync -a --ignore-existing "$f" "$HOST:$GD/"; fi
  done
}

stage_build() {
  log "== build: sync and build both trees on $HOST (shared slot, 16 GB, no GPU)"
  BUILD_JOB=$(fq acquire --host "$HOST" --mem 16 --label "intarsiaml phase1b: build only" --expect 60 --wait | grep -Eo '[0-9]+' | tail -1)
  [[ -n "$BUILD_JOB" ]] || die "could not read the fleetq job id"
  export FLEETQ_JOB=$BUILD_JOB
  rsh "mkdir -p $R/art $R/logs $R/ab $GD; df -h ~ | tail -1"
  rsync_to "$WORK/new" "$R/new"; rsync_to "$WORK/old" "$R/old"
  if [[ $DRY_RUN == 1 ]]; then dry "scp revlist"; else scp -q "$WORK/revlist" "$HOST:${R/#\~/.}/revlist"; fi
  push_goldens
  local nq oq
  nq=$(printf %q "$NEW_STAMP"); oq=$(printf %q "$OLD_STAMP")
  rsh "cd $R/new && env INTARSIA_BUILD_STAMP=$nq CARGO_BUILD_JOBS=$BUILD_JOBS CARGO_TARGET_DIR=target/acc cargo build --release --features cuda -p intarsia-cli 2>&1 | tail -5"
  rsh "cd $R/old && env INTARSIA_BUILD_STAMP=$oq CARGO_BUILD_JOBS=$BUILD_JOBS CARGO_TARGET_DIR=target/acc cargo build --release --features cuda -p intarsia-cli 2>&1 | tail -5"
  rsh "cd $R/new && CARGO_BUILD_JOBS=$BUILD_JOBS CARGO_TARGET_DIR=target/acc cargo test --release -p intarsia-backend-cuda --no-run 2>&1 | tail -5"
  rsh "test -x $NEWBIN && test -x $OLDBIN" || die "a build is missing"
  fq release "$BUILD_JOB" >/dev/null; BUILD_JOB=""; unset FLEETQ_JOB
}

# ---------------------------------------------------------------------------------------------------------------------
# lease: ONE exclusive slot, guidance down
# ---------------------------------------------------------------------------------------------------------------------
stage_lease() {
  log "== lease: exclusive slot on $HOST taking guidance down"
  LEASE_JOB=$(fq acquire --host "$HOST" --exclusive --takes-down guidance --mem "$MEM_LEASE_GB" --expect 360 \
              --label "intarsiaml phase1b acceptance (exclusive, vLLM down)" --wait | grep -Eo '[0-9]+' | tail -1)
  [[ -n "$LEASE_JOB" ]] || die "could not read the fleetq job id"
  export FLEETQ_JOB=$LEASE_JOB
  log "stopping the vLLM container $VLLM_CONTAINER"
  VLLM_STOPPED=1                               # set first: if docker stop half-fails the trap still tries docker start
  docker_remote stop "$VLLM_CONTAINER" >/dev/null
  wait_mem_freed
  rsh_stdin "cat > $R/poll.sh && chmod +x $R/poll.sh" <<'POLL'
#!/bin/sh
# poll.sh <host> <port> <outfile>: once a second, the largest running / waiting request counts seen at /metrics, until <outfile>.run goes away
mr=0; mw=0
while [ -e "$3.run" ]; do
  m=$(curl -s "http://$1:$2/metrics")
  r=$(echo "$m" | awk '/^vllm:num_requests_running/ {print int($2)}' | tail -1)
  w=$(echo "$m" | awk '/^vllm:num_requests_waiting/ {print int($2)}' | tail -1)
  [ "${r:-0}" -gt "$mr" ] && mr=${r:-0}
  [ "${w:-0}" -gt "$mw" ] && mw=${w:-0}
  echo "$mr $mw" > "$3"
  sleep 1
done
POLL
}

# ---------------------------------------------------------------------------------------------------------------------
# kernels
# ---------------------------------------------------------------------------------------------------------------------
stage_kernels() {
  log "== kernels: admission on the real shapes, then the CUDA kernel tests"
  memcheck 8 "kernels"
  rsh "$NEWBIN kernels test --backend cuda 2>&1 | tee $R/logs/kernels.log | tail -40; grep -q ADMITTED $R/logs/kernels.log && ! grep -qE 'NOT ADMITTED|FAILED' $R/logs/kernels.log" \
    || die "kernels test: not every kernel is ADMITTED (tolerances are 1e-3 rel_l2 for bf16 outputs; do not loosen them to pass: read the worst_rel_l2 in $R/logs/kernels.log and decide)"
  rsh "cd $R/new && CARGO_TARGET_DIR=target/acc cargo test --release -p intarsia-backend-cuda --test graph_pos --test quantmoe -- --include-ignored 2>&1 | tail -30" \
    || die "CUDA kernel tests failed"
}

# ---------------------------------------------------------------------------------------------------------------------
# accuracy
# ---------------------------------------------------------------------------------------------------------------------
# name | model | golden | model GB | extra flags
ACC_CASES() {
  cat <<EOF
35b-short|$S1_35B|$GD/qwen36-35b-a3b-nvfp4-vllm-top20.safetensors|$MODEL_GB_35B|--mode both --decode-positions 16 --max-seq 4096 --mem-budget 60
35b-long|$S1_35B|$GD/qwen36-35b-a3b-nvfp4-vllm-long-top20.safetensors|$MODEL_GB_35B|--mode all --decode-positions 64 --max-seq 16384 --mem-budget 70
8b-full|$S1_8B|$GD/qwen3-8b-full.safetensors|$MODEL_GB_8B|--mode both --decode-positions 16 --max-seq 4096 --mem-budget 40
8b-4k|$S1_8B|$GD/qwen3-8b-4k-top20.safetensors|$MODEL_GB_8B|--mode both --decode-positions 64 --max-seq 8192 --mem-budget 40
EOF
}
PROV="--pre-change $PRE_CHANGE --revlist $R/revlist"

stage_accuracy() {
  log "== accuracy: old build records, new build is gated"
  local dflag=""; [[ $ALLOW_DIRTY == 1 ]] && dflag="--allow-dirty"
  local failures=() skipped=()
  while IFS='|' read -r name model golden gb flags; do
    if ! rsh_out "test -s $golden && echo yes" | grep -q yes && [[ $DRY_RUN != 1 ]]; then
      log "SKIPPED $name: golden $golden does not exist on $HOST (build it in its own lease, never beside vLLM; docs/accuracy.md)"
      skipped+=("$name"); continue
    fi
    memcheck "$gb" "accuracy $name"
    local A="$R/art/$name" L="$R/logs/$name"
    # 1. BEFORE: the pre-change build (declared harness overlay) records baseline + self-golden in one run. Its floor may
    #    FAIL (8B prefill top-1 is 99.44% against 99.5%); that is recorded in the baseline and is not this run's verdict.
    log "-- $name: record from the pre-change build ($PRE_CHANGE + harness)"
    rsh "$OLDBIN accuracy --model $model --backend cuda --golden $golden $flags --write-baseline $A.baseline.json --write-self-golden $A.self.safetensors > $L.old.log 2>&1 || true; test -s $A.baseline.json && test -s $A.self.safetensors" \
      || die "$name: the pre-change build wrote no baseline / self-golden (see $L.old.log)"
    # 2. NOISE FLOOR: the same old build, SIMT attention instead of flash/split-KV, against its own dump
    log "-- $name: noise floor (old build, equivalent attention kernels, vs its own dump)"
    rsh "INTARSIA_CUDA_DISABLE=attn_decode_bf16_splitkv,attn_prefill_bf16_splitkv,attn_prefill_bf16_flash $OLDBIN accuracy --model $model --backend cuda --self-golden $A.self.safetensors ${flags/--mode all/--mode both} $PROV --allow-same-code --allow-dirty --self-max-kl 1 --self-max-pos-kl 10 --self-max-flip-pct 100 --self-max-tail 1 > $L.noise.log 2>&1 || true"
    local noise budget
    noise=$(rsh_out "grep SELF-STATS $L.noise.log")
    budget=$(NOISE="$noise" python3 - <<'PY'
import os,re
d=dict(kl=1e-4,pos=5e-3,flip=0.5,tail=5e-4)          # defaults = selfgolden::DEFAULT_BUDGET
m=dict(kl=0,pos=0,flip=0,tail=0)
for l in os.environ.get("NOISE","").splitlines():
    g=dict(re.findall(r'(\w+)=(\S+)',l))
    if not g: continue
    m['kl']=max(m['kl'],float(g['mean_kl21'])); m['pos']=max(m['pos'],float(g['max_kl21']))
    m['flip']=max(m['flip'],float(g['flip_pct'])); m['tail']=max(m['tail'],float(g['tail_abs']))
b={k:max(d[k],3*m[k]) for k in d}
print(f"--self-max-kl {b['kl']:.4g} --self-max-pos-kl {b['pos']:.4g} --self-max-flip-pct {b['flip']:.4g} --self-max-tail {b['tail']:.4g}")
PY
)
    log "-- $name: budgets (max of default and 3x noise): $budget"
    # 3. AFTER: floor + baseline + per-prompt gates, and the new build's own dump for the determinism check
    log "-- $name: gate the new build against the baseline (floor, baseline, per prompt)"
    rsh "$NEWBIN accuracy --model $model --backend cuda --golden $golden $flags --baseline $A.baseline.json $PROV $dflag --write-self-golden $A.new.safetensors > $L.new.log 2>&1; rc=\$?; echo exit \$rc; grep -E '(floor|baseline) verdict|golden gate' $L.new.log | tail -3" \
      | tee "$WORK/acc-$name.new.txt" || true
    # accepted when the baseline verdict is PASS and the floor did not get worse than the old build's floor
    local newl oldl
    newl=$(rsh_out "grep -E 'floor verdict' $L.new.log | tail -1"); oldl=$(rsh_out "grep -E 'floor verdict' $L.old.log | tail -1")
    if ! grep -q 'baseline verdict[^:]*: PASS' <<<"$newl"; then failures+=("$name: baseline gate"); fi
    if grep -q 'floor verdict[^:]*: FAIL' <<<"$newl" && ! grep -q 'floor verdict[^:]*: FAIL' <<<"$oldl"; then failures+=("$name: floor passed before and fails now"); fi
    # reuse gate ran inside --mode all; it is part of the verdict line (prefix-reuse verdict)
    grep -q 'prefix-reuse verdict: FAIL' <<<"$newl" && failures+=("$name: prefix reuse")
    # 4. SAME-WEIGHTS: the new build against the old build's logits
    log "-- $name: self-golden (new vs old logits)"
    rsh "$NEWBIN accuracy --model $model --backend cuda --self-golden $A.self.safetensors ${flags/--mode all/--mode both} $PROV $dflag $budget > $L.self.log 2>&1; echo exit \$?; grep -E 'SELF-STATS|self-golden =>|self-golden verdict' $L.self.log" | tee "$WORK/acc-$name.self.txt" || true
    rsh_out "grep -q 'self-golden verdict[^:]*: PASS' $L.self.log && echo ok" | grep -q ok || [[ $DRY_RUN == 1 ]] || failures+=("$name: self-golden")
    # 5. DETERMINISM (short sets only): the new build against its own dump in a second process must be bit-identical
    if [[ $name == 35b-short || $name == 8b-full ]]; then
      log "-- $name: determinism (new build vs its own dump, second process)"
      rsh "$NEWBIN accuracy --model $model --backend cuda --self-golden $A.new.safetensors ${flags/--mode all/--mode both} $PROV $dflag --allow-same-code --self-max-kl 1e-9 --self-max-pos-kl 1e-6 --self-max-flip-pct 0 --self-max-tail 1e-7 > $L.det.log 2>&1; echo exit \$?; grep -E 'SELF-STATS|self-golden =>' $L.det.log" | tee "$WORK/acc-$name.det.txt" || true
      rsh_out "grep -q 'self-golden verdict[^:]*: PASS' $L.det.log && echo ok" | grep -q ok || [[ $DRY_RUN == 1 ]] || failures+=("$name: run-to-run determinism of the new build (GPU noise: read $L.det.log before deciding anything)")
    fi
  done < <(ACC_CASES)
  printf '%s\n' "${failures[@]:-}" > "$WORK/accuracy.failures"
  (( ${#skipped[@]} )) && log "SKIPPED goldens: ${skipped[*]}"
  if (( ${#failures[@]} )) && [[ -n "${failures[0]}" ]]; then
    log "accuracy FAILED: ${failures[*]}"
    log "bisect before touching code: rerun the failing case with --no-graphs, then with INTARSIA_CUDA_DISABLE=<new kernel names> (a typo is now refused)"
    ACC_FAILED=1
  fi
}
ACC_FAILED=0

# ---------------------------------------------------------------------------------------------------------------------
# graphs: greedy 256 steps, replay vs eager, real 35B
# ---------------------------------------------------------------------------------------------------------------------
stage_graphs() {
  log "== graphs: 256-step greedy decode on the 35B, graph replay vs --no-graphs (bitwise identical token ids)"
  memcheck "$MODEL_GB_35B" "graph-vs-eager"
  local i=0 p replays eager
  for p in "Write a short story about a clockmaker who repairs a clock that runs backwards." \
           "Explain how a B-tree keeps its leaves at the same depth, with an example inserting the keys 1 to 20."; do
    i=$((i+1))
    local common="$NEWBIN generate --model $S1_35B --backend cuda --max-seq 2048 --profile-every 0 -p $(printf %q "$p") --max-tokens 256 --ignore-eos --show-ids"
    rsh "$common > $R/logs/g$i.out 2> $R/logs/g$i.err; $common --no-graphs > $R/logs/e$i.out 2> $R/logs/e$i.err; grep '^ids:' $R/logs/g$i.out > $R/logs/g$i.ids; grep '^ids:' $R/logs/e$i.out > $R/logs/e$i.ids; test -s $R/logs/g$i.ids && cmp $R/logs/g$i.ids $R/logs/e$i.ids" \
      || die "graph replay and eager decode differ (or produced nothing) for prompt $i: diff $R/logs/g$i.ids $R/logs/e$i.ids"
    [[ $DRY_RUN == 1 ]] && continue
    replays=$(rsh_out "sed -n 's/.*graph replays=\([0-9]*\).*/\1/p' $R/logs/g$i.err | tail -1")
    eager=$(rsh_out "sed -n 's/.*eager decode steps=\([0-9]*\).*/\1/p' $R/logs/g$i.err | tail -1")
    (( ${replays:-0} >= 200 )) || die "prompt $i: only ${replays:-0} graph replays in 255 decode steps: decode ran eagerly (capture failed, or the plan is not position-stable: see $R/logs/g$i.err)"
    (( ${eager:-1} == 0 )) || die "prompt $i: $eager eager decode steps with graphs on and --profile-every 0"
    log "prompt $i: graph == eager for 256 steps; replays=$replays eager=$eager"
  done
}

# ---------------------------------------------------------------------------------------------------------------------
# ab: interleaved rounds, vLLM down while ours runs
# ---------------------------------------------------------------------------------------------------------------------
start_server() {                               # start_server <bin> <extra serve flags...>
  local bin=$1; shift
  log "starting the server $bin"
  rsh "nohup $bin serve --backend cuda --model $S1_35B --port $SERVE_PORT --host 127.0.0.1 --served-model-name intarsia-35b --max-seq 8192 --mem-budget 38 --profile-every 0 $* > $R/logs/serve.log 2>&1 < /dev/null & echo \$! > /tmp/intarsia-acc-serve.pid"
  SERVER_UP=1
  [[ $DRY_RUN == 1 ]] && return 0
  local i
  for i in $(seq 1 120); do rsh_out "curl -sf http://127.0.0.1:$SERVE_PORT/health >/dev/null && echo up" | grep -q up && return 0; sleep 3; done
  die "our server did not become healthy (see $R/logs/serve.log)"
}

run_arm() {                                    # run_arm <arm> <round> <port> <served model name> <extra bench flags...>
  local arm=$1 rnd=$2 port=$3 model=$4; shift 4
  local out="$R/ab/$arm-r$rnd" host=127.0.0.1
  [[ $port == "$VLLM_PORT" ]] && host=$VLLM_HOST
  log "-- round $rnd: $arm"
  rsh "mkdir -p $out && touch $out/inflight.run && echo '0 0' > $out/inflight && (setsid $R/poll.sh $host $port $out/inflight > /dev/null 2>&1 < /dev/null &)"
  rsh "$NEWBIN bench speed --url http://$host:$port --model $model --sizes $SIZES --gen $GEN --reps $REPS --warmup 1 --thinking off --label p1b-$arm-r$rnd --out-dir $out $* > $out/bench.log 2>&1; echo \$? > $out/bench.rc; rm -f $out/inflight.run; sleep 2"
}

capture_vllm_info() {
  rsh "curl -s http://$VLLM_HOST:$VLLM_PORT/version > $R/ab/vllm-version.json; curl -s http://$VLLM_HOST:$VLLM_PORT/metrics | grep -E '^vllm:cache_config_info' > $R/ab/vllm-cache-config.txt; true"
}

fetch_ab() { [[ $DRY_RUN == 1 ]] || { mkdir -p "$WORK/ab"; scp -qr "$HOST:${R/#\~/.}/ab/." "$WORK/ab/"; }; }

stage_ab() {
  log "== ab: $ROUNDS interleaved rounds (ours-new, ours-old, vLLM); vLLM is stopped while ours runs"
  local rnd=0 first
  while (( rnd < ROUNDS + 2 )); do               # at most two extra rounds to replace invalid ones
    rnd=$((rnd+1))
    memcheck "$MODEL_GB_35B" "ab round $rnd"
    for a in $( (( rnd % 2 == 1 )) && echo "new old" || echo "old new" ); do
      if [[ $a == new ]]; then
        start_server "$NEWBIN"; run_arm ours-new "$rnd" "$SERVE_PORT" intarsia-35b --require-graphs; stop_server
        if [[ $AB_SHIPPED == 1 && $rnd == 1 ]]; then   # the as-shipped profiling default, reported on its own
          start_server "$NEWBIN" --profile-every 16; run_arm ours-new-shipped "$rnd" "$SERVE_PORT" intarsia-35b; stop_server
        fi
      else
        start_server "$OLDBIN"; run_arm ours-old "$rnd" "$SERVE_PORT" intarsia-35b; stop_server
      fi
    done
    start_vllm                                    # VLLM_STOPPED is 1 here (the lease stopped it); start_vllm clears it
    run_arm vllm "$rnd" "$VLLM_PORT" guidance
    capture_vllm_info
    docker_remote stop "$VLLM_CONTAINER" >/dev/null; VLLM_STOPPED=1; wait_mem_freed
    fetch_ab
    if (( rnd >= ROUNDS )); then
      if [[ $DRY_RUN == 1 ]]; then break; fi
      [[ $(python3 "$WORK/verdict.py" --count-valid "$WORK/ab") -ge $ROUNDS ]] && break
      log "fewer than $ROUNDS valid rounds so far, running another"
    fi
  done
}

# ---------------------------------------------------------------------------------------------------------------------
# verdict (Mac): the decision rule above, applied to the saved bench JSON
# ---------------------------------------------------------------------------------------------------------------------
write_verdict_py() {
cat > "$WORK/verdict.py" <<'PY'
#!/usr/bin/env python3
"""Apply the decision rule of scripts/phase1b-acceptance.sh to <dir>/<arm>-r<N>/*.json. Exit 0 PASS, 1 FAIL, 3 undecided."""
import glob, json, os, re, statistics, sys

def load(d):
    f = glob.glob(os.path.join(d, "*.json"))
    return json.load(open(f[0])) if f else None

def inflight(d):
    try:
        r, w = open(os.path.join(d, "inflight")).read().split()[:2]
        return int(r), int(w)
    except Exception:
        return None

def valid_round(root, arm, n):
    d = os.path.join(root, f"{arm}-r{n}")
    j = load(d)
    if j is None:
        return None, "no result"
    why = []
    inf = inflight(d)
    if inf is None: why.append("no in-flight samples")
    elif inf[0] > 1 or inf[1] > 0: why.append(f"in-flight seen running={inf[0]} waiting={inf[1]}")
    if arm == "ours-new":
        for row in j["rows"]:
            if row.get("skipped"): continue
            g = row.get("graph") or {}
            if not g.get("available"): why.append(f"size {row['target_prompt_tokens']}: no graph metrics")
            elif (g.get("replays") or 0) <= 0: why.append(f"size {row['target_prompt_tokens']}: 0 graph replays")
            elif (g.get("eager_steps") or 0) != 0 or (g.get("profiled_steps") or 0) != 0:
                why.append(f"size {row['target_prompt_tokens']}: eager={g.get('eager_steps')} profiled={g.get('profiled_steps')}")
    return (j, why)

def main():
    args = sys.argv[1:]
    count_only = args and args[0] == "--count-valid"
    root = args[-1]
    rounds = sorted({int(m.group(1)) for p in glob.glob(os.path.join(root, "*-r*")) if (m := re.search(r"-r(\d+)$", p))})
    valid, notes = [], []
    for n in rounds:
        ok = True
        res = {}
        for arm in ("ours-new", "vllm"):
            j, why = valid_round(root, arm, n)
            if j is None or why:
                ok = False; notes.append(f"round {n} {arm}: {why if j is not None else 'missing'}")
            res[arm] = j
        if ok: valid.append((n, res))
    if count_only:
        print(len(valid)); return 0
    sizes = [r["target_prompt_tokens"] for r in valid[0][1]["vllm"]["rows"]] if valid else []
    print(f"valid rounds: {[n for n, _ in valid]} of {rounds}")
    for x in notes: print("  invalid:", x)
    if not valid and any("graph" in x for x in notes):
        print("A/B FAILED: ours-new did not replay CUDA graphs (0 replays, no graph metrics, or eager/profiled decode steps): the numbers would be Phase 1a numbers"); return 1
    verdicts = {}
    for s in sizes:
        def med(arm, res):
            row = next(r for r in res[arm]["rows"] if r["target_prompt_tokens"] == s)
            return row["decode_tok_s"], row.get("decode_tok_s_reps")
        d, line = [], []
        for n, res in valid:
            (a, ar), (b, br) = med("ours-new", res), med("vllm", res)
            if a is None or b is None: continue
            d.append(a - b)
            line.append(f"r{n}: ours {a:.2f} vllm {b:.2f} d {a-b:+.2f}")
        old = []
        for n, _ in valid:
            j, _w = valid_round(root, "ours-old", n)
            if j:
                row = next((r for r in j["rows"] if r["target_prompt_tokens"] == s), None)
                if row and row["decode_tok_s"]: old.append(row["decode_tok_s"])
        if len(d) < 3: v = "INCONCLUSIVE"
        elif all(x >= 0 for x in d): v = "PASS"
        elif statistics.median(d) >= 0: v = "MARGINAL"
        else: v = "FAIL"
        verdicts[s] = v
        rng = f"d min/median/max {min(d):+.2f}/{statistics.median(d):+.2f}/{max(d):+.2f}" if d else "no data"
        print(f"size {s:>5}: {v:<12} {rng}; ours-old rounds {['%.1f' % x for x in old]}")
        for l in line: print("    " + l)
    vs = list(verdicts.values())
    if vs and all(v == "PASS" for v in vs): print("PHASE 1 SPEED GATE: PASS"); return 0
    if any(v == "FAIL" for v in vs): print("PHASE 1 SPEED GATE: FAIL"); return 1
    print("PHASE 1 SPEED GATE: NOT DECIDED (marginal or inconclusive sizes; report the numbers above)"); return 3

sys.exit(main())
PY
chmod +x "$WORK/verdict.py"
}

stage_verdict() {
  log "== verdict"
  write_verdict_py
  local rc=0
  if [[ $DRY_RUN == 1 ]]; then log "DRY_RUN: no bench JSON to judge"; STATUS=0; return 0; fi
  python3 "$WORK/verdict.py" "$WORK/ab" | tee "$WORK/verdict.txt" || rc=$?
  (( ACC_FAILED == 0 )) || { log "accuracy stages failed: $(tr '\n' ' ' < "$WORK/accuracy.failures")"; rc=1; }
  STATUS=$rc
}

# ---------------------------------------------------------------------------------------------------------------------
main() {
  log "phase1b acceptance: stages=$STAGES dry_run=$DRY_RUN host=$HOST pre-change=$PRE_CHANGE work=$WORK"
  write_verdict_py
  : > "$DRYLOG"
  VLLM_CONTAINER=${VLLM_CONTAINER:-}; S1_35B=${S1_35B:-}
  if [[ $DRY_RUN == 1 ]]; then VLLM_CONTAINER=${VLLM_CONTAINER:-dry-run-vllm}; S1_35B=${S1_35B:-/models/35b}; fi
  if have_stage preflight; then maybe_fail preflight; stage_preflight; fi
  load_stamps
  [[ -n "$VLLM_CONTAINER" && -n "$S1_35B" ]] || die "VLLM_CONTAINER and S1_35B must be set"
  have_stage build   && { maybe_fail build;   stage_build; }
  if have_stage lease || have_stage kernels || have_stage accuracy || have_stage graphs || have_stage ab; then
    maybe_fail lease; stage_lease
  fi
  have_stage kernels  && { maybe_fail kernels;  stage_kernels; }
  have_stage accuracy && { maybe_fail accuracy; stage_accuracy; }
  have_stage graphs   && { maybe_fail graphs;   stage_graphs; }
  have_stage ab       && { maybe_fail ab;       stage_ab; }
  have_stage verdict  && { stage_verdict; }
  return 0
}
main "$@"
