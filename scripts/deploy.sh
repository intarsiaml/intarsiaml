#!/usr/bin/env bash
# One-command host-to-host deploy: sync a pinned git ref to a box, build it with the right features in the right
# toolchain, smoke-test it, and print a reproducibility stamp. Run it from endor (or any checkout), INSIDE a fleetq slot:
#
#   fleetq run --host spark1 --mem 20 --label "deploy main to spark1" -- scripts/deploy.sh spark1
#   fleetq run --host spark1 --mem 20 --label "..." -- scripts/deploy.sh spark1 --backend cuda --ref 7536a5e6
#   fleetq run --host tatooine --takes-down radiance ... -- scripts/deploy.sh tatooine --backend hip     # GPU needs the slot
#   fleetq run --host endor --mem 8 --label "..." -- scripts/deploy.sh local --backend metal
#   scripts/deploy.sh spark1 --dry-run              # print what would run, touch nothing (needs no slot)
#   scripts/deploy.sh spark1 --ref v1 -- '$INTARSIA_BIN' bench speed ...    # run a command in the deployed environment
#
# usage: deploy.sh <host> [--backend cuda|hip|metal|auto] [--ref <git ref>] [--tree <name>] [--smoke quick|none]
#                  [--env-file <file>] [--out <dir>] [--no-build] [--dry-run] [-- <cmd...>]
#   host         ssh name (spark1, spark2, tatooine, ...), or `local` / this machine's short hostname to run here.
#   --backend    default auto: probes the host (nvcc or nvidia-smi -> cuda, Darwin -> metal, else /dev/kfd -> hip).
#   --ref        default HEAD. Resolved to a full sha and exported with `git archive`, so only COMMITTED content is deployed
#                (uncommitted changes are reported and ignored: a stamp must name a sha that exists).
#   --tree       directory under ~/intarsia-deploy on the host (default: the backend name). Its target/ dir is kept between
#                deploys so builds are incremental.
#   --smoke      quick (default): `intarsia doctor`, then `intarsia kernels test --quick` (the whole quick sweep, about a
#                second; override the args with SMOKE_KERNELS_ARGS). none: build and stamp only.
#   --env-file   a file of `export INTARSIA_X=...` lines, copied to the tree as DEPLOY_ENV.sh, sourced before the smoke and
#                before <cmd>, and recorded in the stamp.
#   <cmd>        run in the tree with INTARSIA_BIN set (HIP: inside the pinned dev container, binary at /target/release/
#                intarsia) and INTARSIA_REPLAY_OUT set to a fresh directory (relative to the tree).
#
# What runs where: CUDA = the host toolchain on spark1 (rustc and nvcc are stamped); HIP = deploy/tatooine/dev.sh, the ROCm
# image pinned by digest in deploy/tatooine/Dockerfile.hip-dev; Metal = the host's Xcode Metal toolchain. Every build is
# `cargo build --release --locked` through scripts/repro-build.sh (paths remapped, SOURCE_DATE_EPOCH = commit time), so the
# same sha gives the same binary bytes in any tree path: Cargo.lock decides the dependency versions, rust-toolchain.toml the rustc.
# Output: the stamp is printed and written to <tree>/DEPLOY_STAMP.json on the host (scripts/bench-bundle.sh packs it).
# Needs rsync, git and jq locally, bash on the host.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
die() { echo "deploy: $*" >&2; exit 2; }

HOST=""; BACKEND=auto; REF=HEAD; TREE=""; SMOKE=quick; ENVFILE=""; BUILD=1; DRY=0; OUT=""; CMD=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --backend) BACKEND=${2:?}; shift 2 ;;
    --ref) REF=${2:?}; shift 2 ;;
    --tree) TREE=${2:?}; shift 2 ;;
    --smoke) SMOKE=${2:?}; shift 2 ;;
    --env-file) ENVFILE=${2:?}; shift 2 ;;
    --out) OUT=${2:?}; shift 2 ;;
    --no-build) BUILD=0; shift ;;
    --dry-run) DRY=1; shift ;;
    -h|--help) sed -n '2,/^set -euo/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//'; exit 0 ;;
    --) shift; CMD=("$@"); break ;;
    -*) die "unknown option $1" ;;
    *) if [[ -z $HOST ]]; then HOST=$1; else die "unexpected argument $1"; fi; shift ;;
  esac
done
[[ -n $HOST ]] || die "usage: deploy.sh <host> [--backend ..] [--ref ..] (see the header of this file)"
[[ $SMOKE == quick || $SMOKE == none ]] || die "--smoke is quick or none"
[[ -z $ENVFILE || -f $ENVFILE ]] || die "env file $ENVFILE not found"
if [[ $DRY == 0 ]]; then
  : "${FLEETQ_JOB:?run under fleetq run (see the header of this file); --dry-run needs no slot}"
fi
FLEETQ_JOB=${FLEETQ_JOB:-DRYRUN}
command -v jq >/dev/null || die "jq is needed locally"

LOCAL=0
if [[ $HOST == local || $HOST == localhost || $HOST == "$(hostname -s)" ]]; then LOCAL=1; fi

# ---- pin the ref ------------------------------------------------------------------------------------------------------
SHA=$(git -C "$REPO" rev-parse --verify "$REF^{commit}") || die "cannot resolve ref $REF"
TREEHASH=$(git -C "$REPO" rev-parse "$SHA^{tree}")
SUBJECT=$(git -C "$REPO" log -1 --format=%s "$SHA")
if [[ -n $(git -C "$REPO" status --porcelain --untracked-files=no) ]]; then
  echo "deploy: note: the working tree has uncommitted changes; they are NOT deployed (only $SHA is)" >&2
fi
if [[ -z $(git -C "$REPO" branch -r --contains "$SHA" 2>/dev/null | head -1) ]]; then
  echo "deploy: warning: $SHA is on no remote branch; another host cannot replay this stamp until it is pushed" >&2
fi

# ---- run a script (stdin) on the host, or here --------------------------------------------------------------------------
HDR='set -euo pipefail
export PATH="$HOME/.cargo/bin:/usr/local/cuda/bin:/opt/homebrew/bin:$PATH" FLEETQ_JOB='"$FLEETQ_JOB"
rcap() { # run the script on stdin there; its stdout is ours
  local body; body="$(cat)"
  if [[ $LOCAL == 1 ]]; then printf '%s\n%s\n' "$HDR" "$body" | bash -s
  else printf '%s\n%s\n' "$HDR" "$body" | ssh -o BatchMode=yes "$HOST" bash -s; fi
}
rrun() { # rrun <label> (script on stdin); --dry-run prints the script instead
  if [[ $DRY == 1 ]]; then printf '\n--- [%s] %s\n' "$HOST" "$1"; cat; else rcap; fi
}

# ---- backend ----------------------------------------------------------------------------------------------------------
if [[ $BACKEND == auto ]]; then
  if [[ $DRY == 1 ]]; then BACKEND=cuda; echo "deploy: dry run, assuming --backend cuda" >&2
  else
    BACKEND=$(rcap <<'EOF'
if [ "$(uname -s)" = Darwin ]; then echo metal
elif command -v nvcc >/dev/null || command -v nvidia-smi >/dev/null; then echo cuda
elif [ -e /dev/kfd ]; then echo hip
else echo unknown; fi
EOF
    )
    [[ $BACKEND != unknown ]] || die "cannot detect the backend on $HOST; pass --backend"
  fi
fi
case "$BACKEND" in cuda|hip|metal) ;; *) die "--backend is cuda, hip, metal or auto" ;; esac
TREE=${TREE:-$BACKEND}
T="\$HOME/intarsia-deploy/$TREE"                 # remote tree; $HOME is expanded on the host
DEV="$T/deploy/tatooine/dev.sh"
IMAGE="intarsia-hip-dev:rocm7.2.4-rust1.90.0"    # keep equal to INTARSIA_IMAGE's default in deploy/tatooine/dev.sh
case "$BACKEND" in   # PREFIX runs a command in the build environment with the tree as cwd
  hip) BIN=/target/release/intarsia; PREFIX="INTARSIA_REPO=$T $DEV" ;;
  *)   BIN=target/release/intarsia;  PREFIX="" ;;
esac
STAMP_TEXT="head=${SHA:0:7} tree=${TREEHASH:0:12} backend=$BACKEND"   # reaches the binary as INTARSIA_BUILD_STAMP
echo "deploy: $HOST ($BACKEND) <- $SHA ($SUBJECT)"

# ---- sync -------------------------------------------------------------------------------------------------------------
EXPORT=$(mktemp -d); trap 'rm -rf "$EXPORT"' EXIT
git -C "$REPO" archive --format=tar "$SHA" | tar -x -C "$EXPORT"
# -c without -t: unchanged files keep their mtime, changed files get "now". `git archive` stamps every file with the commit
# time, so with -t a deploy of an older ref would look older than the last build and cargo would reuse stale objects.
RSYNC=(rsync -rlpc --delete --itemize-changes --exclude /target --exclude '/DEPLOY_*' --exclude /replay-out)
if [[ $DRY == 1 ]]; then
  echo "--- sync: ${RSYNC[*]} $EXPORT/ $HOST:intarsia-deploy/$TREE/"
else
  if [[ $LOCAL == 1 ]]; then mkdir -p "$HOME/intarsia-deploy/$TREE"; DEST="$HOME/intarsia-deploy/$TREE/"
  else ssh -o BatchMode=yes "$HOST" "mkdir -p ~/intarsia-deploy/$TREE"; DEST="$HOST:intarsia-deploy/$TREE/"; fi
  N=$("${RSYNC[@]}" "$EXPORT/" "$DEST" | grep -cE '^(>f|\*deleting|cd|cL)' || true)
  echo "deploy: synced ($N entries changed)"
fi
# Results of a previous deploy must not leak into this stamp.
rrun "clean previous deploy files" <<EOF
mkdir -p $T && cd $T
rm -f DEPLOY_ENV.sh DEPLOY_DOCTOR.json DEPLOY_SMOKE.log DEPLOY_STAMP.json
EOF
put() { # put <local file> <name in the tree>
  if [[ $DRY == 1 ]]; then echo "--- copy $1 -> $TREE/$2"
  elif [[ $LOCAL == 1 ]]; then cp "$1" "$HOME/intarsia-deploy/$TREE/$2"
  else rsync -a "$1" "$HOST:intarsia-deploy/$TREE/$2"; fi
}
put "$REPO/scripts/deploy-step.sh" DEPLOY_STEP.sh
if [[ -n $ENVFILE ]]; then put "$ENVFILE" DEPLOY_ENV.sh; fi
istep() { # istep <step> <args...>: run a step in the build environment (the container for hip), cwd = the tree
  local q; printf -v q '%q ' "$@"
  rrun "step $1" <<EOF
cd $T
$PREFIX bash DEPLOY_STEP.sh $q
EOF
}
istep_cap() {
  local q; printf -v q '%q ' "$@"
  rcap <<EOF
cd $T
$PREFIX bash DEPLOY_STEP.sh $q
EOF
}

# ---- build ------------------------------------------------------------------------------------------------------------
if [[ $BUILD == 1 ]]; then
  if [[ $BACKEND == hip ]]; then
    rrun "hip image (built from the pinned Dockerfile when missing)" <<EOF
if ! docker image inspect $IMAGE >/dev/null 2>&1; then
  echo "deploy: building $IMAGE from deploy/tatooine/Dockerfile.hip-dev"
  docker build -t $IMAGE -f $T/deploy/tatooine/Dockerfile.hip-dev $T/deploy/tatooine
fi
EOF
  fi
  istep build "$BACKEND" "${CARGO_BUILD_JOBS:-6}" "$STAMP_TEXT" "$(git -C "$REPO" log -1 --format=%ct "$SHA")"
fi

# ---- smoke ------------------------------------------------------------------------------------------------------------
SMOKE_STATUS=skipped
if [[ $SMOKE == quick ]]; then
  # shellcheck disable=SC2086  # the kernels-test args are a word list on purpose
  istep smoke "$BACKEND" "$BIN" ${SMOKE_KERNELS_ARGS:---quick}
  SMOKE_STATUS=pass    # a failing step aborts the script (set -e) before this line
fi

# ---- stamp ------------------------------------------------------------------------------------------------------------
if [[ $DRY == 1 ]]; then istep probe "$BIN"; echo; echo "(dry run: no stamp written)"; exit 0; fi
KV=$(istep_cap probe "$BIN")
IMGINFO=""
if [[ $BACKEND == hip ]]; then
  IMGINFO=$(rcap <<EOF
echo "\$(docker image inspect -f '{{.Id}}' $IMAGE) base=\$(grep -m1 ^FROM $T/deploy/tatooine/Dockerfile.hip-dev | cut -d' ' -f2)"
EOF
  )
fi
DOCTOR=$(rcap <<EOF
cd $T
if [ -f DEPLOY_DOCTOR.json ]; then cat DEPLOY_DOCTOR.json; else echo '{}'; fi
EOF
)
getkv() { printf '%s\n' "$KV" | sed -n "s/^$1=//p" | head -1; }
STAMP=$(jq -n --arg sha "$SHA" --arg ref "$REF" --arg subj "$SUBJECT" --arg tree "$TREEHASH" --arg backend "$BACKEND" \
  --arg host "$HOST" --arg smoke "$SMOKE_STATUS" --arg env "${ENVFILE:+$(cat "$ENVFILE")}" \
  --arg rustc "$(getkv rustc)" --arg cargo "$(getkv cargo)" --arg lock "$(getkv lock_sha256)" \
  --arg tcfile "$(getkv toolchain_file)" --arg bin "$BIN" --arg binsha "$(getkv binary_sha256)" --arg binbytes "$(getkv binary_bytes)" \
  --arg nvcc "$(getkv nvcc)" --arg hipcc "$(getkv hipcc)" --arg rocm "$(getkv rocm)" --arg image "$IMGINFO" \
  --arg created "$(date -u +%FT%TZ)" --arg treedir "intarsia-deploy/$TREE" \
  --argjson doctor "$(printf '%s' "$DOCTOR" | jq -c . 2>/dev/null || echo '{}')" '
  def nn: if . == "" then null else . end;
  {schema: 1, tool: "scripts/deploy.sh", created: $created,
   git: {sha: $sha, ref: $ref, subject: $subj, tree: $tree},
   lock_sha256: $lock, rust_toolchain_file_sha256_16: ($tcfile | nn),
   backend: $backend, features: $backend, deploy_host: $host, tree_dir: $treedir,
   toolchain: {rustc: $rustc, cargo: $cargo, nvcc: ($nvcc | nn), hipcc: ($hipcc | nn), rocm: ($rocm | nn), container: ($image | nn)},
   gpu: ($doctor.gpu // {}),
   device: ($doctor.backend.device // null),
   host: ($doctor.host // {}),
   binary: {path: $bin, sha256: $binsha, bytes: ($binbytes | tonumber? // null), intarsia_version: ($doctor.intarsia_version // null)},
   env_file: ($env | nn),
   smoke: $smoke}')
if [[ $LOCAL == 1 ]]; then printf '%s\n' "$STAMP" > "$HOME/intarsia-deploy/$TREE/DEPLOY_STAMP.json"
else printf '%s\n' "$STAMP" | ssh -o BatchMode=yes "$HOST" "cat > ~/intarsia-deploy/$TREE/DEPLOY_STAMP.json"; fi

echo
echo "=== reproducibility stamp ($HOST:~/intarsia-deploy/$TREE/DEPLOY_STAMP.json) ==="
printf '%s' "$STAMP" | jq -r '
  "git        \(.git.sha)  (\(.git.ref); tree \(.git.tree[0:12]))",
  "subject    \(.git.subject)",
  "backend    \(.backend)",
  "rustc      \(.toolchain.rustc)",
  "cargo      \(.toolchain.cargo)",
  "cargo.lock \(.lock_sha256[0:16])  (built with --locked)",
  (if .toolchain.nvcc then "cuda       \(.toolchain.nvcc)" else empty end),
  (if .toolchain.hipcc then "hipcc      \(.toolchain.hipcc)" else empty end),
  (if .toolchain.rocm then "rocm       \(.toolchain.rocm)" else empty end),
  (if .toolchain.container then "container  \(.toolchain.container)" else empty end),
  (.gpu.nvidia_smi // {}) as $s |
  "gpu        \(.device.name // "n/a")  compute \($s.compute_cap // "-")",
  "driver     \($s.driver_version // .gpu.rocm_version // "n/a")  (cuda toolkit \(.gpu.cuda_toolkit_version // "-"))",
  "host       \(.host.hostname // "n/a")  \(.host.os // "")  \(.host.cpu_model // "")",
  "binary     \(.binary.path)  sha256 \(.binary.sha256[0:16])  \(.binary.bytes) bytes  \(if .binary.intarsia_version then "v" + .binary.intarsia_version else "" end)",
  "smoke      \(.smoke)"'

# ---- optional command -------------------------------------------------------------------------------------------------
if [[ ${#CMD[@]} -gt 0 ]]; then
  OUT=${OUT:-replay-out/$(date -u +%Y%m%dT%H%M%SZ)}
  echo "deploy: running in $HOST:~/intarsia-deploy/$TREE (INTARSIA_BIN=$BIN INTARSIA_REPLAY_OUT=$OUT)"
  istep run "$BIN" "$OUT" "${CMD[@]}"
  echo "deploy: outputs in $HOST:intarsia-deploy/$TREE/$OUT"
fi
