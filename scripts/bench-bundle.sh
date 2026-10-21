#!/usr/bin/env bash
# Bench bundle: pack what a measurement ran on into ONE tarball, so a second host can replay it.
#
#   scripts/bench-bundle.sh pack --host spark1 --backend cuda --out /tmp/r7.tgz \
#       --cmd '$INTARSIA_BIN bench speed -m /models/Qwen3-8B --backend cuda --out $INTARSIA_REPLAY_OUT/speed.json' \
#       --report docs/results/2026-10-21-cuda-round7/speed.json --report spark1:~/bench-out/localeval \
#       --env INTARSIA_CUDA_FP8_GEMM_LT=1 --note "round 7 A/B, vLLM up"
#   scripts/bench-bundle.sh show /tmp/r7.tgz
#   fleetq run --host spark2 --mem 20 --label "replay r7" -- scripts/bench-bundle.sh replay /tmp/r7.tgz spark2
#
# pack    reads the stamp deploy.sh left in <host>:~/intarsia-deploy/<tree>/ (--tree, default the --backend name) and
#         bundles: the stamp (git sha, rustc, CUDA/ROCm, GPU, driver, binary sha256), the device fingerprint (`intarsia
#         doctor` JSON), the env knobs (DEPLOY_ENV.sh from deploy.sh --env-file, plus every --env K=V), the recipes,
#         Cargo.lock and rust-toolchain.toml as built, the command line to replay (--cmd, a shell string run with
#         $INTARSIA_BIN and $INTARSIA_REPLAY_OUT set), and the reports (--report <local path> or <host>:<path>, files or
#         directories: bench-speed JSON, localeval output, logs). --with-binary adds the CUDA/Metal binary itself.
# show    prints the manifest summary.
# replay  deploys the bundle's git sha on <host> with scripts/deploy.sh (same backend, same env knobs), runs the recorded
#         command there, prints which stamp fields match the original (rustc, CUDA/ROCm, driver, GPU, Cargo.lock, binary
#         hash) and copies the command's outputs to ./replay-<sha7>-<time>/. Run it inside a fleetq slot like deploy.sh.
#         --no-deploy skips the deploy and only runs the command in the existing tree.
# Needs jq, tar, rsync, git locally.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
die() { echo "bench-bundle: $*" >&2; exit 2; }
command -v jq >/dev/null || die "jq is needed locally"
sha() { shasum -a 256 "$1" | cut -d' ' -f1; }
rfetch() { # rfetch <host> <remote path relative to ~> <local dest>  (remote path may be a file or a directory)
  if [[ $1 == local || $1 == localhost || $1 == "$(hostname -s)" ]]; then rsync -a "$HOME/$2" "$3"; else rsync -a "$1:$2" "$3"; fi
}

cmd_pack() {
  local HOST="" BACKEND="" TREE="" OUT="" CMD="" NOTE="" WITHBIN=0 REPORTS=() ENVS=()
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --host) HOST=${2:?}; shift 2 ;;
      --backend) BACKEND=${2:?}; shift 2 ;;
      --tree) TREE=${2:?}; shift 2 ;;
      --out) OUT=${2:?}; shift 2 ;;
      --cmd) CMD=${2:?}; shift 2 ;;
      --note) NOTE=${2:?}; shift 2 ;;
      --report) REPORTS+=("${2:?}"); shift 2 ;;
      --env) [[ ${2:?} =~ ^[A-Za-z_][A-Za-z0-9_]*=.*$ ]] || die "--env wants KEY=VALUE, got $2"; ENVS+=("$2"); shift 2 ;;
      --with-binary) WITHBIN=1; shift ;;
      *) die "pack: unknown argument $1" ;;
    esac
  done
  [[ -n $HOST && -n $OUT ]] || die "pack needs --host and --out"
  : "${FLEETQ_JOB:?run under fleetq run --host <host> --mem 0 (pack reads files from the box)}"
  TREE=${TREE:-$BACKEND}; [[ -n $TREE ]] || die "pack needs --backend or --tree (the deploy tree name)"
  local R="intarsia-deploy/$TREE" W; W=$(mktemp -d); trap 'rm -rf "$W"' RETURN
  mkdir -p "$W/b/reports" "$W/b/recipes"
  rfetch "$HOST" "$R/DEPLOY_STAMP.json" "$W/b/stamp.json" || die "no DEPLOY_STAMP.json in $HOST:~/$R (run scripts/deploy.sh first)"
  rfetch "$HOST" "$R/DEPLOY_DOCTOR.json" "$W/b/doctor.json" 2>/dev/null || true
  rfetch "$HOST" "$R/DEPLOY_SMOKE.log" "$W/b/smoke.log" 2>/dev/null || true
  rfetch "$HOST" "$R/Cargo.lock" "$W/b/Cargo.lock"
  rfetch "$HOST" "$R/rust-toolchain.toml" "$W/b/rust-toolchain.toml" 2>/dev/null || true
  rfetch "$HOST" "$R/recipes/" "$W/b/recipes/"
  : > "$W/b/env.sh"
  rfetch "$HOST" "$R/DEPLOY_ENV.sh" "$W/b/env.sh" 2>/dev/null || true
  local e; for e in "${ENVS[@]+"${ENVS[@]}"}"; do printf 'export %s=%q\n' "${e%%=*}" "${e#*=}" >> "$W/b/env.sh"; done
  printf '%s\n' "$CMD" > "$W/b/replay.cmd"
  local r
  for r in "${REPORTS[@]+"${REPORTS[@]}"}"; do
    if [[ -e $r ]]; then rsync -a "$r" "$W/b/reports/"
    elif [[ $r == *:* ]]; then rfetch "${r%%:*}" "${r#*:}" "$W/b/reports/" || die "cannot fetch report $r"
    else die "report $r not found"; fi
  done
  if [[ $WITHBIN == 1 ]]; then
    if [[ $(jq -r .backend "$W/b/stamp.json") == hip ]]; then echo "bench-bundle: --with-binary: the HIP binary lives in a docker volume, not packed" >&2
    else mkdir -p "$W/b/bin"; rfetch "$HOST" "$R/target/release/intarsia" "$W/b/bin/intarsia"; fi
  fi
  local SHA7 BE; SHA7=$(jq -r '.git.sha[0:7]' "$W/b/stamp.json"); BE=$(jq -r .backend "$W/b/stamp.json")
  local NAME="intarsia-bench-bundle-$SHA7-$BE"
  # the manifest lists every file with its hash; verify with `shasum -a 256 -c` after extracting
  ( cd "$W/b" && find . -type f ! -name MANIFEST.json ! -name SHA256SUMS | sort | while read -r f; do printf '%s  %s\n' "$(sha "$f")" "$f"; done > SHA256SUMS )
  jq -n --slurpfile stamp "$W/b/stamp.json" --arg cmd "$CMD" --arg note "$NOTE" --arg created "$(date -u +%FT%TZ)" \
    --arg by "$(hostname -s)" --arg host "$HOST" --arg env "$(cat "$W/b/env.sh")" \
    --argjson reports "$(cd "$W/b/reports" && find . -type f | sort | jq -R . | jq -sc .)" '
    {schema: 1, kind: "intarsia-bench-bundle", created: $created, packed_by: $by, measured_on: $host,
     git_sha: $stamp[0].git.sha, backend: $stamp[0].backend, binary_sha256: $stamp[0].binary.sha256,
     command: $cmd, note: $note, env_knobs: ($env | split("\n") | map(select(length > 0))),
     reports: $reports, stamp: $stamp[0]}' > "$W/b/MANIFEST.json"
  mv "$W/b" "$W/$NAME"
  mkdir -p "$(dirname "$OUT")"
  tar -czf "$OUT" -C "$W" "$NAME"
  echo "bench-bundle: wrote $OUT ($(wc -c < "$OUT" | tr -d ' ') bytes, $(wc -l < "$W/$NAME/SHA256SUMS" | tr -d ' ') files, git ${SHA7}, $BE)"
}

unpack() { # unpack <bundle> <dir>: extracts and prints the bundle directory
  tar -xzf "$1" -C "$2"; local d; d=$(find "$2" -mindepth 1 -maxdepth 1 -type d | head -1); [[ -f $d/MANIFEST.json ]] || die "$1 is not a bench bundle"
  ( cd "$d" && shasum -a 256 -c SHA256SUMS >/dev/null ) || die "$1: checksum mismatch (SHA256SUMS)"
  printf '%s' "$d"
}

cmd_show() {
  local W; W=$(mktemp -d); trap 'rm -rf "$W"' RETURN
  local d; d=$(unpack "${1:?bundle}" "$W")
  jq -r '"bundle     \(.created) packed on \(.packed_by), measured on \(.measured_on)",
    "git        \(.git_sha)",
    "backend    \(.backend)",
    "rustc      \(.stamp.toolchain.rustc)",
    "cuda       \(.stamp.toolchain.nvcc // "-")   rocm \(.stamp.toolchain.rocm // "-")",
    "gpu        \(.stamp.device.name // "n/a")  driver \(.stamp.gpu.nvidia_smi.driver_version // .stamp.gpu.rocm_version // "n/a")",
    "binary     sha256 \(.binary_sha256[0:16])",
    "env        \(.env_knobs | join("; "))",
    "command    \(.command)",
    "reports    \(.reports | join(" "))",
    "note       \(.note)"' "$d/MANIFEST.json"
}

cmd_replay() {
  local BUNDLE=${1:?bundle}; local HOST=${2:?host}; shift 2
  local DEPLOY=1 TREE="" CMDOVR=""
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --no-deploy) DEPLOY=0; shift ;;
      --tree) TREE=${2:?}; shift 2 ;;
      --cmd) CMDOVR=${2:?}; shift 2 ;;
      *) die "replay: unknown argument $1" ;;
    esac
  done
  local W; W=$(mktemp -d); trap 'rm -rf "$W"' RETURN
  local d; d=$(unpack "$BUNDLE" "$W")
  local SHA BE CMD; SHA=$(jq -r .git_sha "$d/MANIFEST.json"); BE=$(jq -r .backend "$d/MANIFEST.json")
  CMD=${CMDOVR:-$(cat "$d/replay.cmd")}
  TREE=${TREE:-$BE}
  git -C "$REPO" cat-file -e "$SHA^{commit}" 2>/dev/null || git -C "$REPO" fetch -q origin "$SHA" || die "commit $SHA is not available locally or on origin"
  local OUTREL="replay-out/replay-${SHA:0:7}-$(date -u +%Y%m%dT%H%M%SZ)"
  local ARGS=("$HOST" --backend "$BE" --ref "$SHA" --tree "$TREE" --out "$OUTREL")
  [[ -s $d/env.sh ]] && ARGS+=(--env-file "$d/env.sh")
  if [[ $DEPLOY == 1 ]]; then
    if [[ -n $CMD ]]; then "$REPO/scripts/deploy.sh" "${ARGS[@]}" -- "$CMD"; else "$REPO/scripts/deploy.sh" "${ARGS[@]}"; fi
  else
    [[ -n $CMD ]] || die "--no-deploy needs a command"
    "$REPO/scripts/deploy.sh" "${ARGS[@]}" --no-build --smoke none -- "$CMD"
  fi
  # compare the replay host's stamp with the bundle's
  rfetch "$HOST" "intarsia-deploy/$TREE/DEPLOY_STAMP.json" "$W/new.json"
  echo
  echo "=== replay vs original stamp ==="
  jq -rn --slurpfile a "$d/stamp.json" --slurpfile b "$W/new.json" '
    def row($k; $x; $y): "\(if $x == $y then "match " else "DIFFER" end)  \($k): \($x | tostring | .[0:70])\(if $x == $y then "" else "  ->  \($y | tostring | .[0:70])" end)";
    $a[0] as $x | $b[0] as $y |
    row("git sha"; $x.git.sha; $y.git.sha),
    row("Cargo.lock"; $x.lock_sha256[0:16]; $y.lock_sha256[0:16]),
    row("rustc"; $x.toolchain.rustc; $y.toolchain.rustc),
    row("cuda"; $x.toolchain.nvcc; $y.toolchain.nvcc),
    row("rocm"; $x.toolchain.rocm; $y.toolchain.rocm),
    row("container"; $x.toolchain.container; $y.toolchain.container),
    row("gpu"; $x.device.name; $y.device.name),
    row("gpu/driver"; $x.gpu; $y.gpu),
    row("binary sha256"; $x.binary.sha256[0:16]; $y.binary.sha256[0:16])'
  echo "(a binary hash can differ with identical sources when the build tree path or the box differs; the sha, lock and toolchain rows are what pin the build)"
  if [[ -n $CMD ]]; then
    local DEST="./replay-${SHA:0:7}-$(date -u +%Y%m%dT%H%M%S)"; mkdir -p "$DEST"
    rfetch "$HOST" "intarsia-deploy/$TREE/$OUTREL/" "$DEST/" && echo "bench-bundle: replay outputs in $DEST"
  fi
}

sub=${1:-}; [[ $# -gt 0 ]] && shift
case "$sub" in
  pack) cmd_pack "$@" ;;
  show) cmd_show "$@" ;;
  replay) cmd_replay "$@" ;;
  *) sed -n '2,/^set -euo/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//'; [[ $sub == -h || $sub == --help ]] || exit 2 ;;
esac
