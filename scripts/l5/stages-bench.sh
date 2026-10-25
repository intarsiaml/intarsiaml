#!/usr/bin/env bash
# Lane l5, ON tatooine inside scripts/l5/window.sh: the standalone harness at several token counts, on card $G. Env: NAME (snapshot), O (out dir), G (card).
IMG=intarsia-hip-dev:rocm7.2.4-rust1.90.0
T=${TS:-"4096 2048 1024 512"}
hb() { # hb <tag> <args...>: f8bench inside the dev image (ROCm libraries) on card $G
  local tag=$1; shift
  docker run --rm --device /dev/kfd --device /dev/dri --group-add "$(stat -c %g /dev/kfd)" --group-add "$(stat -c %g /dev/dri/renderD128)" \
    --group-add video --ipc host --security-opt seccomp=unconfined --user "$(id -u):$(id -g)" -e HOME=/tmp -e HIP_VISIBLE_DEVICES=$G \
    -v "$HOME/intarsiaml-l5-$NAME":/work -w /work $IMG /work/l5-out/f8bench "$@" > "$O/f8bench-$tag.txt" 2>&1 < /dev/null
  echo "f8bench $tag exit $?"; tail -n 25 "$O/f8bench-$tag.txt"
}
for t in $T; do hb "T$t" "$t" 10 0 1; done
hb "T4096-skew" 4096 10 1 1
hb "T4096-abl" 4096 10 0 0 1
