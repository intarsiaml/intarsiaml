#!/usr/bin/env bash
# Window stage (lane ink-hip-bpf), inside window.sh: gate 1 (unit) and gate 4 (hip tiny tests). Card 0: inkling2_packed on hip
# (tiny fixture, then the real checkpoint at real shapes: INTARSIA_INK_PACK_REAL). Card 1 at the same time: inkling2_batch and
# inkling2 on hip (tiny). The test binaries are built (build.sh m:<name>); no env knob besides the tests' own.
set -uo pipefail
REPO=$HOME/intarsiaml-ink-hip-bpf
S=$REPO/.scratch
now() { date '+%F %T'; }
T="cargo test --release -j8 -p intarsia-models"
card0() {
  echo "[stage] card0 inkling2_packed (tiny + real) $(now)"
  HIP_VISIBLE_DEVICES=0 timeout ${CAP0:-230} bash "$S/run.sh" env INTARSIA_REQUIRE_GPU=1 INTARSIA_INKLING2_BACKEND=hip \
    INTARSIA_INK_PACK_REAL=/models/Inkling-2-Mini $T --test inkling2_packed -- --nocapture --test-threads=1 2>&1 |
    grep -aE "^test |test result|panicked|assert|moe_split_rows|SKIPPED|error" | sed 's/^/[card0] /' | head -40
  echo "[stage] card0 done $(now)"
}
card1() {
  echo "[stage] card1 inkling2_batch + inkling2 (tiny hip) $(now)"
  HIP_VISIBLE_DEVICES=1 INK_CONTAINER=ink-hip-bpf-run2 timeout ${CAP1:-200} bash "$S/run.sh" env INTARSIA_REQUIRE_GPU=1 \
    INTARSIA_INKLING2_BACKEND=hip $T --test inkling2_batch --test inkling2 -- --test-threads=1 2>&1 |
    grep -aE "^test |test result|panicked|assert|SKIPPED|error" | sed 's/^/[card1] /' | head -60
  # prime the admission cache (HOME=.scratch/home, keyed by binary, not card) for the next window's servers, one at a time
  for cfg in "${PRIME[@]}"; do
    [ -z "$cfg" ] && continue
    i0=$(date +%H%M%S)
    echo "[stage] card1 prime: $cfg $(now)"
    docker rm -f ink-hip-bpf-run2 >/dev/null 2>&1
    envs=""; set -- $cfg; while [[ "${1:-}" =~ ^[A-Z_][A-Z0-9_]*= ]]; do envs="$envs $1"; shift; done
    HIP_VISIBLE_DEVICES=1 INK_CONTAINER=ink-hip-bpf-run2 nohup bash "$S/run.sh" bash -c "HOME=/work/.scratch/home exec env $envs \
      /work/.scratch/bin/intarsia-$BIN serve -m /models/Inkling-2-Mini --backend hip --host 0.0.0.0 --port 8133 --model-id ink $*" \
      > "$REPO/logs/prime-$i0.log" 2>&1 < /dev/null &
    up=0
    for i in $(seq 1 200); do
      ip=$(docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' ink-hip-bpf-run2 2>/dev/null)
      if [ -n "$ip" ] && curl -sf -m 2 "http://$ip:8133/health" > /dev/null 2>&1; then up=$i; break; fi
      sleep 1
    done
    echo "[stage] card1 prime up after ${up}s (0 = not up) $(now)"
    grep -aE 'admission|admitted|packed prefill|continuous batching|Error|error' "$REPO/logs/prime-$i0.log" | head -6 | sed 's/^/[card1] /'
    docker rm -f ink-hip-bpf-run2 >/dev/null 2>&1
  done
  echo "[stage] card1 done $(now)"
}
BIN=${BIN:-b2}
IFS='|' read -r -a PRIME <<< "${PRIME_CFGS:-}"
card0 &
card1 &
wait
echo "[stage] done $(now)"
