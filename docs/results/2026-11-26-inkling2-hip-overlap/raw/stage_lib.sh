# Helpers of lane ink-hip-overlap's window stages (sourced ON tatooine): srv starts a server, md5 runs `generate`.
REPO=$HOME/intarsiaml-ink-hip-bpf
S=$REPO/.scratch
M=/models/Inkling-2-Mini
now() { date '+%F %T'; }
T0=$(date +%s); CAPT=${BG_CAP:-245}
left() { echo $(( CAPT - ($(date +%s) - T0) )); }
cname() { [ "$1" = 1 ] && echo ink-hip-bpf-run2 || echo ink-hip-bpf-run; }
srv() {  # card name binary serve-args...: start on `card`, port 8133, wait until up (at most WAITS s); env from SENV
  local card=$1 name=$2 b=$3; shift 3
  local C; C=$(cname "$card")
  docker rm -f "$C" >/dev/null 2>&1
  HIP_VISIBLE_DEVICES=$card INK_CONTAINER=$C nohup bash "$S/run.sh" bash -c "HOME=/work/.scratch/home exec env $SENV \
    /work/.scratch/bin/intarsia-$b serve -m $M --backend hip --host 0.0.0.0 --port 8133 --model-id ink $*" > "$L/$name.log" 2>&1 < /dev/null &
  UP=0; IP=""
  local i
  for i in $(seq 1 "${WAITS:-150}"); do
    IP=$(docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' "$C" 2>/dev/null)
    if [ -n "$IP" ] && curl -sf -m 2 "http://$IP:8133/health" > /dev/null 2>&1; then UP=$i; break; fi
    sleep 1
  done
  echo "[$name] ($b, card $card) up after ${UP}s (0 = not up) $(now): $(grep -aE 'admission on plan|context|overlap|continuous batching|decode lane' "$L/$name.log" | tr '\n' ' ' | cut -c1-600)"
  if [ "$UP" != 0 ]; then
    curl -sf -m 5 "http://$IP:8133/metrics" > "$L/$name-metrics.txt" 2>/dev/null
    echo "[$name] $(grep -E '^intarsia_kv_tokens_capacity ' "$L/$name-metrics.txt") VRAM card$card $(( $(cat /sys/class/drm/card$card/device/mem_info_vram_used) / 1048576 )) MiB"
  fi
}
stop() { docker rm -f "$(cname "$1")" >/dev/null 2>&1; }
md5() {  # card binary prompt-name tokens spec-args: greedy `generate`, the md5 of its text (W3's recipe, stage_gate.sh of ink-hip-bpf)
  local card=$1 b=$2 p=$3 n=$4 sp=$5
  local PR; PR="$(cat "$REPO/scripts/cmx/ids-prompts/$p-0.txt")"
  local tag=$b-$p-$n-${sp:-dspark}; tag=${tag// /}
  local f=$L/md5-$tag
  HIP_VISIBLE_DEVICES=$card INK_CONTAINER=$(cname "$card") timeout 60 bash "$S/run.sh" /work/.scratch/bin/intarsia-$b generate -m $M --backend hip \
    --max-seq 8192 $sp --ignore-eos --profile-every 0 --max-tokens "$n" -p "$PR" > "$f.out" 2> "$f.err"
  local rc=$?
  local h; h=$(sed -e 's/finish=length.*//' -e '/steps=/d' -e '/tok\/s/d' "$f.out" | md5sum | cut -c1-32)
  echo "MD5 card$card $b $p $n ${sp:-dspark} $h rc $rc | $(cat "$f.out" "$f.err" | grep -aoE 'prefill [0-9]+ ms \([0-9.]+ tok/s\)' | head -1) | $(cat "$f.out" "$f.err" | grep -aoE 'decode [0-9]+ ms \([0-9.]+ tok/s\)' | head -1) [$(now)]"
}
md5s() {  # card binary spec-args: short / 4k x 64 / 256
  local p n
  for p in short 4k; do for n in 64 256; do
    (( $(left) < 20 )) && { echo "MD5 card$1 $2 $p $n SKIP ($(left) s left)"; continue; }
    md5 "$1" "$2" $p $n "$3"
  done; done
}
tiny() {  # card binary test...: hip tiny suites on the binary's test builds (the tree on tatooine is that build's)
  local card=$1 b=$2; shift 2
  local t
  for t in "$@"; do
    (( $(left) < 30 )) && { echo "[tiny] $t SKIP ($(left) s left)"; continue; }
    HIP_VISIBLE_DEVICES=$card INK_CONTAINER=$(cname "$card") timeout 45 bash "$S/run.sh" env INTARSIA_REQUIRE_GPU=1 INTARSIA_INKLING2_BACKEND=hip \
      cargo test --release -j8 -p intarsia-models --no-fail-fast --test "$t" -- --nocapture --test-threads=1 > "$L/tiny-$t.log" 2>&1
    echo "[tiny] $t ($b tree) exit $? $(now)"
    grep -aE "^test .*(ok|FAILED)$|test result|panicked|overlap, pace" "$L/tiny-$t.log" | cut -c1-220 | head -24 | sed "s/^/[tiny $t] /"
  done
}
admit() {  # card binary op...: `kernels test --op`
  local card=$1 b=$2; shift 2
  local op
  for op in "$@"; do
    (( $(left) < 20 )) && { echo "ADMIT $op SKIP ($(left) s left)"; continue; }
    echo "ADMIT $op ($b) [$(now)]"
    HIP_VISIBLE_DEVICES=$card INK_CONTAINER=$(cname "$card") timeout $(( $(left) - 5 )) bash "$S/run.sh" /work/.scratch/bin/intarsia-$b kernels test --backend hip --op $op 2>&1 \
      | grep -aE "ADMITTED|REJECT|FAIL|NO CASES|Error|panicked" | head -8 | sed "s/^/ADMIT $op: /"
  done
}
g2() {  # card binary short|long: G2 (accuracy --mode prefill vs the oracle W4A16 goldens; stage_g2.sh of ink-hip-bpf)
  local card=$1 b=$2 g=$3
  (( $(left) < 30 )) && { echo "G2 $g $b SKIP ($(left) s left)"; return; }
  echo "G2 $g $b [$(now)] $(HIP_VISIBLE_DEVICES=$card INK_CONTAINER=$(cname "$card") timeout $(( $(left) - 5 )) bash "$S/run.sh" /work/.scratch/bin/intarsia-$b accuracy -m $M \
    --backend hip --max-seq 4096 --mode prefill --golden golden/oracle/inkling2-mini-oracle-w4a16-$g-top20.safetensors 2>&1 \
    | grep -aE "prefill floor|floor verdict|Error|panicked" | cut -c1-300 | tr '\n' ' ')"
}
bw() {  # name mode...: serve_bitwise.py against the server srv started (IP), one json per mode in $L
  local name=$1 m ms; shift
  for m in "$@"; do
    (( $(left) < 15 )) && { echo "[$name] bitwise $m SKIP ($(left) s left)"; continue; }
    ms=${m//:/_}
    timeout $(( $(left) - 3 )) python3 "$S/serve_bitwise.py" "http://$IP:8133/v1" "$m" "$L/$name-bitwise-$ms.json" > "$L/$name-bitwise-$ms.out" 2>&1
    echo "[$name] bitwise $m exit $? $(now): $(head -n 1 "$L/$name-bitwise-$ms.out")"
  done
}
