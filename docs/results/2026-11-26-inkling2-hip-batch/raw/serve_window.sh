#!/usr/bin/env bash
# Lane ink-hip-batch, ON THE MAC, as ONE fleetq command (radiance goes down inside it; window.sh on tatooine restores it on every exit):
#   fleetq run --host tatooine-r9700 --takes-down radiance --quiet gpu,cpu --expect 10 --label "..." -- \
#     env VISITS=<file> OUT=<dir> CAP=<s> bash .scratch/serve_window.sh
# Holds the window on tatooine (window.sh + hold.sh, which also samples VRAM), then per line of VISITS
#   name;bin;serve args;localeval speed args (or "none", or "bitwise:solo|conc[,localeval speed args]")
# starts `intarsia serve` (serve_ctl.sh; a forwarder on tatooine:18131 reaches it), snapshots /metrics, runs localeval speed (inside this fleetq
# job: it inherits FLEETQ_JOB), snapshots /metrics again, stops the server (VRAM peak of the visit). The window ends when the
# visits are done (stop file) or at CAP seconds of radiance-down time (window.sh's cap).
set -uo pipefail
VISITS=${VISITS:?set VISITS}; OUT=${OUT:?set OUT}; CAP=${CAP:-290}
T=tatooine; RR='~/intarsiaml-ink-hip-batch'; LE=/opt/code/deepeval; PORT=18131
SD=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$OUT"
LOG=$OUT/driver.log
say() { echo "[$(date +%T)] $*" | tee -a "$LOG"; }
to() { local s=$1; shift; perl -e 'alarm shift; exec @ARGV or die "exec: $!"' "$s" "$@"; }
t0=$(date +%s)
left() { echo $(( CAP - ($(date +%s) - t0) )); }
mt() { perl -MTime::HiRes=time -e 'printf "%.3f\n", time'; }
ssh -o BatchMode=yes "$T" "CAP=$CAP bash $RR/.scratch/window.sh $RR/.scratch/hold.sh" < /dev/null > "$OUT/win.out" 2>&1 &
WP=$!
TUN=
finish() {
  [ -n "$TUN" ] && kill "$TUN" 2>/dev/null
  ssh -o BatchMode=yes "$T" "bash $RR/.scratch/serve_ctl.sh stop last > /dev/null 2>&1; touch $RR/logs/hold-stop" < /dev/null
  wait "$WP" 2>/dev/null
  say "window ended"; grep -aE '^\[window\]' "$OUT/win.out" | tee -a "$LOG"
}
trap finish EXIT
trap 'say "TERM/INT"; exit 143' TERM INT
for i in $(seq 1 60); do grep -q "radiance stopped" "$OUT/win.out" 2>/dev/null && break; sleep 1; done
grep -q "radiance stopped" "$OUT/win.out" || { say "window did not start: $(cat "$OUT/win.out")"; exit 3; }
t0=$(date +%s)
say "radiance down"
while IFS=';' read -r name bin sargs largs; do
  [ -z "$name" ] && continue
  case "$name" in \#*) continue ;; esac
  if (( $(left) < 45 )); then say "SKIP $name: $(left) s left"; continue; fi
  say "$name: start ($bin; $sargs)"
  if ! ssh -o BatchMode=yes "$T" "bash $RR/.scratch/serve_ctl.sh start $bin $name $sargs" < /dev/null > "$OUT/$name-start.out" 2>&1; then
    say "$name: server did not come up: $(tail -n 6 "$OUT/$name-start.out" | tr '\n' ' ')"
    ssh -o BatchMode=yes "$T" "bash $RR/.scratch/serve_ctl.sh stop $name" < /dev/null >> "$OUT/$name-stop.out" 2>&1
    continue
  fi
  ip=$(awk '/^IP /{print $2}' "$OUT/$name-start.out")
  say "$name: up ($(cat "$OUT/$name-start.out" | tr '\n' ' '); $(left) s left)"
  ssh -o BatchMode=yes "$T" "bash $RR/.scratch/serve_ctl.sh metrics" < /dev/null > "$OUT/$name-metrics0.txt" 2>&1
  ssh -o BatchMode=yes "$T" "bash $RR/.scratch/serve_ctl.sh poll-start $name" < /dev/null > "$OUT/$name-poll.out" 2>&1
  # the two clocks (Mac before / tatooine / Mac after) for the poll log's segmentation (itl_seg.py)
  { mt; ssh -o BatchMode=yes "$T" "bash $RR/.scratch/serve_ctl.sh clock" < /dev/null; mt; } > "$OUT/$name-clock.txt" 2>&1
  case "$largs" in
    bitwise:*)  # bitwise:<solo|conc>[,<localeval speed args>]: the serve-level batched == solo check (serve_bitwise.py) first
      bm=${largs#bitwise:}; largs=none
      case "$bm" in *,*) largs=${bm#*,}; bm=${bm%%,*} ;; esac
      mt > "$OUT/$name-bw.t0"
      to 200 python3 "$SD/serve_bitwise.py" "http://$T:$PORT/v1" "$bm" "$OUT/$name-bitwise.json" < /dev/null > "$OUT/$name-bitwise.out" 2>&1
      mt > "$OUT/$name-bw.t1"
      say "$name: bitwise $bm exit $? ($(left) s left): $(tail -n 1 "$OUT/$name-bitwise.out")" ;;
  esac
  if [ "$largs" != none ]; then
    for r in $(seq 1 "${LEREPS:-1}"); do
      date +%s.%N > "$OUT/$name-r$r.t0"
      if (( $(left) < 40 )); then say "$name: SKIP localeval rep $r: $(left) s left"; break; fi
      cap=$(( $(left) - 25 )); (( cap > 230 )) && cap=230
      ( cd "$LE" && to "$cap" uv run localeval speed --url "http://$T:$PORT/v1" --model ink $largs --label "ink-hip-batch-$name-r$r" \
          --recipe "intarsia serve -m Inkling-2-Mini --backend hip $sargs (tree $bin)" ) < /dev/null 2>&1 \
          | perl -MTime::HiRes=time -ne '$|=1; printf "%.3f %s", time, $_' > "$OUT/$name-r$r-le.out"
      lrc=${PIPESTATUS[0]}
      date +%s.%N > "$OUT/$name-r$r.t1"
      say "$name r$r: localeval exit $lrc ($(left) s left): $(grep -aE 'streams: aggregate' "$OUT/$name-r$r-le.out" | sed 's/^[0-9.]* //' | tr '\n' ' ')"
      res=$(grep -ao 'saved .*result.json' "$OUT/$name-r$r-le.out" | tail -1 | sed 's/^saved //')
      [ -n "$res" ] && [ -f "$LE/$res" ] && cp "$LE/$res" "$OUT/$name-r$r.result.json"
      [ -n "$res" ] && [ -f "$res" ] && cp "$res" "$OUT/$name-r$r.result.json"
      ssh -o BatchMode=yes "$T" "bash $RR/.scratch/serve_ctl.sh metrics" < /dev/null > "$OUT/$name-metrics-r$r.txt" 2>&1
    done
  fi
  ssh -o BatchMode=yes "$T" "bash $RR/.scratch/serve_ctl.sh metrics" < /dev/null > "$OUT/$name-metrics1.txt" 2>&1
  ssh -o BatchMode=yes "$T" "bash $RR/.scratch/serve_ctl.sh poll-stop" < /dev/null >> "$OUT/$name-poll.out" 2>&1
  ssh -o BatchMode=yes "$T" "bash $RR/.scratch/serve_ctl.sh stop $name" < /dev/null > "$OUT/$name-stop.out" 2>&1
  scp -q -o BatchMode=yes "$T:intarsiaml-ink-hip-batch/logs/serve/$name.poll" "$T:intarsiaml-ink-hip-batch/logs/serve/$name.log" "$OUT/" < /dev/null 2>/dev/null
  say "$name: stopped: $(head -n 1 "$OUT/$name-stop.out")"
done < "$VISITS"
say "visits done ($(left) s left)"
