#!/usr/bin/env bash
# Lane ink-hip-bpf, ON THE MAC, as ONE fleetq command (radiance goes down inside it; window.sh on tatooine restores it on every exit):
#   fleetq run --host tatooine-r9700 --takes-down radiance --quiet gpu,cpu --expect 10 --label "..." -- \
#     env VISITS=<file> OUT=<dir> CAP=<s> bash .scratch/serve_window.sh
# Holds the window on tatooine (window.sh + hold.sh, which also samples VRAM), then per line of VISITS
#   name;bin;serve args;localeval speed args (or "none", or "bitwise:solo|conc[,localeval speed args]")
# starts `intarsia serve` (serve_ctl.sh; a forwarder on tatooine:18132 reaches it), snapshots /metrics, runs localeval speed (inside this fleetq
# job: it inherits FLEETQ_JOB), snapshots /metrics again, stops the server (VRAM peak of the visit). The window ends when the
# visits are done (stop file) or at CAP seconds of radiance-down time (window.sh's cap).
set -uo pipefail
VISITS=${VISITS:?set VISITS}; OUT=${OUT:?set OUT}; CAP=${CAP:-290}
T=tatooine; RR='~/intarsiaml-ink-hip-bpf'; LE=/opt/code/deepeval; PORT=18132
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
BGP=
finish() {
  [ -n "$TUN" ] && kill "$TUN" 2>/dev/null
  # a background stage on card 1 (BG=) ends by its own cap (BG_CAP); the window ends only after it
  if [ -n "$BGP" ]; then say "waiting for the background stage"; wait "$BGP" 2>/dev/null; say "background stage ended: $(tail -n 1 "$OUT/bg.out")"; fi
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
if [ -n "${BG:-}" ]; then  # BG=<stage on tatooine>: runs beside the visits (card 1), within BG_CAP s (default CAP - 40)
  bgc=${BG_CAP:-$(( CAP - 40 ))}
  ssh -o BatchMode=yes "$T" "BG_CAP=$bgc bash $RR/.scratch/$BG" < /dev/null > "$OUT/bg.out" 2>&1 &
  BGP=$!
  say "background stage $BG started (cap $bgc s)"
fi
if [ -n "${PRE:-}" ]; then  # PRE=<stage on tatooine> (e.g. stage_w10.sh): run first; PRE_CHECK=<script on the Mac> must pass or no visit runs
  say "pre-stage $PRE ($(left) s left)"
  to $(( $(left) - 20 )) ssh -o BatchMode=yes "$T" "bash $RR/.scratch/$PRE" < /dev/null > "$OUT/pre.out" 2>&1
  say "pre-stage exit $? ($(left) s left)"
  if [ -n "${PRE_CHECK:-}" ]; then
    python3 "$SD/$PRE_CHECK" "$OUT/pre.out" > "$OUT/pre-check.out" 2>&1; pc=$?
    say "pre-check: $(tail -n 1 "$OUT/pre-check.out")"
    if [ $pc != 0 ]; then say "pre-check failed: no visits"; VISITS=/dev/null; fi
  fi
fi
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
  # lane ink-hip-bpf: the visit's actions, ' && '-separated: real:<N,..>:<reps>[:<tag>] | bitwise:<mode>[+<mode>..] | le:<localeval args>[@<tag>]
  IFS=$'\x1f' read -r -a ACTS <<< "${largs// && /$'\x1f'}"
  ai=0
  for act in "${ACTS[@]}"; do
  ai=$((ai + 1))
  largs=$act
  case "$largs" in
    wait:*)  # wait:<file on tatooine>:<max s>: wait (at most max s) until the background stage writes the file (e.g. a binary's
             # admission primed on card 1, so the next visit's server finds it cached)
      wf=${largs#wait:}; wmax=${wf##*:}; wf=${wf%:*}; largs=none
      ssh -o BatchMode=yes "$T" "for i in \$(seq 1 $wmax); do [ -e $wf ] && exit 0; sleep 1; done; exit 1" < /dev/null
      say "$name: wait $wf exit $? ($(left) s left)" ;;
    real:*)  # real:<N,N,..>:<reps>: N of the 16 real prompts at once (serve_bitwise.py conc:N:OFF), rep by rep while time lasts;
             # rep 1 plain (its replies compare with the solo server's), later reps with a nonce in front; levels file for itl_seg.py
      spec=${largs#real:}; lv=${spec%%:*}; rest=${spec#*:}; nr=${rest%%:*}; rtag=""; [[ "$rest" == *:* ]] && rtag=${rest#*:}; largs=none
      for r in $(seq 1 "$nr"); do
        for n in ${lv//,/ }; do
          need=$(( n * 2 + 14 ))
          if (( $(left) < need )); then say "$name: SKIP real rep $r N=$n: $(left) s left (need $need)"; break 2; fi
          # rtag: a later visit's reps (offset and nonce from the tag, so no rep repeats another's prompts or reuses its prefixes)
          # REAL_OFF (env): the windows of a speed series start at different offsets; a tagged level sends a nonce in every rep
          # (no prefix shared between the levels of a visit), an untagged level's rep 1 the plain prompts (compared with solo)
          off=$(( (r - 1 + ${REAL_OFF:-0}) % (16 / n) )); nz=""; { (( r > 1 )) || [ -n "$rtag" ]; } && nz=":${rtag}r${r}n${n}"
          mt > "$OUT/$name-real$rtag-r$r-n$n.t0"
          to 120 python3 "$SD/serve_bitwise.py" "http://$T:$PORT/v1" "conc:$n:$off$nz" "$OUT/$name-real$rtag-r$r-n$n.json" < /dev/null > "$OUT/$name-real$rtag-r$r-n$n.out" 2>&1
          rc=$?
          mt > "$OUT/$name-real$rtag-r$r-n$n.t1"
          echo "$r $n $(cat "$OUT/$name-real$rtag-r$r-n$n.t0") $(cat "$OUT/$name-real$rtag-r$r-n$n.t1")" >> "$OUT/$name-levels.txt"
          say "$name: real rep $r N=$n exit $rc ($(left) s left): $(head -n 1 "$OUT/$name-real$rtag-r$r-n$n.out")"
        done
      done ;;
    bitwise:*)  # bitwise:<solo|conc>[,<localeval speed args>]: the serve-level batched == solo check (serve_bitwise.py) first
      bm=${largs#bitwise:}; largs=none
      case "$bm" in *,*) largs=${bm#*,}; bm=${bm%%,*} ;; esac
      # lane ink-hip-bpf: several modes joined by '+' (e.g. solo+reuse-solo), one json each: <name>-bitwise-<mode>.json
      for m in ${bm//+/ }; do
        if (( $(left) < 30 )); then say "$name: SKIP bitwise $m: $(left) s left"; continue; fi
        ms=${m//:/_}
        mt > "$OUT/$name-bw-$ms.t0"
        to $(( $(left) - 10 )) python3 "$SD/serve_bitwise.py" "http://$T:$PORT/v1" "$m" "$OUT/$name-bitwise-$ms.json" < /dev/null > "$OUT/$name-bitwise-$ms.out" 2>&1
        brc=$?
        mt > "$OUT/$name-bw-$ms.t1"
        say "$name: bitwise $m exit $brc ($(left) s left): $(head -n 1 "$OUT/$name-bitwise-$ms.out")"
      done ;;
  esac
  case "$largs" in le:*) largs=${largs#le:} ;; esac
  if [ "$largs" != none ]; then
    ltag=a$ai; case "$largs" in *@*) ltag=${largs##*@}; largs=${largs%@*} ;; esac
    for r in $(seq 1 "${LEREPS:-1}"); do
      date +%s.%N > "$OUT/$name-$ltag-r$r.t0"
      if (( $(left) < 40 )); then say "$name: SKIP localeval rep $r: $(left) s left"; break; fi
      cap=$(( $(left) - 25 )); (( cap > 230 )) && cap=230
      ( cd "$LE" && to "$cap" env PYTHONUNBUFFERED=1 uv run localeval speed --url "http://$T:$PORT/v1" --model ink $largs --label "ink-hip-bpf-$name-$ltag-r$r" \
          --recipe "intarsia serve -m Inkling-2-Mini --backend hip $sargs (tree $bin)" ) < /dev/null 2>&1 \
          | perl -MTime::HiRes=time -ne '$|=1; printf "%.3f %s", time, $_' > "$OUT/$name-$ltag-r$r-le.out"
      lrc=${PIPESTATUS[0]}
      date +%s.%N > "$OUT/$name-$ltag-r$r.t1"
      say "$name $ltag r$r: localeval exit $lrc ($(left) s left): $(grep -aE 'streams: aggregate' "$OUT/$name-$ltag-r$r-le.out" | sed 's/^[0-9.]* //' | tr '\n' ' ')"
      res=$(grep -ao 'saved .*result.json' "$OUT/$name-$ltag-r$r-le.out" | tail -1 | sed 's/^saved //')
      [ -n "$res" ] && [ -f "$LE/$res" ] && cp "$LE/$res" "$OUT/$name-$ltag-r$r.result.json"
      [ -n "$res" ] && [ -f "$res" ] && cp "$res" "$OUT/$name-$ltag-r$r.result.json"
      ssh -o BatchMode=yes "$T" "bash $RR/.scratch/serve_ctl.sh metrics" < /dev/null > "$OUT/$name-$ltag-metrics-r$r.txt" 2>&1
    done
  fi
  done
  ssh -o BatchMode=yes "$T" "bash $RR/.scratch/serve_ctl.sh metrics" < /dev/null > "$OUT/$name-metrics1.txt" 2>&1
  ssh -o BatchMode=yes "$T" "bash $RR/.scratch/serve_ctl.sh poll-stop" < /dev/null >> "$OUT/$name-poll.out" 2>&1
  ssh -o BatchMode=yes "$T" "bash $RR/.scratch/serve_ctl.sh stop $name" < /dev/null > "$OUT/$name-stop.out" 2>&1
  scp -q -o BatchMode=yes "$T:intarsiaml-ink-hip-bpf/logs/serve/$name.poll" "$T:intarsiaml-ink-hip-bpf/logs/serve/$name.log" "$OUT/" < /dev/null 2>/dev/null
  say "$name: stopped: $(head -n 1 "$OUT/$name-stop.out")"
done < "$VISITS"
say "visits done ($(left) s left)"
