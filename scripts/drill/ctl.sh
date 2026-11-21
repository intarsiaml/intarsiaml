#!/usr/bin/env bash
# drill controller, runs ON spark2 (copied to ~/intarsiaml-drill/out/ctl.sh). Only ever signals processes whose /proc/<pid>/exe is our binary.
#   ctl.sh start <tag>                 launch `intarsia serve` (shipping 35B config; PS=<prefix-store spec> overrides `on`), time launch -> listening / /health / first token
#   ctl.sh term  <tag> <sup|group>     SIGTERM the supervisor (sup) or supervisor + child (group); time exits, port, memory
#   ctl.sh kill9 <tag> <both|child|sup> SIGKILL both, the child only (the supervisor restarts it: timed to /health + first token), or the supervisor only (child orphaned)
#   ctl.sh snap <label> | stop | cleanup
set -u
D=$HOME/intarsiaml-drill; BIN=$D/bin/intarsia; W=$D/out; PORT=8191; S=$W/state; mkdir -p "$S"
M35=/home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5
PS=${PS:-on}
now() { date +%s.%N; }
el() { awk -v a="$1" -v b="$2" 'BEGIN {printf "%.3f", b-a}'; }
ours() { [ -n "${1:-}" ] && [ "$(readlink /proc/$1/exe 2>/dev/null)" = "$(readlink -f $BIN)" ]; }
alive() { [ -n "${1:-}" ] && [ -d /proc/$1 ] && ! grep -q '^State:[[:space:]]*Z' /proc/$1/status 2>/dev/null; }
ours_pids() { local p; for p in $(pgrep -x intarsia); do ours "$p" && echo -n "$p "; done; echo; }
children() { pgrep -P "$1" -x intarsia 2>/dev/null | head -1; }
listen() { ss -ltn "sport = :$PORT" 2>/dev/null | grep -c LISTEN; }
snap() { # one line, key=value
  local f; f=$(free -m | awk '/^Mem:/ {print "used_mb="$3" avail_mb="$7" cache_mb="$6} /^Swap:/ {print "swap_mb="$3}' | tr '\n' ' ')
  echo "SNAP label=$1 t=$(now) $f gpu_apps=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null | tr -d ' ' | tr '\n' ';') listen=$(listen) ours=[$(ours_pids)] shm=$(ls /dev/shm 2>/dev/null | wc -l) vllm_http=$(curl -s -o /dev/null -w '%{http_code}' --max-time 3 http://127.0.0.1:8000/health)"
}
health_code() { curl -s -o /dev/null -w '%{http_code}' --max-time 1 "http://127.0.0.1:$PORT/health"; }
wait_health() { # wait_health <t0> <deadline_s>: sets T_LISTEN T_HEALTH (epoch), returns 1 on timeout
  T_LISTEN=""; T_HEALTH=""; local c i=0
  while (( i < $2 * 10 )); do c=$(health_code)
    [[ $c != 000 && -z $T_LISTEN ]] && T_LISTEN=$(now)
    [[ $c == 200 ]] && { T_HEALTH=$(now); return 0; }
    sleep 0.1; i=$((i+1)); done; return 1
}
first_token() { python3 "$W/ttft.py" $PORT; }
stop_mw() { touch "$S/stop-mw"; [ -s "$S/mwpid" ] && { kill "$(cat $S/mwpid)" 2>/dev/null; : > "$S/mwpid"; }; }
start_mw() { nohup bash -c "while [ ! -e $S/stop-mw ]; do a=\$(awk '/MemAvailable/ {print int(\$2/1024)}' /proc/meminfo); echo \"\$(date +%T) \$a\" >> $W/mem-$1.txt
  if [ \$a -lt 15360 ]; then echo \"KILLED low mem \$a\" >> $W/mem-$1.txt; for p in \$(pgrep -x intarsia); do [ \"\$(readlink /proc/\$p/exe)\" = \"$(readlink -f $BIN)\" ] && kill -9 \$p; done; fi; sleep 2; done" > /dev/null 2>&1 < /dev/null & echo $! > "$S/mwpid"; }
cmd=${1:-}; tag=${2:-x}
case $cmd in
snap) snap "$tag" ;;
start)
  stop_mw; sleep 0.2; rm -f "$S/stop-mw" "$W/serve-$tag.log" "$W/mem-$tag.txt"
  [[ $(listen) != 0 ]] && { echo "RESULT start $tag FAIL port $PORT busy"; exit 1; }
  snap "pre-$tag"
  start_mw "$tag"
  T0=$(now)
  nohup "$BIN" serve --backend cuda --model "$M35" --port $PORT --host 0.0.0.0 --served-model-name guidance --max-seq 131072 --mem-budget 40 --profile-every 0 --max-batch 2 --prefix-store "$PS" > "$W/serve-$tag.log" 2>&1 < /dev/null &
  SP=$!; echo $SP > "$S/spid"
  if ! wait_health "$T0" 400; then echo "RESULT start $tag FAIL no /health in 400 s"; tail -5 "$W/serve-$tag.log"; exit 1; fi
  sleep 0.2; CP=$(children $SP); echo "${CP:-}" > "$S/cpid"
  FT=$(first_token); T2=$(now)
  echo "RESULT start $tag t_launch=$T0 spid=$SP cpid=$CP launch_to_listen_s=$(el $T0 $T_LISTEN) launch_to_health_s=$(el $T0 $T_HEALTH) health_to_first_token_s=$(el $T_HEALTH $T2) first_token_json=$FT"
  snap "up-$tag"
  grep -aE '^loaded |prefix store: |KV cache once|spec mtp|batched MTP' "$W/serve-$tag.log" | cut -c1-220 | sed 's/^/SERVELOG /' ;;
term|kill9)
  how=${3:-}; SP=$(cat "$S/spid" 2>/dev/null); CP=$(children "$SP"); [[ -z $CP ]] && CP=$(cat "$S/cpid" 2>/dev/null)
  ours "$SP" || { echo "RESULT $cmd $tag ABORT supervisor pid '$SP' is not our binary"; exit 1; }
  ours "$CP" || { echo "RESULT $cmd $tag ABORT child pid '$CP' is not our binary"; exit 1; }
  snap "before-$tag"
  if [[ $cmd == term ]]; then sig=TERM; else sig=KILL; fi
  case $how in
    sup) tgt="$SP"; watch="$SP $CP" ;;
    group|both) tgt="$CP $SP"; watch="$SP $CP" ;;
    child) tgt="$CP"; watch="$CP" ;;
    *) echo "RESULT $cmd $tag ABORT bad mode $how"; exit 1 ;;
  esac
  T0=$(now); kill -$sig $tgt; echo "SIGNAL $sig to [$tgt] (spid=$SP cpid=$CP) at t=$T0"
  declare -A TX; i=0; TPORT=""
  while (( i < 3000 )); do   # 150 s at 50 ms
    for p in $watch; do [[ -z ${TX[$p]:-} ]] && ! alive $p && TX[$p]=$(now); done
    [[ -z $TPORT && $(listen) == 0 ]] && TPORT=$(now)
    alldead=1; for p in $watch; do [[ -z ${TX[$p]:-} ]] && alldead=0; done
    if [[ $how == child ]]; then [[ $alldead == 1 ]] && break; else [[ $alldead == 1 && -n $TPORT ]] && break; fi
    sleep 0.05; i=$((i+1)); done
  for p in $watch; do [[ -n ${TX[$p]:-} ]] && echo "RESULT $cmd $tag $how pid=$p exit_after_s=$(el $T0 ${TX[$p]})" || echo "RESULT $cmd $tag $how pid=$p STILL ALIVE after 150 s"; done
  [[ $how != child ]] && { [[ -n $TPORT ]] && echo "RESULT $cmd $tag $how port_freed_after_s=$(el $T0 $TPORT)" || echo "RESULT $cmd $tag $how PORT STILL LISTENING after 150 s"; }
  snap "t0+0-$tag"
  if [[ $how == child ]]; then
    # the supervisor restarts the child: time kill -> new child -> /health -> first token
    NC=""; i=0; while (( i < 600 )); do NC=$(children "$SP"); [[ -n $NC && $NC != $CP ]] && break; NC=""; sleep 0.05; i=$((i+1)); done
    TNC=$(now); [[ -n $NC ]] && echo "RESULT $cmd $tag child restarted by supervisor: new_child=$NC kill_to_new_child_s=$(el $T0 $TNC)" || echo "RESULT $cmd $tag child NOT restarted in 30 s"
    snap "t0+restart-$tag"
    if wait_health "$T0" 400; then FT=$(first_token); T2=$(now); echo "RESULT $cmd $tag child kill_to_listen_s=$(el $T0 $T_LISTEN) kill_to_health_s=$(el $T0 $T_HEALTH) kill_to_first_token_s=$(el $T0 $T2) first_token_json=$FT"
      echo "$NC" > "$S/cpid"; snap "up-after-$tag"
    else echo "RESULT $cmd $tag child restart: NO /health in 400 s"; fi
  else
    sleep 3; snap "t0+3-$tag"
    echo "RESULT $cmd $tag $how leftover ours=[$(ours_pids)]"
    stop_mw
  fi
  grep -a '^\[supervisor\]' "$W/serve-$tag.log" | tail -5 | sed 's/^/SERVELOG /'; tail -3 "$W/serve-$tag.log" | cut -c1-200 | sed 's/^/SERVELOG tail: /' ;;
stop|cleanup)
  stop_mw
  for p in $(pgrep -x intarsia); do ours "$p" && { echo "killing leftover ours $p"; kill -9 $p; }; done; sleep 2; snap "after-cleanup" ;;
*) echo "usage: ctl.sh start|term|kill9|snap|stop"; exit 2 ;;
esac
