#!/bin/bash
# M1 window 5: HIP no-op of the FfnBlock change (Qwen3-8B, base vs head), ONE fleetq run --takes-down radiance, <=15 min.
# The EXIT trap kills our containers and restores radiance (waits for /v1/models plus one chat completion).
set -u
L=/private/tmp/claude-501/-opt-code-intarsiaml/f31ccf91-6032-412f-aeab-eea38852047d/scratchpad/q38/window5.log
log() { echo "$(date +%T) $*" | tee -a $L; }
restore() {
  log "restore: killing m1noop containers"
  ssh tatooine 'for v in base head; do docker ps -q --filter volume=m1noop-target-$v | xargs -r docker kill; done' >/dev/null 2>&1
  log "restore: starting radiance"
  ssh tatooine 'systemctl --user start radiance'
  for i in $(seq 1 90); do
    if ssh tatooine 'curl -sf -m 3 localhost:8080/v1/models' >/dev/null 2>&1; then log "radiance /v1/models up"; break; fi
    sleep 5
  done
  ssh tatooine 'curl -s -m 300 localhost:8080/v1/chat/completions -H "Content-Type: application/json" -d "{\"model\":\"Qwen3.8\",\"messages\":[{\"role\":\"user\",\"content\":\"Say OK.\"}],\"max_tokens\":300}"' \
    | grep -o '"content":"[^"]*"' | head -1 | tee -a $L
  log "restore: done"
}
log "window start (job ${FLEETQ_JOB:-?})"
GAP=$(fleetq log --limit 400 2>/dev/null | python3 -I /private/tmp/claude-501/-opt-code-intarsiaml/f31ccf91-6032-412f-aeab-eea38852047d/scratchpad/scripts/gap.py)
[ "${GAP:-0}" -gt 0 ] && { log "waiting ${GAP}s for the 5-min serving gap"; sleep $GAP; }
trap restore EXIT
ssh tatooine 'systemctl --user stop radiance'
for i in $(seq 1 30); do
  [ "$(ssh tatooine 'pgrep -c -x radiance' 2>/dev/null)" = "0" ] && break; sleep 2
done
log "radiance stopped"
ssh tatooine 'timeout 600 bash /tmp/m1noop-checks.sh' 2>&1 | tee -a $L
log "checks rc=${PIPESTATUS[0]}"
