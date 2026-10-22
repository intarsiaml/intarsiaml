#!/usr/bin/env bash
# fleetq run --host spark2 --mem 34 --label "..." -- scripts/standings-fx3/smoke-sglang.sh <35b|8b>
set -u
: "${FLEETQ_JOB:?}"
HOST=${HOST:-spark2}; M=${1:-35b}; PORT=8200
S="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
rsh() { ssh -n -o BatchMode=yes "$HOST" "export FLEETQ_JOB=$FLEETQ_JOB; $*"; }
trap 'rsh "docker rm -f fx3-sglang >/dev/null 2>&1"' EXIT
scp -q "$S/remote/start-sglang.sh" "$HOST:fx3/start-sglang.sh"
rsh "bash ~/fx3/start-sglang.sh $M $PORT"
for i in $(seq 1 120); do
  if rsh "curl -sf http://127.0.0.1:$PORT/v1/models >/dev/null"; then echo "up after $((i*10))s"; break; fi
  rsh "docker ps -q --filter name=fx3-sglang | grep -q ." || { echo "container died"; break; }
  sleep 10
done
rsh "docker logs fx3-sglang 2>&1 | tail -40" | cut -c1-300
rsh "curl -s http://127.0.0.1:$PORT/v1/chat/completions -H 'content-type: application/json' -d '{\"model\":\"sglang\",\"messages\":[{\"role\":\"user\",\"content\":\"What is 17*23? Answer with the number only.\"}],\"max_tokens\":20,\"temperature\":0,\"chat_template_kwargs\":{\"enable_thinking\":false}}'" | cut -c1-600
rsh "free -g | head -2"
