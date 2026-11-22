#!/usr/bin/env bash
# ink-serve: Inkling-2-Mini through the server on spark1's CPU backend, a handful of short requests. Run under:
#   fleetq run --host spark1 --load cpu --mem 40 --expect 25 --label "intarsiaml inkling serve: cpu smoke" -- bash scripts/inkling2-spark-smoke.sh
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
REPO="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
scp -q -o BatchMode=yes "$REPO/tools/inkling2_serve_smoke_cpu.py" spark1:intarsiaml-ink-serve/smoke.py
ssh -n -o BatchMode=yes spark1 'cd ~/intarsiaml-ink-serve; PORT=8097
./bin/intarsia serve --model model --port $PORT --served-model-name inkling-2-mini --max-seq 1024 --no-supervisor > out/server.log 2>&1 &
SP=$!
for i in $(seq 1 200); do curl -sf http://127.0.0.1:$PORT/health >/dev/null && break; kill -0 $SP 2>/dev/null || { echo "server died"; tail -20 out/server.log; exit 1; }; sleep 3; done
echo "server up after $((i*3)) s"; free -g | sed -n 2p
python3 smoke.py http://127.0.0.1:$PORT/v1 out/smoke-out.json
free -g | sed -n 2p
kill $SP; wait $SP 2>/dev/null; tail -5 out/server.log'
