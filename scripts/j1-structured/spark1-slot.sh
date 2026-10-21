#!/bin/bash
# J1 structured output on spark1 (CUDA, Qwen3-8B BF16), base vs head, non-exclusive with vLLM up. Runs locally (endor)
# and drives spark1 over ssh; one fleetq job:
#   fleetq run --host spark1 --mem 22 --label "J1 CUDA no-op + e2e" -- \
#     bash scripts/j1-structured/spark1-slot.sh OUT BASE HEAD
# 1. builds base and head on spark1 (git archive, CARGO_TARGET_DIR per tree);
# 2. CUDA no-op as docs/results/2026-10-21-m1-qwen38-drill/raw/cuda-noop/m1cuda.sh: KL21 short + long and 256 greedy ids
#    (short, 4k), base then head;
# 3. head server on :8191: e2e.py (50 requests) from here; HTTP greedy ids of one chat request on base and head.
set -u
R=$(cd "$(dirname "$0")/../.." && pwd)
O=$1 BASE=$2 HEAD=$3
H=spark1
PORT=8191
NAME=qwen3-8b
URL=http://spark-4e77:$PORT
RP='export PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH CUDA_PATH=/usr/local/cuda; N=$HOME/j1; M=$HOME/models/Qwen3-8B; G=$HOME/intarsiaml-golden'
mkdir -p "$O"
RPID=
trap '[ -n "$RPID" ] && ssh $H "kill $RPID" 2>/dev/null' EXIT

date +%T
for v in base:$BASE head:$HEAD; do
  n=${v%%:*}; rev=${v#*:}
  git -C "$R" archive "$rev" | ssh $H "rm -rf j1/$n && mkdir -p j1/$n j1/out && tar -x -C j1/$n"
  ssh $H "$RP; cd \$N/$n && CARGO_TARGET_DIR=\$N/target-$n nice cargo build --release -q -p intarsia-cli --features cuda -j 10" \
    > "$O/build-$n.log" 2>&1 || { echo "build $n FAILED"; tail -20 "$O/build-$n.log"; exit 2; }
  echo "built $n $rev $(date +%T)"
done

ssh $H "$RP; P=\$N/head/docs/results/raw/2026-10-14-metal-phase-b; O=\$N/out
  for v in base head; do
    B=\$N/target-\$v/release/intarsia
    \$B accuracy --model \$M --backend cuda --golden \$G/qwen3-8b-fp32.safetensors --gate-class bf16 --max-seq 4096 > \$O/\$v-acc-short.out 2> \$O/\$v-acc-short.err; echo \"\$v accshort rc=\$? \$(date +%T)\"
    \$B accuracy --model \$M --backend cuda --golden \$G/qwen3-8b-long-fp32.safetensors --gate-class bf16 --max-seq 8192 > \$O/\$v-acc-long.out 2> \$O/\$v-acc-long.err; echo \"\$v acclong rc=\$? \$(date +%T)\"
    for p in short 4k; do
      \$B generate --model \$M --backend cuda --max-seq 8192 --temperature 0 --max-tokens 256 --ignore-eos --show-ids --prompt \"\$(cat \$P/prompt-\$p.txt)\" > \$O/\$v-gen-\$p.out 2> \$O/\$v-gen-\$p.err; echo \"\$v gen-\$p rc=\$? \$(date +%T)\"
    done
  done
  cd \$O
  for f in acc-short.out acc-long.out gen-short.out gen-4k.out; do cmp -s base-\$f head-\$f && echo \"\$f: identical\" || echo \"\$f: DIFFERS\"; done
  for p in short 4k; do b=\$(grep '^ids:' base-gen-\$p.err | md5sum | cut -c1-8); h=\$(grep '^ids:' head-gen-\$p.err | md5sum | cut -c1-8); echo \"ids \$p: base \$b head \$h\"; done
  grep -h 'OVERALL\|golden gate' base-acc-*.out head-acc-*.out" 2>&1 | tee "$O/noop.txt"

serve() { # arm
  RPID=$(ssh $H "$RP; nohup \$N/target-$1/release/intarsia serve --backend cuda --model \$M --host 0.0.0.0 --port $PORT \
    --served-model-name $NAME --max-seq 8192 --mem-budget 20 --profile-every 0 --no-supervisor --no-ui \
    > \$N/out/serve-$1.log 2>&1 < /dev/null & echo \$!")
  for _ in $(seq 1 300); do
    curl -sf $URL/v1/models > /dev/null && { echo "serve $1 up $(date +%T)"; return 0; }
    sleep 1
  done
  echo "serve $1 did not come up"; ssh $H "tail -20 j1/out/serve-$1.log"; return 1
}
unserve() { [ -n "$RPID" ] && ssh $H "kill $RPID; sleep 3"; RPID=; }
http_ids() {
  curl -sf $URL/v1/chat/completions -H 'content-type: application/json' -d '{"messages":
    [{"role": "user", "content": "Write a short story about a lighthouse keeper and a robot."}], "max_tokens": 256,
    "temperature": 0, "return_token_ids": true, "chat_template_kwargs": {"enable_thinking": false}}' \
    | python3 -I -c 'import json,sys; print(json.load(sys.stdin)["choices"][0]["token_ids"])' > "$O/http-ids-$1.txt"
  echo "http ids $1: $(md5 -q "$O/http-ids-$1.txt" | cut -c1-8)"
}

if serve head; then
  http_ids head
  uv run --script "$R/scripts/j1-structured/e2e.py" --url $URL --out "$O/e2e-seq.json" --constrained-speed \
    > "$O/e2e-seq.log" 2>&1; echo "e2e seq rc=$? $(tail -1 "$O/e2e-seq.log")"
  unserve
fi
if serve base; then
  http_ids base
  unserve
fi
scp -q "$H:j1/out/*" "$O/" 2>/dev/null
cmp -s "$O/http-ids-base.txt" "$O/http-ids-head.txt" && echo "http ids: identical" || echo "http ids: DIFFER"
date +%T
