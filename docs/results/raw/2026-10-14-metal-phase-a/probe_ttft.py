# one non-streaming /v1/completions call per prompt (max_tokens=1) against a running `intarsia serve`;
# wall time of the call = queue + prefill + 1 decode step. Each prompt starts with a unique nonce so the prefix cache cannot hit.
import json, sys, time, urllib.request, uuid
base = "http://127.0.0.1:8091/v1/completions"
sent = "The quick brown fox jumps over the lazy dog near the river bank while the sun sets slowly over the hills. "
def call(n_sent, label):
    prompt = f"[{uuid.uuid4()}] " + sent * n_sent
    body = json.dumps({"model": "q8", "prompt": prompt, "max_tokens": 1, "temperature": 0}).encode()
    t = time.time()
    r = json.load(urllib.request.urlopen(urllib.request.Request(base, body, {"Content-Type": "application/json"}), timeout=900))
    dt = time.time() - t
    n = r["usage"]["prompt_tokens"]
    print(f"{label:28s} prompt_tokens={n:6d}  wall={dt*1e3:9.1f} ms  {n/dt:8.1f} tok/s", flush=True)
for spec in sys.argv[1:]:
    n, label = spec.split(":")
    call(int(n), label)
