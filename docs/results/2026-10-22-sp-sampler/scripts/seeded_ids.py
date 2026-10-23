"""sp lane: seeded sampled completions, token ids out, for the base/head bit-identity gate.

python3 seeded_ids.py --url U --out F.json   (one JSON object: case -> token id list)
"""
import argparse
import json
import urllib.request

PROMPTS = [
    "Write a short story about a lighthouse keeper who finds a message in a bottle.",
    "Explain how a hash map works, with an example in Python.",
]
# (temperature, top_p, top_k): the ssbench arm, a top-p arm, and a top-k + top-p arm
PARAMS = [(0.7, 1.0, 0), (1.0, 0.95, 0), (0.8, 0.9, 40)]
SEEDS = [1, 2]


def post(url, body):
    req = urllib.request.Request(url, json.dumps(body).encode(), {"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())


ap = argparse.ArgumentParser()
ap.add_argument("--url", default="http://127.0.0.1:8094")
ap.add_argument("--out", required=True)
ap.add_argument("--max-tokens", type=int, default=300)
a = ap.parse_args()
out = {}
for pi, prompt in enumerate(PROMPTS):
    for t, tp, tk in PARAMS:
        for seed in SEEDS:
            body = {"prompt": prompt, "max_tokens": a.max_tokens, "ignore_eos": True, "temperature": t, "top_p": tp,
                    "seed": seed, "return_token_ids": True}
            if tk:
                body["top_k"] = tk
            r = post(a.url + "/v1/completions", body)
            ids = r["choices"][0]["token_ids"]
            key = f"p{pi}-t{t}-tp{tp}-tk{tk}-s{seed}"
            out[key] = ids
            print(key, len(ids), ids[:8], flush=True)
json.dump(out, open(a.out, "w"))
