#!/usr/bin/env python3
"""Lane hgv identity driver: greedy output text of 1 / 8 / 16 streams, same prompt and different prompts, temperature 0, no nonce, so two
servers (main and hgv, tp1 and tp2) get byte-identical requests and their texts can be compared (scripts/hgv/ident_cmp.py).
  cd /opt/code/deepeval && uv run python scripts/hgv/ident.py --url http://tatooine:8090/v1 --model qwen36-35b \
      --docs-list scripts/hgv/ident_docs.txt --root . --gen 384 --out OUT.json
Each phase: N streams started together (the server batches them), ignore_eos + min_tokens = gen. Phases: same1 same8 same16 div1 div8 div16
(div = N different ~1k-token prompts cut from the docs in --docs-list; same = one filler prompt sent N times)."""
import argparse, hashlib, http.client, json, os, threading, time, urllib.parse

ap = argparse.ArgumentParser()
ap.add_argument("--url", required=True)
ap.add_argument("--model", required=True)
ap.add_argument("--docs-list", required=True)
ap.add_argument("--root", required=True, help="the directory the paths in --docs-list are relative to")
ap.add_argument("--gen", type=int, default=384)
ap.add_argument("--phases", default="same1,same8,same16,div1,div8,div16")
ap.add_argument("--chars", type=int, default=3500)
ap.add_argument("--out", required=True)
a = ap.parse_args()

from localeval.speed import INSTRUCTION, filler  # noqa: E402

u = urllib.parse.urlsplit(a.url.rstrip("/") + "/chat/completions")


def chat(content, max_tokens):
    body = {"model": a.model, "messages": [{"role": "user", "content": content}], "max_tokens": max_tokens, "temperature": 0.0,
            "stream": True, "stream_options": {"include_usage": True}, "chat_template_kwargs": {"enable_thinking": False},
            "ignore_eos": True, "min_tokens": max_tokens}
    c = http.client.HTTPConnection(u.hostname, u.port, timeout=900)
    c.request("POST", u.path, body=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    r = c.getresponse()
    if r.status != 200:
        return {"ok": False, "err": r.read().decode()[:300]}
    text, usage = [], None
    for raw in r:
        line = raw.decode("utf-8", "replace").strip()
        if not line.startswith("data:") or line.endswith("[DONE]"):
            continue
        d = json.loads(line[5:])
        if d.get("choices") and d["choices"][0].get("delta", {}).get("content"):
            text.append(d["choices"][0]["delta"]["content"])
        if d.get("usage"):
            usage = d["usage"]
    t = "".join(text)
    return {"ok": True, "text": t, "sha": hashlib.sha256(t.encode()).hexdigest()[:16], "completion": usage and usage["completion_tokens"],
            "prompt": usage and usage["prompt_tokens"]}


same_prompt = filler(70, seed=1024) + INSTRUCTION
docs = [l.strip() for l in open(a.docs_list) if l.strip()]
diverse = []
for f in docs:
    t = open(os.path.join(a.root, f), errors="replace").read()
    s = (len(t) // 3) % max(1, len(t) - a.chars)
    diverse.append(t[s:s + a.chars] + "\n\nSummarize the text above and continue it in detail.")
assert len(diverse) >= 16

chat("warm up", 2)
res = {"gen": a.gen, "phases": {}}
for ph in a.phases.split(","):
    prompts = [same_prompt] * int(ph[4:]) if ph.startswith("same") else diverse[: int(ph[3:])]
    out = [None] * len(prompts)

    def one(i, p):
        out[i] = chat(p, a.gen)

    t0 = time.perf_counter()
    th = [threading.Thread(target=one, args=(i, p)) for i, p in enumerate(prompts)]
    [t.start() for t in th]
    [t.join() for t in th]
    res["phases"][ph] = {"wall_s": round(time.perf_counter() - t0, 1), "streams": out}
    print(json.dumps({"phase": ph, "ok": sum(1 for o in out if o["ok"]), "wall_s": res["phases"][ph]["wall_s"],
                      "sha": [o.get("sha") for o in out], "completion": [o.get("completion") for o in out]}), flush=True)
    time.sleep(3)
json.dump(res, open(a.out, "w"))
