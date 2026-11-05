#!/usr/bin/env python3
"""Other prompt lengths: char-search a prefix of the long prompt to a target token count, then solo vs behind."""
import json, sys, threading, time, urllib.request
port, pr, outp = sys.argv[1:4]; targets = [int(x) for x in sys.argv[4].split(",")]
text = open(pr).read()
def call(p, mt, lp):
    b = {"model": "intarsia", "prompt": p, "max_tokens": mt, "temperature": 0, "ignore_eos": True, "return_token_ids": True}
    if lp: b["logprobs"] = 20
    r = urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions", json.dumps(b).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(r, timeout=900))
res = {}
BG = "Background row.\n" + text[:150]
for tgt in targets:
    head = f"Length probe {tgt}.\n"
    n = int(tgt * 3.9)
    for _ in range(40):
        pl = call(head + text[:n], 1, False)["usage"]["prompt_tokens"]
        if pl == tgt: break
        n += (tgt - pl) if abs(tgt - pl) < 4 else int((tgt - pl) * 3.5)
    if pl != tgt:
        print("no exact length", tgt, pl, file=sys.stderr, flush=True); continue
    P = head + text[:n]; rec = {}
    for m in ("solo", "behind"):
        t = None
        if m == "behind":
            import os; t = threading.Thread(target=lambda: call(BG, int(os.environ.get("BG_MAX", "200")), False)); t.start(); time.sleep(float(os.environ.get("BG_SLEEP", "3")))
        d = call(P, 200, True); c = d["choices"][0]
        rec[m] = {"ids": c["token_ids"], "top": c["logprobs"]["top_logprobs_ids"], "plen": d["usage"]["prompt_tokens"]}
        if t: t.join()
    res[str(tgt)] = rec; json.dump(res, open(outp, "w")); print(time.strftime("%X"), tgt, "done", file=sys.stderr, flush=True)
