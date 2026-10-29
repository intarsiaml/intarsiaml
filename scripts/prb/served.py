#!/usr/bin/env python3
"""Lane prb: served greedy identity of a repeated prompt on one running server (the server is NOT restarted between requests, so a
repeat finds its earlier slot). Prompts are scripts/htk/equal.py's 16. Two phases: SEQ (one request at a time, the order given)
and CONC (rounds of the first four distinct prompts of the order, all at once). Saves a list of {phase, prompt, ids, cached}.
  served.py run URL ORDER OUT      e.g. served.py run http://127.0.0.1:8197/v1 3,4,5,9,13,15,3,4,5,9,13,15 out.json
  served.py compare REF OUT        REF is a run of a server with INTARSIA_CB_NO_REUSE=1 (every request cold)
"""
import json, os, sys, threading
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'htk'))
import equal


def one(url, i):
    body = {"model": "intarsia", "prompt": equal.PROMPTS[i], "max_tokens": 64, "ignore_eos": True, "temperature": 0.0,
            "top_p": 1.0, "seed": 100 + i, "return_token_ids": True}
    r = equal.post(url, body)
    cached = (r.get("usage") or {}).get("prompt_tokens_details", {}) or {}
    return {"prompt": i, "ids": r["choices"][0]["token_ids"], "cached": cached.get("cached_tokens")}


def run(url, order, out):
    recs = []
    for i in order:
        recs.append({"phase": "SEQ", **one(url, i)})
        print("SEQ", i, flush=True)
    first4 = list(dict.fromkeys(order))[:4]
    for rnd in range(3):
        got = [None] * len(first4)
        def w(k, i):
            got[k] = one(url, i)
        ths = [threading.Thread(target=w, args=(k, i)) for k, i in enumerate(first4)]
        [t.start() for t in ths]; [t.join() for t in ths]
        recs += [{"phase": f"CONC{rnd}", **g} for g in got]
        print("CONC", rnd, first4, flush=True)
    json.dump(recs, open(out, "w"))


def compare(ref, out):
    R, O = json.load(open(ref)), json.load(open(out))
    want = {}
    for r in R:
        want.setdefault(r["prompt"], r["ids"])
        assert want[r["prompt"]] == r["ids"], f"reference not stable for prompt {r['prompt']}"
    bad = 0
    for r in O:
        w = want[r["prompt"]]
        pos = next((j for j in range(len(w)) if r["ids"][j] != w[j]), None) if r["ids"] != w else None
        bad += pos is not None
        print(f"{r['phase']:6s} prompt {r['prompt']:2d} cached {r['cached']}: {'IDENTICAL' if pos is None else f'DIFFER at token {pos}'}")
    print(f"{len(O) - bad}/{len(O)} requests identical to the cold reference")
    return bad


if __name__ == "__main__":
    if sys.argv[1] == "run":
        run(sys.argv[2], [int(x) for x in sys.argv[3].split(",")], sys.argv[4])
    else:
        sys.exit(1 if compare(sys.argv[2], sys.argv[3]) else 0)
