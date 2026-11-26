"""Serve-level batched == solo check (lane ink-hip-batch). 16 distinct real-text prompts (scripts/cmx/ids-prompts: the 7 files and
9 prefixes of the two 4k documents, 150 B .. 13 kB), greedy (temperature 0), N forced tokens (ignore_eos + min_tokens).
  python3 serve_bitwise.py URL solo OUT.json      one request at a time (the server's solo path)
  python3 serve_bitwise.py URL conc OUT.json      all 16 at once (a --max-batch 16 server decodes them batched)
  python3 serve_bitwise.py compare A.json B.json  per prompt: equal text and token count?
Records each reply's text, md5, completion / prompt / cached token counts."""
import concurrent.futures as cf
import hashlib
import json
import os
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
P = os.path.join(HERE, "..", "scripts", "cmx", "ids-prompts")
N = 128


def prompts():
    files = ["short-0", "short-1", "short-2", "mid-0", "mid-1", "4k-0", "4k-1"]
    out = [(f, open(os.path.join(P, f + ".txt")).read()) for f in files]
    a, b = (open(os.path.join(P, f + ".txt")).read() for f in ("4k-0", "4k-1"))
    for i, n in enumerate([700, 1500, 2600, 4000, 6000, 8000, 10000, 9500, 5200]):
        src = a if i % 2 == 0 else b
        out.append((f"{'4k-0' if i % 2 == 0 else '4k-1'}[:{n}]", src[:n]))
    return out


def ask(url, text):
    body = {"model": "ink", "messages": [{"role": "user", "content": text + "\n\nSummarize the above in a few sentences."}],
            "temperature": 0, "max_tokens": N, "min_tokens": N, "ignore_eos": True, "stream": False}
    req = urllib.request.Request(url.rstrip("/") + "/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=200) as r:
        d = json.load(r)
    msg = d["choices"][0]["message"]
    text = (msg.get("reasoning_content") or "") + "\x00" + (msg.get("content") or "")
    u = d.get("usage", {})
    return {"md5": hashlib.md5(text.encode()).hexdigest(), "text": text, "completion_tokens": u.get("completion_tokens"),
            "prompt_tokens": u.get("prompt_tokens"), "cached_tokens": (u.get("prompt_tokens_details") or {}).get("cached_tokens")}


def main():
    if sys.argv[1] == "compare":
        a, b = (json.load(open(x)) for x in sys.argv[2:4])
        same = 0
        for k in a:
            eq = a[k]["md5"] == b[k]["md5"] and a[k]["completion_tokens"] == b[k]["completion_tokens"]
            same += eq
            print(f"{'EQUAL' if eq else 'DIFF '} {k:>14}: prompt {a[k]['prompt_tokens']:>5} tok, {a[k]['completion_tokens']} / "
                  f"{b[k]['completion_tokens']} out, md5 {a[k]['md5'][:12]} / {b[k]['md5'][:12]}, cached {a[k]['cached_tokens']} / {b[k]['cached_tokens']}")
        print(f"{same} / {len(a)} replies equal")
        sys.exit(0 if same == len(a) else 1)
    url, mode, out = sys.argv[1:4]
    ps = prompts()
    res = {}
    if mode == "solo":
        for k, t in ps:
            res[k] = ask(url, t)
    else:
        with cf.ThreadPoolExecutor(len(ps)) as ex:
            futs = {k: ex.submit(ask, url, t) for k, t in ps}
            res = {k: f.result() for k, f in futs.items()}
    json.dump(res, open(out, "w"), indent=1)
    print(f"{mode}: {len(res)} replies, {sum(r['completion_tokens'] or 0 for r in res.values())} tokens -> {out}")


if __name__ == "__main__":
    main()
