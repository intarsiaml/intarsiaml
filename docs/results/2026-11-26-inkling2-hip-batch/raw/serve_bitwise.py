"""Serve-level batched == solo check (lane ink-hip-batch). 16 distinct real-text prompts (scripts/cmx/ids-prompts: the 7 files and
9 prefixes of the two 4k documents, 150 B .. 13 kB), greedy (temperature 0), N forced tokens (ignore_eos + min_tokens).
  python3 serve_bitwise.py URL solo OUT.json      one request at a time (the server's solo path)
  python3 serve_bitwise.py URL conc OUT.json      all 16 at once (a --max-batch 16 server decodes them batched)
  python3 serve_bitwise.py URL conc:N:OFF[:nonce] OUT.json   N of them at once: the prompts sorted by length, every (16 / N)-th
                                                  from OFF (so each subset spans short to 4k); with nonce, a rep tag in front
                                                  (defeats any prefix reuse; the replies then match no solo reply)
  python3 serve_bitwise.py compare A.json B.json  per prompt both hold: equal text and token count?
Records each reply's text, md5, completion / prompt / cached token counts."""
import concurrent.futures as cf
import hashlib
import json
import os
import sys
import time
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


def subset(ps, n, off, nonce):
    lens = sorted(ps, key=lambda kt: len(kt[1]))
    step = max(1, len(lens) // n)
    pick = [lens[(off + i * step) % len(lens)] for i in range(n)]
    return [(k, (f"[{nonce}]\n" + t) if nonce else t) for k, t in pick]


def ask(url, text):
    body = {"model": "ink", "messages": [{"role": "user", "content": text + "\n\nSummarize the above in a few sentences."}],
            "temperature": 0, "max_tokens": N, "min_tokens": N, "ignore_eos": True, "stream": False}
    req = urllib.request.Request(url.rstrip("/") + "/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=200) as r:
        d = json.load(r)
    t1 = time.time()
    msg = d["choices"][0]["message"]
    text = (msg.get("reasoning_content") or "") + "\x00" + (msg.get("content") or "")
    u = d.get("usage", {})
    return {"md5": hashlib.md5(text.encode()).hexdigest(), "text": text, "completion_tokens": u.get("completion_tokens"),
            "prompt_tokens": u.get("prompt_tokens"), "cached_tokens": (u.get("prompt_tokens_details") or {}).get("cached_tokens"),
            "t0": t0, "t1": t1}


def main():
    if sys.argv[1] == "compare":
        a, b = (json.load(open(x)) for x in sys.argv[2:4])
        same = 0
        common = [k for k in a if k in b]
        for k in common:
            eq = a[k]["md5"] == b[k]["md5"] and a[k]["completion_tokens"] == b[k]["completion_tokens"]
            same += eq
            ta, tb = a[k]["text"], b[k]["text"]
            first = next((i for i in range(min(len(ta), len(tb))) if ta[i] != tb[i]), min(len(ta), len(tb)))
            div = "" if eq else f", first differing character {first} of {len(ta)} / {len(tb)}"
            cache = " CACHED" if (a[k]["cached_tokens"] or 0) or (b[k]["cached_tokens"] or 0) else ""
            print(f"{'EQUAL' if eq else 'DIFF '} {k:>14}: prompt {a[k]['prompt_tokens']:>5} tok, {a[k]['completion_tokens']} / "
                  f"{b[k]['completion_tokens']} out, md5 {a[k]['md5'][:12]} / {b[k]['md5'][:12]}, cached {a[k]['cached_tokens']} / "
                  f"{b[k]['cached_tokens']}{cache}{div}")
        print(f"{same} / {len(common)} replies equal")
        sys.exit(0 if same == len(common) else 1)
    url, mode, out = sys.argv[1:4]
    ps = prompts()
    res = {}
    t0 = time.time()
    if mode.startswith("conc:"):
        f = mode.split(":")
        ps = subset(ps, int(f[1]), int(f[2]), f[3] if len(f) > 3 else "")
    if mode == "solo":
        for k, t in ps:
            res[k] = ask(url, t)
    else:
        with cf.ThreadPoolExecutor(len(ps)) as ex:
            futs = {k: ex.submit(ask, url, t) for k, t in ps}
            res = {k: f.result() for k, f in futs.items()}
    t1 = time.time()
    wall = t1 - t0
    json.dump(res, open(out, "w"), indent=1)
    toks = sum(r["completion_tokens"] or 0 for r in res.values())
    lat = sorted(r["t1"] - r["t0"] for r in res.values())
    meta = {"mode": mode, "n": len(res), "t0": t0, "t1": t1, "wall_s": round(wall, 3), "completion_tokens": toks,
            "prompt_tokens": sum(r["prompt_tokens"] or 0 for r in res.values()), "aggregate_tok_s": round(toks / wall, 1),
            "latency_s_median": round(lat[len(lat) // 2], 2), "latency_s_max": round(lat[-1], 2),
            "cached_tokens": sum(r["cached_tokens"] or 0 for r in res.values())}
    json.dump(meta, open(out[:-5] + ".meta.json" if out.endswith(".json") else out + ".meta", "w"), indent=1)
    print(f"wall {wall:.2f} s, aggregate {meta['aggregate_tok_s']} tok/s, latency median {meta['latency_s_median']} s max "
          f"{meta['latency_s_max']} s, prompt tokens {meta['prompt_tokens']}, cached {meta['cached_tokens']}")
    print(f"{mode}: {len(res)} replies, {sum(r['completion_tokens'] or 0 for r in res.values())} tokens -> {out}")


if __name__ == "__main__":
    main()
