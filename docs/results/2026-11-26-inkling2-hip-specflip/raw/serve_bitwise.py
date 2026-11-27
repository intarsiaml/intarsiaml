"""Serve-level batched == solo check (lane ink-hip-bpf). 16 distinct real-text prompts (scripts/cmx/ids-prompts: the 7 files and
9 prefixes of the two 4k documents, 150 B .. 13 kB), greedy (temperature 0), N forced tokens (ignore_eos + min_tokens).
  python3 serve_bitwise.py URL solo OUT.json      one request at a time (the server's solo path)
  python3 serve_bitwise.py URL conc OUT.json      all 16 at once (a --max-batch 16 server decodes them batched)
  python3 serve_bitwise.py URL conc:N:OFF[:nonce] OUT.json   N of them at once: the prompts sorted by length, every (16 / N)-th
                                                  from OFF (so each subset spans short to 4k); with nonce, a rep tag in front
                                                  (defeats any prefix reuse; the replies then match no solo reply)
  python3 serve_bitwise.py URL reuse-solo OUT.json  lane ink-hip-bpf, prefix reuse: per prompt one at a time, its base request A
                                                  then B (A's text plus a longer instruction): B resumes from A's cached prefix;
                                                  records B
  python3 serve_bitwise.py URL reuse-conc OUT.json  all 16 A at once, then all 16 B at once (B's resumed chunks prefill together)
  python3 serve_bitwise.py URL reuse2-conc[:K] OUT.json  K conversations' first turns at once, then their second turns (first prompt,
                                                  reply, a new user message) at once: those resume from the first prompt's ring
                                                  snapshot (cached tokens > 0) and their resumed chunks prefill together
  python3 serve_bitwise.py URL pp:K:L[:tag] OUT.json   pure prefill: K distinct prompts of about L tokens (pieces of the two 4k
                                                  documents, a tag in front) at once, 1 token each: prompt tok/s = their
                                                  prompt tokens / (last reply - first request)
  python3 serve_bitwise.py compare A.json B.json  per prompt both hold: equal text and token count?
Records each reply's text, md5, completion / prompt / cached token counts; streamed (lane ink-hip-bpf): time to the first token
and the gaps between tokens, per reply."""
import concurrent.futures as cf
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
def _prompts_dir():
    # .scratch/ on tatooine (../scripts), docs/results/<lane>/raw/ in the worktree (../../../../scripts)
    for up in ("..", os.path.join("..", "..", "..", "..")):
        d = os.path.join(HERE, up, "scripts", "cmx", "ids-prompts")
        if os.path.isdir(d):
            return d
    return os.path.join(HERE, "..", "scripts", "cmx", "ids-prompts")


P = _prompts_dir()
N = 128


FIT = os.environ.get("SB_FIT") == "1"


def prompts():
    # lane ink-hip-specflip: SB_FIT=1 keeps every prompt inside the default --max-batch 16 context (3285 tokens on HIP): the two whole
    # 4k documents (~4k tokens) are replaced by two more prefixes, all <= 9000 characters (<= ~2.9k tokens + 128 generated; 4k-1 is ~2.65 characters a token: its 8600-character prefix, 3247 tokens, hit the b16 context in G2b), still 16 prompts
    files = ["short-0", "short-1", "short-2", "mid-0", "mid-1"] + ([] if FIT else ["4k-0", "4k-1"])
    out = [(f, open(os.path.join(P, f + ".txt")).read()) for f in files]
    a, b = (open(os.path.join(P, f + ".txt")).read() for f in ("4k-0", "4k-1"))
    cuts = [700, 1500, 2600, 3300, 4000, 5200, 6000, 7000, 8000, 7600, 9000] if FIT else [700, 1500, 2600, 4000, 6000, 8000, 10000, 9500, 5200]
    for i, n in enumerate(cuts):
        src = a if i % 2 == 0 else b
        out.append((f"{'4k-0' if i % 2 == 0 else '4k-1'}[:{n}]", src[:n]))
    return out


def subset(ps, n, off, nonce):
    lens = sorted(ps, key=lambda kt: len(kt[1]))
    step = max(1, len(lens) // n)
    pick = [lens[(off + i * step) % len(lens)] for i in range(n)]
    return [(k, (f"[{nonce}]\n" + t) if nonce else t) for k, t in pick]


EXT = ("\n\nSummarize the above in a few sentences. Then list three keywords from the text, say which of them matters most and "
       "why, and end with one question a careful reader might still have after reading it.")


TURN2 = "Now list three keywords from the text and say which of them matters most."


def ask(url, text, ext=False, ntok=None, raw=False, msgs=None):
    content = text if raw else text + (EXT if ext else "\n\nSummarize the above in a few sentences.")
    nt = ntok or N
    body = {"model": "ink", "messages": msgs or [{"role": "user", "content": content}],
            "temperature": 0, "max_tokens": nt, "min_tokens": nt, "ignore_eos": True, "stream": True,
            "stream_options": {"include_usage": True}}
    req = urllib.request.Request(url.rstrip("/") + "/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    rs, cs, u, times = [], [], {}, []
    try:
        r0 = urllib.request.urlopen(req, timeout=200)
    except urllib.error.HTTPError as e:
        print(f"HTTP {e.code} for a prompt of {len(content)} characters: {e.read()[:400]!r}", file=sys.stderr)
        raise
    with r0 as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            d = json.loads(data)
            if d.get("usage"):
                u = d["usage"]
            for ch in d.get("choices") or []:
                dl = ch.get("delta") or {}
                got = False
                if dl.get("reasoning_content"):
                    rs.append(dl["reasoning_content"]); got = True
                if dl.get("content"):
                    cs.append(dl["content"]); got = True
                if got:
                    times.append(time.time())
    t1 = time.time()
    text = "".join(rs) + "\x00" + "".join(cs)
    return {"md5": hashlib.md5(text.encode()).hexdigest(), "text": text, "completion_tokens": u.get("completion_tokens"),
            "prompt_tokens": u.get("prompt_tokens"), "cached_tokens": (u.get("prompt_tokens_details") or {}).get("cached_tokens"),
            "t0": t0, "t1": t1, "ttft": (times[0] - t0) if times else None,
            "gaps": [round(b - a, 5) for a, b in zip(times, times[1:])]}


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
    if mode.startswith("pp:"):
        f = mode.split(":")
        k, L, tag = int(f[1]), int(f[2]), (f[3] if len(f) > 3 else "x")
        a, b = (open(os.path.join(P, x + ".txt")).read() for x in ("4k-0", "4k-1"))
        doc = (a + "\n" + b) * 4
        nch = max(8, int((L - 14) * 3.6))  # ~3.6 characters per token on these documents, ~14 tokens of chat template
        ps = [(f"pp{i}", f"[{tag}-{k}x{L}-{i}] " + doc[(i * 977) % (len(doc) - nch - 1):][:nch]) for i in range(k)]
        with cf.ThreadPoolExecutor(k) as ex:
            futs = {key: ex.submit(ask, url, t, False, 1, True) for key, t in ps}
            res = {key: f.result() for key, f in futs.items()}
        t1 = max(r["t1"] for r in res.values())
        pt = sum(r["prompt_tokens"] or 0 for r in res.values())
        meta = {"mode": mode, "n": k, "t0": t0, "t1": t1, "wall_s": round(t1 - t0, 3), "prompt_tokens": pt,
                "prompt_tok_s": round(pt / (t1 - t0), 1), "cached_tokens": sum(r["cached_tokens"] or 0 for r in res.values()),
                "prompt_tokens_each": sorted(r["prompt_tokens"] or 0 for r in res.values()),
                "ttft_s_median": sorted(r["t1"] - r["t0"] for r in res.values())[k // 2]}
        json.dump(res, open(out, "w"), indent=1)
        json.dump(meta, open(out[:-5] + ".meta.json", "w"), indent=1)
        print(f"pp {k} x {L}: wall {meta['wall_s']} s, prompt tokens {pt} ({meta['prompt_tokens_each'][0]}..{meta['prompt_tokens_each'][-1]}), "
              f"prompt {meta['prompt_tok_s']} tok/s, cached {meta['cached_tokens']}")
        return
    workers = len(ps)
    if mode.startswith("pool:"):
        # all 16, at most N in flight (a worker takes the next prompt as its reply ends)
        workers = int(mode.split(":")[1])
    if mode.startswith("reuse-") or mode.startswith("reuse2-"):
        # reuse-solo[:K] / reuse-conc[:K]: K of the prompts spread by length (default all)
        f = mode.split(":")
        if len(f) > 1:
            ps = subset(ps, int(f[1]), 0, "")
        mode = f[0]
    if mode == "solo":
        for k, t in ps:
            res[k] = ask(url, t)
    elif mode == "reuse-solo":
        for k, t in ps:
            ask(url, t)
            res[k] = ask(url, t, ext=True)
    elif mode == "reuse2-conc":
        # lane ink-hip-bpf: a second turn on each conversation (its first prompt, the reply, a new user message), all at once: the
        # second request starts with the first's whole prompt, so a free slot holding it resumes there (ring snapshot) and the
        # resumed chunks prefill together; records the second turn
        with cf.ThreadPoolExecutor(len(ps)) as ex:
            first = dict(zip([k for k, _ in ps], [f.result() for f in [ex.submit(ask, url, t) for k, t in ps]]))
        t0 = time.time()
        conv = {k: [{"role": "user", "content": t + "\n\nSummarize the above in a few sentences."},
                    {"role": "assistant", "content": first[k]["text"].split("\x00", 1)[1]},
                    {"role": "user", "content": TURN2}] for k, t in ps}
        with cf.ThreadPoolExecutor(len(ps)) as ex:
            futs = {k: ex.submit(ask, url, "", False, None, False, conv[k]) for k, _ in ps}
            res = {k: f.result() for k, f in futs.items()}
    elif mode == "reuse-conc":
        with cf.ThreadPoolExecutor(len(ps)) as ex:
            [f.result() for f in [ex.submit(ask, url, t) for k, t in ps]]
        t0 = time.time()
        with cf.ThreadPoolExecutor(len(ps)) as ex:
            futs = {k: ex.submit(ask, url, t, True) for k, t in ps}
            res = {k: f.result() for k, f in futs.items()}
    else:
        with cf.ThreadPoolExecutor(workers) as ex:
            futs = {k: ex.submit(ask, url, t) for k, t in ps}
            res = {k: f.result() for k, f in futs.items()}
    t1 = time.time()
    wall = t1 - t0
    json.dump(res, open(out, "w"), indent=1)
    toks = sum(r["completion_tokens"] or 0 for r in res.values())
    lat = sorted(r["t1"] - r["t0"] for r in res.values())
    ttft = sorted(r["ttft"] for r in res.values() if r.get("ttft") is not None)
    gaps = sorted(g for r in res.values() for g in r.get("gaps") or [])
    pct = lambda xs, q: round(xs[min(len(xs) - 1, int(q * len(xs)))], 4) if xs else None
    meta = {"mode": mode, "n": len(res), "t0": t0, "t1": t1, "wall_s": round(wall, 3), "completion_tokens": toks,
            "prompt_tokens": sum(r["prompt_tokens"] or 0 for r in res.values()), "aggregate_tok_s": round(toks / wall, 1),
            "latency_s_median": round(lat[len(lat) // 2], 2), "latency_s_max": round(lat[-1], 2),
            "cached_tokens": sum(r["cached_tokens"] or 0 for r in res.values()),
            "ttft_s_median": pct(ttft, 0.5), "ttft_s_max": ttft[-1] if ttft else None,
            "itl_s_p50": pct(gaps, 0.5), "itl_s_p99": pct(gaps, 0.99)}
    json.dump(meta, open(out[:-5] + ".meta.json" if out.endswith(".json") else out + ".meta", "w"), indent=1)
    print(f"wall {wall:.2f} s, aggregate {meta['aggregate_tok_s']} tok/s, latency median {meta['latency_s_median']} s max "
          f"{meta['latency_s_max']} s, prompt tokens {meta['prompt_tokens']}, cached {meta['cached_tokens']}, ttft median "
          f"{meta['ttft_s_median']} max {meta['ttft_s_max']}, itl p50 {meta['itl_s_p50']} p99 {meta['itl_s_p99']}")
    print(f"{mode}: {len(res)} replies, {sum(r['completion_tokens'] or 0 for r in res.values())} tokens -> {out}")


if __name__ == "__main__":
    main()
