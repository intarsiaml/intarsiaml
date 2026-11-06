#!/usr/bin/env python3
"""Lane fh: per-stream TTFT and greedy output of N concurrent streams, each with its own ~1k-token prompt (stdlib only).

  ttft.py --url http://tatooine:8296/v1 --model intarsia --streams 1,4,8,16 --bursts 3 --gen 64 --out out.json

localeval's concurrency rows give only a TTFT median / min / max, so the p50 / p95 over every stream of every burst come from here
(the aggregate tok/s come from localeval). Prompts are deterministic: stream i of burst b is the same text on every server start, so
two arms can be compared text for text (temperature 0, ignore_eos, min_tokens = gen). Each prompt starts with a nonce unique to
(streams, burst, stream), so no burst reuses another's prefix. The text is hashed (sha256) as the stand-in for the greedy ids: the
server returns text, and at temperature 0 equal text means equal ids to within a detokenizer round trip.
After every burst the server's /metrics gives intarsia_batch_pack_passes_total / packed_chunks_total (did the packed pass run?).
"""
import argparse, hashlib, http.client, json, os, statistics, sys, threading, time, urllib.parse

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "hg"))
from stall import filler  # noqa: E402  (lane hg's prompt generator, same text as localeval's filler)


def one(url, model, prompt, gen, res, i):
    u = urllib.parse.urlparse(url)
    c = http.client.HTTPConnection(u.hostname, u.port, timeout=900)
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": gen, "temperature": 0, "stream": True,
            "ignore_eos": True, "min_tokens": gen, "chat_template_kwargs": {"enable_thinking": False}}
    t0 = time.perf_counter()
    c.request("POST", u.path.rstrip("/") + "/chat/completions", json.dumps(body), {"Content-Type": "application/json"})
    r = c.getresponse()
    ttft, text, n = None, [], 0
    for line in r:
        if line.startswith(b"data: {"):
            d = json.loads(line[6:])
            ch = d.get("choices") or []
            piece = (ch[0].get("delta") or {}).get("content") if ch else None
            if piece:
                if ttft is None:
                    ttft = time.perf_counter() - t0
                text.append(piece)
                n += 1
    res[i] = {"ttft_s": ttft, "chunks": n, "end_s": time.perf_counter() - t0, "sha": hashlib.sha256("".join(text).encode()).hexdigest()[:16]}


def metrics(url):
    u = urllib.parse.urlparse(url)
    try:
        c = http.client.HTTPConnection(u.hostname, u.port, timeout=10)
        c.request("GET", "/metrics")
        t = c.getresponse().read().decode()
    except Exception:
        return {}
    out = {}
    for line in t.splitlines():
        for k in ("intarsia_batch_pack_passes_total", "intarsia_batch_packed_chunks_total"):
            if line.startswith(k + " "):
                out[k] = float(line.split()[1])
    return out


def pct(v, p):
    v = sorted(v)
    return v[min(len(v) - 1, round(p / 100 * (len(v) - 1)))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True); ap.add_argument("--model", default="intarsia")
    ap.add_argument("--streams", default="1,4,8,16"); ap.add_argument("--bursts", type=int, default=3)
    ap.add_argument("--gen", type=int, default=64); ap.add_argument("--sentences", type=int, default=45)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    result = {"url": a.url, "gen": a.gen, "rows": []}
    # one discarded warm-up burst: first-use kernel / graph setup
    w = [None]; one(a.url, a.model, f"[warm] " + filler(a.sentences, 999), 8, w, 0)
    for n in [int(x) for x in a.streams.split(",")]:
        ttfts, shas = [], {}
        m0 = metrics(a.url)
        for b in range(a.bursts):
            res = [None] * n
            ts = [threading.Thread(target=one, args=(a.url, a.model, f"[{n}-{b}-{i}] " + filler(a.sentences, 1000 * n + i), a.gen, res, i)) for i in range(n)]
            [t.start() for t in ts]; [t.join() for t in ts]
            ttfts += [r["ttft_s"] for r in res if r and r["ttft_s"] is not None]
            shas[b] = [r["sha"] if r else None for r in res]
            time.sleep(1)
        m1 = metrics(a.url)
        row = {"streams": n, "bursts": a.bursts, "ttft_p50": round(statistics.median(ttfts), 3), "ttft_p95": round(pct(ttfts, 95), 3),
               "ttft_max": round(max(ttfts), 3), "n": len(ttfts), "sha": shas,
               "pack_passes": m1.get("intarsia_batch_pack_passes_total", 0) - m0.get("intarsia_batch_pack_passes_total", 0),
               "packed_chunks": m1.get("intarsia_batch_packed_chunks_total", 0) - m0.get("intarsia_batch_packed_chunks_total", 0)}
        result["rows"].append(row)
        print(f"streams {n}: ttft p50 {row['ttft_p50']} p95 {row['ttft_p95']} max {row['ttft_max']} (n={row['n']}), packed passes {row['pack_passes']:.0f}, packed chunks {row['packed_chunks']:.0f}", flush=True)
    json.dump(result, open(a.out, "w"), indent=1)


main()
