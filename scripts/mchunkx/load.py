#!/usr/bin/env python3
"""Lane msmallm speed client: N closed-loop streams against a running `intarsia serve`, each sending R streamed /v1/completions
requests one after another (raw prompt: a distinct slice of a long text per request, so no prefix-cache hit; greedy; ignore_eos,
exactly --gen tokens). Writes one JSON with per-request send / first-token / last-token times and token counts, and prints:
  aggregate tok/s = all completion tokens / (last token of the cell - first request sent)   <- the lane's decision metric
  TTFT p50 / mean, per-request decode tok/s p50 (tokens after the first / (last - first token)).
  load.py --url http://127.0.0.1:18673/v1 --text prompt-16384.txt --chars 430 --streams 4 --requests 2 --gen 256 --tag x --out f.json"""
import argparse, http.client, json, statistics, threading, time, urllib.parse


def one(url, model, prompt, gen):
    u = urllib.parse.urlsplit(url.rstrip("/") + "/completions")
    body = {"model": model, "prompt": prompt, "max_tokens": gen, "min_tokens": gen, "ignore_eos": True, "temperature": 0.0,
            "stream": True, "stream_options": {"include_usage": True}}
    c = http.client.HTTPConnection(u.hostname, u.port, timeout=1800)
    t0 = time.time()
    c.request("POST", u.path, body=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    r = c.getresponse()
    first = last = None
    usage = {}
    for raw in r:
        line = raw.decode("utf-8", "replace").strip()
        if not line.startswith("data:"):
            continue
        p = line[5:].strip()
        if p == "[DONE]":
            break
        ev = json.loads(p)
        if ev.get("usage"):
            usage = ev["usage"]
        if any(ch.get("text") for ch in ev.get("choices", [])):
            now = time.time()
            first = first or now
            last = now
    return {"send": t0, "first": first, "last": last, "prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": usage.get("completion_tokens")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", default="intarsia")
    ap.add_argument("--text", required=True)
    ap.add_argument("--chars", type=int, default=430)
    ap.add_argument("--offset", type=int, default=0, help="first character of the first request's slice")
    ap.add_argument("--streams", type=int, default=1)
    ap.add_argument("--requests", type=int, default=2)
    ap.add_argument("--gen", type=int, default=256)
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    text = open(a.text).read()
    res = [[None] * a.requests for _ in range(a.streams)]

    def stream(s):
        for j in range(a.requests):
            i = s * a.requests + j
            off = (a.offset + 997 * i) % max(1, len(text) - a.chars)
            res[s][j] = one(a.url, a.model, text[off: off + a.chars], a.gen)

    th = [threading.Thread(target=stream, args=(s,)) for s in range(a.streams)]
    [t.start() for t in th]
    [t.join() for t in th]
    reqs = [r for row in res for r in row]
    t0 = min(r["send"] for r in reqs)
    t1 = max(r["last"] for r in reqs)
    toks = sum(r["completion_tokens"] or 0 for r in reqs)
    ttft = [r["first"] - r["send"] for r in reqs]
    dec = [(r["completion_tokens"] - 1) / (r["last"] - r["first"]) for r in reqs if r["last"] > r["first"]]
    summ = {"tag": a.tag, "streams": a.streams, "requests": len(reqs), "tokens": toks, "wall_s": t1 - t0, "agg_tps": toks / (t1 - t0),
            "ttft_p50": statistics.median(ttft), "ttft_mean": statistics.mean(ttft), "dec_tps_p50": statistics.median(dec),
            "prompt_tokens_p50": statistics.median(r["prompt_tokens"] or 0 for r in reqs)}
    json.dump({"summary": summ, "requests": reqs}, open(a.out, "w"))
    print("CELL " + " ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in summ.items()), flush=True)


if __name__ == "__main__":
    main()
