#!/usr/bin/env python3
"""Lane hprof workload driver (a profiling workload, not a speed benchmark: the unprofiled tok/s come from `localeval speed`).

Phases run one after another with a pause between them (the pause is how the profile is split into phases afterwards, scripts/hprof/analyze.py):
  b1      1 stream, one diverse prompt
  same<N> N streams, every prompt identical apart from a nonce in front (what `localeval speed --concurrency N` sends: filler text)
  div<N>  N streams, N different ~1k-token prompts cut from different repo docs (distinct content, so the streams route differently)
Each stream: ~1k prompt tokens, `gen` forced output tokens (ignore_eos + min_tokens), temperature 0, thinking off. Prints one JSON line
per phase to stdout. Run with the deepeval env so localeval.speed.filler is importable:
  cd /opt/code/deepeval && uv run python /path/to/drive.py --url http://tatooine:8090/v1 --model qwen36-35b --docs /path/to/docs --gen 384
"""
import argparse, glob, http.client, json, os, threading, time, urllib.parse, uuid

ap = argparse.ArgumentParser()
ap.add_argument("--url", required=True)
ap.add_argument("--model", required=True)
ap.add_argument("--docs", required=True, help="directory with the *.md files the diverse prompts are cut from")
ap.add_argument("--gen", type=int, default=384)
ap.add_argument("--phases", default="b1,same8,same16,div8,div16")
ap.add_argument("--pause", type=float, default=4.0)
ap.add_argument("--chars", type=int, default=3500, help="characters of doc text per diverse prompt (~1k tokens)")
a = ap.parse_args()

from localeval.speed import INSTRUCTION, filler  # noqa: E402

u = urllib.parse.urlsplit(a.url.rstrip("/") + "/chat/completions")


def chat(content, max_tokens, nonce=True):
    if nonce:
        content = f"nonce {uuid.uuid4().hex} " + content
    body = {"model": a.model, "messages": [{"role": "user", "content": content}], "max_tokens": max_tokens, "temperature": 0.0,
            "stream": True, "stream_options": {"include_usage": True}, "chat_template_kwargs": {"enable_thinking": False}}
    if max_tokens > 1:
        body.update({"ignore_eos": True, "min_tokens": max_tokens})
    c = http.client.HTTPConnection(u.hostname, u.port, timeout=600)
    c.request("POST", u.path, body=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    r = c.getresponse()
    out = {"ok": r.status == 200, "ttft": None, "prompt": None, "completion": None}
    if r.status != 200:
        out["err"] = r.read().decode()[:300]
        return out
    for raw in r:
        line = raw.decode("utf-8", "replace").strip()
        if not line.startswith("data:") or line.endswith("[DONE]"):
            continue
        d = json.loads(line[5:])
        if out["ttft"] is None and d.get("choices") and d["choices"][0].get("delta", {}).get("content"):
            out["ttft"] = time.perf_counter() - t0
        if d.get("usage"):
            out["prompt"], out["completion"] = d["usage"]["prompt_tokens"], d["usage"]["completion_tokens"]
    out["t_total"] = time.perf_counter() - t0
    return out


# token calibration like localeval: sentences needed for ~1024 prompt tokens
base = chat(INSTRUCTION.strip(), 1)
cal = chat(filler(200, 0) + INSTRUCTION, 1)
per = (cal["prompt"] - base["prompt"]) / 200
n_sent = max(0, round((1024 - base["prompt"]) / per))
same_prompt = filler(n_sent, seed=1024) + INSTRUCTION

files = sorted(f for f in glob.glob(os.path.join(a.docs, "**", "*.md"), recursive=True) if os.path.getsize(f) > 20000)
diverse = []
for i, f in enumerate(files):
    t = open(f, errors="replace").read()
    s = (len(t) // 3) % max(1, len(t) - a.chars)
    diverse.append(t[s:s + a.chars] + "\n\nSummarize the text above and continue it in detail.")
assert len(diverse) >= 16, f"only {len(diverse)} docs with >20 kB under {a.docs}"
print(json.dumps({"calibration": {"base_prompt_tokens": base["prompt"], "per_sentence": per, "same_sentences": n_sent, "diverse_docs": len(diverse), "diverse_files": [os.path.relpath(f, a.docs) for f in files[:16]]}}), flush=True)

for ph in a.phases.split(","):
    if ph == "b1":
        prompts = diverse[:1]
    elif ph.startswith("same"):
        prompts = [same_prompt] * int(ph[4:])
    elif ph.startswith("div"):
        prompts = diverse[: int(ph[3:])]
    else:
        raise SystemExit(f"bad phase {ph}")
    res, lock = [], threading.Lock()

    def one(p):
        r = chat(p, a.gen)
        with lock:
            res.append(r)

    t0 = time.perf_counter()
    th = [threading.Thread(target=one, args=(p,)) for p in prompts]
    [t.start() for t in th]
    [t.join() for t in th]
    wall = time.perf_counter() - t0
    ok = [r for r in res if r["ok"]]
    toks = sum(r["completion"] or 0 for r in ok)
    # decode-only aggregate: tokens after each stream's first token over the span from the last first-token to the end
    last_first = max((r["ttft"] or 0) for r in ok) if ok else 0
    dec = [(r["completion"] - 1) / (r["t_total"] - r["ttft"]) for r in ok if r["ttft"] and r["completion"] and r["t_total"] > r["ttft"]]
    print(json.dumps({"phase": ph, "streams": len(prompts), "ok": len(ok), "wall_s": round(wall, 2), "completion_tokens": toks,
                      "prompt_tokens_median": sorted(r["prompt"] for r in ok)[len(ok) // 2] if ok else None,
                      "ttft_max_s": round(last_first, 2), "per_stream_decode_tok_s_mean": round(sum(dec) / len(dec), 1) if dec else None,
                      "aggregate_decode_tok_s_est": round(sum(dec), 1) if dec else None,
                      "errors": [r.get("err") for r in res if not r["ok"]][:2]}), flush=True)
    time.sleep(a.pause)
