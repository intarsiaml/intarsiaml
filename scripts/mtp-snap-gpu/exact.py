#!/usr/bin/env python3
"""Lane mtp-snap-gpu: exactness client for PR #395 (MTP inner snapshots under macro prefill). stdlib + tokenizers only.

  exact.py prompts --tokenizer tokenizer.json --out prompts.json     deterministic token-id prompts (real repo text)
  exact.py send --url http://tatooine:8090 --prompts prompts.json --scenario c2048 --which A|B --max-tokens N --out FILE
  exact.py table --dir DIR                                           warm vs cold comparison (markdown) from the send outputs

Prompts go to /v1/completions as token-id arrays (temperature 0, return_token_ids), so prompt lengths and the shared prefix are exact. A and B of a
scenario share their first `shared` tokens and differ from there. `send` records usage (prompt, completion, cached tokens), the output ids, and the
/metrics counters before and after the request (speculative steps, proposed, accepted draft tokens; the server is otherwise idle).
Output files are named <bin>-<scenario>-<rule>-warm-A|B.json and <bin>-<scenario>-cold-B.json.
"""
import argparse, json, pathlib, re, sys, time, urllib.request

SCEN = {  # name: (shared prefix tokens, A total, B total)
    "c2048": (2100, 4500, 3000),  # --prefill-chunk 2048: chunk ends of A at 2048 and 4096 inside one macro pass of 8192
    "c4096": (4300, 8700, 5300),  # the HIP recipe's chunk 4096: A's macro pass is 8192 rows, inner chunk end at 4096
}
METRICS = ["intarsia_spec_steps_total", "intarsia_spec_proposed_tokens_total", "intarsia_spec_accepted_tokens_total", "intarsia_tokens_cached_total"]


def prompts(a):
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(a.tokenizer)
    root = pathlib.Path(__file__).resolve().parents[2]
    def ids(p): return tok.encode((root / p).read_text(), add_special_tokens=False).ids
    s1, s2 = ids("docs/tuning.md"), ids("docs/results/2026-11-16-hip-a4k-restore.md")
    out = {}
    for name, (shared, na, nb) in SCEN.items():
        A = s1[:na]
        B = s1[:shared] + s2[:nb - shared]
        assert len(A) == na and len(B) == nb, (name, len(A), len(B))
        assert A[shared] != B[shared], "prompts must diverge at the shared length"
        out[name] = {"shared": shared, "A": A, "B": B}
    pathlib.Path(a.out).write_text(json.dumps(out))
    for k, v in out.items():
        print(k, "shared", v["shared"], "A", len(v["A"]), "B", len(v["B"]))


def metrics(url):
    t = urllib.request.urlopen(url + "/metrics", timeout=10).read().decode()
    r = {}
    for m in METRICS:
        mm = re.search(rf"^{m}(?:{{[^}}]*}})? ([0-9.eE+-]+)$", t, re.M)
        r[m] = float(mm.group(1)) if mm else None
    return r


def send(a):
    P = json.loads(pathlib.Path(a.prompts).read_text())[a.scenario]
    body = {"model": "qwen36-35b", "prompt": P[a.which], "max_tokens": a.max_tokens, "temperature": 0, "return_token_ids": True}
    m0 = metrics(a.url)
    t0 = time.time()
    req = urllib.request.Request(a.url + "/v1/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    resp = json.loads(urllib.request.urlopen(req, timeout=600).read())
    dt = time.time() - t0
    m1 = metrics(a.url)
    u = resp["usage"]
    ch = resp["choices"][0]
    d = {k: (m1[k] - m0[k]) if m0[k] is not None else None for k in METRICS}
    rec = {"scenario": a.scenario, "which": a.which, "prompt_tokens": u["prompt_tokens"], "completion_tokens": u["completion_tokens"],
           "cached_tokens": (u.get("prompt_tokens_details") or {}).get("cached_tokens"), "finish": ch.get("finish_reason"),
           "ids": ch.get("token_ids"), "text": ch.get("text"), "wall_s": round(dt, 3), "metrics_delta": d, "usage_raw": u,
           "prefill_chunk_reported": resp.get("prefill_chunk")}
    pathlib.Path(a.out).write_text(json.dumps(rec))
    print(a.which, a.scenario, "prompt", rec["prompt_tokens"], "cached", rec["cached_tokens"], "completion", rec["completion_tokens"], "spec delta", d)


def table(a):
    d = pathlib.Path(a.dir)
    recs = {}
    for f in sorted(d.glob("*.json")):
        if f.name == "prompts.json":
            continue
        recs[f.stem] = json.loads(f.read_text())

    def acc(r):
        m = r["metrics_delta"]
        p, ac, s = m["intarsia_spec_proposed_tokens_total"], m["intarsia_spec_accepted_tokens_total"], m["intarsia_spec_steps_total"]
        return (f"{ac:.0f}/{p:.0f} = {100*ac/p:.1f}%" if p else "n/a", f"{r['completion_tokens']/s:.3f}" if s else "n/a")

    print("| arm | scenario | rule | B cached | B prompt | warm ids == cold ids | first diff | acceptance warm | acceptance cold | tok/step warm | tok/step cold |")
    print("|---|---|---|---:|---:|---|---|---|---|---|---|")
    for k in sorted(recs):
        m = re.match(r"(.+)-warm-B$", k)
        if not m:
            continue
        pb, sc, rule = m.group(1).split("-")
        w = recs[k]
        cold = recs.get(f"{pb}-{sc}-cold-B")
        if cold is None:
            print(f"| {pb} | {sc} | {rule} | {w['cached_tokens']} | {w['prompt_tokens']} | no cold run | | {acc(w)[0]} | | {acc(w)[1]} | |")
            continue
        n = min(len(w["ids"]), len(cold["ids"]))
        diff = next((i for i in range(n) if w["ids"][i] != cold["ids"][i]), None)
        eq = w["ids"] == cold["ids"]
        print(f"| {pb} | {sc} | {rule} | {w['cached_tokens']} | {w['prompt_tokens']} | {'yes (%d ids)' % len(w['ids']) if eq else 'NO'} | "
              f"{'-' if eq else diff} | {acc(w)[0]} | {acc(cold)[0]} | {acc(w)[1]} | {acc(cold)[1]} |")


ap = argparse.ArgumentParser()
sp = ap.add_subparsers(dest="cmd", required=True)
p = sp.add_parser("prompts"); p.add_argument("--tokenizer", required=True); p.add_argument("--out", required=True); p.set_defaults(f=prompts)
p = sp.add_parser("send"); p.add_argument("--url", required=True); p.add_argument("--prompts", required=True); p.add_argument("--scenario", required=True)
p.add_argument("--which", required=True); p.add_argument("--max-tokens", type=int, default=128); p.add_argument("--out", required=True); p.set_defaults(f=send)
p = sp.add_parser("table"); p.add_argument("--dir", required=True); p.set_defaults(f=table)
a = ap.parse_args()
a.f(a)
