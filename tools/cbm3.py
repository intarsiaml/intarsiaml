#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""M3 gate client (docs/research/continuous-batching-design.md section 9, row M3): speculation under batching.

Streams /v1/chat/completions (greedy, thinking off) against `intarsia serve --max-batch 8 [--spec ngram]` and measures
decode throughput per scenario. Prompts: tools/prompts/m3_grounded.txt (G1's input-grounded Rust edit) and
tools/prompts/m3_open.txt (G1's open-ended story), the same two prompts as docs/results/2026-10-15-metal-spec-g1.md section 5.

  cbm3.py run --url U --arm on --round 1 --out DIR [--scenarios solo-grounded,solo-open,mixed-once,mixed-loop,grounded-n2,..]
        warm-up (unrecorded: every scenario shape once, so kernels are compiled) then one measured pass per scenario
        -> DIR/m3-<arm>-<round>.json
  cbm3.py analyze --dir DIR [--md OUT.md]
        per scenario: median / mean over rounds per arm, paired ratio (arm A / arm B) per round (--num on --den off)

Scenarios (all measured as tokens the server reports in `usage` over wall time):
  solo-grounded / solo-open   B = 1, `--reps` sequential requests, 256 tokens; decode tok/s = (n - 1) / (t_last - t_first)
  grounded-nK / open-nK       K concurrent identical-shape streams (each with its own nonce line), aggregate tok/s
  mixed-once                  1 grounded + 3 open-ended start together (256 tokens each)
  mixed-loop                  3 open-ended streams (640 tokens) + one client that sends grounded requests back to back
                              until the open ones are done (B stays 3..4); counts the tokens of finished requests
"""
import argparse
import http.client
import json
import pathlib
import statistics
import sys
import threading
import time
import urllib.parse
import uuid

HERE = pathlib.Path(__file__).resolve().parent
PROMPTS = HERE / "prompts"
SOLO_TOKENS = 256
LOOP_OPEN_TOKENS = 640


def split_url(url):
    u = urllib.parse.urlparse(url)
    return u.hostname, u.port or 80


def get_json(url, path, timeout=10):
    host, port = split_url(url)
    conn = http.client.HTTPConnection(host, port, timeout=timeout)
    try:
        conn.request("GET", path)
        r = conn.getresponse()
        body = r.read()
        return json.loads(body) if r.status == 200 else None
    except Exception:  # noqa: BLE001
        return None
    finally:
        conn.close()


class Req(threading.Thread):
    """One streaming chat request; `t_first` / `t_last` bracket the content chunks, `usage` is the server's token count."""

    def __init__(self, url, model, kind, max_tokens, ignore_eos=False, tag=""):
        super().__init__(daemon=True)
        self.url, self.kind, self.max_tokens = url, kind, max_tokens
        text = (PROMPTS / ("m3_grounded.txt" if kind == "grounded" else "m3_open.txt")).read_text().strip()
        # nonce line first: no two requests share a prefix (no KV reuse between them); the task text is G1's
        content = f"Request {uuid.uuid4().hex[:12]} {tag}.\n\n{text}"
        self.body = {"model": model, "messages": [{"role": "user", "content": content}], "max_tokens": max_tokens,
                     "temperature": 0, "stream": True, "stream_options": {"include_usage": True},
                     "chat_template_kwargs": {"enable_thinking": False}}
        if ignore_eos:
            self.body["ignore_eos"] = True
        self.t_send = self.t_first = self.t_last = self.t_end = None
        self.chunks, self.usage, self.error, self.finish = 0, None, None, None

    def run(self):
        host, port = split_url(self.url)
        conn = http.client.HTTPConnection(host, port, timeout=300)
        try:
            self.t_send = time.time()
            conn.request("POST", "/v1/chat/completions", body=json.dumps(self.body).encode(),
                         headers={"Content-Type": "application/json", "Connection": "close"})
            resp = conn.getresponse()
            if resp.status != 200:
                self.error = f"HTTP {resp.status}: {resp.read(300)!r}"
                return
            done = False
            while True:
                line = resp.readline()
                if not line:
                    break
                line = line.strip()
                if not line.startswith(b"data:"):
                    continue
                data = line[5:].strip()
                if data == b"[DONE]":
                    done = True
                    break
                obj = json.loads(data)
                if "error" in obj:
                    self.error = json.dumps(obj["error"])[:300]
                    break
                if obj.get("usage"):
                    self.usage = obj["usage"]
                for ch in obj.get("choices") or []:
                    if (ch.get("delta") or {}).get("content"):
                        now = time.time()
                        self.t_first = self.t_first or now
                        self.t_last = now
                        self.chunks += 1
                    if ch.get("finish_reason"):
                        self.finish = ch["finish_reason"]
            if not done and not self.error:
                self.error = "stream ended without [DONE]"
        except Exception as e:  # noqa: BLE001
            self.error = repr(e)[:300]
        finally:
            self.t_end = time.time()
            conn.close()

    @property
    def n(self):
        return (self.usage or {}).get("completion_tokens", 0)

    def decode_tok_s(self):
        if self.error or self.t_first is None or self.t_last is None or self.t_last <= self.t_first or self.n < 2:
            return None
        return (self.n - 1) / (self.t_last - self.t_first)

    def rec(self):
        return {"kind": self.kind, "max_tokens": self.max_tokens, "t_send": self.t_send, "t_first": self.t_first,
                "t_last": self.t_last, "t_end": self.t_end, "tokens": self.n, "chunks": self.chunks,
                "prompt_tokens": (self.usage or {}).get("prompt_tokens"), "finish": self.finish, "error": self.error,
                "decode_tok_s": self.decode_tok_s()}


def agg(reqs, t0, t1):
    toks = sum(r.n for r in reqs)
    return {"tokens": toks, "wall_s": t1 - t0, "agg_tok_s": toks / (t1 - t0) if t1 > t0 else None}


def run_group(url, model, spec):
    """spec: list of (kind, max_tokens, ignore_eos). Concurrent start; returns (reqs, t0, t1)."""
    reqs = [Req(url, model, k, m, ie, tag=f"s{i}") for i, (k, m, ie) in enumerate(spec)]
    t0 = time.time()
    for r in reqs:
        r.start()
    for r in reqs:
        r.join(400)
    return reqs, t0, max(r.t_end or t0 for r in reqs)


def sc_solo(url, model, kind, reps):
    out = []
    for _ in range(reps):
        reqs, t0, t1 = run_group(url, model, [(kind, SOLO_TOKENS, kind == "open")])
        out.append(reqs[0].rec())
    rates = [o["decode_tok_s"] for o in out if o["decode_tok_s"]]
    return {"reqs": out, "decode_tok_s_median": statistics.median(rates) if rates else None,
            "decode_tok_s_mean": statistics.mean(rates) if rates else None}


def sc_group(url, model, kind, k):
    reqs, t0, t1 = run_group(url, model, [(kind, SOLO_TOKENS, kind == "open")] * k)
    return {"reqs": [r.rec() for r in reqs], **agg(reqs, t0, t1)}


def sc_mixed_once(url, model):
    reqs, t0, t1 = run_group(url, model, [("grounded", SOLO_TOKENS, False)] + [("open", SOLO_TOKENS, True)] * 3)
    op = [r.decode_tok_s() for r in reqs[1:] if r.decode_tok_s()]
    return {"reqs": [r.rec() for r in reqs], **agg(reqs, t0, t1),
            "open_decode_tok_s_mean": statistics.mean(op) if op else None,
            "grounded_wall_s": reqs[0].t_end - reqs[0].t_send if reqs[0].t_end else None}


def sc_mixed_loop(url, model):
    opens = [Req(url, model, "open", LOOP_OPEN_TOKENS, True, tag=f"o{i}") for i in range(3)]
    done_grounded, stop = [], threading.Event()

    def looper():
        i = 0
        while not stop.is_set():
            r = Req(url, model, "grounded", SOLO_TOKENS, False, tag=f"g{i}")
            r.start()
            r.join(400)
            done_grounded.append(r)
            i += 1

    t0 = time.time()
    for r in opens:
        r.start()
    th = threading.Thread(target=looper, daemon=True)
    th.start()
    for r in opens:
        r.join(400)
    t1 = max(r.t_end or t0 for r in opens)
    stop.set()
    th.join(60)
    fin = [g for g in done_grounded if g.t_end and g.t_end <= t1 and not g.error]
    allr = opens + fin
    op = [r.decode_tok_s() for r in opens if r.decode_tok_s()]
    return {"reqs": [r.rec() for r in allr], **agg(allr, t0, t1), "grounded_finished": len(fin),
            "grounded_tokens": sum(g.n for g in fin), "open_decode_tok_s_mean": statistics.mean(op) if op else None}


SCENARIOS = {
    "solo-grounded": lambda u, m, a: sc_solo(u, m, "grounded", a.reps),
    "solo-open": lambda u, m, a: sc_solo(u, m, "open", a.reps),
    "mixed-once": lambda u, m, a: sc_mixed_once(u, m),
    "mixed-loop": lambda u, m, a: sc_mixed_loop(u, m),
}
for _k in (2, 3, 4):
    SCENARIOS[f"grounded-n{_k}"] = (lambda k: lambda u, m, a: sc_group(u, m, "grounded", k))(_k)
    SCENARIOS[f"open-n{_k}"] = (lambda k: lambda u, m, a: sc_group(u, m, "open", k))(_k)
DEFAULT = "solo-grounded,solo-open,mixed-once,mixed-loop"


def counters(url):
    """The speculation / batching counters of /metrics (cumulative since server start)."""
    host, port = split_url(url)
    conn = http.client.HTTPConnection(host, port, timeout=10)
    try:
        conn.request("GET", "/metrics")
        txt = conn.getresponse().read().decode()
    except Exception:  # noqa: BLE001
        return {}
    finally:
        conn.close()
    out = {}
    for ln in txt.splitlines():
        if ln.startswith(("intarsia_spec_", "intarsia_batch_decode_steps_total")) and " " in ln and not ln.startswith("#"):
            k, v = ln.rsplit(" ", 1)
            try:
                out[k] = float(v)
            except ValueError:
                pass
    return out


def cmd_run(a):
    j = get_json(a.url, "/v1/models")
    model = (j or {}).get("data", [{}])[0].get("id", "m")
    names = a.scenarios.split(",")
    for n in names:
        if n not in SCENARIOS:
            sys.exit(f"unknown scenario {n}")
    t_all = time.time()
    # warm-up: each scenario shape once, no gap before the measured pass (weights wired, kernels compiled)
    for n in names:
        SCENARIOS[n](a.url, model, argparse.Namespace(reps=1))
    res = {"arm": a.arm, "round": a.round, "url": a.url, "model": model, "t_start": t_all, "scenarios": {}}
    for n in names:
        c0 = counters(a.url)
        res["scenarios"][n] = SCENARIOS[n](a.url, model, a)
        c1 = counters(a.url)
        s = res["scenarios"][n]
        s["counters_delta"] = {k: c1[k] - c0.get(k, 0) for k in c1}
        print(f"[cbm3] {a.arm} r{a.round} {n}: " + ", ".join(f"{k}={v:.1f}" for k, v in s.items()
                                                              if isinstance(v, float))
              + "  " + " ".join(f"{k.replace('intarsia_', '')}={v:.0f}" for k, v in s["counters_delta"].items() if v),
              flush=True)
    res["t_end"] = time.time()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"m3-{a.arm}-{a.round}.json").write_text(json.dumps(res, indent=1))
    errs = [r["error"] for s in res["scenarios"].values() for r in s["reqs"] if r["error"]]
    if errs:
        print("[cbm3] ERRORS:", errs[:3])
        return 1
    return 0


def metric(s):
    if "decode_tok_s_median" in s:
        return s["decode_tok_s_median"]
    return s.get("agg_tok_s")


def cmd_analyze(a):
    d = pathlib.Path(a.dir)
    runs = {}
    for f in sorted(d.glob("m3-*.json")):
        r = json.loads(f.read_text())
        runs.setdefault(r["arm"], {})[r["round"]] = r
    arms = sorted(runs)
    L = [f"Arms: {', '.join(arms)}; rounds per arm: " + ", ".join(f"{x}={len(runs[x])}" for x in arms), ""]
    names = []
    for x in arms:
        for r in runs[x].values():
            for n in r["scenarios"]:
                if n not in names:
                    names.append(n)
    num, den = a.num, a.den
    L.append(f"| scenario | metric | " + " | ".join(f"{x} median (rounds)" for x in arms) +
             f" | paired {num}/{den} per round | median ratio | min ratio |")
    L.append("|---|---|" + "---|" * len(arms) + "---|---|---|")
    for n in names:
        cells, vals = [], {}
        for x in arms:
            v = {rd: metric(r["scenarios"][n]) for rd, r in runs[x].items() if n in r["scenarios"]}
            vals[x] = v
            xs = [y for y in v.values() if y]
            cells.append(f"{statistics.median(xs):.1f} (" + " / ".join(f"{y:.1f}" for _, y in sorted(v.items()) if y) + ")"
                         if xs else "-")
        ratios = []
        if num in vals and den in vals:
            for rd in sorted(set(vals[num]) & set(vals[den])):
                if vals[num][rd] and vals[den][rd]:
                    ratios.append(vals[num][rd] / vals[den][rd])
        mname = "decode tok/s" if any("decode_tok_s_median" in r["scenarios"].get(n, {}) for x in arms
                                      for r in runs[x].values()) else "aggregate tok/s"
        L.append(f"| {n} | {mname} | " + " | ".join(cells) + " | " + " / ".join(f"{q:.3f}" for q in ratios) + " | " +
                 (f"{statistics.median(ratios):.3f}" if ratios else "-") + " | " + (f"{min(ratios):.3f}" if ratios else "-") + " |")
    # every arm against the denominator arm, paired by round
    others = [x for x in arms if x != den]
    if den in runs and others:
        L.append("")
        L.append(f"| scenario | " + " | ".join(f"{x}/{den} per round (median)" for x in others) + " |")
        L.append("|---|" + "---|" * len(others))
        for n in names:
            cells = []
            for x in others:
                rs = []
                for rd in sorted(set(runs[x]) & set(runs[den])):
                    p, q = (metric(runs[x][rd]["scenarios"].get(n, {})) if n in runs[x][rd]["scenarios"] else None,
                            metric(runs[den][rd]["scenarios"][n]) if n in runs[den][rd]["scenarios"] else None)
                    if p and q:
                        rs.append(p / q)
                cells.append(" / ".join(f"{y:.3f}" for y in rs) + (f" ({statistics.median(rs):.3f})" if rs else "-"))
            L.append(f"| {n} | " + " | ".join(cells) + " |")
    # extra detail for the mixed scenarios
    L.append("")
    L.append("| scenario | arm | open-stream decode tok/s (mean over rounds) | grounded wall s / finished (mixed) |")
    L.append("|---|---|---|---|")
    for n in names:
        if not n.startswith("mixed"):
            continue
        for x in arms:
            ss = [r["scenarios"][n] for r in runs[x].values() if n in r["scenarios"]]
            o = [s["open_decode_tok_s_mean"] for s in ss if s.get("open_decode_tok_s_mean")]
            g = [s.get("grounded_wall_s") or s.get("grounded_finished") for s in ss]
            L.append(f"| {n} | {x} | {statistics.mean(o):.1f} | " + " / ".join(f"{y:.1f}" if isinstance(y, float) else str(y) for y in g) + " |")
    text = "\n".join(L)
    print(text)
    if a.md:
        pathlib.Path(a.md).write_text(text + "\n")
    return 0


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--url", required=True)
    r.add_argument("--arm", required=True)
    r.add_argument("--round", type=int, default=1)
    r.add_argument("--out", required=True)
    r.add_argument("--scenarios", default=DEFAULT)
    r.add_argument("--reps", type=int, default=3)
    z = sub.add_parser("analyze")
    z.add_argument("--dir", required=True)
    z.add_argument("--md")
    z.add_argument("--num", default="on")
    z.add_argument("--den", default="off")
    a = ap.parse_args()
    sys.exit(cmd_run(a) if a.cmd == "run" else cmd_analyze(a))


if __name__ == "__main__":
    main()
