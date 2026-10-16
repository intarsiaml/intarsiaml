#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""A/B scenario for the decode-time prefill chunk of the Metal batched engine (`serve --max-batch N --busy-chunk C`).

Scenario (one round, one fresh server): 7 streams (about 1k-token prompts, `ignore_eos`) decode together; once all of
them are past `--settle-tokens` tokens, `--delay-s` later a prompt of about 4k tokens arrives (`max_tokens` 16). The
streams keep decoding while the big prompt prefills in busy-chunk pieces. A warm-up pass of the same shape runs first
(unrecorded, no idle gap before the measured pass, so the weights stay wired and every kernel is compiled).

    cbchunkab.py scenario --url U --arm 256 --round 1 --out DIR            # one round -> DIR/round-<arm>-<round>.json
    cbchunkab.py analyze  --dir DIR [--md OUT.md]                          # aggregate rounds (+ server trace + macmon)

The server must run with `INTARSIA_CB_TRACE=1` (stderr -> DIR/server-<arm>-<round>.log): the `Chunk` trace events
(millisecond wall clock) tell which decode gaps carried a prefill chunk. macmon (`macmon pipe`) output is read from
DIR/macmon.jsonl for GPU busy / frequency.
"""
import argparse
import datetime
import http.client
import json
import pathlib
import re
import statistics
import sys
import threading
import time

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cbsoak  # noqa: E402  (PromptSource, split_url, get_json, percentile)


# ----------------------------------------------------------------------------------------------------------------------
# scenario
# ----------------------------------------------------------------------------------------------------------------------

class Stream(threading.Thread):
    """One streaming /v1/completions request; `times` holds the epoch time of every content chunk as it arrives."""

    def __init__(self, url, body, name):
        super().__init__(daemon=True)
        self.url, self.body, self.name_ = url, body, name
        self.times, self.t_send, self.t_end, self.usage, self.error, self.finish = [], None, None, None, None, None

    def run(self):
        host, port = cbsoak.split_url(self.url)
        conn = http.client.HTTPConnection(host, port, timeout=300)
        try:
            self.t_send = time.time()
            conn.request("POST", "/v1/completions", body=json.dumps(self.body).encode(),
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
                    if ch.get("text"):
                        self.times.append(time.time())
                    if ch.get("finish_reason"):
                        self.finish = ch["finish_reason"]
            if not done and not self.error:
                self.error = "stream ended without [DONE]"
        except Exception as e:  # noqa: BLE001
            self.error = repr(e)[:300]
        finally:
            self.t_end = time.time()
            conn.close()


def make_body(ps, model, tag, i, tokens, max_tokens, seed):
    import random
    rng = random.Random(f"{seed}|{tag}|{i}")
    name = rng.choice([n for n, _ in ps.corpora])
    text = ps.text[name]
    chars = min(int(tokens * ps.cpt[name]), len(text))
    start = rng.randrange(0, len(text) - chars + 1)
    return {"model": model, "prompt": f"[chunkab {seed}-{tag}-{i}] " + text[start:start + chars], "max_tokens": max_tokens,
            "temperature": 0, "stream": True, "stream_options": {"include_usage": True}, "ignore_eos": True}


def run_pass(url, ps, model, tag, a):
    streams = [Stream(url, make_body(ps, model, tag, i, a.stream_ctx, a.stream_tokens, a.seed), f"s{i}") for i in range(a.n_streams)]
    for s in streams:
        s.start()
    t_start = time.time()
    deadline = time.time() + 180
    while time.time() < deadline:
        if any(s.error for s in streams):
            break
        if all(len(s.times) >= a.settle_tokens for s in streams):
            break
        time.sleep(0.01)
    time.sleep(a.delay_s)
    big = Stream(url, make_body(ps, model, tag, 99, a.big_ctx, a.big_tokens, a.seed), "big")
    big.start()
    for s in streams + [big]:
        s.join(300)
    return t_start, streams, big


def cmd_scenario(a):
    model = "m"
    j = cbsoak.get_json(a.url, "/v1/models")
    if j and j.get("data"):
        model = j["data"][0]["id"]
    ps = cbsoak.PromptSource(HERE / "prompts", "docs:1,code:1")
    ps.calibrate(a.url, model)
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    # warm-up pass (same shape, shorter streams), then the measured pass without an idle gap
    wa = argparse.Namespace(**vars(a))
    wa.stream_tokens = a.warmup_stream_tokens
    _, ws, wb = run_pass(a.url, ps, model, f"warm-{a.round}", wa)
    werr = [s.error for s in ws + [wb] if s.error]
    t_start, streams, big = run_pass(a.url, ps, model, f"meas-{a.round}", a)
    rec = {
        "arm": a.arm, "round": a.round, "order": a.order, "cpt": ps.cpt, "t_start": t_start,
        "params": {k: getattr(a, k) for k in ("n_streams", "stream_ctx", "stream_tokens", "big_ctx", "big_tokens", "settle_tokens", "delay_s", "seed")},
        "warmup_errors": werr,
        "streams": [{"t_send": s.t_send, "t_end": s.t_end, "times": s.times, "usage": s.usage, "error": s.error, "finish": s.finish} for s in streams],
        "big": {"t_send": big.t_send, "t_end": big.t_end, "times": big.times, "usage": big.usage, "error": big.error, "finish": big.finish},
    }
    (out / f"round-{a.arm}-{a.round}.json").write_text(json.dumps(rec))
    errs = [s.error for s in streams + [big] if s.error]
    pt = [(s.usage or {}).get("prompt_tokens") for s in streams + [big]]
    print(f"round {a.round} arm {a.arm}: errors={errs or 'none'} warmup_errors={werr or 'none'} prompt_tokens={pt}"
          f" big ttft={(big.times[0] - big.t_send) * 1000 if big.times else None} ms")
    return 1 if errs else 0


# ----------------------------------------------------------------------------------------------------------------------
# analysis
# ----------------------------------------------------------------------------------------------------------------------

TRACE_RE = re.compile(r"cb-trace t=(\d+) (\w+) \{(.*)\}")


def parse_trace(path):
    """[(t_epoch_s, kind, fields-str)] for Chunk / Decode / PrefillStart events."""
    ev = []
    try:
        for line in open(path, errors="replace"):
            m = TRACE_RE.match(line)
            if m and m.group(2) in ("Chunk", "Decode", "PrefillStart"):
                ev.append((int(m.group(1)) / 1000.0, m.group(2), m.group(3)))
    except OSError:
        pass
    return ev


def read_macmon(path):
    out = []
    try:
        for line in open(path):
            try:
                o = json.loads(line)
            except ValueError:
                continue
            ts = datetime.datetime.fromisoformat(o["timestamp"]).timestamp()
            out.append((ts, o["gpu_usage"][0], o["gpu_usage"][1], o.get("gpu_power")))
    except OSError:
        pass
    return out


def pct(xs, p):
    return cbsoak.percentile(sorted(xs), p) if xs else None


def analyze_round(rec, trace, mac, slack=0.005):
    streams, big = rec["streams"], rec["big"]
    r = {"arm": rec["arm"], "round": rec["round"], "order": rec.get("order")}
    errs = [s["error"] for s in streams + [big] if s["error"]]
    r["errors"] = errs
    if errs or not big["times"] or any(len(s["times"]) < 50 for s in streams):
        r["invalid"] = True
        return r
    t_send, t_first = big["t_send"], big["times"][0]
    r["big_ttft_ms"] = (t_first - t_send) * 1000
    r["big_prompt_tokens"] = (big["usage"] or {}).get("prompt_tokens")
    r["stream_prompt_tokens"] = [(s["usage"] or {}).get("prompt_tokens") for s in streams]
    # chunk events inside the big prompt's prefill window (the 7 streams finished prefilling long before)
    chunks = [(t, f) for t, k, f in trace if k == "Chunk" and t_send - 0.05 <= t <= t_first + 0.01]
    r["big_chunks"] = len(chunks)
    lens = [int(re.search(r"len: (\d+)", f).group(1)) for _, f in chunks]
    r["big_chunk_lens"] = lens
    r["big_chunk_tokens"] = sum(lens)
    starts = [(t, f) for t, k, f in trace if k == "PrefillStart" and t_send - 0.05 <= t <= t_first + 0.01]
    r["big_prefill_chunk"] = int(re.search(r"chunk: (\d+)", starts[0][1]).group(1)) if starts else None
    r["big_prefill_decoding"] = int(re.search(r"decoding: (\d+)", starts[0][1]).group(1)) if starts else None
    if chunks:
        r["prefill_wall_ms"] = (t_first - chunks[0][0]) * 1000  # first chunk start (trace) to the first token
    # decode gaps of the 7 streams
    ends = [s["times"][-1] for s in streams]
    settle_t = max(s["times"][rec["params"]["settle_tokens"] - 1] for s in streams)
    big_last = big["times"][-1]
    ctimes = [t for t, _ in chunks]
    pre, ref_before, ref_after, all_gaps = [], [], [], []
    for s in streams:
        ts = s["times"]
        for a, b in zip(ts, ts[1:]):
            g = (b - a) * 1000
            carries = any(a - slack < tc < b - slack for tc in ctimes)
            if carries:
                pre.append(g)
            elif a >= settle_t and b <= t_send:
                ref_before.append(g)
            elif a >= big_last + 0.1 and b <= min(ends) - 0.1:
                ref_after.append(g)
    r["stall_ms"] = pre
    r["itl_noprefill_ms"] = ref_before + ref_after
    r["itl_before_ms"], r["itl_after_ms"] = ref_before, ref_after
    # server-side iteration gaps (Decode events): consecutive decode steps; carries a chunk if a Chunk event lies between
    dec = [t for t, k, _ in trace if k == "Decode" and settle_t <= t <= big_last]
    srv_pre, srv_ref = [], []
    for a, b in zip(dec, dec[1:]):
        g = (b - a) * 1000
        (srv_pre if any(a <= tc < b for tc in ctimes) else srv_ref).append(g)
    r["srv_stall_ms"], r["srv_itl_noprefill_ms"] = srv_pre, srv_ref
    # throughput: tokens delivered at/after the big prompt's send, over the time to the last completion
    toks = sum(1 for s in streams for t in s["times"] if t > t_send) + len(big["times"])
    t_last = max(s["times"][-1] for s in streams + [big])
    r["tokens_after_send"] = toks
    r["agg_tok_s"] = toks / (t_last - t_send)
    r["remaining_s"] = t_last - t_send
    r["decode_tok_s_in_prefill"] = sum(1 for s in streams for t in s["times"] if t_send < t <= t_first) / (t_first - t_send)
    # GPU
    win = [m for m in mac if (chunks[0][0] if chunks else t_send) <= m[0] <= t_first]
    run = [m for m in mac if t_send - 5 <= m[0] <= t_last]
    r["gpu_samples_in_prefill"] = len(win)
    if win:
        r["gpu_busy_prefill"] = statistics.mean(m[2] for m in win)
        r["gpu_mhz_prefill"] = statistics.mean(m[1] for m in win)
    if run:
        r["gpu_busy_run"] = statistics.mean(m[2] for m in run)
        r["gpu_mhz_run"] = statistics.mean(m[1] for m in run)
        r["gpu_mhz_run_min"] = min(m[1] for m in run)
    return r


def fnum(x, nd=1):
    return "-" if x is None else f"{x:.{nd}f}"


def cmd_analyze(a):
    d = pathlib.Path(a.dir)
    mac = read_macmon(d / "macmon.jsonl")
    rounds = []
    for p in sorted(d.glob("round-*.json")):
        rec = json.loads(p.read_text())
        tr = parse_trace(d / f"server-{rec['arm']}-{rec['round']}.log")
        rounds.append(analyze_round(rec, tr, mac))
    (d / "analysis.json").write_text(json.dumps(rounds, indent=1))
    arms = sorted({r["arm"] for r in rounds}, key=int)
    ok = {m: [r for r in rounds if r["arm"] == m and not r.get("invalid")] for m in arms}
    L = []
    L.append(f"Rounds per arm (valid/total): " + ", ".join(f"{m}: {len(ok[m])}/{sum(1 for r in rounds if r['arm'] == m)}" for m in arms))
    L.append("")

    def med(m, k):
        v = [r[k] for r in ok[m] if r.get(k) is not None]
        return statistics.median(v) if v else None

    def mean(m, k):
        v = [r[k] for r in ok[m] if r.get(k) is not None]
        return statistics.mean(v) if v else None

    def sd(m, k):
        v = [r[k] for r in ok[m] if r.get(k) is not None]
        return statistics.stdev(v) if len(v) > 1 else None

    def pooled(m, k):
        return [x for r in ok[m] for x in r[k]]

    rows = [
        ("4k TTFT, ms (median of rounds)", lambda m: med(m, "big_ttft_ms"), 0),
        ("4k TTFT, ms (mean +- sd)", lambda m: f"{fnum(mean(m, 'big_ttft_ms'), 0)} +- {fnum(sd(m, 'big_ttft_ms'), 0)}", None),
        ("4k prefill wall (first chunk to first token), ms (median)", lambda m: med(m, "prefill_wall_ms"), 0),
        ("prefill chunks of the 4k prompt (median)", lambda m: med(m, "big_chunks"), 0),
        ("GPU busy during the prefill (macmon, mean of rounds)", lambda m: mean(m, "gpu_busy_prefill"), 3),
        ("GPU MHz during the prefill (mean)", lambda m: mean(m, "gpu_mhz_prefill"), 0),
        ("GPU busy over the whole scenario (mean)", lambda m: mean(m, "gpu_busy_run"), 3),
        ("GPU MHz over the scenario: mean / min", lambda m: f"{fnum(mean(m, 'gpu_mhz_run'), 0)} / {fnum(min((r['gpu_mhz_run_min'] for r in ok[m] if 'gpu_mhz_run_min' in r), default=None), 0)}", None),
        ("decode-row stall (client gap in a prefill-carrying iteration) p50, ms", lambda m: pct(pooled(m, "stall_ms"), 50), 1),
        ("  p95", lambda m: pct(pooled(m, "stall_ms"), 95), 1),
        ("  max", lambda m: max(pooled(m, "stall_ms"), default=None), 1),
        ("  n gaps", lambda m: len(pooled(m, "stall_ms")), 0),
        ("decode ITL without prefill (reference) p50, ms", lambda m: pct(pooled(m, "itl_noprefill_ms"), 50), 1),
        ("  p95", lambda m: pct(pooled(m, "itl_noprefill_ms"), 95), 1),
        ("  max", lambda m: max(pooled(m, "itl_noprefill_ms"), default=None), 1),
        ("  n gaps", lambda m: len(pooled(m, "itl_noprefill_ms")), 0),
        ("server iteration time with a chunk (Decode-to-Decode) p50 / p95 / max, ms",
         lambda m: " / ".join(fnum(f(pooled(m, "srv_stall_ms")), 1) for f in (lambda x: pct(x, 50), lambda x: pct(x, 95), lambda x: max(x, default=None))), None),
        ("server iteration time without a chunk p50 / p95 / max, ms",
         lambda m: " / ".join(fnum(f(pooled(m, "srv_itl_noprefill_ms")), 1) for f in (lambda x: pct(x, 50), lambda x: pct(x, 95), lambda x: max(x, default=None))), None),
        ("decode tokens/s delivered during the 4k prefill (all 7 streams, mean)", lambda m: mean(m, "decode_tok_s_in_prefill"), 1),
        ("aggregate tok/s from the 4k send to the last completion (mean +- sd)", lambda m: f"{fnum(mean(m, 'agg_tok_s'), 1)} +- {fnum(sd(m, 'agg_tok_s'), 1)}", None),
        ("time from the 4k send to the last completion, s (mean)", lambda m: mean(m, "remaining_s"), 2),
    ]
    L.append("| metric | " + " | ".join(f"busy chunk {m}" for m in arms) + (" | 320 vs 256 |" if {"256", "320"} <= set(arms) else " |"))
    L.append("|---|" + "---|" * len(arms) + ("---|" if {"256", "320"} <= set(arms) else ""))
    for name, fn, nd in rows:
        vals = [fn(m) for m in arms]
        cells = [v if isinstance(v, str) else fnum(v, nd if nd is not None else 1) for v in vals]
        delta = ""
        if {"256", "320"} <= set(arms) and nd is not None and all(isinstance(v, (int, float)) for v in vals):
            v256, v320 = vals[arms.index("256")], vals[arms.index("320")]
            if v256:
                delta = f"{(v320 / v256 - 1) * 100:+.1f}%"
        L.append(f"| {name} | " + " | ".join(cells) + (f" | {delta} |" if {"256", "320"} <= set(arms) else " |"))
    # paired differences per round index (320 - 256)
    if {"256", "320"} <= set(arms):
        L.append("")
        L.append("Paired by round (320 minus 256, same round index, alternating start order):")
        L.append("")
        L.append("| metric | per-round deltas | mean delta | rounds where 320 is lower |")
        L.append("|---|---|---|---|")
        by = {(r["arm"], r["round"]): r for r in rounds if not r.get("invalid")}
        rids = sorted({r["round"] for r in rounds})
        for k, nd, lab in (("big_ttft_ms", 0, "4k TTFT ms"), ("prefill_wall_ms", 0, "prefill wall ms"), ("agg_tok_s", 1, "aggregate tok/s"), ("remaining_s", 2, "time to last completion s"),
                           ("gpu_busy_prefill", 3, "GPU busy in prefill")):
            ds = [by[("320", i)][k] - by[("256", i)][k] for i in rids if ("320", i) in by and ("256", i) in by and k in by[("320", i)] and k in by[("256", i)]]
            if ds:
                L.append(f"| {lab} | {', '.join(f'{x:+.{nd}f}' for x in ds)} | {statistics.mean(ds):+.{nd}f} | {sum(1 for x in ds if x < 0)}/{len(ds)} |")
        for k, lab in (("stall_ms", "stall p95 ms"), ("stall_ms", "stall max ms")):
            ds = []
            for i in rids:
                if ("320", i) in by and ("256", i) in by:
                    f = (lambda x: pct(x, 95)) if "p95" in lab else (lambda x: max(x, default=None))
                    x320, x256 = f(by[("320", i)][k]), f(by[("256", i)][k])
                    if x320 is not None and x256 is not None:
                        ds.append(x320 - x256)
            if ds:
                L.append(f"| {lab} | {', '.join(f'{x:+.1f}' for x in ds)} | {statistics.mean(ds):+.1f} | {sum(1 for x in ds if x < 0)}/{len(ds)} |")
    L.append("")
    L.append("Per round:")
    L.append("")
    L.append("| arm | round | order | big prompt tok | stream prompt tok (min-max) | chunk / decoding at start | chunks | TTFT ms | wall ms | stall p50/p95/max | ITL ref p50 | agg tok/s | GPU busy | MHz |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in sorted(rounds, key=lambda r: (int(r["round"]), r["arm"])):
        if r.get("invalid"):
            L.append(f"| {r['arm']} | {r['round']} | {r.get('order')} | INVALID {r['errors']} |")
            continue
        sp = r["stream_prompt_tokens"]
        L.append(f"| {r['arm']} | {r['round']} | {r.get('order')} | {r['big_prompt_tokens']} | {min(sp)}-{max(sp)} | {r['big_prefill_chunk']} / {r['big_prefill_decoding']} | {r['big_chunks']} | {fnum(r['big_ttft_ms'], 0)} | {fnum(r.get('prefill_wall_ms'), 0)}"
                 f" | {fnum(pct(r['stall_ms'], 50))}/{fnum(pct(r['stall_ms'], 95))}/{fnum(max(r['stall_ms'], default=None))} | {fnum(pct(r['itl_noprefill_ms'], 50))} | {fnum(r['agg_tok_s'])}"
                 f" | {fnum(r.get('gpu_busy_prefill'), 3)} | {fnum(r.get('gpu_mhz_prefill'), 0)} |")
    text = "\n".join(L)
    print(text)
    if a.md:
        pathlib.Path(a.md).write_text(text + "\n")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("scenario")
    s.add_argument("--url", required=True)
    s.add_argument("--arm", required=True, help="busy chunk of the server (label)")
    s.add_argument("--round", type=int, required=True)
    s.add_argument("--order", default="", help="label: position of this arm in the round (first/second)")
    s.add_argument("--out", required=True)
    s.add_argument("--n-streams", type=int, default=7)
    s.add_argument("--stream-ctx", type=int, default=1024, help="prompt tokens of each stream")
    s.add_argument("--stream-tokens", type=int, default=400)
    s.add_argument("--warmup-stream-tokens", type=int, default=250)
    s.add_argument("--big-ctx", type=int, default=4096)
    s.add_argument("--big-tokens", type=int, default=16)
    s.add_argument("--settle-tokens", type=int, default=40)
    s.add_argument("--delay-s", type=float, default=1.0)
    s.add_argument("--seed", type=int, default=20261016)
    s.set_defaults(fn=cmd_scenario)
    an = sub.add_parser("analyze")
    an.add_argument("--dir", required=True)
    an.add_argument("--md")
    an.set_defaults(fn=cmd_analyze)
    a = ap.parse_args()
    sys.exit(a.fn(a))


if __name__ == "__main__":
    main()
