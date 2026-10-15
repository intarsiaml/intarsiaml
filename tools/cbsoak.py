#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Open-loop soak / load harness for `intarsia serve` (gates of milestone M2, docs/research/continuous-batching-design.md).

Poisson arrivals at a target request rate (open loop: the arrival schedule does not wait for responses, so a slow
server builds a queue instead of slowing the clients down), a prompt-length mix (128..4k tokens, corpus slices from
tools/prompts/corpus_*.txt plus the golden_4k.jsonl cases) and a max_tokens mix, optional client-side cancellations,
streaming. Per request it records TTFT, inter-chunk latencies (ITL), completion tokens, finish_reason and the HTTP
status (429/503 count as "shed", everything else non-200 as an error). While it runs it samples the server's memory
(`footprint`/`ps` of --pid) and `/metrics` + `/v1/intarsia/plan` (both optional). The report gives throughput,
TTFT/ITL p50/p95/p99, shed and error rates, memory start/end/max and growth slope (leak indicator), and a per-window
table that shows degradation over time; JSON + markdown. SLO flags turn it into a pass/fail gate (exit 1).

Standard library only (`http.client` + threads):

    uv run tools/cbsoak.py run --url http://127.0.0.1:8099 --rate 0.5 --duration 180 --pid 12345 --out DIR
    uv run tools/cbsoak.py run --url U --rate 4 --duration 1800 --cancel-frac 0.1 \\
        --max-itl-p95-ms 150 --max-error-rate 0.001 --max-mem-growth-mb 200 --out DIR
    uv run tools/cbsoak.py report --raw DIR/raw.json [--max-itl-p95-ms ...]    # re-render / re-judge a finished run
    uv run tools/cbsoak.py selftest                                            # offline: in-process mock server

Exit codes: 0 pass, 1 an SLO threshold was violated, 2 harness-level failure (server unreachable, bad arguments).
Definitions of every metric: docs/results/2026-10-15-metal-cb-soak-harness.md.
"""

import argparse
import collections
import datetime
import http.client
import json
import math
import os
import pathlib
import random
import re
import socket
import subprocess
import sys
import threading
import time
import urllib.parse

HERE = pathlib.Path(__file__).resolve().parent
PROMPTS = HERE / "prompts"
DEFAULT_PROMPT_MIX = "128:2,512:3,1024:3,2048:2,4096:1"
DEFAULT_TOKENS_MIX = "32:2,128:4,256:2"
SHED_STATUSES = (429, 503)


# ----------------------------------------------------------------------------------------------------------------------
# statistics
# ----------------------------------------------------------------------------------------------------------------------

def percentile(xs, p):
    """p in [0, 100]; linear interpolation between closest ranks (numpy's default). None for an empty list."""
    if not xs:
        return None
    s = sorted(xs)
    if len(s) == 1:
        return float(s[0])
    k = (len(s) - 1) * p / 100.0
    lo = int(math.floor(k))
    hi = min(lo + 1, len(s) - 1)
    return float(s[lo] + (s[hi] - s[lo]) * (k - lo))


def summarize(xs):
    if not xs:
        return {"n": 0, "mean": None, "p50": None, "p95": None, "p99": None, "max": None}
    return {"n": len(xs), "mean": sum(xs) / len(xs), "p50": percentile(xs, 50), "p95": percentile(xs, 95),
            "p99": percentile(xs, 99), "max": max(xs)}


def theil_sen(ts, ys, max_points=300):
    """Median of pairwise slopes (robust to allocator spikes). Evenly subsamples to max_points for O(n^2) cost."""
    n = len(ts)
    if n < 2:
        return None
    if n > max_points:
        idx = [int(i * (n - 1) / (max_points - 1)) for i in range(max_points)]
        ts, ys = [ts[i] for i in idx], [ys[i] for i in idx]
        n = len(ts)
    slopes = []
    for i in range(n):
        for j in range(i + 1, n):
            dt = ts[j] - ts[i]
            if dt > 0:
                slopes.append((ys[j] - ys[i]) / dt)
    return percentile(slopes, 50) if slopes else None


def ols(ts, ys):
    """Least-squares (slope, r2). None when under-determined."""
    n = len(ts)
    if n < 2:
        return None, None
    mt, my = sum(ts) / n, sum(ys) / n
    sxx = sum((t - mt) ** 2 for t in ts)
    if sxx == 0:
        return None, None
    sxy = sum((t - mt) * (y - my) for t, y in zip(ts, ys))
    slope = sxy / sxx
    syy = sum((y - my) ** 2 for y in ys)
    r2 = 1.0 if syy == 0 else (sxy * sxy) / (sxx * syy)
    return slope, r2


def analyze_memory(samples, warmup_s=0.0, key="mb", leak_min_mb=100.0, leak_min_r2=0.6):
    """samples: [{"t": seconds since run start, key: MB}]. Slope/growth are fitted on samples with t >= warmup_s."""
    pts = [(s["t"], s[key]) for s in samples if s.get(key) is not None]
    if not pts:
        return {"available": False}
    fit = [(t, y) for t, y in pts if t >= warmup_s] or pts
    ts, ys = [p[0] for p in fit], [p[1] for p in fit]
    span_min = (ts[-1] - ts[0]) / 60.0
    ts_slope = theil_sen(ts, ys)
    ols_slope, r2 = ols(ts, ys)
    slope_min = ts_slope * 60.0 if ts_slope is not None else None  # MB / min
    growth = slope_min * span_min if slope_min is not None else None
    return {
        "available": True, "n": len(pts), "n_fit": len(fit), "warmup_s": warmup_s,
        "start_mb": pts[0][1], "end_mb": pts[-1][1], "max_mb": max(y for _, y in pts), "min_mb": min(y for _, y in pts),
        "end_minus_start_mb": pts[-1][1] - pts[0][1],
        "slope_mb_per_min": slope_min, "slope_ols_mb_per_min": ols_slope * 60.0 if ols_slope is not None else None,
        "r2": r2, "fit_span_min": span_min, "growth_mb": growth,
        "projected_mb_per_hour": slope_min * 60.0 if slope_min is not None else None,
        "leak_suspect": bool(growth is not None and growth >= leak_min_mb and r2 is not None and r2 >= leak_min_r2),
    }


# ----------------------------------------------------------------------------------------------------------------------
# arrival schedule and request mix
# ----------------------------------------------------------------------------------------------------------------------

def parse_mix(spec):
    """'128:2,512:3' -> [(128, 2.0), (512, 3.0)]; a bare '128' has weight 1."""
    out = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        v, _, w = part.partition(":")
        out.append((int(v), float(w) if w else 1.0))
    if not out or any(w <= 0 for _, w in out):
        raise SystemExit(f"bad mix {spec!r}")
    return out


def draw(rng, mix):
    return rng.choices([v for v, _ in mix], weights=[w for _, w in mix])[0]


def poisson_schedule(rate, duration, seed):
    """Arrival times (seconds from start) of a Poisson process with `rate` per second: exponential gaps."""
    rng = random.Random(f"arrivals|{seed}")
    t, out = 0.0, []
    while rate > 0:
        t += rng.expovariate(rate)
        if t >= duration:
            break
        out.append(t)
    return out


def build_plan(args):
    """The full request plan, deterministic in --seed: [{i, t, target_tokens, max_tokens, cancel_ms, golden}]."""
    arrivals = poisson_schedule(args.rate, args.duration, args.seed)
    rng = random.Random(f"mix|{args.seed}")
    pmix, tmix = parse_mix(args.prompt_mix), parse_mix(args.max_tokens_mix)
    plan = []
    for i, t in enumerate(arrivals):
        target = draw(rng, pmix)
        mt = draw(rng, tmix)
        if args.max_seq:
            target = max(16, min(target, args.max_seq - mt - 8))
        cancel = None
        if args.cancel_frac > 0 and rng.random() < args.cancel_frac:
            cancel = rng.uniform(args.cancel_min_ms, args.cancel_max_ms)
        plan.append({"i": i, "t": t, "bucket": draw_bucket(target, pmix), "target_tokens": target, "max_tokens": mt,
                     "cancel_ms": cancel, "golden": rng.random() < args.golden_frac, "jitter": rng.uniform(0.9, 1.1)})
    return plan


def draw_bucket(target, pmix):
    return min((v for v, _ in pmix), key=lambda v: abs(v - target))


class PromptSource:
    """Builds request bodies from the corpus slices and golden_4k cases. Prompt length is set in characters through a
    chars-per-token ratio calibrated against the server's own `usage.prompt_tokens` (the harness has no tokenizer)."""

    def __init__(self, prompts_dir, corpora="docs:1,code:1"):
        self.dir = pathlib.Path(prompts_dir)
        self.corpora = [(n, w) for n, w in ((p.split(":")[0], float(p.split(":")[1]) if ":" in p else 1.0) for p in corpora.split(","))]
        self.text = {}
        for n, _ in self.corpora:
            f = self.dir / f"corpus_{n}.txt"
            if not f.exists():
                raise SystemExit(f"missing corpus file {f}")
            self.text[n] = f.read_text()
        self.cpt = {n: 4.0 for n, _ in self.corpora}  # chars per token, replaced by calibrate()
        self.golden = []
        g = self.dir / "golden_4k.jsonl"
        if g.exists():
            for line in g.read_text().splitlines():
                if line.strip():
                    self.golden.append(json.loads(line))

    def calibrate(self, url, model, timeout=120):
        """One 1-token request per corpus: ratio = characters sent / prompt_tokens the server reports."""
        for n in self.cpt:
            chars = min(6000, len(self.text[n]))
            body = {"model": model, "prompt": self.text[n][:chars], "max_tokens": 1, "temperature": 0}
            try:
                host, port = split_url(url)
                conn = http.client.HTTPConnection(host, port, timeout=timeout)
                conn.request("POST", "/v1/completions", body=json.dumps(body).encode(), headers={"Content-Type": "application/json", "Connection": "close"})
                r = conn.getresponse()
                obj = json.loads(r.read())
                conn.close()
                pt = (obj.get("usage") or {}).get("prompt_tokens")
                if r.status == 200 and pt:
                    self.cpt[n] = chars / pt
            except Exception as e:  # noqa: BLE001
                print(f"calibration for corpus_{n} failed ({e!r}); using {self.cpt[n]} chars/token", file=sys.stderr)
        return dict(self.cpt)

    def _golden_body(self, spec, model, nonce, max_tokens):
        if spec["kind"] == "raw":
            text = (self.dir / spec["file"]).read_text()
            n = next(iter(k for k in self.cpt if spec["file"] == f"corpus_{k}.txt"), None)
            chars = int(spec["max_tokens"] * (self.cpt.get(n) or 4.0))
            return "/v1/completions", {"model": model, "prompt": nonce + text[:chars], "max_tokens": max_tokens}, spec["name"]
        msgs = []
        for m in spec["messages"]:
            if "content_file" in m:
                t = (self.dir / m["content_file"]).read_text()
                s = m.get("skip_chars", 0)
                msgs.append({"role": m["role"], "content": t[s : s + m["max_chars"]]})
            else:
                msgs.append({"role": m["role"], "content": m["content"]})
        msgs[0] = dict(msgs[0], content=nonce + msgs[0]["content"])  # unique system prefix: no KV prefix reuse across requests
        return "/v1/chat/completions", {"model": model, "messages": msgs, "max_tokens": max_tokens,
                                        "chat_template_kwargs": {"enable_thinking": False}}, spec["name"]

    def body(self, req, model, args):
        """(path, body, source) for one planned request."""
        rng = random.Random(f"prompt|{args.seed}|{req['i']}")
        nonce = f"[soak {args.seed}-{req['i']}] " if not args.no_nonce else ""
        if req["golden"] and req["target_tokens"] >= 3000 and self.golden:
            path, body, src = self._golden_body(rng.choice(self.golden), model, nonce, req["max_tokens"])
        else:
            name = rng.choices([n for n, _ in self.corpora], weights=[w for _, w in self.corpora])[0]
            chars = int(req["target_tokens"] * req["jitter"] * self.cpt[name])
            text = self.text[name]
            chars = min(chars, len(text))
            start = rng.randrange(0, len(text) - chars + 1)
            path, body, src = "/v1/completions", {"model": model, "prompt": nonce + text[start : start + chars], "max_tokens": req["max_tokens"]}, f"corpus_{name}"
        body.update({"temperature": args.temperature, "stream": True, "stream_options": {"include_usage": True}})
        if args.ignore_eos:
            body["ignore_eos"] = True
        return path, body, src


# ----------------------------------------------------------------------------------------------------------------------
# HTTP client (streaming, per-chunk timing)
# ----------------------------------------------------------------------------------------------------------------------

class Cancel(Exception):
    pass


def split_url(url):
    u = urllib.parse.urlparse(url)
    return u.hostname, u.port or 80


def http_get(url, path, timeout=10):
    """(status, body bytes) or (None, repr(error))."""
    host, port = split_url(url)
    conn = http.client.HTTPConnection(host, port, timeout=timeout)
    try:
        conn.request("GET", path)
        r = conn.getresponse()
        return r.status, r.read()
    except Exception as e:  # noqa: BLE001
        return None, repr(e).encode()
    finally:
        conn.close()


def get_json(url, path, timeout=10):
    st, data = http_get(url, path, timeout)
    if st != 200:
        return None
    try:
        return json.loads(data)
    except ValueError:
        return None


def stream_request(url, path, body, timeout, cancel_ms=None, total_timeout=None):
    """One streaming request. Times (ms) are relative to the moment before connect. A record:
    status ok|cancelled|shed|error, http_status, ttft_ms, itl_ms (gaps between content chunks), n_chunks, usage,
    finish_reason, error, t_total_ms. A cancel closes the socket mid-stream."""
    host, port = split_url(url)
    rec = {"status": "ok", "http_status": None, "ttft_ms": None, "itl_ms": [], "n_chunks": 0, "usage": None,
           "finish_reason": None, "error": None, "done": False}
    t_send = time.monotonic()
    cancel_at = t_send + cancel_ms / 1000.0 if cancel_ms is not None else None
    hard_at = t_send + total_timeout if total_timeout else None
    conn = http.client.HTTPConnection(host, port, timeout=timeout)

    sock = [None]  # http.client drops conn.sock once a "Connection: close" response arrives; keep our own reference

    def arm():
        if sock[0] is None:
            return
        now = time.monotonic()
        lim = timeout
        if cancel_at is not None:
            if cancel_at - now <= 0:
                raise Cancel("ms")
            lim = min(lim, cancel_at - now)
        if hard_at is not None:
            if hard_at - now <= 0:
                raise socket.timeout()
            lim = min(lim, hard_at - now)
        sock[0].settimeout(lim)

    last = None
    try:
        conn.request("POST", path, body=json.dumps(body).encode(), headers={"Content-Type": "application/json", "Connection": "close"})
        sock[0] = conn.sock
        arm()
        resp = conn.getresponse()
        rec["http_status"] = resp.status
        if resp.status != 200:
            rec["status"] = "shed" if resp.status in SHED_STATUSES else "error"
            rec["error"] = f"HTTP {resp.status}: {resp.read(300).decode(errors='replace')}"
        else:
            while True:
                arm()
                line = resp.readline()
                if not line:
                    break
                line = line.strip()
                if not line.startswith(b"data:"):
                    continue
                data = line[5:].strip()
                if data == b"[DONE]":
                    rec["done"] = True
                    break
                obj = json.loads(data)
                if "error" in obj:
                    rec["status"], rec["error"] = "error", "stream error: " + json.dumps(obj["error"])[:300]
                    break
                if obj.get("usage"):
                    rec["usage"] = obj["usage"]
                for ch in obj.get("choices") or []:
                    d = ch["delta"].get("content") if "delta" in ch else ch.get("text")
                    if d:
                        now = time.monotonic()
                        if last is None:
                            rec["ttft_ms"] = (now - t_send) * 1000.0
                        else:
                            rec["itl_ms"].append((now - last) * 1000.0)
                        last = now
                        rec["n_chunks"] += 1
                    if ch.get("finish_reason"):
                        rec["finish_reason"] = ch["finish_reason"]
            if rec["status"] == "ok" and not rec["done"]:
                rec["status"], rec["error"] = "error", "stream ended without [DONE]"
    except Cancel:
        rec["status"] = "cancelled"
    except socket.timeout:
        if cancel_at is not None and time.monotonic() >= cancel_at - 0.05:
            rec["status"] = "cancelled"
        else:
            rec["status"], rec["error"] = "error", f"timeout (read {timeout}s / total {total_timeout}s)"
    except Exception as e:  # noqa: BLE001
        rec["status"], rec["error"] = "error", repr(e)[:300]
    finally:
        if rec["status"] == "cancelled" and sock[0] is not None:
            try:
                sock[0].shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        conn.close()
    rec["t_total_ms"] = (time.monotonic() - t_send) * 1000.0
    return rec


# ----------------------------------------------------------------------------------------------------------------------
# samplers: process memory, server /metrics
# ----------------------------------------------------------------------------------------------------------------------

_UNITS = {"B": 1.0, "KB": 1024.0, "MB": 1024.0**2, "GB": 1024.0**3, "TB": 1024.0**4}


def parse_footprint(text):
    """MB from `footprint -p PID` output: '... Footprint: 12 GB (16384 bytes per page)'. None when absent."""
    m = re.search(r"Footprint:\s*([\d.]+)\s*([KMGT]?B)\b", text)
    return float(m.group(1)) * _UNITS[m.group(2)] / 1024.0**2 if m else None


def parse_ps_rss(text):
    m = re.search(r"^\s*(\d+)\s*$", text.strip().splitlines()[-1]) if text.strip() else None
    return int(m.group(1)) / 1024.0 if m else None  # ps rss is in KB


def _run(cmd, timeout=15):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.stdout if r.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def sample_process(pid, use_footprint=True):
    """{'rss_mb', 'footprint_mb'} of one process; missing values are None (process gone, tool absent)."""
    out = {"rss_mb": None, "footprint_mb": None}
    t = _run(["ps", "-o", "rss=", "-p", str(pid)])
    if t:
        out["rss_mb"] = parse_ps_rss(t)
    if use_footprint and sys.platform == "darwin":
        t = _run(["footprint", "-p", str(pid), "-f", "bytes"], timeout=30)
        if t:
            out["footprint_mb"] = parse_footprint(t)
    return out


def pid_from_port(url):
    host, port = split_url(url)
    t = _run(["lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN", "-t"])
    return int(t.split()[0]) if t and t.split() else None


def parse_prom(text):
    """Prometheus text -> {'name': value} for unlabeled series, {'name{k=v}': value} for the batch-size counters."""
    out = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        m = re.match(r"^([A-Za-z_:][A-Za-z0-9_:]*)(\{[^}]*\})?\s+(\S+)", line)
        if not m:
            continue
        name, labels, val = m.groups()
        try:
            v = float(val)
        except ValueError:
            continue
        if labels is None:
            if name.startswith(("intarsia_", "vllm:")):
                out[name] = v
        elif name == "intarsia_batch_decode_rows_total":
            rows = re.search(r'rows="(\d+)"', labels)
            if rows:
                out[f"batch_rows:{rows.group(1)}"] = v
        elif name == "intarsia_requests_finished_total":
            reason = re.search(r'reason="([^"]*)"', labels)
            if reason:
                out[f"finished:{reason.group(1)}"] = v
    return out


class Sampler(threading.Thread):
    """Calls fn() every `period` seconds on its own thread; fn returns a dict or None. Samples carry t = seconds since t0."""

    def __init__(self, fn, period, t0):
        super().__init__(daemon=True)
        self.fn, self.period, self.t0 = fn, period, t0
        self.samples = []
        self.stop = threading.Event()

    def run(self):
        while not self.stop.is_set():
            t = time.monotonic() - self.t0
            try:
                v = self.fn()
            except Exception:  # noqa: BLE001
                v = None
            if v is not None:
                v["t"] = t
                self.samples.append(v)
            self.stop.wait(self.period)

    def finish(self):
        self.stop.set()
        self.join(timeout=60)


# ----------------------------------------------------------------------------------------------------------------------
# the run
# ----------------------------------------------------------------------------------------------------------------------

def run_load(args, out_dir=None, log=print):
    """Execute one soak run. Returns the raw dict (meta, requests, mem, server) that `build_report` consumes."""
    url = args.url.rstrip("/")
    st, _ = http_get(url, "/health")
    if st is None:
        raise SystemExit(f"server {url} unreachable (/health)")
    models = get_json(url, "/v1/models") or {}
    model = ((models.get("data") or [{}])[0]).get("id", "intarsia")
    plan_before = get_json(url, "/v1/intarsia/plan")
    src = PromptSource(args.prompts_dir, args.corpora)
    cpt = src.calibrate(url, model) if not args.no_calibrate else dict(src.cpt)
    plan = build_plan(args)
    if not args.no_warmup:  # first use of each prefill/decode shape compiles Metal pipelines: keep that out of the measured run
        pm = parse_mix(args.prompt_mix)
        for k, tgt in enumerate(sorted({min(v for v, _ in pm), max(v for v, _ in pm)})):
            wreq = {"i": -1 - k, "target_tokens": tgt, "max_tokens": 8, "golden": False, "jitter": 1.0}
            path, body, _ = src.body(wreq, model, args)
            w = stream_request(url, path, body, args.timeout, None, args.request_timeout)
            log(f"cbsoak: warmup {tgt} tokens -> {w['status']} ttft={w['ttft_ms'] and round(w['ttft_ms'])} ms")
    log(f"cbsoak: {url} model={model} rate={args.rate}/s duration={args.duration}s -> {len(plan)} requests planned, chars/token={ {k: round(v, 2) for k, v in cpt.items()} }")

    pid = args.pid
    if pid == "auto":
        pid = pid_from_port(url)
        log(f"cbsoak: --pid auto -> {pid}")
    pid = int(pid) if pid else None
    t0 = time.monotonic()
    mem_s = srv_s = None
    if pid:
        mem_s = Sampler(lambda: sample_process(pid, not args.no_footprint), args.sample_s, t0)
        mem_s.start()
    metrics_state = {"ok": None}

    def scrape():
        st, data = http_get(url, "/metrics", timeout=5)
        if st != 200:
            if metrics_state["ok"] is None:
                metrics_state["ok"] = False
            return None
        metrics_state["ok"] = True
        return {"m": parse_prom(data.decode(errors="replace"))}

    if not args.no_metrics:
        srv_s = Sampler(scrape, args.sample_s, t0)
        srv_s.start()

    results = [None] * len(plan)
    lock = threading.Lock()
    inflight = [0]
    threads = []

    def work(req):
        path, body, source = src.body(req, model, args)
        t_send_rel = time.monotonic() - t0
        rec = stream_request(url, path, body, args.timeout, req["cancel_ms"], args.request_timeout)
        rec.update({"i": req["i"], "t_send": t_send_rel, "bucket": req["bucket"], "target_tokens": req["target_tokens"],
                    "max_tokens": req["max_tokens"], "cancel_ms": req["cancel_ms"], "source": source, "path": path,
                    "inflight_at_send": req.get("inflight_at_send"),
                    "prompt_chars": len(body.get("prompt") or json.dumps(body.get("messages")))})
        with lock:
            inflight[0] -= 1
        results[req["i"]] = rec

    dropped = 0
    last_log = 0.0
    for req in plan:
        wait = req["t"] - (time.monotonic() - t0)
        if wait > 0:
            time.sleep(wait)
        with lock:
            if inflight[0] >= args.max_inflight:
                results[req["i"]] = {"i": req["i"], "t_send": time.monotonic() - t0, "status": "client_dropped", "bucket": req["bucket"],
                                     "target_tokens": req["target_tokens"], "max_tokens": req["max_tokens"], "itl_ms": [], "n_chunks": 0,
                                     "http_status": None, "ttft_ms": None, "usage": None, "finish_reason": None, "error": "client in-flight cap", "t_total_ms": 0.0, "source": None}
                dropped += 1
                continue
            req["inflight_at_send"] = inflight[0]  # other requests this client still has open: 0 = arrived to an idle client
            inflight[0] += 1
        th = threading.Thread(target=work, args=(req,), daemon=True)
        th.start()
        threads.append(th)
        now = time.monotonic() - t0
        if now - last_log >= 30:
            last_log = now
            done = sum(1 for r in results if r is not None)
            log(f"cbsoak: t={now:6.0f}s sent={req['i'] + 1} done={done} inflight={inflight[0]}")
    t_sched_end = time.monotonic() - t0
    if t_sched_end < args.duration:
        time.sleep(args.duration - t_sched_end)  # no trailing arrivals: still run to the full duration
    deadline = time.monotonic() + args.drain_s
    for th in threads:
        th.join(timeout=max(0.0, deadline - time.monotonic()))
    stuck = 0
    for req in plan:
        if results[req["i"]] is None:
            stuck += 1
            results[req["i"]] = {"i": req["i"], "t_send": req["t"], "status": "error", "error": f"still running after --drain-s {args.drain_s}",
                                 "bucket": req["bucket"], "target_tokens": req["target_tokens"], "max_tokens": req["max_tokens"], "itl_ms": [],
                                 "n_chunks": 0, "http_status": None, "ttft_ms": None, "usage": None, "finish_reason": None, "t_total_ms": None, "source": None}
    t_end = time.monotonic() - t0
    if mem_s:
        mem_s.finish()
    if srv_s:
        srv_s.finish()
    meta = {"url": url, "model": model, "started_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
            "wall_s": t_end, "duration_s": args.duration, "rate": args.rate, "seed": args.seed, "pid": pid, "chars_per_token": cpt,
            "plan_before": plan_before, "plan_after": get_json(url, "/v1/intarsia/plan"), "metrics_available": metrics_state["ok"],
            "client_dropped": dropped, "stuck_after_drain": stuck, "args": {k: v for k, v in vars(args).items() if k != "func"}}
    raw = {"meta": meta, "requests": results, "mem": mem_s.samples if mem_s else [], "server": srv_s.samples if srv_s else []}
    if out_dir:
        out_dir = pathlib.Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "raw.json").write_text(json.dumps(raw))
        with open(out_dir / "requests.jsonl", "w") as f:
            for r in results:
                f.write(json.dumps({k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items() if k != "itl_ms"} | {"itl_n": len(r.get("itl_ms") or [])}) + "\n")
    return raw


# ----------------------------------------------------------------------------------------------------------------------
# report
# ----------------------------------------------------------------------------------------------------------------------

def req_tokens(r):
    u = r.get("usage") or {}
    if r["status"] == "ok" and u.get("completion_tokens") is not None:
        return u["completion_tokens"]
    return r.get("n_chunks") or 0


def chunk_times(r):
    """Absolute (since run start, seconds) arrival times of every content chunk of one request."""
    if r.get("ttft_ms") is None:
        return []
    t = r["t_send"] + r["ttft_ms"] / 1000.0
    out = [t]
    for g in r.get("itl_ms") or []:
        t += g / 1000.0
        out.append(t)
    return out


def window_of(t, win):
    return int(t // win)


def server_summary(samples, t_from=0.0):
    """Summary of /metrics samples: gauges (queue, running), deltas of counters (rejected, batch-size histogram), server TTFT/ITL means."""
    if not samples:
        return {"available": False}
    ms = [s["m"] for s in samples]

    def series(name):
        return [m[name] for m in ms if name in m]

    def gauge(name):
        v = series(name)
        return {"max": max(v), "mean": sum(v) / len(v), "last": v[-1]} if v else None

    def delta(name):
        v = series(name)
        return v[-1] - v[0] if len(v) >= 2 else None

    rows = collections.OrderedDict()
    names = sorted({k for m in ms for k in m if k.startswith("batch_rows:")}, key=lambda k: int(k.split(":")[1]))
    for k in names:
        d = delta(k)
        if d is None and series(k):
            d = series(k)[-1]
        rows[int(k.split(":")[1])] = d
    tot = sum(v for v in rows.values() if v) or 0
    mean_b = sum(r * v for r, v in rows.items() if v) / tot if tot else None
    out = {"available": True, "n": len(samples),
           "running": gauge("intarsia_batch_running") or gauge("intarsia_requests_running"),
           "queue_depth": gauge("intarsia_batch_queue_depth") or gauge("intarsia_requests_waiting"),
           "queue_depth_max_server": gauge("intarsia_batch_queue_depth_max"),
           "step_ewma_ms": gauge("intarsia_batch_step_ewma_ms"),
           "kv_used_bytes": gauge("intarsia_kv_used_bytes"),
           "rejected_total_delta": delta("intarsia_requests_rejected_total"),
           "batch_rows_delta": dict(rows), "batch_mean_rows": mean_b,
           "prefill_chunks_delta": delta("intarsia_batch_prefill_chunks_total")}
    for nm, key in (("intarsia_ttft_seconds", "server_ttft_mean_ms"), ("intarsia_itl_seconds", "server_itl_mean_ms")):
        s, c = delta(nm + "_sum"), delta(nm + "_count")
        out[key] = s / c * 1000.0 if s is not None and c else None
    return out


def build_report(raw, window_s=30.0, mem_warmup_s=None, slo=None):
    meta, reqs = raw["meta"], raw["requests"]
    wall = max(meta["wall_s"], 1e-9)
    dur = meta["duration_s"]
    by = collections.Counter(r["status"] for r in reqs)
    http = collections.Counter(r["http_status"] for r in reqs if r.get("http_status") is not None)
    ok = [r for r in reqs if r["status"] == "ok"]
    sent = len(reqs) - by["client_dropped"]
    shed = by["shed"]
    err = by["error"]
    counts = {"planned": len(reqs), "sent": sent, "ok": by["ok"], "cancelled": by["cancelled"], "shed": shed,
              "shed_429": http[429], "shed_503": http[503], "error": err, "client_dropped": by["client_dropped"]}
    err_kinds = collections.Counter((r.get("error") or "")[:60] for r in reqs if r["status"] == "error")
    # latency samples: TTFT from every request that produced a first chunk (cancelled ones included), ITL from all chunk gaps
    ttfts = [r["ttft_ms"] for r in reqs if r.get("ttft_ms") is not None and r["status"] in ("ok", "cancelled")]
    itls = [g for r in reqs if r["status"] in ("ok", "cancelled") for g in (r.get("itl_ms") or [])]
    toks = sum(req_tokens(r) for r in reqs if r["status"] in ("ok", "cancelled"))
    chunks = sum(r.get("n_chunks") or 0 for r in ok)
    ct = sum((r.get("usage") or {}).get("completion_tokens") or 0 for r in ok)
    last_done = max([r["t_send"] + (r.get("t_total_ms") or 0) / 1000.0 for r in reqs] or [0.0])
    first_send = min([r["t_send"] for r in reqs] or [0.0])
    busy = max(last_done - first_send, 1e-9)
    per_req_decode = [(r["n_chunks"] - 1) / ((sum(r["itl_ms"])) / 1000.0) for r in ok if r.get("itl_ms") and sum(r["itl_ms"]) > 0]
    rates = {"offered_req_s": sent / dur if dur else None, "completed_req_s": by["ok"] / busy, "out_tok_s": toks / busy,
             "busy_window_s": busy, "shed_rate": shed / sent if sent else 0.0, "error_rate": err / sent if sent else 0.0,
             "cancel_rate": by["cancelled"] / sent if sent else 0.0, "goodput": by["ok"] / sent if sent else 0.0,
             "per_stream_decode_tok_s": summarize(per_req_decode), "chunks_per_completion_token": chunks / ct if ct else None}
    # per prompt-length bucket
    buckets = []
    for b in sorted({r["bucket"] for r in reqs}):
        rs = [r for r in reqs if r["bucket"] == b]
        rok = [r for r in rs if r["status"] in ("ok", "cancelled")]
        buckets.append({"bucket_tokens": b, "n": len(rs), "ok": sum(r["status"] == "ok" for r in rs), "shed": sum(r["status"] == "shed" for r in rs),
                        "error": sum(r["status"] == "error" for r in rs),
                        "prompt_tokens_mean": _mean([(r.get("usage") or {}).get("prompt_tokens") for r in rs]),
                        "ttft_ms": summarize([r["ttft_ms"] for r in rok if r.get("ttft_ms") is not None]),
                        "ttft_solo_ms": summarize([r["ttft_ms"] for r in rok if r.get("ttft_ms") is not None and r.get("inflight_at_send") == 0]),
                        "itl_ms": summarize([g for r in rok for g in (r.get("itl_ms") or [])])})
    # windows
    nwin = max(1, int(math.ceil(max(wall, dur) / window_s)))
    wins = [{"w": i, "t0": i * window_s, "sent": 0, "ok": 0, "shed": 0, "error": 0, "cancelled": 0, "ttft": [], "itl": [], "chunks": 0} for i in range(nwin)]
    for r in reqs:
        if r["status"] == "client_dropped":
            continue
        w = wins[min(nwin - 1, window_of(r["t_send"], window_s))]
        w["sent"] += 1
        if r["status"] in ("ok", "shed", "error", "cancelled"):
            w[r["status"]] += 1
        if r.get("ttft_ms") is not None and r["status"] in ("ok", "cancelled"):
            w["ttft"].append(r["ttft_ms"])
        if r["status"] in ("ok", "cancelled"):
            ts = chunk_times(r)
            for k, t in enumerate(ts):
                ww = wins[min(nwin - 1, window_of(t, window_s))]
                ww["chunks"] += 1
                if k > 0:
                    ww["itl"].append(r["itl_ms"][k - 1])
    mem_by = {}
    for s in raw.get("mem") or []:
        v = s.get("footprint_mb") if s.get("footprint_mb") is not None else s.get("rss_mb")
        if v is not None:
            mem_by[min(nwin - 1, window_of(s["t"], window_s))] = v
    srv_by = collections.defaultdict(list)
    for s in raw.get("server") or []:
        srv_by[min(nwin - 1, window_of(s["t"], window_s))].append(s["m"])
    windows = []
    for w in wins:
        q = [m.get("intarsia_batch_queue_depth", m.get("intarsia_requests_waiting")) for m in srv_by.get(w["w"], [])]
        q = [x for x in q if x is not None]
        windows.append({"window": w["w"], "t0_s": w["t0"], "sent": w["sent"], "ok": w["ok"], "shed": w["shed"], "error": w["error"],
                        "cancelled": w["cancelled"], "tok_s": w["chunks"] / window_s, "ttft_ms": summarize(w["ttft"]),
                        "itl_ms": summarize(w["itl"]), "queue_max": max(q) if q else None, "mem_mb": mem_by.get(w["w"])})
    # memory
    mem_key = "footprint_mb" if any(s.get("footprint_mb") is not None for s in raw.get("mem") or []) else "rss_mb"
    msamples = [{"t": s["t"], "mb": s.get(mem_key)} for s in raw.get("mem") or []]
    warm = mem_warmup_s if mem_warmup_s is not None else 0.2 * dur
    memory = analyze_memory(msamples, warm, "mb")
    memory["source"] = mem_key if memory.get("available") else None
    if memory.get("available") and any(s.get("rss_mb") is not None for s in raw.get("mem") or []):
        rs = [s["rss_mb"] for s in raw["mem"] if s.get("rss_mb") is not None]
        memory["rss_start_mb"], memory["rss_end_mb"], memory["rss_max_mb"] = rs[0], rs[-1], max(rs)
    rep = {"meta": {k: v for k, v in meta.items() if k != "args"} | {"cmdline_args": meta["args"]}, "counts": counts, "error_kinds": dict(err_kinds),
           "http_status": {str(k): v for k, v in http.items()}, "rates": rates, "ttft_ms": summarize(ttfts), "itl_ms": summarize(itls),
           "buckets": buckets, "windows": windows, "window_s": window_s, "memory": memory, "server": server_summary(raw.get("server") or [])}
    rep["slo"] = evaluate_slos(rep, slo or {})
    return rep


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


SLO_KEYS = {
    "max_ttft_p50_ms": ("TTFT p50 (ms)", lambda r: r["ttft_ms"]["p50"], "max"),
    "max_ttft_p95_ms": ("TTFT p95 (ms)", lambda r: r["ttft_ms"]["p95"], "max"),
    "max_ttft_p99_ms": ("TTFT p99 (ms)", lambda r: r["ttft_ms"]["p99"], "max"),
    "max_itl_p50_ms": ("ITL p50 (ms)", lambda r: r["itl_ms"]["p50"], "max"),
    "max_itl_p95_ms": ("ITL p95 (ms)", lambda r: r["itl_ms"]["p95"], "max"),
    "max_itl_p99_ms": ("ITL p99 (ms)", lambda r: r["itl_ms"]["p99"], "max"),
    "max_error_rate": ("error rate", lambda r: r["rates"]["error_rate"], "max"),
    "max_shed_rate": ("shed rate", lambda r: r["rates"]["shed_rate"], "max"),
    "max_mem_growth_mb": ("memory growth (fitted, MB)", lambda r: r["memory"].get("growth_mb") if r["memory"].get("available") else None, "max"),
    "min_out_tok_s": ("output tok/s", lambda r: r["rates"]["out_tok_s"], "min"),
    "min_req_s": ("completed req/s", lambda r: r["rates"]["completed_req_s"], "min"),
}


def evaluate_slos(rep, thresholds):
    """thresholds: {key: value or None}. A threshold whose metric is unmeasurable (no samples) is a violation, never a silent pass."""
    checks = []
    for key, (label, fn, kind) in SLO_KEYS.items():
        lim = thresholds.get(key)
        if lim is None:
            continue
        val = fn(rep)
        if val is None:
            checks.append({"key": key, "label": label, "limit": lim, "value": None, "ok": False, "why": "not measurable"})
        else:
            ok = val <= lim if kind == "max" else val >= lim
            checks.append({"key": key, "label": label, "limit": lim, "value": val, "ok": ok})
    return {"checks": checks, "violations": [c for c in checks if not c["ok"]], "pass": all(c["ok"] for c in checks)}


def f(x, nd=1, none="-"):
    return none if x is None else f"{x:.{nd}f}"


def render_md(rep):
    c, r, m = rep["counts"], rep["rates"], rep["memory"]
    meta = rep["meta"]
    L = [f"# cbsoak report", "",
         f"{meta['url']}  model `{meta['model']}`  started {meta['started_at']}  wall {meta['wall_s']:.0f} s  "
         f"planned rate {meta['rate']} req/s over {meta['duration_s']} s  seed {meta['seed']}", "",
         "## Totals", "",
         "| sent | ok | cancelled | shed (429/503) | error | client_dropped |", "|---|---|---|---|---|---|",
         f"| {c['sent']} | {c['ok']} | {c['cancelled']} | {c['shed']} ({c['shed_429']}/{c['shed_503']}) | {c['error']} | {c['client_dropped']} |", "",
         f"offered {f(r['offered_req_s'], 3)} req/s, completed {f(r['completed_req_s'], 3)} req/s, output {f(r['out_tok_s'])} tok/s "
         f"(over the {r['busy_window_s']:.0f} s from first send to last completion); shed rate {r['shed_rate']:.2%}, error rate {r['error_rate']:.2%}, "
         f"cancel rate {r['cancel_rate']:.2%}, goodput {r['goodput']:.2%}; chunks per completion token {f(r['chunks_per_completion_token'], 3)} "
         f"(1.0 means ITL is per token).", ""]
    if rep["error_kinds"]:
        L += ["Error kinds: " + "; ".join(f"`{k}` x{v}" for k, v in rep["error_kinds"].items()), ""]
    L += ["## Latency", "", "| metric | n | mean | p50 | p95 | p99 | max |", "|---|---|---|---|---|---|---|"]
    for nm, s in (("TTFT (ms)", rep["ttft_ms"]), ("ITL (ms)", rep["itl_ms"]), ("per-stream decode (tok/s)", r["per_stream_decode_tok_s"])):
        L.append(f"| {nm} | {s['n']} | {f(s['mean'])} | {f(s['p50'])} | {f(s['p95'])} | {f(s['p99'])} | {f(s['max'])} |")
    L += ["", "## By prompt length", "", "solo TTFT = requests that found no other request open on the client when they were sent (queue-free reference).", "", "| bucket (tokens) | prompt tok (mean) | n | ok | shed | err | TTFT p50 | TTFT p95 | solo TTFT p50 (n) | ITL p50 | ITL p95 |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for b in rep["buckets"]:
        L.append(f"| {b['bucket_tokens']} | {f(b['prompt_tokens_mean'], 0)} | {b['n']} | {b['ok']} | {b['shed']} | {b['error']} | "
                 f"{f(b['ttft_ms']['p50'])} | {f(b['ttft_ms']['p95'])} | {f(b['ttft_solo_ms']['p50'])} ({b['ttft_solo_ms']['n']}) | {f(b['itl_ms']['p50'])} | {f(b['itl_ms']['p95'])} |")
    L += ["", f"## Per {rep['window_s']:.0f} s window (requests by send time, ITL and tok/s by chunk arrival time)", "",
          "| window | t0 (s) | sent | ok | shed | err | canc | tok/s | TTFT p50 | TTFT p95 | ITL p50 | ITL p95 | ITL p99 | queue max | mem (MB) |", "|" + "---|" * 15]
    for w in rep["windows"]:
        L.append(f"| {w['window']} | {w['t0_s']:.0f} | {w['sent']} | {w['ok']} | {w['shed']} | {w['error']} | {w['cancelled']} | {f(w['tok_s'])} | "
                 f"{f(w['ttft_ms']['p50'], 0)} | {f(w['ttft_ms']['p95'], 0)} | {f(w['itl_ms']['p50'])} | {f(w['itl_ms']['p95'])} | {f(w['itl_ms']['p99'])} | "
                 f"{f(w['queue_max'], 0)} | {f(w['mem_mb'], 0)} |")
    L += ["", "## Memory", ""]
    if m.get("available"):
        L += [f"source `{m['source']}`, {m['n']} samples; start {f(m['start_mb'], 0)} MB, end {f(m['end_mb'], 0)} MB, max {f(m['max_mb'], 0)} MB "
              f"(end - start {f(m['end_minus_start_mb'], 0)} MB).", "",
              f"Growth fit over t >= {m['warmup_s']:.0f} s ({m['n_fit']} samples, {f(m['fit_span_min'], 1)} min): Theil-Sen slope {f(m['slope_mb_per_min'], 2)} MB/min "
              f"(OLS {f(m['slope_ols_mb_per_min'], 2)}, r2 {f(m['r2'], 2)}), fitted growth {f(m['growth_mb'], 0)} MB, projected {f(m['projected_mb_per_hour'], 0)} MB/h. "
              f"Leak suspect: **{'YES' if m['leak_suspect'] else 'no'}** (growth >= 100 MB and r2 >= 0.6)."]
    else:
        L += ["no memory samples (run without --pid, or the process could not be sampled)."]
    s = rep["server"]
    L += ["", "## Server (/metrics)", ""]
    if s.get("available"):
        def g(x):
            return "-" if not x else f"max {f(x['max'], 1)} / mean {f(x['mean'], 2)}"
        L += [f"{s['n']} samples. running: {g(s['running'])}. queue depth: {g(s['queue_depth'])}. step EWMA ms: {g(s['step_ewma_ms'])}. "
              f"rejected (delta): {f(s['rejected_total_delta'], 0)}. server TTFT mean {f(s['server_ttft_mean_ms'])} ms, ITL mean {f(s['server_itl_mean_ms'])} ms. "
              f"prefill chunks (delta) {f(s['prefill_chunks_delta'], 0)}."]
        if s["batch_rows_delta"]:
            tot = sum(v for v in s["batch_rows_delta"].values() if v) or 1
            L += ["", "Batch-size histogram (decode passes by rows, delta over the run):", "", "| rows | passes | share |", "|---|---|---|"]
            L += [f"| {k} | {f(v, 0)} | {f((v or 0) / tot * 100, 1)}% |" for k, v in s["batch_rows_delta"].items()]
            L += ["", f"mean rows per decode pass {f(s['batch_mean_rows'], 2)}"]
    else:
        L += ["`/metrics` not available (or no `intarsia_*` series)."]
    slo = rep["slo"]
    L += ["", "## SLO", ""]
    if slo["checks"]:
        L += ["| check | limit | value | |", "|---|---|---|---|"]
        L += [f"| {k['label']} | {k['limit']} | {f(k['value'], 3) if k['value'] is not None else 'n/a'} | {'ok' if k['ok'] else 'VIOLATED'} |" for k in slo["checks"]]
        L += ["", f"**{'PASS' if slo['pass'] else 'FAIL'}**"]
    else:
        L += ["no thresholds given (informational run)."]
    return "\n".join(L) + "\n"


def slo_from_args(args):
    return {k: getattr(args, k, None) for k in SLO_KEYS}


def emit(rep, out_dir, quiet=False):
    md = render_md(rep)
    if out_dir:
        out_dir = pathlib.Path(out_dir)
        (out_dir / "report.json").write_text(json.dumps(rep, indent=1))
        (out_dir / "report.md").write_text(md)
    if not quiet:
        print(md)
    return 0 if rep["slo"]["pass"] else 1


# ----------------------------------------------------------------------------------------------------------------------
# commands
# ----------------------------------------------------------------------------------------------------------------------

def add_slo_args(p):
    p.add_argument("--max-ttft-p50-ms", type=float)
    p.add_argument("--max-ttft-p95-ms", type=float)
    p.add_argument("--max-ttft-p99-ms", type=float)
    p.add_argument("--max-itl-p50-ms", type=float)
    p.add_argument("--max-itl-p95-ms", type=float)
    p.add_argument("--max-itl-p99-ms", type=float)
    p.add_argument("--max-error-rate", type=float, help="errors / sent (HTTP errors other than 429/503, timeouts, broken streams)")
    p.add_argument("--max-shed-rate", type=float, help="429+503 / sent")
    p.add_argument("--max-mem-growth-mb", type=float, help="fitted (Theil-Sen) memory growth over the post-warmup span")
    p.add_argument("--min-out-tok-s", type=float)
    p.add_argument("--min-req-s", type=float)
    p.add_argument("--window-s", type=float, default=30.0)
    p.add_argument("--mem-warmup-s", type=float, default=None, help="exclude the first N s from the growth fit (default 20%% of --duration)")


def cmd_run(args):
    if args.rate <= 0 or args.duration <= 0:
        print("--rate and --duration must be positive", file=sys.stderr)
        return 2
    try:
        raw = run_load(args, args.out)
    except SystemExit as e:
        print(e, file=sys.stderr)
        return 2
    rep = build_report(raw, args.window_s, args.mem_warmup_s, slo_from_args(args))
    return emit(rep, args.out)


def cmd_report(args):
    raw = json.loads(pathlib.Path(args.raw).read_text())
    rep = build_report(raw, args.window_s, args.mem_warmup_s, slo_from_args(args))
    return emit(rep, args.out or pathlib.Path(args.raw).parent)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="open-loop soak against a running server")
    r.add_argument("--url", required=True)
    r.add_argument("--rate", type=float, required=True, help="Poisson arrival rate, requests/s")
    r.add_argument("--duration", type=float, required=True, help="seconds of arrivals")
    r.add_argument("--seed", type=int, default=1)
    r.add_argument("--prompt-mix", default=DEFAULT_PROMPT_MIX, help="tokens:weight list, default %(default)s")
    r.add_argument("--max-tokens-mix", default=DEFAULT_TOKENS_MIX, help="tokens:weight list, default %(default)s")
    r.add_argument("--max-seq", type=int, default=8192, help="clip prompt+max_tokens to this context (0 = no clip)")
    r.add_argument("--corpora", default="docs:1,code:1", help="corpus_<name>.txt files and weights")
    r.add_argument("--golden-frac", type=float, default=0.25, help="share of >=3000-token requests built from golden_4k.jsonl cases (incl. chat)")
    r.add_argument("--prompts-dir", default=str(PROMPTS))
    r.add_argument("--no-nonce", action="store_true", help="no per-request nonce prefix (allows KV prefix reuse across requests)")
    r.add_argument("--no-warmup", action="store_true", help="skip the two unmeasured warmup requests (smallest and largest prompt bucket)")
    r.add_argument("--no-calibrate", action="store_true", help="skip the chars-per-token calibration requests (4.0 assumed)")
    r.add_argument("--temperature", type=float, default=0.0)
    r.add_argument("--ignore-eos", dest="ignore_eos", action="store_true", default=True, help="force max_tokens output length (default)")
    r.add_argument("--no-ignore-eos", dest="ignore_eos", action="store_false")
    r.add_argument("--cancel-frac", type=float, default=0.0, help="share of requests the client abandons mid-flight")
    r.add_argument("--cancel-min-ms", type=float, default=300.0)
    r.add_argument("--cancel-max-ms", type=float, default=6000.0)
    r.add_argument("--timeout", type=float, default=300.0, help="per-read socket timeout (s): max wait for the first byte / next chunk")
    r.add_argument("--request-timeout", type=float, default=900.0, help="total time per request (s)")
    r.add_argument("--drain-s", type=float, default=300.0, help="after the arrivals end, wait this long for in-flight requests")
    r.add_argument("--max-inflight", type=int, default=512, help="client-side cap; arrivals above it are counted as client_dropped")
    r.add_argument("--pid", default=None, help="server PID (or 'auto': lsof on the URL port) for memory sampling")
    r.add_argument("--no-footprint", action="store_true", help="sample ps RSS only (footprint can take ~0.5 s per call)")
    r.add_argument("--no-metrics", action="store_true", help="do not scrape /metrics")
    r.add_argument("--sample-s", type=float, default=5.0)
    r.add_argument("--out", help="output dir: raw.json, requests.jsonl, report.json, report.md")
    add_slo_args(r)
    r.set_defaults(func=cmd_run)
    rp = sub.add_parser("report", help="re-render / re-judge a finished run from raw.json")
    rp.add_argument("--raw", required=True)
    rp.add_argument("--out")
    add_slo_args(rp)
    rp.set_defaults(func=cmd_report)
    st = sub.add_parser("selftest", help="offline: in-process mock server")
    st.set_defaults(func=lambda a: cmd_selftest(a))
    args = ap.parse_args()
    sys.exit(args.func(args))


# ----------------------------------------------------------------------------------------------------------------------
# mock server (offline self-test): streaming completions with configurable timing, concurrency cap and shedding
# ----------------------------------------------------------------------------------------------------------------------

def make_mock(first_ms=30.0, itl_ms=8.0, max_conc=None, shed_code=429, metrics=True, slow_after=None):
    """slow_after: seconds after server start from which every token takes 4x longer (a degradation to detect)."""
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    st = {"inflight": 0, "lock": threading.Lock(), "shed": 0, "served": 0, "disconnects": 0, "peak": 0, "t0": time.monotonic(), "tokens": 0}

    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):
            pass

        def _send(self, code, body, ctype="application/json", extra=()):
            b = body if isinstance(body, bytes) else json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(b)))
            self.send_header("Connection", "close")
            for k, v in extra:
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(b)

        def do_GET(self):
            if self.path == "/metrics" and metrics:
                with st["lock"]:
                    txt = (f"# TYPE intarsia_requests_running gauge\nintarsia_requests_running {st['inflight']}\n"
                           f"intarsia_requests_waiting 0\nintarsia_requests_rejected_total {st['shed']}\n"
                           f'intarsia_batch_decode_rows_total{{rows="1"}} {st["tokens"]}\n'
                           f'intarsia_batch_decode_rows_total{{rows="2"}} {st["tokens"] // 2}\n'
                           f"intarsia_batch_queue_depth {max(0, st['inflight'] - 1)}\n")
                return self._send(200, txt.encode(), "text/plain")
            if self.path == "/v1/intarsia/plan":
                return self._send(200, {"mock": True})
            if self.path == "/v1/models":
                return self._send(200, {"data": [{"id": "mock"}]})
            if self.path == "/health":
                return self._send(200, {"status": "ok"})
            return self._send(404, {"error": "nope"})

        def do_POST(self):
            req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            text = req.get("prompt") if "prompt" in req else json.dumps(req.get("messages"))
            ptoks = max(1, len(text) // 4)
            with st["lock"]:
                if max_conc is not None and st["inflight"] >= max_conc:
                    st["shed"] += 1
                    return self._send(shed_code, {"error": {"message": "queue full", "type": "rate_limit_error"}}, extra=[("Retry-After", "1")])
                st["inflight"] += 1
                st["peak"] = max(st["peak"], st["inflight"])
            try:
                if not req.get("stream"):
                    return self._send(200, {"choices": [{"text": "x", "finish_reason": "length"}], "usage": {"prompt_tokens": ptoks, "completion_tokens": 1}})
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Connection", "close")
                self.end_headers()
                chat = self.path.endswith("chat/completions")
                n = req.get("max_tokens", 16)
                try:
                    time.sleep(first_ms / 1000.0)
                    for k in range(n):
                        if k:
                            slow = slow_after is not None and time.monotonic() - st["t0"] >= slow_after
                            time.sleep(itl_ms * (4.0 if slow else 1.0) / 1000.0)
                        ch = {"index": 0, "finish_reason": None, **({"delta": {"content": " w"}} if chat else {"text": " w"})}
                        self.wfile.write(b"data: " + json.dumps({"choices": [ch]}).encode() + b"\n\n")
                        self.wfile.flush()
                        with st["lock"]:
                            st["tokens"] += 1
                    fin = {"index": 0, "finish_reason": "length", **({"delta": {}} if chat else {"text": ""})}
                    self.wfile.write(b"data: " + json.dumps({"choices": [fin]}).encode() + b"\n\n")
                    self.wfile.write(b"data: " + json.dumps({"choices": [], "usage": {"prompt_tokens": ptoks, "completion_tokens": n}}).encode() + b"\n\n")
                    self.wfile.write(b"data: [DONE]\n\n")
                    self.wfile.flush()
                    with st["lock"]:
                        st["served"] += 1
                except (BrokenPipeError, ConnectionResetError, OSError):
                    with st["lock"]:
                        st["disconnects"] += 1
            finally:
                with st["lock"]:
                    st["inflight"] -= 1

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    srv.daemon_threads = True
    srv.state = st
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}"


def cmd_selftest(_args):
    fails = []

    def check(name, cond, detail=""):
        print(("ok   " if cond else "FAIL ") + name + (f"   [{detail}]" if detail and not cond else ""), flush=True)
        if not cond:
            fails.append(name)

    def near(a, b, rel):
        return a is not None and abs(a - b) <= rel * abs(b)

    # --- percentile math -------------------------------------------------------------------------------------------
    xs = list(range(1, 101))
    check("percentile p50 of 1..100 = 50.5", percentile(xs, 50) == 50.5)
    check("percentile p95 of 1..100 = 95.05", abs(percentile(xs, 95) - 95.05) < 1e-9)
    check("percentile p99 of 1..100 = 99.01", abs(percentile(xs, 99) - 99.01) < 1e-9)
    check("percentile edge cases", percentile([], 50) is None and percentile([7], 99) == 7.0 and percentile([3, 1, 2], 100) == 3.0 and percentile([3, 1, 2], 0) == 1.0)
    s = summarize([10.0, 20.0, 30.0, 40.0])
    check("summarize", s["n"] == 4 and s["mean"] == 25.0 and s["p50"] == 25.0 and s["max"] == 40.0)

    # --- Poisson schedule ------------------------------------------------------------------------------------------
    rate, dur = 20.0, 400.0
    arr = poisson_schedule(rate, dur, seed=7)
    gaps = [b - a for a, b in zip([0.0] + arr[:-1], arr)]
    mean_gap = sum(gaps) / len(gaps)
    cv = (sum((g - mean_gap) ** 2 for g in gaps) / len(gaps)) ** 0.5 / mean_gap
    frac_long = sum(1 for g in gaps if g > 1.0 / rate) / len(gaps)
    counts = collections.Counter(int(t // 10) for t in arr)  # per-10 s counts: Poisson(200), variance ~ mean
    cm = sum(counts.values()) / 40
    cvar = sum((counts.get(i, 0) - cm) ** 2 for i in range(40)) / 39
    check("poisson: count within 3% of rate*duration", near(len(arr), rate * dur, 0.03), f"{len(arr)} vs {rate * dur}")
    check("poisson: mean gap within 3% of 1/rate", near(mean_gap, 1 / rate, 0.03), f"{mean_gap}")
    check("poisson: gap CV ~ 1 (exponential)", 0.93 < cv < 1.07, f"cv={cv:.3f}")
    check("poisson: P(gap > mean) ~ e^-1", abs(frac_long - math.exp(-1)) < 0.02, f"{frac_long:.3f}")
    check("poisson: per-10s count dispersion (var/mean) ~ 1", 0.5 < cvar / cm < 1.8, f"{cvar / cm:.2f}")
    check("poisson: deterministic in seed, different across seeds", arr == poisson_schedule(rate, dur, 7) and arr != poisson_schedule(rate, dur, 8))
    check("poisson: sorted, inside [0, duration)", arr == sorted(arr) and arr[0] >= 0 and arr[-1] < dur)

    # --- mix sampling ----------------------------------------------------------------------------------------------
    rng = random.Random(3)
    mix = parse_mix("128:1,4096:3")
    dr = collections.Counter(draw(rng, mix) for _ in range(8000))
    check("mix: weights respected", near(dr[4096] / 8000, 0.75, 0.05), str(dr))
    a = argparse.Namespace(rate=10.0, duration=100.0, seed=2, prompt_mix="128:1,4096:1", max_tokens_mix="64:1", max_seq=4200, cancel_frac=0.3,
                           cancel_min_ms=100.0, cancel_max_ms=900.0, golden_frac=0.5)
    plan = build_plan(a)
    cf = sum(1 for p in plan if p["cancel_ms"] is not None) / len(plan)
    check("plan: cancel share ~ --cancel-frac", near(cf, 0.3, 0.15), f"{cf:.3f}")
    check("plan: cancel times inside the configured range", all(100 <= p["cancel_ms"] <= 900 for p in plan if p["cancel_ms"] is not None))
    check("plan: prompt+max_tokens clipped to --max-seq", all(p["target_tokens"] + p["max_tokens"] + 8 <= 4200 for p in plan))
    check("plan: deterministic", build_plan(a) == plan)

    # --- memory parsing and leak slope ----------------------------------------------------------------------------
    check("parse_footprint bytes/GB/KB", near(parse_footprint("zsh [1]: 64-bit    Footprint: 1933720 B (16384 bytes per page)"), 1933720 / 2**20, 1e-9)
          and near(parse_footprint("x Footprint: 12 GB (16384 bytes"), 12288.0, 1e-9) and near(parse_footprint("Footprint: 1888 KB (1"), 1888 / 1024, 1e-9)
          and parse_footprint("nothing") is None)
    check("parse_ps_rss", near(parse_ps_rss("  2048\n"), 2.0, 1e-9) and parse_ps_rss("") is None)
    rng = random.Random(11)
    # growing: 600 MB base, +10 MB/min, noise +-8 MB, samples every 5 s for 30 min, with two big allocator spikes
    grow = [{"t": t, "mb": 600 + 10 * t / 60 + rng.uniform(-8, 8) + (400 if 700 < t < 715 else 0)} for t in range(0, 1800, 5)]
    ga = analyze_memory(grow, warmup_s=120)
    check("leak: slope of a +10 MB/min series recovered within 10%", near(ga["slope_mb_per_min"], 10.0, 0.10), f"{ga['slope_mb_per_min']}")
    check("leak: robust to allocator spikes (Theil-Sen)", near(ga["slope_mb_per_min"], 10.0, 0.10))
    check("leak: growing series is flagged", ga["leak_suspect"] and ga["growth_mb"] > 200, f"growth={ga['growth_mb']}")
    check("leak: start/end/max reported", near(ga["start_mb"], 600, 0.02) and ga["max_mb"] > 1000 and ga["n"] == 360)
    flat = [{"t": t, "mb": 5000 + rng.uniform(-30, 30) + (500 if 300 < t < 310 else 0)} for t in range(0, 1800, 5)]
    fa = analyze_memory(flat, warmup_s=120)
    check("leak: flat noisy series (with a spike) has ~0 slope and is not flagged", abs(fa["slope_mb_per_min"]) < 0.5 and not fa["leak_suspect"], f"{fa['slope_mb_per_min']}")
    plateau = [{"t": t, "mb": 5000 + min(t, 200)} for t in range(0, 1800, 5)]
    pa = analyze_memory(plateau, warmup_s=300)
    check("leak: warmup growth that plateaus before the fit window is ignored", abs(pa["slope_mb_per_min"]) < 0.1 and not pa["leak_suspect"])
    check("leak: no samples -> unavailable", analyze_memory([])["available"] is False)
    me = sample_process(os.getpid())
    check("sample_process reads this process via ps (and footprint on macOS)", me["rss_mb"] and me["rss_mb"] > 5 and (sys.platform != "darwin" or me["footprint_mb"] is None or me["footprint_mb"] > 5), str(me))

    # --- SLO evaluation --------------------------------------------------------------------------------------------
    fake = {"ttft_ms": {"p50": 100, "p95": 400, "p99": 900}, "itl_ms": {"p50": 20, "p95": 80, "p99": 200}, "rates": {"error_rate": 0.0, "shed_rate": 0.1,
            "out_tok_s": 30.0, "completed_req_s": 0.4}, "memory": {"available": True, "growth_mb": 50.0}}
    ok = evaluate_slos(fake, {"max_itl_p95_ms": 100, "max_ttft_p95_ms": 500, "max_error_rate": 0.001, "max_mem_growth_mb": 100, "min_out_tok_s": 25})
    bad = evaluate_slos(fake, {"max_itl_p95_ms": 50, "max_shed_rate": 0.05, "max_mem_growth_mb": 10})
    nom = evaluate_slos({**fake, "memory": {"available": False}}, {"max_mem_growth_mb": 10})
    check("slo: all within limits passes", ok["pass"] and len(ok["checks"]) == 5)
    check("slo: violations listed", not bad["pass"] and {v["key"] for v in bad["violations"]} == {"max_itl_p95_ms", "max_shed_rate", "max_mem_growth_mb"})
    check("slo: unmeasurable metric with a threshold is a violation", not nom["pass"] and nom["violations"][0]["why"] == "not measurable")
    check("slo: no thresholds -> pass", evaluate_slos(fake, {})["pass"])

    # --- end-to-end against the mock ------------------------------------------------------------------------------
    base = dict(prompts_dir=str(PROMPTS), corpora="docs:1,code:1", golden_frac=0.3, no_nonce=False, no_calibrate=False, no_warmup=True, temperature=0.0, ignore_eos=True,
                cancel_frac=0.0, cancel_min_ms=300.0, cancel_max_ms=6000.0, timeout=20.0, request_timeout=60.0, drain_s=20.0, max_inflight=512,
                pid=None, no_footprint=True, no_metrics=False, sample_s=0.5, out=None, max_seq=8192, seed=5, prompt_mix="128:1,1024:1,4096:1", max_tokens_mix="12:1,24:1")
    quiet = lambda *_: None  # noqa: E731

    # (1) clean run: timings, accounting, /metrics, windows, memory sampler on this very process
    srv, url = make_mock(first_ms=40, itl_ms=10)
    a = argparse.Namespace(url=url, **{**base, "rate": 6.0, "duration": 6.0, "pid": str(os.getpid())})
    raw = run_load(a, None, quiet)
    rep = build_report(raw, window_s=2.0, mem_warmup_s=1.0)
    c = rep["counts"]
    check("e2e: every sent request is accounted for", c["sent"] == c["ok"] + c["cancelled"] + c["shed"] + c["error"] and c["sent"] == len(raw["requests"]) and c["sent"] > 15, str(c))
    check("e2e: no errors, no shed, no client drops", c["error"] == 0 and c["shed"] == 0 and c["client_dropped"] == 0, f"{c} {rep['error_kinds']}")
    check("e2e: TTFT p50 ~ mock first-token delay (40 ms)", 35 < rep["ttft_ms"]["p50"] < 120, f"{rep['ttft_ms']['p50']}")
    check("e2e: ITL p50 ~ mock inter-token delay (10 ms)", 9 < rep["itl_ms"]["p50"] < 25, f"{rep['itl_ms']['p50']}")
    check("e2e: chunks per completion token = 1", near(rep["rates"]["chunks_per_completion_token"], 1.0, 0.001))
    check("e2e: output tok/s positive, req/s ~ offered", rep["rates"]["out_tok_s"] > 20 and 0.5 * rep["rates"]["offered_req_s"] < rep["rates"]["completed_req_s"] < 1.6 * rep["rates"]["offered_req_s"], str(rep["rates"]))
    check("e2e: 3 windows of 2 s, sums match", len(rep["windows"]) >= 3 and sum(w["sent"] for w in rep["windows"]) == c["sent"])
    check("e2e: all three prompt buckets present with ~prompt tokens", [b["bucket_tokens"] for b in rep["buckets"]] == [128, 1024, 4096]
          and all(b["prompt_tokens_mean"] and 0.5 * b["bucket_tokens"] < b["prompt_tokens_mean"] < 1.6 * b["bucket_tokens"] for b in rep["buckets"]),
          str([(b["bucket_tokens"], b["prompt_tokens_mean"]) for b in rep["buckets"]]))
    check("e2e: memory sampled from --pid", rep["memory"]["available"] and rep["memory"]["n"] >= 5, str(rep["memory"].get("n")))
    sv = rep["server"]
    check("e2e: /metrics parsed (running gauge, batch-size histogram)", sv["available"] and sv["running"] is not None and set(sv["batch_rows_delta"]) == {1, 2} and sv["batch_mean_rows"] is not None, str(sv))
    check("e2e: markdown renders", "## Per 2 s window" in render_md(rep) and "Batch-size histogram" in render_md(rep))
    srv.shutdown()

    # (1b) warmup: two unmeasured requests ahead of the first arrival, not part of the report
    srv, url = make_mock(first_ms=5, itl_ms=2)
    a = argparse.Namespace(url=url, **{**base, "rate": 3.0, "duration": 2.0, "no_warmup": False, "no_calibrate": True, "max_tokens_mix": "4:1"})
    raw = run_load(a, None, quiet)
    check("warmup: 2 extra requests served, none in the request list", srv.state["served"] == len(raw["requests"]) + 2, f"{srv.state['served']} vs {len(raw['requests'])}")
    srv.shutdown()

    # (2) shedding: cap 2 concurrent, flood -> 429s counted as shed, not errors; a 503 variant too
    for code in (429, 503):
        srv, url = make_mock(first_ms=100, itl_ms=20, max_conc=2, shed_code=code)
        a = argparse.Namespace(url=url, **{**base, "rate": 12.0, "duration": 4.0, "max_tokens_mix": "30:1", "prompt_mix": "128:1", "no_calibrate": True})
        raw = run_load(a, None, quiet)
        rep = build_report(raw, window_s=2.0)
        c = rep["counts"]
        check(f"shed {code}: counted as shed, not as error", c["shed"] > 5 and c["error"] == 0 and c[f"shed_{code}"] == c["shed"], str(c))
        check(f"shed {code}: server-side shed count matches the client's", srv.state["shed"] == c["shed"], f"{srv.state['shed']} vs {c['shed']}")
        check(f"shed {code}: accounting closes and shed rate in (0,1)", c["sent"] == c["ok"] + c["shed"] + c["cancelled"] + c["error"] and 0 < rep["rates"]["shed_rate"] < 1)
        check(f"shed {code}: mock never exceeded its cap", srv.state["peak"] <= 2)
        sl = evaluate_slos(rep, {"max_shed_rate": 0.01})
        check(f"shed {code}: --max-shed-rate violated -> fail", not sl["pass"])
        srv.shutdown()

    # (3) cancellations: every request abandoned at 150-250 ms (first token at 40 ms, then 10 ms ticks, 100 tokens = 1 s)
    srv, url = make_mock(first_ms=40, itl_ms=10)
    a = argparse.Namespace(url=url, **{**base, "rate": 8.0, "duration": 3.0, "cancel_frac": 1.0, "cancel_min_ms": 150.0, "cancel_max_ms": 250.0,
                                       "max_tokens_mix": "100:1", "prompt_mix": "128:1", "no_calibrate": True})
    raw = run_load(a, None, quiet)
    rep = build_report(raw, window_s=2.0)
    c = rep["counts"]
    time.sleep(0.5)
    check("cancel: all requests cancelled, none errored", c["cancelled"] == c["sent"] and c["error"] == 0 and c["sent"] > 8, str(c))
    check("cancel: server saw the disconnects", srv.state["disconnects"] >= 0.9 * c["sent"] and srv.state["served"] == 0, f"{srv.state['disconnects']} / {c['sent']}")
    check("cancel: cancelled requests still contribute TTFT/ITL samples, not completions", rep["ttft_ms"]["n"] > 0 and rep["itl_ms"]["n"] > 0 and rep["counts"]["ok"] == 0)
    check("cancel: cancelled requests are not errors in the rates", rep["rates"]["error_rate"] == 0 and near(rep["rates"]["cancel_rate"], 1.0, 0.001))
    srv.shutdown()
    srv, url = make_mock(first_ms=40, itl_ms=10)
    a = argparse.Namespace(url=url, **{**base, "rate": 10.0, "duration": 6.0, "cancel_frac": 0.5, "cancel_min_ms": 100.0, "cancel_max_ms": 200.0,
                                       "max_tokens_mix": "60:1", "prompt_mix": "128:1", "no_calibrate": True})
    rep = build_report(run_load(a, None, quiet), window_s=2.0)
    c = rep["counts"]
    check("cancel: --cancel-frac 0.5 gives a mixed ok/cancelled run", c["cancelled"] > 0.25 * c["sent"] and c["ok"] > 0.25 * c["sent"] and c["error"] == 0, str(c))
    srv.shutdown()

    # (4) degradation shows up in the per-window table (server becomes 4x slower after 3 s)
    srv, url = make_mock(first_ms=20, itl_ms=8, slow_after=3.0)
    a = argparse.Namespace(url=url, **{**base, "rate": 4.0, "duration": 8.0, "max_tokens_mix": "30:1", "prompt_mix": "128:1", "no_calibrate": True})
    rep = build_report(run_load(a, None, quiet), window_s=2.0)
    w = rep["windows"]
    check("windows: late-window ITL p50 shows the 4x slowdown", w[0]["itl_ms"]["p50"] is not None and w[-2]["itl_ms"]["p50"] is not None and w[-2]["itl_ms"]["p50"] > 2.5 * w[0]["itl_ms"]["p50"],
          f"{w[0]['itl_ms']['p50']} -> {w[-2]['itl_ms']['p50']}")
    srv.shutdown()

    # (5) /metrics absent and server errors / timeouts / refused connection
    srv, url = make_mock(metrics=False)
    a = argparse.Namespace(url=url, **{**base, "rate": 4.0, "duration": 2.0, "max_tokens_mix": "8:1", "prompt_mix": "128:1", "no_calibrate": True})
    raw = run_load(a, None, quiet)
    rep = build_report(raw, window_s=1.0)
    check("no /metrics: run completes, server section says unavailable", rep["server"]["available"] is False and raw["meta"]["metrics_available"] is False and rep["counts"]["ok"] > 0)
    check("no --pid: memory section says unavailable and the --max-mem-growth-mb SLO fails closed", not rep["memory"]["available"] and not evaluate_slos(rep, {"max_mem_growth_mb": 100})["pass"])
    srv.shutdown()
    rec = stream_request(url, "/v1/completions", {"prompt": "x", "stream": True, "max_tokens": 2}, 2.0)
    check("client: refused connection is an error record, not an exception", rec["status"] == "error" and rec["http_status"] is None, str(rec["error"]))
    srv, url = make_mock(first_ms=1500, itl_ms=5)
    rec = stream_request(url, "/v1/completions", {"prompt": "x", "stream": True, "max_tokens": 2}, 0.4)
    check("client: read timeout is an error record", rec["status"] == "error" and "timeout" in rec["error"], str(rec))
    srv.shutdown()

    # (6) CLI exit codes: report from raw.json, SLO violated -> 1, satisfied -> 0
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        srv, url = make_mock(first_ms=20, itl_ms=10)
        a = argparse.Namespace(url=url, **{**base, "rate": 5.0, "duration": 3.0, "max_tokens_mix": "20:1", "prompt_mix": "128:1", "no_calibrate": True, "out": td})
        raw = run_load(a, td, quiet)
        srv.shutdown()
        me = pathlib.Path(__file__).resolve()
        run = lambda *x: subprocess.run([sys.executable, str(me), "report", "--raw", f"{td}/raw.json", *x], capture_output=True, text=True)  # noqa: E731
        check("cli: report passes with satisfied SLOs (exit 0)", run("--max-itl-p95-ms", "500", "--max-error-rate", "0.01").returncode == 0)
        check("cli: report fails with violated SLO (exit 1)", run("--max-itl-p95-ms", "1").returncode == 1)
        check("files: raw.json, requests.jsonl, report.json, report.md written", all((pathlib.Path(td) / n).exists() for n in ("raw.json", "requests.jsonl", "report.json", "report.md")))
    unreachable = subprocess.run([sys.executable, str(pathlib.Path(__file__).resolve()), "run", "--url", "http://127.0.0.1:1", "--rate", "1", "--duration", "1"], capture_output=True, text=True)
    check("cli: unreachable server exits 2", unreachable.returncode == 2, unreachable.stderr[-200:])

    print(f"\nselftest: {'FAILED ' + str(len(fails)) + ': ' + ', '.join(fails) if fails else 'all checks passed'}")
    return 1 if fails else 0


if __name__ == "__main__":
    main()
