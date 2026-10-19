#!/usr/bin/env python3
"""P2' (MoeBlock) soak lane for a max-batch-1 `intarsia serve` (docs/results/2026-10-19-p2prime-soak.md).

HIP serve cannot batch (`--max-batch 1` only), so concurrent clients queue and the engine serves them one at a time.
This lane drives one already-running server (stdlib only; it shells out to tools/cbidentity.py and tools/cbsoak.py):

  1. wait for /health, start a background poller (/health + /metrics every 2 s: 503s, restarts, engine stalls);
  2. decode-speed sentinel (3 greedy 128-token requests; the `last_decode_tokens_per_second` gauge says which kernels ran);
  3. solo reference: cbidentity `run --mode solo --sample N` (flush before each case, token ids);
  4. soak: cbsoak open-loop (Poisson arrivals, mixed prompts, ~30 % client cancels) in the background for the whole soak
     while cbidentity `run --mode concurrent` replays of the same sample run back to back (B 4/8, streamed/not, staggered
     and burst arrivals, its own 20 cancel cases) until `--soak-s` has passed;
  5. drain: wait for running = waiting = 0, check the /metrics invariants;
  6. solo reference again (drift), cbidentity `compare` of every replay and of the second reference against the first.

Every request has a 60 s no-data timeout (`--timeout`); a request that errors or takes longer than 60 s is a hang candidate.
Writes lane_summary.json (+ every raw file) into --out. Exit 0 = pass, 1 = a criterion failed, 2 = harness problem.

    python3 tools/p2prime_soak.py lane --url http://127.0.0.1:8090 --arm blk --out DIR [--soak-s 600] [--sample 60]
    python3 tools/p2prime_soak.py crossref --a DIR1/ref.jsonl --b DIR2/ref.jsonl --out report.json
"""
import argparse
import json
import pathlib
import re
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

HERE = pathlib.Path(__file__).resolve().parent
CBID = HERE / "cbidentity.py"
CBSOAK = HERE / "cbsoak.py"
SCHEDULE = [(4, "mixed", 1.0), (8, "mixed", 0.5), (8, "on", 0.0), (4, "off", 1.0), (8, "mixed", 0.0), (4, "on", 0.25)]
HANG_S = 60.0


def now():
    return time.strftime("%H:%M:%S")


def log(msg):
    print(f"[{now()}] {msg}", flush=True)


def http_get(url, path, timeout=5.0):
    try:
        with urllib.request.urlopen(url + path, timeout=timeout) as r:
            return r.status, r.read().decode(errors="replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode(errors="replace")
    except Exception as e:  # noqa: BLE001
        return None, repr(e)


def parse_prom(text):
    out = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        m = re.match(r"^([^\s{]+)(\{[^}]*\})?\s+(\S+)$", line)
        if not m:
            continue
        try:
            out[m.group(1) + (m.group(2) or "")] = float(m.group(3))
        except ValueError:
            pass
    return out


def metrics(url):
    code, text = http_get(url, "/metrics")
    return parse_prom(text) if code == 200 else None


def finished_total(m):
    return sum(v for k, v in m.items() if k.startswith("intarsia_requests_finished_total"))


def finished_by_reason(m):
    return {re.search(r'reason="([^"]+)"', k).group(1): v for k, v in m.items()
            if k.startswith("intarsia_requests_finished_total") and 'reason="' in k}


# ---------------------------------------------------------------------------------------------------------------- poller
class Poller(threading.Thread):
    def __init__(self, url, path, interval=2.0, stall_s=30.0):
        super().__init__(daemon=True)
        self.url, self.path, self.interval, self.stall_s = url, path, interval, stall_s
        self.stop_ev = threading.Event()
        self.violations = []
        self.max_waiting = 0
        self.max_running = 0
        self.samples = 0
        self._last_uptime = None
        self._steps = None
        self._steps_t = None
        self._stalled = False
        self.decode_tps = []

    def violate(self, kind, detail):
        self.violations.append({"t": now(), "kind": kind, "detail": detail})
        log(f"POLL VIOLATION {kind}: {detail}")

    def run(self):
        with open(self.path, "w") as f:
            while not self.stop_ev.is_set():
                t = time.time()
                hc, ht = http_get(self.url, "/health", 4.0)
                h = {}
                if hc is None:
                    self.violate("health-unreachable", ht)
                else:
                    try:
                        h = json.loads(ht)
                    except Exception:  # noqa: BLE001
                        h = {}
                    if hc != 200:
                        self.violate("health-not-200", f"{hc} {ht[:200]}")
                m = metrics(self.url) or {}
                running = m.get("intarsia_requests_running", h.get("running", 0)) or 0
                waiting = m.get("intarsia_requests_waiting", h.get("waiting", 0)) or 0
                steps = m.get("intarsia_engine_steps_total", h.get("steps_total"))
                up = h.get("uptime_s")
                if up is not None:
                    if self._last_uptime is not None and up + 1.0 < self._last_uptime:
                        self.violate("restart", f"uptime_s {self._last_uptime} -> {up}")
                    self._last_uptime = up
                if m.get("intarsia_engine_failed", 0):
                    self.violate("engine-failed", "intarsia_engine_failed != 0")
                if steps is not None:
                    if self._steps is None or steps != self._steps or (running + waiting) == 0:
                        self._steps, self._steps_t, self._stalled = steps, time.monotonic(), False
                    elif time.monotonic() - self._steps_t > self.stall_s and not self._stalled:
                        self._stalled = True
                        self.violate("stall", f"engine_steps_total stuck at {steps} for {self.stall_s}s with running={running} waiting={waiting}")
                self.max_waiting = max(self.max_waiting, waiting)
                self.max_running = max(self.max_running, running)
                tps = m.get("intarsia_last_decode_tokens_per_second")
                if tps:
                    self.decode_tps.append(tps)
                f.write(json.dumps({"t": round(t, 2), "health": hc, "status": h.get("status"), "uptime_s": up, "steps": steps,
                                    "running": running, "waiting": waiting, "total": m.get("intarsia_requests_total"),
                                    "finished": finished_by_reason(m) if m else None, "rejected": m.get("intarsia_requests_rejected_total"),
                                    "failed": m.get("intarsia_engine_failed"), "no_progress_s": m.get("intarsia_engine_no_progress_seconds"),
                                    "decode_tps": tps}) + "\n")
                f.flush()
                self.samples += 1
                self.stop_ev.wait(self.interval)


# ---------------------------------------------------------------------------------------------------------------- steps
def wait_health(url, timeout_s):
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        c, _ = http_get(url, "/health", 3.0)
        if c == 200:
            return True
        time.sleep(2)
    return False


def speed_check(url, model):
    """3 greedy 128-token requests; the decode-speed gauge after each says whether MoeBlock (~165 tok/s) or the chain (~150) ran."""
    tps = []
    for _ in range(3):
        body = json.dumps({"model": model, "prompt": "Write a long story about a lighthouse keeper.", "max_tokens": 128,
                           "temperature": 0, "ignore_eos": True}).encode()
        try:
            req = urllib.request.Request(url + "/v1/completions", data=body, headers={"Content-Type": "application/json"})
            urllib.request.urlopen(req, timeout=60).read()
        except Exception as e:  # noqa: BLE001
            log(f"speed check request failed: {e!r}")
            continue
        m = metrics(url) or {}
        if m.get("intarsia_last_decode_tokens_per_second"):
            tps.append(m["intarsia_last_decode_tokens_per_second"])
    return tps


def cb(args, out_log, timeout):
    """Run a tools/cbidentity.py command, tee its output into out_log. Returns the exit code (None on harness timeout)."""
    with open(out_log, "w") as f:
        try:
            return subprocess.run([sys.executable, str(CBID)] + args, stdout=f, stderr=subprocess.STDOUT, timeout=timeout).returncode
        except subprocess.TimeoutExpired:
            f.write("\nHARNESS TIMEOUT\n")
            return None


def load_run(path):
    hdr, recs = None, []
    for line in pathlib.Path(path).read_text().splitlines():
        if not line.strip():
            continue
        o = json.loads(line)
        if "_header" in o:
            hdr = o["_header"]
        else:
            recs.append(o)
    return hdr, recs


def replay_stats(path):
    hdr, recs = load_run(path)
    st = {"file": pathlib.Path(path).name, "n": len(recs), "b": hdr.get("b"), "stream": hdr.get("stream"),
          "arrival_scale": hdr.get("arrival_scale"), "wall_s": hdr.get("wall_s"), "peak_client_inflight": hdr.get("peak_client_inflight"),
          "errors": [], "slow": []}
    tot = []
    for r in recs:
        if r["status"] == "error":
            st["errors"].append({"id": r["id"], "error": r.get("error"), "t_total_s": round(r["t_total_ms"] / 1000, 1)})
        if r["t_total_ms"] / 1000 > HANG_S and r["status"] != "cancelled":
            st["slow"].append({"id": r["id"], "t_total_s": round(r["t_total_ms"] / 1000, 1)})
        if r["status"] == "ok":
            tot.append(r["t_total_ms"] / 1000)
    st["cancelled"] = sum(1 for r in recs if r["status"] == "cancelled")
    st["cached_tokens_sum"] = sum(((r.get("usage") or {}).get("prompt_tokens_details") or {}).get("cached_tokens", 0) or 0 for r in recs)
    if tot:
        tot.sort()
        st["latency_s"] = {"p50": round(statistics.median(tot), 2), "p99": round(tot[min(len(tot) - 1, int(0.99 * len(tot)))], 2), "max": round(tot[-1], 2)}
    return st


def drain_and_check(url, wait_s=90):
    """Wait for idle, then the /metrics invariants. Returns (ok, details)."""
    t0 = time.time()
    m = None
    while time.time() - t0 < wait_s:
        m = metrics(url)
        if m is not None and m.get("intarsia_requests_running", 0) == 0 and m.get("intarsia_requests_waiting", 0) == 0:
            break
        time.sleep(2)
    if m is None:
        return False, {"error": "no /metrics"}
    d = {"running": m.get("intarsia_requests_running"), "waiting": m.get("intarsia_requests_waiting"),
         "requests_total": m.get("intarsia_requests_total"), "finished": finished_by_reason(m),
         "finished_sum": finished_total(m), "rejected": m.get("intarsia_requests_rejected_total"),
         "engine_failed": m.get("intarsia_engine_failed"), "drained_after_s": round(time.time() - t0, 1)}
    bad = []
    if d["running"] != 0 or d["waiting"] != 0:
        bad.append("queue did not drain to 0")
    if d["requests_total"] != d["finished_sum"]:
        bad.append(f"requests_total {d['requests_total']} != sum(finished) {d['finished_sum']}")
    if d["rejected"]:
        bad.append(f"rejected_total {d['rejected']}")
    if d["engine_failed"]:
        bad.append("engine_failed")
    d["problems"] = bad
    return not bad, d


def lane(a):
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    url = a.url
    summary = {"arm": a.arm, "url": url, "started": time.strftime("%F %T"), "soak_s": a.soak_s, "sample": a.sample, "criteria": {}}
    fails = []

    def save():
        (out / "lane_summary.json").write_text(json.dumps(summary, indent=1))

    if not wait_health(url, a.health_wait_s):
        summary["criteria"]["serve_up"] = False
        save()
        log("server never answered /health")
        return 2
    code, txt = http_get(url, "/v1/models")
    try:
        model = json.loads(txt)["data"][0]["id"]
    except Exception:  # noqa: BLE001
        model = "intarsia"
    m0 = metrics(url)
    (out / "metrics_start.json").write_text(json.dumps(m0, indent=1))
    c, plan = http_get(url, "/v1/intarsia/plan", 15)
    (out / "plan.json").write_text(plan)

    poller = Poller(url, str(out / "poll.jsonl"))
    poller.start()
    try:
        summary["decode_tps_sentinel"] = speed_check(url, model)
        log(f"decode speed sentinel (tok/s): {summary['decode_tps_sentinel']}")

        sample = ["--sample", str(a.sample)] if a.sample else []
        common = sample + ["--timeout", str(int(HANG_S))]
        log("solo reference")
        rc = cb(["run", "--mode", "solo", "--url", url, "--out", str(out / "ref.jsonl")] + common, out / "ref.log", 1500)
        summary["ref_rc"] = rc
        if rc != 0:
            fails.append(f"solo reference rc={rc}")
            summary["criteria"]["reference_ok"] = False
            save()
            return 2
        _, ref_recs = load_run(out / "ref.jsonl")
        ref_bad = [r["id"] for r in ref_recs if r["status"] != "ok"]
        summary["ref_cases"] = len(ref_recs)
        summary["ref_bad"] = ref_bad
        if ref_bad:
            fails.append(f"reference had failing cases {ref_bad}")

        # background open-loop load with client cancels
        soak_proc = None
        if not a.no_cbsoak:
            cmd = [sys.executable, str(CBSOAK), "run", "--url", url, "--rate", str(a.soak_rate), "--duration", str(a.soak_s),
                   "--cancel-frac", str(a.cancel_frac), "--max-seq", "8192", "--no-footprint", "--seed", str(a.seed),
                   "--timeout", str(int(HANG_S)), "--request-timeout", "180", "--drain-s", "120", "--out", str(out / "cbsoak")]
            log("starting cbsoak: " + " ".join(cmd[2:]))
            soak_proc = subprocess.Popen(cmd, stdout=open(out / "cbsoak.log", "w"), stderr=subprocess.STDOUT)
        t0 = time.time()
        replays = []
        i = 0
        while time.time() - t0 < a.soak_s:
            b, stream, scale = SCHEDULE[i % len(SCHEDULE)]
            name = f"c{i:02d}-b{b}-{stream}-s{scale}"
            log(f"replay {name} (t+{time.time() - t0:.0f}s)")
            rc = cb(["run", "--mode", "concurrent", "--url", url, "--b", str(b), "--stream", stream, "--arrival-scale", str(scale),
                     "--out", str(out / f"{name}.jsonl")] + common, out / f"{name}.log", 600)
            replays.append(name)
            if rc != 0:
                fails.append(f"replay {name} rc={rc}")
            i += 1
        summary["load_wall_s"] = round(time.time() - t0, 1)
        if soak_proc is not None:
            try:
                soak_proc.wait(timeout=a.soak_s + 400)
            except subprocess.TimeoutExpired:
                soak_proc.kill()
                fails.append("cbsoak did not finish")
            summary["cbsoak_rc"] = soak_proc.returncode
        summary["replays"] = [replay_stats(out / f"{n}.jsonl") for n in replays if (out / f"{n}.jsonl").exists()]
        rpt = out / "cbsoak" / "report.json"
        if rpt.exists():
            r = json.loads(rpt.read_text())
            summary["cbsoak"] = {k: r.get(k) for k in ("counts", "error_kinds", "http_status", "ttft_ms", "itl_ms")}
            summary["cbsoak"]["rates"] = {k: v for k, v in (r.get("rates") or {}).items() if not isinstance(v, dict)}

        ok, inv = drain_and_check(url)
        summary["idle_invariants"] = inv
        summary["criteria"]["metrics_invariants_and_drain"] = ok
        if not ok:
            fails.append("metrics: " + "; ".join(inv.get("problems", [])))
        (out / "metrics_end.json").write_text(json.dumps(metrics(url), indent=1))

        log("solo reference again (drift)")
        rc2 = cb(["run", "--mode", "solo", "--url", url, "--out", str(out / "ref2.jsonl")] + common, out / "ref2.log", 1500)
        summary["ref2_rc"] = rc2

        # compare: replays vs reference, second reference vs first
        files = [str(out / f"{n}.jsonl") for n in replays if (out / f"{n}.jsonl").exists()]
        if files:
            summary["compare_rc"] = cb(["compare", "--ref", str(out / "ref.jsonl"), "--test"] + files + ["--report", str(out / "compare.json")],
                                       out / "compare.txt", 300)
        if (out / "ref2.jsonl").exists():
            summary["compare_ref2_rc"] = cb(["compare", "--ref", str(out / "ref.jsonl"), "--test", str(out / "ref2.jsonl"),
                                             "--report", str(out / "compare_ref2.json")], out / "compare_ref2.txt", 300)
    finally:
        poller.stop_ev.set()
        poller.join(timeout=5)
        summary["poll"] = {"samples": poller.samples, "max_waiting": poller.max_waiting, "max_running": poller.max_running,
                           "violations": poller.violations, "decode_tps_median": statistics.median(poller.decode_tps) if poller.decode_tps else None}
        summary["finished"] = time.strftime("%F %T")

    crit = summary["criteria"]
    soak_errors = ((summary.get("cbsoak") or {}).get("counts") or {}).get("error", 0) or 0
    crit["no_hang"] = soak_errors == 0 and not any(r["slow"] or r["errors"] for r in summary.get("replays", [])) and not any(
        v["kind"] in ("stall", "health-not-200", "health-unreachable", "restart", "engine-failed") for v in poller.violations)
    crit["identity_vs_solo"] = summary.get("compare_rc") == 0
    crit["no_drift_solo_vs_solo"] = summary.get("compare_ref2_rc") == 0
    if not crit["no_hang"]:
        fails.append("hang/error/stall (see replays[].errors/slow and poll.violations)")
    if not crit["identity_vs_solo"]:
        fails.append(f"identity: compare rc={summary.get('compare_rc')}")
    if not crit["no_drift_solo_vs_solo"]:
        fails.append(f"drift: compare ref2 rc={summary.get('compare_ref2_rc')}")
    summary["fails"] = fails
    summary["pass"] = not fails and all(v is not False for v in crit.values())
    save()
    log(f"lane {a.arm}: {'PASS' if summary['pass'] else 'FAIL'} {fails}")
    return 0 if summary["pass"] else 1


def crossref(a):
    rc = cb(["compare", "--ref", a.a, "--test", a.b, "--report", a.out], pathlib.Path(a.out).with_suffix(".txt"), 300)
    log(f"cross-arm compare rc={rc}")
    return 0 if rc == 0 else 1


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    l = sub.add_parser("lane")
    l.add_argument("--url", required=True)
    l.add_argument("--arm", required=True)
    l.add_argument("--out", required=True)
    l.add_argument("--soak-s", type=float, default=600.0)
    l.add_argument("--sample", type=int, default=40)
    l.add_argument("--soak-rate", type=float, default=0.3, help="cbsoak open-loop requests/s")
    l.add_argument("--cancel-frac", type=float, default=0.3)
    l.add_argument("--seed", type=int, default=7)
    l.add_argument("--no-cbsoak", action="store_true")
    l.add_argument("--health-wait-s", type=float, default=300.0)
    c = sub.add_parser("crossref")
    c.add_argument("--a", required=True)
    c.add_argument("--b", required=True)
    c.add_argument("--out", required=True)
    a = p.parse_args()
    sys.exit(lane(a) if a.cmd == "lane" else crossref(a))


if __name__ == "__main__":
    main()
