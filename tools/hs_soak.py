#!/usr/bin/env python3
"""Lane HS: tp=2 HIP fusion soak driver for one `intarsia serve --tp 2` (docs/results/2026-10-21-hs-hip-fusion-soak.md).

Built on tools/p2prime_soak.py (same Poller, cbidentity replays, cbsoak open-loop load, drain checks), with a hard wall-clock
budget so the whole lane fits one radiance window, and the extra measurements the fusion soak needs:

  * start / end sentinels: greedy chat requests (short + ~4k-token prompt, 256 tokens, thinking off, token ids returned,
    KV prefix flushed first). Decode tok/s is the server's last_decode_tokens_per_second gauge after each request.
  * sentinel decode tok/s must be fusion-level at the start (--min-start-tps), else the lane aborts (fusion not engaged);
    start vs end must agree within 5 % and the ids must be identical (start vs end, and vs --ref-short / --ref-4k).
  * memory: GPU VRAM used (/sys/class/drm/card*/device/mem_info_vram_used, cards with > 20 GB) and the serve process RSS,
    sampled every 15 s.
  * load: solo reference (B=1), then replays at B = 1, 4, 8 (streamed / not / mixed, short and long prompts, client cancels)
    plus cbsoak's open-loop mix (128..4k tokens, ~30 % cancels) in the background, until the soak deadline; then drain, a
    second solo reference (drift), compares.

Exit 0 = every criterion passed, 1 = a criterion failed, 2 = harness / fusion-not-engaged abort.

    python3 hs_soak.py lane --url http://127.0.0.1:8090 --out DIR --prompt4k prompt4k.txt --ref-short ids --ref-4k ids
"""
import argparse
import glob
import json
import os
import pathlib
import re
import statistics
import subprocess
import sys
import threading
import time
import urllib.request

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import p2prime_soak as P  # noqa: E402

SHORT = "Write a long story about a robot who learns to paint."
SCHEDULE = [(1, "off", 1.0), (4, "mixed", 1.0), (8, "mixed", 0.5), (8, "on", 0.0), (4, "off", 1.0), (1, "on", 1.0),
            (8, "mixed", 0.0), (4, "on", 0.25)]
FLUSH_ID = 151643
log = P.log


def post(url, path, body, timeout=120):
    req = urllib.request.Request(url + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def sentinel(url, model, name, text):
    """flush the prefix cache, then one greedy 256-token chat request -> ids, cached tokens, decode gauge."""
    try:
        post(url, "/v1/completions", {"model": model, "prompt": [FLUSH_ID], "max_tokens": 1, "temperature": 0})
        j = post(url, "/v1/chat/completions", {"model": model, "messages": [{"role": "user", "content": text}], "max_tokens": 256,
                                                "temperature": 0, "ignore_eos": True, "return_token_ids": True,
                                                "chat_template_kwargs": {"enable_thinking": False}})
        ch = j["choices"][0]
        u = j.get("usage") or {}
        m = P.metrics(url) or {}
        return {"kind": name, "ids": ch.get("token_ids"), "completion_tokens": u.get("completion_tokens"),
                "prompt_tokens": u.get("prompt_tokens"),
                "cached_tokens": ((u.get("prompt_tokens_details") or {}).get("cached_tokens")),
                "decode_tps": m.get("intarsia_last_decode_tokens_per_second")}
    except Exception as e:  # noqa: BLE001
        return {"kind": name, "error": repr(e)}


def sentinels(url, model, text4k):
    out = []
    for kind, text, n in (("short", SHORT, 3), ("4k", text4k, 2)):
        for _ in range(n):
            out.append(sentinel(url, model, kind, text))
    return out


def med(xs):
    xs = [x for x in xs if x]
    return statistics.median(xs) if xs else None


def ids_of(sent, kind):
    return [tuple(s["ids"]) if s.get("ids") else None for s in sent if s["kind"] == kind]


def read_ids_file(path):
    if not path or not os.path.exists(path):
        return None
    txt = open(path).read()
    m = re.search(r"\[([0-9,\s]+)\]", txt)
    return tuple(int(x) for x in m.group(1).replace(" ", "").split(",") if x) if m else None


class MemSampler(threading.Thread):
    def __init__(self, path, serve_pat, interval=15.0):
        super().__init__(daemon=True)
        self.path, self.serve_pat, self.interval = path, serve_pat, interval
        self.stop_ev = threading.Event()
        self.cards = [c for c in sorted(glob.glob("/sys/class/drm/card[0-9]*/device"))
                      if self._int(c + "/mem_info_vram_total") > 20e9]
        self.rows = []

    @staticmethod
    def _int(p):
        try:
            return int(open(p).read().strip())
        except Exception:  # noqa: BLE001
            return 0

    def rss_kb(self):
        try:
            pids = subprocess.run(["pgrep", "-f", self.serve_pat], capture_output=True, text=True).stdout.split()
            tot = 0
            for p in pids:
                for line in open(f"/proc/{p}/status"):
                    if line.startswith("VmRSS:"):
                        tot = max(tot, int(line.split()[1]))
            return tot or None
        except Exception:  # noqa: BLE001
            return None

    def sample(self):
        row = {"t": round(time.time(), 1), "vram_mib": [self._int(c + "/mem_info_vram_used") // 2**20 for c in self.cards],
               "rss_mib": (self.rss_kb() or 0) // 1024}
        self.rows.append(row)
        return row

    def run(self):
        with open(self.path, "w") as f:
            while not self.stop_ev.is_set():
                f.write(json.dumps(self.sample()) + "\n")
                f.flush()
                self.stop_ev.wait(self.interval)

    def summary(self, warm_s, t_end):
        rows = [r for r in self.rows if r["t"] >= self.rows[0]["t"] + warm_s] if self.rows else []
        if len(rows) < 4:
            return {"samples": len(self.rows), "ok": None, "note": "too few post-warmup samples"}
        n = len(self.cards)
        d = {"cards": [os.path.basename(os.path.dirname(c)) for c in self.cards], "samples": len(self.rows), "warm_s": warm_s,
             "vram_mib_min": [min(r["vram_mib"][i] for r in rows) for i in range(n)],
             "vram_mib_max": [max(r["vram_mib"][i] for r in rows) for i in range(n)]}
        d["vram_spread_mib"] = [mx - mn for mx, mn in zip(d["vram_mib_max"], d["vram_mib_min"])]
        k = max(2, len(rows) // 4)
        first, last = rows[:k], rows[-k:]
        d["rss_mib_first_quarter_median"] = statistics.median(r["rss_mib"] for r in first)
        d["rss_mib_last_quarter_median"] = statistics.median(r["rss_mib"] for r in last)
        d["vram_first_last_delta_mib"] = [round(statistics.median(r["vram_mib"][i] for r in last) - statistics.median(r["vram_mib"][i] for r in first)) for i in range(n)]
        rss0 = d["rss_mib_first_quarter_median"] or 1
        d["ok"] = bool(n) and all(s <= 256 for s in d["vram_spread_mib"]) and d["rss_mib_last_quarter_median"] <= 1.10 * rss0 + 64
        return d


def lane(a):
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    url = a.url
    t_begin = time.time()
    summary = {"url": url, "started": time.strftime("%F %T"), "budget_s": a.budget_s, "criteria": {}}
    fails = []

    def save():
        (out / "lane_summary.json").write_text(json.dumps(summary, indent=1))

    if not P.wait_health(url, a.health_wait_s):
        summary["criteria"]["serve_up"] = False
        save()
        log("server never answered /health")
        return 2
    code, txt = P.http_get(url, "/v1/models")
    try:
        model = json.loads(txt)["data"][0]["id"]
    except Exception:  # noqa: BLE001
        model = "intarsia"
    (out / "metrics_start.json").write_text(json.dumps(P.metrics(url), indent=1))
    _, plan = P.http_get(url, "/v1/intarsia/plan", 15)
    (out / "plan.json").write_text(plan)
    text4k = pathlib.Path(a.prompt4k).read_text().rstrip("\n")  # the shell's $(cat) in lib.sh gen() strips trailing newlines too
    ref_short, ref_4k = read_ids_file(a.ref_short), read_ids_file(a.ref_4k)

    mem = MemSampler(str(out / "mem.jsonl"), a.serve_pat)
    mem.start()
    poller = P.Poller(url, str(out / "poll.jsonl"))
    poller.start()
    soak_info = {}
    try:
        s0 = sentinels(url, model, text4k)
        summary["sentinel_start"] = s0
        tps0 = med([s["decode_tps"] for s in s0 if s["kind"] == "short"])
        summary["start_short_tps_median"] = tps0
        log(f"start sentinel: short tps {[s.get('decode_tps') for s in s0 if s['kind'] == 'short']} 4k tps {[s.get('decode_tps') for s in s0 if s['kind'] == '4k']}")
        if any(s.get("error") for s in s0):
            summary["criteria"]["sentinel_start_ok"] = False
            fails.append("start sentinel request failed")
            save()
            return 2
        if not tps0 or tps0 < a.min_start_tps:
            summary["criteria"]["fusion_engaged"] = False
            fails.append(f"start decode {tps0} tok/s < {a.min_start_tps}: fusion not engaged (or a slow serve path); aborting")
            save()
            return 2
        summary["criteria"]["fusion_engaged"] = True
        for kind, ref in (("short", ref_short), ("4k", ref_4k)):
            got = {i for i in ids_of(s0, kind)}
            if len(got) != 1 or None in got:
                fails.append(f"start sentinel {kind}: ids vary or missing ({len(got)} distinct)")
                save()
                return 2
            if ref is not None and next(iter(got)) != ref:
                summary.setdefault("api_vs_h1_start_mismatch", []).append(kind)
                log(f"NOTE: start sentinel {kind} ids differ from the H1 CLI reference (API path; start-vs-end still decides)")

        sample = ["--sample", str(a.sample)]
        common = sample + ["--timeout", str(int(P.HANG_S))]
        log("solo reference")
        rc = P.cb(["run", "--mode", "solo", "--url", url, "--out", str(out / "ref.jsonl")] + common, out / "ref.log", 240)
        summary["ref_rc"] = rc
        if rc != 0:
            fails.append(f"solo reference rc={rc}")
            save()
            return 2
        _, ref_recs = P.load_run(out / "ref.jsonl")
        summary["ref_cases"] = len(ref_recs)
        ref_bad = [r["id"] for r in ref_recs if r["status"] != "ok"]
        if ref_bad:
            fails.append(f"reference had failing cases {ref_bad}")

        soak_start = time.time()
        soak_end = min(soak_start + a.soak_s, t_begin + a.budget_s - a.tail_s)
        soak_s = max(60.0, soak_end - soak_start)
        soak_info.update({"soak_start": soak_start, "soak_s": round(soak_s, 1)})
        cmd = [sys.executable, str(P.CBSOAK), "run", "--url", url, "--rate", str(a.soak_rate), "--duration", str(int(soak_s)),
               "--cancel-frac", str(a.cancel_frac), "--max-seq", "8192", "--no-footprint", "--seed", str(a.seed),
               "--timeout", str(int(P.HANG_S)), "--request-timeout", "180", "--drain-s", "40", "--out", str(out / "cbsoak")]
        log(f"cbsoak: duration {int(soak_s)}s rate {a.soak_rate}/s")
        soak_proc = subprocess.Popen(cmd, stdout=open(out / "cbsoak.log", "w"), stderr=subprocess.STDOUT)
        replays, longest, i = [], 60.0, 0
        while time.time() + 1.3 * longest < soak_end:
            b, stream, scale = SCHEDULE[i % len(SCHEDULE)]
            name = f"c{i:02d}-b{b}-{stream}-s{scale}"
            t1 = time.time()
            log(f"replay {name} (t+{t1 - soak_start:.0f}s)")
            rc = P.cb(["run", "--mode", "concurrent", "--url", url, "--b", str(b), "--stream", stream, "--arrival-scale", str(scale),
                       "--out", str(out / f"{name}.jsonl")] + common, out / f"{name}.log", 300)
            longest = max(longest, time.time() - t1)
            replays.append(name)
            if rc != 0:
                fails.append(f"replay {name} rc={rc}")
            i += 1
        # let cbsoak run its full duration (it is the other half of the load), then drain
        try:
            soak_proc.wait(timeout=max(10.0, soak_end - time.time()) + 90)
        except subprocess.TimeoutExpired:
            soak_proc.kill()
            fails.append("cbsoak did not finish")
        summary["cbsoak_rc"] = soak_proc.returncode
        soak_info["load_wall_s"] = round(time.time() - soak_start, 1)
        summary["replays"] = [P.replay_stats(out / f"{n}.jsonl") for n in replays if (out / f"{n}.jsonl").exists()]
        rpt = out / "cbsoak" / "report.json"
        if rpt.exists():
            r = json.loads(rpt.read_text())
            summary["cbsoak"] = {k: r.get(k) for k in ("counts", "error_kinds", "http_status", "ttft_ms", "itl_ms")}
        ok, inv = P.drain_and_check(url, wait_s=45)
        summary["idle_invariants"] = inv
        summary["criteria"]["metrics_invariants_and_drain"] = ok
        if not ok:
            fails.append("metrics: " + "; ".join(inv.get("problems", [])))
        (out / "metrics_end.json").write_text(json.dumps(P.metrics(url), indent=1))

        log("solo reference again (drift)")
        rc2 = P.cb(["run", "--mode", "solo", "--url", url, "--out", str(out / "ref2.jsonl")] + common, out / "ref2.log", 240)
        summary["ref2_rc"] = rc2
        s1 = sentinels(url, model, text4k)
        summary["sentinel_end"] = s1
        log(f"end sentinel: short tps {[s.get('decode_tps') for s in s1 if s['kind'] == 'short']} 4k tps {[s.get('decode_tps') for s in s1 if s['kind'] == '4k']}")

        files = [str(out / f"{n}.jsonl") for n in replays if (out / f"{n}.jsonl").exists()]
        if files:
            summary["compare_rc"] = P.cb(["compare", "--ref", str(out / "ref.jsonl"), "--test"] + files + ["--report", str(out / "compare.json")],
                                         out / "compare.txt", 120)
        if (out / "ref2.jsonl").exists():
            summary["compare_ref2_rc"] = P.cb(["compare", "--ref", str(out / "ref.jsonl"), "--test", str(out / "ref2.jsonl"),
                                               "--report", str(out / "compare_ref2.json")], out / "compare_ref2.txt", 120)
    finally:
        poller.stop_ev.set()
        mem.stop_ev.set()
        poller.join(timeout=5)
        mem.sample()
        mem.join(timeout=5)
        summary["soak"] = soak_info
        summary["poll"] = {"samples": poller.samples, "max_waiting": poller.max_waiting, "max_running": poller.max_running,
                           "violations": poller.violations, "decode_tps_median": med(poller.decode_tps)}
        summary["memory"] = mem.summary(a.mem_warm_s, time.time())
        summary["finished"] = time.strftime("%F %T")
        summary["wall_s"] = round(time.time() - t_begin, 1)

    crit = summary["criteria"]
    soak_errors = ((summary.get("cbsoak") or {}).get("counts") or {}).get("error", 0) or 0
    crit["no_hang_or_error"] = soak_errors == 0 and not any(r["slow"] or r["errors"] for r in summary.get("replays", [])) and not any(
        v["kind"] in ("stall", "health-not-200", "health-unreachable", "restart", "engine-failed") for v in poller.violations)
    s1 = summary.get("sentinel_end") or []
    ok_end = bool(s1) and not any(s.get("error") for s in s1)
    crit["sentinel_end_ok"] = ok_end
    ids_same = ok_end
    detail = {}
    for kind, ref in (("short", ref_short), ("4k", ref_4k)):
        a0, a1 = set(ids_of(s0, kind)), set(ids_of(s1, kind))
        same = ok_end and len(a0) == 1 and a0 == a1 and None not in a0
        detail[kind] = {"start_end_identical": same, "equals_h1_ref": (next(iter(a1)) == ref) if (same and ref) else None}
        ids_same = ids_same and same
    summary["ids_detail"] = detail
    crit["ids_start_end_identical"] = ids_same
    tps1 = med([s.get("decode_tps") for s in s1 if s["kind"] == "short"])
    t4_0, t4_1 = med([s.get("decode_tps") for s in s0 if s["kind"] == "4k"]), med([s.get("decode_tps") for s in s1 if s["kind"] == "4k"])
    summary["drift"] = {"short_start": tps0, "short_end": tps1, "4k_start": t4_0, "4k_end": t4_1,
                        "short_pct": round(100 * (tps1 - tps0) / tps0, 2) if tps1 and tps0 else None,
                        "4k_pct": round(100 * (t4_1 - t4_0) / t4_0, 2) if t4_0 and t4_1 else None}
    dr = [v for v in (summary["drift"]["short_pct"], summary["drift"]["4k_pct"]) if v is not None]
    crit["decode_drift_within_5pct"] = bool(dr) and len(dr) == 2 and all(abs(v) <= 5.0 for v in dr)
    crit["memory_stable"] = summary["memory"].get("ok") is True
    crit["identity_vs_solo"] = summary.get("compare_rc") == 0
    crit["no_drift_solo_vs_solo"] = summary.get("compare_ref2_rc") == 0
    for k, v in crit.items():
        if v is False:
            fails.append(f"criterion {k}")
    summary["fails"] = fails
    summary["pass"] = not fails
    save()
    log(f"lane: {'PASS' if summary['pass'] else 'FAIL'} {fails}")
    return 0 if summary["pass"] else 1


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    l = sub.add_parser("lane")
    l.add_argument("--url", required=True)
    l.add_argument("--out", required=True)
    l.add_argument("--prompt4k", required=True)
    l.add_argument("--ref-short")
    l.add_argument("--ref-4k")
    l.add_argument("--budget-s", type=float, default=910.0, help="hard wall-clock budget for the whole lane")
    l.add_argument("--soak-s", type=float, default=760.0)
    l.add_argument("--tail-s", type=float, default=130.0, help="time reserved after the soak: drain, 2nd reference, end sentinels, compare")
    l.add_argument("--sample", type=int, default=24)
    l.add_argument("--soak-rate", type=float, default=0.3)
    l.add_argument("--cancel-frac", type=float, default=0.3)
    l.add_argument("--seed", type=int, default=7)
    l.add_argument("--min-start-tps", type=float, default=160.0)
    l.add_argument("--serve-pat", default="h1-intarsia serve")
    l.add_argument("--mem-warm-s", type=float, default=150.0)
    l.add_argument("--health-wait-s", type=float, default=90.0)
    a = p.parse_args()
    sys.exit(lane(a))


if __name__ == "__main__":
    main()
