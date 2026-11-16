#!/usr/bin/env python3
"""Lane kvq: knob off/on decode tok/s ABBA-BAAB on endor, fresh server per visit, cooldown between visits.
Run under `fleetq run --host endor --quiet gpu,cpu`. One visit = one server start for one arm, then one localeval
`speed` call per prompt size; the server's /metrics are scraped around each call (server-own decode ms per token and
per verify pass). Writes <out>/visits.jsonl (one line per visit and size) and re-runs flagged visits once.
usage: ab.py --model 35|27 --spec default|off --out DIR [--sizes 4096,32768] [--reps 2] [--cooldown 90] [--order 'ABBA BAAB']
Env KVQ_BIN, KVQ_HOME_MODELS, KVQ_DEEPEVAL, KVQ_PORT override the defaults (dry runs use stubs)."""
import argparse, json, os, re, signal, statistics, subprocess, sys, time, urllib.request

W = "/opt/code/intarsiaml/.worktrees/kvq"
BIN = os.environ.get("KVQ_BIN", W + "/target/kvq-intarsia")
MODELS = {"35": "Qwen3.6-35B-A3B-NVFP4", "27": "Qwen3.8-27B-Quark-AWQ-MXFP4"}
MROOT = os.environ.get("KVQ_HOME_MODELS", os.path.expanduser("~/models"))
DEEP = os.environ.get("KVQ_DEEPEVAL", "/opt/code/deepeval")
PORT = int(os.environ.get("KVQ_PORT", "18641"))
SERVE_TIMEOUT = int(os.environ.get("KVQ_SERVE_TIMEOUT", "300"))
UVRUN = os.environ.get("KVQ_UV", "uv")

def http(path, timeout=5):
    with urllib.request.urlopen(f"http://127.0.0.1:{PORT}{path}", timeout=timeout) as r:
        return r.read().decode()

def scrape():
    out = {}
    try:
        for line in http("/metrics").splitlines():
            if line.startswith("#") or " " not in line:
                continue
            k, v = line.rsplit(" ", 1)
            if k.startswith(("intarsia_itl_seconds_sum", "intarsia_itl_seconds_count", "intarsia_spec_steps_total",
                             "intarsia_spec_proposed_tokens_total", "intarsia_spec_accepted_tokens_total",
                             "intarsia_tokens_generated_total")):
                out[k] = float(v)
    except Exception as e:
        out["error"] = str(e)
    return out

def delta(a, b, k):
    return b.get(k, 0.0) - a.get(k, 0.0)

def start(outdir, arm, model, spec):
    os.makedirs(outdir, exist_ok=True)
    env = dict(os.environ)
    env.pop("INTARSIA_METAL_ATTN_KV_ONCE", None)
    if arm == "B":
        env["INTARSIA_METAL_ATTN_KV_ONCE"] = "1"
    cmd = [BIN, "serve", "--backend", "metal", "--model", f"{MROOT}/{MODELS[model]}", "--served-model-name", "intarsia",
           "--max-seq", "40000", "--profile-every", "0", "--no-supervisor", "--no-ui", "--host", "127.0.0.1", "--port", str(PORT)]
    if spec == "off":
        cmd += ["--spec", "off"]
    p = subprocess.Popen(cmd, env=env, stdout=open(outdir + "/server.out", "w"), stderr=open(outdir + "/server.err", "w"),
                         stdin=subprocess.DEVNULL)
    t0 = time.time()
    while time.time() - t0 < SERVE_TIMEOUT:
        if p.poll() is not None:
            raise RuntimeError("server exited early, see " + outdir + "/server.err")
        try:
            http("/health", 2)
            return p, time.time() - t0, cmd
        except Exception:
            time.sleep(1)
    p.kill()
    raise RuntimeError("server start timeout")

def stop(p):
    p.send_signal(signal.SIGINT)
    for _ in range(60):
        if p.poll() is not None:
            return
        time.sleep(1)
    p.kill()
    p.wait()

ROW = re.compile(r"^\|\s*(\d+)\s*\|(.*)\|\s*$")

def parse_localeval(text):
    """rows of the speed table: size -> list of cells (strings); header kept for the doc."""
    hdr, rows = None, {}
    for line in text.splitlines():
        if line.startswith("|") and hdr is None and not set(line) <= set("|-: "):
            hdr = [c.strip() for c in line.strip("|").split("|")]
        m = ROW.match(line)
        if m:
            rows[int(m.group(1))] = [c.strip() for c in m.group(2).split("|")]
    return hdr, rows

def visit(args, idx, arm, label, outroot):
    out = f"{outroot}/{idx:02d}-{arm}{label}"
    p, load_s, cmd = start(out, arm, args.model, args.spec)
    recs = []
    try:
        for size in [int(s) for s in args.sizes.split(",")]:
            m0 = scrape()
            t0 = time.time()
            r = subprocess.run([UVRUN, "run", "localeval", "speed", "--url", f"http://127.0.0.1:{PORT}/v1", "--model", "intarsia",
                                "--sizes", str(size), "--gen", "256", "--reps", str(args.reps), "--thinking", "off", "--no-unim",
                                "--label", f"kvq-{args.model}-{args.spec}-{arm}", "--recipe", " ".join(cmd) + f" knob={arm}"],
                               cwd=DEEP, capture_output=True, text=True)
            wall = time.time() - t0
            m1 = scrape()
            open(f"{out}/localeval-{size}.out", "w").write(r.stdout)
            open(f"{out}/localeval-{size}.err", "w").write(r.stderr)
            hdr, rows = parse_localeval(r.stdout)
            n = delta(m0, m1, "intarsia_itl_seconds_count")
            s = delta(m0, m1, "intarsia_itl_seconds_sum")
            passes = delta(m0, m1, "intarsia_spec_steps_total")
            rec = dict(idx=idx, arm=arm, rerun=bool(label), model=args.model, spec=args.spec, size=size, exit=r.returncode,
                       load_s=round(load_s, 1), wall_s=round(wall, 1), table_header=hdr, row=rows.get(size),
                       srv_tokens=n, srv_decode_s=s, srv_ms_per_token=(1e3 * s / n if n else None),
                       srv_spec_passes=passes, srv_ms_per_pass=(1e3 * s / passes if passes else None),
                       srv_proposed=delta(m0, m1, "intarsia_spec_proposed_tokens_total"),
                       srv_accepted=delta(m0, m1, "intarsia_spec_accepted_tokens_total"),
                       t=time.strftime("%H:%M:%S"))
            recs.append(rec)
            print(json.dumps(rec), flush=True)
            open(outroot + "/visits.jsonl", "a").write(json.dumps(rec) + "\n")
    finally:
        stop(p)
    return recs

def flagged(allrecs):
    """(arm, size)-median of server ms/token over non-rerun records; visits > 10 % above it."""
    bad = set()
    for size in {r["size"] for r in allrecs}:
        for arm in "AB":
            xs = [r for r in allrecs if r["arm"] == arm and r["size"] == size and r["srv_ms_per_token"] and not r["rerun"]]
            if len(xs) < 3:
                continue
            med = statistics.median(r["srv_ms_per_token"] for r in xs)
            for r in xs:
                if r["srv_ms_per_token"] > 1.10 * med:
                    bad.add((r["idx"], r["arm"]))
    return sorted(bad)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--spec", required=True, choices=["default", "off"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--sizes", default="4096,32768")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--cooldown", type=int, default=90)
    ap.add_argument("--order", default="ABBA BAAB")
    ap.add_argument("--max-rerun", type=int, default=3)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    seq = list(args.order.replace(" ", ""))
    allrecs = []
    for i, arm in enumerate(seq, 1):
        if i > 1:
            time.sleep(args.cooldown)
        allrecs += visit(args, i, arm, "", args.out)
    for idx, arm in flagged(allrecs)[: args.max_rerun]:
        time.sleep(args.cooldown)
        print(f"## rerun of flagged visit {idx} arm {arm}", flush=True)
        allrecs += visit(args, idx, arm, "-rerun", args.out)
    print("## done", flush=True)

if __name__ == "__main__":
    main()
