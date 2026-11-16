#!/usr/bin/env python3
"""itlcreep lane copy of scripts/soak2/replay2.py (itself a copy of scripts/interleave/replay2.py with a per-conversation temperature).
Changes here: each row also carries the raw chunk arrival marks (wall clock, characters) and unweighted gap statistics, `--only-worker W`
runs one worker's conversations (two processes, one per vantage point, then share one server), and LOCALEVAL_SRC points at a copy of
deepeval/src for boxes without the uv project. Everything else (prompts, plan, timing) is unchanged.
--temp-mode mixed (default: conversations with (c // 2) % 2 == 1, i.e. 2, 3, 6, 7, run at
--temperature, the others greedy; with K=2 each worker c % 2 owns both kinds), greedy, or sampled. The soak2 lane does not touch engine code.

ab35c: K-concurrent-stream variant of scripts/ab35b/replay.py (same CLI, plan, trace, prompt and timing semantics).

  cd /opt/code/deepeval && uv run python scripts/ab35c/replay2.py --url http://spark-fa00:8000/v1 --model guidance \
      --arm A --order interleaved --seed 0 --streams 2 --plan PLAN.json --trace TRACE.json --out OUT_DIR

K worker threads; worker w owns conversations c with c % K == w and runs them in the requested order (interleaved: turn t of
each of its conversations, then t+1; sequential: all turns of one conversation, then the next). K requests are therefore in
flight at all times (while every worker still has work) and one conversation's turns never overlap. Prompts and the trace are
identical to replay.py, so a trace recorded at K=1 replays at K=2 and vice versa (history comes from the trace, or, when
recording, from the same conversation's earlier live replies, which the owning worker has always finished).

Busy detection runs once at start (not per request: at K>1 our own streams would count as "someone else"). Plan and trace
files are shared with replay.py; with K>1 and an existing plan the plan is not recomputed.
Output: <arm>-<order>-seed<seed>-k<K>.jsonl / .summary.json. localeval's Client does not expose raw chunk timestamps, only
per-request token-weighted inter-token latency (itl_ms_p50/p99, itl_cv); the smoothness summary aggregates those.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import statistics
import sys
import threading
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "ab35b"))  # scripts/itlcreep/ -> scripts/ab35b/
if os.environ.get("LOCALEVAL_SRC"):  # the box has no uv project: a copy of /opt/code/deepeval/src
    sys.path.insert(0, os.environ["LOCALEVAL_SRC"])
import replay as base  # noqa: E402
import localeval.client as lc  # noqa: E402
from localeval.client import Client, model_info, wait_until_idle  # noqa: E402
from localeval.config import Target  # noqa: E402
from localeval.speed import INSTRUCTION, filler  # noqa: E402


# Raw chunk arrival marks. localeval's Client keeps only the per-request ITL summary; itl_stats(marks, n) is called with the raw
# (arrival time, characters) marks of every content chunk, so wrapping it hands them to this thread without copying _stream.
_tl = threading.local()
_orig_itl_stats = lc.itl_stats


def _itl_stats(marks, n_tokens):
    _tl.marks = list(marks)
    return _orig_itl_stats(marks, n_tokens)


lc.itl_stats = _itl_stats
BURST_S = 0.003  # chunks closer than this belong to one engine step (an MTP burst is sent token by token)


def pctl(v, p):
    v = sorted(v)
    return v[min(len(v) - 1, int(p * len(v)))] if v else None


def raw_gap_stats(marks):
    """Unweighted gaps between consecutive content chunks (ms) and the same after merging bursts into steps."""
    if len(marks) < 3:
        return {}
    t = [m[0] for m in marks]
    gaps = [(b - a) * 1e3 for a, b in zip(t, t[1:])]
    steps = [t[0]]
    for x in t[1:]:
        if x - steps[-1] >= BURST_S:
            steps.append(x)
    sg = [(b - a) * 1e3 for a, b in zip(steps, steps[1:])]
    return {"chunks": len(marks), "gap_p50_ms": round(pctl(gaps, .5), 2), "gap_p99_ms": round(pctl(gaps, .99), 2), "gap_max_ms": round(max(gaps), 1),
            "gaps_gt100": sum(g > 100 for g in gaps), "gaps_gt150": sum(g > 150 for g in gaps), "gaps_gt250": sum(g > 250 for g in gaps),
            "steps": len(steps), "step_p50_ms": round(pctl(sg, .5), 2) if sg else None, "step_p99_ms": round(pctl(sg, .99), 2) if sg else None}


def worker_plan(w, k, convs, turns, order):
    mine = [c for c in range(convs) if c % k == w]
    if order == "interleaved":
        return [(c, t) for t in range(turns) for c in mine]
    return [(c, t) for c in mine for t in range(turns)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--order", choices=["interleaved", "sequential"], default="interleaved")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--streams", type=int, default=2)
    ap.add_argument("--convs", type=int, default=8)
    ap.add_argument("--turns", type=int, default=6)
    ap.add_argument("--first-tokens", type=int, default=8000)
    ap.add_argument("--add-tokens", type=int, default=4800)
    ap.add_argument("--max-tokens", type=int, default=160)
    ap.add_argument("--only-worker", type=int, default=None, help="run only this worker's conversations (c %% streams == w); the other process runs the rest")
    ap.add_argument("--temp-mode", choices=["mixed", "greedy", "sampled"], default="mixed")
    ap.add_argument("--temperature", type=float, default=1.0, help="temperature of the sampled conversations")
    ap.add_argument("--plan", required=True)
    ap.add_argument("--trace", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--wait-idle", type=float, default=600)
    ap.add_argument("--ignore-busy", action="store_true")
    a = ap.parse_args()
    K = a.streams

    target = Target(name=a.arm, url=a.url, model=a.model, thinking="off")
    info = model_info(target)
    if not info.get("reachable"):
        raise SystemExit(f"{a.url}: /models unreachable {info}")
    plan_p, trace_p, out = pathlib.Path(a.plan), pathlib.Path(a.trace), pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    # busy check ONCE, before any request of ours is in flight
    activity = wait_until_idle(target, a.wait_idle, a.ignore_busy, lambda *x: None)["state"]
    client0 = Client(target)
    if plan_p.exists():
        plan = json.loads(plan_p.read_text())
    else:
        plan = base.make_plan(client0, a.first_tokens, a.add_tokens, a.turns)
        plan_p.write_text(json.dumps(plan, indent=1))
    trace = json.loads(trace_p.read_text()) if trace_p.exists() else None
    recording = trace is None
    live = {}
    print(f"[replay2] arm={a.arm} order={a.order} seed={a.seed} streams={K} plan={plan} trace={'RECORD' if recording else 'replay'}", flush=True)

    w = client0.chat([{"role": "user", "content": filler(300, 7) + INSTRUCTION}], max_tokens=8, nonce=True)
    print(f"[replay2] warmup ok={w.ok} prompt={w.prompt_tokens}", flush=True)

    texts = {(c, t): base.user_msg(a.seed, c, t + 1, plan["sentences"][t]) for c in range(a.convs) for t in range(a.turns)}
    rows, lock = [], threading.Lock()
    f = (out / f"{a.arm}-{a.order}-seed{a.seed}-k{K}.jsonl").open("w")
    t_run = time.time()
    t_perf = time.perf_counter()

    epoch0 = time.time() - time.perf_counter()  # perf_counter -> wall clock (the box's clock; the server's trace uses the same clock)

    def run_worker(wi):
        client = Client(target)
        for c, t in worker_plan(wi, K, a.convs, a.turns, a.order):
            msgs = [{"role": "system", "content": base.SYSTEM}]
            for k in range(t):
                msgs.append({"role": "user", "content": texts[(c, k)]})
                msgs.append({"role": "assistant", "content": live[(c, k)] if recording else trace[f"{c}:{k}"]})
            msgs.append({"role": "user", "content": texts[(c, t)]})
            s = time.perf_counter() - t_perf
            temp = a.temperature if (a.temp_mode == "sampled" or (a.temp_mode == "mixed" and (c // 2) % 2 == 1)) else 0.0
            _tl.marks = []
            r = client.chat(msgs, max_tokens=a.max_tokens, temperature=temp, thinking="off")
            marks = list(getattr(_tl, "marks", []))
            e = time.perf_counter() - t_perf
            live[(c, t)] = r.text.strip() if r.ok and r.text.strip() else "(no answer)"
            row = {"arm": a.arm, "order": a.order, "seed": a.seed, "streams": K, "worker": wi, "conv": c, "turn": t + 1,
                   "temperature": temp, "ok": r.ok, "error": r.error, "prompt_tokens": r.prompt_tokens, "cached_tokens": r.cached_tokens or 0,
                   "ttft_s": r.ttft_s, "decode_tok_s": r.decode_tok_s, "peak_decode_tok_s": r.peak_decode_tok_s,
                   "completion_tokens": r.completion_tokens, "finish_reason": r.finish_reason, "total_s": r.total_s,
                   "itl_ms_p50": r.itl_ms_p50, "itl_ms_p99": r.itl_ms_p99, "itl_cv": r.itl_cv,
                   "start_s": round(s, 3), "end_s": round(e, 3), "reply": r.text, "t_start": r.t_start,
                   "epoch_start": round(epoch0 + t_perf + s, 3), "raw": raw_gap_stats(marks),
                   "marks": [[round(m[0] + epoch0, 4), m[1]] for m in marks]}
            with lock:
                rows.append(row)
                f.write(json.dumps(row) + "\n")
                f.flush()
                print(f"[replay2] w{wi} c{c} t{t + 1} prompt={r.prompt_tokens} cached={r.cached_tokens or 0} "
                      f"ttft={r.ttft_s if r.ttft_s is None else round(r.ttft_s, 2)}s "
                      f"decode={r.decode_tok_s if r.decode_tok_s is None else round(r.decode_tok_s, 1)} "
                      f"out={r.completion_tokens} {'' if r.ok else 'ERROR ' + str(r.error)}", flush=True)

    threads = [threading.Thread(target=run_worker, args=(wi,)) for wi in range(K) if a.only_worker in (None, wi)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    wall = time.time() - t_run
    f.close()
    if recording:
        trace_p.write_text(json.dumps({f"{c}:{t}": v for (c, t), v in live.items()}))
    rows.sort(key=lambda x: (x["start_s"]))
    summ = base.summarize(rows, a.turns)
    ok = [x for x in rows if x["ok"]]
    toks = sum(x["completion_tokens"] or 0 for x in ok)
    p50s = [x["itl_ms_p50"] for x in ok if x["itl_ms_p50"] is not None]
    p99s = [x["itl_ms_p99"] for x in ok if x["itl_ms_p99"] is not None]
    cvs = [x["itl_cv"] for x in ok if x["itl_cv"] is not None]
    smooth = ({"note": "per-request token-weighted ITL from localeval Client (raw chunk timestamps not exposed)",
               "itl_ms_p50_median": round(statistics.median(p50s), 2), "itl_ms_p99_median": round(statistics.median(p99s), 2),
               "itl_ms_p99_max": round(max(p99s), 2), "itl_cv_median": round(statistics.median(cvs), 3)}
              if p50s and p99s and cvs else None)
    res = {"arm": a.arm, "order": a.order, "seed": a.seed, "streams": K, "url": a.url, "model": a.model, "plan": plan,
           "trace": "recorded" if recording else "replayed", "wall_s": round(wall, 1), "turns": summ,
           "activity": {activity: 1}, "smoothness": smooth,
           "totals": {"requests": len(rows), "errors": len(rows) - len(ok),
                      "prompt_tokens": sum(x["prompt_tokens"] or 0 for x in ok), "cached_tokens": sum(x["cached_tokens"] for x in ok),
                      "ttft_s_sum": round(sum(x["ttft_s"] for x in ok if x["ttft_s"] is not None), 2),
                      "decode_tok_s_median": round(statistics.median([x["decode_tok_s"] for x in ok if x["decode_tok_s"]]), 1),
                      "completion_tokens": toks, "agg_output_tok_s": round(toks / wall, 2) if wall > 0 else None}}
    (out / f"{a.arm}-{a.order}-seed{a.seed}-k{K}.summary.json").write_text(json.dumps(res, indent=1))
    print("| turn | prompt tok | cached tok (frac) | TTFT s median (min-max) | decode tok/s median (min-max) |", flush=True)
    for s in summ:
        print(f"| {s['turn']} | {s['prompt_tokens_median']:.0f} | {s['cached_tokens_median']:.0f} ({s['cached_frac_median']}) | "
              f"{s['ttft_s_median']} ({s['ttft_s_range'][0]}-{s['ttft_s_range'][1]}) | {s['decode_tok_s_median']} ({s['decode_tok_s_range'][0]}-{s['decode_tok_s_range'][1]}) |", flush=True)
    print(f"[replay2] smoothness {smooth}", flush=True)
    print(f"[replay2] totals {res['totals']} wall {res['wall_s']}s", flush=True)
    return 0 if not res["totals"]["errors"] else 3


if __name__ == "__main__":
    sys.exit(main())
