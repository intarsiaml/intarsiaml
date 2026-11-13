#!/usr/bin/env python3
"""ab35b run 1: multi-turn replay against one OpenAI-compatible endpoint (localeval has no multi-turn mode).

Uses localeval's own streaming Client (same timing definitions as `localeval speed`: TTFT = request sent to first
content chunk, decode tok/s = (completion_tokens - 1) / decode window, cached_tokens from usage.prompt_tokens_details),
so the numbers compare with the localeval runs of ab35. Run it with the localeval environment:

  uv run --project /opt/code/deepeval python scripts/ab35b/replay.py --url http://spark-fa00:8000/v1 --model guidance \
      --arm A --order interleaved --seed 0 --out OUT_DIR --plan PLAN.json --trace TRACE.json

Workload: N conversations, T turns. Turn 1 sends a system prompt plus a ~8k-token notes section and a question; turn k
appends the previous answer and a new ~4.8k-token section plus a question, so a conversation's prompt grows from about
8k to about 32k tokens at turn 6. NO nonce: the prompts are deterministic (seeded filler, `localeval.speed.filler`),
so every arm sees byte-identical requests. The assistant answers placed in the history come from a trace file: the
first run with a given seed records the live replies, every later run with that seed replays them (the model still
generates and is timed on every turn, but its own reply is not fed back), so the prompts cannot diverge between arms.
Use a different --seed for every order and never reuse a seed on one server (it would hit its own cache from the run before).

Orders: `interleaved` = turn 1 of every conversation, then turn 2 of every conversation, ... (other conversations
intervene between a conversation's turns); `sequential` = all turns of conversation 0, then conversation 1, ...
Temperature 0, thinking off, max_tokens 160, one request at a time.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import sys
import time

from localeval.client import Client, model_info, wait_until_idle
from localeval.config import Target
from localeval.speed import INSTRUCTION, filler

SYSTEM = ("You are a careful assistant. The user pastes numbered field notes in sections and asks short questions "
          "about the newest section. Answer in one paragraph of about 100 words.")


def calibrate(client):
    base = client.chat([{"role": "user", "content": INSTRUCTION.strip()}], max_tokens=1, nonce=True)
    cal = client.chat([{"role": "user", "content": filler(200, 0) + INSTRUCTION}], max_tokens=1, nonce=True)
    if not (base.ok and cal.ok and base.prompt_tokens and cal.prompt_tokens):
        raise SystemExit(f"calibration failed: {base.error or cal.error}")
    return (cal.prompt_tokens - base.prompt_tokens) / 200, base.prompt_tokens


def make_plan(client, first_tokens, add_tokens, turns):
    per, base = calibrate(client)
    n_first = round((first_tokens - base - 80) / per)
    n_add = round(add_tokens / per)
    return {"per_sentence": round(per, 3), "base_tokens": base, "sentences": [n_first] + [n_add] * (turns - 1)}


def user_msg(seed, conv, turn, n_sent):
    body = filler(n_sent, seed=seed * 100000 + conv * 100 + turn)
    return (f"Notes section {turn}:\n{body}\n\nQuestion {turn}: in about 100 words, what do the notes of section {turn} say, "
            f"starting from its first note and ending at its last note?")


def pct(vals, p):
    v = sorted(vals)
    return v[min(len(v) - 1, int(p * len(v)))]


def summarize(rows, turns):
    out = []
    for t in range(1, turns + 1):
        r = [x for x in rows if x["turn"] == t and x["ok"]]
        if not r:
            continue
        med = lambda k: statistics.median([x[k] for x in r if x[k] is not None])  # noqa: E731
        rng = lambda k: (min(x[k] for x in r if x[k] is not None), max(x[k] for x in r if x[k] is not None))  # noqa: E731
        out.append({"turn": t, "n": len(r), "prompt_tokens_median": med("prompt_tokens"),
                    "cached_tokens_median": med("cached_tokens"),
                    "cached_frac_median": round(statistics.median([x["cached_tokens"] / x["prompt_tokens"] for x in r]), 3),
                    "ttft_s_median": round(med("ttft_s"), 3), "ttft_s_range": [round(v, 3) for v in rng("ttft_s")],
                    "decode_tok_s_median": round(med("decode_tok_s"), 1),
                    "decode_tok_s_range": [round(v, 1) for v in rng("decode_tok_s")],
                    "completion_tokens_median": med("completion_tokens"), "errors": sum(1 for x in rows if x["turn"] == t and not x["ok"])})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--order", choices=["interleaved", "sequential"], default="interleaved")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--convs", type=int, default=8)
    ap.add_argument("--turns", type=int, default=6)
    ap.add_argument("--first-tokens", type=int, default=8000)
    ap.add_argument("--add-tokens", type=int, default=4800)
    ap.add_argument("--max-tokens", type=int, default=160)
    ap.add_argument("--plan", required=True)
    ap.add_argument("--trace", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--wait-idle", type=float, default=600)
    ap.add_argument("--ignore-busy", action="store_true")
    a = ap.parse_args()

    target = Target(name=a.arm, url=a.url, model=a.model, thinking="off")
    info = model_info(target)
    if not info.get("reachable"):
        raise SystemExit(f"{a.url}: /models unreachable {info}")
    client = Client(target)
    plan_p, trace_p, out = pathlib.Path(a.plan), pathlib.Path(a.trace), pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if plan_p.exists():
        plan = json.loads(plan_p.read_text())
    else:
        plan = make_plan(client, a.first_tokens, a.add_tokens, a.turns)
        plan_p.write_text(json.dumps(plan, indent=1))
    trace = json.loads(trace_p.read_text()) if trace_p.exists() else None
    recording = trace is None
    live = {}
    print(f"[replay] arm={a.arm} order={a.order} seed={a.seed} plan={plan} trace={'RECORD' if recording else 'replay'}", flush=True)

    # warmup (nonce, discarded): the first call on a fresh server pays one-off setup
    w = client.chat([{"role": "user", "content": filler(300, 7) + INSTRUCTION}], max_tokens=8, nonce=True)
    print(f"[replay] warmup ok={w.ok} prompt={w.prompt_tokens}", flush=True)

    texts = {(c, t): user_msg(a.seed, c, t + 1, plan["sentences"][t]) for c in range(a.convs) for t in range(a.turns)}
    if a.order == "interleaved":
        order = [(c, t) for t in range(a.turns) for c in range(a.convs)]
    else:
        order = [(c, t) for c in range(a.convs) for t in range(a.turns)]

    rows, activity = [], []
    f = (out / f"{a.arm}-{a.order}-seed{a.seed}.jsonl").open("w")
    t_run = time.time()
    for c, t in order:
        msgs = [{"role": "system", "content": SYSTEM}]
        for k in range(t):
            msgs.append({"role": "user", "content": texts[(c, k)]})
            hist = live[(c, k)] if recording else trace[f"{c}:{k}"]
            msgs.append({"role": "assistant", "content": hist})
        msgs.append({"role": "user", "content": texts[(c, t)]})
        activity.append(wait_until_idle(target, a.wait_idle, a.ignore_busy, lambda *x: None)["state"])
        r = client.chat(msgs, max_tokens=a.max_tokens, temperature=0.0, thinking="off")
        live[(c, t)] = r.text.strip() if r.ok and r.text.strip() else "(no answer)"
        row = {"arm": a.arm, "order": a.order, "seed": a.seed, "conv": c, "turn": t + 1, "ok": r.ok, "error": r.error,
               "prompt_tokens": r.prompt_tokens, "cached_tokens": r.cached_tokens or 0, "ttft_s": r.ttft_s,
               "decode_tok_s": r.decode_tok_s, "peak_decode_tok_s": r.peak_decode_tok_s, "completion_tokens": r.completion_tokens,
               "finish_reason": r.finish_reason, "total_s": r.total_s, "reply": r.text, "t_start": r.t_start}
        rows.append(row)
        f.write(json.dumps(row) + "\n")
        f.flush()
        print(f"[replay] c{c} t{t + 1} prompt={r.prompt_tokens} cached={r.cached_tokens or 0} "
              f"ttft={r.ttft_s if r.ttft_s is None else round(r.ttft_s, 2)}s decode={r.decode_tok_s if r.decode_tok_s is None else round(r.decode_tok_s, 1)} "
              f"out={r.completion_tokens} {'' if r.ok else 'ERROR ' + str(r.error)}", flush=True)
    f.close()
    if recording:
        trace_p.write_text(json.dumps({f"{c}:{t}": v for (c, t), v in live.items()}))
    summ = summarize(rows, a.turns)
    ok = [x for x in rows if x["ok"]]
    res = {"arm": a.arm, "order": a.order, "seed": a.seed, "url": a.url, "model": a.model, "plan": plan, "trace": "recorded" if recording else "replayed",
           "wall_s": round(time.time() - t_run, 1), "turns": summ, "activity": {s: activity.count(s) for s in set(activity)},
           "totals": {"requests": len(rows), "errors": len(rows) - len(ok),
                      "prompt_tokens": sum(x["prompt_tokens"] or 0 for x in ok), "cached_tokens": sum(x["cached_tokens"] for x in ok),
                      "ttft_s_sum": round(sum(x["ttft_s"] for x in ok if x["ttft_s"] is not None), 2),
                      "decode_tok_s_median": round(statistics.median([x["decode_tok_s"] for x in ok if x["decode_tok_s"]]), 1)}}
    (out / f"{a.arm}-{a.order}-seed{a.seed}.summary.json").write_text(json.dumps(res, indent=1))
    print("| turn | prompt tok | cached tok (frac) | TTFT s median (min-max) | decode tok/s median (min-max) |", flush=True)
    for s in summ:
        print(f"| {s['turn']} | {s['prompt_tokens_median']:.0f} | {s['cached_tokens_median']:.0f} ({s['cached_frac_median']}) | "
              f"{s['ttft_s_median']} ({s['ttft_s_range'][0]}-{s['ttft_s_range'][1]}) | {s['decode_tok_s_median']} ({s['decode_tok_s_range'][0]}-{s['decode_tok_s_range'][1]}) |", flush=True)
    print(f"[replay] totals {res['totals']} wall {res['wall_s']}s", flush=True)
    return 0 if not res["totals"]["errors"] else 3


if __name__ == "__main__":
    sys.exit(main())
