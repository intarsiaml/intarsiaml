#!/usr/bin/env python3
"""restore-acc: GSM8K with a long shared prefix (system prompt + N worked few-shot turns), so every request after the first can
restore that prefix from the prefix store. Same questions/ids/golds as the pinned localeval GSM8K arm (gsm8k_dump.py output).

  cd /opt/code/deepeval && uv run python scripts/soak/gsm8k_prefix.py --url http://spark-fa00:8191/v1 --model guidance \
      --inputs gsm8k-inputs.json --arm W --out OUT_DIR [--concurrency 1] [--shots 32] [--limit 200] [--max-tokens 512]

Messages: system (fixed) | N x (user "**Problem**: <train q>\\n\\n**Answer**:", assistant <train worked solution ending '#### n'>) |
user "**Problem**: <test q>\\n\\n**Answer**:". Thinking off (chat_template_kwargs enable_thinking=false), temperature 0, stream off,
return_token_ids on (intarsia extension: the generated ids, for the first diverging token). Scoring: localeval flexible-v1
(final number or '#### n' marker against gold; a reply cut by max_tokens scores 0).
Output: <arm>.jsonl (one row per question, in id order) and <arm>.summary.json."""
import argparse
import json
import pathlib
import statistics
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from localeval import gsm8k

SYSTEM = ("You are a careful grade-school math tutor. Solve each problem step by step, writing every calculation out in the "
          "same style as the worked examples, and finish with a final line of the form '#### <number>' that holds only the "
          "final numerical answer, with no units and no commas.")


def user_msg(q):
    return {"role": "user", "content": f"**Problem**: {q}\n\n**Answer**:"}


def build_messages(shots, q):
    m = [{"role": "system", "content": SYSTEM}]
    for s in shots:
        m += [user_msg(s["question"]), {"role": "assistant", "content": s["answer"]}]
    return m + [user_msg(q)]


def post(url, body, timeout=180):  # a request takes < 30 s; a hung socket (network blip) must not stall an arm for 15 min
    req = urllib.request.Request(url + "/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def one(args, shots, q):
    body = {"model": args.model, "messages": build_messages(shots, q["question"]), "max_tokens": args.max_tokens, "temperature": 0.0,
            "stream": False, "return_token_ids": True, "chat_template_kwargs": {"enable_thinking": False}}
    t0 = time.time()
    row = {"id": q["id"], "gold": q["gold"], "ok": False}
    for attempt in (0, 1):
        try:
            d = post(args.url, body)
            ch = d["choices"][0]
            u = d.get("usage", {})
            row.update(ok=True, reply=ch["message"].get("content") or "", token_ids=ch.get("token_ids"), finish_reason=ch.get("finish_reason"),
                       prompt_tokens=u.get("prompt_tokens"), completion_tokens=u.get("completion_tokens"),
                       cached_tokens=(u.get("prompt_tokens_details") or {}).get("cached_tokens"),
                       prefill_chunk=(u.get("prompt_tokens_details") or {}).get("prefill_chunk"))
            break
        except Exception as e:  # one retry on a transport error, then scored wrong
            row["error"] = repr(e)[:300]
            time.sleep(2)
    row["latency_s"] = round(time.time() - t0, 3)
    trunc = row.get("finish_reason") == "length"
    row["truncated"] = trunc
    row["extracted"] = gsm8k.extract_answer(row.get("reply"))
    row["correct"] = bool(row["ok"] and gsm8k.flexible_match(q["gold"], row.get("reply")) and not trunc)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--shots", type=int, default=32)
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--resume", action="store_true", help="keep the ok rows already in <arm>.jsonl and send only the missing questions")
    args = ap.parse_args()
    d = json.load(open(args.inputs))
    shots, qs = d["shots"][: args.shots], d["questions"][: args.limit]
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    rows = [None] * len(qs)
    lock = threading.Lock()
    done = [0]
    fails = [0]  # consecutive failed requests: five in a row means the server is gone, stop instead of retrying a dead port
    if args.resume and (out / f"{args.arm}.jsonl").exists():
        have = {r["id"]: r for r in map(json.loads, open(out / f"{args.arm}.jsonl")) if r.get("ok")}
        for i, q in enumerate(qs):
            rows[i] = have.get(q["id"])
        print(f"[{args.arm}] resume: {sum(r is not None for r in rows)} rows kept, {sum(r is None for r in rows)} to send", flush=True)
    live = open(out / f"{args.arm}.jsonl", "a" if args.resume else "w")  # rows are appended as they finish (a killed run keeps what it has)

    def run(i):
        if fails[0] >= 5 or rows[i] is not None:
            return rows[i]
        r = one(args, shots, qs[i])
        rows[i] = r
        with lock:
            live.write(json.dumps(r) + "\n")
            live.flush()
            fails[0] = 0 if r["ok"] else fails[0] + 1
            done[0] += 1
            if done[0] % 20 == 0:
                print(f"[{args.arm}] {done[0]}/{len(qs)} {time.time() - t0:.0f}s correct so far {sum(x['correct'] for x in rows if x)}", flush=True)
        return r

    with ThreadPoolExecutor(args.concurrency) as ex:
        list(ex.map(run, range(len(qs))))
    wall = time.time() - t0
    live.close()
    if fails[0] >= 5:
        print(f"[{args.arm}] ABORTED after 5 consecutive failures: {sum(r is not None for r in rows)} of {len(rows)} requests sent", flush=True)
    rows = [r for r in rows if r is not None]
    with open(out / f"{args.arm}.jsonl", "w") as f:  # final file in question order
        for r in rows:
            f.write(json.dumps(r) + "\n")
    ok = [r for r in rows if r["ok"]]
    pt = [r["prompt_tokens"] for r in ok if r["prompt_tokens"]]
    ct = [r["cached_tokens"] or 0 for r in ok]
    hit = [c for c in ct if c > 0]
    summ = {"arm": args.arm, "n": len(rows), "aborted": fails[0] >= 5, "errors": len(rows) - len(ok), "correct": sum(r["correct"] for r in rows),
            "acc": round(sum(r["correct"] for r in rows) / len(rows), 4), "truncated": sum(r["truncated"] for r in rows),
            "mean_completion_tokens": round(statistics.mean(r["completion_tokens"] for r in ok), 1) if ok else None,
            "prompt_tokens_median": statistics.median(pt) if pt else None, "prompt_tokens_min": min(pt) if pt else None,
            "prompt_tokens_max": max(pt) if pt else None,
            "requests_with_cached_tokens": len(hit), "cached_tokens_median_of_hits": statistics.median(hit) if hit else 0,
            "cached_tokens_sum": sum(ct), "cached_frac": round(sum(ct) / sum(pt), 4) if pt else None,
            "latency_s_median": round(statistics.median(r["latency_s"] for r in ok), 3) if ok else None, "wall_s": round(wall, 1),
            "concurrency": args.concurrency, "shots": len(shots), "max_tokens": args.max_tokens}
    json.dump(summ, open(out / f"{args.arm}.summary.json", "w"), indent=1)
    print(json.dumps(summ))


if __name__ == "__main__":
    sys.exit(main())
