#!/usr/bin/env python3
"""drill: 20 greedy requests (4 multi-turn conversations of the ab35e replay trace, turns in order) -> JSON, for byte-identity checks
across restarts. Prompts are byte-identical every run (plan + trace of docs/results/2026-11-20-ab35e-cutover-scoreboard/raw), temperature 0,
thinking off, 100 tokens, one request at a time. Run in the localeval environment:
  cd /opt/code/deepeval && uv run python scripts/drill/verify.py --url http://spark-fa00:8191/v1 --model guidance --out OUT.json
"""
import argparse, json, pathlib, sys, time

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "ab35b"))
import replay as base  # noqa: E402
from localeval.client import Client  # noqa: E402
from localeval.config import Target  # noqa: E402

RAW = HERE.parent.parent / "docs/results/2026-11-20-ab35e-cutover-scoreboard/raw"
SEQ = [(c, t) for c in range(3) for t in range(6)] + [(3, 0), (3, 1)]  # 20 requests


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=100)
    ap.add_argument("--n", type=int, default=20)
    a = ap.parse_args()
    plan = json.loads((RAW / "replay-plan.json").read_text())
    trace = json.loads((RAW / f"replay-trace-seed{a.seed}.json").read_text())
    texts = {(c, t): base.user_msg(a.seed, c, t + 1, plan["sentences"][t]) for c in range(8) for t in range(6)}
    cl = Client(Target(name="drill", url=a.url, model=a.model, thinking="off"))
    rows = []
    t0 = time.time()
    for i, (c, t) in enumerate(SEQ[: a.n]):
        msgs = [{"role": "system", "content": base.SYSTEM}]
        for k in range(t):
            msgs += [{"role": "user", "content": texts[(c, k)]}, {"role": "assistant", "content": trace[f"{c}:{k}"]}]
        msgs.append({"role": "user", "content": texts[(c, t)]})
        r = cl.chat(msgs, max_tokens=a.max_tokens, temperature=0.0, thinking="off")
        rows.append({"i": i, "conv": c, "turn": t + 1, "ok": r.ok, "error": r.error, "reply": r.text, "raw": r.raw_output,
                     "prompt_tokens": r.prompt_tokens, "cached_tokens": r.cached_tokens or 0, "completion_tokens": r.completion_tokens,
                     "finish_reason": r.finish_reason, "ttft_s": r.ttft_s, "total_s": r.total_s})
        print(f"[verify] {i} c{c} t{t+1} ok={r.ok} prompt={r.prompt_tokens} cached={r.cached_tokens or 0} ttft={r.ttft_s and round(r.ttft_s,2)} out={r.completion_tokens} {r.error or ''}", flush=True)
    json.dump({"url": a.url, "wall_s": round(time.time() - t0, 1), "rows": rows}, open(a.out, "w"), indent=1)
    print(f"[verify] done {len(rows)} requests, {sum(not r['ok'] for r in rows)} errors, wall {time.time()-t0:.1f}s, cached tokens {sum(r['cached_tokens'] for r in rows)}")
    return 0 if all(r["ok"] for r in rows) else 3


if __name__ == "__main__":
    sys.exit(main())
